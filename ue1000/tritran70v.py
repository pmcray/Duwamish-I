"""TRI-TRAN 70V — Forrester/Oliver Vectorized Balanced-Ternary FORTRAN-66 Compiler.

Targets TRIAD-36 assembly on the Duwamish-I (UE-1000, Principles of Operation Rev. 1.5):
  - Authentic USA Standard FORTRAN-66 (X3.9-1966) fixed-form 80-col cards (also accepts
    modern free indentation cleanly when col 1-6 are blank), implicit I-N INTEGER / REAL,
    static storage, 1-based column-major 1D/2D arrays, SUBROUTINE & FUNCTION by reference.
  - Native balanced ternary:
      * 3-valued LOGICAL (.TRUE.=+1, .UNDECIDED./.UNKNOWN.=0, .FALSE.=-1)
      * Kleene .AND./.OR./.EQV./.NOT./.NEQV.; Logical IF fires ONLY on .TRUE. (+1)
      * Arithmetic IF (E) L1, L2, L3 -> single hardware B3 (using FCM for REAL!)
      * FORMAT fields Iw, Fw.d, Ew.d, Lw (T/F/U), Bw (balanced ternary!), Hw (heptavintimal!)
  - 70V Supercomputer Vector & T-Fabric extensions:
      * Vector array-section syntax `C(1:N) = A(1:N) + B(1:N)` (and `-`, `*`, `FMA`, `.AND.`, `.OR.`)
        strip-mined directly into C-Complex VL <= 27 `VSVL/VLD/VADD/VFMA/VST` loops!
      * Vector reductions `VSUM(A, N)`, `VMAX(A, N)`, `VMIN(A, N)`, `VDOT(A, B, N)`
      * T-Fabric dispatch `CALL TGEMM(...)` & `CALL TFFT3(...)` emitting 9-word TTDs +
        `TEXEC A6, S0` / `TWAIT S0, #0` (Rev. 1.5 S0 all-27-tiles shorthand!)
  - Reuses SAL-3/O peephole optimizer & Cray Dual-Parcel 18-trit instruction packer!
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .machine import UE1000Machine
from .sal3 import _pack_dual_parcels, _peephole_optimize
from .ternary import (
    POW3,
    WORD_TRITS,
    char_to_obb729,
    in_range,
    int_to_hept,
    int_to_trit_str,
    sign_trit,
    wrap,
)
from .tfabric import MODE_RADIX3_FFT, MODE_TBFP9_GEMM, TileTaskDescriptor
from .tfloat import (
    TF36,
    db_to_ticks,
    float_to_tfloat,
    int_to_tfloat,
    tfloat_to_float,
    tfloat_to_int,
    ticks_to_db,
)
from .triad36 import AssembledProgram, assemble


class TriTranError(Exception):
    """Compile-time or runtime error in TRI-TRAN 70V."""


# ---------------------------------------------------------------------------
# Card Deck Preprocessor (FORTRAN-66 Columns 1-5 label, 6 cont, 7-72 body)
# ---------------------------------------------------------------------------

@dataclass
class CardLine:
    label: Optional[int]
    text: str
    lineno: int
    vector_hint: bool = False


def preprocess_cards(source: str) -> List[CardLine]:
    """Parse fixed-form 80-column FORTRAN-66 cards with continuation & C$VECTOR hints."""
    cards: List[CardLine] = []
    pending_vector = False

    for lineno, raw in enumerate(source.splitlines(), start=1):
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        col1 = line[0]
        if col1 in ("C", "c", "*", "!"):
            if line[1:].strip().upper().startswith("$VECTOR"):
                pending_vector = True
            continue

        # Truncate at column 72 (FORTRAN-66 card sequence field 73-80 ignored)
        card = line[:72]
        # Detect continuation in column 6 (only when col 1-5 are blank and len >= 6)
        if len(card) >= 6 and card[:5].strip() == "" and card[5] not in (" ", "0"):
            if not cards:
                raise TriTranError(f"Line {lineno}: Continuation card without preceding statement")
            cards[-1].text += " " + card[6:].strip()
            continue

        # Parse optional statement label in columns 1-5
        lbl_field = card[:5].strip()
        body = card[6:].strip() if len(card) >= 6 else ""
        # Also allow cleanly indented free-ish lines where label isn't in cols 1-5
        if lbl_field and lbl_field.isdigit():
            label = int(lbl_field)
        elif lbl_field == "":
            label = None
        else:
            # No leading 6-space margin (e.g., short inline test snippet): split leading int if any
            m = re.match(r"^\s*(\d+)\s+(.*)$", card)
            if m:
                label = int(m.group(1))
                body = m.group(2).strip()
            else:
                label = None
                body = card.strip()

        if not body:
            continue
        cards.append(CardLine(label=label, text=body, lineno=lineno, vector_hint=pending_vector))
        pending_vector = False

    return cards


# ---------------------------------------------------------------------------
# Expression Tokenizer & Recursive-Descent Parser (FORTRAN-66 Precedence)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FTok:
    kind: str
    val: object


_DOT_OPS = {
    ".TRUE.": ("LOG_LIT", 1),
    ".FALSE.": ("LOG_LIT", -1),
    ".UNDECIDED.": ("LOG_LIT", 0),
    ".UNKNOWN.": ("LOG_LIT", 0),
    ".NOT.": ("DOTOP", ".NOT."),
    ".AND.": ("DOTOP", ".AND."),
    ".OR.": ("DOTOP", ".OR."),
    ".EQV.": ("DOTOP", ".EQV."),
    ".NEQV.": ("DOTOP", ".NEQV."),
    ".LT.": ("RELOP", ".LT."),
    ".LE.": ("RELOP", ".LE."),
    ".EQ.": ("RELOP", ".EQ."),
    ".NE.": ("RELOP", ".NE."),
    ".GE.": ("RELOP", ".GE."),
    ".GT.": ("RELOP", ".GT."),
}


def tokenize_fexpr(text: str) -> List[FTok]:
    toks: List[FTok] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in " \t":
            i += 1
            continue
        if ch == "'":
            j = i + 1
            chars = []
            while j < n:
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        chars.append("'")
                        j += 2
                        continue
                    break
                chars.append(text[j])
                j += 1
            toks.append(FTok("STR", "".join(chars)))
            i = j + 1
            continue
        # Dot operators / logical constants: .TRUE., .LT., .AND., etc.
        if ch == ".":
            m = re.match(r"\.[A-Za-z]+\.", text[i:])
            if m:
                key = m.group(0).upper()
                if key in _DOT_OPS:
                    k, v = _DOT_OPS[key]
                    toks.append(FTok(k, v))
                    i += len(m.group(0))
                    continue
        if text[i : i + 2] == "**":
            toks.append(FTok("OP", "**"))
            i += 2
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            is_real = False
            while j < n and text[j].isdigit():
                j += 1
            if j < n and text[j] == ".":
                # Guard against `1.LT.2` where `.` starts a dot-op!
                if not re.match(r"\.[A-Za-z]+\.", text[j:]):
                    is_real = True
                    j += 1
                    while j < n and text[j].isdigit():
                        j += 1
            if j < n and text[j] in "eEdD":
                is_real = True
                j += 1
                if j < n and text[j] in "+-":
                    j += 1
                while j < n and text[j].isdigit():
                    j += 1
            num = text[i:j].replace("D", "E").replace("d", "e")
            if text[j : j + 2].upper() == "DB":
                toks.append(FTok("DB", float(num)))
                i = j + 2
            elif is_real:
                toks.append(FTok("REAL", float(num)))
                i = j
            else:
                toks.append(FTok("INT", int(num)))
                i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            toks.append(FTok("ID", text[i:j].upper()))
            i = j
            continue
        if ch in "+-*/(),:=@":
            toks.append(FTok("OP", ch))
            i += 1
            continue
        raise TriTranError(f"Unexpected character {ch!r} in expression {text!r}")
    toks.append(FTok("EOF", None))
    return toks


# Expression AST
@dataclass
class FLit:
    typ: str   # "INTEGER" | "REAL" | "LOGICAL" | "DECIBAN"
    val: object


@dataclass
class FVar:
    name: str


@dataclass
class FRef:
    name: str
    args: List[object]  # Array subscripts OR function call arguments (disambiguated by symtab)


@dataclass
class FUnary:
    op: str
    operand: object


@dataclass
class FBin:
    op: str
    left: object
    right: object


class FExprParser:
    def __init__(self, toks: List[FTok]) -> None:
        self.toks = toks
        self.pos = 0

    def peek(self) -> FTok:
        return self.toks[self.pos]

    def match(self, kind: str, val: object = None) -> Optional[FTok]:
        t = self.peek()
        if t.kind == kind and (val is None or t.val == val):
            self.pos += 1
            return t
        return None

    def expect(self, kind: str, val: object = None) -> FTok:
        t = self.peek()
        if t.kind != kind or (val is not None and t.val != val):
            raise TriTranError(f"Expected {kind} {val or ''}, got {t}")
        self.pos += 1
        return t

    def parse(self) -> object:
        e = self.parse_eqv()
        if self.peek().kind != "EOF":
            raise TriTranError(f"Trailing tokens in expression: {self.toks[self.pos:]}")
        return e

    def parse_eqv(self) -> object:
        left = self.parse_or()
        while self.peek().kind == "DOTOP" and self.peek().val in (".EQV.", ".NEQV."):
            op = str(self.peek().val)
            self.pos += 1
            left = FBin(op, left, self.parse_or())
        return left

    def parse_or(self) -> object:
        left = self.parse_and()
        while self.match("DOTOP", ".OR."):
            left = FBin(".OR.", left, self.parse_and())
        return left

    def parse_and(self) -> object:
        left = self.parse_not()
        while self.match("DOTOP", ".AND."):
            left = FBin(".AND.", left, self.parse_not())
        return left

    def parse_not(self) -> object:
        if self.match("DOTOP", ".NOT."):
            return FUnary(".NOT.", self.parse_not())
        return self.parse_rel()

    def parse_rel(self) -> object:
        left = self.parse_add()
        if self.peek().kind == "RELOP":
            op = str(self.peek().val)
            self.pos += 1
            return FBin(op, left, self.parse_add())
        return left

    def parse_add(self) -> object:
        left = self.parse_mul()
        while self.peek().kind == "OP" and self.peek().val in ("+", "-"):
            op = str(self.peek().val)
            self.pos += 1
            left = FBin(op, left, self.parse_mul())
        return left

    def parse_mul(self) -> object:
        left = self.parse_unary()
        while self.peek().kind == "OP" and self.peek().val in ("*", "/"):
            op = str(self.peek().val)
            self.pos += 1
            left = FBin(op, left, self.parse_unary())
        return left

    def parse_unary(self) -> object:
        if self.match("OP", "-"):
            return FUnary("-", self.parse_unary())
        if self.match("OP", "+"):
            return self.parse_unary()
        return self.parse_pow()

    def parse_pow(self) -> object:
        base = self.parse_primary()
        if self.match("OP", "**"):
            return FBin("**", base, self.parse_unary())
        return base

    def parse_primary(self) -> object:
        t = self.peek()
        if self.match("INT"):
            return FLit("INTEGER", int(t.val))
        if self.match("REAL"):
            return FLit("REAL", float(t.val))
        if self.match("LOG_LIT"):
            return FLit("LOGICAL", int(t.val))
        if self.match("DB"):
            return FLit("DECIBAN", float(t.val))
        if self.match("OP", "("):
            e = self.parse_eqv()
            self.expect("OP", ")")
            return e
        if self.match("ID"):
            name = str(t.val)
            if self.match("OP", "("):
                args: List[object] = []
                if not self.match("OP", ")"):
                    while True:
                        args.append(self.parse_eqv())
                        if not self.match("OP", ","):
                            break
                    self.expect("OP", ")")
                return FRef(name, args)
            return FVar(name)
        raise TriTranError(f"Unexpected token {t} in FORTRAN expression")


def parse_fexpr(text: str) -> object:
    return FExprParser(tokenize_fexpr(text)).parse()


def _split_top_commas(text: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    in_str = False
    cur: List[str] = []
    for ch in text:
        if ch == "'":
            in_str = not in_str
        elif not in_str:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append("".join(cur).strip())
                cur = []
                continue
        cur.append(ch)
    if cur or parts:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# FORTRAN-66 FORMAT Parser & Runtime Formatter (with Bw Ternary & Hw Hept!)
# ---------------------------------------------------------------------------

def parse_format_spec(fmt_body: str) -> List[Tuple[str, object]]:
    """Parse FORMAT(...) specification into flat list of (kind, args) items."""
    s = fmt_body.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    items: List[Tuple[str, object]] = []
    i = 0
    n = len(s)

    def parse_group(txt: str) -> List[Tuple[str, object]]:
        res: List[Tuple[str, object]] = []
        p = 0
        m = len(txt)
        while p < m:
            while p < m and txt[p] in " \t,":
                p += 1
            if p >= m:
                break
            if txt[p] == "/":
                res.append(("NL", None))
                p += 1
                continue
            if txt[p] == "'":
                q = p + 1
                ch_buf = []
                while q < m:
                    if txt[q] == "'":
                        if q + 1 < m and txt[q + 1] == "'":
                            ch_buf.append("'")
                            q += 2
                            continue
                        break
                    ch_buf.append(txt[q])
                    q += 1
                res.append(("LIT", "".join(ch_buf)))
                p = q + 1
                continue
            # Optional repeat count
            rep_start = p
            while p < m and txt[p].isdigit():
                p += 1
            rep = int(txt[rep_start:p]) if p > rep_start else 1
            if p > rep_start and p < m and txt[p] in "hH":
                # nH Hollerith string (must have explicit leading digit n!)
                hol = txt[p + 1 : p + 1 + rep]
                res.append(("LIT", hol))
                p = p + 1 + rep
                continue
            if p < m and txt[p] in "xX":
                res.append(("LIT", " " * rep))
                p += 1
                continue
            if p < m and txt[p] == "(":
                depth = 1
                q = p + 1
                while q < m and depth > 0:
                    if txt[q] == "(":
                        depth += 1
                    elif txt[q] == ")":
                        depth -= 1
                    q += 1
                sub = parse_group(txt[p + 1 : q - 1])
                for _ in range(rep):
                    res.extend(sub)
                p = q
                continue
            # Data edit descriptor: Iw, Fw.d, Ew.d, Lw, Bw, Hw, Dw
            code = txt[p].upper()
            p += 1
            w_start = p
            while p < m and txt[p].isdigit():
                p += 1
            w = int(txt[w_start:p]) if p > w_start else 0
            d = 0
            if p < m and txt[p] == ".":
                p += 1
                d_start = p
                while p < m and txt[p].isdigit():
                    p += 1
                d = int(txt[d_start:p]) if p > d_start else 0
            for _ in range(rep):
                res.append(("EDIT", (code, w, d)))
        return res

    return parse_group(s)


def format_fortran_record(spec_items: List[Tuple[str, object]], values: List[Tuple[str, int]]) -> str:
    """Format values [(typ, raw_t36_word), ...] using FORTRAN-66 + Bw/Hw/Lw spec."""
    out: List[str] = []
    v_idx = 0
    for kind, arg in spec_items:
        if kind == "LIT":
            out.append(str(arg))
        elif kind == "NL":
            out.append("\n")
        elif kind == "EDIT":
            if v_idx >= len(values):
                break
            code, w, d = arg  # type: ignore[misc]
            v_typ, raw = values[v_idx]
            v_idx += 1
            if code == "I":
                s = str(raw)
                s = "*" * w if (w > 0 and len(s) > w) else s.rjust(w)
            elif code == "F":
                fv = tfloat_to_float(raw, TF36) if v_typ == "REAL" else float(raw)
                s = "UNDECIDED" if math.isnan(fv) else f"{fv:.{d}f}"
                s = "*" * w if (w > 0 and len(s) > w) else s.rjust(w)
            elif code == "E":
                fv = tfloat_to_float(raw, TF36) if v_typ == "REAL" else float(raw)
                s = "UNDECIDED" if math.isnan(fv) else f"{fv:.{d}E}"
                s = "*" * w if (w > 0 and len(s) > w) else s.rjust(w)
            elif code == "L":
                ch = {1: "T", 0: "U", -1: "F"}.get(sign_trit(raw), "?")
                s = ch.rjust(max(1, w))
            elif code == "B":
                # Balanced ternary with T for -1, trimmed of leading zeros!
                full = int_to_trit_str(raw, WORD_TRITS).lstrip("0") or "0"
                s = "*" * w if (w > 0 and len(full) > w) else full.rjust(w)
            elif code == "H":
                # UE Heptavintimal!
                full = int_to_hept(raw, WORD_TRITS).lstrip("0") or "0"
                s = "*" * w if (w > 0 and len(full) > w) else full.rjust(w)
            elif code == "D":
                s = f"{ticks_to_db(raw):+.{d or 2}f}dB"
                s = s.rjust(w)
            else:
                s = str(raw)
            out.append(s)
    # Strip standard FORTRAN carriage-control leading space if present
    line = "".join(out)
    return line[1:] if line.startswith(" ") else line


# ---------------------------------------------------------------------------
# Compiler (Multi-Unit Program -> TRIAD-36 Assembly)
# ---------------------------------------------------------------------------

_CMP_SEL = {
    ".LT.": -11,
    ".LE.": -5,
    ".EQ.": -7,
    ".NE.": 7,
    ".GE.": 11,
    ".GT.": 5,
}

# Exact TF36 transcendental math SVC dispatch codes (handled by C-EXEC math library)
_MATH_SVCS: Dict[str, Tuple[int, str, str]] = {
    "SQRT":   (101, "REAL", "REAL"),
    "EXP":    (102, "REAL", "REAL"),
    "ALOG":   (103, "REAL", "REAL"),
    "ALOG10": (104, "REAL", "REAL"),
    "SIN":    (105, "REAL", "REAL"),
    "COS":    (106, "REAL", "REAL"),
    "TANH":   (107, "REAL", "REAL"),
    "ATAN":   (108, "REAL", "REAL"),
}


@dataclass
class FSymbol:
    name: str
    typ: str                      # "INTEGER" | "REAL" | "LOGICAL" | "DECIBAN"
    dims: Tuple[int, ...] = ()    # () scalar, (N,) 1D, (M, N) 2D
    is_param: bool = False        # FORTRAN pass-by-reference!
    param_idx: int = 0
    init_vals: Optional[List[int]] = None

    @property
    def total_words(self) -> int:
        p = 1
        for d in self.dims:
            p *= d
        return p


@dataclass
class UnitInfo:
    kind: str               # "MAIN" | "SUBROUTINE" | "FUNCTION"
    name: str
    ret_type: str
    params: List[str]
    cards: List[CardLine]
    symbols: Dict[str, FSymbol] = field(default_factory=dict)
    formats: Dict[int, List[Tuple[str, object]]] = field(default_factory=dict)


@dataclass
class TriTranCompileResult:
    assembly: str
    assembled: AssembledProgram
    formats: Dict[int, List[Tuple[str, object]]]
    write_records: Dict[int, Tuple[int, List[str]]]  # svc_tag -> (fmt_label, arg_types)
    vector_loops_emitted: int
    peephole_edits: int
    parcels_packed: int


class TriTran70VCompiler:
    def __init__(self, source: str, opt: bool = True, origin: int = 100) -> None:
        self.cards = preprocess_cards(source)
        self.opt = opt
        self.origin = origin
        self.units: List[UnitInfo] = []
        self.unit_by_name: Dict[str, UnitInfo] = {}
        self.code: List[str] = []
        self.const_pool: Dict[int, str] = {}
        self.ttd_slots: List[str] = []
        self.all_formats: Dict[int, List[Tuple[str, object]]] = {}
        self.write_records: Dict[int, Tuple[int, List[str]]] = {}
        self._write_tag_seq = 200  # SVC #200+ for formatted WRITE records!
        self._lbl_seq = 0
        self.vector_loops_emitted = 0

    def _new_lbl(self, prefix: str = "L") -> str:
        self._lbl_seq += 1
        return f"_F{prefix}{self._lbl_seq}"

    def _emit(self, line: str) -> None:
        self.code.append(line)

    @staticmethod
    def _implicit_type(name: str) -> str:
        return "INTEGER" if name[0].upper() in "IJKLMN" else "REAL"

    def _load_const(self, reg: str, raw: int) -> None:
        if in_range(raw, 7):
            self._emit(f"ADD {reg}, S0, #{raw}")
        elif in_range(raw, 22):
            self._emit(f"LDI {reg}, #{raw}")
        else:
            if raw not in self.const_pool:
                self.const_pool[raw] = f"_FK{len(self.const_pool)}"
            self._emit(f"LD {reg}, {self.const_pool[raw]}(A0)")

    def _split_units(self) -> None:
        cur_cards: List[CardLine] = []
        cur_kind = "MAIN"
        cur_name = "MAIN"
        cur_ret = "INTEGER"
        cur_params: List[str] = []

        for card in self.cards:
            up = card.text.strip().upper()
            if not cur_cards:
                m_prog = re.match(r"^PROGRAM\s+([A-Z][A-Z0-9_]*)$", up)
                m_sub = re.match(r"^SUBROUTINE\s+([A-Z][A-Z0-9_]*)\s*(?:\(([^)]*)\))?$", up)
                m_fn = re.match(
                    r"^(?:(INTEGER|REAL|LOGICAL|DECIBAN)\s+)?FUNCTION\s+([A-Z][A-Z0-9_]*)\s*\(([^)]*)\)$",
                    up,
                )
                if m_prog:
                    cur_kind, cur_name = "MAIN", m_prog.group(1)
                    continue
                if m_sub:
                    cur_kind = "SUBROUTINE"
                    cur_name = m_sub.group(1)
                    cur_params = [p.strip() for p in (m_sub.group(2) or "").split(",") if p.strip()]
                    continue
                if m_fn:
                    cur_kind = "FUNCTION"
                    cur_name = m_fn.group(2)
                    cur_ret = m_fn.group(1) or self._implicit_type(cur_name)
                    cur_params = [p.strip() for p in m_fn.group(3).split(",") if p.strip()]
                    continue
            if up == "END":
                u = UnitInfo(cur_kind, cur_name, cur_ret, cur_params, cur_cards)
                self.units.append(u)
                self.unit_by_name[cur_name] = u
                cur_cards = []
                cur_kind, cur_name, cur_ret, cur_params = "MAIN", "MAIN", "INTEGER", []
                continue
            cur_cards.append(card)

        if cur_cards:
            u = UnitInfo(cur_kind, cur_name, cur_ret, cur_params, cur_cards)
            self.units.append(u)
            self.unit_by_name[cur_name] = u

    def _sym_label(self, unit: UnitInfo, name: str) -> str:
        return f"_U_{unit.name}_{name}"

    def _get_or_create_sym(self, unit: UnitInfo, name: str) -> FSymbol:
        nm = name.upper()
        if nm not in unit.symbols:
            unit.symbols[nm] = FSymbol(name=nm, typ=self._implicit_type(nm))
        return unit.symbols[nm]

    def _parse_declarations(self, unit: UnitInfo) -> List[CardLine]:
        """Extract INTEGER/REAL/LOGICAL/DECIBAN/DIMENSION/DATA/FORMAT; return executable cards."""
        for idx, pname in enumerate(unit.params):
            unit.symbols[pname] = FSymbol(
                name=pname, typ=self._implicit_type(pname), is_param=True, param_idx=idx
            )
        if unit.kind == "FUNCTION":
            unit.symbols[unit.name] = FSymbol(name=unit.name, typ=unit.ret_type)

        exec_cards: List[CardLine] = []
        for card in unit.cards:
            txt = card.text.strip()
            up = txt.upper()

            m_fmt = re.match(r"^FORMAT\s*(\(.*\))$", txt, re.IGNORECASE)
            if m_fmt:
                if card.label is None:
                    raise TriTranError(f"Line {card.lineno}: FORMAT requires a statement label")
                parsed = parse_format_spec(m_fmt.group(1))
                unit.formats[card.label] = parsed
                self.all_formats[card.label] = parsed
                continue

            m_decl = re.match(r"^(INTEGER|REAL|LOGICAL|DECIBAN|DIMENSION)\s+(.*)$", up)
            if m_decl:
                kw = m_decl.group(1)
                rest = m_decl.group(2)
                for item in _split_top_commas(rest):
                    m_arr = re.match(r"^([A-Z][A-Z0-9_]*)\s*(?:\(([^)]+)\))?$", item)
                    if not m_arr:
                        raise TriTranError(f"Line {card.lineno}: Bad declaration item {item!r}")
                    nm = m_arr.group(1)
                    dims: Tuple[int, ...] = ()
                    if m_arr.group(2):
                        dims = tuple(int(d.strip()) for d in m_arr.group(2).split(","))
                    sym = unit.symbols.get(nm)
                    if sym is None:
                        typ = self._implicit_type(nm) if kw == "DIMENSION" else kw
                        unit.symbols[nm] = FSymbol(name=nm, typ=typ, dims=dims)
                    else:
                        if kw != "DIMENSION":
                            sym.typ = kw
                        if dims:
                            sym.dims = dims
                continue

            m_data = re.match(r"^DATA\s+(.*)$", txt, re.IGNORECASE)
            if m_data:
                self._parse_data_stmt(unit, m_data.group(1), card.lineno)
                continue

            exec_cards.append(card)
        return exec_cards

    def _parse_data_stmt(self, unit: UnitInfo, body: str, lineno: int) -> None:
        # Format: VAR1, VAR2 /val1, 3*val2/, ...
        for m in re.finditer(r"([^/]+)/([^/]+)/", body):
            names = [x.strip().upper() for x in m.group(1).strip(", ").split(",") if x.strip()]
            raw_vals: List[int] = []
            for tok in _split_top_commas(m.group(2)):
                m_rep = re.match(r"^(\d+)\s*\*\s*(.+)$", tok)
                rep = int(m_rep.group(1)) if m_rep else 1
                v_str = m_rep.group(2) if m_rep else tok
                lit = parse_fexpr(v_str)
                raw_vals.extend([self._eval_data_lit(lit)] * rep)
            pos = 0
            for nm in names:
                sym = self._get_or_create_sym(unit, nm)
                cnt = sym.total_words
                sym.init_vals = raw_vals[pos : pos + cnt]
                while len(sym.init_vals) < cnt:
                    sym.init_vals.append(0)
                pos += cnt

    def _eval_data_lit(self, expr: object) -> int:
        if isinstance(expr, FLit):
            if expr.typ == "INTEGER":
                return wrap(int(expr.val), WORD_TRITS)
            if expr.typ == "REAL":
                return float_to_tfloat(float(expr.val), TF36)
            if expr.typ == "LOGICAL":
                return int(expr.val)
            if expr.typ == "DECIBAN":
                return db_to_ticks(float(expr.val))
        if isinstance(expr, FUnary) and expr.op == "-":
            inner = self._eval_data_lit(expr.operand)
            if isinstance(expr.operand, FLit) and expr.operand.typ == "REAL":
                return float_to_tfloat(-float(expr.operand.val), TF36)
            return wrap(-inner, WORD_TRITS)
        raise TriTranError(f"Non-constant DATA initializer: {expr}")

    def compile(self) -> TriTranCompileResult:
        self._split_units()
        if not self.units:
            raise TriTranError("Empty TRI-TRAN 70V program")

        # Main entry stub
        main_unit = next((u for u in self.units if u.kind == "MAIN"), self.units[0])
        self._emit("; === TRI-TRAN 70V Compiled TRIAD-36 Image (UE-1000 Rev. 1.5) ===")
        self._emit("LDI S8, #200000")
        self._emit(f"CALL A0, _F_{main_unit.name}")
        self._emit("HLT")

        # Pass 1: Declarations for all units (so cross-unit return types are known!)
        unit_execs: List[Tuple[UnitInfo, List[CardLine]]] = []
        for u in self.units:
            unit_execs.append((u, self._parse_declarations(u)))

        # Pass 2: Compile every program unit
        for u, exec_cards in unit_execs:
            self._compile_unit(u, exec_cards)

        # Pass 3: Emit static FORTRAN storage (all variables/arrays static per X3.9-1966!)
        self._emit("; === Static FORTRAN-66 Storage & Constant Pool ===")
        for u in self.units:
            for sym in u.symbols.values():
                if sym.is_param:
                    continue
                lbl = self._sym_label(u, sym.name)
                vals = sym.init_vals if sym.init_vals is not None else [0] * sym.total_words
                self._emit(f"{lbl}:")
                for i in range(0, len(vals), 9):
                    self._emit(".WORD " + ", ".join(str(x) for x in vals[i : i + 9]))

        for raw_w, lbl in self.const_pool.items():
            self._emit(f"{lbl}: .WORD {raw_w}")
        for ttd_lbl in self.ttd_slots:
            self._emit(f"{ttd_lbl}: .WORD 0, 0, 0, 0, 0, 0, 0, 0, 0")

        lines = self.code
        peep = packed = 0
        if self.opt:
            lines, peep = _peephole_optimize(lines)
            lines, packed = _pack_dual_parcels(lines)

        asm_text = "\n".join(lines) + "\n"
        assembled = assemble(asm_text, origin=self.origin)
        return TriTranCompileResult(
            assembly=asm_text,
            assembled=assembled,
            formats=self.all_formats,
            write_records=self.write_records,
            vector_loops_emitted=self.vector_loops_emitted,
            peephole_edits=peep,
            parcels_packed=packed,
        )

    def _lbl(self, unit: UnitInfo, num: int) -> str:
        return f"_L_{unit.name}_{num}"

    def _compile_unit(self, unit: UnitInfo, cards: List[CardLine]) -> None:
        self._emit(f"_F_{unit.name}:")
        self._emit("PUSH S7")
        self._emit("ADD S7, S8, #0")
        self._emit("S2A A7, S7")

        # Active DO loops: stack of (end_label_num, var_sym, limit_sym_lbl, step_sym_lbl, top_lbl)
        do_stack: List[Tuple[int, FSymbol, str, str, str]] = []

        for card in cards:
            if card.label is not None:
                self._emit(f"{self._lbl(unit, card.label)}:")
            self._compile_stmt(unit, card, do_stack)

            # Close any DO loops terminating at this statement label (bottom-tested per FORTRAN-66!)
            while card.label is not None and do_stack and do_stack[-1][0] == card.label:
                _, v_sym, lim_lbl, stp_lbl, top_lbl = do_stack.pop()
                self._load_scalar(unit, v_sym, "S1")
                self._emit(f"LD S2, {stp_lbl}(A0)")
                self._emit("ADD S1, S1, S2")
                self._store_scalar(unit, v_sym, "S1")
                self._emit(f"LD S2, {lim_lbl}(A0)")
                self._emit("CMP S1, S2")
                self._emit(f"BN {top_lbl}")
                self._emit(f"BZ {top_lbl}")

        # Default return
        if unit.kind == "FUNCTION":
            self._load_scalar(unit, unit.symbols[unit.name], "S1")
        else:
            self._emit("ADD S1, S0, #0")
        self._emit_epilogue(len(unit.params))

    def _emit_epilogue(self, n_params: int) -> None:
        self._emit("ADD S8, S7, #0")
        self._emit("POP S7")
        self._emit("S2A A7, S7")
        self._emit(f"RET #{n_params}")

    def _emit_sym_addr(self, unit: UnitInfo, sym: FSymbol, areg: str = "A1") -> None:
        """Load 18-trit physical address of symbol into `areg` (dereferencing pass-by-ref params!)."""
        if sym.is_param:
            n_params = len(unit.params)
            off = 2 + (n_params - 1 - sym.param_idx)
            self._emit(f"LDA {areg}, {off}(A7)")
        else:
            self._emit(f"LEA {areg}, {self._sym_label(unit, sym.name)}(A0)")

    def _load_scalar(self, unit: UnitInfo, sym: FSymbol, sreg: str = "S1") -> None:
        if sym.is_param:
            self._emit_sym_addr(unit, sym, "A1")
            self._emit(f"LD {sreg}, 0(A1)")
        else:
            self._emit(f"LD {sreg}, {self._sym_label(unit, sym.name)}(A0)")

    def _store_scalar(self, unit: UnitInfo, sym: FSymbol, sreg: str = "S1") -> None:
        if sym.is_param:
            self._emit_sym_addr(unit, sym, "A1")
            self._emit(f"ST {sreg}, 0(A1)")
        else:
            self._emit(f"ST {sreg}, {self._sym_label(unit, sym.name)}(A0)")

    def _compile_stmt(
        self,
        unit: UnitInfo,
        card: CardLine,
        do_stack: List[Tuple[int, FSymbol, str, str, str]],
    ) -> None:
        txt = card.text.strip()
        up = txt.upper()

        if up == "CONTINUE":
            return

        if up == "RETURN":
            if unit.kind == "FUNCTION":
                self._load_scalar(unit, unit.symbols[unit.name], "S1")
            self._emit_epilogue(len(unit.params))
            return

        m_stop = re.match(r"^STOP(?:\s+(\d+))?$", up)
        if m_stop:
            code = int(m_stop.group(1)) if m_stop.group(1) else 0
            self._load_const("S1", code)
            self._emit("HLT")
            return

        # GO TO n
        m_goto = re.match(r"^GO\s*TO\s+(\d+)$", up)
        if m_goto:
            self._emit(f"JMP {self._lbl(unit, int(m_goto.group(1)))}(A0)")
            return

        # DO n I = e1, e2 [, e3]
        m_do = re.match(r"^DO\s+(\d+)\s+([A-Z][A-Z0-9_]*)\s*=\s*(.+)$", up)
        if m_do:
            end_lbl = int(m_do.group(1))
            v_sym = self._get_or_create_sym(unit, m_do.group(2))
            parts = _split_top_commas(m_do.group(3))
            if len(parts) not in (2, 3):
                raise TriTranError(f"Line {card.lineno}: Malformed DO loop {txt!r}")
            # Hidden static limit & step words
            lim_name = f"__DOLIM_{self._lbl_seq}"
            stp_name = f"__DOSTP_{self._lbl_seq}"
            self._lbl_seq += 1
            lim_sym = FSymbol(lim_name, "INTEGER")
            stp_sym = FSymbol(stp_name, "INTEGER")
            unit.symbols[lim_name] = lim_sym
            unit.symbols[stp_name] = stp_sym

            self._compile_expr(unit, parse_fexpr(parts[0]))
            self._store_scalar(unit, v_sym, "S1")
            self._compile_expr(unit, parse_fexpr(parts[1]))
            self._store_scalar(unit, lim_sym, "S1")
            if len(parts) == 3:
                self._compile_expr(unit, parse_fexpr(parts[2]))
            else:
                self._emit("ADD S1, S0, #1")
            self._store_scalar(unit, stp_sym, "S1")

            top_lbl = self._new_lbl("DO")
            self._emit(f"{top_lbl}:")
            do_stack.append(
                (end_lbl, v_sym, self._sym_label(unit, lim_name), self._sym_label(unit, stp_name), top_lbl)
            )
            return

        # Arithmetic IF (E) L1, L2, L3  OR  Logical IF (E) stmt
        if up.startswith("IF") and "(" in txt:
            # Find matching closing paren of IF(...)
            p0 = txt.index("(")
            depth = 0
            p1 = p0
            for idx in range(p0, len(txt)):
                if txt[idx] == "(":
                    depth += 1
                elif txt[idx] == ")":
                    depth -= 1
                    if depth == 0:
                        p1 = idx
                        break
            cond_str = txt[p0 + 1 : p1]
            tail = txt[p1 + 1 :].strip()

            m_arith = re.match(r"^(\d+)\s*,\s*(\d+)\s*,\s*(\d+)$", tail)
            if m_arith:
                # One-instruction ternary arithmetic IF via hardware B3!
                l1, l2, l3 = int(m_arith.group(1)), int(m_arith.group(2)), int(m_arith.group(3))
                et = self._compile_expr(unit, parse_fexpr(cond_str))
                # Rev. 1.5 conformance: REAL uses FCM S1, S0 so UNDECIDED maps cleanly to L2!
                self._emit("FCM S1, S0" if et == "REAL" else "TST S1")
                tbl = self._new_lbl("AIF")
                self._emit(f"B3 {tbl}")
                self._emit(f"JMP {self._lbl(unit, l1)}(A0)")
                self._emit(f"{tbl}: JMP {self._lbl(unit, l2)}(A0)")
                self._emit(f"JMP {self._lbl(unit, l3)}(A0)")
                return
            else:
                # Logical IF: obeys `tail` ONLY when condition is .TRUE. (+1)!
                et = self._compile_expr(unit, parse_fexpr(cond_str))
                self._emit("FCM S1, S0" if et == "REAL" else "TST S1")
                skip = self._new_lbl("LIF")
                self._emit(f"BN {skip}")
                self._emit(f"BZ {skip}")
                self._compile_stmt(unit, CardLine(None, tail, card.lineno), do_stack)
                self._emit(f"{skip}:")
                return

        # CALL SUB(args...) — including T-Fabric intrinsics TGEMM and TFFT3!
        m_call = re.match(r"^CALL\s+([A-Z][A-Z0-9_]*)\s*(?:\((.*)\))?$", txt, re.IGNORECASE)
        if m_call:
            sub_name = m_call.group(1).upper()
            arg_strs = _split_top_commas(m_call.group(2) or "")
            if sub_name == "TGEMM":
                self._compile_tgemm(unit, arg_strs, card.lineno)
                return
            if sub_name == "TFFT3":
                self._compile_tfft3(unit, arg_strs, card.lineno)
                return
            self._compile_user_call(unit, sub_name, arg_strs, card.lineno)
            return

        # WRITE (unit, fmt) list   OR   PRINT fmt, list
        m_wr = re.match(r"^WRITE\s*\(\s*[^,]+\s*,\s*(\d+)\s*\)\s*(.*)$", txt, re.IGNORECASE)
        m_pr = re.match(r"^PRINT\s+(\d+)\s*(?:,\s*(.*))?$", txt, re.IGNORECASE)
        if m_wr or m_pr:
            fmt_num = int((m_wr or m_pr).group(1))
            items_str = ((m_wr or m_pr).group(2) or "").strip()
            self._compile_write(unit, fmt_num, items_str, card.lineno)
            return

        # 70V Vector Array-Section Assignment: C(1:N) = A(1:N) + B(1:N)  or  FMA(A(1:N), B(1:N))
        m_vec = re.match(
            r"^([A-Z][A-Z0-9_]*)\s*\(\s*([^:]+)\s*:\s*([^)]+)\s*\)\s*=\s*(.+)$",
            up,
        )
        if m_vec:
            self._compile_vector_assign(unit, m_vec.group(1), m_vec.group(2), m_vec.group(3), m_vec.group(4), card.lineno)
            return

        # Scalar or Array Element Assignment: LHS = RHS
        if "=" in txt:
            lhs_str, rhs_str = txt.split("=", 1)
            lhs_expr = parse_fexpr(lhs_str.strip())
            rhs_expr = parse_fexpr(rhs_str.strip())

            if isinstance(lhs_expr, FVar):
                sym = self._get_or_create_sym(unit, lhs_expr.name)
                rt = self._compile_expr(unit, rhs_expr)
                self._coerce(rt, sym.typ)
                self._store_scalar(unit, sym, "S1")
                return
            if isinstance(lhs_expr, FRef):
                sym = self._get_or_create_sym(unit, lhs_expr.name)
                self._compile_elem_addr(unit, sym, lhs_expr.args, card.lineno)
                self._emit("PUSH S1")
                rt = self._compile_expr(unit, rhs_expr)
                self._coerce(rt, sym.typ)
                self._emit("POP S2")
                self._emit("S2A A2, S2")
                self._emit("ST S1, 0(A2)")
                return

        raise TriTranError(f"Line {card.lineno}: Unrecognized TRI-TRAN 70V statement: {txt!r}")

    def _compile_vector_assign(
        self, unit: UnitInfo, dst_name: str, lo_s: str, hi_s: str, rhs_s: str, lineno: int
    ) -> None:
        """Emit strip-mined C-Complex VL<=27 vector loop for `C(lo:hi) = A(lo:hi) op B(lo:hi)`."""
        dst_sym = self._get_or_create_sym(unit, dst_name)
        m_fma = re.match(
            r"^FMA\s*\(\s*([A-Z][A-Z0-9_]*)\s*\([^)]+\)\s*,\s*([A-Z][A-Z0-9_]*)\s*\([^)]+\)\s*\)$",
            rhs_s.strip(),
        )
        m_bin = re.match(
            r"^([A-Z][A-Z0-9_]*)\s*\([^)]+\)\s*(\+|-|\*|\.AND\.|\.OR\.)\s*([A-Z][A-Z0-9_]*)\s*\([^)]+\)$",
            rhs_s.strip(),
        )
        if m_fma:
            a_sym = self._get_or_create_sym(unit, m_fma.group(1))
            b_sym = self._get_or_create_sym(unit, m_fma.group(2))
            vop = "FMA"
        elif m_bin:
            a_sym = self._get_or_create_sym(unit, m_bin.group(1))
            vop = m_bin.group(2)
            b_sym = self._get_or_create_sym(unit, m_bin.group(3))
        else:
            raise TriTranError(f"Line {lineno}: Unsupported 70V vector RHS: {rhs_s!r}")

        self.vector_loops_emitted += 1
        # Compute 0-based start offset (lo - 1) in S5 and length (hi - lo + 1) in S6
        self._compile_expr(unit, parse_fexpr(lo_s))
        self._emit("PUSH S1")
        self._compile_expr(unit, parse_fexpr(hi_s))
        self._emit("POP S5")          # S5 = 1-based lo
        self._emit("SUB S6, S1, S5")
        self._emit("ADD S6, S6, #1")  # S6 = remaining length
        self._emit("SUB S5, S5, #1")  # S5 = 0-based offset

        self._emit_sym_addr(unit, dst_sym, "A3")
        self._emit("S2A A1, S5")
        self._emit("AADD A3, A3, A1")
        self._emit_sym_addr(unit, a_sym, "A4")
        self._emit("AADD A4, A4, A1")
        self._emit_sym_addr(unit, b_sym, "A5")
        self._emit("AADD A5, A5, A1")

        l_loop = self._new_lbl("VL")
        l_ok = self._new_lbl("VOK")
        l_end = self._new_lbl("VE")

        self._emit(f"{l_loop}:")
        self._emit("CMP S6, S0")
        self._emit(f"BN {l_end}")
        self._emit(f"BZ {l_end}")
        self._emit("ADD S4, S6, #0")
        self._emit("CMP S4, #27")
        self._emit(f"BN {l_ok}")
        self._emit(f"BZ {l_ok}")
        self._emit("ADD S4, S0, #27")
        self._emit(f"{l_ok}:")
        self._emit("VSVL S4, #0")
        self._emit("VLD V1, 0(A4)")
        self._emit("VLD V2, 0(A5)")
        if vop == "FMA":
            self._emit("VLD V3, 0(A3)")
            self._emit("VFMA V3, V1, V2")
            self._emit("VST V3, 0(A3)")
        elif vop in ("+", "-", "*"):
            tmode = 0 if dst_sym.typ == "REAL" else (-1 if dst_sym.typ == "DECIBAN" else 1)
            vmn = {"+": "VADD", "-": "VSUB", "*": "VMUL"}[vop]
            self._emit(f"{vmn} V3, V1, V2, #{tmode}")
            self._emit("VST V3, 0(A3)")
        else:
            kmode = -1 if vop == ".AND." else 1
            self._emit(f"VKLEEN V3, V1, V2, #{kmode}")
            self._emit("VST V3, 0(A3)")

        self._emit("S2A A1, S4")
        self._emit("AADD A3, A3, A1")
        self._emit("AADD A4, A4, A1")
        self._emit("AADD A5, A5, A1")
        self._emit("SUB S6, S6, S4")
        self._emit(f"BR {l_loop}")
        self._emit(f"{l_end}:")

    def _compile_tgemm(self, unit: UnitInfo, args: List[str], lineno: int) -> None:
        """CALL TGEMM(A, B, C, M, N, K, EA, EB, EC, ISTAT) -> 9-word TTD + TEXEC A6, S0 + TWAIT."""
        if len(args) != 10:
            raise TriTranError(f"Line {lineno}: CALL TGEMM expects 10 args (A,B,C,M,N,K,EA,EB,EC,ISTAT)")
        ttd_lbl = f"_FTTD{len(self.ttd_slots)}"
        self.ttd_slots.append(ttd_lbl)
        a_sym = self._get_or_create_sym(unit, args[0])
        b_sym = self._get_or_create_sym(unit, args[1])
        c_sym = self._get_or_create_sym(unit, args[2])
        stat_sym = self._get_or_create_sym(unit, args[9])

        # Push M, N, K, EA, EB, EC
        for idx in range(3, 9):
            self._compile_expr(unit, parse_fexpr(args[idx]))
            self._emit("PUSH S1")

        self._emit(f"LEA A6, {ttd_lbl}(A0)")
        w0 = TileTaskDescriptor(mode=MODE_TBFP9_GEMM, zgate=1, abft=1, micro_gran=1).pack_9_words()[0]
        self._load_const("S2", w0)
        self._emit("ST S2, 0(A6)")

        # Pack W5 = EA<<30 + EB<<24 + EC<<18
        self._emit("POP S3")
        self._emit("SHF S3, S3, #18")
        self._emit("POP S2")
        self._emit("SHF S2, S2, #24")
        self._emit("ADD S3, S3, S2")
        self._emit("POP S2")
        self._emit("SHF S2, S2, #30")
        self._emit("ADD S3, S3, S2")
        self._emit("ST S3, 5(A6)")

        # Pack W4 = M<<24 + N<<12 + K
        self._emit("POP S3")
        self._emit("POP S2")
        self._emit("SHF S2, S2, #12")
        self._emit("ADD S3, S3, S2")
        self._emit("POP S2")
        self._emit("SHF S2, S2, #24")
        self._emit("ADD S3, S3, S2")
        self._emit("ST S3, 4(A6)")

        for w_idx, sym in ((1, a_sym), (2, b_sym), (3, c_sym)):
            self._emit_sym_addr(unit, sym, "A1")
            self._emit("A2S S2, A1")
            self._emit("SHF S2, S2, #18")
            self._emit(f"ST S2, {w_idx}(A6)")

        self._emit("TEXEC A6, S0")
        self._emit("TWAIT S0, #0")
        self._emit("SEL S1, S0, #8")
        self._store_scalar(unit, stat_sym, "S1")

    def _compile_tfft3(self, unit: UnitInfo, args: List[str], lineno: int) -> None:
        """CALL TFFT3(XIN, XOUT, IPOW3, EIN, ISTAT) -> Radix-3 TBFP-9C FFT on 27 T-Tiles!"""
        if len(args) != 5:
            raise TriTranError(f"Line {lineno}: CALL TFFT3 expects 5 args (XIN,XOUT,IPOW3,EIN,ISTAT)")
        ttd_lbl = f"_FTTD{len(self.ttd_slots)}"
        self.ttd_slots.append(ttd_lbl)
        in_sym = self._get_or_create_sym(unit, args[0])
        out_sym = self._get_or_create_sym(unit, args[1])
        stat_sym = self._get_or_create_sym(unit, args[4])

        self._compile_expr(unit, parse_fexpr(args[2]))  # IPOW3 (p)
        self._emit("PUSH S1")
        self._compile_expr(unit, parse_fexpr(args[3]))  # EIN
        self._emit("PUSH S1")

        self._emit(f"LEA A6, {ttd_lbl}(A0)")
        w0 = TileTaskDescriptor(mode=MODE_RADIX3_FFT, zgate=1, abft=1, micro_gran=1).pack_9_words()[0]
        self._load_const("S2", w0)
        self._emit("ST S2, 0(A6)")

        # W5 = EIN << 30
        self._emit("POP S2")
        self._emit("SHF S2, S2, #30")
        self._emit("ST S2, 5(A6)")
        # W4: M(12t) packs p(6t)|q(6t) -> p << (24 + 6) = p << 30!
        self._emit("POP S2")
        self._emit("SHF S2, S2, #30")
        self._emit("ST S2, 4(A6)")

        self._emit_sym_addr(unit, in_sym, "A1")
        self._emit("A2S S2, A1")
        self._emit("SHF S2, S2, #18")
        self._emit("ST S2, 1(A6)")

        self._emit_sym_addr(unit, out_sym, "A1")
        self._emit("A2S S2, A1")
        self._emit("SHF S2, S2, #18")
        self._emit("ST S2, 3(A6)")

        self._emit("TEXEC A6, S0")
        self._emit("TWAIT S0, #0")
        self._emit("SEL S1, S0, #8")
        self._store_scalar(unit, stat_sym, "S1")

    def _compile_user_call(self, unit: UnitInfo, name: str, arg_strs: List[str], lineno: int) -> None:
        if name not in self.unit_by_name:
            raise TriTranError(f"Line {lineno}: Undefined subprogram {name!r}")
        for a_str in arg_strs:
            e = parse_fexpr(a_str)
            if isinstance(e, FVar):
                sym = self._get_or_create_sym(unit, e.name)
                self._emit_sym_addr(unit, sym, "A1")
                self._emit("A2S S1, A1")
                self._emit("PUSH S1")
            elif isinstance(e, FRef) and e.name in unit.symbols and unit.symbols[e.name].dims:
                self._compile_elem_addr(unit, unit.symbols[e.name], e.args, lineno)
                self._emit("PUSH S1")
            else:
                # Pass temporary static slot address by reference (classic FORTRAN!)
                tmp_name = f"__ARG_{self._lbl_seq}"
                self._lbl_seq += 1
                rt = self._compile_expr(unit, e)
                tmp_sym = FSymbol(tmp_name, rt)
                unit.symbols[tmp_name] = tmp_sym
                self._store_scalar(unit, tmp_sym, "S1")
                self._emit_sym_addr(unit, tmp_sym, "A1")
                self._emit("A2S S1, A1")
                self._emit("PUSH S1")
        self._emit(f"CALL A0, _F_{name}")

    def _compile_write(self, unit: UnitInfo, fmt_num: int, items_str: str, lineno: int) -> None:
        if fmt_num not in unit.formats:
            raise TriTranError(f"Line {lineno}: Undefined FORMAT label {fmt_num}")
        items = _split_top_commas(items_str) if items_str else []
        arg_types: List[str] = []
        # Push each evaluated output item onto stack right-to-left (or left-to-right; SVC reads from SP!)
        for it in items:
            t = self._compile_expr(unit, parse_fexpr(it))
            arg_types.append(t)
            self._emit("PUSH S1")
        tag = self._write_tag_seq
        self._write_tag_seq += 1
        self.write_records[tag] = (fmt_num, arg_types)
        self._emit(f"SVC #{tag}")

    def _compile_elem_addr(
        self, unit: UnitInfo, sym: FSymbol, subs: List[object], lineno: int
    ) -> None:
        """Compute 1-based column-major element address into S1."""
        if len(subs) == 1:
            self._compile_expr(unit, subs[0])
            self._emit("SUB S1, S1, #1")
        elif len(subs) == 2:
            if len(sym.dims) < 1:
                raise TriTranError(f"Line {lineno}: 2D subscript on 1D/scalar {sym.name}")
            rows = sym.dims[0]
            # offset = (i - 1) + (j - 1) * rows
            self._compile_expr(unit, subs[1])
            self._emit("SUB S1, S1, #1")
            self._load_const("S2", rows)
            self._emit("MUL S1, S1, S2")
            self._emit("PUSH S1")
            self._compile_expr(unit, subs[0])
            self._emit("SUB S1, S1, #1")
            self._emit("POP S2")
            self._emit("ADD S1, S1, S2")
        else:
            raise TriTranError(f"Line {lineno}: Only 1D/2D arrays supported for {sym.name}")
        self._emit("S2A A2, S1")
        self._emit_sym_addr(unit, sym, "A1")
        self._emit("AADD A1, A1, A2")
        self._emit("A2S S1, A1")

    def _coerce(self, got: str, want: str) -> None:
        if got == want:
            return
        if got == "INTEGER" and want == "REAL":
            self._emit("FLT S1, S1")
        elif got == "REAL" and want == "INTEGER":
            self._emit("FIX S1, S1")

    def _compile_expr(self, unit: UnitInfo, expr: object) -> str:
        if isinstance(expr, FLit):
            if expr.typ == "INTEGER":
                self._load_const("S1", wrap(int(expr.val), WORD_TRITS))
            elif expr.typ == "REAL":
                self._load_const("S1", float_to_tfloat(float(expr.val), TF36))
            elif expr.typ == "LOGICAL":
                self._emit(f"ADD S1, S0, #{int(expr.val)}")
            elif expr.typ == "DECIBAN":
                self._load_const("S1", db_to_ticks(float(expr.val)))
            return expr.typ

        if isinstance(expr, FVar):
            sym = self._get_or_create_sym(unit, expr.name)
            self._load_scalar(unit, sym, "S1")
            return sym.typ

        if isinstance(expr, FRef):
            name = expr.name.upper()
            # 1. Is it a declared array in this unit?
            if name in unit.symbols and unit.symbols[name].dims:
                sym = unit.symbols[name]
                self._compile_elem_addr(unit, sym, expr.args, 0)
                self._emit("S2A A1, S1")
                self._emit("LD S1, 0(A1)")
                return sym.typ
            # 2. Built-in FORTRAN-66 & 70V intrinsics
            return self._compile_intrinsic_or_func(unit, name, expr.args)

        if isinstance(expr, FUnary):
            t = self._compile_expr(unit, expr.operand)
            if expr.op == "-":
                self._emit("FNEG S1, S1" if t == "REAL" else "NEG S1, S1")
                return t
            if expr.op == ".NOT.":
                self._emit("NEG S1, S1")
                return "LOGICAL"

        if isinstance(expr, FBin):
            if expr.op == "**":
                return self._compile_power(unit, expr.left, expr.right)
            lt = self._compile_expr(unit, expr.left)
            self._emit("PUSH S1")
            rt = self._compile_expr(unit, expr.right)
            self._emit("POP S2")  # S2 = left, S1 = right

            if expr.op in ("+", "-", "*", "/"):
                # Mixed-mode INTEGER + REAL -> REAL per FORTRAN-66 extension
                if lt != rt and {lt, rt} == {"INTEGER", "REAL"}:
                    if lt == "INTEGER":
                        self._emit("FLT S2, S2")
                    else:
                        self._emit("FLT S1, S1")
                    lt = rt = "REAL"
                if lt == "REAL":
                    mn = {"+": "FAD", "-": "FSB", "*": "FMP", "/": "FDV"}[expr.op]
                    self._emit(f"{mn} S1, S2, S1")
                    return "REAL"
                if lt == "DECIBAN" and expr.op in ("+", "-"):
                    self._emit(("DADD" if expr.op == "+" else "DSUB") + " S1, S2, S1")
                    return "DECIBAN"
                mn = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}[expr.op]
                self._emit(f"{mn} S1, S2, S1")
                return "INTEGER"

            if expr.op in _CMP_SEL:
                if lt != rt and {lt, rt} == {"INTEGER", "REAL"}:
                    if lt == "INTEGER":
                        self._emit("FLT S2, S2")
                    else:
                        self._emit("FLT S1, S1")
                    lt = "REAL"
                self._emit("FCM S2, S1" if lt == "REAL" else "CMP S2, S1")
                self._emit(f"SEL S1, S0, #{_CMP_SEL[expr.op]}")
                return "LOGICAL"

            if expr.op == ".AND.":
                self._emit("AND S1, S2, S1")
                return "LOGICAL"
            if expr.op == ".OR.":
                self._emit("OR S1, S2, S1")
                return "LOGICAL"
            if expr.op == ".EQV.":
                self._emit("EQV S1, S2, S1")
                return "LOGICAL"
            if expr.op == ".NEQV.":
                self._emit("EQV S1, S2, S1")
                self._emit("NEG S1, S1")
                return "LOGICAL"

        raise TriTranError(f"Unhandled FORTRAN expression {expr}")

    def _compile_power(self, unit: UnitInfo, base_e: object, exp_e: object) -> str:
        bt = self._compile_expr(unit, base_e)
        self._emit("PUSH S1")
        et = self._compile_expr(unit, exp_e)
        self._emit("POP S2")  # S2 = base, S1 = exp
        if bt == "INTEGER" and et == "INTEGER":
            self._emit("SVC #110")  # exact int**int
            return "INTEGER"
        if bt == "INTEGER":
            self._emit("FLT S2, S2")
        if et == "INTEGER":
            self._emit("FLT S1, S1")
        self._emit("SVC #111")  # exact TF36 real**real
        return "REAL"

    def _compile_intrinsic_or_func(self, unit: UnitInfo, name: str, args: List[object]) -> str:
        if name in _MATH_SVCS:
            svc_id, arg_t, ret_t = _MATH_SVCS[name]
            gt = self._compile_expr(unit, args[0])
            if gt != arg_t:
                raise TriTranError(f"Intrinsic {name} expects {arg_t}, got {gt}")
            self._emit(f"SVC #{svc_id}")
            return ret_t
        if name in ("ABS", "IABS"):
            gt = self._compile_expr(unit, args[0])
            self._emit("FABS S1, S1" if gt == "REAL" else "ABS S1, S1")
            return "INTEGER" if name == "IABS" else gt
        if name == "FLOAT":
            self._compile_expr(unit, args[0])
            self._emit("FLT S1, S1")
            return "REAL"
        if name in ("IFIX", "INT", "NINT"):
            self._compile_expr(unit, args[0])
            self._emit("FIX S1, S1")
            return "INTEGER"
        if name == "MOD":
            self._compile_expr(unit, args[0])
            self._emit("PUSH S1")
            self._compile_expr(unit, args[1])
            self._emit("POP S2")
            self._emit("MOD S1, S2, S1")
            return "INTEGER"
        if name == "FMA":
            self._compile_expr(unit, args[0])
            self._emit("PUSH S1")
            self._compile_expr(unit, args[1])
            self._emit("PUSH S1")
            self._compile_expr(unit, args[2])
            self._emit("POP S2")
            self._emit("POP S3")
            self._emit("FMA S3, S2, S1")
            self._emit("ADD S1, S3, #0")
            return "REAL"
        if name == "ITRIT":
            self._compile_expr(unit, args[0])
            self._emit("PUSH S1")
            self._compile_expr(unit, args[1])
            self._emit("POP S2")
            self._emit("TRT S2, S1")
            self._emit("SEL S1, S0, #8")
            return "INTEGER"
        if name == "IXTR":
            self._compile_expr(unit, args[0])
            p = int(args[1].val)  # type: ignore[attr-defined]
            n = int(args[2].val)  # type: ignore[attr-defined]
            self._emit(f"XTR S1, S1, #{p}, #{n}")
            return "INTEGER"
        if name == "ICLOCK":
            self._emit("ALGD S1, #0")
            return "INTEGER"
        if name == "DLSE":
            self._compile_expr(unit, args[0])
            self._emit("PUSH S1")
            self._compile_expr(unit, args[1])
            self._emit("POP S2")
            self._emit("DLSE S1, S2, S1")
            return "DECIBAN"
        if name == "IWALD":
            self._compile_expr(unit, args[0])
            self._emit("PUSH S1")
            self._compile_expr(unit, args[1])
            self._emit("POP S2")
            self._emit("DWALD S2, S1, #0")
            self._emit("SEL S1, S0, #8")
            return "LOGICAL"
        if name in ("VSUM", "VMAX", "VMIN"):
            sym = self._get_or_create_sym(unit, args[0].name)  # type: ignore[attr-defined]
            self._compile_expr(unit, args[1])
            self._emit("VSVL S1, #0")
            self._emit_sym_addr(unit, sym, "A1")
            self._emit("VLD V1, 0(A1)")
            rm = {"VSUM": 0, "VMAX": 1, "VMIN": -1}[name]
            self._emit(f"VRED S1, V1, #{rm}")
            return sym.typ
        if name == "VDOT":
            # Strip-mined C-Complex hardware vector inner product (Livermore Kernel 3!)
            a_sym = self._get_or_create_sym(unit, args[0].name)  # type: ignore[attr-defined]
            b_sym = self._get_or_create_sym(unit, args[1].name)  # type: ignore[attr-defined]
            self.vector_loops_emitted += 1
            self._compile_expr(unit, args[2])  # N in S1
            self._emit("ADD S6, S1, #0")       # S6 = remaining N
            self._emit("ADD S5, S0, #0")       # S5 = TF36 accumulator (0.0)
            self._emit_sym_addr(unit, a_sym, "A4")
            self._emit_sym_addr(unit, b_sym, "A5")
            l_lp = self._new_lbl("VDL")
            l_ok = self._new_lbl("VDOK")
            l_end = self._new_lbl("VDE")
            self._emit(f"{l_lp}:")
            self._emit("CMP S6, S0")
            self._emit(f"BN {l_end}")
            self._emit(f"BZ {l_end}")
            self._emit("ADD S4, S6, #0")
            self._emit("CMP S4, #27")
            self._emit(f"BN {l_ok}")
            self._emit(f"BZ {l_ok}")
            self._emit("ADD S4, S0, #27")
            self._emit(f"{l_ok}:")
            self._emit("VSVL S4, #0")
            self._emit("VLD V1, 0(A4)")
            self._emit("VLD V2, 0(A5)")
            self._emit("VMUL V3, V1, V2, #0")
            self._emit("VRED S3, V3, #0")
            self._emit("FAD S5, S5, S3")
            self._emit("S2A A1, S4")
            self._emit("AADD A4, A4, A1")
            self._emit("AADD A5, A5, A1")
            self._emit("SUB S6, S6, S4")
            self._emit(f"BR {l_lp}")
            self._emit(f"{l_end}:")
            self._emit("ADD S1, S5, #0")
            return "REAL"

        # User FUNCTION call
        if name in self.unit_by_name:
            fn_u = self.unit_by_name[name]
            # Pass each argument expression cleanly via reference
            self._compile_user_call(unit, name, [_unparse_simple(a) for a in args], 0)
            return fn_u.ret_type

        raise TriTranError(f"Unknown function or undeclared array {name!r}")


def _unparse_simple(e: object) -> str:
    if isinstance(e, FVar):
        return e.name
    if isinstance(e, FLit):
        return str(e.val)
    if isinstance(e, FRef):
        return f"{e.name}(" + ",".join(_unparse_simple(a) for a in e.args) + ")"
    if isinstance(e, FUnary):
        return f"({e.op}{_unparse_simple(e.operand)})"
    if isinstance(e, FBin):
        return f"({_unparse_simple(e.left)}{e.op}{_unparse_simple(e.right)})"
    return "0"


# ---------------------------------------------------------------------------
# Public Compile & Run API
# ---------------------------------------------------------------------------

def compile_tritran(source: str, opt: bool = True, origin: int = 100) -> TriTranCompileResult:
    return TriTran70VCompiler(source, opt=opt, origin=origin).compile()


@dataclass
class TriTranRunResult:
    stop_code: int
    lines: List[str]
    cycles: int
    instructions: int
    istack_hits: int
    compile_result: TriTranCompileResult


def run_tritran(
    source: str,
    machine: Optional[UE1000Machine] = None,
    opt: bool = True,
    max_steps: int = 500_000,
) -> TriTranRunResult:
    """Compile TRI-TRAN 70V source, load onto UE-1000, and run with C-EXEC math & FORMAT I/O."""
    comp = compile_tritran(source, opt=opt, origin=100)
    m = machine if machine is not None else UE1000Machine()
    m.load_program(comp.assembled, cpu_id=0)
    lines_out: List[str] = []

    def _svc_handler(cpu, code: int) -> bool:
        # Exact TF36 transcendental library SVCs (101..111)
        if code in (101, 102, 103, 104, 105, 106, 107, 108):
            x = tfloat_to_float(cpu.s[1], TF36)
            if code == 101:
                y = float("nan") if x < 0 else math.sqrt(x)
            elif code == 102:
                y = math.exp(x)
            elif code == 103:
                y = float("nan") if x <= 0 else math.log(x)
            elif code == 104:
                y = float("nan") if x <= 0 else math.log10(x)
            elif code == 105:
                y = math.sin(x)
            elif code == 106:
                y = math.cos(x)
            elif code == 107:
                y = math.tanh(x)
            else:
                y = math.atan(x)
            cpu.s[1] = float_to_tfloat(y, TF36)
            return True
        if code == 110:  # int ** int
            cpu.s[1] = wrap(int(cpu.s[2] ** cpu.s[1]), WORD_TRITS)
            return True
        if code == 111:  # TF36 ** TF36
            b = tfloat_to_float(cpu.s[2], TF36)
            e = tfloat_to_float(cpu.s[1], TF36)
            cpu.s[1] = float_to_tfloat(b**e, TF36)
            return True

        # Formatted WRITE / PRINT records (SVC #200+)
        if code in comp.write_records:
            fmt_num, arg_types = comp.write_records[code]
            n_args = len(arg_types)
            raw_vals: List[int] = []
            for _ in range(n_args):
                v = cpu._mem_read(cpu.s[8])
                cpu.s[8] = wrap(cpu.s[8] + 1, WORD_TRITS)
                raw_vals.append(v)
            raw_vals.reverse()  # pushed left-to-right -> pop reversed!
            paired = list(zip(arg_types, raw_vals))
            spec = comp.formats[fmt_num]
            lines_out.append(format_fortran_record(spec, paired))
            return True
        return False

    cpu = m.cpu0
    cpu.svc_handler = _svc_handler
    cpu.run(max_steps=max_steps)
    return TriTranRunResult(
        stop_code=cpu.s[1],
        lines=lines_out,
        cycles=cpu.cycles,
        instructions=cpu.instructions,
        istack_hits=cpu.istack_hits,
        compile_result=comp,
    )
