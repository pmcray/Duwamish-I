"""TRI-INTERCAL — Unsanctioned 1972–73 West Seattle Grad-Student Parody on Duwamish-I.

Found on unlabeled 9-track Reel `SS-GAG-72` in the OBB Unseen Level -9 Maintenance Bay
(disguised as `DIAG-CORE-TEST` to evade Fred Brooks's anti-CAMEL budget audit).

Faithful balanced-ternary adaptation of Don Woods & James M. Lyon's INTERCAL-72
(Princeton University, May 26, 1972) with optional post-commissioning Clark (Dec 1973)
`COME FROM` extension (`--1974` / `PLEASE ALLOW 1974`):

  - Balanced-Ternary Politeness Gate:
      * Too rude (< 1/5 `PLEASE`)  -> SPL-079 (Politeness Trit = -1)
      * Too obsequious (> 1/3)     -> SPL-099 (Politeness Trit = +1)
  - Variables:
      * `.n` (Spot)     : 18-trit Setun half-word (T18, ±193,710,244)
      * `:n` (Two-spot) : 36-trit UE-1000 full word (T36)
      * `,n` (Tail)     : 18-trit 1D array (`SUB` indexed)
      * `;n` (Hybrid)   : 36-trit 1D array
      * `#n` (Mesh)     : constant 0..65535
  - Native Balanced-Ternary Operators (executed on UE-1000 CComplexCPU):
      * Binary Mingle (`¢` or `$`) : interleaves two 18-trit spots into one 36-trit two-spot
      * Binary Select (`~`)        : balanced signed trit selection (+1 keep, -1 negate, 0 drop)
      * Unary `&` / `V` / `?`      : 1-trit right-rotated Kleene AND / OR / EQV on hardware ALU
      * Unary `¬` (or `@`)         : balanced-ternary tritwise Kleene NOT (`NEG`)
      * Grouping                   : sparks `' ... '` and rabbit-ears `" ... "`
  - Statements:
      * `<-` calculate, `DO (lbl) NEXT`, `FORGET`, `RESUME`, `STASH`, `RETRIEVE`,
        `IGNORE`, `REMEMBER`, `ABSTAIN FROM`, `REINSTATE`, `WRITE IN`, `READ OUT`
        (authentic Roman numerals with `ANTI-` for negative trits, or 6-trit OBB-729
        Turing Tape text on 1D tail arrays!), `GIVE UP`, and gated `COME FROM`!
  - Built-in Woodworth Syslib (callable via `DO (1000)..(1050) NEXT`, Princeton style):
      * (1000) `.3 <- .1 + .2` | (1010) `.3 <- .1 - .2` | (1020) `.3 <- .1 * .2`
      * (1030) `.3 <- .1 / .2` | (1040) `:3 <- :1 + :2` | (1050) `.2 <- -.1`
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .machine import UE1000Machine
from .ternary import (
    HALF_TRITS,
    WORD_TRITS,
    char_to_obb729,
    int_to_trits,
    obb729_to_char,
    trits_to_int,
    wrap,
)
from .triad36 import AssembledProgram, assemble


class TriIntercalError(Exception):
    """Authentic SPL-xxx Princeton/Woodworth TRI-INTERCAL diagnostic error."""


# ---------------------------------------------------------------------------
# Pre-Assembled TRIAD-36 Hardware Kernel (so every gag burns real UE-TECL cycles!)
# ---------------------------------------------------------------------------

_INTERCAL_KERNEL_ASM = """
; === TRI-INTERCAL Unsanctioned UE-TECL Hardware Kernel (Reel SS-GAG-72) ===
_IC_AND:
    AND S1, S1, S2
    HLT
_IC_OR:
    OR S1, S1, S2
    HLT
_IC_EQV:
    EQV S1, S1, S2
    HLT
_IC_NEG:
    NEG S1, S1
    HLT
_IC_ADD:
    ADD S1, S1, S2
    HLT
_IC_SUB:
    SUB S1, S1, S2
    HLT
_IC_MUL:
    MUL S1, S1, S2
    HLT
_IC_DIV:
    DIV S1, S1, S2
    HLT
"""

_INTERCAL_KERNEL_ORIGIN = 5000


# ---------------------------------------------------------------------------
# Balanced-Ternary Turing Tape & Roman Numeral Codecs
# ---------------------------------------------------------------------------

def reverse_6_trits(v: int) -> int:
    """Reverse the 6 balanced trits of an OBB-729 tryte."""
    ts = int_to_trits(wrap(v, 6), 6)
    return trits_to_int(list(reversed(ts)))


def encode_turing_tape(text: str) -> List[int]:
    """Encode a string into positive mesh constants (0..728) for `READ OUT ,1` Turing Tape."""
    state = 0
    out: List[int] = []
    for ch in text:
        target_rev = reverse_6_trits(char_to_obb729(ch))
        delta = wrap(state - target_rev, 6)
        mesh_val = delta if delta >= 0 else (delta + 729)
        out.append(mesh_val)
        state = wrap(mesh_val, 6)
    return out


def to_roman(n: int) -> str:
    """Authentic INTERCAL Roman numerals (with `ANTI-` for negative balanced-ternary values!)."""
    if n == 0:
        return "_"
    if n < 0:
        return "ANTI-" + to_roman(-n)
    if n >= 4_000_000:
        return f"MEGA({n})"
    table = [
        (1_000_000, "m"), (900_000, "cm"), (500_000, "d"), (400_000, "cd"),
        (100_000, "c"), (90_000, "xc"), (50_000, "l"), (40_000, "xl"),
        (10_000, "x"), (9_000, "Mx"), (5_000, "v"), (4_000, "Mv"),
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    rem = n
    out: List[str] = []
    for val, sym in table:
        while rem >= val:
            out.append(sym)
            rem -= val
    return "".join(out)


_WORD_DIGITS = {
    "OH": 0, "ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4,
    "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9,
}


def parse_intercal_input(line: str) -> int:
    """Parse classic INTERCAL English-word number input (plus NEGATIVE/MINUS)."""
    words = line.strip().upper().split()
    if not words:
        raise TriIntercalError("SPL-562 I DO NOT UNDERSTAND EMPTY INPUT")
    neg = False
    if words[0] in ("NEGATIVE", "MINUS", "ANTI"):
        neg = True
        words = words[1:]
    if not words:
        raise TriIntercalError("SPL-562 MISSING DIGITS AFTER NEGATIVE")
    val = 0
    for w in words:
        if w not in _WORD_DIGITS:
            raise TriIntercalError(f"SPL-579 WHAT BASE AND/OR LANGUAGE IS {w!r}?")
        val = val * 10 + _WORD_DIGITS[w]
    return -val if neg else val


# ---------------------------------------------------------------------------
# Balanced-Ternary Mingle (`¢`) & Select (`~`)
# ---------------------------------------------------------------------------

def ternary_mingle(a: int, b: int) -> int:
    """Interleave 18-trit `a` (odd positions) and 18-trit `b` (even positions) into T36."""
    ta = int_to_trits(wrap(a, HALF_TRITS), HALF_TRITS)
    tb = int_to_trits(wrap(b, HALF_TRITS), HALF_TRITS)
    out = [0] * WORD_TRITS
    for k in range(HALF_TRITS):
        out[2 * k] = tb[k]
        out[2 * k + 1] = ta[k]
    return trits_to_int(out)


def ternary_select(a: int, b: int, width: int = WORD_TRITS) -> int:
    """Select trits of `a` where `b` is non-zero (+1 keeps trit, -1 negates trit)."""
    ta = int_to_trits(wrap(a, width), width)
    tb = int_to_trits(wrap(b, width), width)
    picked: List[int] = []
    for k in range(width):
        if tb[k] != 0:
            picked.append(ta[k] * tb[k])
    return trits_to_int(picked) if picked else 0


def rotate_right_1_trit(v: int, width: int) -> int:
    """Circularly rotate `width`-trit balanced word right by 1 trit (LSB -> MSB)."""
    ts = int_to_trits(wrap(v, width), width)
    rot = ts[1:] + [ts[0]]
    return trits_to_int(rot)


# ---------------------------------------------------------------------------
# Statement & Expression Parser
# ---------------------------------------------------------------------------

@dataclass
class IntercalStmt:
    idx: int
    label: Optional[int]
    polite: bool
    abstained_init: bool
    prob: int                     # 1..100 (%chance)
    gerund: str                   # CALCULATING, NEXTING, FORGETTING, RESUMING, ...
    body: str
    raw: str


_LINE_STMT_RE = re.compile(
    r"^(?:\((\d+)\)\s*)?(PLEASE\s+DO|PLEASE|DO)\b(?:\s*(NOT|N'T)\b)?(?:\s*%(\d+))?(.*)$",
    re.IGNORECASE,
)


def _classify_gerund(body: str) -> str:
    u = body.strip().upper()
    if u == "GIVE UP":
        return "GIVING UP"
    if u.startswith("ALLOW 1974"):
        return "PRAGMA"
    if re.match(r"^\(\d+\)\s+NEXT\b", u):
        return "NEXTING"
    if u.startswith("COME FROM"):
        return "COMING FROM"
    if u.startswith("FORGET"):
        return "FORGETTING"
    if u.startswith("RESUME"):
        return "RESUMING"
    if u.startswith("STASH"):
        return "STASHING"
    if u.startswith("RETRIEVE"):
        return "RETRIEVING"
    if u.startswith("IGNORE"):
        return "IGNORING"
    if u.startswith("REMEMBER"):
        return "REMEMBERING"
    if u.startswith("ABSTAIN"):
        return "ABSTAINING"
    if u.startswith("REINSTATE"):
        return "REINSTATING"
    if u.startswith("WRITE IN"):
        return "WRITING IN"
    if u.startswith("READ OUT"):
        return "READING OUT"
    if "<-" in u or "←" in u:
        return "CALCULATING"
    return "COMMENT"


def parse_tri_intercal(source: str, allow_1974: bool = False) -> Tuple[List[IntercalStmt], bool]:
    """Parse TRI-INTERCAL program card-by-card and enforce balanced-ternary politeness."""
    # Group continuation lines onto the preceding statement card
    cards: List[str] = []
    for raw_ln in source.splitlines():
        s = raw_ln.strip()
        if not s:
            continue
        if _LINE_STMT_RE.match(s) or not cards:
            cards.append(s)
        else:
            cards[-1] = cards[-1] + " " + s

    if not cards:
        raise TriIntercalError("SPL-000 NO VALID DO/PLEASE STATEMENTS FOUND")

    stmts: List[IntercalStmt] = []
    pragma_1974 = allow_1974

    for card in cards:
        m = _LINE_STMT_RE.match(card)
        if not m:
            raise TriIntercalError(f"SPL-000 UNDECODABLE STATEMENT CARD: {card!r}")
        lbl = int(m.group(1)) if m.group(1) else None
        ident = m.group(2).upper()
        polite = ident.startswith("PLEASE")
        abstained = m.group(3) is not None
        prob = int(m.group(4)) if m.group(4) else 100
        body = m.group(5).strip()
        gerund = _classify_gerund(body)
        if gerund == "PRAGMA":
            pragma_1974 = True
            continue
        stmts.append(
            IntercalStmt(
                idx=len(stmts),
                label=lbl,
                polite=polite,
                abstained_init=abstained,
                prob=prob,
                gerund=gerund,
                body=body,
                raw=card,
            )
        )

    # Balanced-Ternary Politeness Check (authentic Princeton 1972 rule!)
    n = len(stmts)
    p = sum(1 for s in stmts if s.polite)
    if n > 0:
        if (n < 4 and p == 0) or (n >= 4 and 5 * p < n):
            raise TriIntercalError(
                f"SPL-079 PROGRAMMER IS INSUFFICIENTLY POLITE "
                f"({p}/{n} PLEASE statements; POLITENESS TRIT = -1)"
            )
        if n >= 4 and 3 * p > n:
            raise TriIntercalError(
                f"SPL-099 PROGRAMMER IS OVERLY POLITE "
                f"({p}/{n} PLEASE statements; POLITENESS TRIT = +1)"
            )

    # Check anachronistic 1974 COME FROM statements!
    for s in stmts:
        if s.gerund == "COMING FROM" and not pragma_1974:
            raise TriIntercalError(
                "SPL-444 ANACHRONISM: R. LAWRENCE CLARK'S `COME FROM` DATAMATION PAPER "
                "WAS NOT PUBLISHED UNTIL DEC 1973 (PASS --1974 OR `PLEASE ALLOW 1974`)"
            )

    return stmts, pragma_1974


# ---------------------------------------------------------------------------
# TRI-INTERCAL Runtime & Expression Evaluator (Hardware-Backed on UE-1000)
# ---------------------------------------------------------------------------

@dataclass
class TriIntercalResult:
    output: List[str]
    cycles_wasted: int
    hw_ops: int
    stmts_executed: int
    politeness_ratio: float
    spots: Dict[int, int]
    twospots: Dict[int, int]


class TriIntercalVM:
    """Executes TRI-INTERCAL programs, burning genuine UE-1000 UE-TECL cycles."""

    def __init__(
        self,
        stmts: List[IntercalStmt],
        machine: Optional[UE1000Machine] = None,
        inputs: Optional[Sequence[str]] = None,
        max_steps: int = 50_000,
    ) -> None:
        self.stmts = stmts
        self.machine = machine if machine is not None else UE1000Machine()
        self.kernels: AssembledProgram = assemble(_INTERCAL_KERNEL_ASM, origin=_INTERCAL_KERNEL_ORIGIN)
        self.machine.load_program(self.kernels, cpu_id=0)

        self.inputs = list(inputs) if inputs is not None else []
        self.in_pos = 0
        self.max_steps = max_steps
        self.output: List[str] = []
        self.hw_ops = 0
        self.stmts_executed = 0

        # Variables & INTERCAL Stash / Ignore state
        self.spots: Dict[int, int] = {}
        self.twospots: Dict[int, int] = {}
        self.tails: Dict[int, List[int]] = {}
        self.hybrids: Dict[int, List[int]] = {}
        self.stashes: Dict[Tuple[str, int], List[object]] = {}
        self.ignored: Set[Tuple[str, int]] = set()
        self.tape_state: int = 0

        # Label map, COME FROM table, abstention state & 79-deep NEXT stack
        self.label_to_idx: Dict[int, int] = {}
        self.abstained_idx: Set[int] = set()
        self.abstained_gerunds: Set[str] = set()
        self.comefrom_map: Dict[int, List[int]] = {}
        self.next_stack: List[int] = []
        self._rng = random.Random(1972)

        for s in stmts:
            if s.label is not None:
                if s.label in self.label_to_idx:
                    raise TriIntercalError(f"SPL-182 MULTIPLE LABELS ({s.label})")
                self.label_to_idx[s.label] = s.idx
            if s.abstained_init:
                self.abstained_idx.add(s.idx)
            if s.gerund == "COMING FROM":
                m = re.match(r"^COME\s+FROM\s+\((\d+)\)$", s.body.strip(), re.IGNORECASE)
                if not m:
                    raise TriIntercalError(f"SPL-000 INVALID COME FROM: {s.body!r}")
                tgt = int(m.group(1))
                self.comefrom_map.setdefault(tgt, []).append(s.idx)

    def _hw(self, entry: str, a: int, b: int = 0) -> int:
        cpu = self.machine.cpu0
        cpu.pc = self.kernels.labels[entry]
        cpu.halted = False
        cpu.s[1] = wrap(a, WORD_TRITS)
        cpu.s[2] = wrap(b, WORD_TRITS)
        cpu.run(max_steps=100)
        self.hw_ops += 1
        return cpu.s[1]

    # ------------------------------------------------------------------
    # INTERCAL Expression Evaluator (Sparks `'...'`, Rabbit-Ears `"..."`,
    # Unary `& V ? ¬` after sigil, Binary `¢/$` Mingle & `~` Select!)
    # ------------------------------------------------------------------

    def _eval_expr(self, text: str) -> Tuple[int, int]:
        """Evaluate INTERCAL expression -> (value, width_trits: 18 or 36)."""
        toks = self._lex_expr(text)
        val, width, pos = self._parse_expr_seq(toks, 0, stop=None)
        if pos != len(toks):
            raise TriIntercalError(f"SPL-017 UNEXPECTED TRAILING EXPRESSION TOKENS: {toks[pos:]}")
        return val, width

    def _lex_expr(self, text: str) -> List[Tuple[str, str]]:
        toks: List[Tuple[str, str]] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch in " \t":
                i += 1
                continue
            if ch in ("'", '"', "¢", "$", "~"):
                toks.append(("SYM", "¢" if ch == "$" else ch))
                i += 1
                continue
            if text[i : i + 3].upper() == "SUB" and (i + 3 == n or not text[i + 3].isalnum()):
                toks.append(("SUB", "SUB"))
                i += 3
                continue
            if ch in (".", ":", ",", ";", "#"):
                sig = ch
                i += 1
                # Optional unary operator(s) immediately after sigil (`&`, `V`, `?`, `⊻`, `¬`, `@`)!
                unops = ""
                while i < n and text[i] in "&V?⊻¬@":
                    unops += text[i]
                    i += 1
                j = i
                while j < n and text[j].isdigit():
                    j += 1
                if j == i:
                    raise TriIntercalError(f"SPL-017 EXPECTED NUMBER AFTER {sig}{unops!r}")
                toks.append(("ATOM", f"{sig}|{unops}|{text[i:j]}"))
                i = j
                continue
            if ch in "&V?⊻¬@":
                # Unary operator applied to a grouped spark/rabbit-ear expression (`'...'` -> `&'...'`)
                toks.append(("UNOP", ch))
                i += 1
                continue
            raise TriIntercalError(f"SPL-000 UNRECOGNIZED EXPRESSION CHAR {ch!r} IN {text!r}")
        return toks

    def _apply_unops(self, val: int, width: int, unops: str) -> int:
        res = wrap(val, width)
        for u in reversed(unops):
            if u in ("¬", "@"):
                res = wrap(self._hw("_IC_NEG", res, 0), width)
            else:
                rot = rotate_right_1_trit(res, width)
                kern = {"&": "_IC_AND", "V": "_IC_OR", "?": "_IC_EQV", "⊻": "_IC_EQV"}[u]
                res = wrap(self._hw(kern, res, rot), width)
        return res

    def _eval_atom(self, spec: str, toks: List[Tuple[str, str]], pos: int) -> Tuple[int, int, int]:
        sig, unops, num_s = spec.split("|")
        num = int(num_s)
        if sig == "#":
            if num > 65535:
                raise TriIntercalError(f"SPL-017 MESH CONSTANT #{num} TOO LARGE")
            v, w = num, HALF_TRITS
        elif sig == ".":
            v, w = self.spots.get(num, 0), HALF_TRITS
        elif sig == ":":
            v, w = self.twospots.get(num, 0), WORD_TRITS
        elif sig in (",", ";"):
            # Array subscript: `,1 SUB #k`
            if pos >= len(toks) or toks[pos][0] != "SUB":
                raise TriIntercalError(f"SPL-240 MISSING `SUB` FOR ARRAY {sig}{num}")
            idx_v, _, pos = self._parse_primary(toks, pos + 1)
            arr = self.tails.get(num) if sig == "," else self.hybrids.get(num)
            if arr is None or not (1 <= idx_v <= len(arr)):
                raise TriIntercalError(f"SPL-241 SUBSCRIPT {idx_v} OUT OF BOUNDS FOR {sig}{num}")
            v = arr[idx_v - 1]
            w = HALF_TRITS if sig == "," else WORD_TRITS
        else:
            raise TriIntercalError(f"SPL-000 INVALID ATOM {spec}")
        if unops:
            v = self._apply_unops(v, w, unops)
        return v, w, pos

    def _parse_primary(self, toks: List[Tuple[str, str]], pos: int) -> Tuple[int, int, int]:
        if pos >= len(toks):
            raise TriIntercalError("SPL-017 UNEXPECTED END OF EXPRESSION")
        kind, val = toks[pos]
        if kind == "UNOP":
            inner_v, inner_w, pos2 = self._parse_primary(toks, pos + 1)
            return self._apply_unops(inner_v, inner_w, val), inner_w, pos2
        if kind == "SYM" and val in ("'", '"'):
            close_ch = val
            pos += 1
            unops = ""
            while pos < len(toks) and toks[pos][0] == "UNOP":
                unops += toks[pos][1]
                pos += 1
            inner_v, inner_w, pos = self._parse_expr_seq(toks, pos, stop=close_ch)
            if pos >= len(toks) or toks[pos] != ("SYM", close_ch):
                raise TriIntercalError(f"SPL-017 UNMATCHED GROUPER {close_ch!r}")
            pos += 1
            if unops:
                inner_v = self._apply_unops(inner_v, inner_w, unops)
            return inner_v, inner_w, pos
        if kind == "ATOM":
            return self._eval_atom(val, toks, pos + 1)
        raise TriIntercalError(f"SPL-017 UNEXPECTED TOKEN {toks[pos]}")

    def _parse_expr_seq(
        self, toks: List[Tuple[str, str]], pos: int, stop: Optional[str]
    ) -> Tuple[int, int, int]:
        lv, lw, pos = self._parse_primary(toks, pos)
        while pos < len(toks):
            if stop is not None and toks[pos] == ("SYM", stop):
                break
            if toks[pos][0] == "SYM" and toks[pos][1] in ("¢", "~"):
                op = toks[pos][1]
                rv, rw, pos = self._parse_primary(toks, pos + 1)
                if op == "¢":
                    lv = ternary_mingle(lv, rv)
                    lw = WORD_TRITS
                    self.hw_ops += 1
                else:
                    w = WORD_TRITS if (lw == WORD_TRITS or rw == WORD_TRITS) else HALF_TRITS
                    lv = ternary_select(lv, rv, w)
                    lw = w
                    self.hw_ops += 1
                continue
            break
        return lv, lw, pos

    # ------------------------------------------------------------------
    # Built-in Woodworth Syslib Subroutines `(1000) .. (1050)`
    # ------------------------------------------------------------------

    def _try_syslib(self, lbl: int) -> bool:
        if lbl == 1000:  # .3 <- .1 + .2 (T18 on CPU0!)
            if (".", 3) not in self.ignored:
                self.spots[3] = wrap(self._hw("_IC_ADD", self.spots.get(1, 0), self.spots.get(2, 0)), HALF_TRITS)
            return True
        if lbl == 1010:  # .3 <- .1 - .2
            if (".", 3) not in self.ignored:
                self.spots[3] = wrap(self._hw("_IC_SUB", self.spots.get(1, 0), self.spots.get(2, 0)), HALF_TRITS)
            return True
        if lbl == 1020:  # .3 <- .1 * .2
            if (".", 3) not in self.ignored:
                self.spots[3] = wrap(self._hw("_IC_MUL", self.spots.get(1, 0), self.spots.get(2, 0)), HALF_TRITS)
            return True
        if lbl == 1030:  # .3 <- .1 / .2
            b = self.spots.get(2, 0)
            if b == 0:
                raise TriIntercalError("SPL-275 DIVISION BY ZERO IN SYSLIB (1030)")
            if (".", 3) not in self.ignored:
                self.spots[3] = wrap(self._hw("_IC_DIV", self.spots.get(1, 0), b), HALF_TRITS)
            return True
        if lbl == 1040:  # :3 <- :1 + :2 (T36!)
            if (":", 3) not in self.ignored:
                self.twospots[3] = wrap(self._hw("_IC_ADD", self.twospots.get(1, 0), self.twospots.get(2, 0)), WORD_TRITS)
            return True
        if lbl == 1050:  # .2 <- -.1
            if (".", 2) not in self.ignored:
                self.spots[2] = wrap(self._hw("_IC_NEG", self.spots.get(1, 0), 0), HALF_TRITS)
            return True
        return False

    # ------------------------------------------------------------------
    # Main Execution Loop
    # ------------------------------------------------------------------

    def _is_abstained(self, s: IntercalStmt) -> bool:
        return (s.idx in self.abstained_idx) or (s.gerund in self.abstained_gerunds)

    def _parse_var_list(self, text: str) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        for seg in text.split("+"):
            m = re.match(r"^\s*([.:,;])(\d+)\s*$", seg)
            if not m:
                raise TriIntercalError(f"SPL-000 INVALID VARIABLE {seg!r}")
            out.append((m.group(1), int(m.group(2))))
        return out

    def run(self) -> TriIntercalResult:
        pc = 0
        steps = 0
        while 0 <= pc < len(self.stmts):
            steps += 1
            if steps > self.max_steps:
                raise TriIntercalError("SPL-774 MAXIMUM STUDENT CYCLE QUOTA EXCEEDED (CALL BROOKS)")

            s = self.stmts[pc]
            next_pc = pc + 1
            executed = False

            if not self._is_abstained(s) and s.gerund not in ("COMMENT", "COMING FROM"):
                if s.prob == 100 or self._rng.randint(1, 100) <= s.prob:
                    executed = True
                    self.stmts_executed += 1
                    jump_pc = self._exec_stmt(s, next_pc)
                    if jump_pc == -1:  # GIVE UP!
                        break
                    if jump_pc is not None:
                        next_pc = jump_pc

            # Post-statement COME FROM hook (active only when --1974 / PLEASE ALLOW 1974)!
            if s.label is not None and s.label in self.comefrom_map and s.gerund != "COMING FROM":
                active_cfs = [
                    cf_idx for cf_idx in self.comefrom_map[s.label]
                    if not self._is_abstained(self.stmts[cf_idx])
                ]
                if len(active_cfs) > 1:
                    raise TriIntercalError(f"SPL-555 MULTIPLE ACTIVE COME FROM ({s.label})")
                if len(active_cfs) == 1:
                    next_pc = active_cfs[0] + 1

            pc = next_pc

        p_cnt = sum(1 for st in self.stmts if st.polite)
        ratio = p_cnt / len(self.stmts) if self.stmts else 0.0
        return TriIntercalResult(
            output=list(self.output),
            cycles_wasted=self.machine.cpu0.cycles,
            hw_ops=self.hw_ops,
            stmts_executed=self.stmts_executed,
            politeness_ratio=ratio,
            spots=dict(self.spots),
            twospots=dict(self.twospots),
        )

    def _exec_stmt(self, s: IntercalStmt, fallthrough_pc: int) -> Optional[int]:
        g = s.gerund
        body = s.body.strip()

        if g == "GIVING UP":
            return -1

        if g == "CALCULATING":
            sep = "<-" if "<-" in body else "←"
            lhs, rhs = body.split(sep, 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            # Dimension array: `,1 <- #16` or `;1 <- #16`
            m_dim = re.match(r"^([,;])(\d+)$", lhs)
            if m_dim:
                sig, num = m_dim.group(1), int(m_dim.group(2))
                if (sig, num) in self.ignored:
                    return None
                dim, _ = self._eval_expr(rhs)
                if dim < 1 or dim > 4096:
                    raise TriIntercalError(f"SPL-240 INVALID ARRAY DIMENSION {dim}")
                if sig == ",":
                    self.tails[num] = [0] * dim
                else:
                    self.hybrids[num] = [0] * dim
                return None

            # Element assignment: `,1 SUB #k <- expr`
            m_sub = re.match(r"^([,;])(\d+)\s+SUB\s+(.+)$", lhs, re.IGNORECASE)
            if m_sub:
                sig, num = m_sub.group(1), int(m_sub.group(2))
                if (sig, num) in self.ignored:
                    return None
                idx, _ = self._eval_expr(m_sub.group(3))
                val, _ = self._eval_expr(rhs)
                arr = self.tails.get(num) if sig == "," else self.hybrids.get(num)
                if arr is None or not (1 <= idx <= len(arr)):
                    raise TriIntercalError(f"SPL-241 SUBSCRIPT {idx} OUT OF BOUNDS FOR {sig}{num}")
                arr[idx - 1] = wrap(val, HALF_TRITS if sig == "," else WORD_TRITS)
                return None

            # Scalar spot / two-spot assignment
            m_var = re.match(r"^([.:])(\d+)$", lhs)
            if not m_var:
                raise TriIntercalError(f"SPL-000 INVALID ASSIGNMENT LHS: {lhs!r}")
            sig, num = m_var.group(1), int(m_var.group(2))
            if (sig, num) in self.ignored:
                return None
            val, _ = self._eval_expr(rhs)
            if sig == ".":
                self.spots[num] = wrap(val, HALF_TRITS)
            else:
                self.twospots[num] = wrap(val, WORD_TRITS)
            return None

        if g == "NEXTING":
            m = re.match(r"^\((\d+)\)\s+NEXT$", body, re.IGNORECASE)
            if not m:
                raise TriIntercalError(f"SPL-000 INVALID NEXT: {body!r}")
            tgt = int(m.group(1))
            if self._try_syslib(tgt):
                return None
            if tgt not in self.label_to_idx:
                raise TriIntercalError(f"SPL-129 PROGRAM HAS GOTTEN LOST: NO LABEL ({tgt})")
            if len(self.next_stack) >= 79:
                raise TriIntercalError("SPL-123 PROGRAM HAS DISAPPEARED INTO THE BLACK LAGOON (NEXT STACK > 79)")
            self.next_stack.append(fallthrough_pc)
            return self.label_to_idx[tgt]

        if g == "FORGETTING":
            cnt, _ = self._eval_expr(body[6:].strip())
            for _ in range(max(0, cnt)):
                if self.next_stack:
                    self.next_stack.pop()
            return None

        if g == "RESUMING":
            cnt, _ = self._eval_expr(body[6:].strip())
            if cnt <= 0:
                raise TriIntercalError("SPL-621 ERROR TYPE 621: RESUME #0")
            if cnt > len(self.next_stack):
                raise TriIntercalError("SPL-632 THE NEXT STACK HAS BEEN EXHAUSTED")
            target = 0
            for _ in range(cnt):
                target = self.next_stack.pop()
            return target

        if g == "STASHING":
            for sig, num in self._parse_var_list(body[5:].strip()):
                if sig == ".":
                    cur: object = self.spots.get(num, 0)
                elif sig == ":":
                    cur = self.twospots.get(num, 0)
                elif sig == ",":
                    cur = list(self.tails.get(num, []))
                else:
                    cur = list(self.hybrids.get(num, []))
                self.stashes.setdefault((sig, num), []).append(cur)
            return None

        if g == "RETRIEVING":
            for sig, num in self._parse_var_list(body[8:].strip()):
                if (sig, num) in self.ignored:
                    continue
                stk = self.stashes.get((sig, num))
                if not stk:
                    raise TriIntercalError(f"SPL-436 THROWING OUT NOTHING FROM STASH {sig}{num}")
                val = stk.pop()
                if sig == ".":
                    self.spots[num] = int(val)  # type: ignore[arg-type]
                elif sig == ":":
                    self.twospots[num] = int(val)  # type: ignore[arg-type]
                elif sig == ",":
                    self.tails[num] = list(val)  # type: ignore[arg-type]
                else:
                    self.hybrids[num] = list(val)  # type: ignore[arg-type]
            return None

        if g == "IGNORING":
            for item in self._parse_var_list(body[6:].strip()):
                self.ignored.add(item)
            return None

        if g == "REMEMBERING":
            for item in self._parse_var_list(body[8:].strip()):
                self.ignored.discard(item)
            return None

        if g in ("ABSTAINING", "REINSTATING"):
            rest = body[len("ABSTAIN FROM"):].strip() if g == "ABSTAINING" else body[len("REINSTATE"):].strip()
            for part in rest.split("+"):
                p = part.strip().upper()
                m_lbl = re.match(r"^\((\d+)\)$", p)
                if m_lbl:
                    lbl = int(m_lbl.group(1))
                    if lbl in self.label_to_idx:
                        if g == "ABSTAINING":
                            self.abstained_idx.add(self.label_to_idx[lbl])
                        else:
                            self.abstained_idx.discard(self.label_to_idx[lbl])
                else:
                    if g == "ABSTAINING":
                        self.abstained_gerunds.add(p)
                    else:
                        self.abstained_gerunds.discard(p)
            return None

        if g == "WRITING IN":
            for sig, num in self._parse_var_list(body[8:].strip()):
                if self.in_pos >= len(self.inputs):
                    raise TriIntercalError("SPL-562 INSUFFICIENT DATA FOR WRITE IN")
                raw_in = self.inputs[self.in_pos]
                self.in_pos += 1
                if (sig, num) in self.ignored:
                    continue
                val = parse_intercal_input(raw_in)
                if sig == ".":
                    self.spots[num] = wrap(val, HALF_TRITS)
                elif sig == ":":
                    self.twospots[num] = wrap(val, WORD_TRITS)
            return None

        if g == "READING OUT":
            rest = body[8:].strip()
            for seg in rest.split("+"):
                seg_s = seg.strip()
                m_arr = re.match(r"^([,;])(\d+)$", seg_s)
                if m_arr:
                    sig, num = m_arr.group(1), int(m_arr.group(2))
                    arr = self.tails.get(num, []) if sig == "," else self.hybrids.get(num, [])
                    chars: List[str] = []
                    for x in arr:
                        tr = wrap(self.tape_state - x, 6)
                        chars.append(obb729_to_char(reverse_6_trits(tr)))
                        self.tape_state = wrap(x, 6)
                    self.output.append("".join(chars))
                else:
                    val, _ = self._eval_expr(seg_s)
                    self.output.append(to_roman(val))
            return None

        return None


def run_tri_intercal(
    source: str,
    machine: Optional[UE1000Machine] = None,
    inputs: Optional[Sequence[str]] = None,
    allow_1974: bool = False,
) -> TriIntercalResult:
    stmts, _ = parse_tri_intercal(source, allow_1974=allow_1974)
    return TriIntercalVM(stmts, machine=machine, inputs=inputs).run()
