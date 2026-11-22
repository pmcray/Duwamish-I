"""APL\\36 — Kenneth Iverson's Balanced-Ternary Array & Tensor Language for Duwamish-I.

Authentic IBM APL\\360 (1966–1972) rank-polymorphic array notation extended natively for
Duwamish-I's 36-trit balanced-ternary hardware:
  - Tier 1 (C-Complex): Vector arithmetic (`+`, `-`, `×`), reductions (`+/`, `⌈/`, `⌊/`),
    Kleene logic (`∧`, `∨`, `≡`), vector dot products (`+.×`), and Wald sequential tests
    (`⎕WALD`) compile/dispatch to pre-loaded native `TRIAD-36` 27-lane vector micro-kernels
    (`VSVL`, `VLD`, `VADD`, `VSUB`, `VMUL`, `VRED`, `VKLEEN`, `VST`, `DWALD`) executing on
    `machine.cpu0` in exact `TF36` unbiased balanced-ternary arithmetic!
  - Tier 2 (T-Fabric): Matrix inner products (`A +.× B`) dispatch directly to the 27-tile
    systolic T-Fabric (`TBFP-9` default `⎕TMODE←1`, `TBFP-3` triad `⎕TMODE←2`, or
    Rosenblatt-Kelvanov `TB1` pure-trit associative `⎕TMODE←3` / `⎕TB1`), and `⎕FFT` runs
    Oliver-Good Radix-3 `TBFP-9C` transforms with native ternary ABFT!
  - Full APL\\360 Semantics:
      * Strict right-to-left evaluation, high-minus `¯` negative literals, strand vectors
      * Monadic & dyadic primitives (`+ - × ÷ * ⍟ ∣ ⌈ ⌊ ! ○ ? ∼ ⍴ ⍳ , ⌽ ⊖ ⍉ ⍋ ⍒ ⌹ ↑ ↓ ∈ / \\ ⊤ ⊥ ⍕`)
      * Higher-order operators: Reduction `f/` `f⌿`, Scan `f\\` `f⍀`, Inner Product `f.g`,
        Outer Product `∘.f` (including 5-char Kleene truth tables `¯1 0 1 ∘.∧ ¯1 0 1`!)
      * Multi-axis indexing `A[i;j]`, elided slices `A[;2]`, indexed assignment `A[i;j]←v`
      * Del-editor user functions (`∇ Z ← A FN B ; L`), labels, recursion, and `→` branching
      * Native `OBB-729` tryte strings (`⎕OBB`), UE Heptavintimal (`⎕H`), Good Decibans (`⎕DB`)
      * Interactive terminal REPL (`python3 -m ue1000 apl --repl`) + system commands `)VARS )FNS )HW`
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .machine import UE1000Machine
from .ternary import (
    WORD_TRITS,
    char_to_obb729,
    int_to_hept,
    int_to_trits,
    obb729_to_char,
    sign_trit,
    trits_to_int,
    wrap,
)
from .tfloat import (
    TF36,
    db_to_ticks,
    deciban_add,
    deciban_exp_float,
    deciban_lse,
    deciban_wald,
    float_to_tfloat,
    tfloat_to_float,
    ticks_to_db,
)
from .triad36 import AssembledProgram, assemble


class APL36Error(Exception):
    """APL\\36 SYNTAX, DOMAIN, RANK, LENGTH, INDEX, or VALUE ERROR."""


class _APLBranch(Exception):
    """Internal flow-control exception for `→ expr` inside user functions."""

    def __init__(self, target: Optional[int]) -> None:
        self.target = target  # None -> fall through, 0 -> return, >0 -> 1-based line


# ---------------------------------------------------------------------------
# Pre-Assembled Native TRIAD-36 C-Complex 27-Lane Vector Micro-Kernels
# ---------------------------------------------------------------------------
# Calling convention (User T-DRAM buffers at +50000, +60000, +70000):
#   S1 = length N, A4 = base A, A5 = base B (or threshold), A6 = base C
#   Returns scalar result (for VRED / VDOT) in S1; vector result written to base C.

_APL_KERNEL_ASM = """
; === APL\\36 Pre-Loaded Native TRIAD-36 C-Complex Vector Micro-Kernels ===
_APL_VADD:
    ADD S6, S1, #0
_VA_LP:
    CMP S6, S0
    BN _VA_DN
    BZ _VA_DN
    ADD S4, S6, #0
    CMP S4, #27
    BN _VA_OK
    BZ _VA_OK
    ADD S4, S0, #27
_VA_OK:
    VSVL S4, #0
    VLD V1, 0(A4)
    VLD V2, 0(A5)
    VADD V3, V1, V2, #0
    VST V3, 0(A6)
    S2A A1, S4
    AADD A4, A4, A1 | AADD A5, A5, A1
    AADD A6, A6, A1 | SUB S6, S6, S4
    BR _VA_LP
_VA_DN:
    HLT

_APL_VSUB:
    ADD S6, S1, #0
_VS_LP:
    CMP S6, S0
    BN _VS_DN
    BZ _VS_DN
    ADD S4, S6, #0
    CMP S4, #27
    BN _VS_OK
    BZ _VS_OK
    ADD S4, S0, #27
_VS_OK:
    VSVL S4, #0
    VLD V1, 0(A4)
    VLD V2, 0(A5)
    VSUB V3, V1, V2, #0
    VST V3, 0(A6)
    S2A A1, S4
    AADD A4, A4, A1 | AADD A5, A5, A1
    AADD A6, A6, A1 | SUB S6, S6, S4
    BR _VS_LP
_VS_DN:
    HLT

_APL_VMUL:
    ADD S6, S1, #0
_VM_LP:
    CMP S6, S0
    BN _VM_DN
    BZ _VM_DN
    ADD S4, S6, #0
    CMP S4, #27
    BN _VM_OK
    BZ _VM_OK
    ADD S4, S0, #27
_VM_OK:
    VSVL S4, #0
    VLD V1, 0(A4)
    VLD V2, 0(A5)
    VMUL V3, V1, V2, #0
    VST V3, 0(A6)
    S2A A1, S4
    AADD A4, A4, A1 | AADD A5, A5, A1
    AADD A6, A6, A1 | SUB S6, S6, S4
    BR _VM_LP
_VM_DN:
    HLT

_APL_VKAND:
    ADD S6, S1, #0
_VKA_LP:
    CMP S6, S0
    BN _VKA_DN
    BZ _VKA_DN
    ADD S4, S6, #0
    CMP S4, #27
    BN _VKA_OK
    BZ _VKA_OK
    ADD S4, S0, #27
_VKA_OK:
    VSVL S4, #0
    VLD V1, 0(A4)
    VLD V2, 0(A5)
    VKLEEN V3, V1, V2, #-1
    VST V3, 0(A6)
    S2A A1, S4
    AADD A4, A4, A1 | AADD A5, A5, A1
    AADD A6, A6, A1 | SUB S6, S6, S4
    BR _VKA_LP
_VKA_DN:
    HLT

_APL_VKOR:
    ADD S6, S1, #0
_VKO_LP:
    CMP S6, S0
    BN _VKO_DN
    BZ _VKO_DN
    ADD S4, S6, #0
    CMP S4, #27
    BN _VKO_OK
    BZ _VKO_OK
    ADD S4, S0, #27
_VKO_OK:
    VSVL S4, #0
    VLD V1, 0(A4)
    VLD V2, 0(A5)
    VKLEEN V3, V1, V2, #1
    VST V3, 0(A6)
    S2A A1, S4
    AADD A4, A4, A1 | AADD A5, A5, A1
    AADD A6, A6, A1 | SUB S6, S6, S4
    BR _VKO_LP
_VKO_DN:
    HLT

_APL_VKEQV:
    ADD S6, S1, #0
_VKE_LP:
    CMP S6, S0
    BN _VKE_DN
    BZ _VKE_DN
    ADD S4, S6, #0
    CMP S4, #27
    BN _VKE_OK
    BZ _VKE_OK
    ADD S4, S0, #27
_VKE_OK:
    VSVL S4, #0
    VLD V1, 0(A4)
    VLD V2, 0(A5)
    VKLEEN V3, V1, V2, #0
    VST V3, 0(A6)
    S2A A1, S4
    AADD A4, A4, A1 | AADD A5, A5, A1
    AADD A6, A6, A1 | SUB S6, S6, S4
    BR _VKE_LP
_VKE_DN:
    HLT

_APL_VRSUM:
    ADD S6, S1, #0 | ADD S5, S0, #0
_VRS_LP:
    CMP S6, S0
    BN _VRS_DN
    BZ _VRS_DN
    ADD S4, S6, #0
    CMP S4, #27
    BN _VRS_OK
    BZ _VRS_OK
    ADD S4, S0, #27
_VRS_OK:
    VSVL S4, #0
    VLD V1, 0(A4)
    VRED S3, V1, #0
    FAD S5, S5, S3
    S2A A1, S4
    AADD A4, A4, A1 | SUB S6, S6, S4
    BR _VRS_LP
_VRS_DN:
    ADD S1, S5, #0
    HLT

_APL_VRMAX:
    ADD S6, S1, #0
    LD S5, 0(A4)
_VRX_LP:
    CMP S6, S0
    BN _VRX_DN
    BZ _VRX_DN
    ADD S4, S6, #0
    CMP S4, #27
    BN _VRX_OK
    BZ _VRX_OK
    ADD S4, S0, #27
_VRX_OK:
    VSVL S4, #0
    VLD V1, 0(A4)
    VRED S3, V1, #1
    FCM S3, S5
    BN _VRX_SK
    BZ _VRX_SK
    ADD S5, S3, #0
_VRX_SK:
    S2A A1, S4
    AADD A4, A4, A1 | SUB S6, S6, S4
    BR _VRX_LP
_VRX_DN:
    ADD S1, S5, #0
    HLT

_APL_VRMIN:
    ADD S6, S1, #0
    LD S5, 0(A4)
_VRN_LP:
    CMP S6, S0
    BN _VRN_DN
    BZ _VRN_DN
    ADD S4, S6, #0
    CMP S4, #27
    BN _VRN_OK
    BZ _VRN_OK
    ADD S4, S0, #27
_VRN_OK:
    VSVL S4, #0
    VLD V1, 0(A4)
    VRED S3, V1, #-1
    FCM S3, S5
    BP _VRN_SK
    BZ _VRN_SK
    ADD S5, S3, #0
_VRN_SK:
    S2A A1, S4
    AADD A4, A4, A1 | SUB S6, S6, S4
    BR _VRN_LP
_VRN_DN:
    ADD S1, S5, #0
    HLT

_APL_VDOT:
    ADD S6, S1, #0 | ADD S5, S0, #0
_VDT_LP:
    CMP S6, S0
    BN _VDT_DN
    BZ _VDT_DN
    ADD S4, S6, #0
    CMP S4, #27
    BN _VDT_OK
    BZ _VDT_OK
    ADD S4, S0, #27
_VDT_OK:
    VSVL S4, #0
    VLD V1, 0(A4)
    VLD V2, 0(A5)
    VMUL V3, V1, V2, #0
    VRED S3, V3, #0
    FAD S5, S5, S3
    S2A A1, S4
    AADD A4, A4, A1 | AADD A5, A5, A1
    SUB S6, S6, S4
    BR _VDT_LP
_VDT_DN:
    ADD S1, S5, #0
    HLT

_APL_VWALD:
    ; S1 = count N, S2 = threshold ticks, A4 = in deciban ticks, A6 = out trits
    ADD S6, S1, #0
_VWD_LP:
    CMP S6, S0
    BN _VWD_DN
    BZ _VWD_DN
    LD S3, 0(A4)
    DWALD S3, S2, #0
    SEL S4, S0, #8
    ST S4, 0(A6)
    AADD A4, A4, #1 | AADD A6, A6, #1
    SUB S6, S6, #1
    BR _VWD_LP
_VWD_DN:
    HLT
"""

_APL_KERNEL_ORIGIN = 1000
_APL_BUF_A = 50000
_APL_BUF_B = 60000
_APL_BUF_C = 70000


# ---------------------------------------------------------------------------
# APLArray Value Representation & Authentic Formatting
# ---------------------------------------------------------------------------

@dataclass
class APLArray:
    """Rank-polymorphic APL\\36 array (rank 0 = scalar, 1 = vector, 2+ = tensor)."""

    data: np.ndarray        # np.int64 (exact T36 int / OBB trytes) or np.float64 (TF36 / NaN=UNDECIDED)
    is_char: bool = False

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(d) for d in self.data.shape)

    @property
    def rank(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return int(self.data.size)

    def is_scalar(self) -> bool:
        return self.data.ndim == 0

    def as_float_arr(self) -> np.ndarray:
        if self.is_char:
            raise APL36Error("DOMAIN ERROR: character array used in numeric operation")
        return self.data.astype(np.float64, copy=False)

    def as_int_arr(self) -> np.ndarray:
        if self.is_char:
            raise APL36Error("DOMAIN ERROR: character array used where integer expected")
        if self.data.dtype == np.int64:
            return self.data
        f = self.data
        if np.any(np.isnan(f)) or np.any(np.abs(f - np.round(f)) > 1e-9):
            raise APL36Error("DOMAIN ERROR: non-integer value")
        return np.round(f).astype(np.int64)

    def as_scalar_float(self) -> float:
        if self.size != 1:
            raise APL36Error("LENGTH ERROR: expected scalar")
        return float(self.as_float_arr().reshape(()))

    def as_scalar_int(self) -> int:
        if self.size != 1:
            raise APL36Error("LENGTH ERROR: expected scalar integer")
        return int(self.as_int_arr().reshape(()))

    def to_python(self):
        if self.is_char:
            flat = [obb729_to_char(int(x)) for x in self.data.ravel()]
            if self.rank == 0:
                return flat[0]
            if self.rank == 1:
                return "".join(flat)
            return np.array(flat).reshape(self.shape).tolist()
        return self.data.tolist()


def apl_scalar(v: Union[int, float]) -> APLArray:
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return APLArray(np.array(int(v), dtype=np.int64))
    fv = float(v)
    if not math.isnan(fv) and not math.isinf(fv) and abs(fv) < 1e15 and fv == math.trunc(fv):
        return APLArray(np.array(int(fv), dtype=np.int64))
    return APLArray(np.array(fv, dtype=np.float64))


def apl_num(arr: np.ndarray) -> APLArray:
    a = np.asarray(arr)
    if a.dtype == np.int64:
        return APLArray(a)
    f = a.astype(np.float64, copy=False)
    if f.size > 0 and not np.any(np.isnan(f)) and not np.any(np.isinf(f)):
        if np.all(np.abs(f) < 1e15) and np.all(f == np.trunc(f)):
            return APLArray(f.astype(np.int64))
    return APLArray(f)


def apl_chars(text: str) -> APLArray:
    trytes = [char_to_obb729(ch) for ch in text]
    if len(trytes) == 1:
        return APLArray(np.array(trytes[0], dtype=np.int64), is_char=True)
    return APLArray(np.array(trytes, dtype=np.int64), is_char=True)


def _fmt_num(val: float, pp: int = 6) -> str:
    if math.isnan(val):
        return "?"
    if math.isinf(val):
        return "∞" if val > 0 else "¯∞"
    if abs(val) < 1e-12:
        return "0"
    neg = val < 0
    av = abs(val)
    if av < 1e15 and abs(av - round(av)) < 1e-10:
        s = str(int(round(av)))
    else:
        s = f"{av:.{pp}g}"
        if "e" in s:
            m, e = s.split("e")
            ei = int(e)
            s = f"{m}E{'¯' if ei < 0 else ''}{abs(ei)}"
    return ("¯" + s) if neg else s


def format_apl(arr: APLArray, pp: int = 6) -> str:
    """Format an APLArray in authentic APL\\360 / APL\\36 output layout."""
    if arr.size == 0:
        return ""
    if arr.is_char:
        if arr.rank == 0:
            return obb729_to_char(int(arr.data))
        if arr.rank == 1:
            return "".join(obb729_to_char(int(x)) for x in arr.data)
        if arr.rank == 2:
            return "\n".join(
                "".join(obb729_to_char(int(x)) for x in row) for row in arr.data
            )
        planes = arr.data.reshape((-1, arr.shape[-2], arr.shape[-1]))
        return "\n\n".join(
            "\n".join("".join(obb729_to_char(int(x)) for x in row) for row in plane)
            for plane in planes
        )

    if arr.rank == 0:
        return _fmt_num(float(arr.data), pp)
    if arr.rank == 1:
        return " ".join(_fmt_num(float(x), pp) for x in arr.data)
    if arr.rank == 2:
        grid = [[_fmt_num(float(x), pp) for x in row] for row in arr.data]
        n_cols = arr.shape[1]
        widths = [max(len(grid[r][c]) for r in range(arr.shape[0])) for c in range(n_cols)]
        return "\n".join(
            " ".join(grid[r][c].rjust(widths[c]) for c in range(n_cols))
            for r in range(arr.shape[0])
        )
    planes = arr.data.reshape((-1, arr.shape[-2], arr.shape[-1]))
    rendered_planes = [format_apl(APLArray(p), pp) for p in planes]
    return "\n\n".join(rendered_planes)


# ---------------------------------------------------------------------------
# Lexer & Right-to-Left AST/Token Evaluator
# ---------------------------------------------------------------------------

_PRIM_VERBS = set(
    "+-×÷*⍟∣⌈⌊!○?∼¬<≤=≥>≠∧∨⍲⍱≡⍴⍳,⌽⊖⍉⍋⍒⌹↑↓∈/\\⌿⍀⊤⊥⍕⍎"
)
_REDUCE_OPS = {"/", "\\", "⌿", "⍀"}


@dataclass
class ATok:
    kind: str       # NUM, STR, ID, SYS, VERB, OP, DOT, JOT, ASSIGN, BRANCH, LPAR, RPAR, LBRK, RBRK, SEMI, DIAMOND, EOF
    val: object


def tokenize_apl(line: str) -> List[ATok]:
    """Tokenize an APL\\36 line (supporting Unicode glyphs + ASCII teletype digraphs)."""
    toks: List[ATok] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch in " \t\r":
            i += 1
            continue
        # Comments: `⍝` or `NB.`
        if ch == "⍝" or (line[i : i + 3] == "NB."):
            break
        if ch == "⋄":
            toks.append(ATok("DIAMOND", "⋄"))
            i += 1
            continue
        # Strings '...' with '' escape
        if ch == "'":
            j = i + 1
            buf: List[str] = []
            while j < n:
                if line[j] == "'":
                    if j + 1 < n and line[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    break
                buf.append(line[j])
                j += 1
            if j >= n:
                raise APL36Error("SYNTAX ERROR: unterminated string literal")
            toks.append(ATok("STR", "".join(buf)))
            i = j + 1
            continue
        # ASCII digraphs for non-APL keyboards
        two = line[i : i + 2]
        if two == "<-":
            toks.append(ATok("ASSIGN", "←"))
            i += 2
            continue
        if two == "->":
            toks.append(ATok("BRANCH", "→"))
            i += 2
            continue
        if two == "<=":
            toks.append(ATok("VERB", "≤"))
            i += 2
            continue
        if two == ">=":
            toks.append(ATok("VERB", "≥"))
            i += 2
            continue
        if two == "!=":
            toks.append(ATok("VERB", "≠"))
            i += 2
            continue
        if two == "o." and (i == 0 or not line[i - 1].isalnum()):
            toks.append(ATok("JOT", "∘"))
            toks.append(ATok("DOT", "."))
            i += 2
            continue
        # Numeric literals (including high-minus `¯`, balanced-ternary `3t1T01`, Good `12.5db`)
        if ch == "¯" or ch.isdigit() or (ch == "." and i + 1 < n and line[i + 1].isdigit()):
            j = i
            neg = False
            if line[j] == "¯":
                neg = True
                j += 1
            # Balanced ternary literal: `3t1T01`
            if line[j : j + 2].lower() == "3t":
                j += 2
                k = j
                while k < n and line[k] in "10Tt_":
                    k += 1
                s = line[j:k].replace("_", "")
                if not s:
                    raise APL36Error("SYNTAX ERROR: empty 3t balanced-ternary literal")
                val_i = 0
                for c in s:
                    val_i = val_i * 3 + (-1 if c in "Tt" else int(c))
                toks.append(ATok("NUM", -val_i if neg else val_i))
                i = k
                continue
            start_d = j
            is_flt = False
            while j < n and line[j].isdigit():
                j += 1
            if j < n and line[j] == ".":
                # Only treat `.` as decimal point if followed by digit (not inner product!)
                if j + 1 < n and line[j + 1].isdigit():
                    is_flt = True
                    j += 1
                    while j < n and line[j].isdigit():
                        j += 1
            if j == start_d:
                raise APL36Error("SYNTAX ERROR: stray high-minus `¯`")
            if j < n and line[j] in "eE" and (
                (j + 1 < n and (line[j + 1].isdigit() or line[j + 1] in "+-¯"))
            ):
                is_flt = True
                j += 1
                if j < n and line[j] in "+-¯":
                    j += 1
                while j < n and line[j].isdigit():
                    j += 1
            raw_s = line[start_d:j].replace("¯", "-")
            if line[j : j + 2].lower() == "db":
                fv = float(raw_s)
                ticks = db_to_ticks(-fv if neg else fv)
                toks.append(ATok("NUM", ticks))
                i = j + 2
                continue
            if is_flt:
                fv = float(raw_s)
                toks.append(ATok("NUM", -fv if neg else fv))
            else:
                iv = int(raw_s)
                toks.append(ATok("NUM", -iv if neg else iv))
            i = j
            continue
        # System names `⎕...`
        if ch == "⎕":
            j = i + 1
            while j < n and (line[j].isalnum() or line[j] == "_"):
                j += 1
            toks.append(ATok("SYS", line[i:j].upper()))
            i = j
            continue
        # Identifiers
        if ch.isalpha() or ch in "_∆⍙":
            j = i + 1
            while j < n and (line[j].isalnum() or line[j] in "_∆⍙"):
                j += 1
            toks.append(ATok("ID", line[i:j].upper()))
            i = j
            continue
        # Single-char delimiters & primitives
        if ch == "←":
            toks.append(ATok("ASSIGN", "←"))
        elif ch == "→":
            toks.append(ATok("BRANCH", "→"))
        elif ch == "(":
            toks.append(ATok("LPAR", "("))
        elif ch == ")":
            toks.append(ATok("RPAR", ")"))
        elif ch == "[":
            toks.append(ATok("LBRK", "["))
        elif ch == "]":
            toks.append(ATok("RBRK", "]"))
        elif ch == ";":
            toks.append(ATok("SEMI", ";"))
        elif ch == ".":
            toks.append(ATok("DOT", "."))
        elif ch == "∘":
            toks.append(ATok("JOT", "∘"))
        elif ch == "~":
            toks.append(ATok("VERB", "∼"))
        elif ch == "|":
            toks.append(ATok("VERB", "∣"))
        elif ch in _PRIM_VERBS:
            toks.append(ATok("VERB", ch))
        else:
            raise APL36Error(f"SYNTAX ERROR: unrecognized symbol {ch!r}")
        i += 1
    toks.append(ATok("EOF", None))
    return toks


# Structured tokens after grouping parens, brackets, strands, and operators
@dataclass
class _NodeNum:
    arr: APLArray


@dataclass
class _NodeVar:
    name: str       # variable or system name `⎕...`


@dataclass
class _NodeGroup:
    items: List[object]


@dataclass
class _NodeIndex:
    base: object
    axes: List[Optional[List[object]]]  # None means elided axis `[;2]`


@dataclass
class _NodeVerb:
    kind: str       # "PRIM", "SYS_FN", "USER_FN", "REDUCE", "SCAN", "INNER", "OUTER"
    val: object
    axis: Optional[List[object]] = None


@dataclass
class _NodeAssign:
    pass


@dataclass
class _NodeBranch:
    pass


# ---------------------------------------------------------------------------
# User-Defined Function (`∇` Del-Editor)
# ---------------------------------------------------------------------------

@dataclass
class APLUserFunc:
    name: str
    result_var: Optional[str]
    left_arg: Optional[str]
    right_arg: Optional[str]
    locals_list: List[str]
    lines: List[str]                    # 1-indexed body lines (lines[0] is line 1)
    labels: Dict[str, int]              # label -> 1-based line number

    @property
    def arity(self) -> int:
        if self.left_arg and self.right_arg:
            return 2
        if self.right_arg:
            return 1
        return 0


def parse_del_header(header: str) -> APLUserFunc:
    """Parse authentic APL\\360 `∇` header: `[Z ←] [A] FN [B] [; L1 ; L2]`."""
    s = header.strip()
    if s.startswith("∇"):
        s = s[1:].strip()
    if s.endswith("∇"):
        s = s[:-1].strip()
    s = s.replace("<-", "←")
    parts = [p.strip() for p in s.split(";")]
    sig = parts[0]
    locals_list = [p.upper() for p in parts[1:] if p.strip()]

    result_var: Optional[str] = None
    if "←" in sig:
        lhs, rhs = sig.split("←", 1)
        result_var = lhs.strip().upper()
        sig = rhs.strip()
    words = [w.upper() for w in sig.split() if w.strip()]
    if len(words) == 1:
        fn_name, left_a, right_a = words[0], None, None
    elif len(words) == 2:
        fn_name, left_a, right_a = words[0], None, words[1]
    elif len(words) == 3:
        left_a, fn_name, right_a = words[0], words[1], words[2]
    else:
        raise APL36Error(f"SYNTAX ERROR: invalid ∇ function header {header!r}")

    return APLUserFunc(
        name=fn_name,
        result_var=result_var,
        left_arg=left_a,
        right_arg=right_a,
        locals_list=locals_list,
        lines=[],
        labels={},
    )


# ---------------------------------------------------------------------------
# System Function Names
# ---------------------------------------------------------------------------

_SYS_VERBS = {"⎕H", "⎕HEPT", "⎕OBB", "⎕DB", "⎕WALD", "⎕FFT", "⎕TB1", "⎕TRITS", "⎕UNTRITS", "⎕LSE"}
_SYS_VARS = {"⎕IO", "⎕KL", "⎕TMODE", "⎕PP", "⎕CYCLE", "⎕TOPS", "⎕U", "⎕"}


# ---------------------------------------------------------------------------
# APL\36 Hardware-Backed Session & Evaluator
# ---------------------------------------------------------------------------

class APL36Session:
    """Interactive & scriptable APL\\36 workspace backed by real UE-1000 Tiers 1 & 2."""

    def __init__(self, machine: Optional[UE1000Machine] = None) -> None:
        self.machine = machine if machine is not None else UE1000Machine()
        self.kernels: AssembledProgram = assemble(_APL_KERNEL_ASM, origin=_APL_KERNEL_ORIGIN)
        self.machine.load_program(self.kernels, cpu_id=0)

        self.vars: Dict[str, APLArray] = {}
        self.functions: Dict[str, APLUserFunc] = {}
        self._call_stack: List[Dict[str, Optional[APLArray]]] = []

        # System variables
        self.io: int = 1        # ⎕IO: index origin (1 default like APL\360, 0, or ¯1 balanced)
        self.kl: int = 0        # ⎕KL: 0=classic 0/1 comparisons (Kleene auto on ¯1), 1=strict ¯1/0/1
        self.tmode: int = 1     # ⎕TMODE: 1=TBFP-9, 2=TBFP-3, 3=TB1, 0=exact TF36 CPU
        self.pp: int = 6        # ⎕PP: print precision digits

        # Telemetry counters
        self.hw_vec_calls: int = 0
        self.hw_tfabric_calls: int = 0
        self.output_lines: List[str] = []
        self._rng = np.random.default_rng(1971)  # deterministic Byurakan seed

    # ------------------------------------------------------------------
    # Tier 1 C-Complex & Tier 2 T-Fabric Hardware Dispatch
    # ------------------------------------------------------------------

    def _run_kernel(self, entry_sym: str, n: int, a_addr: int, b_addr: int, c_addr: int, s2_val: int = 0) -> int:
        cpu = self.machine.cpu0
        cpu.pc = self.kernels.labels[entry_sym]
        cpu.halted = False
        cpu.s[1] = n
        cpu.s[2] = s2_val
        cpu.a[4] = a_addr
        cpu.a[5] = b_addr
        cpu.a[6] = c_addr
        cpu.run(max_steps=max(50_000, n * 30))
        self.hw_vec_calls += 1
        return cpu.s[1]

    def _hw_vec_binop(self, op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Execute 1D vector `+`, `-`, `×`, `∧`, `∨`, `≡` natively on C-Complex 27-lane pipes!"""
        n = int(a.size)
        is_kleene = op in ("∧", "∨", "≡")
        if is_kleene:
            ia = np.round(a).astype(np.int64)
            ib = np.round(b).astype(np.int64)
            for i in range(n):
                self.machine.mem.write(_APL_BUF_A + i, int(ia[i]), supervisor=True)
                self.machine.mem.write(_APL_BUF_B + i, int(ib[i]), supervisor=True)
            sym = {"∧": "_APL_VKAND", "∨": "_APL_VKOR", "≡": "_APL_VKEQV"}[op]
            self._run_kernel(sym, n, _APL_BUF_A, _APL_BUF_B, _APL_BUF_C)
            return np.array(
                [self.machine.mem.read(_APL_BUF_C + i, supervisor=True)[0] for i in range(n)],
                dtype=np.int64,
            )

        # TF36 exact balanced-ternary vector ALU!
        for i in range(n):
            self.machine.mem.write(_APL_BUF_A + i, float_to_tfloat(float(a[i]), TF36), supervisor=True)
            self.machine.mem.write(_APL_BUF_B + i, float_to_tfloat(float(b[i]), TF36), supervisor=True)
        sym = {"+": "_APL_VADD", "-": "_APL_VSUB", "×": "_APL_VMUL"}[op]
        self._run_kernel(sym, n, _APL_BUF_A, _APL_BUF_B, _APL_BUF_C)
        return np.array(
            [tfloat_to_float(self.machine.mem.read(_APL_BUF_C + i, supervisor=True)[0], TF36) for i in range(n)],
            dtype=np.float64,
        )

    def _hw_vec_reduce(self, op: str, a: np.ndarray) -> float:
        n = int(a.size)
        for i in range(n):
            self.machine.mem.write(_APL_BUF_A + i, float_to_tfloat(float(a[i]), TF36), supervisor=True)
        sym = {"+": "_APL_VRSUM", "⌈": "_APL_VRMAX", "⌊": "_APL_VRMIN"}[op]
        res_w = self._run_kernel(sym, n, _APL_BUF_A, _APL_BUF_B, _APL_BUF_C)
        return tfloat_to_float(res_w, TF36)

    def _hw_vec_dot(self, a: np.ndarray, b: np.ndarray) -> float:
        n = int(a.size)
        for i in range(n):
            self.machine.mem.write(_APL_BUF_A + i, float_to_tfloat(float(a[i]), TF36), supervisor=True)
            self.machine.mem.write(_APL_BUF_B + i, float_to_tfloat(float(b[i]), TF36), supervisor=True)
        res_w = self._run_kernel("_APL_VDOT", n, _APL_BUF_A, _APL_BUF_B, _APL_BUF_C)
        return tfloat_to_float(res_w, TF36)

    def _hw_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Dispatch 2D matrix multiply onto Tier 2 T-Fabric (TBFP-9 / TBFP-3 / TB1) or Tier 1 TF36."""
        if self.tmode == 1:
            out, _ = self.machine.tfabric.gemm_tbfp9(a, b, zgate=True, abft=True)
            self.hw_tfabric_calls += 1
            return out
        if self.tmode == 2:
            out, _ = self.machine.tfabric.gemm_tbfp3(a, b)
            self.hw_tfabric_calls += 1
            return out
        if self.tmode == 3:
            out, _ = self.machine.tfabric.assoc_tb1(a, b.T)
            self.hw_tfabric_calls += 1
            return out
        # ⎕TMODE = 0: Exact TF36 on C-Complex VDOT!
        m, _k = a.shape
        n = b.shape[1]
        out = np.zeros((m, n), dtype=np.float64)
        for r in range(m):
            for c in range(n):
                out[r, c] = self._hw_vec_dot(a[r, :], b[:, c])
        return out

    # ------------------------------------------------------------------
    # Variable Lookup & Scoped Binding
    # ------------------------------------------------------------------

    def _get_var(self, name: str) -> APLArray:
        if name == "⎕IO":
            return apl_scalar(self.io)
        if name == "⎕KL":
            return apl_scalar(self.kl)
        if name == "⎕TMODE":
            return apl_scalar(self.tmode)
        if name == "⎕PP":
            return apl_scalar(self.pp)
        if name == "⎕CYCLE":
            return apl_scalar(self.machine.cpu0.cycles)
        if name == "⎕TOPS":
            return apl_scalar(self.machine.tfabric.total_ops)
        if name == "⎕U":
            return APLArray(np.array(float("nan"), dtype=np.float64))
        for frame in reversed(self._call_stack):
            if name in frame:
                v = frame[name]
                if v is None:
                    raise APL36Error(f"VALUE ERROR: local variable {name} not initialized")
                return v
        if name in self.vars:
            return self.vars[name]
        raise APL36Error(f"VALUE ERROR: {name}")

    def _set_var(self, name: str, val: APLArray) -> APLArray:
        if name == "⎕":
            s = format_apl(val, self.pp)
            self.output_lines.append(s)
            return val
        if name == "⎕IO":
            v = val.as_scalar_int()
            if v not in (-1, 0, 1):
                raise APL36Error("DOMAIN ERROR: ⎕IO must be ¯1, 0, or 1")
            self.io = v
            return val
        if name == "⎕KL":
            v = val.as_scalar_int()
            if v not in (0, 1):
                raise APL36Error("DOMAIN ERROR: ⎕KL must be 0 or 1")
            self.kl = v
            return val
        if name == "⎕TMODE":
            v = val.as_scalar_int()
            if v not in (0, 1, 2, 3):
                raise APL36Error("DOMAIN ERROR: ⎕TMODE must be 0..3")
            self.tmode = v
            return val
        if name == "⎕PP":
            self.pp = max(1, min(15, val.as_scalar_int()))
            return val
        for frame in reversed(self._call_stack):
            if name in frame:
                frame[name] = val
                return val
        self.vars[name] = val
        return val

    # ------------------------------------------------------------------
    # Grouping Tokens into Structured APL Items (Parens, Brackets, Strands, Operators)
    # ------------------------------------------------------------------

    def _is_verb_node(self, node: object) -> bool:
        return isinstance(node, _NodeVerb)

    def _is_noun_node(self, node: object) -> bool:
        return isinstance(node, (_NodeNum, _NodeVar, _NodeGroup, _NodeIndex))

    def _structure_tokens(self, toks: List[ATok]) -> List[object]:
        raw_items, pos = self._parse_seq(toks, 0, stop_kinds={"EOF"})
        return self._bind_strands_and_ops(raw_items)

    def _parse_seq(
        self, toks: List[ATok], pos: int, stop_kinds: set
    ) -> Tuple[List[object], int]:
        items: List[object] = []
        while pos < len(toks) and toks[pos].kind not in stop_kinds:
            t = toks[pos]
            if t.kind == "NUM":
                items.append(_NodeNum(apl_scalar(t.val)))  # type: ignore[arg-type]
                pos += 1
            elif t.kind == "STR":
                items.append(_NodeNum(apl_chars(str(t.val))))
                pos += 1
            elif t.kind == "SYS":
                sname = str(t.val)
                if sname in _SYS_VERBS:
                    items.append(_NodeVerb("SYS_FN", sname))
                else:
                    items.append(_NodeVar(sname))
                pos += 1
            elif t.kind == "ID":
                name = str(t.val)
                if name in self.functions and self.functions[name].arity > 0:
                    items.append(_NodeVerb("USER_FN", name))
                else:
                    items.append(_NodeVar(name))
                pos += 1
            elif t.kind == "VERB":
                items.append(_NodeVerb("PRIM", str(t.val)))
                pos += 1
            elif t.kind in ("DOT", "JOT"):
                items.append(t)
                pos += 1
            elif t.kind == "ASSIGN":
                items.append(_NodeAssign())
                pos += 1
            elif t.kind == "BRANCH":
                items.append(_NodeBranch())
                pos += 1
            elif t.kind == "LPAR":
                sub, pos = self._parse_seq(toks, pos + 1, stop_kinds={"RPAR"})
                if pos >= len(toks) or toks[pos].kind != "RPAR":
                    raise APL36Error("SYNTAX ERROR: unmatched `(`")
                pos += 1
                items.append(_NodeGroup(self._bind_strands_and_ops(sub)))
            elif t.kind == "LBRK":
                if not items:
                    raise APL36Error("SYNTAX ERROR: `[` without target")
                axes: List[Optional[List[object]]] = []
                pos += 1
                while True:
                    sub, pos = self._parse_seq(toks, pos, stop_kinds={"SEMI", "RBRK"})
                    axes.append(self._bind_strands_and_ops(sub) if sub else None)
                    if pos < len(toks) and toks[pos].kind == "SEMI":
                        pos += 1
                        continue
                    if pos < len(toks) and toks[pos].kind == "RBRK":
                        pos += 1
                        break
                    raise APL36Error("SYNTAX ERROR: unmatched `[`")
                prev = items.pop()
                if isinstance(prev, _NodeVerb) and prev.val in _REDUCE_OPS:
                    if len(axes) != 1 or axes[0] is None:
                        raise APL36Error("SYNTAX ERROR: invalid axis bracket on operator")
                    prev.axis = axes[0]
                    items.append(prev)
                else:
                    items.append(_NodeIndex(prev, axes))
            else:
                raise APL36Error(f"SYNTAX ERROR: unexpected token {t}")
        return items, pos

    def _bind_strands_and_ops(self, items: List[object]) -> List[object]:
        # Pass 1: coalesce adjacent numeric literals into strand vector nouns (`1 2 3`)
        s1: List[object] = []
        i = 0
        while i < len(items):
            if isinstance(items[i], _NodeNum) and not items[i].arr.is_char and items[i].arr.is_scalar():
                run = [items[i].arr]
                j = i + 1
                while (
                    j < len(items)
                    and isinstance(items[j], _NodeNum)
                    and not items[j].arr.is_char
                    and items[j].arr.is_scalar()
                ):
                    run.append(items[j].arr)
                    j += 1
                if len(run) == 1:
                    s1.append(items[i])
                else:
                    if all(a.data.dtype == np.int64 for a in run):
                        vec = np.array([int(a.data) for a in run], dtype=np.int64)
                    else:
                        vec = np.array([float(a.data) for a in run], dtype=np.float64)
                    s1.append(_NodeNum(APLArray(vec)))
                i = j
            else:
                s1.append(items[i])
                i += 1

        # Pass 2: bind Outer Product (`∘ . g`), Inner Product (`f . g`), and Reduction/Scan (`f /`, `f \`)
        out: List[object] = []
        i = 0
        while i < len(s1):
            cur = s1[i]
            # Outer product: JOT DOT VERB
            if (
                isinstance(cur, ATok)
                and cur.kind == "JOT"
                and i + 2 < len(s1)
                and isinstance(s1[i + 1], ATok)
                and s1[i + 1].kind == "DOT"
                and isinstance(s1[i + 2], _NodeVerb)
            ):
                out.append(_NodeVerb("OUTER", s1[i + 2]))
                i += 3
                continue
            # Inner product: VERB DOT VERB
            if (
                isinstance(cur, _NodeVerb)
                and i + 2 < len(s1)
                and isinstance(s1[i + 1], ATok)
                and s1[i + 1].kind == "DOT"
                and isinstance(s1[i + 2], _NodeVerb)
            ):
                out.append(_NodeVerb("INNER", (cur, s1[i + 2])))
                i += 3
                continue
            # Reduce / Scan: VERB followed by `/`, `\`, `⌿`, `⍀` (when previous token is a verb!)
            if (
                isinstance(cur, _NodeVerb)
                and cur.kind == "PRIM"
                and cur.val in _REDUCE_OPS
                and out
                and isinstance(out[-1], _NodeVerb)
            ):
                f_verb = out.pop()
                op_ch = str(cur.val)
                kind = "REDUCE" if op_ch in ("/", "⌿") else "SCAN"
                first_ax = op_ch in ("⌿", "⍀")
                out.append(_NodeVerb(kind, (f_verb, first_ax), axis=cur.axis))
                i += 1
                continue
            out.append(cur)
            i += 1
        return out

    # ------------------------------------------------------------------
    # Right-to-Left Expression Evaluator
    # ------------------------------------------------------------------

    def _eval_noun(self, node: object) -> APLArray:
        if isinstance(node, _NodeNum):
            return node.arr
        if isinstance(node, _NodeVar):
            # Niladic user function invoked by name
            if node.name in self.functions and self.functions[node.name].arity == 0:
                res = self._call_user_func(self.functions[node.name], None, None)
                if res is None:
                    raise APL36Error(f"VALUE ERROR: function {node.name} returned no value")
                return res
            return self._get_var(node.name)
        if isinstance(node, _NodeGroup):
            res = self._eval_items(node.items)
            if res is None:
                raise APL36Error("VALUE ERROR: empty parenthesized expression")
            return res
        if isinstance(node, _NodeIndex):
            base = self._eval_noun(node.base)
            return self._index_get(base, node.axes)
        raise APL36Error(f"SYNTAX ERROR: expected value, got {node}")

    def _eval_items(self, items: List[object]) -> Optional[APLArray]:
        if not items:
            return None

        # Handle top-level branch `→ expr`
        if isinstance(items[0], _NodeBranch):
            if len(items) == 1:
                raise _APLBranch(0)
            target_arr = self._eval_items(items[1:])
            if target_arr is None or target_arr.size == 0:
                raise _APLBranch(None)
            t_int = int(target_arr.as_int_arr().ravel()[0])
            raise _APLBranch(t_int)

        idx = len(items) - 1
        # Evaluate rightmost primary noun
        if not self._is_noun_node(items[idx]):
            raise APL36Error(f"SYNTAX ERROR: expression ends with non-value {items[idx]}")
        val: Optional[APLArray] = self._eval_noun(items[idx])
        idx -= 1

        while idx >= 0:
            cur = items[idx]
            if isinstance(cur, _NodeAssign):
                if idx == 0 or val is None:
                    raise APL36Error("SYNTAX ERROR: invalid assignment `←`")
                target = items[idx - 1]
                if isinstance(target, _NodeVar):
                    val = self._set_var(target.name, val)
                elif isinstance(target, _NodeIndex) and isinstance(target.base, _NodeVar):
                    self._index_set(target.base.name, target.axes, val)
                else:
                    raise APL36Error("SYNTAX ERROR: invalid assignment target")
                idx -= 2
                continue

            if isinstance(cur, _NodeVerb):
                if val is None:
                    raise APL36Error("VALUE ERROR: missing right operand")
                # Check whether immediate left neighbor is a noun -> DYADIC, else MONADIC
                if idx > 0 and self._is_noun_node(items[idx - 1]):
                    left_val = self._eval_noun(items[idx - 1])
                    val = self._apply_dyadic(cur, left_val, val)
                    idx -= 2
                else:
                    val = self._apply_monadic(cur, val)
                    idx -= 1
                continue

            raise APL36Error(f"SYNTAX ERROR: unexpected token sequence near {cur}")

        return val

    # ------------------------------------------------------------------
    # Multi-Axis Indexing `A[i;j]` & Indexed Assignment `A[i;j] ← V`
    # ------------------------------------------------------------------

    def _resolve_axis_index(self, ax_expr: Optional[List[object]], dim_len: int):
        if ax_expr is None:
            return slice(None)
        idx_arr = self._eval_items(ax_expr)
        if idx_arr is None:
            raise APL36Error("INDEX ERROR: empty index expression")
        raw = idx_arr.as_int_arr()
        if self.io == -1:
            offset = dim_len // 2
            zero_based = raw + offset
        else:
            zero_based = raw - self.io
        if np.any(zero_based < 0) or np.any(zero_based >= dim_len):
            raise APL36Error(f"INDEX ERROR: index out of range for axis length {dim_len}")
        return int(zero_based) if raw.ndim == 0 else zero_based

    def _index_get(self, base: APLArray, axes: List[Optional[List[object]]]) -> APLArray:
        if len(axes) != base.rank:
            raise APL36Error(f"RANK ERROR: {len(axes)} indices for rank-{base.rank} array")
        resolved = [self._resolve_axis_index(ax, base.shape[k]) for k, ax in enumerate(axes)]
        # APL outer-indexing semantics across multi-axis array indices
        cur = base.data
        out_axis = 0
        for r in resolved:
            if isinstance(r, slice):
                out_axis += 1
            elif isinstance(r, int):
                cur = np.take(cur, r, axis=out_axis)
            else:
                cur = np.take(cur, r, axis=out_axis)
                out_axis += r.ndim
        return APLArray(np.asarray(cur), is_char=base.is_char)

    def _index_set(self, var_name: str, axes: List[Optional[List[object]]], val: APLArray) -> None:
        base = self._get_var(var_name)
        if len(axes) != base.rank:
            raise APL36Error(f"RANK ERROR: {len(axes)} indices for rank-{base.rank} array")
        resolved = tuple(self._resolve_axis_index(ax, base.shape[k]) for k, ax in enumerate(axes))
        arr = base.data.copy()
        if not base.is_char and (arr.dtype == np.int64 and val.data.dtype == np.float64):
            arr = arr.astype(np.float64)
        arr[resolved] = val.data
        self._set_var(var_name, APLArray(arr, is_char=base.is_char))

    # ------------------------------------------------------------------
    # Scalar Broadcast Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _broadcast_pair(a: APLArray, b: APLArray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
        if a.shape == b.shape:
            return a.data, b.data, a.shape
        if a.size == 1:
            return np.full(b.shape, a.data.reshape(()), dtype=a.data.dtype), b.data, b.shape
        if b.size == 1:
            return a.data, np.full(a.shape, b.data.reshape(()), dtype=b.data.dtype), a.shape
        raise APL36Error(f"LENGTH/RANK ERROR: shapes {a.shape} and {b.shape} do not conform")

    # ------------------------------------------------------------------
    # Monadic Verbs
    # ------------------------------------------------------------------

    def _apply_monadic(self, verb: _NodeVerb, right: APLArray) -> Optional[APLArray]:
        if verb.kind == "USER_FN":
            return self._call_user_func(self.functions[str(verb.val)], None, right)
        if verb.kind == "SYS_FN":
            return self._apply_sys_monadic(str(verb.val), right)
        if verb.kind == "REDUCE":
            return self._apply_reduce(verb, right)
        if verb.kind == "SCAN":
            return self._apply_scan(verb, right)
        if verb.kind != "PRIM":
            raise APL36Error(f"SYNTAX ERROR: {verb.kind} cannot be used monadically")

        op = str(verb.val)
        if op == "⍴":
            return APLArray(np.array(right.shape, dtype=np.int64))
        if op == ",":
            return APLArray(right.data.ravel().copy(), is_char=right.is_char)
        if op == "⍳":
            n = right.as_scalar_int()
            if n < 0:
                raise APL36Error("DOMAIN ERROR: ⍳ requires non-negative integer")
            if self.io == -1:
                half = n // 2
                return APLArray(np.arange(-half, -half + n, dtype=np.int64))
            return APLArray(np.arange(self.io, self.io + n, dtype=np.int64))
        if op == "⌽":
            if right.rank == 0:
                return right
            return APLArray(np.flip(right.data, axis=-1).copy(), is_char=right.is_char)
        if op == "⊖":
            if right.rank == 0:
                return right
            return APLArray(np.flip(right.data, axis=0).copy(), is_char=right.is_char)
        if op == "⍉":
            if right.rank <= 1:
                return right
            return APLArray(np.transpose(right.data).copy(), is_char=right.is_char)
        if op == "⍋":
            if right.rank != 1:
                raise APL36Error("RANK ERROR: ⍋ requires vector")
            order = np.argsort(right.data, kind="stable")
            offset = -(right.size // 2) if self.io == -1 else self.io
            return APLArray(order.astype(np.int64) + offset)
        if op == "⍒":
            if right.rank != 1:
                raise APL36Error("RANK ERROR: ⍒ requires vector")
            order = np.argsort(-right.as_float_arr(), kind="stable")
            offset = -(right.size // 2) if self.io == -1 else self.io
            return APLArray(order.astype(np.int64) + offset)
        if op == "⍕":
            return apl_chars(format_apl(right, self.pp))
        if op == "⍎":
            if not right.is_char:
                raise APL36Error("DOMAIN ERROR: ⍎ expects character vector")
            text = str(right.to_python())
            res = self.eval_line(text)
            if res is None:
                raise APL36Error("VALUE ERROR: ⍎ expression returned no value")
            return res
        if op == "⌹":
            # Matrix inverse (Gauss-Jordan / numpy.linalg.inv)
            a = right.as_float_arr()
            if a.ndim == 0:
                return apl_scalar(1.0 / float(a))
            if a.ndim != 2 or a.shape[0] != a.shape[1]:
                raise APL36Error("DOMAIN ERROR: monadic ⌹ requires square matrix")
            try:
                inv = np.linalg.inv(a)
            except np.linalg.LinAlgError as exc:
                raise APL36Error("DOMAIN ERROR: singular matrix in ⌹") from exc
            return apl_num(inv)
        if op == "⊤":
            # Monadic ⊤: minimal MSB-first balanced-ternary trit vector of integer
            v = right.as_scalar_int()
            if v == 0:
                return APLArray(np.array([0], dtype=np.int64))
            ts = int_to_trits(v, WORD_TRITS)
            while len(ts) > 1 and ts[-1] == 0:
                ts.pop()
            return APLArray(np.array(list(reversed(ts)), dtype=np.int64))
        if op == "⊥":
            # Monadic ⊥: decode balanced-ternary trit vector (MSB-first) to T36 integer
            iv = right.as_int_arr().ravel()
            val = 0
            for t in iv:
                val = val * 3 + int(t)
            return apl_scalar(val)

        # Elementwise numeric monadics
        f = right.as_float_arr()
        if op == "+":
            return right
        if op == "-":
            return apl_num(-f)
        if op == "×":
            # Iverson monadic `×` IS literally the balanced signum trit (-1, 0, +1)!
            out = np.where(np.isnan(f), 0.0, np.sign(f))
            return APLArray(out.astype(np.int64))
        if op == "÷":
            with np.errstate(divide="raise"):
                if np.any(f == 0):
                    raise APL36Error("DOMAIN ERROR: division by zero in ÷")
                return apl_num(1.0 / f)
        if op == "*":
            return apl_num(np.exp(f))
        if op == "⍟":
            if np.any(f <= 0):
                raise APL36Error("DOMAIN ERROR: ⍟ requires positive argument")
            return apl_num(np.log(f))
        if op == "∣":
            return apl_num(np.abs(f))
        if op == "⌈":
            return apl_num(np.ceil(f))
        if op == "⌊":
            return apl_num(np.floor(f))
        if op == "!":
            vec_g = np.vectorize(lambda x: math.gamma(x + 1.0))
            return apl_num(vec_g(f))
        if op == "○":
            return apl_num(math.pi * f)
        if op == "?":
            # Roll: random integer in origin..n
            iv = right.as_int_arr()
            out = np.zeros_like(iv)
            it = np.nditer(iv, flags=["multi_index"])
            for x in it:
                k = int(x)
                if k < 1:
                    raise APL36Error("DOMAIN ERROR: ? requires positive integer")
                r = int(self._rng.integers(0, k))
                out[it.multi_index] = r + (-(k // 2) if self.io == -1 else self.io)
            return APLArray(out)
        if op in ("∼", "¬"):
            # Kleene NOT if ⎕KL=1 or any negative trit present; else classic boolean 1-x
            use_kleene = (self.kl == 1) or bool(np.any(f < 0))
            if use_kleene:
                return apl_num(-f)
            return apl_num(1.0 - f)

        raise APL36Error(f"SYNTAX ERROR: unsupported monadic primitive {op!r}")

    # ------------------------------------------------------------------
    # Dyadic Verbs
    # ------------------------------------------------------------------

    def _apply_dyadic(self, verb: _NodeVerb, left: APLArray, right: APLArray) -> Optional[APLArray]:
        if verb.kind == "USER_FN":
            return self._call_user_func(self.functions[str(verb.val)], left, right)
        if verb.kind == "SYS_FN":
            return self._apply_sys_dyadic(str(verb.val), left, right)
        if verb.kind == "OUTER":
            return self._apply_outer(verb.val, left, right)  # type: ignore[arg-type]
        if verb.kind == "INNER":
            f_v, g_v = verb.val  # type: ignore[misc]
            return self._apply_inner(f_v, g_v, left, right)
        if verb.kind == "REDUCE":
            # Dyadic windowed reduce: N f/ V (e.g. 2 +/ 1 2 3 4 -> 3 5 7)
            return self._apply_window_reduce(verb, left, right)
        if verb.kind != "PRIM":
            raise APL36Error(f"SYNTAX ERROR: {verb.kind} cannot be used dyadically")

        op = str(verb.val)

        # Structural dyadic primitives
        if op == "⍴":
            dims = tuple(int(x) for x in left.as_int_arr().ravel())
            if any(d < 0 for d in dims):
                raise APL36Error("DOMAIN ERROR: negative dimension in ⍴")
            total = 1
            for d in dims:
                total *= d
            if total == 0:
                return APLArray(np.zeros(dims, dtype=right.data.dtype), is_char=right.is_char)
            if right.size == 0:
                fill = char_to_obb729(" ") if right.is_char else 0
                return APLArray(np.full(dims, fill, dtype=right.data.dtype), is_char=right.is_char)
            flat = right.data.ravel()
            tiled = np.resize(flat, total).reshape(dims)
            return APLArray(tiled.copy(), is_char=right.is_char)

        if op == ",":
            if left.is_char != right.is_char:
                raise APL36Error("DOMAIN ERROR: cannot catenate character and numeric arrays")
            if left.rank <= 1 and right.rank <= 1:
                la = left.data.ravel()
                ra = right.data.ravel()
                return APLArray(np.concatenate([la, ra]), is_char=left.is_char)
            # Catenate along last axis (classic APL\360)
            a, b = left.data, right.data
            if a.ndim == b.ndim - 1:
                a = np.expand_dims(a, axis=-1)
            elif b.ndim == a.ndim - 1:
                b = np.expand_dims(b, axis=-1)
            if a.ndim != b.ndim or a.shape[:-1] != b.shape[:-1]:
                raise APL36Error(f"LENGTH/RANK ERROR in `,`: {left.shape} vs {right.shape}")
            return APLArray(np.concatenate([a, b], axis=-1), is_char=left.is_char)

        if op == "↑":
            counts = left.as_int_arr().ravel()
            arr = right.data
            if arr.ndim == 0:
                arr = arr.reshape((1,))
            if len(counts) != arr.ndim:
                raise APL36Error("RANK ERROR in ↑")
            fill = char_to_obb729(" ") if right.is_char else 0
            out = arr
            for ax, c in enumerate(counts):
                cur_len = out.shape[ax]
                ac = abs(int(c))
                if ac <= cur_len:
                    sl = [slice(None)] * out.ndim
                    sl[ax] = slice(0, ac) if c >= 0 else slice(cur_len - ac, cur_len)
                    out = out[tuple(sl)]
                else:
                    pad = [(0, 0)] * out.ndim
                    pad[ax] = (0, ac - cur_len) if c >= 0 else (ac - cur_len, 0)
                    out = np.pad(out, pad, mode="constant", constant_values=fill)
            return APLArray(out.copy(), is_char=right.is_char)

        if op == "↓":
            counts = left.as_int_arr().ravel()
            arr = right.data
            if arr.ndim == 0:
                arr = arr.reshape((1,))
            if len(counts) != arr.ndim:
                raise APL36Error("RANK ERROR in ↓")
            sl = []
            for ax, c in enumerate(counts):
                n = arr.shape[ax]
                ci = int(c)
                if ci >= 0:
                    sl.append(slice(min(ci, n), n))
                else:
                    sl.append(slice(0, max(0, n + ci)))
            return APLArray(arr[tuple(sl)].copy(), is_char=right.is_char)

        if op in ("⌽", "⊖"):
            k = left.as_scalar_int()
            ax = -1 if op == "⌽" else 0
            return APLArray(np.roll(right.data, -k, axis=ax), is_char=right.is_char)

        if op == "⍉":
            perm = [int(x) - self.io for x in left.as_int_arr().ravel()]
            return APLArray(np.transpose(right.data, axes=perm).copy(), is_char=right.is_char)

        if op == "⍳":
            # Index-of: find each element of right in vector left (honoring APL\360 1e-11 fuzz on floats!)
            if left.rank != 1:
                raise APL36Error("RANK ERROR: dyadic ⍳ left arg must be vector")
            offset = -(left.size // 2) if self.io == -1 else self.io
            not_found = left.size + offset
            out = np.empty(right.shape, dtype=np.int64)
            if left.is_char or (left.data.dtype == np.int64 and right.data.dtype == np.int64):
                lut: Dict[int, int] = {}
                for i, x in enumerate(left.data):
                    k = int(x)
                    if k not in lut:
                        lut[k] = i + offset
                it = np.nditer(right.data, flags=["multi_index"])
                for x in it:
                    out[it.multi_index] = lut.get(int(x), not_found)
            else:
                lf = left.as_float_arr()
                rf = right.as_float_arr()
                it = np.nditer(rf, flags=["multi_index"])
                for x in it:
                    xv = float(x)
                    matches = np.where(np.abs(lf - xv) <= 1e-10 * max(1.0, abs(xv)))[0]
                    out[it.multi_index] = int(matches[0]) + offset if matches.size > 0 else not_found
            return APLArray(out)

        if op == "∈":
            # Membership
            rset = set(right.data.ravel().tolist())
            out = np.empty(left.shape, dtype=np.int64)
            f_val = -1 if self.kl == 1 else 0
            it = np.nditer(left.data, flags=["multi_index"])
            for x in it:
                out[it.multi_index] = 1 if x.item() in rset else f_val
            return APLArray(out)

        if op in ("/", "⌿"):
            # Compress / replicate! (Keeps elements where mask > 0; replicates k times for k > 1)
            mask = left.as_int_arr().ravel()
            ax = -1 if op == "/" else 0
            arr = right.data
            if arr.ndim == 0:
                arr = arr.reshape((1,))
            if mask.size == 1:
                mask = np.full(arr.shape[ax], int(mask[0]), dtype=np.int64)
            if mask.size != arr.shape[ax]:
                raise APL36Error(f"LENGTH ERROR in compress `/`: {mask.size} vs {arr.shape[ax]}")
            reps = np.maximum(mask, 0)
            return APLArray(np.repeat(arr, reps, axis=ax), is_char=right.is_char)

        if op in ("\\", "⍀"):
            # Expand!
            mask = left.as_int_arr().ravel()
            ax = -1 if op == "\\" else 0
            arr = right.data
            if arr.ndim == 0:
                arr = arr.reshape((1,))
            pos_count = int(np.sum(mask > 0))
            if pos_count != arr.shape[ax]:
                raise APL36Error("LENGTH ERROR in expand `\\`")
            fill = char_to_obb729(" ") if right.is_char else 0
            new_shape = list(arr.shape)
            new_shape[ax] = mask.size
            out = np.full(new_shape, fill, dtype=arr.dtype)
            src_i = 0
            for dst_i, m in enumerate(mask):
                if m > 0:
                    sl_dst = [slice(None)] * arr.ndim
                    sl_src = [slice(None)] * arr.ndim
                    sl_dst[ax] = dst_i
                    sl_src[ax] = src_i
                    out[tuple(sl_dst)] = arr[tuple(sl_src)]
                    src_i += 1
            return APLArray(out, is_char=right.is_char)

        if op == "⌹":
            # Matrix divide / linear least-squares solve `b ⌹ A` -> solves A x = b!
            b = left.as_float_arr()
            a = right.as_float_arr()
            if a.ndim != 2:
                raise APL36Error("RANK ERROR: dyadic ⌹ right arg must be rank-2 matrix")
            try:
                if a.shape[0] == a.shape[1]:
                    sol = np.linalg.solve(a, b)
                else:
                    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
            except np.linalg.LinAlgError as exc:
                raise APL36Error("DOMAIN ERROR: singular matrix in ⌹") from exc
            return apl_num(sol)

        if op == "⊤":
            # Encode / Representation: negative radices (`¯3 ¯3 ¯3 ⊤ N`) = balanced ternary!
            rad = left.as_int_arr().ravel()
            vals = right.as_int_arr()
            out_shape = (rad.size,) + vals.shape
            out = np.zeros(out_shape, dtype=np.int64)
            it = np.nditer(vals, flags=["multi_index"])
            for x in it:
                v = int(x)
                digits = [0] * rad.size
                for k in range(rad.size - 1, -1, -1):
                    r = int(rad[k])
                    if r == -3:
                        # Balanced-ternary digit {-1, 0, +1}!
                        rem = v % 3
                        if rem == 2:
                            rem = -1
                        digits[k] = rem
                        v = (v - rem) // 3
                    elif r < 0:
                        # Balanced odd base |r|
                        b_abs = -r
                        half = (b_abs - 1) // 2
                        rem = (v + half) % b_abs - half
                        digits[k] = rem
                        v = (v - rem) // b_abs
                    elif r == 0:
                        digits[k] = v
                        v = 0
                    else:
                        v, rem = divmod(v, r)
                        digits[k] = rem
                for k in range(rad.size):
                    out[(k,) + it.multi_index] = digits[k]
            return APLArray(out)

        if op == "⊥":
            # Decode / Base value (Horner evaluation along first axis of right)
            rad = left.as_float_arr().ravel()
            r_arr = right.as_float_arr()
            if r_arr.ndim == 0:
                return right
            n = r_arr.shape[0]
            if rad.size == 1:
                rad = np.full(n, float(rad[0]), dtype=np.float64)
            elif rad.size != n:
                raise APL36Error("LENGTH ERROR in ⊥")
            acc = np.zeros(r_arr.shape[1:], dtype=np.float64)
            for k in range(n):
                acc = acc * abs(rad[k]) + r_arr[k]
            return apl_num(acc)

        # Elementwise dyadic operations (with broadcast!)
        if left.is_char and right.is_char and op in ("=", "≠"):
            a, b, _ = self._broadcast_pair(left, right)
            eq = (a == b) if op == "=" else (a != b)
            f_val = -1 if self.kl == 1 else 0
            return APLArray(np.where(eq, 1, f_val).astype(np.int64))

        a_raw, b_raw, shape = self._broadcast_pair(left, right)
        if left.is_char or right.is_char:
            raise APL36Error(f"DOMAIN ERROR: character operand in `{op}`")

        # Hardware C-Complex 27-lane vector acceleration for 1D vectors (`n >= 2`)!
        if len(shape) == 1 and shape[0] >= 2 and op in ("+", "-", "×", "∧", "∨", "≡"):
            if op in ("∧", "∨", "≡"):
                # Hardware VKLEEN executes tritwise Kleene AND/OR/EQV on {-1, 0, +1}!
                if np.all(np.isin(a_raw, (-1, 0, 1))) and np.all(np.isin(b_raw, (-1, 0, 1))):
                    return APLArray(self._hw_vec_binop(op, a_raw, b_raw))
            else:
                res = self._hw_vec_binop(op, a_raw.astype(np.float64), b_raw.astype(np.float64))
                return apl_num(res)

        a = a_raw.astype(np.float64, copy=False)
        b = b_raw.astype(np.float64, copy=False)

        if op == "+":
            return apl_num(a + b)
        if op == "-":
            return apl_num(a - b)
        if op == "×":
            return apl_num(a * b)
        if op == "÷":
            if np.any(b == 0):
                raise APL36Error("DOMAIN ERROR: division by zero in ÷")
            return apl_num(a / b)
        if op == "*":
            return apl_num(np.power(a, b))
        if op == "⍟":
            return apl_num(np.log(b) / np.log(a))
        if op == "∣":
            # APL residue: `a ∣ b` is b mod a (0 ∣ b is b)
            return apl_num(np.where(a == 0, b, np.mod(b, a)))
        if op == "⌈":
            return apl_num(np.maximum(a, b))
        if op == "⌊":
            return apl_num(np.minimum(a, b))
        if op == "!":
            vf = np.vectorize(lambda x, y: math.gamma(y + 1.0) / (math.gamma(x + 1.0) * math.gamma(y - x + 1.0)))
            return apl_num(vf(a, b))
        if op == "○":
            # Iconic APL\360 circular functions!
            out = np.empty_like(b, dtype=np.float64)
            it = np.nditer([a, b], flags=["multi_index"])
            for av, bv in it:
                k = int(round(float(av)))
                x = float(bv)
                if k == 0:
                    y = math.sqrt(max(0.0, 1.0 - x * x))
                elif k == 1:
                    y = math.sin(x)
                elif k == 2:
                    y = math.cos(x)
                elif k == 3:
                    y = math.tan(x)
                elif k == -1:
                    y = math.asin(x)
                elif k == -2:
                    y = math.acos(x)
                elif k == -3:
                    y = math.atan(x)
                elif k == 5:
                    y = math.sinh(x)
                elif k == 6:
                    y = math.cosh(x)
                elif k == 7:
                    y = math.tanh(x)
                else:
                    raise APL36Error(f"DOMAIN ERROR: unsupported circle selector {k}○x")
                out[it.multi_index] = y
            return apl_num(out)

        # Relational operators (propagate UNDECIDED NaN -> 0!)
        if op in ("<", "≤", "=", "≥", ">", "≠"):
            und = np.isnan(a) | np.isnan(b)
            with np.errstate(invalid="ignore"):
                if op == "<":
                    cond = a < b
                elif op == "≤":
                    cond = a <= b
                elif op == "=":
                    cond = np.abs(a - b) <= 1e-11
                elif op == "≥":
                    cond = a >= b
                elif op == ">":
                    cond = a > b
                else:
                    cond = np.abs(a - b) > 1e-11
            f_val = -1 if self.kl == 1 else 0
            res = np.where(und, 0, np.where(cond, 1, f_val)).astype(np.int64)
            return APLArray(res)

        # Kleene Three-Valued & Boolean Logic (min/max/product work identically on {0,1} AND {-1,0,+1}!)
        if op == "∧":
            return APLArray(np.minimum(a, b).astype(np.int64))
        if op == "∨":
            return APLArray(np.maximum(a, b).astype(np.int64))
        if op == "⍲":
            m = np.minimum(a, b)
            return APLArray((-m if (self.kl == 1 or np.any(m < 0)) else 1 - m).astype(np.int64))
        if op == "⍱":
            m = np.maximum(a, b)
            return APLArray((-m if (self.kl == 1 or np.any(m < 0)) else 1 - m).astype(np.int64))
        if op == "≡":
            # Balanced-ternary Kleene Equivalence: t_a * t_b!
            return APLArray(np.round(a * b).astype(np.int64))

        raise APL36Error(f"SYNTAX ERROR: unsupported dyadic primitive {op!r}")

    # ------------------------------------------------------------------
    # Higher-Order APL Operators (Reduce, Scan, Inner Product, Outer Product)
    # ------------------------------------------------------------------

    def _resolve_op_axis(self, verb: _NodeVerb, arr: APLArray, first_ax: bool) -> int:
        if verb.axis is not None:
            ax_val = self._eval_items(verb.axis)
            if ax_val is None:
                raise APL36Error("SYNTAX ERROR: empty operator axis")
            ax = ax_val.as_scalar_int() - (0 if self.io == 0 else 1)
            if not (0 <= ax < arr.rank):
                raise APL36Error(f"INDEX ERROR: axis {ax} out of bounds for rank {arr.rank}")
            return ax
        return 0 if first_ax else (arr.rank - 1)

    def _apply_reduce(self, verb: _NodeVerb, right: APLArray) -> APLArray:
        f_verb, first_ax = verb.val  # type: ignore[misc]
        if right.rank == 0:
            return right
        ax = self._resolve_op_axis(verb, right, first_ax)
        op_ch = str(f_verb.val) if f_verb.kind == "PRIM" else ""

        # Hardware C-Complex VRED acceleration for 1D `+/` `⌈/` `⌊/`!
        if right.rank == 1 and right.size >= 2 and op_ch in ("+", "⌈", "⌊"):
            val = self._hw_vec_reduce(op_ch, right.as_float_arr())
            return apl_scalar(val)

        # Empty axis identities
        if right.shape[ax] == 0:
            ident = {"+": 0, "-": 0, "×": 1, "÷": 1, "∨": -1 if self.kl else 0, "∧": 1}.get(op_ch)
            if ident is None:
                raise APL36Error(f"DOMAIN ERROR: no identity element for {op_ch}/ on empty axis")
            new_shape = right.shape[:ax] + right.shape[ax + 1 :]
            return APLArray(np.full(new_shape, ident, dtype=np.int64))

        # General right-associative APL reduction along `ax`
        moved = np.moveaxis(right.data, ax, -1)
        out_shape = moved.shape[:-1]
        n = moved.shape[-1]
        out_vals: List[APLArray] = []
        flat_rows = moved.reshape((-1, n))
        for r in range(flat_rows.shape[0]):
            acc = APLArray(np.array(flat_rows[r, n - 1]), is_char=right.is_char)
            for k in range(n - 2, -1, -1):
                cur = APLArray(np.array(flat_rows[r, k]), is_char=right.is_char)
                acc = self._apply_dyadic(f_verb, cur, acc)  # type: ignore[assignment]
            out_vals.append(acc)
        if not out_shape:
            return out_vals[0]
        arr_out = np.array([v.data.item() for v in out_vals]).reshape(out_shape)
        return apl_num(arr_out) if not right.is_char else APLArray(arr_out, is_char=True)

    def _apply_window_reduce(self, verb: _NodeVerb, left: APLArray, right: APLArray) -> APLArray:
        f_verb, first_ax = verb.val  # type: ignore[misc]
        w = left.as_scalar_int()
        if right.rank != 1 or w < 1 or w > right.size:
            raise APL36Error("LENGTH/RANK ERROR in windowed reduce")
        out = []
        for i in range(right.size - w + 1):
            sl = APLArray(right.data[i : i + w], is_char=right.is_char)
            out.append(self._apply_reduce(_NodeVerb("REDUCE", (f_verb, first_ax)), sl).data.item())
        return apl_num(np.array(out))

    def _apply_scan(self, verb: _NodeVerb, right: APLArray) -> APLArray:
        f_verb, first_ax = verb.val  # type: ignore[misc]
        if right.rank == 0:
            return right
        ax = self._resolve_op_axis(verb, right, first_ax)
        moved = np.moveaxis(right.data, ax, -1)
        n = moved.shape[-1]
        flat = moved.reshape((-1, n))
        res_rows = []
        for r in range(flat.shape[0]):
            row_res = []
            for k in range(1, n + 1):
                sl = APLArray(flat[r, :k], is_char=right.is_char)
                val = self._apply_reduce(_NodeVerb("REDUCE", (f_verb, False)), sl)
                row_res.append(val.data.item())
            res_rows.append(row_res)
        arr = np.array(res_rows).reshape(moved.shape)
        arr = np.moveaxis(arr, -1, ax)
        return apl_num(arr)

    def _apply_outer(self, f_verb: _NodeVerb, left: APLArray, right: APLArray) -> APLArray:
        l_flat = left.data.ravel()
        r_flat = right.data.ravel()
        out_shape = left.shape + right.shape
        out_items = []
        for lv in l_flat:
            la = APLArray(np.array(lv), is_char=left.is_char)
            for rv in r_flat:
                ra = APLArray(np.array(rv), is_char=right.is_char)
                res = self._apply_dyadic(f_verb, la, ra)
                out_items.append(res.data.item())  # type: ignore[union-attr]
        arr = np.array(out_items).reshape(out_shape)
        return apl_num(arr)

    def _apply_inner(self, f_verb: _NodeVerb, g_verb: _NodeVerb, left: APLArray, right: APLArray) -> APLArray:
        """Generalized APL Inner Product `A f.g B` with C-Complex & T-Fabric acceleration!"""
        la = left.data if left.rank >= 1 else left.data.reshape((1,))
        ra = right.data if right.rank >= 1 else right.data.reshape((1,))
        if la.shape[-1] != ra.shape[0]:
            raise APL36Error(f"LENGTH ERROR in inner product: {left.shape} vs {right.shape}")

        # Hardware fast paths for `+.×`!
        if f_verb.kind == "PRIM" and f_verb.val == "+" and g_verb.kind == "PRIM" and g_verb.val == "×":
            lf = left.as_float_arr()
            rf = right.as_float_arr()
            if lf.ndim == 1 and rf.ndim == 1:
                return apl_scalar(self._hw_vec_dot(lf, rf))
            if lf.ndim == 2 and rf.ndim == 2:
                return apl_num(self._hw_matmul(lf, rf))
            if lf.ndim == 2 and rf.ndim == 1:
                res = self._hw_matmul(lf, rf.reshape((-1, 1))).reshape((-1,))
                return apl_num(res)
            if lf.ndim == 1 and rf.ndim == 2:
                res = self._hw_matmul(lf.reshape((1, -1)), rf).reshape((-1,))
                return apl_num(res)

        # General `f.g` over arbitrary ranks (e.g., `⌈.+` tropical or `∨.∧` Kleene closure!)
        out_shape = la.shape[:-1] + ra.shape[1:]
        k_len = la.shape[-1]
        l_rows = la.reshape((-1, k_len))
        r_cols = np.moveaxis(ra, 0, -1).reshape((-1, k_len))
        red_verb = _NodeVerb("REDUCE", (f_verb, False))
        results = []
        for i in range(l_rows.shape[0]):
            lv = APLArray(l_rows[i], is_char=left.is_char)
            for j in range(r_cols.shape[0]):
                rv = APLArray(r_cols[j], is_char=right.is_char)
                pair = self._apply_dyadic(g_verb, lv, rv)
                red = self._apply_reduce(red_verb, pair)  # type: ignore[arg-type]
                results.append(red.data.item())
        if not out_shape:
            return apl_scalar(results[0])
        return apl_num(np.array(results).reshape(out_shape))

    # ------------------------------------------------------------------
    # Duwamish-I Native System Functions (`⎕H`, `⎕OBB`, `⎕DB`, `⎕WALD`, `⎕FFT`, `⎕TB1`)
    # ------------------------------------------------------------------

    def _apply_sys_monadic(self, name: str, right: APLArray) -> APLArray:
        if name in ("⎕H", "⎕HEPT"):
            return apl_chars(int_to_hept(right.as_scalar_int(), WORD_TRITS))
        if name == "⎕OBB":
            if right.is_char:
                return APLArray(right.data.copy(), is_char=False)
            return APLArray(right.as_int_arr().copy(), is_char=True)
        if name == "⎕DB":
            # Convert deciban float values -> exact DB36 ticks (81 ticks/dB)
            f = right.as_float_arr()
            vf = np.vectorize(lambda x: db_to_ticks(float(x)), otypes=[np.int64])
            return APLArray(vf(f))
        if name == "⎕LSE":
            # Good deciban log-sum-exp reduction over deciban values (dB)
            ticks = [db_to_ticks(float(x)) for x in right.as_float_arr().ravel()]
            if not ticks:
                raise APL36Error("LENGTH ERROR in ⎕LSE")
            acc = ticks[0]
            for t in ticks[1:]:
                acc = deciban_lse(acc, t)
            return apl_scalar(ticks_to_db(acc))
        if name == "⎕FFT":
            # Oliver-Good Radix-3 TBFP-9C FFT on the 27-tile T-Fabric!
            vec = right.as_float_arr().ravel().astype(np.complex128)
            n = vec.size
            p = 1
            while p < n:
                p *= 3
            if p != n:
                vec = np.pad(vec, (0, p - n))
            spec, _ = self.machine.tfabric.fft_radix3_tbfp9c(vec)
            self.hw_tfabric_calls += 1
            return apl_num(np.abs(spec))
        if name == "⎕TRITS":
            return self._apply_monadic(_NodeVerb("PRIM", "⊤"), right)
        if name == "⎕UNTRITS":
            return self._apply_monadic(_NodeVerb("PRIM", "⊥"), right)
        raise APL36Error(f"DOMAIN ERROR: {name} requires two arguments")

    def _apply_sys_dyadic(self, name: str, left: APLArray, right: APLArray) -> APLArray:
        if name == "⎕WALD":
            # `threshold_db ⎕WALD evidence_db_vec` -> hardware DWALD on C-Complex CPU!
            thresh_ticks = db_to_ticks(left.as_scalar_float())
            ev_f = right.as_float_arr()
            flat = ev_f.ravel()
            n = int(flat.size)
            for i in range(n):
                self.machine.mem.write(_APL_BUF_A + i, db_to_ticks(float(flat[i])), supervisor=True)
            self._run_kernel("_APL_VWALD", n, _APL_BUF_A, 0, _APL_BUF_C, s2_val=thresh_ticks)
            trits = np.array(
                [self.machine.mem.read(_APL_BUF_C + i, supervisor=True)[0] for i in range(n)],
                dtype=np.int64,
            ).reshape(ev_f.shape)
            return APLArray(trits)
        if name == "⎕TB1":
            # `A ⎕TB1 B` -> Rosenblatt-Kelvanov Mode +3 Pure-Trit Associative Inner Product on T-Fabric!
            a = left.as_float_arr()
            b = right.as_float_arr()
            if a.ndim == 1:
                a = a.reshape((1, -1))
            if b.ndim == 1:
                b = b.reshape((-1, 1))
            out, _ = self.machine.tfabric.assoc_tb1(a, b.T)
            self.hw_tfabric_calls += 1
            return apl_num(out)
        raise APL36Error(f"DOMAIN ERROR: {name} is monadic only")

    # ------------------------------------------------------------------
    # User-Defined Function Execution (`∇`)
    # ------------------------------------------------------------------

    def _call_user_func(
        self,
        fn: APLUserFunc,
        left: Optional[APLArray],
        right: Optional[APLArray],
    ) -> Optional[APLArray]:
        frame: Dict[str, Optional[APLArray]] = {}
        if fn.result_var:
            frame[fn.result_var] = None
        if fn.left_arg:
            if left is None:
                raise APL36Error(f"SYNTAX ERROR: {fn.name} requires left argument")
            frame[fn.left_arg] = left
        if fn.right_arg:
            if right is None:
                raise APL36Error(f"SYNTAX ERROR: {fn.name} requires right argument")
            frame[fn.right_arg] = right
        for loc in fn.locals_list:
            frame[loc] = None
        for lbl, lno in fn.labels.items():
            frame[lbl] = apl_scalar(lno)

        self._call_stack.append(frame)
        try:
            pc = 1
            steps = 0
            while 1 <= pc <= len(fn.lines):
                steps += 1
                if steps > 100_000:
                    raise APL36Error(f"LIMIT ERROR: infinite loop in function {fn.name}")
                line = fn.lines[pc - 1]
                try:
                    self._exec_statement_line(line, print_bare=False)
                    pc += 1
                except _APLBranch as br:
                    if br.target is None:
                        pc += 1
                    elif br.target == 0 or br.target > len(fn.lines):
                        break
                    else:
                        pc = br.target
            return frame.get(fn.result_var) if fn.result_var else None
        finally:
            self._call_stack.pop()

    # ------------------------------------------------------------------
    # Line & Script Execution + System Commands
    # ------------------------------------------------------------------

    def _exec_statement_line(self, line: str, print_bare: bool = True) -> Optional[APLArray]:
        toks = tokenize_apl(line)
        # Split on `⋄` statement diamonds
        stmts: List[List[ATok]] = []
        cur: List[ATok] = []
        for t in toks:
            if t.kind == "EOF":
                break
            if t.kind == "DIAMOND":
                if cur:
                    stmts.append(cur + [ATok("EOF", None)])
                    cur = []
            else:
                cur.append(t)
        if cur:
            stmts.append(cur + [ATok("EOF", None)])

        last_val: Optional[APLArray] = None
        for st in stmts:
            structured = self._structure_tokens(st)
            if not structured:
                continue
            # Suppress auto-print if top-level statement is pure assignment or branch
            is_bare_assign = (
                len(structured) >= 2
                and isinstance(structured[1], _NodeAssign)
                and not (isinstance(structured[0], _NodeVar) and structured[0].name == "⎕")
            )
            is_branch = isinstance(structured[0], _NodeBranch)
            val = self._eval_items(structured)
            last_val = val
            if print_bare and val is not None and not is_bare_assign and not is_branch:
                # Avoid double-printing `⎕←` which already appended to output_lines
                if not (
                    len(structured) >= 2
                    and isinstance(structured[0], _NodeVar)
                    and structured[0].name == "⎕"
                    and isinstance(structured[1], _NodeAssign)
                ):
                    self.output_lines.append(format_apl(val, self.pp))
        return last_val

    def eval_line(self, line: str) -> Optional[APLArray]:
        """Evaluate a single APL\\36 line or system command (`)VARS`, `)FNS`, `)HW`, `)CLEAR`)."""
        s = line.strip()
        if not s or s.startswith("⍝") or s.startswith("NB."):
            return None
        if s.startswith(")"):
            cmd = s[1:].strip().upper()
            if cmd == "VARS":
                self.output_lines.append("  ".join(sorted(self.vars.keys())) or "(no variables)")
            elif cmd == "FNS":
                self.output_lines.append("  ".join(sorted(self.functions.keys())) or "(no functions)")
            elif cmd == "CLEAR":
                self.vars.clear()
                self.functions.clear()
                self.output_lines.append("CLEAR WS")
            elif cmd == "HW":
                self.output_lines.append(
                    f"C-Complex: {self.machine.cpu0.cycles:,} cyc ({self.hw_vec_calls} vec kernels, "
                    f"{self.machine.cpu0.istack_hits:,} I-stack hits) | "
                    f"T-Fabric: {self.machine.tfabric.total_ops:,} ops ({self.hw_tfabric_calls} dispatches, "
                    f"ABFT={self.machine.tfabric.total_abft_checks})"
                )
            else:
                raise APL36Error(f"UNKNOWN SYSTEM COMMAND: ){cmd}")
            return None
        return self._exec_statement_line(s, print_bare=True)

    def eval_script(self, source: str) -> Tuple[List[str], Optional[APLArray]]:
        """Execute a multi-line APL\\36 script (including `∇ ... ∇` function definitions)."""
        start_idx = len(self.output_lines)
        last_val: Optional[APLArray] = None
        lines = source.splitlines()
        i = 0
        while i < len(lines):
            raw = lines[i].strip()
            i += 1
            if not raw or raw.startswith("⍝") or raw.startswith("NB."):
                continue
            # Del-editor function definition block `∇ ... ∇`
            if raw.startswith("∇"):
                fn = parse_del_header(raw)
                # Single-line closing `∇`?
                if raw.count("∇") >= 2:
                    self.functions[fn.name] = fn
                    continue
                while i < len(lines):
                    body_ln = lines[i].strip()
                    i += 1
                    if body_ln == "∇":
                        break
                    # Strip leading [n] line number if present
                    body_ln = re.sub(r"^\[\d+\]\s*", "", body_ln)
                    if not body_ln or body_ln.startswith("⍝"):
                        continue
                    # Check label `LBL: stmt` (make sure not inside quotes)
                    m_lbl = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", body_ln)
                    if m_lbl:
                        fn.labels[m_lbl.group(1).upper()] = len(fn.lines) + 1
                        body_ln = m_lbl.group(2).strip()
                    fn.lines.append(body_ln)
                self.functions[fn.name] = fn
                continue
            last_val = self.eval_line(raw)
        return self.output_lines[start_idx:], last_val


def run_apl_script(
    source: str,
    machine: Optional[UE1000Machine] = None,
) -> Tuple[List[str], APL36Session]:
    sess = APL36Session(machine=machine)
    out, _ = sess.eval_script(source)
    return out, sess


def run_apl_repl(machine: Optional[UE1000Machine] = None) -> None:
    """Interactive APL\\36 terminal on OBB Level -9 vector-CRT console."""
    sess = APL36Session(machine=machine)
    print("APL\\36 — Duwamish-I (UE-1000) Balanced-Ternary Array Workspace")
    print("  CLEAR WS | C-Complex 27-lane vectors + 27-tile T-Fabric (⎕TMODE=1 TBFP-9)")
    print("  Commands: )VARS  )FNS  )HW  )CLEAR  )OFF  (use ∇...∇ for functions)")
    while True:
        try:
            line = input("      ")
        except (EOFError, KeyboardInterrupt):
            print("\n)OFF")
            break
        if line.strip().upper() in (")OFF", "QUIT", "EXIT"):
            print("  SESSION CLOSED.")
            break
        if line.strip().startswith("∇"):
            buf = [line]
            if line.count("∇") < 2:
                ln_no = 1
                while True:
                    try:
                        nxt = input(f"[{ln_no}]   ")
                    except (EOFError, KeyboardInterrupt):
                        break
                    buf.append(nxt)
                    if nxt.strip() == "∇":
                        break
                    ln_no += 1
            try:
                sess.eval_script("\n".join(buf))
            except APL36Error as exc:
                print(f"  {exc}")
            continue
        start = len(sess.output_lines)
        try:
            sess.eval_line(line)
            for out_ln in sess.output_lines[start:]:
                print(out_ln)
        except APL36Error as exc:
            print(f"  {exc}")
