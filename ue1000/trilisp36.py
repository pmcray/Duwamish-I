"""TRILISP-36 — Classic LISP 1.5 Native TRIAD-36 Compiler & Heap VM for Duwamish-I (UE-1000).

One Trit, Three Primary Types (exact balanced-ternary LSB tag):
  +1 (trit 0 = '1') : Immediate 35-trit Fixnum integer `3*n + 1` (±2.50e16)
   0 (trit 0 = '0') : Heap Pointer `3*i`
                        i > 0 -> Cons cell in T-DRAM at HEAP_BASE + 2*i : [CAR, CDR]
                        i < 0 -> Boxed cell in T-DRAM at BOX_BASE + 2*(-i): [TYPE_SYM, PAYLOAD]
                                 (TF36 float, DB36 deciban, or compiled CLOSURE!)
  -1 (trit 0 = 'T') : Interned Symbol `3*j - 1`
                        j = 0 (word -1) -> NIL (FALSE)
                        j = 1 (word +2) -> T   (TRUE)
                        j = 2 (word +5) -> U   (UNDECIDED / UNKNOWN)
                        j >= 3          -> Interned symbol with value slot in SYMVAL_BASE + j

Balanced-Ternary Unboxing Miracle:
  Because 1/3 < 1/2, a single 1-cycle `SHF Sd, Ss1, #-1` rounds:
    (3*n + 1) >> 1 == n    (Fixnum)
    (3*i + 0) >> 1 == i    (Heap pointer)
    (3*j - 1) >> 1 == j    (Symbol index)
  Zero tag masking or subtraction required!

Everything — CONS, CAR, CDR, RPLACA/D, fixnum arithmetic, Kleene logic, SIGN3 (B3),
lexical closures, tail-call-optimized recursion, and mark-and-sweep GC — compiles directly
to TRIAD-36 assembly (optimized with SAL-3/O peephole + Cray Dual-Parcel packing) and
executes natively on UE1000Machine.cpu0 & TernaryMemory!
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .machine import UE1000Machine
from .sal3 import _pack_dual_parcels, _peephole_optimize
from .ternary import WORD_TRITS, in_range, shift_trits, sign_trit, wrap
from .tfloat import TF36, db_to_ticks, float_to_tfloat, tfloat_to_float, ticks_to_db
from .triad36 import AssembledProgram, assemble


class TriLispError(Exception):
    """Compile-time or runtime error in TRILISP-36."""


# ---------------------------------------------------------------------------
# Tagged Word Representation Constants (in User T-DRAM)
# ---------------------------------------------------------------------------

TAG_NUM = 1
TAG_PTR = 0
TAG_SYM = -1

# Predefined Symbol IDs (word = 3*id - 1)
SYM_NIL = 0        # word = -1
SYM_T = 1          # word = +2
SYM_U = 2          # word = +5  (Kleene UNDECIDED)
SYM_CLOSURE = 3    # word = +8  (boxed closure type tag)
SYM_FLOAT = 4      # word = +11 (boxed TF36 float type tag)
SYM_DB = 5         # word = +14 (boxed DB36 deciban type tag)
SYM_SUBR = 6       # word = +17 (boxed built-in primitive function tag)

W_NIL = 3 * SYM_NIL - 1         # -1
W_T = 3 * SYM_T - 1             # +2
W_U = 3 * SYM_U - 1             # +5
W_CLOSURE = 3 * SYM_CLOSURE - 1 # +8
W_FLOAT = 3 * SYM_FLOAT - 1     # +11
W_DB = 3 * SYM_DB - 1           # +14
W_SUBR = 3 * SYM_SUBR - 1       # +17

# User T-DRAM Layout (well inside [+1 .. +41,629,544])
CODE_ORIGIN = 100
SYMVAL_BASE = 100_000     # Global symbol value table (up to 10,000 symbols)
HEAP_BASE = 200_000       # Cons cells: cell i (1..HEAP_MAX) at HEAP_BASE + 2*i (CAR, CDR)
MARK_BASE = 250_000       # Parallel 1-word mark bits for native TRIAD-36 mark-and-sweep GC!
BOX_BASE = 300_000        # Boxed cells: cell k (1..BOX_MAX) at BOX_BASE + 2*k (TYPE, PAYLOAD)
GC_STACK_BASE = 350_000   # Explicit iterative DFS mark stack for native GC (zero recursion!)
STACK_TOP = 500_000       # C-Complex machine stack SP (S8) growing downward
DEFAULT_HEAP_CELLS = 4096 # Default cons heap size (configurable for GC stress tests!)


def tag_num(n: int) -> int:
    return wrap(3 * n + 1, WORD_TRITS)


def tag_sym(j: int) -> int:
    return wrap(3 * j - 1, WORD_TRITS)


def tag_cons(idx: int) -> int:
    return wrap(3 * idx, WORD_TRITS)


def tag_box(idx: int) -> int:
    return wrap(-3 * idx, WORD_TRITS)


def get_tag(w: int) -> int:
    r = w % 3
    return -1 if r == 2 else r


def untag(w: int) -> int:
    """Exact balanced-ternary unbox: (3*x + t) >> 1 == x for all t in {-1, 0, +1}."""
    return shift_trits(w, -1, WORD_TRITS)


# ---------------------------------------------------------------------------
# S-Expression Reader (Classic LISP 1.5 + 'x + dotted pairs + floats + dB)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sym:
    name: str


@dataclass(frozen=True)
class FltLit:
    val: float


@dataclass(frozen=True)
class DbLit:
    db: float


@dataclass(frozen=True)
class ConsLit:
    car: object
    cdr: object


def _tokenize_lisp(src: str) -> List[str]:
    toks: List[str] = []
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch == ";":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if ch in "()'":
            toks.append(ch)
            i += 1
            continue
        if ch == '"':
            j = i + 1
            buf = ['"']
            while j < n and src[j] != '"':
                if src[j] == "\\" and j + 1 < n:
                    buf.append(src[j + 1])
                    j += 2
                else:
                    buf.append(src[j])
                    j += 1
            if j >= n:
                raise TriLispError("Unterminated string literal")
            buf.append('"')
            toks.append("".join(buf))
            i = j + 1
            continue
        j = i
        while j < n and src[j] not in " \t\r\n();'\"":
            j += 1
        toks.append(src[i:j])
        i = j
    return toks


def read_sexprs(src: str) -> List[object]:
    toks = _tokenize_lisp(src)
    pos = 0

    def parse_one() -> object:
        nonlocal pos
        if pos >= len(toks):
            raise TriLispError("Unexpected EOF while reading S-expression")
        t = toks[pos]
        pos += 1
        if t == "'":
            return ConsLit(Sym("QUOTE"), ConsLit(parse_one(), Sym("NIL")))
        if t == "(":
            items: List[object] = []
            tail: object = Sym("NIL")
            while True:
                if pos >= len(toks):
                    raise TriLispError("Unclosed '(' in S-expression")
                if toks[pos] == ")":
                    pos += 1
                    break
                if toks[pos] == ".":
                    pos += 1
                    tail = parse_one()
                    if pos >= len(toks) or toks[pos] != ")":
                        raise TriLispError("Expected ')' after dotted pair tail")
                    pos += 1
                    break
                items.append(parse_one())
            res = tail
            for it in reversed(items):
                res = ConsLit(it, res)
            return res
        if t == ")":
            raise TriLispError("Unexpected ')'")
        # String literal -> self-evaluating symbol (as in TRILISP!)
        if t.startswith('"') and t.endswith('"'):
            return Sym(t)
        # Deciban literal e.g. 20.0db, -5.5db
        if len(t) > 2 and t[-2:].lower() == "db":
            try:
                return DbLit(float(t[:-2]))
            except ValueError:
                pass
        # Integer literal
        try:
            return int(t)
        except ValueError:
            pass
        # Float literal
        if "." in t or "e" in t.lower():
            try:
                return FltLit(float(t))
            except ValueError:
                pass
        return Sym(t.upper())

    forms: List[object] = []
    while pos < len(toks):
        forms.append(parse_one())
    return forms


def sexpr_to_pylist(node: object) -> Tuple[List[object], object]:
    """Flatten ConsLit chain -> (elements, proper_or_dotted_tail)."""
    elems: List[object] = []
    cur = node
    while isinstance(cur, ConsLit):
        elems.append(cur.car)
        cur = cur.cdr
    return elems, cur


def pylist_to_sexpr(items: Sequence[object], tail: object = Sym("NIL")) -> object:
    res = tail
    for it in reversed(items):
        res = ConsLit(it, res)
    return res


# ---------------------------------------------------------------------------
# Native TRIAD-36 Runtime Assembly (Heap, Mark-and-Sweep GC, Arithmetic, Closures)
# ---------------------------------------------------------------------------

def _build_runtime_asm(heap_cells: int) -> str:
    """Generate the resident native TRIAD-36 LISP 1.5 runtime & Mark-and-Sweep GC."""
    return f"""
; ============================================================================
; TRILISP-36 NATIVE TRIAD-36 RUNTIME & MARK-AND-SWEEP GC (UE-1000 Rev. 1.5)
; Calling convention: args pushed L->R on S8; result in S1; S7/A7 = FP.
; ============================================================================

; --- _TL_CONS(car, cdr) -> tagged cons pointer 3*idx in S1 ---
; Args: 3(A7)=car, 2(A7)=cdr
_TL_CONS:
    PUSH S7
    ADD S7, S8, #0
    S2A A7, S7
_TL_CONS_RETRY:
    ; Check free list head _RT_FREE first
    LD S2, _RT_FREE(A0)
    CMP S2, S0
    BZ _TL_CONS_BUMP
    ; Pop cell idx S2 from free list: _RT_FREE <- M[HEAP_BASE + 2*S2 + 1]
    ADD S3, S2, S2
    LDI S4, #{HEAP_BASE}
    ADD S3, S3, S4
    S2A A1, S3
    LD S4, 1(A1)
    ST S4, _RT_FREE(A0)
    BR _TL_CONS_INIT
_TL_CONS_BUMP:
    LD S2, _RT_NEXT(A0)
    LDI S3, #{heap_cells}
    CMP S2, S3
    BP _TL_CONS_DO_GC
    ADD S4, S2, #1
    ST S4, _RT_NEXT(A0)
    ADD S3, S2, S2
    LDI S4, #{HEAP_BASE}
    ADD S3, S3, S4
    S2A A1, S3
_TL_CONS_INIT:
    LD S4, 3(A7)          ; car
    ST S4, 0(A1)
    LD S4, 2(A7)          ; cdr
    ST S4, 1(A1)
    LD S4, _RT_USED(A0)
    ADD S4, S4, #1
    ST S4, _RT_USED(A0)
    SHF S1, S2, #1        ; tagged cons ptr = 3 * idx (tag = 0!)
    ADD S8, S7, #0
    POP S7
    S2A A7, S7
    RET #2
_TL_CONS_DO_GC:
    CALL A0, _TL_GC_INNER
    ; Check if GC freed at least one cell
    LD S2, _RT_FREE(A0)
    CMP S2, S0
    BN _TL_CONS_RETRY
    BP _TL_CONS_RETRY
    ; Out of heap memory -> SVC #309 fatal error
    SVC #309
    HLT

; --- _TL_BOX(type_sym, payload_word) -> negative tagged box ptr -3*k in S1 ---
_TL_BOX:
    PUSH S7
    ADD S7, S8, #0
    S2A A7, S7
    LD S2, _RT_BOXNEXT(A0)
    ADD S4, S2, #1
    ST S4, _RT_BOXNEXT(A0)
    ADD S3, S2, S2
    LDI S4, #{BOX_BASE}
    ADD S3, S3, S4
    S2A A1, S3
    LD S4, 3(A7)          ; type_sym
    ST S4, 0(A1)
    LD S4, 2(A7)          ; payload
    ST S4, 1(A1)
    NEG S2, S2
    SHF S1, S2, #1        ; -3 * k (tag = 0, negative!)
    ADD S8, S7, #0
    POP S7
    S2A A7, S7
    RET #2

; --- _TL_CAR(x) -> S1 = CAR(x), or NIL if x == NIL ---
_TL_CAR:
    PUSH S7
    ADD S7, S8, #0
    S2A A7, S7
    LD S1, 2(A7)
    CMP S1, #-1           ; NIL?
    BZ _TL_CAR_RET
    SHF S2, S1, #-1       ; idx = w >> 1 (1-cycle balanced unbox!)
    CMP S2, S0
    BN _TL_CAR_NIL
    BZ _TL_CAR_NIL
    ADD S2, S2, S2
    LDI S3, #{HEAP_BASE}
    ADD S2, S2, S3
    S2A A1, S2
    LD S1, 0(A1)
    BR _TL_CAR_RET
_TL_CAR_NIL:
    ADD S1, S0, #-1
_TL_CAR_RET:
    ADD S8, S7, #0
    POP S7
    S2A A7, S7
    RET #1

; --- _TL_CDR(x) -> S1 = CDR(x), or NIL if x == NIL ---
_TL_CDR:
    PUSH S7
    ADD S7, S8, #0
    S2A A7, S7
    LD S1, 2(A7)
    CMP S1, #-1
    BZ _TL_CDR_RET
    SHF S2, S1, #-1
    CMP S2, S0
    BN _TL_CDR_NIL
    BZ _TL_CDR_NIL
    ADD S2, S2, S2
    LDI S3, #{HEAP_BASE}
    ADD S2, S2, S3
    S2A A1, S2
    LD S1, 1(A1)
    BR _TL_CDR_RET
_TL_CDR_NIL:
    ADD S1, S0, #-1
_TL_CDR_RET:
    ADD S8, S7, #0
    POP S7
    S2A A7, S7
    RET #1

; --- _TL_RPLACA(cell, val) -> mutates CAR, returns cell ---
_TL_RPLACA:
    PUSH S7
    ADD S7, S8, #0
    S2A A7, S7
    LD S1, 3(A7)
    SHF S2, S1, #-1
    ADD S2, S2, S2
    LDI S3, #{HEAP_BASE}
    ADD S2, S2, S3
    S2A A1, S2
    LD S4, 2(A7)
    ST S4, 0(A1)
    ADD S8, S7, #0
    POP S7
    S2A A7, S7
    RET #2

; --- _TL_RPLACD(cell, val) -> mutates CDR, returns cell ---
_TL_RPLACD:
    PUSH S7
    ADD S7, S8, #0
    S2A A7, S7
    LD S1, 3(A7)
    SHF S2, S1, #-1
    ADD S2, S2, S2
    LDI S3, #{HEAP_BASE}
    ADD S2, S2, S3
    S2A A1, S2
    LD S4, 2(A7)
    ST S4, 1(A1)
    ADD S8, S7, #0
    POP S7
    S2A A7, S7
    RET #2

; --- _TL_EQUAL(a, b) -> T (+2) if structurally equal, else NIL (-1) ---
_TL_EQUAL:
    PUSH S7
    ADD S7, S8, #0
    S2A A7, S7
    LD S1, 3(A7)
    LD S2, 2(A7)
    CMP S1, S2
    BZ _TL_EQ_T
    ; Both must be positive cons cells (tag == 0 and > 0) to recurse!
    TRT S1, S0, #0
    BN _TL_EQ_F
    BP _TL_EQ_F
    CMP S1, S0
    BN _TL_EQ_F
    BZ _TL_EQ_F
    TRT S2, S0, #0
    BN _TL_EQ_F
    BP _TL_EQ_F
    CMP S2, S0
    BN _TL_EQ_F
    BZ _TL_EQ_F
    ; Compare CAR(a) with CAR(b)
    PUSH S1
    CALL A0, _TL_CAR
    PUSH S1
    LD S2, 2(A7)
    PUSH S2
    CALL A0, _TL_CAR
    PUSH S1
    CALL A0, _TL_EQUAL
    CMP S1, #-1
    BZ _TL_EQ_F
    ; Compare CDR(a) with CDR(b)
    LD S1, 3(A7)
    PUSH S1
    CALL A0, _TL_CDR
    PUSH S1
    LD S2, 2(A7)
    PUSH S2
    CALL A0, _TL_CDR
    PUSH S1
    CALL A0, _TL_EQUAL
    BR _TL_EQ_DONE
_TL_EQ_T:
    ADD S1, S0, #2        ; T (+2)
    BR _TL_EQ_DONE
_TL_EQ_F:
    ADD S1, S0, #-1       ; NIL (-1)
_TL_EQ_DONE:
    ADD S8, S7, #0
    POP S7
    S2A A7, S7
    RET #2

; --- _TL_ASSOC(key, alist) -> matching pair or NIL ---
_TL_ASSOC:
    PUSH S7
    ADD S7, S8, #0
    S2A A7, S7
    LD S5, 2(A7)          ; cur alist
_TL_AS_LP:
    CMP S5, #-1
    BZ _TL_AS_NIL
    PUSH S5
    CALL A0, _TL_CAR      ; S1 = pair
    PUSH S1               ; save pair on stack
    PUSH S1
    CALL A0, _TL_CAR      ; S1 = caar
    LD S2, 3(A7)          ; key
    CMP S1, S2
    BZ _TL_AS_HIT
    POP S1                ; discard pair
    PUSH S5
    CALL A0, _TL_CDR
    ADD S5, S1, #0
    BR _TL_AS_LP
_TL_AS_HIT:
    POP S1                ; return matching pair
    BR _TL_AS_DONE
_TL_AS_NIL:
    ADD S1, S0, #-1
_TL_AS_DONE:
    ADD S8, S7, #0
    POP S7
    S2A A7, S7
    RET #2

; --- _TL_GC_INNER: Native TRIAD-36 Mark-and-Sweep Garbage Collector! ---
; Traces ALL global symbol value cells + conservative scan of machine stack S8..STACK_TOP.
; Uses iterative explicit mark stack at GC_STACK_BASE (zero recursion).
; Returns count of reclaimed cells (raw int) in S1; updates _RT_FREE, _RT_USED, _RT_GC_RUNS.
_TL_GC_INNER:
    PUSH S7
    ADD S7, S8, #0
    S2A A7, S7
    ; Save S2..S6 on stack so caller registers survive GC transparently!
    PUSH S2
    PUSH S3
    PUSH S4
    PUSH S5
    PUSH S6

    ; 1. Clear mark table MARK_BASE[1 .. _RT_NEXT-1]
    LD S5, _RT_NEXT(A0)
    ADD S2, S0, #1
    LDI S6, #{MARK_BASE}
_GC_CLR_LP:
    CMP S2, S5
    BP _GC_CLR_DONE
    BZ _GC_CLR_DONE
    ADD S3, S6, S2
    S2A A1, S3
    ST S0, 0(A1)
    ADD S2, S2, #1
    BR _GC_CLR_LP
_GC_CLR_DONE:
    ; Initialize mark stack pointer A6 = GC_STACK_BASE (grows upward)
    LDI S6, #{GC_STACK_BASE}
    S2A A6, S6

    ; 2. Push global symbol value roots SYMVAL_BASE[0 .. _RT_NSYMS-1]
    LD S5, _RT_NSYMS(A0)
    ADD S2, S0, #0
    LDI S4, #{SYMVAL_BASE}
_GC_SYM_LP:
    CMP S2, S5
    BP _GC_SYM_DONE
    BZ _GC_SYM_DONE
    ADD S3, S4, S2
    S2A A1, S3
    LD S1, 0(A1)
    CALL A0, _GC_PUSH_IF_CONS
    ADD S2, S2, #1
    BR _GC_SYM_LP
_GC_SYM_DONE:

    ; 3. Conservative stack scan: S8 .. STACK_TOP (every word looking like a cons ptr!)
    ADD S2, S8, #0
    LDI S5, #{STACK_TOP}
_GC_STK_LP:
    CMP S2, S5
    BP _GC_STK_DONE
    S2A A1, S2
    LD S1, 0(A1)
    CALL A0, _GC_PUSH_IF_CONS
    ADD S2, S2, #1
    BR _GC_STK_LP
_GC_STK_DONE:

    ; 4. Iterative mark loop: pop cons idx from A6, trace CAR & CDR (including boxed closures!)
    LDI S6, #{GC_STACK_BASE}
_GC_MRK_LP:
    A2S S3, A6
    CMP S3, S6
    BZ _GC_MRK_DONE
    BN _GC_MRK_DONE
    AADD A6, A6, #-1
    LD S2, 0(A6)          ; S2 = cell idx (1.._RT_NEXT-1)
    ADD S3, S2, S2
    LDI S4, #{HEAP_BASE}
    ADD S3, S3, S4
    S2A A2, S3
    LD S1, 0(A2)          ; CAR
    CALL A0, _GC_PUSH_IF_CONS
    LD S1, 1(A2)          ; CDR
    CALL A0, _GC_PUSH_IF_CONS
    BR _GC_MRK_LP
_GC_MRK_DONE:

    ; 5. Sweep: rebuild _RT_FREE list from unmarked cells 1 .. _RT_NEXT-1; count live & freed
    ST S0, _RT_FREE(A0)
    ADD S4, S0, #0        ; live count
    ADD S5, S0, #0        ; reclaimed count
    LD S6, _RT_NEXT(A0)
    ADD S2, S0, #1
_GC_SWP_LP:
    CMP S2, S6
    BP _GC_SWP_DONE
    BZ _GC_SWP_DONE
    LDI S3, #{MARK_BASE}
    ADD S3, S3, S2
    S2A A1, S3
    LD S1, 0(A1)
    CMP S1, S0
    BP _GC_SWP_LIVE
    ; Unmarked -> push idx S2 onto free list!
    ADD S3, S2, S2
    LDI S1, #{HEAP_BASE}
    ADD S3, S3, S1
    S2A A1, S3
    LD S1, _RT_FREE(A0)
    ST S1, 1(A1)          ; cell.cdr = old _RT_FREE
    ST S2, _RT_FREE(A0)   ; _RT_FREE = S2
    ADD S5, S5, #1
    BR _GC_SWP_NXT
_GC_SWP_LIVE:
    ADD S4, S4, #1
_GC_SWP_NXT:
    ADD S2, S2, #1
    BR _GC_SWP_LP
_GC_SWP_DONE:
    ST S4, _RT_USED(A0)
    LD S3, _RT_GC_RUNS(A0)
    ADD S3, S3, #1
    ST S3, _RT_GC_RUNS(A0)
    ADD S1, S5, #0        ; return reclaimed count

    POP S6
    POP S5
    POP S4
    POP S3
    POP S2
    ADD S8, S7, #0
    POP S7
    S2A A7, S7
    RET #0

; Helper: if S1 is a valid unmarked cons pointer (or boxed closure pointing to cons env), mark & push!
_GC_PUSH_IF_CONS:
    PUSH S7
    ADD S7, S8, #0
    S2A A7, S7
    TRT S1, S0, #0
    BN _GCPF_RET
    BP _GCPF_RET
    SHF S3, S1, #-1       ; S3 = signed cell idx
    CMP S3, S0
    BZ _GCPF_RET
    BP _GCPF_CONS
    ; Negative boxed pointer: if TYPE == W_CLOSURE, trace its cons payload!
    NEG S3, S3
    LD S4, _RT_BOXNEXT(A0)
    CMP S3, S4
    BP _GCPF_RET
    BZ _GCPF_RET
    ADD S3, S3, S3
    LDI S4, #{BOX_BASE}
    ADD S3, S3, S4
    S2A A1, S3
    LD S4, 0(A1)
    CMP S4, #{W_CLOSURE}
    BN _GCPF_RET
    BP _GCPF_RET
    LD S1, 1(A1)          ; payload is cons (code_addr . env_alist)
    SHF S3, S1, #-1
    CMP S3, S0
    BN _GCPF_RET
    BZ _GCPF_RET
_GCPF_CONS:
    LD S4, _RT_NEXT(A0)
    CMP S3, S4
    BP _GCPF_RET
    BZ _GCPF_RET
    LDI S4, #{MARK_BASE}
    ADD S4, S4, S3
    S2A A1, S4
    LD S4, 0(A1)
    CMP S4, S0
    BP _GCPF_RET          ; already marked!
    ADD S4, S0, #1
    ST S4, 0(A1)          ; mark = 1
    ST S3, 0(A6)          ; push idx onto mark stack
    AADD A6, A6, #1
_GCPF_RET:
    ADD S8, S7, #0
    POP S7
    S2A A7, S7
    RET #0

; --- _TL_CALL_CLOSURE(closure_obj, arg_list) -> calls compiled closure with (env, arg_list) ---
_TL_CALL_CLOSURE:
    PUSH S7
    ADD S7, S8, #0
    S2A A7, S7
    LD S1, 3(A7)          ; closure_obj (boxed -3*k)
    SHF S2, S1, #-1
    NEG S2, S2
    ADD S2, S2, S2
    LDI S3, #{BOX_BASE}
    ADD S2, S2, S3
    S2A A1, S2
    LD S4, 1(A1)          ; payload = cons(code_addr_fixnum, env_alist)
    PUSH S4
    CALL A0, _TL_CAR      ; S1 = tagged code_addr
    SHF S5, S1, #-1       ; S5 = raw code word address!
    PUSH S4
    CALL A0, _TL_CDR      ; S1 = env_alist
    PUSH S1               ; arg 1 to lambda: env
    LD S2, 2(A7)          ; arg 2 to lambda: arg_list
    PUSH S2
    ; Indirect call via stack push of return address + JMP (A5)
    S2A A5, S5
    LEA A1, _TL_CC_RET(A0)
    A2S S1, A1
    PUSH S1
    JMP 0(A5)
_TL_CC_RET:
    ADD S8, S7, #0
    POP S7
    S2A A7, S7
    RET #2

; --- Kleene 3-Valued Logic Helpers (NIL=-1 -> -1, U=+5 -> 0, T=+2 -> +1) ---
_TL_TO_TRIT:
    ; S1 (LISP truth obj) -> S1 in [-1, 0, +1]
    CMP S1, #-1
    BZ _TTT_N
    CMP S1, #5
    BZ _TTT_Z
    ADD S1, S0, #1
    RET #0
_TTT_N:
    ADD S1, S0, #-1
    RET #0
_TTT_Z:
    ADD S1, S0, #0
    RET #0

_TL_FROM_TRIT:
    ; S1 in [-1, 0, +1] -> S1 in [NIL=-1, U=+5, T=+2] via single B3!
    TST S1
    B3 _TFT_0
    BR _TFT_N | NOP
_TFT_0: BR _TFT_Z | NOP
    BR _TFT_P | NOP
_TFT_N:
    ADD S1, S0, #-1 | RET #0
_TFT_Z:
    ADD S1, S0, #5  | RET #0
_TFT_P:
    ADD S1, S0, #2  | RET #0
"""


# ---------------------------------------------------------------------------
# Native TRIAD-36 LISP 1.5 Compiler
# ---------------------------------------------------------------------------

@dataclass
class TriLispCompileResult:
    assembly: str
    assembled: AssembledProgram
    symbols: List[str]
    peephole_edits: int
    parcels_packed: int
    heap_cells: int


class TriLisp36Compiler:
    """Compiles classic LISP 1.5 S-expressions directly to native TRIAD-36 assembly."""

    def __init__(self, source: str, opt: bool = True, heap_cells: int = DEFAULT_HEAP_CELLS) -> None:
        self.forms = read_sexprs(source)
        self.opt = opt
        self.heap_cells = heap_cells

        self.sym_to_id: Dict[str, int] = {
            "NIL": SYM_NIL,
            "T": SYM_T,
            "U": SYM_U,
            "*CLOSURE*": SYM_CLOSURE,
            "*FLOAT*": SYM_FLOAT,
            "*DB*": SYM_DB,
            "*SUBR*": SYM_SUBR,
        }
        self.id_to_sym: List[str] = [
            "NIL", "T", "U", "*CLOSURE*", "*FLOAT*", "*DB*", "*SUBR*"
        ]
        self.macros: Dict[str, Tuple[List[str], Optional[str], List[object]]] = {}
        self.code: List[str] = []
        self.lambda_code: List[str] = []
        self.const_pool: Dict[int, str] = {}
        self._lbl_seq = 0
        self._defined_procs: Dict[str, Tuple[str, int, bool]] = {}  # name -> (label, fixed_arity, has_rest)

    def intern(self, name: str) -> int:
        if name not in self.sym_to_id:
            self.sym_to_id[name] = len(self.id_to_sym)
            self.id_to_sym.append(name)
        return self.sym_to_id[name]

    def _mangle_fn(self, fname: str) -> str:
        sid = self.intern(fname)
        safe = re.sub(r"[^A-Za-z0-9_]", "_", fname)
        return f"_LF_{safe}_{sid}"

    def _new_lbl(self, pfx: str = "L") -> str:
        self._lbl_seq += 1
        return f"_TL_{pfx}{self._lbl_seq}"

    def _load_raw(self, buf: List[str], reg: str, val: int) -> None:
        val = wrap(val, WORD_TRITS)
        if in_range(val, 7):
            buf.append(f"ADD {reg}, S0, #{val}")
        elif in_range(val, 22):
            buf.append(f"LDI {reg}, #{val}")
        else:
            if val not in self.const_pool:
                self.const_pool[val] = f"_LK{len(self.const_pool)}"
            buf.append(f"LD {reg}, {self.const_pool[val]}(A0)")

    # ------------------------------------------------------------------
    # Compile-Time Macro Expansion (DEFMACRO)
    # ------------------------------------------------------------------

    def _expand_macros(self, form: object) -> object:
        if not isinstance(form, ConsLit):
            return form
        elems, tail = sexpr_to_pylist(form)
        if not elems or tail != Sym("NIL"):
            return form
        head = elems[0]
        if isinstance(head, Sym) and head.name == "QUOTE":
            return form
        if isinstance(head, Sym) and head.name in self.macros:
            params, rest_p, body = self.macros[head.name]
            raw_args = elems[1:]
            env: Dict[str, object] = {}
            for i, p in enumerate(params):
                env[p] = raw_args[i] if i < len(raw_args) else Sym("NIL")
            if rest_p:
                env[rest_p] = pylist_to_sexpr(raw_args[len(params) :])
            expanded: object = Sym("NIL")
            for b in body:
                expanded = self._macro_eval(b, env)
            return self._expand_macros(expanded)
        return pylist_to_sexpr([self._expand_macros(e) for e in elems])

    def _macro_eval(self, expr: object, env: Dict[str, object]) -> object:
        """Minimal compile-time LISP 1.5 evaluator for DEFMACRO bodies."""
        if isinstance(expr, (int, FltLit, DbLit)):
            return expr
        if isinstance(expr, Sym):
            if expr.name in env:
                return env[expr.name]
            return expr
        elems, _ = sexpr_to_pylist(expr)
        if not elems:
            return Sym("NIL")
        op = elems[0].name if isinstance(elems[0], Sym) else ""
        if op == "QUOTE":
            return elems[1]
        if op == "LIST":
            return pylist_to_sexpr([self._macro_eval(a, env) for a in elems[1:]])
        if op == "CONS":
            return ConsLit(self._macro_eval(elems[1], env), self._macro_eval(elems[2], env))
        if op == "CAR":
            v = self._macro_eval(elems[1], env)
            return v.car if isinstance(v, ConsLit) else Sym("NIL")
        if op == "CDR":
            v = self._macro_eval(elems[1], env)
            return v.cdr if isinstance(v, ConsLit) else Sym("NIL")
        if op == "APPEND":
            out: List[object] = []
            for a in elems[1:]:
                lst, _ = sexpr_to_pylist(self._macro_eval(a, env))
                out.extend(lst)
            return pylist_to_sexpr(out)
        if op == "IF":
            c = self._macro_eval(elems[1], env)
            return self._macro_eval(elems[2], env) if c not in (Sym("NIL"), Sym("U")) else (
                self._macro_eval(elems[3], env) if len(elems) > 3 else Sym("NIL")
            )
        raise TriLispError(f"Unsupported form {op!r} inside DEFMACRO body")

    # ------------------------------------------------------------------
    # Top-Level Compilation
    # ------------------------------------------------------------------

    def compile(self) -> TriLispCompileResult:
        # Pass 0: Register DEFMACRO definitions & pre-scan top-level DEFINE/DEFUN names
        top_forms: List[object] = []
        for f in self.forms:
            if isinstance(f, ConsLit) and isinstance(f.car, Sym) and f.car.name == "DEFMACRO":
                elems, _ = sexpr_to_pylist(f)
                sig, rest_p = sexpr_to_pylist(elems[1])
                mname = sig[0].name  # type: ignore[union-attr]
                mparams = [s.name for s in sig[1:]]  # type: ignore[union-attr]
                mrest = rest_p.name if isinstance(rest_p, Sym) and rest_p.name != "NIL" else None
                self.macros[mname] = (mparams, mrest, elems[2:])
                continue
            exp = self._expand_macros(f)
            top_forms.append(exp)
            # Pre-register top-level named functions for mutual recursion!
            if isinstance(exp, ConsLit) and isinstance(exp.car, Sym):
                h = exp.car.name
                elems, _ = sexpr_to_pylist(exp)
                if h == "DEFINE" and len(elems) >= 3 and isinstance(elems[1], ConsLit):
                    sig, tail = sexpr_to_pylist(elems[1])
                    fname = sig[0].name  # type: ignore[union-attr]
                    self.intern(fname)
                    has_rest = isinstance(tail, Sym) and tail.name != "NIL"
                    self._defined_procs[fname] = (self._mangle_fn(fname), len(sig) - 1, has_rest)
                elif h == "DEFUN" and len(elems) >= 4 and isinstance(elems[1], Sym):
                    fname = elems[1].name
                    self.intern(fname)
                    ps, tail = sexpr_to_pylist(elems[2])
                    has_rest = isinstance(tail, Sym) and tail.name != "NIL"
                    self._defined_procs[fname] = (self._mangle_fn(fname), len(ps), has_rest)

        # Startup header: initialize SP, NIL/T/U symbol cells, _RT_NSYMS, then eval top-level forms
        main_buf: List[str] = [
            "; === TRILISP-36 Compiled Native TRIAD-36 Image (UE-1000 Rev. 1.5) ===",
            f"LDI S8, #{STACK_TOP}",
            f"LDI S1, #{W_NIL}",
            f"LDI S2, #{SYMVAL_BASE}",
            "S2A A1, S2",
            "ST S1, 0(A1)",       # NIL -> NIL
            f"ADD S1, S0, #{W_T}",
            "ST S1, 1(A1)",       # T -> T
            f"ADD S1, S0, #{W_U}",
            "ST S1, 2(A1)",       # U -> U
            "ADD S1, S0, #-1",    # default result = NIL
        ]

        for form in top_forms:
            self._compile_expr(form, env={}, buf=main_buf, tail=False)

        main_buf.append("HLT")

        # Assemble full program: main + compiled lambdas/defuns + native runtime + data
        all_lines: List[str] = []
        all_lines.extend(main_buf)
        all_lines.extend(self.lambda_code)
        all_lines.extend(_build_runtime_asm(self.heap_cells).splitlines())
        all_lines.append("; === Runtime State & Constant Pool ===")
        all_lines.append("_RT_NEXT:    .WORD 1")
        all_lines.append("_RT_FREE:    .WORD 0")
        all_lines.append("_RT_USED:    .WORD 0")
        all_lines.append("_RT_BOXNEXT: .WORD 1")
        all_lines.append("_RT_GC_RUNS: .WORD 0")
        all_lines.append(f"_RT_NSYMS:   .WORD {len(self.id_to_sym)}")
        for raw_w, lbl in self.const_pool.items():
            all_lines.append(f"{lbl}: .WORD {raw_w}")

        peep = packed = 0
        if self.opt:
            all_lines, peep = _peephole_optimize(all_lines)
            all_lines, packed = _pack_dual_parcels(all_lines)

        asm_text = "\n".join(all_lines) + "\n"
        assembled = assemble(asm_text, origin=CODE_ORIGIN)
        return TriLispCompileResult(
            assembly=asm_text,
            assembled=assembled,
            symbols=list(self.id_to_sym),
            peephole_edits=peep,
            parcels_packed=packed,
            heap_cells=self.heap_cells,
        )

    # ------------------------------------------------------------------
    # Environment Representation in Compiled Frames:
    #   Direct-call functions use lexical FP slots (offset in env[name] = int)
    #   Closures (LAMBDA) also keep a cons alist in a hidden local slot `ENV_SLOT`
    #   so captured variables work cleanly across arbitrary nesting!
    # ------------------------------------------------------------------

    def _emit_lookup_sym_global(self, name: str, buf: List[str]) -> None:
        sid = self.intern(name)
        buf.append(f"LDI S2, #{SYMVAL_BASE + sid}")
        buf.append("S2A A1, S2")
        buf.append("LD S1, 0(A1)")

    def _emit_store_sym_global(self, name: str, buf: List[str]) -> None:
        sid = self.intern(name)
        buf.append(f"LDI S2, #{SYMVAL_BASE + sid}")
        buf.append("S2A A1, S2")
        buf.append("ST S1, 0(A1)")

    def _compile_quote(self, datum: object, buf: List[str]) -> None:
        if isinstance(datum, int):
            self._load_raw(buf, "S1", tag_num(datum))
            return
        if isinstance(datum, FltLit):
            self._load_raw(buf, "S1", W_FLOAT)
            buf.append("PUSH S1")
            self._load_raw(buf, "S1", float_to_tfloat(datum.val, TF36))
            buf.append("PUSH S1")
            buf.append("CALL A0, _TL_BOX")
            return
        if isinstance(datum, DbLit):
            self._load_raw(buf, "S1", W_DB)
            buf.append("PUSH S1")
            self._load_raw(buf, "S1", db_to_ticks(datum.db))
            buf.append("PUSH S1")
            buf.append("CALL A0, _TL_BOX")
            return
        if isinstance(datum, Sym):
            sid = self.intern(datum.name)
            self._load_raw(buf, "S1", tag_sym(sid))
            return
        if isinstance(datum, ConsLit):
            self._compile_quote(datum.car, buf)
            buf.append("PUSH S1")
            self._compile_quote(datum.cdr, buf)
            buf.append("PUSH S1")
            buf.append("CALL A0, _TL_CONS")
            return
        raise TriLispError(f"Cannot QUOTE datum {datum!r}")

    def _compile_closure_body(
        self,
        params: List[str],
        rest_param: Optional[str],
        body: List[object],
        label: Optional[str] = None,
        self_name: Optional[str] = None,
    ) -> str:
        """Compile a LAMBDA / DEFUN / DEFINE into a uniform closure-callable TRIAD-36 proc.

        Calling convention for ALL compiled LISP functions (uniform for direct & closure calls!):
          FP + 3: captured env alist (tagged cons list of (sym . val) pairs, or NIL)
          FP + 2: argument list (tagged cons list of evaluated arguments)
          Returns result in S1 and does `RET #2`!
        Because every function has the exact same 2-arg stack signature (env, args),
        tail calls overwrite `3(A7)` and `2(A7)` and `JMP` directly — TRUE O(1) TCO!
        """
        lbl = label or self._new_lbl("FN")
        b: List[str] = []
        b.append(f"{lbl}:")
        b.append("PUSH S7")
        b.append("ADD S7, S8, #0")
        b.append("S2A A7, S7")
        # Local slot FP - 1 holds the active lexical env alist (`__ENV`)!
        b.append("ADD S8, S8, #-1")
        b.append("LD S1, 3(A7)")      # start from captured env
        b.append("ST S1, -1(A7)")

        # Bind parameters from arg_list in 2(A7) onto lexical env alist in -1(A7)!
        if params or rest_param:
            b.append("LD S5, 2(A7)")  # S5 walks remaining arg_list (preserved via stack!)
            b.append("PUSH S5")       # FP - 2 holds remaining arg_list cursor
            for pname in params:
                sid = self.intern(pname)
                # Extract CAR(cursor) -> val, advance cursor = CDR(cursor)
                b.append("LD S1, -2(A7)")
                b.append("PUSH S1")
                b.append("CALL A0, _TL_CAR")
                b.append("PUSH S1")   # save val
                b.append("LD S1, -2(A7)")
                b.append("PUSH S1")
                b.append("CALL A0, _TL_CDR")
                b.append("ST S1, -2(A7)")
                # Bind (sym . val) onto env alist: env = cons(cons(sym, val), env)
                self._load_raw(b, "S2", tag_sym(sid))
                b.append("POP S3")    # val
                b.append("PUSH S2")
                b.append("PUSH S3")
                b.append("CALL A0, _TL_CONS")
                b.append("PUSH S1")
                b.append("LD S2, -1(A7)")
                b.append("PUSH S2")
                b.append("CALL A0, _TL_CONS")
                b.append("ST S1, -1(A7)")
            if rest_param:
                sid = self.intern(rest_param)
                self._load_raw(b, "S2", tag_sym(sid))
                b.append("LD S3, -2(A7)")
                b.append("PUSH S2")
                b.append("PUSH S3")
                b.append("CALL A0, _TL_CONS")
                b.append("PUSH S1")
                b.append("LD S2, -1(A7)")
                b.append("PUSH S2")
                b.append("CALL A0, _TL_CONS")
                b.append("ST S1, -1(A7)")
            b.append("POP S5")        # discard cursor slot

        # If LABEL recursive name given, bind self_name -> closure(lbl, env) into env!
        if self_name:
            sid = self.intern(self_name)
            self._emit_make_closure(lbl, b, has_env=True)
            b.append("PUSH S1")       # closure
            self._load_raw(b, "S2", tag_sym(sid))
            b.append("POP S3")
            b.append("PUSH S2")
            b.append("PUSH S3")
            b.append("CALL A0, _TL_CONS")
            b.append("PUSH S1")
            b.append("LD S2, -1(A7)")
            b.append("PUSH S2")
            b.append("CALL A0, _TL_CONS")
            b.append("ST S1, -1(A7)")

        # Compile body forms (last form is in tail position!)
        sub_env = {"__IN_LEX__": 1}
        for idx, bform in enumerate(body):
            is_tail = (idx == len(body) - 1)
            self._compile_expr(bform, env=sub_env, buf=b, tail=is_tail)

        b.append("ADD S8, S7, #0")
        b.append("POP S7")
        b.append("S2A A7, S7")
        b.append("RET #2")
        self.lambda_code.extend(b)
        return lbl

    def _emit_make_closure(self, lbl: str, buf: List[str], has_env: bool) -> None:
        """Emit code creating boxed closure object in S1: BOX(W_CLOSURE, CONS(tag_num(lbl), env))."""
        buf.append(f"LEA A1, {lbl}(A0)")
        buf.append("A2S S1, A1")
        buf.append("SHF S1, S1, #1")
        buf.append("ADD S1, S1, #1")   # tag_num(code_addr)
        buf.append("PUSH S1")
        if has_env:
            buf.append("LD S1, -1(A7)")
        else:
            buf.append("ADD S1, S0, #-1")  # NIL env at top level
        buf.append("PUSH S1")
        buf.append("CALL A0, _TL_CONS")    # S1 = cons(code_addr, env)
        buf.append("PUSH S1")
        self._load_raw(buf, "S2", W_CLOSURE)
        buf.append("POP S3")
        buf.append("PUSH S2")
        buf.append("PUSH S3")
        buf.append("CALL A0, _TL_BOX")     # S1 = boxed closure

    def _compile_expr(self, expr: object, env: Dict[str, int], buf: List[str], tail: bool = False) -> None:
        has_lex = "__IN_LEX__" in env

        # Literals
        if isinstance(expr, int):
            self._load_raw(buf, "S1", tag_num(expr))
            return
        if isinstance(expr, (FltLit, DbLit)):
            self._compile_quote(expr, buf)
            return
        if isinstance(expr, Sym):
            if expr.name == "NIL":
                buf.append("ADD S1, S0, #-1")
                return
            if expr.name == "T":
                buf.append("ADD S1, S0, #2")
                return
            if expr.name == "U":
                buf.append("ADD S1, S0, #5")
                return
            if expr.name.startswith('"') and expr.name.endswith('"'):
                # Strings are self-evaluating symbols (TRILISP convention!)
                self._load_raw(buf, "S1", tag_sym(self.intern(expr.name)))
                return
            # Variable lookup: check lexical env alist (-1(A7)) first if inside a proc, else global!
            sid = self.intern(expr.name)
            if has_lex:
                l_glob = self._new_lbl("VG")
                l_done = self._new_lbl("VD")
                self._load_raw(buf, "S1", tag_sym(sid))
                buf.append("PUSH S1")
                buf.append("LD S2, -1(A7)")
                buf.append("PUSH S2")
                buf.append("CALL A0, _TL_ASSOC")
                buf.append("CMP S1, #-1")
                buf.append(f"BZ {l_glob}")
                buf.append("PUSH S1")
                buf.append("CALL A0, _TL_CDR")
                buf.append(f"BR {l_done}")
                buf.append(f"{l_glob}:")
                self._emit_lookup_sym_global(expr.name, buf)
                buf.append(f"{l_done}:")
            else:
                self._emit_lookup_sym_global(expr.name, buf)
            return

        if not isinstance(expr, ConsLit):
            raise TriLispError(f"Invalid S-expression {expr!r}")

        elems, tail_node = sexpr_to_pylist(expr)
        if not elems or tail_node != Sym("NIL"):
            raise TriLispError(f"Malformed application list: {expr!r}")

        head = elems[0]
        args = elems[1:]
        hname = head.name if isinstance(head, Sym) else ""

        # ---------------- Special Forms ----------------
        if hname == "QUOTE":
            self._compile_quote(args[0], buf)
            return

        if hname == "DEFINE":
            if isinstance(args[0], ConsLit):
                sig, rest_t = sexpr_to_pylist(args[0])
                fname = sig[0].name  # type: ignore[union-attr]
                fparams = [s.name for s in sig[1:]]  # type: ignore[union-attr]
                frest = rest_t.name if isinstance(rest_t, Sym) and rest_t.name != "NIL" else None
                lbl = self._defined_procs.get(fname, (self._mangle_fn(fname), 0, False))[0]
                self._compile_closure_body(fparams, frest, args[1:], label=lbl)
                self._emit_make_closure(lbl, buf, has_env=has_lex)
                self._emit_store_sym_global(fname, buf)
                self._load_raw(buf, "S1", tag_sym(self.intern(fname)))
                return
            vname = args[0].name  # type: ignore[union-attr]
            self._compile_expr(args[1], env, buf, tail=False)
            self._emit_store_sym_global(vname, buf)
            return

        if hname == "DEFUN":
            fname = args[0].name  # type: ignore[union-attr]
            ps, rest_t = sexpr_to_pylist(args[1])
            fparams = [s.name for s in ps]  # type: ignore[union-attr]
            frest = rest_t.name if isinstance(rest_t, Sym) and rest_t.name != "NIL" else None
            lbl = self._defined_procs.get(fname, (self._mangle_fn(fname), 0, False))[0]
            self._compile_closure_body(fparams, frest, args[2:], label=lbl)
            self._emit_make_closure(lbl, buf, has_env=has_lex)
            self._emit_store_sym_global(fname, buf)
            self._load_raw(buf, "S1", tag_sym(self.intern(fname)))
            return

        if hname == "LAMBDA":
            ps, rest_t = sexpr_to_pylist(args[0])
            fparams = [s.name for s in ps]  # type: ignore[union-attr]
            frest = rest_t.name if isinstance(rest_t, Sym) and rest_t.name != "NIL" else None
            lbl = self._compile_closure_body(fparams, frest, args[1:])
            self._emit_make_closure(lbl, buf, has_env=has_lex)
            return

        if hname == "LABEL":
            # Classic LISP 1.5 (LABEL name (LAMBDA (args) body))
            lname = args[0].name  # type: ignore[union-attr]
            lam_elems, _ = sexpr_to_pylist(args[1])
            ps, rest_t = sexpr_to_pylist(lam_elems[1])
            fparams = [s.name for s in ps]  # type: ignore[union-attr]
            frest = rest_t.name if isinstance(rest_t, Sym) and rest_t.name != "NIL" else None
            lbl = self._compile_closure_body(fparams, frest, lam_elems[2:], self_name=lname)
            self._emit_make_closure(lbl, buf, has_env=has_lex)
            return

        if hname == "PROGN":
            if not args:
                buf.append("ADD S1, S0, #-1")
                return
            for i, a in enumerate(args):
                self._compile_expr(a, env, buf, tail=(tail and i == len(args) - 1))
            return

        if hname == "IF":
            # Strictly 3-valued: ONLY non-NIL AND non-U takes then-branch!
            l_else = self._new_lbl("IE")
            l_end = self._new_lbl("ID")
            self._compile_expr(args[0], env, buf, tail=False)
            buf.append("CMP S1, #-1")  # NIL?
            buf.append(f"BZ {l_else}")
            buf.append("CMP S1, #5")   # U (UNDECIDED)?
            buf.append(f"BZ {l_else}")
            self._compile_expr(args[1], env, buf, tail=tail)
            buf.append(f"BR {l_end}")
            buf.append(f"{l_else}:")
            if len(args) >= 3:
                self._compile_expr(args[2], env, buf, tail=tail)
            else:
                buf.append("ADD S1, S0, #-1")
            buf.append(f"{l_end}:")
            return

        if hname == "SIGN3":
            # Native 3-way hardware B3 branch: (SIGN3 expr neg-expr zero-expr pos-expr)
            # Works on Fixnums (<0, 0, >0) AND Kleene truth values (NIL=-1, U=0, T=+1)!
            self._compile_expr(args[0], env, buf, tail=False)
            l_fix = self._new_lbl("S3F")
            l_go = self._new_lbl("S3G")
            buf.append("TRT S1, S0, #0")
            buf.append(f"BP {l_fix}")     # fixnum (+1) -> unbox sign
            buf.append("CALL A0, _TL_TO_TRIT")
            buf.append("TST S1")
            buf.append(f"BR {l_go}")
            buf.append(f"{l_fix}:")
            buf.append("SHF S1, S1, #-1")
            buf.append("TST S1")
            buf.append(f"{l_go}:")
            l_tbl = self._new_lbl("S3T")
            l_n = self._new_lbl("S3N")
            l_z = self._new_lbl("S3Z")
            l_p = self._new_lbl("S3P")
            l_end = self._new_lbl("S3E")
            buf.append(f"B3 {l_tbl}")
            buf.append(f"BR {l_n} | NOP")
            buf.append(f"{l_tbl}: BR {l_z} | NOP")
            buf.append(f"BR {l_p} | NOP")
            buf.append(f"{l_n}:")
            self._compile_expr(args[1], env, buf, tail=tail)
            buf.append(f"BR {l_end}")
            buf.append(f"{l_z}:")
            self._compile_expr(args[2], env, buf, tail=tail)
            buf.append(f"BR {l_end}")
            buf.append(f"{l_p}:")
            self._compile_expr(args[3], env, buf, tail=tail)
            buf.append(f"{l_end}:")
            return

        if hname == "COND":
            # Desugar (COND (c1 e1...) (c2 e2...) ...) cleanly into nested IFs (preserving tail!)
            nested: object = Sym("NIL")
            for clause in reversed(args):
                c_elems, _ = sexpr_to_pylist(clause)
                test = c_elems[0]
                body_e = pylist_to_sexpr([Sym("PROGN")] + c_elems[1:]) if len(c_elems) > 2 else (
                    c_elems[1] if len(c_elems) == 2 else test
                )
                if isinstance(test, Sym) and test.name in ("T", "ELSE"):
                    nested = body_e
                else:
                    nested = pylist_to_sexpr([Sym("IF"), test, body_e, nested])
            self._compile_expr(nested, env, buf, tail=tail)
            return

        if hname == "LET":
            # Desugar (LET ((v1 e1) (v2 e2)) body...) -> ((LAMBDA (v1 v2) body...) e1 e2)
            binds, _ = sexpr_to_pylist(args[0])
            pnames: List[object] = []
            inits: List[object] = []
            for b in binds:
                be, _ = sexpr_to_pylist(b)
                pnames.append(be[0])
                inits.append(be[1])
            lam = pylist_to_sexpr([Sym("LAMBDA"), pylist_to_sexpr(pnames)] + args[1:])
            call = pylist_to_sexpr([lam] + inits)
            self._compile_expr(call, env, buf, tail=tail)
            return

        if hname == "LET*":
            binds, _ = sexpr_to_pylist(args[0])
            body_e: object = pylist_to_sexpr([Sym("PROGN")] + args[1:])
            for b in reversed(binds):
                body_e = pylist_to_sexpr([Sym("LET"), pylist_to_sexpr([b]), body_e])
            self._compile_expr(body_e, env, buf, tail=tail)
            return

        if hname == "SETQ":
            vname = args[0].name  # type: ignore[union-attr]
            self._compile_expr(args[1], env, buf, tail=False)
            if has_lex:
                # Update lexical binding cell in-place via RPLACD if found, else update global!
                sid = self.intern(vname)
                l_glob = self._new_lbl("SQG")
                l_done = self._new_lbl("SQD")
                buf.append("PUSH S1")  # save new val
                self._load_raw(buf, "S2", tag_sym(sid))
                buf.append("PUSH S2")
                buf.append("LD S3, -1(A7)")
                buf.append("PUSH S3")
                buf.append("CALL A0, _TL_ASSOC")
                buf.append("CMP S1, #-1")
                buf.append(f"BZ {l_glob}")
                # Mutate cell CDR in-place!
                buf.append("POP S2")   # new val
                buf.append("PUSH S2")  # keep new val as return
                buf.append("PUSH S1")  # cell
                buf.append("PUSH S2")  # val
                buf.append("CALL A0, _TL_RPLACD")
                buf.append("POP S1")
                buf.append(f"BR {l_done}")
                buf.append(f"{l_glob}:")
                buf.append("POP S1")
                self._emit_store_sym_global(vname, buf)
                buf.append(f"{l_done}:")
            else:
                self._emit_store_sym_global(vname, buf)
            return

        if hname == "AND":
            if not args:
                buf.append("ADD S1, S0, #2")
                return
            l_done = self._new_lbl("AND")
            for i, a in enumerate(args):
                is_last = (i == len(args) - 1)
                self._compile_expr(a, env, buf, tail=(tail and is_last))
                if not is_last:
                    buf.append("CMP S1, #-1")
                    buf.append(f"BZ {l_done}")
                    buf.append("CMP S1, #5")
                    buf.append(f"BZ {l_done}")
            buf.append(f"{l_done}:")
            return

        if hname == "OR":
            if not args:
                buf.append("ADD S1, S0, #-1")
                return
            l_done = self._new_lbl("OR")
            for i, a in enumerate(args):
                is_last = (i == len(args) - 1)
                self._compile_expr(a, env, buf, tail=(tail and is_last))
                if not is_last:
                    # Short-circuit if NOT NIL (preserves T, truthy value, or U!)
                    buf.append("CMP S1, #-1")
                    buf.append(f"BN {l_done}")
                    buf.append(f"BP {l_done}")
            buf.append(f"{l_done}:")
            return

        # ---------------- Native Hardware & List Primitives ----------------
        if self._try_compile_primitive(hname, args, env, buf):
            return

        # ---------------- General Function / Closure Application (with TCO!) ----------------
        # Evaluate function expression first, then args STRICTLY LEFT-TO-RIGHT (LISP 1.5!),
        # then pop args in reverse to build (a1 a2 ... an) cons list!
        self._compile_expr(head, env, buf, tail=False)
        buf.append("PUSH S1")  # save closure obj
        for a in args:
            self._compile_expr(a, env, buf, tail=False)
            buf.append("PUSH S1")
        buf.append("ADD S1, S0, #-1")  # NIL tail in S1
        for _ in args:
            buf.append("POP S2")       # next arg (last -> first)
            buf.append("PUSH S2")      # car = arg
            buf.append("PUSH S1")      # cdr = tail so far
            buf.append("CALL A0, _TL_CONS")
        buf.append("ADD S2, S1, #0")   # S2 = arg_list
        buf.append("POP S1")           # S1 = closure_obj

        if tail and has_lex:
            # PROPER TAIL-CALL OPTIMIZATION (O(1) stack!):
            # Unbox closure_obj -> (code_addr, new_env), overwrite 3(A7) & 2(A7), unwind frame, JMP!
            buf.append("ST S2, 2(A7)")       # overwrite arg_list in current frame!
            buf.append("SHF S3, S1, #-1")
            buf.append("NEG S3, S3")
            buf.append("ADD S3, S3, S3")
            buf.append(f"LDI S4, #{BOX_BASE}")
            buf.append("ADD S3, S3, S4")
            buf.append("S2A A1, S3")
            buf.append("LD S4, 1(A1)")       # cons(code_addr, env)
            buf.append("PUSH S4")
            buf.append("CALL A0, _TL_CDR")   # S1 = new_env
            buf.append("ST S1, 3(A7)")       # overwrite env in current frame!
            buf.append("PUSH S4")
            buf.append("CALL A0, _TL_CAR")   # S1 = tagged code_addr
            buf.append("SHF S5, S1, #-1")
            buf.append("S2A A5, S5")
            # Restore caller's FP & SP pointing at saved return PC, then JMP directly to callee!
            buf.append("ADD S8, S7, #0")
            buf.append("POP S7")
            buf.append("S2A A7, S7")
            buf.append("JMP 0(A5)")
        else:
            buf.append("PUSH S1")
            buf.append("PUSH S2")
            buf.append("CALL A0, _TL_CALL_CLOSURE")

    def _try_compile_primitive(
        self, name: str, args: List[object], env: Dict[str, int], buf: List[str]
    ) -> bool:
        # List & Pair Primitives
        if name == "CONS":
            self._compile_expr(args[0], env, buf)
            buf.append("PUSH S1")
            self._compile_expr(args[1], env, buf)
            buf.append("PUSH S1")
            buf.append("CALL A0, _TL_CONS")
            return True
        if name == "CAR":
            self._compile_expr(args[0], env, buf)
            buf.append("PUSH S1")
            buf.append("CALL A0, _TL_CAR")
            return True
        if name == "CDR":
            self._compile_expr(args[0], env, buf)
            buf.append("PUSH S1")
            buf.append("CALL A0, _TL_CDR")
            return True
        if name in ("CAAR", "CADR", "CDAR", "CDDR", "CADDR", "CADAR", "CADDDR"):
            self._compile_expr(args[0], env, buf)
            for ch in reversed(name[1:-1]):
                buf.append("PUSH S1")
                buf.append("CALL A0, _TL_CAR" if ch == "A" else "CALL A0, _TL_CDR")
            return True
        if name == "LIST":
            for a in args:
                self._compile_expr(a, env, buf)
                buf.append("PUSH S1")
            buf.append("ADD S1, S0, #-1")  # NIL tail
            for _ in args:
                buf.append("POP S2")
                buf.append("PUSH S2")
                buf.append("PUSH S1")
                buf.append("CALL A0, _TL_CONS")
            return True
        if name in ("RPLACA", "RPLACD"):
            self._compile_expr(args[0], env, buf)
            buf.append("PUSH S1")
            self._compile_expr(args[1], env, buf)
            buf.append("PUSH S1")
            buf.append(f"CALL A0, _TL_{name}")
            return True

        # Type & Equality Predicates (native 1-trit TRT!)
        if name == "EQ":
            self._compile_expr(args[0], env, buf)
            buf.append("PUSH S1")
            self._compile_expr(args[1], env, buf)
            buf.append("POP S2")
            buf.append("CMP S2, S1")
            # C==0 -> T (+2), else NIL (-1)
            l_t, l_d = self._new_lbl("EQT"), self._new_lbl("EQD")
            buf.append(f"BZ {l_t}")
            buf.append("ADD S1, S0, #-1")
            buf.append(f"BR {l_d}")
            buf.append(f"{l_t}: ADD S1, S0, #2 | NOP")
            buf.append(f"{l_d}:")
            return True
        if name == "EQUAL":
            self._compile_expr(args[0], env, buf)
            buf.append("PUSH S1")
            self._compile_expr(args[1], env, buf)
            buf.append("PUSH S1")
            buf.append("CALL A0, _TL_EQUAL")
            return True
        if name in ("NULL", "NOT"):
            self._compile_expr(args[0], env, buf)
            l_t, l_u, l_d = self._new_lbl("NT"), self._new_lbl("NU"), self._new_lbl("ND")
            buf.append("CMP S1, #-1")
            buf.append(f"BZ {l_t}")
            if name == "NOT":
                buf.append("CMP S1, #5")
                buf.append(f"BZ {l_u}")
            buf.append("ADD S1, S0, #-1")
            buf.append(f"BR {l_d}")
            buf.append(f"{l_u}: ADD S1, S0, #5 | BR {l_d}")
            buf.append(f"{l_t}: ADD S1, S0, #2 | NOP")
            buf.append(f"{l_d}:")
            return True
        if name == "NUMBERP":
            self._compile_expr(args[0], env, buf)
            l_t, l_d = self._new_lbl("NPT"), self._new_lbl("NPD")
            buf.append("TRT S1, S0, #0")
            buf.append(f"BP {l_t}")
            buf.append("ADD S1, S0, #-1")
            buf.append(f"BR {l_d}")
            buf.append(f"{l_t}: ADD S1, S0, #2 | NOP")
            buf.append(f"{l_d}:")
            return True
        if name == "SYMBOLP":
            self._compile_expr(args[0], env, buf)
            l_t, l_d = self._new_lbl("SPT"), self._new_lbl("SPD")
            buf.append("TRT S1, S0, #0")
            buf.append(f"BN {l_t}")
            buf.append("ADD S1, S0, #-1")
            buf.append(f"BR {l_d}")
            buf.append(f"{l_t}: ADD S1, S0, #2 | NOP")
            buf.append(f"{l_d}:")
            return True
        if name in ("CONSP", "ATOM"):
            self._compile_expr(args[0], env, buf)
            l_cons, l_done = self._new_lbl("CP"), self._new_lbl("CPD")
            buf.append("TRT S1, S0, #0")
            buf.append(f"BN {l_done}_F")
            buf.append(f"BP {l_done}_F")
            buf.append("CMP S1, S0")
            buf.append(f"BP {l_cons}")
            buf.append(f"{l_done}_F:")
            buf.append("ADD S1, S0, #2" if name == "ATOM" else "ADD S1, S0, #-1")
            buf.append(f"BR {l_done}")
            buf.append(f"{l_cons}:")
            buf.append("ADD S1, S0, #-1" if name == "ATOM" else "ADD S1, S0, #2")
            buf.append(f"{l_done}:")
            return True

        # Fast Tagged Fixnum Arithmetic! (3*x+1 + 3*y+1 - 1 = 3*(x+y)+1 -> zero unboxing!)
        if name in ("+", "-", "*", "/", "MOD"):
            if name == "-" and len(args) == 1:
                self._compile_expr(args[0], env, buf)
                buf.append("SHF S1, S1, #-1")
                buf.append("NEG S1, S1")
                buf.append("SHF S1, S1, #1")
                buf.append("ADD S1, S1, #1")
                return True
            self._compile_expr(args[0], env, buf)
            for a in args[1:]:
                buf.append("PUSH S1")
                self._compile_expr(a, env, buf)
                buf.append("POP S2")  # S2 = left, S1 = right
                if name == "+":
                    buf.append("ADD S1, S2, S1")
                    buf.append("SUB S1, S1, #1")
                elif name == "-":
                    buf.append("SUB S1, S2, S1")
                    buf.append("ADD S1, S1, #1")
                elif name == "*":
                    buf.append("SHF S2, S2, #-1")
                    buf.append("SHF S1, S1, #-1")
                    buf.append("MUL S1, S2, S1")
                    buf.append("SHF S1, S1, #1")
                    buf.append("ADD S1, S1, #1")
                elif name in ("/", "MOD"):
                    buf.append("SHF S2, S2, #-1")
                    buf.append("SHF S1, S1, #-1")
                    buf.append(("DIV" if name == "/" else "MOD") + " S1, S2, S1")
                    buf.append("SHF S1, S1, #1")
                    buf.append("ADD S1, S1, #1")
            return True

        # Fixnum Comparisons (exact on tagged 3*x+1 because 3*x+1 is strictly monotonic!)
        if name in ("=", "<", ">", "<=", ">="):
            self._compile_expr(args[0], env, buf)
            buf.append("PUSH S1")
            self._compile_expr(args[1], env, buf)
            buf.append("POP S2")
            buf.append("CMP S2, S1")
            tbl = {"=": -7, "<": -11, ">": 5, "<=": -5, ">=": 11}[name]
            buf.append(f"SEL S1, S0, #{tbl}")
            buf.append("CALL A0, _TL_FROM_TRIT")
            return True

        # Kleene Three-Valued Logic Primitives (KAND, KOR, KEQV, KNOT) on NIL / U / T
        if name in ("KAND", "KOR", "KEQV"):
            self._compile_expr(args[0], env, buf)
            buf.append("CALL A0, _TL_TO_TRIT")
            buf.append("PUSH S1")
            self._compile_expr(args[1], env, buf)
            buf.append("CALL A0, _TL_TO_TRIT")
            buf.append("POP S2")
            op_mn = {"KAND": "AND", "KOR": "OR", "KEQV": "EQV"}[name]
            buf.append(f"{op_mn} S1, S2, S1")
            buf.append("CALL A0, _TL_FROM_TRIT")
            return True
        if name == "KNOT":
            self._compile_expr(args[0], env, buf)
            buf.append("CALL A0, _TL_TO_TRIT")
            buf.append("NEG S1, S1")
            buf.append("CALL A0, _TL_FROM_TRIT")
            return True

        # Boxed Good Deciban Primitives (DB+, DLSE, WALD)
        if name in ("DB+", "DLSE", "WALD"):
            # Unbox payloads of two boxed DB36 values (M[BOX_BASE + 2*k + 1])
            for a in args[:2]:
                self._compile_expr(a, env, buf)
                buf.append("SHF S1, S1, #-1")
                buf.append("NEG S1, S1")
                buf.append("ADD S1, S1, S1")
                buf.append(f"LDI S2, #{BOX_BASE}")
                buf.append("ADD S1, S1, S2")
                buf.append("S2A A1, S1")
                buf.append("LD S1, 1(A1)")
                buf.append("PUSH S1")
            buf.append("POP S1")  # b
            buf.append("POP S2")  # a
            if name == "WALD":
                buf.append("DWALD S2, S1, #0")
                buf.append("SEL S1, S0, #8")
                buf.append("CALL A0, _TL_FROM_TRIT")
                return True
            buf.append(("DADD" if name == "DB+" else "DLSE") + " S1, S2, S1")
            buf.append("PUSH S1")
            self._load_raw(buf, "S2", W_DB)
            buf.append("POP S3")
            buf.append("PUSH S2")
            buf.append("PUSH S3")
            buf.append("CALL A0, _TL_BOX")
            return True

        # Low-level & Hardware Introspection Primitives (TAG, WORD, CLOCK, GC, HEAP-USED, PRINT)
        if name == "TAG":
            self._compile_expr(args[0], env, buf)
            buf.append("TRT S1, S0, #0")
            buf.append("SEL S1, S0, #8")
            buf.append("SHF S1, S1, #1")
            buf.append("ADD S1, S1, #1")
            return True
        if name == "WORD":
            self._compile_expr(args[0], env, buf)
            buf.append("SHF S1, S1, #1")
            buf.append("ADD S1, S1, #1")
            return True
        if name == "CLOCK":
            buf.append("ALGD S1, #0")
            buf.append("SHF S1, S1, #1")
            buf.append("ADD S1, S1, #1")
            return True
        if name == "GC":
            buf.append("CALL A0, _TL_GC_INNER")
            buf.append("SHF S1, S1, #1")
            buf.append("ADD S1, S1, #1")
            return True
        if name == "HEAP-USED":
            buf.append("LD S1, _RT_USED(A0)")
            buf.append("SHF S1, S1, #1")
            buf.append("ADD S1, S1, #1")
            return True
        if name == "ASSOC":
            self._compile_expr(args[0], env, buf)
            buf.append("PUSH S1")
            self._compile_expr(args[1], env, buf)
            buf.append("PUSH S1")
            buf.append("CALL A0, _TL_ASSOC")
            return True
        if name in ("PRINT", "PRIN1"):
            self._compile_expr(args[0], env, buf)
            buf.append("SVC #301" if name == "PRINT" else "SVC #302")
            return True
        if name == "APPLY":
            self._compile_expr(args[0], env, buf)
            buf.append("PUSH S1")
            self._compile_expr(args[1], env, buf)
            buf.append("PUSH S1")
            buf.append("CALL A0, _TL_CALL_CLOSURE")
            return True
        if name == "EVAL":
            # Dynamic EVAL bridged directly onto the live T-DRAM heap via SVC #305!
            self._compile_expr(args[0], env, buf)
            buf.append("SVC #305")
            return True

        return False


# ---------------------------------------------------------------------------
# T-DRAM S-Expression Formatter & Dynamic EVAL Bridge (C-EXEC SVC #301..309)
# ---------------------------------------------------------------------------

def format_tdram_sexpr(mem, word: int, id_to_sym: List[str], depth: int = 0) -> str:
    """Format any live tagged T-DRAM LISP 1.5 object as a readable S-expression."""
    if depth > 64:
        return "..."
    t = get_tag(word)
    u = untag(word)
    if t == TAG_NUM:
        return str(u)
    if t == TAG_SYM:
        if 0 <= u < len(id_to_sym):
            return id_to_sym[u]
        return f"#<SYM:{u}>"
    # TAG_PTR (0)
    if u == 0:
        return "#<EMPTY>"
    if u < 0:
        bidx = -u
        addr = BOX_BASE + 2 * bidx
        btype = mem.read(addr, supervisor=True)[0]
        payload = mem.read(addr + 1, supervisor=True)[0]
        if btype == W_FLOAT:
            fv = tfloat_to_float(payload, TF36)
            return "UNDECIDED_F" if math.isnan(fv) else f"{fv:.6g}"
        if btype == W_DB:
            return f"{ticks_to_db(payload):+.2f}db"
        if btype == W_CLOSURE:
            return "#<CLOSURE>"
        return f"#<BOX:{btype}>"
    # Positive Cons cell
    parts: List[str] = []
    cur = word
    steps = 0
    while get_tag(cur) == TAG_PTR and untag(cur) > 0 and steps < 256:
        idx = untag(cur)
        addr = HEAP_BASE + 2 * idx
        car = mem.read(addr, supervisor=True)[0]
        cdr = mem.read(addr + 1, supervisor=True)[0]
        parts.append(format_tdram_sexpr(mem, car, id_to_sym, depth + 1))
        cur = cdr
        steps += 1
    if cur == W_NIL:
        return "(" + " ".join(parts) + ")"
    return "(" + " ".join(parts) + " . " + format_tdram_sexpr(mem, cur, id_to_sym, depth + 1) + ")"


def compile_trilisp(
    source: str, opt: bool = True, heap_cells: int = DEFAULT_HEAP_CELLS
) -> TriLispCompileResult:
    return TriLisp36Compiler(source, opt=opt, heap_cells=heap_cells).compile()


@dataclass
class TriLispRunResult:
    result_word: int
    result_str: str
    printed: List[str]
    cycles: int
    instructions: int
    istack_hits: int
    heap_used: int
    gc_runs: int
    compile_result: TriLispCompileResult


def run_trilisp(
    source: str,
    machine: Optional[UE1000Machine] = None,
    opt: bool = True,
    heap_cells: int = DEFAULT_HEAP_CELLS,
    max_steps: int = 2_000_000,
) -> TriLispRunResult:
    """Compile TRILISP-36 to native TRIAD-36, load onto UE-1000, and execute on CPU 0."""
    comp = compile_trilisp(source, opt=opt, heap_cells=heap_cells)
    m = machine if machine is not None else UE1000Machine()
    m.load_program(comp.assembled, cpu_id=0)
    printed: List[str] = []

    def _svc_handler(cpu, code: int) -> bool:
        if code in (301, 302):
            s = format_tdram_sexpr(m.mem, cpu.s[1], comp.symbols)
            printed.append(s)
            return True
        if code == 305:
            # Dynamic EVAL of a live T-DRAM S-expression!
            expr_str = format_tdram_sexpr(m.mem, cpu.s[1], comp.symbols)
            sub_res = run_trilisp(expr_str, opt=True)
            cpu.s[1] = sub_res.result_word
            return True
        if code == 309:
            raise TriLispError("TRILISP-36 heap exhausted after GC")
        return False

    cpu = m.cpu0
    cpu.svc_handler = _svc_handler
    cpu.run(max_steps=max_steps)

    used_addr = comp.assembled.labels["_RT_USED"]
    gc_addr = comp.assembled.labels["_RT_GC_RUNS"]
    heap_used = m.mem.read(used_addr, supervisor=True)[0]
    gc_runs = m.mem.read(gc_addr, supervisor=True)[0]
    res_word = cpu.s[1]

    return TriLispRunResult(
        result_word=res_word,
        result_str=format_tdram_sexpr(m.mem, res_word, comp.symbols),
        printed=printed,
        cycles=cpu.cycles,
        instructions=cpu.instructions,
        istack_hits=cpu.istack_hits,
        heap_used=heap_used,
        gc_runs=gc_runs,
        compile_result=comp,
    )
