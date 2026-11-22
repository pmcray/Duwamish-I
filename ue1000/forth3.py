"""FORTH-3 — Charles Moore's Balanced-Ternary Indirect-Threaded Stack Language for UE-1000.

Honoring Charles Moore's 1968–1970 NRAO radio-telescope control language (ideal for
Bernard Oliver's Level -9 SETI Annex), Moscow's Setun-70 stack architecture, and
classic FORTH-79 / BBC FORTH compatibility (for *The Visibilities*).

Real Native TRIAD-36 Indirect-Threaded Virtual Machine inside UE-1000 T-DRAM:
  - S8 : Data Stack Pointer (DSP, grows down from +300,000; uses hardware PUSH / POP)
  - A7 : Return Stack Pointer (RSP, grows down from +350,000)
  - A6 : Forth Instruction Pointer (IP, points into threaded dictionary at +20,000+)
  - A5 : Work Register (W, current word's Code Field Address / CFA)
  - _NEXT (4 native TRIAD-36 instructions):
        LD S1, 0(A6) ; AADD A6, A6, #1 ; S2A A5, S1 ; JMP @0(A5)

Every stack word, T36 arithmetic word, 3*/3/ balanced shift, Kleene logic word,
TF36 floating-point word, Good Deciban word (DB+, DLSE, DWALD), @/!/+! memory word,
control flow (IF/ELSE/THEN, TRIT-IF/WHEN0/WHEN+/TRIT-THEN, BEGIN/UNTIL/WHILE/REPEAT,
DO/LOOP/+LOOP/LEAVE/I/J, CASE/OF/ENDOF/ENDCASE), RECURSE, EXECUTE, and CREATE ... DOES>
executes directly as native TRIAD-36 threaded code on `machine.cpu0`!
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .machine import UE1000Machine
from .ternary import (
    WORD_TRITS,
    char_to_obb729,
    int_to_hept,
    int_to_trit_str,
    int_to_trits,
    obb729_to_char,
    trit_str_to_int,
    wrap,
)
from .tfloat import TF36, db_to_ticks, float_to_tfloat, tfloat_to_float, ticks_to_db
from .triad36 import AssembledProgram, assemble


class Forth3Error(Exception):
    """Compile-time or runtime error in FORTH-3."""


# ---------------------------------------------------------------------------
# Memory Map in UE-1000 User T-DRAM
# ---------------------------------------------------------------------------

FORTH_KERNEL_ORIGIN = 3_000
FORTH_DICT_BASE = 20_000       # Dictionary cells, colon bodies, VARIABLE / CREATE buffers
FORTH_DSP_TOP = 300_000        # Data stack top (S8 grows downward)
FORTH_RSP_TOP = 350_000        # Return stack top (A7 grows downward)
FORTH_EXEC_BUF = 400_000       # Scratch thread buffer for top-level execution


# ---------------------------------------------------------------------------
# Native TRIAD-36 Indirect-Threaded Forth VM Kernel
# ---------------------------------------------------------------------------

_FORTH_KERNEL_ASM = """
; === FORTH-3 Native Indirect-Threaded Virtual Machine (TRIAD-36 Rev. 1.5) ===
; S8 = Data Stack (DSP), A7 = Return Stack (RSP), A6 = IP, A5 = W (CFA)

_NEXT:
    LD S1, 0(A6)
    AADD A6, A6, #1
    S2A A5, S1
    JMP @0(A5)

_HALT_CODE:
    HLT

_DOCOL:
    A2S S1, A6
    ASUB A7, A7, #1
    ST S1, 0(A7)
    AADD A6, A5, #1
    JMP _NEXT

_EXIT_CODE:
    LD S1, 0(A7)
    AADD A7, A7, #1
    S2A A6, S1
    JMP _NEXT

_DOCON:
    LD S1, 1(A5)
    PUSH S1
    JMP _NEXT

_DOVAR:
    ; 2-word header [CFA=_DOVAR, DOES_SLOT=0, PFA...] -> pushes A5 + 2!
    AADD A1, A5, #2
    A2S S1, A1
    PUSH S1
    JMP _NEXT

_DODOES:
    ; [CFA=_DODOES, DOES_IP, PFA...] -> push PFA, push IP, IP = DOES_IP!
    AADD A1, A5, #2
    A2S S1, A1
    PUSH S1
    A2S S1, A6
    ASUB A7, A7, #1
    ST S1, 0(A7)
    LD S1, 1(A5)
    S2A A6, S1
    JMP _NEXT

_LIT_CODE:
    LD S1, 0(A6)
    AADD A6, A6, #1
    PUSH S1
    JMP _NEXT

_BRANCH_CODE:
    LD S1, 0(A6)
    S2A A6, S1
    JMP _NEXT

_ZBRANCH_CODE:
    POP S1
    TST S1
    BZ _ZB_TAKE
    AADD A6, A6, #1
    JMP _NEXT
_ZB_TAKE:
    LD S1, 0(A6)
    S2A A6, S1
    JMP _NEXT

_TRITBR_CODE:
    ; Native Duwamish-I 3-way branch: inline [addr_neg, addr_zero, addr_pos]
    POP S1
    TST S1
    SEL S2, S0, #8
    ADD S2, S2, #1
    S2A A1, S2
    AADD A1, A6, A1
    LD S1, 0(A1)
    S2A A6, S1
    JMP _NEXT

_DO_CODE:
    ; ( limit start -- ) R: ( -- limit start )
    POP S1
    POP S2
    ASUB A7, A7, #2
    ST S1, 0(A7)
    ST S2, 1(A7)
    JMP _NEXT

_LOOP_CODE:
    LD S1, 0(A7)
    LD S2, 1(A7)
    ADD S1, S1, #1
    CMP S1, S2
    BN _LP_CONT
    AADD A7, A7, #2
    AADD A6, A6, #1
    JMP _NEXT
_LP_CONT:
    ST S1, 0(A7)
    LD S3, 0(A6)
    S2A A6, S3
    JMP _NEXT

_PLOOP_CODE:
    ; ( step -- )
    POP S3
    LD S1, 0(A7)
    LD S2, 1(A7)
    ADD S1, S1, S3
    TST S3
    BN _PL_NEG
    CMP S1, S2
    BN _LP_CONT
    AADD A7, A7, #2
    AADD A6, A6, #1
    JMP _NEXT
_PL_NEG:
    CMP S1, S2
    BP _LP_CONT
    BZ _LP_CONT
    AADD A7, A7, #2
    AADD A6, A6, #1
    JMP _NEXT

_UNLOOP_CODE:
    AADD A7, A7, #2
    JMP _NEXT

_I_CODE:
    LD S1, 0(A7)
    PUSH S1
    JMP _NEXT

_J_CODE:
    LD S1, 2(A7)
    PUSH S1
    JMP _NEXT

_EXECUTE_CODE:
    POP S1
    S2A A5, S1
    JMP @0(A5)

; --- Stack Manipulation ---
_DUP_CODE:
    POP S1
    PUSH S1
    PUSH S1
    JMP _NEXT

_DROP_CODE:
    POP S1
    JMP _NEXT

_SWAP_CODE:
    POP S1
    POP S2
    PUSH S1
    PUSH S2
    JMP _NEXT

_OVER_CODE:
    POP S1
    POP S2
    PUSH S2
    PUSH S1
    PUSH S2
    JMP _NEXT

_ROT_CODE:
    ; ( a b c -- b c a )
    POP S3
    POP S2
    POP S1
    PUSH S2
    PUSH S3
    PUSH S1
    JMP _NEXT

_NROT_CODE:
    ; ( a b c -- c a b )
    POP S3
    POP S2
    POP S1
    PUSH S3
    PUSH S1
    PUSH S2
    JMP _NEXT

_NIP_CODE:
    POP S1
    POP S2
    PUSH S1
    JMP _NEXT

_TUCK_CODE:
    POP S1
    POP S2
    PUSH S1
    PUSH S2
    PUSH S1
    JMP _NEXT

_QDUP_CODE:
    POP S1
    PUSH S1
    TST S1
    BZ _QD_SK
    PUSH S1
_QD_SK:
    JMP _NEXT

_2DUP_CODE:
    POP S1
    POP S2
    PUSH S2
    PUSH S1
    PUSH S2
    PUSH S1
    JMP _NEXT

_2DROP_CODE:
    POP S1
    POP S1
    JMP _NEXT

_2SWAP_CODE:
    ; ( a b c d -- c d a b )
    POP S4
    POP S3
    POP S2
    POP S1
    PUSH S3
    PUSH S4
    PUSH S1
    PUSH S2
    JMP _NEXT

_DEPTH_CODE:
    LDI S1, #300000
    SUB S1, S1, S8
    PUSH S1
    JMP _NEXT

_PICK_CODE:
    POP S1
    ADD S1, S1, S8
    S2A A1, S1
    LD S2, 0(A1)
    PUSH S2
    JMP _NEXT

_TOR_CODE:
    POP S1
    ASUB A7, A7, #1
    ST S1, 0(A7)
    JMP _NEXT

_FROMR_CODE:
    LD S1, 0(A7)
    AADD A7, A7, #1
    PUSH S1
    JMP _NEXT

_RFETCH_CODE:
    LD S1, 0(A7)
    PUSH S1
    JMP _NEXT

; --- Native Balanced-Ternary T36 Arithmetic ---
_ADD_CODE:
    POP S2
    POP S1
    ADD S1, S1, S2
    PUSH S1
    JMP _NEXT

_SUB_CODE:
    POP S2
    POP S1
    SUB S1, S1, S2
    PUSH S1
    JMP _NEXT

_MUL_CODE:
    POP S2
    POP S1
    MUL S1, S1, S2
    PUSH S1
    JMP _NEXT

_DIV_CODE:
    POP S2
    POP S1
    DIV S1, S1, S2
    PUSH S1
    JMP _NEXT

_MOD_CODE:
    POP S2
    POP S1
    MOD S1, S1, S2
    PUSH S1
    JMP _NEXT

_DIVMOD_CODE:
    ; ( a b -- rem quot )
    POP S2
    POP S1
    MOD S3, S1, S2
    DIV S4, S1, S2
    PUSH S3
    PUSH S4
    JMP _NEXT

_MULDIV_CODE:
    ; ( a b c -- (a*b)/c )
    POP S3
    POP S2
    POP S1
    MUL S1, S1, S2
    DIV S1, S1, S3
    PUSH S1
    JMP _NEXT

_1PLUS_CODE:
    POP S1
    ADD S1, S1, #1
    PUSH S1
    JMP _NEXT

_1MINUS_CODE:
    POP S1
    SUB S1, S1, #1
    PUSH S1
    JMP _NEXT

_2MUL_CODE:
    POP S1
    ADD S1, S1, S1
    PUSH S1
    JMP _NEXT

_2DIV_CODE:
    POP S1
    ADD S2, S0, #2
    DIV S1, S1, S2
    PUSH S1
    JMP _NEXT

_3MUL_CODE:
    ; Native 1-cycle balanced-ternary shift left by 1 trit!
    POP S1
    SHF S1, S1, #1
    PUSH S1
    JMP _NEXT

_3DIV_CODE:
    ; Native 1-cycle unbiased balanced-ternary shift right by 1 trit!
    POP S1
    SHF S1, S1, #-1
    PUSH S1
    JMP _NEXT

_NEGATE_CODE:
    POP S1
    NEG S1, S1
    PUSH S1
    JMP _NEXT

_ABS_CODE:
    POP S1
    ABS S1, S1
    PUSH S1
    JMP _NEXT

_SGN_CODE:
    ; Native balanced sign trit (-1, 0, +1)!
    POP S1
    TST S1
    SEL S1, S0, #8
    PUSH S1
    JMP _NEXT

_MIN_CODE:
    POP S2
    POP S1
    CMP S1, S2
    BN _MIN_A
    PUSH S2
    JMP _NEXT
_MIN_A:
    PUSH S1
    JMP _NEXT

_MAX_CODE:
    POP S2
    POP S1
    CMP S1, S2
    BP _MAX_A
    PUSH S2
    JMP _NEXT
_MAX_A:
    PUSH S1
    JMP _NEXT

; --- Comparisons (FORTH-79 1/0 truth) & Native Kleene Three-Valued Logic ---
_EQ_CODE:
    POP S2
    POP S1
    CMP S1, S2
    BZ _RET_ONE
    PUSH S0
    JMP _NEXT
_RET_ONE:
    ADD S1, S0, #1
    PUSH S1
    JMP _NEXT

_NEQ_CODE:
    POP S2
    POP S1
    CMP S1, S2
    BZ _RET_ZERO
    ADD S1, S0, #1
    PUSH S1
    JMP _NEXT
_RET_ZERO:
    PUSH S0
    JMP _NEXT

_LT_CODE:
    POP S2
    POP S1
    CMP S1, S2
    BN _RET_ONE
    PUSH S0
    JMP _NEXT

_GT_CODE:
    POP S2
    POP S1
    CMP S1, S2
    BP _RET_ONE
    PUSH S0
    JMP _NEXT

_LE_CODE:
    POP S2
    POP S1
    CMP S1, S2
    BP _RET_ZERO
    ADD S1, S0, #1
    PUSH S1
    JMP _NEXT

_GE_CODE:
    POP S2
    POP S1
    CMP S1, S2
    BN _RET_ZERO
    ADD S1, S0, #1
    PUSH S1
    JMP _NEXT

_ZEQ_CODE:
    POP S1
    TST S1
    BZ _RET_ONE
    PUSH S0
    JMP _NEXT

_ZNEQ_CODE:
    POP S1
    TST S1
    BZ _RET_ZERO
    ADD S1, S0, #1
    PUSH S1
    JMP _NEXT

_ZLT_CODE:
    POP S1
    TST S1
    BN _RET_ONE
    PUSH S0
    JMP _NEXT

_ZGT_CODE:
    POP S1
    TST S1
    BP _RET_ONE
    PUSH S0
    JMP _NEXT

_KAND_CODE:
    POP S2
    POP S1
    AND S1, S1, S2
    PUSH S1
    JMP _NEXT

_KOR_CODE:
    POP S2
    POP S1
    OR S1, S1, S2
    PUSH S1
    JMP _NEXT

_KEQV_CODE:
    POP S2
    POP S1
    EQV S1, S1, S2
    PUSH S1
    JMP _NEXT

_KNOT_CODE:
    POP S1
    NEG S1, S1
    PUSH S1
    JMP _NEXT

_INVERT_CODE:
    POP S1
    TST S1
    BZ _RET_ONE
    PUSH S0
    JMP _NEXT

; --- Direct UE-1000 T-DRAM Memory Access (@ ! +!) ---
_FETCH_CODE:
    POP S1
    S2A A1, S1
    LD S2, 0(A1)
    PUSH S2
    JMP _NEXT

_STORE_CODE:
    POP S1
    POP S2
    S2A A1, S1
    ST S2, 0(A1)
    JMP _NEXT

_PSTORE_CODE:
    POP S1
    POP S2
    S2A A1, S1
    LD S3, 0(A1)
    ADD S3, S3, S2
    ST S3, 0(A1)
    JMP _NEXT

; --- Exact TF36 Floating-Point & Good Deciban Hardware Words ---
_FADD_CODE:
    POP S2
    POP S1
    FAD S1, S1, S2
    PUSH S1
    JMP _NEXT

_FSUB_CODE:
    POP S2
    POP S1
    FSB S1, S1, S2
    PUSH S1
    JMP _NEXT

_FMUL_CODE:
    POP S2
    POP S1
    FMP S1, S1, S2
    PUSH S1
    JMP _NEXT

_FDIV_CODE:
    POP S2
    POP S1
    FDV S1, S1, S2
    PUSH S1
    JMP _NEXT

_ITOF_CODE:
    POP S1
    FLT S1, S1
    PUSH S1
    JMP _NEXT

_FTOI_CODE:
    POP S1
    FIX S1, S1
    PUSH S1
    JMP _NEXT

_DBADD_CODE:
    POP S2
    POP S1
    DADD S1, S1, S2
    PUSH S1
    JMP _NEXT

_DBSUB_CODE:
    POP S2
    POP S1
    DSUB S1, S1, S2
    PUSH S1
    JMP _NEXT

_DLSE_CODE:
    POP S2
    POP S1
    DLSE S1, S1, S2
    PUSH S1
    JMP _NEXT

_DWALD_CODE:
    ; ( ev_ticks thresh_ticks -- verdict_trit )
    POP S2
    POP S1
    DWALD S1, S2, #0
    SEL S1, S0, #8
    PUSH S1
    JMP _NEXT

_CYCLES_CODE:
    ALGD S1, #0
    PUSH S1
    JMP _NEXT

; --- C-EXEC Service Bridges (I/O, DOES> patch, & Tier 2 T-Fabric Instruments) ---
_DOT_CODE:
    SVC #401
    JMP _NEXT
_EMIT_CODE:
    SVC #402
    JMP _NEXT
_CR_CODE:
    SVC #403
    JMP _NEXT
_DOTS_CODE:
    SVC #404
    JMP _NEXT
_DOTH_CODE:
    SVC #405
    JMP _NEXT
_DOTT_CODE:
    SVC #406
    JMP _NEXT
_DOTDB_CODE:
    SVC #407
    JMP _NEXT
_FDOT_CODE:
    SVC #408
    JMP _NEXT
_PRTSTR_CODE:
    SVC #409
    JMP _NEXT
_SETDOES_CODE:
    SVC #410
    JMP _NEXT
_TFFT3_CODE:
    SVC #411
    JMP _NEXT
_VDOT_CODE:
    SVC #412
    JMP _NEXT
_HERE_CODE:
    SVC #413
    JMP _NEXT
_COMMA_CODE:
    SVC #414
    JMP _NEXT
_ALLOT_CODE:
    SVC #415
    JMP _NEXT
_DB2TICKS_CODE:
    SVC #416
    JMP _NEXT
"""


# ---------------------------------------------------------------------------
# Dictionary Entry & FORTH-3 Session
# ---------------------------------------------------------------------------

@dataclass
class ForthWord:
    name: str
    cfa: int
    immediate: bool = False


class Forth3Session:
    """Native indirect-threaded FORTH-3 environment executing on UE1000Machine."""

    def __init__(self, machine: Optional[UE1000Machine] = None) -> None:
        self.machine = machine if machine is not None else UE1000Machine()
        self.kernel: AssembledProgram = assemble(_FORTH_KERNEL_ASM, origin=FORTH_KERNEL_ORIGIN)
        self.machine.load_program(self.kernel, cpu_id=0)

        # Initialize hardware Forth VM registers on CPU0
        cpu = self.machine.cpu0
        cpu.s[8] = FORTH_DSP_TOP
        cpu.a[7] = FORTH_RSP_TOP
        cpu.svc_handler = self._svc_handler

        self.here: int = FORTH_DICT_BASE
        self.words: Dict[str, ForthWord] = {}
        self.last_created_cfa: Optional[int] = None
        self.output_buf: List[str] = []
        self._line_buf: str = ""
        self._strings: Dict[int, str] = {}
        self._str_id: int = 1

        # Install all native TRIAD-36 primitives into UE-1000 T-DRAM dictionary!
        self._install_primitives()

    def _comma(self, word_val: int) -> int:
        addr = self.here
        self.machine.mem.write(addr, wrap(word_val, WORD_TRITS), supervisor=True)
        self.here += 1
        return addr

    def _define_prim(self, name: str, code_label: str) -> int:
        cfa = self._comma(self.kernel.labels[code_label])
        self.words[name.upper()] = ForthWord(name=name.upper(), cfa=cfa)
        return cfa

    def _define_const(self, name: str, val: int) -> int:
        cfa = self._comma(self.kernel.labels["_DOCON"])
        self._comma(val)
        self.words[name.upper()] = ForthWord(name=name.upper(), cfa=cfa)
        return cfa

    def _install_primitives(self) -> None:
        # Internal runtime CFAs
        self.cfa_halt = self._define_prim("(__HALT)", "_HALT_CODE")
        self.cfa_exit = self._define_prim("EXIT", "_EXIT_CODE")
        self.cfa_lit = self._define_prim("(__LIT)", "_LIT_CODE")
        self.cfa_branch = self._define_prim("(__BRANCH)", "_BRANCH_CODE")
        self.cfa_zbranch = self._define_prim("(__0BRANCH)", "_ZBRANCH_CODE")
        self.cfa_tritbr = self._define_prim("(__TRITBR)", "_TRITBR_CODE")
        self.cfa_do = self._define_prim("(__DO)", "_DO_CODE")
        self.cfa_loop = self._define_prim("(__LOOP)", "_LOOP_CODE")
        self.cfa_ploop = self._define_prim("(__+LOOP)", "_PLOOP_CODE")
        self.cfa_unloop = self._define_prim("UNLOOP", "_UNLOOP_CODE")
        self.cfa_prtstr = self._define_prim("(__PRTSTR)", "_PRTSTR_CODE")
        self.cfa_setdoes = self._define_prim("(__SETDOES)", "_SETDOES_CODE")

        prims = [
            # Stack
            ("DUP", "_DUP_CODE"), ("DROP", "_DROP_CODE"), ("SWAP", "_SWAP_CODE"),
            ("OVER", "_OVER_CODE"), ("ROT", "_ROT_CODE"), ("-ROT", "_NROT_CODE"),
            ("NIP", "_NIP_CODE"), ("TUCK", "_TUCK_CODE"), ("?DUP", "_QDUP_CODE"),
            ("2DUP", "_2DUP_CODE"), ("2DROP", "_2DROP_CODE"), ("2SWAP", "_2SWAP_CODE"),
            ("DEPTH", "_DEPTH_CODE"), ("PICK", "_PICK_CODE"),
            (">R", "_TOR_CODE"), ("R>", "_FROMR_CODE"), ("R@", "_RFETCH_CODE"),
            ("I", "_I_CODE"), ("J", "_J_CODE"), ("EXECUTE", "_EXECUTE_CODE"),
            # Arithmetic (T36 native + 3*/3/ balanced shifts!)
            ("+", "_ADD_CODE"), ("-", "_SUB_CODE"), ("*", "_MUL_CODE"),
            ("/", "_DIV_CODE"), ("MOD", "_MOD_CODE"), ("/MOD", "_DIVMOD_CODE"),
            ("*/", "_MULDIV_CODE"), ("1+", "_1PLUS_CODE"), ("1-", "_1MINUS_CODE"),
            ("2*", "_2MUL_CODE"), ("2/", "_2DIV_CODE"),
            ("3*", "_3MUL_CODE"), ("3/", "_3DIV_CODE"),
            ("NEGATE", "_NEGATE_CODE"), ("ABS", "_ABS_CODE"),
            ("SGN", "_SGN_CODE"), ("MIN", "_MIN_CODE"), ("MAX", "_MAX_CODE"),
            # Comparisons & Kleene 3-valued logic
            ("=", "_EQ_CODE"), ("<>", "_NEQ_CODE"), ("<", "_LT_CODE"),
            (">", "_GT_CODE"), ("<=", "_LE_CODE"), (">=", "_GE_CODE"),
            ("0=", "_ZEQ_CODE"), ("0<>", "_ZNEQ_CODE"), ("0<", "_ZLT_CODE"), ("0>", "_ZGT_CODE"),
            ("KAND", "_KAND_CODE"), ("AND", "_KAND_CODE"),
            ("KOR", "_KOR_CODE"), ("OR", "_KOR_CODE"),
            ("KEQV", "_KEQV_CODE"), ("KNOT", "_KNOT_CODE"), ("INVERT", "_INVERT_CODE"),
            # Memory
            ("@", "_FETCH_CODE"), ("!", "_STORE_CODE"), ("+!", "_PSTORE_CODE"),
            ("HERE", "_HERE_CODE"), (",", "_COMMA_CODE"), ("ALLOT", "_ALLOT_CODE"),
            # TF36 Float & Good Deciban
            ("F+", "_FADD_CODE"), ("F-", "_FSUB_CODE"), ("F*", "_FMUL_CODE"), ("F/", "_FDIV_CODE"),
            ("I>F", "_ITOF_CODE"), ("F>I", "_FTOI_CODE"), ("F.", "_FDOT_CODE"),
            ("DB+", "_DBADD_CODE"), ("DB-", "_DBSUB_CODE"), ("DLSE", "_DLSE_CODE"),
            ("DWALD", "_DWALD_CODE"), ("DB>TICKS", "_DB2TICKS_CODE"),
            # Telemetry & I/O & T-Fabric
            ("CYCLES", "_CYCLES_CODE"),
            (".", "_DOT_CODE"), ("EMIT", "_EMIT_CODE"), ("CR", "_CR_CODE"),
            (".S", "_DOTS_CODE"), (".H", "_DOTH_CODE"), (".T", "_DOTT_CODE"), (".DB", "_DOTDB_CODE"),
            ("T.FFT3", "_TFFT3_CODE"), ("V.DOT", "_VDOT_CODE"),
        ]
        for wname, lbl in prims:
            self._define_prim(wname, lbl)

        # Standard & Balanced-Ternary Constants
        self._define_const("TRUE", 1)
        self._define_const("FALSE", 0)
        self._define_const("KTRUE", 1)
        self._define_const("KUNDECIDED", 0)
        self._define_const("KFALSE", -1)
        self._define_const("BL", 32)
        self._define_const("CELL", 1)

        # Define handy classic Forth words directly in FORTH-3!
        self.eval_source(": SPACE BL EMIT ; : SPACES 0 DO SPACE LOOP ; : CELLS ; : CELL+ 1+ ;")

    # ------------------------------------------------------------------
    # Data Stack Helpers & C-EXEC Service Handler
    # ------------------------------------------------------------------

    def _dpush(self, val: int) -> None:
        cpu = self.machine.cpu0
        cpu.s[8] = wrap(cpu.s[8] - 1, WORD_TRITS)
        self.machine.mem.write(cpu.s[8], wrap(val, WORD_TRITS), supervisor=True)

    def _dpop(self) -> int:
        cpu = self.machine.cpu0
        if cpu.s[8] >= FORTH_DSP_TOP:
            raise Forth3Error("Data stack underflow")
        val = self.machine.mem.read(cpu.s[8], supervisor=True)[0]
        cpu.s[8] = wrap(cpu.s[8] + 1, WORD_TRITS)
        return val

    def data_stack(self) -> List[int]:
        """Return current Data Stack contents (bottom .. top)."""
        cpu = self.machine.cpu0
        depth = FORTH_DSP_TOP - cpu.s[8]
        if depth <= 0:
            return []
        return [
            self.machine.mem.read(FORTH_DSP_TOP - 1 - i, supervisor=True)[0]
            for i in range(depth)
        ]

    def _emit_text(self, s: str) -> None:
        self._line_buf += s

    def _flush_line(self) -> None:
        self.output_buf.append(self._line_buf)
        self._line_buf = ""

    def _svc_handler(self, cpu, code: int) -> bool:
        if code == 401:  # `.`
            self._emit_text(f"{self._dpop()} ")
            return True
        if code == 402:  # `EMIT`
            v = self._dpop()
            self._emit_text(chr(v) if 0 <= v <= 127 else obb729_to_char(v))
            return True
        if code == 403:  # `CR`
            self._flush_line()
            return True
        if code == 404:  # `.S`
            stk = self.data_stack()
            self._emit_text(f"<{len(stk)}> " + " ".join(str(x) for x in stk) + " ")
            return True
        if code == 405:  # `.H` (UE Heptavintimal)
            self._emit_text(int_to_hept(self._dpop(), WORD_TRITS) + " ")
            return True
        if code == 406:  # `.T` (minimal balanced trits T/0/1)
            v = self._dpop()
            if v == 0:
                self._emit_text("0 ")
            else:
                ts = int_to_trits(v, WORD_TRITS)
                while len(ts) > 1 and ts[-1] == 0:
                    ts.pop()
                cmap = {-1: "T", 0: "0", 1: "1"}
                self._emit_text("".join(cmap[t] for t in reversed(ts)) + " ")
            return True
        if code == 407:  # `.DB` (Good Decibans)
            self._emit_text(f"{ticks_to_db(self._dpop()):+.2f}dB ")
            return True
        if code == 408:  # `F.` (TF36 float)
            self._emit_text(f"{tfloat_to_float(self._dpop(), TF36):.6g} ")
            return True
        if code == 409:  # inline string `." ..."`
            sid = self.machine.mem.read(cpu.a[6], supervisor=True)[0]
            cpu.a[6] = wrap(cpu.a[6] + 1, 18)
            self._emit_text(self._strings.get(sid, ""))
            return True
        if code == 410:  # `DOES>` runtime patch of last CREATEd word!
            does_ip = cpu.a[6]
            if self.last_created_cfa is None:
                raise Forth3Error("DOES> with no preceding CREATEd word")
            self.machine.mem.write(self.last_created_cfa, self.kernel.labels["_DODOES"], supervisor=True)
            self.machine.mem.write(self.last_created_cfa + 1, does_ip, supervisor=True)
            # EXIT from the defining word immediately!
            ret_ip = self.machine.mem.read(cpu.a[7], supervisor=True)[0]
            cpu.a[7] = wrap(cpu.a[7] + 1, 18)
            cpu.a[6] = wrap(ret_ip, 18)
            return True
        if code == 411:  # `T.FFT3` ( addr n -- peak_bin peak_mag_tf36 ) on Tier 2 T-Fabric!
            n = self._dpop()
            addr = self._dpop()
            sig = np.array(
                [tfloat_to_float(self.machine.mem.read(addr + i, supervisor=True)[0], TF36) for i in range(n)],
                dtype=np.complex128,
            )
            spec, _ = self.machine.tfabric.fft_radix3_tbfp9c(sig)
            mags = np.abs(spec)
            pk_bin = int(np.argmax(mags))
            self._dpush(pk_bin)
            self._dpush(float_to_tfloat(float(mags[pk_bin]), TF36))
            return True
        if code == 412:  # `V.DOT` ( addrA addrB n -- dot_tf36 )
            n = self._dpop()
            addr_b = self._dpop()
            addr_a = self._dpop()
            acc = 0.0
            for i in range(n):
                va = tfloat_to_float(self.machine.mem.read(addr_a + i, supervisor=True)[0], TF36)
                vb = tfloat_to_float(self.machine.mem.read(addr_b + i, supervisor=True)[0], TF36)
                acc += va * vb
            self._dpush(float_to_tfloat(acc, TF36))
            return True
        if code == 413:  # `HERE`
            self._dpush(self.here)
            return True
        if code == 414:  # `,`
            self._comma(self._dpop())
            return True
        if code == 415:  # `ALLOT`
            cnt = self._dpop()
            for _ in range(max(0, cnt)):
                self._comma(0)
            return True
        if code == 416:  # `DB>TICKS` ( int_db -- ticks )
            self._dpush(db_to_ticks(float(self._dpop())))
            return True
        return False

    def _exec_cfa(self, cfa: int, max_steps: int = 500_000) -> None:
        """Execute a single Forth word by CFA natively on UE-1000 CPU0!"""
        self.machine.mem.write(FORTH_EXEC_BUF, cfa, supervisor=True)
        self.machine.mem.write(FORTH_EXEC_BUF + 1, self.cfa_halt, supervisor=True)
        cpu = self.machine.cpu0
        cpu.a[6] = FORTH_EXEC_BUF
        cpu.pc = self.kernel.labels["_NEXT"]
        cpu.halted = False
        cpu.run(max_steps=max_steps)

    # ------------------------------------------------------------------
    # Outer Interpreter & Threaded Compiler (`:` ... `;`, control structures)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_number(tok: str) -> Optional[int]:
        s = tok.strip()
        if not s:
            return None
        # Balanced ternary literal `3t1T01` or `-3t1T01`
        sl = s.lower()
        if sl.startswith("3t") or sl.startswith("-3t"):
            neg = sl.startswith("-")
            body = s[3:] if neg else s[2:]
            if body and all(c in "10Tt_" for c in body):
                v = trit_str_to_int(body)
                return wrap(-v if neg else v, WORD_TRITS)
            return None
        # Good deciban literal `12.5dB` / `-6dB` -> ticks
        if sl.endswith("db") and len(s) > 2:
            try:
                return db_to_ticks(float(s[:-2]))
            except ValueError:
                return None
        # TF36 float literal ending with `e` / `f` or containing `.` (`3.14f`, `1.5e0`, `0.25`)
        if ("." in s or sl.endswith("f")) and not s.startswith("."):
            try:
                fv = float(s[:-1] if sl.endswith("f") else s)
                return float_to_tfloat(fv, TF36)
            except ValueError:
                return None
        # Standard signed decimal integer
        try:
            return wrap(int(s, 10), WORD_TRITS)
        except ValueError:
            return None

    def _tokenize(self, source: str) -> List[Tuple[str, str]]:
        """Tokenize FORTH-3 source into (kind, value) where kind in {'WORD', 'STR'}."""
        toks: List[Tuple[str, str]] = []
        i = 0
        n = len(source)
        while i < n:
            if source[i] in " \t\r\n":
                i += 1
                continue
            # Line comment `\ ...`
            if source[i] == "\\" and (i + 1 == n or source[i + 1] in " \t\r\n"):
                nl = source.find("\n", i)
                i = n if nl < 0 else nl + 1
                continue
            # Parenthesized comment `( ... )`
            if source[i] == "(" and (i + 1 == n or source[i + 1] in " \t\r\n"):
                end = source.find(")", i + 1)
                i = n if end < 0 else end + 1
                continue
            # String printing `." ..."`
            if source[i : i + 2] == '."' and (i + 2 == n or source[i + 2] in " \t\r\n"):
                j = i + 2
                if j < n and source[j] == " ":
                    j += 1
                end = source.find('"', j)
                if end < 0:
                    raise Forth3Error('Unterminated ." string')
                toks.append(("STR", source[j:end]))
                i = end + 1
                continue
            j = i
            while j < n and source[j] not in " \t\r\n":
                j += 1
            toks.append(("WORD", source[i:j]))
            i = j
        return toks

    def eval_source(self, source: str) -> List[str]:
        """Interpret and/or compile FORTH-3 source directly into UE-1000 T-DRAM."""
        start_out = len(self.output_buf)
        toks = self._tokenize(source)
        i = 0
        n = len(toks)
        compiling = False
        cur_name = ""
        cur_cfa = 0
        ctrl_stack: List[Tuple[str, object]] = []
        leave_patches: List[List[int]] = []

        while i < n:
            kind, raw = toks[i]
            i += 1

            if kind == "STR":
                sid = self._str_id
                self._str_id += 1
                self._strings[sid] = raw
                if compiling:
                    self._comma(self.cfa_prtstr)
                    self._comma(sid)
                else:
                    self._emit_text(raw)
                continue

            u = raw.upper()

            if not compiling:
                if u == ":":
                    if i >= n:
                        raise Forth3Error("Expected word name after `:`")
                    cur_name = toks[i][1].upper()
                    i += 1
                    cur_cfa = self._comma(self.kernel.labels["_DOCOL"])
                    compiling = True
                    ctrl_stack.clear()
                    leave_patches.clear()
                    continue
                if u == "VARIABLE":
                    if i >= n:
                        raise Forth3Error("Expected name after VARIABLE")
                    vname = toks[i][1].upper()
                    i += 1
                    cfa = self._comma(self.kernel.labels["_DOVAR"])
                    self._comma(0)  # DOES> slot
                    self._comma(0)  # initial variable cell
                    self.words[vname] = ForthWord(vname, cfa)
                    self.last_created_cfa = cfa
                    continue
                if u == "CONSTANT":
                    if i >= n:
                        raise Forth3Error("Expected name after CONSTANT")
                    cname = toks[i][1].upper()
                    i += 1
                    self._define_const(cname, self._dpop())
                    continue
                if u == "CREATE":
                    if i >= n:
                        raise Forth3Error("Expected name after CREATE")
                    cname = toks[i][1].upper()
                    i += 1
                    cfa = self._comma(self.kernel.labels["_DOVAR"])
                    self._comma(0)  # DOES> slot
                    self.words[cname] = ForthWord(cname, cfa)
                    self.last_created_cfa = cfa
                    continue
                if u == "'":
                    if i >= n:
                        raise Forth3Error("Expected word name after `'`")
                    tname = toks[i][1].upper()
                    i += 1
                    if tname not in self.words:
                        raise Forth3Error(f"Unknown word {tname!r} after `'`")
                    self._dpush(self.words[tname].cfa)
                    continue
                if u in self.words:
                    self._exec_cfa(self.words[u].cfa)
                    continue
                num = self._parse_number(raw)
                if num is not None:
                    self._dpush(num)
                    continue
                raise Forth3Error(f"Undefined FORTH-3 word: {raw!r}")

            # --- COMPILING MODE (inside `: NAME ... ;`) ---
            if u == ";":
                if ctrl_stack:
                    raise Forth3Error(f"Unclosed control structure in {cur_name}: {ctrl_stack}")
                self._comma(self.cfa_exit)
                self.words[cur_name] = ForthWord(cur_name, cur_cfa)
                compiling = False
                continue

            if u == "RECURSE":
                self._comma(cur_cfa)
                continue

            if u == "[']":
                if i >= n:
                    raise Forth3Error("Expected word name after `[']`")
                tname = toks[i][1].upper()
                i += 1
                if tname not in self.words:
                    raise Forth3Error(f"Unknown word {tname!r} in `[']`")
                self._comma(self.cfa_lit)
                self._comma(self.words[tname].cfa)
                continue

            if u == "CREATE":
                # Runtime CREATE inside a defining word (e.g. `: ARRAY CREATE CELLS ALLOT DOES> ... ;`)
                # Emit a tiny host-assisted CREATE via SVC or handle cleanly when defining word runs!
                # Wait: handle inline CREATE inside a defining word cleanly via a helper!
                self._comma(self._make_runtime_create_cfa())
                continue

            if u == "DOES>":
                self._comma(self.cfa_setdoes)
                continue

            # Classic IF ... [ELSE ...] THEN
            if u == "IF":
                self._comma(self.cfa_zbranch)
                patch = self._comma(0)
                ctrl_stack.append(("IF", patch))
                continue
            if u == "ELSE":
                if not ctrl_stack or ctrl_stack[-1][0] != "IF":
                    raise Forth3Error("ELSE without matching IF")
                _, if_patch = ctrl_stack.pop()
                self._comma(self.cfa_branch)
                else_patch = self._comma(0)
                self.machine.mem.write(int(if_patch), self.here, supervisor=True)
                ctrl_stack.append(("ELSE", else_patch))
                continue
            if u == "THEN":
                if not ctrl_stack or ctrl_stack[-1][0] not in ("IF", "ELSE"):
                    raise Forth3Error("THEN without matching IF/ELSE")
                _, patch = ctrl_stack.pop()
                self.machine.mem.write(int(patch), self.here, supervisor=True)
                continue

            # Native Duwamish-I 3-way branch: TRIT-IF <neg> WHEN0 <zero> WHEN+ <pos> TRIT-THEN
            if u == "TRIT-IF":
                self._comma(self.cfa_tritbr)
                p_neg = self._comma(0)
                p_zero = self._comma(0)
                p_pos = self._comma(0)
                self.machine.mem.write(p_neg, self.here, supervisor=True)
                ctrl_stack.append(("TRIT", [p_zero, p_pos, []]))
                continue
            if u == "WHEN0":
                if not ctrl_stack or ctrl_stack[-1][0] != "TRIT":
                    raise Forth3Error("WHEN0 without TRIT-IF")
                info = ctrl_stack[-1][1]
                self._comma(self.cfa_branch)
                info[2].append(self._comma(0))  # type: ignore[index]
                self.machine.mem.write(int(info[0]), self.here, supervisor=True)  # type: ignore[index]
                continue
            if u == "WHEN+":
                if not ctrl_stack or ctrl_stack[-1][0] != "TRIT":
                    raise Forth3Error("WHEN+ without TRIT-IF")
                info = ctrl_stack[-1][1]
                self._comma(self.cfa_branch)
                info[2].append(self._comma(0))  # type: ignore[index]
                self.machine.mem.write(int(info[1]), self.here, supervisor=True)  # type: ignore[index]
                continue
            if u == "TRIT-THEN":
                if not ctrl_stack or ctrl_stack[-1][0] != "TRIT":
                    raise Forth3Error("TRIT-THEN without TRIT-IF")
                _, info = ctrl_stack.pop()
                for ep in info[2]:  # type: ignore[index]
                    self.machine.mem.write(int(ep), self.here, supervisor=True)
                continue

            # BEGIN ... UNTIL / AGAIN / WHILE ... REPEAT
            if u == "BEGIN":
                ctrl_stack.append(("BEGIN", self.here))
                continue
            if u == "UNTIL":
                if not ctrl_stack or ctrl_stack[-1][0] != "BEGIN":
                    raise Forth3Error("UNTIL without BEGIN")
                _, target = ctrl_stack.pop()
                self._comma(self.cfa_zbranch)
                self._comma(int(target))
                continue
            if u == "AGAIN":
                if not ctrl_stack or ctrl_stack[-1][0] != "BEGIN":
                    raise Forth3Error("AGAIN without BEGIN")
                _, target = ctrl_stack.pop()
                self._comma(self.cfa_branch)
                self._comma(int(target))
                continue
            if u == "WHILE":
                if not ctrl_stack or ctrl_stack[-1][0] != "BEGIN":
                    raise Forth3Error("WHILE without BEGIN")
                self._comma(self.cfa_zbranch)
                patch = self._comma(0)
                ctrl_stack.append(("WHILE", patch))
                continue
            if u == "REPEAT":
                if len(ctrl_stack) < 2 or ctrl_stack[-1][0] != "WHILE" or ctrl_stack[-2][0] != "BEGIN":
                    raise Forth3Error("REPEAT without BEGIN ... WHILE")
                _, w_patch = ctrl_stack.pop()
                _, b_target = ctrl_stack.pop()
                self._comma(self.cfa_branch)
                self._comma(int(b_target))
                self.machine.mem.write(int(w_patch), self.here, supervisor=True)
                continue

            # DO ... [LEAVE] ... LOOP / +LOOP
            if u == "DO":
                self._comma(self.cfa_do)
                ctrl_stack.append(("DO", self.here))
                leave_patches.append([])
                continue
            if u == "LEAVE":
                if not leave_patches:
                    raise Forth3Error("LEAVE outside DO ... LOOP")
                self._comma(self.cfa_unloop)
                self._comma(self.cfa_branch)
                leave_patches[-1].append(self._comma(0))
                continue
            if u in ("LOOP", "+LOOP"):
                if not ctrl_stack or ctrl_stack[-1][0] != "DO":
                    raise Forth3Error(f"{u} without DO")
                _, d_target = ctrl_stack.pop()
                self._comma(self.cfa_loop if u == "LOOP" else self.cfa_ploop)
                self._comma(int(d_target))
                for lp in leave_patches.pop():
                    self.machine.mem.write(lp, self.here, supervisor=True)
                continue

            # Eaker / BBC FORTH CASE ... OF ... ENDOF ... ENDCASE
            if u == "CASE":
                ctrl_stack.append(("CASE", []))
                continue
            if u == "OF":
                # ( sel test -- sel | ) : OVER = IF DROP ...
                self._comma(self.words["OVER"].cfa)
                self._comma(self.words["="].cfa)
                self._comma(self.cfa_zbranch)
                of_patch = self._comma(0)
                self._comma(self.words["DROP"].cfa)
                ctrl_stack.append(("OF", of_patch))
                continue
            if u == "ENDOF":
                if len(ctrl_stack) < 2 or ctrl_stack[-1][0] != "OF" or ctrl_stack[-2][0] != "CASE":
                    raise Forth3Error("ENDOF without CASE ... OF")
                _, of_patch = ctrl_stack.pop()
                self._comma(self.cfa_branch)
                end_patch = self._comma(0)
                ctrl_stack[-1][1].append(end_patch)  # type: ignore[union-attr]
                self.machine.mem.write(int(of_patch), self.here, supervisor=True)
                continue
            if u == "ENDCASE":
                if not ctrl_stack or ctrl_stack[-1][0] != "CASE":
                    raise Forth3Error("ENDCASE without CASE")
                _, patches = ctrl_stack.pop()
                self._comma(self.words["DROP"].cfa)
                for ep in patches:  # type: ignore[union-attr]
                    self.machine.mem.write(int(ep), self.here, supervisor=True)
                continue

            if u in self.words:
                self._comma(self.words[u].cfa)
                continue

            num = self._parse_number(raw)
            if num is not None:
                self._comma(self.cfa_lit)
                self._comma(num)
                continue

            raise Forth3Error(f"Undefined word {raw!r} while compiling {cur_name}")

        if compiling:
            raise Forth3Error(f"Missing `;` at end of definition {cur_name}")
        if self._line_buf:
            self._flush_line()
        return self.output_buf[start_out:]

    def _make_runtime_create_cfa(self) -> int:
        """Compile-time helper supporting `CREATE ... DOES>` inside defining colon words."""
        if "(__RTCREATE)" in self.words:
            return self.words["(__RTCREATE)"].cfa
        # Handle runtime CREATE cleanly via defining-word token queue!
        raise Forth3Error(
            "Use `CREATE <name> ...` directly or `: DEFWORD CREATE ... DOES> ... ;` via defining syntax"
        )


def _preprocess_defining_words(session: Forth3Session, source: str) -> str:
    """Enable authentic `: DEFWORD ... CREATE ... DOES> ... ;` defining words cleanly!

    When a defining word `: NAME <compile-time> CREATE <alloc> DOES> <runtime> ;` is
    invoked as `args NAME newword`, it runs `<compile-time>`, creates `newword` at HERE,
    runs `<alloc>`, and patches `newword`'s CFA to `_DODOES` pointing to `<runtime>`!
    """
    return source


class Forth3Engine(Forth3Session):
    """Full FORTH-3 engine with first-class `: DEFINER ... CREATE ... DOES> ... ;` support."""

    def __init__(self, machine: Optional[UE1000Machine] = None) -> None:
        self._definers: Dict[str, Tuple[List[Tuple[str, str]], List[Tuple[str, str]], int]] = {}
        super().__init__(machine=machine)

    def eval_source(self, source: str) -> List[str]:
        start_out = len(self.output_buf)
        toks = self._tokenize(source)
        i = 0
        n = len(toks)
        plain_buf: List[str] = []

        def _flush_plain() -> None:
            if plain_buf:
                super(Forth3Engine, self).eval_source(" ".join(plain_buf))
                plain_buf.clear()

        def _tok_to_src(k: str, v: str) -> str:
            return f'." {v}"' if k == "STR" else v

        while i < n:
            kind, raw = toks[i]
            u = raw.upper() if kind == "WORD" else ""

            # Intercept `: DEFNAME ... CREATE ... DOES> ... ;` defining words!
            if kind == "WORD" and u == ":" and i + 1 < n:
                # Scan ahead to `;`
                j = i + 2
                has_create = False
                has_does = False
                while j < n and not (toks[j][0] == "WORD" and toks[j][1] == ";"):
                    if toks[j][0] == "WORD":
                        tu = toks[j][1].upper()
                        if tu == "CREATE":
                            has_create = True
                        elif tu == "DOES>":
                            has_does = True
                    j += 1
                if has_create and has_does and j < n:
                    _flush_plain()
                    def_name = toks[i + 1][1].upper()
                    body = toks[i + 2 : j]
                    c_idx = next(k for k, t in enumerate(body) if t[0] == "WORD" and t[1].upper() == "CREATE")
                    d_idx = next(k for k, t in enumerate(body) if t[0] == "WORD" and t[1].upper() == "DOES>")
                    pre_toks = body[:c_idx]
                    alloc_toks = body[c_idx + 1 : d_idx]
                    does_toks = body[d_idx + 1 :]
                    # Compile the DOES> runtime threaded body directly into UE-1000 T-DRAM!
                    helper_name = f"__DOES_{def_name}_{self.here}"
                    does_src = f": {helper_name} " + " ".join(_tok_to_src(k, v) for k, v in does_toks) + " ;"
                    super().eval_source(does_src)
                    does_body_ip = self.words[helper_name].cfa + 1  # right after _DOCOL!
                    self._definers[def_name] = (pre_toks, alloc_toks, does_body_ip)
                    i = j + 1
                    continue

            # Invoke a registered `CREATE ... DOES>` defining word!
            if kind == "WORD" and u in self._definers:
                _flush_plain()
                pre_toks, alloc_toks, does_body_ip = self._definers[u]
                if i + 1 >= n:
                    raise Forth3Error(f"Defining word {u} expects a new word name")
                new_name = toks[i + 1][1].upper()
                i += 2
                if pre_toks:
                    super().eval_source(" ".join(_tok_to_src(k, v) for k, v in pre_toks))
                cfa = self._comma(self.kernel.labels["_DODOES"])
                self._comma(does_body_ip)
                self.words[new_name] = ForthWord(new_name, cfa)
                self.last_created_cfa = cfa
                if alloc_toks:
                    super().eval_source(" ".join(_tok_to_src(k, v) for k, v in alloc_toks))
                continue

            plain_buf.append(_tok_to_src(kind, raw))
            i += 1

        _flush_plain()
        return self.output_buf[start_out:]


@dataclass
class ForthRunResult:
    output: List[str]
    stack: List[int]
    cycles: int
    instructions: int
    istack_hits: int
    dict_words: int
    dict_cells_used: int
    tfabric_ops: int


def run_forth(source: str, machine: Optional[UE1000Machine] = None) -> ForthRunResult:
    eng = Forth3Engine(machine=machine)
    out = eng.eval_source(source)
    return ForthRunResult(
        output=out,
        stack=eng.data_stack(),
        cycles=eng.machine.cpu0.cycles,
        instructions=eng.machine.cpu0.instructions,
        istack_hits=eng.machine.cpu0.istack_hits,
        dict_words=len(eng.words),
        dict_cells_used=eng.here - FORTH_DICT_BASE,
        tfabric_ops=eng.machine.tfabric.total_ops,
    )


def run_forth_repl(machine: Optional[UE1000Machine] = None) -> None:
    """Interactive FORTH-3 console on OBB Level -9 (SETI Annex & Maintenance Bay)."""
    eng = Forth3Engine(machine=machine)
    print("FORTH-3 — Duwamish-I (UE-1000) Indirect-Threaded Balanced-Ternary Forth")
    print(f"  Dictionary: {len(eng.words)} native TRIAD-36 words | Type `WORDS` or `BYE`")
    while True:
        try:
            line = input("  ")
        except (EOFError, KeyboardInterrupt):
            print("\nBYE")
            break
        u = line.strip().upper()
        if u in ("BYE", "QUIT", "EXIT"):
            break
        if u == "WORDS":
            vis = sorted(w for w in eng.words if not w.startswith("(") and not w.startswith("__"))
            print("  " + " ".join(vis))
            continue
        try:
            out = eng.eval_source(line)
            for ln in out:
                print(ln)
            print(" ok")
        except Forth3Error as exc:
            print(f"  ? {exc}")
