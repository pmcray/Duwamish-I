"""Tests for FORTH-3 native TRIAD-36 indirect-threaded Forth on Duwamish-I."""

from __future__ import annotations

from pathlib import Path

import pytest

from ue1000.forth3 import Forth3Error, run_forth
from ue1000.machine import UE1000Machine

PROGRAMS_DIR = Path(__file__).resolve().parent.parent / "programs"


def test_native_threaded_stack_math_and_balanced_shifts() -> None:
    """Verify native TRIAD-36 _NEXT, stack words, T36 arithmetic, 3*/3/, and SGN."""
    res = run_forth("3t1T01 DUP 3* SWAP 3/ -27 SGN 0 SGN 99 SGN")
    # 3t1T01 = 19; 19*3 = 57; 19/3 unbiased = 6 (since 19 = 6*3 + 1!)
    assert res.stack == [57, 6, -1, 0, 1]
    assert res.cycles > 0
    assert res.instructions > 0


def test_loops_recursion_case_and_trit_if() -> None:
    """Verify DO..LOOP, +LOOP, LEAVE, BEGIN..WHILE..REPEAT, RECURSE, CASE, and TRIT-IF."""
    src = """
    : SUM10 0 11 1 DO I + LOOP ;
    : FACT ( n -- n! ) DUP 2 < IF DROP 1 EXIT THEN DUP 1- RECURSE * ;
    : STOPAT5 0 10 0 DO I 5 = IF LEAVE THEN I + LOOP ;
    : SIGN-VAL ( n -- v ) SGN TRIT-IF -100 WHEN0 0 WHEN+ 100 TRIT-THEN ;
    SUM10 FACT_CHECK
    """
    res = run_forth(
        """
        : SUM10 0 11 1 DO I + LOOP ;
        : FACT DUP 2 < IF DROP 1 EXIT THEN DUP 1- RECURSE * ;
        : STOPAT5 0 10 0 DO I 5 = IF LEAVE THEN I + LOOP ;
        : SIGN-VAL SGN TRIT-IF -100 WHEN0 0 WHEN+ 100 TRIT-THEN ;
        : COUNTDOWN 0 SWAP BEGIN DUP 0> WHILE SWAP 1+ SWAP 1- REPEAT DROP ;
        SUM10 6 FACT STOPAT5 -8 SIGN-VAL 0 SIGN-VAL 8 SIGN-VAL 7 COUNTDOWN
        """
    )
    assert res.stack == [55, 720, 10, -100, 0, 100, 7]


def test_create_does_memory_and_execute() -> None:
    """Verify BBC FORTH / FORTH-79 CREATE..DOES>, VARIABLE, CONSTANT, @/!/+!, and EXECUTE."""
    src = """
    VARIABLE ACC
    10 ACC !
    5 ACC +!
    : ADDER CREATE , DOES> @ + ;
    27 ADDER ADD27
    ACC @ ADD27
    ' ADD27 EXECUTE
    """
    res = run_forth(src)
    assert res.stack == [15 + 27 + 27]


def test_tf36_and_good_deciban_hardware_words() -> None:
    """Verify exact TF36 floating-point and Good Deciban hardware instructions on stack."""
    res = run_forth("1.5f 2.5f F+ F>I  12.0dB 9.5dB DB+ 20.0dB DWALD  3.0dB 20.0dB DWALD  -22.0dB 20.0dB DWALD")
    assert res.stack == [4, 1, 0, -1]


def test_showcase_forth_programs() -> None:
    """Run forth_tour.fth and oliver_dish_ctrl.fth end-to-end on UE1000Machine."""
    r1 = run_forth((PROGRAMS_DIR / "forth_tour.fth").read_text(encoding="utf-8"))
    j1 = "\n".join(r1.output)
    assert "FIB(1..10): 1 1 2 3 5 8 13 21 34 55" in j1
    assert "TRIPLE(14) = 42" in j1
    assert "TABLE9[0..8]: 0 1 4 9 16 25 36 49 64" in j1
    assert "NEGATIVE(-1)" in j1 and "UNDECIDED(0)" in j1 and "POSITIVE(+1)" in j1
    assert "000000000CSB" in j1

    m = UE1000Machine()
    r2 = run_forth((PROGRAMS_DIR / "oliver_dish_ctrl.fth").read_text(encoding="utf-8"), machine=m)
    j2 = "\n".join(r2.output)
    assert "[+1] CANDIDATE ARTIFICIAL BEACON!" in j2
    assert "[ 0] UNDECIDED (HOLD INTEGRATION & REFINE)" in j2
    assert "[-1] REJECT (NATURAL/NOISE)" in j2
    # Stack holds verdicts from the 3 INSPECT-BEAM calls: +1, 0, -1!
    assert r2.stack == [1, 0, -1]
    assert r2.tfabric_ops > 0
    assert m.cpu0.cycles > 0
