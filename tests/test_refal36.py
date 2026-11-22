"""Tests for REFAL-36 (Turchin 1968–71 Markov pattern-rewriting language on UE-1000)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ue1000.machine import UE1000Machine
from ue1000.refal36 import Refal36Error, run_refal

PROGRAMS_DIR = Path(__file__).resolve().parent.parent / "programs"


def test_patterns_backtracking_and_where_conditions() -> None:
    """Verify s./t./e. variables, repeated variables, and backtracking where-clauses."""
    src = """
    $ENTRY Go {
      = <Prout <Pal 'LEVEL'>>
        <Prout <Pal 'SETUN'>>
        <Prout <Split ',alpha,beta'>>
        <Prout <Dup 'abccde'>>;
    }
    Pal { = True; s.1 = True; s.1 e.2 s.1 = <Pal e.2>; e.1 = False; }
    Split {
      e.1 ',' e.2, <Lenw e.1> : s.N e.3, <Compare s.N 0> : '+' = (e.1) (e.2);
    }
    Dup { e.1 s.X s.X e.2 = s.X; e.1 = None; }
    """
    res = run_refal(src)
    assert res.output == ["True", "False", "(,alpha)(beta)", "c"]
    assert res.hw_alu_calls >= 1
    assert res.cpu_cycles > 0


def test_hardware_t36_alu_kleene_and_wald() -> None:
    """Verify native C-Complex T36 arithmetic, 3-way Compare, Kleene logic, and DWALD."""
    m = UE1000Machine()
    src = """
    $ENTRY Go {
      = <Prout <Add 100 -37>>
        <Prout <Mul 3t1T01 3>>
        <Prout <Compare -5 5> <Compare 7 7> <Compare 9 2>>
        <Prout <KleeneAnd True Undecided> <KleeneOr False Undecided> <KleeneEqv False False>>
        <Prout <Wald 1800 1620> <Wald 500 1620> <Wald -1800 1620>>;
    }
    """
    res = run_refal(src, machine=m)
    assert res.output[0] == "63"
    assert res.output[1] == "57"       # 3t1T01 = 27 - 9 + 1 = 19; 19 * 3 = 57
    assert res.output[2] == "-0+"
    assert res.output[3] == "Undecided Undecided True"
    assert res.output[4] == "True Undecided False"
    assert m.cpu0.cycles > 0
    assert res.hw_alu_calls >= 10


def test_showcase_refal_tour() -> None:
    """Run programs/refal_tour.ref: 1969 differentiation, 3-way Qsort, burial store, Cyrillic."""
    src = (PROGRAMS_DIR / "refal_tour.ref").read_text(encoding="utf-8")
    res = run_refal(src)
    joined = "\n".join(res.output)
    assert "Reverse(SETUN-70): 07-NUTES" in joined
    assert "Cyrillic echo    : СЕТУНЬ" in joined
    assert "d/dx(x^2 + 3*x + 5) = ((x+x)+3)" in joined
    assert "Sorted: -27 -9 -1 0 3 9 13 27 81" in joined
    assert "Buried site : OBB Level -9" in joined
    assert "Trits(42)   : 1TTT0 | Untrits(1TTT0) = 42" in joined
    assert "Hept(1973)  : 000000000CSB" in joined
    assert "Wald(+22dB) : True" in joined
    assert res.store_ops >= 3


def test_futamura_first_projection_mix() -> None:
    """Run programs/refal_mix.ref: specialized residual Power_5(3)=243 runs in far fewer steps."""
    src = (PROGRAMS_DIR / "refal_mix.ref").read_text(encoding="utf-8")
    res = run_refal(src)
    interp_line = next(ln for ln in res.output if ln.startswith("Interpreted"))
    comp_line = next(ln for ln in res.output if ln.startswith("Compiled"))
    assert "Result = 243" in interp_line
    assert "Result = 243" in comp_line
    steps_interp = int(interp_line.split("Refal steps = ")[1])
    steps_comp = int(comp_line.split("Refal steps = ")[1])
    assert steps_comp < steps_interp  # Futamura's 1st projection eliminates loop interpretation!


def test_byurakan_ceti_grammar_decoder() -> None:
    """Run programs/byurakan_decode.ref reading framed rules & seed from $END card deck."""
    src = (PROGRAMS_DIR / "byurakan_decode.ref").read_text(encoding="utf-8")
    res = run_refal(src)
    joined = "\n".join(res.output)
    assert "BYURAKAN-CETI-1971-TRIT-V1" in joined
    assert "Hardware Wald: True" in joined
    assert "Decoded trits:" in joined
    # Ensure Markov normal algorithm reached a fixed point with none of the LHS patterns remaining
    dec_line = next(ln for ln in res.output if ln.startswith("Decoded trits: "))
    decoded = dec_line.split("Decoded trits: ", 1)[1].strip()
    for pat in ("TT", "T1", "00", "1T"):
        assert pat not in decoded
