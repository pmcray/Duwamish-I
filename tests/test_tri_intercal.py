"""Tests for TRI-INTERCAL (unsanctioned 1972-73 West Seattle grad-student gag on UE-1000)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ue1000.machine import UE1000Machine
from ue1000.tri_intercal import (
    TriIntercalError,
    encode_turing_tape,
    run_tri_intercal,
    ternary_mingle,
    ternary_select,
    to_roman,
)

PROGRAMS_DIR = Path(__file__).resolve().parent.parent / "programs"


def test_balanced_ternary_politeness_enforcement() -> None:
    """Rude (<1/5) triggers SPL-079 (-1); obsequious (>1/3) triggers SPL-099 (+1)."""
    rude = """
    DO .1 <- #1
    DO .2 <- #2
    DO .3 <- #3
    DO .4 <- #4
    DO .5 <- #5
    DO GIVE UP
    """
    with pytest.raises(TriIntercalError, match="SPL-079"):
        run_tri_intercal(rude)

    obsequious = """
    PLEASE DO .1 <- #1
    PLEASE DO .2 <- #2
    PLEASE DO .3 <- #3
    PLEASE GIVE UP
    """
    with pytest.raises(TriIntercalError, match="SPL-099"):
        run_tri_intercal(obsequious)


def test_ternary_mingle_select_unary_kleene_and_roman() -> None:
    """Verify balanced-ternary mingle, signed select, hardware unary Kleene, and ANTI-Roman."""
    assert ternary_mingle(1, 0) == 3
    assert ternary_mingle(1, 1) == 4
    assert ternary_select(19, 13) == 19 - 27 + 9 - 9  # 19 = 1T01_3; low 3 trits = T01_3 = -9+1 = -8!
    assert ternary_select(19, -13) == 8               # signed -1 mask negates selected trits!
    assert to_roman(1973) == "MCMLXXIII"
    assert to_roman(-42) == "ANTI-XLII"
    assert to_roman(0) == "_"

    src = """
    PLEASE DO .1 <- #19
    DO .2 <- #13
    DO .3 <- .1~.2
    DO READ OUT .3
    DO .4 <- .¬1
    DO READ OUT .4
    PLEASE GIVE UP
    """
    res = run_tri_intercal(src)
    assert res.output == ["ANTI-VIII", "ANTI-XIX"]
    assert res.cycles_wasted > 0


def test_turing_tape_obb729_text_and_write_in() -> None:
    """Verify OBB-729 6-trit balanced-ternary Turing Tape text output and English WRITE IN."""
    msg = "OBB -9"
    vals = encode_turing_tape(msg)
    lines = ["PLEASE DO ,1 <- #" + str(len(vals))]
    for idx, v in enumerate(vals, start=1):
        lines.append(f"DO ,1 SUB #{idx} <- #{v}")
    lines.append("PLEASE READ OUT ,1")
    lines.append("DO WRITE IN .5")
    lines.append("DO READ OUT .5")
    lines.append("PLEASE GIVE UP")
    res = run_tri_intercal("\n".join(lines), inputs=["NEGATIVE FOUR TWO"])
    assert res.output == [msg, "ANTI-XLII"]


def test_comefrom_anachronism_gate_and_1974_mode() -> None:
    """COME FROM must fail with SPL-444 in 1972 mode and run cleanly with allow_1974=True."""
    src_no_pragma = """
    PLEASE DO .1 <- #1
    DO COME FROM (10)
    DO .1 <- #99
    (10) DO .2 <- #2
    DO READ OUT .1
    PLEASE GIVE UP
    """
    with pytest.raises(TriIntercalError, match="SPL-444"):
        run_tri_intercal(src_no_pragma, allow_1974=False)

    src_clean = """
    PLEASE DO .1 <- #7
    DO (10) NEXT
    DO GIVE UP
    (10) DO .2 <- #3
    DO ABSTAIN FROM (20)
    (20) DO COME FROM (10)
    DO READ OUT .1
    PLEASE GIVE UP
    """
    res = run_tri_intercal(src_clean, allow_1974=True)
    assert res.output == ["VII"]


def test_showcase_tri_intercal_programs() -> None:
    """Run hello_woodworth.i3, grad_gag_add.i3, and comefrom_1974.i3 on UE1000Machine."""
    m = UE1000Machine()
    r_hello = run_tri_intercal((PROGRAMS_DIR / "hello_woodworth.i3").read_text(encoding="utf-8"), machine=m)
    assert any("HELLO LEVEL -9" in ln for ln in r_hello.output)
    assert r_hello.cycles_wasted > 0

    r_add = run_tri_intercal((PROGRAMS_DIR / "grad_gag_add.i3").read_text(encoding="utf-8"))
    # .1=19 + .2=23 -> XLII (42), abstained .3 stays XLII, 1¢0 -> III (3), -42 -> ANTI-XLII, retrieved -> XLII
    assert r_add.output == ["XLII", "XLII", "III", "ANTI-XLII", "XLII"]

    r_cf = run_tri_intercal((PROGRAMS_DIR / "comefrom_1974.i3").read_text(encoding="utf-8"), allow_1974=True)
    assert "IV" in r_cf.output and "III" in r_cf.output
