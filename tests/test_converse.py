"""Comprehensive tests for CONVERSE (Pask / Licklider / Good interactive language)."""

from pathlib import Path
import pytest

from ue1000.converse import ConverseError, ConverseSession, run_converse_script

PROGRAMS_DIR = Path(__file__).resolve().parent.parent / "programs"


def test_converse_tb1_associative_search_and_negation():
    sess = ConverseSession()
    out1 = sess.execute_line('FIND "Setun Kelvanov Byurakan ternary" TOP 2')
    assert "TB1 SEARCH" in out1[0]
    assert "Kelvanov" in out1[1] or "Brusentsov" in out1[1]

    # Negative query term (`NOT`) uses balanced-ternary -1 weights in TB1!
    out2 = sess.execute_line('FIND "LISP McCarthy NOT SETI" TOP 1')
    assert "McCarthy" in out2[1]


def test_converse_good_decibans_pask_entailment_and_sign_branching():
    src = """
    THRESHOLD 20.0db
    ENTAIL HYPOTHESIS FROM PREM_A, PREM_B
    ASSERT PREM_A AT +25.0db SOURCE "direct measurement"
    ASSERT PREM_B AT +6.0db SOURCE "preliminary"
    DIAGNOSE HYPOTHESIS
    SIGN PREM_B OF FALSE: PRINT "NO" | UNDECIDED: PRINT "STILL_UNDECIDED" | TRUE: PRINT "YES"
    EVIDENCE PREM_B ADD +18.0db SOURCE "confirmation"
    COMBINE BOTH = LSE(PREM_A, PREM_B)
    DIAGNOSE HYPOTHESIS
    IF PREM_B IS TRUE THEN PRINT "RESOLVED_TRUE"
    """
    out, sess = run_converse_script(src)
    joined = "\n".join(out)
    assert "entailment closure = UNDECIDED (0)" in joined
    assert "gather 14.0 dB on UNDECIDED premise `PREM_B`" in joined
    assert "STILL_UNDECIDED" in joined
    assert "PASK CONVERSATIONAL LOOP CLOSED: `HYPOTHESIS` is entailed as TRUE (+1)." in joined
    assert "RESOLVED_TRUE" in joined
    assert sess.claims["PREM_B"].verdict(20.0) == 1
    assert sess.claims["BOTH"].decibans > 25.0


def test_converse_showcase_byurakan_inquiry():
    src = (PROGRAMS_DIR / "byurakan_inquiry.cv").read_text(encoding="utf-8")
    out, sess = run_converse_script(src)
    joined = "\n".join(out)
    assert "Byurakan" in joined
    assert "gather 12.0 dB on UNDECIDED premise `DOPPLER_COHERENT`" in joined
    assert "ARTIFICIAL BEACON (+1)" in joined
    assert "PASK CONVERSATIONAL LOOP CLOSED: `BEACON_ARTIFICIAL` is entailed as TRUE (+1)." in joined
    assert "VERDICT LOCKED" in joined
    assert sess.claims["DOPPLER_COHERENT"].verdict(20.0) == 1


def test_converse_showcase_obb_full_session():
    src = (PROGRAMS_DIR / "obb_session.cv").read_text(encoding="utf-8")
    out, sess = run_converse_script(src)
    joined = "\n".join(out)
    assert "Kelvanov" in joined
    assert "SAEW [TBFP9 · 1-TILE]: CAT-001 SETUN" in joined
    assert "SAEW [TBFP3 · 1-TILE]: CAT-001 SETUN" in joined
    assert "[POLICY ALERT] Trajectory collapses" in joined
    assert "[POLICY CONFIRMED] Ashby-Beer Ultrastable Equilibrium reached!" in joined
    assert "ASHBY CUTOVER EXECUTED" in joined
    assert len(sess.machine.vcomplex.cutovers) == 1
    assert sess.machine.vcomplex.cutovers[0].hw_cutover_us < 75.0


def test_converse_rejects_unknown_command():
    with pytest.raises(ConverseError):
        run_converse_script("FROBNICATE EVERYTHING")
