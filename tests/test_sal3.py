"""Comprehensive conformance tests for the SAL-3 compiler, optimizer, and runtime."""

from pathlib import Path
import pytest

from ue1000.machine import UE1000Machine
from ue1000.sal3 import SAL3Error, compile_sal3, run_sal3

PROGRAMS_DIR = Path(__file__).resolve().parent.parent / "programs"


def test_sal3_type_checker_rejects_mismatches():
    # Mixing INT and FLOAT without explicit flt()/fix() must fail compile-time type check
    bad_add = """
    proc main() -> INT
    begin
      var x : INT := 3
      var y : FLOAT := 2.5
      return x + y
    end
    """
    with pytest.raises(SAL3Error, match="Type mismatch"):
        compile_sal3(bad_add)

    # Assigning DECIBAN to INT must fail
    bad_assign = """
    proc main() -> INT
    begin
      var x : INT := 10.0db
      return x
    end
    """
    with pytest.raises(SAL3Error, match="Type error"):
        compile_sal3(bad_assign)


def test_sal3_bloop_certifier_rejects_while_and_recursion():
    while_prog = """
    bloop
    proc main() -> INT
    begin
      var x : INT := 5
      while x > 0 do x -:= 1
      return x
    end
    """
    with pytest.raises(SAL3Error, match="BlooP violation"):
        compile_sal3(while_prog)

    rec_prog = """
    bloop
    proc fact(n : INT) -> INT
    begin
      if n <= 1 then return 1
      return n * fact(n - 1)
    end
    proc main() -> INT = fact(5)
    """
    with pytest.raises(SAL3Error, match="recursion"):
        compile_sal3(rec_prog)

    # Without `bloop`, FlooP recursion runs cleanly and computes 5! = 120!
    floop_prog = rec_prog.replace("bloop", "")
    res = run_sal3(floop_prog)
    assert res.return_value == 120
    assert res.compile_result.bloop_certified is False


def test_sal3_three_valued_logic_and_if_strictness():
    prog = """
    proc main() -> INT
    begin
      var score : INT := 0
      -- `if undecided` must NOT take then branch!
      if undecided then score +:= 100 else score +:= 1
      if false then score +:= 100 else score +:= 2
      if true then score +:= 4 else score +:= 100
      -- McCarthy sequential `and`/`or` preserve UNDECIDED
      if (undecided and true) = undecided then score +:= 8
      if (false or undecided) = undecided then score +:= 16
      -- Kleene strong `&` and `|`
      if (false & undecided) = false then score +:= 32
      if (true | undecided) = true then score +:= 64
      return score
    end
    """
    res = run_sal3(prog)
    assert res.return_value == 1 + 2 + 4 + 8 + 16 + 32 + 64  # 127


def test_sal3_optimizer_equivalence_and_dual_parcel_packing():
    src = (PROGRAMS_DIR / "bloop_primes.sal3").read_text(encoding="utf-8")
    r_plain = run_sal3(src, opt=False)
    r_opt = run_sal3(src, opt=True)

    assert r_plain.return_value == 1521
    assert r_opt.return_value == 1521
    assert r_plain.stdout == r_opt.stdout
    assert r_opt.compile_result.bloop_certified is True
    assert r_opt.compile_result.peephole_edits > 0
    assert r_opt.compile_result.parcels_packed > 0
    assert len(r_opt.compile_result.assembled.words) < len(r_plain.compile_result.assembled.words)


def test_sal3_showcase_wald_seti():
    src = (PROGRAMS_DIR / "wald_seti.sal3").read_text(encoding="utf-8")
    res = run_sal3(src, opt=True)
    assert res.return_value == 1
    assert res.stdout[:3] == ["TRUE", "FALSE", "UNDECIDED"]
    assert res.stdout[3].startswith("+13.0")


def test_sal3_showcase_vec_cliodynamics_and_nan_safety():
    src = (PROGRAMS_DIR / "vec_cliodynamics.sal3").read_text(encoding="utf-8")
    res = run_sal3(src, opt=True)
    # 27 * 1.20 = 32.40 -> fix(32.40) = 32, and undecided_f classified cleanly as UNDECIDED!
    assert res.return_value == 32
    assert res.stdout[1] == "UNDECIDED"
    assert res.stdout[2] == "TRUE"
    assert "VLD V1" in res.compile_result.assembly
    assert "VFMA V3, V1, V2" in res.compile_result.assembly
    assert "FCM S1, S0" in res.compile_result.assembly


def test_sal3_showcase_tfabric_gemm_and_live_ashby_cutover():
    src = (PROGRAMS_DIR / "tfabric_gemm.sal3").read_text(encoding="utf-8")

    # 1. Clean run: status = TRUE (+1), result = 20
    res_clean = run_sal3(src, opt=True)
    assert res_clean.return_value == 20
    assert res_clean.stdout == ["TRUE", "10", "10"]

    # 2. Live fault injection on logical tile 7: Ashby <75 us cutover heals transparently!
    m = UE1000Machine()
    m.tfabric.inject_tile_fault(logical_tile=7, reason="ABFT_SAL3_TEST_FAULT")
    res_fault = run_sal3(src, machine=m, opt=True)
    # TWAIT returned C = -1 (RECOVERED_BY_CUTOVER -> FALSE in print_trit), math still exact (20)!
    assert res_fault.return_value == 20
    assert res_fault.stdout == ["FALSE", "10", "10"]
    assert len(m.vcomplex.cutovers) == 1
    assert m.vcomplex.cutovers[0].hw_cutover_us < 75.0
