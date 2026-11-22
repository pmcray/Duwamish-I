"""Comprehensive tests for TRI-TRAN 70V (FORTRAN-66 + 3-valued + Vector + T-Fabric)."""

from pathlib import Path
import pytest

from ue1000.machine import UE1000Machine
from ue1000.tritran70v import TriTranError, compile_tritran, run_tritran

PROGRAMS_DIR = Path(__file__).resolve().parent.parent / "programs"


def test_tritran_tour_kleene_b3_format_and_nan_safety():
    src = (PROGRAMS_DIR / "tritran_tour.f70").read_text(encoding="utf-8")
    res = run_tritran(src, opt=True)
    assert res.stop_code == 101
    assert "KLEENE T&U=U F|U=U ~U=U ICNT=  1" in res.lines[0]
    assert "B3 COUNTS NEG= 2 ZERO= 1 POS= 2" in res.lines[1]
    # 42 in balanced ternary is 1TTT0 (27-9-3-1? Wait: 81-27-9-3=42 -> 1TTT0? Check: 81-27-9-3=42 -> 1TTT0! Wait: 3^4 - 3^3 - 3^2 - 3^1 = 81-27-9-3=42!)
    assert "42=B( 1TTT0)" in res.lines[2]
    assert "-42=B( T1110)" in res.lines[2]
    # SQRT(-1.0) produced TF36 UNDECIDED -> arithmetic IF via FCM routed cleanly to middle label (52)!
    assert "SQRT2= 1.414214" in res.lines[3]
    assert "NAN= 101" in res.lines[3]


def test_tritran_livermore70v_vector_speedup_and_exact_agreement():
    src = (PROGRAMS_DIR / "livermore70v.f70").read_text(encoding="utf-8")
    res = run_tritran(src, opt=True)
    assert res.stop_code == 1
    assert res.compile_result.vector_loops_emitted >= 3
    assert "VLD V1" in res.compile_result.assembly
    assert "VFMA V3, V1, V2" in res.compile_result.assembly
    assert "VRED S3, V3" in res.compile_result.assembly
    # Verify vector VDOT ran in significantly fewer cycles than scalar DO loop!
    cyc_line = res.lines[1]
    parts = [int(tok) for tok in cyc_line.replace("=", " ").split() if tok.isdigit()]
    cyc_scalar, cyc_vec = parts[0], parts[1]
    assert cyc_vec < cyc_scalar * 0.25


def test_tritran_oliver_fft70v_and_live_ashby_cutover():
    src = (PROGRAMS_DIR / "oliver_fft70v.f70").read_text(encoding="utf-8")

    # 1. Clean run: peak at bin 9, ISTAT = +1 -> STOP 9
    res_ok = run_tritran(src, opt=True)
    assert res_ok.stop_code == 9
    assert "PEAK BIN=  9" in res_ok.lines[0]
    assert "STAT= 1" in res_ok.lines[0]

    # 2. Injected hardware tile fault during Radix-3 FFT -> Ashby <75 us cutover -> ISTAT = -1 -> STOP 200!
    m = UE1000Machine()
    m.tfabric.inject_tile_fault(logical_tile=3, reason="FREON_TEMP_SPIKE_FFT")
    res_cut = run_tritran(src, machine=m, opt=True)
    assert res_cut.stop_code == 200
    assert "PEAK BIN=  9" in res_cut.lines[0]
    assert "STAT=-1" in res_cut.lines[0]
    assert len(m.vcomplex.cutovers) == 1
    assert m.vcomplex.cutovers[0].hw_cutover_us < 75.0


def test_tritran_fortran66_card_continuation_2d_arrays_and_functions():
    src = """
C     TEST CONTINUATION CARDS, 2D COLUMN-MAJOR ARRAYS, AND REAL FUNCTION
      PROGRAM F66TST
      REAL A(3, 3), TOTAL, SQPLUS
      INTEGER I, J
      DO 20 J = 1, 3
        DO 10 I = 1, 3
          A(I, J) = FLOAT(I * 10 + J)
   10   CONTINUE
   20 CONTINUE
      TOTAL = SQPLUS(A(2, 3))
     +      + A(3, 1)
      WRITE (6, 100) TOTAL
  100 FORMAT (1X, F8.2)
      IF (ABS(TOTAL - 561.0) .LT. 1.0E-6) STOP 7
      STOP 0
      END
      REAL FUNCTION SQPLUS(X)
      REAL X
      SQPLUS = X * X + 1.0
      RETURN
      END
    """
    # A(2,3) = 23.0 -> SQPLUS(23) = 530.0; A(3,1) = 31.0 -> TOTAL = 561.0!
    res = run_tritran(src, opt=True)
    assert res.stop_code == 7
    assert "561.00" in res.lines[0]


def test_tritran_call_tgemm_from_fortran():
    src = """
      PROGRAM TGM70V
      INTEGER A(9), B(9), C(9), I, ISTAT, C00
      DATA A /39366, 0, 0, 0, 0, 0, 0, 0, 0/
      DO 10 I = 1, 9
        B(I) = 0
   10 CONTINUE
C     A(1) HAS NONET 3 = 2? WAIT: 2 * 3^27 IS LARGE -> PACK VIA ARITHMETIC!
      A(1) = 2 * 19683 * 19683 * 19683
      B(1) = 7 * 19683 * 19683 * 19683
      CALL TGEMM(A, B, C, 6, 6, 6, 8, 8, 8, ISTAT)
      C00 = IXTR(C(1), 27, 9)
      WRITE (6, 100) C00, ISTAT
  100 FORMAT (1X, 'C00=', I4, ' STAT=', I2)
      IF (C00 .EQ. 14 .AND. ISTAT .EQ. 1) STOP 14
      STOP 0
      END
    """
    res = run_tritran(src, opt=True)
    assert res.stop_code == 14
    assert "C00=  14 STAT= 1" in res.lines[0]
