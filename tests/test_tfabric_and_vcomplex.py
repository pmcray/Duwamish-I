"""Tests for Tier 2 T-Fabric (GEMM, TBFP-3, TB1 Assoc, Radix-3 FFT) & Tier 4 Ashby Cutover."""

import numpy as np
import pytest

from ue1000.machine import UE1000Machine
from ue1000.tfabric import TFabricUnrecoverableFault
from ue1000.vcomplex import AlgedonicPacket


def test_tbfp9_gemm_accuracy_and_kelvanov_gating():
    m = UE1000Machine()
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1.0, (18, 24))
    b = rng.normal(0, 1.0, (24, 18))
    ref = a @ b

    out, stats = m.tfabric.gemm_tbfp9(a, b, zgate=True, abft=True)
    rel_err = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    assert rel_err < 1e-3
    assert stats.abft_checks >= 9
    assert stats.zero_trit_suppressed_pct > 10.0

    # Verify Rosenblatt-Kelvanov 3-valued attention dead-zone zeroes out small cross-terms
    out_gated, _ = m.tfabric.gemm_tbfp9(a, b, wgate_deadzone=(-2.0, 2.0))
    assert np.sum(out_gated == 0.0) > np.sum(out == 0.0)


def test_tbfp3_and_tb1_associative_modes():
    m = UE1000Machine()
    rng = np.random.default_rng(11)
    a = rng.normal(0, 1.0, (12, 18))
    b = rng.normal(0, 1.0, (18, 12))
    out3, st3 = m.tfabric.gemm_tbfp3(a, b)
    # TBFP-3 correlates strongly with full precision (cos sim > 0.98) at 3x speed
    cos = float(np.sum(out3 * (a @ b)) / (np.linalg.norm(out3) * np.linalg.norm(a @ b)))
    assert cos > 0.98

    # Pure-trit TB1 associative matching finds exact planted key
    keys = rng.choice([-1, 0, 1], size=(64, 36))
    query = keys[19].astype(np.float64)[None, :]
    scores, st1 = m.tfabric.assoc_tb1(query, keys)
    assert int(np.argmax(scores[0])) == 19
    assert st1.ops == 64 * 36


def test_radix3_tbfp9c_fft_matches_exact_dft():
    m = UE1000Machine()
    n = 3**6  # 729 points
    t = np.arange(n, dtype=np.float64)
    sig = 0.8 * np.exp(2j * np.pi * 81 * t / n) + 0.3 * np.exp(2j * np.pi * 243 * t / n)

    spec_t, stats = m.tfabric.fft_radix3_tbfp9c(sig)
    spec_ref = np.fft.fft(sig)
    rel_err = np.linalg.norm(spec_t - spec_ref) / np.linalg.norm(spec_ref)
    assert rel_err < 2e-3
    assert int(np.argmax(np.abs(spec_t))) == 81
    assert stats.elapsed_ns > 0


def test_ashby_three_spares_survive_and_fourth_fault_raises_trap12():
    """Verify all 3 hot spares cut over strictly < 75 us; 4th fault triggers Trap 12 (TFAULT)."""
    m = UE1000Machine()
    a = np.ones((6, 6))
    b = np.eye(6) * 2.0

    for idx, lt in enumerate((2, 14, 26), start=1):
        m.tfabric.inject_tile_fault(logical_tile=lt, reason=f"THERMAL_DRIFT_#{idx}")
        out, stats = m.tfabric.gemm_tbfp9(a, b, abft=True)
        assert np.allclose(out, 2.0 * a, atol=1e-2)
        assert stats.recovered_by_cutover is True
        assert m.vcomplex.cutovers[-1].hw_cutover_us < 75.0
        assert len(m.vcomplex.spare_tiles) == 3 - idx

    # 4th tile fault with 0 spares left must raise TFabricUnrecoverableFault (Trap 12)!
    m.tfabric.inject_tile_fault(logical_tile=0, reason="ABFT_4TH_FAULT")
    with pytest.raises(TFabricUnrecoverableFault):
        m.tfabric.gemm_tbfp9(a, b, abft=True)
    assert m.vcomplex.cutovers[-1].kind == "SPARES_EXHAUSTED"

    # Verify 4-word Algedonic Telemetry Packet packs cleanly into 36-trit words
    pkt = m.vcomplex.packets[0]
    w0, w1, w2, w3 = pkt.pack_words()
    assert all(isinstance(w, int) for w in (w0, w1, w2, w3))
