"""Tests for balanced-ternary primitives, Kleene logic, OBB-729, TF36/TF18 & Good Decibans."""

import math
import numpy as np
import pytest

from ue1000.ternary import (
    HEPT_ALPHABET,
    WMAX_36,
    char_to_obb729,
    extract_field,
    hept_to_int,
    int_to_hept,
    int_to_trits,
    kleene_and,
    kleene_eqv,
    kleene_not,
    kleene_or,
    kleene_sel,
    obb729_to_char,
    pack_trytes_to_words,
    shift_trits,
    sign_trit,
    trits_to_int,
    unpack_words_to_trytes,
)
from ue1000.tfloat import (
    TF18,
    TF36,
    db_to_ticks,
    deciban_add,
    deciban_exp_float,
    deciban_lse,
    deciban_wald,
    float_to_tfloat,
    int_to_tfloat,
    is_inf,
    is_undecided,
    pack_dual_tf18,
    quantize_tbfp,
    dequantize_tbfp,
    tfloat_add,
    tfloat_cmp,
    tfloat_div,
    tfloat_fma,
    tfloat_mul,
    tfloat_neg,
    tfloat_sub,
    tfloat_to_float,
    tfloat_to_int,
    ticks_to_db,
    unpack_dual_tf18,
)


def test_trits_and_heptavintimal_roundtrip():
    assert len(HEPT_ALPHABET) == 27
    for v in (0, 1, -1, 13, -13, 364, -364, 9841, -9841, 1234567890123, -1234567890123, WMAX_36, -WMAX_36):
        ts = int_to_trits(v, 36)
        assert trits_to_int(ts) == v
        h = int_to_hept(v, 36)
        assert len(h) == 12
        assert hept_to_int(h) == v


def test_unbiased_truncation_rounding_shift():
    # In balanced ternary, shifting right by 1 rounds exact nearest multiple of 3 (no ties!)
    assert shift_trits(4, -1) == 1   # 4/3 = 1.333 -> 1
    assert shift_trits(5, -1) == 2   # 5/3 = 1.667 -> 2
    assert shift_trits(-4, -1) == -1
    assert shift_trits(-5, -1) == -2
    assert shift_trits(14, -2) == 2  # 14/9 = 1.556 -> 2
    assert shift_trits(13, -2) == 1  # 13/9 = 1.444 -> 1


def test_kleene_three_valued_logic_and_sel():
    assert kleene_and(-1, 1, 1) == -1
    assert kleene_and(0, 1, 1) == 0
    assert kleene_and(0, -1, 1) == -1
    assert kleene_or(0, 1, 1) == 1
    assert kleene_or(0, -1, 1) == 0
    assert kleene_eqv(-1, -1, 1) == 1
    assert kleene_eqv(1, -1, 1) == -1
    assert kleene_eqv(0, 1, 1) == 0
    assert kleene_not(1, 1) == -1
    # SEL truth table: trits (-1 -> T, 0 -> 0, +1 -> 1) -> triad 1*9 + 0*3 + (-1) = 8
    assert kleene_sel(-1, 8) == -1
    assert kleene_sel(0, 8) == 0
    assert kleene_sel(1, 8) == 1


def test_obb729_tryte_partition_and_packing():
    assert char_to_obb729("\x00") == 0
    for ch in (" ", "A", "z", "Ж", "я", "∑", "∇"):
        tr = char_to_obb729(ch)
        assert tr > 0  # Printable glyphs are strictly positive!
        assert obb729_to_char(tr) == ch
    for ctrl in ("\n", "\r", "\t"):
        assert char_to_obb729(ctrl) < 0  # Control codes are strictly negative!
    trytes = [char_to_obb729(c) for c in "SETUN\n"]
    words = pack_trytes_to_words(trytes)
    assert len(words) == 1
    assert unpack_words_to_trytes(words) == trytes


def test_tf36_arithmetic_precision_and_sign_invariant():
    a = float_to_tfloat(math.pi, TF36)
    b = float_to_tfloat(-2.718281828459045, TF36)
    assert math.isclose(tfloat_to_float(a, TF36), math.pi, rel_tol=1e-13)
    assert math.isclose(tfloat_to_float(b, TF36), -2.718281828459045, rel_tol=1e-13)

    # Sign invariant holds for all finite and infinite values!
    assert sign_trit(a) == 1
    assert sign_trit(b) == -1
    assert sign_trit(TF36.zero_word) == 0
    assert sign_trit(TF36.pos_inf_word) == 1
    assert sign_trit(TF36.neg_inf_word) == -1

    # Arithmetic accuracy (~47.55 bits -> < 1e-13 rel err)
    s, _, _ = tfloat_add(a, b, TF36)
    d, _, _ = tfloat_sub(a, b, TF36)
    p, _, _ = tfloat_mul(a, b, TF36)
    q, _, _, dz = tfloat_div(a, b, TF36)
    f, _, _ = tfloat_fma(a, b, a, TF36)
    assert not dz
    assert math.isclose(tfloat_to_float(s, TF36), math.pi - math.e, rel_tol=1e-12)
    assert math.isclose(tfloat_to_float(d, TF36), math.pi + math.e, rel_tol=1e-12)
    assert math.isclose(tfloat_to_float(p, TF36), -math.pi * math.e, rel_tol=1e-13)
    assert math.isclose(tfloat_to_float(q, TF36), -math.pi / math.e, rel_tol=1e-13)
    assert math.isclose(tfloat_to_float(f, TF36), math.pi - math.e * math.pi, rel_tol=1e-12)

    # Round-trip FLT/FIX & dual TF18 packing
    assert tfloat_to_int(int_to_tfloat(-987654321, TF36), TF36) == -987654321
    pk = pack_dual_tf18(a, b)
    assert math.isclose(tfloat_to_float(unpack_dual_tf18(pk, True), TF36), math.pi, rel_tol=1e-5)
    assert math.isclose(tfloat_to_float(unpack_dual_tf18(pk, False), TF36), -math.e, rel_tol=1e-5)


def test_tf36_undecided_tst_vs_fcm_normative_contract():
    """Verify Rev. 1.5 §2.3 & §4.3A caveat: raw TST on UNDECIDED is -1, FCM returns 0 + FU."""
    u = TF36.undecided_word
    assert is_undecided(u, TF36)
    assert u == -364
    assert sign_trit(u) == -1  # Raw integer TST sees -364 -> -1!
    # Normative FCM explicitly detects UNDECIDED and returns C = 0, saw_undecided = True!
    c, und = tfloat_cmp(u, TF36.zero_word, TF36)
    assert c == 0 and und is True
    # FNEG preserves UNDECIDED
    assert is_undecided(tfloat_neg(u, TF36), TF36)
    # INF - INF produces UNDECIDED
    res, _, und2 = tfloat_sub(TF36.pos_inf_word, TF36.pos_inf_word, TF36)
    assert is_undecided(res, TF36) and und2 is True


def test_good_deciban_dlse_and_wald():
    # Exact log-sum-exp check: 10*log10(10^1 + 10^1) = 13.0103 dB
    a = db_to_ticks(10.0)
    b = db_to_ticks(10.0)
    lse = deciban_lse(a, b)
    assert abs(ticks_to_db(lse) - 13.0103) < 0.02
    assert math.isclose(deciban_exp_float(db_to_ticks(20.0)), 100.0, rel_tol=1e-4)

    # DWALD works identically via raw ticks register OR whole-dB simm!
    assert deciban_wald(db_to_ticks(21.0), reg_ticks=1620, simm_db=0) == 1
    assert deciban_wald(db_to_ticks(21.0), reg_ticks=0, simm_db=20) == 1
    assert deciban_wald(db_to_ticks(5.0), reg_ticks=0, simm_db=20) == 0
    assert deciban_wald(db_to_ticks(-20.5), reg_ticks=0, simm_db=20) == -1
    # Saturation check (no wrap!)
    assert deciban_add(WMAX_36, 1000) == WMAX_36


def test_tbfp9_quantization_precision():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1.0, (12, 12))
    q, e = quantize_tbfp(x, mant_trits=9)
    xr = dequantize_tbfp(q, e, mant_trits=9)
    # 9-trit = 14.26 bits -> max relative error < 2e-4
    assert np.max(np.abs(x - xr)) / np.max(np.abs(x)) < 2e-4
