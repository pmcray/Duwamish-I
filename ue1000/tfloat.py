"""Native balanced-ternary floating-point, block-floating-point & Good Deciban units.

Conforms strictly to Principles of Operation (Rev. 1.5), Sections 2.3–2.5:
  - TF36 : [ mant : 30 trits (35..6) | exp : 6 trits (5..0) ], value = m * 3^(e - 29)
           47.55-bit mantissa, ±10^173.2 range, unbiased single-round truncation.
           UNDECIDED (NaN) = (mant=0, exp=-364).
           Sign invariant holds for finite and ±INF; UNDECIDED requires FCM!
  - TF18 : [ mant : 13 trits (17..5) | exp : 5 trits (4..0) ], value = m * 3^(e - 12)
           20.60-bit mantissa, ±10^57.3 range. UNDECIDED = (mant=0, exp=-121).
  - TBFP-9 / TBFP-3 / TB1: T-Fabric block-floating-point & pure-trit formats.
  - Good Deciban (DB18 / DB36): 81 ticks/dB, saturating add/sub, 243-entry ROM DLSE & DEXP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .ternary import (
    POW3,
    WMAX_6,
    WMAX_9,
    WMAX_18,
    WMAX_36,
    WORD_TRITS,
    clamp,
    extract_field,
    sign_trit,
    wrap,
)

# ---------------------------------------------------------------------------
# Generic Balanced-Ternary Floating-Point Engine (TF36 and TF18)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TFloatSpec:
    name: str
    total_trits: int
    mant_trits: int
    exp_trits: int

    @property
    def mant_max(self) -> int:
        return (POW3[self.mant_trits] - 1) // 2

    @property
    def mant_min_norm(self) -> int:
        # Leading trit non-zero -> |m| >= (3^(M-1) + 1) / 2
        return (POW3[self.mant_trits - 1] + 1) // 2

    @property
    def exp_special(self) -> int:
        return (POW3[self.exp_trits] - 1) // 2  # +364 for TF36, +121 for TF18

    @property
    def exp_max(self) -> int:
        return self.exp_special - 1             # +363 for TF36, +120 for TF18

    @property
    def exp_min(self) -> int:
        return -self.exp_max                    # -363 for TF36, -120 for TF18

    @property
    def zero_word(self) -> int:
        return 0

    @property
    def undecided_word(self) -> int:
        return -self.exp_special  # mant=0, exp=-exp_special

    @property
    def pos_inf_word(self) -> int:
        return self.mant_max * POW3[self.exp_trits] + self.exp_special

    @property
    def neg_inf_word(self) -> int:
        return -self.mant_max * POW3[self.exp_trits] + self.exp_special


TF36 = TFloatSpec("TF36", total_trits=36, mant_trits=30, exp_trits=6)
TF18 = TFloatSpec("TF18", total_trits=18, mant_trits=13, exp_trits=5)


def unpack_tfloat(word: int, spec: TFloatSpec = TF36) -> Tuple[int, int]:
    """Return (mant, exp) from a mantissa-high balanced-ternary float word."""
    w = wrap(word, spec.total_trits)
    exp = extract_field(w, 0, spec.exp_trits, spec.total_trits)
    mant = extract_field(w, spec.exp_trits, spec.mant_trits, spec.total_trits)
    return mant, exp


def pack_tfloat(mant: int, exp: int, spec: TFloatSpec = TF36) -> int:
    """Pack (mant, exp) into a mantissa-high balanced-ternary float word."""
    return mant * POW3[spec.exp_trits] + exp


def is_undecided(word: int, spec: TFloatSpec = TF36) -> bool:
    mant, exp = unpack_tfloat(word, spec)
    return mant == 0 and exp == -spec.exp_special


def is_inf(word: int, spec: TFloatSpec = TF36) -> bool:
    mant, exp = unpack_tfloat(word, spec)
    return exp == spec.exp_special and abs(mant) == spec.mant_max


def is_zero(word: int, spec: TFloatSpec = TF36) -> bool:
    mant, exp = unpack_tfloat(word, spec)
    return mant == 0 and exp != -spec.exp_special


def _round_div_pow3(val: int, k: int) -> int:
    """Exact tie-free unbiased round-to-nearest of val / 3^k for k >= 0."""
    if k <= 0:
        return val * (3 ** (-k))
    p = 3**k
    return (val + (p - 1) // 2) // p


def _normalize_and_pack(mant: int, exp: int, spec: TFloatSpec) -> Tuple[int, bool]:
    """Normalize exact integer `mant * 3^(exp - (M-1))` with single unbiased round.

    Returns (encoded_word, overflowed_flag).
    """
    if mant == 0:
        return spec.zero_word, False

    m_max = spec.mant_max
    m_min = spec.mant_min_norm

    # Shift left if subnormal magnitude
    while abs(mant) < m_min:
        mant *= 3
        exp -= 1
        if exp < spec.exp_min - spec.mant_trits:
            return spec.zero_word, False

    # Compute excess trits if magnitude exceeds m_max, then round ONCE!
    if abs(mant) > m_max:
        # Determine minimal k such that |round(mant / 3^k)| <= m_max
        k = 0
        tmp = abs(mant)
        while tmp > m_max:
            tmp = (tmp + 1) // 3
            k += 1
        mant = _round_div_pow3(mant, k)
        exp += k
        # Rounding up could theoretically push |mant| to m_max + 1; re-check once
        if abs(mant) > m_max:
            mant = _round_div_pow3(mant, 1)
            exp += 1

    if exp > spec.exp_max:
        return (spec.pos_inf_word if mant > 0 else spec.neg_inf_word), True
    if exp < spec.exp_min:
        return spec.zero_word, False

    return pack_tfloat(mant, exp, spec), False


def float_to_tfloat(x: float, spec: TFloatSpec = TF36) -> int:
    """Convert IEEE Python float to exact balanced-ternary TF36 or TF18."""
    if math.isnan(x):
        return spec.undecided_word
    if math.isinf(x):
        return spec.pos_inf_word if x > 0 else spec.neg_inf_word
    if x == 0.0:
        return spec.zero_word

    # Exact rational conversion via frexp: x = f * 2^p with exact integer num / den
    sign = -1 if x < 0 else 1
    ax = abs(x)
    # Estimate base-3 exponent so ax * 3^(M-1 - e) lies near 3^(M-1)
    e = int(math.floor(math.log(ax, 3.0)))
    # Use Python's exact integer ratio (no precision loss for IEEE float64!)
    num, den = ax.as_integer_ratio()
    shift = (spec.mant_trits - 1) - e
    if shift >= 0:
        scaled_num = num * (3**shift)
        scaled_den = den
    else:
        scaled_num = num
        scaled_den = den * (3 ** (-shift))
    # Round to nearest integer
    q, rem = divmod(scaled_num, scaled_den)
    if 2 * rem >= scaled_den:
        q += 1
    word, _ = _normalize_and_pack(sign * q, e, spec)
    return word


def tfloat_to_float(word: int, spec: TFloatSpec = TF36) -> float:
    """Convert TF36 or TF18 word to Python float."""
    if is_undecided(word, spec):
        return float("nan")
    if is_inf(word, spec):
        mant, _ = unpack_tfloat(word, spec)
        return float("inf") if mant > 0 else float("-inf")
    mant, exp = unpack_tfloat(word, spec)
    if mant == 0:
        return 0.0
    p = exp - (spec.mant_trits - 1)
    if p >= 0:
        return float(mant) * (3.0**p)
    return float(mant) / (3.0 ** (-p))


def int_to_tfloat(v: int, spec: TFloatSpec = TF36) -> int:
    """FLT: Convert signed integer to TF36 / TF18 with single unbiased round."""
    if v == 0:
        return spec.zero_word
    word, _ = _normalize_and_pack(v, spec.mant_trits - 1, spec)
    return word


def tfloat_to_int(word: int, spec: TFloatSpec = TF36, width: int = WORD_TRITS) -> int:
    """FIX: Convert TF36 / TF18 to signed balanced integer (unbiased nearest)."""
    if is_undecided(word, spec) or is_zero(word, spec):
        return 0
    if is_inf(word, spec):
        mant, _ = unpack_tfloat(word, spec)
        return clamp(mant, width)
    mant, exp = unpack_tfloat(word, spec)
    p = exp - (spec.mant_trits - 1)
    if p >= 0:
        return clamp(mant * (3**p), width)
    return clamp(_round_div_pow3(mant, -p), width)


def tfloat_neg(word: int, spec: TFloatSpec = TF36) -> int:
    """FNEG: Invert high mantissa trits only; UNDECIDED remains UNDECIDED."""
    if is_undecided(word, spec):
        return spec.undecided_word
    mant, exp = unpack_tfloat(word, spec)
    if mant == 0:
        return spec.zero_word
    return pack_tfloat(-mant, exp, spec)


def tfloat_abs(word: int, spec: TFloatSpec = TF36) -> int:
    """FABS: Absolute value; UNDECIDED remains UNDECIDED."""
    if is_undecided(word, spec):
        return spec.undecided_word
    mant, exp = unpack_tfloat(word, spec)
    if mant == 0:
        return spec.zero_word
    return pack_tfloat(abs(mant), exp, spec)


def tfloat_cmp(a: int, b: int, spec: TFloatSpec = TF36) -> Tuple[int, bool]:
    """FCM: Compare a vs b. Returns (cond_trit, saw_undecided).

    If either operand is UNDECIDED, returns (0, True) per Rev. 1.5 §2.3 & §4.3B.
    """
    if is_undecided(a, spec) or is_undecided(b, spec):
        return 0, True
    if is_inf(a, spec) or is_inf(b, spec):
        ma, _ = unpack_tfloat(a, spec)
        mb, _ = unpack_tfloat(b, spec)
        sa = sign_trit(ma) if is_inf(a, spec) else 0
        sb = sign_trit(mb) if is_inf(b, spec) else 0
        if is_inf(a, spec) and is_inf(b, spec):
            return sign_trit(sa - sb), False
        if is_inf(a, spec):
            return sa, False
        return -sb, False

    ma, ea = unpack_tfloat(a, spec)
    mb, eb = unpack_tfloat(b, spec)
    if ma == 0 and mb == 0:
        return 0, False
    if ma == 0:
        return -sign_trit(mb), False
    if mb == 0:
        return sign_trit(ma), False
    if sign_trit(ma) != sign_trit(mb):
        return sign_trit(ma), False
    # Same sign: compare exponents, then mantissas (since normalized!)
    if ea != eb:
        return (1 if ea > eb else -1) * sign_trit(ma), False
    return sign_trit(ma - mb), False


def tfloat_add(a: int, b: int, spec: TFloatSpec = TF36) -> Tuple[int, bool, bool]:
    """FAD: Exact single-round balanced-ternary add. Returns (word, overflow, undecided)."""
    if is_undecided(a, spec) or is_undecided(b, spec):
        return spec.undecided_word, False, True
    if is_inf(a, spec) or is_inf(b, spec):
        ma, _ = unpack_tfloat(a, spec)
        mb, _ = unpack_tfloat(b, spec)
        if is_inf(a, spec) and is_inf(b, spec):
            if sign_trit(ma) != sign_trit(mb):
                return spec.undecided_word, False, True  # INF - INF -> UNDECIDED
            return a, False, False
        return (a if is_inf(a, spec) else b), False, False

    ma, ea = unpack_tfloat(a, spec)
    mb, eb = unpack_tfloat(b, spec)
    if ma == 0:
        return b, False, False
    if mb == 0:
        return a, False, False

    # Ensure ea >= eb
    if ea < eb:
        ma, ea, mb, eb = mb, eb, ma, ea

    diff = ea - eb
    # If diff > M + 3, b cannot alter the rounded M-trit result of a!
    if diff > spec.mant_trits + 4:
        return pack_tfloat(ma, ea, spec), False, False

    # Exact integer sum aligned to eb!
    exact_sum = ma * (3**diff) + mb
    w, ovf = _normalize_and_pack(exact_sum, eb, spec)
    return w, ovf, False


def tfloat_sub(a: int, b: int, spec: TFloatSpec = TF36) -> Tuple[int, bool, bool]:
    """FSB: Exact single-round subtract."""
    return tfloat_add(a, tfloat_neg(b, spec), spec)


def tfloat_mul(a: int, b: int, spec: TFloatSpec = TF36) -> Tuple[int, bool, bool]:
    """FMP: Exact single-round multiply."""
    if is_undecided(a, spec) or is_undecided(b, spec):
        return spec.undecided_word, False, True
    if is_inf(a, spec) or is_inf(b, spec):
        if is_zero(a, spec) or is_zero(b, spec):
            return spec.undecided_word, False, True  # 0 * INF -> UNDECIDED
        ma, _ = unpack_tfloat(a, spec)
        mb, _ = unpack_tfloat(b, spec)
        s = sign_trit(ma) * sign_trit(mb)
        return (spec.pos_inf_word if s > 0 else spec.neg_inf_word), False, False

    ma, ea = unpack_tfloat(a, spec)
    mb, eb = unpack_tfloat(b, spec)
    if ma == 0 or mb == 0:
        return spec.zero_word, False, False

    # (ma * 3^(ea - (M-1))) * (mb * 3^(eb - (M-1))) = (ma * mb) * 3^((ea + eb - (M-1)) - (M-1))
    w, ovf = _normalize_and_pack(ma * mb, ea + eb - (spec.mant_trits - 1), spec)
    return w, ovf, False


def tfloat_fma(acc: int, a: int, b: int, spec: TFloatSpec = TF36) -> Tuple[int, bool, bool]:
    """FMA: True single-round fused multiply-add: acc + a * b."""
    if is_undecided(acc, spec) or is_undecided(a, spec) or is_undecided(b, spec):
        return spec.undecided_word, False, True
    if is_inf(acc, spec) or is_inf(a, spec) or is_inf(b, spec):
        p, ovf1, und1 = tfloat_mul(a, b, spec)
        if und1:
            return spec.undecided_word, False, True
        return tfloat_add(acc, p, spec)

    ma, ea = unpack_tfloat(a, spec)
    mb, eb = unpack_tfloat(b, spec)
    mc, ec = unpack_tfloat(acc, spec)
    if ma == 0 or mb == 0:
        return acc, False, False
    prod_m = ma * mb
    prod_e = ea + eb - (spec.mant_trits - 1)
    if mc == 0:
        w, ovf = _normalize_and_pack(prod_m, prod_e, spec)
        return w, ovf, False

    # Align exact product and exact acc at common exponent
    # Effective exponent of prod_m is prod_e (magnitude ~ 3^(2M-2) -> shift +M-1)
    prod_eff_e = prod_e + (spec.mant_trits - 1)
    diff = ec - prod_eff_e
    if diff > spec.mant_trits + 5:
        return acc, False, False
    if diff < -(2 * spec.mant_trits + 5):
        w, ovf = _normalize_and_pack(prod_m, prod_e, spec)
        return w, ovf, False

    # Align both to min(ec, prod_e)
    base_e = min(ec, prod_e)
    exact_sum = mc * (3 ** (ec - base_e)) + prod_m * (3 ** (prod_e - base_e))
    w, ovf = _normalize_and_pack(exact_sum, base_e, spec)
    return w, ovf, False


def tfloat_div(a: int, b: int, spec: TFloatSpec = TF36) -> Tuple[int, bool, bool, bool]:
    """FDV: Exact single-round divide. Returns (word, overflow, undecided, div_by_zero)."""
    if is_undecided(a, spec) or is_undecided(b, spec):
        return spec.undecided_word, False, True, False
    if is_zero(b, spec):
        if is_zero(a, spec):
            return spec.undecided_word, False, True, True
        return spec.undecided_word, False, False, True
    if is_inf(a, spec) and is_inf(b, spec):
        return spec.undecided_word, False, True, False
    if is_inf(a, spec):
        ma, _ = unpack_tfloat(a, spec)
        mb, _ = unpack_tfloat(b, spec)
        s = sign_trit(ma) * sign_trit(mb)
        return (spec.pos_inf_word if s > 0 else spec.neg_inf_word), False, False, False
    if is_inf(b, spec):
        return spec.zero_word, False, False, False

    ma, ea = unpack_tfloat(a, spec)
    mb, eb = unpack_tfloat(b, spec)
    if ma == 0:
        return spec.zero_word, False, False, False

    sign = sign_trit(ma) * sign_trit(mb)
    ama, amb = abs(ma), abs(mb)
    # Scale numerator by 3^(M + 2) for guaranteed extra guard precision, then round once!
    guard = spec.mant_trits + 2
    num = ama * (3**guard)
    q, rem = divmod(num, amb)
    if 2 * rem >= amb:
        q += 1
    # Exact value = sign * (ama/amb) * 3^(ea - eb) = (sign * q) * 3^(ea - eb - guard)
    # Express as m * 3^(exp - (M-1)) -> exp = ea - eb - guard + (M-1)
    w, ovf = _normalize_and_pack(sign * q, ea - eb - guard + (spec.mant_trits - 1), spec)
    return w, ovf, False, False


def pack_dual_tf18(hi_tf36: int, lo_tf36: int) -> int:
    """F18PK: Convert two TF36 values to TF18 and pack into one T36 word (H1 | H0)."""
    hi18 = float_to_tfloat(tfloat_to_float(hi_tf36, TF36), TF18)
    lo18 = float_to_tfloat(tfloat_to_float(lo_tf36, TF36), TF18)
    return hi18 * POW3[18] + lo18


def unpack_dual_tf18(word36: int, high: bool = True) -> int:
    """F18UP: Unpack high or low TF18 half-word to TF36."""
    h18 = extract_field(word36, 18 if high else 0, 18, WORD_TRITS)
    return float_to_tfloat(tfloat_to_float(h18, TF18), TF36)


# ---------------------------------------------------------------------------
# TBFP-9 / TBFP-3 / TB1 Vectorized Block Quantization (T-Fabric Arithmetic)
# ---------------------------------------------------------------------------

def quantize_tbfp(arr: np.ndarray, mant_trits: int = 9) -> Tuple[np.ndarray, int]:
    """Quantize a float array into balanced-ternary block-floating-point (int64 mantissas, E_blk).

    Element value = m_i * 3^(E_blk - (mant_trits - 1)), with |m_i| <= (3^mant_trits - 1)//2.
    """
    arr = np.asarray(arr, dtype=np.float64)
    max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
    if max_abs == 0.0:
        return np.zeros(arr.shape, dtype=np.int64), 0

    m_max = (3**mant_trits - 1) // 2
    # Choose smallest integer block exponent E_blk such that max_abs / 3^(E_blk - (M-1)) <= m_max
    # Wait: 3^(E_blk - (M-1)) * m_max >= max_abs -> E_blk >= log3(max_abs / m_max) + (M - 1)
    e_blk = int(math.ceil(math.log(max_abs / m_max, 3.0))) + (mant_trits - 1)
    e_blk = max(-WMAX_6, min(WMAX_6, e_blk))
    scale = 3.0 ** (e_blk - (mant_trits - 1))
    q = np.rint(arr / scale).astype(np.int64)
    q = np.clip(q, -m_max, m_max)
    return q, e_blk


def dequantize_tbfp(q: np.ndarray, e_blk: int, mant_trits: int = 9) -> np.ndarray:
    """Reconstruct float64 array from balanced-ternary block-floating-point mantissas."""
    scale = 3.0 ** (e_blk - (mant_trits - 1))
    return q.astype(np.float64) * scale


def quantize_tb1(arr: np.ndarray, threshold_ratio: float = 0.35) -> Tuple[np.ndarray, int]:
    """Quantize float array to pure-trit TB1 {-1, 0, +1} with shared 6-trit scale exponent."""
    arr = np.asarray(arr, dtype=np.float64)
    max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
    if max_abs == 0.0:
        return np.zeros(arr.shape, dtype=np.int8), 0
    e_blk = int(round(math.log(max_abs, 3.0)))
    e_blk = max(-WMAX_6, min(WMAX_6, e_blk))
    thr = threshold_ratio * max_abs
    out = np.zeros(arr.shape, dtype=np.int8)
    out[arr >= thr] = 1
    out[arr <= -thr] = -1
    return out, e_blk


def pack_nonets_to_word(n3: int, n2: int, n1: int, n0: int) -> int:
    """Pack 4 x 9-trit nonets (TBFP-9 elements or SAEW tokens) into one T36 word."""
    return (
        wrap(n3, 9) * POW3[27]
        + wrap(n2, 9) * POW3[18]
        + wrap(n1, 9) * POW3[9]
        + wrap(n0, 9)
    )


def unpack_word_to_nonets(w: int) -> Tuple[int, int, int, int]:
    """Unpack one T36 word into 4 x 9-trit nonets (n3, n2, n1, n0)."""
    return (
        extract_field(w, 27, 9, WORD_TRITS),
        extract_field(w, 18, 9, WORD_TRITS),
        extract_field(w, 9, 9, WORD_TRITS),
        extract_field(w, 0, 9, WORD_TRITS),
    )


def pack_triads_to_word(triads: Tuple[int, ...] | list[int]) -> int:
    """Pack up to 12 x 3-trit triads (TBFP-3 elements, ±13) into one T36 word (MSB first)."""
    w = 0
    n = min(12, len(triads))
    for i in range(n):
        w += wrap(int(triads[i]), 3) * POW3[33 - 3 * i]
    return w


def unpack_word_to_triads(w: int) -> Tuple[int, ...]:
    """Unpack one T36 word into 12 x 3-trit triads (MSB first)."""
    return tuple(extract_field(w, 33 - 3 * i, 3, WORD_TRITS) for i in range(12))


def pack_trits_to_word(trits: Tuple[int, ...] | list[int]) -> int:
    """Pack up to 36 pure trits (-1/0/+1, TB1 weights) into one T36 word (trit 35..0 MSB first)."""
    w = 0
    n = min(WORD_TRITS, len(trits))
    for i in range(n):
        t = int(trits[i])
        t = 1 if t > 0 else (-1 if t < 0 else 0)
        w += t * POW3[35 - i]
    return w


def unpack_word_to_trits(w: int) -> Tuple[int, ...]:
    """Unpack one T36 word into 36 pure trits (trit 35..0 MSB first)."""
    return tuple(extract_field(w, 35 - i, 1, WORD_TRITS) for i in range(WORD_TRITS))


# ---------------------------------------------------------------------------
# Good Deciban Fixed-Point ALU (DB18 / DB36, 81 ticks/dB, 243-entry ROM)
# ---------------------------------------------------------------------------

DECIBAN_TICKS_PER_DB: int = 81  # 3^4 ticks per deciban

# Hardware 243-entry ROM for log-sum-exp delta: f(d) = 10*log10(1 + 10^(-d/10))
# Tabulated for d in [0 .. 24.2 dB] at step 0.1 dB (actually 243 entries covering 0..30 dB,
# step = 30.0 / 242 dB -> linear interpolation error < 0.002 dB!).
_DLSE_ROM_MAX_DB = 30.0
_DLSE_ROM_N = 243
_DLSE_STEP_TICKS = (_DLSE_ROM_MAX_DB * DECIBAN_TICKS_PER_DB) / (_DLSE_ROM_N - 1)
_DLSE_ROM = tuple(
    int(round(10.0 * math.log10(1.0 + 10.0 ** (-(i * _DLSE_ROM_MAX_DB / (_DLSE_ROM_N - 1)) / 10.0)) * DECIBAN_TICKS_PER_DB))
    for i in range(_DLSE_ROM_N)
)


def db_to_ticks(db: float, width: int = WORD_TRITS) -> int:
    return clamp(int(round(db * DECIBAN_TICKS_PER_DB)), width)


def ticks_to_db(ticks: int) -> float:
    return ticks / DECIBAN_TICKS_PER_DB


def deciban_add(a: int, b: int, width: int = WORD_TRITS) -> int:
    """DADD: Saturating balanced deciban addition."""
    return clamp(a + b, width)


def deciban_sub(a: int, b: int, width: int = WORD_TRITS) -> int:
    """DSUB: Saturating balanced deciban subtraction."""
    return clamp(a - b, width)


def deciban_lse(a: int, b: int, width: int = WORD_TRITS) -> int:
    """DLSE: Hardware 243-entry ROM + linear interpolation log-sum-exp in deciban ticks."""
    hi = a if a >= b else b
    diff = abs(a - b)
    max_diff = int(_DLSE_ROM_MAX_DB * DECIBAN_TICKS_PER_DB)
    if diff >= max_diff:
        return clamp(hi, width)
    pos = diff / _DLSE_STEP_TICKS
    idx = int(pos)
    frac = pos - idx
    if idx >= _DLSE_ROM_N - 1:
        delta = _DLSE_ROM[-1]
    else:
        delta = int(round(_DLSE_ROM[idx] * (1.0 - frac) + _DLSE_ROM[idx + 1] * frac))
    return clamp(hi + delta, width)


def deciban_exp_float(ticks: int) -> float:
    """DEXP: Convert deciban ticks to linear ratio 10^(db / 10)."""
    db = ticks / DECIBAN_TICKS_PER_DB
    if db > 1500.0:
        return float("inf")
    if db < -1500.0:
        return 0.0
    return 10.0 ** (db / 10.0)


def deciban_wald(evidence_ticks: int, reg_ticks: int = 0, simm_db: int = 0) -> int:
    """DWALD: Good-Wald 3-way sequential verdict in {-1, 0, +1}.

    Threshold T = |reg_ticks + 81 * simm_db| per Rev. 1.5 §4.3B.
    """
    thr = abs(reg_ticks + DECIBAN_TICKS_PER_DB * simm_db)
    if evidence_ticks >= thr:
        return 1
    if evidence_ticks <= -thr:
        return -1
    return 0
