"""Tests for [42, 36, 4]_3 Ternary SEC-DED + Triple-Erasure ECC & Tier 3a Memory."""

import random
import pytest

from ue1000.ecc import decode_word_42, encode_word_42
from ue1000.memory import (
    FERRITE_MIN_ADDR,
    SUP_TDRAM_MIN,
    USER_TDRAM_MAX,
    MemoryDEDError,
    MemoryProtectionError,
    TernaryMemory,
)
from ue1000.ternary import WMAX_36


def test_ecc_all_84_single_trit_errors_corrected():
    """Exhaustively test ALL 42 positions x 2 non-zero error polarity flips = 84 SEC cases."""
    word = 12_345_678_901_234_567
    cw = encode_word_42(word)
    assert decode_word_42(cw).status == "CLEAN"
    assert decode_word_42(cw).data_word == word

    for pos in range(42):
        for delta in (1, -1):
            corrupt = list(cw)
            r = (corrupt[pos] + delta) % 3
            corrupt[pos] = -1 if r == 2 else r
            res = decode_word_42(corrupt)
            assert res.status == "SEC_CORRECTED"
            assert res.data_word == word
            assert res.corrected_positions == (pos,)


def test_ecc_double_trit_errors_detected_ded():
    """Verify d_min = 4 guarantees detection of ANY 2 arbitrary trit errors."""
    rng = random.Random(1972)
    word = -55_555_555_555_555
    cw = encode_word_42(word)
    for _ in range(300):
        p1, p2 = rng.sample(range(42), 2)
        corrupt = list(cw)
        for p in (p1, p2):
            d = rng.choice((1, -1))
            r = (corrupt[p] + d) % 3
            corrupt[p] = -1 if r == 2 else r
        res = decode_word_42(corrupt)
        assert res.status == "UNCORRECTABLE_DED"


def test_ecc_one_two_and_three_erasures_uniquely_recovered():
    """Verify UE-TD2000 (1,1) hardware erasure flags recover up to 3 simultaneous erasures!"""
    rng = random.Random(1973)
    word = WMAX_36 - 987654321
    cw = encode_word_42(word)
    for k in (1, 2, 3):
        for _ in range(80):
            epos = rng.sample(range(42), k)
            # Corrupt erased positions arbitrarily (decoder ignores their raw value)
            corrupt = list(cw)
            for p in epos:
                corrupt[p] = rng.choice((-1, 0, 1))
            res = decode_word_42(corrupt, erasure_positions=epos)
            assert res.status == "ERASURE_CORRECTED"
            assert res.data_word == word
            assert res.erasure_count == k

    # 4 erasures must raise UNCORRECTABLE_DED
    res4 = decode_word_42(cw, erasure_positions=(0, 7, 19, 41))
    assert res4.status == "UNCORRECTABLE_DED"


def test_memory_protection_ferrite_and_bank_cutover():
    mem = TernaryMemory(strict_ecc=True)
    # Ferrite Vault write/read in supervisor mode
    mem.write(-1, 777, supervisor=True)
    val, lat = mem.read(-1, supervisor=True)
    assert val == 777 and lat == 600

    # User mode forbidden from negative addresses & M[0]
    with pytest.raises(MemoryProtectionError):
        mem.read(-1, supervisor=False)
    with pytest.raises(MemoryProtectionError):
        mem.read(0, supervisor=False)
    with pytest.raises(MemoryProtectionError):
        mem.write(0, 1, supervisor=True)  # Zero-Guard forbids even supervisor writes!

    # User arena accommodates full SAEW-32M footprint inside positive addresses!
    mem.write(40_000_000, -424242, supervisor=False)
    assert mem.read(40_000_000, supervisor=False)[0] == -424242

    # Inject 3 erasures -> transparently corrected; then scrub-cutover bank onto a hot spare!
    addr = 10_000
    mem.write(addr, 314159265, supervisor=True)
    mem.inject_erasure(addr, 3)
    mem.inject_erasure(addr, 17)
    mem.inject_erasure(addr, 39)
    assert mem.read(addr, supervisor=True)[0] == 314159265
    lbank = mem.logical_bank(addr)
    old_p, new_p, scrubbed = mem.scrub_and_cutover_bank(lbank)
    assert old_p != new_p and scrubbed >= 1
    assert mem.bank_health[old_p].retired is True
    assert mem.read(addr, supervisor=True)[0] == 314159265

    # Inject 2 unflagged trit flips -> MEMDED!
    mem.inject_trit_flip(addr, 2, +1)
    mem.inject_trit_flip(addr, 25, -1)
    with pytest.raises(MemoryDEDError):
        mem.read(addr, supervisor=True)
