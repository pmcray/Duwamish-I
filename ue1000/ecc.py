"""Proven [42, 36, 4]_3 Ternary SEC-DED + Triple-Erasure ECC Hardware Model.

Conforms strictly to Principles of Operation (Rev. 1.5), Section 3.3 & Appendix A.
Uses the exact systematic parity-check matrix H = [P | I_6] over GF(3) whose 6 rows
of P are given in UE Heptavintimal (MSB-first trits 35..0):
    c0: LCNSPSGHXODC   c1: LSFDELFVLAQW   c2: GOTXTDZXFHFF
    c3: WMRQGXKSZJUY   c4: AMMM000AMENN   c5: 0000MMMMMMMM

Combined with UE-TD2000 differential 2-cell hardware erasure flags ((1,1) forbidden state):
  - |E| = 0: Corrects any 1 arbitrary trit error (SEC), detects any 2 arbitrary trit errors (DED).
  - 1 <= |E| <= 3: Uniquely solves and corrects 1, 2, or 3 simultaneous hardware erasures!
  - |E| >= 4: Raises uncorrectable erasure fault (MEMDED -> Ashby bank cutover).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .ternary import (
    ECC_CHECK_TRITS,
    MEM_WORD_TRITS,
    WORD_TRITS,
    hept_to_trits_msb,
    int_to_trits,
    trits_to_int,
)

ECC_P_ROWS_HEPT: Tuple[str, ...] = (
    "LCNSPSGHXODC",
    "LSFDELFVLAQW",
    "GOTXTDZXFHFF",
    "WMRQGXKSZJUY",
    "AMMM000AMENN",
    "0000MMMMMMMM",
)


def _mod3_to_bal(x: int) -> int:
    r = x % 3
    return -1 if r == 2 else r


def _build_h_matrix() -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
    """Build exact 6x42 matrix H = [P | I_6] over GF(3) ({0, 1, 2}) and its 42 columns."""
    p_rows = [[t % 3 for t in hept_to_trits_msb(rh)] for rh in ECC_P_ROWS_HEPT]
    h_rows = [
        tuple(p_rows[r] + [1 if c == r else 0 for c in range(ECC_CHECK_TRITS)])
        for r in range(ECC_CHECK_TRITS)
    ]
    h_cols = tuple(tuple(h_rows[r][c] for r in range(ECC_CHECK_TRITS)) for c in range(MEM_WORD_TRITS))
    return tuple(h_rows), h_cols


H_ROWS, H_COLS = _build_h_matrix()

# Precomputed single-trit syndrome lookup: syndrome_tuple -> (col_index, error_trit_in_{+1,-1})
_SEC_LUT: Dict[Tuple[int, ...], Tuple[int, int]] = {}
for _col_idx, _col in enumerate(H_COLS):
    _SEC_LUT[_col] = (_col_idx, 1)
    _neg_col = tuple((2 * x) % 3 for x in _col)
    _SEC_LUT[_neg_col] = (_col_idx, -1)


@dataclass
class ECCDecodeResult:
    data_word: int
    status: str  # "CLEAN" | "SEC_CORRECTED" | "ERASURE_CORRECTED" | "UNCORRECTABLE_DED"
    corrected_positions: Tuple[int, ...] = ()
    erasure_count: int = 0
    syndrome: Tuple[int, ...] = (0, 0, 0, 0, 0, 0)


def encode_word_42(data_word: int) -> List[int]:
    """Encode a 36-trit signed integer into a 42-trit MSB-first codeword [d35..d0, c0..c5]."""
    d_lsb = int_to_trits(data_word, WORD_TRITS)
    d_msb = list(reversed(d_lsb))  # index 0 = trit 35, index 35 = trit 0
    checks: List[int] = []
    for r in range(ECC_CHECK_TRITS):
        s = sum(H_ROWS[r][j] * (d_msb[j] % 3) for j in range(WORD_TRITS)) % 3
        checks.append(_mod3_to_bal((-s) % 3))
    return d_msb + checks


def compute_syndrome(codeword_42: Sequence[int]) -> Tuple[int, ...]:
    """Compute 6-trit syndrome s = H * w^T mod 3 in {0, 1, 2}^6."""
    w_mod = [t % 3 for t in codeword_42]
    return tuple(
        sum(H_ROWS[r][c] * w_mod[c] for c in range(MEM_WORD_TRITS)) % 3
        for r in range(ECC_CHECK_TRITS)
    )


def _solve_erasures_gf3(
    erasure_positions: Sequence[int], syndrome: Tuple[int, ...]
) -> Optional[List[int]]:
    """Solve H_E * e_E = syndrome (mod 3) for 1 <= |E| <= 3 and check all 6 equations!

    Because d_min = 4, H_E has full column rank |E|, so a solution exists iff no
    additional silent error corrupted the word outside E.
    """
    k = len(erasure_positions)
    # Build 6 x (k + 1) augmented matrix over GF(3)
    aug = [
        [H_COLS[pos][r] for pos in erasure_positions] + [syndrome[r]]
        for r in range(ECC_CHECK_TRITS)
    ]
    row = 0
    for col in range(k):
        piv = next((r for r in range(row, ECC_CHECK_TRITS) if aug[r][col] != 0), None)
        if piv is None:
            return None
        aug[row], aug[piv] = aug[piv], aug[row]
        inv = 1 if aug[row][col] == 1 else 2
        aug[row] = [(x * inv) % 3 for x in aug[row]]
        for r in range(ECC_CHECK_TRITS):
            if r != row and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [(aug[r][c] - f * aug[row][c]) % 3 for c in range(k + 1)]
        row += 1

    # Verify remaining (6 - k) redundant parity equations equal 0 mod 3!
    for r in range(k, ECC_CHECK_TRITS):
        if aug[r][k] != 0:
            return None

    return [_mod3_to_bal(aug[r][k]) for r in range(k)]


def decode_word_42(
    codeword_42: Sequence[int],
    erasure_positions: Sequence[int] = (),
) -> ECCDecodeResult:
    """Decode a 42-trit codeword with optional UE-TD2000 hardware erasure positions.

    Erased positions have their unknown values substituted with 0 prior to syndrome check.
    """
    cw = list(codeword_42)
    e_pos = tuple(sorted(set(erasure_positions)))
    e_cnt = len(e_pos)

    if e_cnt >= 4:
        return ECCDecodeResult(
            data_word=0,
            status="UNCORRECTABLE_DED",
            erasure_count=e_cnt,
            syndrome=compute_syndrome(cw),
        )

    # Zero out hardware-flagged erased positions before computing syndrome
    for pos in e_pos:
        cw[pos] = 0

    syn = compute_syndrome(cw)

    # Case 2: 1..3 known hardware erasures from UE-TD2000 (1,1) cells
    if e_cnt > 0:
        sol = _solve_erasures_gf3(e_pos, syn)
        if sol is None:
            return ECCDecodeResult(
                data_word=0,
                status="UNCORRECTABLE_DED",
                erasure_count=e_cnt,
                syndrome=syn,
            )
        # Note: H * (cw_true + err) = syn where cw_read = cw_true - missing_val -> cw_true = cw - e?
        # Check sign: H * (cw + delta) = 0 -> H * delta = -syn! Wait: H * e = syn -> cw_true = cw - e (mod 3)!
        for pos, err_val in zip(e_pos, sol):
            cw[pos] = _mod3_to_bal((cw[pos] - err_val) % 3)
        data = trits_to_int(list(reversed(cw[:WORD_TRITS])))
        return ECCDecodeResult(
            data_word=data,
            status="ERASURE_CORRECTED",
            corrected_positions=e_pos,
            erasure_count=e_cnt,
            syndrome=syn,
        )

    # Case 0: Clean read
    if syn == (0, 0, 0, 0, 0, 0):
        data = trits_to_int(list(reversed(cw[:WORD_TRITS])))
        return ECCDecodeResult(data_word=data, status="CLEAN", syndrome=syn)

    # Case 1: Single-trit error (SEC) vs Double-trit error (DED)
    match = _SEC_LUT.get(syn)
    if match is not None:
        col_idx, err_trit = match
        cw[col_idx] = _mod3_to_bal((cw[col_idx] - err_trit) % 3)
        data = trits_to_int(list(reversed(cw[:WORD_TRITS])))
        return ECCDecodeResult(
            data_word=data,
            status="SEC_CORRECTED",
            corrected_positions=(col_idx,),
            syndrome=syn,
        )

    return ECCDecodeResult(
        data_word=0,
        status="UNCORRECTABLE_DED",
        syndrome=syn,
    )
