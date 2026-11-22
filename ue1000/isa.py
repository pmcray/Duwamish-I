"""Duwamish-I (UE-1000) Instruction Set Architecture Encoder & Decoder (Rev. 1.5).

Implements exact trit-level encoding/decoding for:
  - Long 36-trit instructions (F = +1):
      [ F=+1(1) | op(5) | rd(2) | rs1(2) | rs2(2) | mode(2) | disp/imm(22) ]
  - Short 18-trit parcels (F = 0 scalar/logic/branch, F = -1 float/vector/deciban):
      [ F(1) | sop(4) | rd(2) | rs1(2) | rs2(2) | simm(7) ]
  - Dual-parcel word: Parcel A (trits 35..18) + Parcel B (trits 17..0).
    If F_A in {0, -1} and F_B == +1 -> hardware Trap 1 (ILLOP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .ternary import (
    HALF_TRITS,
    POW3,
    WMAX_7,
    WMAX_22,
    WORD_TRITS,
    extract_field,
    in_range,
)

# Register index (0..8) <-> 2-trit balanced code (-4..+4)
_REG_TO_CODE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: -1, 6: -2, 7: -3, 8: -4}
_CODE_TO_REG = {v: k for k, v in _REG_TO_CODE.items()}

# Long instruction addressing/width modes (2 trits: -4..+4)
MODE_DIRECT = 0      # M[EA] full T36
MODE_IMM = 1         # EA itself (immediate)
MODE_INDIRECT = -1   # M[M[EA]]
MODE_H1 = 2          # High T18 half-word
MODE_H0 = 3          # Low T18 half-word
MODE_TRYTE = 4       # 6-trit tryte indexed by rs1 (0..5)
MODE_NONET = -4      # 9-trit nonet indexed by rs1 (0..3)
MODE_VUNIT = -3      # Vector unit-stride (+1)
MODE_VSTRIDE = -2    # Vector strided (stride = A[rs1])

# Unique Opcodes per Principles of Operation Rev. 1.5, Section 4.3
SHORT0_OPS: Dict[str, int] = {
    "NOP": 0, "ADD": 1, "SUB": 2, "MUL": 3, "MULH": 4, "DIV": 5, "MOD": 6,
    "NEG": 7, "ABS": 8, "SHF": 9, "AND": 10, "OR": 11, "EQV": 12, "XTR": 13,
    "TRT": 14, "SEL": 15, "CMP": 16, "TST": 17, "AADD": 18, "ASUB": 19,
    "A2S": 20, "S2A": 21, "BR": 22, "B3": 23, "BN": 24, "BZ": 25, "BP": 26,
    "BCT": 27, "PUSH": 28, "POP": 29, "RET": 30,
}

SHORTT_OPS: Dict[str, int] = {
    "FAD": 1, "FSB": 2, "FMP": 3, "FDV": 4, "FMA": 5, "FNEG": 6, "FABS": 7,
    "FCM": 8, "FLT": 9, "FIX": 10, "F18PK": 11, "F18UP": 12, "FAD18": 13,
    "FMP18": 14, "DADD": 15, "DSUB": 16, "DLSE": 17, "DEXP": 18, "DWALD": 19,
    "VADD": 20, "VSUB": 21, "VMUL": 22, "VFMA": 23, "VF18OP": 24, "VRED": 25,
    "VKLEEN": 26, "VSVL": 27,
}

LONG_OPS: Dict[str, int] = {
    "LD": 1, "LDA": 2, "ST": 3, "STA": 4, "LEA": 5, "LDI": 6, "VLD": 7,
    "VST": 8, "VGTH": 9, "VSCT": 10, "JMP": 11, "J3": 12, "CALL": 13,
    "TEXEC": 20, "TWAIT": 21, "TSTAT": 22,
    "SVC": 30, "RTI": 31, "CHIO": 32, "ALGD": 33, "HLT": 34,
}

SHORT0_BY_CODE = {v: k for k, v in SHORT0_OPS.items()}
SHORTT_BY_CODE = {v: k for k, v in SHORTT_OPS.items()}
LONG_BY_CODE = {v: k for k, v in LONG_OPS.items()}


def reg_to_code(reg: int) -> int:
    if reg not in _REG_TO_CODE:
        raise ValueError(f"Register index {reg} out of range 0..8")
    return _REG_TO_CODE[reg]


def code_to_reg(code: int) -> int:
    return _CODE_TO_REG[code]


def encode_xtr_simm(p: int, n: int) -> int:
    """Centered bijective XTR encoding per Rev. 1.5 §4.3A: simm = 36*(n-18) + (p-18)."""
    if not (0 <= p <= 35 and 1 <= n <= 36):
        raise ValueError(f"XTR parameters out of range: p={p}, n={n}")
    return 36 * (n - 18) + (p - 18)


def decode_xtr_simm(simm: int) -> Tuple[int, int]:
    """Decode centered XTR simm -> (p, n)."""
    u = simm + 36 * 18 + 18
    n, p = divmod(u, 36)
    return p, n


@dataclass(frozen=True)
class DecodedInstruction:
    fmt: int           # +1 (Long 36t), 0 (Short F=0), -1 (Short F=T)
    op: int            # op or sop integer
    mnemonic: str      # resolved mnemonic or "ILLEGAL"
    rd: int            # 0..8
    rs1: int           # 0..8
    rs2: int           # 0..8
    mode: int          # 2-trit mode (Long only; 0 for Short)
    imm: int           # 22-trit disp/imm (Long) or 7-trit simm (Short)


def encode_short(
    fmt: int,
    sop: int,
    rd: int = 0,
    rs1: int = 0,
    rs2: int = 0,
    simm: int = 0,
) -> int:
    """Encode an 18-trit Short Parcel (fmt in {0, -1})."""
    if fmt not in (0, -1):
        raise ValueError(f"Short parcel fmt must be 0 or -1, got {fmt}")
    if not in_range(sop, 4):
        raise ValueError(f"Short sop {sop} out of 4-trit range [-40, +40]")
    if not in_range(simm, 7):
        raise ValueError(f"Short simm {simm} out of 7-trit range [-{WMAX_7}, +{WMAX_7}]")
    return (
        fmt * POW3[17]
        + sop * POW3[13]
        + reg_to_code(rd) * POW3[11]
        + reg_to_code(rs1) * POW3[9]
        + reg_to_code(rs2) * POW3[7]
        + simm
    )


def pack_dual_parcel(parcel_a: int, parcel_b: int) -> int:
    """Pack Parcel A (trits 35..18) and Parcel B (trits 17..0) into one 36-trit word."""
    fa = extract_field(parcel_a, 17, 1, HALF_TRITS)
    fb = extract_field(parcel_b, 17, 1, HALF_TRITS)
    if fa == 1 or fb == 1:
        raise ValueError("Cannot pack a Long (F=+1) instruction into a dual-parcel word")
    return parcel_a * POW3[18] + parcel_b


def encode_long(
    op: int,
    rd: int = 0,
    rs1: int = 0,
    rs2: int = 0,
    mode: int = 0,
    imm: int = 0,
) -> int:
    """Encode a 36-trit Long Instruction (F = +1)."""
    if not in_range(op, 5):
        raise ValueError(f"Long op {op} out of 5-trit range [-121, +121]")
    if not in_range(mode, 2):
        raise ValueError(f"Long mode {mode} out of 2-trit range [-4, +4]")
    if not in_range(imm, 22):
        raise ValueError(f"Long imm/disp {imm} out of 22-trit range [-{WMAX_22}, +{WMAX_22}]")
    return (
        1 * POW3[35]
        + op * POW3[30]
        + reg_to_code(rd) * POW3[28]
        + reg_to_code(rs1) * POW3[26]
        + reg_to_code(rs2) * POW3[24]
        + mode * POW3[22]
        + imm
    )


def decode_short(parcel: int) -> DecodedInstruction:
    """Decode an 18-trit short parcel."""
    fmt = extract_field(parcel, 17, 1, HALF_TRITS)
    sop = extract_field(parcel, 13, 4, HALF_TRITS)
    rd = code_to_reg(extract_field(parcel, 11, 2, HALF_TRITS))
    rs1 = code_to_reg(extract_field(parcel, 9, 2, HALF_TRITS))
    rs2 = code_to_reg(extract_field(parcel, 7, 2, HALF_TRITS))
    simm = extract_field(parcel, 0, 7, HALF_TRITS)
    if fmt == 0:
        mn = SHORT0_BY_CODE.get(sop, "ILLEGAL")
    elif fmt == -1:
        mn = SHORTT_BY_CODE.get(sop, "ILLEGAL")
    else:
        mn = "ILLEGAL"  # F_B == +1 in parcel B is ILLOP!
    return DecodedInstruction(fmt=fmt, op=sop, mnemonic=mn, rd=rd, rs1=rs1, rs2=rs2, mode=0, imm=simm)


def decode_long(word: int) -> DecodedInstruction:
    """Decode a 36-trit Long Instruction (trit 35 == +1)."""
    op = extract_field(word, 30, 5, WORD_TRITS)
    rd = code_to_reg(extract_field(word, 28, 2, WORD_TRITS))
    rs1 = code_to_reg(extract_field(word, 26, 2, WORD_TRITS))
    rs2 = code_to_reg(extract_field(word, 24, 2, WORD_TRITS))
    mode = extract_field(word, 22, 2, WORD_TRITS)
    imm = extract_field(word, 0, 22, WORD_TRITS)
    mn = LONG_BY_CODE.get(op, "ILLEGAL")
    return DecodedInstruction(fmt=1, op=op, mnemonic=mn, rd=rd, rs1=rs1, rs2=rs2, mode=mode, imm=imm)


def decode_word(word: int) -> Tuple[DecodedInstruction, ...]:
    """Decode a 36-trit instruction word -> 1 Long instruction or 2 Short parcels."""
    fa = extract_field(word, 35, 1, WORD_TRITS)
    if fa == 1:
        return (decode_long(word),)
    pa = extract_field(word, 18, 18, WORD_TRITS)
    pb = extract_field(word, 0, 18, WORD_TRITS)
    return (decode_short(pa), decode_short(pb))
