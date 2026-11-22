"""Tests for Rev. 1.5 ISA encoding, TRIAD-36 assembler, C-Complex CPU & Section 8 examples."""

import numpy as np
import pytest

from ue1000.cpu import TRAP_ILLOP, TRAP_PROT, TRAP_TFAULT
from ue1000.isa import (
    SHORT0_OPS,
    decode_word,
    decode_xtr_simm,
    encode_short,
    encode_xtr_simm,
)
from ue1000.machine import UE1000Machine
from ue1000.ternary import POW3, WMAX_7
from ue1000.tfabric import MODE_TBFP9_GEMM, TileTaskDescriptor
from ue1000.tfloat import TF36, db_to_ticks, float_to_tfloat, pack_nonets_to_word, unpack_word_to_nonets
from ue1000.triad36 import AssemblyError, assemble


def test_centered_xtr_bijective_over_all_p_and_n():
    for n in range(1, 37):
        for p in range(36):
            code = encode_xtr_simm(p, n)
            assert -WMAX_7 <= code <= WMAX_7
            assert decode_xtr_simm(code) == (p, n)


def test_assembler_rejects_ldi_overflow_and_illegal_parcel_packing():
    # Bug 1 regression check: 27-trit immediate overflows LDI's 22-trit field!
    with pytest.raises(ValueError):
        assemble("LDI S2, #3812798742493")

    # Bug 2 regression check: LDI (36t Long) cannot pack beside RET (18t Short)!
    with pytest.raises(AssemblyError):
        assemble("LDI S4, #+1 | RET #0")

    # DWALD simm overflow check (> 1093 dB rejected!)
    with pytest.raises(ValueError):
        assemble("DWALD S1, S0, #1620")


def test_dual_parcel_illegal_fb_plus1_triggers_illop_trap():
    """Verify Rev. 1.5 §4.2: if F_A in {0, T} and F_B == +1 -> Trap 1 (ILLOP)."""
    m = UE1000Machine()
    pa = encode_short(0, SHORT0_OPS["NOP"])
    bad_pb = 1 * POW3[17]  # F_B == +1 illegal in Parcel B!
    bad_word = pa * POW3[18] + bad_pb
    m.mem.write(100, bad_word, supervisor=True)
    m.cpu0.psw.mode = 1  # User mode
    m.cpu0.pc = 100
    m.cpu0.step()
    assert m.cpu0.last_trap == TRAP_ILLOP


def test_section_8_1_good_wald_detector_both_register_and_whole_db_modes():
    """Assemble and execute the exact Rev. 1.5 Section 8.1 Good-Wald detector."""
    asm = """
              ADD     S1, S0, #0        | NOP
    LOOP:     LD      S3, 0(A1)
              DADD    S1, S1, S3        | AADD A1, A1, #1
              DWALD   S1, S0, #20       | B3   WALD_0
    WALD_N:   BR      REJECT            | NOP
    WALD_0:   BCT     A2, #LOOP         | BR   UNDECIDED
    WALD_P:   ADD     S4, S0, #+1       | RET  #0
    REJECT:   ADD     S4, S0, #-1       | RET  #0
    UNDECIDED:ADD     S4, S0, #0        | RET  #0
    """
    m = UE1000Machine()
    prog = m.assemble_and_load(asm, origin=100)
    m.mem.write(7000, 0, supervisor=True)

    for stream_db, expected_trit in [
        ([+8.0, +7.0, +9.0], +1),      # +24 dB >= +20 dB -> ARTIFICIAL (+1)
        ([-9.0, -8.0, -7.0], -1),      # -24 dB <= -20 dB -> NATURAL (-1)
        ([+4.0, -3.0, +5.0], 0),       #  +6 dB in (-20, +20) -> UNDECIDED (0)
    ]:
        for idx, db in enumerate(stream_db):
            m.mem.write(500 + idx, db_to_ticks(db), supervisor=True)
        cpu = m.cpu0
        cpu.pc = prog.origin
        cpu.halted = False
        cpu.a[1] = 500
        cpu.a[2] = len(stream_db)
        cpu.s[8] = 7000
        cpu.run()
        assert cpu.s[4] == expected_trit
        assert cpu.c == expected_trit


def test_section_8_2_texec_twait_s0_shorthand_and_cutover_status():
    """Assemble and execute Section 8.2 TEXEC A1, S0 / TWAIT S0, #0 with live fault cutover!"""
    m = UE1000Machine()
    # Store 6x6 identity-like TBFP-9 matrices A and B at addr 1000 and 1100, output C at 1200
    a_addr, b_addr, c_addr, ttd_addr = 1000, 1100, 1200, 900
    # E_blk = 8 -> scale = 3^(8-8) = 1.0, so nonets hold exact small integers!
    mat_a = np.eye(6, dtype=np.int64) * 3
    mat_b = np.arange(1, 37, dtype=np.int64).reshape(6, 6)
    for base, mat in ((a_addr, mat_a), (b_addr, mat_b)):
        flat = mat.ravel()
        for i in range(0, flat.size, 4):
            m.mem.write(base + i // 4, pack_nonets_to_word(int(flat[i]), int(flat[i+1]), int(flat[i+2]), int(flat[i+3])), supervisor=True)

    ttd = TileTaskDescriptor(
        mode=MODE_TBFP9_GEMM, zgate=1, abft=1, wgate=0,
        base_a=a_addr, base_b=b_addr, base_c=c_addr,
        m=6, n=6, k=6, e_a=8, e_b=8, e_c=8,
    )
    for idx, w in enumerate(ttd.pack_9_words()):
        m.mem.write(ttd_addr + idx, w, supervisor=True)

    # Inject a hardware tile fault on logical tile 5 to verify B3 lands on RECOVERED (-1)!
    m.tfabric.inject_tile_fault(logical_tile=5, reason="ABFT_CHECKSUM_MISMATCH")

    asm = f"""
              LEA     A1, {ttd_addr}(A0)
              TEXEC   A1, S0
              TWAIT   S0, #0
              B3      CHK_0             | NOP
    CHK_NEG:  ADD     S5, S0, #-1       | RET  #0
    CHK_0:    ADD     S5, S0, #0        | RET  #0
    CHK_POS:  ADD     S5, S0, #+1       | RET  #0
    """
    m.assemble_and_load(asm, origin=200)
    m.mem.write(7000, 0, supervisor=True)
    m.cpu0.s[8] = 7000
    m.cpu0.run()

    # TWAIT returned C = -1 (recovered by Ashby <75 us cutover!), and C == 3 * B exactly!
    assert m.cpu0.s[5] == -1
    assert len(m.vcomplex.cutovers) == 1
    assert m.vcomplex.cutovers[0].hw_cutover_us < 75.0
    got = []
    for i in range(9):
        got.extend(unpack_word_to_nonets(m.mem.read(c_addr + i, supervisor=True)[0]))
    assert np.array_equal(np.array(got).reshape(6, 6), 3 * mat_b)


def test_cpu_tst_vs_fcm_on_undecided_and_vector_fma():
    m = UE1000Machine()
    m.mem.write(600, TF36.undecided_word, supervisor=True)
    asm = """
              LD      S1, 600(A0)
              TST     S1                | SEL S2, S0, #8
              FCM     S1, S0            | SEL S3, S0, #8
              HLT
    """
    m.assemble_and_load(asm, origin=300)
    m.cpu0.run()
    # Raw TST on UNDECIDED (-364) gave -1; FCM explicitly gave 0 (UNDECIDED) and latched PSW.FU!
    assert m.cpu0.s[2] == -1
    assert m.cpu0.s[3] == 0
    assert m.cpu0.psw.fu == 1
