"""Comprehensive conformance tests for the completed Rev. 1.5 Hardware Spec (Tiers 1–4).

Covers:
  - Tier 3b Asynchronous Data Channels (CHIO, 6-word CCB chaining, Disc, 9-Track Tape +
    READ_BACKWARD, Alexandrian Archival Silo + WEAR provenance security, Channel 8 SETI
    Baseband Ring, Level -8 Digitisation Chute, async DMA BUSY -> OK lifecycle)
  - Tier 2 T-Fabric: Authentic Good-Thomas PFA Mixed-Radix-6 FFT, all 6 TTD modes
    (ELEM, TBFP-9 with stride/accum/bias, TBFP-3 12-triads/word, TB1 36-trits/word,
    Radix-3 batch FFT, Radix-6 PFA FFT), truly asynchronous TEXEC/TWAIT overlap & timeout,
    and extended TSTAT/ALGD commands
  - Tier 3a 81-bank crossbar stride contention (coprime to 3 = 0 conflicts vs mod-3 stalls),
    proactive Ashby bank scrub-cutover, and live MMIO control apertures
  - Tier 1 C-Complex: VF18OP (54 floats/vector), packed 2xDB18 DADD/DSUB, full RTI trap
    frame restore (PC/PSW/C/VL), watchdog TRAP_TIMER, PSW.ov / TRAP_OVF, and concurrent
    multi-CPU (CPU 0..7) execution with partitioned T-Fabric tiles
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from ue1000.channels import (
    CH_BUSY,
    CH_MEDIA_ERR,
    CH_OK,
    CH_PROVENANCE_DENIED,
    ChannelControlBlock,
    DEV_DISC,
    DEV_L8_CHUTE,
    DEV_SETI_RING,
    DEV_SILO,
    DEV_TAPE,
    Level8DigitisationChute,
    OP_MARK,
    OP_READ,
    OP_READ_BACKWARD,
    OP_REWIND,
    OP_SEEK,
    OP_VERIFY,
    OP_WRITE,
    decode_4tape_frames_to_word,
    decode_8flux_to_5trit,
    encode_5trit_to_8flux,
    encode_word_to_4tape_frames,
)
from ue1000.cpu import (
    PSW,
    TRAP_OVF,
    TRAP_PRIV,
    TRAP_TFAULT,
    TRAP_TIMER,
)
from ue1000.machine import (
    MMIO_BANK_BASE,
    MMIO_BANK_CUTOVER_CMD,
    MMIO_CHANNEL_BASE,
    MMIO_SYS_ID,
    MMIO_TILE_BASE,
    MMIO_VCOMPLEX_SUMMARY,
    UE1000Machine,
)
from ue1000.ternary import POW3, WMAX_18, WMAX_36, extract_field, int_to_trits, trits_to_int
from ue1000.tfabric import (
    MODE_ELEM,
    MODE_RADIX3_FFT,
    MODE_RADIX6_FFT,
    MODE_TB1_ASSOC,
    MODE_TBFP3_GEMM,
    MODE_TBFP9_GEMM,
    TileTaskDescriptor,
)
from ue1000.tfloat import (
    TF18,
    TF36,
    float_to_tfloat,
    pack_dual_tf18,
    pack_nonets_to_word,
    pack_triads_to_word,
    pack_trits_to_word,
    tfloat_to_float,
    unpack_dual_tf18,
    unpack_word_to_nonets,
    unpack_word_to_triads,
)


def _write_ccb(m: UE1000Machine, addr: int, ccb: ChannelControlBlock) -> None:
    for i, w in enumerate(ccb.pack_6_words()):
        m.mem.write(addr + i, w, supervisor=True)


def test_tier3b_media_coding_disc_and_tape_read_backward():
    """Verify 5t/8-flux disc coding, 4-frame 9-track tape coding, and CHIO Disc + Tape."""
    # Exhaustive 5-trit-per-8-flux group coding check (all 243 balanced 5-trit states!)
    seen = set()
    for v in range(-121, 122):
        t5 = int_to_trits(v, 5)
        fb = encode_5trit_to_8flux(t5)
        assert 0 <= fb <= 242
        seen.add(fb)
        assert decode_8flux_to_5trit(fb) == t5
    assert len(seen) == 243

    # 9-track 4-frame tape word roundtrip
    sample_word = -12_345_678_901_234_567
    frames = encode_word_to_4tape_frames(sample_word)
    assert len(frames) == 4 and all(len(f) == 9 for f in frames)
    assert decode_4tape_frames_to_word(frames) == sample_word

    m = UE1000Machine()
    payload = [111, -222, 333, -444, 555, -666]
    for i, v in enumerate(payload):
        m.mem.write(2000 + i, v, supervisor=True)

    # CCB 1 (addr 500): Disc Unit 3 WRITE 6 words at sector offset 100
    _write_ccb(m, 500, ChannelControlBlock(
        device_type=DEV_DISC, unit=3, op=OP_WRITE, mem_addr=2000, word_count=6, media_pos=100,
    ))
    # CCB 2 (addr 510): Disc Unit 3 READ back into 2100
    _write_ccb(m, 510, ChannelControlBlock(
        device_type=DEV_DISC, unit=3, op=OP_READ, mem_addr=2100, word_count=6, media_pos=100,
    ))
    # CCB 3 (addr 520): Disc Unit 3 VERIFY against 2100
    _write_ccb(m, 520, ChannelControlBlock(
        device_type=DEV_DISC, unit=3, op=OP_VERIFY, mem_addr=2100, word_count=6, media_pos=100,
    ))
    # CCB 4 (addr 530): 9-Track Tape Unit 0 WRITE record from 2000
    _write_ccb(m, 530, ChannelControlBlock(
        device_type=DEV_TAPE, unit=0, op=OP_WRITE, mem_addr=2000, word_count=6,
    ))
    # CCB 5 (addr 540): 9-Track Tape Unit 0 READ_BACKWARD into 2200 (Setun/IBM 2401 reverse!)
    _write_ccb(m, 540, ChannelControlBlock(
        device_type=DEV_TAPE, unit=0, op=OP_READ_BACKWARD, mem_addr=2200, word_count=6,
    ))
    # CCB 6 (addr 550): Tape REWIND + forward READ into 2300 via chain!
    _write_ccb(m, 550, ChannelControlBlock(
        device_type=DEV_TAPE, unit=0, op=OP_REWIND, chain_ptr=560,
    ))
    _write_ccb(m, 560, ChannelControlBlock(
        device_type=DEV_TAPE, unit=0, op=OP_READ, mem_addr=2300, word_count=6,
    ))

    asm = """
        LEA  A1, 500(A0)
        CHIO A1, #1
        LEA  A1, 510(A0)
        CHIO A1, #1
        LEA  A1, 520(A0)
        CHIO A1, #1
        LEA  A1, 530(A0)
        CHIO A1, #2
        LEA  A1, 540(A0)
        CHIO A1, #2
        LEA  A1, 550(A0)
        CHIO A1, #2
        HLT
    """
    m.assemble_and_load(asm, origin=100)
    m.cpu0.run()

    assert m.cpu0.c == 1
    assert [m.mem.read(2100 + i, supervisor=True)[0] for i in range(6)] == payload
    assert [m.mem.read(2200 + i, supervisor=True)[0] for i in range(6)] == list(reversed(payload))
    assert [m.mem.read(2300 + i, supervisor=True)[0] for i in range(6)] == payload

    # Verify CHIO in User mode raises TRAP_PRIV (Trap 4)!
    m2 = UE1000Machine()
    m2.assemble_and_load("CHIO A1, #1\nHLT", origin=100)
    m2.cpu0.psw.mode = 1
    m2.cpu0.run()
    assert m2.cpu0.last_trap == TRAP_PRIV


def test_tier3b_silo_provenance_seti_ring_l8_chute_and_async_dma():
    """Verify Alexandrian Silo WEAR provenance, Channel 8 SETI ring, L-8 Chute, & async DMA."""
    m = UE1000Machine()

    # 1. Alexandrian Silo with WEAR compartment clearance 7777
    secret_words = [1971, 1973, -999, 4242]
    m.io_channels.silo.mount_cartridge(
        cartridge_id=9, words=secret_words, label="WEAR-BYURAKAN-71", provenance_compartment=7777,
    )
    # Unauthorized attempt (tag 1234 != 7777) -> PROVENANCE_DENIED (-2) + Beer VSM audit!
    _write_ccb(m, 600, ChannelControlBlock(
        device_type=DEV_SILO, op=OP_READ, mem_addr=3000, word_count=4, media_pos=9, provenance_tag=1234,
    ))
    # Authorized attempt (tag 7777) -> OK (+1)
    _write_ccb(m, 610, ChannelControlBlock(
        device_type=DEV_SILO, op=OP_READ, mem_addr=3100, word_count=4, media_pos=9, provenance_tag=7777,
    ))
    m.assemble_and_load(
        "LEA A1, 600(A0)\nCHIO A1, #3\nLEA A1, 610(A0)\nCHIO A1, #3\nHLT", origin=100
    )
    m.cpu0.run()

    st_bad, n_bad = ChannelControlBlock.unpack_status(m.mem.read(605, supervisor=True)[0])
    st_ok, n_ok = ChannelControlBlock.unpack_status(m.mem.read(615, supervisor=True)[0])
    assert st_bad == CH_PROVENANCE_DENIED and n_bad == 0
    assert [m.mem.read(3000 + i, supervisor=True)[0] for i in range(4)] == [0, 0, 0, 0]
    assert m.io_channels.silo.provenance_denials == 1
    assert m.vcomplex.algedonic_alarms >= 1
    assert st_ok == CH_OK and n_ok == 4
    assert [m.mem.read(3100 + i, supervisor=True)[0] for i in range(4)] == secret_words

    # 2. Level -8 Digitisation Chute (OBB-729 text) chained to Disc staging
    text = "JAMESON CAT-73: SETUN-70 BYURAKAN"
    n_words = m.io_channels.l8_chute.feed_text_record(text, provenance_tag=555)
    _write_ccb(m, 620, ChannelControlBlock(
        device_type=DEV_L8_CHUTE, op=OP_READ, mem_addr=3200, word_count=n_words, chain_ptr=630,
    ))
    _write_ccb(m, 630, ChannelControlBlock(
        device_type=DEV_DISC, unit=0, op=OP_WRITE, mem_addr=3200, word_count=n_words, media_pos=0,
    ))
    m.assemble_and_load("LEA A1, 620(A0)\nCHIO A1, #4\nHLT", origin=120)
    m.cpu0.run()
    got_words = [m.mem.read(3200 + i, supervisor=True)[0] for i in range(n_words)]
    assert Level8DigitisationChute.decode_words_to_text(got_words) == text
    assert m.mem.read(623, supervisor=True)[0] == 555  # Chute provenance stamped into CCB[3]

    # 3. Oliver SETI Baseband Ring enforced on dedicated Channel 8
    t = np.arange(27, dtype=np.float64)
    sig = np.exp(2j * np.pi * 5 * t / 27.0)
    m.io_channels.seti_ring.ingest_complex_samples(sig)
    w_count = m.io_channels.seti_ring.buffered_words
    _write_ccb(m, 640, ChannelControlBlock(
        device_type=DEV_SETI_RING, op=OP_READ, mem_addr=3400, word_count=w_count,
    ))
    # Reject on Channel 1 (wrong channel!), succeed on Channel 8!
    m.assemble_and_load("LEA A1, 640(A0)\nCHIO A1, #1\nHLT", origin=140)
    m.cpu0.run()
    assert ChannelControlBlock.unpack_status(m.mem.read(645, supervisor=True)[0])[0] == CH_MEDIA_ERR

    m.assemble_and_load("LEA A1, 640(A0)\nCHIO A1, #8\nHLT", origin=150)
    m.cpu0.run()
    st8, transferred8 = ChannelControlBlock.unpack_status(m.mem.read(645, supervisor=True)[0])
    assert st8 == CH_OK and transferred8 == w_count

    # 4. Async DMA mode: BUSY (0) immediately after CHIO -> OK (+1) after simulated time advances
    m_async = UE1000Machine(async_dma=True)
    m_async.io_channels.seti_ring.ingest_words([10, 20, 30, 40])
    _write_ccb(m_async, 700, ChannelControlBlock(
        device_type=DEV_SETI_RING, op=OP_READ, mem_addr=4000, word_count=4,
    ))
    m_async.assemble_and_load("LEA A1, 700(A0)\nCHIO A1, #8\nHLT", origin=100)
    m_async.cpu0.run()
    # Immediately after 4-cycle CHIO + HLT, 1600 ns transfer is still BUSY!
    st_now, _ = ChannelControlBlock.unpack_status(m_async.mem.read(705, supervisor=True)[0])
    assert st_now == CH_BUSY
    # Advance simulated time past transfer duration -> commits OK!
    m_async.io_channels.advance_to_ns(10_000)
    st_done, cnt_done = ChannelControlBlock.unpack_status(m_async.mem.read(705, supervisor=True)[0])
    assert st_done == CH_OK and cnt_done == 4
    assert [m_async.mem.read(4000 + i, supervisor=True)[0] for i in range(4)] == [10, 20, 30, 40]


def test_good_thomas_radix6_pfa_fft_and_all_six_ttd_modes():
    """Verify genuine Good-Thomas PFA (3^p * 2^q) FFT and all 6 TTD modes via TEXEC/TWAIT."""
    m = UE1000Machine()

    # 1. Direct Good-Thomas PFA accuracy for N = 3^3 * 2^3 = 216 and N = 3^4 * 2^2 = 324
    for n, k1, k2 in [(216, 17, 91), (324, 29, 140)]:
        t = np.arange(n, dtype=np.float64)
        sig = 0.7 * np.exp(2j * np.pi * k1 * t / n) + 0.35 * np.exp(2j * np.pi * k2 * t / n)
        spec_pfa, stats = m.tfabric.fft_radix6_pfa_tbfp9c(sig)
        spec_ref = np.fft.fft(sig)
        rel_err = np.linalg.norm(spec_pfa - spec_ref) / np.linalg.norm(spec_ref)
        assert stats.mode == MODE_RADIX6_FFT
        assert rel_err < 2e-3
        assert int(np.argmax(np.abs(spec_pfa))) == k1

    # 2. TTD Mode +1 (TBFP-9 GEMM) with accum=+1, W6 bias_addr, and non-zero stride
    a_base, b_base, c_base, bias_base, ttd_base = 5000, 5100, 5200, 5300, 4900
    # 2x4 A * 4x4 B with stride = 2 words per row
    A = np.array([[1, 2, 0, -1], [0, 1, 2, 1]], dtype=np.int64)
    B = np.eye(4, dtype=np.int64) * 2
    C_init = np.ones((2, 4), dtype=np.int64) * 5
    bias = np.array([10, 20, 30, 40], dtype=np.int64)
    for r in range(2):
        m.mem.write(a_base + r * 2, pack_nonets_to_word(*map(int, A[r])), supervisor=True)
        m.mem.write(c_base + r * 2, pack_nonets_to_word(*map(int, C_init[r])), supervisor=True)
    for r in range(4):
        m.mem.write(b_base + r * 2, pack_nonets_to_word(*map(int, B[r])), supervisor=True)
    m.mem.write(bias_base, pack_nonets_to_word(*map(int, bias)), supervisor=True)

    ttd1 = TileTaskDescriptor(
        mode=MODE_TBFP9_GEMM, accum=1, base_a=a_base, stride_a=2, base_b=b_base, stride_b=2,
        base_c=c_base, stride_c=2, m=2, n=4, k=4, e_a=8, e_b=8, e_c=8, bias_addr=bias_base,
    )
    for i, w in enumerate(ttd1.pack_9_words()):
        m.mem.write(ttd_base + i, w, supervisor=True)
    m.assemble_and_load(f"LEA A1, {ttd_base}(A0)\nTEXEC A1, S0\nTWAIT S0, #0\nHLT", origin=200)
    m.cpu0.run()
    expected = C_init + (A @ B) + bias[None, :]
    for r in range(2):
        got = list(unpack_word_to_nonets(m.mem.read(c_base + r * 2, supervisor=True)[0]))
        assert got == list(expected[r])

    # 3. TTD Mode +2 (TBFP-3 12-triads/word GEMM) with live fault cutover!
    a3, b3, c3, ttd3_addr = 6000, 6010, 6020, 5900
    # 2x3 * 3x2 exact small integers in [-13..+13] with e=2 (scale = 3^(2-2) = 1)
    A3 = np.array([[1, 2, -1], [0, 1, 2]], dtype=np.int64)
    B3 = np.array([[2, 0], [1, -1], [0, 2]], dtype=np.int64)
    m.mem.write(a3, pack_triads_to_word(A3.ravel().tolist()), supervisor=True)
    m.mem.write(b3, pack_triads_to_word(B3.ravel().tolist()), supervisor=True)
    ttd3 = TileTaskDescriptor(
        mode=MODE_TBFP3_GEMM, abft=1, base_a=a3, base_b=b3, base_c=c3,
        m=2, n=2, k=3, e_a=2, e_b=2, e_c=2,
    )
    for i, w in enumerate(ttd3.pack_9_words()):
        m.mem.write(ttd3_addr + i, w, supervisor=True)
    m.tfabric.inject_tile_fault(logical_tile=3, reason="TBFP3_ABFT_FAULT")
    m.assemble_and_load(f"LEA A1, {ttd3_addr}(A0)\nTEXEC A1, S0\nTWAIT S0, #0\nHLT", origin=220)
    m.cpu0.run()
    assert m.cpu0.c == -1  # Recovered by cutover!
    got3 = list(unpack_word_to_triads(m.mem.read(c3, supervisor=True)[0]))[:4]
    assert got3 == (A3 @ B3).ravel().tolist()

    # 4. TTD Mode +3 (TB1 Pure-Trit 36-trits/word Associative Search)
    q1_addr, k1_addr, c1_addr, ttd_tb1 = 6100, 6110, 6120, 6090
    rng = np.random.default_rng(42)
    keys = rng.choice([-1, 0, 1], size=(6, 36))
    query = keys[4].copy()
    m.mem.write(q1_addr, pack_trits_to_word(query.tolist()), supervisor=True)
    for r in range(6):
        m.mem.write(k1_addr + r, pack_trits_to_word(keys[r].tolist()), supervisor=True)
    ttd_assoc = TileTaskDescriptor(
        mode=MODE_TB1_ASSOC, base_a=q1_addr, base_b=k1_addr, base_c=c1_addr, m=1, n=6, k=36,
    )
    for i, w in enumerate(ttd_assoc.pack_9_words()):
        m.mem.write(ttd_tb1 + i, w, supervisor=True)
    m.assemble_and_load(f"LEA A1, {ttd_tb1}(A0)\nTEXEC A1, S0\nTWAIT S0, #0\nHLT", origin=240)
    m.cpu0.run()
    scores = [m.mem.read(c1_addr + i, supervisor=True)[0] for i in range(6)]
    assert int(np.argmax(scores)) == 4
    assert scores[4] == int(np.sum(query * query))

    # 5. TTD Mode 0 (Elementwise scale + Wald dead-zone gate)
    ea, ec, ttd_el = 6200, 6210, 6190
    m.mem.write(ea, pack_nonets_to_word(90, 9, -6, -81), supervisor=True)
    ttd0 = TileTaskDescriptor(
        mode=MODE_ELEM, wgate=1, base_a=ea, base_c=ec, m=1, n=4,
        e_a=8, e_c=8, rshift=1, tau_neg=-5, tau_pos=5,  # /3 -> [30, 3, -2, -27], gate (-5,+5) -> [30, 0, 0, -27]
    )
    for i, w in enumerate(ttd0.pack_9_words()):
        m.mem.write(ttd_el + i, w, supervisor=True)
    m.assemble_and_load(f"LEA A1, {ttd_el}(A0)\nTEXEC A1, S0\nTWAIT S0, #0\nHLT", origin=260)
    m.cpu0.run()
    assert list(unpack_word_to_nonets(m.mem.read(ec, supervisor=True)[0])) == [30, 0, 0, -27]

    # 6. TTD Mode -2 (Radix-6 Good-Thomas PFA FFT via TEXEC: p=3, q=2 -> N=108, batch=2)
    f_in, f_out, ttd_f6 = 6400, 6600, 6390
    n_pts = (3**3) * (2**2)  # 108
    t = np.arange(n_pts, dtype=np.float64)
    sigs = [np.exp(2j * np.pi * 11 * t / n_pts), np.exp(2j * np.pi * 37 * t / n_pts)]
    words_per_fft = n_pts // 2  # 54
    for b_idx, s in enumerate(sigs):
        for idx in range(0, n_pts, 2):
            w = pack_nonets_to_word(
                int(round(s[idx].real * 1000)), int(round(s[idx].imag * 1000)),
                int(round(s[idx + 1].real * 1000)), int(round(s[idx + 1].imag * 1000)),
            )
            m.mem.write(f_in + b_idx * words_per_fft + idx // 2, w, supervisor=True)
    packed_pq = 3 * POW3[6] + 2  # p=3 (high 6t), q=2 (low 6t)
    ttd_fft6 = TileTaskDescriptor(
        mode=MODE_RADIX6_FFT, base_a=f_in, base_c=f_out, m=packed_pq, n=2, e_a=8,
    )
    for i, w in enumerate(ttd_fft6.pack_9_words()):
        m.mem.write(ttd_f6 + i, w, supervisor=True)
    m.assemble_and_load(f"LEA A1, {ttd_f6}(A0)\nTEXEC A1, S0\nTWAIT S0, #0\nHLT", origin=280)
    m.cpu0.run()
    for b_idx, expected_peak in enumerate((11, 37)):
        mags = []
        for idx in range(0, n_pts, 2):
            n3, n2, n1, n0 = unpack_word_to_nonets(
                m.mem.read(f_out + b_idx * words_per_fft + idx // 2, supervisor=True)[0]
            )
            mags.extend([abs(n3 + 1j * n2), abs(n1 + 1j * n0)])
        assert int(np.argmax(mags)) == expected_peak


def test_async_texec_twait_overlap_timeout_tstat_and_algd():
    """Verify genuine TEXEC/TWAIT CPU overlap, TWAIT cycle timeout, TSTAT selectors & ALGD cmds."""
    def _setup_gemm_machine() -> Tuple[UE1000Machine, int]:
        m = UE1000Machine()
        a_addr, b_addr, c_addr, ttd_addr = 10000, 11000, 12000, 9900
        # 24x24 * 24x24 on 1 tile (tile 0 only via mask=1) -> ~1200+ CPU cycles!
        A = np.eye(24, dtype=np.int64) * 2
        B = np.eye(24, dtype=np.int64) * 3
        for base, mat in ((a_addr, A), (b_addr, B)):
            flat = mat.ravel()
            for i in range(0, flat.size, 4):
                m.mem.write(base + i // 4, pack_nonets_to_word(*map(int, flat[i : i + 4])), supervisor=True)
        ttd = TileTaskDescriptor(
            mode=MODE_TBFP9_GEMM, base_a=a_addr, base_b=b_addr, base_c=c_addr,
            m=24, n=24, k=24, e_a=8, e_b=8, e_c=8,
        )
        for i, w in enumerate(ttd.pack_9_words()):
            m.mem.write(ttd_addr + i, w, supervisor=True)
        return m, ttd_addr

    # Sequential: TEXEC -> TWAIT -> 80-iteration CPU loop
    m_seq, ttd_addr = _setup_gemm_machine()
    asm_seq = f"""
        LEA   A1, {ttd_addr}(A0)
        ADD   S2, S0, #1          | NOP
        TEXEC A1, S2
        TWAIT S2, #0
        AADD  A2, A0, #80         | ADD S3, S0, #0
    LP: ADD   S3, S3, #1          | BCT A2, #LP
        HLT
    """
    m_seq.assemble_and_load(asm_seq, origin=100)
    m_seq.cpu0.run()

    # Overlapped: TEXEC -> 80-iteration CPU loop runs IN PARALLEL -> TWAIT!
    m_ovl, ttd_addr = _setup_gemm_machine()
    asm_ovl = f"""
        LEA   A1, {ttd_addr}(A0)
        ADD   S2, S0, #1          | NOP
        TEXEC A1, S2
        AADD  A2, A0, #80         | ADD S3, S0, #0
    LP: ADD   S3, S3, #1          | BCT A2, #LP
        TWAIT S2, #0
        HLT
    """
    m_ovl.assemble_and_load(asm_ovl, origin=100)
    m_ovl.cpu0.run()
    assert m_ovl.cpu0.s[3] == 80 and m_seq.cpu0.s[3] == 80
    assert m_ovl.cpu0.cycles < m_seq.cpu0.cycles - 100

    # Timeout + TSTAT busy-mask + ALGD commands!
    m_to, ttd_addr = _setup_gemm_machine()
    asm_to = f"""
        LEA   A1, {ttd_addr}(A0)
        ADD   S2, S0, #1          | AADD A3, A0, #2
        TEXEC A1, S2
        TSTAT S4, A3
        TWAIT S2, #5
        SEL   S5, S0, #8          | NOP
        TWAIT S2, #0
        SEL   S6, S0, #8          | AADD A4, A0, #3
        TSTAT S7, A4
        ALGD  S1, #3
        HLT
    """
    m_to.assemble_and_load(asm_to, origin=100)
    m_to.cpu0.run()
    assert m_to.cpu0.s[4] == 1     # Tile 0 was busy right after TEXEC!
    assert m_to.cpu0.s[5] == 0     # First TWAIT timed out (C = 0)!
    assert m_to.cpu0.s[6] == 1     # Second TWAIT completed cleanly (C = +1)!
    assert m_to.cpu0.s[7] == 3     # TSTAT #3: 3 hot spares remaining!
    assert m_to.cpu0.s[1] == 1     # ALGD #3: system health trit = +1 NOMINAL!


def test_81_bank_crossbar_stride_contention_vf18op_and_packed_db18():
    """Verify 81-bank stride conflict timing, VF18OP (54 floats/vec), and 2xDB18 DADD/DSUB."""
    def _run_vld_stride(stride: int) -> Tuple[int, int]:
        m = UE1000Machine()
        for i in range(27):
            m.mem.write(2000 + i * stride, i + 1, supervisor=True)
        asm = f"""
            VSVL  S0, #27         | AADD A1, A0, #{stride}
            VLD   V1, 2000(A0), A1
            HLT
        """
        m.assemble_and_load(asm, origin=100)
        m.cpu0.run()
        return m.cpu0.cycles, m.mem.crossbar_conflicts

    cyc_s1, conf_s1 = _run_vld_stride(1)
    cyc_s5, conf_s5 = _run_vld_stride(5)    # gcd(5, 81) = 1 -> 0 conflicts!
    cyc_s27, conf_s27 = _run_vld_stride(27) # hits only 3 banks -> bank conflicts!
    cyc_s81, conf_s81 = _run_vld_stride(81) # hits 1 bank 27 times -> max conflicts!

    assert conf_s1 == 0 and conf_s5 == 0
    assert cyc_s1 == cyc_s5
    assert conf_s27 > 0 and cyc_s27 > cyc_s1
    assert conf_s81 == 26 and cyc_s81 > cyc_s27 * 2

    # VF18OP (sop=24): 2-way packed TF18 vector add (#0), mul (#1), and fma (#2)!
    m = UE1000Machine()
    for i in range(6):
        w_a = pack_dual_tf18(float_to_tfloat(float(i + 1), TF36), float_to_tfloat(float(-(i + 1)), TF36))
        w_b = pack_dual_tf18(float_to_tfloat(2.0, TF36), float_to_tfloat(3.0, TF36))
        w_c = pack_dual_tf18(float_to_tfloat(10.0, TF36), float_to_tfloat(100.0, TF36))
        m.mem.write(3000 + i, w_a, supervisor=True)
        m.mem.write(3100 + i, w_b, supervisor=True)
        m.mem.write(3200 + i, w_c, supervisor=True)

    asm_vf18 = """
        VSVL   S0, #6             | NOP
        VLD    V1, 3000(A0)
        VLD    V2, 3100(A0)
        VLD    V3, 3200(A0)
        VF18OP V4, V1, V2, #0     | VF18OP V5, V1, V2, #1
        VF18OP V3, V1, V2, #2     | NOP
        VST    V4, 3300(A0)
        VST    V5, 3400(A0)
        VST    V3, 3500(A0)
        HLT
    """
    m.assemble_and_load(asm_vf18, origin=100)
    m.cpu0.run()
    for i in range(6):
        # Add: (i+1)+2 and -(i+1)+3
        w_add = m.mem.read(3300 + i, supervisor=True)[0]
        assert tfloat_to_float(unpack_dual_tf18(w_add, high=True), TF36) == pytest.approx((i + 1) + 2.0, rel=1e-4)
        assert tfloat_to_float(unpack_dual_tf18(w_add, high=False), TF36) == pytest.approx(-(i + 1) + 3.0, rel=1e-4)
        # FMA: 10 + (i+1)*2 and 100 + (-(i+1))*3
        w_fma = m.mem.read(3500 + i, supervisor=True)[0]
        assert tfloat_to_float(unpack_dual_tf18(w_fma, high=True), TF36) == pytest.approx(10.0 + (i + 1) * 2.0, rel=1e-4)
        assert tfloat_to_float(unpack_dual_tf18(w_fma, high=False), TF36) == pytest.approx(100.0 - (i + 1) * 3.0, rel=1e-4)

    # Packed 2xDB18 DADD / DSUB (#1) with independent 18-trit saturation per lane!
    m_db = UE1000Machine()
    m_db.mem.write(4000, WMAX_18 * POW3[18] + (-100), supervisor=True)
    m_db.mem.write(4001, 500 * POW3[18] + 250, supervisor=True)
    asm_db = """
        LD   S1, 4000(A0)
        LD   S2, 4001(A0)
        DADD S3, S1, S2, #1       | DSUB S4, S1, S2, #1
        HLT
    """
    m_db.assemble_and_load(asm_db, origin=100)
    m_db.cpu0.run()
    assert extract_field(m_db.cpu0.s[3], 18, 18, 36) == WMAX_18       # High lane saturated cleanly!
    assert extract_field(m_db.cpu0.s[3], 0, 18, 36) == 150            # Low lane: -100 + 250 = +150
    assert extract_field(m_db.cpu0.s[4], 18, 18, 36) == WMAX_18 - 500
    assert extract_field(m_db.cpu0.s[4], 0, 18, 36) == -350


def test_full_rti_restore_watchdog_timer_proactive_bank_cutover_and_mmio():
    """Verify Ferrite Vault trap vector, full RTI restore (PSW/C/VL), TRAP_TIMER, OVF, & MMIO."""
    # PSW pack/unpack roundtrip
    psw_in = PSW(mode=1, ft=1, fu=1, ov=1, ie=1, cpuid=5, ov_trap=1)
    assert PSW.unpack(psw_in.pack()) == psw_in

    m = UE1000Machine(strict_ecc=True, auto_cutover_erasure_threshold=4)

    # Install a real C-EXEC trap handler in Non-Volatile Ferrite Vault at -100, vector at -27
    # Handler counts traps in Supervisor T-DRAM [-200000] and records last trap code at [-200001], then RTI!
    handler_asm = """
        .ORG -100
        LD   S6, -200000(A0)
        ADD  S6, S6, #1           | NOP
        ST   S6, -200000(A0)
        LD   S5, -3(A0)
        ST   S5, -200001(A0)
        RTI
    """
    m.assemble_and_load(handler_asm, origin=-100)
    m.mem.write(-27, -100, supervisor=True)  # CPU 0 C-EXEC trap entry vector!

    # User program sets VL=13, C=+1, invokes SVC #42, runs loop triggering TRAP_TIMER, then halts via SVC #99
    user_asm = """
        VSVL S0, #13              | ADD S1, S0, #+7
        SVC  #42
        AADD A1, A0, #8           | ADD S2, S0, #0
    LP: ADD  S2, S2, #1           | BCT A1, #LP
        SVC  #99
    """
    m.assemble_and_load(user_asm, origin=500)
    m.cpu0.psw = PSW(mode=1, ft=1, ie=1, cpuid=0)
    m.cpu0.timer_quantum = 4  # Fire TRAP_TIMER every 4 User instruction words!
    # Intercept SVC #99 only to halt cleanly after testing SVC #42 and TRAP_TIMER through real Ferrite handler
    m.cpu0.svc_handler = lambda cpu, arg: (setattr(cpu, "halted", True) or True) if arg == 99 else False

    m.cpu0.run(max_steps=500)
    assert m.cpu0.halted
    assert m.cpu0.s[2] == 8
    # Full RTI preserved User mode, PSW.ft, C, and VL == 13!
    assert m.cpu0.psw.mode == 1 and m.cpu0.psw.ft == 1
    assert m.cpu0.vl == 13
    # At least 1 SVC (#42) + multiple TRAP_TIMERs handled cleanly in Ferrite Vault!
    trap_count = m.mem.read(-200000, supervisor=True)[0]
    assert trap_count >= 3
    assert m.mem.read(-200001, supervisor=True)[0] == TRAP_TIMER

    # Signed integer overflow sticky flag + optional TRAP_OVF
    m_ov = UE1000Machine()
    m_ov.mem.write(700, WMAX_36, supervisor=True)
    m_ov.assemble_and_load("LD S1, 700(A0)\nADD S2, S1, #1 | NOP\nHLT", origin=100)
    m_ov.cpu0.psw = PSW(mode=1, ov_trap=1, cpuid=0)
    m_ov.cpu0.run()
    assert m_ov.cpu0.psw.ov == 1
    assert m_ov.cpu0.last_trap == TRAP_OVF

    # Proactive Ashby bank scrub-cutover when bank erasures cross threshold (4) across two rows
    addr = 12_345
    addr2 = addr + 81  # Same logical bank (12345 and 12426 mod 81)!
    m.mem.write(addr, 987654321, supervisor=True)
    m.mem.write(addr2, -112233445, supervisor=True)
    lbank = m.mem.logical_bank(addr)
    assert m.mem.logical_bank(addr2) == lbank
    orig_pbank = m.mem.physical_bank(addr)
    m.mem.inject_erasure(addr, 1)
    m.mem.inject_erasure(addr, 10)
    assert m.mem.read(addr, supervisor=True)[0] == 987654321  # 2 erasures (<4)
    assert m.mem.physical_bank(addr) == orig_pbank
    m.mem.inject_erasure(addr2, 20)
    m.mem.inject_erasure(addr2, 30)
    assert m.mem.read(addr2, supervisor=True)[0] == -112233445  # 4 bank erasures -> proactive cutover!
    assert m.mem.physical_bank(addr) != orig_pbank
    assert m.mem.bank_cutovers == 1
    assert any(ev.kind == "BANK_CUTOVER" for ev in m.vcomplex.cutovers)
    # Both rows scrubbed cleanly onto new physical bank!
    assert m.mem.read(addr, supervisor=True)[0] == 987654321
    assert m.mem.read(addr2, supervisor=True)[0] == -112233445

    # Live MMIO control apertures via TRIAD-36 LD / ST!
    mmio_asm = f"""
        LDI S4, #{MMIO_SYS_ID}
        S2A A1, S4                | NOP
        LD  S1, 0(A1)
        LDI S4, #{MMIO_VCOMPLEX_SUMMARY}
        S2A A2, S4                | NOP
        LD  S2, 0(A2)
        LDI S4, #{MMIO_BANK_CUTOVER_CMD}
        S2A A3, S4                | ADD S3, S0, #{(lbank + 1) % 81}
        ST  S3, 0(A3)
        HLT
    """
    m.cpu0.psw = PSW(mode=-1, cpuid=0)
    m.assemble_and_load(mmio_asm, origin=800)
    m.cpu0.run()
    assert m.cpu0.s[1] == 100015
    assert m.cpu0.s[2] != 0
    assert m.mem.bank_cutovers == 2  # MMIO write triggered a 2nd live bank cutover!

    # Verify TRAP_TFAULT via TEXEC/TWAIT when all 3 hot spares are exhausted!
    m_tf = UE1000Machine()
    for lt in (0, 1, 2):
        m_tf.vcomplex.trigger_tile_cutover(lt, f"PRE_FAULT_{lt}")
    assert len(m_tf.vcomplex.spare_tiles) == 0
    m_tf.mem.write(2000, pack_nonets_to_word(1, 0, 0, 1), supervisor=True)
    m_tf.mem.write(2100, pack_nonets_to_word(1, 0, 0, 1), supervisor=True)
    for i, w in enumerate(
        TileTaskDescriptor(
            mode=MODE_TBFP9_GEMM, base_a=2000, base_b=2100, base_c=2200, m=2, n=2, k=2, e_a=8, e_b=8, e_c=8
        ).pack_9_words()
    ):
        m_tf.mem.write(1900 + i, w, supervisor=True)
    m_tf.tfabric.inject_tile_fault(logical_tile=4, reason="4TH_UNRECOVERABLE_FAULT")
    m_tf.assemble_and_load("LEA A1, 1900(A0)\nTEXEC A1, S0\nTWAIT S0, #0\nHLT", origin=100)
    m_tf.cpu0.psw = PSW(mode=1, cpuid=0)
    m_tf.cpu0.run()
    assert m_tf.cpu0.last_trap == TRAP_TFAULT


def test_multi_cpu_concurrent_execution_and_partitioned_tfabric():
    """Run CPUs 0..3 concurrently via run_all() with partitioned T-Fabric tiles & shared T-DRAM."""
    m = UE1000Machine()

    # Prepare GEMM TTD for CPU 0 (using tiles 0..12 only!)
    a0, b0, c0, ttd0_addr = 20000, 20100, 20200, 19900
    A = np.eye(6, dtype=np.int64) * 4
    B = np.arange(1, 37, dtype=np.int64).reshape(6, 6)
    for base, mat in ((a0, A), (b0, B)):
        flat = mat.ravel()
        for i in range(0, flat.size, 4):
            m.mem.write(base + i // 4, pack_nonets_to_word(*map(int, flat[i : i + 4])), supervisor=True)
    for i, w in enumerate(
        TileTaskDescriptor(
            mode=MODE_TBFP9_GEMM, base_a=a0, base_b=b0, base_c=c0, m=6, n=6, k=6, e_a=8, e_b=8, e_c=8
        ).pack_9_words()
    ):
        m.mem.write(ttd0_addr + i, w, supervisor=True)

    # Prepare Radix-3 FFT TTD for CPU 1 (using tiles 13..26 — disjoint partition!)
    f1_in, f1_out, ttd1_addr = 21000, 21100, 20900
    n_pts = 27
    t = np.arange(n_pts, dtype=np.float64)
    sig = np.exp(2j * np.pi * 7 * t / n_pts)
    for idx in range(0, n_pts, 2):
        r0, i0 = int(round(sig[idx].real * 1000)), int(round(sig[idx].imag * 1000))
        r1, i1 = (int(round(sig[idx + 1].real * 1000)), int(round(sig[idx + 1].imag * 1000))) if idx + 1 < n_pts else (0, 0)
        m.mem.write(f1_in + idx // 2, pack_nonets_to_word(r0, i0, r1, i1), supervisor=True)
    for i, w in enumerate(
        TileTaskDescriptor(
            mode=MODE_RADIX3_FFT, base_a=f1_in, base_c=f1_out, m=3 * POW3[6], n=1, e_a=8
        ).pack_9_words()
    ):
        m.mem.write(ttd1_addr + i, w, supervisor=True)

    # Store disjoint 27-trit tile masks in T-DRAM (0..12 for CPU 0, 13..26 for CPU 1)
    mask_cpu0 = trits_to_int([1] * 13 + [0] * 14)
    mask_cpu1 = trits_to_int([0] * 13 + [1] * 14)
    m.mem.write(19800, mask_cpu0, supervisor=True)
    m.mem.write(19801, mask_cpu1, supervisor=True)

    # CPU 0: Run GEMM on tiles 0..12, then set shared barrier flag M[25000] = +1
    m.assemble_and_load(f"""
        LEA   A1, {ttd0_addr}(A0)
        LD    S2, 19800(A0)
        TEXEC A1, S2
        TWAIT S2, #0
        ADD   S1, S0, #+1         | NOP
        ST    S1, 25000(A0)
        HLT
    """, origin=100, cpu_id=0)

    # CPU 1: Run Radix-3 FFT on tiles 13..26 simultaneously!
    m.assemble_and_load(f"""
        LEA   A1, {ttd1_addr}(A0)
        LD    S2, 19801(A0)
        TEXEC A1, S2
        TWAIT S2, #0
        ADD   S1, S0, #+1         | NOP
        ST    S1, 25001(A0)
        HLT
    """, origin=200, cpu_id=1)

    # CPU 2: Compute sum 1..20 and write result to M[25002]
    m.assemble_and_load("""
        AADD  A1, A0, #20         | ADD S1, S0, #0
    LP: A2S   S2, A1              | ADD S1, S1, S2
        BCT   A1, #LP             | NOP
        ST    S1, 25002(A0)
        HLT
    """, origin=300, cpu_id=2)

    # CPU 3: Spin-wait on CPU 0's shared T-DRAM flag M[25000] == +1, then record M[25003] = 777!
    m.assemble_and_load("""
    WT: LD    S1, 25000(A0)
        BZ    WT                  | NOP
        ADD   S3, S0, #777        | NOP
        ST    S3, 25003(A0)
        HLT
    """, origin=400, cpu_id=3)

    steps = m.run_all(cpu_ids=[0, 1, 2, 3], max_steps=10_000)
    assert steps > 0
    assert all(m.cpus[i].halted for i in range(4))
    assert m.mem.read(25000, supervisor=True)[0] == 1
    assert m.mem.read(25001, supervisor=True)[0] == 1
    assert m.mem.read(25002, supervisor=True)[0] == 210  # sum(1..20)
    assert m.mem.read(25003, supervisor=True)[0] == 777  # CPU 3 observed CPU 0's commit!

    # Verify CPU 0 GEMM output (4 * B)
    got_gemm = []
    for i in range(9):
        got_gemm.extend(unpack_word_to_nonets(m.mem.read(c0 + i, supervisor=True)[0]))
    assert np.array_equal(np.array(got_gemm).reshape(6, 6), 4 * B)

    # Verify CPU 1 FFT peak bin == 7
    mags = []
    for idx in range(0, n_pts, 2):
        n3, n2, n1, n0 = unpack_word_to_nonets(m.mem.read(f1_out + idx // 2, supervisor=True)[0])
        mags.append(abs(n3 + 1j * n2))
        if idx + 1 < n_pts:
            mags.append(abs(n1 + 1j * n0))
    assert int(np.argmax(mags)) == 7

    # Verify telemetry snapshot reports all new hardware fields cleanly
    snap = m.telemetry_snapshot()
    assert snap["c_complex"]["total_instructions"] > snap["c_complex"]["cpu0_instructions"]
    assert "io_channels" in snap
    assert "crossbar_conflicts" in snap["memory"]
