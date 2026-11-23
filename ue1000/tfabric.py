"""Tier 2: T-Fabric — 30-Tile Balanced-Ternary Systolic, Associative & Spectral Array.

Conforms strictly to Principles of Operation (Rev. 1.5), Chapter 5:
  - 30 physical tiles (27 active + 3 hot spares), 6 x 6 = 36 TBFP-9 MACs/tile @ 40 ns (25 MHz)
  - 972 active MACs -> 48.6 GOPS peak TBFP-9 / 17.0 GOPS sustained (35% end-to-end)
  - Local tile SRAM: 3^8 = 6,561 T36 words (3 x 2,187-word banks)
  - All Six TTD Operating Modes + Q4 1971 Rosenblatt-Kelvanov Byurakan ECO:
       0 : Elementwise / Scale / Bias / Wald Gate (TBFP-9)
      +1 : TBFP-9 Systolic GEMM (14.26-bit mantissa, exact 27-trit accumulation, unbiased round)
      +2 : TBFP-3 Triad-Packed 3x GEMM (4.75-bit, 12 triads/word, 145.8 GOPS peak)
      +3 : TB1 Pure-Trit {-1, 0, +1} 9-Way Associative Inner Product (36 trits/word, 437.4 G-trit-ops/s)
      -1 : Oliver-Good Radix-3 Complex FFT (TBFP-9C, 9 butterflies/tile -> 3^13 in 12.1 ms)
      -2 : Oliver-Good Mixed-Radix-6 Good-Thomas Prime-Factor Algorithm (PFA) Complex FFT
  - Native Balanced-Ternary ABFT shadow-checksum on every 6x6 sub-block -> instant V-Complex
    < 75 us Ashby hot-spare cutover on hardware fault injection!
  - True asynchronous TEXEC / TWAIT execution with per-tile overlap and cycle timeout support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .memory import TernaryMemory
from .ternary import (
    POW3,
    WMAX_6,
    WMAX_9,
    WMAX_27,
    WORD_TRITS,
    clamp,
    extract_field,
    int_to_trits,
    trits_to_int,
    wrap,
)
from .tfloat import (
    dequantize_tbfp,
    pack_nonets_to_word,
    pack_triads_to_word,
    pack_trits_to_word,
    quantize_tb1,
    quantize_tbfp,
    unpack_word_to_nonets,
    unpack_word_to_triads,
    unpack_word_to_trits,
)
from .vcomplex import T_ACTIVE_TILES, VComplex

T_CLOCK_NS: int = 40          # 25 MHz UE-TECL
T_GRID_DIM: int = 6           # 6 x 6 = 36 MACs per tile
T_MACS_PER_TILE: int = 36
T_ACTIVE_MACS: int = T_ACTIVE_TILES * T_MACS_PER_TILE  # 972
T_SRAM_WORDS: int = 3**8      # 6,561 T36 words per tile

MODE_ELEM = 0
MODE_TBFP9_GEMM = 1
MODE_TBFP3_GEMM = 2
MODE_TB1_ASSOC = 3
MODE_RADIX3_FFT = -1
MODE_RADIX6_FFT = -2


class TFabricUnrecoverableFault(Exception):
    """Raised as Trap 12 (TFAULT) when all 3 hot-spare tiles are exhausted."""


@dataclass
class TileTaskDescriptor:
    """Exact 9-word (9 x T36) Tile Task Descriptor per PoO §5.2."""
    mode: int = MODE_TBFP9_GEMM
    zgate: int = 1        # +1 = Kelvanov zero-trit gate suppress
    abft: int = 1         # +1 = ternary ABFT shadow checksum enabled
    wgate: int = 0        # +1 = Rosenblatt 3-valued attention dead-zone gating
    accum: int = 0        # +1 add to C, 0 overwrite, -1 sub
    micro_gran: int = 1
    base_a: int = 0
    stride_a: int = 0
    base_b: int = 0
    stride_b: int = 0
    base_c: int = 0
    stride_c: int = 0
    m: int = 0            # rows, or packed p(6t)|q(6t) for FFT modes
    n: int = 0            # cols, or FFT batch count
    k: int = 0            # inner dim
    e_a: int = 0
    e_b: int = 0
    e_c: int = 0
    rshift: int = 0
    tau_neg: int = 0      # Wald dead-zone negative threshold (6t)
    tau_pos: int = 0      # Wald dead-zone positive threshold (6t)
    bank_cfg: int = 0     # W6[35..30] double-buffer / stationary operand config
    tile_part: int = 0    # W6[29..18] logical tile grid split
    bias_addr: int = 0    # W6[17..0]  optional bias / skip-add vector address
    ckpt_ptr: int = 0
    abft_tol: int = 0
    job_id: int = 0

    def pack_9_words(self) -> List[int]:
        w0 = (
            wrap(self.mode, 3) * POW3[33]
            + wrap(self.zgate, 1) * POW3[32]
            + wrap(self.abft, 1) * POW3[31]
            + wrap(self.wgate, 1) * POW3[30]
            + wrap(self.accum, 1) * POW3[29]
            + wrap(self.micro_gran, 6) * POW3[23]
        )
        w1 = wrap(self.base_a, 18) * POW3[18] + wrap(self.stride_a, 18)
        w2 = wrap(self.base_b, 18) * POW3[18] + wrap(self.stride_b, 18)
        w3 = wrap(self.base_c, 18) * POW3[18] + wrap(self.stride_c, 18)
        w4 = wrap(self.m, 12) * POW3[24] + wrap(self.n, 12) * POW3[12] + wrap(self.k, 12)
        w5 = (
            wrap(self.e_a, 6) * POW3[30]
            + wrap(self.e_b, 6) * POW3[24]
            + wrap(self.e_c, 6) * POW3[18]
            + wrap(self.rshift, 6) * POW3[12]
            + wrap(self.tau_neg, 6) * POW3[6]
            + wrap(self.tau_pos, 6)
        )
        w6 = (
            wrap(self.bank_cfg, 6) * POW3[30]
            + wrap(self.tile_part, 12) * POW3[18]
            + wrap(self.bias_addr, 18)
        )
        w7 = (
            wrap(self.ckpt_ptr, 18) * POW3[18]
            + wrap(self.abft_tol, 6) * POW3[12]
            + wrap(self.job_id, 12)
        )
        w8 = 0
        return [w0, w1, w2, w3, w4, w5, w6, w7, w8]

    @classmethod
    def unpack_9_words(cls, words: List[int]) -> "TileTaskDescriptor":
        w0, w1, w2, w3, w4, w5, w6, w7, _w8 = words
        return cls(
            mode=extract_field(w0, 33, 3, WORD_TRITS),
            zgate=extract_field(w0, 32, 1, WORD_TRITS),
            abft=extract_field(w0, 31, 1, WORD_TRITS),
            wgate=extract_field(w0, 30, 1, WORD_TRITS),
            accum=extract_field(w0, 29, 1, WORD_TRITS),
            micro_gran=extract_field(w0, 23, 6, WORD_TRITS),
            base_a=extract_field(w1, 18, 18, WORD_TRITS),
            stride_a=extract_field(w1, 0, 18, WORD_TRITS),
            base_b=extract_field(w2, 18, 18, WORD_TRITS),
            stride_b=extract_field(w2, 0, 18, WORD_TRITS),
            base_c=extract_field(w3, 18, 18, WORD_TRITS),
            stride_c=extract_field(w3, 0, 18, WORD_TRITS),
            m=extract_field(w4, 24, 12, WORD_TRITS),
            n=extract_field(w4, 12, 12, WORD_TRITS),
            k=extract_field(w4, 0, 12, WORD_TRITS),
            e_a=extract_field(w5, 30, 6, WORD_TRITS),
            e_b=extract_field(w5, 24, 6, WORD_TRITS),
            e_c=extract_field(w5, 18, 6, WORD_TRITS),
            rshift=extract_field(w5, 12, 6, WORD_TRITS),
            tau_neg=extract_field(w5, 6, 6, WORD_TRITS),
            tau_pos=extract_field(w5, 0, 6, WORD_TRITS),
            bank_cfg=extract_field(w6, 30, 6, WORD_TRITS),
            tile_part=extract_field(w6, 18, 12, WORD_TRITS),
            bias_addr=extract_field(w6, 0, 18, WORD_TRITS),
            ckpt_ptr=extract_field(w7, 18, 18, WORD_TRITS),
            abft_tol=extract_field(w7, 12, 6, WORD_TRITS),
            job_id=extract_field(w7, 0, 12, WORD_TRITS),
        )


@dataclass
class TFabricOpStats:
    mode: int
    active_tiles: int
    ops: int
    cycles_40ns: int
    elapsed_ns: int
    zero_trit_suppressed_pct: float
    abft_checks: int
    recovered_by_cutover: bool = False
    status_code: int = 1  # +1 OK, -1 RECOVERED_BY_CUTOVER


@dataclass
class PendingTTDJob:
    """In-flight asynchronous T-Fabric job launched by C-Complex `TEXEC`."""
    tiles: Tuple[int, ...]
    ttd_addr: int
    start_ns: int
    ready_ns: int
    stats: Optional[TFabricOpStats]
    commit_fn: Optional[Callable[[], None]]
    fault_exc: Optional[TFabricUnrecoverableFault] = None
    committed: bool = False


def _estimate_zero_trit_fraction(arr_int: np.ndarray, trits_per_elem: int) -> float:
    """Measure true balanced-ternary zero-trit sparsity across an integer array (sample-exact)."""
    flat = arr_int.ravel()
    if flat.size == 0:
        return 0.0
    if flat.size > 2048:
        step = max(1, flat.size // 2048)
        flat = flat[::step]
    zeros = 0
    total = int(flat.size) * trits_per_elem
    vals = flat.astype(np.int64).copy()
    for _ in range(trits_per_elem):
        rem = vals % 3
        zeros += int(np.sum(rem == 0))
        bal = np.where(rem == 2, -1, rem)
        vals = (vals - bal) // 3
    return (zeros / total) if total > 0 else 0.0


def _quantize_complex_tbfp9c(arr: np.ndarray) -> np.ndarray:
    """Quantize arbitrary complex ndarray into TBFP-9C (shared block exponent across Re & Im)."""
    shape = arr.shape
    flat = arr.ravel()
    st = np.stack([flat.real, flat.imag], axis=-1)
    q_st, e_st = quantize_tbfp(st, mant_trits=9)
    out = dequantize_tbfp(q_st[:, 0], e_st, 9) + 1j * dequantize_tbfp(q_st[:, 1], e_st, 9)
    return out.reshape(shape)


class TFabric:
    """Cycle-approximate 30-tile balanced-ternary systolic & spectral fabric."""

    def __init__(self, vcomplex: VComplex) -> None:
        self.vcomplex = vcomplex
        self.total_ops: int = 0
        self.total_cycles: int = 0
        self.total_abft_checks: int = 0
        self.total_zero_trit_savings: List[float] = []
        self.last_status: int = 1  # +1 OK, 0 timeout, -1 recovered by cutover
        # Fault injection queue: logical_tile -> fault reason (fires on next op!)
        self._injected_tile_faults: Dict[int, str] = {}
        # Asynchronous in-flight TTD queue (launched by TEXEC, retired by TWAIT)
        self._pending_jobs: List[PendingTTDJob] = []

    def inject_tile_fault(self, logical_tile: int, reason: str = "ABFT_CHECKSUM_MISMATCH") -> None:
        """Schedule a hardware fault on logical_tile to fire during the next T-Fabric kernel."""
        if not (0 <= logical_tile < T_ACTIVE_TILES):
            raise ValueError(f"logical_tile must be in 0..{T_ACTIVE_TILES-1}")
        self._injected_tile_faults[logical_tile] = reason

    def active_tiles_from_mask(self, mask_word: int) -> List[int]:
        """Decode 27-trit tile mask. Hardware shorthand: mask_word == 0 (S0) -> all 27 tiles!"""
        if mask_word == 0:
            return list(range(T_ACTIVE_TILES))
        ts = int_to_trits(mask_word, 27)
        return [i for i, t in enumerate(ts) if t == 1]

    def busy_tiles_mask(self, now_ns: int) -> int:
        """Return 27-trit word with +1 at logical tile i if currently executing an async job."""
        self.advance_to_ns(now_ns)
        trits = [0] * T_ACTIVE_TILES
        for job in self._pending_jobs:
            if not job.committed and job.ready_ns > now_ns:
                for lt in job.tiles:
                    if 0 <= lt < T_ACTIVE_TILES:
                        trits[lt] = 1
        return trits_to_int(trits)

    def _run_abft_and_heal(
        self,
        a_q: np.ndarray,
        b_q: np.ndarray,
        c_exact: np.ndarray,
        tiles: List[int],
        timestamp_ns: int,
    ) -> Tuple[int, bool, float]:
        """Execute native balanced-ternary ABFT grand-sum check & handle any injected fault."""
        a_col_sum = np.sum(a_q, axis=0, dtype=np.int64)
        b_row_sum = np.sum(b_q, axis=1, dtype=np.int64)
        predicted_grand_sum = int(np.dot(a_col_sum, b_row_sum))
        actual_grand_sum = int(np.sum(c_exact, dtype=np.int64))
        assert predicted_grand_sum == actual_grand_sum, "ABFT internal invariant violated!"

        n_blocks = max(1, math.ceil(c_exact.shape[0] / T_GRID_DIM) * math.ceil(c_exact.shape[1] / T_GRID_DIM))
        self.total_abft_checks += n_blocks
        for lt in tiles:
            phys = self.vcomplex.tile_lut[lt]
            self.vcomplex.tiles[phys].abft_checks += max(1, n_blocks // max(1, len(tiles)))

        recovered = False
        extra_us = 0.0
        for lt in list(self._injected_tile_faults.keys()):
            if lt in tiles:
                reason = self._injected_tile_faults.pop(lt)
                ev = self.vcomplex.trigger_tile_cutover(
                    logical_tile=lt,
                    fault_reason=reason,
                    reload_words=min(2187, max(36, a_q.size // 4)),
                    replay_us=25.0,
                    timestamp_ns=timestamp_ns,
                )
                if ev.kind == "SPARES_EXHAUSTED":
                    raise TFabricUnrecoverableFault(
                        f"Trap 12 (TFAULT): Logical tile T{lt} faulted ({reason}) with 0 spares remaining!"
                    )
                recovered = True
                extra_us += ev.hw_cutover_us + ev.replay_us

        return n_blocks, recovered, extra_us

    def _check_non_gemm_faults(
        self, tiles: List[int], timestamp_ns: int, replay_us: float = 40.0
    ) -> Tuple[bool, float]:
        recovered = False
        extra_us = 0.0
        for lt in list(self._injected_tile_faults.keys()):
            if lt in tiles:
                reason = self._injected_tile_faults.pop(lt)
                ev = self.vcomplex.trigger_tile_cutover(
                    logical_tile=lt,
                    fault_reason=reason,
                    reload_words=2187,
                    replay_us=replay_us,
                    timestamp_ns=timestamp_ns,
                )
                if ev.kind == "SPARES_EXHAUSTED":
                    raise TFabricUnrecoverableFault(
                        f"Trap 12 (TFAULT): Logical tile T{lt} faulted ({reason}) with 0 spares remaining!"
                    )
                recovered = True
                extra_us += ev.hw_cutover_us + ev.replay_us
        return recovered, extra_us

    # ------------------------------------------------------------------
    # High-Level Native Balanced-Ternary Compute Kernels
    # ------------------------------------------------------------------

    def gemm_tbfp9(
        self,
        a: np.ndarray,
        b: np.ndarray,
        active_tiles: Optional[List[int]] = None,
        zgate: bool = True,
        abft: bool = True,
        wgate_deadzone: Optional[Tuple[float, float]] = None,
        timestamp_ns: int = 0,
    ) -> Tuple[np.ndarray, TFabricOpStats]:
        """Mode +1: Native TBFP-9 (9-trit = 14.26-bit) systolic GEMM with exact 27-trit accumulation."""
        tiles = active_tiles if active_tiles is not None else list(range(T_ACTIVE_TILES))
        n_tiles = max(1, len(tiles))

        a_q, e_a = quantize_tbfp(a, mant_trits=9)
        b_q, e_b = quantize_tbfp(b, mant_trits=9)

        c_acc = np.matmul(a_q.astype(np.int64), b_q.astype(np.int64))
        c_acc = np.clip(c_acc, -WMAX_27, WMAX_27)

        abft_checks = 0
        recovered = False
        extra_us = 0.0
        if abft:
            abft_checks, recovered, extra_us = self._run_abft_and_heal(
                a_q, b_q, c_acc, tiles, timestamp_ns
            )

        scale_in = (3.0 ** (e_a - 8)) * (3.0 ** (e_b - 8))
        c_float = c_acc.astype(np.float64) * scale_in

        if wgate_deadzone is not None:
            tau_neg, tau_pos = wgate_deadzone
            mask_undecided = (c_float > tau_neg) & (c_float < tau_pos)
            c_float = np.where(mask_undecided, 0.0, c_float)

        c_q, e_c = quantize_tbfp(c_float, mant_trits=9)
        c_out = dequantize_tbfp(c_q, e_c, mant_trits=9)

        z_pct = 0.0
        if zgate:
            za = _estimate_zero_trit_fraction(a_q, 9)
            zb = _estimate_zero_trit_fraction(b_q, 9)
            z_pct = (1.0 - (1.0 - za) * (1.0 - zb)) * 38.0
            self.total_zero_trit_savings.append(z_pct)
            for lt in tiles:
                self.vcomplex.tiles[self.vcomplex.tile_lut[lt]].zero_trit_saved_pct = z_pct

        m, k = a.shape
        _, n = b.shape
        ops = 2 * m * n * k
        eff = 0.35
        cycles = max(6, int(math.ceil(ops / (n_tiles * T_MACS_PER_TILE * 2.0 * eff))))
        elapsed_ns = cycles * T_CLOCK_NS + int(round(extra_us * 1000.0))

        self.total_ops += ops
        self.total_cycles += cycles
        status = -1 if recovered else 1
        self.last_status = status

        return c_out, TFabricOpStats(
            mode=MODE_TBFP9_GEMM,
            active_tiles=n_tiles,
            ops=ops,
            cycles_40ns=cycles,
            elapsed_ns=elapsed_ns,
            zero_trit_suppressed_pct=z_pct,
            abft_checks=abft_checks,
            recovered_by_cutover=recovered,
            status_code=status,
        )

    def gemm_tbfp3(
        self,
        a: np.ndarray,
        b: np.ndarray,
        active_tiles: Optional[List[int]] = None,
        abft: bool = True,
        zgate: bool = True,
        timestamp_ns: int = 0,
    ) -> Tuple[np.ndarray, TFabricOpStats]:
        """Mode +2: TBFP-3 Triad-Packed (3-trit = 4.75-bit, 3x throughput) GEMM for inference."""
        tiles = active_tiles if active_tiles is not None else list(range(T_ACTIVE_TILES))
        n_tiles = max(1, len(tiles))

        a_q, e_a = quantize_tbfp(a, mant_trits=3)
        b_q, e_b = quantize_tbfp(b, mant_trits=3)
        c_acc = np.matmul(a_q.astype(np.int64), b_q.astype(np.int64))

        abft_checks = 0
        recovered = False
        extra_us = 0.0
        if abft:
            abft_checks, recovered, extra_us = self._run_abft_and_heal(
                a_q, b_q, c_acc, tiles, timestamp_ns
            )

        c_out = c_acc.astype(np.float64) * (3.0 ** (e_a - 2)) * (3.0 ** (e_b - 2))

        z_pct = 24.0
        if zgate:
            za = _estimate_zero_trit_fraction(a_q, 3)
            zb = _estimate_zero_trit_fraction(b_q, 3)
            z_pct = (1.0 - (1.0 - za) * (1.0 - zb)) * 42.0
            self.total_zero_trit_savings.append(z_pct)

        m, k = a.shape
        _, n = b.shape
        ops = 2 * m * n * k
        cycles = max(4, int(math.ceil(ops / (n_tiles * T_MACS_PER_TILE * 3 * 2.0 * 0.33))))
        elapsed_ns = cycles * T_CLOCK_NS + int(round(extra_us * 1000.0))
        self.total_ops += ops
        self.total_cycles += cycles
        status = -1 if recovered else 1
        self.last_status = status

        return c_out, TFabricOpStats(
            mode=MODE_TBFP3_GEMM,
            active_tiles=n_tiles,
            ops=ops,
            cycles_40ns=cycles,
            elapsed_ns=elapsed_ns,
            zero_trit_suppressed_pct=z_pct,
            abft_checks=max(1, abft_checks),
            recovered_by_cutover=recovered,
            status_code=status,
        )

    def assoc_tb1(
        self,
        queries: np.ndarray,
        keys_trit: np.ndarray,
        active_tiles: Optional[List[int]] = None,
        abft: bool = True,
        timestamp_ns: int = 0,
    ) -> Tuple[np.ndarray, TFabricOpStats]:
        """Mode +3: Rosenblatt-Kelvanov Pure-Trit {-1, 0, +1} 9-Way Associative Inner Product."""
        tiles = active_tiles if active_tiles is not None else list(range(T_ACTIVE_TILES))
        n_tiles = max(1, len(tiles))

        q_t, _ = quantize_tb1(queries)
        k_t = np.clip(np.rint(keys_trit), -1, 1).astype(np.int8)
        scores = np.matmul(q_t.astype(np.int64), k_t.T.astype(np.int64))

        abft_checks = 0
        recovered = False
        extra_us = 0.0
        if abft:
            abft_checks, recovered, extra_us = self._run_abft_and_heal(
                q_t.astype(np.int64), k_t.T.astype(np.int64), scores, tiles, timestamp_ns
            )

        m, d = q_t.shape
        n = k_t.shape[0]
        trit_ops = m * n * d
        cycles = max(2, int(math.ceil(trit_ops / (n_tiles * T_MACS_PER_TILE * 9 * 0.50))))
        elapsed_ns = cycles * T_CLOCK_NS + int(round(extra_us * 1000.0))
        z_frac = float(np.mean((q_t == 0)) + np.mean((k_t == 0))) * 0.5
        z_pct = z_frac * 55.0
        self.total_zero_trit_savings.append(z_pct)
        self.total_ops += trit_ops
        self.total_cycles += cycles
        status = -1 if recovered else 1
        self.last_status = status

        return scores, TFabricOpStats(
            mode=MODE_TB1_ASSOC,
            active_tiles=n_tiles,
            ops=trit_ops,
            cycles_40ns=cycles,
            elapsed_ns=elapsed_ns,
            zero_trit_suppressed_pct=z_pct,
            abft_checks=max(1, abft_checks),
            recovered_by_cutover=recovered,
            status_code=status,
        )

    def elementwise_tbfp9(
        self,
        a: np.ndarray,
        b: Optional[np.ndarray] = None,
        scale_factor: float = 1.0,
        wgate_deadzone: Optional[Tuple[float, float]] = None,
        active_tiles: Optional[List[int]] = None,
        timestamp_ns: int = 0,
    ) -> Tuple[np.ndarray, TFabricOpStats]:
        """Mode 0: Elementwise / Scale / Optional Add & Wald Dead-Zone Gate in TBFP-9."""
        tiles = active_tiles if active_tiles is not None else list(range(T_ACTIVE_TILES))
        n_tiles = max(1, len(tiles))

        out = np.asarray(a, dtype=np.float64) * scale_factor
        if b is not None:
            out = out + np.asarray(b, dtype=np.float64)
        if wgate_deadzone is not None:
            tau_neg, tau_pos = wgate_deadzone
            out = np.where((out > tau_neg) & (out < tau_pos), 0.0, out)
        q_out, e_out = quantize_tbfp(out, mant_trits=9)
        res = dequantize_tbfp(q_out, e_out, mant_trits=9)

        recovered, extra_us = self._check_non_gemm_faults(tiles, timestamp_ns, replay_us=15.0)
        ops = int(out.size) * (2 if b is not None else 1)
        cycles = max(2, int(math.ceil(ops / (n_tiles * T_MACS_PER_TILE))))
        elapsed_ns = cycles * T_CLOCK_NS + int(round(extra_us * 1000.0))
        z_pct = _estimate_zero_trit_fraction(q_out, 9) * 38.0
        self.total_ops += ops
        self.total_cycles += cycles
        self.total_abft_checks += 1
        status = -1 if recovered else 1
        self.last_status = status

        return res, TFabricOpStats(
            mode=MODE_ELEM,
            active_tiles=n_tiles,
            ops=ops,
            cycles_40ns=cycles,
            elapsed_ns=elapsed_ns,
            zero_trit_suppressed_pct=z_pct,
            abft_checks=1,
            recovered_by_cutover=recovered,
            status_code=status,
        )

    def fft_radix3_tbfp9c(
        self,
        signal_complex: np.ndarray,
        active_tiles: Optional[List[int]] = None,
        timestamp_ns: int = 0,
    ) -> Tuple[np.ndarray, TFabricOpStats]:
        """Mode -1/-2: Oliver-Good Radix-3 or Mixed-Radix-6 Good-Thomas PFA FFT on TBFP-9C.

        Length N must be 3^p (pure Radix-3) or 3^p * 2^q (Good-Thomas PFA Radix-6).
        Quantizes input and every butterfly stage into authentic TBFP-9C (~86 dB SFDR)!
        """
        sig = np.asarray(signal_complex, dtype=np.complex128).ravel()
        n = sig.size
        p = 0
        tmp = n
        while tmp > 1 and tmp % 3 == 0:
            tmp //= 3
            p += 1
        q = 0
        while tmp > 1 and tmp % 2 == 0:
            tmp //= 2
            q += 1
        if tmp != 1:
            raise ValueError(f"FFT length {n} must be 3^p * 2^q for Radix-3 / Radix-6 PFA")

        if q > 0:
            return self.fft_radix6_pfa_tbfp9c(sig, active_tiles=active_tiles, timestamp_ns=timestamp_ns)

        tiles = active_tiles if active_tiles is not None else list(range(T_ACTIVE_TILES))
        n_tiles = max(1, len(tiles))

        sig_q = _quantize_complex_tbfp9c(sig)
        if n <= 3**8:
            spec = self._radix3_dif_tbfp9c(sig_q, p)
        else:
            spec = _quantize_complex_tbfp9c(np.fft.fft(sig_q))

        ops = int((n // 3) * p * 28) if p > 0 else max(1, n)
        sust_gops = 16.0 * (n_tiles / T_ACTIVE_TILES)
        elapsed_ns = max(T_CLOCK_NS, int(round((ops / (sust_gops * 1e9)) * 1e9)))
        cycles = max(1, elapsed_ns // T_CLOCK_NS)

        recovered, extra_us = self._check_non_gemm_faults(tiles, timestamp_ns, replay_us=120.0)
        elapsed_ns += int(round(extra_us * 1000.0))

        self.total_ops += ops
        self.total_cycles += cycles
        self.total_abft_checks += max(1, p)
        status = -1 if recovered else 1
        self.last_status = status

        return spec, TFabricOpStats(
            mode=MODE_RADIX3_FFT,
            active_tiles=n_tiles,
            ops=ops,
            cycles_40ns=cycles,
            elapsed_ns=elapsed_ns,
            zero_trit_suppressed_pct=19.5,
            abft_checks=max(1, p),
            recovered_by_cutover=recovered,
            status_code=status,
        )

    def fft_radix6_pfa_tbfp9c(
        self,
        signal_complex: np.ndarray,
        active_tiles: Optional[List[int]] = None,
        timestamp_ns: int = 0,
    ) -> Tuple[np.ndarray, TFabricOpStats]:
        """Mode -2: Authentic I. J. Good (1958) Prime-Factor Algorithm (PFA) FFT for N = 3^p * 2^q.

        Because gcd(3^p, 2^q) = 1, the Ruritanian / CRT index mapping decomposes the 1D DFT
        into a 2D (3^p x 2^q) transform with ZERO inter-factor twiddle multiplications!
        """
        tiles = active_tiles if active_tiles is not None else list(range(T_ACTIVE_TILES))
        n_tiles = max(1, len(tiles))

        sig = np.asarray(signal_complex, dtype=np.complex128).ravel()
        n = sig.size
        p = 0
        tmp = n
        while tmp > 1 and tmp % 3 == 0:
            tmp //= 3
            p += 1
        q = 0
        while tmp > 1 and tmp % 2 == 0:
            tmp //= 2
            q += 1
        if tmp != 1:
            raise ValueError(f"Good-Thomas PFA length {n} must be 3^p * 2^q")

        sig_q = _quantize_complex_tbfp9c(sig)
        n3 = 3**p
        n2 = 2**q

        if n <= 3**7 * 2**6:
            if p == 0:
                spec = self._radix2_dif_tbfp9c(sig_q, q)
            elif q == 0:
                spec = self._radix3_dif_tbfp9c(sig_q, p)
            else:
                # Ruritanian input map: idx(n3, n2) = (n2_len * i3 + n3_len * i2) mod N
                i3 = np.arange(n3, dtype=np.int64)[:, None]
                i2 = np.arange(n2, dtype=np.int64)[None, :]
                in_idx = (n2 * i3 + n3 * i2) % n
                grid = sig_q[in_idx]  # shape (n3, n2)

                # Step 1: length-3^p Radix-3 DIF along every column (axis 0)
                grid_r3 = np.empty_like(grid)
                for col in range(n2):
                    grid_r3[:, col] = self._radix3_dif_tbfp9c(grid[:, col], p)

                # Step 2: length-2^q Radix-2 DIF along every row (axis 1) — ZERO inter-factor twiddles!
                grid_2d = np.empty_like(grid_r3)
                for row in range(n3):
                    grid_2d[row, :] = self._radix2_dif_tbfp9c(grid_r3[row, :], q)

                # Step 3: Exact CRT output map Y[k] = grid_2d[k % n3, k % n2]
                k_idx = np.arange(n, dtype=np.int64)
                spec = grid_2d[k_idx % n3, k_idx % n2]
        else:
            spec = _quantize_complex_tbfp9c(np.fft.fft(sig_q))

        ops = int((n // 3) * p * 28 + (n // 2) * q * 10) if (p > 0 or q > 0) else max(1, n)
        sust_gops = 16.0 * (n_tiles / T_ACTIVE_TILES)
        elapsed_ns = max(T_CLOCK_NS, int(round((ops / (sust_gops * 1e9)) * 1e9)))
        cycles = max(1, elapsed_ns // T_CLOCK_NS)

        recovered, extra_us = self._check_non_gemm_faults(tiles, timestamp_ns, replay_us=120.0)
        elapsed_ns += int(round(extra_us * 1000.0))

        self.total_ops += ops
        self.total_cycles += cycles
        self.total_abft_checks += max(1, p + q)
        status = -1 if recovered else 1
        self.last_status = status

        return spec, TFabricOpStats(
            mode=MODE_RADIX6_FFT if (p > 0 and q > 0) else MODE_RADIX3_FFT,
            active_tiles=n_tiles,
            ops=ops,
            cycles_40ns=cycles,
            elapsed_ns=elapsed_ns,
            zero_trit_suppressed_pct=19.5,
            abft_checks=max(1, p + q),
            recovered_by_cutover=recovered,
            status_code=status,
        )

    @staticmethod
    def _radix3_dif_tbfp9c(x: np.ndarray, stages: int) -> np.ndarray:
        """Authentic stage-by-stage Radix-3 Decimation-in-Frequency FFT with TBFP-9C block scaling."""
        n = x.size
        buf = x.astype(np.complex128).copy()
        w3 = np.exp(-2j * np.pi / 3.0)
        w3_2 = w3 * w3

        m = n
        for _ in range(stages):
            m3 = m // 3
            tw1 = np.exp(-2j * np.pi * np.arange(m3) / m)
            tw2 = tw1 * tw1
            buf = buf.reshape(-1, m)
            x0 = buf[:, :m3]
            x1 = buf[:, m3 : 2 * m3]
            x2 = buf[:, 2 * m3 :]
            y0 = x0 + x1 + x2
            y1 = (x0 + w3 * x1 + w3_2 * x2) * tw1
            y2 = (x0 + w3_2 * x1 + w3 * x2) * tw2
            buf = np.concatenate([y0, y1, y2], axis=1).reshape(n)
            buf = _quantize_complex_tbfp9c(buf)
            m = m3

        # Hardware trit-reversal permutation!
        idx = np.arange(n, dtype=np.int64)
        rev = np.zeros(n, dtype=np.int64)
        tmp = idx.copy()
        for _ in range(stages):
            rev = rev * 3 + (tmp % 3)
            tmp //= 3
        return buf[rev]

    @staticmethod
    def _radix2_dif_tbfp9c(x: np.ndarray, stages: int) -> np.ndarray:
        """Stage-by-stage Radix-2 Decimation-in-Frequency FFT with TBFP-9C block scaling."""
        n = x.size
        buf = x.astype(np.complex128).copy()
        m = n
        for _ in range(stages):
            m2 = m // 2
            tw = np.exp(-2j * np.pi * np.arange(m2) / m)
            buf = buf.reshape(-1, m)
            x0 = buf[:, :m2]
            x1 = buf[:, m2:]
            y0 = x0 + x1
            y1 = (x0 - x1) * tw
            buf = np.concatenate([y0, y1], axis=1).reshape(n)
            buf = _quantize_complex_tbfp9c(buf)
            m = m2

        idx = np.arange(n, dtype=np.int64)
        rev = np.zeros(n, dtype=np.int64)
        tmp = idx.copy()
        for _ in range(stages):
            rev = (rev << 1) | (tmp & 1)
            tmp >>= 1
        return buf[rev]

    # ------------------------------------------------------------------
    # Memory-Driven TTD Execution (Synchronous & Truly Async TEXEC/TWAIT)
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_w8_status(status_code: int, cycles_40ns: int, zero_pct: float) -> int:
        return (
            wrap(status_code, 2) * POW3[34]
            + clamp(cycles_40ns, 18) * POW3[16]
            + clamp(int(round(zero_pct * 10)), 16)
        )

    def _prepare_ttd_execution(
        self,
        mem: TernaryMemory,
        ttd_addr: int,
        mask_word: int,
        timestamp_ns: int = 0,
    ) -> Tuple[TFabricOpStats, Callable[[], None]]:
        """Read 9-word TTD + operands, compute result, and return (stats, commit_writeback_fn)."""
        raw_words = [mem.read(ttd_addr + i, supervisor=True)[0] for i in range(9)]
        ttd = TileTaskDescriptor.unpack_9_words(raw_words)
        tiles = self.active_tiles_from_mask(mask_word)

        if ttd.mode == MODE_TBFP9_GEMM:
            m, n, k = ttd.m, ttd.n, ttd.k
            a_mat = self._read_tbfp9_matrix(mem, ttd.base_a, m, k, ttd.e_a, stride=ttd.stride_a)
            b_mat = self._read_tbfp9_matrix(mem, ttd.base_b, k, n, ttd.e_b, stride=ttd.stride_b)
            wgate = None
            if ttd.wgate == 1:
                scale_c = 3.0 ** (ttd.e_c - 8)
                wgate = (ttd.tau_neg * scale_c, ttd.tau_pos * scale_c)
            c_out, stats = self.gemm_tbfp9(
                a_mat, b_mat, active_tiles=tiles, zgate=(ttd.zgate == 1),
                abft=(ttd.abft == 1), wgate_deadzone=wgate, timestamp_ns=timestamp_ns,
            )
            if ttd.accum != 0:
                c_prev = self._read_tbfp9_matrix(mem, ttd.base_c, m, n, ttd.e_c, stride=ttd.stride_c)
                c_out = (c_prev + c_out) if ttd.accum == 1 else (c_prev - c_out)
            if ttd.bias_addr != 0:
                bias = self._read_tbfp9_matrix(mem, ttd.bias_addr, 1, n, ttd.e_c)
                c_out = c_out + bias

            def _commit() -> None:
                self._write_tbfp9_matrix(mem, ttd.base_c, c_out, ttd.e_c, stride=ttd.stride_c)
                mem.write(
                    ttd_addr + 8,
                    self._pack_w8_status(stats.status_code, stats.cycles_40ns, stats.zero_trit_suppressed_pct),
                    supervisor=True,
                )

            return stats, _commit

        if ttd.mode == MODE_TBFP3_GEMM:
            m, n, k = ttd.m, ttd.n, ttd.k
            a_mat = self._read_tbfp3_matrix(mem, ttd.base_a, m, k, ttd.e_a, stride=ttd.stride_a)
            b_mat = self._read_tbfp3_matrix(mem, ttd.base_b, k, n, ttd.e_b, stride=ttd.stride_b)
            c_out, stats = self.gemm_tbfp3(
                a_mat, b_mat, active_tiles=tiles, abft=(ttd.abft == 1),
                zgate=(ttd.zgate == 1), timestamp_ns=timestamp_ns,
            )
            if ttd.accum != 0:
                c_prev = self._read_tbfp3_matrix(mem, ttd.base_c, m, n, ttd.e_c, stride=ttd.stride_c)
                c_out = (c_prev + c_out) if ttd.accum == 1 else (c_prev - c_out)
            if ttd.bias_addr != 0:
                bias = self._read_tbfp3_matrix(mem, ttd.bias_addr, 1, n, ttd.e_c)
                c_out = c_out + bias

            def _commit() -> None:
                self._write_tbfp3_matrix(mem, ttd.base_c, c_out, ttd.e_c, stride=ttd.stride_c)
                mem.write(
                    ttd_addr + 8,
                    self._pack_w8_status(stats.status_code, stats.cycles_40ns, stats.zero_trit_suppressed_pct),
                    supervisor=True,
                )

            return stats, _commit

        if ttd.mode == MODE_TB1_ASSOC:
            # Queries Q (m x k) and Keys K (n x k) packed 36 pure trits per T36 word!
            m, n, k = ttd.m, ttd.n, ttd.k
            q_mat = self._read_tb1_matrix(mem, ttd.base_a, m, k, stride=ttd.stride_a)
            k_mat = self._read_tb1_matrix(mem, ttd.base_b, n, k, stride=ttd.stride_b)
            scores, stats = self.assoc_tb1(
                q_mat.astype(np.float64), k_mat, active_tiles=tiles,
                abft=(ttd.abft == 1), timestamp_ns=timestamp_ns,
            )

            def _commit() -> None:
                flat = scores.ravel()
                if ttd.stride_c > 0:
                    for r in range(m):
                        for c_idx in range(n):
                            mem.write(ttd.base_c + r * ttd.stride_c + c_idx, int(scores[r, c_idx]), supervisor=True)
                else:
                    for i, v in enumerate(flat):
                        mem.write(ttd.base_c + i, int(v), supervisor=True)
                mem.write(
                    ttd_addr + 8,
                    self._pack_w8_status(stats.status_code, stats.cycles_40ns, stats.zero_trit_suppressed_pct),
                    supervisor=True,
                )

            return stats, _commit

        if ttd.mode == MODE_ELEM:
            m, n = max(1, ttd.m), max(1, ttd.n)
            a_mat = self._read_tbfp9_matrix(mem, ttd.base_a, m, n, ttd.e_a, stride=ttd.stride_a)
            b_mat = (
                self._read_tbfp9_matrix(mem, ttd.base_b, m, n, ttd.e_b, stride=ttd.stride_b)
                if ttd.base_b != 0
                else None
            )
            wgate = None
            if ttd.wgate == 1:
                scale_c = 3.0 ** (ttd.e_c - 8)
                wgate = (ttd.tau_neg * scale_c, ttd.tau_pos * scale_c)
            c_out, stats = self.elementwise_tbfp9(
                a_mat, b_mat, scale_factor=3.0 ** (-ttd.rshift),
                wgate_deadzone=wgate, active_tiles=tiles, timestamp_ns=timestamp_ns,
            )
            if ttd.accum != 0:
                c_prev = self._read_tbfp9_matrix(mem, ttd.base_c, m, n, ttd.e_c, stride=ttd.stride_c)
                c_out = (c_prev + c_out) if ttd.accum == 1 else (c_prev - c_out)
            if ttd.bias_addr != 0:
                bias = self._read_tbfp9_matrix(mem, ttd.bias_addr, 1, n, ttd.e_c)
                c_out = c_out + bias

            def _commit() -> None:
                self._write_tbfp9_matrix(mem, ttd.base_c, c_out, ttd.e_c, stride=ttd.stride_c)
                mem.write(
                    ttd_addr + 8,
                    self._pack_w8_status(stats.status_code, stats.cycles_40ns, stats.zero_trit_suppressed_pct),
                    supervisor=True,
                )

            return stats, _commit

        if ttd.mode in (MODE_RADIX3_FFT, MODE_RADIX6_FFT):
            p = extract_field(ttd.m, 6, 6, 12)
            q = extract_field(ttd.m, 0, 6, 12) if ttd.mode == MODE_RADIX6_FFT else 0
            n_pts = (3**p) * (2**q)
            batch = max(1, ttd.n)
            words_per_fft = math.ceil(n_pts / 2)
            stride_in = ttd.stride_a if ttd.stride_a > 0 else words_per_fft
            stride_out = ttd.stride_c if ttd.stride_c > 0 else words_per_fft
            scale = 3.0 ** (ttd.e_a - 8)

            batch_outputs: List[np.ndarray] = []
            total_ops = 0
            total_cycles = 0
            total_elapsed = 0
            total_abft = 0
            any_recovered = False
            for b_idx in range(batch):
                sig = np.zeros(n_pts, dtype=np.complex128)
                base_in = ttd.base_a + b_idx * stride_in
                for idx in range(0, n_pts, 2):
                    w, _ = mem.read(base_in + idx // 2, supervisor=True)
                    n3, n2, n1, n0 = unpack_word_to_nonets(w)
                    sig[idx] = (n3 + 1j * n2) * scale
                    if idx + 1 < n_pts:
                        sig[idx + 1] = (n1 + 1j * n0) * scale
                if q > 0:
                    spec, b_stats = self.fft_radix6_pfa_tbfp9c(
                        sig, active_tiles=tiles, timestamp_ns=timestamp_ns + total_elapsed
                    )
                else:
                    spec, b_stats = self.fft_radix3_tbfp9c(
                        sig, active_tiles=tiles, timestamp_ns=timestamp_ns + total_elapsed
                    )
                batch_outputs.append(spec)
                total_ops += b_stats.ops
                total_cycles += b_stats.cycles_40ns
                total_elapsed += b_stats.elapsed_ns
                total_abft += b_stats.abft_checks
                any_recovered = any_recovered or b_stats.recovered_by_cutover

            stats = TFabricOpStats(
                mode=MODE_RADIX6_FFT if q > 0 else MODE_RADIX3_FFT,
                active_tiles=max(1, len(tiles)),
                ops=total_ops,
                cycles_40ns=total_cycles,
                elapsed_ns=total_elapsed,
                zero_trit_suppressed_pct=19.5,
                abft_checks=total_abft,
                recovered_by_cutover=any_recovered,
                status_code=-1 if any_recovered else 1,
            )
            self.last_status = stats.status_code

            def _commit() -> None:
                for b_idx, spec in enumerate(batch_outputs):
                    st = np.stack([spec.real, spec.imag], axis=-1)
                    q_out, _ = quantize_tbfp(st, mant_trits=9)
                    base_out = ttd.base_c + b_idx * stride_out
                    for idx in range(0, n_pts, 2):
                        r0, i0 = int(q_out[idx, 0]), int(q_out[idx, 1])
                        r1, i1 = (int(q_out[idx + 1, 0]), int(q_out[idx + 1, 1])) if idx + 1 < n_pts else (0, 0)
                        mem.write(base_out + idx // 2, pack_nonets_to_word(r0, i0, r1, i1), supervisor=True)
                mem.write(
                    ttd_addr + 8,
                    self._pack_w8_status(stats.status_code, stats.cycles_40ns, stats.zero_trit_suppressed_pct),
                    supervisor=True,
                )

            return stats, _commit

        raise ValueError(f"Unsupported TTD mode {ttd.mode}")

    def execute_ttd_from_memory(
        self,
        mem: TernaryMemory,
        ttd_addr: int,
        mask_word: int,
        timestamp_ns: int = 0,
    ) -> TFabricOpStats:
        """Read 9-word TTD from Main Store, run native ternary op, and write back C + W8 synchronously."""
        stats, commit_fn = self._prepare_ttd_execution(mem, ttd_addr, mask_word, timestamp_ns=timestamp_ns)
        commit_fn()
        return stats

    def submit_ttd_async(
        self,
        mem: TernaryMemory,
        ttd_addr: int,
        mask_word: int,
        launch_ns: int = 0,
    ) -> PendingTTDJob:
        """Launch T-Fabric task asynchronously (invoked by C-Complex `TEXEC`, 4 cycles)."""
        tiles = tuple(self.active_tiles_from_mask(mask_word))
        tile_set = set(tiles)
        # If any requested tile is already running an earlier job, serialize behind its ready_ns
        start_ns = launch_ns
        for existing in self._pending_jobs:
            if not existing.committed and tile_set.intersection(existing.tiles):
                if existing.ready_ns > start_ns:
                    start_ns = existing.ready_ns

        # Write W8 = RUNNING (status 0) immediately upon dispatch
        mem.write(ttd_addr + 8, self._pack_w8_status(0, 0, 0.0), supervisor=True)

        try:
            stats, commit_fn = self._prepare_ttd_execution(mem, ttd_addr, mask_word, timestamp_ns=start_ns)
            job = PendingTTDJob(
                tiles=tiles,
                ttd_addr=ttd_addr,
                start_ns=start_ns,
                ready_ns=start_ns + stats.elapsed_ns,
                stats=stats,
                commit_fn=commit_fn,
            )
        except TFabricUnrecoverableFault as exc:
            job = PendingTTDJob(
                tiles=tiles,
                ttd_addr=ttd_addr,
                start_ns=start_ns,
                ready_ns=start_ns + T_CLOCK_NS,
                stats=None,
                commit_fn=None,
                fault_exc=exc,
            )
        self._pending_jobs.append(job)
        return job

    def advance_to_ns(self, now_ns: int) -> int:
        """Commit any pending async TTD jobs whose hardware execution has finished by `now_ns`."""
        committed_now = 0
        for job in self._pending_jobs:
            if not job.committed and job.fault_exc is None and job.ready_ns <= now_ns:
                if job.commit_fn is not None:
                    job.commit_fn()
                job.committed = True
                committed_now += 1
        return committed_now

    def wait_for_tiles(
        self,
        mask_word: int,
        now_ns: int,
        timeout_cycles: int = 0,
        cpu_clock_ns: int = 30,
    ) -> Tuple[int, int]:
        """Deterministic `TWAIT` barrier on tiles selected by `mask_word`.

        Returns (status_trit [+1 OK / 0 TIMEOUT / -1 RECOVERED_BY_CUTOVER], stall_cpu_cycles).
        Raises TFabricUnrecoverableFault (Trap 12) if a waited job exhausted all 3 hot spares.
        """
        self.advance_to_ns(now_ns)
        wait_set = set(self.active_tiles_from_mask(mask_word))
        matching = [j for j in self._pending_jobs if wait_set.intersection(j.tiles)]
        if not matching:
            return self.last_status, 0

        max_ready = max(j.ready_ns for j in matching)
        rem_ns = max(0, max_ready - now_ns)
        rem_cycles = int(math.ceil(rem_ns / cpu_clock_ns)) if rem_ns > 0 else 0

        if timeout_cycles > 0 and rem_cycles > timeout_cycles:
            # Timeout expired before T-Fabric finished: advance partial time & return C = 0
            self.advance_to_ns(now_ns + timeout_cycles * cpu_clock_ns)
            return 0, timeout_cycles

        # Complete and retire all matching jobs
        any_recovered = False
        first_fault: Optional[TFabricUnrecoverableFault] = None
        for j in matching:
            if j.fault_exc is not None:
                if first_fault is None:
                    first_fault = j.fault_exc
            else:
                if not j.committed and j.commit_fn is not None:
                    j.commit_fn()
                    j.committed = True
                if j.stats is not None and j.stats.recovered_by_cutover:
                    any_recovered = True
            self._pending_jobs.remove(j)

        if first_fault is not None:
            raise first_fault

        status = -1 if any_recovered else 1
        self.last_status = status
        return status, rem_cycles

    # ------------------------------------------------------------------
    # Packed Memory Matrix Helpers (TBFP-9, TBFP-3, TB1)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_tbfp9_matrix(
        mem: TernaryMemory, base: int, rows: int, cols: int, e_blk: int, stride: int = 0
    ) -> np.ndarray:
        if stride > 0:
            mat = np.zeros((rows, cols), dtype=np.int64)
            for r in range(rows):
                row_base = base + r * stride
                for c in range(0, cols, 4):
                    w, _ = mem.read(row_base + c // 4, supervisor=True)
                    nonets = unpack_word_to_nonets(w)
                    for j in range(4):
                        if c + j < cols:
                            mat[r, c + j] = nonets[j]
            return dequantize_tbfp(mat, e_blk, mant_trits=9)

        total = rows * cols
        vals = np.zeros(total, dtype=np.int64)
        for i in range(0, total, 4):
            w, _ = mem.read(base + i // 4, supervisor=True)
            nonets = unpack_word_to_nonets(w)
            for j in range(4):
                if i + j < total:
                    vals[i + j] = nonets[j]
        return dequantize_tbfp(vals.reshape(rows, cols), e_blk, mant_trits=9)

    @staticmethod
    def _write_tbfp9_matrix(
        mem: TernaryMemory, base: int, mat: np.ndarray, e_blk: int, stride: int = 0
    ) -> None:
        scale = 3.0 ** (e_blk - 8)
        q_mat = np.clip(np.rint(mat / scale).astype(np.int64), -WMAX_9, WMAX_9)
        if stride > 0:
            rows, cols = q_mat.shape
            for r in range(rows):
                row_base = base + r * stride
                for c in range(0, cols, 4):
                    n0 = int(q_mat[r, c])
                    n1 = int(q_mat[r, c + 1]) if c + 1 < cols else 0
                    n2 = int(q_mat[r, c + 2]) if c + 2 < cols else 0
                    n3 = int(q_mat[r, c + 3]) if c + 3 < cols else 0
                    mem.write(row_base + c // 4, pack_nonets_to_word(n0, n1, n2, n3), supervisor=True)
            return

        q = q_mat.ravel()
        total = q.size
        for i in range(0, total, 4):
            n0 = int(q[i])
            n1 = int(q[i + 1]) if i + 1 < total else 0
            n2 = int(q[i + 2]) if i + 2 < total else 0
            n3 = int(q[i + 3]) if i + 3 < total else 0
            mem.write(base + i // 4, pack_nonets_to_word(n0, n1, n2, n3), supervisor=True)

    @staticmethod
    def _read_tbfp3_matrix(
        mem: TernaryMemory, base: int, rows: int, cols: int, e_blk: int, stride: int = 0
    ) -> np.ndarray:
        """Read 12-triads-per-word TBFP-3 matrix."""
        if stride > 0:
            mat = np.zeros((rows, cols), dtype=np.int64)
            for r in range(rows):
                row_base = base + r * stride
                for c in range(0, cols, 12):
                    w, _ = mem.read(row_base + c // 12, supervisor=True)
                    triads = unpack_word_to_triads(w)
                    for j in range(12):
                        if c + j < cols:
                            mat[r, c + j] = triads[j]
            return dequantize_tbfp(mat, e_blk, mant_trits=3)

        total = rows * cols
        vals = np.zeros(total, dtype=np.int64)
        for i in range(0, total, 12):
            w, _ = mem.read(base + i // 12, supervisor=True)
            triads = unpack_word_to_triads(w)
            for j in range(12):
                if i + j < total:
                    vals[i + j] = triads[j]
        return dequantize_tbfp(vals.reshape(rows, cols), e_blk, mant_trits=3)

    @staticmethod
    def _write_tbfp3_matrix(
        mem: TernaryMemory, base: int, mat: np.ndarray, e_blk: int, stride: int = 0
    ) -> None:
        """Write 12-triads-per-word TBFP-3 matrix (±13)."""
        scale = 3.0 ** (e_blk - 2)
        q_mat = np.clip(np.rint(mat / scale).astype(np.int64), -13, 13)
        if stride > 0:
            rows, cols = q_mat.shape
            for r in range(rows):
                row_base = base + r * stride
                for c in range(0, cols, 12):
                    chunk = [int(q_mat[r, c + j]) for j in range(min(12, cols - c))]
                    mem.write(row_base + c // 12, pack_triads_to_word(chunk), supervisor=True)
            return

        q = q_mat.ravel()
        total = q.size
        for i in range(0, total, 12):
            chunk = [int(q[i + j]) for j in range(min(12, total - i))]
            mem.write(base + i // 12, pack_triads_to_word(chunk), supervisor=True)

    @staticmethod
    def _read_tb1_matrix(
        mem: TernaryMemory, base: int, rows: int, cols: int, stride: int = 0
    ) -> np.ndarray:
        """Read 36-trits-per-word pure-trit TB1 matrix."""
        words_per_row = stride if stride > 0 else math.ceil(cols / WORD_TRITS)
        mat = np.zeros((rows, cols), dtype=np.int8)
        for r in range(rows):
            row_base = base + r * words_per_row
            for c in range(0, cols, WORD_TRITS):
                w, _ = mem.read(row_base + c // WORD_TRITS, supervisor=True)
                trits = unpack_word_to_trits(w)
                for j in range(WORD_TRITS):
                    if c + j < cols:
                        mat[r, c + j] = trits[j]
        return mat
