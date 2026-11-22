"""Tier 2: T-Fabric — 30-Tile Balanced-Ternary Systolic, Associative & Spectral Array.

Conforms strictly to Principles of Operation (Rev. 1.5), Chapter 5:
  - 30 physical tiles (27 active + 3 hot spares), 6 x 6 = 36 TBFP-9 MACs/tile @ 40 ns (25 MHz)
  - 972 active MACs -> 48.6 GOPS peak TBFP-9 / 17.0 GOPS sustained (35% end-to-end)
  - Local tile SRAM: 3^8 = 6,561 T36 words (3 x 2,187-word banks)
  - Four Operating Modes + Q4 1971 Rosenblatt-Kelvanov Byurakan ECO:
      +1 : TBFP-9 Systolic GEMM (14.26-bit mantissa, exact 27-trit accumulation, unbiased round)
      +2 : TBFP-3 Triad-Packed 3x GEMM (4.75-bit, 145.8 GOPS peak)
      +3 : TB1 Pure-Trit {-1, 0, +1} 9-Way Associative Inner Product (437.4 G-trit-ops/s peak)
      -1 : Oliver-Good Radix-3 Complex FFT (TBFP-9C, 9 butterflies/tile -> 3^13 in 12.1 ms)
      -2 : Oliver-Good Mixed-Radix-6 Good-Thomas PFA Complex FFT
  - Native Balanced-Ternary ABFT shadow-checksum on every 6x6 sub-block -> instant V-Complex
    < 75 us Ashby hot-spare cutover on hardware fault injection!
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    wrap,
)
from .tfloat import (
    dequantize_tbfp,
    pack_nonets_to_word,
    quantize_tb1,
    quantize_tbfp,
    unpack_word_to_nonets,
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
        w6 = 0
        w7 = (
            wrap(self.ckpt_ptr, 18) * POW3[18]
            + wrap(self.abft_tol, 6) * POW3[12]
            + wrap(self.job_id, 12)
        )
        w8 = 0
        return [w0, w1, w2, w3, w4, w5, w6, w7, w8]

    @classmethod
    def unpack_9_words(cls, words: List[int]) -> "TileTaskDescriptor":
        w0, w1, w2, w3, w4, w5, _w6, w7, _w8 = words
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


def _estimate_zero_trit_fraction(arr_int: np.ndarray, trits_per_elem: int) -> float:
    """Measure true balanced-ternary zero-trit sparsity across an integer array (sample-exact)."""
    flat = arr_int.ravel()
    if flat.size == 0:
        return 0.0
    # Sample up to 2,048 elements for speed if huge
    if flat.size > 2048:
        step = max(1, flat.size // 2048)
        flat = flat[::step]
    zeros = 0
    total = int(flat.size) * trits_per_elem
    vals = flat.astype(np.int64).copy()
    for _ in range(trits_per_elem):
        rem = vals % 3
        zeros += int(np.sum(rem == 0))
        # Balanced digit: 2 -> -1
        bal = np.where(rem == 2, -1, rem)
        vals = (vals - bal) // 3
    return (zeros / total) if total > 0 else 0.0


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

    def _run_abft_and_heal(
        self,
        a_q: np.ndarray,
        b_q: np.ndarray,
        c_exact: np.ndarray,
        tiles: List[int],
        timestamp_ns: int,
    ) -> Tuple[int, bool, float]:
        """Execute native balanced-ternary ABFT grand-sum check & handle any injected fault."""
        # Shadow checksum: (sum_rows A) . (sum_cols B) == sum(C) in exact T27 arithmetic!
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
        # Check if any participating logical tile has an injected physical fault!
        for lt in list(self._injected_tile_faults.keys()):
            if lt in tiles:
                reason = self._injected_tile_faults.pop(lt)
                # ABFT detects corrupted block sum BEFORE Main Store writeback!
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

        # Exact 27-trit balanced accumulation (no overflow for K <= 10^4)
        c_acc = np.matmul(a_q.astype(np.int64), b_q.astype(np.int64))
        c_acc = np.clip(c_acc, -WMAX_27, WMAX_27)

        # ABFT shadow-checksum verification + Ashby cutover on hardware fault
        abft_checks = 0
        recovered = False
        extra_us = 0.0
        if abft:
            abft_checks, recovered, extra_us = self._run_abft_and_heal(
                a_q, b_q, c_acc, tiles, timestamp_ns
            )

        # Exact unbiased re-quantization from 27-trit accumulator back to TBFP-9!
        scale_in = (3.0 ** (e_a - 8)) * (3.0 ** (e_b - 8))
        c_float = c_acc.astype(np.float64) * scale_in

        # Optional Rosenblatt-Kelvanov Three-Valued Attention Dead-Zone Gating (ECO-71-09)
        if wgate_deadzone is not None:
            tau_neg, tau_pos = wgate_deadzone
            mask_undecided = (c_float > tau_neg) & (c_float < tau_pos)
            c_float = np.where(mask_undecided, 0.0, c_float)

        c_q, e_c = quantize_tbfp(c_float, mant_trits=9)
        c_out = dequantize_tbfp(c_q, e_c, mant_trits=9)

        # Measure true Kelvanov zero-trit power gating savings
        z_pct = 0.0
        if zgate:
            za = _estimate_zero_trit_fraction(a_q, 9)
            zb = _estimate_zero_trit_fraction(b_q, 9)
            # Either operand trit == 0 suppresses current steering: P(zero) = 1 - (1-za)(1-zb)
            # Effective dynamic tile power reduction scales ~0.38x of suppressed trit-products
            z_pct = (1.0 - (1.0 - za) * (1.0 - zb)) * 38.0
            self.total_zero_trit_savings.append(z_pct)
            for lt in tiles:
                self.vcomplex.tiles[self.vcomplex.tile_lut[lt]].zero_trit_saved_pct = z_pct

        m, k = a.shape
        _, n = b.shape
        ops = 2 * m * n * k
        # Realistic sustained efficiency ~35% on full GEMM, tiled across n_tiles * 36 MACs @ 40 ns
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
        timestamp_ns: int = 0,
    ) -> Tuple[np.ndarray, TFabricOpStats]:
        """Mode +2: TBFP-3 Triad-Packed (3-trit = 4.75-bit, 3x throughput) GEMM for inference."""
        tiles = active_tiles if active_tiles is not None else list(range(T_ACTIVE_TILES))
        n_tiles = max(1, len(tiles))

        a_q, e_a = quantize_tbfp(a, mant_trits=3)
        b_q, e_b = quantize_tbfp(b, mant_trits=3)
        c_acc = np.matmul(a_q.astype(np.int64), b_q.astype(np.int64))
        c_out = c_acc.astype(np.float64) * (3.0 ** (e_a - 2)) * (3.0 ** (e_b - 2))

        m, k = a.shape
        _, n = b.shape
        ops = 2 * m * n * k
        # 3 triads per 9-trit MAC lane -> 3x peak (145.8 GOPS), ~33% sustained
        cycles = max(4, int(math.ceil(ops / (n_tiles * T_MACS_PER_TILE * 3 * 2.0 * 0.33))))
        self.total_ops += ops
        self.total_cycles += cycles
        self.last_status = 1

        return c_out, TFabricOpStats(
            mode=MODE_TBFP3_GEMM,
            active_tiles=n_tiles,
            ops=ops,
            cycles_40ns=cycles,
            elapsed_ns=cycles * T_CLOCK_NS,
            zero_trit_suppressed_pct=24.0,
            abft_checks=1,
            status_code=1,
        )

    def assoc_tb1(
        self,
        queries: np.ndarray,
        keys_trit: np.ndarray,
        active_tiles: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, TFabricOpStats]:
        """Mode +3: Rosenblatt-Kelvanov Pure-Trit {-1, 0, +1} 9-Way Associative Inner Product."""
        tiles = active_tiles if active_tiles is not None else list(range(T_ACTIVE_TILES))
        n_tiles = max(1, len(tiles))

        q_t, _ = quantize_tb1(queries)
        k_t = np.clip(np.rint(keys_trit), -1, 1).astype(np.int8)
        scores = np.matmul(q_t.astype(np.int64), k_t.T.astype(np.int64))

        m, d = q_t.shape
        n = k_t.shape[0]
        trit_ops = m * n * d
        # 9 trit-products per PE per cycle (437.4 G-trit-ops/s peak across 27 tiles)
        cycles = max(2, int(math.ceil(trit_ops / (n_tiles * T_MACS_PER_TILE * 9 * 0.50))))
        z_frac = float(np.mean((q_t == 0)) + np.mean((k_t == 0))) * 0.5
        self.total_ops += trit_ops
        self.total_cycles += cycles
        self.last_status = 1

        return scores, TFabricOpStats(
            mode=MODE_TB1_ASSOC,
            active_tiles=n_tiles,
            ops=trit_ops,
            cycles_40ns=cycles,
            elapsed_ns=cycles * T_CLOCK_NS,
            zero_trit_suppressed_pct=z_frac * 55.0,
            abft_checks=1,
            status_code=1,
        )

    def fft_radix3_tbfp9c(
        self,
        signal_complex: np.ndarray,
        active_tiles: Optional[List[int]] = None,
        timestamp_ns: int = 0,
    ) -> Tuple[np.ndarray, TFabricOpStats]:
        """Mode -1: Oliver-Good Radix-3 Complex FFT on TBFP-9C (9t Re + 9t Im, ~86 dB SFDR).

        Length N must be a power of 3 (3^p) for pure Radix-3, or 3^p * 2^q for Radix-6 PFA.
        Quantizes input and each radix-3 stage into authentic TBFP-9C precision!
        """
        tiles = active_tiles if active_tiles is not None else list(range(T_ACTIVE_TILES))
        n_tiles = max(1, len(tiles))

        sig = np.asarray(signal_complex, dtype=np.complex128)
        n = sig.size
        # Check radix-3 power
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

        # Quantize input complex signal to TBFP-9C (shared block exponent across Re & Im)
        stacked = np.stack([sig.real, sig.imag], axis=-1)
        q_in, e_in = quantize_tbfp(stacked, mant_trits=9)
        sig_q = dequantize_tbfp(q_in[:, 0], e_in, 9) + 1j * dequantize_tbfp(q_in[:, 1], e_in, 9)

        # Execute exact Cooley-Tukey / Good-Thomas DFT and re-quantize output to TBFP-9C
        # (For moderate N <= 3^9, we run stage-quantized radix-3 butterflies directly!)
        if q == 0 and n <= 3**8:
            spec = self._radix3_dif_tbfp9c(sig_q, p)
        else:
            spec_raw = np.fft.fft(sig_q)
            st_out = np.stack([spec_raw.real, spec_raw.imag], axis=-1)
            q_out, e_out = quantize_tbfp(st_out, mant_trits=9)
            spec = dequantize_tbfp(q_out[:, 0], e_out, 9) + 1j * dequantize_tbfp(q_out[:, 1], e_out, 9)

        # Exact radix-3 operation count per PoO §5.3: 28 real ops per radix-3 butterfly
        ops = int((n // 3) * p * 28 + (n // 2) * q * 10) if p > 0 else int(5 * n * q)
        # Sustained 16.0 GOPS across 27 tiles (~33% of 48.6 GOPS peak)
        sust_gops = 16.0 * (n_tiles / T_ACTIVE_TILES)
        elapsed_ns = max(T_CLOCK_NS, int(round((ops / (sust_gops * 1e9)) * 1e9)))
        cycles = max(1, elapsed_ns // T_CLOCK_NS)

        # Check injected tile faults
        recovered = False
        for lt in list(self._injected_tile_faults.keys()):
            if lt in tiles:
                reason = self._injected_tile_faults.pop(lt)
                ev = self.vcomplex.trigger_tile_cutover(lt, reason, reload_words=2187, replay_us=120.0, timestamp_ns=timestamp_ns)
                if ev.kind == "SPARES_EXHAUSTED":
                    raise TFabricUnrecoverableFault(f"Trap 12 (TFAULT) during FFT on tile T{lt}")
                recovered = True
                elapsed_ns += int(round((ev.hw_cutover_us + ev.replay_us) * 1000.0))

        self.total_ops += ops
        self.total_cycles += cycles
        self.total_abft_checks += max(1, p + q)
        status = -1 if recovered else 1
        self.last_status = status

        return spec, TFabricOpStats(
            mode=MODE_RADIX3_FFT if q == 0 else MODE_RADIX6_FFT,
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
            # Re-quantize stage output to TBFP-9C (unbiased balanced-ternary rounding!)
            st = np.stack([buf.real, buf.imag], axis=-1)
            q_st, e_st = quantize_tbfp(st, mant_trits=9)
            buf = dequantize_tbfp(q_st[:, 0], e_st, 9) + 1j * dequantize_tbfp(q_st[:, 1], e_st, 9)
            m = m3

        # Hardware trit-reversal permutation!
        idx = np.arange(n, dtype=np.int64)
        rev = np.zeros(n, dtype=np.int64)
        tmp = idx.copy()
        for _ in range(stages):
            rev = rev * 3 + (tmp % 3)
            tmp //= 3
        return buf[rev]

    # ------------------------------------------------------------------
    # Memory-Driven TTD Execution (Invoked by C-Complex TEXEC!)
    # ------------------------------------------------------------------

    def execute_ttd_from_memory(
        self,
        mem: TernaryMemory,
        ttd_addr: int,
        mask_word: int,
        timestamp_ns: int = 0,
    ) -> TFabricOpStats:
        """Read 9-word TTD from Main Store, run native ternary op, and write back C + W8 STATUS."""
        raw_words = [mem.read(ttd_addr + i, supervisor=True)[0] for i in range(9)]
        ttd = TileTaskDescriptor.unpack_9_words(raw_words)
        tiles = self.active_tiles_from_mask(mask_word)

        if ttd.mode in (MODE_TBFP9_GEMM, MODE_TBFP3_GEMM):
            m, n, k = ttd.m, ttd.n, ttd.k
            # Read packed TBFP-9 matrices (4 nonets per T36 word)
            a_mat = self._read_tbfp9_matrix(mem, ttd.base_a, m, k, ttd.e_a)
            b_mat = self._read_tbfp9_matrix(mem, ttd.base_b, k, n, ttd.e_b)
            wgate = None
            if ttd.wgate == 1:
                scale_c = 3.0 ** (ttd.e_c - 8)
                wgate = (ttd.tau_neg * scale_c, ttd.tau_pos * scale_c)
            if ttd.mode == MODE_TBFP9_GEMM:
                c_out, stats = self.gemm_tbfp9(
                    a_mat, b_mat, active_tiles=tiles, zgate=(ttd.zgate == 1),
                    abft=(ttd.abft == 1), wgate_deadzone=wgate, timestamp_ns=timestamp_ns
                )
            else:
                c_out, stats = self.gemm_tbfp3(a_mat, b_mat, active_tiles=tiles, timestamp_ns=timestamp_ns)
            self._write_tbfp9_matrix(mem, ttd.base_c, c_out, ttd.e_c)
        elif ttd.mode == MODE_RADIX3_FFT:
            p = extract_field(ttd.m, 6, 6, 12)
            n_pts = 3**p
            # Read packed 18t complex TBFP-9C pairs (2 complex samples per T36 word)
            sig = np.zeros(n_pts, dtype=np.complex128)
            scale = 3.0 ** (ttd.e_a - 8)
            for idx in range(0, n_pts, 2):
                w, _ = mem.read(ttd.base_a + idx // 2, supervisor=True)
                n3, n2, n1, n0 = unpack_word_to_nonets(w)
                sig[idx] = (n3 + 1j * n2) * scale
                if idx + 1 < n_pts:
                    sig[idx + 1] = (n1 + 1j * n0) * scale
            spec, stats = self.fft_radix3_tbfp9c(sig, active_tiles=tiles, timestamp_ns=timestamp_ns)
            st = np.stack([spec.real, spec.imag], axis=-1)
            q_out, _ = quantize_tbfp(st, mant_trits=9)
            for idx in range(0, n_pts, 2):
                r0, i0 = int(q_out[idx, 0]), int(q_out[idx, 1])
                r1, i1 = (int(q_out[idx + 1, 0]), int(q_out[idx + 1, 1])) if idx + 1 < n_pts else (0, 0)
                mem.write(ttd.base_c + idx // 2, pack_nonets_to_word(r0, i0, r1, i1), supervisor=True)
        else:
            raise ValueError(f"Unsupported TTD mode {ttd.mode}")

        # Write completion status word W8 atomically
        w8 = (
            wrap(stats.status_code, 2) * POW3[34]
            + clamp(stats.cycles_40ns, 18) * POW3[16]
            + clamp(int(round(stats.zero_trit_suppressed_pct * 10)), 16)
        )
        mem.write(ttd_addr + 8, w8, supervisor=True)
        return stats

    @staticmethod
    def _read_tbfp9_matrix(
        mem: TernaryMemory, base: int, rows: int, cols: int, e_blk: int
    ) -> np.ndarray:
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
        mem: TernaryMemory, base: int, mat: np.ndarray, e_blk: int
    ) -> None:
        scale = 3.0 ** (e_blk - 8)
        q = np.clip(np.rint(mat / scale).astype(np.int64), -WMAX_9, WMAX_9).ravel()
        total = q.size
        for i in range(0, total, 4):
            n0 = int(q[i])
            n1 = int(q[i + 1]) if i + 1 < total else 0
            n2 = int(q[i + 2]) if i + 2 < total else 0
            n3 = int(q[i + 3]) if i + 3 < total else 0
            mem.write(base + i // 4, pack_nonets_to_word(n0, n1, n2, n3), supervisor=True)
