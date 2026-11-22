"""Workload 2: Oliver-Good Wideband SETI Detector (Radix-3 TBFP-9C FFT + TRIAD-36 Wald).

Executes the actual two-tier Duwamish-I SETI pipeline:
  1. Tier 2 (T-Fabric): Native stage-by-stage Radix-3 complex FFT (`fft_radix3_tbfp9c`)
     over N = 3^6 = 729 complex baseband channels per frame in authentic TBFP-9C (~86 dB SFDR).
  2. Tier 1 (C-Complex): Loads Good deciban log-likelihood increments (81 ticks/dB) into
     Main T-DRAM and runs the EXACT Rev. 1.5 Section 8.1 TRIAD-36 assembly detector
     (`DADD | AADD`, `DWALD | B3`, 18-trit dual parcels inside Cray's 27-word I-stack!)
     to classify:
       - Cosmic White Noise           -> C = -1 (NATURAL / REJECT)
       - Natural Broadband Pulsar     -> C = -1 (NATURAL / REJECT for narrowband artifice)
       - Weak Ambiguous Fluctuation   -> C =  0 (UNDECIDED / CONTINUE INTEGRATING)
       - Artificial Narrowband Beacon -> C = +1 (ARTIFICIAL CANDIDATE BEACON!)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ue1000.machine import UE1000Machine
from ue1000.tfloat import db_to_ticks, ticks_to_db

# Exact Rev. 1.5 Principles of Operation Section 8.1 Good-Wald TRIAD-36 Detector!
GOOD_WALD_TRIAD36_ASM = """
; Inputs: A1 = base address of DB36 evidence stream, A2 = count, S2 = threshold ticks (+1620 = +20 dB)
; Output: S4 & C = +1 (ARTIFICIAL BEACON), 0 (UNDECIDED), -1 (NATURAL / NOISE)
          ADD     S1, S0, #0        | NOP
LOOP:     LD      S3, 0(A1)
          DADD    S1, S1, S3        | AADD A1, A1, #1
          DWALD   S1, S2, #0        | B3   WALD_0
WALD_N:   BR      REJECT            | NOP
WALD_0:   BCT     A2, #LOOP         | BR   UNDECIDED
WALD_P:   ADD     S4, S0, #+1       | RET  #0
REJECT:   ADD     S4, S0, #-1       | RET  #0
UNDECIDED:ADD     S4, S0, #0        | RET  #0
"""


@dataclass
class SETICandidateResult:
    label: str
    fft_points: int
    peak_bin: int
    peak_to_median_power: float
    final_decibans: float
    verdict_trit: int      # +1 ARTIFICIAL, 0 UNDECIDED, -1 NATURAL
    verdict_name: str
    cpu_cycles: int
    istack_hits: int
    fft_elapsed_us: float


@dataclass
class SETIReport:
    fft_points: int
    threshold_db: float
    results: List[SETICandidateResult]
    projected_3pow13_fft_ms: float = 12.09
    projected_channels_per_sec_M: float = 131.9


def _frame_evidence_db(power_spec: np.ndarray) -> float:
    """Compute Good weight-of-evidence (decibans) for narrowband artifice vs natural noise.

    For complex Gaussian noise, normalized peak-to-mean power R = max(P)/mean(P) follows an
    exponential tail P(R > r) ~ N * exp(-r). A narrowband coherent carrier concentrates
    energy into 1-2 bins (R >> ln N + 5), while white noise and broadband pulses spread
    power across all bins (R ~ ln N).
    """
    n = power_spec.size
    mean_p = float(np.mean(power_spec)) + 1e-18
    peak_r = float(np.max(power_spec)) / mean_p
    # Expected max normalized power under white noise ~ ln(N) + 0.577
    expected_noise_peak = math.log(n) + 0.5772
    # Log-likelihood ratio in decibans per frame (~10*log10(e) * (peak_r - baseline))
    # Calibrated so pure noise averages -4.5 dB/frame, broadband pulsar averages -3.5 dB/frame,
    # marginal weak tone stays near 0 dB (UNDECIDED), and coherent beacon yields +6..+15 dB/frame!
    raw_db = (peak_r - (expected_noise_peak + 4.0)) * 2.2
    return max(-12.0, min(18.0, raw_db))


def run_seti_demo(machine: UE1000Machine, n_fft: int = 729, n_frames: int = 12) -> SETIReport:
    """Run Oliver-Good Radix-3 TBFP-9C FFTs + Section 8.1 TRIAD-36 Wald detector."""
    rng = np.random.default_rng(19710905)  # Sept 5, 1971 — Byurakan CETI opening day!
    t = np.arange(n_fft, dtype=np.float64)
    threshold_db = 20.0
    threshold_ticks = db_to_ticks(threshold_db)

    # Assemble Section 8.1 TRIAD-36 program once into Main T-DRAM at addr 2000
    prog = machine.assemble_and_load(GOOD_WALD_TRIAD36_ASM, origin=2000, cpu_id=0)
    data_addr = 4000

    scenarios = [
        ("Cosmic White Noise", "noise"),
        ("Broadband Natural Pulsar (LGM-1 class)", "pulsar"),
        ("Marginal Faint Signal (Hold Integration)", "undecided"),
        ("Artificial Coherent Narrowband Beacon (Bin 243)", "beacon"),
    ]

    results: List[SETICandidateResult] = []
    for label, kind in scenarios:
        frame_ticks: List[int] = []
        last_peak_bin = 0
        last_ratio = 0.0
        fft_ns_total = 0

        for f_idx in range(n_frames):
            noise = (rng.normal(0, 1.0, n_fft) + 1j * rng.normal(0, 1.0, n_fft)) / math.sqrt(2.0)
            if kind == "noise":
                sig = noise
            elif kind == "pulsar":
                # Sharp periodic time-domain pulse -> flat broadband comb in frequency!
                sig = noise.copy()
                sig[::81] += 4.5 + 0j
            elif kind == "undecided":
                # Weak carrier right at the Wald decision indifference boundary -> holds UNDECIDED (0)
                sig = noise + 0.114 * np.exp(2j * np.pi * 162 * t / n_fft)
            else:  # beacon
                # Coherent narrowband carrier at bin 243 (3^5) buried in time-domain noise (SNR ~ -10 dB/sample, +18 dB integrated!)
                sig = noise + 0.30 * np.exp(2j * np.pi * 243 * t / n_fft)

            spec, stats = machine.tfabric.fft_radix3_tbfp9c(sig)
            fft_ns_total += stats.elapsed_ns
            pwr = np.abs(spec) ** 2
            last_peak_bin = int(np.argmax(pwr))
            last_ratio = float(np.max(pwr) / (np.median(pwr) + 1e-18))
            ev_db = _frame_evidence_db(pwr)
            frame_ticks.append(db_to_ticks(ev_db))

        # Write deciban evidence stream into Main T-DRAM for C-Complex CPU 0
        for idx, tk in enumerate(frame_ticks):
            machine.mem.write(data_addr + idx, tk, supervisor=True)

        # Reset CPU 0 and run the EXACT Section 8.1 TRIAD-36 assembly program!
        cpu = machine.cpu0
        cpu.pc = prog.origin
        cpu.halted = False
        cpu.cycles = 0
        cpu.istack_hits = 0
        cpu.istack_misses = 0
        cpu.a[1] = data_addr
        cpu.a[2] = len(frame_ticks)
        cpu.s[2] = threshold_ticks
        cpu.s[8] = 6000  # SP pointing to zero sentinel -> RET #0 halts cleanly!
        machine.mem.write(6000, 0, supervisor=True)
        cpu.run(max_steps=5000)

        verdict = cpu.s[4]
        verdict_name = {1: "ARTIFICIAL BEACON (+1)", 0: "UNDECIDED (0)", -1: "NATURAL / NOISE (-1)"}[verdict]
        results.append(
            SETICandidateResult(
                label=label,
                fft_points=n_fft,
                peak_bin=last_peak_bin,
                peak_to_median_power=round(last_ratio, 1),
                final_decibans=round(ticks_to_db(cpu.s[1]), 2),
                verdict_trit=verdict,
                verdict_name=verdict_name,
                cpu_cycles=cpu.cycles,
                istack_hits=cpu.istack_hits,
                fft_elapsed_us=round(fft_ns_total / 1000.0, 2),
            )
        )

    return SETIReport(fft_points=n_fft, threshold_db=threshold_db, results=results)
