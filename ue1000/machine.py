"""Integrated Duwamish-I (UE-1000) System Model (OBB Unseen Level -9).

Combines all four orthogonal tiers under WEAR / UE Special Systems governance:
  - Tier 1: 8 x 36-trit C-Complex Vector-Scalar CPUs (30 ns UE-TECL), multi-CPU scheduler
  - Tier 2: 30-Tile T-Fabric (27 active + 3 hot spares, 40 ns UE-TECL, async TEXEC/TWAIT)
  - Tier 3a: 43,046,721-word (3^16) T-DRAM (81 active + 9 spare banks, 42t SEC-DED, 81-bank crossbar)
             + 177,147-word (3^11) Non-Volatile Dual-Core Ferrite Executive Vault
  - Tier 3b: 8 Autonomous DMA Data Channels (CHIO) + Alexandrian Vault (Disc, 9-Track Tape,
             Archival Silo with WEAR provenance, Channel 8 SETI Ring, Level -8 Digitisation Chute)
  - Tier 4: Beer/Ashby V-Complex (Algedonic Telemetry Bus + <75 us Hot-Spare Cutover)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .channels import IOChannelSubsystem, NUM_CHANNELS
from .cpu import C_CLOCK_NS, CComplexCPU
from .memory import (
    FERRITE_WORDS,
    MMIO_MIN,
    TDRAM_BANKS_ACTIVE,
    TDRAM_BANKS_SPARE,
    TDRAM_BANKS_TOTAL,
    TOTAL_TDRAM_WORDS,
    TernaryMemory,
)
from .ternary import POW3, clamp
from .tfabric import T_ACTIVE_MACS, T_CLOCK_NS, TFabric
from .triad36 import AssembledProgram, assemble
from .vcomplex import T_ACTIVE_TILES, T_SPARE_TILES, T_TOTAL_TILES, VComplex

# Locked OBB Unseen Level -9 Cellular Floorplan (Rev. 2 Project Plan)
OBB_LEVEL_NINE_CELLS: Dict[str, int] = {
    "Central Core (Lifts / Stairs / WCs)": 2,
    "Compute Vault (C-Complex + T-Fabric + Crossbar)": 2,
    "Main Ternary Memory Vault (Tier 3a)": 6,
    "Alexandrian Vault (Tier 3b — J. U. Jameson)": 8,
    "V-Complex & Beer Operations Gallery (Tier 4)": 4,
    "Researcher Terminal Suites (Pask / Licklider)": 3,
    "Oliver SETI Baseband Annex": 3,
    "UE Special Systems Maintenance & Staging Bay": 2,
}

# Memory-Mapped Control Aperture Offsets from MMIO_MIN (+100,000,000; PoO §3.1)
MMIO_SYS_ID = MMIO_MIN + 0                # Rev 1.5 system identifier (100015)
MMIO_GLOBAL_CLOCK = MMIO_MIN + 1          # Global 30 ns cycle clock (max across CPUs)
MMIO_CHANNEL_BASE = MMIO_MIN + 10         # +10..+17 : Data Channels 1..8 live status
MMIO_TILE_BASE = MMIO_MIN + 100           # +100..+129 : Physical Tiles 0..29 telemetry
MMIO_BANK_BASE = MMIO_MIN + 200           # +200..+289 : Physical T-DRAM Banks 0..89 health
MMIO_VCOMPLEX_SUMMARY = MMIO_MIN + 300    # Packed alarms | spares | cutovers
MMIO_BANK_CUTOVER_CMD = MMIO_MIN + 301    # Write logical bank 0..80 -> scrub-cutover!


class UE1000Machine:
    """Top-level integrated Duwamish-I (UE-1000) simulator."""

    def __init__(
        self,
        strict_ecc: bool = False,
        async_dma: bool = False,
        auto_cutover_erasure_threshold: Optional[int] = None,
    ) -> None:
        self.vcomplex = VComplex()
        self.mem = TernaryMemory(
            strict_ecc=strict_ecc,
            vcomplex=self.vcomplex,
            auto_cutover_erasure_threshold=auto_cutover_erasure_threshold,
        )
        self.tfabric = TFabric(self.vcomplex)
        self.io_channels = IOChannelSubsystem(vcomplex=self.vcomplex, async_dma=async_dma)
        self.cpus: List[CComplexCPU] = [
            CComplexCPU(
                cpu_id=i,
                mem=self.mem,
                tfabric=self.tfabric,
                io_channels=self.io_channels,
                vcomplex=self.vcomplex,
            )
            for i in range(8)
        ]
        # Wire live MMIO apertures (+100,000,000 .. +100,000,399)
        self.mem.mmio_read_handler = self._mmio_read
        self.mem.mmio_write_handler = self._mmio_write

    @property
    def cpu0(self) -> CComplexCPU:
        return self.cpus[0]

    @property
    def global_cycles(self) -> int:
        return max((c.cycles for c in self.cpus), default=0)

    def _mmio_read(self, phys: int) -> Optional[int]:
        if phys == MMIO_SYS_ID:
            return 100015  # UE-1000 Rev. 1.5
        if phys == MMIO_GLOBAL_CLOCK:
            return self.global_cycles
        if MMIO_CHANNEL_BASE <= phys < MMIO_CHANNEL_BASE + NUM_CHANNELS:
            ch = (phys - MMIO_CHANNEL_BASE) + 1
            return self.io_channels.channel_status_word(ch, now_ns=self.global_cycles * C_CLOCK_NS)
        if MMIO_TILE_BASE <= phys < MMIO_TILE_BASE + T_TOTAL_TILES:
            pid = phys - MMIO_TILE_BASE
            st = self.vcomplex.tiles[pid]
            role_trit = 1 if st.role == "ACTIVE" else (0 if st.role == "HOT_SPARE" else -1)
            return role_trit * POW3[24] + clamp(st.abft_faults, 6) * POW3[18] + clamp(st.abft_checks, 18)
        if MMIO_BANK_BASE <= phys < MMIO_BANK_BASE + TDRAM_BANKS_TOTAL:
            bid = phys - MMIO_BANK_BASE
            bh = self.mem.bank_health[bid]
            ret_trit = -1 if bh.retired else 1
            return ret_trit * POW3[24] + clamp(bh.erasure_count, 12) * POW3[12] + clamp(bh.sec_count, 12)
        if phys == MMIO_VCOMPLEX_SUMMARY:
            return (
                self.vcomplex.system_health_trit() * POW3[30]
                + clamp(len(self.vcomplex.spare_tiles), 6) * POW3[24]
                + clamp(len(self.vcomplex.cutovers), 12) * POW3[12]
                + clamp(self.vcomplex.algedonic_alarms, 12)
            )
        return None

    def _mmio_write(self, phys: int, val: int) -> bool:
        if phys == MMIO_BANK_CUTOVER_CMD:
            if 0 <= val < TDRAM_BANKS_ACTIVE:
                self.mem.scrub_and_cutover_bank(val)
            return True
        return False

    def load_program(self, prog: AssembledProgram, cpu_id: int = 0) -> None:
        for addr, word in prog.words.items():
            self.mem.write(addr, word, supervisor=True)
        self.cpus[cpu_id].pc = prog.origin
        self.cpus[cpu_id].halted = False

    def assemble_and_load(self, source: str, origin: int = 100, cpu_id: int = 0) -> AssembledProgram:
        prog = assemble(source, origin=origin)
        self.load_program(prog, cpu_id=cpu_id)
        return prog

    # ------------------------------------------------------------------
    # Multi-CPU Cycle-Stepped Execution (CPU 0..7 + Async Tier 2 & 3b)
    # ------------------------------------------------------------------

    def step_all(self, cpu_ids: Optional[Sequence[int]] = None) -> int:
        """Advance the active C-Complex CPUs by stepping the lowest-cycle runnable CPU.

        Keeps global simulated time monotonic across CPUs, async T-Fabric jobs, and DMA channels.
        Returns the cpu_id stepped, or -1 if all selected CPUs are halted.
        """
        ids = list(cpu_ids) if cpu_ids is not None else list(range(len(self.cpus)))
        runnable = [self.cpus[i] for i in ids if not self.cpus[i].halted]
        if not runnable:
            return -1
        # Event-driven cycle ordering: step whichever runnable CPU is earliest in simulated time
        nxt = min(runnable, key=lambda c: (c.cycles, c.cpu_id))
        nxt.step()
        now_ns = self.global_cycles * C_CLOCK_NS
        if self.tfabric._pending_jobs:
            self.tfabric.advance_to_ns(now_ns)
        if self.io_channels._pending:
            self.io_channels.advance_to_ns(now_ns)
        return nxt.cpu_id

    def run_all(
        self,
        cpu_ids: Optional[Sequence[int]] = None,
        max_steps: int = 100_000,
    ) -> int:
        """Run selected C-Complex CPUs concurrently in cycle-accurate order until all halt."""
        ids = list(cpu_ids) if cpu_ids is not None else list(range(len(self.cpus)))
        steps = 0
        while steps < max_steps:
            stepped = self.step_all(ids)
            if stepped < 0:
                break
            steps += 1
        return steps

    def telemetry_snapshot(self) -> Dict[str, object]:
        """Export structured live state for CLI reporting and the Beer VSM HTML Gallery."""
        active_tiles = [t for t in self.vcomplex.tiles.values() if t.role == "ACTIVE"]
        spare_tiles = [t for t in self.vcomplex.tiles.values() if t.role == "HOT_SPARE"]
        faulted_tiles = [t for t in self.vcomplex.tiles.values() if t.role == "FAULTED_BAY_LIT"]
        avg_zsave = (
            sum(self.tfabric.total_zero_trit_savings) / len(self.tfabric.total_zero_trit_savings)
            if self.tfabric.total_zero_trit_savings
            else 0.0
        )
        return {
            "system": "Duwamish-I (UE-1000) — Rev. 1.5",
            "site": "OBB Unseen Level -9 (28 Structural Cells + 2 Central Core Cells)",
            "level_9_cells": OBB_LEVEL_NINE_CELLS,
            "c_complex": {
                "cpus": len(self.cpus),
                "clock_ns": C_CLOCK_NS,
                "cpu0_cycles": self.cpu0.cycles,
                "cpu0_instructions": self.cpu0.instructions,
                "global_cycles": self.global_cycles,
                "total_instructions": sum(c.instructions for c in self.cpus),
                "istack_hits": sum(c.istack_hits for c in self.cpus),
                "istack_misses": sum(c.istack_misses for c in self.cpus),
            },
            "t_fabric": {
                "clock_ns": T_CLOCK_NS,
                "active_tiles": len(active_tiles),
                "hot_spares_remaining": len(spare_tiles),
                "faulted_bays": [t.physical_id for t in faulted_tiles],
                "active_macs": T_ACTIVE_MACS,
                "total_ops": self.tfabric.total_ops,
                "total_cycles": self.tfabric.total_cycles,
                "abft_checks": self.tfabric.total_abft_checks,
                "avg_zero_trit_power_saving_pct": round(avg_zsave, 2),
                "tiles": [
                    {
                        "physical_id": t.physical_id,
                        "logical_id": t.logical_id,
                        "role": t.role,
                        "temp_C": round(t.temp_C, 2),
                        "abft_checks": t.abft_checks,
                        "abft_faults": t.abft_faults,
                        "zero_trit_saved_pct": round(t.zero_trit_saved_pct, 1),
                    }
                    for t in self.vcomplex.tiles.values()
                ],
            },
            "memory": {
                "tdram_words": TOTAL_TDRAM_WORDS,
                "ferrite_words": FERRITE_WORDS,
                "active_banks": TDRAM_BANKS_ACTIVE,
                "spare_banks_remaining": len(self.mem.spare_banks),
                "reads": self.mem.reads,
                "writes": self.mem.writes,
                "crossbar_conflicts": self.mem.crossbar_conflicts,
                "bank_cutovers": self.mem.bank_cutovers,
                "total_sec_corrections": sum(b.sec_count for b in self.mem.bank_health.values()),
                "total_erasure_corrections": sum(b.erasure_count for b in self.mem.bank_health.values()),
            },
            "io_channels": self.io_channels.telemetry_summary(),
            "v_complex": {
                "health_trit": self.vcomplex.system_health_trit(),
                "algedonic_alarms": self.vcomplex.algedonic_alarms,
                "cutovers": [
                    {
                        "event_id": ev.event_id,
                        "kind": ev.kind,
                        "logical_id": ev.logical_id,
                        "old_physical_id": ev.old_physical_id,
                        "new_physical_id": ev.new_physical_id,
                        "fault_reason": ev.fault_reason,
                        "hw_cutover_us": ev.hw_cutover_us,
                        "replay_us": ev.replay_us,
                        "phases_us": ev.phases_us,
                    }
                    for ev in self.vcomplex.cutovers
                ],
                "gallery_lamps": list(self.vcomplex.gallery_lamps),
            },
        }
