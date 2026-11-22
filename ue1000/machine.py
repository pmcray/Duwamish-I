"""Integrated Duwamish-I (UE-1000) System Model (OBB Unseen Level -9).

Combines all four orthogonal tiers under WEAR / UE Special Systems governance:
  - Tier 1: 8 x 36-trit C-Complex Vector-Scalar CPUs (30 ns UE-TECL)
  - Tier 2: 30-Tile T-Fabric (27 active + 3 hot spares, 40 ns UE-TECL)
  - Tier 3: 43,046,721-word (3^16) T-DRAM (81 active + 9 spare banks, 42t SEC-DED)
            + 177,147-word (3^11) Non-Volatile Dual-Core Ferrite Executive Vault
  - Tier 4: Beer/Ashby V-Complex (Algedonic Telemetry Bus + <75 us Hot-Spare Cutover)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .cpu import C_CLOCK_NS, CComplexCPU
from .memory import (
    FERRITE_WORDS,
    TDRAM_BANKS_ACTIVE,
    TDRAM_BANKS_SPARE,
    TOTAL_TDRAM_WORDS,
    TernaryMemory,
)
from .tfabric import T_ACTIVE_MACS, T_CLOCK_NS, TFabric
from .triad36 import AssembledProgram, assemble
from .vcomplex import T_ACTIVE_TILES, T_SPARE_TILES, VComplex

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


class UE1000Machine:
    """Top-level integrated Duwamish-I (UE-1000) simulator."""

    def __init__(self, strict_ecc: bool = False) -> None:
        self.vcomplex = VComplex()
        self.mem = TernaryMemory(strict_ecc=strict_ecc)
        self.tfabric = TFabric(self.vcomplex)
        self.cpus: List[CComplexCPU] = [
            CComplexCPU(cpu_id=i, mem=self.mem, tfabric=self.tfabric) for i in range(8)
        ]

    @property
    def cpu0(self) -> CComplexCPU:
        return self.cpus[0]

    def load_program(self, prog: AssembledProgram, cpu_id: int = 0) -> None:
        for addr, word in prog.words.items():
            self.mem.write(addr, word, supervisor=True)
        self.cpus[cpu_id].pc = prog.origin
        self.cpus[cpu_id].halted = False

    def assemble_and_load(self, source: str, origin: int = 100, cpu_id: int = 0) -> AssembledProgram:
        prog = assemble(source, origin=origin)
        self.load_program(prog, cpu_id=cpu_id)
        return prog

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
                "istack_hits": self.cpu0.istack_hits,
                "istack_misses": self.cpu0.istack_misses,
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
                "bank_cutovers": self.mem.bank_cutovers,
                "total_sec_corrections": sum(b.sec_count for b in self.mem.bank_health.values()),
                "total_erasure_corrections": sum(b.erasure_count for b in self.mem.bank_health.values()),
            },
            "v_complex": {
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
