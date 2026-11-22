"""Tier 4: V-Complex — Stafford Beer Algedonic Telemetry & W. Ross Ashby Cutover.

Conforms strictly to Principles of Operation (Rev. 1.5), Chapter 7:
  - Out-of-band Algedonic Telemetry Bus emitting 4-word (4 x T36) ATP packets
    with zero compute-cycle theft.
  - Ashby Ultrastable Hot-Spare Tile Cutover:
      0-5 us   : Algedonic hard-fault wire fires (ABFT mismatch / thermal / rail)
      5-15 us  : Freeze crossbar port k, gate tile power/clock, block corrupt writeback
      15-35 us : Ashby variety LUT selects spare s in {27, 28, 29}; remap crossbar LUT
      35-65 us : DMA reload TTD + checkpoint block (<= 2,187 words @ 150 Mwords/s ~ 15 us)
      65-75 us : Release tile s on interrupted micro-block + ABFT verify
      Total hardware cutover strictly < 75 us (60 us nominal)!
  - Hot-Spare T-DRAM Bank Scrub-Cutover across 9 spare banks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ternary import POW3, WORD_TRITS, clamp, wrap

CUTOVER_HW_MAX_US: float = 75.0
T_ACTIVE_TILES: int = 27
T_SPARE_TILES: int = 3
T_TOTAL_TILES: int = 30


@dataclass
class AlgedonicPacket:
    """4-word (4 x T36) out-of-band Algedonic Telemetry Packet per Chapter 7.1."""
    unit_id: int        # 9t
    health_trit: int    # +1 NOMINAL, 0 WARNING, -1 ALGEDONIC PAIN
    fault_code: int     # 6t (0=NONE, 1=ABFT_FAIL, 2=THERMAL, 3=RAIL_DRIFT, 4=MEM_ERASURE)
    temp_mC: int        # 12t cold-plate junction delta in 0.01 C
    freon_dP: int       # 8t pressure delta
    sec_count: int      # 18t
    erasure_count: int  # 18t
    abft_checks: int    # 18t
    abft_faults: int    # 6t
    zero_trit_pct: int  # 12t (0..1000 tenths of a percent)
    timestamp: int      # 36t global 30 ns clock

    def pack_words(self) -> Tuple[int, int, int, int]:
        """Pack into exact 4 x T36 words per PoO §7.1 (9+1+6+12+8 = 36, etc.)."""
        w0 = (
            clamp(self.unit_id, 9) * POW3[27]
            + clamp(self.health_trit, 1) * POW3[26]
            + clamp(self.fault_code, 6) * POW3[20]
            + clamp(self.temp_mC, 12) * POW3[8]
            + clamp(self.freon_dP, 8)
        )
        w1 = clamp(self.sec_count, 18) * POW3[18] + clamp(self.erasure_count, 18)
        w2 = (
            clamp(self.abft_checks, 18) * POW3[18]
            + clamp(self.abft_faults, 6) * POW3[12]
            + clamp(self.zero_trit_pct, 12)
        )
        w3 = wrap(self.timestamp, WORD_TRITS)
        return (w0, w1, w2, w3)


@dataclass
class CutoverEvent:
    event_id: int
    kind: str  # "TILE_CUTOVER" | "BANK_CUTOVER" | "SPARES_EXHAUSTED"
    logical_id: int
    old_physical_id: int
    new_physical_id: Optional[int]
    fault_reason: str
    hw_cutover_us: float
    replay_us: float
    timestamp_ns: int
    phases_us: Dict[str, float] = field(default_factory=dict)


@dataclass
class TilePhysicalState:
    physical_id: int
    role: str              # "ACTIVE" | "HOT_SPARE" | "FAULTED_BAY_LIT"
    logical_id: Optional[int]
    temp_C: float = 22.0   # Freon cold-plate nominal 22.0 C (±3 C envelope)
    abft_checks: int = 0
    abft_faults: int = 0
    zero_trit_saved_pct: float = 0.0


class VComplex:
    """Stafford Beer VSM Supervisory Ring & Ashby Ultrastable Fault-Healing Controller."""

    def __init__(self) -> None:
        # Logical tile 0..26 -> physical tile 0..26; hot spares = 27, 28, 29
        self.tile_lut: List[int] = list(range(T_ACTIVE_TILES))
        self.spare_tiles: List[int] = list(range(T_ACTIVE_TILES, T_TOTAL_TILES))
        self.tiles: Dict[int, TilePhysicalState] = {}
        for p in range(T_ACTIVE_TILES):
            self.tiles[p] = TilePhysicalState(physical_id=p, role="ACTIVE", logical_id=p)
        for p in range(T_ACTIVE_TILES, T_TOTAL_TILES):
            self.tiles[p] = TilePhysicalState(physical_id=p, role="HOT_SPARE", logical_id=None)

        self.packets: List[AlgedonicPacket] = []
        self.cutovers: List[CutoverEvent] = []
        self.gallery_lamps: List[str] = []
        self.algedonic_alarms: int = 0

    def emit_telemetry(self, pkt: AlgedonicPacket) -> None:
        self.packets.append(pkt)
        if pkt.health_trit == -1:
            self.algedonic_alarms += 1

    def trigger_tile_cutover(
        self,
        logical_tile: int,
        fault_reason: str,
        reload_words: int = 2187,
        replay_us: float = 25.0,
        timestamp_ns: int = 0,
    ) -> CutoverEvent:
        """Execute Ashby < 75 us Hot-Spare Tile Cutover state machine."""
        old_phys = self.tile_lut[logical_tile]
        self.algedonic_alarms += 1

        # Emit hard-fault Algedonic Telemetry Packet
        self.emit_telemetry(
            AlgedonicPacket(
                unit_id=old_phys,
                health_trit=-1,
                fault_code=1 if "ABFT" in fault_reason else 2,
                temp_mC=int(round((self.tiles[old_phys].temp_C - 22.0) * 100)),
                freon_dP=0,
                sec_count=0,
                erasure_count=0,
                abft_checks=self.tiles[old_phys].abft_checks,
                abft_faults=self.tiles[old_phys].abft_faults + 1,
                zero_trit_pct=int(round(self.tiles[old_phys].zero_trit_saved_pct * 10)),
                timestamp=timestamp_ns // 30,
            )
        )

        if not self.spare_tiles:
            self.tiles[old_phys].role = "FAULTED_BAY_LIT"
            self.tiles[old_phys].abft_faults += 1
            ev = CutoverEvent(
                event_id=len(self.cutovers) + 1,
                kind="SPARES_EXHAUSTED",
                logical_id=logical_tile,
                old_physical_id=old_phys,
                new_physical_id=None,
                fault_reason=fault_reason,
                hw_cutover_us=0.0,
                replay_us=0.0,
                timestamp_ns=timestamp_ns,
            )
            self.cutovers.append(ev)
            self.gallery_lamps.append(f"CRITICAL: Tile Bay {old_phys} faulted — ALL 3 SPARES EXHAUSTED (Trap 12)")
            return ev

        new_phys = self.spare_tiles.pop(0)

        # Phase-by-phase hardware timing budget per Chapter 7.2
        t_detect = 4.0
        t_isolate = 9.0
        t_remap = 14.0
        # DMA reload <= 2,187 words @ 150 Mwords/s (~14.6 us for full bank)
        t_reload = max(4.0, (min(reload_words, 2187) / 150.0))
        t_verify_release = 8.0
        hw_total = round(t_detect + t_isolate + t_remap + t_reload + t_verify_release, 2)
        assert hw_total < CUTOVER_HW_MAX_US, f"Cutover {hw_total} us exceeded 75 us spec!"

        # Update physical states & Crossbar LUT
        self.tiles[old_phys].role = "FAULTED_BAY_LIT"
        self.tiles[old_phys].logical_id = None
        self.tiles[old_phys].abft_faults += 1

        self.tiles[new_phys].role = "ACTIVE"
        self.tiles[new_phys].logical_id = logical_tile
        self.tile_lut[logical_tile] = new_phys

        ev = CutoverEvent(
            event_id=len(self.cutovers) + 1,
            kind="TILE_CUTOVER",
            logical_id=logical_tile,
            old_physical_id=old_phys,
            new_physical_id=new_phys,
            fault_reason=fault_reason,
            hw_cutover_us=hw_total,
            replay_us=replay_us,
            timestamp_ns=timestamp_ns,
            phases_us={
                "algedonic_detect": t_detect,
                "isolate_port": t_isolate,
                "ashby_lut_remap": t_remap,
                "sram_dma_reload": round(t_reload, 2),
                "verify_release": t_verify_release,
            },
        )
        self.cutovers.append(ev)
        self.gallery_lamps.append(
            f"BAY {old_phys:02d} SWAP LAMP LIT: Logical T{logical_tile:02d} -> Hot Spare S{new_phys} "
            f"in {hw_total:.1f} us ({fault_reason})"
        )
        return ev

    def record_bank_cutover(
        self,
        logical_bank: int,
        old_phys: int,
        new_phys: int,
        words_scrubbed: int,
        timestamp_ns: int = 0,
    ) -> CutoverEvent:
        ev = CutoverEvent(
            event_id=len(self.cutovers) + 1,
            kind="BANK_CUTOVER",
            logical_id=logical_bank,
            old_physical_id=old_phys,
            new_physical_id=new_phys,
            fault_reason=f"UE-TD2000 erasure threshold ({words_scrubbed} words scrubbed)",
            hw_cutover_us=12.0,
            replay_us=25_000.0,  # ~25 ms background scrub for 531,441-word bank
            timestamp_ns=timestamp_ns,
        )
        self.cutovers.append(ev)
        self.gallery_lamps.append(
            f"MEM BANK {old_phys:02d} RETIRED: Logical B{logical_bank:02d} -> Spare Bank {new_phys:02d}"
        )
        return ev
