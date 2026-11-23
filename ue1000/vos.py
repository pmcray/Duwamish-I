"""Tier 4 Supervisory Operating System (`V-OS`) & Interactive OBB Level -9 Console Shell.

Conforms to Duwamish-I Project Plan (Rev. 2) §2.4, §4.3, §6.1 & PoO Rev. 1.5 Chapters 6–7:
  - Runs strictly on the isolated Tier 4 V-Complex, physically decoupled from C-Complex's
    bare-metal Ferrite Vault `C-EXEC`.
  - WEAR Compartment Capability Security (authenticated principals, clearance levels,
    compartment tags enforced across Jameson's Alexandrian Vault and privileged commands).
  - Alexandrian Vault Filesystem (`DISC:`, `SILO:`, `TAPE:`, `CHUTE:`, `PROGRAMS:`) backed
    directly by Tier 3b `CHIO` DMA Channels 1..8!
  - Multi-CPU `C-EXEC` batch job submission, preemptive round-robin scheduling, and
    hardware-isolated `[BASE, BASE+LIMIT)` User execution.
  - Unified interactive OBB Level -9 console launcher tying together Beer VSM telemetry,
    all 9 languages (`SAL-3`, `TRI-TRAN 70V`, `TRILISP-36`, `CONVERSE`, `DYNAMO-X`,
    `APL\\36`, `REFAL-36`, `FORTH-3`, `TRI-INTERCAL`), and all 3 anchor workloads.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .cexec import CExecHypervisor, TRAP_NAMES
from .channels import (
    CH_OK,
    CH_PROVENANCE_DENIED,
    ChannelControlBlock,
    DEV_DISC,
    DEV_L8_CHUTE,
    DEV_SILO,
    DEV_TAPE,
    Level8DigitisationChute,
    OP_READ,
    OP_READ_BACKWARD,
    OP_REWIND,
    OP_WRITE,
)
from .machine import (
    MMIO_BANK_CUTOVER_CMD,
    MMIO_SYS_ID,
    MMIO_VCOMPLEX_SUMMARY,
    UE1000Machine,
)
from .ternary import char_to_obb729, obb729_to_char, pack_trytes_to_words, unpack_words_to_trytes

# Supervisor staging scratch addresses inside T-DRAM used by V-OS CHIO channel programs
_VOS_CCB_ADDR = -260_000
_VOS_IO_BUF_ADDR = -270_000


class VOSError(Exception):
    """Raised on invalid V-OS command syntax or WEAR capability denial."""


@dataclass(frozen=True)
class WearPrincipal:
    username: str
    role: str
    clearance: int               # 1=CARREL, 2=INVESTIGATOR, 3=CUSTODIAN/SYSOP
    compartments: Tuple[int, ...]  # Allowed WEAR compartment tags (0 = open/unrestricted)
    can_fault_inject: bool = False


DEFAULT_PRINCIPALS: Dict[str, WearPrincipal] = {
    "BROOKS": WearPrincipal("BROOKS", "Chief System Architect (SYSOP)", 3, (0, 1971, 2000, 3000, 7777), True),
    "JAMESON": WearPrincipal("JAMESON", "University Librarian / Vault Custodian", 3, (0, 1971, 7777), True),
    "ROSENBLATT": WearPrincipal("ROSENBLATT", "T-Fabric & Byurakan AI Lead", 2, (0, 1971), True),
    "OLIVER": WearPrincipal("OLIVER", "SETI Baseband Annex Lead", 2, (0, 1971, 3000), True),
    "FORRESTER": WearPrincipal("FORRESTER", "Cliodynamics & Store Lead", 2, (0, 2000), False),
    "KNUTH": WearPrincipal("KNUTH", "ISA & Compilers Lead", 2, (0, 1971, 2000), False),
    "GUEST": WearPrincipal("GUEST", "OBB Research Carrel Terminal", 1, (0,), False),
}


@dataclass
class DiscFileEntry:
    name: str
    spindle: int
    sector: int
    word_count: int
    owner: str
    compartment: int = 0


@dataclass
class SiloFileEntry:
    name: str
    cartridge_id: int
    word_count: int
    compartment: int
    label: str


class VOSSession:
    """Interactive & scriptable Tier 4 V-OS Supervisory Console Session."""

    def __init__(
        self,
        machine: Optional[UE1000Machine] = None,
        user: str = "BROOKS",
        programs_dir: Optional[Path] = None,
    ) -> None:
        self.machine = machine if machine is not None else UE1000Machine()
        self.cexec = CExecHypervisor(self.machine)
        self.cexec.boot()

        self.principals: Dict[str, WearPrincipal] = dict(DEFAULT_PRINCIPALS)
        uname = user.upper()
        if uname not in self.principals:
            raise VOSError(f"Unknown WEAR principal {user!r}")
        self.current_user: WearPrincipal = self.principals[uname]
        self.active_compartment: int = 0

        self.programs_dir = programs_dir or (Path(__file__).resolve().parent.parent / "programs")
        self.disc_catalog: Dict[str, DiscFileEntry] = {}
        self.silo_catalog: Dict[str, SiloFileEntry] = {}
        self._next_disc_sector: int = 0
        self.audit_log: List[str] = []
        self._apl_session = None
        self._converse_session = None

        self._seed_alexandrian_vault()

    # ------------------------------------------------------------------
    # Alexandrian Vault Seeding & Hardware CHIO Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_text_words(text: str) -> List[int]:
        trytes = [char_to_obb729(ch) for ch in text] + [0]
        return pack_trytes_to_words(trytes)

    @staticmethod
    def _decode_text_words(words: List[int]) -> str:
        chars: List[str] = []
        for tr in unpack_words_to_trytes(words):
            if tr == 0:
                break
            chars.append(obb729_to_char(tr))
        return "".join(chars)

    def _run_chio(self, channel: int, ccb: ChannelControlBlock) -> Tuple[int, int]:
        """Dispatch a real hardware CHIO CCB through Tier 3b IOChannelSubsystem."""
        for i, w in enumerate(ccb.pack_6_words()):
            self.machine.mem.write(_VOS_CCB_ADDR + i, w, supervisor=True)
        status, words, _ = self.machine.io_channels.execute_chio(
            channel=channel,
            ccb_addr=_VOS_CCB_ADDR,
            mem=self.machine.mem,
            now_ns=self.machine.global_cycles * 30,
        )
        return status, words

    def _seed_alexandrian_vault(self) -> None:
        """Populate DISC: and SILO: with period-authentic OBB Level -9 records via Tier 3b."""
        self.write_disc_file(
            "MOTD.TXT",
            "DUWAMISH-I (UE-1000) LEVEL -9 V-OS REV 1.5 — WEAR CHARTER ACTIVE. OBEY ANTI-CAMEL.",
            compartment=0,
            system_seed=True,
        )
        self.write_disc_file(
            "CATALOGUE.IDX",
            "CAT-001 SETUN-70 | CAT-002 WOODWORTH ALEXANDRIAN VAULT | CAT-003 BYURAKAN CETI 1971",
            compartment=0,
            system_seed=True,
        )
        # Seed Alexandrian Silo cartridges (open + restricted WEAR compartments!)
        silo_seeds = [
            ("OPEN_NOTES.TXT", 1, 0, "OBB-OPEN-01",
             "OBB LEVEL -9 FLOORPLAN: 2 CORE + 28 STRUCTURAL CELLS. 27 ACTIVE + 3 SPARE T-TILES."),
            ("BYURAKAN_71.MEMO", 71, 1971, "WEAR-CETI-1971",
             "ROSENBLATT-KELVANOV MEMO (SEPT 1971): ECO-71-09 ZERO-TRIT GATE-SUPPRESS & TB1 MODE."),
            ("WEAR_CHARTER.SEC", 99, 7777, "WEAR-COMPARTMENT-7777",
             "EYES ONLY — WEAR CHARTER 1.000 G$ CAPTIVE TERNARY FOUNDRY & LEVEL -9 VAULT AUDIT."),
        ]
        for name, cid, comp, label, body in silo_seeds:
            words = self._encode_text_words(body)
            self.machine.io_channels.silo.mount_cartridge(
                cartridge_id=cid, words=words, label=label, provenance_compartment=comp
            )
            self.silo_catalog[name] = SiloFileEntry(
                name=name, cartridge_id=cid, word_count=len(words), compartment=comp, label=label
            )

        # Preload one record in Jameson's Level -8 Digitisation Chute
        self.machine.io_channels.l8_chute.feed_text_record(
            "L8-SCAN-0412: BRUSENTSOV (1970) SETUN-70 MICROFORM REEL #19 — VERIFIED JAMESON",
            provenance_tag=1971,
        )

    def write_disc_file(
        self, name: str, text: str, compartment: int = 0, system_seed: bool = False
    ) -> DiscFileEntry:
        clean_name = name.upper()
        words = self._encode_text_words(text)
        sector = self._next_disc_sector
        self._next_disc_sector += len(words)
        for i, w in enumerate(words):
            self.machine.mem.write(_VOS_IO_BUF_ADDR + i, w, supervisor=True)
        ccb = ChannelControlBlock(
            device_type=DEV_DISC,
            unit=0,
            op=OP_WRITE,
            mem_addr=_VOS_IO_BUF_ADDR,
            word_count=len(words),
            media_pos=sector,
            provenance_tag=compartment,
        )
        status, written = self._run_chio(channel=1, ccb=ccb)
        if status != CH_OK or written != len(words):
            raise VOSError(f"Channel 1 DISC write failed for {clean_name} (status={status})")
        entry = DiscFileEntry(
            name=clean_name,
            spindle=0,
            sector=sector,
            word_count=len(words),
            owner="SYSTEM" if system_seed else self.current_user.username,
            compartment=compartment,
        )
        self.disc_catalog[clean_name] = entry
        return entry

    def read_disc_file(self, name: str) -> str:
        clean_name = name.upper()
        entry = self.disc_catalog.get(clean_name)
        if entry is None:
            raise VOSError(f"DISC:{clean_name} not found")
        if entry.compartment != 0 and entry.compartment not in self.current_user.compartments:
            self.audit_log.append(f"DENIED {self.current_user.username} READ DISC:{clean_name}")
            raise VOSError(f"WEAR COMPARTMENT DENIED: DISC:{clean_name} requires tag {entry.compartment}")
        ccb = ChannelControlBlock(
            device_type=DEV_DISC,
            unit=entry.spindle,
            op=OP_READ,
            mem_addr=_VOS_IO_BUF_ADDR,
            word_count=entry.word_count,
            media_pos=entry.sector,
            provenance_tag=entry.compartment,
        )
        status, n = self._run_chio(channel=1, ccb=ccb)
        if status != CH_OK:
            raise VOSError(f"Channel 1 DISC read failed for {clean_name} (status={status})")
        words = [self.machine.mem.read(_VOS_IO_BUF_ADDR + i, supervisor=True)[0] for i in range(n)]
        return self._decode_text_words(words)

    def read_silo_file(self, name: str) -> str:
        clean_name = name.upper()
        entry = self.silo_catalog.get(clean_name)
        if entry is None:
            raise VOSError(f"SILO:{clean_name} not found")
        # Present user's active compartment tag (or auto-present if user holds required tag!)
        presented = self.active_compartment
        if entry.compartment in self.current_user.compartments and presented == 0:
            presented = entry.compartment
        if presented not in self.current_user.compartments:
            presented = 0  # Principal cannot forge a compartment tag they don't hold!
        ccb = ChannelControlBlock(
            device_type=DEV_SILO,
            unit=0,
            op=OP_READ,
            mem_addr=_VOS_IO_BUF_ADDR,
            word_count=entry.word_count,
            media_pos=entry.cartridge_id,
            provenance_tag=presented,
        )
        status, n = self._run_chio(channel=3, ccb=ccb)
        if status == CH_PROVENANCE_DENIED:
            msg = (
                f"WEAR PROVENANCE DENIED (-2): {self.current_user.username} lacks compartment "
                f"{entry.compartment} for SILO:{clean_name} (logged to Beer Algedonic Bus)"
            )
            self.audit_log.append(msg)
            raise VOSError(msg)
        if status != CH_OK:
            raise VOSError(f"Channel 3 SILO read failed for {clean_name} (status={status})")
        words = [self.machine.mem.read(_VOS_IO_BUF_ADDR + i, supervisor=True)[0] for i in range(n)]
        self.audit_log.append(f"OK {self.current_user.username} READ SILO:{clean_name} (tag={presented})")
        return self._decode_text_words(words)

    # ------------------------------------------------------------------
    # Command Execution Engine
    # ------------------------------------------------------------------

    @property
    def prompt(self) -> str:
        return f"V-OS [L-9] {self.current_user.username}> "

    def execute_line(self, raw_line: str) -> List[str]:
        """Execute a single V-OS command line and return output lines."""
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            return []
        first_word = line.split(None, 1)[0].upper()
        raw_rest = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
        if first_word in ("CONVERSE", "APL", "FORTH"):
            cmd = first_word
            args: List[str] = []
        else:
            try:
                tokens = shlex.split(line)
            except ValueError as exc:
                raise VOSError(f"Command parse error: {exc}") from exc
            if not tokens:
                return []
            cmd = tokens[0].upper()
            args = tokens[1:]

        if cmd in ("HELP", "?"):
            return [
                "V-OS Rev. 1.5 Commands (OBB Unseen Level -9 — WEAR Charter):",
                "  Session  : LOGIN <user> | WHO | COMP <tag> | AUDIT | HELP | QUIT",
                "  Hardware : VSM | TILES | MEMORY | CHANNELS | MMIO | IPL | ACK",
                "             FAULT TILE <0..26> [reason] | CUTOVER BANK <0..80>",
                "  Vault    : LS [DISC:|SILO:|PROGRAMS:] | CAT <vol:file> | WRITE <DISC:file> <text>",
                "             STAGE <SILO:file> <DISC:file> | TAPE WRITE|READ|READBACK|REWIND",
                "             CHUTE FEED <text> | CHUTE INGEST <DISC:file>",
                "  C-EXEC   : SUBMIT <name> <asm_text_or_file> [--cpu N] [--quantum Q]",
                "             PS | DISPATCH [rounds] | CEXEC",
                "  Compute  : RUN <program_path> | CONVERSE <cmd> | APL <expr> | FORTH <words> | DEMO <saew|seti|clio>",
            ]

        if cmd == "LOGIN":
            if not args:
                raise VOSError("Usage: LOGIN <username>")
            uname = args[0].upper()
            if uname not in self.principals:
                raise VOSError(f"Unknown WEAR principal {uname!r} (valid: {', '.join(sorted(self.principals))})")
            self.current_user = self.principals[uname]
            self.active_compartment = 0
            self.audit_log.append(f"LOGIN {uname} (clearance={self.current_user.clearance})")
            return [
                f"Authenticated: {self.current_user.username} — {self.current_user.role} "
                f"[Clearance {self.current_user.clearance}, Compartments {self.current_user.compartments}]"
            ]

        if cmd == "WHO":
            u = self.current_user
            return [
                f"Principal    : {u.username} ({u.role})",
                f"Clearance    : {u.clearance} | Active Compartment: {self.active_compartment} | Allowed: {u.compartments}",
                f"Site         : OBB Unseen Level -9 (V-Complex Supervisory Console)",
            ]

        if cmd == "COMP":
            if not args:
                return [f"Active WEAR compartment tag: {self.active_compartment} (allowed: {self.current_user.compartments})"]
            tag = int(args[0])
            if tag not in self.current_user.compartments:
                self.audit_log.append(f"DENIED {self.current_user.username} COMP {tag}")
                raise VOSError(f"WEAR DENIAL: {self.current_user.username} not cleared for compartment {tag}")
            self.active_compartment = tag
            return [f"Active WEAR compartment switched to {tag}"]

        if cmd == "AUDIT":
            if not self.audit_log:
                return ["(No WEAR audit entries recorded yet)"]
            return ["=== WEAR & Alexandrian Vault Audit Log ==="] + [f"  {e}" for e in self.audit_log[-20:]]

        if cmd == "IPL":
            if self.current_user.clearance < 2:
                raise VOSError("WEAR DENIAL: IPL requires clearance >= 2")
            st = self.cexec.boot()
            return [
                f"C-EXEC Cold-Boot IPL complete: {st['kernel_words']} words in Non-Volatile Ferrite Vault "
                f"(magic={st['boot_magic']}, MMIO_SYS_ID=100015)"
            ]

        if cmd in ("VSM", "STATUS"):
            snap = self.machine.telemetry_snapshot()
            vc = snap["v_complex"]
            tf = snap["t_fabric"]
            mem = snap["memory"]
            cc = snap["c_complex"]
            health_name = {1: "+1 NOMINAL", 0: "0 WARNING", -1: "-1 ALGEDONIC PAIN"}[vc["health_trit"]]
            out = [
                f"=== Stafford Beer VSM Operations Status [{health_name}] ===",
                f"  C-Complex : {cc['cpus']} CPUs @ {cc['clock_ns']} ns | Global cycles={cc['global_cycles']:,} | Insns={cc['total_instructions']:,}",
                f"  T-Fabric  : {tf['active_tiles']} active + {tf['hot_spares_remaining']} hot spares | Ops={tf['total_ops']:,} | ABFT={tf['abft_checks']:,}",
                f"  T-DRAM    : {mem['active_banks']} active + {mem['spare_banks_remaining']} spare banks | Conflicts={mem['crossbar_conflicts']} | Bank cutovers={mem['bank_cutovers']}",
                f"  Algedonic : Alarms={vc['algedonic_alarms']} (unacked={self.machine.vcomplex.unacked_alarms}) | Cutovers={len(vc['cutovers'])}",
            ]
            if vc["gallery_lamps"]:
                out.append("  Lamps     : " + " | ".join(vc["gallery_lamps"][-3:]))
            return out

        if cmd == "TILES":
            rows = ["=== T-Fabric Physical Tile Ring (27 Active + 3 Hot Spares) ==="]
            for start in range(0, 30, 6):
                chunk = []
                for p in range(start, start + 6):
                    st = self.machine.vcomplex.tiles[p]
                    tag = f"T{st.logical_id:02d}" if st.logical_id is not None else ("SPR" if st.role == "HOT_SPARE" else "FLT")
                    chunk.append(f"P{p:02d}:{tag}")
                rows.append("  " + "  ".join(chunk))
            return rows

        if cmd == "MEMORY":
            mem = self.machine.mem
            return [
                "=== Tier 3a Hierarchical Store ===",
                f"  T-DRAM      : 43,046,721 words (3^16) | 81 active + {len(mem.spare_banks)} hot spares",
                f"  Ferrite Core: 177,147 words (3^11) | C-EXEC booted={self.cexec.booted}",
                f"  Counters    : reads={mem.reads:,} writes={mem.writes:,} crossbar_conflicts={mem.crossbar_conflicts:,} bank_cutovers={mem.bank_cutovers}",
            ]

        if cmd == "CHANNELS":
            summary = self.machine.io_channels.telemetry_summary()
            out = ["=== Tier 3b Asynchronous Data Channels (1..8) ==="]
            for ch, st in summary["channels"].items():
                out.append(f"  CH{ch}: status={st['last_status']:+d} ops={st['total_ops']} words={st['total_words']:,}")
            out.append(
                f"  Media: Disc(R={summary['disc_reads']},W={summary['disc_writes']}) | "
                f"Tape(R={summary['tape_words_read']},W={summary['tape_words_written']}) | "
                f"Silo(carts={summary['silo_cartridges']},denials={summary['silo_provenance_denials']}) | "
                f"SETI(w={summary['seti_words_streamed']}) | L8Chute(rec={summary['l8_records_ingested']})"
            )
            return out

        if cmd == "MMIO":
            sys_id = self.machine.mem.read(MMIO_SYS_ID, supervisor=True)[0]
            vsum = self.machine.mem.read(MMIO_VCOMPLEX_SUMMARY, supervisor=True)[0]
            return [
                f"MMIO_SYS_ID (+100,000,000)         = {sys_id}",
                f"MMIO_VCOMPLEX_SUMMARY (+100,000,300)= {vsum}",
            ]

        if cmd == "ACK":
            prev = self.machine.vcomplex.acknowledge_alarms()
            return [f"Acknowledged {prev} Algedonic alarm(s). Health trit = {self.machine.vcomplex.system_health_trit():+d}"]

        if cmd == "FAULT":
            if not self.current_user.can_fault_inject:
                raise VOSError(f"WEAR DENIAL: {self.current_user.username} cannot inject hardware faults")
            if len(args) < 2 or args[0].upper() != "TILE":
                raise VOSError("Usage: FAULT TILE <0..26> [reason]")
            lt = int(args[1])
            reason = args[2] if len(args) > 2 else "MANUAL_VOS_DRILL"
            ev = self.machine.vcomplex.trigger_tile_cutover(
                logical_tile=lt, fault_reason=reason, timestamp_ns=self.machine.global_cycles * 30
            )
            if ev.kind == "SPARES_EXHAUSTED":
                return [f"CRITICAL: Logical T{lt} faulted — ALL 3 SPARES EXHAUSTED (Trap 12)"]
            return [
                f"Ashby Cutover #{ev.event_id}: Logical T{lt} (Bay {ev.old_physical_id}) -> "
                f"Hot Spare S{ev.new_physical_id} in {ev.hw_cutover_us:.1f} us ({reason})"
            ]

        if cmd == "CUTOVER":
            if not self.current_user.can_fault_inject:
                raise VOSError(f"WEAR DENIAL: {self.current_user.username} cannot trigger bank cutover")
            if len(args) < 2 or args[0].upper() != "BANK":
                raise VOSError("Usage: CUTOVER BANK <0..80>")
            lbank = int(args[1])
            self.machine.mem.write(MMIO_BANK_CUTOVER_CMD, lbank, supervisor=True)
            return [f"Ashby Bank Cutover triggered via MMIO for logical bank {lbank} (total cutovers={self.machine.mem.bank_cutovers})"]

        if cmd == "LS":
            vol = args[0].upper() if args else "ALL"
            out: List[str] = []
            if vol in ("ALL", "DISC:", "DISC"):
                out.append("DISC: (Online Ternary Disc Array — Channel 1)")
                for e in sorted(self.disc_catalog.values(), key=lambda x: x.name):
                    out.append(f"  DISC:{e.name:20s}  {e.word_count:4d}w  owner={e.owner:10s}  comp={e.compartment}")
            if vol in ("ALL", "SILO:", "SILO"):
                out.append("SILO: (Alexandrian Archival Silo — Channel 3)")
                for e in sorted(self.silo_catalog.values(), key=lambda x: x.name):
                    lock = f"WEAR-{e.compartment}" if e.compartment else "OPEN"
                    out.append(f"  SILO:{e.name:20s}  cart={e.cartridge_id:3d}  {e.word_count:4d}w  [{lock}]")
            if vol in ("PROGRAMS:", "PROGRAMS"):
                out.append(f"PROGRAMS: ({self.programs_dir})")
                if self.programs_dir.exists():
                    for p in sorted(self.programs_dir.iterdir()):
                        if p.is_file():
                            out.append(f"  PROGRAMS:{p.name}")
            return out

        if cmd == "CAT":
            if not args:
                raise VOSError("Usage: CAT <DISC:name | SILO:name>")
            target = args[0]
            if ":" not in target:
                return [self.read_disc_file(target)]
            vol, fname = target.split(":", 1)
            vol_u = vol.upper()
            if vol_u == "DISC":
                return [self.read_disc_file(fname)]
            if vol_u == "SILO":
                return [self.read_silo_file(fname)]
            if vol_u == "PROGRAMS":
                p = self.programs_dir / fname
                if not p.exists():
                    raise VOSError(f"PROGRAMS:{fname} not found")
                return p.read_text(encoding="utf-8").splitlines()
            raise VOSError(f"Unknown volume {vol!r}")

        if cmd == "WRITE":
            if len(args) < 2:
                raise VOSError("Usage: WRITE <DISC:name> <text...>")
            target = args[0]
            fname = target.split(":", 1)[1] if ":" in target else target
            text = " ".join(args[1:])
            entry = self.write_disc_file(fname, text, compartment=self.active_compartment)
            return [f"Wrote DISC:{entry.name} ({entry.word_count} T36 words via CHIO #1, sector {entry.sector})"]

        if cmd == "STAGE":
            if len(args) < 2:
                raise VOSError("Usage: STAGE <SILO:src> <DISC:dst>")
            src_name = args[0].split(":", 1)[1] if ":" in args[0] else args[0]
            dst_name = args[1].split(":", 1)[1] if ":" in args[1] else args[1]
            content = self.read_silo_file(src_name)
            entry = self.write_disc_file(dst_name, content, compartment=self.silo_catalog[src_name.upper()].compartment)
            return [f"Staged SILO:{src_name.upper()} -> DISC:{entry.name} ({entry.word_count} words)"]

        if cmd == "TAPE":
            if not args:
                raise VOSError("Usage: TAPE <WRITE text | READ | READBACK | REWIND>")
            sub = args[0].upper()
            if sub == "WRITE":
                text = " ".join(args[1:])
                words = self._encode_text_words(text)
                for i, w in enumerate(words):
                    self.machine.mem.write(_VOS_IO_BUF_ADDR + i, w, supervisor=True)
                st, n = self._run_chio(2, ChannelControlBlock(
                    device_type=DEV_TAPE, unit=0, op=OP_WRITE, mem_addr=_VOS_IO_BUF_ADDR, word_count=len(words)
                ))
                return [f"CHIO #2 TAPE WRITE: status={st:+d}, {n} words ({n * 4} nine-track frames)"]
            if sub == "REWIND":
                st, _ = self._run_chio(2, ChannelControlBlock(device_type=DEV_TAPE, unit=0, op=OP_REWIND))
                return [f"CHIO #2 TAPE REWIND: status={st:+d}, head=0"]
            if sub in ("READ", "READBACK"):
                op = OP_READ if sub == "READ" else OP_READ_BACKWARD
                st, n = self._run_chio(2, ChannelControlBlock(
                    device_type=DEV_TAPE, unit=0, op=op, mem_addr=_VOS_IO_BUF_ADDR, word_count=256
                ))
                if st != CH_OK:
                    raise VOSError(f"CHIO #2 TAPE {sub} failed (status={st})")
                words = [self.machine.mem.read(_VOS_IO_BUF_ADDR + i, supervisor=True)[0] for i in range(n)]
                if sub == "READBACK":
                    # Show raw reversed word count + decoded reversed recovery
                    return [f"CHIO #2 TAPE READ_BACKWARD ({n} words reversed): {self._decode_text_words(list(reversed(words)))}"]
                return [f"CHIO #2 TAPE READ ({n} words): {self._decode_text_words(words)}"]
            raise VOSError(f"Unknown TAPE subcommand {sub!r}")

        if cmd == "CHUTE":
            if not args:
                raise VOSError("Usage: CHUTE <FEED text | INGEST DISC:name>")
            sub = args[0].upper()
            if sub == "FEED":
                text = " ".join(args[1:])
                n = self.machine.io_channels.l8_chute.feed_text_record(text, provenance_tag=self.active_compartment)
                return [f"Level -8 Chute queued record ({n} T36 words, pending={self.machine.io_channels.l8_chute.pending_records})"]
            if sub == "INGEST":
                if len(args) < 2:
                    raise VOSError("Usage: CHUTE INGEST <DISC:name>")
                dst = args[1].split(":", 1)[1] if ":" in args[1] else args[1]
                st, n = self._run_chio(4, ChannelControlBlock(
                    device_type=DEV_L8_CHUTE, unit=0, op=OP_READ, mem_addr=_VOS_IO_BUF_ADDR, word_count=256
                ))
                if st != CH_OK:
                    raise VOSError(f"Level -8 Chute empty or error (status={st})")
                words = [self.machine.mem.read(_VOS_IO_BUF_ADDR + i, supervisor=True)[0] for i in range(n)]
                text = Level8DigitisationChute.decode_words_to_text(words)
                tag = self.machine.mem.read(_VOS_CCB_ADDR + 3, supervisor=True)[0]
                entry = self.write_disc_file(dst, text, compartment=0)
                return [f"Ingested Level -8 record (provenance={tag}) -> DISC:{entry.name}: {text}"]
            raise VOSError(f"Unknown CHUTE subcommand {sub!r}")

        if cmd == "SUBMIT":
            return self._cmd_submit(args)

        if cmd in ("PS", "JOBS"):
            if not self.cexec.jobs:
                return ["(No C-EXEC jobs submitted)"]
            rows = ["ID   NAME                 OWNER       STATE      CPU   CYCLES  PREEMPT  RESULT"]
            for j in self.cexec.jobs.values():
                cpu_s = f"C{j.assigned_cpu}" if j.assigned_cpu is not None else "-"
                if j.state == "EXITED":
                    res_s = f"exit={j.exit_code}"
                elif j.state == "ABORTED":
                    tname = TRAP_NAMES.get(j.abort_trap or 0, str(j.abort_trap))
                    res_s = f"TRAP {j.abort_trap} ({tname})"
                else:
                    res_s = "-"
                rows.append(
                    f"{j.job_id:<4d} {j.name[:20]:20s} {j.owner[:10]:10s}  "
                    f"{j.state:9s}  {cpu_s:4s}  {j.cycles_used:6d}  {j.preemptions:7d}  {res_s}"
                )
            return rows

        if cmd in ("DISPATCH", "WAIT"):
            max_cpus = int(args[0]) if args else 8
            self.cexec.run_scheduler(max_cpus=max_cpus)
            out = [f"C-EXEC scheduler completed (max_cpus={max_cpus}):"]
            for j in self.cexec.jobs.values():
                detail = f"exit={j.exit_code}" if j.state == "EXITED" else (
                    f"ABORT TRAP {j.abort_trap} ({TRAP_NAMES.get(j.abort_trap or 0, '?')})" if j.state == "ABORTED" else j.state
                )
                out.append(f"  Job #{j.job_id} [{j.name}] -> {j.state} ({detail}, {j.cycles_used} cyc, {j.preemptions} preempts)")
                for ln in j.output_lines:
                    out.append(f"    | {ln}")
            return out

        if cmd == "CEXEC":
            st = self.cexec.kernel_stats()
            return [
                "=== Tier 1 C-EXEC Resident Ferrite Vault Telemetry ===",
                f"  Booted={bool(st['booted'])} | Kernel={st['kernel_words']} words | Magic={st['boot_magic']}",
                f"  Traps={st['total_traps']} | SVCs={st['total_svcs']} | Timer Preemptions={st['total_preemptions']} | User Aborts={st['total_aborts']}",
            ]

        if cmd == "RUN":
            if not args:
                raise VOSError("Usage: RUN <path_or_PROGRAMS:file>")
            return self._cmd_run_language(args[0])

        if cmd == "CONVERSE":
            if self._converse_session is None:
                from .converse import ConverseSession
                self._converse_session = ConverseSession(machine=self.machine)
            return self._converse_session.execute_line(raw_rest)

        if cmd == "APL":
            if self._apl_session is None:
                from .apl36 import APL36Session
                self._apl_session = APL36Session(machine=self.machine)
            lines, _ = self._apl_session.eval_script(raw_rest)
            return lines

        if cmd == "FORTH":
            from .forth3 import run_forth
            res = run_forth(raw_rest, machine=self.machine)
            out = list(res.output)
            if res.stack:
                out.append("Stack: " + str(res.stack))
            return out

        if cmd == "DEMO":
            target = args[0].lower() if args else "seti"
            if target == "seti":
                from workloads.seti import run_seti_demo
                rep = run_seti_demo(self.machine)
                return [f"SETI {r.label}: {r.final_decibans:+.2f} dB -> {r.verdict_name}" for r in rep.results]
            if target == "saew":
                from workloads.saew import run_saew_demo
                rep = run_saew_demo(self.machine)
                return [
                    f"SAEW loss {rep.initial_loss:.3f} -> {rep.final_loss:.3f} | "
                    f"Cutovers survived={rep.cutovers_survived} | Zero-trit saved={rep.avg_zero_trit_power_saving_pct:.1f}%"
                ]
            if target == "clio":
                from workloads.cliodynamics import run_cliodynamics_demo
                rep = run_cliodynamics_demo(self.machine)
                return [f"CLIO {tr.policy_name}: Pop={tr.final_population:.2f} -> {tr.stability_label}" for tr in rep.trajectories]
            raise VOSError(f"Unknown DEMO target {target!r} (choose saew, seti, clio)")

        if cmd in ("QUIT", "EXIT", "LOGOUT"):
            return ["LOGOUT"]

        raise VOSError(f"Unknown V-OS command {cmd!r} (type HELP)")

    def _cmd_submit(self, args: List[str]) -> List[str]:
        if len(args) < 2:
            raise VOSError("Usage: SUBMIT <job_name> <asm_source_or_DISC:file> [--cpu N] [--quantum Q]")
        name = args[0]
        src_arg = args[1]
        cpu: Optional[int] = None
        quantum = 20
        i = 2
        while i < len(args):
            if args[i] == "--cpu" and i + 1 < len(args):
                cpu = int(args[i + 1])
                i += 2
            elif args[i] == "--quantum" and i + 1 < len(args):
                quantum = int(args[i + 1])
                i += 2
            else:
                i += 1

        if src_arg.upper().startswith("DISC:"):
            asm_src = self.read_disc_file(src_arg.split(":", 1)[1])
        else:
            asm_src = src_arg.replace("\\n", "\n")

        job = self.cexec.submit_assembly(
            name=name,
            asm_source=asm_src,
            owner=self.current_user.username,
            quantum=quantum,
            preferred_cpu=cpu,
        )
        return [
            f"Submitted C-EXEC Job #{job.job_id} [{job.name}] "
            f"(BASE={job.base}, LIMIT={job.limit}, quantum={job.quantum})"
        ]

    def _cmd_run_language(self, target: str) -> List[str]:
        if target.upper().startswith("PROGRAMS:"):
            path = self.programs_dir / target.split(":", 1)[1]
        else:
            p = Path(target)
            path = p if p.exists() else (self.programs_dir / target)
        if not path.exists():
            raise VOSError(f"Program file not found: {target}")
        src = path.read_text(encoding="utf-8")
        ext = path.suffix.lower()

        if ext == ".sal3":
            from .sal3 import run_sal3
            r = run_sal3(src)
            return [f"[SAL-3 {path.name}] return={r.return_value} (C={r.return_trit:+d}) | " + " | ".join(r.stdout)]
        if ext == ".f70":
            from .tritran70v import run_tritran
            r = run_tritran(src)
            return [f"[TRI-TRAN 70V {path.name}] STOP={r.stop_code}"] + r.lines
        if ext == ".lsp":
            from .trilisp36 import run_trilisp
            r = run_trilisp(src)
            return [f"[TRILISP-36 {path.name}] => {r.result_str}"] + r.printed
        if ext == ".cv":
            from .converse import run_converse_script
            out, _ = run_converse_script(src)
            return [f"[CONVERSE {path.name}]"] + out
        if ext == ".dyn":
            from .dynamox import run_dynamo
            r = run_dynamo(src)
            return [f"[DYNAMO-X {path.name}] {r.compile_result.model.title}"] + r.rendered_tables
        if ext == ".apl":
            from .apl36 import run_apl_script
            out, _ = run_apl_script(src)
            return [f"[APL\\36 {path.name}]"] + out
        if ext == ".ref":
            from .refal36 import run_refal
            r = run_refal(src)
            return [f"[REFAL-36 {path.name}] => {r.result_str}"] + r.output
        if ext == ".fth":
            from .forth3 import run_forth
            r = run_forth(src)
            return [f"[FORTH-3 {path.name}]"] + r.output
        if ext == ".i3":
            from .tri_intercal import run_tri_intercal
            r = run_tri_intercal(src)
            return [f"[TRI-INTERCAL {path.name} — BROOKS AUDIT!]"] + r.output
        raise VOSError(f"Unsupported program extension {ext!r}")


def run_vos_script(
    script: str,
    user: str = "BROOKS",
    machine: Optional[UE1000Machine] = None,
) -> Tuple[List[str], VOSSession]:
    """Execute a multi-line V-OS command script and return (output_lines, session)."""
    sess = VOSSession(machine=machine, user=user)
    output: List[str] = []
    for raw in script.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(";"):
            continue
        output.append(f"{sess.prompt}{stripped}")
        try:
            res = sess.execute_line(stripped)
        except VOSError as exc:
            output.append(f"  [ERROR] {exc}")
            continue
        if res == ["LOGOUT"]:
            output.append("  Session closed.")
            break
        for ln in res:
            output.append("  " + ln)
    return output, sess


def run_vos_repl(user: str = "BROOKS") -> None:
    """Start an interactive OBB Level -9 V-OS terminal REPL."""
    sess = VOSSession(user=user)
    print("=========================================================================")
    print("  DUWAMISH-I (UE-1000) — V-OS Rev. 1.5 Supervisory Console (Level -9)")
    print("  WEAR Charter · Special Systems Section, Universal Entropics")
    print(f"  C-EXEC booted in Non-Volatile Ferrite Vault ({sess.cexec.kernel_stats()['kernel_words']} words)")
    print("  Type HELP for commands, QUIT to logout.")
    print("=========================================================================")
    while True:
        try:
            raw = input(sess.prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nLogout.")
            break
        try:
            res = sess.execute_line(raw)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            continue
        if res == ["LOGOUT"]:
            print("Logout.")
            break
        for ln in res:
            print("  " + ln)
