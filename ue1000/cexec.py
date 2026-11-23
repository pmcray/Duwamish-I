"""Tier 1 Resident Bare-Metal Executive (`C-EXEC`) in Non-Volatile Ferrite Core.

Conforms strictly to Principles of Operation (Rev. 1.5), §3.1 & §4.4:
  - Assembled as authentic native TRIAD-36 code resident in the Non-Volatile Dual-Core
    Ferrite Vault (`-177,147 .. -1`) and backed by Supervisor T-DRAM (`-1,594,323 .. -177,148`).
  - Installs per-CPU Trap Entry Vectors for all 8 C-Complex CPUs (`-27 * cpuid - 27`).
  - Executes Cold-Boot IPL verifying MMIO_SYS_ID (100015) and stamping C_EXEC_MAGIC.
  - Native TRIAD-36 Trap Dispatcher handles:
      * SVC #1 (SYS_EXIT)    : Record user exit status in CPU mailbox & HLT cleanly
      * SVC #2 (SYS_PUTINT)  : Append T36 integer to Supervisor Console Ring & RTI
      * SVC #3 (SYS_PUTCHAR) : Append OBB-729 tryte to Supervisor Console Ring & RTI
      * SVC #4 (SYS_CLOCK)   : Read ALGD #0 (30 ns TIM) into user S1 & RTI
      * SVC #5 (SYS_HEALTH)  : Read ALGD #3 (Beer VSM health trit) into user S1 & RTI
      * SVC #7 (SYS_YIELD)   : Cooperative yield -> save context for round-robin switch
      * TRAP_TIMER (10)      : Watchdog quantum preemption -> lossless context switch
      * Fault Traps (1..14)  : Hardware-isolated job abort without crashing Supervisor!
  - Hardware BASE/LIMIT arena allocator giving every User-mode job an isolated T-DRAM
    partition inside `+1 .. +41,629,544` so concurrent jobs on CPUs 0..7 cannot collide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .cpu import C_VEC_LEN, PSW
from .machine import MMIO_SYS_ID, UE1000Machine
from .memory import USER_TDRAM_MAX
from .ternary import WORD_TRITS, extract_field, obb729_to_char
from .triad36 import AssembledProgram, assemble

# Non-Volatile Ferrite Vault Kernel Addresses (-1 .. -177,147)
FERRITE_STUB_BASE = -500      # 8 per-CPU stubs, 16 words each (-500 .. -627)
FERRITE_IPL_ADDR = -1000      # Cold-Boot IPL routine
FERRITE_COMMON_ADDR = -1100   # Shared native TRIAD-36 Trap & SVC Dispatcher

# Supervisor T-DRAM Control Block Addresses (-1,594,323 .. -177,148)
SUP_BOOT_MAGIC_ADDR = -200_000
SUP_TRAP_COUNT_ADDR = -200_001
SUP_SVC_COUNT_ADDR = -200_002
SUP_PREEMPT_COUNT_ADDR = -200_003
SUP_ABORT_COUNT_ADDR = -200_004
SUP_MAILBOX_BASE = -210_000   # CPU k mailbox at -210_000 - 16*k
SUP_CONSOLE_COUNT_ADDR = -220_000
SUP_CONSOLE_RING_BASE = -220_001  # 3 words/entry growing downward: kind, val, mailbox_addr

C_EXEC_MAGIC = 36_1973

# Mailbox Status Codes written by native C-EXEC Trap Dispatcher (0(A6))
MB_IDLE = 0
MB_EXITED = 1
MB_PREEMPTED = 2
MB_YIELDED = 3
MB_ABORTED = -1

# Standard C-EXEC SVC Numbers
SYS_EXIT = 1
SYS_PUTINT = 2
SYS_PUTCHAR = 3
SYS_CLOCK = 4
SYS_HEALTH = 5
SYS_YIELD = 7

# Trap code human names (PoO §4.4)
TRAP_NAMES: Dict[int, str] = {
    1: "ILLOP",
    2: "PROT",
    3: "DIVZ",
    4: "PRIV",
    5: "OVF",
    6: "SVC",
    7: "FPOVF",
    8: "FPUND",
    10: "TIMER",
    12: "TFAULT",
    14: "MEMDED",
    18: "MCHK",
}


def build_cexec_ferrite_assembly() -> str:
    """Generate the complete native TRIAD-36 assembly image for resident C-EXEC."""
    lines: List[str] = [
        "; ==========================================================================",
        "; Duwamish-I (UE-1000) C-EXEC Resident Ferrite Vault Executive (Rev. 1.5)",
        "; ==========================================================================",
    ]

    # Per-CPU Trap Entry Stubs (CPU 0..7)
    for k in range(8):
        stub_addr = FERRITE_STUB_BASE - 16 * k
        slot = -27 * k
        mb = SUP_MAILBOX_BASE - 16 * k
        lines.extend([
            f"        .ORG {stub_addr}",
            f"STUB_{k}: STA  A7, {slot - 12}(A0)",
            f"        STA  A6, {slot - 13}(A0)",
            f"        ST   S5, {slot - 14}(A0)",
            f"        ST   S6, {slot - 15}(A0)",
            f"        ST   S7, {slot - 16}(A0)",
            f"        STA  A5, {slot - 17}(A0)",
            f"        LEA  A7, {slot}(A0)",
            f"        LEA  A6, {mb}(A0)",
            f"        JMP  CEXEC_COMMON(A0)",
        ])

    # Cold-Boot IPL routine at FERRITE_IPL_ADDR (-1000)
    lines.extend([
        f"        .ORG {FERRITE_IPL_ADDR}",
        "CEXEC_IPL:",
        f"        LDI  S1, #{MMIO_SYS_ID}",
        "        S2A  A1, S1               | NOP",
        "        LD   S2, 0(A1)",
        "        LDI  S3, #100015",
        "        CMP  S2, S3               | BZ IPL_OK",
        "        HLT",
        "IPL_OK:",
        f"        LDI  S4, #{C_EXEC_MAGIC}",
        f"        ST   S4, {SUP_BOOT_MAGIC_ADDR}(A0)",
        f"        ST   S0, {SUP_TRAP_COUNT_ADDR}(A0)",
        f"        ST   S0, {SUP_SVC_COUNT_ADDR}(A0)",
        f"        ST   S0, {SUP_PREEMPT_COUNT_ADDR}(A0)",
        f"        ST   S0, {SUP_ABORT_COUNT_ADDR}(A0)",
        f"        ST   S0, {SUP_CONSOLE_COUNT_ADDR}(A0)",
        "        HLT",
    ])

    # Shared Trap & SVC Dispatcher at FERRITE_COMMON_ADDR (-1100)
    lines.extend([
        f"        .ORG {FERRITE_COMMON_ADDR}",
        "CEXEC_COMMON:",
        # Increment global trap counter
        f"        LD   S5, {SUP_TRAP_COUNT_ADDR}(A0)",
        "        ADD  S5, S5, #1           | NOP",
        f"        ST   S5, {SUP_TRAP_COUNT_ADDR}(A0)",
        # Increment per-CPU trap counter M[4(A6)]
        "        LD   S5, 4(A6)",
        "        ADD  S5, S5, #1           | NOP",
        "        ST   S5, 4(A6)",
        # Load trap code (slot-3) and trap arg (slot-4), log in mailbox[2..3]
        "        LD   S5, -3(A7)",
        "        LD   S6, -4(A7)",
        "        ST   S5, 2(A6)",
        "        ST   S6, 3(A6)",
        # Dispatch: Trap 6 = SVC, Trap 10 = TIMER, else Fault Abort
        "        CMP  S5, #6               | BZ DO_SVC",
        "        CMP  S5, #10              | BZ DO_TIMER",
        "        BR   DO_FAULT             | NOP",
        "",
        "DO_SVC:",
        f"        LD   S5, {SUP_SVC_COUNT_ADDR}(A0)",
        "        ADD  S5, S5, #1           | NOP",
        f"        ST   S5, {SUP_SVC_COUNT_ADDR}(A0)",
        "        CMP  S6, #1               | BZ SVC_EXIT",
        "        CMP  S6, #2               | BZ SVC_PUTINT",
        "        CMP  S6, #3               | BZ SVC_PUTCHAR",
        "        CMP  S6, #4               | BZ SVC_CLOCK",
        "        CMP  S6, #5               | BZ SVC_HEALTH",
        "        CMP  S6, #7               | BZ SVC_YIELD",
        "        BR   DO_FAULT             | NOP",
        "",
        "SVC_EXIT:",
        f"        ADD  S5, S0, #{MB_EXITED} | NOP",
        "        ST   S5, 0(A6)",
        "        ST   S1, 1(A6)",
        "        BR   RESTORE_HLT          | NOP",
        "",
        "SVC_PUTINT:",
        f"        LD   S5, {SUP_CONSOLE_COUNT_ADDR}(A0)",
        "        ADD  S7, S5, S5           | NOP",
        "        ADD  S7, S7, S5           | NOP",
        "        NEG  S7, S7               | S2A A5, S7",
        "        ADD  S7, S0, #+1          | NOP",
        f"        ST   S7, {SUP_CONSOLE_RING_BASE}(A5)",
        f"        ST   S1, {SUP_CONSOLE_RING_BASE - 1}(A5)",
        f"        STA  A6, {SUP_CONSOLE_RING_BASE - 2}(A5)",
        "        ADD  S5, S5, #1           | NOP",
        f"        ST   S5, {SUP_CONSOLE_COUNT_ADDR}(A0)",
        "        BR   RESTORE_RTI          | NOP",
        "",
        "SVC_PUTCHAR:",
        f"        LD   S5, {SUP_CONSOLE_COUNT_ADDR}(A0)",
        "        ADD  S7, S5, S5           | NOP",
        "        ADD  S7, S7, S5           | NOP",
        "        NEG  S7, S7               | S2A A5, S7",
        "        ADD  S7, S0, #-1          | NOP",
        f"        ST   S7, {SUP_CONSOLE_RING_BASE}(A5)",
        f"        ST   S1, {SUP_CONSOLE_RING_BASE - 1}(A5)",
        f"        STA  A6, {SUP_CONSOLE_RING_BASE - 2}(A5)",
        "        ADD  S5, S5, #1           | NOP",
        f"        ST   S5, {SUP_CONSOLE_COUNT_ADDR}(A0)",
        "        BR   RESTORE_RTI          | NOP",
        "",
        "SVC_CLOCK:",
        "        ALGD S1, #0",
        "        BR   RESTORE_RTI          | NOP",
        "",
        "SVC_HEALTH:",
        "        ALGD S1, #3",
        "        BR   RESTORE_RTI          | NOP",
        "",
        "SVC_YIELD:",
        f"        ADD  S5, S0, #{MB_YIELDED} | NOP",
        "        ST   S5, 0(A6)",
        "        BR   RESTORE_HLT          | NOP",
        "",
        "DO_TIMER:",
        f"        LD   S5, {SUP_PREEMPT_COUNT_ADDR}(A0)",
        "        ADD  S5, S5, #1           | NOP",
        f"        ST   S5, {SUP_PREEMPT_COUNT_ADDR}(A0)",
        f"        ADD  S5, S0, #{MB_PREEMPTED} | NOP",
        "        ST   S5, 0(A6)",
        "        BR   RESTORE_HLT          | NOP",
        "",
        "DO_FAULT:",
        f"        LD   S5, {SUP_ABORT_COUNT_ADDR}(A0)",
        "        ADD  S5, S5, #1           | NOP",
        f"        ST   S5, {SUP_ABORT_COUNT_ADDR}(A0)",
        f"        ADD  S5, S0, #{MB_ABORTED} | NOP",
        "        ST   S5, 0(A6)",
        "        BR   RESTORE_HLT          | NOP",
        "",
        "RESTORE_RTI:",
        "        LDA  A5, -17(A7)",
        "        LD   S7, -16(A7)",
        "        LD   S6, -15(A7)",
        "        LD   S5, -14(A7)",
        "        LDA  A6, -13(A7)",
        "        LDA  A7, -12(A7)",
        "        RTI",
        "",
        "RESTORE_HLT:",
        "        LDA  A5, -17(A7)",
        "        LD   S7, -16(A7)",
        "        LD   S6, -15(A7)",
        "        LD   S5, -14(A7)",
        "        LDA  A6, -13(A7)",
        "        LDA  A7, -12(A7)",
        "        HLT",
    ])
    return "\n".join(lines)


@dataclass
class JobContext:
    """Saved architectural state for lossless preemptive context switching."""
    s: List[int] = field(default_factory=lambda: [0] * 9)
    a: List[int] = field(default_factory=lambda: [0] * 9)
    v: List[List[int]] = field(default_factory=lambda: [[0] * C_VEC_LEN for _ in range(9)])
    pc: int = 100
    c: int = 0
    vl: int = C_VEC_LEN
    psw: PSW = field(default_factory=lambda: PSW(mode=1, ie=1))


@dataclass
class CExecJob:
    """User-mode batch job managed by C-EXEC inside an isolated [BASE, BASE+LIMIT) arena."""
    job_id: int
    name: str
    owner: str
    base: int
    limit: int
    quantum: int
    preferred_cpu: Optional[int] = None
    state: str = "QUEUED"  # QUEUED | RUNNING | EXITED | ABORTED
    assigned_cpu: Optional[int] = None
    exit_code: Optional[int] = None
    abort_trap: Optional[int] = None
    abort_arg: Optional[int] = None
    preemptions: int = 0
    yields: int = 0
    cycles_used: int = 0
    instructions_used: int = 0
    output_lines: List[str] = field(default_factory=list)
    _char_buf: str = ""
    context: JobContext = field(default_factory=JobContext)


class CExecHypervisor:
    """Bare-metal Tier 1 Executive residing in the Non-Volatile Ferrite Vault."""

    def __init__(self, machine: UE1000Machine, arena_start: int = 1_000_000, default_arena_size: int = 65_536) -> None:
        self.machine = machine
        self.arena_cursor = arena_start
        self.default_arena_size = default_arena_size
        self.booted: bool = False
        self.kernel_prog: Optional[AssembledProgram] = None

        self.jobs: Dict[int, CExecJob] = {}
        self._next_job_id: int = 1
        self._ready_queue: List[int] = []
        self._running_on_cpu: Dict[int, int] = {}  # cpu_id -> job_id
        self._console_read_idx: int = 0

    def boot(self) -> Dict[str, int]:
        """Cold-Boot IPL: assemble C-EXEC into Ferrite Vault, install 8 trap vectors, run IPL."""
        src = build_cexec_ferrite_assembly()
        prog = assemble(src, origin=FERRITE_IPL_ADDR)
        self.kernel_prog = prog
        for addr, w in prog.words.items():
            self.machine.mem.write(addr, w, supervisor=True)

        # Install all 8 per-CPU trap entry vectors in Non-Volatile Ferrite Core
        for k in range(8):
            slot = -27 * k
            stub_addr = FERRITE_STUB_BASE - 16 * k
            self.machine.mem.write(slot - 27, stub_addr, supervisor=True)
            mb = SUP_MAILBOX_BASE - 16 * k
            for off in range(6):
                self.machine.mem.write(mb + off, 0, supervisor=True)

        # Execute Cold-Boot IPL natively on CPU 0 in Supervisor mode!
        cpu0 = self.machine.cpu0
        cpu0.psw = PSW(mode=-1, cpuid=0)
        cpu0.pc = FERRITE_IPL_ADDR
        cpu0.halted = False
        cpu0.run(max_steps=200)

        magic = self.machine.mem.read(SUP_BOOT_MAGIC_ADDR, supervisor=True)[0]
        if magic != C_EXEC_MAGIC:
            raise RuntimeError(f"C-EXEC Cold-Boot IPL failed (magic={magic}, expected {C_EXEC_MAGIC})")
        self.booted = True
        self._console_read_idx = 0
        return self.kernel_stats()

    def kernel_stats(self) -> Dict[str, int]:
        mem = self.machine.mem
        return {
            "booted": int(self.booted),
            "kernel_words": len(self.kernel_prog.words) if self.kernel_prog else 0,
            "boot_magic": mem.read(SUP_BOOT_MAGIC_ADDR, supervisor=True)[0],
            "total_traps": mem.read(SUP_TRAP_COUNT_ADDR, supervisor=True)[0],
            "total_svcs": mem.read(SUP_SVC_COUNT_ADDR, supervisor=True)[0],
            "total_preemptions": mem.read(SUP_PREEMPT_COUNT_ADDR, supervisor=True)[0],
            "total_aborts": mem.read(SUP_ABORT_COUNT_ADDR, supervisor=True)[0],
            "console_events": mem.read(SUP_CONSOLE_COUNT_ADDR, supervisor=True)[0],
        }

    def allocate_arena(self, size: Optional[int] = None) -> Tuple[int, int]:
        sz = size or self.default_arena_size
        base = self.arena_cursor
        if base + sz > USER_TDRAM_MAX:
            raise RuntimeError("User T-DRAM arena space exhausted")
        self.arena_cursor += sz + 1024  # 1024-word guard band between partitions
        return base, sz

    def submit_assembly(
        self,
        name: str,
        asm_source: str,
        owner: str = "BROOKS",
        origin: int = 100,
        arena_size: Optional[int] = None,
        quantum: int = 20,
        preferred_cpu: Optional[int] = None,
    ) -> CExecJob:
        """Assemble a User-mode TRIAD-36 program into an isolated [BASE, BASE+LIMIT) arena."""
        if not self.booted:
            self.boot()
        prog = assemble(asm_source, origin=origin)
        return self.submit_program(
            name=name,
            words=prog.words,
            entry_pc=prog.origin,
            owner=owner,
            arena_size=arena_size,
            quantum=quantum,
            preferred_cpu=preferred_cpu,
        )

    def submit_program(
        self,
        name: str,
        words: Dict[int, int],
        entry_pc: int = 100,
        owner: str = "BROOKS",
        arena_size: Optional[int] = None,
        quantum: int = 20,
        preferred_cpu: Optional[int] = None,
    ) -> CExecJob:
        """Load pre-assembled words into a hardware-isolated User partition and enqueue."""
        if not self.booted:
            self.boot()
        max_addr = max(words.keys(), default=entry_pc) + 4096
        limit = max(arena_size or self.default_arena_size, max_addr + 1024)
        base, limit = self.allocate_arena(limit)

        for logical_addr, word in words.items():
            if logical_addr <= 0 or logical_addr > limit:
                raise ValueError(f"Program word at logical {logical_addr} outside arena [1..{limit}]")
            self.machine.mem.write(base + logical_addr, word, supervisor=True)

        jid = self._next_job_id
        self._next_job_id += 1
        ctx = JobContext(pc=entry_pc, psw=PSW(mode=1, ie=1))
        ctx.s[8] = limit - 64  # User stack pointer near top of isolated arena
        job = CExecJob(
            job_id=jid,
            name=name,
            owner=owner,
            base=base,
            limit=limit,
            quantum=quantum,
            preferred_cpu=preferred_cpu,
            state="QUEUED",
            context=ctx,
        )
        self.jobs[jid] = job
        self._ready_queue.append(jid)
        return job

    def _load_job_onto_cpu(self, job: CExecJob, cpu_id: int) -> None:
        cpu = self.machine.cpus[cpu_id]
        ctx = job.context
        cpu.s = list(ctx.s)
        cpu.a = list(ctx.a)
        cpu.v = [list(row) for row in ctx.v]
        cpu.pc = ctx.pc
        cpu.c = ctx.c
        cpu.vl = ctx.vl
        cpu.base = job.base
        cpu.limit = job.limit
        cpu.psw = PSW(
            mode=1,
            ft=ctx.psw.ft,
            fu=ctx.psw.fu,
            ov=ctx.psw.ov,
            ie=ctx.psw.ie,
            cpuid=cpu_id,
            ov_trap=ctx.psw.ov_trap,
        )
        # Invalidate Cray I-stack when switching user address spaces!
        cpu._istack_base = None
        cpu._istack_slots.clear()
        cpu.timer_quantum = job.quantum
        cpu._timer_counter = 0
        cpu.halted = False
        cpu.last_trap = None

        # Clear CPU mailbox status word
        mb = SUP_MAILBOX_BASE - 16 * cpu_id
        self.machine.mem.write(mb, MB_IDLE, supervisor=True)

        job.state = "RUNNING"
        job.assigned_cpu = cpu_id
        self._running_on_cpu[cpu_id] = job.job_id

    def _save_job_from_cpu(self, job: CExecJob, cpu_id: int) -> None:
        cpu = self.machine.cpus[cpu_id]
        slot = -27 * cpu_id
        saved_pc = self.machine.mem.read(slot - 1, supervisor=True)[0]
        saved_psw = self.machine.mem.read(slot - 2, supervisor=True)[0]
        saved_cvl = self.machine.mem.read(slot - 5, supervisor=True)[0]

        ctx = job.context
        ctx.s = list(cpu.s)
        ctx.a = list(cpu.a)
        ctx.v = [list(row) for row in cpu.v]
        ctx.pc = saved_pc
        ctx.psw = PSW.unpack(saved_psw, default_cpuid=cpu_id)
        ctx.c = extract_field(saved_cvl, 6, 1, WORD_TRITS)
        ctx.vl = max(0, min(C_VEC_LEN, extract_field(saved_cvl, 0, 6, WORD_TRITS)))

    def _drain_console_ring(self) -> List[Tuple[int, str]]:
        """Read any new SVC #2 (PUTINT) / SVC #3 (PUTCHAR) records from Supervisor T-DRAM."""
        mem = self.machine.mem
        total = mem.read(SUP_CONSOLE_COUNT_ADDR, supervisor=True)[0]
        emitted: List[Tuple[int, str]] = []
        while self._console_read_idx < total:
            off = -3 * self._console_read_idx
            kind = mem.read(SUP_CONSOLE_RING_BASE + off, supervisor=True)[0]
            val = mem.read(SUP_CONSOLE_RING_BASE - 1 + off, supervisor=True)[0]
            mb_addr = mem.read(SUP_CONSOLE_RING_BASE - 2 + off, supervisor=True)[0]
            cpuid = (SUP_MAILBOX_BASE - mb_addr) // 16
            jid = self._running_on_cpu.get(cpuid)
            job = self.jobs.get(jid) if jid is not None else None
            if kind == 1:
                text = str(val)
                if job is not None:
                    job.output_lines.append(text)
                emitted.append((jid or 0, text))
            elif kind == -1:
                ch = obb729_to_char(val)
                if job is not None:
                    if ch == "\n":
                        job.output_lines.append(job._char_buf)
                        emitted.append((job.job_id, job._char_buf))
                        job._char_buf = ""
                    else:
                        job._char_buf += ch
            self._console_read_idx += 1
        return emitted

    def _flush_job_char_buf(self, job: CExecJob) -> None:
        if job._char_buf:
            job.output_lines.append(job._char_buf)
            job._char_buf = ""

    def dispatch_ready(self, max_cpus: int = 8) -> int:
        """Assign queued jobs to idle C-Complex CPUs (honoring preferred_cpu if free)."""
        launched = 0
        remaining: List[int] = []
        for jid in self._ready_queue:
            job = self.jobs[jid]
            chosen: Optional[int] = None
            if job.preferred_cpu is not None and 0 <= job.preferred_cpu < max_cpus:
                if job.preferred_cpu not in self._running_on_cpu:
                    chosen = job.preferred_cpu
            else:
                for cid in range(max_cpus):
                    if cid not in self._running_on_cpu:
                        chosen = cid
                        break
            if chosen is not None:
                self._load_job_onto_cpu(job, chosen)
                launched += 1
            else:
                remaining.append(jid)
        self._ready_queue = remaining
        return launched

    def run_scheduler(
        self,
        max_cpus: int = 8,
        max_rounds: int = 200,
        steps_per_round: int = 2_000,
    ) -> List[CExecJob]:
        """Run all queued/active jobs across up to `max_cpus` CPUs with preemptive time-slicing."""
        if not self.booted:
            self.boot()

        for _ in range(max_rounds):
            self.dispatch_ready(max_cpus=max_cpus)
            if not self._running_on_cpu:
                break

            active_cpus = sorted(self._running_on_cpu.keys())
            start_cycles = {cid: self.machine.cpus[cid].cycles for cid in active_cpus}
            start_insns = {cid: self.machine.cpus[cid].instructions for cid in active_cpus}

            # Step active CPUs concurrently in cycle-accurate order until at least one halts (trap/SVC/timer)
            steps = 0
            while steps < steps_per_round:
                if any(self.machine.cpus[cid].halted for cid in active_cpus):
                    break
                stepped = self.machine.step_all(active_cpus)
                if stepped < 0:
                    break
                steps += 1

            # Drain any console output produced while CPUs were mapped to their jobs
            self._drain_console_ring()

            # Inspect every halted CPU's mailbox written natively by C-EXEC in the Ferrite Vault!
            for cid in active_cpus:
                cpu = self.machine.cpus[cid]
                jid = self._running_on_cpu[cid]
                job = self.jobs[jid]
                job.cycles_used += cpu.cycles - start_cycles[cid]
                job.instructions_used += cpu.instructions - start_insns[cid]

                if not cpu.halted:
                    continue

                mb = SUP_MAILBOX_BASE - 16 * cid
                status = self.machine.mem.read(mb, supervisor=True)[0]
                exit_code = self.machine.mem.read(mb + 1, supervisor=True)[0]
                trap_code = self.machine.mem.read(mb + 2, supervisor=True)[0]
                trap_arg = self.machine.mem.read(mb + 3, supervisor=True)[0]

                del self._running_on_cpu[cid]
                job.assigned_cpu = None

                if status == MB_EXITED:
                    self._flush_job_char_buf(job)
                    job.state = "EXITED"
                    job.exit_code = exit_code
                elif status in (MB_PREEMPTED, MB_YIELDED):
                    self._save_job_from_cpu(job, cid)
                    if status == MB_PREEMPTED:
                        job.preemptions += 1
                    else:
                        job.yields += 1
                    job.state = "QUEUED"
                    self._ready_queue.append(jid)
                else:
                    self._flush_job_char_buf(job)
                    job.state = "ABORTED"
                    job.abort_trap = trap_code
                    job.abort_arg = trap_arg

        return list(self.jobs.values())
