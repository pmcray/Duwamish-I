"""Tier 1: C-Complex 36-Trit Hardwired Vector-Scalar Processor (Rev. 1.5).

Conforms strictly to Principles of Operation (Rev. 1.5), Chapter 4:
  - 30 ns (33.33 MHz) UE-TECL clock
  - Registers: S0-S8 (36-trit, S0 == 0), A0-A8 (18-trit, A0 == 0), V0-V8 (27 x 36-trit)
  - 27-word (54-parcel) Cray Instruction Stack
  - Dual-parcel 18-trit (F=0, F=T) & Long 36-trit (F=+1) execution
  - Exact traps 1..18 saved to Non-Volatile Ferrite Core (-1..-27 per CPU)
  - Native Kleene logic, TF36/TF18 exact unbiased float, Good Deciban ALU, and T-Fabric dispatch
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional

from .isa import (
    DecodedInstruction,
    MODE_DIRECT,
    MODE_H0,
    MODE_H1,
    MODE_IMM,
    MODE_INDIRECT,
    MODE_NONET,
    MODE_TRYTE,
    MODE_VSTRIDE,
    MODE_VUNIT,
    decode_word,
    decode_xtr_simm,
)
from .memory import (
    FERRITE_MAX_ADDR,
    USER_TDRAM_MAX,
    MemoryDEDError,
    MemoryProtectionError,
    TernaryMemory,
)
from .ternary import (
    HALF_TRITS,
    POW3,
    WMAX_36,
    WORD_TRITS,
    extract_field,
    get_trit,
    kleene_and,
    kleene_eqv,
    kleene_or,
    kleene_sel,
    set_field,
    shift_trits,
    sign_trit,
    wrap,
)
from .tfloat import (
    TF18,
    TF36,
    deciban_add,
    deciban_exp_float,
    deciban_lse,
    deciban_sub,
    deciban_wald,
    float_to_tfloat,
    int_to_tfloat,
    pack_dual_tf18,
    tfloat_abs,
    tfloat_add,
    tfloat_cmp,
    tfloat_div,
    tfloat_fma,
    tfloat_mul,
    tfloat_neg,
    tfloat_sub,
    tfloat_to_int,
    unpack_dual_tf18,
)
from .tfabric import TFabric, TFabricUnrecoverableFault

if TYPE_CHECKING:
    from .channels import IOChannelSubsystem
    from .vcomplex import VComplex

C_CLOCK_NS: int = 30
C_VEC_LEN: int = 27
ISTACK_WORDS: int = 27

# Hardware Trap Numbers (Chapter 4.4)
TRAP_ILLOP = 1
TRAP_PROT = 2
TRAP_DIVZ = 3
TRAP_PRIV = 4
TRAP_OVF = 5
TRAP_SVC = 6
TRAP_FPOVF = 7
TRAP_FPUND = 8
TRAP_TIMER = 10
TRAP_TFAULT = 12
TRAP_MEMDED = 14
TRAP_MCHK = 18


class CPUTrap(Exception):
    def __init__(self, code: int, arg: int = 0, msg: str = "") -> None:
        super().__init__(msg or f"Trap {code} (arg={arg})")
        self.code = code
        self.arg = arg


@dataclass
class PSW:
    mode: int = -1       # -1 = Supervisor, +1 = User
    ft: int = 0          # +1 = trap on float overflow/undecided
    fu: int = 0          # sticky float UNDECIDED flag
    ov: int = 0          # sticky integer overflow flag
    ie: int = 1          # interrupt enable
    cpuid: int = 0
    ov_trap: int = 0     # +1 = raise Trap 5 (OVF) on signed integer overflow

    def pack(self) -> int:
        return (
            self.ov_trap * POW3[11]
            + self.mode * POW3[10]
            + self.ft * POW3[9]
            + self.fu * POW3[8]
            + self.ov * POW3[7]
            + self.ie * POW3[6]
            + wrap(self.cpuid, 6)
        )

    @classmethod
    def unpack(cls, word: int, default_cpuid: Optional[int] = None) -> "PSW":
        return cls(
            mode=extract_field(word, 10, 1, WORD_TRITS) or -1,
            ft=extract_field(word, 9, 1, WORD_TRITS),
            fu=extract_field(word, 8, 1, WORD_TRITS),
            ov=extract_field(word, 7, 1, WORD_TRITS),
            ie=extract_field(word, 6, 1, WORD_TRITS),
            cpuid=default_cpuid if default_cpuid is not None else extract_field(word, 0, 6, WORD_TRITS),
            ov_trap=extract_field(word, 11, 1, WORD_TRITS),
        )


class CComplexCPU:
    """Single 36-trit hardwired vector-scalar C-Complex processor."""

    def __init__(
        self,
        cpu_id: int,
        mem: TernaryMemory,
        tfabric: TFabric,
        io_channels: Optional["IOChannelSubsystem"] = None,
        vcomplex: Optional["VComplex"] = None,
    ) -> None:
        self.cpu_id = cpu_id
        self.mem = mem
        self.tfabric = tfabric
        self.io_channels = io_channels
        self.vcomplex = vcomplex if vcomplex is not None else tfabric.vcomplex

        self.s: List[int] = [0] * 9
        self.a: List[int] = [0] * 9
        self.v: List[List[int]] = [[0] * C_VEC_LEN for _ in range(9)]
        self.pc: int = 100
        self.c: int = 0
        self.vl: int = C_VEC_LEN
        self.base: int = 0
        self.limit: int = USER_TDRAM_MAX
        self.psw = PSW(mode=-1, cpuid=cpu_id)

        # 27-word Cray Instruction Stack window: [base .. base + 26]
        self._istack_base: Optional[int] = None
        self._istack_slots: dict[int, int] = {}
        self.istack_hits: int = 0
        self.istack_misses: int = 0

        # Optional C-EXEC watchdog timer quantum (instruction words before TRAP_TIMER)
        self.timer_quantum: Optional[int] = None
        self._timer_counter: int = 0

        self.cycles: int = 0
        self.instructions: int = 0
        self.halted: bool = False
        self.last_trap: Optional[int] = None
        self.svc_log: List[int] = []
        self.svc_handler: Optional[Callable[["CComplexCPU", int], bool]] = None

    def _set_s(self, reg: int, val: int, set_c: bool = True) -> None:
        w = wrap(val, WORD_TRITS)
        if reg != 0:
            self.s[reg] = w
        if set_c:
            self.c = sign_trit(w)

    def _set_a(self, reg: int, val: int) -> None:
        if reg != 0:
            self.a[reg] = wrap(val, HALF_TRITS)

    def _is_sup(self) -> bool:
        return self.psw.mode == -1

    def _mem_read(self, addr: int) -> int:
        try:
            val, lat_ns = self.mem.read(addr, supervisor=self._is_sup(), base=self.base, limit=self.limit)
        except MemoryProtectionError as exc:
            raise CPUTrap(TRAP_PROT, addr, str(exc)) from exc
        except MemoryDEDError as exc:
            raise CPUTrap(TRAP_MEMDED, addr, str(exc)) from exc
        self.cycles += max(1, lat_ns // C_CLOCK_NS)
        return val

    def _mem_write(self, addr: int, val: int) -> None:
        try:
            lat_ns = self.mem.write(addr, val, supervisor=self._is_sup(), base=self.base, limit=self.limit)
        except MemoryProtectionError as exc:
            raise CPUTrap(TRAP_PROT, addr, str(exc)) from exc
        # Invalidate instruction stack if store hits cached window
        if self._istack_base is not None and self._istack_base <= addr < self._istack_base + ISTACK_WORDS:
            self._istack_base = None
            self._istack_slots.clear()
        self.cycles += max(1, lat_ns // C_CLOCK_NS)

    def _fetch_word(self, addr: int) -> int:
        if self._istack_base is not None and self._istack_base <= addr < self._istack_base + ISTACK_WORDS:
            if addr in self._istack_slots:
                self.istack_hits += 1
                return self._istack_slots[addr]
            self.istack_misses += 1
            w = self._mem_read(addr)
            self._istack_slots[addr] = w
            return w
        self.istack_misses += 1
        w = self._mem_read(addr)
        self._istack_base = addr
        self._istack_slots = {addr: w}
        return w

    def _handle_trap(self, trap: CPUTrap) -> None:
        self.last_trap = trap.code
        slot = -27 * self.cpu_id  # CPU 0 uses -1..-27 in Ferrite Core!
        # Save trap frame into non-volatile Ferrite Vault
        self.mem.write(slot - 1, self.pc, supervisor=True)
        self.mem.write(slot - 2, self.psw.pack(), supervisor=True)
        self.mem.write(slot - 3, trap.code, supervisor=True)
        self.mem.write(slot - 4, trap.arg, supervisor=True)
        self.mem.write(slot - 5, self.c * POW3[6] + self.vl, supervisor=True)

        handler = self.mem.read(slot - 27, supervisor=True)[0]
        if self._is_sup() and trap.code != TRAP_SVC:
            # Trap in Supervisor mode = Machine Check (Trap 18) -> HALT
            self.last_trap = TRAP_MCHK
            self.halted = True
            return
        if handler == 0:
            # No C-EXEC vector installed: halt cleanly with trap code recorded
            self.halted = True
            return
        self.psw.mode = -1
        self.pc = handler

    def step(self) -> None:
        """Fetch and execute one 36-trit instruction word (1 Long or 2 Short parcels)."""
        if self.halted:
            return
        pc_curr = self.pc
        try:
            # Watchdog quantum expiry check on clean instruction boundary (User mode, IE=+1)
            if (
                not self._is_sup()
                and self.psw.ie == 1
                and self.timer_quantum is not None
                and self._timer_counter >= self.timer_quantum
            ):
                self._timer_counter = 0
                raise CPUTrap(TRAP_TIMER, pc_curr, "Watchdog quantum expired")

            raw_word = self._fetch_word(pc_curr)
            decoded = decode_word(raw_word)
            self.pc = pc_curr + 1  # default sequential next word

            for ins in decoded:
                if ins.mnemonic == "ILLEGAL":
                    raise CPUTrap(TRAP_ILLOP, raw_word, f"Illegal instruction/parcel at PC={pc_curr}")
                branched = self._exec_one(ins, pc_curr)
                self.instructions += 1
                if self.halted or branched:
                    # Branch in Parcel A skips Parcel B and transfers to Parcel A of target word
                    break

            if not self._is_sup() and self.timer_quantum is not None:
                self._timer_counter += 1

            # Advance any in-flight asynchronous T-Fabric or Tier 3b DMA completions
            now_ns = self.cycles * C_CLOCK_NS
            if self.tfabric._pending_jobs:
                self.tfabric.advance_to_ns(now_ns)
            if self.io_channels is not None and self.io_channels._pending:
                self.io_channels.advance_to_ns(now_ns)
        except CPUTrap as trap:
            self._handle_trap(trap)

    def run(self, max_steps: int = 100_000) -> int:
        steps = 0
        while not self.halted and steps < max_steps:
            self.step()
            steps += 1
        return steps

    def _exec_one(self, ins: DecodedInstruction, pc_curr: int) -> bool:
        """Execute one decoded Long or Short instruction. Returns True if PC branched."""
        mn = ins.mnemonic
        rd, rs1, rs2, mode, imm = ins.rd, ins.rs1, ins.rs2, ins.mode, ins.imm

        # ---------------- SHORT F = 0 (Scalar / Logic / Branch) ----------------
        if ins.fmt == 0:
            if mn == "NOP":
                self.cycles += 1
            elif mn == "ADD":
                raw_sum = self.s[rs1] + self.s[rs2] + imm
                self._set_s(rd, raw_sum)
                self.cycles += 1
                if raw_sum < -WMAX_36 or raw_sum > WMAX_36:
                    self.psw.ov = 1
                    if self.psw.ov_trap == 1:
                        raise CPUTrap(TRAP_OVF, pc_curr, "Signed integer overflow in ADD")
            elif mn == "SUB":
                raw_diff = self.s[rs1] - self.s[rs2] - imm
                self._set_s(rd, raw_diff)
                self.cycles += 1
                if raw_diff < -WMAX_36 or raw_diff > WMAX_36:
                    self.psw.ov = 1
                    if self.psw.ov_trap == 1:
                        raise CPUTrap(TRAP_OVF, pc_curr, "Signed integer overflow in SUB")
            elif mn == "MUL":
                raw_prod = self.s[rs1] * self.s[rs2]
                self._set_s(rd, raw_prod)
                self.cycles += 4
                if raw_prod < -WMAX_36 or raw_prod > WMAX_36:
                    self.psw.ov = 1
                    if self.psw.ov_trap == 1:
                        raise CPUTrap(TRAP_OVF, pc_curr, "Signed integer overflow in MUL")
            elif mn == "MULH":
                prod = self.s[rs1] * self.s[rs2]
                self._set_s(rd, shift_trits(prod, -WORD_TRITS, WORD_TRITS * 2))
                self.cycles += 4
            elif mn in ("DIV", "MOD"):
                if self.s[rs2] == 0:
                    raise CPUTrap(TRAP_DIVZ, pc_curr, "Integer divide by zero")
                # Balanced truncating quotient toward zero
                a, b = self.s[rs1], self.s[rs2]
                q = int(a / b) if abs(a) < 10**15 else (abs(a) // abs(b)) * (1 if (a >= 0) == (b >= 0) else -1)
                self._set_s(rd, q if mn == "DIV" else (a - q * b))
                self.cycles += 14
            elif mn == "NEG":
                self._set_s(rd, -self.s[rs1])
                self.cycles += 1
            elif mn == "ABS":
                self._set_s(rd, abs(self.s[rs1]))
                self.cycles += 1
            elif mn == "SHF":
                self._set_s(rd, shift_trits(self.s[rs1], self.s[rs2] + imm, WORD_TRITS))
                self.cycles += 1
            elif mn == "AND":
                self._set_s(rd, kleene_and(self.s[rs1], self.s[rs2]))
                self.cycles += 1
            elif mn == "OR":
                self._set_s(rd, kleene_or(self.s[rs1], self.s[rs2]))
                self.cycles += 1
            elif mn == "EQV":
                self._set_s(rd, kleene_eqv(self.s[rs1], self.s[rs2]))
                self.cycles += 1
            elif mn == "XTR":
                p, n = decode_xtr_simm(imm)
                self._set_s(rd, extract_field(self.s[rs1], p, n, WORD_TRITS))
                self.cycles += 1
            elif mn == "TRT":
                self.c = get_trit(self.s[rs1], self.s[rs2] + imm, WORD_TRITS)
                self.cycles += 1
            elif mn == "SEL":
                self._set_s(rd, kleene_sel(self.c, self.s[rs1] + imm))
                self.cycles += 1
            elif mn == "CMP":
                self.c = sign_trit(self.s[rs1] - (self.s[rs2] + imm))
                self.cycles += 1
            elif mn == "TST":
                self.c = sign_trit(self.s[rs1])
                self.cycles += 1
            elif mn == "AADD":
                self._set_a(rd, self.a[rs1] + self.a[rs2] + imm)
                self.cycles += 1
            elif mn == "ASUB":
                self._set_a(rd, self.a[rs1] - self.a[rs2] - imm)
                self.cycles += 1
            elif mn == "A2S":
                self._set_s(rd, self.a[rs1])
                self.cycles += 1
            elif mn == "S2A":
                self._set_a(rd, self.s[rs1])
                self.cycles += 1
            elif mn == "BR":
                self.pc = pc_curr + imm
                self.cycles += 1
                return True
            elif mn == "B3":
                self.pc = pc_curr + imm + self.c
                self.cycles += 1
                return True
            elif mn == "BN":
                self.cycles += 1
                if self.c == -1:
                    self.pc = pc_curr + imm
                    return True
            elif mn == "BZ":
                self.cycles += 1
                if self.c == 0:
                    self.pc = pc_curr + imm
                    return True
            elif mn == "BP":
                self.cycles += 1
                if self.c == 1:
                    self.pc = pc_curr + imm
                    return True
            elif mn == "BCT":
                new_val = wrap(self.a[rd] - 1, HALF_TRITS)
                self._set_a(rd, new_val)
                self.cycles += 1
                if new_val > 0:
                    self.pc = pc_curr + imm
                    return True
            elif mn == "PUSH":
                sp = wrap(self.s[8] - 1, WORD_TRITS)
                self.s[8] = sp
                self._mem_write(sp, self.s[rs1])
                self.cycles += 2
            elif mn == "POP":
                val = self._mem_read(self.s[8])
                self.s[8] = wrap(self.s[8] + 1, WORD_TRITS)
                self._set_s(rd, val)
                self.cycles += 2
            elif mn == "RET":
                ret_pc = self._mem_read(self.s[8])
                self.s[8] = wrap(self.s[8] + 1 + imm, WORD_TRITS)
                self.cycles += 2
                if ret_pc == 0:
                    self.halted = True
                    return True
                self.pc = ret_pc
                return True
            else:
                raise CPUTrap(TRAP_ILLOP, ins.op, f"Unhandled F=0 op {mn}")
            return False

        # ---------------- SHORT F = -1 (Float / Vector / Deciban) ----------------
        if ins.fmt == -1:
            if mn in ("FAD", "FSB", "FMP", "FMA"):
                if mn == "FAD":
                    w, ovf, und = tfloat_add(self.s[rs1], self.s[rs2], TF36)
                    self.cycles += 4
                elif mn == "FSB":
                    w, ovf, und = tfloat_sub(self.s[rs1], self.s[rs2], TF36)
                    self.cycles += 4
                elif mn == "FMP":
                    w, ovf, und = tfloat_mul(self.s[rs1], self.s[rs2], TF36)
                    self.cycles += 5
                else:
                    w, ovf, und = tfloat_fma(self.s[rd], self.s[rs1], self.s[rs2], TF36)
                    self.cycles += 6
                if und:
                    self.psw.fu = 1
                    if self.psw.ft == 1:
                        raise CPUTrap(TRAP_FPUND, pc_curr)
                if ovf and self.psw.ft == 1:
                    raise CPUTrap(TRAP_FPOVF, pc_curr)
                self._set_s(rd, w, set_c=False)
                self.c = 0 if und else sign_trit(w)
            elif mn == "FDV":
                w, ovf, und, divz = tfloat_div(self.s[rs1], self.s[rs2], TF36)
                self.cycles += 16
                if divz:
                    raise CPUTrap(TRAP_DIVZ, pc_curr, "TF36 divide by zero")
                if und:
                    self.psw.fu = 1
                    if self.psw.ft == 1:
                        raise CPUTrap(TRAP_FPUND, pc_curr)
                if ovf and self.psw.ft == 1:
                    raise CPUTrap(TRAP_FPOVF, pc_curr)
                self._set_s(rd, w, set_c=False)
                self.c = 0 if und else sign_trit(w)
            elif mn == "FNEG":
                w = tfloat_neg(self.s[rs1], TF36)
                self._set_s(rd, w, set_c=False)
                self.cycles += 1
            elif mn == "FABS":
                w = tfloat_abs(self.s[rs1], TF36)
                self._set_s(rd, w, set_c=False)
                self.cycles += 1
            elif mn == "FCM":
                c, und = tfloat_cmp(self.s[rs1], self.s[rs2], TF36)
                self.c = c
                if und:
                    self.psw.fu = 1
                self.cycles += 2
            elif mn == "FLT":
                self._set_s(rd, int_to_tfloat(self.s[rs1], TF36))
                self.cycles += 3
            elif mn == "FIX":
                self._set_s(rd, tfloat_to_int(self.s[rs1], TF36))
                self.cycles += 3
            elif mn == "F18PK":
                self._set_s(rd, pack_dual_tf18(self.s[rs1], self.s[rs2]), set_c=False)
                self.cycles += 2
            elif mn == "F18UP":
                self._set_s(rd, unpack_dual_tf18(self.s[rs1], high=(imm != 0)), set_c=False)
                self.cycles += 2
            elif mn in ("FAD18", "FMP18"):
                fn = tfloat_add if mn == "FAD18" else tfloat_mul
                h1 = extract_field(self.s[rs1], 18, 18, WORD_TRITS)
                l1 = extract_field(self.s[rs1], 0, 18, WORD_TRITS)
                h2 = extract_field(self.s[rs2], 18, 18, WORD_TRITS)
                l2 = extract_field(self.s[rs2], 0, 18, WORD_TRITS)
                rh, _, _ = fn(h1, h2, TF18)
                rl, _, _ = fn(l1, l2, TF18)
                self._set_s(rd, rh * POW3[18] + rl, set_c=False)
                self.cycles += 3 if mn == "FAD18" else 4
            elif mn == "DADD":
                if imm == 1:
                    # Packed 2xDB18 saturating deciban add (PoO §4.3B)
                    h1 = extract_field(self.s[rs1], 18, 18, WORD_TRITS)
                    l1 = extract_field(self.s[rs1], 0, 18, WORD_TRITS)
                    h2 = extract_field(self.s[rs2], 18, 18, WORD_TRITS)
                    l2 = extract_field(self.s[rs2], 0, 18, WORD_TRITS)
                    rh = deciban_add(h1, h2, width=HALF_TRITS)
                    rl = deciban_add(l1, l2, width=HALF_TRITS)
                    self._set_s(rd, rh * POW3[18] + rl)
                else:
                    self._set_s(rd, deciban_add(self.s[rs1], self.s[rs2]))
                self.cycles += 1
            elif mn == "DSUB":
                if imm == 1:
                    # Packed 2xDB18 saturating deciban sub (PoO §4.3B)
                    h1 = extract_field(self.s[rs1], 18, 18, WORD_TRITS)
                    l1 = extract_field(self.s[rs1], 0, 18, WORD_TRITS)
                    h2 = extract_field(self.s[rs2], 18, 18, WORD_TRITS)
                    l2 = extract_field(self.s[rs2], 0, 18, WORD_TRITS)
                    rh = deciban_sub(h1, h2, width=HALF_TRITS)
                    rl = deciban_sub(l1, l2, width=HALF_TRITS)
                    self._set_s(rd, rh * POW3[18] + rl)
                else:
                    self._set_s(rd, deciban_sub(self.s[rs1], self.s[rs2]))
                self.cycles += 1
            elif mn == "DLSE":
                self._set_s(rd, deciban_lse(self.s[rs1], self.s[rs2]))
                self.cycles += 3
            elif mn == "DEXP":
                val = deciban_exp_float(self.s[rs1])
                self._set_s(rd, float_to_tfloat(val, TF36), set_c=False)
                self.cycles += 3
            elif mn == "DWALD":
                self.c = deciban_wald(self.s[rs1], reg_ticks=self.s[rs2], simm_db=imm)
                self.cycles += 1
            elif mn in ("VADD", "VSUB", "VMUL", "VFMA"):
                for i in range(self.vl):
                    a, b = self.v[rs1][i], self.v[rs2][i]
                    if mn == "VADD":
                        res = wrap(a + b) if imm == 1 else (deciban_add(a, b) if imm == -1 else tfloat_add(a, b, TF36)[0])
                    elif mn == "VSUB":
                        res = wrap(a - b) if imm == 1 else (deciban_sub(a, b) if imm == -1 else tfloat_sub(a, b, TF36)[0])
                    elif mn == "VMUL":
                        res = wrap(a * b) if imm == 1 else tfloat_mul(a, b, TF36)[0]
                    else:
                        res = tfloat_fma(self.v[rd][i], a, b, TF36)[0]
                    self.v[rd][i] = res
                self.cycles += (5 if mn in ("VMUL", "VFMA") else 4) + self.vl
            elif mn == "VF18OP":
                # Packed 2-way TF18 vector ops (54 floats/vector; PoO §4.3B):
                # imm == 0: add, imm == +1: mul, imm == +2: fma (Vd = Vd + Vs1 * Vs2)
                for i in range(self.vl):
                    h1 = extract_field(self.v[rs1][i], 18, 18, WORD_TRITS)
                    l1 = extract_field(self.v[rs1][i], 0, 18, WORD_TRITS)
                    h2 = extract_field(self.v[rs2][i], 18, 18, WORD_TRITS)
                    l2 = extract_field(self.v[rs2][i], 0, 18, WORD_TRITS)
                    if imm == 0:
                        rh, _, _ = tfloat_add(h1, h2, TF18)
                        rl, _, _ = tfloat_add(l1, l2, TF18)
                    elif imm == 1:
                        rh, _, _ = tfloat_mul(h1, h2, TF18)
                        rl, _, _ = tfloat_mul(l1, l2, TF18)
                    elif imm == 2:
                        hd = extract_field(self.v[rd][i], 18, 18, WORD_TRITS)
                        ld = extract_field(self.v[rd][i], 0, 18, WORD_TRITS)
                        rh, _, _ = tfloat_fma(hd, h1, h2, TF18)
                        rl, _, _ = tfloat_fma(ld, l1, l2, TF18)
                    else:
                        raise CPUTrap(TRAP_ILLOP, imm, f"Illegal VF18OP mode {imm}")
                    self.v[rd][i] = rh * POW3[18] + rl
                self.cycles += 4 + self.vl
            elif mn == "VRED":
                acc = self.v[rs1][0] if self.vl > 0 else 0
                for i in range(1, self.vl):
                    x = self.v[rs1][i]
                    if imm == 0:
                        acc = tfloat_add(acc, x, TF36)[0]
                    elif imm == 1:
                        acc = x if tfloat_cmp(x, acc, TF36)[0] > 0 else acc
                    elif imm == -1:
                        acc = x if tfloat_cmp(x, acc, TF36)[0] < 0 else acc
                    elif imm == 2:
                        acc = deciban_lse(acc, x)
                self._set_s(rd, acc, set_c=False)
                self.cycles += 8 + self.vl
            elif mn == "VKLEEN":
                for i in range(self.vl):
                    a, b = self.v[rs1][i], self.v[rs2][i]
                    if imm == -1:
                        self.v[rd][i] = kleene_and(a, b)
                    elif imm == 1:
                        self.v[rd][i] = kleene_or(a, b)
                    elif imm == 0:
                        self.v[rd][i] = kleene_eqv(a, b)
                    else:
                        self.v[rd][i] = wrap(-a)
                self.cycles += 2 + self.vl
            elif mn == "VSVL":
                val = self.s[rs1] + imm
                self.vl = max(0, min(C_VEC_LEN, val))
                self.cycles += 1
            else:
                raise CPUTrap(TRAP_ILLOP, ins.op, f"Unhandled F=-1 op {mn}")
            return False

        # ---------------- LONG F = +1 (Memory / Vector DMA / Control / T-Fabric) ----------------
        ea = wrap(self.a[rs2] + imm, HALF_TRITS)
        if mn in ("LD", "LDA"):
            if mode == MODE_IMM:
                val = ea
            elif mode == MODE_INDIRECT:
                ptr = self._mem_read(ea)
                val = self._mem_read(wrap(ptr, HALF_TRITS))
            else:
                raw = self._mem_read(ea)
                if mode == MODE_H1:
                    val = extract_field(raw, 18, 18, WORD_TRITS)
                elif mode == MODE_H0:
                    val = extract_field(raw, 0, 18, WORD_TRITS)
                elif mode == MODE_TRYTE:
                    slot = max(0, min(5, self.s[rs1]))
                    val = extract_field(raw, slot * 6, 6, WORD_TRITS)
                elif mode == MODE_NONET:
                    slot = max(0, min(3, self.s[rs1]))
                    val = extract_field(raw, slot * 9, 9, WORD_TRITS)
                else:
                    val = raw
            if mn == "LD":
                self._set_s(rd, val)
            else:
                self._set_a(rd, val)
        elif mn in ("ST", "STA"):
            src = self.s[rs1] if mn == "ST" else self.a[rs1]
            target = wrap(self._mem_read(ea), HALF_TRITS) if mode == MODE_INDIRECT else ea
            if mode in (MODE_H1, MODE_H0, MODE_TRYTE, MODE_NONET):
                cur = self._mem_read(target)
                if mode == MODE_H1:
                    src = set_field(cur, 18, 18, src)
                elif mode == MODE_H0:
                    src = set_field(cur, 0, 18, src)
                elif mode == MODE_TRYTE:
                    slot = max(0, min(5, self.s[rd]))
                    src = set_field(cur, slot * 6, 6, src)
                elif mode == MODE_NONET:
                    slot = max(0, min(3, self.s[rd]))
                    src = set_field(cur, slot * 9, 9, src)
            self._mem_write(target, src)
        elif mn == "LEA":
            self._set_a(rd, ea)
            self.cycles += 1
        elif mn == "LDI":
            self._set_s(rd, imm)
            self.cycles += 1
        elif mn == "VLD":
            stride = self.a[rs1] if mode == MODE_VSTRIDE else 1
            start_cyc = self.cycles
            phys_addrs: List[int] = []
            for i in range(self.vl):
                a_i = wrap(ea + i * stride, HALF_TRITS)
                self.v[rd][i] = self._mem_read(a_i)
                phys_addrs.append(a_i if self._is_sup() else self.base + a_i)
            self.cycles = start_cyc
            if self.vl > 0:
                burst_ns, _ = self.mem.schedule_vector_burst(phys_addrs, start_ns=start_cyc * C_CLOCK_NS)
                self.cycles += max(1, int(math.ceil(burst_ns / C_CLOCK_NS)))
            else:
                self.cycles += 1
        elif mn == "VST":
            stride = self.a[rs1] if mode == MODE_VSTRIDE else 1
            start_cyc = self.cycles
            phys_addrs = []
            for i in range(self.vl):
                a_i = wrap(ea + i * stride, HALF_TRITS)
                self._mem_write(a_i, self.v[rd][i])
                phys_addrs.append(a_i if self._is_sup() else self.base + a_i)
            self.cycles = start_cyc
            if self.vl > 0:
                burst_ns, _ = self.mem.schedule_vector_burst(phys_addrs, start_ns=start_cyc * C_CLOCK_NS)
                self.cycles += max(1, int(math.ceil(burst_ns / C_CLOCK_NS)))
            else:
                self.cycles += 1
        elif mn == "VGTH":
            start_cyc = self.cycles
            phys_addrs = []
            for i in range(self.vl):
                a_i = wrap(ea + self.v[rs1][i], HALF_TRITS)
                self.v[rd][i] = self._mem_read(a_i)
                phys_addrs.append(a_i if self._is_sup() else self.base + a_i)
            self.cycles = start_cyc
            if self.vl > 0:
                burst_ns, _ = self.mem.schedule_vector_burst(phys_addrs, start_ns=start_cyc * C_CLOCK_NS)
                self.cycles += 2 + max(1, int(math.ceil(burst_ns / C_CLOCK_NS)))
            else:
                self.cycles += 2
        elif mn == "VSCT":
            start_cyc = self.cycles
            phys_addrs = []
            for i in range(self.vl):
                a_i = wrap(ea + self.v[rs1][i], HALF_TRITS)
                self._mem_write(a_i, self.v[rd][i])
                phys_addrs.append(a_i if self._is_sup() else self.base + a_i)
            self.cycles = start_cyc
            if self.vl > 0:
                burst_ns, _ = self.mem.schedule_vector_burst(phys_addrs, start_ns=start_cyc * C_CLOCK_NS)
                self.cycles += 2 + max(1, int(math.ceil(burst_ns / C_CLOCK_NS)))
            else:
                self.cycles += 2
        elif mn == "JMP":
            self.pc = wrap(self._mem_read(ea), HALF_TRITS) if mode == MODE_INDIRECT else ea
            self.cycles += 2
            return True
        elif mn == "J3":
            base_target = wrap(self._mem_read(ea), HALF_TRITS) if mode == MODE_INDIRECT else ea
            self.pc = base_target + self.c
            self.cycles += 2
            return True
        elif mn == "CALL":
            target = wrap(self._mem_read(ea), HALF_TRITS) if mode == MODE_INDIRECT else ea
            ret = pc_curr + 1
            if rd == 0:
                sp = wrap(self.s[8] - 1, WORD_TRITS)
                self.s[8] = sp
                self._mem_write(sp, ret)
            else:
                self._set_a(rd, ret)
            self.pc = target
            self.cycles += 2
            return True
        elif mn == "TEXEC":
            ttd_addr = self.a[rs1]
            mask = self.s[rs2]  # S0 == 0 -> all 27 active tiles!
            # Truly asynchronous launch: 4 cycles dispatch, overlapped with subsequent CPU ops!
            self.tfabric.submit_ttd_async(
                self.mem, ttd_addr, mask, launch_ns=self.cycles * C_CLOCK_NS
            )
            self.cycles += 4
        elif mn == "TWAIT":
            mask = self.s[rs2]  # S0 == 0 -> wait on all 27 active tiles; imm = optional cycle timeout
            try:
                status, wait_cyc = self.tfabric.wait_for_tiles(
                    mask,
                    now_ns=self.cycles * C_CLOCK_NS,
                    timeout_cycles=imm,
                    cpu_clock_ns=C_CLOCK_NS,
                )
            except TFabricUnrecoverableFault as exc:
                raise CPUTrap(TRAP_TFAULT, self.a[rs1], str(exc)) from exc
            self.c = status
            self.cycles += wait_cyc + 2
        elif mn == "TSTAT":
            sel = self.a[rs1]
            if sel == 0:
                val = self.tfabric.total_abft_checks
            elif sel == 1:
                zs = self.tfabric.total_zero_trit_savings
                val = int(round((sum(zs) / len(zs)) * 10)) if zs else 0
            elif sel == 2:
                val = self.tfabric.busy_tiles_mask(self.cycles * C_CLOCK_NS)
            elif sel == 3:
                val = len(self.vcomplex.spare_tiles)
            elif -27 <= sel <= -1:
                lt = -sel - 1
                phys = self.vcomplex.tile_lut[lt]
                st = self.vcomplex.tiles[phys]
                temp_mC = int(round((st.temp_C - 22.0) * 100))
                val = temp_mC * POW3[18] + st.abft_checks
            else:
                val = self.tfabric.total_abft_checks
            self._set_s(rd, val)
            self.cycles += 2
        elif mn == "SVC":
            self.svc_log.append(imm)
            self.cycles += 6
            if self.svc_handler is not None and self.svc_handler(self, imm):
                return False
            raise CPUTrap(TRAP_SVC, imm, f"SVC #{imm}")
        elif mn == "RTI":
            if not self._is_sup():
                raise CPUTrap(TRAP_PRIV, pc_curr, "RTI in User mode")
            slot = -27 * self.cpu_id
            self.pc = self.mem.read(slot - 1, supervisor=True)[0]
            saved_psw = self.mem.read(slot - 2, supervisor=True)[0]
            saved_cvl = self.mem.read(slot - 5, supervisor=True)[0]
            self.psw = PSW.unpack(saved_psw, default_cpuid=self.cpu_id)
            self.c = extract_field(saved_cvl, 6, 1, WORD_TRITS)
            vl_restored = extract_field(saved_cvl, 0, 6, WORD_TRITS)
            self.vl = max(0, min(C_VEC_LEN, vl_restored))
            self.cycles += 6
            return True
        elif mn == "CHIO":
            if not self._is_sup():
                raise CPUTrap(TRAP_PRIV, pc_curr, "CHIO in User mode")
            if self.io_channels is None:
                raise CPUTrap(TRAP_ILLOP, ins.op, "No Tier 3b IOChannelSubsystem attached")
            ccb_addr = self.a[rs1]
            status, _words, _dur = self.io_channels.execute_chio(
                channel=imm, ccb_addr=ccb_addr, mem=self.mem, now_ns=self.cycles * C_CLOCK_NS
            )
            self.c = sign_trit(status)
            self.cycles += 4
        elif mn == "ALGD":
            if imm == 0:
                val = self.cycles
            elif imm == 1:
                val = self.vcomplex.algedonic_alarms
            elif imm == 2:
                val = len(self.vcomplex.cutovers)
            elif imm == 3:
                val = self.vcomplex.system_health_trit()
            elif imm == 4:
                val = self.vcomplex.acknowledge_alarms()
            elif imm == 5:
                val = self.vcomplex.packets[-1].pack_words()[0] if self.vcomplex.packets else 0
            else:
                val = self.cycles
            self._set_s(rd, val)
            self.cycles += 2
        elif mn == "HLT":
            if not self._is_sup():
                raise CPUTrap(TRAP_PRIV, pc_curr, "HLT in User mode")
            self.halted = True
            return True
        else:
            raise CPUTrap(TRAP_ILLOP, ins.op, f"Unhandled Long op {mn}")
        return False
