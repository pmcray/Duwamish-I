# `C-EXEC` & `V-OS` — Bare-Metal Ferrite Vault Executive & Level -9 Supervisory OS

**System:** Duwamish-I (`UE-1000`, Rev. 1.5) — OBB Unseen Level -9  
**Architectural Split (Brooks Anti-CAMEL Decoupling):**
- **`C-EXEC`** ([`ue1000/cexec.py`](../ue1000/cexec.py)) — Native `TRIAD-36` bare-metal hypervisor assembled directly into the **Non-Volatile Dual-Core Ferrite Vault** (`-177,147 .. -1`), backed by Supervisor `T-DRAM` (`-1,594,323 .. -177,148`).
- **`V-OS`** ([`ue1000/vos.py`](../ue1000/vos.py)) — Tier 4 V-Complex supervisory console providing WEAR capability security, Jameson's Alexandrian Vault filesystem over real `CHIO` channels, multi-CPU `C-EXEC` batch scheduling, Beer VSM telemetry, and a unified 9-language launcher.

## Quick Start
```bash
# Run the showcase Level -9 V-OS / C-EXEC operations script
python3 -m ue1000 vos programs/level9_ops.vos

# Launch an interactive OBB Level -9 V-OS terminal
python3 -m ue1000 vos --repl --user BROOKS
```

## `C-EXEC` Ferrite Vault Memory Map & `SVC` ABI
| Address Range | Purpose |
|---|---|
| `-1 .. -216` | Per-CPU Hardware Trap Frames (`-27 * cpuid - 1..-27`) & Trap Entry Vectors |
| `-500 .. -627` | 8 Per-CPU Entry Stubs saving `A5..A7, S5..S7` into Ferrite scratch slots |
| `-1,000` | Cold-Boot IPL (verifies `MMIO_SYS_ID = 100015`, stamps `C_EXEC_MAGIC`) |
| `-1,100` | Shared Native `TRIAD-36` Trap & `SVC` Dispatcher |
| `-200,000 .. -200,004` | Boot Magic, Global Trap / `SVC` / `TRAP_TIMER` Preemption / Abort Counters |
| `-210,000 - 16*cpu` | Per-CPU Job Mailboxes (`status`, `exit_code`, `trap_code`, `trap_arg`, `traps`) |
| `-220,000 ..` | Supervisor Console Output Ring (`SVC #2 PUTINT`, `SVC #3 PUTCHAR`) |

### Supported User-Mode `SVC` Calls
| `SVC` | Name | Contract |
|---:|---|---|
| `#1` | `SYS_EXIT` | Record `S1` as job exit code in mailbox & terminate job cleanly |
| `#2` | `SYS_PUTINT` | Emit signed `T36` integer `S1` to Supervisor Console Ring & `RTI` |
| `#3` | `SYS_PUTCHAR` | Emit `OBB-729` tryte `S1` to Supervisor Console Ring & `RTI` |
| `#4` | `SYS_CLOCK` | Return 30 ns global `TIM` clock (`ALGD #0`) in `S1` & `RTI` |
| `#5` | `SYS_HEALTH` | Return Beer VSM health trit (`ALGD #3`: `+1/0/-1`) in `S1` & `RTI` |
| `#7` | `SYS_YIELD` | Cooperative context-switch yield back to `C-EXEC` round-robin queue |

## `V-OS` WEAR Principals & Compartments
| Principal | Role | Clearance | Allowed Compartments |
|---|---|---:|---|
| `BROOKS` | Chief System Architect (`SYSOP`) | 3 | `0, 1971, 2000, 3000, 7777` |
| `JAMESON` | University Librarian / Vault Custodian | 3 | `0, 1971, 7777` |
| `ROSENBLATT` | T-Fabric & Byurakan AI Lead | 2 | `0, 1971` |
| `OLIVER` | SETI Baseband Annex Lead | 2 | `0, 1971, 3000` |
| `FORRESTER` | Cliodynamics & Store Lead | 2 | `0, 2000` |
| `KNUTH` | ISA & Compilers Lead | 2 | `0, 1971, 2000` |
| `GUEST` | OBB Research Carrel Terminal | 1 | `0` (open only) |
