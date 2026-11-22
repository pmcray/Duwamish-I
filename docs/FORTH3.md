# FORTH-3 — Balanced-Ternary Indirect-Threaded Stack Language Manual

**System:** Duwamish-I (`UE-1000`, Rev. 1.5) — OBB Unseen Level -9  
**Lineage:** Charles Moore's NRAO Kitt Peak radio-telescope FORTH (1968–1970), Moscow's Setun-70 stack philosophy (1970), and classic FORTH-79 / BBC FORTH compatibility  
**Primary Users:** Bernard Oliver's Level -9 SETI Baseband Annex & UE Special Systems Maintenance/Staging Bay  
**Implementation:** [ue1000/forth3.py](../ue1000/forth3.py)

---

## 1. Native `TRIAD-36` Indirect-Threaded Virtual Machine

`FORTH-3` is not a Python interpreter — its inner interpreter (`_NEXT`), dictionary, colon definitions, data stack, return stack, and ALU primitives live directly inside `UE-1000` `T-DRAM` and execute natively on `machine.cpu0`:

| Hardware Register | Forth VM Role | Address Space |
|---|---|---|
| `S8` | Data Stack Pointer (`DSP`, native `PUSH`/`POP`) | `+300,000` (grows downward) |
| `A7` | Return Stack Pointer (`RSP`) | `+350,000` (grows downward) |
| `A6` | Instruction Pointer (`IP`) | Dictionary `+20,000+` |
| `A5` | Work Register (`W` / CFA) | Indirect jump `JMP @0(A5)` |

The entire inner interpreter is **4 native `TRIAD-36` instructions**:
```asm
_NEXT:  LD S1, 0(A6)
        AADD A6, A6, #1
        S2A A5, S1
        JMP @0(A5)
```

---

## 2. Core Wordset (FORTH-79 / BBC FORTH Compatible + Balanced Ternary)

- **Stack:** `DUP`, `DROP`, `SWAP`, `OVER`, `ROT`, `-ROT`, `NIP`, `TUCK`, `?DUP`, `2DUP`, `2DROP`, `2SWAP`, `DEPTH`, `PICK`, `>R`, `R>`, `R@`
- **Balanced-Ternary & Integer Math (`T36`):** `+`, `-`, `*`, `/`, `MOD`, `/MOD`, `*/`, `1+`, `1-`, `2*`, `2/`, **`3*`** *(1-cycle `SHF #1`)*, **`3/`** *(1-cycle unbiased `SHF #-1`)*, `NEGATE`, `ABS`, **`SGN`** *(`-1/0/+1`)*, `MIN`, `MAX`
- **Literals:** Decimal integers (`42`, `-9`), balanced ternary (`3t1T01`), Good decibans (`12.5dB` → ticks), `TF36` floats (`3.14f` or `0.25`)
- **Logic & Comparisons:** `=`, `<>`, `<`, `>`, `<=`, `>=`, `0=`, `0<>`, `0<`, `0>`, `INVERT`, **`KAND`/`AND`**, **`KOR`/`OR`**, **`KEQV`**, **`KNOT`**, `TRUE`/`FALSE`, **`KTRUE` (`+1`) / `KUNDECIDED` (`0`) / `KFALSE` (`-1`)**
- **Control Flow:** `IF ... [ELSE ...] THEN`, **`TRIT-IF <neg> WHEN0 <zero> WHEN+ <pos> TRIT-THEN`** *(single 3-way dispatch!)*, `BEGIN ... UNTIL/AGAIN/WHILE ... REPEAT`, `DO ... [LEAVE] ... LOOP/+LOOP` (`I`, `J`, `UNLOOP`), `CASE ... OF ... ENDOF ... ENDCASE`, `RECURSE`
- **Memory & Defining Words:** `@`, `!`, `+!`, `HERE`, `,`, `ALLOT`, `VARIABLE`, `CONSTANT`, `CREATE ... DOES>`, `'`, `[']`, `EXECUTE`
- **I/O & Formatting:** `.`, `EMIT`, `CR`, `SPACE`, `SPACES`, `." ..."`, `.S`, **`.H`** *(UE Heptavintimal)*, **`.T`** *(balanced trits)*, **`.DB`** *(decibans)*, `F.`
- **Hardware & SETI Instrument Words:** `F+`, `F-`, `F*`, `F/`, `I>F`, `F>I`, `DB+`, `DB-`, `DLSE`, `DWALD`, `DB>TICKS`, **`T.FFT3`** *(Tier 2 Radix-3 `TBFP-9C` FFT)*, **`V.DOT`** *(Tier 1 dot product)*, **`CYCLES`** *(hardware `ALGD`)*

---

## 3. CLI & Reference Programs

```bash
python3 -m ue1000 forth programs/forth_tour.fth        # Stack, loops, RECURSE, CREATE..DOES>, TRIT-IF
python3 -m ue1000 forth programs/oliver_dish_ctrl.fth  # Level -9 SETI Annex dish & T-Fabric FFT controller
python3 -m ue1000 forth --repl                         # Interactive FORTH-3 console
```
