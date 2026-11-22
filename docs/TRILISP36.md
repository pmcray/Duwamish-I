# `TRILISP-36` — Classic LISP 1.5 Native `TRIAD-36` Compiler & Heap VM

**Target Architecture:** Duwamish-I (`UE-1000`), `TRIAD-36` Assembly (Rev. 1.5)  
**Lineage:** McCarthy's LISP 1.5 (1962) + UE Special Systems `TRILISP` ternary tagging

```bash
python3 -m ue1000 trilisp programs/trilisp_tour.lsp       # Compile & run natively on UE-1000
python3 -m ue1000 trilisp programs/trilisp_tour.lsp --asm # Inspect native TRIAD-36 + GC code
```

## 1. One Trit, Three Types — And 1-Cycle Balanced Unboxing
Every `TRILISP-36` value is a single 36-trit `T36` word tagged by its **least significant trit (`trit 0`)**, dispatched in one cycle by hardware `TRT S1, S0, #0` + `B3`:

| LSB Trit | Domain | Encoding | Balanced-Ternary Unbox (`SHF S1, #-1`) |
|---:|---|---|---|
| **`+1` (`1`)** | **Fixnum (`INT`)** | `3*n + 1` (35-trit integer, `±2.50 × 10^{16}`) | `round((3*n + 1)/3) == n` |
| **`0` (`0`)** | **Heap Pointer** | `3*i`: `i > 0` cons cell (`[CAR, CDR]`); `i < 0` boxed `TF36` float, `DB36` deciban, or lexical `CLOSURE` | `round(3*i / 3) == i` |
| **`-1` (`T`)** | **Symbol** | `3*j - 1`: `j=0` → **`NIL` (`-1`)**, `j=1` → **`T` (`+2`)**, `j=2` → **`U` (`+5`, `UNDECIDED`)** | `round((3*j - 1)/3) == j` |

Because `1/3 < 1/2` in balanced ternary, a single 1-cycle **`SHF Sd, Ss1, #-1`** unboxes **any** tag (`+1`, `0`, or `-1`) with zero bit-masking or tag subtraction — and tagged fixnum addition/subtraction is simply `ADD + SUB #1` / `SUB + ADD #1`.

## 2. Special Forms & Proper Tail-Call Optimization
- **LISP 1.5 Core:** `QUOTE` / `'x`, `DEFINE`, `DEFUN`, `DEFMACRO`, `LAMBDA` (with dotted rest parameters `(lambda (a . rest) ...)`), **`LABEL`**, `IF`, `COND` (`T` / `ELSE`), `LET`, `LET*`, `SETQ`, `PROGN`, `AND`, `OR`, `APPLY`, `EVAL`.
- **Three-Valued Ternary Extensions:**
  - `NIL` (`FALSE`), `T` (`TRUE`), **`U` (`UNDECIDED`)** — `IF`/`COND` treat both `NIL` and `U` as non-true.
  - **`SIGN3`** (`(sign3 expr neg-branch zero-branch pos-branch)`) compiles directly to a single hardware **`B3`** 3-way jump table on both fixnums and Kleene truth values.
  - Kleene connectives **`KAND`**, **`KOR`**, **`KEQV`**, **`KNOT`** map directly to hardware `AND`/`OR`/`EQV`/`NEG`.
  - Good Deciban primitives **`DB+`**, **`DLSE`**, **`WALD`** (`20.0db` literals → `DWALD` returning `T / U / NIL`).
- **Proper TCO:** All compiled functions share a uniform 2-argument frame `(env, arg-list)` so tail calls overwrite the frame in place and **`JMP`** directly, running arbitrarily deep tail recursion in `O(1)` stack space.

## 3. Native `TRIAD-36` Mark-and-Sweep Garbage Collector
The cons heap (`HEAP_BASE = 200,000`), mark table (`MARK_BASE = 250,000`), and iterative mark stack (`GC_STACK_BASE = 350,000`) live directly in User `T-DRAM`. When the heap fills or `(GC)` is called, `_TL_GC_INNER` runs natively on C-Complex CPU 0, conservatively tracing all global symbol values and the live machine stack (`S8..STACK_TOP`), marking reachable cons cells and closure environments, sweeping dead cells onto `_RT_FREE`, and returning the reclaimed count.
