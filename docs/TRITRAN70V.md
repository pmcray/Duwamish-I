# `TRI-TRAN 70V` — Vectorized Balanced-Ternary FORTRAN-66 Manual

**Target Architecture:** Duwamish-I (`UE-1000`), `TRIAD-36` Assembly (Rev. 1.5)  
**Leads:** Jay W. Forrester · Bernard M. Oliver

```bash
python3 -m ue1000 tritran programs/livermore70v.f70        # Compile & run
python3 -m ue1000 tritran programs/livermore70v.f70 --asm  # Inspect TRIAD-36 code
```

## 1. FORTRAN-66 Heritage
- **80-column fixed-form cards:** `C` / `*` in column 1 = comment; columns 1–5 = numeric statement label; non-blank/`0` in column 6 = continuation; columns 7–72 = statement body (columns 73–80 ignored).
- **Implicit typing & static storage:** `I–N` default to `INTEGER` (`T36`), `A–H, O–Z` default to `REAL` (`TF36`, 47.55-bit mantissa). All variables and 1D/2D column-major arrays are statically allocated; `SUBROUTINE` and `FUNCTION` arguments pass by reference.
- **Formatted Output (`WRITE (6, fmt)` / `PRINT fmt`):** Supports `Iw`, `Fw.d`, `Ew.d`, `nX`, `nH...`, `'...'`, `/`, repeat groups, plus three native ternary descriptors:
  - **`Lw`**: prints `T` (`.TRUE.`), **`U`** (`.UNDECIDED.`), or `F` (`.FALSE.`).
  - **`Bw`**: prints balanced ternary (`42 → 1TTT0`, `-42 → T1110`).
  - **`Hw`**: prints UE heptavintimal (`0, A..M, N..Z`).

## 2. Three-Valued Logic & One-Instruction Arithmetic `IF`
- `LOGICAL` holds `.TRUE.` (`+1`), `.UNDECIDED.` / `.UNKNOWN.` (`0`), or `.FALSE.` (`-1`).
- `.AND.`, `.OR.`, `.EQV.`, `.NOT.`, and `.NEQV.` map directly to single Kleene hardware instructions (`AND`, `OR`, `EQV`, `NEG`).
- **Logical `IF (Q) stmt`** executes `stmt` **only when `Q = .TRUE.` (`+1`)** — ignoring `.UNDECIDED.` (`0`).
- **Arithmetic `IF (E) L1, L2, L3`** emits **`FCM S1, S0`** (for `REAL`, safely routing `UNDECIDED` NaN to `L2`!) or `TST S1`, followed by a **single hardware `B3`** into a 3-word branch table.

## 3. `70V` Vector & T-Fabric Extensions
- **C-Complex Hardware Vector Slices:** `C(1:N) = A(1:N) + B(1:N)` (also `-`, `*`, `.AND.`, `.OR.`, and `FMA(A(1:N), B(1:N))`) automatically strip-mines across any length `N` into `VL ≤ 27` (`3^3`) `VSVL` / `VLD` / `VADD` / `VFMA` / `VST` loops.
- **Hardware Vector Reductions:** `VSUM(A, N)`, `VMAX(A, N)`, `VMIN(A, N)`, and strip-mined **`VDOT(A, B, N)`** (`VLD` + `VMUL` + `VRED` + `FAD`).
- **T-Fabric Dispatch:**
  - `CALL TGEMM(A, B, C, M, N, K, EA, EB, EC, ISTAT)` — dispatches 27-tile `TBFP-9` GEMM with Kelvanov `zgate=+1` and ternary `abft=+1` via `TEXEC A6, S0` / `TWAIT S0, #0`.
  - `CALL TFFT3(XIN, XOUT, IPOW3, EIN, ISTAT)` — dispatches Oliver–Good **Radix-3 `TBFP-9C` Complex FFT** ($N = 3^{\text{IPOW3}}$) on all 27 active tiles!
