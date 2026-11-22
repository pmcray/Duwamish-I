# `SAL-3` — Balanced-Ternary Systems & Vector Language Reference

**Target Architecture:** Duwamish-I (`UE-1000`), `TRIAD-36` Assembly (Rev. 1.5)  
**Designer:** Donald E. Knuth (evolving the UE Special Systems `SALISH` lineage to 36-trit vector and T-Fabric scale)

```bash
python3 -m ue1000 sal3 programs/wald_seti.sal3         # Compile & run on UE-1000
python3 -m ue1000 sal3 programs/wald_seti.sal3 --asm   # Inspect dual-parcel TRIAD-36 code
```

## 1. Strong Type System
Unlike typeless 1967 `SALISH`, `SAL-3` is **strongly typed** so the compiler dispatches `INT`, `FLOAT`, `TRIT`, `DECIBAN`, and `ADDR` operations to the exact C-Complex pipeline and prevents `UNDECIDED` (`NaN`) float polarity bugs at compile time:

| Type | Hardware Width | Semantics & Hardware Mapping |
|---|---|---|
| `INT` | `T36` (`36t`) | Balanced signed integer (`±7.50 × 10^{16}`); `<<` / `>>` emit unbiased ternary `SHF`. |
| `FLOAT` | `TF36` (`30t+6t`) | 47.55-bit mantissa balanced float; `sign` & comparisons always emit **`FCM`** (Rev. 1.5 `UNDECIDED` safety!). |
| `TRIT` | `1t` (`T/0/1`) | Three-valued truth (`false=-1`, `undecided=0`, `true=+1`); Kleene `&`, `|`, `not`, `eqv`; McCarthy `and`, `or`. |
| `DECIBAN` | `DB36` (`81/dB`) | Good weight-of-evidence (`20.0db`); `+`/`-` emit saturating `DADD`/`DSUB`; `lse`, `dexp`, `wald`. |
| `ADDR` | `T18` (`18t`) | Word address (`@var`, array base, `peek`/`poke`, T-Fabric matrix pointers). |

## 2. Three-Valued Control Flow (`sign` → Single `B3`)
`if cond then ...` executes the `then` branch **only when `cond = +1` (`true`)**; `0` (`undecided`) and `-1` (`false`) take the `else` path. Native three-way trichotomy uses `sign`, compiling directly to a single 3-way hardware **`B3`** jump table:

```sal3
sign wald(ev, 20.0db) of
  false     : return false
  undecided : return undecided
  true      : return true
end
```

## 3. Native C-Complex Vector Slices & T-Fabric Dispatch
- **Vector Slices:** `dst[lo..hi] := a[lo..hi] + b[lo..hi]` (also `-`, `*`, `&`, `|`, `fma(a[lo..hi], b[lo..hi])`) automatically strip-mines any slice length into hardware `VL ≤ 27` (`3^3`) `VSVL` / `VLD` / `VADD` / `VFMA` / `VST` loops.
- **Vector Reductions:** `vsum(arr, n)`, `vmax(arr, n)`, `vmin(arr, n)`, `vlse(arr, n)` compile directly to `VLD` + `VRED`.
- **T-Fabric Dispatch:** `tfabric_gemm(A, B, C, M, N, K, eA, eB, eC)` populates a 9-word `TTD` (`TBFP-9`, Kelvanov `zgate=+1`, ternary `abft=+1`) and emits **`TEXEC A6, S0`** + **`TWAIT S0, #0`** (Rev. 1.5 `S0` all-27-tiles shorthand), returning the hardware `TRIT` status (`+1 OK`, `0 TIMEOUT`, `-1 RECOVERED_BY_CUTOVER`).

## 4. Hofstadter `bloop` Termination Certification
Placing `bloop` at the top of a file statically certifies that `main()` and all reachable procedures contain zero `while` loops, zero recursion, and zero loop-variable mutation — proving guaranteed termination before code generation.

## 5. `SAL-3/O` Peephole & Cray Dual-Parcel Packer
Enabled by default (`--no-opt` to compare plain output):
1. Collapses `PUSH S1 ; POP S2` into single 18-trit `ADD S2, S1, #0`, drops redundant reloads (`ST`/`LD`) and dead branches.
2. Pairs consecutive 18-trit short parcels (`F ∈ {0, T}`) into dual-parcel 36-trit words (`instA | instB`), increasing Cray 27-word Instruction Stack residency.
