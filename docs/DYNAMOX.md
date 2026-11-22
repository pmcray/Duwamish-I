# DYNAMO-X — Forrester / Ashby System Dynamics & Cliodynamics Language Manual

**System:** Duwamish-I (`UE-1000`, Rev. 1.5) — OBB Unseen Level -9  
**Designers:** Jay W. Forrester (MIT System Dynamics) & W. Ross Ashby (Ultrastability Diagnostics)  
**Compiler & Runtime:** [ue1000/dynamox.py](../ue1000/dynamox.py)  
**Target:** Native `TRIAD-36` assembly (`TF36` 30-trit / 47.55-bit unbiased balanced-ternary floating-point on Tier 1 `C-Complex`, plus optional multi-region spatial coupling on Tier 2 `T-Fabric` `TBFP-9` GEMM with ternary ABFT).

---

## 1. Purpose & Historical Lineage

`DYNAMO-X` is Duwamish-I's native descendant of MIT's **DYNAMO II** (Pugh 1970; Forrester *World Dynamics* 1971; Meadows et al. *Limits to Growth* 1972), promised in Section 6 of [PROJECT_PLAN.md](PROJECT_PLAN.md#L320) for Workload 3 (OBB Cliodynamics).

Unlike imperative FORTRAN loops, DYNAMO-X models are **declarative equation decks**:
- Authors write stock-and-flow differential equations (`L` Levels, `R` Rates, `A` Auxiliaries, `N` Initializers, `C` Constants, `T` Lookup Tables) in **any order**.
- The compiler automatically expands Forrester delay macros (`SMOOTH`, `DELAY1`, `DELAY3`), **topologically sorts** all equations, rejects algebraic simultaneous loops without a Level/delay, and emits optimized, dual-parcel-packed **`TRIAD-36`** code running directly on `CComplexCPU` in exact **`TF36`**.
- Because balanced-ternary `TF36` rounding is **exact and tie-free unbiased**, 100–200 year stiff integrations avoid systematic IEEE/IBM-hexadecimal rounding bias.

---

## 2. Equation Cards & Syntax

| Card | Form | Meaning |
|---|---|---|
| `*` / `NOTE` | `* Title comment` | Comment / model title card |
| `L` | `L POP.K = POP.J + DT * (BIRTH.JK - DEATH.JK)` | **Level** (state stock); supports `.K`/`.J`/`.JK`/`.KL` time subscripts |
| `N` | `N POP = 1.00` | **Initial** condition (topologically ordered at `T = TSTART`) |
| `A` | `A REFF.K = TABHL(TRES, RES.K, 0.0, 1.5, 0.25)` | **Auxiliary** algebraic variable (topologically sorted) |
| `R` | `R BIRTH.KL = POP.K * 0.058 * REFF.K` | **Rate** (flow evaluated over interval `.KL`) |
| `C` | `C CONS = 0.05` | **Constant** parameter (patched in-place in `T-DRAM` for `RERUN`/`SWEEP`) |
| `T` | `T TRES = 0.02 / 0.32 / 0.65 / 0.85 / 1.00` | Piecewise-linear lookup table (compiled into `_DYN_TABHL`) |
| `SPEC` | `SPEC DT=1.0, LENGTH=120.0, TSTART=1970.0, PRTPER=20.0, PLTPER=8.0, METHOD=RK2, REGIONS=6` | Integration cadence, `EULER` or `RK2` midpoint solver, and spatial region count (`1..27`) |
| `PRINT` | `PRINT POP, CAP, RES, POL, LEG` | Tabular report columns |
| `PLOT` | `PLOT POP=P, RES=R, POL=Z, LEG=L` | Authentic 1971 line-printer / CRT time-series plot |
| `RERUN` | `RERUN SCENARIO NAME` + `C ...` overrides | Counterfactual rerun with zero recompilation |

### Multi-Line Continuation
Leading `X ` in column 1 (classic MIT DYNAMO II continuation card) or trailing `\` joins lines.

---

## 3. Forrester Macros & Built-in Functions

- **`TABHL(TNAME, x, xlo, xhi, xinc)` / `TABLE(...)`** — Clamped piecewise-linear interpolation compiled into the native `TRIAD-36` `_DYN_TABHL` routine (using balanced-ternary `FIX` + floor adjustment and `FMA`).
- **`SMOOTH(input, delay)`** — Synthesizes a 1st-order exponential smoothing Level (`_SMk`).
- **`DELAY1(input, delay)`** — Synthesizes a 1st-order material pipeline delay Level.
- **`DELAY3(input, delay)`** — Synthesizes a **3rd-order Forrester Erlang cascade** (`_D3_k_1`, `_D3_k_2`, `_D3_k_3`), capturing authentic phase lag and overshoot dynamics.
- **`CLIP(p, q, a, b)`** (`p` if `a >= b` else `q`), **`SWITCH(p, q, s)`** (`p` if `s == 0` else `q`), **`STEP(h, t0)`**, **`RAMP(slope, t0)`**.
- **`MAX`**, **`MIN`**, **`ABS`**, **`SQRT`**, **`EXP`**, **`LOG`**, **`LOG10`**, **`SIN`**, **`COS`**, **`TANH`**, **`ATAN`**, and **`SIGN3(x)`** (native Rev. 1.5 `FCM` against zero returning `-1.0 / 0.0 / +1.0`).

---

## 4. Duwamish-I `-X` Cybernetic Extensions (Ashby / Beer / T-Fabric)

1. **Multi-Region Spatial Coupling (`REGIONS N` + `COUPLE VAR = k`):**
   Allocates an `N`-region state array per variable (`1..27`) and dispatches inter-regional Laplacian trade/resource diffusion matrices onto Tier 2 **`TFabric.gemm_tbfp9`** (`TBFP-9` systolic GEMM with ternary ABFT) at each step.
2. **Ashby Essential Variable Corridors (`ASHBY VAR IN [lo, hi]`):**
   Evaluates every run against Ashby's viability bounds and emits a native Kleene three-valued verdict:
   - **`+1` (`VIABLE EQUILIBRIUM`)** — All essential variables remain inside corridor and settle smoothly.
   - **`0` (`METASTABLE / OSCILLATORY — UNDECIDED`)** — Inside bounds at end, but oscillating or transiently breached.
   - **`-1` (`OVERSHOOT / COLLAPSE`)** — Essential variable collapsed outside corridor or diverged.
3. **Attractor-Basin Bifurcation Sweeps (`SWEEP P1 = a..b STEP s [, P2 = c..d STEP t]`):**
   Patches constant slots directly in `UE-1000` `T-DRAM` across a 1D or 2D policy grid and renders a native balanced-ternary (`+` / `0` / `-`) phase-space viability diagram.

---

## 5. CLI & Reference Programs

```bash
python3 -m ue1000 dynamo programs/world2_forrester.dyn       # World2/3 + DELAY3 + 3 RERUNs + ASCII plot
python3 -m ue1000 dynamo programs/ashby_sweep.dyn            # 6-region TBFP-9 coupled 2D bifurcation map
python3 -m ue1000 dynamo programs/predator_delay.dyn --asm   # Inspect compiled TRIAD-36 assembly
```
