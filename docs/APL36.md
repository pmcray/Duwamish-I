# APL\36 — Balanced-Ternary Array & Tensor Language Manual

**System:** Duwamish-I (`UE-1000`, Rev. 1.5) — OBB Unseen Level -9  
**Lineage:** Kenneth E. Iverson (*A Programming Language*, 1962; IBM `APL\360`, 1966) adapted for native 36-trit balanced-ternary vectors, Kleene logic, and 27-tile T-Fabric tensor ops  
**Implementation:** [ue1000/apl36.py](../ue1000/apl36.py)  
**Character Encoding:** All 32 core APL glyphs are mapped natively into `OBB-729` trytes `+193..+224` ([ue1000/ternary.py](../ue1000/ternary.py)).

---

## 1. Hardware-Backed Array Execution

`APL\36` executes interactively or in batch directly atop Duwamish-I's Tiers 1 and 2:

1. **Tier 1 (`C-Complex` 27-Lane Vector Pipes):**
   On workspace startup, `APL\36` assembles and loads native `TRIAD-36` strip-mined vector micro-kernels (`_APL_VADD`, `_APL_VSUB`, `_APL_VMUL`, `_APL_VKAND`, `_APL_VKOR`, `_APL_VKEQV`, `_APL_VRSUM`, `_APL_VRMAX`, `_APL_VRMIN`, `_APL_VDOT`, `_APL_VWALD`) into `UE-1000` memory (`+1000`). Vector arithmetic (`+`, `-`, `×`), Kleene logic (`∧`, `∨`, `≡`), reductions (`+/`, `⌈/`, `⌊/`), vector dot products (`+.×`), and Wald tests (`⎕WALD`) run directly on `machine.cpu0` in exact `TF36` / `VKLEEN` / `DWALD` hardware, recording real cycles and Cray Instruction Stack hits (`⎕CYCLE`).
2. **Tier 2 (`T-Fabric` 27-Tile Systolic & Spectral Array):**
   Rank-2 inner products (`A +.× B`) dispatch directly onto `machine.tfabric` according to `⎕TMODE`:
   - `⎕TMODE ← 1` *(default)*: **`TBFP-9`** systolic GEMM with Kelvanov zero-trit gate-suppress & ternary ABFT
   - `⎕TMODE ← 2`: **`TBFP-3`** triad-packed 3× throughput GEMM
   - `⎕TMODE ← 3`: **`TB1`** Rosenblatt–Kelvanov pure-trit associative mode
   - `⎕TMODE ← 0`: Exact **`TF36`** (`47.55-bit` mantissa) on Tier 1 C-Complex `VDOT`
   - `⎕FFT X`: Oliver–Good **Radix-3 `TBFP-9C` FFT** on `T-Fabric`
   - `A ⎕TB1 B`: Explicit Mode `+3` pure-trit associative inner product

---

## 2. Core Syntax & Right-to-Left Rules

- **Strict Right-to-Left Evaluation:** Zero operator precedence except parentheses `(...)`. Thus `2 × 3 + 4` equals `14`.
- **High-Minus `¯`:** Negative literals use `¯` (`¯1 0 1`, `¯3.5E¯2`), keeping `-` strictly as subtraction/negation.
- **Literals:** Strands (`1 2 3`), balanced-ternary (`3t1T01`), Good decibans (`12.5db` → ticks), `OBB-729` strings (`'SETUN'`), comment `⍝` (or `NB.`), statement separator `⋄`.
- **ASCII Teletype Digraphs:** `<-` (`←`), `->` (`→`), `<=` (`≤`), `>=` (`≥`), `!=` (`≠`), `~` (`∼`), `|` (`∣`), `o.` (`∘.`).

---

## 3. Primitives, Operators & Ternary Extensions

| Category | Monadic | Dyadic |
|---|---|---|
| **Arithmetic** | `+` `-` `×` *(balanced signum `¯1/0/1`!)* `÷` `*` `⍟` `∣` `⌈` `⌊` `!` `○` `?` | `+` `-` `×` `÷` `*` `⍟` `∣` `⌈` `⌊` `!` `○` *(1=sin, 2=cos, 7=tanh)* |
| **Kleene / Relational** | `∼` / `¬` *(auto Kleene `-x` on trits or `⎕KL=1`)* | `< ≤ = ≥ > ≠` *(NaN/`⎕U` → `0` UNDECIDED)* · `∧` *(min)* `∨` *(max)* `≡` *(product)* `⍲` `⍱` |
| **Structural** | `⍴` `,` `⍳` *(honors `⎕IO ∈ {¯1,0,1}`!)* `⌽` `⊖` `⍉` `⍋` `⍒` `⍕` `⍎` | `⍴` `,` `↑` `↓` `⌽` `⊖` `⍉` `⍳` `∈` `/` `\` `⌿` `⍀` |
| **Linear & Radix** | `⌹` *(matrix inverse)* · `⊤` *(minimal balanced trits)* · `⊥` *(decode balanced trits)* | `⌹` *(linear solve `b ⌹ A`)* · `⊤` *(`¯3` = balanced-ternary fixed width!)* · `⊥` |
| **Operators** | Reduction `f/` `f⌿`, Scan `f\` `f⍀` (optional `[axis]`) | Inner Product `f.g` (`+.×`, `⌈.+`, `∨.∧`), Outer Product `∘.f`, Windowed `N f/ V` |
| **Duwamish-I `⎕`** | `⎕H` *(Heptavintimal)*, `⎕OBB` *(tryte codes)*, `⎕DB` *(deciban ticks)*, `⎕LSE`, `⎕FFT` | `T ⎕WALD V` *(hardware `DWALD`)*, `A ⎕TB1 B` *(Mode `+3` associative)* |

### System Variables & Commands
- **Variables:** `⎕IO` (`1`, `0`, or `¯1` balanced centered indices), `⎕KL` (`0` classic / `1` strict `¯1/0/1` relations), `⎕TMODE` (`0..3`), `⎕PP`, `⎕CYCLE`, `⎕TOPS`, `⎕U` (`UNDECIDED`), `⎕←` (output).
- **Commands:** `)VARS`, `)FNS`, `)HW` (prints live Tier 1 & Tier 2 counters), `)CLEAR`, `)OFF`.
- **Del-Editor Functions:** `∇ Z ← A FN B ; L1 ; L2` ... labels `LP:` ... branches `→(COND)/LP`, `→0` ... closing `∇`.

---

## 4. CLI & Reference Programs

```bash
python3 -m ue1000 apl programs/apl_tour.apl        # Primes, Kleene tables, balanced ⊤/⊥, Hilbert ⌹, ∇
python3 -m ue1000 apl programs/saew_attention.apl  # 8-line Rosenblatt-Kelvanov attention head on T-Fabric
python3 -m ue1000 apl programs/oliver_dft_apl.apl  # Radix-3 TBFP-9C FFT + hardware DWALD SETI pipeline
python3 -m ue1000 apl --repl                       # Interactive APL\36 console
```
