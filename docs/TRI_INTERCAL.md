# TRI-INTERCAL — Unsanctioned Grad-Student Gag Language (Reel `SS-GAG-72`)

**Status:** **STRICTLY UNOFFICIAL / OFF-LEDGER** *(found on 9-track tape reel `SS-GAG-72`, disguised as `DIAG-CORE-TEST` in the OBB Level -9 Maintenance/Staging Bay — DO NOT SHOW TO F. P. BROOKS)*  
**Origin:** West Seattle graduate-student port of Don Woods & James M. Lyon's **INTERCAL-72** (Princeton University, May 26, 1972) to Duwamish-I's native balanced-ternary hardware  
**Implementation:** [ue1000/tri_intercal.py](../ue1000/tri_intercal.py)

---

## 1. Balanced-Ternary Politeness Enforcement

Like Princeton's INTERCAL-72, `TRI-INTERCAL` enforces strict manners using balanced-ternary polarity over the ratio `p = PLEASE / total_statements`:
- **`p < 1/5` → `SPL-079 PROGRAMMER IS INSUFFICIENTLY POLITE (POLITENESS TRIT = -1)`**
- **`p > 1/3` (for ≥ 4 statements) → `SPL-099 PROGRAMMER IS OVERLY POLITE (POLITENESS TRIT = +1)`**

---

## 2. Ternary Data Types, Operators & Hardware Execution

| Construct | Meaning on Duwamish-I (`UE-1000`) |
|---|---|
| `.n` *(spot)* / `,n` *(tail)* | 18-trit Setun half-word (`T18`, `±193,710,244`) scalar / 1D array |
| `:n` *(two-spot)* / `;n` *(hybrid)* | 36-trit full `UE-1000` word (`T36`) scalar / 1D array |
| `#n` *(mesh)* | Non-negative constant `0..65535` |
| `a ¢ b` *(or `$`)* — **Mingle** | Interleaves two 18-trit spots into one 36-trit two-spot |
| `a ~ b` — **Select** | Balanced signed selection (`+1` keeps trit, `-1` negates trit, `0` drops) |
| `.&n`, `.Vn`, `.?n`, `.¬n` | Hardware `AND`/`OR`/`EQV` (1-trit right circular rotation) & `NEG` on `CComplexCPU`! |
| `READ OUT .1` | Authentic Roman numerals (`_` for `0`, **`ANTI-`** prefix for negative trits!) |
| `READ OUT ,1` | 6-trit **`OBB-729` Balanced-Ternary Turing Tape** text output! |
| `DO (1000)..(1050) NEXT` | Built-in Woodworth Syslib (`1000` Add, `1010` Sub, `1020` Mul, `1030` Div, `1040` T36 Add, `1050` Neg) |
| `DO COME FROM (lbl)` | **Anachronism-gated (`SPL-444`)** until Clark's Dec 1973 *Datamation* paper — pass `--1974` or `PLEASE ALLOW 1974`! |

---

## 3. CLI & Smuggled Tape Programs

```bash
python3 -m ue1000 intercal programs/hello_woodworth.i3          # Polite Level -9 Turing-Tape Hello World
python3 -m ue1000 intercal programs/grad_gag_add.i3             # Mingle/Select, ABSTAIN, STASH & Roman numerals
python3 -m ue1000 intercal programs/comefrom_1974.i3 --1974     # Post-commissioning Clark 1973 COME FROM loop
```
