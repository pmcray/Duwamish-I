# Duwamish-I (`UE-1000`) — Principles of Operation

**Document Control:** Universal Entropics — Special Systems Section, Form `SS-1000-01`  
**Revision:** **Rev. 1.5 (Conformance-Audited Freeze — Incorporating Q4 1971 Byurakan `ECO-71-09` & ISA Errata Patch)**  
**Sponsor & Charter Authority:** **WEAR** (Woodworth's UARC/FFRDC, A. J. Woodworth III University, West Seattle)  
**Chief System Architect & Final-Draft Editor:** Frederick P. Brooks Jr.  
**Word/ISA Co-Signatories:** Donald E. Knuth · Seymour R. Cray  
**Subsystem Architecture Sign-Off:** I. J. Good · Frank Rosenblatt · W. Ross Ashby · Stafford Beer · Gordon Pask · Jay W. Forrester · Daniel L. Slotnick · Bernard M. Oliver · Gordon E. Moore · J. Ulysses Jameson *(with J. C. R. Licklider, external consultant)*  
**Installation Site:** Unseen Level -9, Oral Bland Building (OBB), West Seattle — **28 Structural Cells + 2 Central Core Cells**  

> [!IMPORTANT]
> **Normative Engineering Specification:** This manual defines the exact, bit-exact (**trit-exact**) programming, arithmetic, memory, fault-tolerance, and hardware-interface architecture of the **Duwamish-I (`UE-1000`)**. Every register width, trit field, parity-check row, timing parameter, descriptor word, and cutover state transition is locked to the **Revision 2 Project Plan** (`1.000 G$` 1967 USD, Q2 1973 commissioning) and serves as the authoritative conformance contract for compilers (`SAL-3`, `TRI-TRAN 70V`, `TRIAD-36`), the `C-EXEC` / `V-OS` executives, and the cycle-approximate simulator (`duwamish_ue1000`).

---

## Chapter 1. Architectural Overview & System Organization

### 1.1 The Balanced-Ternary Charter & Conceptual Integrity
Mainline **Universal Entropics** builds denary (base-10) commercial and institutional data-processing systems. The **UE Special Systems Section** exists under WEAR charter specifically to design and fabricate **native balanced-ternary (`-1, 0, +1`, written `T, 0, 1`)** scientific and cybernetic supercomputers, extending the Soviet **Setun** (1958) and **Setun-70** (1970) ternary lineage and implementing Kleene's strong three-valued logic (`FALSE [-1] | UNDECIDED [0] | TRUE [+1]`) directly in hardware.

Under Chief Architect Frederick P. Brooks Jr.'s **anti-CAMEL rule**, every hardware mechanism in the `UE-1000` serves at least two of the facility's three anchor missions (**`SAEW` Library-Scale Autoregressive AI**, **Oliver–Good Wideband SETI**, and **Forrester–Ashby OBB Cliodynamics**). Four strictly orthogonal tiers prevent feature creep and preserve deterministic compute timing:

| Tier | Designation | Lead Architects | Physical Hardware & Clock | Architectural Role |
|---|---|---|---|---|
| **Tier 1** | **C-Complex** | Cray, Knuth, Forrester | 8 symmetric 36-trit hardwired vector–scalar CPUs (`UE-TECL`, **30 ns / 33.33 MHz**) | High-precision `TF36` stiff ODE integration, scalar/symbolic indexing, Good deciban sequential tests, and deterministic T-Fabric job dispatch (`C-EXEC`). |
| **Tier 2** | **T-Fabric** | Rosenblatt, Slotnick, Oliver | 30 liquid-cooled systolic tiles (**27 active + 3 hot spares**); $6 \times 6 = 36$ `TBFP-9` MACs/tile (`UE-TECL`, **40 ns / 25 MHz**) | Dense `TBFP-9` matrix/neural training (**48.6 GOPS peak / 17.0 GOPS sustained**), radix-3/6 complex FFTs, `TBFP-3` inference, and Rosenblatt–Kelvanov pure-trit associative matching. |
| **Tier 3** | **Hierarchical Store** | Forrester, Moore, Jameson, Oliver | **3a:** $3^{16} = 43{,}046{,}721$ `T36` words (`UE-TD2000` T-DRAM, 81 active + 9 spare banks, 240 ns, 42-trit SEC-DED) + $3^{11} = 177{,}147$ dual-core ferrite words (600 ns). **3b:** 32 ternary discs (`768M` words) + Alexandrian archival silo (~100 GB equiv). | Uniform 81-way interleaved crossbar (**1.35 Gwords/s = 9.63 GB/s equiv**) feeding C-Complex, T-Fabric, and Jameson's Level -9 Alexandrian Vault. |
| **Tier 4** | **V-Complex** | Beer, Ashby, Pask, Licklider | 4 fault-tolerant 36-trit supervisory nodes (`V-OS`), out-of-band **Algedonic Telemetry Bus**, 64 interactive vector-CRT consoles | Virtual memory, WEAR capability security, Beer's VSM System 3–5 health monitoring, `< 75 µs` Ashby hot-spare cutover, and interactive researcher consoles. |

### 1.2 Physical Stratigraphy & Unseen Level -9 Cell Assignment
The `UE-1000` occupies **Unseen Level -9** of the Oral Bland Building (**2 Central Core cells** for lifts, fire stairs, WCs, and utility chases + **28 reinforced structural cells**):
- **2 Cells — Compute Vault:** Hardened copper-shielded octagonal bay directly adjacent to the central core and over Level -10 Freon/power risers; houses C-Complex, 30 T-Fabric tiles, and the 81-bank Ternary Crossbar (`< 2.5 m` compute cable runs).
- **6 Cells — Main Ternary Memory Vault (Tier 3a):** 90 `UE-TD2000` banks + dual-core ferrite Executive vault ringing the Compute Vault (`< 5 m` runs).
- **8 Cells — The Alexandrian Vault (Tier 3b, J. Ulysses Jameson):** 32 ternary disc spindles, Ampex TBM & Photo-Digital archival silos, 32 nine-track ternary tape drives, and the receiving lock for the vertical digitization feed shaft from **Level -8 (Library Staff & Digitisation)** and **Level -7 (Special Collections)** above.
- **4 Cells — V-Complex & Beer's Operations Gallery (Tier 4):** 4 supervisory nodes, glass-walled algedonic wall scopes, and spare-tile maintenance annunciators.
- **3 Cells — Researcher Terminal Suites:** Acoustically isolated interactive vector-CRT/light-pen bays.
- **3 Cells — Oliver SETI Baseband Annex:** Dedicated Faraday-isolated digitizer, atomic frequency standard, and baseband tape rooms.
- **2 Cells — UE Special Systems Maintenance & Staging Bay:** Live blade-swap test rigs, Freon purge manifold, and spare-module vault.

*Vertical Plant & Isolation Boundary:* **Level -10** directly below supplies 2.6 MW dual conditioned power, flywheel/diesel UPS, primary Freon condensers, and cryogenic test support up into Level -9. Below Level -10, a **30 m solid rock gap** provides absolute electrical, RF, and air-gap isolation from **Level -11 (Experimental Hall)** and **Level -12 (WEAR Bunker)**, which operate an independent, out-of-scope two-machine stack (a ruggedized mainline **UE Denary** life-support/archive controller plus a compact Special Systems balanced-ternary array wired directly to Level -11). No `UE-1000` cable, bus, or address aperture crosses the 30 m rock gap.

---

## Chapter 2. Number Systems, Data Formats & Character Encoding

### 2.1 Balanced-Ternary Notation & UE Heptavintimal Alphabet
A **trit** takes values in $\{-1, 0, +1\}$, written **`T`**, **`0`**, **`1`**. Each trit carries $\log_2 3 \approx 1.5849625\text{ bits}$. Trits within an $n$-trit quantity are indexed from **0 (least significant, weight $3^0$)** to **$n-1$ (most significant, weight $3^{n-1}$)**:
$$v = \sum_{i=0}^{n-1} t_i \cdot 3^i, \qquad v \in \left[-\frac{3^n - 1}{2},\; +\frac{3^n - 1}{2}\right]$$

Three fundamental arithmetic identities hold in hardware across every width:
1. **Sign-Free Negation:** $-v$ is formed by parallel tritwise inversion (`T ↔ 1`, `0 ↔ 0`) with zero carry propagation and no negative zero.
2. **Exact Unbiased Rounding by Truncation:** Dropping the $k$ least-significant trits of $v$ (arithmetic right shift by $k$) computes $\operatorname{round}(v / 3^k)$ to the **unique nearest integer**. Because $3^k$ is always odd, half-way ties **cannot occur**, and statistical rounding bias is identically zero.
3. **Single-Trit Trichotomy:** The sign of any non-zero balanced-ternary number is the value of its highest non-zero trit (`T` = negative, `0` = zero, `1` = positive).

Three contiguous trits form one **triad**, printed in UE listings using the house **heptavintimal** (balanced base-27) alphabet (`0`, `A..M` = `+1..+13`, `N..Z` = `-13..-1`):

| Val | Trits | Ch | Val | Trits | Ch | Val | Trits | Ch |
|---:|:---:|:---:|---:|:---:|:---:|---:|:---:|:---:|
| `-13` | `TTT` | **`N`** | `-4` | `0TT` | **`W`** | `+5` | `1TT` | **`E`** |
| `-12` | `TT0` | **`O`** | `-3` | `0T0` | **`X`** | `+6` | `1T0` | **`F`** |
| `-11` | `TT1` | **`P`** | `-2` | `0T1` | **`Y`** | `+7` | `1T1` | **`G`** |
| `-10` | `T0T` | **`Q`** | `-1` | `00T` | **`Z`** | `+8` | `10T` | **`H`** |
| `-9` | `T00` | **`R`** | `0` | `000` | **`0`** | `+9` | `100` | **`I`** |
| `-8` | `T01` | **`S`** | `+1` | `001` | **`A`** | `+10` | `101` | **`J`** |
| `-7` | `T1T` | **`T`** | `+2` | `01T` | **`B`** | `+11` | `11T` | **`K`** |
| `-6` | `T10` | **`U`** | `+3` | `010` | **`C`** | `+12` | `110` | **`L`** |
| `-5` | `T11` | **`V`** | `+4` | `011` | **`D`** | `+13` | `111` | **`M`** |

### 2.2 Word & Sub-Word Hierarchy (Knuth–Cray Sign-Off)
The fundamental machine word is **36 trits (`T36`, 57.059 bits)**, partitioning without remainder into every Setun, character, and T-Fabric sub-unit:

```text
Trit Index:  35       30 29       24 23       18 17       12 11        6 5         0
            +-----------+-----------+-----------+-----------+-----------+-----------+
T36 Word:   |  Tryte 5  |  Tryte 4  |  Tryte 3  |  Tryte 2  |  Tryte 1  |  Tryte 0  |  (6 trytes = 12 hept. digits)
            +-----------+-----------+-----------+-----------+-----------+-----------+
T18 Half:   |<---- High Setun Half-Word (H1) --->|<---- Low Setun Half-Word (H0) --->|  (2x 18t Setun-compatible)
            +-----------------+-----------------+-----------------+-----------------+
T9 Nonet:   |    Nonet 3      |    Nonet 2      |    Nonet 1      |    Nonet 0      |  (4x 9t TBFP-9 / Token IDs)
            +-----------------+-----------------+-----------------+-----------------+
```

| Format Name | Width | Exact Signed Range | Bit Equiv. | Packing / `T36` | Primary Architectural Use |
|---|---:|---|---:|---:|---|
| **`T6` (Tryte)** | 6 trits | `[-364, +364]` ($3^6 = 729$) | 9.510 b | 6 | `OBB-729` character code; Setun-70 syllable; ECC check tryte. |
| **`T9` (Nonet)** | 9 trits | `[-9,841, +9,841]` ($3^9 = 19{,}683$) | 14.265 b | 4 | `TBFP-9` mantissa; `SAEW` subword token ID; Setun (1958) half-word. |
| **`T18` (Half-Word)** | 18 trits | `[-193,710,244, +193,710,244]` | 28.529 b | 2 | Compact instruction parcel; Setun/Setun-70 word; complex `TBFP-9` pair; `TF18` float; physical address. |
| **`T27` (Accum.)** | 27 trits | `[-3,812,798,742,493, +3,812,798,742,493]` | 42.794 b | internal | T-Fabric exact MAC accumulator (zero overflow over $10^4+$ terms). |
| **`T36` (Full Word)** | 36 trits | `[-75,047,317,648,499,560, +75,047,317,648,499,560]` | 57.059 b | 1 | C-Complex scalar/vector register; `TF36` float; memory word (stored as 42t with 6t ECC). |

### 2.3 Floating-Point Formats (`TF36` and `TF18`)
To guarantee that **word sign equals floating-point value sign** (so scalar `TST` and `J3` test float polarity in 1 cycle with zero field unpacking!), both `TF36` and `TF18` place the **mantissa in the high trits** and the **balanced exponent in the low trits**:

```text
TF36 (36 trits):  |        mantissa : 30 trits (35..6)        | exp : 6 trits (5..0) |
TF18 (18 trits):  |   mantissa : 13 trits (17..5)   | exp : 5 trits (4..0) |
```

- **Value Representation:**
  - For **`TF36`**: $v = m \cdot 3^{e - 29}$, where normalized $m$ has leading trit $m_{29} \in \{-1, +1\}$ (so $\frac{3^{29}+1}{2} \le |m| \le \frac{3^{30}-1}{2}$, yielding **47.55 bits** of precision — matching Cray's CDC 7600 48-bit mantissa) and $e \in [-363, +363]$ (dynamic range $\approx 10^{\pm 173.2}$).
  - For **`TF18`**: $v = m \cdot 3^{e - 12}$, where normalized $m$ has $m_{12} \in \{-1, +1\}$ (**20.60 bits** of precision) and $e \in [-120, +120]$ (dynamic range $\approx 10^{\pm 57.3}$). Two `TF18` floats pack into one `T36` word for vector `TF18` ops and `SAEW` master weights/momentum.
- **Canonical Special Encodings (Kleene-Compatible):**
  - **Canonical Zero (`0`):** `mant = 0, exp = 0` (entire word is all `0` trits).
  - **Saturated Infinity (`+INF` / `-INF`):** `mant = ±M_max`, `exp = +364` (`TF36`) or `+121` (`TF18`). Produced on overflow when `PSW.FT` (Float Trap Enable) is `0`; raises Trap 7 (`FPOVF`) when `PSW.FT = +1`.
  - **`UNDECIDED` / `NaN` (`U`):** `mant = 0`, `exp = -364` (`TF36`) or `-121` (`TF18`). Produced by `0 / 0`, `INF - INF`, `√(-x)`, or Kleene propagation. Any arithmetic operation with an `UNDECIDED` operand yields `UNDECIDED` (silent) or Trap 8 (`FPUND`) if `PSW.FT = +1`. Floating comparison `FCM` against `UNDECIDED` sets `C = 0` (`UNDECIDED`) and sets `PSW.FU = +1`.
- **Rounding Law:** Every `TF36` and `TF18` operation computes the exact mathematical result internally (with 4 guard trits), normalizes so the leading mantissa trit is non-zero, and **truncates once** — which in balanced ternary is exact, tie-free, unbiased round-to-nearest! `FNEG` inverts only the high mantissa trits (`35..6` for `TF36`, `17..5` for `TF18`).
- **Sign Invariant & `UNDECIDED` Caveat (Normative):** For every **finite** and **infinite (`±INF`)** floating-point value, $|m| \ge \frac{3^{29}+1}{2}$ (`TF36`) or $\frac{3^{12}+1}{2}$ (`TF18`), so the high mantissa strictly dominates the low exponent and `sign(word) == sign(float)` holds identically (allowing 1-cycle scalar `TST` polarity checks). **However**, `UNDECIDED` (`mant = 0, exp = -364` / `-121`) has raw signed integer value `-364` (`-121`), so scalar `TST` returns `C = -1` (negative), *not* `0` (`UNDECIDED`). Whenever an operand may be `UNDECIDED`, software **must** use **`FCM Ss1, S0`** (2 cycles), which explicitly detects `UNDECIDED`, sets `C = 0`, and latches `PSW.FU ← +1`.

### 2.4 T-Fabric Block-Floating-Point & Associative Formats
| Format | Element Width | Elements / `T36` | Shared Block Exponent | Exact Range & Equivalent Precision |
|---|---:|---:|---|---|
| **`TBFP-9`** | 9 trits | **4** | `E_blk` (6 trits, `±364`, per $6\times 6$ block) | $m_i \in [-9{,}841, +9{,}841]$, value $= m_i \cdot 3^{E_{\text{blk}} - 8}$ (**14.26 bits**, ~86 dB SFDR). Primary training & SETI FFT format. |
| **`TBFP-9C`** | 18 trits (`9t Re \| 9t Im`) | **2** | `E_blk` (6 trits, shared Re/Im) | Complex baseband/FFT sample; 2 complex samples per `T36` word. |
| **`TBFP-3`** | 3 trits (1 triad) | **12** | `E_blk` (6 trits) | $m_i \in [-13, +13]$ (**4.75 bits**, 27 levels). 3× throughput interactive `SAEW` inference & coarse SETI correlation. |
| **`TB1`** | 1 trit (`T, 0, 1`) | **36** | `E_blk` (6 trits) | Pure balanced-ternary synaptic weights (`-1, 0, +1`). Rosenblatt–Kelvanov associative inner-product & card-catalogue matching mode. |

### 2.5 Good Deciban Fixed-Point (`DB18` / `DB36`)
To accelerate I. J. Good's sequential probability-ratio tests (SETI) and log-odds attention normalization (`SAEW`), the C-Complex includes a hardware **Deciban Unit**:
- **Scale:** $1\text{ deciban} = 3^4 = 81\text{ ticks}$ (`1 tick = 0.012346 dB`, where $+10\text{ dB} = 10\times$ odds ratio).
  - **`DB18`** (packed 2 per `T36`): spans $\pm 2.39 \times 10^6\text{ dB}$ with `0.012 dB` precision.
  - **`DB36`** (full word): spans $\pm 9.26 \times 10^{14}\text{ dB}$.
- **Saturating Semantics:** `DADD` / `DSUB` clamp cleanly to `±WMAX` on overflow rather than wrapping (preventing catastrophic sign reversal on overwhelming evidence!).
- **Hardware Transcendental Assist:** A dedicated **243-entry (`3^5`) ROM lookup + linear interpolator** executes **`DLSE`** ($\operatorname{log-sum-exp}$ in decibans: $10\log_{10}(10^{a/10} + 10^{b/10})$) in **3 cycles** and **`DEXP`** ($10^{a/10} \to \text{TF18/TBFP-9}$) in **3 cycles** to $< 0.005\text{ dB}$ max error.

### 2.6 Kleene Strong Three-Valued Logic
All tritwise logical operations execute across all 36 trits simultaneously in **1 clock cycle** (`T = -1 = FALSE`, `0 = UNDECIDED`, `1 = TRUE`):

| `a \ b` | `AND (min)` `T / 0 / 1` | `OR (max)` `T / 0 / 1` | `EQV (a·b)` `T / 0 / 1` | `NOT (-a)` |
|:---:|:---:|:---:|:---:|:---:|
| **`T` (-1)** | `T / T / T` | `T / 0 / 1` | `1 / 0 / T` | **`1`** |
| **`0` (0)** | `T / 0 / 0` | `0 / 0 / 1` | `0 / 0 / 0` | **`0`** |
| **`1` (+1)** | `T / 0 / 1` | `1 / 1 / 1` | `T / 0 / 1` | **`T`** |

The **`SEL rd, rs1, #t`** instruction extracts trit number `(C + 1) ∈ {0, 1, 2}` from the low triad of `rs1` (or `#t`), implementing any arbitrary 1-trit unary function or relational truth table (`<, =, >, ≤, ≥, ≠`) in 1 cycle with zero branches.

### 2.7 The `OBB-729` Tryte Character Set (Jameson–Knuth–Licklider)
Each **6-trit tryte** (`-364 .. +364`) encodes one character or control symbol such that **one `TST` / `J3` instruction partitions the entire code space**:
- **`0` (`00`):** `NUL` / String Terminator / Zero-Pad.
- **`+1 .. +364` (Printable Glyphs — `TST` yields `+1`):**
  - `+1 .. +95`: Printable ASCII-1968 Latin, digits, punctuation (`+1 = SPACE`, `+33..+58 = A..Z`, `+65..+90 = a..z`).
  - `+96 .. +163`: Full **Cyrillic** uppercase/lowercase (GOST-1967 / Setun-70 & Akademgorodok journals).
  - `+164 .. +224`: **Greek** alphabet & mathematical/logical operators (`∂, ∇, ∑, ∫, ≤, ≥, ≠, ∧, ∨, ¬, ⊤, ⊥, ∈, ∞`).
  - `+225 .. +300`: European Latin diacritics, phonetic & archival typographic marks.
  - `+301 .. +364`: **Jameson MARC / Alexandrian Catalogue Delimiters** (author/title/LC-class/microform-reel/frame markers).
- **`-364 .. -1` (Control & Interactive Console Commands — `TST` yields `-1`):**
  - `-1 .. -32`: Standard teleprinter/line controls (`CR, LF, TAB, FF, BEL, ETX`).
  - `-33 .. -121`: **Licklider–Pask Vector-CRT & Light-Pen Protocol** (move/draw vector, beam intensity, character scale, light-pen pick tag).
  - `-122 .. -364`: `V-OS` terminal session, capability-compartment switches, and `CONVERSE` epistemic uncertainty markers.

---

## Chapter 3. Memory Organization, Addressing & `[42, 36, 4]_3` Ternary ECC

### 3.1 18-Trit Physical Address Space & Protection Model
Physical word addresses are **18-trit balanced integers** (`-193,710,244 .. +193,710,244`). Preserving the Duwamish/Setun convention, **all negative addresses are hardware-protected Supervisor space**: any User-mode fetch, load, or store to an address $\le 0$ immediately raises **Trap 2 (`PROT`)** without modifying memory.

| Physical Address Range (`T18`) | Word Count | Physical Medium | Access & Function |
|---|---:|---|---|
| **`-1,594,323 .. -177,148`** | `1,417,176` (`~9.6 MB` eq.) | **Supervisor `T-DRAM`** (81-bank interleaved) | Privileged `C-EXEC` TTD queues, I/O channel rings, ABFT audit logs, V-Complex shared mailboxes. Lower boundary adjustable via `FENCE_S`. |
| **`-177,147 .. -1`** | **`177,147` ($3^{11}$)** | **Non-Volatile Dual-Core Ferrite Vault** (600 ns) | Privileged, power-loss-immune resident store: Trap vectors (`-1 .. -27`), Cold-Boot IPL, `C-EXEC` kernel, Ashby cutover recovery tables. |
| **`0`** | `1` | **Hardware Null-Guard** | Read in Supervisor returns `0`; any User access or Supervisor write raises **Trap 2 (`PROT`)** (instant null-pointer catcher). |
| **`+1 .. +41,629,544`** | **`41,629,544` (`~283.2 MB` eq.)** | **User & Tensor `T-DRAM`** (81-bank interleaved) | **96.7% of all `43,046,721` ($3^{16}$) `T-DRAM` words!** Hardware Base/Bounds (`BASE`, `LIMIT`) relocated for bare-metal deterministic jobs (`SAEW-18M` uses `37.60M` user words; `SAEW-32M` uses `38.26M` user words — both fit 100% resident!). |
| **`+100,000,000 .. +100,999,999`** | `1,000,000` (sparse) | **Memory-Mapped Control Apertures** | Privileged / capability-gated registers for T-Fabric Tile Controllers, Data Channels `1..8`, SETI DMA ring, and Algedonic status readback. |

*Exact Total `T-DRAM` Accounting:* `1,417,176 (Sup) + 1 (Zero-Guard) + 41,629,544 (User) = 43,046,721 = 3^{16}` words (`258.28 Mtrytes ≈ 292.8 MB` equiv), separate from the `177,147`-word non-volatile Ferrite Vault!

### 3.2 81-Bank Interleaving & Crossbar Timing
- **Bank Mapping:** Physical `T-DRAM` word address `A` maps linearly to `logical_bank = (A + 1,594,323) mod 81`, `row = (A + 1,594,323) // 81` (`0 .. 531,440 = 3^{12} - 1`).
- **Hot-Spare Bank LUT:** An 81-entry hardware lookup table inside the Crossbar maps each `logical_bank ∈ [0..80]` to one of **90 physical banks** (`0..89`, i.e., **81 active + 9 hot spares**).
- **Timing:** Bank read/write cycle = **240 ns** (`8` C-Complex clocks / `6` T-Fabric clocks); non-blocking crossbar pipelined burst = **60 ns** per bank → **$81 / 60\text{ ns} = 1.35\text{ Gwords/s}$** (`8.10 Gtrytes/s ≈ 9.63 GB/s` equiv peak; ~6.5 GB/s sustained). Strided vector accesses with stride coprime to `3` achieve zero bank conflicts!

### 3.3 `UE-TD2000` Differential Erasure Flagging & Proven `[42, 36, 4]_3` SEC-DED Code
Each `UE-TD2000` chip stores $3^7 = 2{,}187$ trits across `4,374` matched MOS capacitors (two per trit):

| Capacitor Pair `(C+, C-)` | Logical Trit | Hardware Sense-Amp Flag |
|:---:|:---:|---|
| `(1, 0)` | **`+1` (`1`)** | Valid |
| `(0, 0)` | **`0` (`0`)** | Valid |
| `(0, 1)` | **`-1` (`T`)** | Valid |
| **`(1, 1)`** | **Forbidden** | **`ERASE = +1` (Known Hardware Erasure Location!)** |

Every 36-trit data word $d = (d_{35}, \dots, d_0)$ is stored in `T-DRAM` as a **42-trit (`7-tryte`) systematic codeword** $w = (d_{35}, \dots, d_0, c_0, \dots, c_5)$ satisfying $H \cdot w^T \equiv \mathbf{0} \pmod 3$, where $H = [P \mid I_6]$ is a **proven $[42, 36, 4]_3$ linear code** ($d_{\min} = 4$; all $\binom{42}{3} = 11{,}480$ column triples verified linearly independent over $\mathrm{GF}(3)$).

#### Exact Systematic Parity-Check Submatrix $P$ ($6 \times 36$, UE Heptavintimal, Trits `35..0` MSB-First)
Each check trit is computed on write as $c_r \equiv -\sum_{j=0}^{35} P_{r,j} \cdot d_{35-j} \pmod 3$ (balanced `{-1, 0, +1}`). Each row has fan-in `23–25` trits, settling in **3 levels of 3-input mod-3 adders (~10 ns)** inside the 240 ns memory cycle:

| Check Trit | Fan-In | Exact 12-Digit UE Heptavintimal Mask of $P_r$ (Trits `35..0`) |
|:---:|---:|:---:|
| **`c0`** | 24 | `LCNSPSGHXODC` |
| **`c1`** | 25 | `LSFDELFVLAQW` |
| **`c2`** | 24 | `GOTXTDZXFHFF` |
| **`c3`** | 24 | `WMRQGXKSZJUY` |
| **`c4`** | 23 | `AMMM000AMENN` |
| **`c5`** | 24 | `0000MMMMMMMM` |

#### Hardware Decode & Recovery Algorithm (Executed Every Read)
Let $E = \{i \in 0..41 : \text{ERASE}_i = +1\}$ be the hardware erasure set (temporarily substituted with `0`), and let $s = H \cdot w^T \pmod 3$ be the 6-trit syndrome:
1. **`|E| = 0, s = 0`:** Clean read.
2. **`|E| = 0, s ≠ 0`:** Match $s$ against $\pm h_i$ (columns of $H$):
   - Unique match $s = \sigma \cdot h_i$ ($\sigma \in \{+1, -1\}$) → **Single-Trit Correction (`SEC`)**: subtract $\sigma$ from trit $i$ (mod 3), deliver clean word, increment bank soft-error counter on Algedonic Bus.
   - No match → **Double-Trit Error Detected (`DED`)** (guaranteed for all 2-trit errors since $d_{\min} = 4$): abort read, signal **Trap 14 (`MEMDED`)** & Algedonic Bus alert.
3. **`1 ≤ |E| ≤ 3`:** Because any $\le 3$ columns of $H$ are linearly independent over $\mathrm{GF}(3)$, hardware Gaussian elimination over the tiny $6 \times |E|$ submatrix $H_E$ solves $H_E \cdot e_E \equiv s \pmod 3$ **uniquely** and verifies consistency across the remaining $6 - |E|$ check equations! If consistent → **Corrects 1, 2, or 3 simultaneous capacitor erasures** on the fly and notifies Beer's monitor; if inconsistent → extra silent error present → raise `MEMDED`.
4. **`|E| ≥ 4`:** Immediate Uncorrectable Bank Erasure → raise `MEMDED` and trigger **Ashby Hot-Spare Bank Cutover**.

---

## Chapter 4. Tier 1: C-Complex Processor Architecture & Instruction Set

### 4.1 Programmer-Visible State (Per C-Complex CPU, `CPU 0..7`)
- **Scalar Registers (`S0–S8`):** Nine 36-trit general-purpose registers (2-trit specifier `-4..+4` mapped `0→S0, +1..+4→S1..S4, -1..-4→S5..S8`; **`S0` is hardwired `0`**, writes discarded). By convention `S7 = FP`, `S8 = SP`.
- **Address/Index Registers (`A0–A8`):** Nine 18-trit signed address/stride registers (`A0` hardwired `0`). Keeping addresses separate from scalar data (à la Cray's CDC 6600/7600) keeps address arithmetic off the floating-point/ALU pipeline.
- **Vector Registers (`V0–V8`):** Nine vector registers, each holding **$3^3 = 27$ elements of `T36`** (or `54` packed `TF18` / `108` packed `TBFP-9` elements), with full **Cray-style vector chaining** (result element $k$ forwards immediately to a dependent vector pipe on the next 30 ns cycle!).
- **Control & Status Registers:**
  - `PC` (18-trit word address + 1-trit parcel selector `P ∈ {+1 [high parcel A], -1 [low parcel B], 0 [36t long]}`).
  - `C` (**1-trit Condition Code**: `T = -1 [NEG/FALSE]`, `0 [ZERO/UNDECIDED]`, `1 = +1 [POS/TRUE]`).
  - `VL` (Vector Length, `0..27`).
  - `BASE`, `LIMIT` (18-trit User relocation & bounds registers; checked in parallel with address add).
  - `PSW` (36-trit Processor Status Word: `MODE ∈ {+1 User, -1 Supervisor}`, `FT` Float-Trap enable, `FU` Float-Undecided sticky trit, `OV` Integer-Overflow sticky trit, `IE` Interrupt mask, `CPUID ∈ [0..7]`).
- **Instruction Stack (Cray Look-Ahead Buffer):** **27-word (`54`-parcel) contiguous circular buffer** in 30 ns bipolar SRAM. Loops fitting inside 27 words execute at full 30 ns logic speed with zero `T-DRAM` instruction fetches!

### 4.2 Dual-Parcel 18-Trit & Long 36-Trit Instruction Formats
Trit `35` (the Format Trit `F_A`) of every fetched 36-trit instruction word selects between **one 36-trit Long Instruction (`F_A = +1`)** or **two sequential 18-trit Short Parcels (`F_A ∈ {0, T}`)** — Parcel A (`35..18`) executed first, followed by Parcel B (`17..0`, whose format trit `F_B` at trit `17` must also lie in `{0, T}`; if `F_A ∈ {0, T}` and `F_B = +1`, hardware raises **Trap 1 [`ILLOP`]**). All branch, jump, and call targets align strictly to **36-trit word boundaries (Parcel A)**; a standalone 18-trit parcel at a branch target or word boundary is padded with `| NOP` in Parcel B.

```text
LONG 36-TRIT INSTRUCTION (F = +1):
 35  34..30   29..28  27..26  25..24  23..22          21..0
+---+--------+-------+-------+-------+-------+-------------------------+
| 1 |  op(5) | rd(2) |rs1(2) |rs2(2) |mod(2) |     imm / disp (22t)    |  = 36 trits
+---+--------+-------+-------+-------+-------+-------------------------+
  op: 243 opcodes (-121..+121) | imm/disp: ±15,690,529,804 (covers full 18t address space!)

SHORT 18-TRIT PARCEL (F = 0 [Scalar/Logic/Branch] or F = T [Float/Vector/Deciban]):
 17   16..13   12..11  10..9    8..7       6..0
+---+--------+-------+-------+-------+---------------+
|0/T| sop(4) | rd(2) |rs1(2) |rs2(2) |   simm (7t)   |                    = 18 trits
+---+--------+-------+-------+-------+---------------+
  sop: 81 opcodes per class (-40..+40) -> 162 short opcodes! | simm: ±1,093
```

**Long-Instruction `mode` Field (2 trits = 9 modes, `EA = A[rs2] + disp`):**
- `00` (`0`): Direct `M[EA]` (36-trit full word `T36`)
- `01` (`+1`): Immediate (`EA` itself, sign-extended to 36 trits)
- `0T` (`-1`): Indirect `M[M[EA]]`
- `1T` (`+2`): High Half-Word `T18` (`H1`, trits `35..18`, sign-extended / packed)
- `10` (`+3`): Low Half-Word `T18` (`H0`, trits `17..0`)
- `11` (`+4`): Tryte-Indexed (`T6`, `EA` selects word + `rs1` selects tryte `0..5`)
- `TT` (`-4`): Nonet-Indexed (`T9`, `rs1` selects nonet `0..3`)
- `T0` (`-3`): Vector Unit-Stride (`stride = +1`)
- `T1` (`-2`): Vector Strided (`stride = A[rs1]`)

### 4.3 Complete Orthogonal Opcode Table (Core Factory Set — Unique Opcodes)

#### A. Short Class `F = 0` — Scalar Integer, Address, Kleene Logic & Short Branches (18-Trit Parcels)
*PC-relative branch offsets (`simm ∈ [-1,093, +1,093]`) are signed 36-trit word displacements relative to the word address `PC_curr` of the branching instruction, targeting Parcel A.*

| `sop` | Mnemonic | Operands | Semantics (Sets `C ← sign(result)` unless noted) | Cycles |
|---:|---|---|---|---:|
| `0` | `NOP` | — | No operation (parcel pad). | 1 |
| `1` | `ADD` | `Sd, Ss1, Ss2, #simm` | `Sd ← Ss1 + Ss2 + simm` (36-trit balanced add; `ADD Sd, S0, #simm` loads small immediates in 18t!). | 1 |
| `2` | `SUB` | `Sd, Ss1, Ss2, #simm` | `Sd ← Ss1 - Ss2 - simm`. | 1 |
| `3` | `MUL` | `Sd, Ss1, Ss2` | `Sd ← low36(Ss1 * Ss2)` (pipelined balanced-ternary tree, no Booth). | 4 |
| `4` | `MULH` | `Sd, Ss1, Ss2` | `Sd ← high36(Ss1 * Ss2)` (upper half of 72-trit product). | 4 |
| `5` | `DIV` | `Sd, Ss1, Ss2` | `Sd ← trunc_quot(Ss1 / Ss2)` (Trap 3 on `Ss2 = 0`). | 14 |
| `6` | `MOD` | `Sd, Ss1, Ss2` | Balanced remainder `Sd = Ss1 - q*Ss2`. | 14 |
| `7` | `NEG` | `Sd, Ss1` | `Sd ← -Ss1` (tritwise `T ↔ 1` inversion). | 1 |
| `8` | `ABS` | `Sd, Ss1` | `Sd ← |Ss1|`. | 1 |
| `9` | `SHF` | `Sd, Ss1, Ss2, #simm` | `Sd ← Ss1 * 3^(Ss2 + simm)` (negative shift rounds unbiased to nearest!). | 1 |
| `10` | `AND` | `Sd, Ss1, Ss2` | Tritwise Kleene strong AND: `min(t1, t2)`. | 1 |
| `11` | `OR` | `Sd, Ss1, Ss2` | Tritwise Kleene strong OR: `max(t1, t2)`. | 1 |
| `12` | `EQV` | `Sd, Ss1, Ss2` | Tritwise Kleene equivalence / product: `t1 * t2`. | 1 |
| `13` | `XTR` | `Sd, Ss1, #p, #n` | Extract `n ∈ [1..36]` trits starting at `p ∈ [0..35]`; centered `simm = 36*(n - 18) + (p - 18) ∈ [-630, +665]`. | 1 |
| `14` | `TRT` | `Ss1, Ss2, #simm` | `C ← trit[Ss2 + simm]` of `Ss1`. | 1 |
| `15` | `SEL` | `Sd, Ss1, #simm` | `Sd ← trit[C + 1]` of `(Ss1 + simm)` (branchless 3-way truth table!). | 1 |
| `16` | `CMP` | `Ss1, Ss2, #simm` | `C ← sign(Ss1 - (Ss2 + simm))` (no register write). | 1 |
| `17` | `TST` | `Ss1` | `C ← sign(Ss1)`. Guarantees float polarity for **finite and `±INF` `TF36` values only**; if `UNDECIDED` (`NaN`) is possible, software must use **`FCM Ss1, S0`**. | 1 |
| `18` | `AADD` | `Ad, As1, As2, #simm` | 18-trit Address register add (`C` unchanged). | 1 |
| `19` | `ASUB` | `Ad, As1, As2, #simm` | 18-trit Address register subtract (`C` unchanged). | 1 |
| `20` | `A2S` | `Sd, As1` | Sign-extend 18-trit `As1` into 36-trit `Sd` (`C ← sign`). | 1 |
| `21` | `S2A` | `Ad, Ss1` | Copy low 18 trits of `Ss1` into `Ad` (`C` unchanged). | 1 |
| `22` | `BR` | `#simm` | PC-relative short branch (`PC_word ← PC_curr + simm`, Parcel A). | 1 |
| `23` | `B3` | `#simm` | **Three-Way Short Branch (`J3`):** `PC_word ← PC_curr + simm + C` (`-1 / 0 / +1` words, Parcel A). | 1 |
| `24` | `BN` | `#simm` | Conditional short branch if `C = -1` (`PC_word ← PC_curr + simm`). | 1 |
| `25` | `BZ` | `#simm` | Conditional short branch if `C = 0` (`PC_word ← PC_curr + simm`). | 1 |
| `26` | `BP` | `#simm` | Conditional short branch if `C = +1` (`PC_word ← PC_curr + simm`). | 1 |
| `27` | `BCT` | `Ad, #simm` | Decrement `Ad ← Ad - 1`; if `Ad > 0`, `PC_word ← PC_curr + simm` (1-cycle loop close in Instruction Stack!). | 1 |
| `28` | `PUSH` | `Ss1` | `S8 ← S8 - 1; M[S8] ← Ss1`. | 2+mem |
| `29` | `POP` | `Sd` | `Sd ← M[S8]; S8 ← S8 + 1; C ← sign(Sd)`. | 2+mem |
| `30` | `RET` | `#simm` | Pop return `PC` from `M[S8]`, advance `S8 ← S8 + 1 + simm`, jump to Parcel A. | 2+mem |

#### B. Short Class `F = T` — `TF36`/`TF18` Floating Point, Vector Ops & Good Deciban ALU (18-Trit Parcels)
| `sop` | Mnemonic | Operands | Semantics (Pipelined: 1 new op/cycle; Latency shown) | Lat. |
|---:|---|---|---|---:|
| `1` | `FAD` | `Sd, Ss1, Ss2` | `TF36` add (exact single-round unbiased!). | 4 |
| `2` | `FSB` | `Sd, Ss1, Ss2` | `TF36` subtract (exact single-round unbiased!). | 4 |
| `3` | `FMP` | `Sd, Ss1, Ss2` | `TF36` multiply (30t × 30t balanced tree, unbiased round). | 5 |
| `4` | `FDV` | `Sd, Ss1, Ss2` | `TF36` divide (radix-3 SRT recurrence, unbiased round). | 16 |
| `5` | `FMA` | `Sd, Ss1, Ss2` | Fused `Sd ← Sd + Ss1 * Ss2` (`TF36`, single round!). | 6 |
| `6` | `FNEG` | `Sd, Ss1` | `TF36` negate (inverts high 30 mantissa trits `35..6` only; leaves `UNDECIDED` intact). | 1 |
| `7` | `FABS` | `Sd, Ss1` | `TF36` absolute value (leaves `UNDECIDED` intact). | 1 |
| `8` | `FCM` | `Ss1, Ss2` | `TF36` compare → `C ∈ {T, 0, 1}`. If either operand is `UNDECIDED`, explicitly sets **`C ← 0`** and latches **`PSW.FU ← +1`**. | 2 |
| `9` | `FLT` | `Sd, Ss1` | Convert 36-trit integer `Ss1` → `TF36` (unbiased round). | 3 |
| `10` | `FIX` | `Sd, Ss1` | Convert `TF36` `Ss1` → 36-trit integer (unbiased round to nearest). | 3 |
| `11` | `F18PK` | `Sd, Ss1, Ss2` | Pack two `TF36` (`Ss1 → H1`, `Ss2 → H0`) into dual `TF18` in `Sd`. | 2 |
| `12` | `F18UP` | `Sd, Ss1, #simm` | Unpack `TF18` half (`simm = +1` High `H1`, `simm = 0` Low `H0`) → `TF36` in `Sd`. | 2 |
| `13` | `FAD18` | `Sd, Ss1, Ss2` | Packed 2-way `TF18` add (`2×` throughput!). | 3 |
| `14` | `FMP18` | `Sd, Ss1, Ss2` | Packed 2-way `TF18` multiply (`2×` throughput!). | 4 |
| `15` | `DADD` | `Sd, Ss1, Ss2` | **Good Deciban** saturating add (`DB36`; `simm = +1` selects packed `2×DB18`). | 1 |
| `16` | `DSUB` | `Sd, Ss1, Ss2` | **Good Deciban** saturating subtract (`DB36`; `simm = +1` selects packed `2×DB18`). | 1 |
| `17` | `DLSE` | `Sd, Ss1, Ss2` | **Good Log-Sum-Exp** in decibans (243-entry ROM + linear interp). | 3 |
| `18` | `DEXP` | `Sd, Ss1` | Convert `DB18`/`DB36` decibans → `TF18`/`TBFP-9` linear weight. | 3 |
| `19` | `DWALD` | `Ss1, Ss2, #simm` | **Good–Wald Sequential Test:** `T = |Ss2 + 81 * simm|` (ticks); sets `C ← +1` if `Ss1 ≥ T`, `-1` if `Ss1 ≤ -T`, `0` otherwise. Accepts register threshold in ticks (`Ss2`, `simm=0`) *or* immediate threshold in **whole decibans** (`Ss2=S0`, `#simm ∈ [-1,093, +1,093]` dB)! | 1 |
| `20` | `VADD` | `Vd, Vs1, Vs2, #simm` | Vector add across `0..VL-1` (`simm = 0`: `TF36`, `+1`: `T36` int, `-1`: `DB36`), chainable! | 4+VL |
| `21` | `VSUB` | `Vd, Vs1, Vs2, #simm` | Vector subtract across `0..VL-1`, chainable! | 4+VL |
| `22` | `VMUL` | `Vd, Vs1, Vs2, #simm` | Vector multiply across `0..VL-1` (`simm = 0`: `TF36`, `+1`: `T36`), chainable! | 5+VL |
| `23` | `VFMA` | `Vd, Vs1, Vs2` | Vector fused `Vd[i] ← Vd[i] + Vs1[i] * Vs2[i]` (`TF36`, 2 ops/clk chained!). | 6+VL |
| `24` | `VF18OP` | `Vd, Vs1, Vs2, #simm` | Packed 2-way `TF18` vector ops (`simm = 0` add, `+1` mul, `+2` fma; 54 floats/vector!). | 4+VL |
| `25` | `VRED` | `Sd, Vs1, #simm` | Vector reduction (`simm = 0` `TF36` sum, `+1` max, `-1` min, `+2` `DLSE` softmax denom!). | 8+VL |
| `26` | `VKLEEN` | `Vd, Vs1, Vs2, #simm` | Vector tritwise Kleene ops (`simm = -1` `AND`, `+1` `OR`, `0` `EQV`, `+2` `NOT`). | 2+VL |
| `27` | `VSVL` | `Ss1, #simm` | Set Vector Length `VL ← clamp(Ss1 + simm, 0, 27)`. | 1 |

#### C. Long Class `F = +1` — Full-Address Memory, Vector DMA, Control & T-Fabric Dispatch (36-Trit Words)
| `op` | Mnemonic | Operands | Semantics | Cycles |
|---:|---|---|---|---:|
| `1` | `LD` | `Sd, disp(As2)` | Load `T36` (or `T18`/`T9`/`T6` per `mode`) from `M[EA]` into `Sd`; `C ← sign(Sd)`. | 8 (T-DRAM) |
| `2` | `LDA` | `Ad, disp(As2)` | Load 18-trit address from `M[EA]` into `Ad` (`C` unchanged). | 8 (T-DRAM) |
| `3` | `ST` | `Ss1, disp(As2)` | Store `T36` (`T18`/`T9`/`T6` per `mode`) from `Ss1` into `M[EA]`. | 8 (posted) |
| `4` | `STA` | `As1, disp(As2)` | Store 18-trit `As1` (sign-extended to `T36`) into `M[EA]`. | 8 (posted) |
| `5` | `LEA` | `Ad, disp(As2)` | Load Effective Address `Ad ← As2 + disp` (18-trit). | 1 |
| `6` | `LDI` | `Sd, #imm22` | Load 22-trit signed immediate (`±15,690,529,804`) into `Sd`; `C ← sign(Sd)`. | 1 |
| `7` | `VLD` | `Vd, disp(As2), As1` | Vector Load `VL` words (`mode`: unit-stride or stride `As1`; streams 1 word/2 clocks across 81 banks!). | 8+2·VL |
| `8` | `VST` | `Vs1, disp(As2), As1` | Vector Store `VL` words (`mode`: unit-stride or stride `As1`; note `rd` holds source `Vs1`, `rs1` holds stride `As1`, `rs2` holds base `As2`). | 8+2·VL |
| `9` | `VGTH` | `Vd, disp(As2), Vs1` | Vector Gather (`EA_i = disp + As2 + Vs1[i]`, sparse cliodynamics Jacobians!). | 10+2·VL |
| `10` | `VSCT` | `Vs2, disp(As2), Vs1` | Vector Scatter (`M[disp + As2 + Vs1[i]] ← Vs2[i]`). | 10+2·VL |
| `11` | `JMP` | `disp(As2)` | Unconditional full-address jump (`PC_word ← EA`, Parcel A). | 2 |
| `12` | `J3` | `disp(As2)` | Full-address **Three-Way Jump** (`PC_word ← EA + C`, Parcel A). | 2 |
| `13` | `CALL` | `Ad, disp(As2)` | Save return `PC_word + 1` into `Ad` (or push onto `M[S8]` if `Ad = A0`), jump to `EA`. | 2 |
| `20` | **`TEXEC`** | `As1, Ss2` | **Dispatch T-Fabric Task:** Enqueue 9-word TTD at `M[As1]` across active tiles selected by 27-trit mask `Ss2` (`+1` run, `0` ignore, `-1` barrier). **Hardware Shorthand:** `Ss2 = S0` (hardwired `0`) selects **all 27 active tiles** (`+1` in trits `0..26`). | 4 (async) |
| `21` | **`TWAIT`** | `Ss2, #imm22` | Wait deterministically on T-Fabric tile mask `Ss2` (**`Ss2 = S0` waits on all 27 active tiles**; optional cycle timeout `#imm22`, `0` = no timeout); sets `C ← +1` (OK), `0` (timeout), or `-1` (recovered via cutover). | barrier |
| `22` | **`TSTAT`** | `Sd, As1` | Read live tile status / zero-trit power-saved counter / ABFT pass count. | 2 |
| `30` | `SVC` | `#imm22` | Supervisor Call (raises Trap 6 into `C-EXEC` with argument `imm22`). | 6 |
| `31` | `RTI` | — | Return from Trap *(Privileged)*; restores `PC`, `PSW`, `C` from Ferrite Core. | 6 |
| `32` | `CHIO` | `As1, #ch` | Start Data Channel `ch ∈ [1..8]` on 6-word CCB at `M[As1]` *(Privileged)*. | 4 |
| `33` | `ALGD` | `Sd, #cmd` | Query/acknowledge Beer Algedonic Bus or read `TIM` 36-trit cycle clock. | 2 |
| `34` | `HLT` | — | Halt C-Complex CPU and signal V-Complex *(Privileged)*. | — |

### 4.4 Hardware Trap & Exception Model (Non-Volatile Ferrite Core `-1 .. -27`)
When a trap occurs in User mode (`PSW.MODE = +1`), hardware atomically saves the CPU frame into the per-CPU Ferrite Vault slot (offset `-27 * CPUID`) and jumps to the `C-EXEC` handler at `-27`:
- `-1`: Resume `PC` · `-2`: Previous `PSW` · `-3`: Trap Code (`1..18`) · `-4`: Faulting `EA` / `SVC` Arg · `-5`: Saved `C` & `VL` · `-6..-26`: Scratch save slots · `-27`: `C-EXEC` Trap Entry Vector.

| Trap | Name | Trigger Condition | Recovery Action |
|---:|---|---|---|
| `1` | `ILLOP` | Undefined opcode or illegal `mode` trit | Terminate user job; report to `V-OS`. |
| `2` | `PROT` | User access to `EA ≤ 0` or outside `[BASE, BASE+LIMIT)`, or write to `M[0]` | Terminate user job; log WEAR compartment audit. |
| `3` | `DIVZ` | Integer or `TF36`/`TF18` division by zero | Job trap handler or abort. |
| `4` | `PRIV` | Privileged opcode executed in User mode | Terminate user job. |
| `5` | `OVF` | Signed integer overflow (when `PSW.OV` trap enabled) | Trap handler. |
| `6` | `SVC` | Intentional Supervisor Call (`SVC #arg`) | `C-EXEC` services I/O, checkpoint, or `V-OS` IPC. |
| `7` | `FPOVF` | `TF36`/`TF18` exponent overflow (when `PSW.FT = +1`) | Trap handler (else saturates to `±INF`). |
| `8` | `FPUND` | `TF36`/`TF18` `UNDECIDED` (`NaN`) produced (when `PSW.FT = +1`) | Trap handler (else propagates `UNDECIDED`). |
| `10` | `TIMER` | Watchdog quantum expired (`C-EXEC` batch scheduling) | Save context cleanly on instruction boundary. |
| `12` | `TFAULT` | Unrecoverable T-Fabric fault (all 3 hot spares exhausted!) | Checkpoint job to disc; alert Beer Operations Gallery. |
| `14` | `MEMDED` | Uncorrectable `[42, 36, 4]_3` double-trit error or `|E| ≥ 4` erasures | Trigger Ashby hot-spare bank cutover + micro-step replay; halt only if replay fails. |
| `18` | `MCHK` | Trap occurring inside Supervisor mode (`PSW.MODE = -1`) | Immediate CPU halt; Algedonic hard alarm to V-Complex! |

---

## Chapter 5. Tier 2: T-Fabric Systolic, Associative & Spectral Architecture

### 5.1 Tile Organization & Local Bipolar Ternary SRAM
The **T-Fabric** consists of **30 identical liquid-cooled field-replaceable blades** (`27` active logical tiles `T0..T26` + `3` hot spares `S0..S2`) clocked at **40 ns (25 MHz)**:
- **Systolic Grid:** **$6 \times 6 = 36$ pipelined `TBFP-9` MAC Processing Elements (PEs)** per tile (`972` active MACs total). Each PE contains:
  - A `9 × 9`-trit balanced multiplier (zero Booth recoding, 9 partial products, tritwise `T` inversion, and **Q4 1971 Kelvanov Zero-Trit Gate-Suppress** switching off current steering whenever either input trit is `0`!).
  - An exact **27-trit (`T27`) accumulator** (`±3.81 × 10^{12}`).
  - Output shift-and-truncate unit (unbiased round-to-nearest `T27 → TBFP-9` with block exponent tracking) + programmable **Wald Three-Valued Gate (`τ_neg, τ_pos`)**.
- **Local Tile SRAM (`3^8 = 6,561` `T36` words ≈ 46.8 KB equiv per tile, 40 ns):** Partitioned into **3 × `2,187`-word (`3^7`) independent banks** (`SRAM-A`, `SRAM-B`, `SRAM-C`) allowing simultaneous double-buffered DMA from Main `T-DRAM` while feeding all 36 MACs at `100%` utilization (**145 GB/s equiv** internal aggregate operand bandwidth across 27 tiles!).

### 5.2 The 9-Word Tile Task Descriptor (`TTD`, Aligned to 9-Word Boundary)
C-Complex dispatches work to any subset of the 27 logical tiles via `TEXEC A_ttd, S_mask`. Each `TTD` occupies exactly **9 × `T36` words**:

| Word | Name | Exact Trit-Field Layout (`35..0`) & Semantics |
|---:|---|---|
| **`W0`** | `CTRL` | `mode(3t)` (`+1`: `TBFP-9 GEMM`, `+2`: `TBFP-3 Triad GEMM`, `+3`: `TB1 Pure-Trit Assoc`, `-1`: `Radix-3 FFT`, `-2`: `Radix-6 PFA FFT`, `0`: `Elementwise/Scale`) · `zgate(1t)` (`+1` Kelvanov zero-trit suppress) · `abft(1t)` (`+1` enable ternary ABFT checksum) · `wgate(1t)` (`+1` enable Rosenblatt 3-valued attention dead-zone) · `accum(1t)` (`+1` add to existing `C`, `0` overwrite `C`, `-1` subtract from `C`) · `micro_gran(6t)` (micro-block commit cadence in tile steps) · `flags(23t)`. |
| **`W1`** | `ADDR_A` | `base_A(18t)` (physical `T36` address of Matrix `A` or FFT Input) · `stride_A(18t)` (row stride in `T36` words). |
| **`W2`** | `ADDR_B` | `base_B(18t)` (Matrix `B` or FFT Twiddle Table) · `stride_B(18t)`. |
| **`W3`** | `ADDR_C` | `base_C(18t)` (Output Matrix `C` or FFT Output) · `stride_C(18t)`. |
| **`W4`** | `DIMS` | `M(12t)` (output rows, or in FFT modes packed `p(6t) \| q(6t)` for $N_{\text{FFT}} = 3^p \cdot 2^q$) · `N(12t)` (output cols / FFT batch count) · `K(12t)` (inner reduction dimension, multiple of 6). |
| **`W5`** | `SCALE` | `E_A(6t)` · `E_B(6t)` · `E_C(6t)` (shared 6-trit block exponents) · `rshift(6t)` (`T27 → TBFP-9` unbiased right-shift) · `tau_neg(6t)` · `tau_pos(6t)` (Wald dead-zone thresholds for 3-valued attention gating!). |
| **`W6`** | `SRAM` | `bank_cfg(6t)` (double-buffer mode & stationary operand: `A`, `B`, or `C`) · `tile_part(12t)` (logical grid split across participating tiles) · `bias_addr(18t)` (optional bias/skip-add vector address). |
| **`W7`** | `CKPT` | `ckpt_ptr(18t)` (Main Store micro-step checkpoint pointer for `< 75 µs` Ashby cutover) · `abft_tol(6t)` (max allowed ternary checksum residual, `0` for exact integer/fixed) · `job_id(12t)`. |
| **`W8`** | `STATUS` | Written atomically by Tile Controller on completion: `code(2t)` (`+1 OK`, `0 RUNNING`, `-1 RECOVERED_BY_CUTOVER`, `-2 ABFT_FAIL`) · `cycles(18t)` · `zero_saved(16t)` (count of suppressed zero-trit products for Beer thermal telemetry!). |

### 5.3 Operating Modes & Q4 1971 Byurakan ECO Refinements

1. **Mode `+1` — `TBFP-9` Systolic GEMM (Training & Cliodynamics Jacobians):**
   - Computes $C_{M \times N} \leftarrow \alpha C + A_{M \times K} \cdot B_{K \times N}$ tiled in $6 \times 6$ sub-blocks.
   - Each PE completes 1 `TBFP-9` MAC per 40 ns clock → **48.6 GOPS peak** across 27 active tiles (**17.0 GOPS sustained** end-to-end training including `T-DRAM` traffic, C-Complex `DLSE` softmax, and checkpointing).
   - Every `6` rows/columns carries an automatic **ABFT ternary checksum** (Section 5.4).
2. **Mode `+2` — `TBFP-3` Triad-Packed GEMM (Interactive Inference & Coarse Correlation):**
   - Splits each 9-trit MAC lane cleanly into **3 independent 3-trit heptavintimal triads** (`±13`, 4.75 bits) accumulating into 3 × 9-trit sub-accumulators → **145.8 GOPS peak** (~48 GOPS sustained).
   - Enables **1-tile `SAEW-18M` inference at 50.8 tokens/sec** (27 concurrent interactive users!) and coarse wideband SETI multi-beam correlation.
3. **Mode `+3` — Rosenblatt–Kelvanov Pure-Trit Associative Mode (`ECO-71-09`, Q4 1971):**
   - Directly informed by Frank Rosenblatt's September 1971 Byurakan CETI discussions with **Pavel Kelvanov** on Soviet Setun-70 sparse ternary weights and three-valued associative retrieval:
     - **Pure-Trit Inner Product (`TB1`):** Each 9-trit MAC bypasses its multiplier and executes **9 parallel tritwise `EQV` (`a_i · w_i ∈ {-1, 0, +1}`)** summed via a 9-input balanced counter tree → **9 trit-MACs/cycle/PE = 437.4 G-trit-ops/s peak** across 27 tiles! Used to pre-filter Jameson's 2M-volume card catalogue and execute Setun-compatible ternary associative memory layers.
     - **Zero-Trit Gate-Suppress (`zgate = +1`):** Active across Modes `+1`, `+2`, and `+3`: whenever a weight or activation trit is `0`, `UE-TECL` current steering stays at its symmetric zero-current rest state, reducing dynamic tile thermal dissipation by **18%–22%** on regularized `SAEW` layers.
     - **Hardware Three-Valued Attention Gating (`wgate = +1`):** Before writing attention compatibility scores $q_i \cdot k_j$ out of a PE, hardware compares the score against `[tau_neg, tau_pos]` in `TTD.W5`: scores inside `(tau_neg, tau_pos)` clamp to exact **`0` (`UNDECIDED / ABSTAIN`)**, pruning noisy cross-context retrieval and compounding zero-trit power savings on the downstream value-projection GEMM!
4. **Modes `-1` & `-2` — Oliver–Good Spectral Butterfly Mode (`Radix-3` & `Radix-6 PFA` FFTs):**
   - Reconfigures each $6 \times 6$ tile into **9 complex radix-3 butterfly engines** (or **6 mixed-radix-6 Good–Thomas Prime-Factor Algorithm engines**) operating on packed 18-trit `TBFP-9C` complex samples (`9t Re + 9t Im`, ~86 dB SFDR).
   - Hardware **trit-reversed address generator** streams twiddle factors from `SRAM-B` with zero C-Complex intervention.
   - **Verified Throughput:** A **$3^{13} = 1{,}594{,}323$-point complex radix-3 FFT** (`1.934 × 10^8` ops) completes across all 27 tiles in **12.1 ms** (**131.9M spectral channels/sec** sustained @ 16.0 GOPS).

### 5.4 Native Balanced-Ternary ABFT (Algorithm-Based Fault Tolerance)
Because `UE-TECL` operates with 0.4 V step margins across 972 high-speed MACs over 29-day to 95-day runs, hardware enforces **exact ternary ABFT** on every $6 \times 6$ tile micro-block:
- Before streaming a $6 \times 6$ block of $A$ and $B$ into a tile, the tile ingress unit computes a 7th **column-checksum row** $a_{\Sigma, k} = \sum_{r=0}^5 A_{r, k}$ and 7th **row-checksum column** $b_{k, \Sigma} = \sum_{c=0}^5 B_{k, c}$ in exact `T27` arithmetic.
- A dedicated 7th shadow-checksum MAC accumulator inside each tile computes the predicted block grand sum $\hat{S} = \sum_k a_{\Sigma, k} b_{k, \Sigma}$ in parallel with the $6 \times 6$ grid (costing `< 3%` extra tile gates and **0% cycle overhead**!).
- At micro-block commit (every `~25 µs`–`250 µs`), hardware compares $\hat{S}$ against $\sum_{r,c} C_{r,c}$:
  - Any discrepancy exceeding `TTD.W7.abft_tol` **blocks the corrupt `C` write-back to Main Store** and fires an immediate **Algedonic Hard Fault** to Tier 4 (`V-Complex`), triggering `< 75 µs` Ashby cutover onto a hot-spare tile!

---

## Chapter 6. Tier 3b: I/O Channels, Peripherals & Jameson's Alexandrian Vault

### 6.1 Asynchronous Data Channels (`Channels 1..8`)
All backing storage and wideband SETI streams transfer directly between devices and `T-DRAM` over **8 autonomous DMA Data Channels**, invoked by `C-EXEC` via `CHIO As1, #ch` using a **6-word Channel Control Block (`CCB`)**:
- `CCB[0]`: `device_type(3t)` (`+1` Disc, `+2` 9-Track Tape, `+3` Alexandrian Silo, `-1` SETI Baseband Ring, `-2` Level -8 Digitisation Chute) · `unit(6t)` · `op(4t)` (`READ`, `WRITE`, `READ_BACKWARD`, `SEEK`, `REWIND`, `MARK`, `VERIFY`) · `flags(23t)`.
- `CCB[1]`: Starting `T-DRAM` physical word address (`18t`) + word count (`18t`).
- `CCB[2]`: Media position (disc cylinder/surface/sector, tape block, or silo cartridge ID).
- `CCB[3]`: Provenance & WEAR compartment tag (`36t`, audited by Jameson's archival cataloguer).
- `CCB[4]`: Chain pointer (`18t`, allows scatter-gather channel programs without interrupting `C-EXEC`).
- `CCB[5]`: Completion status (`+1 OK`, `0 BUSY`, `-1 MEDIA_ERR`, `-2 PROVENANCE_DENIED`) + actual transferred words.

### 6.2 Native Ternary Media Specifications
| Device Subsystem | Physical Units & Level -9 Location | Native Ternary Recording & Capacity | Sustained Transfer & Timing |
|---|---|---|---|
| **Online Ternary Disc Array** | 32 UE-modified 3330-class spindles (Alexandrian Vault, 8 DMA channels) | **5-trit-per-8-flux group coding** (~99% coding eff.): `24M` `T36` words/spindle → **`768M` `T36` words (`4.61 Gtrytes ≈ 5.2 GB` equiv)** | **~20 MB/s equiv** aggregate across 8 channels; 30 ms avg seek, 16.7 ms rotation. Holds full 1.0B-token corpus (`250M` words) 3× over. |
| **Nine-Track Ternary Tape** | 32 drives (Alexandrian Vault) | **1 trit per track** (`+flux / 0 / -flux`), 9 tracks = 1 nonet frame, 4 frames = 1 `T36` word; supports **read-backward** (Setun/IBM 2401 heritage). | 600 frames/inch @ 112.5 ips → `16,875` `T36` words/sec/drive (~120 KB/s equiv/drive). |
| **Alexandrian Archival Silo** | Ampex TBM (1971) video-tape cartridge silo + IBM 1360-class photo-digital store (Jameson Vault) | **`14.7G` `T36` words (~88 Gtrytes ≈ ~100 GB equiv)**; fed directly via vertical hoist & optical-tape chute from **Level -8 Digitisation**. | Cartridge mount 10–15 s; streaming staging to disc at ~1.5 MB/s equiv. |
| **Oliver SETI Baseband Ingest** | 3 Faraday-isolated Annex cells, Channel 8 dedicated ring | 9-trit (`TBFP-9`) dual-quadrature A/D digitizers disciplined by rubidium/cesium atomic frequency standard. | Continuous streaming up to **`5.0M` complex `TBFP-9C` samples/sec** directly into `T-DRAM` double buffers. |

---

## Chapter 7. Tier 4: V-Complex, Algedonic Telemetry & Ashby Cutover

### 7.1 Out-of-Band Algedonic Telemetry Bus (Stafford Beer VSM)
The **Algedonic Bus** is a physically separate, low-speed, triple-modular-redundant (TMR) serial ternary ring connecting every T-Fabric tile, C-Complex CPU, `T-DRAM` bank controller, and Level -10 Freon/power sensor directly to the **4 V-Complex supervisory nodes** (`V-OS`). Polling steals **zero cycles** from the 81-bank crossbar or compute pipelines.

Every **1 ms** (and instantaneously on any hardware hard-fault wire!), each node emits a **4-word (`4 × T36`) Algedonic Telemetry Packet**:
- `ATP[0]`: `unit_id(9t)` · `health_trit(1t)` (`+1 NOMINAL`, `0 DEGRADED/WARNING`, `-1 ALGEDONIC PAIN/FAULT`) · `fault_code(6t)` · `temp_mC(12t)` (cold-plate junction delta in `0.01 °C`) · `freon_dP(8t)`.
- `ATP[1]`: `sec_count(18t)` (ECC single-trit corrections since last epoch) · `erasure_count(18t)` (`UE-TD2000` `(1,1)` hardware erasures).
- `ATP[2]`: `abft_checks(18t)` · `abft_faults(6t)` · `zero_trit_pct(12t)` (live Kelvanov zero-trit power-saving ratio).
- `ATP[3]`: `timestamp(36t)` (global 30 ns system clock).

### 7.2 Ashby Ultrastable Hot-Spare Tile Cutover (`< 75 µs` Hardware Cutover)
When a T-Fabric tile raises an Algedonic Hard Fault (`ABFT_FAIL`, `UE-TECL` voltage rail drift, or Freon $\Delta T > 3.0^\circ\text{C}$), the V-Complex executes an autonomous, hardware-sequenced cutover **completely transparent to `C-EXEC` and user software**:

```mermaid
stateDiagram-v2
    [*] --> NORMAL : "27 Active + 3 Hot Spares Ready"
    NORMAL --> ISOLATE : "0-5 us: Algedonic Hard-Fault Wire Fires\n(ABFT Mismatch / Thermal / Rail Fault)"
    ISOLATE --> SELECT_SPARE : "5-15 us: Freeze Crossbar Port k,\nGate Tile k Power/Clock, Block Corrupt Writeback"
    SELECT_SPARE --> REMAP_RELOAD : "15-35 us: Ashby Variety LUT Picks Spare s;\nCrossbar LUT Remaps Logical k -> Physical s"
    REMAP_RELOAD --> REPLAY : "35-65 us: DMA Reload TTD + Checkpoint Block\n(<= 2,187 Words @ ~150 Mwords/s -> ~15 us)"
    REPLAY --> VERIFY_RESUME : "65-75 us: Release Tile s on Interrupted Micro-Block\n(Deterministic Replay: 25 us - 1 ms) + ABFT Verify"
    VERIFY_RESUME --> NORMAL : "Cutover Complete! Beer Gallery Lights Bay k Lamp\nfor Hot Blade Swap; C-Complex Job Never Aborted"
    SELECT_SPARE --> CHECKPOINT_HALT : "All 3 Spares Exhausted -> Raise Trap 12 (TFAULT)"
```

- **Exact Timing Guarantee:** Hardware isolation, crossbar LUT remapping, and local `SRAM` reload onto the hot-spare blade complete strictly in **`< 75 µs`** (`~60 µs` nominal). Because `ABFT` gates every micro-block write-back *before* corrupt data enters Main `T-DRAM`, the spare tile then deterministically replays the single interrupted micro-block (`25 µs` for a $6 \times 6$ inner block up to `1 ms` for a large FFT stage), verifies ABFT parity, and rejoins the System 2 barrier ring — allowing **29-day (`SAEW-18M`) and 95-day (`SAEW-32M`) runs** to survive multiple physical blade failures with zero manual restarts!
- **Hot-Spare `T-DRAM` Bank Cutover (9 Spare Banks):** When a physical bank's `UE-TD2000` erasure rate crosses Beer's warning threshold (`health_trit = 0`), V-Complex allocates a spare bank, runs a low-priority background hardware scrub-copy (`531,441` words corrected through `[42, 36, 4]_3` ECC in **~25 ms**), and performs an atomic 1-cycle Crossbar Bank-LUT flip — retiring the degrading bank cleanly *before* an uncorrectable `|E| ≥ 4` failure ever occurs!

---

## Chapter 8. Reference `TRIAD-36` Assembly Examples

### 8.1 Good–Wald Sequential Deciban Detector (Packed 18-Trit Parcels, Instruction Stack)
Scans `A2` spectral peaks in `DB36` decibans, accumulating log-odds in `S1` and stopping via a single-instruction 3-way Wald branch (`B3`) against `±20 dB` (`±1,620` ticks at `81` ticks/dB, passed in `S2` — or equivalently via `DWALD S1, S0, #20` using whole-deciban `simm`!). Every branch target is strictly word-aligned (Parcel A):

```asm
; Inputs: A1 = pointer to DB36 peak stream, A2 = peak count, S2 = 1620 (+20 dB in ticks)
; Output: S4 & C = +1 (ARTIFICIAL BEACON), 0 (UNDECIDED - KEEP INTEGRATING), -1 (NATURAL NOISE)
          ADD     S1, S0, #0        | NOP                ; [18t+18t = 1w] S1 <- 0 dB initial log-odds
LOOP:     LD      S3, 0(A1)                              ; [36t     = 1w] Load next deciban increment
          DADD    S1, S1, S3        | AADD A1, A1, #1    ; [18t+18t = 1w] Saturating deciban add || ptr++
          DWALD   S1, S2, #0        | B3   WALD_0        ; [18t+18t = 1w] C <- Wald verdict || 3-way jump!
                                                         ; (equivalently: DWALD S1, S0, #20 for ±20 dB!)
WALD_N:   BR      REJECT            | NOP                ; [18t+18t = 1w] WALD_0 - 1 (C = -1): Natural/Noise
WALD_0:   BCT     A2, #LOOP         | BR   UNDECIDED     ; [18t+18t = 1w] WALD_0 + 0 (C =  0): Loop or abstain
WALD_P:   ADD     S4, S0, #+1       | RET  #0            ; [18t+18t = 1w] WALD_0 + 1 (C = +1): Candidate beacon!
REJECT:   ADD     S4, S0, #-1       | RET  #0            ; [18t+18t = 1w] Return -1 (sets C = -1)
UNDECIDED:ADD     S4, S0, #0        | RET  #0            ; [18t+18t = 1w] Return  0 (sets C =  0)
```
*Performance:* Entire 9-word routine (5-word loop body) fits cleanly inside Cray's 27-word Instruction Stack — zero instruction fetches after iteration 1, executing in **13 clocks (390 ns)** per spectral candidate (`8` `LD` + `2` `DADD|AADD` + `2` `DWALD|B3` + `1` `BCT`)!

### 8.2 Dispatching a 27-Tile `SAEW` Associative-Attention GEMM (`TBFP-9` + Kelvanov Gating)
Uses the hardware shorthand **`Ss2 = S0`** on `TEXEC` and `TWAIT` to address all 27 active T-Tiles cleanly in one instruction without overflowing `LDI`'s 22-trit immediate (custom sub-partition 27-trit masks are loaded from memory via `LD` or synthesized with `LDI` + `SHF` + `ADD`):

```asm
; Assumes 9-word TTD at label ATTN_TTD has:
;   W0: mode=+1 (TBFP-9 GEMM), zgate=+1 (Kelvanov zero-trit suppress), abft=+1, wgate=+1
;   W4: M=243 (context T), N=243, K=36 (d_k per head), W5: tau_neg=-27, tau_pos=+27
        LEA     A1, ATTN_TTD(A0)                         ; [36t     = 1w] A1 <- physical address of 9-word TTD
        TEXEC   A1, S0                                   ; [36t     = 1w] Async launch across ALL 27 active tiles (S0 shorthand!)
        ; ... C-Complex overlaps prior-head DLSE softmax normalization in parallel ...
        TWAIT   S0, #0                                   ; [36t     = 1w] Deterministic barrier on all 27 tiles (no timeout)
        B3      CHECK_STATUS        | NOP                ; [18t+18t = 1w] C = +1 OK, 0 timeout, -1 recovered by cutover
```

---

## Appendix A. Simulator Conformance Constants (`duwamish_ue1000` Step 3 Contract)

Every parameter below is normative for the clean Step 3 Python simulator (`duwamish_ue1000`):

| Constant Symbol | Normative Value | Notes |
|---|---|---|
| `TRIT_VALUES` | `(-1, 0, +1)` (`'T', '0', '1'`) | Balanced ternary radix 3 |
| `HEPT_ALPHABET` | `"NOPQRSTUVWXYZ0ABCDEFGHIJKLM"` | Index `= triad_val + 13` (`-13..+13`) |
| `TRYTE_TRITS` / `NONET_TRITS` / `HALF_TRITS` / `WORD_TRITS` | `6` / `9` / `18` / `36` | Exact divisors of `36` |
| `ECC_CHECK_TRITS` / `MEM_WORD_TRITS` | `6` / `42` | Proven `[42, 36, 4]_3` SEC-DED + 3-erasure |
| `ECC_P_ROWS_HEPT` | `("LCNSPSGHXODC", "LSFDELFVLAQW", "GOTXTDZXFHFF", "WMRQGXKSZJUY", "AMMM000AMENN", "0000MMMMMMMM")` | Exact $6 \times 36$ parity submatrix $P$ (MSB-first trits `35..0`) |
| `TF36_MANT_TRITS` / `TF36_EXP_TRITS` | `30` (`35..6`) / `6` (`5..0`) | 47.55-bit mantissa; `exp = -364, mant = 0` is `UNDECIDED` |
| `TF18_MANT_TRITS` / `TF18_EXP_TRITS` | `13` (`17..5`) / `5` (`4..0`) | 20.60-bit mantissa; `exp = -121, mant = 0` is `UNDECIDED` |
| `TBFP9_MANT_TRITS` / `TBFP9_ACC_TRITS` | `9` / `27` | 14.26-bit mantissa; exact 27-trit accumulator |
| `DECIBAN_TICKS_PER_DB` | `81` ($3^4$) | Saturating `DB18`/`DB36`; 243-entry `DLSE`/`DEXP` ROM |
| `C_COMPLEX_CPUS` / `C_CLOCK_NS` / `C_VEC_LEN` | `8` / `30` (`33.33 MHz`) / `27` ($3^3$) | `533 MFLOPS` peak `TF36`, `27`-word I-stack |
| `T_ACTIVE_TILES` / `T_SPARE_TILES` / `T_GRID` / `T_CLOCK_NS` | `27` / `3` / `6 × 6` (`36` MACs) / `40` (`25 MHz`) | `48.6 GOPS` peak / `17.0 GOPS` sust `TBFP-9`; `6,561`-word SRAM/tile |
| `TDRAM_BANKS_ACTIVE` / `TDRAM_BANKS_SPARE` / `TDRAM_BANK_WORDS` | `81` ($3^4$) / `9` ($3^2$) / `531,441` ($3^{12}$) | Total `43,046,721` ($3^{16}$) `T36` words (`240 ns` cyc, `60 ns` burst) |
| `FERRITE_CORE_WORDS` / `FERRITE_CYCLE_NS` | `177,147` ($3^{11}$) / `600` | Mapped at `[-177,147 .. -1]`; non-volatile `C-EXEC` + trap frame |
| `USER_TDRAM_RANGE` | `[+1 .. +41,629,544]` | `41.63M` user words (fits `SAEW-18M` & `SAEW-32M` 100% resident) |
| `CUTOVER_HW_MAX_US` | `75` µs | Isolation + LUT remap + SRAM reload; followed by deterministic micro-block replay |
| `SAEW_VOCAB` / `SAEW_CONTEXT` | `19,683` ($3^9$) / `243` ($3^5$) | `SAEW-18M`: `L=9, d=324, h=9, d_k=36` (28.6 d); `SAEW-32M`: `L=8, d=486, h=9, d_k=54` (94.8 d) |
