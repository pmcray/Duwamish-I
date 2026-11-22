⍝ === OLIVER-GOOD WIDEBAND SETI SPECTRUM & WALD DETECTOR IN APL\36 ===
⍝ Synthesizes a 27-sample baseband signal with a coherent tone at bin 4,
⍝ runs Oliver-Good Radix-3 TBFP-9C FFT on Tier 2 T-Fabric (`⎕FFT`),
⍝ converts peak SNR to Good Decibans, and emits hardware `DWALD` verdicts!

⎕IO ← 0
N ← 27
T ← ⍳ N
SIGNAL ← (1 ○ 2 × ○ 4 × T ÷ N) + 0.12 × (2 ○ 2 × ○ 9 × T ÷ N)

⎕← '=== 1. RADIX-3 TBFP-9C MAGNITUDE SPECTRUM (T-FABRIC ⎕FFT) ==='
SPEC ← ⎕FFT SIGNAL
⎕← SPEC

⎕← '=== 2. PEAK SPECTRAL BIN & MAGNITUDE ==='
PEAK_MAG ← ⌈/ SPEC
PEAK_BIN ← SPEC ⍳ PEAK_MAG
⎕← PEAK_BIN
⎕← PEAK_MAG

⎕← '=== 3. GOOD DECIBAN EVIDENCE & HARDWARE DWALD VERDICTS (±10 dB) ==='
MEAN_BG ← (+/ SPEC) ÷ N
DB_EVID ← 10 × (10 ⍟ (SPEC + 0.01) ÷ MEAN_BG)
VERDICTS ← 10 ⎕WALD DB_EVID
⎕← 'Top 9 channel verdicts (+1 BEACON, 0 UNDECIDED, ¯1 REJECTED):'
⎕← 9 ↑ VERDICTS

)HW
