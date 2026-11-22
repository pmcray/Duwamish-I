⍝ === APL\36 — Duwamish-I (UE-1000) Balanced-Ternary Array Showcase ===
⍝ Runs natively on Tier 1 C-Complex 27-lane vectors & Tier 2 T-Fabric!

⎕← '=== 1. ICONIC 1966 APL ONE-LINE PRIME SIEVE (<= 30) ==='
PRIMES ← (∼ T ∈ T ∘.× T) / T ← 1 ↓ ⍳ 30
⎕← PRIMES

⎕← '=== 2. KLEENE THREE-VALUED LOGIC TRUTH TABLES (5 CHARS EACH!) ==='
TRITS ← ¯1 0 1
⎕← 'KLEENE AND (¯1 0 1 ∘.∧ ¯1 0 1):'
⎕← TRITS ∘.∧ TRITS
⎕← 'KLEENE OR  (¯1 0 1 ∘.∨ ¯1 0 1):'
⎕← TRITS ∘.∨ TRITS
⎕← 'KLEENE EQV (¯1 0 1 ∘.≡ ¯1 0 1):'
⎕← TRITS ∘.≡ TRITS

⎕← '=== 3. BALANCED-TERNARY ENCODE (⊤), DECODE (⊥) & HEPTAVINTIMAL (⎕H) ==='
⎕← 'Minimal balanced trits of 42 (⊤ 42):'
⎕← ⊤ 42
⎕← 'Fixed 6-trit tryte of ¯19 (6⍴¯3 ⊤ ¯19):'
TRYTE ← (6 ⍴ ¯3) ⊤ ¯19
⎕← TRYTE
⎕← 'Decode back (⊥ TRYTE):'
⎕← ⊥ TRYTE
⎕← 'Heptavintimal of 1973 (⎕H 1973):'
⎕← ⎕H 1973

⎕← '=== 4. C-COMPLEX 27-LANE HARDWARE VECTOR OPS & SCAN ==='
X ← ⍳ 12
⎕← 'Triangular numbers (+\ ⍳ 12):'
⎕← +\ X
⎕← 'Dot product X +.× X (C-Complex VDOT):'
⎕← X +.× X

⎕← '=== 5. HILBERT LINEAR SYSTEM SOLVE (b ⌹ H) IN EXACT TF36 (⎕TMODE←0) ==='
⎕TMODE ← 0
H ← ÷ ¯1 + (⍳ 4) ∘.+ ⍳ 4
B ← 1 1 1 1
SOL ← B ⌹ H
⎕← 'Solution x to H x = 1:'
⎕← SOL
⎕← 'Residual norm |H x - b|:'
⎕← +/ ∣ (H +.× SOL) - B
⎕TMODE ← 1

⎕← '=== 6. DEL-EDITOR USER FUNCTION (FACTORIAL) ==='
∇ Z ← FACT N; I
  Z ← 1
  I ← 1
LP: →(I > N)/0
  Z ← Z × I
  I ← I + 1
  →LP
∇
⎕← 'FACT 7 ='
⎕← FACT 7

)HW
