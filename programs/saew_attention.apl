⍝ === ROSENBLATT-GOOD-KELVANOV SAEW ATTENTION HEAD IN APL\36 ===
⍝ Executes Q +.× ⍉K on Tier 2 T-Fabric (TBFP-9), applies Rosenblatt-Kelvanov
⍝ three-valued dead-zone gating (+1 retrieve / 0 abstain / ¯1 inhibit),
⍝ normalizes Good Bayesian weights, and retrieves W +.× V!

⎕TMODE ← 1   ⍝ Tier 2 T-Fabric TBFP-9 systolic GEMM (ABFT + zero-trit gate!)

Q ← 4 6 ⍴ 1.0 ¯0.5 0.0 0.8 ¯0.2 0.4  0.0 1.0 ¯0.6 0.2 0.9 ¯0.1  ¯0.8 0.3 1.0 0.0 ¯0.5 0.7  0.5 0.5 ¯0.2 1.0 0.0 ¯0.8
K ← 4 6 ⍴ 1.0 ¯0.4 0.1 0.9 ¯0.1 0.3  0.1 0.9 ¯0.7 0.1 0.8 0.0  ¯0.9 0.2 1.0 0.1 ¯0.4 0.8  0.4 0.6 ¯0.1 0.9 0.1 ¯0.7
V ← 4 4 ⍴ 10 0 0 0   0 20 0 0   0 0 30 0   0 0 0 40

⎕← '=== 1. COMPATIBILITY LOGITS (Q +.× ⍉K) ÷ √6 ON T-FABRIC TBFP-9 ==='
SCORES ← (Q +.× ⍉ K) ÷ 6 * 0.5
⎕← SCORES

⎕← '=== 2. KELVANOV THREE-VALUED DEAD-ZONE GATE (|score| >= 0.45) ==='
GATE ← (× SCORES) × (∣ SCORES) ≥ 0.45
⎕← GATE

⎕← '=== 3. SPARSE GATED BAYESIAN RETRIEVAL WEIGHTS ==='
EXPW ← (* SCORES) × (GATE > 0)
ROW_SUM ← +/ EXPW
W ← EXPW ÷ ⍉ 4 4 ⍴ ROW_SUM
⎕← W

⎕← '=== 4. ATTENDED OUTPUT (W +.× V) ON T-FABRIC TBFP-9 ==='
OUT ← W +.× V
⎕← OUT

⎕← '=== 5. PURE-TRIT MODE +3 ASSOCIATIVE PROBE (Q ⎕TB1 ⍉K) ==='
⎕← (× Q) ⎕TB1 ⍉ (× K)

)HW
