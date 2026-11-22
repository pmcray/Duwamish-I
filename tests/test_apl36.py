"""Tests for APL\\36 balanced-ternary array & tensor language on Duwamish-I."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ue1000.apl36 import APL36Error, APL36Session, run_apl_script
from ue1000.machine import UE1000Machine
from ue1000.ternary import char_to_obb729, obb729_to_char

PROGRAMS_DIR = Path(__file__).resolve().parent.parent / "programs"


def test_obb729_apl_glyph_roundtrip() -> None:
    """Every APL\\36 glyph must encode/decode cleanly in a single 6-trit OBB-729 tryte."""
    glyphs = "←→×÷⌈⌊⍴⍳⌽⊖⍉⍋⍒○⍟⌹∘↑↓∼∣≡⍲⍱⌿⍀⍎⍕⍝¯⎕⋄∇≤≥≠∧∨¬⊤⊥∈"
    for g in glyphs:
        tr = char_to_obb729(g)
        assert -364 <= tr <= 364
        assert obb729_to_char(tr) == g


def test_right_to_left_strands_and_prime_sieve() -> None:
    """Verify strict APL right-to-left evaluation and the iconic 1966 prime sieve."""
    s = APL36Session()
    assert s.eval_line("2 × 3 + 4").to_python() == 14
    assert s.eval_line("¯3 + 1 2 3").to_python() == [-2, -1, 0]
    primes = s.eval_line("(∼ T ∈ T ∘.× T) / T ← 1 ↓ ⍳ 30").to_python()
    assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_kleene_three_valued_tables_and_hardware_vkleen() -> None:
    """Verify 5-char Kleene 3x3 truth tables, UNDECIDED propagation, and C-Complex VKLEEN."""
    s = APL36Session()
    t_and = s.eval_line("¯1 0 1 ∘.∧ ¯1 0 1").to_python()
    t_or = s.eval_line("¯1 0 1 ∘.∨ ¯1 0 1").to_python()
    t_eqv = s.eval_line("¯1 0 1 ∘.≡ ¯1 0 1").to_python()
    assert t_and == [[-1, -1, -1], [-1, 0, 0], [-1, 0, 1]]
    assert t_or == [[-1, 0, 1], [0, 0, 1], [1, 1, 1]]
    assert t_eqv == [[1, 0, -1], [0, 0, 0], [-1, 0, 1]]

    # 1D vector Kleene ops dispatch to hardware VKLEEN on C-Complex!
    before = s.hw_vec_calls
    res = s.eval_line("¯1 0 1 1 ∧ 1 0 ¯1 1").to_python()
    assert res == [-1, 0, -1, 1]
    assert s.hw_vec_calls > before

    # Strict ⎕KL ← 1 relational mode + UNDECIDED (⎕U) -> 0
    s.eval_line("⎕KL ← 1")
    assert s.eval_line("3 < 5").to_python() == 1
    assert s.eval_line("5 < 3").to_python() == -1
    assert s.eval_line("⎕U < 5").to_python() == 0
    assert s.eval_line("∼ ¯1 0 1").to_python() == [1, 0, -1]


def test_balanced_ternary_encode_decode_hept_and_io() -> None:
    """Verify monadic/dyadic ⊤ & ⊥ balanced-ternary coding, ⎕H, and ⎕IO ← ¯1."""
    s = APL36Session()
    assert s.eval_line("⊤ 5").to_python() == [1, -1, -1]        # 9 - 3 - 1 = 5
    assert s.eval_line("⊥ 1 ¯1 ¯1").to_python() == 5
    assert s.eval_line("(6 ⍴ ¯3) ⊤ ¯19").to_python() == [0, 0, -1, 1, 0, -1]  # -27 + 9 - 1 = -19
    assert s.eval_line("⊥ (6 ⍴ ¯3) ⊤ ¯19").to_python() == -19
    assert s.eval_line("24 60 60 ⊤ 3661").to_python() == [1, 1, 1]
    assert s.eval_line("⎕H 13").to_python() == "00000000000M"

    # Balanced centered index origin ⎕IO ← ¯1!
    s.eval_line("⎕IO ← ¯1")
    assert s.eval_line("⍳ 5").to_python() == [-2, -1, 0, 1, 2]
    s.eval_line("V ← 10 20 30 40 50")
    assert s.eval_line("V[0]").to_python() == 30
    assert s.eval_line("V[¯2]").to_python() == 10


def test_hardware_ccomplex_vectors_and_tfabric_matmul() -> None:
    """Verify C-Complex 27-lane vector ops, T-Fabric TBFP-9/TB1 matmul, and exact TF36 mode."""
    m = UE1000Machine()
    s = APL36Session(machine=m)
    # 30-element vector forces strip-mining across VL=27 + VL=3 chunks!
    s.eval_line("A ← ⍳ 30")
    s.eval_line("B ← 2 × A")
    total = s.eval_line("+/ B").to_python()
    assert total == 930
    dot = s.eval_line("A +.× A").to_python()
    assert dot == 9455
    assert m.cpu0.cycles > 0
    assert s.hw_vec_calls >= 3

    # 2D matrix multiply dispatches to Tier 2 T-Fabric TBFP-9!
    s.eval_line("M1 ← 3 4 ⍴ ⍳ 12")
    s.eval_line("M2 ← 4 2 ⍴ 1 0 0 1 1 1 2 ¯1")
    out_tbfp9 = np.array(s.eval_line("M1 +.× M2").to_python(), dtype=np.float64)
    expected = np.arange(1, 13).reshape(3, 4) @ np.array([[1, 0], [0, 1], [1, 1], [2, -1]])
    assert np.allclose(out_tbfp9, expected, atol=1e-2)
    assert s.hw_tfabric_calls >= 1
    assert m.tfabric.total_abft_checks >= 1

    # Exact TF36 mode (⎕TMODE ← 0)
    s.eval_line("⎕TMODE ← 0")
    out_tf36 = np.array(s.eval_line("M1 +.× M2").to_python(), dtype=np.float64)
    assert np.allclose(out_tf36, expected, atol=1e-9)


def test_del_functions_recursion_indexing_and_ascii_digraphs() -> None:
    """Verify ∇ recursive functions, multi-axis indexing/assignment, and ASCII teletype digraphs."""
    src = """
    ∇ Z <- FIB N
      ->(N <= 1)/BASE
      Z <- (FIB N - 1) + FIB N - 2
      ->0
    BASE: Z <- N
    ∇
    F8 <- FIB 8
    M <- 3 3 ⍴ ⍳ 9
    M[2;2] <- 99
    COL2 <- M[;2]
    """
    _, sess = run_apl_script(src)
    assert sess.vars["F8"].to_python() == 21
    assert sess.vars["COL2"].to_python() == [2, 99, 8]


def test_showcase_apl_scripts() -> None:
    """Run all three showcase .apl programs end-to-end and verify real hardware metrics."""
    # 1. apl_tour.apl
    out1, sess1 = run_apl_script((PROGRAMS_DIR / "apl_tour.apl").read_text(encoding="utf-8"))
    assert any("2 3 5 7 11 13 17 19 23 29" in ln for ln in out1)
    assert any("5040" in ln for ln in out1)
    assert sess1.vars["SOL"].to_python() is not None
    assert sess1.machine.cpu0.cycles > 0

    # 2. saew_attention.apl (Tier 2 TBFP-9 + Mode +3 TB1!)
    out2, sess2 = run_apl_script((PROGRAMS_DIR / "saew_attention.apl").read_text(encoding="utf-8"))
    assert sess2.hw_tfabric_calls >= 3
    assert sess2.machine.tfabric.total_abft_checks >= 2
    out_mat = np.array(sess2.vars["OUT"].to_python(), dtype=np.float64)
    assert out_mat.shape == (4, 4)
    # Strong diagonal self-attention: row i attends primarily to value row i!
    assert np.argmax(out_mat, axis=1).tolist() == [0, 1, 2, 3]

    # 3. oliver_dft_apl.apl (Tier 2 Radix-3 FFT + C-Complex hardware DWALD!)
    out3, sess3 = run_apl_script((PROGRAMS_DIR / "oliver_dft_apl.apl").read_text(encoding="utf-8"))
    assert sess3.vars["PEAK_BIN"].to_python() in (4, 23)  # symmetric real-signal bin 4 or 27-4=23
    verdicts = sess3.vars["VERDICTS"].to_python()
    assert verdicts[4] == 1  # hardware DWALD flags bin 4 as +1 BEACON!
    assert sess3.hw_tfabric_calls >= 1
    assert sess3.hw_vec_calls >= 1
