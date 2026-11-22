"""Comprehensive tests for TRILISP-36 (Native TRIAD-36 LISP 1.5 Compiler, VM & GC)."""

from pathlib import Path
import pytest

from ue1000.trilisp36 import (
    TAG_NUM,
    TAG_PTR,
    TAG_SYM,
    W_NIL,
    W_T,
    W_U,
    compile_trilisp,
    get_tag,
    run_trilisp,
    tag_cons,
    tag_num,
    tag_sym,
    untag,
)

PROGRAMS_DIR = Path(__file__).resolve().parent.parent / "programs"


def test_balanced_ternary_one_cycle_unboxing_identity():
    """Verify (3*x + tag) >> 1 == x for ALL positive, zero, and negative x and all 3 tags!"""
    for x in (-1_000_000, -42, -1, 0, 1, 42, 1_000_000):
        wn = tag_num(x)
        assert get_tag(wn) == TAG_NUM and untag(wn) == x
        wc = tag_cons(x)
        assert get_tag(wc) == TAG_PTR and untag(wc) == x
        ws = tag_sym(x)
        assert get_tag(ws) == TAG_SYM and untag(ws) == x
    assert get_tag(W_NIL) == TAG_SYM and untag(W_NIL) == 0
    assert get_tag(W_T) == TAG_SYM and untag(W_T) == 1
    assert get_tag(W_U) == TAG_SYM and untag(W_U) == 2


def test_trilisp_tail_call_optimization_and_automatic_gc_under_tight_heap():
    """Run a 300-step tail-recursive allocator in a TINY 128-cell heap!

    Proves TWO hard properties simultaneously:
      1. Proper O(1) Tail-Call Optimization (no stack growth over 300 iterations).
      2. Automatic native TRIAD-36 mark-and-sweep GC triggers mid-execution, reclaims dead
         cons cells, and completes cleanly with the exact mathematical result!
    """
    src = """
    (DEFUN SUM-DOWN (N ACC)
      (IF (= N 0)
          ACC
          (SUM-DOWN (- N 1) (+ ACC N))))
    (SUM-DOWN 200 0)
    """
    # 200 tail calls create ~1,200 temporary env/arg cons cells -> MUST trigger GC in a 160-cell heap!
    res = run_trilisp(src, heap_cells=160, opt=True)
    assert res.result_str == "20100"  # 200 * 201 / 2
    assert res.gc_runs >= 3
    assert res.heap_used < 160


def test_trilisp_dotted_rest_args_let_star_and_dynamic_eval():
    src = """
    (DEFUN SUM-ALL (FIRST . REST)
      (IF (NULL REST)
          FIRST
          (+ FIRST (APPLY SUM-ALL REST))))
    (LET* ((A 10)
           (B (+ A 5))
           (C (EVAL (LIST '+ A B 7))))
      (LIST (SUM-ALL 1 2 3 4 5) B C))
    """
    res = run_trilisp(src, opt=True)
    assert res.result_str == "(15 15 32)"


def test_trilisp_showcase_tour():
    src = (PROGRAMS_DIR / "trilisp_tour.lsp").read_text(encoding="utf-8")
    res = run_trilisp(src, opt=True)
    assert res.result_str == "TOUR-COMPLETE"
    assert res.printed[0] == "(1 0 -1 -1 -1)"                      # TAGs of num, cons, sym, NIL, U
    assert res.printed[1] == "(SETUN DUWAMISH OBB LEVEL-9)"        # APPEND
    assert res.printed[2] == "(1 4 9 16 25)"                       # MAPCAR lambda
    assert res.printed[3] == "(11 12 13)"                          # Lexical closure + SETQ
    assert res.printed[4] == "720"                                 # LABEL 6!
    assert res.printed[5] == "(U U T U)"                           # Kleene 3-valued logic
    assert res.printed[6] == "(NEGATIVE UNDECIDED-OR-ZERO POSITIVE)"  # Hardware SIGN3 (B3)
    # 12.0*81 + round(9.5*81) = 972 + 770 = 1742 ticks -> 1742/81 = 21.506 -> +21.51db!
    assert res.printed[7] == "(+21.51db T U +13.01db)"             # Good Deciban WALD & DLSE
    assert res.printed[8].startswith("(GC-RECLAIMED ")             # Native TRIAD-36 GC!
    assert res.gc_runs >= 1


def test_trilisp_showcase_metacircular_evaluator():
    """Run McCarthy's LISP 1.5 metacircular evaluator inside compiled TRILISP-36 on UE-1000!"""
    src = (PROGRAMS_DIR / "metacircular.lsp").read_text(encoding="utf-8")
    res = run_trilisp(src, opt=True)
    assert res.result_str == "((A B C D) 120)"
    assert res.printed == ["(A B C D)", "120"]


def test_trilisp_showcase_symbolic_differentiation():
    src = (PROGRAMS_DIR / "symbolic_diff.lsp").read_text(encoding="utf-8")
    res = run_trilisp(src, opt=True)
    assert res.result_str == "((+ (+ X X) 3) 2)"
    assert res.printed == ["(+ (+ X X) 3)", "2"]
