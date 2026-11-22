"""Tests for DYNAMO-X Forrester/Ashby System Dynamics & Cliodynamics compiler."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ue1000.dynamox import (
    DynamoXError,
    compile_dynamo,
    parse_dynamo_source,
    run_dynamo,
)
from ue1000.machine import UE1000Machine

PROGRAMS_DIR = Path(__file__).resolve().parent.parent / "programs"


def test_exponential_growth_euler_and_rk2_exact_tf36() -> None:
    """dX/dt = 0.1 * X from X(0)=1.0 over t=0..10 -> exact e^1 = 2.718281828..."""
    src_euler = """
    * EXPONENTIAL GROWTH EULER CHECK
    SPEC DT=0.25, LENGTH=10.0, TSTART=0.0, PRTPER=5.0, METHOD=EULER
    C RATE = 0.1
    N X = 1.0
    R DX.KL = RATE * X.K
    L X.K = X.J + DT * DX.JK
    PRINT X
    """
    src_rk2 = src_euler.replace("METHOD=EULER", "METHOD=RK2")
    res_e = run_dynamo(src_euler)
    res_rk = run_dynamo(src_rk2)
    x_e = res_e.runs[0].series["X"][-1]
    x_rk = res_rk.runs[0].series["X"][-1]
    assert abs(x_rk - math.e) < 5e-4, f"RK2 TF36 drifted: {x_rk} vs {math.e}"
    assert abs(x_rk - math.e) < abs(x_e - math.e)  # RK2 strictly beats Euler!
    assert res_e.runs[0].cycles > 0
    assert res_e.runs[0].istack_hits > 0           # Tight Euler loop fits in 27-word Cray I-stack!
    assert res_rk.compile_result.parcels_packed > 0


def test_topological_sort_and_algebraic_cycle_rejection() -> None:
    """DYNAMO-X must accept out-of-order A/R equations and reject algebraic feedback cycles."""
    out_of_order = """
    SPEC DT=1.0, LENGTH=5.0, TSTART=0.0
    N X = 2.0
    R FLOW.KL = B.K
    A B.K = A1.K * 2.0
    A A1.K = X.K + 1.0
    L X.K = X.J + DT * FLOW.JK
    """
    res = run_dynamo(out_of_order)
    # At t=0: A1=3, B=6 -> X(1)=8; monotonic growth verified
    assert res.runs[0].series["X"][0] == pytest.approx(2.0, abs=1e-6)
    assert res.runs[0].series["X"][1] == pytest.approx(8.0, abs=1e-6)

    cyclic = """
    SPEC DT=1.0, LENGTH=5.0
    N X = 1.0
    A P.K = Q.K + 1.0
    A Q.K = P.K * 2.0
    L X.K = X.J + DT * P.K
    """
    with pytest.raises(DynamoXError, match="simultaneous loop"):
        parse_dynamo_source(cyclic)


def test_tabhl_clip_step_ramp_and_sign3() -> None:
    """Verify native TRIAD-36 _DYN_TABHL interpolation, STEP, RAMP, CLIP, and SIGN3."""
    src = """
    SPEC DT=1.0, LENGTH=4.0, TSTART=0.0
    T CURVE = 10.0 / 20.0 / 40.0 / 80.0
    N X = 0.0
    A LOOK.K = TABHL(CURVE, TIME, 0.0, 3.0, 1.0)
    A MID.K  = TABHL(CURVE, 1.5, 0.0, 3.0, 1.0)
    A ST.K   = STEP(5.0, 2.0)
    A RM.K   = RAMP(3.0, 1.0)
    A CL.K   = CLIP(100.0, -50.0, TIME, 2.0)
    A SG.K   = SIGN3(CL.K)
    L X.K = X.J + DT * 1.0
    PRINT LOOK, MID, ST, RM, CL, SG
    """
    res = run_dynamo(src)
    s = res.runs[0].series
    assert s["LOOK"][0] == pytest.approx(10.0, abs=1e-5)
    assert s["LOOK"][2] == pytest.approx(40.0, abs=1e-5)
    assert s["LOOK"][4] == pytest.approx(80.0, abs=1e-5)  # clamped at hi
    assert s["MID"][0] == pytest.approx(30.0, abs=1e-5)    # midpoint between 20 and 40
    assert s["ST"][1] == pytest.approx(0.0, abs=1e-6)
    assert s["ST"][2] == pytest.approx(5.0, abs=1e-6)
    assert s["RM"][3] == pytest.approx(6.0, abs=1e-6)
    assert s["SG"][1] == pytest.approx(-1.0, abs=1e-6)
    assert s["SG"][2] == pytest.approx(1.0, abs=1e-6)


def test_smooth_and_delay3_macro_expansion() -> None:
    """Verify SMOOTH and 3rd-order Forrester DELAY3 expand into real Levels and lag a step."""
    src = (PROGRAMS_DIR / "predator_delay.dyn").read_text(encoding="utf-8")
    res = run_dynamo(src)
    # 1 SMOOTH level + 3 DELAY3 stages = 4 synthesized hidden levels!
    assert len(res.compile_result.model.synthesized_levels) == 4
    s = res.runs[0].series
    # Before shock (t=0): steady state orders=100, forecast=100, arrivals=100, inv=400
    assert s["ORDERS"][0] == pytest.approx(100.0, abs=1e-4)
    assert s["ARRIVALS"][0] == pytest.approx(100.0, abs=1e-4)
    # After step shock to 125 at t=5, inventory dips then overshoots above 450 due to DELAY3 lag!
    assert min(s["INV"]) < 395.0
    assert max(s["INV"]) > 480.0
    assert res.rendered_plots and "DYNAMO-X PLOT" in res.rendered_plots[0]


def test_world2_forrester_reruns_separate_three_regimes() -> None:
    """Showcase World2/Cliodynamics model with 3 RERUNs must yield distinct Ashby outcomes."""
    src = (PROGRAMS_DIR / "world2_forrester.dyn").read_text(encoding="utf-8")
    res = run_dynamo(src)
    assert len(res.runs) == 3
    bau, mid, opt = res.runs
    # Unchecked BAU depletes resources & legitimacy -> COLLAPSE (-1)
    assert bau.stability_trit == -1
    # Ashby-Beer equilibrium corridor stays viable (+1) with healthier resources & legitimacy
    assert opt.stability_trit == 1
    assert opt.series["RES"][-1] > bau.series["RES"][-1]
    assert opt.series["LEG"][-1] > bau.series["LEG"][-1]
    assert len(res.rendered_tables) == 3
    assert len(res.rendered_plots) == 3


def test_ashby_2d_sweep_with_tfabric_multi_region_coupling() -> None:
    """Showcase 6-region coupled 2D sweep must invoke T-Fabric TBFP-9 and map viable vs collapse."""
    src = (PROGRAMS_DIR / "ashby_sweep.dyn").read_text(encoding="utf-8")
    m = UE1000Machine()
    res = run_dynamo(src, machine=m)
    assert res.sweep is not None
    assert len(res.sweep.grid_trits) == 5
    assert len(res.sweep.grid_trits[0]) == 5
    # Must contain both collapse (-1) at low conservation/abatement and viable (+1) at high!
    assert res.sweep.collapse_count >= 1
    assert res.sweep.viable_count >= 1
    assert res.sweep.grid_trits[0][0] == -1   # CONS=0.10, ABAT=0.10 -> collapse
    assert res.sweep.grid_trits[-1][-1] == 1  # CONS=0.90, ABAT=0.90 -> viable equilibrium
    # Confirm real Tier 2 T-Fabric TBFP-9 coupling GEMMs and ABFT checks executed!
    assert m.tfabric.total_ops > 0
    assert m.tfabric.total_abft_checks > 0
    assert res.runs[0].tbfp9_calls > 0
