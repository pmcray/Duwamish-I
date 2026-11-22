"""End-to-end integration tests for all three anchor workloads & Beer VSM Operations Gallery."""

from pathlib import Path

from ue1000.machine import OBB_LEVEL_NINE_CELLS, UE1000Machine
from workloads.cliodynamics import run_cliodynamics_demo
from workloads.gallery import render_vsm_gallery_html
from workloads.saew import run_saew_demo
from workloads.seti import run_seti_demo


def test_locked_obb_level_nine_cell_counts():
    assert sum(OBB_LEVEL_NINE_CELLS.values()) == 30
    assert OBB_LEVEL_NINE_CELLS["Central Core (Lifts / Stairs / WCs)"] == 2
    assert OBB_LEVEL_NINE_CELLS["Alexandrian Vault (Tier 3b — J. U. Jameson)"] == 8
    assert OBB_LEVEL_NINE_CELLS["Compute Vault (C-Complex + T-Fabric + Crossbar)"] == 2
    assert OBB_LEVEL_NINE_CELLS["Researcher Terminal Suites (Pask / Licklider)"] == 3
    assert OBB_LEVEL_NINE_CELLS["UE Special Systems Maintenance & Staging Bay"] == 2


def test_all_three_anchor_workloads_and_gallery(tmp_path: Path):
    m = UE1000Machine()

    # 1. Mini-SAEW training inside native TBFP-9 with live mid-training fault cutover
    saew = run_saew_demo(m, steps=60, inject_fault_at_step=20)
    assert saew.final_loss < saew.initial_loss * 0.45
    assert saew.cutovers_survived == 1
    assert saew.avg_zero_trit_power_saving_pct > 10.0
    assert len(saew.sample_tbfp9_output) > len(saew.sample_prompt)

    # 2. Oliver-Good SETI (Radix-3 TBFP-9C FFT + Section 8.1 TRIAD-36 assembly)
    seti = run_seti_demo(m, n_fft=729, n_frames=12)
    verdicts = {r.label: r.verdict_trit for r in seti.results}
    assert verdicts["Cosmic White Noise"] == -1
    assert verdicts["Broadband Natural Pulsar (LGM-1 class)"] == -1
    assert verdicts["Marginal Faint Signal (Hold Integration)"] == 0
    assert verdicts["Artificial Coherent Narrowband Beacon (Bin 243)"] == 1
    assert all(r.istack_hits > 0 for r in seti.results)

    # 3. Forrester-Ashby Cliodynamics (Exact TF36 + TBFP-9 coupling)
    clio = run_cliodynamics_demo(m, years=150, dt=2.5)
    stabs = {tr.policy_name: tr.stability_trit for tr in clio.trajectories}
    assert stabs["Business as Usual (Unchecked Extraction)"] == -1
    assert stabs["Partial Mitigation (Metastable Frontier)"] == 0
    assert stabs["Ashby-Beer Ultrastable Policy Corridor"] == 1
    assert clio.tf36_ops > 10_000

    # 4. Beer VSM Operations Gallery HTML generation
    out_html = render_vsm_gallery_html(m.telemetry_snapshot(), saew, seti, clio, tmp_path / "gallery.html")
    html = out_html.read_text(encoding="utf-8")
    assert "DUWAMISH-I (UE-1000)" in html
    assert "Alexandrian Vault" in html
    assert "ASHBY CUTOVER" in html
