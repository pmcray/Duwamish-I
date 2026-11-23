"""CLI entry point for the Duwamish-I (UE-1000) simulator & anchor workloads.

Usage:
    python3 -m ue1000 demo all       # Run SAEW + SETI + Cliodynamics + generate Beer VSM Gallery
    python3 -m ue1000 demo saew
    python3 -m ue1000 demo seti
    python3 -m ue1000 demo clio
    python3 -m ue1000 info
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .machine import OBB_LEVEL_NINE_CELLS, UE1000Machine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ue1000", description="Duwamish-I (UE-1000) Simulator CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("info", help="Print locked UE-1000 Rev. 1.5 architectural parameters")
    d = sub.add_parser("demo", help="Run reference anchor workloads")
    d.add_argument("target", choices=["saew", "seti", "clio", "all"], default="all", nargs="?")
    d.add_argument("--out", default="out/vsm_operations_gallery.html", help="Output HTML gallery path")

    s3 = sub.add_parser("sal3", help="Compile and/or run a SAL-3 program on Duwamish-I")
    s3.add_argument("file", help="Path to .sal3 source file")
    s3.add_argument("--asm", action="store_true", help="Print generated TRIAD-36 assembly")
    s3.add_argument("--no-opt", action="store_true", help="Disable peephole & dual-parcel packing")

    f70 = sub.add_parser("tritran", help="Compile and/or run a TRI-TRAN 70V FORTRAN program")
    f70.add_argument("file", help="Path to .f70 source file")
    f70.add_argument("--asm", action="store_true", help="Print generated TRIAD-36 assembly")
    f70.add_argument("--no-opt", action="store_true", help="Disable peephole & dual-parcel packing")

    tl = sub.add_parser("trilisp", help="Compile and/or run a TRILISP-36 LISP 1.5 program")
    tl.add_argument("file", help="Path to .lsp source file")
    tl.add_argument("--asm", action="store_true", help="Print generated TRIAD-36 assembly")
    tl.add_argument("--no-opt", action="store_true", help="Disable peephole & dual-parcel packing")
    tl.add_argument("--heap", type=int, default=4096, help="Cons cell heap capacity")

    cv = sub.add_parser("converse", help="Run a CONVERSE procognitive script (.cv) or interactive REPL")
    cv.add_argument("file", nargs="?", help="Optional path to .cv script (omit or --repl for interactive terminal)")
    cv.add_argument("--repl", action="store_true", help="Start interactive CONVERSE terminal")

    dx = sub.add_parser("dynamo", help="Compile and/or run a DYNAMO-X Cliodynamics model (.dyn)")
    dx.add_argument("file", help="Path to .dyn source file")
    dx.add_argument("--asm", action="store_true", help="Print generated TRIAD-36 assembly")
    dx.add_argument("--no-opt", action="store_true", help="Disable peephole & dual-parcel packing")

    ap = sub.add_parser("apl", help="Run an APL\\36 balanced-ternary array script (.apl) or interactive REPL")
    ap.add_argument("file", nargs="?", help="Optional path to .apl script (omit or --repl for interactive workspace)")
    ap.add_argument("--repl", action="store_true", help="Start interactive APL\\36 workspace REPL")

    rf = sub.add_parser("refal", help="Run a REFAL-36 pattern-rewriting program (.ref)")
    rf.add_argument("file", help="Path to .ref source file")

    f3 = sub.add_parser("forth", help="Run a FORTH-3 indirect-threaded program (.fth) or interactive REPL")
    f3.add_argument("file", nargs="?", help="Optional path to .fth source file (omit or --repl for interactive console)")
    f3.add_argument("--repl", action="store_true", help="Start interactive FORTH-3 console")

    ic = sub.add_parser("intercal", help="Run an unsanctioned TRI-INTERCAL program (.i3 — Reel SS-GAG-72)")
    ic.add_argument("file", help="Path to .i3 source file")
    ic.add_argument("--1974", dest="allow_1974", action="store_true", help="Allow post-Dec-1973 Clark COME FROM")

    vo = sub.add_parser("vos", help="Run a V-OS / C-EXEC Level -9 console script (.vos) or interactive shell")
    vo.add_argument("file", nargs="?", help="Optional path to .vos script (omit or --repl for interactive shell)")
    vo.add_argument("--repl", action="store_true", help="Start interactive V-OS Level -9 terminal")
    vo.add_argument("--user", default="BROOKS", help="Initial WEAR login principal (default: BROOKS)")

    gl = sub.add_parser("gallery", help="Launch live Web VSM Operations Gallery & OBB-729 Vector-CRT server")
    gl.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    gl.add_argument("--port", type=int, default=8089, help="Bind port (default: 8089)")
    gl.add_argument("--user", default="BROOKS", help="Initial V-OS principal (default: BROOKS)")
    gl.add_argument("--prewarm", action="store_true", help="Pre-run Cliodynamics & SETI scopes on startup")

    args = parser.parse_args(argv)
    if args.cmd == "gallery":
        from .gallery_server import serve_gallery

        serve_gallery(host=args.host, port=args.port, user=args.user, prewarm=args.prewarm)
        return 0
    if args.cmd == "vos":
        from .vos import run_vos_repl, run_vos_script

        if args.repl or not args.file:
            run_vos_repl(user=args.user)
            return 0
        src = Path(args.file).read_text(encoding="utf-8")
        out, _ = run_vos_script(src, user=args.user)
        print(f"=== V-OS Level -9 Session: {args.file} ===")
        for ln in out:
            print(ln)
        return 0
    if args.cmd == "intercal":
        from .tri_intercal import run_tri_intercal

        src = Path(args.file).read_text(encoding="utf-8")
        res = run_tri_intercal(src, allow_1974=args.allow_1974)
        print(f"=== TRI-INTERCAL (Reel SS-GAG-72): {args.file} ===")
        for ln in res.output:
            print("  " + ln)
        print(
            f"  [BROOKS AUDIT WARNING: {res.cycles_wasted:,} UE-TECL cycles burned on unsanctioned humor | "
            f"Politeness={res.politeness_ratio*100:.1f}% | Stmts={res.stmts_executed}]"
        )
        return 0
    if args.cmd == "forth":
        from .forth3 import run_forth, run_forth_repl

        if args.repl or not args.file:
            run_forth_repl()
            return 0
        src = Path(args.file).read_text(encoding="utf-8")
        res = run_forth(src)
        print(f"=== FORTH-3 Execution: {args.file} ===")
        print(
            f"  Dictionary: {res.dict_words} words ({res.dict_cells_used} T-DRAM cells) | "
            f"{res.cycles:,} cycles ({res.istack_hits:,} I-stack hits) | T-Fabric ops: {res.tfabric_ops:,}"
        )
        for ln in res.output:
            print("  " + ln)
        if res.stack:
            print("  Data Stack:", res.stack)
        return 0
    if args.cmd == "refal":
        from .refal36 import run_refal

        src = Path(args.file).read_text(encoding="utf-8")
        res = run_refal(src)
        print(f"=== REFAL-36 Execution: {args.file} ===")
        print(
            f"  Steps: {res.steps:,} | Hardware ALU calls: {res.hw_alu_calls:,} | "
            f"C-Complex cycles: {res.cpu_cycles:,} | Burial store ops: {res.store_ops}"
        )
        for ln in res.output:
            print("  " + ln)
        if res.result_str:
            print(f"  Result: {res.result_str}")
        return 0
    if args.cmd == "apl":
        from .apl36 import run_apl_repl, run_apl_script

        if args.repl or not args.file:
            run_apl_repl()
            return 0
        src = Path(args.file).read_text(encoding="utf-8")
        out, sess = run_apl_script(src)
        print(f"=== APL\\36 Execution: {args.file} ===")
        for ln in out:
            print(ln)
        return 0
    if args.cmd == "dynamo":
        from .dynamox import compile_dynamo, run_dynamo

        src = Path(args.file).read_text(encoding="utf-8")
        opt = not args.no_opt
        if args.asm:
            res = compile_dynamo(src, opt=opt)
            print(res.assembly)
            print(
                f"; --- Words: {len(res.assembled.words)} | Levels: {len(res.model.levels)} "
                f"(synthesized={len(res.model.synthesized_levels)}) | "
                f"Peephole: {res.peephole_edits} | Dual-parcels: {res.parcels_packed} ---"
            )
            return 0
        run_res = run_dynamo(src, opt=opt)
        cr = run_res.compile_result
        print(f"=== DYNAMO-X Execution: {args.file} ({cr.model.title}) ===")
        print(
            f"  Code size: {len(cr.assembled.words)} words (peephole={cr.peephole_edits}, "
            f"dual-parcels={cr.parcels_packed}) | Method={cr.model.spec['METHOD']} | "
            f"Regions={cr.model.spec['REGIONS']} | Levels={len(cr.model.levels)} "
            f"(macros expanded: {len(cr.model.synthesized_levels)})"
        )
        for tbl in run_res.rendered_tables:
            print("\n" + tbl)
        for plt in run_res.rendered_plots:
            print("\n" + plt)
        if run_res.sweep is not None:
            print("\n" + run_res.sweep.grid_ascii)
            print(
                f"  Sweep Totals: VIABLE(+1)={run_res.sweep.viable_count} | "
                f"METASTABLE(0)={run_res.sweep.metastable_count} | "
                f"COLLAPSE(-1)={run_res.sweep.collapse_count}"
            )
        return 0
    if args.cmd == "converse":
        from .converse import run_converse_repl, run_converse_script

        if args.repl or not args.file:
            run_converse_repl()
            return 0
        src = Path(args.file).read_text(encoding="utf-8")
        out, _ = run_converse_script(src)
        print(f"=== CONVERSE Session: {args.file} ===")
        for ln in out:
            print("  " + ln)
        return 0
    if args.cmd == "trilisp":
        from .trilisp36 import compile_trilisp, run_trilisp

        src = Path(args.file).read_text(encoding="utf-8")
        opt = not args.no_opt
        if args.asm:
            res = compile_trilisp(src, opt=opt, heap_cells=args.heap)
            print(res.assembly)
            print(
                f"; --- Words: {len(res.assembled.words)} | Symbols: {len(res.symbols)} | "
                f"Peephole: {res.peephole_edits} | Dual-parcels: {res.parcels_packed} ---"
            )
            return 0
        run_res = run_trilisp(src, opt=opt, heap_cells=args.heap)
        cr = run_res.compile_result
        print(f"=== TRILISP-36 Execution: {args.file} ===")
        print(
            f"  Code size: {len(cr.assembled.words)} words (peephole={cr.peephole_edits}, "
            f"dual-parcels={cr.parcels_packed}) | {run_res.cycles:,} cycles | "
            f"Heap: {run_res.heap_used}/{cr.heap_cells} cells (GC runs: {run_res.gc_runs})"
        )
        for p in run_res.printed:
            print("  > " + p)
        print(f"  Result   : {run_res.result_str}")
        return 0
    if args.cmd == "tritran":
        from .tritran70v import compile_tritran, run_tritran

        src = Path(args.file).read_text(encoding="utf-8")
        opt = not args.no_opt
        if args.asm:
            res = compile_tritran(src, opt=opt)
            print(res.assembly)
            print(
                f"; --- Words: {len(res.assembled.words)} | Vector loops: {res.vector_loops_emitted} | "
                f"Peephole: {res.peephole_edits} | Dual-parcels: {res.parcels_packed} ---"
            )
            return 0
        run_res = run_tritran(src, opt=opt)
        cr = run_res.compile_result
        print(f"=== TRI-TRAN 70V Execution: {args.file} ===")
        print(
            f"  Code size: {len(cr.assembled.words)} words (vec_loops={cr.vector_loops_emitted}, "
            f"peephole={cr.peephole_edits}, dual-parcels={cr.parcels_packed}) | "
            f"{run_res.cycles:,} cycles ({run_res.istack_hits} I-stack hits)"
        )
        for ln in run_res.lines:
            print("  " + ln)
        print(f"  STOP code: {run_res.stop_code}")
        return 0
    if args.cmd == "sal3":
        from .sal3 import compile_sal3, run_sal3

        src = Path(args.file).read_text(encoding="utf-8")
        opt = not args.no_opt
        if args.asm:
            res = compile_sal3(src, opt=opt)
            print(res.assembly)
            print(
                f"; --- Words: {len(res.assembled.words)} | Peephole edits: {res.peephole_edits} | "
                f"Dual parcels packed: {res.parcels_packed} | BlooP: {res.bloop_certified} ---"
            )
            return 0
        run_res = run_sal3(src, opt=opt)
        cr = run_res.compile_result
        print(f"=== SAL-3 Execution: {args.file} ===")
        if cr.bloop_certified:
            print("  [BlooP Certificate: TERMINATION STATICALLY PROVEN]")
        print(
            f"  Code size: {len(cr.assembled.words)} words (peephole={cr.peephole_edits}, "
            f"dual-parcels={cr.parcels_packed}) | {run_res.cycles:,} cycles ({run_res.istack_hits} I-stack hits)"
        )
        if run_res.stdout:
            print("  Output   :", " | ".join(run_res.stdout))
        print(f"  Return   : {run_res.return_value} (C={run_res.return_trit:+d})")
        return 0
    if args.cmd == "info" or args.cmd is None:
        print("Duwamish-I (UE-1000) — Special Systems Section, Universal Entropics (WEAR Charter)")
        print("  Word / Tryte  : 36 trits (6 x 6-trit trytes = 4 x 9-trit nonets = 2 x 18-trit Setun)")
        print("  Memory Word   : 42 trits ([42, 36, 4]_3 SEC-DED + 3-erasure UE-TD2000)")
        print("  C-Complex     : 8 x 36-trit CPUs @ 30 ns (33.33 MHz), 533 MFLOPS peak TF36 (47.55b mant)")
        print("  T-Fabric      : 30 tiles (27 active + 3 hot spares), 6x6 TBFP-9 MACs @ 40 ns (48.6 GOPS peak / 17.0 GOPS sust)")
        print("  Main Store    : 43,046,721 T36 words (3^16, 81+9 banks, 292.8 MB eq.) + 177,147 Ferrite Core words")
        print("  OBB Level -9  : 2 Central Core + 28 Structural Cells:")
        for k, v in OBB_LEVEL_NINE_CELLS.items():
            print(f"    - {k}: {v}")
        return 0

    from workloads.cliodynamics import run_cliodynamics_demo
    from workloads.gallery import render_vsm_gallery_html
    from workloads.saew import run_saew_demo
    from workloads.seti import run_seti_demo

    machine = UE1000Machine()
    saew_rep = seti_rep = clio_rep = None

    if args.target in ("saew", "all"):
        print("\n=== WORKLOAD 1: Mini-SAEW Autoregressive Model (TBFP-9 + Kelvanov Gating + Live Fault Cutover) ===")
        saew_rep = run_saew_demo(machine)
        print(f"  Params: {saew_rep.param_count:,} | Loss: {saew_rep.initial_loss:.3f} -> {saew_rep.final_loss:.3f} ({saew_rep.steps} steps)")
        print(f"  Kelvanov Zero-Trit Power Saving: {saew_rep.avg_zero_trit_power_saving_pct:.1f}% | ABFT checks: {saew_rep.abft_checks} | Ashby Cutovers Survived: {saew_rep.cutovers_survived}")
        print(f"  1-Tile TBFP-9 Sample : {saew_rep.sample_tbfp9_output}")
        print(f"  1-Tile TBFP-3 Sample : {saew_rep.sample_tbfp3_output}")

    if args.target in ("seti", "all"):
        print("\n=== WORKLOAD 2: Oliver-Good Wideband SETI (Radix-3 TBFP-9C FFT + Section 8.1 TRIAD-36 Wald) ===")
        seti_rep = run_seti_demo(machine)
        for r in seti_rep.results:
            print(f"  {r.label:48s} | {r.final_decibans:+7.2f} dB | {r.verdict_name:22s} | {r.cpu_cycles:4d} CPU cyc ({r.istack_hits} I-stack hits)")

    if args.target in ("clio", "all"):
        print("\n=== WORKLOAD 3: Forrester-Ashby Cliodynamics (Exact TF36 + TBFP-9 Regional Coupling) ===")
        clio_rep = run_cliodynamics_demo(machine)
        print(f"  Executed {clio_rep.tf36_ops:,} exact TF36 ops + {clio_rep.tbfp9_coupling_calls} TBFP-9 coupling GEMMs across {clio_rep.regions} regions ({clio_rep.years_simulated} yrs):")
        for tr in clio_rep.trajectories:
            print(f"  {tr.policy_name:44s} | Pop={tr.final_population:.2f} Res={tr.final_resources:.2f} Leg={tr.final_legitimacy:.2f} | {tr.stability_label}")

    if args.target == "all" and saew_rep and seti_rep and clio_rep:
        out_path = render_vsm_gallery_html(
            machine.telemetry_snapshot(), saew_rep, seti_rep, clio_rep, Path(args.out)
        )
        print(f"\n=== STAGE 4: Beer VSM Operations Gallery written to {out_path.resolve()} ===")
        for lamp in machine.vcomplex.gallery_lamps:
            print(f"  [ALGEDONIC] {lamp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
