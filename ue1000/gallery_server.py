"""Live Local Web Server & Interactive Pask–Licklider Vector-CRT Operations Gallery.

Zero third-party dependencies (Python stdlib `http.server.ThreadingHTTPServer`).
Backs the Swiss-modernist Universal Entropics Operations Gallery with a LIVE running
`VOSSession` (`UE1000Machine` + Ferrite Vault `C-EXEC` + Tier 3b `CHIO` + Tier 4 `V-Complex`),
streaming real `OBB-729` Vector-CRT display lists (`-33 .. -121`), handling interactive
Licklider–Pask Light-Pen picks, real `< 75 us` Ashby hot-spare cutovers, live anchor
workload execution, and an embedded OBB Level -9 `V-OS` console terminal.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .machine import MMIO_BANK_CUTOVER_CMD, UE1000Machine
from .vcomplex import T_ACTIVE_TILES, T_TOTAL_TILES, TilePhysicalState
from .vector_crt import (
    CLIO_POLICY_BOX,
    CLIO_POLICY_PICK_TAG,
    CRTFrame,
    SAEW_TRAIN_PICK_TAG,
    SETI_SCENARIOS,
    light_pen_pick,
    render_clio_phase_crt,
    render_saew_crt,
    render_seti_scope_crt,
    render_vsm_ring_crt,
    run_live_clio_probe,
    run_live_seti_probe,
)
from .vos import VOSError, VOSSession


class GalleryController:
    """Thread-safe controller owning a live UE-1000 / V-OS / C-EXEC session and Vector-CRT."""

    def __init__(
        self,
        user: str = "BROOKS",
        prewarm: bool = False,
        session: Optional[VOSSession] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.vos = session if session is not None else VOSSession(user=user)
        self.active_scope: str = "vsm"
        self.clio_trajectories: List[Dict[str, object]] = []
        self.seti_probe: Optional[Dict[str, object]] = None
        self.saew_state: Optional[Dict[str, object]] = None
        self.last_lightpen_msg: str = "Light-pen ready. Click any active tile or scope control."

        if prewarm:
            self.ensure_scope_data("clio")
            self.ensure_scope_data("seti")

    @property
    def machine(self) -> UE1000Machine:
        return self.vos.machine

    def ensure_scope_data(self, scope: str) -> None:
        with self._lock:
            if scope == "clio" and not self.clio_trajectories:
                self.clio_trajectories = [
                    run_live_clio_probe(self.machine, 0.08, 0.08, label="BAU Extraction"),
                    run_live_clio_probe(self.machine, 0.60, 0.56, label="Metastable Frontier"),
                    run_live_clio_probe(self.machine, 0.85, 0.85, label="Ashby Corridor"),
                ]
            elif scope == "seti" and self.seti_probe is None:
                self.seti_probe = run_live_seti_probe(self.machine, kind="beacon")

    def render_crt(self, scope: Optional[str] = None) -> CRTFrame:
        with self._lock:
            sc = (scope or self.active_scope).lower()
            if sc not in ("vsm", "clio", "seti", "saew"):
                sc = "vsm"
            self.active_scope = sc
            if sc == "vsm":
                return render_vsm_ring_crt(self.machine)
            if sc == "clio":
                self.ensure_scope_data("clio")
                return render_clio_phase_crt(self.clio_trajectories)
            if sc == "seti":
                self.ensure_scope_data("seti")
                return render_seti_scope_crt(self.seti_probe or {})
            return render_saew_crt(self.saew_state)

    def get_state(self, scope: Optional[str] = None) -> Dict[str, object]:
        with self._lock:
            frame = self.render_crt(scope)
            snap = self.machine.telemetry_snapshot()
            jobs = [
                {
                    "job_id": j.job_id,
                    "name": j.name,
                    "owner": j.owner,
                    "state": j.state,
                    "cycles_used": j.cycles_used,
                    "preemptions": j.preemptions,
                    "exit_code": j.exit_code,
                    "abort_trap": j.abort_trap,
                }
                for j in self.vos.cexec.jobs.values()
            ]
            return {
                "telemetry": snap,
                "cexec": self.vos.cexec.kernel_stats(),
                "vos": {
                    "user": self.vos.current_user.username,
                    "role": self.vos.current_user.role,
                    "clearance": self.vos.current_user.clearance,
                    "prompt": self.vos.prompt,
                    "audit_tail": self.vos.audit_log[-10:],
                },
                "jobs": jobs,
                "active_scope": self.active_scope,
                "last_lightpen_msg": self.last_lightpen_msg,
                "crt": frame.to_dict(),
            }

    def handle_lightpen(self, scope: str, x: float, y: float, pick_tag: Optional[int] = None) -> Dict[str, object]:
        """Process an authentic Licklider–Pask Light-Pen pick on the active Vector-CRT."""
        with self._lock:
            frame = self.render_crt(scope)
            tag = pick_tag if (pick_tag is not None and pick_tag != 0) else light_pen_pick(frame, x, y)
            msg = f"Light-pen @ ({int(round(x))}, {int(round(y))}): no pick target hit."

            if frame.scope == "vsm" and tag is not None and 1 <= tag <= T_TOTAL_TILES:
                phys_id = tag - 1
                st = self.machine.vcomplex.tiles[phys_id]
                if st.role == "ACTIVE" and st.logical_id is not None:
                    ev = self.machine.vcomplex.trigger_tile_cutover(
                        logical_tile=st.logical_id,
                        fault_reason="CRT_LIGHTPEN_ABFT_FAULT",
                        timestamp_ns=self.machine.global_cycles * 30,
                    )
                    if ev.kind == "SPARES_EXHAUSTED":
                        msg = f"CRITICAL: Bay P{phys_id:02d} faulted — ALL 3 HOT SPARES EXHAUSTED (Trap 12)!"
                    else:
                        msg = (
                            f"LIGHT-PEN FAULT -> Ashby Cutover #{ev.event_id}: Bay P{phys_id:02d} (L{ev.logical_id:02d}) "
                            f"-> Hot Spare P{ev.new_physical_id:02d} in {ev.hw_cutover_us:.1f} us!"
                        )
                else:
                    msg = f"Bay P{phys_id:02d} is {st.role} (pick an ACTIVE green blade to inject a fault)."

            elif frame.scope == "clio" and tag == CLIO_POLICY_PICK_TAG:
                bx0, by0, bx1, by1 = CLIO_POLICY_BOX
                cons = max(0.02, min(0.98, (x - bx0) / (bx1 - bx0)))
                abat = max(0.02, min(0.98, (y - by0) / (by1 - by0)))
                probe = run_live_clio_probe(self.machine, cons, abat)
                self.clio_trajectories.append(probe)
                if len(self.clio_trajectories) > 10:
                    self.clio_trajectories = self.clio_trajectories[-10:]
                msg = (
                    f"LIGHT-PEN CLIO PROBE (C={cons:.2f}, A={abat:.2f}) -> "
                    f"Pop={probe['final_population']} Res={probe['final_resources']} [{probe['stability_label']}]"
                )

            elif frame.scope == "seti" and tag in SETI_SCENARIOS:
                kind_key, lbl = SETI_SCENARIOS[tag]
                self.seti_probe = run_live_seti_probe(self.machine, kind=kind_key)
                msg = (
                    f"LIGHT-PEN SETI [{lbl}] -> Peak bin {self.seti_probe['peak_bin']}, "
                    f"{self.seti_probe['final_db']:+.2f} dB -> {self.seti_probe['verdict_label']}"
                )

            elif frame.scope == "saew" and tag == SAEW_TRAIN_PICK_TAG:
                from workloads.saew import run_saew_demo
                rep = run_saew_demo(self.machine, steps=60)
                self.saew_state = asdict(rep)
                msg = (
                    f"LIGHT-PEN SAEW TRAINED: loss {rep.initial_loss:.3f} -> {rep.final_loss:.3f} | "
                    f"zero-trit saved {rep.avg_zero_trit_power_saving_pct:.1f}% | cutovers={rep.cutovers_survived}"
                )

            self.last_lightpen_msg = msg
            state = self.get_state(frame.scope)
            state["pick_tag"] = tag
            return state

    def inject_tile_fault(self, logical_tile: int, reason: str = "GALLERY_ABFT_FAULT") -> Dict[str, object]:
        with self._lock:
            ev = self.machine.vcomplex.trigger_tile_cutover(
                logical_tile=logical_tile,
                fault_reason=reason,
                timestamp_ns=self.machine.global_cycles * 30,
            )
            if ev.kind == "SPARES_EXHAUSTED":
                self.last_lightpen_msg = f"CRITICAL: Logical T{logical_tile} faulted — ALL 3 SPARES EXHAUSTED (Trap 12)!"
            else:
                self.last_lightpen_msg = (
                    f"Ashby Cutover #{ev.event_id}: T{logical_tile:02d} (Bay {ev.old_physical_id:02d}) -> "
                    f"Spare {ev.new_physical_id:02d} in {ev.hw_cutover_us:.1f} us"
                )
            return self.get_state()

    def reset_tiles(self) -> Dict[str, object]:
        """Maintenance Bay blade hot-swap restoring 27 active tiles + 3 hot spares."""
        with self._lock:
            vc = self.machine.vcomplex
            vc.tile_lut = list(range(T_ACTIVE_TILES))
            vc.spare_tiles = list(range(T_ACTIVE_TILES, T_TOTAL_TILES))
            for p in range(T_ACTIVE_TILES):
                vc.tiles[p] = TilePhysicalState(physical_id=p, role="ACTIVE", logical_id=p)
            for p in range(T_ACTIVE_TILES, T_TOTAL_TILES):
                vc.tiles[p] = TilePhysicalState(physical_id=p, role="HOT_SPARE", logical_id=None)
            vc.gallery_lamps.append("MAINTENANCE BAY: Faulted UE-TECL blades swapped; 3 hot spares restored.")
            self.last_lightpen_msg = "Maintenance Bay: all 27 active tiles + 3 hot spares restored."
            return self.get_state()

    def trigger_bank_cutover(self, logical_bank: int) -> Dict[str, object]:
        with self._lock:
            self.machine.mem.write(MMIO_BANK_CUTOVER_CMD, logical_bank, supervisor=True)
            self.last_lightpen_msg = f"MMIO Ashby Bank Cutover completed for logical bank {logical_bank}."
            return self.get_state()

    def ack_alarms(self) -> Dict[str, object]:
        with self._lock:
            n = self.machine.vcomplex.acknowledge_alarms()
            self.last_lightpen_msg = f"Acknowledged {n} Beer Algedonic alarm(s)."
            return self.get_state()

    def run_vos_command(self, command: str) -> Dict[str, object]:
        with self._lock:
            prompt_before = self.vos.prompt
            error: Optional[str] = None
            lines: List[str] = []
            try:
                lines = self.vos.execute_line(command)
            except Exception as exc:
                error = str(exc)
            state = self.get_state()
            state["vos_result"] = {
                "prompt": prompt_before,
                "command": command,
                "lines": lines,
                "error": error,
            }
            return state

    def run_workload(self, target: str) -> Dict[str, object]:
        with self._lock:
            t = target.lower()
            if t == "saew":
                from workloads.saew import run_saew_demo
                rep = run_saew_demo(self.machine, steps=60)
                self.saew_state = asdict(rep)
                self.active_scope = "saew"
                self.last_lightpen_msg = f"Mini-SAEW complete: loss {rep.initial_loss:.3f} -> {rep.final_loss:.3f}"
            elif t == "seti":
                self.seti_probe = run_live_seti_probe(self.machine, kind="beacon")
                self.active_scope = "seti"
                self.last_lightpen_msg = f"Oliver-Good SETI complete: {self.seti_probe['verdict_label']}"
            elif t == "clio":
                self.clio_trajectories = []
                self.ensure_scope_data("clio")
                self.active_scope = "clio"
                self.last_lightpen_msg = "Forrester-Ashby Cliodynamics 3-regime TF36 sweep complete."
            else:
                raise ValueError(f"Unknown workload target {target!r}")
            return self.get_state(self.active_scope)


# ---------------------------------------------------------------------------
# Standalone Zero-Dependency Swiss-Modernist Web + Phosphor Vector-CRT UI
# ---------------------------------------------------------------------------

def render_live_gallery_page() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>UE-1000 Duwamish-I · Live Operations Gallery & OBB-729 Vector-CRT</title>
<style>
  :root {
    --black: #161614;
    --panel: #21211e;
    --panel2: #191916;
    --amber: #e8962e;
    --card: #f4efe4;
    --muted: #9f998c;
    --green: #4fb477;
    --red: #d9534f;
    --border: #38362f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 18px 22px; background: var(--black); color: var(--card);
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; line-height: 1.4;
  }
  header {
    border-bottom: 3px solid var(--amber); padding-bottom: 12px; margin-bottom: 18px;
    display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 12px;
  }
  .brand-top { letter-spacing: 0.22em; font-weight: 700; font-size: 12px; text-transform: uppercase; }
  .brand-sub { letter-spacing: 0.14em; font-size: 11px; color: var(--muted); text-transform: uppercase; }
  .brand-sub span { color: var(--amber); font-weight: 700; }
  h1 { margin: 4px 0 0 0; font-size: 22px; letter-spacing: 0.04em; }
  .badge {
    display: inline-block; padding: 3px 8px; font-family: monospace; font-size: 11px; font-weight: 700;
    border: 1px solid var(--border); background: var(--panel2);
  }
  .badge.ok { color: var(--green); border-color: var(--green); }
  .badge.warn { color: var(--amber); border-color: var(--amber); }
  .badge.pain { color: var(--red); border-color: var(--red); }
  .layout { display: grid; grid-template-columns: 1.25fr 1fr; gap: 18px; }
  @media (max-width: 1080px) { .layout { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 4px; padding: 14px;
    margin-bottom: 16px;
  }
  .card h2 {
    margin: 0 0 10px 0; font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--amber); border-bottom: 1px solid var(--border); padding-bottom: 6px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
  .btn {
    background: var(--amber); color: var(--black); border: none; padding: 6px 11px;
    font-weight: 700; font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase; cursor: pointer;
  }
  .btn:hover { opacity: 0.9; }
  .btn.sec { background: #34332d; color: var(--card); border: 1px solid var(--border); }
  .btn.active { background: var(--card); color: var(--black); }
  .crt-bezel {
    background: #0b0d0b; border: 3px solid #3a372f; border-radius: 10px; padding: 10px;
    box-shadow: inset 0 0 42px rgba(232,150,46,0.12);
    position: relative;
  }
  canvas#crt {
    width: 100%; height: auto; aspect-ratio: 1 / 1; display: block;
    background: radial-gradient(circle at 50% 50%, #111611 0%, #070907 100%);
    cursor: crosshair; border-radius: 6px;
  }
  .crt-meta {
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
    font-family: monospace; font-size: 11px; color: var(--muted); margin-top: 8px;
  }
  .hept-bar {
    font-family: monospace; font-size: 10px; color: var(--amber);
    background: #121210; border: 1px solid var(--border); padding: 5px 8px; margin-top: 6px;
    overflow-x: auto; white-space: nowrap;
  }
  .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 10px; }
  .kpi { background: var(--panel2); border: 1px solid var(--border); padding: 7px 9px; }
  .kpi .lbl { font-size: 9px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
  .kpi .val { font-size: 16px; font-weight: 700; font-family: monospace; }
  .tile-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 5px; margin: 8px 0; }
  .tile {
    padding: 6px 3px; text-align: center; font-family: monospace; font-size: 10px;
    border: 1px solid var(--border); cursor: pointer;
  }
  .tile.ACTIVE { background: #1e3124; border-color: var(--green); color: #d5f5e0; }
  .tile.HOT_SPARE { background: #2c261b; border-color: var(--amber); color: var(--amber); }
  .tile.FAULTED_BAY_LIT { background: #3d1c1b; border-color: var(--red); color: #ffb3b0; font-weight: 700; }
  .log {
    background: #11110f; border: 1px solid var(--border); padding: 8px;
    font-family: monospace; font-size: 11px; max-height: 140px; overflow-y: auto; color: #dcd6c8;
  }
  .term-out {
    background: #0d0e0c; border: 1px solid var(--border); padding: 10px;
    font-family: monospace; font-size: 11px; height: 210px; overflow-y: auto;
    color: #e6dfd0; white-space: pre-wrap;
  }
  .term-in { display: flex; gap: 6px; margin-top: 8px; }
  .term-in input {
    flex: 1; background: #11110f; color: var(--card); border: 1px solid var(--amber);
    padding: 7px 10px; font-family: monospace; font-size: 12px;
  }
  .lp-banner {
    background: #151713; border-left: 3px solid var(--amber); padding: 6px 10px;
    font-family: monospace; font-size: 11px; margin-top: 8px; color: var(--card);
  }
</style>
</head>
<body>
<header>
  <div>
    <div class="brand-top">Universal Entropics · Special Systems Section</div>
    <h1>DUWAMISH-I (UE-1000) — LIVE VSM &amp; OBB-729 VECTOR-CRT GALLERY</h1>
    <div class="brand-sub"><span>WEST SEATTLE</span> · OBB UNSEEN LEVEL -9 · LIVE HARDWARE BACKEND (TIER 1–4 + C-EXEC + V-OS)</div>
  </div>
  <div style="text-align:right;font-family:monospace;font-size:12px;">
    <div id="health-badge" class="badge ok">VSM: +1 NOMINAL</div>
    <div style="color:var(--muted);margin-top:4px;" id="user-badge">PRINCIPAL: BROOKS</div>
  </div>
</header>

<div class="layout">
  <div>
    <div class="card">
      <h2>
        <span>Pask–Licklider OBB-729 Phosphor Vector-CRT</span>
        <span id="crt-coords" style="font-family:monospace;font-size:11px;color:var(--muted);">BEAM (0, 0)</span>
      </h2>
      <div class="tabs">
        <button class="btn sec" id="tab-vsm" onclick="switchScope('vsm')">1. VSM Tile Ring</button>
        <button class="btn sec" id="tab-clio" onclick="switchScope('clio')">2. Cliodynamics TF36</button>
        <button class="btn sec" id="tab-seti" onclick="switchScope('seti')">3. SETI Radix-3</button>
        <button class="btn sec" id="tab-saew" onclick="switchScope('saew')">4. SAEW AI Scope</button>
      </div>
      <div class="crt-bezel">
        <canvas id="crt" width="760" height="760"></canvas>
      </div>
      <div class="lp-banner" id="lp-msg">Light-pen ready.</div>
      <div class="crt-meta">
        <span id="crt-stats">OBB-729: 0 trytes (0 T36 words)</span>
        <span id="crt-title">Scope</span>
      </div>
      <div class="hept-bar" id="crt-hept">HEPTAVINTIMAL DISPLAY LIST: —</div>
    </div>
  </div>

  <div>
    <div class="card">
      <h2>Live Hardware Telemetry (Tiers 1–4)</h2>
      <div class="kpi-row">
        <div class="kpi"><div class="lbl">Global Cycles</div><div class="val" id="k-cyc">0</div></div>
        <div class="kpi"><div class="lbl">Active / Spares</div><div class="val" id="k-tiles">27/3</div></div>
        <div class="kpi"><div class="lbl">Spare Banks</div><div class="val" id="k-banks">9</div></div>
        <div class="kpi"><div class="lbl">C-EXEC Traps</div><div class="val" id="k-traps">0</div></div>
      </div>
      <div class="tile-grid" id="tile-grid"></div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin:8px 0;">
        <button class="btn" onclick="faultRandomTile()">Inject Live Tile Fault</button>
        <button class="btn sec" onclick="postJSON('/api/fault/bank', {logical_bank: 7})">Bank Cutover #7</button>
        <button class="btn sec" onclick="postJSON('/api/reset/tiles', {})">Hot-Swap Blades</button>
        <button class="btn sec" onclick="postJSON('/api/ack', {})">Ack Alarms</button>
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;">
        <button class="btn sec" onclick="runWorkload('seti')">Run Live SETI</button>
        <button class="btn sec" onclick="runWorkload('clio')">Run Live Cliodynamics</button>
        <button class="btn sec" onclick="runWorkload('saew')">Train Mini-SAEW (TBFP-9)</button>
      </div>
      <div class="log" id="algedonic-log"></div>
    </div>

    <div class="card">
      <h2>
        <span>Tier 4 V-OS Level -9 Console</span>
        <span style="font-size:10px;color:var(--muted);">CHIO · C-EXEC · 9 LANGUAGES</span>
      </h2>
      <div class="term-out" id="term-out">DUWAMISH-I (UE-1000) V-OS REV 1.5 LIVE CONSOLE READY.
Type HELP, VSM, LS ALL, CAT SILO:BYURAKAN_71.MEMO, APL +/ 1 2 3 4 5, FORTH 6 7 * ., or CONVERSE FIND "SETUN".</div>
      <form class="term-in" onsubmit="sendVOS(event)">
        <input id="vos-input" type="text" placeholder="V-OS command (e.g. LS ALL, VSM, APL +/ ⍳ 9, FORTH 6 7 * .)" autocomplete="off" />
        <button class="btn" type="submit">Send</button>
      </form>
    </div>
  </div>
</div>

<script>
let STATE = null;
const canvas = document.getElementById('crt');
const ctx = canvas.getContext('2d');
const PHOSPHOR = {
  '-1': '#ff5c57',
  '0':  '#5e5948',
  '1':  '#e8962e',
  '2':  '#57d687',
  '3':  '#fff3d1',
};

function toPx(x, y) {
  const pad = 16;
  const span = canvas.width - 2 * pad;
  return [
    pad + ((x + 364) / 728) * span,
    pad + ((364 - y) / 728) * span,
  ];
}

function fromPx(px, py) {
  const rect = canvas.getBoundingClientRect();
  const padFrac = 16 / canvas.width;
  const nx = (px - rect.left) / rect.width;
  const ny = (py - rect.top) / rect.height;
  const u = (nx - padFrac) / (1 - 2 * padFrac);
  const v = (ny - padFrac) / (1 - 2 * padFrac);
  return [
    Math.max(-364, Math.min(364, Math.round(u * 728 - 364))),
    Math.max(-364, Math.min(364, Math.round(364 - v * 728))),
  ];
}

function drawCRT(frame) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Subtle phosphor graticule
  ctx.save();
  ctx.strokeStyle = 'rgba(232,150,46,0.06)';
  ctx.lineWidth = 1;
  for (let g = -300; g <= 300; g += 100) {
    const [x1, y1] = toPx(g, -364); const [x2, y2] = toPx(g, 364);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    const [a1, b1] = toPx(-364, g); const [a2, b2] = toPx(364, g);
    ctx.beginPath(); ctx.moveTo(a1, b1); ctx.lineTo(a2, b2); ctx.stroke();
  }
  ctx.restore();

  function setBeam(inten) {
    const col = PHOSPHOR[String(inten)] || PHOSPHOR['1'];
    ctx.strokeStyle = col;
    ctx.fillStyle = col;
    ctx.shadowColor = col;
    ctx.shadowBlur = inten === 0 ? 0 : (inten === 3 ? 12 : 7);
    ctx.lineWidth = inten === 0 ? 1 : 2;
  }

  for (const r of frame.rects || []) {
    setBeam(r.intensity);
    const [x1, y1] = toPx(r.x1, r.y1);
    const [x2, y2] = toPx(r.x2, r.y2);
    ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
  }
  for (const s of frame.strokes || []) {
    setBeam(s.intensity);
    const [x1, y1] = toPx(s.x1, s.y1);
    const [x2, y2] = toPx(s.x2, s.y2);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  for (const c of frame.circles || []) {
    setBeam(c.intensity);
    const [cx, cy] = toPx(c.cx, c.cy);
    const pr = (c.r / 728) * (canvas.width - 32);
    ctx.beginPath(); ctx.arc(cx, cy, pr, 0, Math.PI * 2); ctx.stroke();
  }
  for (const p of frame.points || []) {
    setBeam(p.intensity);
    const [px, py] = toPx(p.x, p.y);
    ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
  }
  for (const t of frame.texts || []) {
    setBeam(t.intensity);
    const [tx, ty] = toPx(t.x, t.y);
    ctx.font = 'bold ' + (10 + (t.scale || 1) * 2) + 'px monospace';
    ctx.fillText(t.text, tx, ty);
  }
  ctx.shadowBlur = 0;

  document.getElementById('crt-stats').textContent =
    'OBB-729 STREAM: ' + frame.tryte_count + ' trytes (' + frame.t36_word_count + ' T36 words)';
  document.getElementById('crt-title').textContent = frame.title;
  document.getElementById('crt-hept').textContent =
    'HEPTAVINTIMAL T36 DISPLAY LIST: ' + (frame.hept_preview || []).join('  ');
}

function renderState(st) {
  STATE = st;
  drawCRT(st.crt);
  document.getElementById('lp-msg').textContent = st.last_lightpen_msg || '';
  ['vsm', 'clio', 'seti', 'saew'].forEach(sc => {
    document.getElementById('tab-' + sc).className = 'btn ' + (st.active_scope === sc ? 'active' : 'sec');
  });

  const vc = st.telemetry.v_complex;
  const hb = document.getElementById('health-badge');
  if (vc.health_trit > 0) { hb.textContent = 'VSM: +1 NOMINAL'; hb.className = 'badge ok'; }
  else if (vc.health_trit === 0) { hb.textContent = 'VSM: 0 WARNING'; hb.className = 'badge warn'; }
  else { hb.textContent = 'VSM: -1 ALGEDONIC PAIN'; hb.className = 'badge pain'; }
  document.getElementById('user-badge').textContent =
    'PRINCIPAL: ' + st.vos.user + ' (CL ' + st.vos.clearance + ')';

  document.getElementById('k-cyc').textContent = st.telemetry.c_complex.global_cycles.toLocaleString();
  document.getElementById('k-tiles').textContent =
    st.telemetry.t_fabric.active_tiles + '/' + st.telemetry.t_fabric.hot_spares_remaining;
  document.getElementById('k-banks').textContent = st.telemetry.memory.spare_banks_remaining;
  document.getElementById('k-traps').textContent = st.cexec.total_traps;

  const grid = document.getElementById('tile-grid');
  grid.innerHTML = '';
  st.telemetry.t_fabric.tiles.forEach(t => {
    const div = document.createElement('div');
    div.className = 'tile ' + t.role;
    const tag = t.role === 'ACTIVE' ? ('L' + String(t.logical_id).padStart(2,'0')) : (t.role === 'HOT_SPARE' ? 'SPR' : 'FLT');
    div.innerHTML = 'P' + String(t.physical_id).padStart(2,'0') + '<br>' + tag;
    if (t.role === 'ACTIVE') {
      div.onclick = () => postJSON('/api/fault/tile', {logical_tile: t.logical_id, reason: 'GALLERY_CLICK_FAULT'});
    }
    grid.appendChild(div);
  });

  const log = document.getElementById('algedonic-log');
  const lamps = vc.gallery_lamps || [];
  log.innerHTML = lamps.length
    ? lamps.slice().reverse().map(l => '&gt; ' + l).join('<br>')
    : '&gt; ALL 27 ACTIVE TILES + 3 HOT SPARES + 81 BANKS NOMINAL.';
}

async function fetchState(scope) {
  const url = scope ? ('/api/state?scope=' + encodeURIComponent(scope)) : '/api/state';
  const r = await fetch(url);
  renderState(await r.json());
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  });
  const data = await r.json();
  renderState(data);
  return data;
}

function switchScope(sc) { fetchState(sc); }

function faultRandomTile() {
  if (!STATE) return;
  const act = STATE.telemetry.t_fabric.tiles.filter(t => t.role === 'ACTIVE');
  if (!act.length) return;
  const pick = act[Math.floor(Math.random() * act.length)];
  postJSON('/api/fault/tile', {logical_tile: pick.logical_id, reason: 'RANDOM_DRILL_ABFT'});
}

function runWorkload(target) {
  document.getElementById('lp-msg').textContent = 'Running live ' + target.toUpperCase() + ' workload on UE-1000...';
  postJSON('/api/workload', {target});
}

canvas.addEventListener('mousemove', ev => {
  const [tx, ty] = fromPx(ev.clientX, ev.clientY);
  document.getElementById('crt-coords').textContent = 'BEAM (' + tx + ', ' + ty + ')';
});

canvas.addEventListener('click', ev => {
  if (!STATE) return;
  const [tx, ty] = fromPx(ev.clientX, ev.clientY);
  postJSON('/api/lightpen', {scope: STATE.active_scope, x: tx, y: ty});
});

async function sendVOS(ev) {
  ev.preventDefault();
  const inp = document.getElementById('vos-input');
  const cmd = inp.value.trim();
  if (!cmd) return;
  inp.value = '';
  const data = await postJSON('/api/vos', {command: cmd});
  const vr = data.vos_result || {};
  const term = document.getElementById('term-out');
  term.textContent += '\\n' + (vr.prompt || 'V-OS> ') + cmd;
  if (vr.error) term.textContent += '\\n  [ERROR] ' + vr.error;
  for (const ln of vr.lines || []) term.textContent += '\\n  ' + ln;
  term.scrollTop = term.scrollHeight;
}

fetchState('vsm');
</script>
</body>
</html>
"""


def create_gallery_server(
    host: str = "127.0.0.1",
    port: int = 8089,
    user: str = "BROOKS",
    prewarm: bool = False,
    controller: Optional[GalleryController] = None,
) -> Tuple[ThreadingHTTPServer, GalleryController]:
    """Create a ThreadingHTTPServer serving the live Operations Gallery & OBB-729 Vector-CRT."""
    ctrl = controller if controller is not None else GalleryController(user=user, prewarm=prewarm)

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return  # Keep terminal output clean

        def _send_json(self, obj: Dict[str, object], status: int = HTTPStatus.OK) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_html(self, html: str) -> None:
            raw = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json_body(self) -> Dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            if parsed.path in ("/", "/index.html"):
                self._send_html(render_live_gallery_page())
                return
            if parsed.path == "/api/state":
                scope = qs.get("scope", [None])[0]
                self._send_json(ctrl.get_state(scope))
                return
            if parsed.path == "/api/crt":
                scope = qs.get("scope", [None])[0]
                self._send_json(ctrl.render_crt(scope).to_dict())
                return
            self._send_json({"error": f"Not found: {parsed.path}"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                body = self._read_json_body()
                if parsed.path == "/api/lightpen":
                    res = ctrl.handle_lightpen(
                        scope=str(body.get("scope", ctrl.active_scope)),
                        x=float(body.get("x", 0.0)),
                        y=float(body.get("y", 0.0)),
                        pick_tag=int(body["pick_tag"]) if body.get("pick_tag") is not None else None,
                    )
                    self._send_json(res)
                    return
                if parsed.path == "/api/fault/tile":
                    res = ctrl.inject_tile_fault(
                        logical_tile=int(body.get("logical_tile", 0)),
                        reason=str(body.get("reason", "GALLERY_ABFT_FAULT")),
                    )
                    self._send_json(res)
                    return
                if parsed.path == "/api/fault/bank":
                    res = ctrl.trigger_bank_cutover(int(body.get("logical_bank", 0)))
                    self._send_json(res)
                    return
                if parsed.path == "/api/reset/tiles":
                    self._send_json(ctrl.reset_tiles())
                    return
                if parsed.path == "/api/ack":
                    self._send_json(ctrl.ack_alarms())
                    return
                if parsed.path == "/api/vos":
                    self._send_json(ctrl.run_vos_command(str(body.get("command", ""))))
                    return
                if parsed.path == "/api/workload":
                    self._send_json(ctrl.run_workload(str(body.get("target", "seti"))))
                    return
                self._send_json({"error": f"Not found: {parsed.path}"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    server = ThreadingHTTPServer((host, port), _Handler)
    return server, ctrl


def serve_gallery(
    host: str = "127.0.0.1",
    port: int = 8089,
    user: str = "BROOKS",
    prewarm: bool = False,
) -> None:
    """Run the live Operations Gallery & OBB-729 Vector-CRT server until Ctrl-C."""
    server, _ = create_gallery_server(host=host, port=port, user=user, prewarm=prewarm)
    actual_host, actual_port = server.server_address
    print("=========================================================================")
    print("  DUWAMISH-I (UE-1000) — Live VSM Operations Gallery & OBB-729 Vector-CRT")
    print(f"  Listening on http://{actual_host}:{actual_port}  (Principal: {user.upper()})")
    print("  Press Ctrl-C to stop.")
    print("=========================================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Operations Gallery server.")
    finally:
        server.server_close()
