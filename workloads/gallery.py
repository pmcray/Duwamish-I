"""Stage 4: Interactive Stafford Beer VSM Operations Gallery HTML Console.

Renders a standalone, zero-dependency HTML/JS Operations Gallery in Universal Entropics'
Special Systems Section Swiss modernist house style (Process Black #1b1b1b, Entropic Amber
#e8962e, Card Stock #f4efe4) populated with live telemetry from all three workloads, the
locked OBB Unseen Level -9 30-cell floorplan, and interactive client-side tile fault
injection + < 75 us Ashby hot-spare cutover simulation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from .cliodynamics import CliodynamicsReport
from .saew import SAEWTrainingReport
from .seti import SETIReport


def render_vsm_gallery_html(
    telemetry: Dict[str, object],
    saew: SAEWTrainingReport,
    seti: SETIReport,
    clio: CliodynamicsReport,
    output_path: Path,
) -> Path:
    payload = {
        "telemetry": telemetry,
        "saew": asdict(saew),
        "seti": asdict(seti),
        "clio": asdict(clio),
    }
    data_json = json.dumps(payload, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>UE-1000 Duwamish-I · Beer VSM Operations Gallery (OBB Level -9)</title>
<style>
  :root {{
    --black: #1b1b1b;
    --panel: #242422;
    --amber: #e8962e;
    --card: #f4efe4;
    --muted: #a6a094;
    --green: #4fb477;
    --red: #d9534f;
    --border: #3a3833;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px; background: var(--black); color: var(--card);
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; line-height: 1.45;
  }}
  header {{
    border-bottom: 3px solid var(--amber); padding-bottom: 14px; margin-bottom: 22px;
    display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 12px;
  }}
  .brand-top {{ letter-spacing: 0.22em; font-weight: 700; font-size: 13px; text-transform: uppercase; }}
  .brand-sub {{ letter-spacing: 0.16em; font-size: 11px; color: var(--muted); text-transform: uppercase; }}
  .brand-sub span {{ color: var(--amber); font-weight: 700; }}
  h1 {{ margin: 4px 0 0 0; font-size: 24px; letter-spacing: 0.04em; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
  .card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 4px; padding: 16px;
  }}
  .card h2 {{
    margin: 0 0 12px 0; font-size: 14px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--amber); border-bottom: 1px solid var(--border); padding-bottom: 6px;
  }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px; }}
  .kpi {{ background: #191917; border: 1px solid var(--border); padding: 8px 10px; }}
  .kpi .lbl {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
  .kpi .val {{ font-size: 18px; font-weight: 700; color: var(--card); font-family: monospace; }}
  .tile-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 6px; margin: 10px 0; }}
  .tile {{
    padding: 8px 4px; text-align: center; font-family: monospace; font-size: 11px;
    border: 1px solid var(--border); cursor: pointer; transition: transform 0.1s;
  }}
  .tile:hover {{ transform: scale(1.04); border-color: var(--amber); }}
  .tile.ACTIVE {{ background: #203126; border-color: var(--green); color: #d5f5e0; }}
  .tile.HOT_SPARE {{ background: #2c271d; border-color: var(--amber); color: var(--amber); }}
  .tile.FAULTED_BAY_LIT {{ background: #3d1d1c; border-color: var(--red); color: #ffb3b0; font-weight: 700; }}
  .btn {{
    background: var(--amber); color: var(--black); border: none; padding: 7px 12px;
    font-weight: 700; font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; cursor: pointer;
  }}
  .btn:hover {{ opacity: 0.9; }}
  .log {{
    background: #141412; border: 1px solid var(--border); padding: 10px;
    font-family: monospace; font-size: 11px; max-height: 150px; overflow-y: auto; color: #dcd6c8;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; }}
  .trit-pos {{ color: var(--green); font-weight: 700; font-family: monospace; }}
  .trit-zero {{ color: var(--amber); font-weight: 700; font-family: monospace; }}
  .trit-neg {{ color: var(--red); font-weight: 700; font-family: monospace; }}
  .cell-bar {{ display: flex; justify-content: space-between; font-size: 12px; padding: 4px 0; border-bottom: 1px dotted var(--border); }}
  svg {{ width: 100%; height: 150px; background: #141412; border: 1px solid var(--border); }}
  pre.sample {{ background: #141412; padding: 8px; font-size: 11px; white-space: pre-wrap; border-left: 3px solid var(--amber); }}
</style>
</head>
<body>
<header>
  <div>
    <div class="brand-top">Universal Entropics · Special Systems Section</div>
    <h1>DUWAMISH-I (UE-1000) — VSM OPERATIONS GALLERY</h1>
    <div class="brand-sub"><span>WEST SEATTLE</span> · WEAR UARC/FFRDC · OBB UNSEEN LEVEL -9 (28 STRUCTURAL + 2 CENTRAL CORE CELLS)</div>
  </div>
  <div style="text-align:right;font-family:monospace;font-size:12px;">
    <div>WORD: 36 TRITS (6 TRYTES) · ECC: [42,36,4]&#8323;</div>
    <div style="color:var(--amber);">T-FABRIC: 48.6 GOPS PEAK / 17.0 GOPS SUST (TBFP-9)</div>
  </div>
</header>

<div class="grid">
  <div class="card">
    <h2>Tier 2 &amp; 4 · 30-Tile T-Fabric &amp; Ashby &lt;75 &mu;s Cutover</h2>
    <div class="kpi-row">
      <div class="kpi"><div class="lbl">Active Tiles</div><div class="val" id="k-active">27</div></div>
      <div class="kpi"><div class="lbl">Hot Spares</div><div class="val" id="k-spares">3</div></div>
      <div class="kpi"><div class="lbl">Zero-Trit Saved</div><div class="val" id="k-zsave">0%</div></div>
    </div>
    <div style="font-size:11px;color:var(--muted);margin-bottom:6px;">
      Click any <b style="color:var(--green)">ACTIVE</b> blade below to inject an ABFT checksum fault and watch V-Complex System 3* trigger live Ashby &lt;75 &mu;s hot-spare cutover:
    </div>
    <div class="tile-grid" id="tile-grid"></div>
    <div style="margin-bottom:8px;display:flex;gap:8px;">
      <button class="btn" onclick="injectRandomFault()">Inject Random Tile Fault</button>
      <button class="btn" style="background:#444;color:#eee;" onclick="resetTiles()">Swap Faulted Blades</button>
    </div>
    <div class="log" id="algedonic-log"></div>
  </div>

  <div class="card">
    <h2>OBB Unseen Level -9 · Locked 30-Cell Stratigraphy</h2>
    <div style="font-size:11px;color:var(--muted);margin-bottom:8px;">
      Above: L-7 Special Collections &rarr; L-8 Digitisation (J. U. Jameson) &rarr; vertical chute.<br/>
      Below: L-10 Plant/Cryogenics (2.6 MW / Freon). 30 m rock gap isolates air-gapped L-11/L-12.
    </div>
    <div id="cell-list"></div>
    <h2 style="margin-top:16px;">Tier 3a · Main Ternary Store &amp; ECC</h2>
    <div class="kpi-row">
      <div class="kpi"><div class="lbl">T-DRAM Words</div><div class="val">43.05M</div></div>
      <div class="kpi"><div class="lbl">Ferrite Core</div><div class="val">177.1k</div></div>
      <div class="kpi"><div class="lbl">Spare Banks</div><div class="val" id="k-banks">9</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Workload 1 · Mini-SAEW (TBFP-9 + Kelvanov Gating)</h2>
    <div class="kpi-row">
      <div class="kpi"><div class="lbl">Start Loss</div><div class="val" id="s-loss0"></div></div>
      <div class="kpi"><div class="lbl">Final Loss</div><div class="val" id="s-loss1"></div></div>
      <div class="kpi"><div class="lbl">Cutovers Survived</div><div class="val" id="s-cut"></div></div>
    </div>
    <svg id="saew-svg" viewBox="0 0 360 150"></svg>
    <pre class="sample" id="saew-sample"></pre>
    <div style="font-size:11px;color:var(--muted);">
      Honest 1973 Full-Scale Targets: <b>SAEW-18M</b> = 28.6 days (39.6M/43.05M words) · <b>SAEW-32M</b> = 94.8 days (1 quarter).
    </div>
  </div>

  <div class="card">
    <h2>Workload 2 · Oliver-Good SETI (Radix-3 + TRIAD-36 Wald)</h2>
    <table>
      <thead><tr><th>Signal</th><th>Peak/Med</th><th>Decibans</th><th>TRIAD-36 Verdict</th></tr></thead>
      <tbody id="seti-body"></tbody>
    </table>
    <div style="font-size:11px;color:var(--muted);margin-top:10px;">
      Full 27-Tile Scale: 3&#185;&#179; (1,594,323-pt) complex radix-3 FFT in <b>12.1 ms</b> (~131.9M ch/sec @ 16.0 GOPS TBFP-9C).
    </div>
  </div>

  <div class="card" style="grid-column: 1 / -1;">
    <h2>Workload 3 · Forrester-Ashby Cliodynamics (Exact TF36 + TBFP-9 Coupling)</h2>
    <table>
      <thead>
        <tr><th>Policy Regime</th><th>Conserv.</th><th>Abate.</th><th>Final Pop</th><th>Final Res</th><th>Final Legit</th><th>Ashby Stability Verdict</th></tr>
      </thead>
      <tbody id="clio-body"></tbody>
    </table>
  </div>
</div>

<script>
const DATA = {data_json};
let tiles = JSON.parse(JSON.stringify(DATA.telemetry.t_fabric.tiles));
let lamps = [...DATA.telemetry.v_complex.gallery_lamps];

function renderTiles() {{
  const grid = document.getElementById('tile-grid');
  grid.innerHTML = '';
  let active = 0, spares = 0;
  tiles.forEach(t => {{
    if (t.role === 'ACTIVE') active++;
    if (t.role === 'HOT_SPARE') spares++;
    const div = document.createElement('div');
    div.className = 'tile ' + t.role;
    const tag = t.role === 'ACTIVE' ? ('L' + String(t.logical_id).padStart(2,'0')) : (t.role === 'HOT_SPARE' ? 'SPARE' : 'SWAP!');
    div.innerHTML = 'P' + String(t.physical_id).padStart(2,'0') + '<br>' + tag;
    div.title = 'Physical Bay ' + t.physical_id + ' | ' + t.role + ' | ' + t.temp_C + ' C';
    if (t.role === 'ACTIVE') div.onclick = () => triggerFault(t.physical_id);
    grid.appendChild(div);
  }});
  document.getElementById('k-active').textContent = active;
  document.getElementById('k-spares').textContent = spares;
  document.getElementById('k-zsave').textContent = DATA.telemetry.t_fabric.avg_zero_trit_power_saving_pct + '%';
  document.getElementById('k-banks').textContent = DATA.telemetry.memory.spare_banks_remaining;
  const log = document.getElementById('algedonic-log');
  log.innerHTML = lamps.length ? lamps.map(l => '&gt; ' + l).join('<br>') : '&gt; ALL 27 ACTIVE TILES + 3 HOT SPARES NOMINAL.';
}}

function triggerFault(physId) {{
  const t = tiles.find(x => x.physical_id === physId);
  if (!t || t.role !== 'ACTIVE') return;
  const spare = tiles.find(x => x.role === 'HOT_SPARE');
  const lt = t.logical_id;
  t.role = 'FAULTED_BAY_LIT';
  t.logical_id = null;
  if (!spare) {{
    lamps.unshift('CRITICAL: Bay P' + physId + ' (Logical L' + lt + ') FAULTED — ALL 3 SPARES EXHAUSTED (TRAP 12 TFAULT!)');
  }} else {{
    spare.role = 'ACTIVE';
    spare.logical_id = lt;
    const us = (46 + Math.random() * 14).toFixed(1);
    lamps.unshift('ASHBY CUTOVER [' + us + ' us < 75 us]: Bay P' + physId + ' ABFT fault -> Logical L' + lt + ' remapped to Spare P' + spare.physical_id + ' (0 job aborts)');
  }}
  renderTiles();
}}

function injectRandomFault() {{
  const act = tiles.filter(x => x.role === 'ACTIVE');
  if (act.length) triggerFault(act[Math.floor(Math.random() * act.length)].physical_id);
}}

function resetTiles() {{
  tiles = Array.from({{length: 30}}, (_, i) => ({{
    physical_id: i, logical_id: i < 27 ? i : null, role: i < 27 ? 'ACTIVE' : 'HOT_SPARE', temp_C: 22.0
  }}));
  lamps.unshift('MAINTENANCE BAY: Faulted UE-TECL blades hot-swapped; 3 hot spares restored.');
  renderTiles();
}}

// Render Level -9 cells
const cl = document.getElementById('cell-list');
Object.entries(DATA.telemetry.level_9_cells).forEach(([k, v]) => {{
  cl.innerHTML += '<div class="cell-bar"><span>' + k + '</span><b style="color:var(--amber);font-family:monospace">' + v + ' cells</b></div>';
}});

// Render SAEW loss curve
document.getElementById('s-loss0').textContent = DATA.saew.initial_loss.toFixed(2);
document.getElementById('s-loss1').textContent = DATA.saew.final_loss.toFixed(2);
document.getElementById('s-cut').textContent = DATA.saew.cutovers_survived;
document.getElementById('saew-sample').textContent =
  'TBFP-9 (1-Tile): ' + DATA.saew.sample_tbfp9_output + '\\nTBFP-3 (Triad):  ' + DATA.saew.sample_tbfp3_output;
const lh = DATA.saew.loss_history;
const maxL = Math.max(...lh), minL = Math.min(...lh);
const pts = lh.map((v, i) => {{
  const x = 15 + (i / Math.max(1, lh.length - 1)) * 330;
  const y = 135 - ((v - minL) / Math.max(1e-6, maxL - minL)) * 115;
  return x.toFixed(1) + ',' + y.toFixed(1);
}}).join(' ');
document.getElementById('saew-svg').innerHTML =
  '<polyline fill="none" stroke="#e8962e" stroke-width="2.5" points="' + pts + '" />' +
  '<text x="15" y="16" fill="#a6a094" font-size="10" font-family="monospace">TBFP-9 Cross-Entropy Loss (' + lh.length + ' steps)</text>';

// Render SETI table
const sb = document.getElementById('seti-body');
DATA.seti.results.forEach(r => {{
  const cls = r.verdict_trit > 0 ? 'trit-pos' : (r.verdict_trit < 0 ? 'trit-neg' : 'trit-zero');
  sb.innerHTML += '<tr><td>' + r.label + '</td><td style="font-family:monospace">' + r.peak_to_median_power + 'x</td>' +
    '<td style="font-family:monospace">' + (r.final_decibans >= 0 ? '+' : '') + r.final_decibans + ' dB</td>' +
    '<td class="' + cls + '">' + r.verdict_name + '</td></tr>';
}});

// Render Cliodynamics table
const cb = document.getElementById('clio-body');
DATA.clio.trajectories.forEach(tr => {{
  const cls = tr.stability_trit > 0 ? 'trit-pos' : (tr.stability_trit < 0 ? 'trit-neg' : 'trit-zero');
  cb.innerHTML += '<tr><td>' + tr.policy_name + '</td><td>' + tr.conservation + '</td><td>' + tr.abatement + '</td>' +
    '<td style="font-family:monospace">' + tr.final_population + '</td><td style="font-family:monospace">' + tr.final_resources + '</td>' +
    '<td style="font-family:monospace">' + tr.final_legitimacy + '</td><td class="' + cls + '">' + tr.stability_label + '</td></tr>';
}});

renderTiles();
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
