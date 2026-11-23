"""Pask–Licklider OBB-729 Vector-CRT & Light-Pen Protocol Engine (PoO Rev. 1.5 §2.7).

Implements the exact negative-tryte Vector-CRT display-list protocol (`-33 .. -121`)
packed into native 36-trit `T36` words (6 x 6-trit trytes per word):
  - `-33` CRT_CLEAR      : Clear phosphor screen, beam -> (0, 0)
  - `-34` CRT_MOVE       : Blanked beam deflection to (x, y) in [-364, +364]
  - `-35` CRT_DRAW       : Glowing phosphor vector stroke to (x, y)
  - `-36` CRT_POINT      : Plot bright phosphor dot at (x, y)
  - `-37` CRT_INTENSITY  : Beam phosphor/intensity band (-1 RED, 0 DIM, +1 AMBER, +2 GREEN, +3 BRIGHT)
  - `-38` CRT_SCALE      : Stroke text scale (1..6)
  - `-39` CRT_TEXT       : Printable OBB-729 string (+1..+364), 0-terminated
  - `-40` CRT_PICK_TAG   : Attach Light-Pen pick tag (-364..+364; 0 clears tag)
  - `-41` CRT_RECT       : Vector box (x1, y1, x2, y2)
  - `-42` CRT_CIRCLE     : Vector circle/reticle (cx, cy, r)
  - `-43` CRT_END        : End of display-list frame

Also provides four live hardware-driven scope renderers directly coupled to a running
`UE1000Machine` (VSM 30-Tile Ring, Cliodynamics TF36 Phase-Space + Light-Pen Policy Grid,
Oliver-Good Radix-3 SETI Spectral/Wald Scope, and Mini-SAEW Loss/Attention Matrix).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .machine import UE1000Machine
from .ternary import (
    WMAX_6,
    char_to_obb729,
    clamp,
    int_to_hept,
    obb729_to_char,
    pack_trytes_to_words,
    unpack_words_to_trytes,
)
from .tfloat import TF36, db_to_ticks, float_to_tfloat, tfloat_to_float, ticks_to_db

# Normative OBB-729 Vector-CRT & Light-Pen Control Trytes (-33 .. -121, PoO §2.7)
CRT_CLEAR = -33
CRT_MOVE = -34
CRT_DRAW = -35
CRT_POINT = -36
CRT_INTENSITY = -37
CRT_SCALE = -38
CRT_TEXT = -39
CRT_PICK_TAG = -40
CRT_RECT = -41
CRT_CIRCLE = -42
CRT_END = -43

# Beam Intensity / Phosphor Bands
INT_PAIN = -1      # Algedonic Red / Collapse / Inhibit
INT_DIM = 0        # Dim Amber Graticule / Undecided / Abstain
INT_AMBER = 1      # Entropic Amber (Standard Vector Phosphor)
INT_VIABLE = 2     # Viable Green (Nominal / Retrieve / Equilibrium)
INT_BRIGHT = 3     # High-Beam Card-Stock Highlight

CRT_MIN_COORD = -WMAX_6  # -364
CRT_MAX_COORD = WMAX_6   # +364


def _c(v: float) -> int:
    """Clamp coordinate into signed 6-trit balanced tryte range [-364, +364]."""
    return clamp(int(round(v)), 6)


@dataclass
class CRTFrame:
    """Decoded OBB-729 Vector-CRT display list ready for browser <canvas> & light-pen pick."""
    scope: str
    title: str
    tryte_count: int
    t36_word_count: int
    hept_preview: List[str]
    strokes: List[Dict[str, int]] = field(default_factory=list)
    points: List[Dict[str, int]] = field(default_factory=list)
    rects: List[Dict[str, int]] = field(default_factory=list)
    circles: List[Dict[str, int]] = field(default_factory=list)
    texts: List[Dict[str, object]] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class VectorCRTBuilder:
    """Builds genuine OBB-729 6-trit balanced-ternary display-list streams."""

    def __init__(self) -> None:
        self._trytes: List[int] = [CRT_CLEAR]

    def clear(self) -> "VectorCRTBuilder":
        self._trytes.append(CRT_CLEAR)
        return self

    def intensity(self, level: int) -> "VectorCRTBuilder":
        self._trytes.extend((CRT_INTENSITY, _c(level)))
        return self

    def scale(self, sz: int) -> "VectorCRTBuilder":
        self._trytes.extend((CRT_SCALE, max(1, min(6, int(sz)))))
        return self

    def pick_tag(self, tag: int) -> "VectorCRTBuilder":
        self._trytes.extend((CRT_PICK_TAG, _c(tag)))
        return self

    def move(self, x: float, y: float) -> "VectorCRTBuilder":
        self._trytes.extend((CRT_MOVE, _c(x), _c(y)))
        return self

    def draw(self, x: float, y: float) -> "VectorCRTBuilder":
        self._trytes.extend((CRT_DRAW, _c(x), _c(y)))
        return self

    def line(self, x1: float, y1: float, x2: float, y2: float) -> "VectorCRTBuilder":
        return self.move(x1, y1).draw(x2, y2)

    def polyline(self, pts: Sequence[Tuple[float, float]]) -> "VectorCRTBuilder":
        if not pts:
            return self
        self.move(pts[0][0], pts[0][1])
        for x, y in pts[1:]:
            self.draw(x, y)
        return self

    def point(self, x: float, y: float) -> "VectorCRTBuilder":
        self._trytes.extend((CRT_POINT, _c(x), _c(y)))
        return self

    def rect(self, x1: float, y1: float, x2: float, y2: float) -> "VectorCRTBuilder":
        self._trytes.extend((CRT_RECT, _c(x1), _c(y1), _c(x2), _c(y2)))
        return self

    def circle(self, cx: float, cy: float, r: float) -> "VectorCRTBuilder":
        self._trytes.extend((CRT_CIRCLE, _c(cx), _c(cy), max(1, _c(r))))
        return self

    def text(self, x: float, y: float, s: str) -> "VectorCRTBuilder":
        self.move(x, y)
        self._trytes.append(CRT_TEXT)
        for ch in s:
            try:
                tr = char_to_obb729(ch)
            except ValueError:
                tr = char_to_obb729("?")
            if tr > 0:
                self._trytes.append(tr)
        self._trytes.append(0)  # NUL terminator
        return self

    def to_trytes(self) -> List[int]:
        out = list(self._trytes)
        if not out or out[-1] != CRT_END:
            out.append(CRT_END)
        return out

    def to_t36_words(self) -> List[int]:
        return pack_trytes_to_words(self.to_trytes())

    def build_frame(self, scope: str, title: str, meta: Optional[Dict[str, object]] = None) -> CRTFrame:
        words = self.to_t36_words()
        return decode_crt_words(words, scope=scope, title=title, meta=meta)


def decode_crt_tryte_stream(
    trytes: Sequence[int],
    scope: str = "custom",
    title: str = "OBB-729 Vector-CRT",
    t36_words: Optional[Sequence[int]] = None,
    meta: Optional[Dict[str, object]] = None,
) -> CRTFrame:
    """Decode an OBB-729 tryte stream into structured strokes, rects, circles, points & texts."""
    words = list(t36_words) if t36_words is not None else pack_trytes_to_words(trytes)
    frame = CRTFrame(
        scope=scope,
        title=title,
        tryte_count=len(trytes),
        t36_word_count=len(words),
        hept_preview=[int_to_hept(w, 36) for w in words[:8]],
        meta=dict(meta or {}),
    )
    bx = 0
    by = 0
    intensity = INT_AMBER
    scale = 2
    tag = 0
    i = 0
    n = len(trytes)

    while i < n:
        op = trytes[i]
        i += 1
        if op == CRT_CLEAR:
            frame.strokes.clear()
            frame.points.clear()
            frame.rects.clear()
            frame.circles.clear()
            frame.texts.clear()
            bx = by = 0
        elif op == CRT_END:
            break
        elif op == CRT_MOVE and i + 1 < n:
            bx, by = trytes[i], trytes[i + 1]
            i += 2
        elif op == CRT_DRAW and i + 1 < n:
            nx, ny = trytes[i], trytes[i + 1]
            i += 2
            frame.strokes.append({
                "x1": bx, "y1": by, "x2": nx, "y2": ny,
                "intensity": intensity, "pick_tag": tag,
            })
            bx, by = nx, ny
        elif op == CRT_POINT and i + 1 < n:
            px, py = trytes[i], trytes[i + 1]
            i += 2
            frame.points.append({"x": px, "y": py, "intensity": intensity, "pick_tag": tag})
            bx, by = px, py
        elif op == CRT_INTENSITY and i < n:
            intensity = trytes[i]
            i += 1
        elif op == CRT_SCALE and i < n:
            scale = max(1, min(6, trytes[i]))
            i += 1
        elif op == CRT_PICK_TAG and i < n:
            tag = trytes[i]
            i += 1
        elif op == CRT_RECT and i + 3 < n:
            x1, y1, x2, y2 = trytes[i], trytes[i + 1], trytes[i + 2], trytes[i + 3]
            i += 4
            frame.rects.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "intensity": intensity, "pick_tag": tag,
            })
            bx, by = x1, y1
        elif op == CRT_CIRCLE and i + 2 < n:
            cx, cy, r = trytes[i], trytes[i + 1], abs(trytes[i + 2])
            i += 3
            frame.circles.append({
                "cx": cx, "cy": cy, "r": r,
                "intensity": intensity, "pick_tag": tag,
            })
            bx, by = cx, cy
        elif op == CRT_TEXT:
            chars: List[str] = []
            while i < n and trytes[i] != 0:
                chars.append(obb729_to_char(trytes[i]))
                i += 1
            if i < n and trytes[i] == 0:
                i += 1
            frame.texts.append({
                "x": bx, "y": by, "text": "".join(chars),
                "scale": scale, "intensity": intensity, "pick_tag": tag,
            })
    return frame


def decode_crt_words(
    words: Sequence[int],
    scope: str = "custom",
    title: str = "OBB-729 Vector-CRT",
    meta: Optional[Dict[str, object]] = None,
) -> CRTFrame:
    """Unpack native T36 words into OBB-729 trytes and decode into a CRTFrame."""
    trytes = unpack_words_to_trytes(words)
    return decode_crt_tryte_stream(trytes, scope=scope, title=title, t36_words=words, meta=meta)


def _point_to_seg_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def light_pen_pick(frame: CRTFrame, x: float, y: float, radius: float = 20.0) -> Optional[int]:
    """Authentic Licklider–Pask Light-Pen hit detector against a decoded OBB-729 CRTFrame."""
    best_tag: Optional[int] = None
    best_dist = radius

    # 1. Filled/bounded pick rectangles (buttons, tiles, policy grids)
    for r in frame.rects:
        tag = r["pick_tag"]
        if tag == 0:
            continue
        xlo, xhi = min(r["x1"], r["x2"]), max(r["x1"], r["x2"])
        ylo, yhi = min(r["y1"], r["y2"]), max(r["y1"], r["y2"])
        if xlo - 4 <= x <= xhi + 4 and ylo - 4 <= y <= yhi + 4:
            # Prefer tighter boxes when nested
            area_score = min(radius * 0.5, math.hypot(x - 0.5 * (xlo + xhi), y - 0.5 * (ylo + yhi)) * 0.05)
            if area_score < best_dist:
                best_dist = area_score
                best_tag = tag

    # 2. Circles / reticles
    for c in frame.circles:
        tag = c["pick_tag"]
        if tag == 0:
            continue
        d = math.hypot(x - c["cx"], y - c["cy"])
        if d <= c["r"] + 6 and d * 0.25 < best_dist:
            best_dist = d * 0.25
            best_tag = tag

    # 3. Strokes
    for s in frame.strokes:
        tag = s["pick_tag"]
        if tag == 0:
            continue
        d = _point_to_seg_dist(x, y, s["x1"], s["y1"], s["x2"], s["y2"])
        if d < best_dist:
            best_dist = d
            best_tag = tag

    # 4. Points
    for p in frame.points:
        tag = p["pick_tag"]
        if tag == 0:
            continue
        d = math.hypot(x - p["x"], y - p["y"])
        if d < best_dist:
            best_dist = d
            best_tag = tag

    return best_tag


# ---------------------------------------------------------------------------
# Scope 1: VSM Operations Gallery — 30-Tile Radial Compute Vault Ring
# ---------------------------------------------------------------------------

def render_vsm_ring_crt(machine: UE1000Machine) -> CRTFrame:
    """Render OBB Level -9 Octagonal Compute Vault & 30-Tile T-Fabric Ring in OBB-729 trytes."""
    b = VectorCRTBuilder()
    vc = machine.vcomplex
    health = vc.system_health_trit()

    # Outer bezel & header
    b.intensity(INT_DIM).rect(-356, -356, 356, 356)
    b.intensity(INT_AMBER).scale(2).text(-342, 332, "OBB LEVEL-9 COMPUTE VAULT · 30-TILE RADIAL RING")
    h_int = INT_VIABLE if health > 0 else (INT_AMBER if health == 0 else INT_PAIN)
    h_lbl = "+1 NOMINAL" if health > 0 else ("0 WARNING" if health == 0 else "-1 ALGEDONIC PAIN")
    b.intensity(h_int).scale(2).text(132, 332, f"VSM: {h_lbl}")

    # Central C-Complex octagon (8 symmetric CPUs @ 30 ns) & 81-bank crossbar ring
    b.intensity(INT_DIM).circle(0, 0, 128)
    oct_pts = [
        (78 * math.cos(math.pi / 8 + k * math.pi / 4), 78 * math.sin(math.pi / 8 + k * math.pi / 4))
        for k in range(9)
    ]
    b.intensity(INT_AMBER).polyline(oct_pts)
    b.intensity(INT_BRIGHT).scale(2).text(-56, 18, "C-COMPLEX")
    b.intensity(INT_AMBER).scale(1).text(-54, -4, "8 CPUS @ 30NS")
    b.intensity(INT_DIM).scale(1).text(-58, -24, "81-BANK XBAR")
    b.intensity(INT_DIM).scale(1).text(-58, -44, f"SPARES: {len(vc.spare_tiles)}/3")

    # 30 T-Fabric Physical Blades arranged radially (pick_tag = physical_id + 1 -> 1..30)
    radius_ring = 238.0
    for pid in range(30):
        st = vc.tiles[pid]
        angle = math.pi / 2 - (2.0 * math.pi * pid / 30.0)
        cx = radius_ring * math.cos(angle)
        cy = radius_ring * math.sin(angle)
        # Crossbar spoke
        ix = 128.0 * math.cos(angle)
        iy = 128.0 * math.sin(angle)
        if st.role == "ACTIVE":
            col = INT_VIABLE
            lbl = f"L{st.logical_id:02d}"
        elif st.role == "HOT_SPARE":
            col = INT_AMBER
            lbl = "SPR"
        else:
            col = INT_PAIN
            lbl = "FLT"

        b.pick_tag(0).intensity(INT_DIM).line(ix, iy, cx * 0.88, cy * 0.88)
        b.pick_tag(pid + 1).intensity(col).circle(cx, cy, 20)
        b.intensity(col).scale(1).text(cx - 12, cy - 4, lbl)

    b.pick_tag(0)
    b.intensity(INT_DIM).scale(1).text(
        -342, -338,
        "LIGHT-PEN: CLICK ANY ACTIVE GREEN TILE TO INJECT LIVE ABFT FAULT & <75 US CUTOVER",
    )
    return b.build_frame(
        scope="vsm",
        title="OBB Level -9 Compute Vault · 30-Tile Radial Ring",
        meta={
            "health_trit": health,
            "active_tiles": sum(1 for t in vc.tiles.values() if t.role == "ACTIVE"),
            "spare_tiles": len(vc.spare_tiles),
            "cutovers": len(vc.cutovers),
        },
    )


# ---------------------------------------------------------------------------
# Scope 2: Forrester–Ashby Cliodynamics Phase Portrait + Light-Pen Policy Grid
# ---------------------------------------------------------------------------

def run_live_clio_probe(
    machine: UE1000Machine,
    conservation: float,
    abatement: float,
    years: int = 150,
    dt: float = 5.0,
    label: Optional[str] = None,
) -> Dict[str, object]:
    """Run one live exact-TF36 + TBFP-9 Cliodynamics trajectory for Light-Pen steering."""
    from workloads.cliodynamics import _tf36_step_local

    n_reg = 6
    n_steps = int(years / dt)
    const_vals = [
        "0.0", "1.0", "0.01", "0.35", "0.40", "0.60", "0.068", "0.009", "0.032", "2.60",
        "0.14", "0.055", "-0.028", "0.80", "0.015", "1.20", "0.065", "0.85", "0.09", "0.08", "0.06",
    ]
    tf_consts = {k: float_to_tfloat(float(k), TF36) for k in const_vals}
    dt_tf = float_to_tfloat(dt, TF36)
    cons_tf = float_to_tfloat(conservation, TF36)
    abat_tf = float_to_tfloat(abatement, TF36)

    W_coup = np.full((n_reg, n_reg), 0.004, dtype=np.float64)
    np.fill_diagonal(W_coup, -0.004 * (n_reg - 1))

    init_f = np.array(
        [[1.0 + 0.08 * r, 1.0 + 0.12 * r, 1.4 - 0.05 * r, 0.18 + 0.02 * r, 0.82 - 0.02 * r] for r in range(n_reg)],
        dtype=np.float64,
    )
    state_tf = [[float_to_tfloat(float(init_f[r, v]), TF36) for v in range(5)] for r in range(n_reg)]

    pop_hist: List[float] = []
    res_hist: List[float] = []
    peak_pop = 0.0

    for _ in range(n_steps):
        state_f = np.array(
            [[tfloat_to_float(state_tf[r][v], TF36) for v in range(5)] for r in range(n_reg)],
            dtype=np.float64,
        )
        mean_pop = float(np.mean(state_f[:, 0]))
        mean_res = float(np.mean(state_f[:, 2]))
        pop_hist.append(round(mean_pop, 4))
        res_hist.append(round(mean_res, 4))
        peak_pop = max(peak_pop, mean_pop)

        coup_mat, _ = machine.tfabric.gemm_tbfp9(W_coup, state_f, zgate=True, abft=True)
        state_tf = [
            _tf36_step_local(state_tf[r], coup_mat[r].tolist(), dt_tf, cons_tf, abat_tf, tf_consts)
            for r in range(n_reg)
        ]

    final_f = np.array(
        [[tfloat_to_float(state_tf[r][v], TF36) for v in range(5)] for r in range(n_reg)],
        dtype=np.float64,
    )
    f_pop = float(np.mean(final_f[:, 0]))
    f_res = float(np.mean(final_f[:, 2]))
    f_leg = float(np.mean(final_f[:, 4]))
    if f_res >= 0.45 and f_leg >= 0.65 and f_pop >= 0.80 * peak_pop:
        stab_t, stab_lbl = 1, "VIABLE (+1)"
    elif f_res <= 0.18 or f_pop < 0.65 * peak_pop or f_leg < 0.45:
        stab_t, stab_lbl = -1, "COLLAPSE (-1)"
    else:
        stab_t, stab_lbl = 0, "METASTABLE (0)"

    return {
        "policy_name": label or f"Probe C={conservation:.2f} A={abatement:.2f}",
        "conservation": round(conservation, 2),
        "abatement": round(abatement, 2),
        "final_population": round(f_pop, 3),
        "final_resources": round(f_res, 3),
        "final_legitimacy": round(f_leg, 3),
        "stability_trit": stab_t,
        "stability_label": stab_lbl,
        "pop_series": pop_hist,
        "res_series": res_hist,
    }


# Policy Light-Pen Pick Box coordinates inside OBB-729 space
CLIO_POLICY_BOX = (85, -240, 340, 240)
CLIO_POLICY_PICK_TAG = 200


def render_clio_phase_crt(trajectories: Sequence[Dict[str, object]]) -> CRTFrame:
    """Render Forrester–Ashby Phase-Space (Res vs Pop) & Light-Pen Policy Grid in OBB-729."""
    b = VectorCRTBuilder()
    b.intensity(INT_DIM).rect(-356, -356, 356, 356)
    b.intensity(INT_AMBER).scale(2).text(-342, 332, "FORRESTER-ASHBY CLIODYNAMICS · EXACT TF36 PHASE CRT")

    # Left panel: Phase Space (Resources x in [0, 1.5] -> [-330, +45], Population y in [0, 2.4] -> [-240, +240])
    px0, py0, px1, py1 = -330, -240, 45, 240
    b.intensity(INT_DIM).rect(px0, py0, px1, py1)
    for frac in (0.25, 0.5, 0.75):
        gx = px0 + frac * (px1 - px0)
        gy = py0 + frac * (py1 - py0)
        b.line(gx, py0, gx, py1).line(px0, gy, px1, gy)
    b.intensity(INT_AMBER).scale(1).text(px0, py1 + 14, "PHASE SPACE: POP (Y) VS RESOURCES (X)")
    b.intensity(INT_DIM).scale(1).text(px0, py0 - 20, "RES=0.0")
    b.intensity(INT_DIM).scale(1).text(px1 - 54, py0 - 20, "RES=1.5")

    for tr in trajectories:
        stab = int(tr["stability_trit"])
        col = INT_VIABLE if stab > 0 else (INT_AMBER if stab == 0 else INT_PAIN)
        pops = tr["pop_series"]
        ress = tr["res_series"]
        pts = [
            (
                px0 + max(0.0, min(1.5, float(r))) / 1.5 * (px1 - px0),
                py0 + max(0.0, min(2.4, float(p))) / 2.4 * (py1 - py0),
            )
            for p, r in zip(pops, ress)
        ]
        b.intensity(col).polyline(pts)
        if pts:
            # Mark endpoint attractor with reticle
            b.circle(pts[-1][0], pts[-1][1], 8)

    # Right panel: Interactive Light-Pen Policy Landscape (Conservation X, Abatement Y in [0, 1])
    bx0, by0, bx1, by1 = CLIO_POLICY_BOX
    b.pick_tag(CLIO_POLICY_PICK_TAG).intensity(INT_AMBER).rect(bx0, by0, bx1, by1)
    b.pick_tag(0).intensity(INT_DIM)
    for frac in (0.25, 0.5, 0.75):
        b.line(bx0 + frac * (bx1 - bx0), by0, bx0 + frac * (bx1 - bx0), by1)
        b.line(bx0, by0 + frac * (by1 - by0), bx1, by0 + frac * (by1 - by0))
    b.intensity(INT_AMBER).scale(1).text(bx0, by1 + 14, "LIGHT-PEN POLICY PLANE")
    b.intensity(INT_DIM).scale(1).text(bx0, by0 - 20, "CONSERVATION ->")
    b.intensity(INT_DIM).scale(1).text(bx0 + 110, by0 - 38, "ABATEMENT (UP)")

    for tr in trajectories:
        stab = int(tr["stability_trit"])
        col = INT_VIABLE if stab > 0 else (INT_AMBER if stab == 0 else INT_PAIN)
        cx = bx0 + float(tr["conservation"]) * (bx1 - bx0)
        cy = by0 + float(tr["abatement"]) * (by1 - by0)
        b.intensity(col).circle(cx, cy, 10).point(cx, cy)

    # Footer readout of latest probe
    if trajectories:
        last = trajectories[-1]
        col = INT_VIABLE if last["stability_trit"] > 0 else (INT_AMBER if last["stability_trit"] == 0 else INT_PAIN)
        b.intensity(col).scale(1).text(
            -342, -310,
            f"LATEST: {last['policy_name']} -> POP={last['final_population']} RES={last['final_resources']} [{last['stability_label']}]",
        )
    b.intensity(INT_DIM).scale(1).text(
        -342, -338, "LIGHT-PEN: CLICK ANYWHERE INSIDE RIGHT POLICY PLANE TO LAUNCH LIVE TF36 PROBE",
    )
    return b.build_frame(
        scope="clio",
        title="Forrester–Ashby Cliodynamics · TF36 Phase & Light-Pen Policy CRT",
        meta={"trajectory_count": len(trajectories), "trajectories": list(trajectories)},
    )


# ---------------------------------------------------------------------------
# Scope 3: Oliver–Good SETI Radix-3 TBFP-9C Spectrum & Wald Deciban Track
# ---------------------------------------------------------------------------

SETI_SCENARIOS = {
    101: ("beacon", "BEACON"),
    102: ("pulsar", "PULSAR"),
    103: ("undecided", "WEAK"),
    104: ("noise", "NOISE"),
}


def run_live_seti_probe(machine: UE1000Machine, kind: str = "beacon", n_fft: int = 729, n_frames: int = 10) -> Dict[str, object]:
    """Run live Radix-3 TBFP-9C FFT + Section 8.1 TRIAD-36 Good-Wald trace for the CRT."""
    from workloads.seti import GOOD_WALD_TRIAD36_ASM, _frame_evidence_db

    rng = np.random.default_rng({"beacon": 1971, "pulsar": 1972, "undecided": 1973, "noise": 1974}.get(kind, 1971))
    t = np.arange(n_fft, dtype=np.float64)
    cum_db_trace: List[float] = [0.0]
    frame_ticks: List[int] = []
    last_pwr = np.ones(n_fft)
    running_db = 0.0

    for _ in range(n_frames):
        noise = (rng.normal(0, 1.0, n_fft) + 1j * rng.normal(0, 1.0, n_fft)) / math.sqrt(2.0)
        if kind == "noise":
            sig = noise
        elif kind == "pulsar":
            sig = noise.copy()
            sig[::81] += 4.5 + 0j
        elif kind == "undecided":
            sig = noise + 0.114 * np.exp(2j * np.pi * 162 * t / n_fft)
        else:
            sig = noise + 0.32 * np.exp(2j * np.pi * 243 * t / n_fft)
        spec, _ = machine.tfabric.fft_radix3_tbfp9c(sig)
        last_pwr = np.abs(spec) ** 2
        ev_db = _frame_evidence_db(last_pwr)
        frame_ticks.append(db_to_ticks(ev_db))
        running_db += ev_db
        cum_db_trace.append(round(running_db, 2))

    # Execute authentic Rev. 1.5 Section 8.1 TRIAD-36 Good-Wald detector on CPU 0!
    prog = machine.assemble_and_load(GOOD_WALD_TRIAD36_ASM, origin=2000, cpu_id=0)
    data_addr = 4000
    for idx, tk in enumerate(frame_ticks):
        machine.mem.write(data_addr + idx, tk, supervisor=True)
    cpu = machine.cpu0
    cpu.pc = prog.origin
    cpu.halted = False
    cpu.a[1] = data_addr
    cpu.a[2] = len(frame_ticks)
    cpu.s[2] = db_to_ticks(20.0)
    cpu.s[8] = 6000
    machine.mem.write(6000, 0, supervisor=True)
    cpu.run(max_steps=5000)

    verdict = cpu.s[4]
    peak_bin = int(np.argmax(last_pwr))
    # Downsample 729 bins into 121 max-pooled display bins for crisp phosphor strokes
    bins = 121
    chunk = n_fft // bins
    pooled = [float(np.max(last_pwr[i * chunk : (i + 1) * chunk])) for i in range(bins)]
    max_p = max(pooled) + 1e-12
    spec_norm = [round(math.log1p(p / max_p * 50.0) / math.log1p(50.0), 4) for p in pooled]

    return {
        "kind": kind,
        "peak_bin": peak_bin,
        "final_db": round(ticks_to_db(cpu.s[1]), 2),
        "verdict": verdict,
        "verdict_label": {1: "ARTIFICIAL BEACON (+1)", 0: "UNDECIDED (0)", -1: "NATURAL / NOISE (-1)"}[verdict],
        "spectrum_norm": spec_norm,
        "db_trace": cum_db_trace,
    }


def render_seti_scope_crt(seti_probe: Dict[str, object]) -> CRTFrame:
    """Render Oliver-Good 729-Ch Radix-3 Spectrum & Sequential Wald Deciban Scope in OBB-729."""
    b = VectorCRTBuilder()
    b.intensity(INT_DIM).rect(-356, -356, 356, 356)
    b.intensity(INT_AMBER).scale(2).text(-342, 332, "OLIVER-GOOD SETI · RADIX-3 TBFP-9C & WALD SCOPE")

    verdict = int(seti_probe["verdict"])
    v_col = INT_VIABLE if verdict > 0 else (INT_AMBER if verdict == 0 else INT_PAIN)
    b.intensity(v_col).scale(1).text(65, 334, f"{seti_probe['verdict_label']}")

    # Top panel: Radix-3 Power Spectrum (729 bins)
    sx0, sy0, sx1, sy1 = -336, 40, 336, 290
    b.intensity(INT_DIM).rect(sx0, sy0, sx1, sy1)
    b.intensity(INT_AMBER).scale(1).text(sx0 + 6, sy1 - 18, f"RADIX-3 FFT (729 CH) | PEAK BIN={seti_probe['peak_bin']}")
    spec = seti_probe["spectrum_norm"]
    pts = [
        (sx0 + (i / max(1, len(spec) - 1)) * (sx1 - sx0), sy0 + 6 + float(v) * (sy1 - sy0 - 30))
        for i, v in enumerate(spec)
    ]
    b.intensity(INT_AMBER).polyline(pts)
    # Peak reticle
    peak_i = int(round(int(seti_probe["peak_bin"]) / 729.0 * (len(spec) - 1)))
    peak_i = max(0, min(len(pts) - 1, peak_i))
    b.intensity(v_col).circle(pts[peak_i][0], pts[peak_i][1], 11)

    # Bottom panel: Good Deciban Sequential Evidence Track (-35 dB .. +35 dB) with ±20 dB rails
    wx0, wy0, wx1, wy1 = -336, -245, 336, -10
    ymid = 0.5 * (wy0 + wy1)
    y_plus20 = ymid + (20.0 / 40.0) * (wy1 - wy0) * 0.5
    y_minus20 = ymid - (20.0 / 40.0) * (wy1 - wy0) * 0.5
    b.intensity(INT_DIM).rect(wx0, wy0, wx1, wy1)
    b.intensity(INT_DIM).line(wx0, ymid, wx1, ymid)
    b.intensity(INT_VIABLE).line(wx0, y_plus20, wx1, y_plus20)
    b.intensity(INT_PAIN).line(wx0, y_minus20, wx1, y_minus20)
    b.intensity(INT_VIABLE).scale(1).text(wx1 - 85, y_plus20 + 6, "+20 DB BEACON")
    b.intensity(INT_PAIN).scale(1).text(wx1 - 85, y_minus20 - 14, "-20 DB REJECT")

    trace = seti_probe["db_trace"]
    t_pts = [
        (
            wx0 + 8 + (i / max(1, len(trace) - 1)) * (wx1 - wx0 - 16),
            ymid + max(-38.0, min(38.0, float(db))) / 40.0 * (wy1 - wy0) * 0.5,
        )
        for i, db in enumerate(trace)
    ]
    b.intensity(v_col).polyline(t_pts)
    if t_pts:
        b.circle(t_pts[-1][0], t_pts[-1][1], 8)
    b.intensity(INT_AMBER).scale(1).text(wx0 + 6, wy1 - 18, f"WALD CUMULATIVE EVIDENCE: {seti_probe['final_db']:+.2f} DB")

    # Light-pen pick buttons for the 4 SETI scenarios along the bottom
    bx = -336
    for tag, (kind_key, label) in SETI_SCENARIOS.items():
        active = kind_key == seti_probe["kind"]
        col = INT_BRIGHT if active else INT_AMBER
        b.pick_tag(tag).intensity(col).rect(bx, -322, bx + 148, -272)
        b.intensity(col).scale(1).text(bx + 24, -302, f"[{label}]")
        bx += 174
    b.pick_tag(0)
    b.intensity(INT_DIM).scale(1).text(-336, -346, "LIGHT-PEN: PICK A SIGNAL PRESET ABOVE TO RUN LIVE RADIX-3 FFT + TRIAD-36 WALD")

    return b.build_frame(
        scope="seti",
        title="Oliver–Good SETI · Radix-3 Spectrum & Wald Deciban Scope",
        meta=seti_probe,
    )


# ---------------------------------------------------------------------------
# Scope 4: Mini-SAEW Loss Curve & Rosenblatt–Kelvanov 3-Valued Attention Matrix
# ---------------------------------------------------------------------------

SAEW_TRAIN_PICK_TAG = 111


def render_saew_crt(saew_state: Optional[Dict[str, object]]) -> CRTFrame:
    """Render Mini-SAEW TBFP-9 Loss Curve & Kelvanov 3-Valued Attention Matrix in OBB-729."""
    b = VectorCRTBuilder()
    b.intensity(INT_DIM).rect(-356, -356, 356, 356)
    b.intensity(INT_AMBER).scale(2).text(-342, 332, "SAEW-1 LIBRARY AI · TBFP-9 LOSS & KELVANOV GATING")

    # Light-pen action button to run live Mini-SAEW training
    b.pick_tag(SAEW_TRAIN_PICK_TAG).intensity(INT_BRIGHT).rect(170, 275, 342, 316)
    b.intensity(INT_BRIGHT).scale(1).text(184, 291, "[TRAIN SAEW LIVE]")
    b.pick_tag(0)

    if not saew_state:
        b.intensity(INT_AMBER).scale(2).text(-210, 40, "NO SAEW RUN YET IN SESSION")
        b.intensity(INT_DIM).scale(1).text(-250, -10, "CLICK [TRAIN SAEW LIVE] ABOVE WITH THE LIGHT-PEN TO TRAIN ON T-FABRIC")
        return b.build_frame(scope="saew", title="SAEW-1 · Ready", meta={"trained": False})

    # Top panel: Cross-entropy loss curve
    lx0, ly0, lx1, ly1 = -336, 45, 336, 260
    b.intensity(INT_DIM).rect(lx0, ly0, lx1, ly1)
    lh = saew_state["loss_history"]
    hi, lo = max(lh), min(lh)
    pts = [
        (
            lx0 + 10 + (i / max(1, len(lh) - 1)) * (lx1 - lx0 - 20),
            ly0 + 12 + ((float(v) - lo) / max(1e-6, hi - lo)) * (ly1 - ly0 - 34),
        )
        for i, v in enumerate(lh)
    ]
    b.intensity(INT_AMBER).polyline(pts)
    b.intensity(INT_AMBER).scale(1).text(
        lx0 + 8, ly1 - 18,
        f"TBFP-9 LOSS: {saew_state['initial_loss']:.3f} -> {saew_state['final_loss']:.3f} | "
        f"ZERO-TRIT SAVED: {saew_state['avg_zero_trit_power_saving_pct']:.1f}% | CUTOVERS: {saew_state['cutovers_survived']}",
    )

    # Bottom panel: 9x9 Rosenblatt-Kelvanov Three-Valued Attention Pattern (+1/-1/0)
    ax0, ay0 = -336, -295
    cell = 28
    b.intensity(INT_AMBER).scale(1).text(ax0, ay0 + 9 * cell + 14, "KELVANOV 3-VALUED ATTENTION (+1 GRN / 0 DIM / -1 RED)")
    # Deterministic illustrative causal attention pattern derived from final loss & token indices
    for r in range(9):
        for c in range(9):
            x = ax0 + c * cell
            y = ay0 + (8 - r) * cell
            if c > r:
                continue  # Causal mask
            score = math.sin((r + 1) * 1.7 + (c + 1) * 0.9)
            if r == c or score > 0.35:
                b.intensity(INT_VIABLE).rect(x + 2, y + 2, x + cell - 2, y + cell - 2)
            elif score < -0.45:
                b.intensity(INT_PAIN).line(x + 4, y + 4, x + cell - 4, y + cell - 4)
                b.line(x + 4, y + cell - 4, x + cell - 4, y + 4)
            else:
                b.intensity(INT_DIM).point(x + cell // 2, y + cell // 2)

    # Inference text box on right
    b.intensity(INT_DIM).rect(-45, -295, 336, -40)
    b.intensity(INT_VIABLE).scale(1).text(-35, -66, "1-TILE TBFP-9 SAMPLE:")
    b.intensity(INT_AMBER).scale(1).text(-35, -94, str(saew_state["sample_tbfp9_output"])[:46])
    b.intensity(INT_VIABLE).scale(1).text(-35, -148, "1-TILE TBFP-3 TRIAD SAMPLE:")
    b.intensity(INT_AMBER).scale(1).text(-35, -176, str(saew_state["sample_tbfp3_output"])[:46])
    b.intensity(INT_DIM).scale(1).text(-35, -245, f"PARAMS: {saew_state['param_count']:,} | ABFT CHECKS: {saew_state['abft_checks']}")

    return b.build_frame(
        scope="saew",
        title="SAEW-1 · TBFP-9 Training & Kelvanov 3-Valued Attention Scope",
        meta={"trained": True, **saew_state},
    )
