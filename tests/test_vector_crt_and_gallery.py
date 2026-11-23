"""Tests for OBB-729 Pask–Licklider Vector-CRT Protocol & Live Web Operations Gallery Server."""

from __future__ import annotations

import json
import threading
import urllib.request

from ue1000.gallery_server import GalleryController, create_gallery_server
from ue1000.machine import UE1000Machine
from ue1000.vector_crt import (
    CLIO_POLICY_BOX,
    CLIO_POLICY_PICK_TAG,
    CRT_CLEAR,
    CRT_END,
    INT_AMBER,
    INT_PAIN,
    INT_VIABLE,
    SAEW_TRAIN_PICK_TAG,
    VectorCRTBuilder,
    decode_crt_words,
    light_pen_pick,
)


def test_obb729_crt_protocol_tdram_roundtrip_and_light_pen_pick():
    """Verify OBB-729 (-33..-121) tryte stream packs to T36 words in T-DRAM and decodes cleanly."""
    b = VectorCRTBuilder()
    b.intensity(INT_VIABLE).scale(2).pick_tag(17).rect(-120, -60, -20, 60)
    b.intensity(INT_AMBER).pick_tag(29).circle(140, 80, 35)
    b.intensity(INT_PAIN).pick_tag(42).line(-300, -200, 300, -200)
    b.pick_tag(55).point(0, 250)
    b.pick_tag(0).intensity(INT_AMBER).text(-100, 300, "OBB-729 CRT TEST")

    trytes = b.to_trytes()
    assert trytes[0] == CRT_CLEAR and trytes[-1] == CRT_END
    assert all(-364 <= tr <= 364 for tr in trytes)

    # Store native T36 display-list words directly inside UE-1000 T-DRAM and read back!
    words = b.to_t36_words()
    m = UE1000Machine()
    for i, w in enumerate(words):
        m.mem.write(8000 + i, w, supervisor=True)
    reloaded = [m.mem.read(8000 + i, supervisor=True)[0] for i in range(len(words))]

    frame = decode_crt_words(reloaded, scope="custom", title="Roundtrip")
    assert frame.t36_word_count == len(words)
    assert len(frame.hept_preview) == min(8, len(words))
    assert len(frame.rects) == 1 and frame.rects[0]["pick_tag"] == 17
    assert len(frame.circles) == 1 and frame.circles[0]["pick_tag"] == 29
    assert len(frame.strokes) == 1 and frame.strokes[0]["pick_tag"] == 42
    assert len(frame.points) == 1 and frame.points[0]["pick_tag"] == 55
    assert len(frame.texts) == 1 and frame.texts[0]["text"] == "OBB-729 CRT TEST"

    # Authentic light-pen hit testing across all primitive types
    assert light_pen_pick(frame, -70, 0) == 17      # Inside rect
    assert light_pen_pick(frame, 145, 75) == 29     # Inside circle
    assert light_pen_pick(frame, 0, -198) == 42     # Near stroke
    assert light_pen_pick(frame, 2, 248) == 55      # Near point
    assert light_pen_pick(frame, -340, 340) is None # Empty phosphor


def test_all_four_hardware_crt_scopes_and_light_pen_interactions():
    """Verify VSM Ring, Cliodynamics TF36, SETI Radix-3, and SAEW scopes with Light-Pen picks."""
    ctrl = GalleryController(user="BROOKS")

    # 1. VSM 30-Tile Radial Ring: Light-Pen pick on physical tile 0 circle triggers Ashby cutover!
    vsm_frame = ctrl.render_crt("vsm")
    tile0_circle = next(c for c in vsm_frame.circles if c["pick_tag"] == 1)
    st = ctrl.handle_lightpen("vsm", tile0_circle["cx"], tile0_circle["cy"])
    assert st["pick_tag"] == 1
    assert st["telemetry"]["t_fabric"]["hot_spares_remaining"] == 2
    assert len(st["telemetry"]["v_complex"]["cutovers"]) == 1

    # 2. Cliodynamics TF36 Phase Portrait: Light-Pen pick inside policy box launches live probe!
    ctrl.render_crt("clio")
    before = len(ctrl.clio_trajectories)
    bx0, by0, bx1, by1 = CLIO_POLICY_BOX
    # High conservation & abatement -> viable equilibrium (+1)
    st_clio = ctrl.handle_lightpen("clio", bx0 + 0.9 * (bx1 - bx0), by0 + 0.9 * (by1 - by0))
    assert st_clio["pick_tag"] == CLIO_POLICY_PICK_TAG
    assert len(ctrl.clio_trajectories) == before + 1
    assert ctrl.clio_trajectories[-1]["stability_trit"] == 1

    # 3. Oliver-Good SETI Scope: Light-Pen pick NOISE (104) vs BEACON (101)
    ctrl.render_crt("seti")
    st_noise = ctrl.handle_lightpen("seti", 0, 0, pick_tag=104)
    assert st_noise["crt"]["meta"]["verdict"] == -1
    st_beacon = ctrl.handle_lightpen("seti", 0, 0, pick_tag=101)
    assert st_beacon["crt"]["meta"]["verdict"] == 1

    # 4. SAEW AI Scope: Light-Pen pick TRAIN (111) runs live TBFP-9 training!
    ctrl.reset_tiles()  # Restore 3 spares before SAEW live fault-injection training
    st_saew = ctrl.handle_lightpen("saew", 0, 0, pick_tag=SAEW_TRAIN_PICK_TAG)
    assert st_saew["crt"]["meta"]["trained"] is True
    assert st_saew["crt"]["meta"]["final_loss"] < st_saew["crt"]["meta"]["initial_loss"]


def test_live_http_gallery_server_end_to_end():
    """Spin up ThreadingHTTPServer on an ephemeral port and exercise HTML + REST JSON API."""
    server, ctrl = create_gallery_server(host="127.0.0.1", port=0, user="BROOKS")
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()

    def _get(path: str) -> bytes:
        with urllib.request.urlopen(f"{base_url}{path}", timeout=5) as resp:
            assert resp.status == 200
            return resp.read()

    def _post_json(path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            return json.loads(resp.read().decode("utf-8"))

    try:
        # 1. HTML Gallery Page
        html = _get("/").decode("utf-8")
        assert '<canvas id="crt"' in html
        assert "DUWAMISH-I (UE-1000)" in html

        # 2. Initial State JSON & OBB-729 CRT display list
        st0 = json.loads(_get("/api/state?scope=vsm").decode("utf-8"))
        assert st0["vos"]["user"] == "BROOKS"
        assert st0["crt"]["tryte_count"] > 100
        assert st0["crt"]["t36_word_count"] > 20
        assert len(st0["crt"]["hept_preview"]) == 8

        # 3. Live tile fault via POST /api/fault/tile -> real Ashby <75 us cutover!
        st1 = _post_json("/api/fault/tile", {"logical_tile": 5, "reason": "HTTP_DRILL"})
        assert st1["telemetry"]["t_fabric"]["hot_spares_remaining"] == 2
        assert st1["telemetry"]["v_complex"]["cutovers"][-1]["hw_cutover_us"] < 75.0

        # 4. Live MMIO bank cutover via POST /api/fault/bank
        st2 = _post_json("/api/fault/bank", {"logical_bank": 3})
        assert st2["telemetry"]["memory"]["spare_banks_remaining"] == 8

        # 5. Acknowledge Algedonic alarms
        st3 = _post_json("/api/ack", {})
        assert ctrl.machine.vcomplex.unacked_alarms == 0
        assert st3["telemetry"]["v_complex"]["algedonic_alarms"] >= 1

        # 6. Embedded V-OS Terminal commands over HTTP (CHIO Silo read + APL\36!)
        st_silo = _post_json("/api/vos", {"command": "CAT SILO:BYURAKAN_71.MEMO"})
        assert st_silo["vos_result"]["error"] is None
        assert any("KELVANOV" in ln for ln in st_silo["vos_result"]["lines"])

        st_apl = _post_json("/api/vos", {"command": "APL +/ 1 2 3 4 5"})
        assert st_apl["vos_result"]["error"] is None
        assert any("15" in ln for ln in st_apl["vos_result"]["lines"])

        # 7. Maintenance Bay blade hot-swap restores all 3 hot spares
        st_rst = _post_json("/api/reset/tiles", {})
        assert st_rst["telemetry"]["t_fabric"]["hot_spares_remaining"] == 3
    finally:
        server.shutdown()
        server.server_close()
        th.join(timeout=2)
