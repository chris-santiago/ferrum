"""Packed-instance alignment under the figure-title band (round-3 T3).

The composite render entry shifts a composite's scene-graph mark nodes and
each panel's ``plot_area`` / ``clip`` / axes down by the figure-title header
band (``header_h``).  High-cardinality (>1000-mark) circle / rect batches do NOT
travel as scene-graph nodes — they are packed into a binary GPU-instance
sidecar (``CircleInstance`` / ``RectInstance``), bypassing the scene JSON.

Before the fix, that sidecar was never offset: GPU marks rendered ``header_h``
px above the shifted-down axes and clipped into the title band.  These tests
parse the packed bytes directly and assert the Y field of every instance is
shifted by exactly ``header_h`` (the same shift applied to the scene nodes),
while X / radius / size / style fields are untouched.

Binary layout (mirrors ``ferrum-core`` ``pack_instances.rs`` and the
``CircleInstance`` / ``RectInstance`` structs in ``ferrum-wasm`` ``scene_load.rs``):

    [header 20B: panel_idx, batch_idx, kind, count, flags (all u32 LE)]
    [instance_data][data_indices?][tooltips?]

  * kind 0 = CircleInstance: 16 f32 (64-byte stride); cy is f32[1] (byte 4).
  * kind 1 = RectInstance:   18 f32 (72-byte stride);  y is f32[1] (byte 4).
"""

from __future__ import annotations

import struct

import polars as pl
import pytest

import ferrum as fm

_INSTANCE_SIZES = {0: 64, 1: 72}  # kind -> sizeof(Instance) in bytes
_Y_FIELD_OFFSET = 4  # byte offset of the Y f32 within each instance


def _iter_packed_batches(packed: bytes):
    """Yield ``(panel_idx, kind, count, instance_start)`` for each packed batch.

    ``instance_start`` is the byte index where the batch's instance data begins
    (just past the 20-byte header).

    Tooltip scan mirrors ``scene_load.rs::unpack_binary_instances`` (lines ~708-735):
        [num_fields: u32]
        num_fields   × [len: u32][name bytes]
        count×num_fields × [len: u32][value bytes]
    No overall length prefix.
    """
    pos = 0
    while pos + 20 <= len(packed):
        panel_idx, batch_idx, kind, count, flags = struct.unpack_from("<5I", packed, pos)
        if kind not in _INSTANCE_SIZES:
            break
        instance_start = pos + 20
        batch_end = instance_start + count * _INSTANCE_SIZES[kind]
        if flags & 0x2:
            batch_end += count * 4
        if flags & 0x1:
            # Scan the string table field-by-field (no overall length prefix).
            if batch_end + 4 > len(packed):
                break
            num_fields = struct.unpack_from("<I", packed, batch_end)[0]
            batch_end += 4
            for _ in range(num_fields):
                if batch_end + 4 > len(packed):
                    break
                slen = struct.unpack_from("<I", packed, batch_end)[0]
                batch_end += 4 + slen
                if batch_end > len(packed):
                    break
            for _ in range(count * num_fields):
                if batch_end + 4 > len(packed):
                    break
                slen = struct.unpack_from("<I", packed, batch_end)[0]
                batch_end += 4 + slen
                if batch_end > len(packed):
                    break
        if batch_end > len(packed):
            break
        yield panel_idx, kind, count, instance_start
        pos = batch_end


def _packed_ys(packed: bytes) -> list[float]:
    """Return the Y coordinate of every packed instance, in batch order."""
    ys: list[float] = []
    for _panel_idx, kind, count, instance_start in _iter_packed_batches(packed):
        stride = _INSTANCE_SIZES[kind]
        for i in range(count):
            y_pos = instance_start + i * stride + _Y_FIELD_OFFSET
            (y,) = struct.unpack_from("<f", packed, y_pos)
            ys.append(y)
    return ys


def _packed_xs(packed: bytes) -> list[float]:
    """Return the X coordinate (f32[0], byte 0) of every packed instance."""
    xs: list[float] = []
    for _panel_idx, kind, count, instance_start in _iter_packed_batches(packed):
        stride = _INSTANCE_SIZES[kind]
        for i in range(count):
            (x,) = struct.unpack_from("<f", packed, instance_start + i * stride)
            xs.append(x)
    return xs


def _header_h_for(composite) -> float:
    """Compute the header band height the composite's figure chrome injects.

    Feeds the merged pre-chrome body width/height to the same Rust layout
    helper (``figure_title_nodes``) the composite render entry uses internally.
    """
    import json

    from ferrum._chrome import chrome_kwargs, merge_configure_layers
    from ferrum._core import figure_title_nodes

    base = composite
    # Strip the figure title so we measure the pre-chrome merged body.
    base_json, _ = base._render_interactive()
    base_scene = json.loads(base_json)
    chrome = chrome_kwargs(merge_configure_layers(getattr(composite, "_configure_layers", None)))
    _nodes, header_h, _footer_h = figure_title_nodes(
        width=float(base_scene["width"]),
        height=float(base_scene["height"]),
        title=composite._figure_title,
        subtitle=composite._figure_subtitle,
        caption=composite._figure_caption,
        **chrome,
    )
    return float(header_h)


@pytest.fixture
def big_point_chart():
    """A >1000-mark point chart so its circle batch packs into binary."""
    n = 1500
    df = pl.DataFrame({"x": [float(i) for i in range(n)], "y": [float(i % 50) for i in range(n)]})
    return fm.Chart(df).mark_point().encode(x="x", y="y")


@pytest.fixture
def big_bar_chart():
    """A >1000-mark bar chart so its rect batch packs into binary."""
    n = 1500
    df = pl.DataFrame(
        {"x": [float(i) for i in range(n)], "y": [float(i % 50 + 1) for i in range(n)]}
    )
    return fm.Chart(df).mark_bar().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# Sanity: the children actually pack (else the regression test is vacuous).
# ---------------------------------------------------------------------------


def test_big_point_child_packs(big_point_chart):
    composite = big_point_chart | big_point_chart
    _scene, packed = composite._render_interactive()
    batches = list(_iter_packed_batches(packed))
    assert batches, "a >1000-mark point composite must produce packed circle batches"
    assert all(kind == 0 for _p, kind, _c, _s in batches), "circle batches expected"


def test_big_bar_child_packs(big_bar_chart):
    composite = big_bar_chart | big_bar_chart
    _scene, packed = composite._render_interactive()
    batches = list(_iter_packed_batches(packed))
    assert batches, "a >1000-mark bar composite must produce packed rect batches"
    assert all(kind == 1 for _p, kind, _c, _s in batches), "rect batches expected"


# ---------------------------------------------------------------------------
# T3 core: packed Y shifts by header_h under a figure title.
# ---------------------------------------------------------------------------


def test_titled_composite_shifts_packed_circle_y_by_header_h(big_point_chart):
    composite = big_point_chart | big_point_chart
    _base_json, base_packed = composite._render_interactive()
    titled = composite.properties(title="PackedBand")
    _titled_json, titled_packed = titled._render_interactive()

    header_h = _header_h_for(titled)
    assert header_h > 0, "a figure title must produce a non-zero header band"

    base_ys = _packed_ys(base_packed)
    titled_ys = _packed_ys(titled_packed)
    assert len(base_ys) == len(titled_ys) > 0

    # Every packed instance's Y is shifted by exactly header_h (f32 precision).
    for by, ty in zip(base_ys, titled_ys):
        assert ty == pytest.approx(by + header_h, abs=1e-3), (
            f"packed circle cy must shift by header_h ({header_h}): base={by} titled={ty}"
        )


def test_titled_composite_shifts_packed_rect_y_by_header_h(big_bar_chart):
    composite = big_bar_chart | big_bar_chart
    _base_json, base_packed = composite._render_interactive()
    titled = composite.properties(title="RectBand")
    _titled_json, titled_packed = titled._render_interactive()

    header_h = _header_h_for(titled)
    assert header_h > 0

    base_ys = _packed_ys(base_packed)
    titled_ys = _packed_ys(titled_packed)
    assert len(base_ys) == len(titled_ys) > 0
    for by, ty in zip(base_ys, titled_ys):
        assert ty == pytest.approx(by + header_h, abs=1e-3)


def test_titled_composite_packed_x_unchanged(big_point_chart):
    """Only Y shifts — X (and by extension radius/size/style) are untouched."""
    composite = big_point_chart | big_point_chart
    _base_json, base_packed = composite._render_interactive()
    titled = composite.properties(title="XStable")
    _titled_json, titled_packed = titled._render_interactive()

    base_xs = _packed_xs(base_packed)
    titled_xs = _packed_xs(titled_packed)
    assert len(base_xs) == len(titled_xs) > 0
    for bx, tx in zip(base_xs, titled_xs):
        assert tx == pytest.approx(bx, abs=1e-3), "packed X must not shift"


def test_packed_y_matches_scene_node_shift(big_point_chart):
    """The packed Y shift equals the plot_area Y shift (one post-offset frame).

    The plot_area Y in the titled scene is shifted down by header_h relative to
    the untitled scene; the packed instances must share that exact offset so GPU
    marks and axes line up.
    """
    import json

    composite = big_point_chart | big_point_chart
    base_json, base_packed = composite._render_interactive()
    titled = composite.properties(title="FrameMatch")
    titled_json, titled_packed = titled._render_interactive()

    base_scene = json.loads(base_json)
    titled_scene = json.loads(titled_json)

    base_plot_y = base_scene["panels"][0]["plot_area"]["y"]
    titled_plot_y = titled_scene["panels"][0]["plot_area"]["y"]
    plot_shift = titled_plot_y - base_plot_y
    assert plot_shift > 0, "the plot_area must shift down under the title band"

    base_ys = _packed_ys(base_packed)
    titled_ys = _packed_ys(titled_packed)
    packed_shift = titled_ys[0] - base_ys[0]

    assert packed_shift == pytest.approx(plot_shift, abs=1e-3), (
        f"packed Y shift ({packed_shift}) must equal plot_area shift ({plot_shift})"
    )


# ---------------------------------------------------------------------------
# Backward compatibility.
# ---------------------------------------------------------------------------


def test_non_titled_packed_composite_byte_identical(big_point_chart):
    """No figure title -> packed bytes byte-identical (only panel_idx remap)."""
    composite = big_point_chart | big_point_chart
    _before_json, before_packed = composite._render_interactive()
    _after_json, after_packed = composite.properties()._render_interactive()
    assert before_packed == after_packed


def test_caption_only_composite_packed_byte_identical(big_point_chart):
    """A caption is a footer band (header_h == 0): packed bytes unchanged."""
    composite = big_point_chart | big_point_chart
    _before_json, before_packed = composite._render_interactive()
    captioned = composite.properties(caption="FooterOnly")
    _after_json, after_packed = captioned._render_interactive()
    # Caption grows the canvas downward (footer) but applies no header offset,
    # so the packed instance bytes are byte-identical.
    assert before_packed == after_packed


def test_small_data_titled_composite_packed_empty_and_scene_offset():
    """Small-data composite: no packing; scene nodes still offset by header_h."""
    import json

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 3.0, 2.0, 1.0]})
    a = fm.Chart(df).mark_point().encode(x="x", y="y")
    b = fm.Chart(df).mark_point().encode(x="x", y="y")
    composite = a | b

    base_json, base_packed = composite._render_interactive()
    titled = composite.properties(title="SmallBand")
    titled_json, titled_packed = titled._render_interactive()

    # No packing for a 4-row chart.
    assert base_packed == b""
    assert titled_packed == b""

    # The scene-graph plot_area still shifts down (existing behavior preserved).
    base_plot_y = json.loads(base_json)["panels"][0]["plot_area"]["y"]
    titled_plot_y = json.loads(titled_json)["panels"][0]["plot_area"]["y"]
    assert titled_plot_y > base_plot_y


# ---------------------------------------------------------------------------
# Grid / sparse composites also offset their packed instances.
# ---------------------------------------------------------------------------


def test_vconcat_titled_shifts_packed_y(big_point_chart):
    composite = big_point_chart & big_point_chart
    _base_json, base_packed = composite._render_interactive()
    titled = composite.properties(title="VBand")
    _titled_json, titled_packed = titled._render_interactive()

    header_h = _header_h_for(titled)
    assert header_h > 0

    base_ys = _packed_ys(base_packed)
    titled_ys = _packed_ys(titled_packed)
    assert len(base_ys) == len(titled_ys) > 0
    for by, ty in zip(base_ys, titled_ys):
        assert ty == pytest.approx(by + header_h, abs=1e-3)


def test_concat_grid_titled_shifts_packed_y(big_point_chart):
    composite = fm.ConcatChart(big_point_chart, big_point_chart, columns=2)
    _base_json, base_packed = composite._render_interactive()
    titled = composite.properties(title="GridBand")
    _titled_json, titled_packed = titled._render_interactive()

    header_h = _header_h_for(titled)
    assert header_h > 0

    base_ys = _packed_ys(base_packed)
    titled_ys = _packed_ys(titled_packed)
    assert len(base_ys) == len(titled_ys) > 0
    for by, ty in zip(base_ys, titled_ys):
        assert ty == pytest.approx(by + header_h, abs=1e-3)


# ---------------------------------------------------------------------------
# Finding-B discriminating tests: per-panel (dx, dy) offset in packed data.
# ---------------------------------------------------------------------------


def _packed_xs_by_panel(packed: bytes) -> dict[int, list[float]]:
    """Return packed instance X coords grouped by panel_idx."""
    by_panel: dict[int, list[float]] = {}
    for panel_idx, kind, count, instance_start in _iter_packed_batches(packed):
        xs: list[float] = []
        stride = _INSTANCE_SIZES[kind]
        for i in range(count):
            (x,) = struct.unpack_from("<f", packed, instance_start + i * stride)
            xs.append(x)
        by_panel.setdefault(panel_idx, []).extend(xs)
    return by_panel


def _packed_ys_by_panel(packed: bytes) -> dict[int, list[float]]:
    """Return packed instance Y coords grouped by panel_idx."""
    by_panel: dict[int, list[float]] = {}
    for panel_idx, kind, count, instance_start in _iter_packed_batches(packed):
        ys: list[float] = []
        stride = _INSTANCE_SIZES[kind]
        for i in range(count):
            y_pos = instance_start + i * stride + _Y_FIELD_OFFSET
            (y,) = struct.unpack_from("<f", packed, y_pos)
            ys.append(y)
        by_panel.setdefault(panel_idx, []).extend(ys)
    return by_panel


def test_hconcat_packed_x_offset_by_panel_dx(big_point_chart):
    """Finding-B headline test: hconcat panel-1 packed instances are shifted
    by the panel's dx relative to panel-0 instances.

    Before the fix: panel-0 and panel-1 X coordinates are identical (dx never
    applied) → the assertion fails.  After the fix: panel-1 min X ≈ panel-0
    min X + (panel1_plot_x - panel0_plot_x).
    """
    import json

    composite = big_point_chart | big_point_chart
    scene_json, packed = composite._render_interactive()
    scene = json.loads(scene_json)

    xs_by_panel = _packed_xs_by_panel(packed)
    assert set(xs_by_panel.keys()) == {0, 1}, (
        f"expected packed data for panel 0 and 1; got panels {set(xs_by_panel.keys())}"
    )
    assert len(xs_by_panel[0]) > 1000, "panel 0 must have >1000 packed instances"
    assert len(xs_by_panel[1]) > 1000, "panel 1 must have >1000 packed instances"

    panel0_plot_x = scene["panels"][0]["plot_area"]["x"]
    panel1_plot_x = scene["panels"][1]["plot_area"]["x"]
    dx = panel1_plot_x - panel0_plot_x
    assert dx > 0, "hconcat: panel 1 plot_area.x must be greater than panel 0"

    panel0_min_x = min(xs_by_panel[0])
    panel1_min_x = min(xs_by_panel[1])
    assert panel1_min_x == pytest.approx(panel0_min_x + dx, abs=1e-2), (
        f"panel-1 packed min X ({panel1_min_x:.3f}) must equal panel-0 min X "
        f"({panel0_min_x:.3f}) + dx ({dx:.3f}); diff = "
        f"{panel1_min_x - panel0_min_x:.3f} vs expected {dx:.3f}"
    )


def test_vconcat_packed_y_offset_by_panel_dy(big_point_chart):
    """Finding-B: vconcat panel-1 packed instances are shifted by the panel's
    dy relative to panel-0 instances.

    Before the fix: panel-0 and panel-1 Y coordinates are identical (dy never
    applied to packed data, only the scene nodes) → the assertion fails.
    After the fix: panel-1 min Y ≈ panel-0 min Y + (panel1_plot_y - panel0_plot_y).
    """
    import json

    composite = big_point_chart & big_point_chart
    scene_json, packed = composite._render_interactive()
    scene = json.loads(scene_json)

    ys_by_panel = _packed_ys_by_panel(packed)
    assert set(ys_by_panel.keys()) == {0, 1}, (
        f"expected packed data for panel 0 and 1; got panels {set(ys_by_panel.keys())}"
    )
    assert len(ys_by_panel[0]) > 1000, "panel 0 must have >1000 packed instances"
    assert len(ys_by_panel[1]) > 1000, "panel 1 must have >1000 packed instances"

    panel0_plot_y = scene["panels"][0]["plot_area"]["y"]
    panel1_plot_y = scene["panels"][1]["plot_area"]["y"]
    dy = panel1_plot_y - panel0_plot_y
    assert dy > 0, "vconcat: panel 1 plot_area.y must be greater than panel 0"

    panel0_min_y = min(ys_by_panel[0])
    panel1_min_y = min(ys_by_panel[1])
    assert panel1_min_y == pytest.approx(panel0_min_y + dy, abs=1e-2), (
        f"panel-1 packed min Y ({panel1_min_y:.3f}) must equal panel-0 min Y "
        f"({panel0_min_y:.3f}) + dy ({dy:.3f}); diff = "
        f"{panel1_min_y - panel0_min_y:.3f} vs expected {dy:.3f}"
    )


def test_hconcat_packed_marks_within_panel_plot_area(big_point_chart):
    """Alignment sanity: every packed instance X must fall within its panel's
    plot_area X range (allowing a small radius/pixel margin).

    Before the fix: panel-1 marks render at panel-0 coordinates, outside their
    scissor rect → every panel-1 instance violates this bound.
    After the fix: all instances fall within their panel's plot_area.
    """
    import json

    composite = big_point_chart | big_point_chart
    scene_json, packed = composite._render_interactive()
    scene = json.loads(scene_json)

    xs_by_panel = _packed_xs_by_panel(packed)
    assert xs_by_panel, "hconcat must produce packed instances"

    margin = 10.0  # allow a small margin for circle radius overhang
    for panel_idx, xs in xs_by_panel.items():
        plot_area = scene["panels"][panel_idx]["plot_area"]
        plot_x_min = plot_area["x"] - margin
        plot_x_max = plot_area["x"] + plot_area["w"] + margin
        out_of_bounds = [x for x in xs if x < plot_x_min or x > plot_x_max]
        assert not out_of_bounds, (
            f"panel {panel_idx}: {len(out_of_bounds)} packed instances fall outside "
            f"plot_area X range [{plot_area['x']:.1f}, "
            f"{plot_area['x'] + plot_area['w']:.1f}] (margin={margin}); "
            f"sample out-of-bounds: {out_of_bounds[:5]}"
        )
