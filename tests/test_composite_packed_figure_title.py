"""Packed-instance alignment under the figure-title band (round-3 T3).

`_inject_figure_chrome` shifts a composite's scene-graph mark nodes and each
panel's ``plot_area`` / ``clip`` / axes down by the figure-title header band
(``header_h``).  High-cardinality (>1000-mark) circle / rect batches do NOT
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

    Mirrors :func:`ferrum.composition._inject_figure_chrome`: feed the merged
    pre-chrome body width/height to the same Rust layout helper.
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
