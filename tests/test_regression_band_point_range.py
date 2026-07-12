"""Regression tests for issue #39: BandScale/PointScale `range=` silently dropped.

Before the fix, ``ScaleSpec::Band``/``ScaleSpec::Point`` had no ``range`` field on
the Rust side, so a user-supplied ``fr.BandScale(domain=[...], range=[lo, hi])`` (or
``fr.PointScale(...)``) round-tripped through ``_to_scale_spec_dict()`` /
``chart.to_json()`` with the range silently stripped, and the positional resolver
fell back to the full panel extent regardless of what the caller asked for.

The fix adds the field, emits it from ``to_scale_spec``, and honors it in the
positional resolver (falling back to the panel extent only when ``range`` is
absent). These tests check three things:

1. **Wire-level** — ``chart.to_json()`` carries the requested ``range`` under the
   band/point scale on the x encoding.
2. **Render-level** — with an explicit sub-panel ``range``, all rendered mark
   positions (bar x/width for BandScale, circle cx for PointScale) fall inside
   that sub-range rather than spanning the full panel.
3. **Fallback** — a band scale built *without* ``range`` has no ``"range"`` key
   in its wire dict, so the panel-extent fallback path is untouched by the fix.
"""

from __future__ import annotations

import json
import re

import polars as pl

import ferrum as fr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data_bar_rects(svg: str) -> list[dict[str, str]]:
    """Return attr dicts for <rect> elements that are data bars.

    Background/panel rects either carry the theme background fill or omit
    ``fill`` entirely (inheriting from a parent group); data bars always carry
    an explicit non-background fill.
    """
    rects = []
    for m in re.finditer(r"<rect([^/]+)/>", svg):
        attrs = dict(re.findall(r'([\w-]+)="([^"]+)"', m.group(1)))
        fill = attrs.get("fill")
        if fill and fill != "#faf7f2":
            rects.append(attrs)
    return rects


def _circle_cxs(svg: str) -> list[float]:
    """Return sorted cx values from all <circle> elements in the SVG."""
    return sorted(float(v) for v in re.findall(r'<circle[^>]*cx="([^"]+)"', svg))


# ---------------------------------------------------------------------------
# 1. Wire-level: range survives to_json()
# ---------------------------------------------------------------------------


def test_band_scale_range_present_in_wire_json():
    """chart.to_json() carries BandScale(range=[...]) under the x scale.

    Regression: issue #39 — ScaleSpec::Band had no range field, so this key
    was silently absent from the wire dict.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [10.0, 20.0, 30.0, 40.0]})
    chart = (
        fr.Chart(df)
        .mark_bar()
        .encode(
            x=fr.X("cat", scale=fr.BandScale(domain=["a", "b", "c", "d"], range=[40.0, 260.0])),
            y="val",
        )
    )
    spec = json.loads(chart.to_json())
    x_scale = spec["encoding"]["x"]["scale"]
    assert x_scale.get("type") == "band"
    assert x_scale.get("range") == [40.0, 260.0], (
        f"BandScale range was not emitted in to_json(); got scale dict {x_scale!r}"
    )


def test_point_scale_range_present_in_wire_json():
    """chart.to_json() carries PointScale(range=[...]) under the x scale.

    Regression: issue #39 — ScaleSpec::Point had no range field, so this key
    was silently absent from the wire dict.
    """
    df = pl.DataFrame({"cat": ["x", "y", "z"], "val": [5.0, 10.0, 15.0]})
    chart = (
        fr.Chart(df)
        .mark_point()
        .encode(
            x=fr.X("cat", scale=fr.PointScale(domain=["x", "y", "z"], range=[40.0, 260.0])),
            y="val",
        )
    )
    spec = json.loads(chart.to_json())
    x_scale = spec["encoding"]["x"]["scale"]
    assert x_scale.get("type") == "point"
    assert x_scale.get("range") == [40.0, 260.0], (
        f"PointScale range was not emitted in to_json(); got scale dict {x_scale!r}"
    )


# ---------------------------------------------------------------------------
# 2. Render-level: mark positions honor the requested sub-panel range
# ---------------------------------------------------------------------------


def test_band_scale_range_constrains_bar_positions():
    """Bars render within the explicit BandScale range, not the full panel.

    Regression: issue #39 — the dropped range field meant the positional
    resolver fell back to the full panel extent (~0-570px on a 600px-wide
    chart), spilling far outside the requested [40, 260] sub-range.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [10.0, 20.0, 30.0, 40.0]})
    chart = (
        fr.Chart(df)
        .mark_bar()
        .encode(
            x=fr.X("cat", scale=fr.BandScale(domain=["a", "b", "c", "d"], range=[40.0, 260.0])),
            y="val",
        )
        .properties(width=600, height=400)
    )
    svg = chart.to_svg()
    bars = _data_bar_rects(svg)
    assert len(bars) == 4, f"Expected 4 data bars, got {len(bars)}: {bars}"

    tol = 0.5
    for attrs in bars:
        x0 = float(attrs["x"])
        x1 = x0 + float(attrs["width"])
        assert x0 >= 40.0 - tol and x1 <= 260.0 + tol, (
            f"Bar rect [{x0}, {x1}] falls outside BandScale range [40.0, 260.0]: {attrs!r}"
        )


def test_point_scale_range_constrains_point_positions():
    """Points render within the explicit PointScale range, not the full panel.

    Regression: issue #39 — same drop as BandScale but for PointScale, so
    mark_point circle centers spilled outside the requested sub-range.
    """
    df = pl.DataFrame({"cat": ["x", "y", "z"], "val": [5.0, 10.0, 15.0]})
    chart = (
        fr.Chart(df)
        .mark_point()
        .encode(
            x=fr.X("cat", scale=fr.PointScale(domain=["x", "y", "z"], range=[40.0, 260.0])),
            y="val",
        )
        .properties(width=600, height=400)
    )
    svg = chart.to_svg()
    cxs = _circle_cxs(svg)
    assert len(cxs) == 3, f"Expected 3 points, got {len(cxs)}: {cxs}"

    tol = 0.5
    for cx in cxs:
        assert 40.0 - tol <= cx <= 260.0 + tol, (
            f"Circle cx={cx} falls outside PointScale range [40.0, 260.0]: {cxs!r}"
        )


# ---------------------------------------------------------------------------
# 3. Fallback: no range specified -> no "range" key, panel-extent path unchanged
# ---------------------------------------------------------------------------


def test_band_scale_without_range_omits_range_key():
    """A BandScale built without range has no 'range' key in its wire dict.

    Guards the fallback-to-panel-extent path: this must remain untouched by
    the #39 fix. tests/test_scale_spec_parity.py::TestByteIdentity already
    freezes the full no-range band-chart to_json() output; this test only
    checks the narrower invariant that "range" is absent.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [10.0, 20.0, 30.0, 40.0]})
    chart = (
        fr.Chart(df)
        .mark_bar()
        .encode(
            x=fr.X("cat", scale=fr.BandScale(domain=["a", "b", "c", "d"], padding=0.1)),
            y="val",
        )
    )
    spec = json.loads(chart.to_json())
    x_scale = spec["encoding"]["x"]["scale"]
    assert x_scale.get("type") == "band"
    assert "range" not in x_scale, (
        f"BandScale without range= should not emit a 'range' key; got {x_scale!r}"
    )


# ---------------------------------------------------------------------------
# 4. Geometry / axis-alignment: phase 2 (GH #39) — band geometry consumers
#    (bar/box/heatmap/tick width, categorical axis tick placement) must also
#    derive from the resolved scale's explicit range, not the panel extent.
#    These are the discriminating RED tests for design-docs/superpowers/specs/
#    2026-07-11-band-point-range-geometry-design.md §9; they fail today
#    (phase 1 fixes mark *position* only) and must pass after Tasks 3-4.
# ---------------------------------------------------------------------------

BAND_RANGE = [40.0, 260.0]
# domain of 4 categories over BAND_RANGE: step = (260 - 40) / 4 = 55,
# centers at range[0] + step * (i + 0.5) for i in 0..4.
BAND_STEP = 55.0
BAND_CENTERS = [67.5, 122.5, 177.5, 232.5]
_TOL = 0.5


def _axis_tick_label_pos(svg: str, label: str, *, coord: str) -> float:
    """Return the pixel position (``coord`` = "x" or "y") of the axis <text>
    tick whose content is exactly ``label``.

    Category axis tick labels are plain SVG <text> elements; this assumes
    ``label`` is unique among the SVG's rendered text nodes, true for the
    single-letter category names used by these tests.
    """
    match = re.search(rf"<text([^>]*)>{re.escape(label)}</text>", svg)
    assert match is not None, f"no axis tick <text>{label}</text> found in SVG"
    attrs = dict(re.findall(r'([\w-]+)="([^"]+)"', match.group(1)))
    return float(attrs[coord])


def _mark_lines(svg: str, stroke: str = "#2563eb") -> list[dict[str, float]]:
    """Return numeric x1/y1/x2/y2 for <line> elements with the given stroke.

    Data-mark lines (mark_tick) carry the resolved mark color; axis and grid
    lines use the theme's neutral gray strokes, so filtering by stroke
    isolates data marks from chrome.
    """
    lines = []
    for m in re.finditer(r"<line([^/]+)/>", svg):
        attrs = dict(re.findall(r'([\w-]+)="([^"]+)"', m.group(1)))
        if attrs.get("stroke") != stroke:
            continue
        lines.append({k: float(attrs[k]) for k in ("x1", "y1", "x2", "y2")})
    return lines


def _heatmap_cell_rects(svg: str) -> list[dict[str, str]]:
    """Return attr dicts for heatmap data-cell <rect> elements.

    Excludes the panel background (theme fill) and the color-scale legend
    swatch, which is filled with a ``url(#...)`` gradient reference rather
    than a flat color.
    """
    cells = []
    for m in re.finditer(r"<rect([^/]+)/>", svg):
        attrs = dict(re.findall(r'([\w-]+)="([^"]+)"', m.group(1)))
        fill = attrs.get("fill")
        if not fill or fill == "#faf7f2" or fill.startswith("url("):
            continue
        cells.append(attrs)
    return cells


def test_ordinal_y_range_constrains_bar_heights_and_tick_labels():
    """Horizontal bars (ordinal y) honor an explicit BandScale range on y, and
    the y-axis tick labels align to the same band centers as the bars.

    Regression: issue #39 phase 2 — bar geometry and categorical axis tick
    placement are independently re-derived from the full panel extent, so an
    explicit y range constrains mark *position* only (phase 1); bar height
    still spills past the range and tick labels stay at panel-uniform
    centers instead of the band centers.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [10.0, 20.0, 30.0, 40.0]})
    chart = (
        fr.Chart(df)
        .mark_bar()
        .encode(
            y=fr.Y("cat", scale=fr.BandScale(domain=["a", "b", "c", "d"], range=BAND_RANGE)),
            x="val",
        )
        .properties(width=600, height=400)
    )
    svg = chart.to_svg()
    bars = _data_bar_rects(svg)
    assert len(bars) == 4, f"Expected 4 data bars, got {len(bars)}: {bars}"

    for attrs in bars:
        y0 = float(attrs["y"])
        y1 = y0 + float(attrs["height"])
        assert y0 >= BAND_RANGE[0] - _TOL and y1 <= BAND_RANGE[1] + _TOL, (
            f"Bar rect y-extent [{y0}, {y1}] falls outside BandScale range {BAND_RANGE}: {attrs!r}"
        )

    # y-axis tick <text> elements carry a baked-in baseline offset (+0.333em,
    # ~3.667px at the default font-size 11) relative to the geometric band
    # center -- the x-axis sibling check below doesn't need this because
    # text-anchor="middle" centers each label horizontally with no baseline
    # offset. Two checks avoid both false negatives (raw offset vs. center)
    # and false positives (too-loose a tolerance could pass by accident):
    # (a) each label falls within its own band interval -- the offset
    # (~3.67px) is far smaller than the half-step (27.5px), so this can't
    # pass under panel-uniform placement by coincidence; and (b) the
    # label-minus-center delta is uniform across all four labels, which
    # locks in true band-center alignment without hardcoding the font
    # metric as an exact oracle.
    deltas = []
    for i, label in enumerate(["a", "b", "c", "d"]):
        actual = _axis_tick_label_pos(svg, label, coord="y")
        band_lo = BAND_RANGE[0] + BAND_STEP * i
        band_hi = BAND_RANGE[0] + BAND_STEP * (i + 1)
        assert band_lo <= actual <= band_hi, (
            f"y tick label {label!r} at y={actual} falls outside its own band "
            f"interval [{band_lo}, {band_hi}] under range {BAND_RANGE}"
        )
        deltas.append(actual - BAND_CENTERS[i])
    assert max(deltas) - min(deltas) <= _TOL, (
        f"y tick label offsets from band centers are not uniform: {deltas} "
        "(labels aren't consistently aligned to the same band centers as the bars)"
    )


def test_heatmap_cell_extent_matches_explicit_range():
    """mark_rect (heatmap) cell extent on a ranged categorical axis equals
    |range| / n_categories; the unranged axis is unaffected.

    Regression: issue #39 phase 2 — heatmap cell width/height is
    independently re-derived from the full panel extent
    (``panel.w / n_categories``-style arithmetic), ignoring an explicit
    BandScale range on one axis.
    """
    df = pl.DataFrame(
        {
            "row": ["a", "a", "b", "b"],
            "col": ["x", "y", "x", "y"],
            "val": [1.0, 2.0, 3.0, 4.0],
        }
    )

    def _render(*, ranged: bool) -> str:
        x_scale = (
            fr.BandScale(domain=["a", "b"], range=BAND_RANGE)
            if ranged
            else fr.BandScale(domain=["a", "b"])
        )
        chart = (
            fr.Chart(df)
            .mark_rect()
            .encode(x=fr.X("row", scale=x_scale), y="col", color="val")
            .properties(width=600, height=400)
        )
        return chart.to_svg()

    ranged_cells = _heatmap_cell_rects(_render(ranged=True))
    baseline_cells = _heatmap_cell_rects(_render(ranged=False))
    assert len(ranged_cells) == 4, (
        f"Expected 4 heatmap cells, got {len(ranged_cells)}: {ranged_cells}"
    )
    assert len(baseline_cells) == 4, (
        f"Expected 4 baseline heatmap cells, got {len(baseline_cells)}: {baseline_cells}"
    )

    expected_width = abs(BAND_RANGE[1] - BAND_RANGE[0]) / 2  # 2 row categories
    for attrs in ranged_cells:
        width = float(attrs["width"])
        assert abs(width - expected_width) <= _TOL, (
            f"Heatmap cell width {width} does not match |range|/n_categories "
            f"({expected_width}) for explicit x range {BAND_RANGE}: {attrs!r}"
        )

    # The unranged y (col) axis extent must be unaffected by the x-axis range fix.
    ranged_heights = sorted(float(a["height"]) for a in ranged_cells)
    baseline_heights = sorted(float(a["height"]) for a in baseline_cells)
    assert ranged_heights == baseline_heights, (
        f"y-axis (unranged) cell heights changed by an x-axis range fix: "
        f"{ranged_heights} vs baseline {baseline_heights}"
    )


def test_tick_mark_half_extent_scales_with_explicit_range():
    """mark_tick's band-dimension half-extent derives from the explicit range
    extent, not the full panel width; ticks nest within the requested range.

    Regression: issue #39 phase 2 — tick.rs's ordinal-x + quantitative-y mode
    (``tick_half = panel.w / n_cats / n_groups * band_size``) is
    panel-derived and ignores ``range=``; today it spills ticks outside
    [40, 260] and sizes them far wider than a 55px band step.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [10.0, 20.0, 30.0, 40.0]})
    chart = (
        fr.Chart(df)
        .mark_tick()
        .encode(
            x=fr.X("cat", scale=fr.BandScale(domain=["a", "b", "c", "d"], range=BAND_RANGE)),
            y="val",
        )
        .properties(width=600, height=400)
    )
    svg = chart.to_svg()
    lines = _mark_lines(svg)
    assert len(lines) == 4, f"Expected 4 tick marks, got {len(lines)}: {lines}"

    step = (BAND_RANGE[1] - BAND_RANGE[0]) / 4  # 4 categories
    for line in lines:
        x0, x1 = sorted((line["x1"], line["x2"]))
        assert x0 >= BAND_RANGE[0] - _TOL and x1 <= BAND_RANGE[1] + _TOL, (
            f"Tick mark x-extent [{x0}, {x1}] falls outside BandScale range {BAND_RANGE}: {line!r}"
        )
        assert x1 - x0 <= step + _TOL, (
            f"Tick mark width {x1 - x0} exceeds the band step {step} for range {BAND_RANGE}: {line!r}"
        )


def test_x_axis_tick_labels_align_with_band_centers():
    """Under an explicit BandScale range, x tick label centers equal the mark
    band centers, not the uniform-over-panel placement used today.

    Regression: issue #39 phase 2 — categorical axis tick placement is
    independently derived by dividing the full panel width evenly, ignoring
    the resolved scale's band centers under an explicit range.
    """
    df = pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [10.0, 20.0, 30.0, 40.0]})
    chart = (
        fr.Chart(df)
        .mark_bar()
        .encode(
            x=fr.X("cat", scale=fr.BandScale(domain=["a", "b", "c", "d"], range=BAND_RANGE)),
            y="val",
        )
        .properties(width=600, height=400)
    )
    svg = chart.to_svg()
    for label, expected_center in zip(["a", "b", "c", "d"], BAND_CENTERS):
        actual = _axis_tick_label_pos(svg, label, coord="x")
        assert abs(actual - expected_center) <= _TOL, (
            f"x tick label {label!r} at x={actual} does not align with mark band "
            f"center {expected_center} under range {BAND_RANGE}"
        )


def test_dodged_bars_non_overlapping_within_explicit_range():
    """Dodged sub-bars stay within an explicit BandScale range and do not
    overlap each other, matching the band geometry they subdivide.

    Regression: issue #39 phase 2 — Dodge's sub-band offsets are correct
    (bandwidth()-derived, the in-tree precedent per the design spec), but bar
    *width* is still panel-derived, so sub-bars both spill outside the range
    and overlap adjacent dodge groups.
    """
    df = pl.DataFrame(
        {
            "cat": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "group": ["g1", "g2"] * 4,
            "val": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
        }
    )
    chart = (
        fr.Chart(df)
        .mark_bar(position=fr.Dodge())
        .encode(
            x=fr.X("cat", scale=fr.BandScale(domain=["a", "b", "c", "d"], range=BAND_RANGE)),
            y="val",
            color="group",
        )
        .properties(width=600, height=400)
    )
    svg = chart.to_svg()
    bars = _data_bar_rects(svg)
    assert len(bars) == 8, f"Expected 8 dodged bars, got {len(bars)}: {bars}"

    intervals = []
    for attrs in bars:
        x0 = float(attrs["x"])
        x1 = x0 + float(attrs["width"])
        assert x0 >= BAND_RANGE[0] - _TOL and x1 <= BAND_RANGE[1] + _TOL, (
            f"Dodged bar rect [{x0}, {x1}] falls outside BandScale range {BAND_RANGE}: {attrs!r}"
        )
        intervals.append((x0, x1))

    intervals.sort()
    for (x0, x1), (nx0, nx1) in zip(intervals, intervals[1:]):
        assert x1 <= nx0 + _TOL, f"Dodged sub-bars overlap: [{x0}, {x1}] vs [{nx0}, {nx1}]"
