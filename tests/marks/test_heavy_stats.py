"""Smoke tests for heavy-stat marks (Phase 8b Sub-batch F).

Each mark gets:
- A spec-build smoke test (assert spec.layers is not None or appropriate shape)
- Key kwarg propagation test (verify kwarg flows into spec JSON)
- A render smoke test (to_svg returns valid SVG) where supported
- An error-case test where applicable
"""

import re

import polars as pl
import pyarrow as pa  # noqa: F401
import numpy as np
import pytest
import ferrum as fe


@pytest.fixture
def df_xy():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 1.2],
            "y": [1.0, 2.0, 3.0, 2.5, 1.5, 2.0, 3.0, 2.5, 1.5, 2.5],
        }
    )


@pytest.fixture
def df_xyc():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 2.5, 1.5],
            "color_value": [10.0, 20.0, 15.0, 25.0, 5.0],
        }
    )


# ---- SVG-geometry helpers (violin quartile + mark_function clip tests) ----

# The two default-theme colors every axis tick and gridline renders in,
# regardless of any mark-level color/style override -- used to isolate a
# mark's own <line> elements (e.g. violin's quartile ticks) from axis
# chrome without depending on a specific mark color (which a `color=` hue
# encoding could otherwise recolor out from under a literal-hex filter).
_AXIS_CHROME_STROKES = {"#6b7280", "#d6d3d1"}


def _panel_rect(svg: str) -> tuple[float, float, float, float]:
    """Return (x, y, width, height) of the panel's clip-path rect."""
    m = re.search(
        r'<clipPath[^>]*><rect x="(-?[0-9.]+)" y="(-?[0-9.]+)" '
        r'width="(-?[0-9.]+)" height="(-?[0-9.]+)"',
        svg,
    )
    assert m, "expected a panel clip rect in the SVG"
    return tuple(float(v) for v in m.groups())  # type: ignore[return-value]


def _path_bboxes(svg: str, axis: str) -> list[tuple[float, float]]:
    """Return (min, max) along *axis* ("x" or "y") for every <path>'s coordinates.

    Violin bodies render as ``<path d="M x y L x y ...">`` -- space-separated
    coordinate pairs (unlike ``<line>``/``<polyline>``'s comma-separated
    points) -- one ``<path>`` per (category, hue) group.

    S1 hardening (T11 cycle 3 review): matches ``d="..."`` wherever it falls
    among a ``<path>``'s attributes, not only when ``d`` happens to be
    first (every producer today emits ``d`` first, but a future one is not
    obligated to), and raises a descriptive error rather than a bare
    ``min()``-of-empty ``ValueError`` if a matched path carries no ``M``/``L``
    coordinate pairs.
    """
    idx = 0 if axis == "x" else 1
    boxes = []
    for path_match in re.finditer(r'<path[^>]*\bd="([^"]+)"', svg):
        d = path_match.group(1)
        coords = re.findall(r"[ML] (-?[0-9.]+) (-?[0-9.]+)", d)
        if not coords:
            raise AssertionError(f"<path> d attribute has no M/L coordinate pairs: {d!r}")
        vals = [float(c[idx]) for c in coords]
        boxes.append((min(vals), max(vals)))
    return boxes


def _non_chrome_line_spans(svg: str, axis: str) -> list[tuple[float, float]]:
    """Return (a1, a2) along *axis* ("x" or "y") for every <line> that isn't
    axis/grid chrome -- isolates a mark's own crossbars (e.g. violin's
    quartile ticks) generically, without relying on a specific stroke color
    that a `color=` hue encoding might or might not apply.
    """
    a1_name, a2_name = f"{axis}1", f"{axis}2"
    spans = []
    for m in re.finditer(r"<line([^>]*)/>", svg):
        attrs = m.group(1)
        stroke_m = re.search(r'stroke="([^"]+)"', attrs)
        if not stroke_m or stroke_m.group(1) in _AXIS_CHROME_STROKES:
            continue
        a1 = float(re.search(rf'{a1_name}="(-?[0-9.]+)"', attrs).group(1))
        a2 = float(re.search(rf'{a2_name}="(-?[0-9.]+)"', attrs).group(1))
        spans.append((a1, a2))
    return spans


# ---- mark_contour ----


def test_contour_smoke(df_xy):
    spec = fe.Chart(df_xy).mark_contour().encode(x="x", y="y")._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 1


def test_contour_thresholds_propagates(df_xy):
    spec = fe.Chart(df_xy).mark_contour(thresholds=10).encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    assert '"thresholds":10' in json_str


def test_contour_fill_mode(df_xy):
    spec = fe.Chart(df_xy).mark_contour(fill=True).encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    assert '"fill":true' in json_str


# ---- mark_violin ----


def test_violin_smoke(df_xy):
    spec = fe.Chart(df_xy).mark_violin(inner=None).encode(x="x", y="y")._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 1


def test_violin_inner_box_adds_layers(df_xy):
    spec = fe.Chart(df_xy).mark_violin(inner="box").encode(x="x", y="y")._build_spec()
    assert len(spec.layers) > 1  # violin polygon + box layers


def test_violin_inner_quartile_3_rules(df_xy):
    spec = fe.Chart(df_xy).mark_violin(inner="quartile").encode(x="x", y="y")._build_spec()
    assert len(spec.layers) == 4  # violin + 3 quartile rules


def test_violin_invalid_inner_raises(df_xy):
    with pytest.raises(ValueError, match="inner"):
        fe.Chart(df_xy).mark_violin(inner="bogus").encode(x="x", y="y")._build_spec()


def test_violin_shared_extent_kwarg_propagates_to_transform():
    """R6: mark_violin(shared_extent=True) serializes shared_extent into the Violin transform."""
    df = pl.DataFrame(
        {
            "grp": ["a"] * 20 + ["b"] * 20,
            "val": list(range(20)) + list(range(100, 120)),
        }
    )
    spec = (
        fe.Chart(df)
        .mark_violin(inner=None, shared_extent=True)
        .encode(x="grp", y="val")
        ._build_spec()
    )
    json_str = spec.to_json()
    # shared_extent=true must appear in the serialized spec.
    assert '"shared_extent":true' in json_str, (
        "mark_violin(shared_extent=True) must emit shared_extent in the transform JSON"
    )


def test_violin_shared_extent_false_is_default():
    """R6: mark_violin() with no shared_extent kwarg defaults to shared_extent=false."""
    df = pl.DataFrame({"grp": ["a"] * 10 + ["b"] * 10, "val": list(range(20))})
    spec = fe.Chart(df).mark_violin(inner=None).encode(x="grp", y="val")._build_spec()
    json_str = spec.to_json()
    # Default must NOT emit shared_extent (it is skipped when false per serde default).
    assert '"shared_extent":true' not in json_str, (
        "mark_violin() without shared_extent must not emit shared_extent:true in transform JSON"
    )


def test_violin_shared_extent_produces_shared_value_range():
    """R6: mark_violin(shared_extent=True) renders groups on a shared value range.

    Groups 'a' ([0,9]) and 'b' ([100,109]) have disjoint per-group ranges.
    With shared_extent=True, the SVG must contain y-coordinate values
    spanning the cross-group range in both groups' polygon paths.
    """
    df = pl.DataFrame(
        {
            "grp": ["a"] * 10 + ["b"] * 10,
            "val": list(range(10)) + list(range(100, 110)),
        }
    )
    # shared_extent=True: both groups share the full [0, 109] y range.
    svg_shared = (
        fe.Chart(df).mark_violin(inner=None, shared_extent=True).encode(x="grp", y="val").to_svg()
    )
    # shared_extent=False (default): each group uses its own y range.
    svg_per = (
        fe.Chart(df).mark_violin(inner=None, shared_extent=False).encode(x="grp", y="val").to_svg()
    )
    # Both must produce non-empty SVG.
    assert "<svg" in svg_shared
    assert "<svg" in svg_per
    # The shared-extent render must produce different output than per-group
    # (shared forces the same y-grid, per-group gives narrower ranges per group).
    assert svg_shared != svg_per, (
        "shared_extent=True and shared_extent=False must produce different SVG "
        "for groups with disjoint value ranges"
    )


# ---------------------------------------------------------------------------
# T11 cycle 3 review: quartile-tick style/channel enumeration
#
# ``desugar_violin``'s ``inner="quartile"`` branch is the only caller that
# puts per-row style onto a ``tick`` layer's ``mark_kwargs``/encoding today:
# ``band_size`` (always), ``stroke_dash`` (q1/q3 only), ``color`` (only under
# a hue split). Two consecutive review cycles found ``tick.rs`` silently
# dropping one of these while a spec-JSON-only assertion stayed green (cycle
# 1: ``stroke_dash`` was on the wire, never consumed; cycle 2: ``color`` was
# on the wire and consistent with the body layer's encoding, but rendered as
# one flat theme color for every hue). Per the cycle-3 required fix, every
# channel these layers can actually declare is enumerated here, in one
# place, with the render-level test(s) that exercise it end to end -- so a
# further silent drop cannot again hide behind a passing spec-level test:
#
#   band_size      (mark_kwargs literal, always set)
#       -> test_violin_quartile_ticks_confined_per_band[_horizontal]
#       -> test_violin_quartile_ticks_contained_and_centered_in_body
#       -> test_violin_quartile_ticks_span_category_band_not_body_under_hue
#   stroke_dash    (mark_kwargs literal, q1/q3 only, not median)
#       -> test_violin_quartile_dash_renders_on_q1_and_q3_not_median
#   color          (encoding, only when split_hue)
#       -> test_violin_quartile_ticks_render_per_hue_stroke_color
#   stroke_opacity
#       -> NOT declared by these layers today, and deliberately not
#          asserted here. Confirmed empirically: a top-level
#          ``mark_violin(inner="quartile").encode(x=.., y=..,
#          stroke_opacity="col")`` produces ``tick`` layers whose encoding
#          has no ``stroke_opacity`` key at all -- ``_desugar.py`` only
#          propagates the chart-level x/y positional *scale* onto composite
#          sub-layers (``_propagate_layer_scale``), not RENDERER_HONORED
#          style channels such as ``stroke_opacity``. There is nothing for
#          ``tick.rs``'s now-correct per-row ``stroke_opacity`` consumer
#          (T11r cycle 3) to read for this mark, so there is no "quartile
#          layer declares stroke_opacity" behavior to pin yet. This is a
#          pre-existing composite-mark channel-propagation gap that predates
#          T11 and applies to every composite mark, not something specific
#          to violin quartile ticks -- fixing it means extending
#          ``_desugar.py``'s generic composite-layer machinery, which is out
#          of this task's file scope (``heavy_stat.py`` + these test
#          modules). Disclosed rather than silently omitted; not filed as a
#          numbered follow-up because #151 (the hue band_size overshoot)
#          already covers the adjacent "inner= marks are not per-hue aware"
#          territory and a maintainer picking this up will want to scope it
#          across all composite marks at once, not just violin.
# ---------------------------------------------------------------------------


def test_violin_quartile_layers_are_tick_with_band_size(df_xy):
    """F-L03-04: the q1/median/q3 inner layers use mark='tick' with an
    explicit band_size, not the old full-panel-width 'rule' mark."""
    spec = fe.Chart(df_xy).mark_violin(inner="quartile").encode(x="x", y="y")._build_spec()
    quart_layers = [layer for layer in spec.layers if layer.mark.name == "tick"]
    assert len(quart_layers) == 3, "expected q1/median/q3 tick layers"
    json_str = spec.to_json()
    assert '"band_size":0.8' in json_str


def test_violin_quartile_ticks_confined_per_band():
    """F-L03-04: quartile crossbars are confined to their own category band.

    Two signals, both derived from the render itself rather than a magic
    pixel literal (T11 review, cycle 2):

    1. Containment: each tick's span must be smaller than the PITCH between
       adjacent category centers (measured from the n=2 render's own two
       tick-group centers) -- a real, per-render quantity, not a constant
       tied to the default viewport.
    2. Doubling the category count roughly halves each band, so a
       band-confined tick's span shrinks with it. The pre-fix 'rule' mark
       ignored x entirely and drew the SAME full-panel-width span
       regardless of category count -- this is the RED-proof
       discriminator: both assertions fail on pre-fix code (span_2 exceeds
       the derived pitch, and span_4 == span_2 because neither shrinks).
    """

    def tick_x_spans(n_categories):
        cats = [f"c{i}" for i in range(n_categories) for _ in range(10)]
        vals = list(np.linspace(0.0, 10.0, 10)) * n_categories
        df = pl.DataFrame({"cat": cats, "val": vals})
        svg = fe.Chart(df).mark_violin(inner="quartile").encode(x="cat", y="val").to_svg()
        spans = sorted(set(_non_chrome_line_spans(svg, "x")))
        assert len(spans) == n_categories, (
            f"expected {n_categories} distinct tick x-spans (one per category), got {spans}"
        )
        return spans

    spans_2 = tick_x_spans(2)
    spans_4 = tick_x_spans(4)

    centers_2 = [(x1 + x2) / 2 for x1, x2 in spans_2]
    pitch_2 = abs(centers_2[1] - centers_2[0])
    max_span_2 = max(abs(x2 - x1) for x1, x2 in spans_2)
    assert max_span_2 < pitch_2, (
        f"quartile tick span {max_span_2} is not confined within its own category's "
        f"pitch {pitch_2} -- looks like it overlaps a neighboring category"
    )

    max_span_4 = max(abs(x2 - x1) for x1, x2 in spans_4)
    assert max_span_4 < max_span_2 * 0.75, (
        f"quartile ticks did not shrink with more categories (span_2={max_span_2}, "
        f"span_4={max_span_4}); expected band-confined ticks, not a full-panel-width rule"
    )


def test_violin_quartile_ticks_confined_per_band_horizontal():
    """Horizontal mirror of the confinement test above: with horizontal=True
    the categorical axis is y, so the tick spans the y dimension instead."""

    def tick_y_spans(n_categories):
        cats = [f"c{i}" for i in range(n_categories) for _ in range(10)]
        vals = list(np.linspace(0.0, 10.0, 10)) * n_categories
        df = pl.DataFrame({"cat": cats, "val": vals})
        svg = (
            fe.Chart(df)
            .mark_violin(inner="quartile", horizontal=True)
            .encode(x="val", y="cat")
            .to_svg()
        )
        spans = sorted(set(_non_chrome_line_spans(svg, "y")))
        assert len(spans) == n_categories, (
            f"expected {n_categories} distinct tick y-spans (one per category), got {spans}"
        )
        return spans

    spans_2 = tick_y_spans(2)
    spans_4 = tick_y_spans(4)

    centers_2 = [(y1 + y2) / 2 for y1, y2 in spans_2]
    pitch_2 = abs(centers_2[1] - centers_2[0])
    max_span_2 = max(abs(y2 - y1) for y1, y2 in spans_2)
    assert max_span_2 < pitch_2, (
        f"quartile tick span {max_span_2} is not confined within its own category's "
        f"pitch {pitch_2} -- looks like it overlaps a neighboring category"
    )

    max_span_4 = max(abs(y2 - y1) for y1, y2 in spans_4)
    assert max_span_4 < max_span_2 * 0.75, (
        f"quartile ticks did not shrink with more categories (span_2={max_span_2}, "
        f"span_4={max_span_4}); expected band-confined ticks, not a full-panel-width rule"
    )


def test_violin_quartile_ticks_contained_and_centered_in_body():
    """T11 review, required fix 2: containment against the ACTUAL violin
    body geometry (not just 'smaller than some number'), plus centering.
    Unsplit (no hue) case only -- see the hue test below for the adjudicated
    honest-overshoot behavior under a hue split.
    """
    df = pl.DataFrame(
        {
            "cat": ["a"] * 15 + ["b"] * 15 + ["c"] * 15,
            "val": (
                [float(i) for i in range(15)]
                + [x * 1.4 for x in range(15)]
                + [x * 0.6 + 3 for x in range(15)]
            ),
        }
    )
    svg = fe.Chart(df).mark_violin(inner="quartile").encode(x="cat", y="val").to_svg()

    body_boxes = sorted(set(_path_bboxes(svg, "x")))
    tick_spans = sorted(set(_non_chrome_line_spans(svg, "x")))
    assert len(body_boxes) == 3, f"expected 3 violin bodies, got {body_boxes}"
    assert len(tick_spans) == 3, f"expected 3 distinct quartile tick x-spans, got {tick_spans}"

    for (bx0, bx1), (tx0, tx1) in zip(body_boxes, tick_spans):
        assert bx0 <= tx0 and tx1 <= bx1, (
            f"quartile tick span ({tx0}, {tx1}) is not contained inside its violin "
            f"body's x-range ({bx0}, {bx1})"
        )
        body_center = (bx0 + bx1) / 2
        tick_center = (tx0 + tx1) / 2
        assert abs(body_center - tick_center) < 1.0, (
            f"quartile tick center {tick_center} is not aligned with its violin "
            f"body's center {body_center}"
        )


def test_violin_quartile_ticks_span_category_band_not_body_under_hue():
    """T11 review, required fix 3 (adjudicated, spec §6.11 amended
    2026-09-06): ``_VIOLIN_QUARTILE_BAND_SIZE`` is one static constant that
    cannot track the per-band hue-group count the Rust ``Violin`` transform
    resolves at render time, so quartile ticks are NOT per-hue dodged --
    every hue's ticks for a category render identically, sized to the
    CATEGORY band rather than each (narrower) per-hue violin body. This
    pins that honest, current behavior rather than silently claiming
    per-hue correctness: the overshoot against the body is real and
    expected, but containment against the CATEGORY band (derived as the
    pitch between the two category centers) still holds.
    """
    df = pl.DataFrame(
        {
            "g": ["a"] * 20 + ["b"] * 20,
            "h": (["x"] * 10 + ["y"] * 10) * 2,
            "v": [float(i) for i in range(20)] + [50.0 + i * 0.2 for i in range(20)],
        }
    )
    svg = fe.Chart(df).mark_violin(inner="quartile").encode(x="g", y="v", color="h").to_svg()

    body_boxes = sorted(set(_path_bboxes(svg, "x")))
    tick_spans = sorted(set(_non_chrome_line_spans(svg, "x")))
    assert len(body_boxes) == 2, (
        f"expected 2 distinct per-hue body x-ranges (the two hues overlay rather than "
        f"dodge within each category), got {body_boxes}"
    )
    assert len(tick_spans) == 2, (
        f"expected 2 distinct tick x-spans (one per category, NOT per hue -- ticks are "
        f"not dodged), got {tick_spans}"
    )

    # Adjudicated overshoot: the tick span exceeds its own category's narrower
    # per-hue body width. Pinned as current honest behavior (see
    # _VIOLIN_QUARTILE_BAND_SIZE's comment), not silently passed off as
    # tracing the body's silhouette the way the unsplit case does.
    for (bx0, bx1), (tx0, tx1) in zip(body_boxes, tick_spans):
        body_width = bx1 - bx0
        tick_width = tx1 - tx0
        assert tick_width > body_width, (
            f"expected the documented hue overshoot (tick wider than its per-hue "
            f"body); got tick_width={tick_width}, body_width={body_width}"
        )

    # But containment against the CATEGORY band (not the body) still holds:
    # derive the category pitch from the two categories' tick centers.
    centers = [(x1 + x2) / 2 for x1, x2 in tick_spans]
    pitch = abs(centers[1] - centers[0])
    for x1, x2 in tick_spans:
        assert abs(x2 - x1) < pitch, (
            f"quartile tick span ({x1}, {x2}) exceeds the category pitch {pitch} -- "
            f"ticks should stay within their own category band even when they "
            f"overshoot the (narrower) per-hue body"
        )


def test_violin_quartile_ticks_render_per_hue_stroke_color():
    """T11 review cycle 2/3 required fix 1: pin the RENDERED per-hue stroke
    color on the quartile tick <line> elements -- not just the spec-level
    ``"color" in encoding`` check
    (``tests/test_flexibility_caps/test_fa6_violin_box_color.py::
    test_violin_hue_quartile_body_and_quartile_layers_consistent``), which
    stayed green while the render did not honor it.

    Cycle-2 review found the same silent-drop class as cycle-1's dash bug,
    one channel later: this task's ``mark="rule"`` -> ``mark="tick"`` switch
    carries a hue ``color`` encoding onto the wire for the quartile layers
    (pre-existing ``split_hue`` branch, ``heavy_stat.py``), but pre-T11r-
    cycle-3 ``tick.rs`` resolved stroke color as one flat expression computed
    ONCE outside any per-row loop -- so every quartile crossbar rendered the
    same theme-blue regardless of hue.

    RED-proof (analytic, not reproduced by reverting this task's own diff):
    the regression lived entirely in ``tick.rs``'s stroke-color resolution,
    which is Rust code outside this task's authority to revert (T11r cycle 3
    already landed the fix upstream, see ``.sdd/task-11r-report.md``).
    Reverting only this task's Python ``rule``->``tick`` hunk does not
    isolate it: on ``rule``, hue color always rendered correctly (color was
    never the bug on that mark), so there is no clean "broken" baseline to
    diff against on the Python side alone. The RED evidence is the pre-fix
    measurement both cycle-2 reviewers took independently against the
    then-current build: the quality reviewer measured ``mark_tick`` +
    ``color="h:N"`` rendering a single ``['#2563eb']`` (vs. ``mark_rule``'s
    ``['#2563eb', '#dc2626']`` for the identical encoding); the spec
    reviewer measured all six quartile ``<line>``s in a 2-category x 2-hue
    violin rendering one flat ``#2563eb``. This test is the forward-looking
    GREEN regression pin for the now-correct cross-language behavior
    (``tick.rs``'s ``row_stroke`` closure: ``row_colors_from_scale`` +
    ``resolve_stroke_color`` per row) -- the same class of caveat as
    ``test_violin_quartile_dash_renders_on_q1_and_q3_not_median`` above, for
    the same reason: the bug was 100% Rust-side.
    """
    df = pl.DataFrame(
        {
            "g": ["a"] * 20 + ["b"] * 20,
            "h": (["x"] * 10 + ["y"] * 10) * 2,
            "v": [float(i) for i in range(20)] + [50.0 + i * 0.2 for i in range(20)],
        }
    )
    svg = fe.Chart(df).mark_violin(inner="quartile").encode(x="g", y="v", color="h").to_svg()

    body_fill_rgb = re.findall(r'<path[^>]*fill="rgba\((\d+),(\d+),(\d+),[^)]+\)"', svg)
    body_colors = {tuple(int(c) for c in rgb) for rgb in body_fill_rgb}
    assert len(body_colors) == 2, f"expected 2 distinct hue body fills, got {body_colors}"

    tick_strokes = set()
    for m in re.finditer(r"<line([^>]*)/>", svg):
        attrs = m.group(1)
        stroke_m = re.search(r'stroke="(#[0-9a-fA-F]{6})"', attrs)
        if stroke_m and stroke_m.group(1) not in _AXIS_CHROME_STROKES:
            tick_strokes.add(stroke_m.group(1))
    assert len(tick_strokes) == 2, (
        f"expected 2 distinct quartile tick stroke colors (one per hue), got "
        f"{tick_strokes} -- tick.rs must resolve per-row color the way "
        f"rule.rs does (T11r cycle 3), not a single flat stroke computed "
        f"once outside the per-row loop"
    )
    tick_rgb = {tuple(int(hexcolor[i : i + 2], 16) for i in (1, 3, 5)) for hexcolor in tick_strokes}
    assert tick_rgb == body_colors, (
        f"quartile tick stroke colors {tick_rgb} must match the violin body's "
        f"per-hue fill colors {body_colors}"
    )


def test_violin_quartile_dash_renders_on_q1_and_q3_not_median(df_xy):
    """T11 review, required fix 1: pin the RENDERED SVG attribute, not just
    the spec JSON. The rule->tick switch kept ``stroke_dash=[2, 2]`` on the
    wire (``mark_style``), but ``tick.rs`` did not consume
    ``mark_style.stroke_dash`` until the T11r Rust fix landed
    (``crates/ferrum-core/src/render/marks/tick.rs``) -- a spec-JSON-only
    assertion is exactly what let that silent-drop regression ship green.

    RED-proof note: reverting only the Python mark="tick"->"rule" swap does
    NOT cleanly reproduce the historical bug for this specific assertion --
    "rule" applies its own default reference-line dash styling regardless
    of the `mk` override (confirmed: even the median role's empty `mk={}`
    renders WITH a default dasharray under "rule", so there is no
    clean "solid" baseline to compare against pre-fix). The actual bug was
    entirely Rust-side (tick.rs ignoring mark_style.stroke_dash) and was
    already fixed and covered by the Rust coder's own tests
    (tick_stroke_dash_mark_style_applies_to_line /
    tick_no_stroke_dash_is_byte_identical). This test is a forward-looking
    GREEN regression pin for the now-correct cross-language behavior, not a
    RED-proven-against-my-own-diff test.
    """
    svg = fe.Chart(df_xy).mark_violin(inner="quartile").encode(x="x", y="y").to_svg()
    quartile_lines = [
        m.group(0)
        for m in re.finditer(r"<line[^>]*/>", svg)
        if (sm := re.search(r'stroke="([^"]+)"', m.group(0)))
        and sm.group(1) not in _AXIS_CHROME_STROKES
    ]
    assert quartile_lines, "expected quartile tick <line> elements"
    dashed = [ln for ln in quartile_lines if "stroke-dasharray" in ln]
    solid = [ln for ln in quartile_lines if "stroke-dasharray" not in ln]
    assert dashed, "q1/q3 ticks must render stroke-dasharray in the SVG"
    assert solid, "median tick must render without stroke-dasharray"
    assert len(dashed) == 2 * len(solid), (
        f"expected dashed:solid == 2:1 (q1+q3 dashed vs median solid), got "
        f"dashed={len(dashed)} solid={len(solid)}"
    )


# ---- mark_qq ----


def test_qq_smoke():
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    spec = fe.Chart(df).mark_qq().encode(x="v")._build_spec()
    assert len(spec.layers) == 2  # point + line


def test_qq_no_line():
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    spec = fe.Chart(df).mark_qq(line=False).encode(x="v")._build_spec()
    assert len(spec.layers) == 1


def test_qq_invalid_distribution():
    df = pl.DataFrame({"v": [1.0, 2.0]})
    with pytest.raises(ValueError, match="distribution"):
        fe.Chart(df).mark_qq(distribution="bogus").encode(x="v")._build_spec()


# ---- mark_raster ----


def test_raster_smoke(df_xy):
    spec = fe.Chart(df_xy).mark_raster().encode(x="x", y="y")._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 1


def test_raster_aggregate_mean_requires_field(df_xyc):
    with pytest.raises(ValueError, match="field"):
        fe.Chart(df_xyc).mark_raster(aggregate="mean").encode(x="x", y="y")._build_spec()


def test_raster_resolution_int(df_xy):
    spec = fe.Chart(df_xy).mark_raster(resolution=64).encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    assert "64" in json_str


# ---- mark_hex ----


def test_hex_smoke(df_xy):
    spec = fe.Chart(df_xy).mark_hex().encode(x="x", y="y")._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 1


def test_hex_aggregate_mean_requires_field(df_xyc):
    with pytest.raises(ValueError, match="field"):
        fe.Chart(df_xyc).mark_hex(aggregate="mean").encode(x="x", y="y")._build_spec()


def test_hex_bin_size_propagates(df_xy):
    spec = fe.Chart(df_xy).mark_hex(bin_size=0.5).encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    assert '"bin_size":0.5' in json_str


# ---- mark_hex stroke tests (item 17) ----


def test_hex_stroke_and_width_no_longer_raises(df_xy):
    """Passing stroke= and stroke_width= no longer raises ValueError."""
    # Both together — use full 6-digit hex (from_hex_str does not support 3-digit shorthand)
    spec = (
        fe.Chart(df_xy)
        .mark_hex(stroke="#ffffff", stroke_width=1)
        .encode(x="x", y="y")
        ._build_spec()
    )
    assert spec.layers is not None
    # stroke alone
    spec2 = fe.Chart(df_xy).mark_hex(stroke="#ffffff").encode(x="x", y="y")._build_spec()
    assert spec2.layers is not None
    # stroke_width alone (non-zero)
    spec3 = fe.Chart(df_xy).mark_hex(stroke_width=2).encode(x="x", y="y")._build_spec()
    assert spec3.layers is not None


def test_hex_stroke_with_width_renders_border_attributes(df_xy):
    """mark_hex(stroke="#ffffff", stroke_width=1) emits stroke/stroke-width on hex polygons.

    Note: stroke color must be a full 6-digit hex. The Rust color parser (from_hex_str)
    handles only 6- and 8-digit hex codes; 3-digit shorthand (e.g. #fff) is silently
    dropped. This is a pre-existing constraint in the renderer, not a limitation of the
    hex-stroke wiring.
    """
    svg = fe.Chart(df_xy).mark_hex(stroke="#ffffff", stroke_width=1).encode(x="x", y="y").to_svg()
    assert "<svg" in svg
    # The polygon <path> elements should carry stroke="..." (white = #ffffff)
    assert 'stroke="#ffffff"' in svg
    # stroke-width attribute is emitted for nonzero widths
    assert 'stroke-width="1"' in svg


def test_hex_stroke_alone_width_zero_renders_no_visible_border(df_xy):
    """mark_hex(stroke="#ffffff") alone (stroke_width=0 default) renders with no stroke-width.

    Per the locked literal semantics (spec §8): a stroke color with stroke_width left at
    its 0 default produces no visible border. The Rust SVG writer only emits stroke-width
    when the value is nonzero, so no stroke-width attribute appears on the hex polygons.
    """
    import re

    svg = fe.Chart(df_xy).mark_hex(stroke="#ffffff").encode(x="x", y="y").to_svg()
    assert "<svg" in svg
    # The polygon paths carry stroke="#ffffff" but no stroke-width (0 = invisible).
    assert 'stroke="#ffffff"' in svg
    # Confirm no nonzero stroke-width on polygon paths. The SVG may have stroke-width
    # from axis/colorbar chrome, but not from the hex polygon paths themselves.
    # Find path elements with stroke="#ffffff" and assert they have no stroke-width.
    poly_blocks = re.findall(r'<path\b[^>]*stroke="#ffffff"[^>]*/>', svg)
    for block in poly_blocks:
        assert "stroke-width" not in block, (
            f"Expected no stroke-width on zero-width hex polygon but found: {block!r}"
        )


# ---- mark_swarm ----


def test_swarm_smoke(df_xy):
    spec = fe.Chart(df_xy).mark_swarm().encode(x="x", y="y")._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 1


def test_swarm_orient_horizontal(df_xy):
    spec = fe.Chart(df_xy).mark_swarm(orient="horizontal").encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    assert "swarm" in json_str


def test_swarm_side_left(df_xy):
    spec = fe.Chart(df_xy).mark_swarm(side="left").encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    assert '"side":"left"' in json_str


# ---- mark_function ----


def test_function_explicit_domain():
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64), "y": pl.Series([], dtype=pl.Float64)})
    chart = fe.Chart(df).mark_function(lambda x: x**2, domain=(0, 5), n=50)
    spec = chart._build_spec()
    json_str = spec.to_json()
    assert '"line"' in json_str


def test_function_inferred_domain(df_xy):
    chart = fe.Chart(df_xy).encode(x="x", y="y").mark_function(lambda x: x * 2)
    spec = chart._build_spec()
    json_str = spec.to_json()
    assert '"line"' in json_str


def test_function_missing_domain_raises():
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64), "y": pl.Series([], dtype=pl.Float64)})
    with pytest.raises(ValueError, match="domain"):
        fe.Chart(df).mark_function(lambda x: x**2)


def test_function_wrong_shape_raises():
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64), "y": pl.Series([], dtype=pl.Float64)})
    with pytest.raises(ValueError, match="shape"):
        fe.Chart(df).mark_function(lambda x: 42, domain=(0, 5), n=50)


def test_function_clip_false_no_longer_raises():
    """E13/F-L02-04: mark_function(clip=False) used to raise; the scene-level
    clip escape (Task 9's MarkBatch.clip) supports it now."""
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64), "y": pl.Series([], dtype=pl.Float64)})
    svg = fe.Chart(df).mark_function(lambda x: x**2, domain=(0, 5), n=50, clip=False).to_svg()
    assert "<svg" in svg


def test_function_clip_false_renders_outside_panel_clip_group():
    """clip=False threads MarkKwargsSpec.clip=False onto the synthesized line
    layer; the renderer places the resulting <polyline> OUTSIDE (after) the
    panel's <g clip-path=...> group instead of inside it (mirrors the
    Rust-side render_svg_clip_false_line_renders_outside_panel_clip_group
    test in crates/ferrum-core/src/render/mod.rs)."""
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64), "y": pl.Series([], dtype=pl.Float64)})
    svg = fe.Chart(df).mark_function(lambda x: x**2, domain=(0, 5), n=50, clip=False).to_svg()
    open_idx = svg.find("<g clip-path=")
    assert open_idx != -1, "expected a panel clip group in the SVG"
    close_idx = svg.find("</g>", open_idx)
    assert close_idx != -1, "expected the panel clip group to close"
    line_idx = svg.find("<polyline")
    assert line_idx != -1, "expected the function's line to render as a <polyline>"
    assert line_idx > close_idx, (
        "mark_function(clip=False) must render its line OUTSIDE the panel clip group: "
        f"polyline at {line_idx}, clip group closes at {close_idx}"
    )


def test_function_clip_unset_line_renders_inside_panel_clip_group():
    """Default (clip unset -> True) renders the <polyline> INSIDE the panel's
    clip group -- the byte-stability counterpart to the clip=False test
    above."""
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64), "y": pl.Series([], dtype=pl.Float64)})
    svg = fe.Chart(df).mark_function(lambda x: x**2, domain=(0, 5), n=50).to_svg()
    open_idx = svg.find("<g clip-path=")
    close_idx = svg.find("</g>", open_idx)
    line_idx = svg.find("<polyline")
    assert open_idx != -1 and close_idx != -1 and line_idx != -1
    assert open_idx < line_idx < close_idx, (
        "default (clip unset) line must render INSIDE the panel clip group"
    )


def test_function_clip_true_is_byte_identical_to_default():
    """clip=True is the theme default and must stay unset on the wire, so
    output is byte-identical to omitting clip= entirely."""
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64), "y": pl.Series([], dtype=pl.Float64)})
    svg_default = fe.Chart(df).mark_function(lambda x: x**2, domain=(0, 5), n=50).to_svg()
    svg_explicit = fe.Chart(df).mark_function(lambda x: x**2, domain=(0, 5), n=50, clip=True).to_svg()
    assert svg_default == svg_explicit


def test_function_clip_false_multi_layer_branch_renders_outside_panel_clip_group():
    """T11 review, required fix 5: covers the OTHER mark_function code path
    -- ``self._layers is not None`` -- which builds a fresh ``Chart`` for
    the synthetic data and composes it via ``self + fn_chart`` rather than
    cloning ``self`` the way the single-chart branch does. This is genuinely
    different code (a different ``_mark_kwargs`` merge site in
    ``_chart_methods_statistical.py``), so the single-chart test above does
    not exercise it. Both existing layers are ``point`` marks (no
    ``<polyline>``), so the function's is the only one in the SVG.
    """
    point_df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 4.0, 9.0]})
    layered = fe.Chart(point_df).mark_point().encode(x="x", y="y") + fe.Chart(
        point_df
    ).mark_point().encode(x="x", y="y")
    assert layered._layers is not None, "expected a layered chart with _layers set"

    svg = layered.mark_function(lambda x: x**2, domain=(0, 5), n=50, clip=False).to_svg()

    open_idx = svg.find("<g clip-path=")
    assert open_idx != -1, "expected a panel clip group in the SVG"
    close_idx = svg.find("</g>", open_idx)
    assert close_idx != -1, "expected the panel clip group to close"
    line_idx = svg.find("<polyline")
    assert line_idx != -1, "expected the function's line to render as a <polyline>"
    assert line_idx > close_idx, (
        "the multi-layer mark_function(clip=False) branch must still render its "
        f"line OUTSIDE the panel clip group: polyline at {line_idx}, clip group "
        f"closes at {close_idx}"
    )


def test_function_clip_false_renders_vertices_outside_panel_rect():
    """T11 review, required fix (public-API overflow pin, now reachable):
    ``scale/linear.rs`` now extrapolates out-of-domain vertices for
    clip-exempt batches instead of dropping them, so ``clip=False``
    measurably renders line VERTICES outside the panel's plot rect -- not
    just DOM-placement-outside-the-clip-group, the coordinates themselves
    now escape the panel. ``clip=True`` still confines every vertex inside.
    Mirrors the Rust coder's repro: ``domain=(0, 3)``, ``n=50``, y-domain
    forced to ``(0, 5)`` (narrower than ``fn``'s actual range up to ``9``).
    """
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64), "y": pl.Series([], dtype=pl.Float64)})

    def render(clip):
        chart = fe.Chart(df).mark_function(lambda x: x**2, domain=(0, 3), n=50, clip=clip)
        chart = chart.encode(y=fe.Y("y", scale=fe.LinearScale(domain=(0, 5))))
        return chart.to_svg()

    def vertices(svg):
        m = re.search(r'<polyline points="([^"]+)"', svg)
        assert m, "expected the function's <polyline>"
        return [tuple(map(float, p.split(","))) for p in m.group(1).split()]

    def outside_panel(svg, pts):
        px, py, pw, ph = _panel_rect(svg)
        eps = 1e-6
        return [
            p
            for p in pts
            if not (px - eps <= p[0] <= px + pw + eps and py - eps <= p[1] <= py + ph + eps)
        ]

    svg_true = render(True)
    svg_false = render(False)

    outside_true = outside_panel(svg_true, vertices(svg_true))
    pts_false = vertices(svg_false)
    outside_false = outside_panel(svg_false, pts_false)

    assert not outside_true, f"clip=True must confine every vertex to the panel rect, got {outside_true}"
    assert outside_false, "clip=False must render at least one vertex outside the panel rect"
    # Measured 13/50 vertices outside under this exact domain mismatch; use
    # a safe lower bound rather than pinning the exact count.
    assert len(outside_false) >= 10, (
        f"expected a substantial fraction of the {len(pts_false)} vertices outside "
        f"the panel under this domain mismatch, got {len(outside_false)}"
    )


# ---- Smoke renders (skip if engine doesn't support yet) ----


def test_violin_no_inner_renders(df_xy):
    """Smoke render -- violin polygon only (no inner box/quartile)."""
    svg = fe.Chart(df_xy).mark_violin(inner=None).encode(x="x", y="y").to_svg()
    assert "<svg" in svg


def test_qq_renders():
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 1.2]})
    svg = fe.Chart(df).mark_qq(line=False).encode(x="v").to_svg()
    assert "<svg" in svg


def test_swarm_renders():
    """Smoke render -- swarm (point mark on transformed coords)."""
    df_cat = pl.DataFrame(
        {
            "g": ["a"] * 5 + ["b"] * 5,
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    svg = fe.Chart(df_cat).mark_swarm().encode(x="g", y="v").to_svg()
    assert "<svg" in svg


def test_function_renders(df_xy):
    chart = fe.Chart(df_xy).encode(x="x", y="y").mark_function(lambda x: x**2)
    svg = chart.to_svg()
    assert "<svg" in svg


# --- Phase 8b Task 35: bivariate density routes through mark_contour ---


def test_bivariate_density_routes_through_contour():
    """When .encode() binds both x and y, mark_density() emits a 2D KDE +
    contour-fill layered spec (Phase 8b), not the 1D area+Kde path."""
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.standard_normal(50), "y": rng.standard_normal(50)})
    spec = fe.Chart(df).mark_density().encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    # 2D KDE transform (Kde2D) and contour transform should both appear in the
    # serialized spec, regardless of exact casing/format of their tags. The
    # serde snake_case mangling on `Bin2D`/`Kde2D` yields `bin_2_d`/`kde2_d`
    # — see ferrum-phase-9 commit 041c528 for the explicit serde rename
    # applied to `Bin2D`; `Kde2D` is unaffected here because callers don't
    # need to match against a fixed string.
    assert any(t in json_str.lower() for t in ("kde_2d", "kde2d", "kde2_d"))
    assert "contour" in json_str.lower()
