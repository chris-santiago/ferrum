"""Regression tests for GH #77 — horizontal composite-mark Stack value axis.

``mark_histogram``/``mark_density`` with ``orient="horizontal"`` remap the
value channel onto ``encoding.x`` directly (``desugar_histogram``/
``desugar_density``'s ``encoding_remap``) without setting ``CoordFlip``, so
Rust's ``apply_stack`` (``crates/ferrum-core/src/render/position.rs``) used
to guess the wrong value axis from ``coord_flipped`` alone (always ``False``
for this path): ``mark_histogram(orient="horizontal", multiple="stack")``
raised ``ValueError`` for every input, and the ``mark_density`` equivalent
silently produced corrupted geometry.

The fix threads an explicit ``value_axis: "x" | "y" | None`` through the
``Stack`` position adjustment (``src/ferrum/position.py``), set by
``desugar_histogram``/``desugar_density`` (``src/ferrum/marks/statistical.py``)
whenever ``orientation == "horizontal"``. The Rust wire only emits the key
when non-``None`` (``skip_serializing_if``), so every pre-#77 vertical Stack
spec stays byte-identical.

Sibling-sweep disposition (every ``Stack(`` construction under
``src/ferrum/``, see the GH #77 fix report):

* ``marks/statistical.py`` (``desugar_density``, ``desugar_histogram``) —
  fixed here (the orientation-swapping producers).
* ``marks/diagnostic/_classification.py``
  (``desugar_class_prediction_error``) — vertical-only, ``x`` is always the
  fixed ``"actual"`` category field; not orientation-swappable.
* ``plots/distribution.py`` (``_multiple_to_position``, used by
  ``displot`` only) — vertical-only; ``catplot``'s ``orient="h"`` routes
  through real ``CoordFlip`` (the pre-existing ``coord_flipped`` fallback),
  never through this helper.

Known, deliberately-xfail'd divergence discovered while writing these tests
(NOT part of GH #77's scope — see ``test_vertical_stacked_histogram_unchanged_and_correct``):
the vertical x2-binned bar drawer (``render::marks::bar::build_quantitative``,
used for a plain vertical ``mark_bar`` with ``x``/``x2`` bin edges) ignores
``__stack_y_base__`` entirely and always draws every segment from the panel
baseline, so stacked segments in the same bin fully overlap instead of
tiling. This is a pre-existing, separate gap (not introduced by GH #77, and
not touched by the GH #77 Rust diff) — not fixed here.

A second, broader pre-existing gap surfaced incidentally while first writing
these tests, and was FIXED in the same sweep (before this file's tests were
finalized): Rust's stacked-domain widening (``position::axis_batch_for_y``)
used to widen only the **Y**-labeled scale slot to the post-stack cumulative
total, with no ``axis_batch_for_x`` counterpart. Whenever the stacked
*value* lived on X — every horizontal composite-mark Stack, and also real
``CoordFlip`` + Stack (``axis_batch_for_y`` matches on the **pre-flip** Y
field, which is the *category* once swapped, so the old Y-only widening
never applied there either) — the X-axis domain used to resolve from the
pre-stack per-group values only, and any row whose stacked cumulative value
exceeded that narrow domain mapped through ``LinearScaleData::scale`` to
``NaN`` (out-of-domain returns ``NaN``, not an extrapolated pixel, when
``clamp=false`` — the auto-resolved default) and got silently dropped by
every mark drawer's per-row loop (``to_pixel_f64() -> None`` => ``continue``
=> skip this row). ``position::axis_batch_for_x`` (mirroring
``axis_batch_for_y``, both now gated through the shared
``resolve_stack_value_on_x(value_axis, coord_flipped)`` triad instead of a
per-axis guess) closes this: the auto-domain case is now widened correctly,
confirmed end-to-end below by
``test_horizontal_stacked_histogram_auto_domain_tiles_along_x``.

Several tests below still route the value channel's scale through an
explicit ``scale={"domain": [...]}`` (propagated into the composite-mark's
remapped channel via the existing chart-level scale-propagation path,
``_desugar.py::_propagate_positional_scale`` — a supported mechanism, not a
private-API workaround). Now that the widening gap above is closed, these
explicit-domain variants are no longer load-bearing for correctness, but
they remain useful as widening-independent geometry checks: they pin the
GH #77 value_axis wiring + ``apply_stack``/mark-drawer tiling formula in
isolation from ``axis_batch_for_x``/``axis_batch_for_y``'s domain-inference
logic, so a future regression in either subsystem is attributable to the
subsystem that actually broke.
"""

from __future__ import annotations

import json
import re

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding import X, Y
from ferrum.position import Stack

_THEME_BACKGROUND = "#faf7f2"


# ---------------------------------------------------------------------------
# Helpers (SVG rect/path parsing conventions mirror
# tests/test_bug_hunt_band_point_range.py's _svg_elements/_data_bar_rects)
# ---------------------------------------------------------------------------


def _svg_rects(svg: str) -> list[dict[str, str]]:
    out = []
    for m in re.finditer(r"<rect([^/]+)/>", svg):
        out.append(dict(re.findall(r'([\w-]+)="([^"]+)"', m.group(1))))
    return out


def _data_rects(svg: str) -> list[dict[str, float | str]]:
    """Filled, non-background ``<rect>`` elements (excludes the SVG
    background and the unfilled plot-area rect), with numeric attrs parsed.
    """
    out = []
    for attrs in _svg_rects(svg):
        fill = attrs.get("fill")
        if not fill or fill == _THEME_BACKGROUND or fill.startswith("url("):
            continue
        out.append(
            {
                **attrs,
                "x": float(attrs["x"]),
                "y": float(attrs["y"]),
                "width": float(attrs["width"]),
                "height": float(attrs["height"]),
            }
        )
    return out


def _group_rects_by_bin(rects: list[dict], *, axis: str) -> dict[tuple[float, float], list[dict]]:
    """Group rects sharing the same cross-``axis`` span (i.e. the same bin —
    same row/column, different stacked segments)."""
    cross_pos_key, cross_size_key = ("y", "height") if axis == "x" else ("x", "width")
    groups: dict[tuple[float, float], list[dict]] = {}
    for r in rects:
        key = (round(r[cross_pos_key], 1), round(r[cross_size_key], 1))
        groups.setdefault(key, []).append(r)
    return groups


def _assert_tiles_without_overlap(rects: list[dict], *, axis: str, eps: float = 0.5) -> None:
    """Group same-bin rects (sharing the cross-axis span) and assert their
    along-``axis`` spans are disjoint and contiguous (base-to-value
    stacking): each segment starts exactly where the previous one ends.
    """
    pos_key, size_key = ("x", "width") if axis == "x" else ("y", "height")
    groups = _group_rects_by_bin(rects, axis=axis)

    assert groups, "no data rects to check"
    for key, segs in groups.items():
        segs = sorted(segs, key=lambda r: r[pos_key])
        for i in range(len(segs) - 1):
            end_i = segs[i][pos_key] + segs[i][size_key]
            start_next = segs[i + 1][pos_key]
            assert abs(end_i - start_next) < eps, (
                f"segments in bin {key} are not contiguous (axis={axis!r}): "
                f"segment {i} spans [{segs[i][pos_key]}, {end_i}], "
                f"segment {i + 1} starts at {start_next}"
            )


def _area_paths_in_draw_order(svg: str) -> list[list[tuple[float, float]]]:
    """Area-fill ``<path>`` point lists, in draw order (== group order:
    first-appearance of the ``by``/color field). Fill is either an opaque
    hex color (stacked areas) or a translucent ``rgba(...)`` (layered
    areas) — matched loosely so both modes parse the same way.
    """
    out: list[list[tuple[float, float]]] = []
    for m in re.finditer(r'<path[^>]*\sd="([^"]+)"[^>]*fill="([^"]+)"', svg):
        d, fill = m.group(1), m.group(2)
        if fill == _THEME_BACKGROUND or fill.startswith("url("):
            continue
        out.append([(float(x), float(y)) for x, y in re.findall(r"(-?[\d.]+)[ ,]+(-?[\d.]+)", d)])
    return out


# ---------------------------------------------------------------------------
# Test 1 — mark_histogram(orient="horizontal", multiple="stack")
# ---------------------------------------------------------------------------


def test_horizontal_stacked_histogram_renders_and_stacks_along_x():
    """GH #77's literal repro: RED pre-fix was ``ValueError: Stack: x column
    must be Float64 or Utf8`` for every input (Rust guessed bin_start, a
    Float64 category column, as the value column since coord_flipped was
    always False on this path). Post-fix: renders, and same-bin segments
    tile base-to-value without overlap.

    This variant pins the value_axis wiring + tiling geometry via an
    explicit value-axis scale domain (propagated via
    ``_propagate_positional_scale``), isolated from
    ``axis_batch_for_x``'s auto-domain inference (see module docstring).
    ``test_horizontal_stacked_histogram_auto_domain_tiles_along_x`` below
    covers the auto-domain, end-to-end case.
    """
    # Part A: the literal bug-report shape — auto domain, must not raise.
    bug_report_df = pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.5, 3.5],
            "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )
    svg = (
        fm.Chart(bug_report_df)
        .mark_histogram(orient="horizontal", multiple="stack")
        .encode(y="v", color="g")
        .to_svg()
    )
    assert "<svg" in svg

    # Part B: tiling geometry, with an explicit value-axis scale domain so
    # the assertion isolates the GH #77 geometry fix (value_axis wiring +
    # apply_stack/mark-drawer tiling formula) from axis_batch_for_x's
    # separate domain-inference logic.
    df = pl.DataFrame(
        {
            "v": [0.5, 0.5, 0.5, 1.5, 1.5],
            "g": ["a", "a", "b", "a", "b"],
        }
    )
    svg = (
        fm.Chart(df)
        .mark_histogram(orient="horizontal", multiple="stack", bin_count=2, extent=(0.0, 2.0))
        .encode(x=X("_scale_domain_only", scale={"domain": [0, 4]}), y="v", color="g")
        .to_svg()
    )
    rects = _data_rects(svg)
    assert len(rects) == 4, f"expected 4 bar segments (2 bins x 2 groups); got {len(rects)}"
    _assert_tiles_without_overlap(rects, axis="x")

    # Every bin's first segment must start at the plot-area left edge (base
    # of the stack is 0), not some arbitrary offset.
    plot_area_x = _svg_rects(svg)[1]["x"]
    starts = sorted({r["x"] for r in rects})
    assert abs(min(starts) - float(plot_area_x)) < 0.5, (
        f"first stacked segment must start at the plot-area left edge "
        f"({plot_area_x}); got {min(starts)}"
    )


def test_horizontal_stacked_histogram_auto_domain_tiles_along_x():
    """End-to-end coverage for ``position::axis_batch_for_x`` (the GH #77
    follow-up X-domain widening, mirroring ``axis_batch_for_y``, gated by
    the shared ``resolve_stack_value_on_x`` triad): with NO explicit
    ``scale=`` override, the value channel's auto-inferred domain must
    still widen to the post-stack cumulative total, so every group's
    segments render (not just the first, pre-widening-fix group) and tile
    base-to-value without overlap. Reuses the bug report's literal dataset
    from Part A above — pre-widening-fix this rendered only the first
    group's bars (the cumulative "b" segments fell outside the narrow
    auto-inferred domain and were silently dropped, one per bin).
    """
    df = pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.5, 3.5],
            "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )
    svg = (
        fm.Chart(df)
        .mark_histogram(orient="horizontal", multiple="stack")
        .encode(y="v", color="g")
        .to_svg()
    )
    rects = _data_rects(svg)
    fills = {r["fill"] for r in rects}
    assert len(fills) == 2, f"expected both groups' fills present; got {fills}"
    assert len(rects) == 6, f"expected 6 bar segments (3 bins x 2 groups); got {len(rects)}"
    _assert_tiles_without_overlap(rects, axis="x")

    # Every bin's first segment must start at the SAME x (base of the stack
    # is 0 in every bin) — a consistent base-alignment check that doesn't
    # pin the exact plot-area edge, since the auto-inferred domain applies
    # its own (unrelated to stacking) expand/nice padding.
    bins = _group_rects_by_bin(rects, axis="x")
    first_starts = {round(min(r["x"] for r in segs), 1) for segs in bins.values()}
    assert len(first_starts) == 1, (
        f"every bin's first segment must start at the same x; got {first_starts}"
    )


# ---------------------------------------------------------------------------
# Test 2 — mark_density(orient="horizontal", multiple="stack")
# ---------------------------------------------------------------------------


def test_horizontal_stacked_density_stacks_not_overlays():
    """Pre-fix this silently corrupted geometry rather than raising (Rust
    treated the density curve's own value column as the grouping category
    and vice versa). Post-fix: ``multiple="stack"`` must differ from
    ``multiple="layer"``, and the later-stacked group's area base must be
    offset from the panel-left baseline for every point (unlike ``"a"``,
    the first group in stack order, whose base is always exactly the
    baseline — offset 0).

    Density has no analog of the histogram's ``X(scale=...)`` domain-
    propagation workaround (setting a chart-level ``x=`` alongside ``y=``
    for a 1D density trips ``desugar_density``'s bivariate-KDE routing
    check), so this test exercises the plain auto-domain path — now that
    ``axis_batch_for_x`` widens the auto-inferred domain to the post-stack
    cumulative total (see module docstring), both curves render in full
    (1024 points each; pre-widening-fix only ~206/1024 of the second
    curve's points survived the narrow domain), so the assertion below
    checks every point rather than "at least some".
    """
    df = pl.DataFrame(
        {
            "v": ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.5, 3.5] * 3),
            "g": ((["a"] * 4 + ["b"] * 4) * 3),
        }
    )

    def _build(multiple: str) -> str:
        return (
            fm.Chart(df)
            .mark_density(orient="horizontal", multiple=multiple, bandwidth=0.6)
            .encode(y="v", color="g")
            .to_svg()
        )

    svg_stack = _build("stack")
    svg_layer = _build("layer")
    assert svg_stack != svg_layer

    paths_stack = _area_paths_in_draw_order(svg_stack)
    paths_layer = _area_paths_in_draw_order(svg_layer)
    assert len(paths_stack) == 2, f"expected 2 stacked density curves; got {len(paths_stack)}"
    assert len(paths_layer) == 2, f"expected 2 layered density curves; got {len(paths_layer)}"
    assert len(paths_stack[0]) == len(paths_stack[1]), (
        "both stacked curves must have the same point count (the widened "
        f"domain must not drop rows from either group); got "
        f"{len(paths_stack[0])} vs {len(paths_stack[1])}"
    )

    plot_area_x = float(_svg_rects(svg_stack)[1]["x"])

    def _bottom_half_xs(points: list[tuple[float, float]]) -> list[float]:
        half = len(points) // 2
        return [x for x, _y in points[half:]]

    # "a" (first group in stack order) always has base=0 -> every
    # bottom-half point sits exactly at the panel-left baseline.
    stack_a_bottom = _bottom_half_xs(paths_stack[0])
    assert all(abs(x - plot_area_x) < 1e-6 for x in stack_a_bottom), (
        "first-stacked group's base must equal the panel-left baseline for "
        f"every point; got x range [{min(stack_a_bottom)}, {max(stack_a_bottom)}]"
    )

    # "b" (second group) is stacked on top of "a", so its base must be
    # offset from the baseline for EVERY point (not just some).
    stack_b_bottom = _bottom_half_xs(paths_stack[1])
    assert all(abs(x - plot_area_x) > 1e-6 for x in stack_b_bottom), (
        "stacked group's base must differ from the panel-left baseline for "
        f"every point; got x range "
        f"[{min(stack_b_bottom)}, {max(stack_b_bottom)}], baseline={plot_area_x}"
    )


# ---------------------------------------------------------------------------
# Test 3 — vertical stacked histogram: unaffected by GH #77, correct probe
# ---------------------------------------------------------------------------


def test_vertical_stacked_histogram_unchanged_and_correct():
    """Vertical orientation must stay byte-stable: no ``value_axis`` key on
    the wire, since ``desugar_histogram``/``desugar_density`` only set it
    for ``orientation == "horizontal"``.

    Probe (c): whether same-bin stacked segments actually tile without
    overlap along y. Rust-side investigation found that
    ``render::marks::bar::build_quantitative`` (the vertical x2-binned bar
    drawer, dispatched whenever ``x``/``x2`` are both quantitative and
    ``y2`` is absent — i.e. every plain vertical histogram) reads
    ``baseline_y`` and ``top_y`` only; it never reads
    ``__stack_y_base__``. Every segment therefore draws from the panel
    bottom to its own (possibly cumulative) top, so segments in the same
    bin fully overlap instead of tiling — confirmed below by asserting
    disjoint y-spans, which fails. This is a pre-existing gap, unrelated to
    GH #77's value_axis wire (the vertical desugar path never sets
    value_axis, so ``apply_stack`` correctly cumulates via the y column;
    the bug is entirely in the *drawer* ignoring the correctly-computed
    base). Not fixed here per the task's Rust-boundary constraint.
    """
    df = pl.DataFrame(
        {
            "v": [0.5, 0.5, 0.5, 1.5, 1.5],
            "g": ["a", "a", "b", "a", "b"],
        }
    )

    # (a) renders fine.
    svg = (
        fm.Chart(df)
        .mark_histogram(multiple="stack", bin_count=2, extent=(0.0, 2.0))
        .encode(x="v", color="g")
        .to_svg()
    )
    assert "<svg" in svg

    # (b) no "value_axis" key in the vertical Stack position JSON.
    spec = (
        fm.Chart(df)
        .mark_histogram(multiple="stack", bin_count=2, extent=(0.0, 2.0))
        .encode(x="v", color="g")
        .to_spec()
    )
    position = json.loads(spec.to_json())["position"]
    assert "value_axis" not in position, position

    # (c) probe: same-bin segments must tile without overlap along y.
    # KNOWN GAP (pre-existing, not GH #77): build_quantitative ignores
    # __stack_y_base__, so this currently FAILS with large y-overlaps.
    rects = _data_rects(svg)
    assert len(rects) == 4, f"expected 4 bar segments (2 bins x 2 groups); got {len(rects)}"
    _assert_tiles_without_overlap(rects, axis="y")


test_vertical_stacked_histogram_unchanged_and_correct = pytest.mark.xfail(
    reason=(
        "vertical build_quantitative ignores __stack_y_base__ — tracked "
        "separately (pre-existing, not GH #77)"
    ),
    strict=True,
)(test_vertical_stacked_histogram_unchanged_and_correct)


# ---------------------------------------------------------------------------
# Test 4 — primitive path: mark_bar(orient="horizontal") + explicit Stack()
# ---------------------------------------------------------------------------


def test_horizontal_stacked_histogram_coord_flip_primitive_still_works():
    """Primitive (non-composite) horizontal stacking via real ``CoordFlip``.

    ``orient="horizontal"`` on a plain ``mark_bar`` sets real ``CoordFlip``
    (``_coord="flip"``), which is a DIFFERENT code path than the composite
    desugar's ``value_axis`` wire (real CoordFlip never sets
    ``value_axis``; ``apply_stack`` falls back to its pre-#77
    ``coord_flipped``-only convention, which the GH #77 Rust diff also
    taught ``build_quantitative_horizontal`` to read via the
    ``STACK_VALUE_ON_X_KEY`` schema stamp).

    Shape choice: a plain categorical y (``x=``category, real CoordFlip)
    dispatches to ``build_ordinal_y``, a SIBLING drawer this task's Rust
    diff did not touch — it never reads ``__stack_y_base__`` either (same
    class of gap as test 3's ``build_quantitative`` finding, discovered
    incidentally; not fixed here). The shape that genuinely exercises the
    GH #77-fixed ``build_quantitative_horizontal`` path is a quantitative
    x/x2 bin-edge + y2 (post-flip) bar — i.e. encode as if a normal
    vertical binned bar (``x``=bin_lo, ``x2``=bin_hi, ``y``=count) and let
    ``orient="horizontal"`` flip it, exactly mirroring the Rust GH #77
    tests (``stack_value_axis_none_coord_flipped_true_preserved``,
    ``horizontal_stacked_bar_segments_span_base_to_value_no_overlap``).
    """
    df = pl.DataFrame(
        {
            "bin_lo": [0.0, 0.0, 1.0, 1.0],
            "bin_hi": [1.0, 1.0, 2.0, 2.0],
            "count": [3.0, 5.0, 2.0, 4.0],
            "g": ["a", "b", "a", "b"],
        }
    )
    svg = (
        fm.Chart(df)
        .mark_bar(orient="horizontal", position=Stack(by="g"))
        .encode(x="bin_lo", x2="bin_hi", y=Y("count", scale={"domain": [0, 10]}), color="g")
        .to_svg()
    )
    rects = _data_rects(svg)
    assert len(rects) == 4, f"expected 4 bar segments (2 bins x 2 groups); got {len(rects)}"
    _assert_tiles_without_overlap(rects, axis="x")


# ---------------------------------------------------------------------------
# Test 5 — wire byte-stability guard
# ---------------------------------------------------------------------------


def test_stack_value_axis_wire_absent_when_unset():
    """A plain vertical ``Stack()`` (no ``value_axis``) must not emit the
    key at all — byte-stability guard for every pre-#77 spec-JSON fixture.
    ``tests/test_scale_spec_parity.py`` is re-verified in the same run (see
    the GH #77 fix report's full pass output) rather than invoked from
    inside this test.
    """
    df = pl.DataFrame({"x": ["a", "b"], "y": [1.0, 2.0], "g": ["p", "q"]})
    spec = fm.Chart(df).mark_bar(position=Stack()).encode(x="x", y="y", color="g").to_spec()
    position = json.loads(spec.to_json())["position"]
    assert "value_axis" not in position, position
