"""Regression coverage for F-L09-01 + F-L09-10 (batch F task 4).

F-L09-01: ``displot(kind="kde", cumulative=True)`` must actually produce a
CDF (for the primary ``kind="kde"`` mark and for the ``kind="hist", kde=True``
overlay curve).

F-L09-10: the ``kind="hist", kde=True`` overlay must be visible and honest:

- count-scaled onto the bars' own count axis under ``stat="count"`` (not an
  unscaled density, which renders as a flat ~2% sliver against the bars);
- grouped and colored per ``hue`` level (not one curve pooled across every
  level in a single theme color);
- independent of the primary mark's position adjustment (a density curve is
  never meaningfully stacked, dodged, or normalized -- see
  ``TestOverlayPositionIndependence``);
- scaled using the *same* bin width the rendered bars actually used
  (``TestOverlayBinWidthParity``), and *only* requested when the primary
  mark's value axis is actually a count axis -- excluded for
  ``kind="ecdf"``/``"rug"`` (no matching count axis) and
  ``multiple="fill"`` (the axis is a ``[0, 1]`` proportion, not a count,
  regardless of ``stat``) -- see ``TestOverlayCountScaleAxisScope``;
- and ``stat=`` must validate instead of silently degrading to count
  semantics for any unrecognized value.

These tests parse rendered SVG geometry (bar ``<rect>`` heights, overlay
``<polyline>`` points) rather than asserting on the spec dict alone, per
this finding's "must be visible" framing -- a wiring-only check would not
catch a scale/geometry regression.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._svg_rects import data_bar_rects as _data_bar_rects
from tests._svg_rects import svg_elements as _svg_elements


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unimodal_df():
    """A single-mode normal sample -- for ungrouped displot calls."""
    rng = np.random.default_rng(0)
    return pl.DataFrame({"x": rng.normal(size=300)})


@pytest.fixture
def two_group_df():
    """Two normal samples at different means, tagged by ``g`` -- for hue=."""
    rng = np.random.default_rng(0)
    n = 300
    return pl.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, n), rng.normal(3, 1, n)]),
            "g": ["a"] * n + ["b"] * n,
        }
    )


@pytest.fixture
def two_facet_df():
    """Two normal samples at different means, tagged by ``facet`` -- for col=."""
    rng = np.random.default_rng(0)
    n = 200
    return pl.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, n), rng.normal(3, 1, n)]),
            "facet": ["p1"] * n + ["p2"] * n,
        }
    )


# ---------------------------------------------------------------------------
# SVG geometry helpers. `_data_bar_rects` comes from the shared
# `tests/_svg_rects.py` module (same convention as `tests/_svg_extents.py`);
# the polyline helpers below are specific to this file's line-mark overlay.
# ---------------------------------------------------------------------------


def _polyline_points(svg: str) -> list[list[tuple[float, float]]]:
    """Return the ``(x, y)`` point lists for every ``<polyline>`` in *svg*."""
    out = []
    for pts in re.findall(r'<polyline[^>]*points="([^"]+)"', svg):
        out.append([tuple(map(float, p.split(","))) for p in pts.split()])
    return out


def _polyline_strokes(svg: str) -> list[str]:
    return re.findall(r'<polyline[^>]*stroke="([^"]+)"', svg)


def _polyline_peak_by_stroke(svg: str) -> dict[str, float]:
    """Map each ``<polyline>``'s ``stroke`` color to its peak (min-y) point.

    Since the same hue level gets the same color across two separately
    rendered charts (fixed nominal-color domain order), this lets a test
    compare "this group's curve" between two renders without depending on
    ``<polyline>`` emission order.
    """
    out = {}
    for pts, stroke in re.findall(r'<polyline points="([^"]+)"[^>]*stroke="([^"]+)"', svg):
        ys = [float(p.split(",")[1]) for p in pts.split()]
        out[stroke] = min(ys)
    return out


def _curve_to_bar_ratio(bars: list[dict[str, str]], points: list[tuple[float, float]]) -> float:
    """Ratio of a curve's peak height (px) to the tallest bar's height (px).

    Both are measured from the shared baseline (max of ``y + height`` across
    the bars) so the two panels being compared use the same zero line.
    """
    heights = [float(b["height"]) for b in bars]
    baseline = max(float(b["y"]) + float(b["height"]) for b in bars)
    max_bar_height = max(heights)
    curve_top = min(y for _, y in points)
    return (baseline - curve_top) / max_bar_height


def _rug_tick_x1_by_stroke(svg: str) -> dict[str, set[float]]:
    """Map each rug tick `<line>`'s `stroke` color to its set of x1 positions.

    Excludes the theme axis-line/gridline strokes (`#6b7280`, `#d6d3d1`) so
    only the per-hue rug-tick lines remain -- lets a test compare "this
    group's rug ticks" between two renders without depending on emission
    order or on the theme's own chrome lines.
    """
    out: dict[str, set[float]] = {}
    for attrs in _svg_elements(
        svg, "line", lambda a: a.get("stroke") not in {"#6b7280", "#d6d3d1"}
    ):
        out.setdefault(attrs["stroke"], set()).add(round(float(attrs["x1"]), 3))
    return out


def _kde_overlay_transform(spec: dict) -> dict:
    return next(t for t in spec["transforms"] if t.get("name") == "kde_overlay")


def _kde_overlay_layer(spec: dict) -> dict:
    return next(layer for layer in spec["layers"] if layer.get("data_source") == "kde_overlay")


def _assert_finite_svg(svg: str) -> None:
    assert "NaN" not in svg
    assert "Infinity" not in svg


# ---------------------------------------------------------------------------
# F-L09-01: cumulative= reaches both the primary kind="kde" mark and the
# kind="hist", kde=True overlay curve.
# ---------------------------------------------------------------------------


class TestCumulativeThreading:
    def test_kde_primary_cumulative_toggle_changes_output(self, unimodal_df):
        """cumulative=True vs cumulative=False must render different SVGs for kind="kde"."""
        svg_pdf = fm.displot(unimodal_df, x="x", kind="kde", cumulative=False).to_svg()
        svg_cdf = fm.displot(unimodal_df, x="x", kind="kde", cumulative=True).to_svg()
        _assert_finite_svg(svg_pdf)
        _assert_finite_svg(svg_cdf)
        assert svg_pdf != svg_cdf

    def test_kde_primary_grouped_cumulative_is_monotonic_per_group(self, two_group_df):
        """A grouped CDF must be monotonically non-decreasing within each group.

        Rendered as a per-group `<polyline>` (fill=False so the mark is a
        line, not an area path); SVG y increases downward, so a correct CDF's
        y-coordinates are monotonically non-increasing as x increases.
        """
        chart = fm.displot(two_group_df, x="x", hue="g", kind="kde", cumulative=True, fill=False)
        svg = chart.to_svg()
        _assert_finite_svg(svg)
        point_lists = _polyline_points(svg)
        assert len(point_lists) == 2, f"expected one curve per hue level, got {len(point_lists)}"
        for points in point_lists:
            ordered = sorted(points, key=lambda p: p[0])
            ys = [y for _, y in ordered]
            # Allow a hair of numerical slack; the CDF must not have a real
            # (>0.5px) reversal anywhere along its length.
            violations = sum(1 for a, b in zip(ys, ys[1:]) if b - a > 0.5)
            assert violations == 0, f"non-monotonic CDF curve: {violations} reversals"

    def test_hist_kde_overlay_cumulative_plateau_tracks_tallest_cumulative_bar(self, unimodal_df):
        """cumulative= must also reach the kde=True overlay, not just the
        primary kind="kde" path. Deleting `cumulative=cumulative` from the
        overlay's `Kde(...)` call would leave every other test in this file
        green, since none of them render a `kind="hist", kde=True,
        cumulative=True` chart.

        Discriminating assertion: a cumulative histogram's tallest bar is
        its last (rightmost) bin, holding the full cumulative count; a
        correctly cumulative, count-scaled overlay's plateau (its curve
        peak, since a CDF rises monotonically) must land within 5% of that
        bar's height, not stay an unscaled, non-cumulative PDF shape.
        """
        chart = fm.displot(unimodal_df, x="x", kind="hist", kde=True, stat="count", cumulative=True)
        spec = json.loads(chart.to_spec().to_json())
        assert _kde_overlay_transform(spec)["cumulative"] is True

        svg = chart.to_svg()
        _assert_finite_svg(svg)
        bars = _data_bar_rects(svg)
        baseline = max(float(b["y"]) + float(b["height"]) for b in bars)
        tallest_bar_height = baseline - min(float(b["y"]) for b in bars)

        point_lists = _polyline_points(svg)
        assert len(point_lists) == 1
        curve_height = baseline - min(y for _, y in point_lists[0])
        assert curve_height == pytest.approx(tallest_bar_height, rel=0.05), (
            f"cumulative overlay plateau ({curve_height:.1f}px) does not track the "
            f"tallest cumulative bar ({tallest_bar_height:.1f}px)"
        )


# ---------------------------------------------------------------------------
# F-L09-10: the overlay is visible on a count axis under stat="count", and
# unscaled + byte-stable under stat="density".
# ---------------------------------------------------------------------------


class TestOverlayCountAxisVisibility:
    def test_overlay_visible_against_bars_under_stat_count(self, unimodal_df):
        """Under stat="count" (the default), the overlay curve must be a
        substantial fraction of the bars' height, not a flat ~2% sliver.
        """
        chart = fm.displot(unimodal_df, x="x", kind="hist", kde=True, stat="count")
        svg = chart.to_svg()
        _assert_finite_svg(svg)
        bars = _data_bar_rects(svg)
        point_lists = _polyline_points(svg)
        assert len(point_lists) == 1
        ratio = _curve_to_bar_ratio(bars, point_lists[0])
        assert ratio > 0.3, f"overlay curve/bar ratio too small: {ratio:.3f}"

    def test_overlay_visible_per_panel_under_facets(self, two_facet_df):
        """Faceted displot(col=..., kde=True, stat="count"): each panel's
        overlay must be comparably scaled to that panel's own bars -- a
        count-scaled overlay computed from the wrong (e.g. pooled or
        mismatched) bin width would only show up per-panel.
        """
        chart = fm.displot(two_facet_df, x="x", col="facet", kind="hist", kde=True, stat="count")
        svg = chart.to_svg()
        _assert_finite_svg(svg)
        bars = _data_bar_rects(svg)
        point_lists = _polyline_points(svg)
        assert len(point_lists) == 2, (
            f"expected one overlay curve per facet panel, got {len(point_lists)}"
        )

        # Partition bars into the two panels by an x-midpoint split -- col
        # faceting lays panels out side by side, so each panel's bars form a
        # contiguous x-cluster distinct from the other panel's.
        xs = sorted(float(b["x"]) for b in bars)
        mid = (xs[0] + xs[-1]) / 2
        panels = [
            [b for b in bars if float(b["x"]) < mid],
            [b for b in bars if float(b["x"]) >= mid],
        ]
        # Match each curve to its panel by x-overlap (curve x-range must sit
        # inside that panel's bar x-range).
        for points in point_lists:
            curve_x = [x for x, _ in points]
            curve_mid = (min(curve_x) + max(curve_x)) / 2
            panel_bars = panels[0] if curve_mid < mid else panels[1]
            assert panel_bars, "no bars found for this panel"
            ratio = _curve_to_bar_ratio(panel_bars, points)
            assert ratio > 0.3, f"panel overlay curve/bar ratio too small: {ratio:.3f}"

    def test_stat_density_overlay_unscaled_and_byte_stable(self, unimodal_df):
        """stat="density" must leave the overlay unscaled: no `count_scale`
        on the wire, and the rendered SVG must match a pinned byte-identity
        hash captured against the pre-fix baseline.
        """
        chart = fm.displot(unimodal_df, x="x", kind="hist", kde=True, stat="density")
        spec = json.loads(chart.to_spec().to_json())
        assert "count_scale" not in _kde_overlay_transform(spec)

        svg = chart.to_svg()
        _assert_finite_svg(svg)
        digest = hashlib.sha256(svg.encode()).hexdigest()
        # Digest updated (T5r, 2026-09-10): a `kind="hist", kde=True` overlay
        # is a layered spec (the histogram `bar` layer plus the KDE `line`
        # layer) whose density y-axis previously stopped at the auto-inferred
        # union of the two layers' extents -- the SAME truncated-axis bug
        # `catplot(kind="bar")` had, fixed in the same place:
        # `numeric_domain_union`'s bar zero-anchor
        # (`crates/ferrum-core/src/render/scale_resolve/domain.rs`). The
        # y-axis now reaches exactly 0 instead of stopping just above it, so
        # this pinned baseline moved; the pre-fix digest was
        # "118ad56b5e7a9e4655f1619a9615a3bed007e9b2aa9c5907051c8203b705d15f".
        # Re-verified visually (rasterized PNG) that the histogram + KDE
        # overlay render correctly at the new baseline.
        #
        # Cycle-2 note (rust-quality-review): the FIRST version of this fix
        # (cycle 1) over-applied the anchor to the bar layer's POSITION
        # channel too (any numeric channel a bar layer bound, not just its
        # value channel), which would have moved THIS digest for the wrong
        # reason on off-center data. `unimodal_df` (`rng.normal(size=300)`,
        # zero-centered) cannot reveal that bug either way: its x extent
        # already straddles zero, so the (over-broad, cycle-1) x-anchor was
        # a structural no-op for this fixture specifically (`mn` was already
        # negative, `mx` already positive -- neither zero-extension branch
        # could fire). Confirmed after narrowing to the value-channel-only
        # rule (cycle 2): this digest is UNCHANGED from cycle 1, and the
        # rendered x extent (-3.5..3.5) is identical before and after
        # narrowing -- so the "density y-axis alone" attribution above holds
        # precisely, not just plausibly. The off-center case the narrowing
        # actually fixes (`displot(x=..., kind="hist", kde=True)` on data
        # like `normal(100, 5)`, where the bug visibly squeezed bars into a
        # corner of the panel) is pinned at the Rust level instead --
        # `bar_position_channel_never_zero_extends` and
        # `coord_flipped_bar_position_channel_never_zero_extends` in
        # `domain.rs` -- because this fixture is structurally blind to it.
        assert digest == "84afa395a92ad173597898b22adb9b7615996dbd82112e667ce7102a42f09e4f", (
            "stat='density' overlay output moved -- expected byte-identity "
            "with the T5r baseline"
        )

    def test_stat_count_sets_count_scale_matching_bins(self, unimodal_df):
        """stat="count" must set `count_scale` from the same bin-count rule
        the histogram's own `Bin` transform actually used: `bin_count` when
        `bins` is an int, else "sturges" (the 1-D `Bin`'s only auto rule --
        it has no Freedman-Diaconis fallback, so a raw non-int `bins` string
        would silently diverge from what the bars actually used).
        """
        chart = fm.displot(unimodal_df, x="x", kind="hist", kde=True, stat="count", bins=12)
        spec = json.loads(chart.to_spec().to_json())
        assert _kde_overlay_transform(spec)["count_scale"] == {"kind": "fixed", "n": 12}

        chart_default = fm.displot(unimodal_df, x="x", kind="hist", kde=True, stat="count")
        spec_default = json.loads(chart_default.to_spec().to_json())
        assert _kde_overlay_transform(spec_default)["count_scale"] == {"kind": "sturges"}


# ---------------------------------------------------------------------------
# F-L09-10: the overlay is grouped and colored per hue level.
# ---------------------------------------------------------------------------


class TestOverlayPerHueColor:
    def test_overlay_curves_are_distinctly_colored_per_hue(self, two_group_df):
        """hue= must produce one overlay curve per level, each in its own
        color -- not one pooled curve in a single theme color.
        """
        chart = fm.displot(
            two_group_df, x="x", hue="g", kind="hist", kde=True, stat="count", fill=False
        )
        svg = chart.to_svg()
        _assert_finite_svg(svg)
        strokes = _polyline_strokes(svg)
        assert len(strokes) == 2, (
            f"expected 2 overlay curves (one per hue level), got {len(strokes)}"
        )
        assert len(set(strokes)) == 2, f"overlay curves are not distinctly colored: {strokes}"

    def test_overlay_layer_carries_color_channel_matching_hue(self, two_group_df):
        chart = fm.displot(two_group_df, x="x", hue="g", kind="hist", kde=True)
        spec = json.loads(chart.to_spec().to_json())
        color = _kde_overlay_layer(spec)["encoding"]["color"]
        field = color.get("field") if isinstance(color, dict) else color
        assert field == "g"


# ---------------------------------------------------------------------------
# F-L09-10 / spec §6: the overlay's grouped bin width must match the bars'.
# ---------------------------------------------------------------------------


class TestOverlayBinWidthParity:
    """The overlay's `shared_extent` must equal what the histogram's `Bin`
    transform actually receives on the displot path -- which is always
    `False` today, since `_displot_build` never forwards `multiple=` into
    `mark_histogram` -- not a re-derivation of the `multiple in ("stack",
    "fill")` rule that never reaches that call.
    """

    def test_wire_shared_extent_matches_the_real_bin_transform(self, two_group_df):
        """For every `multiple=`, the kde-overlay transform's `shared_extent`
        on the wire must equal the real `Bin` transform's `shared_extent`.
        """
        for multiple in ("layer", "stack", "fill", "dodge"):
            chart = fm.displot(
                two_group_df,
                x="x",
                hue="g",
                kind="hist",
                kde=True,
                stat="count",
                multiple=multiple,
            )
            spec = json.loads(chart.to_spec().to_json())
            kde_xform = _kde_overlay_transform(spec)
            bin_xform = next(t for t in spec["transforms"] if t.get("type") == "bin")
            assert kde_xform["shared_extent"] == bin_xform["shared_extent"], (
                f"multiple={multiple!r}: kde shared_extent="
                f"{kde_xform['shared_extent']!r} != bin shared_extent="
                f"{bin_xform['shared_extent']!r}"
            )

    def test_stack_mode_curve_scale_matches_layer_mode(self, two_group_df):
        """Per-hue overlay curve geometry must be identical between
        multiple="stack" and multiple="layer" for the same data -- asserts
        the parity directly rather than pinning either render's absolute
        pixel constants.
        """
        layer_svg = fm.displot(
            two_group_df,
            x="x",
            hue="g",
            kind="hist",
            kde=True,
            stat="count",
            multiple="layer",
            fill=False,
        ).to_svg()
        stack_svg = fm.displot(
            two_group_df,
            x="x",
            hue="g",
            kind="hist",
            kde=True,
            stat="count",
            multiple="stack",
            fill=False,
        ).to_svg()
        _assert_finite_svg(layer_svg)
        _assert_finite_svg(stack_svg)

        layer_peaks = _polyline_peak_by_stroke(layer_svg)
        stack_peaks = _polyline_peak_by_stroke(stack_svg)
        assert len(layer_peaks) == 2, f"expected 2 hue curves, got {len(layer_peaks)}"
        assert set(layer_peaks) == set(stack_peaks), (
            "stack/layer overlay curves are colored differently -- cannot compare"
        )
        for stroke, layer_peak in layer_peaks.items():
            stack_peak = stack_peaks[stroke]
            assert stack_peak == pytest.approx(layer_peak, abs=0.5), (
                f"stroke {stroke}: multiple='stack' overlay peak {stack_peak:.2f} "
                f"diverges from multiple='layer' peak {layer_peak:.2f} -- the "
                "count-scaled bin width used by the overlay no longer matches "
                "the one the bars actually use"
            )


# ---------------------------------------------------------------------------
# F-L09-10: count_scale is requested only when the primary mark's value axis
# is actually a count axis.
# ---------------------------------------------------------------------------


class TestOverlayCountScaleAxisScope:
    """`count_scale` may only be requested when the histogram's y-axis is
    actually in counts: excluded when the primary mark isn't histogram bars
    at all (`kind="ecdf"`), and excluded when the bars are normalized to a
    `[0, 1]` proportion regardless of `stat` (`multiple="fill"`).
    """

    def test_ecdf_kde_overlay_omits_count_scale(self, unimodal_df):
        """kind="ecdf" renders a cumulative-count step line that a
        non-cumulative, count-scaled `Kde` overlay does not match, so
        `kind="ecdf", kde=True` must NOT set `count_scale`.
        """
        chart = fm.displot(unimodal_df, x="x", kind="ecdf", kde=True, stat="count")
        spec = json.loads(chart.to_spec().to_json())
        assert "count_scale" not in _kde_overlay_transform(spec)

        svg = chart.to_svg()
        _assert_finite_svg(svg)

    def test_fill_mode_omits_count_scale(self, two_group_df):
        """multiple="fill" normalizes bars to [0, 1] regardless of `stat`,
        so `count_scale` (which targets a count axis) must be omitted even
        under `stat="count"`.
        """
        chart = fm.displot(
            two_group_df, x="x", hue="g", kind="hist", kde=True, stat="count", multiple="fill"
        )
        spec = json.loads(chart.to_spec().to_json())
        assert "count_scale" not in _kde_overlay_transform(spec)

    def test_fill_mode_bars_match_kde_false_geometry(self, two_group_df):
        """The fill-mode bars must render at (approximately) the same height
        with kde=True as with kde=False -- i.e. still on a ~[0, 1] proportion
        axis, not collapsed by a count-scaled overlay sharing the domain.
        Asserts the parity between kde=True and kde=False, not either
        render's absolute pixel constants.
        """
        svg_kde = fm.displot(
            two_group_df, x="x", hue="g", kind="hist", kde=True, stat="count", multiple="fill"
        ).to_svg()
        svg_no_kde = fm.displot(
            two_group_df, x="x", hue="g", kind="hist", kde=False, stat="count", multiple="fill"
        ).to_svg()
        _assert_finite_svg(svg_kde)
        _assert_finite_svg(svg_no_kde)

        max_h_kde = max(float(b["height"]) for b in _data_bar_rects(svg_kde))
        max_h_no_kde = max(float(b["height"]) for b in _data_bar_rects(svg_no_kde))
        assert max_h_kde == pytest.approx(max_h_no_kde, rel=0.1), (
            f"multiple='fill' bar height with kde=True ({max_h_kde:.1f}px) diverges "
            f"from kde=False ({max_h_no_kde:.1f}px) -- the count-scaled overlay is "
            "stretching the shared proportion axis"
        )

    def test_stack_and_dodge_still_get_count_scale(self, two_group_df):
        """Only `multiple="fill"` is excluded -- `"stack"` and `"dodge"`
        bars stay counts and must keep the count-scaled overlay.
        """
        for multiple in ("stack", "dodge"):
            chart = fm.displot(
                two_group_df,
                x="x",
                hue="g",
                kind="hist",
                kde=True,
                stat="count",
                multiple=multiple,
            )
            spec = json.loads(chart.to_spec().to_json())
            assert "count_scale" in _kde_overlay_transform(spec), (
                f"multiple={multiple!r} should keep count_scale"
            )


# ---------------------------------------------------------------------------
# F-L09-10: the overlay must not inherit the primary mark's position
# adjustment -- a density curve is never meaningfully stacked, dodged, or
# normalized.
# ---------------------------------------------------------------------------


class TestOverlayPositionIndependence:
    def test_fill_mode_curves_are_real_not_flattened(self, two_group_df):
        """multiple="fill" overlay curves must be real per-group density
        curves, not degenerate flat lines pinned to the panel edge.

        Regression: once the overlay batch carries the hue column
        (`groupby=hue`), an unset `position` on the overlay `_Layer` let it
        inherit the chart-level `Stack(offset="normalize")`, flattening
        both curves to a single constant y (every point identical). Fixed
        by setting `position=Identity()` explicitly on the overlay layer.
        """
        svg = fm.displot(
            two_group_df,
            x="x",
            hue="g",
            kind="hist",
            kde=True,
            stat="count",
            multiple="fill",
            fill=False,
        ).to_svg()
        _assert_finite_svg(svg)
        point_lists = _polyline_points(svg)
        assert len(point_lists) == 2, f"expected 2 hue curves, got {len(point_lists)}"
        for points in point_lists:
            ys = [y for _, y in points]
            y_range = max(ys) - min(ys)
            assert y_range > 10, (
                f"overlay curve is flat (y range {y_range:.2f}px) -- likely inherited "
                "the chart's normalize-stack position"
            )

    def test_stack_and_dodge_curves_are_not_flattened_either(self, two_group_df):
        """Same check as fill mode, for the two other grouped-position
        modes -- guards against the `position=Identity()` fix having been
        scoped too narrowly (e.g. only to `multiple="fill"`).
        """
        for multiple in ("stack", "dodge"):
            svg = fm.displot(
                two_group_df,
                x="x",
                hue="g",
                kind="hist",
                kde=True,
                stat="count",
                multiple=multiple,
                fill=False,
            ).to_svg()
            _assert_finite_svg(svg)
            point_lists = _polyline_points(svg)
            assert len(point_lists) == 2, f"multiple={multiple!r}: expected 2 hue curves"
            for points in point_lists:
                ys = [y for _, y in points]
                y_range = max(ys) - min(ys)
                assert y_range > 10, (
                    f"multiple={multiple!r}: overlay curve is flat (y range {y_range:.2f}px)"
                )

    def test_rug_overlay_renders_under_stack_and_fill_with_hue(self, two_group_df):
        """rug=True + hue= + multiple="stack"/"fill" must render, not raise.

        Regression (pre-existing at HEAD, before this task's `kde=True`
        work): the rug overlay's `_Layer` had no explicit `position=`, so
        once its `identity_transform` batch carried the hue column, it
        inherited the chart-level `Stack(...)` -- and `Stack`'s zero/
        normalize offsets require a `y` encoding the rug's `mark="tick"`
        layer (x-only) never has, so `displot(..., hue=..., rug=True,
        multiple="stack")` (and `"fill"`) raised `ValueError: Stack: value
        (y) encoding required`. Fixed by the same `position=Identity()`
        applied to the kde overlay above (same class of bug).
        """
        for multiple in ("stack", "fill"):
            svg = fm.displot(two_group_df, x="x", hue="g", rug=True, multiple=multiple).to_svg()
            _assert_finite_svg(svg)

    def test_rug_tick_positions_independent_of_multiple(self, two_group_df):
        """Rug tick x-positions must not be dodged, stacked, or filled by
        `multiple=` -- a rug tick marks each row's raw x value, not an
        aggregated bar. `multiple="dodge"` changes the *bars'* rendered
        bytes (they shift side by side); it must leave the rug ticks' x
        positions identical to `multiple="layer"`.
        """
        layer_svg = fm.displot(two_group_df, x="x", hue="g", rug=True, multiple="layer").to_svg()
        dodge_svg = fm.displot(two_group_df, x="x", hue="g", rug=True, multiple="dodge").to_svg()
        layer_ticks = _rug_tick_x1_by_stroke(layer_svg)
        dodge_ticks = _rug_tick_x1_by_stroke(dodge_svg)
        assert dodge_ticks, "no rug tick lines found -- helper filter may be wrong"
        assert dodge_ticks == layer_ticks, (
            "rug tick x-positions differ between multiple='dodge' and "
            "multiple='layer' -- the rug overlay appears to have inherited "
            "the primary mark's dodge position"
        )


# ---------------------------------------------------------------------------
# F-L09-10: stat= validates instead of silently degrading to count.
# ---------------------------------------------------------------------------


class TestDisplotStatValidation:
    def test_stat_probability_refuses(self, unimodal_df):
        with pytest.raises(ValueError, match="stat"):
            fm.displot(unimodal_df, x="x", stat="probability")

    def test_stat_count_and_density_still_accepted(self, unimodal_df):
        fm.displot(unimodal_df, x="x", stat="count").to_svg()
        fm.displot(unimodal_df, x="x", stat="density").to_svg()
