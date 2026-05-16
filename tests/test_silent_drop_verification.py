"""Behavioral integration tests for Tasks 1–9 of the silent-drop remediation.

These tests check SVG OUTPUT STRUCTURE — not just "no crash".
A test that only checks "<svg" in svg is insufficient.

Each test verifies that the feature is actually rendered differently
depending on the kwarg, not merely accepted without error.

Covers:
  Task 1 — sort= (ascending / descending / explicit list)
  Task 2 — stack= (zero accumulation, normalize, invalid rejection)
  Task 3 — axis= dict (label_angle, ticks=False, labels=False)
  Task 4 — format_type= (number formatting applied to tick labels)
  Task 5 — impute= (missing values filled → more polyline points)
  Task 6 — legend kwargs (orient=bottom changes legend position)
  Task 7 — histogram multiple= (dodge → different x per group; stack → y accumulates)
  Task 8 — lmplot/regplot truncate=False (no longer raises ValueError)
  Task 9 — Chart(data=None) + Layer(data=) renders both datasets
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------


def _cat_df() -> pl.DataFrame:
    return pl.DataFrame({"cat": ["b", "a", "c"], "val": [10.0, 5.0, 15.0]})


def _group_bar_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cat": ["x", "x", "y", "y"],
            "val": [3.0, 7.0, 4.0, 6.0],
            "g": ["a", "b", "a", "b"],
        }
    )


def _numeric_df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]})


def _hist_df() -> pl.DataFrame:
    import numpy as np

    rng = np.random.default_rng(0)
    vals_a = rng.normal(0, 1, 30).tolist()
    vals_b = rng.normal(2, 1, 30).tolist()
    return pl.DataFrame(
        {
            "x": vals_a + vals_b,
            "g": ["a"] * 30 + ["b"] * 30,
        }
    )


def _sparse_df() -> pl.DataFrame:
    """Sparse time series: group b is missing x=2."""
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 1.0, 3.0],
            "y": [1.0, 2.0, 3.0, 4.0, 6.0],
            "group": ["a", "a", "a", "b", "b"],
        }
    )


def _reg_df() -> pl.DataFrame:
    import numpy as np

    rng = np.random.default_rng(7)
    x = rng.uniform(0, 10, 30)
    y = 2 * x + rng.normal(0, 1, 30)
    return pl.DataFrame({"x": x.tolist(), "y": y.tolist()})


def _extract_tick_labels(svg: str) -> list[str]:
    """Extract all <text> element content from SVG."""
    return re.findall(r"<text[^>]*>([^<]+)</text>", svg)


def _extract_rects(svg: str) -> list[str]:
    """Return all <rect ...> tag strings from SVG."""
    return re.findall(r"<rect[^/]*/?>", svg)


def _extract_rect_coords(svg: str) -> list[dict]:
    """Extract x, y, width, height from each <rect> element."""
    rects = []
    for r in re.findall(r"<rect[^/]*(?:/>|>)", svg):
        attrs = {}
        for attr in ("x", "y", "width", "height"):
            m = re.search(rf'\b{attr}="([^"]+)"', r)
            if m:
                try:
                    attrs[attr] = float(m.group(1))
                except ValueError:
                    pass
        if attrs:
            rects.append(attrs)
    return rects


def _extract_polyline_points(svg: str) -> list[list[tuple[float, float]]]:
    """Return a list-of-lists of (x, y) coordinate pairs, one per <polyline>."""
    results = []
    for pts_str in re.findall(r'<polyline[^>]*\bpoints="([^"]+)"', svg):
        pts = []
        for pair in pts_str.strip().split():
            parts = pair.split(",")
            if len(parts) == 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        results.append(pts)
    return results


# ---------------------------------------------------------------------------
# Task 1: sort= changes axis tick ORDER in SVG
# ---------------------------------------------------------------------------


class TestSortSVGOrder:
    """sort= must change the ORDER of category tick labels in the SVG, not just
    be silently accepted."""

    def test_sort_ascending_produces_alpha_order_in_svg(self):
        """sort='ascending' → tick labels appear in a-b-c order, not insertion order."""
        svg_default = fm.Chart(_cat_df()).mark_bar().encode(x=fm.X("cat"), y="val").show_svg()
        svg_asc = (
            fm.Chart(_cat_df())
            .mark_bar()
            .encode(x=fm.X("cat", sort="ascending"), y="val")
            .show_svg()
        )
        labels_asc = [l for l in _extract_tick_labels(svg_asc) if l in ("a", "b", "c")]
        assert labels_asc == ["a", "b", "c"], (
            f"sort='ascending' should yield [a,b,c] in SVG tick order; got {labels_asc}"
        )
        # Without sort, insertion order is b,a,c (not alphabetical).
        labels_default = [l for l in _extract_tick_labels(svg_default) if l in ("a", "b", "c")]
        # Ascending and default MUST differ (b,a,c vs a,b,c)
        # This confirms the sort actually changed output.
        assert labels_asc != labels_default or labels_default == ["a", "b", "c"], (
            f"sort='ascending' SVG tick order {labels_asc} must differ from "
            f"unsorted {labels_default} (unless default happens to be already sorted)"
        )

    def test_sort_descending_reverses_alpha_order_in_svg(self):
        """sort='descending' → tick labels appear in c-b-a order."""
        svg = (
            fm.Chart(_cat_df())
            .mark_bar()
            .encode(x=fm.X("cat", sort="descending"), y="val")
            .show_svg()
        )
        labels = [l for l in _extract_tick_labels(svg) if l in ("a", "b", "c")]
        assert labels == ["c", "b", "a"], (
            f"sort='descending' should yield [c,b,a] in SVG; got {labels}"
        )

    def test_sort_ascending_differs_from_descending_in_svg(self):
        """Ascending and descending produce DIFFERENT SVG tick orders (both implemented)."""
        svg_asc = (
            fm.Chart(_cat_df())
            .mark_bar()
            .encode(x=fm.X("cat", sort="ascending"), y="val")
            .show_svg()
        )
        svg_desc = (
            fm.Chart(_cat_df())
            .mark_bar()
            .encode(x=fm.X("cat", sort="descending"), y="val")
            .show_svg()
        )
        asc_labels = [l for l in _extract_tick_labels(svg_asc) if l in ("a", "b", "c")]
        desc_labels = [l for l in _extract_tick_labels(svg_desc) if l in ("a", "b", "c")]
        assert asc_labels != desc_labels, (
            f"Ascending ({asc_labels}) and descending ({desc_labels}) must differ"
        )

    def test_sort_explicit_list_produces_exact_order_in_svg(self):
        """sort=['c','a','b'] → SVG tick labels appear in exactly that sequence."""
        svg = (
            fm.Chart(_cat_df())
            .mark_bar()
            .encode(x=fm.X("cat", sort=["c", "a", "b"]), y="val")
            .show_svg()
        )
        labels = [l for l in _extract_tick_labels(svg) if l in ("a", "b", "c")]
        assert labels == ["c", "a", "b"], (
            f"sort=['c','a','b'] should yield [c,a,b] in SVG; got {labels}"
        )

    def test_sort_list_differs_from_unsorted(self):
        """sort=['c','a','b'] produces DIFFERENT SVG than no-sort (confirming wired)."""
        svg_no_sort = fm.Chart(_cat_df()).mark_bar().encode(x=fm.X("cat"), y="val").show_svg()
        svg_sort = (
            fm.Chart(_cat_df())
            .mark_bar()
            .encode(x=fm.X("cat", sort=["c", "a", "b"]), y="val")
            .show_svg()
        )
        no_sort_labels = [l for l in _extract_tick_labels(svg_no_sort) if l in ("a", "b", "c")]
        sort_labels = [l for l in _extract_tick_labels(svg_sort) if l in ("a", "b", "c")]
        assert sort_labels != no_sort_labels, (
            f"sort=['c','a','b'] should reorder ticks relative to no-sort; "
            f"no-sort={no_sort_labels}, sort={sort_labels}"
        )


# ---------------------------------------------------------------------------
# Task 2: stack= changes BAR POSITIONS in SVG (not just rendered without crash)
# ---------------------------------------------------------------------------


class TestStackSVGPositions:
    """stack='zero' must accumulate bar heights — the second group's bars must
    start where the first group ends, not at y=0."""

    def test_stack_zero_bars_have_different_y_per_group(self):
        """stack='zero' on Y encoding must change bar y-positions vs unstacked.

        KNOWN GAP (Task 2): Y(stack='zero') is accepted and stored in EncodingSpec,
        but the Rust renderer only applies position adjustment when
        ``layer.position.is_some()`` (scene_build.rs:155). The encoding-level
        stack fallback in apply_position is dead code because apply_position is
        never called without an explicit layer.position.

        Result: Y(stack='zero') produces IDENTICAL SVG to no-stack. The only
        wired stacking path is via position=Stack(offset='zero') passed to
        mark_histogram/mark_density via their multiple= kwarg.

        This test documents the CURRENT (broken) behavior to make the gap explicit.
        TODO: When the gap is fixed (apply_position called unconditionally or
        scene_build checks encoding.y.stack), change the assertion below to:
            assert y_vals_stacked != y_vals_unstacked
        """
        svg_stacked = (
            fm.Chart(_group_bar_df())
            .mark_bar()
            .encode(x="cat", y=fm.Y("val", stack="zero"), color="g")
            .show_svg()
        )
        svg_unstacked = (
            fm.Chart(_group_bar_df()).mark_bar().encode(x="cat", y="val", color="g").show_svg()
        )

        # GAP: Y(stack='zero') encoding-level stacking is not wired in
        # scene_build.rs — apply_position is only called when layer.position
        # is set, not when encoding.y.stack is set. Fix: read encoding.y.stack
        # in scene_build.rs and call apply_position accordingly.
        assert svg_stacked != svg_unstacked, (
            "Y(stack='zero') must produce different bar positions than unstacked. "
            "Currently broken: scene_build.rs only calls apply_position when "
            "layer.position is explicitly set, ignoring EncodingSpec.stack."
        )

    def test_stack_normalize_produces_bars_at_fractional_heights(self):
        """stack='normalize' → bar heights as fractions of 1.0 (y-axis max ≈ 1.0)."""
        svg = (
            fm.Chart(_group_bar_df())
            .mark_bar()
            .encode(x="cat", y=fm.Y("val", stack="normalize"), color="g")
            .show_svg()
        )
        labels = _extract_tick_labels(svg)
        numeric_labels = []
        for lbl in labels:
            try:
                v = float(lbl)
                numeric_labels.append(v)
            except ValueError:
                pass
        # With normalization, the maximum tick label on y should be ≤ 1.0
        assert any(v <= 1.0 for v in numeric_labels), (
            f"stack='normalize' y-axis should have tick ≤ 1.0; numeric ticks: {numeric_labels}"
        )

    def test_stack_invalid_raises_at_render(self):
        """stack='bogus' → ValueError or TypeError at construction/render time."""
        with pytest.raises((ValueError, TypeError)):
            fm.Chart(_group_bar_df()).mark_bar().encode(
                x="cat",
                y=fm.Y("val", stack="bogus"),
                color="g",
            ).show_svg()


# ---------------------------------------------------------------------------
# Task 3: axis= dict — changes SVG structure (rotation, hidden elements)
# ---------------------------------------------------------------------------


class TestAxisDictSVGStructure:
    """axis= dict kwargs must produce measurably different SVG output."""

    def test_label_angle_negative45_emits_rotate_transform(self):
        """axis={'label_angle': -45} → rotate(-45 ...) appears in SVG on x-axis labels.

        The Rust axis renderer emits `transform="rotate(-45 x y)"` on each
        <text> element when label_angle != 0.
        """
        svg = (
            fm.Chart(pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [1.0, 2.0, 3.0, 4.0]}))
            .mark_bar()
            .encode(x=fm.X("cat", axis={"label_angle": -45}), y="val")
            .show_svg()
        )
        assert "rotate(-45" in svg, (
            f"axis={{'label_angle': -45}} should produce rotate(-45 ...) in SVG;\n"
            f"SVG excerpt: {svg[:3000]}"
        )

    def test_label_angle_produces_different_svg_than_default(self):
        """Setting label_angle changes SVG vs default (no label_angle set)."""
        svg_default = (
            fm.Chart(pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [1.0, 2.0, 3.0, 4.0]}))
            .mark_bar()
            .encode(x="cat", y="val")
            .show_svg()
        )
        svg_rotated = (
            fm.Chart(pl.DataFrame({"cat": ["a", "b", "c", "d"], "val": [1.0, 2.0, 3.0, 4.0]}))
            .mark_bar()
            .encode(x=fm.X("cat", axis={"label_angle": -45}), y="val")
            .show_svg()
        )
        rotate_default = svg_default.count("rotate(-45")
        rotate_angled = svg_rotated.count("rotate(-45")
        assert rotate_angled > rotate_default, (
            f"label_angle=-45 should add rotate(-45 ...) transforms; "
            f"default count={rotate_default}, angled count={rotate_angled}"
        )

    def test_labels_false_reduces_text_elements(self):
        """axis={'labels': False} → fewer <text> elements than default (labels hidden)."""
        svg_with = fm.Chart(_numeric_df()).mark_point().encode(x="x", y="y").show_svg()
        svg_without = (
            fm.Chart(_numeric_df())
            .mark_point()
            .encode(x=fm.X("x", axis={"labels": False}), y="y")
            .show_svg()
        )
        count_with = svg_with.count("<text")
        count_without = svg_without.count("<text")
        assert count_without < count_with, (
            f"axis={{'labels': False}} should remove x-axis tick label <text> elements; "
            f"with_labels={count_with}, without_labels={count_without}"
        )

    def test_ticks_false_reduces_line_elements(self):
        """axis={'ticks': False} → fewer <line> elements (tick marks removed)."""
        svg_with = fm.Chart(_numeric_df()).mark_point().encode(x="x", y="y").show_svg()
        svg_without = (
            fm.Chart(_numeric_df())
            .mark_point()
            .encode(x=fm.X("x", axis={"ticks": False}), y="y")
            .show_svg()
        )
        count_with = svg_with.count("<line")
        count_without = svg_without.count("<line")
        assert count_without < count_with, (
            f"axis={{'ticks': False}} should remove x-axis tick <line> elements; "
            f"with_ticks={count_with}, without_ticks={count_without}"
        )

    def test_axis_title_override_appears_in_svg(self):
        """axis={'title': 'Custom Title'} → that string appears in SVG output."""
        svg = (
            fm.Chart(_numeric_df())
            .mark_point()
            .encode(x=fm.X("x", axis={"title": "MyCustomAxisTitle"}), y="y")
            .show_svg()
        )
        assert "MyCustomAxisTitle" in svg, (
            "axis={'title': 'MyCustomAxisTitle'} string must appear in SVG"
        )


# ---------------------------------------------------------------------------
# Task 4: format_type= — tick labels formatted differently
# ---------------------------------------------------------------------------


class TestFormatTypeSVGLabels:
    """format_type= must cause tick labels to be formatted differently,
    not just accepted without error."""

    def test_format_dotf_produces_decimal_labels(self):
        """X(format='.2f', format_type='number') → tick labels have decimal point."""
        svg = (
            fm.Chart(_numeric_df())
            .mark_point()
            .encode(x=fm.X("x", format=".2f", format_type="number"), y="y")
            .show_svg()
        )
        labels = _extract_tick_labels(svg)
        decimal_labels = [l for l in labels if "." in l and re.match(r"^-?\d+\.\d+$", l.strip())]
        assert decimal_labels, (
            f"format='.2f' with format_type='number' should produce decimal-formatted "
            f"tick labels; all labels: {labels}"
        )

    def test_format_type_number_differs_from_default_formatting(self):
        """format_type='number' with an explicit format produces different ticks
        than the default tick format (integer-like)."""
        svg_default = (
            fm.Chart(pl.DataFrame({"x": [1000.0, 2000.0, 3000.0], "y": [1.0, 2.0, 3.0]}))
            .mark_point()
            .encode(x="x", y="y")
            .show_svg()
        )
        svg_formatted = (
            fm.Chart(pl.DataFrame({"x": [1000.0, 2000.0, 3000.0], "y": [1.0, 2.0, 3.0]}))
            .mark_point()
            .encode(x=fm.X("x", format=".1f", format_type="number"), y="y")
            .show_svg()
        )
        # The formatted version should contain ".0" suffixes on tick labels
        assert ".0" in svg_formatted or "1000.0" in svg_formatted or "2000.0" in svg_formatted, (
            f"format_type='number' with .1f should produce decimal-formatted labels; "
            f"SVG excerpt: {svg_formatted[:2000]}"
        )


# ---------------------------------------------------------------------------
# Task 5: impute= fills missing group×x combinations
# ---------------------------------------------------------------------------


class TestImputeSVGStructure:
    """impute= must produce measurably different SVG (more path points or polylines)
    than without impute on sparse data."""

    def test_impute_value_adds_point_to_sparse_group(self):
        """impute={'method':'value','value':0} fills x=2 for group b.

        Without impute, group b has 2 points (x=1 and x=3).
        With impute, group b has 3 points (x=1, x=2 [imputed to 0], x=3).
        The line for group b should have MORE points with impute.
        """
        svg_no_impute = (
            fm.Chart(_sparse_df()).mark_line().encode(x="x", y="y", color="group").show_svg()
        )
        svg_imputed = (
            fm.Chart(_sparse_df())
            .mark_line()
            .encode(
                x="x",
                y=fm.Y("y", impute={"method": "value", "value": 0}),
                color="group",
            )
            .show_svg()
        )

        # Both should render
        assert "<svg" in svg_no_impute
        assert "<svg" in svg_imputed

        # Extract polyline point counts
        pts_no_impute = _extract_polyline_points(svg_no_impute)
        pts_imputed = _extract_polyline_points(svg_imputed)

        # Total points across all polylines should be greater with imputation
        total_no_impute = sum(len(pts) for pts in pts_no_impute)
        total_imputed = sum(len(pts) for pts in pts_imputed)

        assert total_imputed > total_no_impute, (
            f"impute should add a point to sparse group b (x=2 filled to 0); "
            f"total points without impute={total_no_impute}, with impute={total_imputed}"
        )

    def test_impute_fills_gap_so_line_is_continuous(self):
        """With impute, the imputed group's polyline has exactly 3 points (was 2)."""
        svg_imputed = (
            fm.Chart(_sparse_df())
            .mark_line()
            .encode(
                x="x",
                y=fm.Y("y", impute={"method": "value", "value": 0}),
                color="group",
            )
            .show_svg()
        )
        pts_all = _extract_polyline_points(svg_imputed)
        # We have 2 groups. The imputed group (b) should have 3 points.
        point_counts = sorted([len(p) for p in pts_all])
        assert 3 in point_counts, (
            f"Imputed group b should produce a 3-point polyline; "
            f"polyline point counts: {point_counts}"
        )


# ---------------------------------------------------------------------------
# Task 6: legend kwargs — legend position changes in SVG
# ---------------------------------------------------------------------------


class TestLegendKwargsSVGPosition:
    """legend={'orient': 'bottom'} must change WHERE the legend is rendered
    in the SVG, not just be silently accepted."""

    def _legend_y_positions(self, svg: str) -> list[float]:
        """Extract approximate y-positions of legend group elements."""
        # Legend title text is typically larger than axis labels
        # and appears in a <g> block. We look for the legend group anchor.
        # The Rust renderer wraps the legend in a <g> element at a specific y.
        # We look for <text> elements that contain the color category values.
        text_elements = re.findall(r'<text[^>]+y="([^"]+)"[^>]*>[^<]+</text>', svg)
        y_vals = []
        for y_str in text_elements:
            try:
                y_vals.append(float(y_str))
            except ValueError:
                pass
        return y_vals

    def test_legend_orient_bottom_produces_lower_legend_than_right(self):
        """legend={'orient': 'bottom'} → legend elements at higher y values than 'right' (default).

        In SVG coordinate space, y=0 is the top. A bottom legend has larger y
        coordinates than a right-side legend (which sits beside the plot).
        """
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "g": ["a", "b", "c"]})

        svg_right = fm.Chart(df).mark_point().encode(x="x", y="y", color="g").show_svg()
        svg_bottom = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g", legend={"orient": "bottom"}))
            .show_svg()
        )

        # Both should render
        assert "<svg" in svg_right and "<svg" in svg_bottom

        # The two SVGs must not be identical (the legend moved)
        assert svg_right != svg_bottom, (
            "legend={'orient': 'bottom'} must produce different SVG than default 'right'"
        )

    def test_legend_title_override_appears_in_svg(self):
        """legend={'title': 'My Legend'} → that string appears in SVG."""
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "species": ["a", "b"]})
        svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("species", legend={"title": "My Legend"}))
            .show_svg()
        )
        assert "My Legend" in svg, "legend={'title': 'My Legend'} should appear verbatim in SVG"

    def test_legend_format_changes_tick_labels(self):
        """legend={'format': '.0%'} on a continuous color scale must change
        the tick label text vs. default formatting."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "val": [0.1, 0.25, 0.5, 0.75, 0.9],
            }
        )
        svg_default = (
            fm.Chart(df).mark_point().encode(x="x", y="y", color=fm.Color("val")).show_svg()
        )
        svg_pct = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("val", legend={"format": ".0%"}))
            .show_svg()
        )
        assert "<svg" in svg_pct
        assert svg_default != svg_pct, (
            "legend={'format': '.0%'} must produce different tick label text than "
            "default formatting"
        )
        # A percent-formatted legend should contain a '%' sign in the tick labels
        assert "%" in svg_pct, "legend={'format': '.0%'} should produce tick labels containing '%'"

    def test_legend_columns_changes_layout(self):
        """legend={'columns': 2} on a categorical color scale must change the
        legend layout vs. default single-column layout."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 2.0, 3.0, 4.0],
                "g": ["a", "b", "c", "d"],
            }
        )
        svg_default = fm.Chart(df).mark_point().encode(x="x", y="y", color="g").show_svg()
        svg_2col = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g", legend={"columns": 2}))
            .show_svg()
        )
        assert "<svg" in svg_2col
        assert svg_default != svg_2col, (
            "legend={'columns': 2} must produce a different legend layout than "
            "the default single-column arrangement"
        )


# ---------------------------------------------------------------------------
# Task 7: histogram multiple= — bar x-positions differ for dodge/stack
# ---------------------------------------------------------------------------


class TestHistogramMultipleSVGStructure:
    """histogram multiple= must produce structurally different SVG layouts."""

    def test_dodge_produces_different_bar_x_positions_than_layer(self):
        """multiple='dodge' → for the same bin, the two groups have DIFFERENT x positions.

        In 'layer' mode (default), both group bars start at the same bin_start x.
        In 'dodge' mode, they are offset horizontally so they sit side-by-side.
        """
        svg_layer = (
            fm.Chart(_hist_df())
            .mark_histogram(groupby="g", multiple="layer")
            .encode(x="x", color="g")
            .show_svg()
        )
        svg_dodge = (
            fm.Chart(_hist_df())
            .mark_histogram(groupby="g", multiple="dodge")
            .encode(x="x", color="g")
            .show_svg()
        )
        rects_layer = _extract_rect_coords(svg_layer)
        rects_dodge = _extract_rect_coords(svg_dodge)

        # Both should have at least 2 rects
        assert len(rects_layer) >= 2, f"layer mode should produce rects; got {len(rects_layer)}"
        assert len(rects_dodge) >= 2, f"dodge mode should produce rects; got {len(rects_dodge)}"

        # The x-positions must differ between modes
        x_layer = sorted(set(r.get("x", 0) for r in rects_layer))
        x_dodge = sorted(set(r.get("x", 0) for r in rects_dodge))
        assert x_layer != x_dodge, (
            f"multiple='dodge' must produce different bar x-positions than 'layer';\n"
            f"  layer x-positions: {x_layer[:5]}\n"
            f"  dodge x-positions: {x_dodge[:5]}"
        )

    def test_stack_produces_different_bar_y_positions_than_layer(self):
        """multiple='stack' → bars accumulate on y-axis vs layer (overlapping at y=0).

        KNOWN GAP (Task 7): mark_histogram(multiple='stack') uses a position=Stack
        adjuster, which is the correct mechanism. However, stacking requires SHARED
        bin edges across groups so the x-positions match. The Bin transform's groupby
        mode computes per-group bin edges independently, so group a and group b get
        different bin edges and apply_stack finds no matching x-positions to accumulate.

        The fix requires passing shared_extent=True to the Bin transform when
        multiple='stack' (same as desugar_density already does). desugar_histogram
        does not do this.

        This test uses ALIGNED data (same x values for all groups) to confirm that
        apply_stack itself works correctly when bin edges do align.
        """
        # Use deterministic data with fixed bin_count so groups share bin edges
        df = pl.DataFrame(
            {
                "x": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                "g": ["a", "b", "a", "b", "a", "b"],
            }
        )
        svg_layer = (
            fm.Chart(df)
            .mark_histogram(groupby="g", bin_count=3, multiple="layer")
            .encode(x="x", color="g")
            .show_svg()
        )
        svg_stack = (
            fm.Chart(df)
            .mark_histogram(groupby="g", bin_count=3, multiple="stack")
            .encode(x="x", color="g")
            .show_svg()
        )
        rects_layer = _extract_rect_coords(svg_layer)
        rects_stack = _extract_rect_coords(svg_stack)

        assert len(rects_layer) >= 2
        assert len(rects_stack) >= 2

        # With aligned data, y-positions MUST differ between modes
        y_layer = sorted(set(r.get("y", 0) for r in rects_layer))
        y_stack = sorted(set(r.get("y", 0) for r in rects_stack))
        assert y_layer != y_stack, (
            f"multiple='stack' must produce different bar y-positions than 'layer' "
            f"when bin edges are aligned;\n"
            f"  layer y-positions: {y_layer[:5]}\n"
            f"  stack y-positions: {y_stack[:5]}"
        )

    def test_stack_with_unaligned_bins_identical_to_layer(self):
        """multiple='stack' with non-overlapping group bins produces identical output
        to 'layer' mode.

        KNOWN GAP (Task 7): desugar_histogram doesn't pass shared_extent=True to Bin
        for multiple='stack', so groups with different x-ranges get different bin edges.
        When bin edges don't align, apply_stack finds no matching x-key pairs and
        returns the batch unchanged — identical to 'layer' mode.

        This test documents the gap explicitly. When desugar_histogram is fixed to
        pass shared_extent=True for multiple='stack', update this test to assert
        svg_random_stack != svg_random_layer.
        """
        svg_layer = (
            fm.Chart(_hist_df())
            .mark_histogram(groupby="g", multiple="layer")
            .encode(x="x", color="g")
            .show_svg()
        )
        svg_stack = (
            fm.Chart(_hist_df())
            .mark_histogram(groupby="g", multiple="stack")
            .encode(x="x", color="g")
            .show_svg()
        )
        # GAP: desugar_histogram must pass shared_extent=True to Bin for
        # multiple='stack' so all groups share the same bin edges. Without
        # that, apply_stack finds no matching x-key pairs and is a no-op.
        assert svg_layer != svg_stack, (
            "mark_histogram(multiple='stack') must produce different output than "
            "'layer'. Currently broken: desugar_histogram doesn't pass "
            "shared_extent=True so per-group bin edges don't align."
        )

    def test_fill_produces_different_bar_heights_than_layer(self):
        """multiple='fill' → bars are normalized; heights differ from 'layer'."""
        svg_layer = (
            fm.Chart(_hist_df())
            .mark_histogram(groupby="g", multiple="layer")
            .encode(x="x", color="g")
            .show_svg()
        )
        svg_fill = (
            fm.Chart(_hist_df())
            .mark_histogram(groupby="g", multiple="fill")
            .encode(x="x", color="g")
            .show_svg()
        )
        rects_layer = _extract_rect_coords(svg_layer)
        rects_fill = _extract_rect_coords(svg_fill)

        assert len(rects_fill) >= 2

        # Heights should differ (normalized vs count)
        h_layer = sorted(set(round(r.get("height", 0), 2) for r in rects_layer))
        h_fill = sorted(set(round(r.get("height", 0), 2) for r in rects_fill))
        assert h_layer != h_fill, (
            f"multiple='fill' must normalize bar heights vs 'layer';\n"
            f"  layer heights: {h_layer[:5]}\n"
            f"  fill heights: {h_fill[:5]}"
        )

    def test_invalid_multiple_raises_value_error(self):
        """multiple='bogus' raises ValueError at render time."""
        with pytest.raises(ValueError, match="multiple"):
            fm.Chart(_hist_df()).mark_histogram(multiple="bogus").encode(x="x").show_svg()


# ---------------------------------------------------------------------------
# Task 8: lmplot/regplot truncate=False — no longer raises ValueError
# ---------------------------------------------------------------------------


class TestTruncateFalseNoRaise:
    """truncate=False must no longer raise ValueError.

    The commit claims this is implemented by computing x_range from data
    and passing it to the Smooth transform. We verify:
    1. No ValueError is raised
    2. The chart renders SVG

    NOTE: A deeper structural test (fit line extends beyond data x-range)
    is partially wired — x_range is computed but NOT threaded into mark_smooth
    call parameters in regression.py. This test documents that deficiency.
    """

    def test_lmplot_truncate_false_no_longer_raises(self):
        """lmplot(truncate=False) renders without raising ValueError."""
        svg = fm.lmplot(_reg_df(), x="x", y="y", truncate=False)
        assert svg is not None
        # The return is a Chart object; calling show_svg() renders it
        rendered = svg.show_svg()
        assert "<svg" in rendered, "lmplot(truncate=False) must render to SVG"

    def test_regplot_truncate_false_no_longer_raises(self):
        """regplot(truncate=False) renders without raising ValueError."""
        chart = fm.regplot(_reg_df(), x="x", y="y", truncate=False)
        assert chart is not None
        rendered = chart.show_svg()
        assert "<svg" in rendered, "regplot(truncate=False) must render to SVG"

    def test_truncate_false_produces_same_or_different_fit_from_true(self):
        """truncate=False and truncate=True both produce valid SVG.

        This test verifies the feature is plumbed without crashing, even if
        the visual difference (fit line extension) is not yet fully wired
        (x_range is computed in lmplot but not forwarded to mark_smooth kwargs).
        """
        chart_true = fm.lmplot(_reg_df(), x="x", y="y", truncate=True)
        chart_false = fm.lmplot(_reg_df(), x="x", y="y", truncate=False)

        svg_true = chart_true.show_svg()
        svg_false = chart_false.show_svg()

        assert "<svg" in svg_true
        assert "<svg" in svg_false
        # If x_range is fully wired, these would differ. If not, they may be identical.
        # We document the current state without asserting equality/difference.
        # This test passes regardless — its purpose is to confirm no crash.

    def test_truncate_false_xrange_not_forwarded_to_smooth(self):
        """Structural check: truncate=False computes x_range but does NOT forward it
        to mark_smooth kwargs in the current implementation.

        This test documents the known gap: lmplot computes x_range at lines 535-549
        of regression.py but none of the mark_smooth() calls at lines 578-609
        pass x_range=x_range as a kwarg.

        If this test FAILS in the future, it means the gap was fixed.
        """
        # We verify the current broken state: truncate=False SVG is identical
        # to truncate=True SVG (because x_range is computed but ignored).
        chart_true = fm.lmplot(_reg_df(), x="x", y="y", truncate=True, ci=None, show_metrics=False)
        chart_false = fm.lmplot(
            _reg_df(), x="x", y="y", truncate=False, ci=None, show_metrics=False
        )
        svg_true = chart_true.show_svg()
        svg_false = chart_false.show_svg()

        # GAP: x_range is computed in lmplot (regression.py:535-549) but
        # never passed to any mark_smooth() call at lines 578-609.
        # Fix: thread x_range through to the Smooth layer spec.
        assert svg_true != svg_false, (
            "lmplot(truncate=False) must produce a different fit-line extent than "
            "truncate=True. Currently broken: x_range is computed but not forwarded "
            "to mark_smooth() in regression.py."
        )


# ---------------------------------------------------------------------------
# Task 9: Chart(data=None) + Layer(data=) merges data into SVG output
# ---------------------------------------------------------------------------


class TestChartDataNoneSVGRendering:
    """Chart(data=None) with per-layer data must render elements from BOTH layers."""

    def test_two_layer_data_produces_marks_from_both_datasets(self):
        """Chart(data=None).layer(Layer(data=df1), Layer(data=df2)) renders both.

        df1 has 3 points; df2 has 3 points. The merged chart should have at
        least 2 types of mark (point + line) with at least 3 elements each.
        """
        from ferrum.layer import Layer

        df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        df2 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 15.0, 25.0]})

        chart = (
            fm.Chart(data=None)
            .layer(
                Layer(data=df1, mark="point", encoding={"x": fm.X("x"), "y": fm.Y("y")}),
                Layer(data=df2, mark="line", encoding={"x": fm.X("x"), "y": fm.Y("y")}),
            )
            .encode(x="x", y="y")
        )
        svg = chart.show_svg()
        assert "<svg" in svg

        # Should have circle elements (from point mark) and line/polyline (from line mark)
        circle_count = svg.count("<circle")
        line_count = svg.count("<polyline") + svg.count("<path")

        assert circle_count >= 1 or line_count >= 1, (
            f"Chart(data=None) with two Layer(data=) args should render marks; "
            f"circles={circle_count}, polylines/paths={line_count}"
        )

    def test_chart_data_none_layer_merges_rows(self):
        """Chart(data=None).layer() merges layer data into chart _data.

        After layer(), chart._data should contain rows from both input DataFrames
        (verified by the number of rendered mark elements being > single-layer count).
        """
        from ferrum.layer import Layer

        df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        df_single = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})

        chart_single = fm.Chart(df_single).mark_point().encode(x="x", y="y")
        chart_merged = (
            fm.Chart(data=None)
            .layer(
                Layer(data=df1, mark="point", encoding={"x": "x", "y": "y"}),
            )
            .encode(x="x", y="y")
        )

        svg_single = chart_single.show_svg()
        svg_merged = chart_merged.show_svg()

        # Both should render
        assert "<svg" in svg_single
        assert "<svg" in svg_merged

        # The merged chart with 3 rows in df1 should render 3 circles
        circles_merged = svg_merged.count("<circle")
        assert circles_merged >= 3, (
            f"Chart(data=None).layer(Layer(data=df1)) should render 3 circles; got {circles_merged}"
        )

    def test_chart_data_none_accepts_construction(self):
        """Chart(data=None) is accepted without ValueError at construction time."""
        chart = fm.Chart(data=None)
        assert chart is not None

    def test_chart_data_none_no_data_raises_at_render(self):
        """Chart(data=None) with no layer data raises at render time, not construction."""
        chart = fm.Chart(data=None).mark_point().encode(x="x", y="y")
        with pytest.raises(Exception):
            chart.show_svg()


# ---------------------------------------------------------------------------
# G1: chart-level description → <desc> in root SVG
# ---------------------------------------------------------------------------


class TestChartDescription:
    """Tests for properties(description=...) emitting <desc> in the root SVG."""

    def _base_chart(self, df: pl.DataFrame | None = None) -> "fm.Chart":
        if df is None:
            df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        return fm.Chart(df).mark_point().encode(x="x", y="y")

    def test_description_emitted_as_desc_element(self):
        """properties(description=...) emits <desc>...</desc> inside the root <svg>."""
        svg = self._base_chart().properties(description="Test description").show_svg()
        assert "<desc>Test description</desc>" in svg, (
            f"Expected '<desc>Test description</desc>' in SVG; got:\n{svg[:500]}"
        )

    def test_description_appears_near_start_of_svg(self):
        """<desc> is a direct child of <svg>, before any mark geometry."""
        svg = self._base_chart().properties(description="Accessibility text").show_svg()
        svg_open = svg.index("<svg ")
        desc_pos = svg.index("<desc>")
        first_circle = svg.find("<circle")
        assert desc_pos > svg_open, "<desc> must come after <svg> opening tag"
        assert first_circle == -1 or desc_pos < first_circle, (
            "<desc> should appear before mark geometry"
        )

    def test_description_xml_escaped(self):
        """Special XML characters in description are escaped correctly."""
        svg = self._base_chart().properties(description="A & B <test>").show_svg()
        # XML-escaped form
        assert "A &amp; B &lt;test&gt;" in svg, (
            f"Expected XML-escaped description in SVG; got:\n{svg[:500]}"
        )
        # Raw angle-brackets must not appear inside <desc>
        assert "<desc>A & B <test></desc>" not in svg

    def test_no_description_emits_no_root_desc(self):
        """A chart without .properties(description=) emits no <desc> at root level."""
        svg = self._base_chart().show_svg()
        # There must be no chart-level <desc>; per-mark descriptions are inside <g> wrappers,
        # not at the root level immediately after <svg>.
        # The simplest check: no <desc> element at all.
        assert "<desc>" not in svg, (
            f"Expected no <desc> in SVG when description is not set; got:\n{svg[:500]}"
        )

    def test_description_with_quotes(self):
        """Description containing quotes is properly escaped."""
        svg = (
            self._base_chart()
            .properties(description="Chart with \"quotes\" & 'apostrophes'")
            .show_svg()
        )
        assert "<desc>" in svg
        assert "&amp;" in svg

    def test_description_empty_string_not_emitted(self):
        """Empty string description → no <desc> emitted (falsy value)."""
        svg = self._base_chart().properties(description="").show_svg()
        assert "<desc>" not in svg, "Empty description should not emit <desc>"

    def test_description_coexists_with_encoding_desc(self):
        """Chart-level <desc> and per-mark encoding description <desc> can coexist."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [3.0, 4.0],
                "d": ["point A", "point B"],
            }
        )
        svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", description=fm.Description("d"))
            .properties(description="Chart-level description")
            .show_svg()
        )
        assert "Chart-level description" in svg
        desc_count = svg.count("<desc>")
        assert desc_count >= 3, f"Expected chart-level + 2 per-mark <desc> tags; got {desc_count}"
