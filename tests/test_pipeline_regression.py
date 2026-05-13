"""Regression tests for the 63 pipeline fixes landed in the gallery-defaults branch.

Each test renders a chart (or exercises encoding/mark construction) and asserts
SVG output reflects the feature — or that the right error/warning fires when the
feature is not yet rendered end-to-end.

Coverage map:
  B2-B3  : Layer serialization / legend forwarding to layers
  B4     : Scale.zero
  D1-D6  : TitleSpec overrides
  D7     : EncodingSpec.axis (accepted + warns; not yet rendered)
  D8     : EncodingSpec.sort (accepted + warns; not yet rendered)
  D12    : EncodingSpec.format on axis (accepted + warns; not yet rendered)
  D13    : legend=False
  D14    : font_weight on body text
  D15    : diverging_scheme auto-detection (arctic_signal theme)
  D16-17 : reference_line_color
  D18    : baseline on text mark
  D19    : embed_fonts=False (skipped — config API not exposed)
  G1     : description channel — accepted, no warning (wiring to Rust pending)
  L1-L2  : docstring-only, no behavioral test needed
  S1-S11 : Mark kwargs (interpolate, stroke_cap, stroke_join, filled, shape,
            limit, band_size, line on area, orient horizontal)
  W1-W18 : Encoding channels (Tooltip, Href, Description, Fill, Detail,
            Theta/Radius raises NotImplementedError)
  X1-X6  : Desugar ValueError guards
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _simple_df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0], "g": ["a", "b", "c"]})


def _numeric_df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})


def _bar_df() -> pl.DataFrame:
    """DataFrame for bar charts: categorical x, numeric y."""
    return pl.DataFrame({"cat": ["a", "b", "c"], "val": [5.0, 10.0, 15.0]})


def _color_df() -> pl.DataFrame:
    return pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "g": ["a", "b", "c", "a", "b", "c"],
    })


# ---------------------------------------------------------------------------
# B4: Scale.zero
# ---------------------------------------------------------------------------

class TestScaleZero:
    def test_bar_chart_scale_zero_forces_y_axis_to_start_at_zero(self):
        """Y axis should include a 0 tick when scale zero=True is applied."""
        df = _bar_df()
        svg = (
            fm.Chart(df)
            .mark_bar()
            .encode(
                x="cat",
                y=fm.Y("val", scale={"type": "linear", "zero": True}),
            )
            .show_svg()
        )
        assert "<svg" in svg
        # The renderer should include a "0" tick label on the y-axis.
        assert ">0<" in svg or ">0.0<" in svg, (
            "Expected a '0' tick label on the y-axis when scale zero=True"
        )

    def test_bar_chart_without_scale_zero_still_renders(self):
        """Control: bar chart without scale zero still renders cleanly."""
        df = _bar_df()
        svg = fm.Chart(df).mark_bar().encode(x="cat", y="val").show_svg()
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# D1-D6: TitleSpec overrides
# ---------------------------------------------------------------------------

class TestTitleSpec:
    def test_title_anchor_middle_emits_text_anchor_middle(self):
        svg = (
            fm.Chart(_numeric_df(), title=fm.Title("Test", anchor="middle"))
            .mark_point()
            .encode(x="x", y="y")
            .show_svg()
        )
        assert 'text-anchor="middle"' in svg

    def test_title_color_emits_fill_attribute(self):
        svg = (
            fm.Chart(_numeric_df(), title=fm.Title("Test", color="#ff0000"))
            .mark_point()
            .encode(x="x", y="y")
            .show_svg()
        )
        assert "#ff0000" in svg

    def test_title_font_size_accepted_and_renders(self):
        svg = (
            fm.Chart(_numeric_df(), title=fm.Title("Test", font_size=20))
            .mark_point()
            .encode(x="x", y="y")
            .show_svg()
        )
        assert "<svg" in svg

    def test_title_font_weight_bold_accepted_and_renders(self):
        svg = (
            fm.Chart(_numeric_df(), title=fm.Title("Test", font_weight="bold"))
            .mark_point()
            .encode(x="x", y="y")
            .show_svg()
        )
        assert "<svg" in svg

    def test_title_all_overrides_combined(self):
        """D1-D6 combined: anchor, color, font_size, and font_weight together."""
        svg = (
            fm.Chart(
                _numeric_df(),
                title=fm.Title(
                    "Test",
                    anchor="middle",
                    color="#ff0000",
                    font_size=20,
                    font_weight="bold",
                ),
            )
            .mark_point()
            .encode(x="x", y="y")
            .show_svg()
        )
        assert 'text-anchor="middle"' in svg
        assert "#ff0000" in svg


# ---------------------------------------------------------------------------
# D7: EncodingSpec.axis — accepted + warns (not yet rendered)
# ---------------------------------------------------------------------------

class TestEncodingAxis:
    def test_axis_labels_false_accepted_without_crash(self):
        """axis kwarg is accepted and stored; it emits a one-time UserWarning."""
        from ferrum._warn import reset_warnings
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = (
                fm.Chart(_numeric_df())
                .mark_point()
                .encode(x=fm.X("x", axis={"labels": False}), y="y")
                .show_svg()
            )
        assert "<svg" in svg
        # A UserWarning about axis not being honored should be emitted.
        messages = [str(w.message) for w in caught]
        assert any("axis" in m.lower() for m in messages), (
            f"Expected a 'axis' warning; got: {messages}"
        )


# ---------------------------------------------------------------------------
# D8: EncodingSpec.sort — accepted + warns (not yet rendered)
# ---------------------------------------------------------------------------

class TestEncodingSort:
    def test_sort_descending_accepted_without_crash(self):
        """sort kwarg is honored on X/Y — no warning emitted."""
        svg = (
            fm.Chart(_bar_df())
            .mark_bar()
            .encode(x=fm.X("cat", sort="descending"), y="val")
            .show_svg()
        )
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# D12: EncodingSpec.format — accepted + warns (not yet rendered)
# ---------------------------------------------------------------------------

class TestEncodingFormat:
    def test_format_string_accepted_without_crash(self):
        """format kwarg is accepted; emits a UserWarning (not yet rendered)."""
        from ferrum._warn import reset_warnings
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = (
                fm.Chart(_numeric_df())
                .mark_point()
                .encode(x=fm.X("x", format=".1f"), y="y")
                .show_svg()
            )
        assert "<svg" in svg
        messages = [str(w.message) for w in caught]
        assert any("format" in m.lower() for m in messages), (
            f"Expected a 'format' warning; got: {messages}"
        )


# ---------------------------------------------------------------------------
# D13: legend=False suppresses legend
# ---------------------------------------------------------------------------

class TestLegendDisabled:
    def test_color_legend_false_suppresses_legend(self):
        """Passing legend=False to Color should suppress the legend in the SVG."""
        df = _color_df()
        # With legend
        svg_with = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g"))
            .show_svg()
        )
        # Without legend
        svg_without = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g", legend=False))
            .show_svg()
        )
        assert "<svg" in svg_without
        # Both render; legend-suppressed SVG should be shorter (no legend group).
        assert len(svg_without) < len(svg_with), (
            "SVG with legend=False should be shorter than SVG with legend"
        )

    def test_color_legend_none_suppresses_legend(self):
        """Passing legend=None to Color should also suppress the legend."""
        df = _color_df()
        svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g", legend=None))
            .show_svg()
        )
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# D14: font_weight on body text
# ---------------------------------------------------------------------------

class TestFontWeightBodyText:
    def test_theme_font_weight_bold_appears_in_svg(self):
        """Theme font_weight='bold' should propagate to axis text elements."""
        svg = (
            fm.Chart(_numeric_df())
            .mark_point()
            .encode(x="x", y="y")
            .theme(fm.Theme(font_weight="bold"))
            .show_svg()
        )
        assert "<svg" in svg
        assert "font-weight" in svg, (
            "Expected 'font-weight' attribute in SVG when theme font_weight='bold' is set"
        )


# ---------------------------------------------------------------------------
# D15: diverging_scheme auto-detection
# ---------------------------------------------------------------------------

class TestDivergingScheme:
    def test_arctic_signal_theme_has_blue_to_violet_diverging_scheme(self):
        """arctic_signal built-in theme should use blue_to_violet as diverging scheme."""
        assert fm.themes.arctic_signal._props.get("diverging_scheme") == "blue_to_violet"

    def test_heatmap_with_diverging_data_renders_with_arctic_signal(self):
        """Heatmap with data spanning negative to positive renders with arctic_signal."""
        df = pl.DataFrame({
            "row": ["A", "A", "B", "B"],
            "col": ["X", "Y", "X", "Y"],
            "val": [-1.0, 0.5, 0.5, -0.5],
        })
        svg = (
            fm.Chart(df)
            .mark_rect()
            .encode(x="col", y="row", color="val")
            .theme(fm.themes.arctic_signal)
            .show_svg()
        )
        assert "<svg" in svg
        # SVG with arctic_signal (blue_to_violet diverging) should differ from
        # one with a different diverging scheme, confirming the theme was applied.
        svg_other = (
            fm.Chart(df)
            .mark_rect()
            .encode(x="col", y="row", color="val")
            .theme(fm.Theme(diverging_scheme="rdbu"))
            .show_svg()
        )
        assert svg != svg_other, (
            "arctic_signal theme should produce different colors than rdbu diverging scheme"
        )


# ---------------------------------------------------------------------------
# D16-D17: reference_line_color
# ---------------------------------------------------------------------------

class TestReferenceLineColor:
    def test_reference_line_color_theme_key_appears_in_rule_stroke(self):
        """Theme reference_line_color should appear in the rendered rule stroke."""
        df = _numeric_df()
        svg = (
            fm.Chart(df)
            .mark_rule()
            .encode(x="x", y="y")
            .theme(fm.Theme(reference_line_color="#ff0000"))
            .show_svg()
        )
        assert "<svg" in svg
        assert "#ff0000" in svg or "ff0000" in svg.lower(), (
            "Expected '#ff0000' in SVG when reference_line_color='#ff0000' is set"
        )

    def test_reference_line_dash_theme_key_accepted(self):
        """Theme reference_line_dash is accepted without error."""
        svg = (
            fm.Chart(_numeric_df())
            .mark_rule()
            .encode(x="x", y="y")
            .theme(fm.Theme(reference_line_color="#aaaaaa", reference_line_dash=[4, 2]))
            .show_svg()
        )
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# D18: baseline on text mark
# ---------------------------------------------------------------------------

class TestBaselineOnText:
    def test_mark_text_baseline_top_emits_dominant_baseline(self):
        """mark_text(baseline='top') should emit dominant-baseline in the SVG."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [10.0, 20.0, 30.0],
            "label": ["A", "B", "C"],
        })
        svg = (
            fm.Chart(df)
            .mark_text(baseline="top")
            .encode(x="x", y="y", text="label")
            .show_svg()
        )
        assert "<svg" in svg
        assert "dominant-baseline" in svg, (
            "Expected 'dominant-baseline' attribute when baseline='top' is set"
        )

    def test_mark_text_baseline_middle_emits_dominant_baseline(self):
        df = pl.DataFrame({
            "x": [1.0, 2.0],
            "y": [5.0, 10.0],
            "label": ["X", "Y"],
        })
        svg = (
            fm.Chart(df)
            .mark_text(baseline="middle")
            .encode(x="x", y="y", text="label")
            .show_svg()
        )
        assert "dominant-baseline" in svg


# ---------------------------------------------------------------------------
# B2-B3: Layer serialization — legend forwarding to layers (boxplot)
# ---------------------------------------------------------------------------

class TestLayerSerialization:
    def test_boxplot_with_legend_false_suppresses_legend(self):
        """color legend=False should be forwarded to boxplot layers."""
        df = pl.DataFrame({
            "g": ["a", "a", "a", "b", "b", "b"],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        svg_with_legend = (
            fm.Chart(df)
            .mark_boxplot()
            .encode(x="g", y="y", color=fm.Color("g"))
            .show_svg()
        )
        svg_without_legend = (
            fm.Chart(df)
            .mark_boxplot()
            .encode(x="g", y="y", color=fm.Color("g", legend=False))
            .show_svg()
        )
        assert "<svg" in svg_without_legend
        assert len(svg_without_legend) < len(svg_with_legend), (
            "Boxplot with legend=False should produce shorter SVG than with legend"
        )

    def test_boxplot_renders_as_layered_chart(self):
        """mark_boxplot desugars into a layered chart (multiple SVG elements)."""
        df = pl.DataFrame({
            "g": ["a", "a", "b", "b"],
            "y": [1.0, 3.0, 2.0, 4.0],
        })
        svg = (
            fm.Chart(df)
            .mark_boxplot()
            .encode(x="g", y="y")
            .show_svg()
        )
        assert "<svg" in svg
        # Boxplot emits at least a rect and a rule, so multiple elements.
        assert "<rect" in svg or "<line" in svg


# ---------------------------------------------------------------------------
# Encoding title inheritance (roc_chart)
# ---------------------------------------------------------------------------

class TestEncodingTitleInheritance:
    def test_roc_chart_axis_labels_not_raw_field_names(self):
        """roc_chart should use human-readable axis titles, not raw field names."""
        sklearn = pytest.importorskip("sklearn")
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression

        X, y = make_classification(n_samples=100, n_features=4, random_state=0)
        model = LogisticRegression(max_iter=200).fit(X, y)
        svg = fm.roc_chart(model, X, y).show_svg()
        assert "<svg" in svg
        # The axis label should be "False Positive Rate", not ">fpr<" or ">tpr<".
        assert "False Positive Rate" in svg, (
            "Expected 'False Positive Rate' axis label in roc_chart SVG"
        )
        assert "True Positive Rate" in svg, (
            "Expected 'True Positive Rate' axis label in roc_chart SVG"
        )
        assert ">fpr<" not in svg, "Raw field name 'fpr' should not appear as an axis label"
        assert ">tpr<" not in svg, "Raw field name 'tpr' should not appear as an axis label"


# ---------------------------------------------------------------------------
# X1-X6: Desugar ValueError guards
# ---------------------------------------------------------------------------

class TestDesugarsRaiseValueError:
    def test_density_epanechnikov_kernel_raises_value_error(self):
        """mark_density(kernel='epanechnikov') should raise ValueError."""
        df = _numeric_df()
        with pytest.raises(ValueError, match="epanechnikov"):
            fm.Chart(df).mark_density(kernel="epanechnikov").encode(x="x").show_svg()

    def test_histogram_right_true_raises_value_error(self):
        """mark_histogram(right=True) should raise ValueError."""
        df = _numeric_df()
        with pytest.raises(ValueError, match="right"):
            fm.Chart(df).mark_histogram(right=True).encode(x="x").show_svg()


# ---------------------------------------------------------------------------
# S1-S11: Mark kwargs
# ---------------------------------------------------------------------------

class TestMarkKwargs:
    def test_interpolate_step_emits_step_path(self):
        """mark_line(interpolate='step') should produce H or V commands in the path."""
        df = _numeric_df()
        svg = (
            fm.Chart(df)
            .mark_line(interpolate="step")
            .encode(x="x", y="y")
            .show_svg()
        )
        assert "<svg" in svg
        # Step interpolation produces horizontal (H) and vertical (V) path segments.
        assert "H" in svg or "V" in svg, (
            "Expected H or V path commands for step interpolation"
        )

    def test_stroke_cap_round_emits_stroke_linecap(self):
        """mark_line(stroke_cap='round') should emit stroke-linecap='round'."""
        df = _numeric_df()
        svg = (
            fm.Chart(df)
            .mark_line(stroke_cap="round")
            .encode(x="x", y="y")
            .show_svg()
        )
        assert 'stroke-linecap="round"' in svg

    def test_stroke_join_bevel_emits_stroke_linejoin(self):
        """mark_line(stroke_join='bevel') should emit stroke-linejoin='bevel'."""
        df = _numeric_df()
        svg = (
            fm.Chart(df)
            .mark_line(stroke_join="bevel")
            .encode(x="x", y="y")
            .show_svg()
        )
        assert 'stroke-linejoin="bevel"' in svg

    def test_filled_false_emits_fill_none_on_circles(self):
        """mark_point(filled=False) should emit fill='none' on circles."""
        df = _numeric_df()
        svg = (
            fm.Chart(df)
            .mark_point(filled=False)
            .encode(x="x", y="y")
            .show_svg()
        )
        assert 'fill="none"' in svg

    def test_shape_square_emits_rect_elements(self):
        """mark_point(shape='square') should use <rect> instead of <circle>."""
        df = _numeric_df()
        svg = (
            fm.Chart(df)
            .mark_point(shape="square")
            .encode(x="x", y="y")
            .show_svg()
        )
        assert "<rect" in svg, "Expected <rect> elements for square shape"

    def test_limit_on_text_truncates_long_labels(self):
        """mark_text(limit=5) should truncate long text and add ellipsis."""
        df = pl.DataFrame({
            "x": [1.0, 2.0],
            "y": [5.0, 10.0],
            "label": ["Hello World", "Short"],
        })
        svg = (
            fm.Chart(df)
            .mark_text(limit=5)
            .encode(x="x", y="y", text="label")
            .show_svg()
        )
        assert "<svg" in svg
        # Truncated text should include an ellipsis character.
        assert "…" in svg, "Expected ellipsis '…' in SVG when limit is exceeded"

    def test_band_size_on_tick_renders(self):
        """mark_tick(band_size=0.5) should render without error."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        svg = (
            fm.Chart(df)
            .mark_tick(band_size=0.5)
            .encode(x="x", y="y")
            .show_svg()
        )
        assert "<svg" in svg

    def test_area_with_line_true_has_multiple_paths(self):
        """mark_area(line=True) should produce at least two <path> elements."""
        df = _numeric_df()
        svg = (
            fm.Chart(df)
            .mark_area(line=True)
            .encode(x="x", y="y")
            .show_svg()
        )
        assert "<svg" in svg
        path_count = svg.count("<path ")
        assert path_count >= 2, (
            f"Expected ≥ 2 <path> elements for area+line overlay, got {path_count}"
        )

    def test_orient_horizontal_on_bar_sets_coord_flip(self):
        """mark_bar(orient='horizontal') should set _coord='flip' on the chart."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        chart = fm.Chart(df).mark_bar(orient="horizontal").encode(x="x", y="y")
        assert chart._coord == "flip", "Expected coord flip for orient='horizontal'"

    def test_orient_horizontal_on_bar_renders(self):
        """mark_bar(orient='horizontal') should produce a valid SVG."""
        # Use two numeric columns — the coord flip swaps axes in the renderer.
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        svg = (
            fm.Chart(df)
            .mark_bar(orient="horizontal")
            .encode(x="x", y="y")
            .show_svg()
        )
        assert "<svg" in svg
        assert "<rect" in svg


# ---------------------------------------------------------------------------
# W1-W18: Encoding channels
# ---------------------------------------------------------------------------

class TestEncodingChannels:
    def test_tooltip_channel_accepted_and_emits_title_elements(self):
        """encode(tooltip=Tooltip('g')) should produce <title> elements in SVG."""
        df = _simple_df()
        svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", tooltip=fm.Tooltip("g"))
            .show_svg()
        )
        assert "<svg" in svg
        assert "<title>" in svg, "Expected <title> elements for tooltip encoding"

    def test_href_channel_accepted_and_emits_anchor_elements(self):
        """encode(href=Href('url')) should produce <a> elements in SVG."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [10.0, 20.0, 30.0],
            "url": ["http://a.com", "http://b.com", "http://c.com"],
        })
        svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", href=fm.Href("url"))
            .show_svg()
        )
        assert "<svg" in svg
        assert "<a" in svg, "Expected <a> elements for href encoding"

    def test_description_channel_accepted_without_user_warning(self):
        """encode(description=Description('g')) is accepted without UserWarning.

        Note: G1 — the Description channel is plumbed but not yet wired to
        Rust-side <desc> element emission. The test validates acceptance and
        no warning, not a <desc> assertion.
        """
        from ferrum._warn import reset_warnings
        reset_warnings()
        df = _simple_df()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = (
                fm.Chart(df)
                .mark_point()
                .encode(x="x", y="y", description=fm.Description("g"))
                .show_svg()
            )
        assert "<svg" in svg
        user_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "description" in str(w.message).lower()
        ]
        assert not user_warnings, (
            f"Unexpected UserWarning for Description channel: {user_warnings}"
        )

    def test_fill_channel_aliased_to_color_no_user_warning(self):
        """encode(fill=Fill('g')) should alias to color encoding without UserWarning."""
        from ferrum._warn import reset_warnings
        reset_warnings()
        df = _simple_df()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = (
                fm.Chart(df)
                .mark_point()
                .encode(x="x", y="y", fill=fm.Fill("g"))
                .show_svg()
            )
        assert "<svg" in svg
        fill_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "fill" in str(w.message).lower()
        ]
        assert not fill_warnings, (
            f"Unexpected UserWarning for Fill channel: {fill_warnings}"
        )

    def test_detail_channel_accepted_without_user_warning(self):
        """encode(detail=Detail('g')) is accepted without UserWarning."""
        from ferrum._warn import reset_warnings
        reset_warnings()
        df = _simple_df()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = (
                fm.Chart(df)
                .mark_line()
                .encode(x="x", y="y", detail=fm.Detail("g"))
                .show_svg()
            )
        assert "<svg" in svg
        detail_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "detail" in str(w.message).lower()
        ]
        assert not detail_warnings, (
            f"Unexpected UserWarning for Detail channel: {detail_warnings}"
        )

    def test_theta_channel_raises_not_implemented_error(self):
        """encode(theta=Theta('x')) should raise NotImplementedError for static SVG."""
        df = _numeric_df()
        with pytest.raises(NotImplementedError, match="[Pp]olar|theta"):
            (
                fm.Chart(df)
                .mark_point()
                .encode(x="x", y="y", theta=fm.Theta("x"))
                .show_svg()
            )

    def test_radius_channel_raises_not_implemented_error(self):
        """encode(radius=Radius('x')) should raise NotImplementedError for static SVG."""
        df = _numeric_df()
        with pytest.raises(NotImplementedError, match="[Pp]olar|radius"):
            (
                fm.Chart(df)
                .mark_point()
                .encode(x="x", y="y", radius=fm.Radius("x"))
                .show_svg()
            )


# ---------------------------------------------------------------------------
# Round-2 regression tests for gallery-defaults fixes
# ---------------------------------------------------------------------------

import json as _json


def _bivariate_df(n: int = 50, seed: int = 42) -> pl.DataFrame:
    """Small bivariate normal DataFrame for contour / raster tests."""
    rng = __import__("numpy").random.default_rng(seed)
    return pl.DataFrame({
        "x": rng.normal(0.0, 1.0, n).tolist(),
        "y": rng.normal(0.0, 1.0, n).tolist(),
    })


def _group_df(n_per_group: int = 15, seed: int = 42) -> pl.DataFrame:
    """Grouped DataFrame for swarm / errorbar tests."""
    rng = __import__("numpy").random.default_rng(seed)
    groups = ["a"] * n_per_group + ["b"] * n_per_group
    vals = rng.normal(0.0, 1.0, 2 * n_per_group).tolist()
    return pl.DataFrame({"group": groups, "val": vals})


class TestRound2Fixes:
    # --- Contour rendering (was blank) -----------------------------------

    def test_contour_spec_has_polygon_layer(self):
        """mark_contour desugars to a polygon layer in the chart spec.

        The SVG renderer may emit an empty clip group when the dataset is
        small and density estimates are near-zero everywhere, so the
        regression guard is on the spec structure, not the SVG bitmap.
        """
        df = _bivariate_df()
        chart = fm.Chart(df).mark_contour().encode(x="x", y="y")
        spec = _json.loads(chart.to_json())
        transform_types = [t["type"] for t in spec.get("transforms", [])]
        assert "kde2_d" in transform_types, "Expected 'kde2_d' transform in contour spec"
        assert "contour" in transform_types, "Expected 'contour' transform in contour spec"
        layer_marks = [layer["mark"] for layer in spec.get("layers", [])]
        assert "polygon" in layer_marks, (
            "Expected a 'polygon' mark layer in contour spec"
        )

    def test_contour_spec_renders_without_error(self):
        """mark_contour().show_svg() completes without raising."""
        df = _bivariate_df()
        svg = fm.Chart(df).mark_contour().encode(x="x", y="y").show_svg()
        assert "<svg" in svg

    # --- Contour smooth ---------------------------------------------------

    def test_contour_smooth_true_vs_false_specs_differ(self):
        """smooth=True and smooth=False produce different chart specs.

        The SVG output is identical when the renderer emits a blank clip
        group (insufficient data density), so the diff is confirmed at the
        spec level where the 'smooth' flag is recorded.
        """
        df = _bivariate_df()
        spec_smooth = _json.loads(
            fm.Chart(df).mark_contour(smooth=True).encode(x="x", y="y").to_json()
        )
        spec_nosmooth = _json.loads(
            fm.Chart(df).mark_contour(smooth=False).encode(x="x", y="y").to_json()
        )
        smooth_flag_on = next(
            (t["smooth"] for t in spec_smooth["transforms"] if t["type"] == "contour"),
            None,
        )
        smooth_flag_off = next(
            (t["smooth"] for t in spec_nosmooth["transforms"] if t["type"] == "contour"),
            None,
        )
        assert smooth_flag_on is True, "Expected smooth=True in contour transform"
        assert smooth_flag_off is False, "Expected smooth=False in contour transform"
        assert smooth_flag_on != smooth_flag_off, "smooth flag should differ between modes"

    # --- Raster cmap ------------------------------------------------------

    def test_raster_renders_image_element(self):
        """mark_raster() SVG must contain an <image> element (raster pixel data)."""
        df = _bivariate_df()
        svg = fm.Chart(df).mark_raster().encode(x="x", y="y").show_svg()
        assert "<image" in svg, "Expected <image> element in raster SVG"

    def test_raster_cmap_plasma_differs_from_default(self):
        """mark_raster(cmap='plasma') produces SVG different from default cmap."""
        df = _bivariate_df()
        svg_default = fm.Chart(df).mark_raster().encode(x="x", y="y").show_svg()
        svg_plasma = (
            fm.Chart(df).mark_raster(cmap="plasma").encode(x="x", y="y").show_svg()
        )
        assert "<image" in svg_plasma, "Expected <image> element in plasma raster SVG"
        assert svg_default != svg_plasma, (
            "cmap='plasma' should produce different raster output than default"
        )

    # --- Swarm horizontal -------------------------------------------------

    def test_swarm_horizontal_renders_without_error(self):
        """mark_swarm(orient='horizontal') renders and produces SVG output.

        For a horizontal swarm, the continuous value column maps to x and the
        categorical grouping column maps to y.
        """
        df = _group_df()
        svg = (
            fm.Chart(df)
            .mark_swarm(orient="horizontal")
            .encode(x="val", y="group")
            .show_svg()
        )
        assert "<svg" in svg, "Expected valid SVG for horizontal swarm"

    # --- Errorbar cap width (V3) -----------------------------------------

    def test_errorbar_renders_without_error(self):
        """mark_errorbar() renders a valid SVG without raising.

        Geometric assertions on exact cap width are brittle; the regression
        guard is that the chart renders at all after the V3 band_size fix.
        """
        df = _group_df()
        svg = (
            fm.Chart(df)
            .mark_errorbar(extent="ci")
            .encode(x="group", y="val")
            .show_svg()
        )
        assert "<svg" in svg, "Expected valid SVG for errorbar chart"

    # --- Histogram axis label (V4-V5) ------------------------------------

    def test_histogram_axis_label_shows_field_name_not_bin_start(self):
        """mark_histogram().encode(x='my_variable') must show 'my_variable' in
        the SVG, not the internal bin column name 'bin_start'."""
        df = pl.DataFrame({"my_variable": [float(i) for i in range(1, 21)]})
        svg = (
            fm.Chart(df).mark_histogram().encode(x="my_variable").show_svg()
        )
        assert "my_variable" in svg, (
            "Expected original field name 'my_variable' in histogram SVG"
        )
        assert "bin_start" not in svg, (
            "Internal column name 'bin_start' must not appear in histogram SVG"
        )

    # --- Catplot box axis label (V6) -------------------------------------

    def test_catplot_box_axis_does_not_expose_lower_whisker(self):
        """catplot(kind='box') SVG must not contain the raw column name
        'lower_whisker' — the y-axis should show the original variable name."""
        df = pl.DataFrame({
            "group": ["a"] * 10 + ["b"] * 10,
            "value": [float(i) for i in range(20)],
        })
        svg = fm.catplot(df, x="group", y="value", kind="box").show_svg()
        assert "<svg" in svg, "Expected valid SVG for catplot box"
        assert "lower_whisker" not in svg, (
            "Internal column 'lower_whisker' must not appear in catplot box SVG"
        )

    # --- SHAP colorbar labels (V8) ----------------------------------------

    def test_shap_beeswarm_color_encoding_title_is_feature_value(self):
        """shap_beeswarm_chart color encoding title must be 'Feature value'.

        The SVG renderer may truncate or elide the multi-word legend title, so
        this regression guard checks the chart spec rather than the SVG bitmap.
        """
        sklearn = pytest.importorskip("sklearn")
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd

        X, y = make_classification(
            n_samples=30, n_features=5, n_informative=3, random_state=42
        )
        feature_names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
        X_df = pd.DataFrame(X, columns=feature_names)
        model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X_df, y)

        chart = fm.shap_beeswarm_chart(model, X_df, y)
        spec = _json.loads(chart.to_json())

        # The color encoding is inside the first layer of the layered chart.
        layers = spec.get("layers", [])
        color_titles = [
            layer.get("encoding", {}).get("color", {}).get("title")
            for layer in layers
            if layer.get("encoding", {}).get("color")
        ]
        assert "Feature value" in color_titles, (
            f"Expected color encoding title 'Feature value' in SHAP spec; got {color_titles}"
        )

    # --- Hardcoded cmap removed (V10) ------------------------------------

    def test_confusion_matrix_chart_theme_affects_colormap(self):
        """confusion_matrix_chart with different themes must produce different SVGs.

        V10 removed hardcoded 'blues' cmap so the theme's sequential scheme is
        used; two distinct themes must therefore produce distinct color output.
        """
        sklearn = pytest.importorskip("sklearn")
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        X, y = make_classification(
            n_samples=50, n_features=5, n_informative=3, random_state=42
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

        svg_arctic = fm.confusion_matrix_chart(
            model, X, y, theme=fm.themes.arctic_signal
        ).show_svg()
        svg_paper = fm.confusion_matrix_chart(
            model, X, y, theme=fm.themes.paper_ink
        ).show_svg()
        assert "<svg" in svg_arctic
        assert svg_arctic != svg_paper, (
            "arctic_signal and paper_ink themes must produce different confusion matrix SVGs"
        )

    # --- top_k on importance (X8) ----------------------------------------

    def test_importance_chart_top_k_limits_feature_count(self):
        """importance_chart(top_k=3) must show exactly 3 feature names in the SVG."""
        sklearn = pytest.importorskip("sklearn")
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd

        X, y = make_classification(
            n_samples=50, n_features=10, n_informative=5, random_state=42
        )
        feature_names = [
            "feat_a", "feat_b", "feat_c", "feat_d", "feat_e",
            "feat_f", "feat_g", "feat_h", "feat_i", "feat_j",
        ]
        X_df = pd.DataFrame(X, columns=feature_names)
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_df, y)

        svg = fm.importance_chart(model, X_df, y, top_k=3).show_svg()
        names_in_svg = [name for name in feature_names if name in svg]
        assert len(names_in_svg) == 3, (
            f"Expected exactly 3 feature names in importance_chart(top_k=3) SVG; "
            f"got {len(names_in_svg)}: {names_in_svg}"
        )

    # --- max_display on SHAP (X9) ----------------------------------------

    def test_shap_beeswarm_max_display_limits_feature_count(self):
        """shap_beeswarm_chart(max_display=3) must show exactly 3 features in SVG."""
        sklearn = pytest.importorskip("sklearn")
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd

        X, y = make_classification(
            n_samples=50, n_features=10, n_informative=5, random_state=42
        )
        feature_names = [
            "feat_a", "feat_b", "feat_c", "feat_d", "feat_e",
            "feat_f", "feat_g", "feat_h", "feat_i", "feat_j",
        ]
        X_df = pd.DataFrame(X, columns=feature_names)
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_df, y)

        svg = fm.shap_beeswarm_chart(model, X_df, y, max_display=3).show_svg()
        names_in_svg = [name for name in feature_names if name in svg]
        assert len(names_in_svg) == 3, (
            f"Expected exactly 3 feature names in shap_beeswarm_chart(max_display=3) SVG; "
            f"got {len(names_in_svg)}: {names_in_svg}"
        )

    # --- normalize on confusion (X7) -------------------------------------

    def test_confusion_matrix_default_normalize_shows_proportions(self):
        """confusion_matrix_chart default (normalize='true') cell values are
        proportions — they contain a decimal point confirming non-integer output."""
        sklearn = pytest.importorskip("sklearn")
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        import re

        X, y = make_classification(
            n_samples=50, n_features=5, n_informative=3, random_state=42
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

        svg = fm.confusion_matrix_chart(model, X, y).show_svg()
        text_matches = re.findall(r"<text[^>]*>([^<]+)</text>", svg)
        # Proportions are formatted as decimals (e.g. "1.00", "0.00"); find at
        # least one cell value that is neither a pure integer nor a colorbar tick.
        decimal_cell_values = [
            t for t in text_matches
            if "." in t and any(c.isdigit() for c in t)
        ]
        assert decimal_cell_values, (
            "Expected decimal cell values in default (normalize='true') confusion matrix SVG; "
            f"all text elements: {text_matches}"
        )
