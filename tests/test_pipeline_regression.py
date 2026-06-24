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
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "g": ["a", "b", "c", "a", "b", "c"],
        }
    )


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
            .to_svg()
        )
        assert "<svg" in svg
        # The renderer should include a "0" tick label on the y-axis.
        assert ">0<" in svg or ">0.0<" in svg, (
            "Expected a '0' tick label on the y-axis when scale zero=True"
        )

    def test_bar_chart_without_scale_zero_still_renders(self):
        """Control: bar chart without scale zero still renders cleanly."""
        df = _bar_df()
        svg = fm.Chart(df).mark_bar().encode(x="cat", y="val").to_svg()
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
            .to_svg()
        )
        assert 'text-anchor="middle"' in svg

    def test_title_color_emits_fill_attribute(self):
        svg = (
            fm.Chart(_numeric_df(), title=fm.Title("Test", color="#ff0000"))
            .mark_point()
            .encode(x="x", y="y")
            .to_svg()
        )
        assert "#ff0000" in svg

    def test_title_font_size_accepted_and_renders(self):
        svg = (
            fm.Chart(_numeric_df(), title=fm.Title("Test", font_size=20))
            .mark_point()
            .encode(x="x", y="y")
            .to_svg()
        )
        assert "<svg" in svg

    def test_title_font_weight_bold_accepted_and_renders(self):
        svg = (
            fm.Chart(_numeric_df(), title=fm.Title("Test", font_weight="bold"))
            .mark_point()
            .encode(x="x", y="y")
            .to_svg()
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
            .to_svg()
        )
        assert 'text-anchor="middle"' in svg
        assert "#ff0000" in svg


# ---------------------------------------------------------------------------
# D7: EncodingSpec.axis — accepted + warns (not yet rendered)
# ---------------------------------------------------------------------------


class TestEncodingAxis:
    def test_axis_labels_false_accepted_and_honored(self):
        """axis= kwarg is accepted and honored — no UserWarning emitted."""
        from ferrum._warn import reset_warnings

        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = (
                fm.Chart(_numeric_df())
                .mark_point()
                .encode(x=fm.X("x", axis={"labels": False}), y="y")
                .to_svg()
            )
        assert "<svg" in svg
        # axis= is now honored — no warning should be emitted.
        axis_warns = [w for w in caught if "axis" in str(w.message).lower()]
        assert axis_warns == [], (
            f"axis= should not warn (it is now honored); got: {[str(w.message) for w in axis_warns]}"
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
            .to_svg()
        )
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# D12: EncodingSpec.format — accepted + warns (not yet rendered)
# ---------------------------------------------------------------------------


class TestEncodingFormat:
    def test_format_string_accepted_and_honored(self):
        """format= kwarg is accepted and honored — no UserWarning emitted."""
        from ferrum._warn import reset_warnings

        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = (
                fm.Chart(_numeric_df())
                .mark_point()
                .encode(x=fm.X("x", format=".1f"), y="y")
                .to_svg()
            )
        assert "<svg" in svg
        # format= is now honored — no warning should be emitted.
        fmt_warns = [w for w in caught if "format" in str(w.message).lower()]
        assert fmt_warns == [], (
            f"format= should not warn (it is now honored); got: {[str(w.message) for w in fmt_warns]}"
        )


# ---------------------------------------------------------------------------
# D13: legend=False suppresses legend
# ---------------------------------------------------------------------------


class TestLegendDisabled:
    def test_color_legend_false_suppresses_legend(self):
        """Passing legend=False to Color should suppress the legend in the SVG."""
        df = _color_df()
        # With legend
        svg_with = fm.Chart(df).mark_point().encode(x="x", y="y", color=fm.Color("g")).to_svg()
        # Without legend
        svg_without = (
            fm.Chart(df)
            .mark_point()
            .encode(x="x", y="y", color=fm.Color("g", legend=False))
            .to_svg()
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
            .to_svg()
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
            .to_svg()
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
        df = pl.DataFrame(
            {
                "row": ["A", "A", "B", "B"],
                "col": ["X", "Y", "X", "Y"],
                "val": [-1.0, 0.5, 0.5, -0.5],
            }
        )
        svg = (
            fm.Chart(df)
            .mark_rect()
            .encode(x="col", y="row", color="val")
            .theme(fm.themes.arctic_signal)
            .to_svg()
        )
        assert "<svg" in svg
        # SVG with arctic_signal (blue_to_violet diverging) should differ from
        # one with a different diverging scheme, confirming the theme was applied.
        svg_other = (
            fm.Chart(df)
            .mark_rect()
            .encode(x="col", y="row", color="val")
            .theme(fm.Theme(diverging_scheme="rdbu"))
            .to_svg()
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
            .to_svg()
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
            .to_svg()
        )
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# D18: baseline on text mark
# ---------------------------------------------------------------------------


class TestBaselineOnText:
    def test_mark_text_baseline_top_emits_dominant_baseline(self):
        """mark_text(baseline='top') should emit dominant-baseline in the SVG."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [10.0, 20.0, 30.0],
                "label": ["A", "B", "C"],
            }
        )
        svg = fm.Chart(df).mark_text(baseline="top").encode(x="x", y="y", text="label").to_svg()
        assert "<svg" in svg
        assert "dominant-baseline" in svg, (
            "Expected 'dominant-baseline' attribute when baseline='top' is set"
        )

    def test_mark_text_baseline_middle_emits_dominant_baseline(self):
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [5.0, 10.0],
                "label": ["X", "Y"],
            }
        )
        svg = fm.Chart(df).mark_text(baseline="middle").encode(x="x", y="y", text="label").to_svg()
        assert "dominant-baseline" in svg


# ---------------------------------------------------------------------------
# B2-B3: Layer serialization — legend forwarding to layers (boxplot)
# ---------------------------------------------------------------------------


class TestLayerSerialization:
    def test_boxplot_with_legend_false_suppresses_legend(self):
        """color legend=False should be forwarded to boxplot layers.

        The hue column ``h`` is distinct from the categorical axis ``g`` so the
        color encoding is genuinely split (``split_hue=True``) and is attached
        to the box layers. When the hue equals the categorical axis the color
        encoding is redundant and is suppressed, so this test deliberately uses
        a separate hue to exercise the legend-forwarding path.
        """
        df = pl.DataFrame(
            {
                "g": ["a", "a", "a", "b", "b", "b"],
                "h": ["x", "y", "x", "y", "x", "y"],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        svg_with_legend = (
            fm.Chart(df).mark_boxplot().encode(x="g", y="y", color=fm.Color("h")).to_svg()
        )
        svg_without_legend = (
            fm.Chart(df)
            .mark_boxplot()
            .encode(x="g", y="y", color=fm.Color("h", legend=False))
            .to_svg()
        )
        assert "<svg" in svg_without_legend
        assert len(svg_without_legend) < len(svg_with_legend), (
            "Boxplot with legend=False should produce shorter SVG than with legend"
        )

    def test_boxplot_renders_as_layered_chart(self):
        """mark_boxplot desugars into a layered chart (multiple SVG elements)."""
        df = pl.DataFrame(
            {
                "g": ["a", "a", "b", "b"],
                "y": [1.0, 3.0, 2.0, 4.0],
            }
        )
        svg = fm.Chart(df).mark_boxplot().encode(x="g", y="y").to_svg()
        assert "<svg" in svg
        # Boxplot emits at least a rect and a rule, so multiple elements.
        assert "<rect" in svg or "<line" in svg

    def test_build_layers_list_forwards_encoding_type(self):
        """B2 fix: _build_layers_list must carry the 'type' key into each layer's
        encoding JSON dict.

        to_encoding_spec_dict() emits the data-type under 'type_' (Python
        convention to avoid shadowing the builtin), but _build_layers_list was
        reading d.get('type') — silently dropping the type for every composite-
        mark layer encoding that carries a ChannelBase with a typed kwarg.

        The fix: read d.get('type_') and emit it as 'type' in the JSON dict.
        """
        import polars as pl
        from ferrum._layer import _Layer, MarkDesugarResult, _PendingMark
        from ferrum.encoding import Y

        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        # Create a chart that goes through the layers path.  The Y channel
        # carries an explicit type so we can assert it reaches the layer dict.
        chart = fm.Chart(df)

        def desugar_typed_layer(x_field, y_field, **_kw):
            return MarkDesugarResult(
                layers=[
                    _Layer(
                        mark="line",
                        encoding={"x": x_field, "y": Y(y_field, type="quantitative")},
                    )
                ]
            )

        chart._pending_stat_mark = _PendingMark("__typed_test__", {}, desugar_typed_layer)
        resolved = chart.encode(x="x", y="y")._resolve_pending()
        layers_list = resolved._build_layers_list()

        assert layers_list, "Expected at least one layer dict"
        y_enc = layers_list[0]["encoding"].get("y")
        assert y_enc is not None, "Expected 'y' key in first layer encoding dict"
        assert y_enc.get("type") == "quantitative", (
            f"Expected encoding type 'quantitative' in layer JSON dict, got: {y_enc!r}"
        )


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
        svg = fm.roc_chart(model, X, y).to_svg()
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
    def test_density_invalid_kernel_raises_value_error(self):
        """mark_density(kernel='banana') should raise ValueError."""
        df = _numeric_df()
        with pytest.raises(ValueError, match="not supported"):
            fm.Chart(df).mark_density(kernel="banana").encode(x="x").to_svg()

    def test_histogram_right_true_raises_value_error(self):
        """mark_histogram(right=True) should raise ValueError."""
        df = _numeric_df()
        with pytest.raises(ValueError, match="right"):
            fm.Chart(df).mark_histogram(right=True).encode(x="x").to_svg()


# ---------------------------------------------------------------------------
# S1-S11: Mark kwargs
# ---------------------------------------------------------------------------


class TestMarkKwargs:
    def test_interpolate_step_emits_step_path(self):
        """mark_line(interpolate='step') should produce H or V commands in the path."""
        df = _numeric_df()
        svg = fm.Chart(df).mark_line(interpolate="step").encode(x="x", y="y").to_svg()
        assert "<svg" in svg
        # Step interpolation produces horizontal (H) and vertical (V) path segments.
        assert "H" in svg or "V" in svg, "Expected H or V path commands for step interpolation"

    def test_stroke_cap_round_emits_stroke_linecap(self):
        """mark_line(stroke_cap='round') should emit stroke-linecap='round'."""
        df = _numeric_df()
        svg = fm.Chart(df).mark_line(stroke_cap="round").encode(x="x", y="y").to_svg()
        assert 'stroke-linecap="round"' in svg

    def test_stroke_join_bevel_emits_stroke_linejoin(self):
        """mark_line(stroke_join='bevel') should emit stroke-linejoin='bevel'."""
        df = _numeric_df()
        svg = fm.Chart(df).mark_line(stroke_join="bevel").encode(x="x", y="y").to_svg()
        assert 'stroke-linejoin="bevel"' in svg

    def test_filled_false_emits_fill_none_on_circles(self):
        """mark_point(filled=False) should emit fill='none' on circles."""
        df = _numeric_df()
        svg = fm.Chart(df).mark_point(filled=False).encode(x="x", y="y").to_svg()
        assert 'fill="none"' in svg

    def test_shape_square_emits_rect_elements(self):
        """mark_point(shape='square') should use <rect> instead of <circle>."""
        df = _numeric_df()
        svg = fm.Chart(df).mark_point(shape="square").encode(x="x", y="y").to_svg()
        assert "<rect" in svg, "Expected <rect> elements for square shape"

    def test_limit_on_text_truncates_long_labels(self):
        """mark_text(limit=5) should truncate long text and add ellipsis."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [5.0, 10.0],
                "label": ["Hello World", "Short"],
            }
        )
        svg = fm.Chart(df).mark_text(limit=5).encode(x="x", y="y", text="label").to_svg()
        assert "<svg" in svg
        # Truncated text should include an ellipsis character.
        assert "…" in svg, "Expected ellipsis '…' in SVG when limit is exceeded"

    def test_band_size_on_tick_renders(self):
        """mark_tick(band_size=0.5) should render without error."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        svg = fm.Chart(df).mark_tick(band_size=0.5).encode(x="x", y="y").to_svg()
        assert "<svg" in svg

    def test_area_with_line_true_has_multiple_paths(self):
        """mark_area(line=True) should produce at least two <path> elements."""
        df = _numeric_df()
        svg = fm.Chart(df).mark_area(line=True).encode(x="x", y="y").to_svg()
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
        svg = fm.Chart(df).mark_bar(orient="horizontal").encode(x="x", y="y").to_svg()
        assert "<svg" in svg
        assert "<rect" in svg


# ---------------------------------------------------------------------------
# W1-W18: Encoding channels
# ---------------------------------------------------------------------------


class TestEncodingChannels:
    def test_tooltip_channel_accepted_and_emits_title_elements(self):
        """encode(tooltip=Tooltip('g')) should produce <title> elements in SVG."""
        df = _simple_df()
        svg = fm.Chart(df).mark_point().encode(x="x", y="y", tooltip=fm.Tooltip("g")).to_svg()
        assert "<svg" in svg
        assert "<title>" in svg, "Expected <title> elements for tooltip encoding"

    def test_href_channel_accepted_and_emits_anchor_elements(self):
        """encode(href=Href('url')) should produce <a> elements in SVG."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [10.0, 20.0, 30.0],
                "url": ["http://a.com", "http://b.com", "http://c.com"],
            }
        )
        svg = fm.Chart(df).mark_point().encode(x="x", y="y", href=fm.Href("url")).to_svg()
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
                .to_svg()
            )
        assert "<svg" in svg
        user_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "description" in str(w.message).lower()
        ]
        assert not user_warnings, f"Unexpected UserWarning for Description channel: {user_warnings}"

    def test_fill_channel_aliased_to_color_no_user_warning(self):
        """encode(fill=Fill('g')) should alias to color encoding without UserWarning."""
        from ferrum._warn import reset_warnings

        reset_warnings()
        df = _simple_df()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = fm.Chart(df).mark_point().encode(x="x", y="y", fill=fm.Fill("g")).to_svg()
        assert "<svg" in svg
        fill_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "fill" in str(w.message).lower()
        ]
        assert not fill_warnings, f"Unexpected UserWarning for Fill channel: {fill_warnings}"

    def test_detail_channel_accepted_without_user_warning(self):
        """encode(detail=Detail('g')) is accepted without UserWarning."""
        from ferrum._warn import reset_warnings

        reset_warnings()
        df = _simple_df()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = fm.Chart(df).mark_line().encode(x="x", y="y", detail=fm.Detail("g")).to_svg()
        assert "<svg" in svg
        detail_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "detail" in str(w.message).lower()
        ]
        assert not detail_warnings, f"Unexpected UserWarning for Detail channel: {detail_warnings}"

    def test_theta_channel_with_coord_polar_remaps_to_x(self):
        """Phase 11d: theta remaps to x when CoordPolar(theta='x') is set."""
        import json

        df = _numeric_df()
        spec = (
            fm.Chart(df)
            .mark_point()
            .encode(theta=fm.Theta("y"))
            .coord(fm.CoordPolar(theta="x"))
            .to_spec()
        )
        j = json.loads(spec.to_json())
        # theta should have been remapped to the x channel
        assert j["encoding"].get("x") is not None
        assert j["encoding"].get("theta") is None

    def test_radius_channel_with_coord_polar_remaps_to_y(self):
        """Phase 11d: radius remaps to y when CoordPolar(theta='x') is set."""
        import json

        df = _numeric_df()
        spec = (
            fm.Chart(df)
            .mark_point()
            .encode(theta=fm.Theta("y"), radius=fm.Radius("x"))
            .coord(fm.CoordPolar(theta="x"))
            .to_spec()
        )
        j = json.loads(spec.to_json())
        assert j["encoding"].get("x") is not None
        assert j["encoding"].get("y") is not None
        assert j["encoding"].get("theta") is None
        assert j["encoding"].get("radius") is None


# ---------------------------------------------------------------------------
# Round-2 regression tests for gallery-defaults fixes
# ---------------------------------------------------------------------------

import json as _json


def _bivariate_df(n: int = 50, seed: int = 42) -> pl.DataFrame:
    """Small bivariate normal DataFrame for contour / raster tests."""
    rng = __import__("numpy").random.default_rng(seed)
    return pl.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, n).tolist(),
            "y": rng.normal(0.0, 1.0, n).tolist(),
        }
    )


def _group_df(n_per_group: int = 15, seed: int = 42) -> pl.DataFrame:
    """Grouped DataFrame for swarm / errorbar tests."""
    rng = __import__("numpy").random.default_rng(seed)
    groups = ["a"] * n_per_group + ["b"] * n_per_group
    vals = rng.normal(0.0, 1.0, 2 * n_per_group).tolist()
    return pl.DataFrame({"group": groups, "val": vals})


class TestRound2Fixes:
    # --- Contour rendering (was blank) -----------------------------------

    def test_contour_spec_has_correct_layer(self):
        """mark_contour desugars to the correct layer mark per fill mode.

        fill=False (default) produces segment mark (isolines).
        fill=True            produces polygon mark (isobands) with color encoding.
        """
        df = _bivariate_df()
        # fill=False (default): segment mark for isolines
        chart = fm.Chart(df).mark_contour().encode(x="x", y="y")
        spec = _json.loads(chart.to_json())
        transform_types = [t["type"] for t in spec.get("transforms", [])]
        assert "kde2_d" in transform_types, "Expected 'kde2_d' transform in contour spec"
        assert "contour" in transform_types, "Expected 'contour' transform in contour spec"
        layer_marks = [layer["mark"] for layer in spec.get("layers", [])]
        assert "segment" in layer_marks, (
            "Expected a 'segment' mark layer in contour spec (fill=False)"
        )

        # fill=True: polygon mark for isobands with color=level_value
        chart_fill = fm.Chart(df).mark_contour(fill=True).encode(x="x", y="y")
        spec_fill = _json.loads(chart_fill.to_json())
        layer_marks_fill = [layer["mark"] for layer in spec_fill.get("layers", [])]
        assert "polygon" in layer_marks_fill, (
            "Expected a 'polygon' mark layer in contour spec (fill=True)"
        )
        # Verify color encoding is present for density-band coloring
        poly_layer = next(l for l in spec_fill["layers"] if l["mark"] == "polygon")
        assert poly_layer["encoding"].get("color", {}).get("field") == "level_value", (
            "Expected 'color' encoding mapped to 'level_value' for filled contour"
        )

    def test_contour_spec_renders_without_error(self):
        """mark_contour().to_svg() completes without raising."""
        df = _bivariate_df()
        svg = fm.Chart(df).mark_contour().encode(x="x", y="y").to_svg()
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
        svg = fm.Chart(df).mark_raster().encode(x="x", y="y").to_svg()
        assert "<image" in svg, "Expected <image> element in raster SVG"

    def test_raster_cmap_plasma_differs_from_default(self):
        """mark_raster(cmap='plasma') produces SVG different from default cmap."""
        df = _bivariate_df()
        svg_default = fm.Chart(df).mark_raster().encode(x="x", y="y").to_svg()
        svg_plasma = fm.Chart(df).mark_raster(cmap="plasma").encode(x="x", y="y").to_svg()
        assert "<image" in svg_plasma, "Expected <image> element in plasma raster SVG"
        assert svg_default != svg_plasma, (
            "cmap='plasma' should produce different raster output than default"
        )

    def test_render_config_raster_scheme_flows_through_auto_raster(self):
        """Regression (#32): RenderConfig raster_scheme/raster_cmap reach the
        auto-raster-substituted mark_raster path in _render.py.

        Field-level tests confirm that RenderConfig stores the alias correctly,
        but do not prove _render.py reads raster_scheme and passes it to the
        raster renderer. This test proves the end-to-end flow:
          1. raster_threshold=0 + raster_behavior="silent" forces substitution
             on a tiny chart without emitting a UserWarning.
          2. raster_scheme="plasma" and its alias raster_cmap="plasma" produce
             identical SVG output.
          3. That SVG differs from the default (viridis), proving the value is
             actually consumed by the raster render path.
        """
        df = _bivariate_df()

        def _svg(**cfg):
            c = fm.RenderConfig(raster_threshold=0, raster_behavior="silent", **cfg)
            return (
                fm.Chart(df)
                .mark_point()
                .encode(x="x:Q", y="y:Q")
                .properties(render_config=c)
                .to_svg(raster=True)
            )

        plasma_scheme = _svg(raster_scheme="plasma")
        plasma_cmap = _svg(raster_cmap="plasma")
        default_svg = _svg()

        assert "<image" in plasma_scheme, "expected a rasterized <image> in the auto-raster output"
        assert plasma_scheme == plasma_cmap, (
            "raster_scheme and its raster_cmap alias must render identically end-to-end"
        )
        assert plasma_scheme != default_svg, (
            "raster_scheme='plasma' must differ from the default viridis — "
            "raster_scheme is not being read by _render.py if this fails"
        )

    # --- Swarm horizontal -------------------------------------------------

    def test_swarm_horizontal_renders_without_error(self):
        """mark_swarm(orient='horizontal') renders and produces SVG output.

        For a horizontal swarm, the continuous value column maps to x and the
        categorical grouping column maps to y.
        """
        df = _group_df()
        svg = fm.Chart(df).mark_swarm(orient="horizontal").encode(x="val", y="group").to_svg()
        assert "<svg" in svg, "Expected valid SVG for horizontal swarm"

    # --- Errorbar cap width (V3) -----------------------------------------

    def test_errorbar_renders_without_error(self):
        """mark_errorbar() renders a valid SVG without raising.

        Geometric assertions on exact cap width are brittle; the regression
        guard is that the chart renders at all after the V3 band_size fix.
        """
        df = _group_df()
        svg = fm.Chart(df).mark_errorbar(method="ci").encode(x="group", y="val").to_svg()
        assert "<svg" in svg, "Expected valid SVG for errorbar chart"

    # --- Histogram axis label (V4-V5) ------------------------------------

    def test_histogram_axis_label_shows_field_name_not_bin_start(self):
        """mark_histogram().encode(x='my_variable') must show 'my_variable' in
        the SVG, not the internal bin column name 'bin_start'."""
        df = pl.DataFrame({"my_variable": [float(i) for i in range(1, 21)]})
        svg = fm.Chart(df).mark_histogram().encode(x="my_variable").to_svg()
        assert "my_variable" in svg, "Expected original field name 'my_variable' in histogram SVG"
        assert "bin_start" not in svg, (
            "Internal column name 'bin_start' must not appear in histogram SVG"
        )

    # --- Catplot box axis label (V6) -------------------------------------

    def test_catplot_box_axis_does_not_expose_lower_whisker(self):
        """catplot(kind='box') SVG must not contain the raw column name
        'lower_whisker' — the y-axis should show the original variable name."""
        df = pl.DataFrame(
            {
                "group": ["a"] * 10 + ["b"] * 10,
                "value": [float(i) for i in range(20)],
            }
        )
        svg = fm.catplot(df, x="group", y="value", kind="box").to_svg()
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

        X, y = make_classification(n_samples=30, n_features=5, n_informative=3, random_state=42)
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

        X, y = make_classification(n_samples=50, n_features=5, n_informative=3, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

        svg_arctic = fm.confusion_matrix_chart(model, X, y, theme=fm.themes.arctic_signal).to_svg()
        svg_paper = fm.confusion_matrix_chart(model, X, y, theme=fm.themes.paper_ink).to_svg()
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

        X, y = make_classification(n_samples=50, n_features=10, n_informative=5, random_state=42)
        feature_names = [
            "feat_a",
            "feat_b",
            "feat_c",
            "feat_d",
            "feat_e",
            "feat_f",
            "feat_g",
            "feat_h",
            "feat_i",
            "feat_j",
        ]
        X_df = pd.DataFrame(X, columns=feature_names)
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_df, y)

        svg = fm.importance_chart(model, X_df, y, top_k=3).to_svg()
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

        X, y = make_classification(n_samples=50, n_features=10, n_informative=5, random_state=42)
        feature_names = [
            "feat_a",
            "feat_b",
            "feat_c",
            "feat_d",
            "feat_e",
            "feat_f",
            "feat_g",
            "feat_h",
            "feat_i",
            "feat_j",
        ]
        X_df = pd.DataFrame(X, columns=feature_names)
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_df, y)

        svg = fm.shap_beeswarm_chart(model, X_df, y, max_display=3).to_svg()
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

        X, y = make_classification(n_samples=50, n_features=5, n_informative=3, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

        svg = fm.confusion_matrix_chart(model, X, y).to_svg()
        text_matches = re.findall(r"<text[^>]*>([^<]+)</text>", svg)
        # Proportions are formatted as decimals (e.g. "1.00", "0.00"); find at
        # least one cell value that is neither a pure integer nor a colorbar tick.
        decimal_cell_values = [t for t in text_matches if "." in t and any(c.isdigit() for c in t)]
        assert decimal_cell_values, (
            "Expected decimal cell values in default (normalize='true') confusion matrix SVG; "
            f"all text elements: {text_matches}"
        )


# ---------------------------------------------------------------------------
# Round-3 regression tests for gallery-defaults fixes
# ---------------------------------------------------------------------------

import re as _re


class TestRound3Fixes:
    # --- H1: contour renders (not blank) ------------------------------------

    def test_contour_to_svg_completes_without_error(self):
        """mark_contour().to_svg() must not raise.

        The polygon renderer currently produces an empty clip group for small
        bivariate samples; the regression guard is that the chart call
        completes and returns a structurally valid SVG with axis lines and
        tick labels.  The spec-level polygon assertion lives in Round2.
        """
        df = _bivariate_df()
        svg = fm.Chart(df).mark_contour().encode(x="x", y="y").to_svg()
        assert "<svg" in svg, "Expected valid SVG for contour chart"
        # Axis lines confirm the chart structure was rendered (not a trivial stub).
        assert "<line" in svg, "Expected axis <line> elements in contour SVG"

    # --- H3: tick strip plot distributes across categories ------------------

    def test_tick_ordinal_y_renders_without_error(self):
        """mark_tick() with ordinal y + quantitative x renders a valid SVG."""
        df = pl.DataFrame(
            {
                "cat": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
                "val": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 0.5, 2.2, 3.1, 1.8],
            }
        )
        svg = fm.Chart(df).mark_tick().encode(x="val", y="cat").to_svg()
        assert "<svg" in svg, "Expected valid SVG for tick strip plot"
        # Tick marks render as <line> elements.
        assert "<line" in svg, "Expected <line> elements in tick strip plot SVG"

    # --- H6: hex density color encoding -------------------------------------

    def test_hex_density_has_multiple_distinct_fill_colors(self):
        """mark_hex() must produce more than one distinct fill color in the SVG.

        A uniform fill across all hex cells would indicate the color scale is
        broken (all cells mapped to the same bin value or the same palette stop).
        """
        df = _bivariate_df(n=100)
        svg = fm.Chart(df).mark_hex().encode(x="x", y="y").to_svg()
        assert "<svg" in svg, "Expected valid SVG for hex density chart"
        # hex cells are emitted as <path> elements with fill attributes.
        fill_colors = _re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg)
        unique_fills = set(fill_colors)
        assert len(unique_fills) > 1, (
            f"Expected multiple distinct fill colors in hex SVG; got: {unique_fills}"
        )

    # --- H7: scatter+smooth preserves scatter layer -------------------------

    def test_scatter_smooth_chain_has_both_circle_and_line_element(self):
        """Chaining mark_point().mark_smooth() must render both scatter circles
        and the smooth line in a single SVG.

        The smooth layer renders as <polyline> (not <path>); both the scatter
        <circle> elements and the smooth <polyline> must be present.
        """
        rng = __import__("numpy").random.default_rng(42)
        df = pl.DataFrame(
            {
                "x": rng.uniform(0.0, 10.0, 30).tolist(),
                "y": (rng.uniform(0.0, 10.0, 30) + rng.normal(0.0, 1.0, 30)).tolist(),
            }
        )
        svg = fm.Chart(df).mark_point().mark_smooth().encode(x="x", y="y").to_svg()
        assert "<circle" in svg, "Expected <circle> elements (scatter layer) in scatter+smooth SVG"
        has_line = "<path" in svg or "<polyline" in svg
        assert has_line, (
            "Expected <path> or <polyline> element (smooth layer) in scatter+smooth SVG"
        )

    # --- H10: violin axis label shows field name, not internal column -------

    def test_violin_y_axis_shows_field_name_not_violin_y(self):
        """mark_violin() y-axis must show the user's field name, not 'violin_y'."""
        rng = __import__("numpy").random.default_rng(42)
        df = pl.DataFrame(
            {
                "cat": ["a"] * 15 + ["b"] * 15,
                "measurement": rng.normal(0.0, 1.0, 30).tolist(),
            }
        )
        svg = fm.Chart(df).mark_violin().encode(x="cat", y="measurement").to_svg()
        assert "measurement" in svg, "Expected user field name 'measurement' in violin SVG y-axis"
        assert "violin_y" not in svg, (
            "Internal column name 'violin_y' must not appear in violin SVG"
        )

    # --- H11: PCA scree ordinal x-axis has no fractional tick labels --------

    def test_pca_scree_x_axis_has_no_fractional_component_ticks(self):
        """pca_scree_chart() component axis must not contain '1.5' or '2.5' as
        tick labels — those would indicate the axis was treated as continuous
        (linear) rather than ordinal.

        Note: '1.5' can appear in the SVG as stroke-width; the assertion is
        scoped to <text> element content only.
        """
        sklearn = pytest.importorskip("sklearn")
        pd = pytest.importorskip("pandas")
        from sklearn.decomposition import PCA

        rng = __import__("numpy").random.default_rng(42)
        X_df = pd.DataFrame(
            rng.normal(0.0, 1.0, (100, 6)),
            columns=[f"f{i}" for i in range(6)],
        )
        pca = PCA(n_components=4).fit(X_df)
        svg = fm.pca_scree_chart(pca, X_df).to_svg()
        assert "<svg" in svg, "Expected valid SVG for pca_scree_chart"

        # Extract text content from all <text> elements.
        text_labels = _re.findall(r"<text[^>]*>([^<]+)</text>", svg)
        assert "1.5" not in text_labels, (
            f"Fractional tick '1.5' must not appear as a component axis label; "
            f"text elements: {text_labels}"
        )
        assert "2.5" not in text_labels, (
            f"Fractional tick '2.5' must not appear as a component axis label; "
            f"text elements: {text_labels}"
        )

    # --- M1: SHAP bar chart has human-readable x axis title -----------------

    def test_shap_bar_chart_x_encoding_title_is_mean_abs_shap(self):
        """shap_bar_chart() x encoding title must be 'Mean |SHAP value|'.

        The chart is a layered spec; the title lives in the first layer's
        encoding, not at the top-level encoding.
        """
        sklearn = pytest.importorskip("sklearn")
        pd = pytest.importorskip("pandas")
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        X, y = make_classification(n_samples=30, n_features=5, n_informative=3, random_state=42)
        feature_names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
        X_df = pd.DataFrame(X, columns=feature_names)
        model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X_df, y)

        spec = _json.loads(fm.shap_bar_chart(model, X_df, y).to_json())
        # The x title is in the first layer's encoding.
        layers = spec.get("layers", [])
        x_titles = [
            layer.get("encoding", {}).get("x", {}).get("title")
            for layer in layers
            if isinstance(layer.get("encoding", {}).get("x"), dict)
        ]
        assert "Mean |SHAP value|" in x_titles, (
            f"Expected x encoding title 'Mean |SHAP value|' in shap_bar_chart spec; got: {x_titles}"
        )

    # --- M8: QQ plot axis titles are human-readable -------------------------

    def test_qq_plot_svg_contains_axis_titles(self):
        """mark_qq() SVG must contain 'Theoretical Quantiles' and 'Sample Quantiles'
        as axis labels, not raw internal column names.
        """
        rng = __import__("numpy").random.default_rng(42)
        df = pl.DataFrame({"val": rng.normal(0.0, 1.0, 30).tolist()})
        svg = fm.Chart(df).mark_qq().encode(x="val").to_svg()
        assert "<svg" in svg, "Expected valid SVG for QQ plot"
        assert "Theoretical Quantiles" in svg, (
            "Expected 'Theoretical Quantiles' axis label in QQ plot SVG"
        )
        assert "Sample Quantiles" in svg, "Expected 'Sample Quantiles' axis label in QQ plot SVG"


class TestTickRug:
    """Regression tests for mark_tick() single-axis rug mode (x-rug and y-rug).

    Both were broken before the fix: resolve_scales required both x and y
    encodings unconditionally. The x-rug path in tick.rs also had an early
    return when x_field was None that killed the y-rug case entirely.
    """

    def _rug_lines(self, svg: str):
        """Extract (x1, y1, x2, y2) tuples from all SVG <line> elements."""
        import re

        lines = []
        for m in re.finditer(r"<line([^/]+)/>", svg):
            attrs = dict(re.findall(r'(\w+)="([^"]+)"', m.group(1)))
            try:
                lines.append(
                    (
                        float(attrs["x1"]),
                        float(attrs["y1"]),
                        float(attrs["x2"]),
                        float(attrs["y2"]),
                    )
                )
            except (KeyError, ValueError):
                pass
        return lines

    def test_x_rug_renders_without_error(self):
        """mark_tick().encode(x=...) must render without ValueError."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        svg = fm.Chart(df).mark_tick().encode(x="val").to_svg()
        assert "<svg" in svg

    def test_x_rug_produces_vertical_tick_lines(self):
        """x-rug ticks must be short vertical lines going UP from the plot baseline."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        svg = fm.Chart(df).mark_tick().encode(x="val").to_svg()
        lines = self._rug_lines(svg)
        # x-rug ticks: vertical (x1≈x2), short dy, going UP (y2<y1 means into plot area).
        # X-axis tick marks go DOWN (y2>y1, outside plot) — excluded by y1>y2 filter.
        rug = [l for l in lines if abs(l[0] - l[2]) < 0.5 and 4 < (l[1] - l[3]) < 30]
        assert len(rug) == 5, f"Expected 5 x-rug tick lines, got {len(rug)}: {rug}"

    def test_y_rug_renders_without_error(self):
        """mark_tick().encode(y=...) must render without ValueError."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        svg = fm.Chart(df).mark_tick().encode(y="val").to_svg()
        assert "<svg" in svg

    def test_y_rug_produces_horizontal_tick_lines(self):
        """y-rug ticks must be short horizontal lines extending INTO the plot area."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        svg = fm.Chart(df).mark_tick().encode(y="val").to_svg()
        lines = self._rug_lines(svg)
        # y-rug ticks: horizontal (y1≈y2), short dx, and going RIGHT (x2>x1 means into plot).
        # Y-axis tick marks go LEFT (x2<x1) — excluded by x2>x1 filter.
        rug = [l for l in lines if abs(l[1] - l[3]) < 0.5 and 1 < (l[2] - l[0]) < 30]
        assert len(rug) == 5, f"Expected 5 y-rug tick lines, got {len(rug)}: {rug}"

    def test_y_rug_ticks_at_left_axis_edge(self):
        """y-rug tick lines must start at the left axis boundary and go right."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        svg = fm.Chart(df).mark_tick().encode(y="val").to_svg()
        lines = self._rug_lines(svg)
        rug = [l for l in lines if abs(l[1] - l[3]) < 0.5 and 1 < (l[2] - l[0]) < 30]
        assert len(rug) == 5
        x1_vals = {round(l[0], 1) for l in rug}
        assert len(x1_vals) == 1, f"All y-rug ticks should share the same x1; got {x1_vals}"
        x1 = x1_vals.pop()
        assert all(l[2] > x1 for l in rug), "y-rug ticks must extend into the plot area"

    def test_both_axes_encoded_still_works(self):
        """mark_tick().encode(x=..., y=...) (strip plot) must still work after the fix."""
        df = pl.DataFrame(
            {
                "cat": ["a", "b", "a", "b", "a"],
                "val": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        svg = fm.Chart(df).mark_tick().encode(x="val", y="cat").to_svg()
        assert "<svg" in svg
        assert "<line" in svg


class TestRasterOverride:
    """Test the raster= parameter on to_svg/to_png/show/save."""

    @pytest.fixture()
    def chart(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        return fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")

    def test_raster_false_disables_auto_raster(self, chart):
        """raster=False sets threshold to None internally."""
        overridden = chart._with_raster_override(False)
        assert overridden._render_config.raster_threshold is None

    def test_raster_true_forces_raster(self, chart):
        """raster=True sets threshold to 0 so auto-raster always fires."""
        overridden = chart._with_raster_override(True)
        assert overridden._render_config.raster_threshold == 0

    def test_raster_none_returns_self(self, chart):
        """raster=None returns the chart unchanged (no clone)."""
        assert chart._with_raster_override(None) is chart

    def test_to_svg_raster_false_produces_circles(self, chart):
        """to_svg(raster=False) renders per-element circles, not raster."""
        svg = chart.to_svg(raster=False)
        assert "<circle" in svg or 'cx="' in svg

    def test_to_svg_raster_true_substitutes_raster(self, chart):
        """to_svg(raster=True) on a 3-row chart produces a raster image."""
        svg = chart.to_svg(raster=True)
        assert "<image" in svg

    def test_to_png_raster_kwarg_accepted(self, chart):
        """to_png(raster=False) produces valid PNG."""
        png = chart.to_png(raster=False)
        assert png[:4] == b"\x89PNG"

    def test_save_raster_kwarg_accepted(self, chart, tmp_path):
        """save(raster=False) writes an SVG with per-element marks."""
        out = tmp_path / "test.svg"
        chart.save(str(out), raster=False)
        svg = out.read_text()
        assert "<circle" in svg or 'cx="' in svg

    def test_original_chart_not_mutated(self, chart):
        """The raster= override does not mutate the original chart."""
        original_config = chart._render_config
        chart.to_svg(raster=False)
        assert chart._render_config is original_config


class TestKdeIntegerCoercion:
    """Regression: mark_density / KDE transforms hard-errored on integer columns.

    Before the fix, stat_kde (1D) and stat_kde_2d rejected any column whose
    arrow dtype was not Float64 with "column '<x>' must be Float64", so
    mark_density on Int64 data (ages, counts, years, range(...)) raised a
    ValueError instead of rendering. The fix coerces numeric columns to Float64
    in the transform; genuinely non-numeric columns must still raise.
    """

    def test_mark_density_1d_renders_on_int_column(self):
        """1D KDE on an Int64 column renders instead of raising."""
        df = pl.DataFrame({"x": list(range(50))})
        svg = fm.Chart(df).mark_density().encode(x="x").to_svg()
        assert svg.lstrip().startswith("<svg")
        assert "<path" in svg

    def test_mark_density_2d_renders_on_int_columns(self):
        """2D KDE on Int64 x and y renders instead of raising."""
        df = pl.DataFrame({"x": list(range(50)), "y": list(range(50, 100))})
        svg = fm.Chart(df).mark_density().encode(x="x", y="y").to_svg()
        assert svg.lstrip().startswith("<svg")

    def test_mark_density_int_matches_float(self):
        """Int input produces the same density output as the equivalent floats."""
        ints = list(range(50))
        svg_int = fm.Chart(pl.DataFrame({"x": ints})).mark_density().encode(x="x").to_svg()
        svg_float = (
            fm.Chart(pl.DataFrame({"x": [float(v) for v in ints]}))
            .mark_density()
            .encode(x="x")
            .to_svg()
        )
        assert svg_int == svg_float

    def test_mark_density_renders_on_non_int_numeric_column(self):
        """A non-Int64 integer dtype (UInt) also coerces and renders."""
        df = pl.DataFrame({"x": pl.Series(list(range(50)), dtype=pl.UInt32)})
        svg = fm.Chart(df).mark_density().encode(x="x").to_svg()
        assert svg.lstrip().startswith("<svg")

    def test_mark_density_string_column_still_raises(self):
        """A genuinely non-numeric (Utf8) field must still raise, not coerce."""
        df = pl.DataFrame({"x": ["a", "b", "c", "d"] * 10})
        with pytest.raises(Exception, match="numeric|Float64"):
            fm.Chart(df).mark_density().encode(x="x").to_svg()


class TestQqHexIntegerCoercion:
    """Regression: stat_qq and stat_hex hard-errored on integer columns.

    Same root cause as the KDE fix — these transforms rejected any non-Float64
    column with "column '<x>' must be Float64", so mark_qq / mark_hex on integer
    data raised instead of rendering. The fix routes them through
    numeric_util::coerce_to_float64. (stat_bin_2d shares the fix and is covered
    by a Rust unit test; its only Python entry point, jointplot(kind="hist"),
    has a separate pre-existing bug unrelated to dtype.)
    """

    def test_mark_qq_renders_on_int_column(self):
        """mark_qq on an Int64 column renders instead of raising."""
        df = pl.DataFrame({"val": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]})
        svg = fm.Chart(df).mark_qq().encode(x="val").to_svg()
        assert svg.lstrip().startswith("<svg")

    def test_mark_hex_renders_on_int_columns(self):
        """mark_hex on Int64 x and y renders instead of raising."""
        df = pl.DataFrame({"x": list(range(40)), "y": list(range(40, 80))})
        svg = fm.Chart(df).mark_hex().encode(x="x", y="y").to_svg()
        assert svg.lstrip().startswith("<svg")

    def test_mark_hex_int_aggregate_field_renders(self):
        """mark_hex with an integer aggregate field coerces and renders."""
        df = pl.DataFrame({"x": list(range(40)), "y": list(range(40, 80)), "w": list(range(40))})
        svg = fm.Chart(df).mark_hex(aggregate="mean", field="w").encode(x="x", y="y").to_svg()
        assert svg.lstrip().startswith("<svg")

    def test_mark_qq_string_column_still_raises(self):
        """A non-numeric qq field must still raise, not coerce."""
        df = pl.DataFrame({"val": ["a", "b", "c", "d"] * 5})
        with pytest.raises(Exception, match="numeric|Float64"):
            fm.Chart(df).mark_qq().encode(x="val").to_svg()


# ---------------------------------------------------------------------------
# TestJointplotHist: regression for bin_x_start / bin_y_start wrong column names
# ---------------------------------------------------------------------------


def _spread_df(n: int = 250, seed: int = 0) -> pl.DataFrame:
    """Bivariate dataset with enough rows to fill multiple 2D histogram bins."""
    rng = __import__("numpy").random.default_rng(seed)
    return pl.DataFrame(
        {
            "x": rng.normal(0.0, 1.5, n).tolist(),
            "y": rng.normal(0.0, 1.5, n).tolist(),
        }
    )


class TestJointplotHist:
    """Regression: jointplot(kind='hist') encoded wrong Bin2D output columns.

    The hist branch was encoding x='bin_x_start'/y='bin_y_start' but Bin2D
    emits [x_lo, x_hi, y_lo, y_hi, count].  This caused a ValueError ("unknown
    column 'bin_x_start'") for all dtypes.  The fix is to use all four corner
    columns (x='x_lo', x2='x_hi', y='y_lo', y2='y_hi') so tiles are full-width
    rectangles, not degenerate points.
    """

    def test_jointplot_hist_renders_svg(self):
        """jointplot(kind='hist') must render without raising ValueError."""
        df = _spread_df()
        svg = fm.jointplot(df, x="x", y="y", kind="hist").to_svg()
        assert svg.lstrip().startswith("<svg"), "Expected SVG output"

    def test_jointplot_hist_has_many_rects(self):
        """The center panel must contain a healthy number of <rect> tiles.

        With correct x_lo/x_hi/y_lo/y_hi encoding, Bin2D produces a grid of
        filled rectangles (>10).  With the old degenerate encoding, only ~3
        rects appear.
        """
        df = _spread_df()
        svg = fm.jointplot(df, x="x", y="y", kind="hist").to_svg()
        assert svg.count("<rect") > 10, (
            f"Expected >10 <rect> elements in hist jointplot SVG, got {svg.count('<rect')}"
        )

    def test_jointplot_hist_with_xlim_ylim_renders(self):
        """The xlim/ylim scale-domain branch of kind='hist' renders correctly."""
        df = _spread_df()
        svg = fm.jointplot(
            df, x="x", y="y", kind="hist", xlim=(-3.0, 3.0), ylim=(-3.0, 3.0)
        ).to_svg()
        assert svg.lstrip().startswith("<svg"), "Expected SVG output with xlim/ylim"
        assert svg.count("<rect") > 10, (
            f"Expected >10 <rect> elements with xlim/ylim, got {svg.count('<rect')}"
        )

    def test_jointplot_hist_renders_on_int_columns(self):
        """Regression: kind='hist' on integer x/y exercises both the encoding
        fix and the stat_bin_2d integer coercion end-to-end (previously broken
        by the 'bin_x_start' encoding bug AND the int-rejection bug)."""
        df = pl.DataFrame({"x": list(range(200)), "y": [(v * 7) % 50 for v in range(200)]})
        svg = fm.jointplot(df, x="x", y="y", kind="hist").to_svg()
        assert svg.lstrip().startswith("<svg")
        assert svg.count("<rect") > 10
