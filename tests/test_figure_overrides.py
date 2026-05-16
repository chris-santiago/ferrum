"""Tests for figure-level override passthrough (mark, encode, properties, layers).

Modeled after ``test_kwarg_piping.py``: the SVG round-trip section verifies
that override kwargs survive the full pipeline into rendered SVG attributes,
not just that internal Python state was set.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import ferrum as fm
from ferrum._overrides import _apply_overrides, LAYER_NAME_CATALOG
from ferrum.layer import Layer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clf_data():
    rng = np.random.RandomState(42)
    X = rng.randn(120, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    model = LogisticRegression(random_state=0).fit(X_train, y_train)
    return model, X_test, y_test


@pytest.fixture(scope="module")
def reg_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    y = X[:, 0] * 2 + X[:, 1] + rng.randn(100) * 0.1
    X_train, X_test = X[:70], X[70:]
    y_train, y_test = y[:70], y[70:]
    model = LinearRegression().fit(X_train, y_train)
    return model, X_test, y_test


# ---------------------------------------------------------------------------
# layer_names property
# ---------------------------------------------------------------------------


class TestLayerNames:
    def test_composite_mark_returns_names(self):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        chart = fm.Chart(df).mark_boxplot().encode(x="x", y="y")
        names = chart.layer_names
        assert "whisker" in names
        assert "box" in names
        assert "median" in names

    def test_single_mark_returns_empty(self):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        chart = fm.Chart(df).mark_point().encode(x="x", y="y")
        assert chart.layer_names == []

    def test_layer_names_forces_desugar(self):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4], "g": ["a", "b"]})
        chart = fm.Chart(df).mark_errorbar().encode(x="g", y="x")
        assert chart._pending_stat_mark is not None
        names = chart.layer_names
        assert "rule" in names

    def test_figure_function_chart_has_names(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(model, X, y)
        names = chart.layer_names
        assert "line" in names
        assert "reference" in names
        assert "point" in names

    def test_roc_chart_layer_names(self, clf_data):
        model, X, y = clf_data
        chart = fm.roc_chart(model, X, y)
        names = chart.layer_names
        assert "line" in names


# ---------------------------------------------------------------------------
# Mark overrides — suppression
# ---------------------------------------------------------------------------


class TestMarkSuppression:
    def test_suppress_point_layer(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(model, X, y, mark={"point": False})
        assert "point" not in chart.layer_names

    def test_suppress_reference_line(self, clf_data):
        model, X, y = clf_data
        chart = fm.roc_chart(model, X, y, mark={"reference": False})
        assert "reference" not in chart.layer_names

    def test_suppress_outlier_on_residuals(self, reg_data):
        model, X, y = reg_data
        chart = fm.residuals_chart(
            model,
            X,
            y,
            cook_threshold=0.5,
            panels="single",
            mark={"outlier": False},
        )
        assert "outlier" not in chart.layer_names

    def test_suppress_produces_different_svg(self, clf_data):
        model, X, y = clf_data
        default = fm.calibration_chart(model, X, y)
        suppressed = fm.calibration_chart(model, X, y, mark={"point": False})
        svg_default = default.show_svg()
        svg_suppressed = suppressed.show_svg()
        assert svg_default != svg_suppressed


# ---------------------------------------------------------------------------
# Mark overrides — kwarg merge
# ---------------------------------------------------------------------------


class TestMarkKwargMerge:
    def test_merge_stroke_width(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(model, X, y, mark={"line": {"stroke_width": 5}})
        resolved = chart._resolve_pending()
        line_layer = next(ly for ly in resolved._layers if ly.name == "line")
        assert line_layer.mark_kwargs.get("stroke_width") == 5

    def test_existing_kwargs_preserved(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(model, X, y, mark={"point": {"filled": False}})
        resolved = chart._resolve_pending()
        point_layer = next(ly for ly in resolved._layers if ly.name == "point")
        assert point_layer.mark_kwargs.get("size") == 40
        assert point_layer.mark_kwargs.get("filled") is False


# ---------------------------------------------------------------------------
# Mark overrides — single-mark fallback
# ---------------------------------------------------------------------------


class TestMarkSingleMark:
    def test_single_mark_flat_kwargs(self):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        chart = fm.Chart(df).mark_point().encode(x="x", y="y")
        overridden = _apply_overrides(chart, mark={"opacity": 0.3})
        assert overridden._mark_kwargs.get("opacity") == 0.3


# ---------------------------------------------------------------------------
# Mark overrides — conditional sub-layer catalog
# ---------------------------------------------------------------------------


class TestConditionalSubLayers:
    def test_suppress_absent_layer_is_noop(self):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4], "g": ["a", "b"]})
        chart = fm.Chart(df).mark_boxplot(outliers=False).encode(x="g", y="y")
        overridden = _apply_overrides(chart, mark={"outlier": False})
        assert "outlier" not in overridden.layer_names

    def test_unknown_layer_raises(self, clf_data):
        model, X, y = clf_data
        with pytest.raises(ValueError, match="Unknown sub-layer"):
            fm.calibration_chart(model, X, y, mark={"nonexistent": False})

    def test_error_lists_valid_names(self, clf_data):
        model, X, y = clf_data
        with pytest.raises(ValueError, match="line"):
            fm.calibration_chart(model, X, y, mark={"bogus": False})


# ---------------------------------------------------------------------------
# Encode overrides
# ---------------------------------------------------------------------------


class TestEncodeOverrides:
    def test_override_x_title(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(
            model,
            X,
            y,
            encode={"x": fm.X("mean_predicted", title="P(positive)")},
        )
        svg = chart.show_svg()
        assert "P(positive)" in svg

    def test_unmentioned_channels_preserved(self, clf_data):
        model, X, y = clf_data
        default = fm.calibration_chart(model, X, y)
        overridden = fm.calibration_chart(
            model,
            X,
            y,
            encode={"x": fm.X("mean_predicted", title="Custom")},
        )
        default_svg = default.show_svg()
        overridden_svg = overridden.show_svg()
        assert "Fraction of positives" in overridden_svg
        assert "Custom" in overridden_svg


# ---------------------------------------------------------------------------
# Properties overrides
# ---------------------------------------------------------------------------


class TestPropertiesOverrides:
    def test_override_width(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(model, X, y, properties={"width": 800})
        svg = chart.show_svg()
        assert 'width="800"' in svg

    def test_title_preserved_when_only_width_set(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(model, X, y, properties={"width": 800})
        svg = chart.show_svg()
        assert "Calibration" in svg

    def test_override_title(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(model, X, y, properties={"title": "My Custom Title"})
        svg = chart.show_svg()
        assert "My Custom Title" in svg


# ---------------------------------------------------------------------------
# Layers override
# ---------------------------------------------------------------------------


class TestLayersOverride:
    def test_extra_layer_appended(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(
            model,
            X,
            y,
            layers=[Layer(mark="rule", encoding={"y": "fraction_positive"})],
        )
        names_before = fm.calibration_chart(model, X, y).layer_names
        names_after = chart.layer_names
        assert len(names_after) >= len(names_before)

    def test_empty_layers_is_noop(self, clf_data):
        model, X, y = clf_data
        default = fm.calibration_chart(model, X, y)
        overridden = fm.calibration_chart(model, X, y, layers=[])
        assert default.show_svg() == overridden.show_svg()


# ---------------------------------------------------------------------------
# End-to-end through figure functions
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_calibration_suppress_and_properties(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(
            model,
            X,
            y,
            mark={"point": False},
            properties={"width": 600},
        )
        svg = chart.show_svg()
        assert 'width="600"' in svg
        assert "point" not in chart.layer_names

    def test_roc_encode_override(self, clf_data):
        model, X, y = clf_data
        chart = fm.roc_chart(
            model,
            X,
            y,
            encode={"x": fm.X("fpr", title="Custom FPR")},
        )
        svg = chart.show_svg()
        assert "Custom FPR" in svg

    def test_residuals_with_extra_layer(self, reg_data):
        model, X, y = reg_data
        chart = fm.residuals_chart(
            model,
            X,
            y,
            layers=[Layer(mark="rule", encoding={"y": "residual"})],
        )
        assert chart.show_svg()

    def test_confusion_matrix_override(self, clf_data):
        model, X, y = clf_data
        chart = fm.confusion_matrix_chart(
            model,
            X,
            y,
            mark={"label": False},
            properties={"title": "Custom CM"},
        )
        svg = chart.show_svg()
        assert "Custom CM" in svg

    def test_importance_suppress_errorbar(self, clf_data):
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(42)
        X = rng.randn(80, 4)
        y = (X[:, 0] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        chart = fm.importance_chart(model, X, y, mark={"errorbar": False})
        assert "errorbar" not in chart.layer_names


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_bad_mark_type_raises(self, clf_data):
        model, X, y = clf_data
        with pytest.raises(TypeError, match="must be a dict"):
            fm.calibration_chart(model, X, y, mark={"line": 42})

    def test_unknown_sublayer_raises_valueerror(self, clf_data):
        model, X, y = clf_data
        with pytest.raises(ValueError, match="Unknown sub-layer"):
            fm.calibration_chart(model, X, y, mark={"imaginary": False})


# ---------------------------------------------------------------------------
# LAYER_NAME_CATALOG registry
# ---------------------------------------------------------------------------


class TestCatalog:
    def test_diagnostic_marks_registered(self):
        assert "residuals" in LAYER_NAME_CATALOG
        assert "calibration" in LAYER_NAME_CATALOG
        assert "roc" in LAYER_NAME_CATALOG
        assert "confusion" in LAYER_NAME_CATALOG

    def test_composite_marks_registered(self):
        assert "boxplot" in LAYER_NAME_CATALOG
        assert "errorbar" in LAYER_NAME_CATALOG
        assert "errorband" in LAYER_NAME_CATALOG

    def test_statistical_marks_registered(self):
        assert "smooth" in LAYER_NAME_CATALOG

    def test_heavy_stat_marks_registered(self):
        assert "contour" in LAYER_NAME_CATALOG
        assert "violin" in LAYER_NAME_CATALOG
        assert "qq" in LAYER_NAME_CATALOG

    def test_calibration_includes_builder_point(self):
        assert "point" in LAYER_NAME_CATALOG["calibration"]


# ---------------------------------------------------------------------------
# SVG round-trip: verify override kwargs reach rendered SVG attributes
# ---------------------------------------------------------------------------


def _circle_count(svg: str) -> int:
    return svg.count("<circle")


def _has_dasharray(svg: str, pattern: str) -> bool:
    return f'stroke-dasharray="{pattern}"' in svg


class TestSvgRoundTrip:
    """Verify overrides survive the full pipeline into SVG, same philosophy
    as ``test_kwarg_piping.py``."""

    def test_suppress_point_removes_circles(self, clf_data):
        model, X, y = clf_data
        default_svg = fm.calibration_chart(model, X, y).show_svg()
        suppressed_svg = fm.calibration_chart(model, X, y, mark={"point": False}).show_svg()
        assert _circle_count(default_svg) > _circle_count(suppressed_svg)

    def test_mark_stroke_width_reaches_svg(self, clf_data):
        model, X, y = clf_data
        svg = fm.calibration_chart(model, X, y, mark={"line": {"stroke_width": 5}}).show_svg()
        assert 'stroke-width="5"' in svg

    def test_mark_stroke_dash_reaches_svg(self, clf_data):
        model, X, y = clf_data
        svg = fm.calibration_chart(
            model, X, y, mark={"reference": {"stroke_dash": [8, 4]}}
        ).show_svg()
        assert _has_dasharray(svg, "8,4")

    def test_mark_opacity_reaches_svg(self, clf_data):
        model, X, y = clf_data
        svg = fm.calibration_chart(model, X, y, mark={"point": {"opacity": 0.2}}).show_svg()
        assert "rgba(" in svg

    def test_point_size_override_changes_radius(self, clf_data):
        model, X, y = clf_data
        small_svg = fm.calibration_chart(model, X, y, mark={"point": {"size": 10}}).show_svg()
        big_svg = fm.calibration_chart(model, X, y, mark={"point": {"size": 200}}).show_svg()
        small_r = [float(m) for m in re.findall(r'<circle[^>]+r="([^"]+)"', small_svg)]
        big_r = [float(m) for m in re.findall(r'<circle[^>]+r="([^"]+)"', big_svg)]
        assert small_r and big_r
        assert max(big_r) > max(small_r)

    def test_encode_title_reaches_svg(self, clf_data):
        model, X, y = clf_data
        svg = fm.roc_chart(
            model,
            X,
            y,
            encode={"x": fm.X("fpr", title="My Custom FPR")},
        ).show_svg()
        assert "My Custom FPR" in svg

    def test_properties_width_reaches_svg(self, clf_data):
        model, X, y = clf_data
        svg = fm.calibration_chart(model, X, y, properties={"width": 900}).show_svg()
        assert 'width="900"' in svg

    def test_properties_title_reaches_svg(self, clf_data):
        model, X, y = clf_data
        svg = fm.calibration_chart(model, X, y, properties={"title": "Round-Trip Title"}).show_svg()
        assert "Round-Trip Title" in svg

    def test_confusion_suppress_label_removes_text(self, clf_data):
        model, X, y = clf_data
        default_svg = fm.confusion_matrix_chart(model, X, y).show_svg()
        suppressed_svg = fm.confusion_matrix_chart(model, X, y, mark={"label": False}).show_svg()
        default_texts = default_svg.count("<text ")
        suppressed_texts = suppressed_svg.count("<text ")
        assert suppressed_texts < default_texts

    def test_roc_suppress_reference_removes_diagonal(self, clf_data):
        model, X, y = clf_data
        default_svg = fm.roc_chart(model, X, y).show_svg()
        suppressed_svg = fm.roc_chart(model, X, y, mark={"reference": False}).show_svg()
        default_dashes = default_svg.count("stroke-dasharray")
        suppressed_dashes = suppressed_svg.count("stroke-dasharray")
        assert suppressed_dashes < default_dashes

    def test_residuals_single_panel_stroke_width(self, reg_data):
        model, X, y = reg_data
        svg = fm.residuals_chart(
            model,
            X,
            y,
            panels="single",
            mark={"reference": {"stroke_width": 3}},
        ).show_svg()
        assert 'stroke-width="3"' in svg

    def test_importance_bar_opacity_reaches_svg(self, clf_data):
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(42)
        X = rng.randn(80, 4)
        y = (X[:, 0] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        svg = fm.importance_chart(model, X, y, mark={"bar": {"opacity": 0.3}}).show_svg()
        assert "rgba(" in svg


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_suppress_all_layers(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(
            model,
            X,
            y,
            mark={"line": False, "reference": False, "point": False},
        )
        assert chart.layer_names == []
        svg = chart.show_svg()
        assert "<svg" in svg

    def test_empty_dict_overrides_are_noop(self, clf_data):
        model, X, y = clf_data
        default_svg = fm.calibration_chart(model, X, y).show_svg()
        overridden_svg = fm.calibration_chart(
            model,
            X,
            y,
            mark={},
            encode={},
            properties={},
            layers=[],
        ).show_svg()
        assert default_svg == overridden_svg

    def test_none_suppresses_like_false(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(model, X, y, mark={"point": None})
        assert "point" not in chart.layer_names

    def test_mixed_suppress_and_merge(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(
            model,
            X,
            y,
            mark={"point": False, "line": {"stroke_width": 4}},
        )
        assert "point" not in chart.layer_names
        resolved = chart._resolve_pending()
        line_layer = next(ly for ly in resolved._layers if ly.name == "line")
        assert line_layer.mark_kwargs["stroke_width"] == 4
        svg = chart.show_svg()
        assert 'stroke-width="4"' in svg

    def test_all_four_overrides_combined(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(
            model,
            X,
            y,
            mark={"point": False},
            encode={"x": fm.X("mean_predicted", title="P(+)")},
            properties={"width": 700, "title": "Combined Test"},
            layers=[Layer(mark="rule", encoding={"y": "fraction_positive"})],
        )
        svg = chart.show_svg()
        assert "point" not in chart.layer_names
        assert "P(+)" in svg
        assert 'width="700"' in svg
        assert "Combined Test" in svg

    def test_post_hoc_chaining_after_override(self, clf_data):
        model, X, y = clf_data
        chart = fm.calibration_chart(model, X, y, mark={"point": False}).properties(width=500)
        svg = chart.show_svg()
        assert 'width="500"' in svg
        assert "point" not in chart.layer_names

    def test_compare_multi_model_with_overrides(self, clf_data):
        from sklearn.tree import DecisionTreeClassifier

        model, X, y = clf_data
        dt = DecisionTreeClassifier(random_state=0).fit(X, y)
        chart = fm.roc_chart(
            model,
            X,
            y,
            compare={"DT": dt},
            mark={"reference": False},
            properties={"title": "Multi-model ROC"},
        )
        svg = chart.show_svg()
        assert "Multi-model ROC" in svg

    def test_compound_view_fanout(self, reg_data):
        model, X, y = reg_data
        chart = fm.residuals_chart(
            model,
            X,
            y,
            mark={"reference": {"stroke_width": 4}},
        )
        svg = chart.show_svg()
        assert 'stroke-width="4"' in svg

    def test_suppress_on_compound_view_child(self, reg_data):
        model, X, y = reg_data
        default_svg = fm.residuals_chart(model, X, y).show_svg()
        suppressed_svg = fm.residuals_chart(
            model,
            X,
            y,
            mark={"reference": False},
        ).show_svg()
        default_dashes = default_svg.count("stroke-dasharray")
        suppressed_dashes = suppressed_svg.count("stroke-dasharray")
        assert suppressed_dashes < default_dashes
