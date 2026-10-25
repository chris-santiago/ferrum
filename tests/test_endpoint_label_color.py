"""Endpoint direct labels draw in their own series' color.

``gain_chart``, ``lift_chart``, ``learning_curve_chart`` and
``validation_curve_chart`` all replace a categorical legend with endpoint
text labels via ``_charts_with_endpoint_labels``. A text mark draws only a
color it binds itself (it never inherits one), so without an own binding
every label falls through to the theme's label ink regardless of which
series it names -- leaving a reader unable to tell which line a label
belongs to by color alone. These tests pin that the shared helper binds
the label layer's own ``color`` to the series field, with no legend of its
own, so each label reads in the color of the line it names.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import ferrum
from ferrum import Chart
from ferrum._direct_label import _direct_label_endpoint
from ferrum.encoding import Color
from ferrum.plots._helpers import _charts_with_endpoint_labels
from tests._svg_extents import ordered_text_entries

#: The theme's title ink and tick ink -- every axis title, axis tick and
#: chart title renders in one of these two, never a series color. A text
#: run with any other fill is a drawn data label, which is what lets the
#: helpers below tell a direct label apart from an axis tick that happens
#: to share its text (e.g. a binary gain chart's "0"/"1" class labels sit at
#: the same y values as its "0"/"1" axis ticks).
_TITLE_INK = "#1f2937"
_TICK_INK = "#6b7280"
_CHROME_INK = {_TITLE_INK, _TICK_INK, ""}


def _text_fill_pairs(svg: str) -> list[tuple[str, str]]:
    """Return ``(text, fill)`` for every non-empty ``<text>`` element, in document order."""
    out = []
    for attrs_str, text in re.findall(r"<text\s+([^>]*)>([^<]*)</text>", svg):
        text = text.strip()
        if not text:
            continue
        fill = dict(re.findall(r'([\w-]+)="([^"]+)"', attrs_str)).get("fill", "")
        out.append((text, fill))
    return out


def _colored_label_fills(svg: str) -> dict[str, str]:
    """Map each rendered data-label text to its own color, chrome excluded.

    Filtering on fill (not on document position) is what lets a label
    whose text coincides with an axis tick's (the binary gain/lift "0"/"1"
    case) resolve to the label's own color rather than the tick's.
    """
    return {text: fill for text, fill in _text_fill_pairs(svg) if fill not in _CHROME_INK}


def _make_binary_gain_lift_fixture():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 10)
    rng = np.random.default_rng(1)
    y_pred = np.clip(y_true + rng.normal(0, 0.3, size=len(y_true)), 0, 1)
    return y_true, y_pred


def _make_learning_curve_source():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return ferrum.ModelSource(LogisticRegression(), X, y)


def _make_class_names_source():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression().fit(X, y)
    return ferrum.ModelSource(clf, X, y, class_names=["malignant", "benign"])


def _make_iris_source():
    iris = load_iris()
    clf = LogisticRegression(max_iter=1000).fit(iris.data, iris.target)
    return ferrum.ModelSource(clf, iris.data, iris.target)


# ---------------------------------------------------------------------------
# Structural: the shared helper binds the label layer's own color.
# ---------------------------------------------------------------------------


def test_charts_with_endpoint_labels_binds_own_color_with_no_legend():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 1, 2, 3],
            "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "split": ["a", "a", "a", "b", "b", "b"],
        }
    )
    base = Chart(df).encode(x="x", y="y", color=Color("split", legend=None)).mark_line()
    chart = _charts_with_endpoint_labels(base, label_field="split", x_col="x", y_col="y")

    text_layer = chart._layers[-1]
    assert text_layer.mark == "text"
    color_binding = text_layer.encoding["color"]
    assert isinstance(color_binding, Color)
    assert color_binding.field == "split"
    assert color_binding.option("legend") is None


def test_direct_label_endpoint_without_color_field_keeps_default_ink():
    """Negative control: calling the lower-level helper with ``color_field``
    omitted (``None``) draws every label in the theme's label ink. This is
    what proves the colored result in the other tests comes from the
    ``color_field`` binding itself, not from some unrelated scale or theme
    default that colors every label regardless.
    """
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 1, 2, 3],
            "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "split": ["a", "a", "a", "b", "b", "b"],
        }
    )
    base = Chart(df).encode(x="x", y="y", color=Color("split", legend=None)).mark_line()
    chart = _direct_label_endpoint(base, label_field="split", x_col="x", y_col="y")
    svg = chart.to_svg()

    fills = dict(_text_fill_pairs(svg))
    assert fills["a"] == _TITLE_INK
    assert fills["b"] == _TITLE_INK
    # With no own color binding, "a" and "b" are indistinguishable from the
    # axis title ink, so the chrome filter drops them entirely.
    assert _colored_label_fills(svg) == {}


def test_direct_label_endpoint_color_field_preserves_text_position_and_order():
    """Binding color must not move a label or change its stagger order."""
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 1, 2, 3],
            "y": [0.1, 0.2, 0.3, 0.4, 0.3, 0.6],
            "split": ["a", "a", "a", "b", "b", "b"],
        }
    )
    base = Chart(df).encode(x="x", y="y", color=Color("split", legend=None)).mark_line()

    plain = _direct_label_endpoint(base, label_field="split", x_col="x", y_col="y")
    colored = _direct_label_endpoint(
        base, label_field="split", x_col="x", y_col="y", color_field="split"
    )

    plain_positions = [e for e in ordered_text_entries(plain.to_svg()) if e[2] in ("a", "b")]
    colored_positions = [e for e in ordered_text_entries(colored.to_svg()) if e[2] in ("a", "b")]
    assert colored_positions == plain_positions


# ---------------------------------------------------------------------------
# Rendered production charts: gain, lift, learning-curve and
# validation-curve, binary and multiclass, class_names= and real names.
# ---------------------------------------------------------------------------


def test_gain_chart_binary_endpoint_labels_match_line_colors():
    y_true, y_pred = _make_binary_gain_lift_fixture()
    svg = ferrum.gain_chart(y_true=y_true, y_pred=y_pred).to_svg()
    assert _colored_label_fills(svg) == {
        "0": "#2563eb",
        "1": "#dc2626",
        "Baseline": "#d4a017",
    }
    assert ">Class<" not in svg and ">class<" not in svg


def test_lift_chart_binary_endpoint_labels_match_line_colors():
    y_true, y_pred = _make_binary_gain_lift_fixture()
    svg = ferrum.lift_chart(y_true=y_true, y_pred=y_pred).to_svg()
    assert _colored_label_fills(svg) == {
        "0": "#2563eb",
        "1": "#dc2626",
        "Baseline": "#d4a017",
    }
    assert ">Class<" not in svg and ">class<" not in svg


def test_learning_curve_chart_endpoint_labels_match_line_colors():
    source = _make_learning_curve_source()
    svg = ferrum.learning_curve_chart(source, cv=3, train_sizes=[0.3, 0.6, 1.0]).to_svg()
    assert _colored_label_fills(svg) == {
        "Training Score": "#dc2626",
        "Cross-Validation Score": "#2563eb",
    }
    assert ">split<" not in svg


def test_validation_curve_chart_endpoint_labels_match_line_colors():
    source = _make_learning_curve_source()
    svg = ferrum.validation_curve_chart(source, param="C", values=[0.1, 1.0, 10.0], cv=3).to_svg()
    assert _colored_label_fills(svg) == {
        "Training Score": "#dc2626",
        "Cross-Validation Score": "#2563eb",
    }
    assert ">split<" not in svg


def test_gain_chart_class_names_endpoint_labels_colored():
    """class_names= real names ship colored, not just as bare digits."""
    source = _make_class_names_source()
    svg = ferrum.gain_chart(source).to_svg()
    assert _colored_label_fills(svg) == {
        "malignant": "#2563eb",
        "benign": "#dc2626",
        "Baseline": "#d4a017",
    }


def test_lift_chart_class_names_endpoint_labels_colored():
    source = _make_class_names_source()
    svg = ferrum.lift_chart(source).to_svg()
    assert _colored_label_fills(svg) == {
        "malignant": "#2563eb",
        "benign": "#dc2626",
        "Baseline": "#d4a017",
    }


def test_gain_chart_multiclass_endpoint_labels_colored():
    """Multiclass (iris) keeps its real integer class names, each colored."""
    source = _make_iris_source()
    svg = ferrum.gain_chart(source).to_svg()
    assert _colored_label_fills(svg) == {
        "0": "#2563eb",
        "1": "#dc2626",
        "2": "#d4a017",
        "Baseline": "#0f766e",
    }


def test_lift_chart_multiclass_endpoint_labels_colored():
    """Multiclass (iris) on the lift chart too, not just gain."""
    source = _make_iris_source()
    svg = ferrum.lift_chart(source).to_svg()
    assert _colored_label_fills(svg) == {
        "0": "#2563eb",
        "1": "#dc2626",
        "2": "#d4a017",
        "Baseline": "#0f766e",
    }


# ---------------------------------------------------------------------------
# Control: ROC/PR never route through the endpoint-label helper.
# ---------------------------------------------------------------------------


def test_roc_and_pr_chart_unaffected_by_endpoint_label_change():
    y_true, y_pred = _make_binary_gain_lift_fixture()
    roc_svg = ferrum.roc_chart(y_true=y_true, y_pred=y_pred).to_svg()
    pr_svg = ferrum.pr_chart(y_true=y_true, y_pred=y_pred).to_svg()
    # Both keep their categorical "Class" legend -- the direct-label path
    # this change touches is never reached by roc_chart/pr_chart.
    assert ">Class<" in roc_svg
    assert ">Class<" in pr_svg
