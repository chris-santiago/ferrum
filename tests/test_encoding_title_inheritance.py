"""Regression tests for encoding title inheritance in multi-layer charts.

The Rust renderer must display encoding `title` (not the raw field name)
on axes, even for composite/multi-layer charts where layers inherit
chart-level encodings via ``Encoding::inherit_from()``.

Bug discovered 2026-05-12: ``inherit_from()`` only propagated ``scale``,
silently dropping ``title``, ``scheme``, and all other metadata fields.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm


def _svg_axis_labels(svg: str) -> list[str]:
    """Extract axis title text elements from SVG."""
    return re.findall(r'class="axis-title"[^>]*>([^<]+)</text>', svg)


class TestSingleLayerTitle:
    """Single-layer charts should render encoding titles (baseline)."""

    def test_point_chart_renders_x_title(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        svg = (
            fm.Chart(df)
            .mark_point()
            .encode(x=fm.X("a", title="Custom X"), y=fm.Y("b", title="Custom Y"))
            .to_svg()
        )
        assert "Custom X" in svg
        assert "Custom Y" in svg


class TestMultiLayerTitleInheritance:
    """Multi-layer (composite) charts must inherit titles from chart-level encoding."""

    def test_boxplot_inherits_axis_titles(self) -> None:
        df = pl.DataFrame(
            {
                "group": ["a"] * 20 + ["b"] * 20,
                "value": [float(v) for v in range(20)] + [float(v) for v in range(5, 25)],
            }
        )
        svg = (
            fm.Chart(df)
            .mark_boxplot()
            .encode(
                x=fm.X("group", title="Category"),
                y=fm.Y("value", title="Measurement"),
            )
            .to_svg()
        )
        assert "Category" in svg
        assert "Measurement" in svg

    def test_roc_chart_renders_human_readable_labels(self) -> None:
        pytest.importorskip("sklearn")
        from sklearn.datasets import load_breast_cancer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X, y = load_breast_cancer(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        model = LogisticRegression(max_iter=2000, random_state=0).fit(Xtr, ytr)
        svg = fm.roc_chart(model, Xte, yte).to_svg()
        assert "False Positive Rate" in svg
        assert "True Positive Rate" in svg
        assert ">fpr<" not in svg
        assert ">tpr<" not in svg

    def test_pr_chart_renders_human_readable_labels(self) -> None:
        pytest.importorskip("sklearn")
        from sklearn.datasets import load_breast_cancer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X, y = load_breast_cancer(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        model = LogisticRegression(max_iter=2000, random_state=0).fit(Xtr, ytr)
        svg = fm.pr_chart(model, Xte, yte).to_svg()
        assert "Recall" in svg
        assert "Precision" in svg
