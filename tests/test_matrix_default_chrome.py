"""``heatmap`` and ``clustermap`` default chrome: chart title, axis titles,
colorbar title, and a colorbar whose tick labels stay inside the canvas.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm


@pytest.fixture
def df_wide() -> pl.DataFrame:
    rng = np.random.default_rng(2)
    return pl.DataFrame(
        {
            "gene": [f"g{i}" for i in range(5)],
            **{c: rng.normal(0, 1, 5).tolist() for c in "ABCD"},
        }
    )


def _texts(svg: str) -> list[str]:
    svg = re.sub(r"<style>.*?</style>", "", svg, flags=re.S)
    return re.findall(r"<text\b[^>]*>([^<]*)</text>", svg)


def _chrome(svg: str) -> list[str]:
    """Every text that is not a tick label (number, row id, or value column)."""
    return [t for t in _texts(svg) if not re.fullmatch(r"-?[\d.]+|g\d|[ABCD]", t)]


FIGURES = {"heatmap": (fm.heatmap, "Heatmap"), "clustermap": (fm.clustermap, "Clustermap")}


@pytest.mark.parametrize("name", list(FIGURES))
def test_default_titles_with_id_column(df_wide, name):
    fn, title = FIGURES[name]
    assert sorted(_chrome(fn(df_wide).to_svg())) == sorted([title, "gene", "Column", "Value"])


@pytest.mark.parametrize("name", list(FIGURES))
def test_default_titles_without_id_column(df_wide, name):
    fn, title = FIGURES[name]
    chrome = _chrome(fn(df_wide.drop("gene")).to_svg())
    # Synthetic row ids are digits; drop them with the other tick labels.
    assert sorted(t for t in chrome if not t.isdigit()) == sorted([title, "Row", "Column", "Value"])


def test_single_value_column_titles_the_colorbar_with_it(df_wide):
    # clustermap needs at least two value columns to cluster the column axis.
    df = df_wide.select("gene", "A").rename({"A": "expr"})
    chrome = [t for t in _texts(fm.heatmap(df).to_svg()) if not re.fullmatch(r"-?[\d.]+|g\d", t)]
    # "expr" twice: once as the column-axis tick label, once as the colorbar title.
    assert sorted(chrome) == sorted(["Heatmap", "gene", "Column", "expr", "expr"])


@pytest.mark.parametrize("name", list(FIGURES))
def test_user_title_replaces_the_default(df_wide, name):
    fn, title = FIGURES[name]
    chrome = _chrome(fn(df_wide, properties={"title": "Expression"}).to_svg())
    assert "Expression" in chrome
    assert title not in chrome


@pytest.mark.parametrize("name", list(FIGURES))
def test_empty_title_suppresses_the_default(df_wide, name):
    fn, _ = FIGURES[name]
    chrome = _chrome(fn(df_wide, properties={"title": ""}).to_svg())
    assert sorted(t for t in chrome if t) == sorted(["gene", "Column", "Value"])


def test_clustermap_heatmap_cell_is_unscaled_and_colorbar_labels_fit(df_wide):
    """The heatmap cell renders at its native size, so its colorbar (placed
    in the cell's native coordinates) lands inside the canvas: every colorbar
    tick label ends left of the right canvas edge."""
    svg = fm.clustermap(df_wide).to_svg()
    canvas_w = float(re.search(r'<svg[^>]*\bwidth="([\d.]+)"', svg).group(1))
    assert " scale(" not in svg
    labels = re.findall(
        r'<text x="([\d.]+)"[^>]*font-size="([\d.]+)"[^>]*text-anchor="start">(-?[\d.]+)</text>',
        svg,
    )
    assert len(labels) >= 3
    for x, size, text in labels:
        # 0.5em per glyph is a lower bound for these digits in Inter.
        assert float(x) + 0.5 * float(size) * len(text) <= canvas_w, (text, x, canvas_w)
