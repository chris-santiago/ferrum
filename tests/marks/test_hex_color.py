"""``mark_hex``'s chart-level ``color=`` is honored or refused, never inert.

The cells are always colored by the ``Hex`` transform's per-cell aggregate.
A ``color=`` naming the aggregated column supplies ``field`` when it is
omitted and styles the cell binding (title, scheme, legend); one naming the
transform's ``value`` output column styles it too. Any other ``color=``
would be drawn nowhere, so it refuses with a ``ValueError`` naming
``mark_hex`` and the column received. The cell legend is titled with the
aggregated column, or ``"Count"``, unless the caller's channel titles it.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm

_LEGEND_TITLE_RE = re.compile(r'font-weight="600"[^>]*>([^<]+)</text>')


@pytest.fixture()
def df() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    return pl.DataFrame(
        {
            "x": rng.normal(size=200).tolist(),
            "y": rng.normal(size=200).tolist(),
            "w": rng.normal(size=200).tolist(),
            "cat": ["a", "b"] * 100,
        }
    )


def _legend_title(chart: fm.Chart) -> str:
    """The one bold text that is neither axis title."""
    titles = [t for t in _LEGEND_TITLE_RE.findall(chart.to_svg()) if t not in ("x", "y")]
    assert len(titles) == 1, titles
    return titles[0]


@pytest.mark.parametrize(
    "mark_kwargs,color,received,aggregated",
    [
        ({}, "cat:N", "cat", "aggregate='count'"),
        ({"aggregate": "mean", "field": "w"}, "x", "x", "aggregate='mean' of field='w'"),
    ],
    ids=["count", "other-than-field"],
)
def test_a_color_drawn_nowhere_refuses_naming_mark_hex(
    df, mark_kwargs, color, received, aggregated
):
    chart = fm.Chart(df).mark_hex(**mark_kwargs).encode(x="x", y="y", color=color)
    with pytest.raises(ValueError) as err:
        chart.to_svg()
    assert str(err.value) == (
        f"mark_hex: color={received!r} is drawn nowhere; the cells are colored by "
        f"{aggregated}. Bind color= to the aggregated column, or drop it."
    )


def test_a_color_naming_the_aggregated_column_supplies_the_field(df):
    via_color = fm.Chart(df).mark_hex(aggregate="mean").encode(x="x", y="y", color="w")
    via_field = fm.Chart(df).mark_hex(aggregate="mean", field="w").encode(x="x", y="y")
    assert via_color.to_svg() == via_field.to_svg()
    assert _legend_title(via_color) == "w"


@pytest.mark.parametrize(
    "mark_kwargs,color,title",
    [
        ({}, None, "Count"),
        ({}, fm.Color("value", title="Cells"), "Cells"),
        ({"aggregate": "mean"}, fm.Color("w", title="Average w"), "Average w"),
        ({"aggregate": "max", "field": "w"}, "w", "w"),
    ],
    ids=["count-default", "output-column-title", "aggregated-column-title", "named-field"],
)
def test_the_cell_legend_title(df, mark_kwargs, color, title):
    enc = {"x": "x", "y": "y"} if color is None else {"x": "x", "y": "y", "color": color}
    assert _legend_title(fm.Chart(df).mark_hex(**mark_kwargs).encode(**enc)) == title


def test_a_color_scheme_paints_the_cells_like_cmap(df):
    """The scheme reaches the cells, not only the legend: the render equals
    ``mark_hex(cmap=)``'s, and differs from the unschemed one."""
    base = fm.Chart(df).mark_hex(aggregate="mean")
    plain = base.encode(x="x", y="y", color="w").to_svg()
    schemed = base.encode(x="x", y="y", color=fm.Color("w", scheme="magma")).to_svg()
    via_cmap = (
        fm.Chart(df)
        .mark_hex(aggregate="mean", cmap="magma")
        .encode(x="x", y="y", color=fm.Color("w", scheme="magma"))
        .to_svg()
    )
    assert schemed != plain
    assert schemed == via_cmap


def test_a_scheme_and_a_different_cmap_refuse(df):
    chart = (
        fm.Chart(df)
        .mark_hex(aggregate="mean", cmap="viridis")
        .encode(x="x", y="y", color=fm.Color("w", scheme="magma"))
    )
    with pytest.raises(ValueError) as err:
        chart.to_svg()
    assert str(err.value) == (
        "mark_hex: cmap='viridis' and color=fm.Color(..., scheme='magma') both set the "
        "cell colormap; pass only one."
    )


def test_a_color_drawn_nowhere_names_the_calling_entry(df):
    """A figure function stamps its own name onto a pending ``mark_hex``
    (``Chart._with_entry``), so the refusal speaks as the caller's entry,
    not the ``mark_hex`` method used to build it."""
    chart = fm.Chart(df).mark_hex().encode(x="x", y="y", color="cat:N")._with_entry("some_figure")
    with pytest.raises(ValueError) as err:
        chart.to_svg()
    assert str(err.value) == (
        "some_figure: color='cat' is drawn nowhere; the cells are colored by "
        "aggregate='count'. Bind color= to the aggregated column, or drop it."
    )


def test_a_scheme_and_a_different_cmap_refuse_naming_the_calling_entry(df):
    chart = (
        fm.Chart(df)
        .mark_hex(aggregate="mean", cmap="viridis")
        .encode(x="x", y="y", color=fm.Color("w", scheme="magma"))
        ._with_entry("some_figure")
    )
    with pytest.raises(ValueError) as err:
        chart.to_svg()
    assert str(err.value) == (
        "some_figure: cmap='viridis' and color=fm.Color(..., scheme='magma') both set the "
        "cell colormap; pass only one."
    )
