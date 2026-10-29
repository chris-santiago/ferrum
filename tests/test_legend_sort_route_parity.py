"""A legend's data-aware sort reads the drawing member's own rows on every route.

``Color("g", sort={"field": "w", "op": "sum", "order": "descending"})`` beside
``y="sum(v)"`` sorts the legend by a raw column the aggregate's own output
drops: ``g``'s groups (a: w=5+4=9, b: w=0.5+0.5=1, c: w=3+2=5) order
descending as ``a, c, b``, distinct from the aggregate's own insertion order
``a, b, c`` so the two are never accidentally equal. A static ``fm.LayerChart``
overlay reads that sort from the member's own pre-statistic rows, the same
rows the flat chart, ``+`` and ``.interactive()`` read, whether the sorted
member is the first or the second member of the overlay, and whether it is
declared as one layer of a two-layer member (``bar.layer(text)``). Rejects
falling back to the aggregate's drawn output, which does not carry ``w``, and
warning that the sort could not be applied with the legend left in insertion
order; also rejects reading a different member's rows when the sorted member
is not the overlay's first member.

Three further cases pin a legend sort that reaches its order by a different
route than the one above: a sort on the member's own drawn output
(``sort="-y"``), a non-aggregate member's shape/stroke_dash data-aware sort
over its own raw rows, and an overlay's binned color legend, whose edges and
bar fills come from a lineage lookup distinct from the sort's.
"""

from __future__ import annotations

import warnings
from typing import Callable

import polars as pl
import pytest

import ferrum as fm
from tests._legends import interactive_legend_set, legend_set
from tests._svg_rects import data_bar_rects

#: Three categories whose raw ``w`` sums (a=9, c=5, b=1) order descending as
#: a, c, b -- distinct from the insertion order a, b, c the aggregate's own
#: output would fall back to.
_SORT_BATCH = pl.DataFrame(
    {
        "g": ["a", "a", "b", "b", "c", "c"],
        "v": [1.0, 2.0, 5.0, 6.0, 3.0, 4.0],
        "w": [5.0, 4.0, 0.5, 0.5, 3.0, 2.0],
    }
)

#: A companion member sharing the categorical ``g`` domain, so every member
#: order in the cases below maps to the same (categorical) axis type.
_COMPANION = (
    fm.Chart(pl.DataFrame({"g": ["a", "b", "c"], "v2": [10.0, 20.0, 30.0]}))
    .mark_point()
    .encode(x="g", y="v2")
)

_SORT_W_SUM_DESC = {"field": "w", "op": "sum", "order": "descending"}


def _sorted_bar() -> fm.Chart:
    """A bar whose own ``y`` is ``sum(v)``, and whose color legend sorts by raw ``w``."""
    return (
        fm.Chart(_SORT_BATCH)
        .mark_bar()
        .encode(x="g", y="sum(v)", color=fm.Color("g", sort=_SORT_W_SUM_DESC))
    )


def _legend_g(svg: str) -> tuple[str, ...] | None:
    return legend_set(svg).get("g")


#: Each route under test, keyed by case id, as a builder of the chart that
#: route renders.
_ROUTES: dict[str, Callable[[], fm.Chart | fm.LayerChart]] = {
    "flat": _sorted_bar,
    "plus-bar-first": lambda: _sorted_bar() + _COMPANION,
    "plus-bar-second": lambda: _COMPANION + _sorted_bar(),
    "LCs-bar-first": lambda: fm.LayerChart(_sorted_bar(), _COMPANION),
    "LCs-bar-second": lambda: fm.LayerChart(_COMPANION, _sorted_bar()),
    "LCs-layered-member": lambda: fm.LayerChart(
        _sorted_bar().layer(fm.Layer(mark="text", encoding={"text": "sum(v)"})), _COMPANION
    ),
}


@pytest.mark.parametrize("case", sorted(_ROUTES))
def test_a_color_legends_data_aware_sort_reads_the_members_own_rows(case: str) -> None:
    """Every route sorts the legend a, c, b from the member's own raw ``w``, never falling back.

    Discriminates against reading the already-drawn aggregate output (which
    lacks ``w``) by asserting the descending-by-``w`` order a, c, b, not the
    aggregate's insertion order a, b, c, and against the fallback's own
    symptom by asserting no sort-ignored warning fires.
    """
    build = _ROUTES[case]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = build().to_svg()
    assert caught == [], f"{case}: expected no warning, got {[str(w.message) for w in caught]}"
    assert _legend_g(svg) == ("a", "c", "b"), f"{case}: legend order {_legend_g(svg)}"


@pytest.mark.parametrize("case", sorted(_ROUTES))
def test_an_interactive_routes_color_legend_reads_the_members_own_rows(case: str) -> None:
    """Every route's ``.interactive()`` scene sorts the legend a, c, b from the member's own raw ``w``.

    The same statement as the static-route cases above, read off the
    interactive scene legend instead of the SVG.
    """
    build = _ROUTES[case]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = build()
        legends = interactive_legend_set(chart)
    assert caught == [], f"{case}: expected no warning, got {[str(w.message) for w in caught]}"
    assert legends.get("g") == ("a", "c", "b"), f"{case}: legend order {legends.get('g')}"


def test_an_overlay_legend_sorted_on_the_drawn_output_orders_by_that_output() -> None:
    """``sort="-y"`` on an overlay member's color legend orders by the drawn aggregate.

    ``v``'s per-group sums (a=3, b=11, c=7) order descending as b, c, a. This
    sort reads the member's own drawn output, not its pre-statistic rows,
    which is the mechanism the color legend cases above read; pinning both
    here states each by its own order (b, c, a vs a, c, b).
    """
    bar = fm.Chart(_SORT_BATCH).mark_bar().encode(x="g", y="sum(v)", color=fm.Color("g", sort="-y"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = fm.LayerChart(bar, _COMPANION).to_svg()
    assert caught == []
    assert _legend_g(svg) == ("b", "c", "a")


@pytest.mark.parametrize("channel", ["shape", "stroke_dash"])
def test_an_overlay_members_shape_or_stroke_dash_sorts_its_own_raw_rows(channel: str) -> None:
    """A non-aggregate member's shape/stroke_dash legend sorts by its own raw rows.

    Neither member here runs a statistic, so the legend's rows are the
    member's own batch with no lineage lookup involved. Both channels sort
    by mean(w) descending, giving the same a, c, b order as the color cases
    above.
    """
    sort_mean_w_desc = {"field": "w", "op": "mean", "order": "descending"}
    if channel == "shape":
        member = (
            fm.Chart(_SORT_BATCH)
            .mark_point()
            .encode(x="g", y="v", shape=fm.Shape("g", sort=sort_mean_w_desc))
        )
    else:
        member = (
            fm.Chart(_SORT_BATCH)
            .mark_line()
            .encode(
                x="g",
                y="v",
                detail="g",
                stroke_dash=fm.StrokeDash("g", sort=sort_mean_w_desc),
            )
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = fm.LayerChart(member, _COMPANION).to_svg()
    assert caught == []
    assert _legend_g(svg) == ("a", "c", "b")


def _binned_color_bar(values: list[float]) -> fm.Chart:
    """A bar whose statistic groups by ``v`` and whose color bins the same raw ``v``."""
    data = pl.DataFrame({"v": values, "y": values})
    return fm.Chart(data).mark_bar().encode(x="v", y="mean(y)", color=fm.Color("v", bin=True))


def test_a_binned_overlay_legend_cuts_edges_from_the_rows_consumed() -> None:
    """A static overlay's binned color legend cuts its edges from the rows the members consumed.

    Two bar members over disjoint ranges (0..3 and 10..20) share one
    bin-ordinal color scale; six Sturges edges (0, 4, 8, 12, 16, 20, five
    bins) come from the 16 rows the two members' statistics together
    consumed, read through ``published_origin`` -- a lineage lookup distinct
    from the sort's ``pre_statistic_rows``. The edges place each member's
    rows into four of the five buckets (member one's four groups all fall in
    the lowest bucket; member two's four groups split across the remaining
    three), so the four bars drawn carry four distinct fills in bucket
    order. Rejects cutting the edges from the members' drawn (grouped)
    output instead of the rows their statistics consumed: that batch holds
    8 distinct ``v`` values rather than 16 rows and cuts different edges,
    which would place member two's v=10 and v=12 groups in the same bucket
    (sharing one fill) instead of two.
    """
    bar1 = _binned_color_bar([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    bar2 = _binned_color_bar([10.0, 10.0, 12.0, 12.0, 15.0, 15.0, 20.0, 20.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = fm.LayerChart(bar1, bar2).to_svg()
    assert caught == []
    assert legend_set(svg)["v"] == ("0", "4", "8", "12", "16", "20")
    bars = sorted(data_bar_rects(svg), key=lambda r: float(r["x"]))
    assert [r["fill"] for r in bars] == ["#eff6ff", "#60a5fa", "#2159e2", "#1e3a8a"]
