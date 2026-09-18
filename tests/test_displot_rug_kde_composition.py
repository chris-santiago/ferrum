"""``displot(kind="rug", kde=True)`` draws its curve on a real density axis (#201).

The docstring advertised this composition and the call raised at render:
``ValueError: encoding 'y' expected EncodingSpec, got None``. ``kind="rug"`` is
the only kind whose root encoding lacks a ``y`` -- ``hist`` remaps ``y`` onto
the ``Bin`` count column, ``ecdf`` binds the cumulative count, and
``kind="kde"`` never enters the overlay block at all -- so it was the only kind
whose layer promotion left the chart with no value axis to offer.

**What this module proves, and what it deliberately does not.** #201 was fixed
in two independent halves, and the plan required each to be provable without
the other. The other half (Task #201-D, ``tests/test_single_axis_eligibility.py``
and the Rust resolver) made a layered spec's positional axes resolvable from
the layer that binds them. This module pins the COMPOSITION: the chart's own
root encoding takes its ``y`` from the KDE overlay's output column, so the
density axis is the chart's, not a fallback read out of a layer.

The observable that separates them is the axis TITLE. An axis resolved from a
layer's binding has no chart-level encoding to take a title from and renders
untitled (measured at the sibling half's commit: correct geometry, no y title);
an axis the chart itself binds is titled ``density``, exactly as ``kind="kde"``
titles the same quantity. The geometry assertions here are not that proof --
they are the guard against the candidate this fix rejected.

**The rejected candidate.** Keeping the rug's ``tick`` as the root mark also
renders, by qualifying the spec for the single-axis exemption -- and the
exemption synthesizes a ``0 … 1`` unit scale for the absent axis, which
squashed the density curve (peak ~0.36) into the bottom ~6% of the panel under
a meaningless ladder. So every test below asserts both halves of "the curve is
read correctly": its vertical extent spans the panel, and the ticks it is read
against are in density units. Neither holds under the fabricated scale.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum._registry import registry_output_columns
from tests._svg_extents import axis_tick_labels, y_axis_title
from tests._svg_rects import polylines, rug_tick_x1_by_stroke

#: Canvas height of a default-sized displot, in px. The curve extent
#: assertions are stated against it so "occupies the panel" is not a bare
#: magic number.
_CANVAS_HEIGHT_PX = 480.0

#: The peak density of a standard normal is 1/sqrt(2*pi) ~ 0.399, so a real
#: density axis tops out just under 0.4. The synthesized unit scale tops out
#: at exactly 1, and a count axis at the sample size -- this band excludes
#: both.
_DENSITY_AXIS_TOP = (0.2, 0.6)

_N_ROWS = 200


@pytest.fixture
def normal_df() -> pl.DataFrame:
    """A seeded standard-normal sample with a two-level group column."""
    rng = np.random.default_rng(0)
    return pl.DataFrame({"v": rng.normal(size=_N_ROWS), "g": ["a", "b"] * (_N_ROWS // 2)})


def _curve_y_extent(svg: str) -> tuple[float, float]:
    """``(min, max)`` screen y over every rendered curve vertex."""
    ys = [y for _, points in polylines(svg) for _, y in points]
    assert ys, "expected at least one rendered density curve"
    return min(ys), max(ys)


def _assert_density_axis(svg: str) -> list[float]:
    """Assert the y axis is a titled density axis and return its tick values.

    The two facts the rejected candidate got wrong, asserted together: the
    axis is titled by the quantity it shows, and its ticks are in that
    quantity's units rather than the fabricated ``0 … 1`` ladder.
    """
    assert y_axis_title(svg) == "density", (
        "the y axis carries no density title -- the chart's own encoding does "
        f"not bind it, got {y_axis_title(svg)!r}"
    )
    labels = axis_tick_labels(svg, axis="y")
    values = [float(label) for label in labels]
    assert values, "no y tick labels rendered"
    lo, hi = _DENSITY_AXIS_TOP
    assert lo < max(values) < hi, (
        f"y ticks are not in density units (a standard normal peaks near 0.4); "
        f"a fabricated unit scale tops out at 1 and a count axis at {_N_ROWS}. got {labels}"
    )
    return values


def _assert_curve_spans_the_panel(svg: str) -> None:
    """The curve must occupy the panel, not a sliver at its foot.

    Pre-fix measurement of the rejected candidate on this same canvas: the
    curve spanned ~26px at the very bottom (y in [374.6, 400.3] of 480).
    """
    top, bottom = _curve_y_extent(svg)
    assert bottom - top > _CANVAS_HEIGHT_PX * 0.6, (
        f"the density curve is squashed: it spans {bottom - top:.1f}px of a "
        f"{_CANVAS_HEIGHT_PX:.0f}px canvas"
    )
    assert top < _CANVAS_HEIGHT_PX * 0.2, (
        f"the density curve never rises into the panel: its highest point is y={top:.1f}"
    )


def test_rug_with_a_kde_overlay_draws_a_titled_density_axis(normal_df: pl.DataFrame) -> None:
    """The composition renders: ticks at the base, curve over a density axis.

    The assertion this fix makes true is the axis title (``density``); the
    extent and unit assertions guard the rejected candidate, which drew the
    curve correctly-coloured and wrongly-scaled.
    """
    svg = fm.displot(normal_df, x="v", kind="rug", kde=True).to_svg()

    _assert_density_axis(svg)
    _assert_curve_spans_the_panel(svg)

    ticks = rug_tick_x1_by_stroke(svg)
    assert [len(positions) for positions in ticks.values()] == [_N_ROWS], (
        f"expected one rug tick per row in a single colour, got "
        f"{ {stroke: len(p) for stroke, p in ticks.items()} }"
    )


def test_the_rug_ticks_sit_at_the_base_of_the_density_axis(normal_df: pl.DataFrame) -> None:
    """The ticks mark values along x at the axis's foot, below the whole curve.

    The rug keeps its own x-only encoding through layer promotion, so binding
    the chart's ``y`` must not lift the ticks into the plot area or drop the
    curve onto them.
    """
    from tests._svg_rects import svg_elements

    svg = fm.displot(normal_df, x="v", kind="rug", kde=True).to_svg()
    _, curve_bottom = _curve_y_extent(svg)
    tick_tops = [
        min(float(attrs["y1"]), float(attrs["y2"]))
        for attrs in svg_elements(svg, "line", lambda a: a.get("stroke") == "#2563eb")
    ]
    assert len(tick_tops) == _N_ROWS
    assert min(tick_tops) >= curve_bottom - 1.0, (
        f"rug ticks reach above the density axis's base (topmost tick y="
        f"{min(tick_tops):.1f}, curve foot y={curve_bottom:.1f})"
    )


def test_the_density_axis_is_bound_by_the_chart_not_read_off_a_layer(
    normal_df: pl.DataFrame,
) -> None:
    """The root encoding binds ``y`` to the KDE transform's own density column.

    This is the structural half of the title assertion above, and the line
    between this fix and Task #201-D's. Both of that task's mechanisms are
    reached only through a channel the chart's own encoding leaves unbound:
    the resolver takes its binding from ``spec.encoding`` when there is one
    and consults the layers otherwise, and it synthesizes a scale only when
    exactly one of the two channels is bound anywhere. This spec binds both
    at the root, so neither the layer fallback nor the single-axis exemption
    is reachable for it, whatever eligibility would have answered.

    The column name is compared against the registry rather than spelled
    literally, so a Rust-side rename of the ``Kde`` output fails here instead
    of silently retitling the axis off the data it describes.
    """
    spec = fm.displot(normal_df, x="v", kind="rug", kde=True).to_dict()
    density_column = registry_output_columns("kde", ["value", "density"])["density"]

    assert spec["mark"] == "point", (
        "layer promotion is expected to leave the placeholder root mark in "
        "place; if that changed, this test's premise needs re-reasoning"
    )
    assert spec["encoding"]["y"]["field"] == density_column
    assert spec["encoding"]["x"]["field"] == "v"

    layers = {layer["name"]: layer for layer in spec["layers"]}
    assert layers["kde"]["encoding"]["y"]["field"] == density_column, (
        "the chart's y axis and the curve must name one column by construction"
    )
    assert "y" not in layers["rug"]["encoding"], (
        "the promoted rug layer must keep its own x-only encoding -- the ticks "
        "are drawn at the axis base, not plotted against density"
    )


def test_it_places_the_curve_where_the_kde_rug_route_does(normal_df: pl.DataFrame) -> None:
    """The two routes place the same two marks identically, under ``fill=False``.

    Both spell "a density curve over a rug of the same values", and the second
    route worked all along. Pinning them against each other is what keeps the
    newly-composable route honest: an axis that merely *renders* could still
    disagree with the established one about where the curve goes.

    Two bounds on the claim, so it is not read as stronger than it is measured.
    ``fill=False`` is required: ``kind="kde", rug=True`` defaults to a filled
    ``area``, which draws no polyline to compare against (it titles the axis
    ``density`` either way, which is what the title decision rests on). And
    even with the flag the two SVGs are not byte-identical -- the curve is
    painted before the rug ticks by one route and after them by the other,
    with character-identical vertices. The equivalence asserted here is
    placement: vertices, tick positions, both tick ladders and the title.
    """
    rug_first = fm.displot(normal_df, x="v", kind="rug", kde=True).to_svg()
    kde_first = fm.displot(normal_df, x="v", kind="kde", rug=True, fill=False).to_svg()

    assert y_axis_title(rug_first) == y_axis_title(kde_first)
    assert axis_tick_labels(rug_first, axis="y") == axis_tick_labels(kde_first, axis="y")
    assert axis_tick_labels(rug_first, axis="x") == axis_tick_labels(kde_first, axis="x")
    assert [points for _, points in polylines(rug_first)] == [
        points for _, points in polylines(kde_first)
    ], "the two routes place the density curve differently"
    assert rug_tick_x1_by_stroke(rug_first) == rug_tick_x1_by_stroke(kde_first)


def test_it_composes_with_hue(normal_df: pl.DataFrame) -> None:
    """One curve and one rug per ``hue`` level, on a shared density axis.

    Listed in the research as a shape that failed for this reason; it is the
    combination where the promoted layer, the overlay layer and the colour
    channel all have to agree about the same grouping column.
    """
    svg = fm.displot(normal_df, x="v", kind="rug", kde=True, hue="g").to_svg()

    _assert_density_axis(svg)
    _assert_curve_spans_the_panel(svg)

    strokes = {stroke for stroke, _ in polylines(svg)}
    assert len(strokes) == 2, f"expected one curve per hue level, got strokes {strokes}"

    ticks = rug_tick_x1_by_stroke(svg)
    assert sorted(len(positions) for positions in ticks.values()) == [
        _N_ROWS // 2,
        _N_ROWS // 2,
    ], f"expected each hue level's rug ticks, got { {s: len(p) for s, p in ticks.items()} }"
    assert strokes <= set(ticks), (
        f"curve colours {strokes} are not the rug colours {set(ticks)} -- the "
        "two marks disagree about the hue levels"
    )


@pytest.mark.parametrize(
    ("kind", "expected_y"),
    [("hist", "count"), ("ecdf", "count")],
    ids=["hist", "ecdf"],
)
def test_a_kind_that_binds_its_own_value_axis_keeps_it(
    normal_df: pl.DataFrame, kind: str, expected_y: str
) -> None:
    """The overlay only supplies ``y`` where the primary mark left none.

    ``hist`` puts the bin count on the value axis and ``ecdf`` the cumulative
    count; overwriting either with the overlay's density would rescale the
    primary mark off its own axis. The guard is "the root encoding has no
    ``y``", and these are the kinds that prove it is not vacuous.
    """
    spec = fm.displot(normal_df, x="v", kind=kind, kde=True).to_dict()
    assert spec["encoding"]["y"]["field"] == expected_y
