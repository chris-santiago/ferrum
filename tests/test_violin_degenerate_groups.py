"""A density statistic names a group it cannot estimate: ``Violin`` and ``QQ``.

A violin category, hue group or facet panel whose values cannot be estimated (no usable value, one
value or an infinite one, tied values, a zero density) has no body to draw. The statistic keeps its
slot, legend entry or panel and names it once through ``DegenerateFit``: the count reason is
``Kde``'s wording for the same data, the tied-value and zero-density reasons are the violin's own.
A chart that estimates every group is byte-identical and warns nothing. ``QQ`` does the same for an
all-null hue group or facet panel beside one with values, and still refuses
(``stat_qq: empty input after filter``) when nothing can be drawn: an unfaceted call with no usable
value, grouped or not, and a faceted call whose every panel has none. Rows for a report shared
with the render's census (the violin's inner box layers) are in
``test_null_group_warning.py``.
"""

from __future__ import annotations

import warnings
from typing import Callable

import polars as pl
import pytest

import ferrum as fm
from tests.test_null_group_warning import (
    FIT_BANDWIDTH,
    FIT_FEW,
    FIT_NONE,
    _census,
    _fit_message,
    _fit_warnings,
    _panel_fit_message,
    _render_capturing,
    _small_slot_frame,
)

_TIED = "its usable values are all equal"
_ZERO_DENSITY = "its density is zero across the evaluation range"

# Category ``z`` is the one the statistic cannot estimate; ``a`` and ``b`` hold two rows each.
_Z_CASES = [
    pytest.param([3.0], FIT_FEW, id="one"),
    pytest.param([3.0, 3.0, 3.0, 3.0], _TIED, id="tied"),
    pytest.param([None, None, None], FIT_FEW, id="null"),
]
_INNERS = ["none", "box", "quartile", "point"]
# (door, inner): the figure draws the default inner box and has no knob for it.
_DOORS = [pytest.param("mark", inner, id=f"mark-{inner}") for inner in _INNERS] + [
    pytest.param("catplot", "box", id="catplot")
]


def _violin_door(door: str, df: pl.DataFrame, inner: str, hue: str | None = None) -> fm.Chart:
    """The violin of ``w`` by category ``c`` through ``door``: the mark method or the figure.

    The figure draws the default inner box and has no ``inner`` knob, so ``_DOORS`` pairs it
    with ``"box"`` alone. A horizontal catplot is not built here: it still names the
    lowering's channel."""
    if door == "mark":
        chart = fm.Chart(df).mark_violin(inner=inner)
        return chart.encode(x="c", y="w", **({} if hue is None else {"color": hue}))
    return fm.catplot(df, x="c", y="w", kind="violin", **({} if hue is None else {"hue": hue}))


def _faceted_violin(door: str, inner: str, df: pl.DataFrame) -> fm.Chart:
    if door == "mark":
        return fm.Chart(df).mark_violin(inner=inner).encode(x="c", y="w").facet(col="f")
    return fm.catplot(df, x="c", y="w", kind="violin", col="f")


def _hue_frame() -> pl.DataFrame:
    """Categories ``a`` and ``b`` of hue levels ``p`` (four usable values) and ``q`` (all null)."""
    rows = [
        (category, level, None if level == "q" else value)
        for category in ("a", "b")
        for level, values in (("p", (1.0, 2.0, 3.0, 4.0)), ("q", (5.0, 6.0, 7.0, 8.0)))
        for value in values
    ]
    return pl.DataFrame(rows, schema={"c": pl.Utf8, "g": pl.Utf8, "w": pl.Float64}, orient="row")


def _panel_frame() -> pl.DataFrame:
    """Facet panels ``p`` (categories ``a`` and ``b`` of four values) and ``q`` (all null)."""
    rows = [
        (panel, category, None if panel == "q" else value)
        for panel in ("p", "q")
        for category in ("a", "b")
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    return pl.DataFrame(rows, schema={"f": pl.Utf8, "c": pl.Utf8, "w": pl.Float64}, orient="row")


# ---------------------------------------------------------------------------
# Violin: a category it cannot estimate is named by its own statistic.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("door,inner", _DOORS)
@pytest.mark.parametrize("z_values,reason", [c for c in _Z_CASES if c.id != "null"])
def test_a_violin_group_it_cannot_estimate_is_named(
    z_values: list[float | None], reason: str, inner: str, door: str
):
    """One value, or values all equal, give a category no density. It keeps its slot, the
    statistic names it once whatever the inner mark, and the inner marks, which have a value
    to draw, add no census report."""
    chart = _violin_door(door, _small_slot_frame(z_values), inner)
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert _fit_warnings(shown, "Violin") == [_fit_message("Violin", "c=z", reason)]
    assert _census(shown) == []
    assert ">z<" in svg


def test_an_all_null_violin_category_keeps_its_slot_and_is_named_once_by_violin():
    """With ``inner='none'`` the body is the only layer over the category, so the statistic names
    it and the census, which defers to it by lineage, adds nothing on the body. (The figure
    always draws an inner box, so its census rows are in ``test_null_group_warning.py``.)"""
    chart = _violin_door("mark", _small_slot_frame([None, None, None]), "none")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_fit_message("Violin", "c=z", FIT_FEW)]
    assert ">z<" in svg


@pytest.mark.parametrize("inner", _INNERS[1:])
def test_an_all_null_violin_category_is_named_by_violin_whatever_the_inner_mark(inner: str):
    """The statistic's warning does not depend on which inner marks sit over the category."""
    chart = _violin_door("mark", _small_slot_frame([None, None, None]), inner)
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert _fit_warnings(shown, "Violin") == [_fit_message("Violin", "c=z", FIT_FEW)]
    assert ">z<" in svg


@pytest.mark.parametrize("door,inner", [_DOORS[0], _DOORS[1], _DOORS[-1]])
def test_a_violin_hue_group_a_null_column_empties_is_named(door: str, inner: str):
    """Hue level ``q`` is null in every category: each category's ``q`` group is named once,
    and the legend keeps the level."""
    chart = _violin_door(door, _hue_frame(), inner, hue="g")
    svg, shown = _render_capturing(chart.to_svg, action="default")
    assert sorted(_fit_warnings(shown, "Violin")) == [
        _fit_message("Violin", f"c={category}, g=q", FIT_FEW) for category in "ab"
    ]
    assert ">q<" in svg
    if inner == "none":
        assert _census(shown) == []


@pytest.mark.parametrize("door,inner", [_DOORS[0], _DOORS[-1]])
def test_a_violin_facet_panel_a_null_column_empties_is_named(door: str, inner: str):
    """Panel ``q`` holds only nulls: each of its categories is named once with the panel clause,
    and the panel is still drawn (its strip title stays)."""
    svg, shown = _render_capturing(
        _faceted_violin(door, inner, _panel_frame()).to_svg, action="default"
    )
    assert sorted(_fit_warnings(shown, "Violin")) == [
        _fit_message("Violin", f"c={category}", FIT_FEW, panel="f=q") for category in "ab"
    ]
    assert ">q<" in svg
    if inner == "none":
        assert _census(shown) == []


# ---------------------------------------------------------------------------
# What does not change.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("z_values,reason", _Z_CASES)
def test_a_density_mark_on_the_same_data_keeps_its_own_warning(
    z_values: list[float | None], reason: str
):
    """``mark_density`` reads the same estimate through ``Kde``, whose warning is unchanged.
    Tied values read ``Kde``'s own cause (the bandwidth), the one place its wording and the
    violin's differ."""
    kde_reason = FIT_BANDWIDTH if reason == _TIED else reason
    chart = fm.Chart(_small_slot_frame(z_values)).mark_density().encode(x="w", color="c")
    _, shown = _render_capturing(chart.to_svg, action="default")
    assert shown == [_fit_message("Kde", "c=z", kde_reason)]


@pytest.mark.parametrize("door,inner", _DOORS)
def test_a_well_formed_violin_warns_nothing(door: str, inner: str):
    df = pl.DataFrame(
        {
            "c": ["a"] * 4 + ["b"] * 4 + ["z"] * 4,
            "w": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 6.0, 3.0, 4.0, 5.0, 8.0],
        }
    )
    svg, shown = _render_capturing(_violin_door(door, df, inner).to_svg, action="always")
    assert shown == []
    assert ">z<" in svg


# ---------------------------------------------------------------------------
# QQ: an all-null hue group or facet panel.
# ---------------------------------------------------------------------------


def _qq_hue_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"v": [1.0, 2.0, 3.0, 4.0, None, None, None, None], "g": ["a"] * 4 + ["c"] * 4},
        schema={"v": pl.Float64, "g": pl.Utf8},
    )


def _qq_panel_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"v": [1.0, 2.0, 3.0, 4.0, None, None, None, None], "f": ["p"] * 4 + ["q"] * 4},
        schema={"v": pl.Float64, "f": pl.Utf8},
    )


def test_a_qq_group_a_null_column_empties_is_named():
    """Hue group ``c`` has no usable value: the statistic names it once and the legend keeps it."""
    chart = fm.Chart(_qq_hue_frame()).mark_qq().encode(x="v", color="g")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_fit_message("QQ", "g=c", FIT_NONE)]
    assert ">c<" in svg


def test_a_qq_facet_panel_a_null_column_empties_is_named():
    """An all-null panel draws empty and is named, instead of failing the whole faceted chart."""
    chart = fm.Chart(_qq_panel_frame()).mark_qq().encode(x="v").facet(col="f")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert shown == [_panel_fit_message("QQ", FIT_NONE)]
    assert ">p<" in svg and ">q<" in svg


def _refusal_warnings(chart: fm.Chart) -> list[str]:
    """Every warning raised on the way to ``chart`` refusing with the empty-input sentence."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="stat_qq: empty input after filter"):
            chart.to_svg()
    return [str(w.message) for w in caught]


def test_a_qq_with_no_group_and_no_panel_and_no_usable_value_still_refuses():
    """The refusal is the whole outcome: no sentence claims the call degraded instead of failing."""
    df = pl.DataFrame({"v": [None, None]}, schema={"v": pl.Float64})
    assert _refusal_warnings(fm.Chart(df).mark_qq().encode(x="v")) == []


def test_a_grouped_qq_with_no_panel_and_no_usable_value_refuses_without_a_warning():
    df = pl.DataFrame(
        {"v": [None] * 4, "g": ["a", "b"] * 2}, schema={"v": pl.Float64, "g": pl.Utf8}
    )
    assert _refusal_warnings(fm.Chart(df).mark_qq().encode(x="v", color="g")) == []


def test_a_well_formed_qq_warns_nothing():
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 5.0, 8.0], "g": ["a"] * 4 + ["c"] * 4})
    chart: Callable[[], str] = fm.Chart(df).mark_qq().encode(x="v", color="g").to_svg
    assert _render_capturing(chart, action="always")[1] == []


def test_a_violin_group_holding_an_infinity_is_named_in_kdes_non_finite_wording():
    """An infinite value leaves a group with no finite extent to estimate over. The category keeps
    its slot and the statistic names it, with the cause ``Kde`` gives the same data."""
    df = pl.DataFrame(
        {"c": ["a"] * 4 + ["z"] * 4, "w": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, float("inf"), 3.0]}
    )
    reason = FIT_BANDWIDTH
    svg, shown = _render_capturing(_violin_door("mark", df, "none").to_svg, action="always")
    assert shown == [_fit_message("Violin", "c=z", reason)]
    assert ">z<" in svg
    _, kde = _render_capturing(
        fm.Chart(df).mark_density().encode(x="w", color="c").to_svg, action="always"
    )
    assert kde == [_fit_message("Kde", "c=z", reason)]


def test_a_qq_facet_panel_that_is_emptied_in_every_group_names_each_group():
    """With a hue the panel holds one group per level, so the statistic names each emptied group
    (with the panel clause) instead of the panel; the panel and its legend entries stay."""
    df = pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, None, None, None, None] * 2,
            "f": (["p"] * 4 + ["q"] * 4) * 2,
            "g": ["a"] * 8 + ["b"] * 8,
        },
        schema={"v": pl.Float64, "f": pl.Utf8, "g": pl.Utf8},
    )
    chart = fm.Chart(df).mark_qq().encode(x="v", color="g").facet(col="f")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert sorted(shown) == [
        _fit_message("QQ", f"g={level}, f=q", FIT_NONE, panel="f=q") for level in ("a", "b")
    ]
    assert ">q<" in svg


def test_a_violin_category_of_nan_values_is_named_like_a_null_one():
    """A NaN is no usable value, as a null is: the category keeps its slot and is named."""
    df = pl.DataFrame({"c": ["a"] * 4 + ["z"] * 3, "w": [1.0, 2.0, 3.0, 4.0] + [float("nan")] * 3})
    svg, shown = _render_capturing(_violin_door("mark", df, "none").to_svg, action="always")
    assert shown == [_fit_message("Violin", "c=z", FIT_FEW)]
    assert ">z<" in svg


def test_a_violin_whose_density_is_zero_everywhere_names_each_group():
    """A bandwidth so small every density underflows to zero leaves no body to draw; each category
    is named with the cause, and keeps its slot."""
    df = pl.DataFrame({"c": ["a"] * 4 + ["b"] * 4, "w": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0]})
    chart = fm.Chart(df).mark_violin(inner="none", bandwidth=1e-320).encode(x="c", y="w")
    svg, shown = _render_capturing(chart.to_svg, action="always")
    assert sorted(shown) == [_fit_message("Violin", f"c={c}", _ZERO_DENSITY) for c in "ab"]
    assert ">a<" in svg and ">b<" in svg


def test_a_violin_group_of_one_infinite_value_is_named_with_the_reason_kde_gives():
    """One infinite value is one value, not a degenerate bandwidth: the count is the cause,
    as ``Kde`` reads the same data."""
    df = pl.DataFrame({"c": ["a"] * 4 + ["z"], "w": [1.0, 2.0, 3.0, 4.0, float("inf")]})
    svg, shown = _render_capturing(_violin_door("mark", df, "none").to_svg, action="always")
    assert shown == [_fit_message("Violin", "c=z", FIT_FEW)]
    assert ">z<" in svg
    _, kde = _render_capturing(
        fm.Chart(df).mark_density().encode(x="w", color="c").to_svg, action="always"
    )
    assert kde == [_fit_message("Kde", "c=z", FIT_FEW)]


def test_a_qq_hue_group_of_nan_values_is_named_like_a_null_one():
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0] + [float("nan")] * 4, "g": ["a"] * 4 + ["c"] * 4})
    svg, shown = _render_capturing(
        fm.Chart(df).mark_qq().encode(x="v", color="g").to_svg, action="always"
    )
    assert shown == [_fit_message("QQ", "g=c", FIT_NONE)]
    assert ">c<" in svg


@pytest.mark.parametrize("hue", [None, "g"], ids=["ungrouped", "grouped"])
def test_a_faceted_qq_with_no_usable_value_in_any_panel_refuses_as_the_whole_frame_does(
    hue: str | None,
):
    """Every panel empty is the whole frame empty: the call refuses, with the unfaceted
    call's own sentence and with no warning, instead of drawing blank panels, failing on an
    internal lookup, or saying each panel degraded instead of failing the whole chart."""
    df = pl.DataFrame(
        {"v": [None] * 8, "f": ["p"] * 4 + ["q"] * 4, "g": ["a", "b"] * 4},
        schema={"v": pl.Float64, "f": pl.Utf8, "g": pl.Utf8},
    )
    encoding = {"x": "v"} if hue is None else {"x": "v", "color": hue}
    chart = fm.Chart(df).mark_qq().encode(**encoding).facet(col="f")
    assert _refusal_warnings(chart) == []
