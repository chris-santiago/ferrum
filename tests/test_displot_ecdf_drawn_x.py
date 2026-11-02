"""``displot(kind="ecdf")`` counts exactly the x the chart draws, through every door.

A keyword, an ``encode=`` entry, a post-hoc ``.encode()``, and a shorthand
string (``"v:Q"``) must all count and draw the same column -- the one the
x axis shows -- never a column the ``x=`` keyword named that a later door
re-points elsewhere. The window's grouping must read the same effective
encoding too: its own bindings, then a ``+``/``LayerChart`` composition's,
on every channel the chart itself leaves unbound.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from tests._svg_extents import value_axis_ladder, value_for_pixel
from tests._svg_rects import step_after_paths
from tests._wire_steps import all_steps


@pytest.fixture
def two_col_df():
    return pl.DataFrame({"v": list(range(1, 11)), "w": list(range(10, 0, -1))})


@pytest.fixture
def grouped_df():
    """Two grouping columns (`g` for a composition, `h` for the ecdf's own
    hue), five rows per level, so a pooled count (6..10) is distinct from
    each group's own (0..5, the riser included)."""
    return pl.DataFrame(
        {
            "v": list(range(1, 6)) * 2,
            "g": ["p1"] * 5 + ["p2"] * 5,
            "h": ["x1"] * 5 + ["x2"] * 5,
        }
    )


def _window_groupby(chart) -> list[str] | None:
    """The ecdf's `data_window` op's own `groupby`, or `None` when unset."""
    spec = json.loads(chart.to_spec().to_json())
    (window,) = [step for step in all_steps(spec) if step["type"] == "data_window"]
    return window.get("groupby")


# ---------------------------------------------------------------------------
# The window counts the x the chart draws, through every door.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build,plain",
    [
        pytest.param(
            lambda df: fm.displot(df, x="v", kind="ecdf", encode={"x": "w"}),
            lambda df: fm.displot(df, x="w", kind="ecdf"),
            id="encode",
        ),
        pytest.param(
            lambda df: fm.displot(df, x="v", kind="ecdf").encode(x="w"),
            lambda df: fm.displot(df, x="w", kind="ecdf"),
            id="posthoc",
        ),
        pytest.param(
            lambda df: fm.displot(df, x="v:Q", kind="ecdf"),
            lambda df: fm.displot(df, x="v", kind="ecdf"),
            id="shorthand",
        ),
        pytest.param(
            lambda df: fm.displot(df, x="v", kind="ecdf", encode={"x": "v:Q"}),
            lambda df: fm.displot(df, x="v", kind="ecdf"),
            id="encode_shorthand",
        ),
        pytest.param(
            lambda df: fm.displot(df, x="v:Q", kind="ecdf", log_scale=True),
            lambda df: fm.displot(df, x="v", kind="ecdf", log_scale=True),
            id="shorthand_log_scale",
        ),
    ],
)
def test_the_ecdf_counts_the_x_it_draws(two_col_df, build, plain):
    """Every door counts and draws the column the chart's x axis shows,
    byte-identical to building that column directly."""
    assert build(two_col_df).to_svg() == plain(two_col_df).to_svg()


def test_the_ecdf_counts_the_x_it_draws_encode_none_is_a_negative_control(two_col_df):
    """``encode={"x": None}`` binds nothing (``Chart.encode``'s documented
    contract), so it leaves the keyword's own x bound -- byte-identical to
    the call without it, never drawing ``None``."""
    with_none = fm.displot(two_col_df, x=fm.X("v"), kind="ecdf", encode={"x": None}).to_svg()
    without = fm.displot(two_col_df, x=fm.X("v"), kind="ecdf").to_svg()
    assert with_none == without


@pytest.mark.parametrize(
    "kind,overlay",
    [
        ("hist", "kde"),
        ("hist", "rug"),
        ("ecdf", "kde"),
        ("ecdf", "rug"),
        ("kde", "rug"),
    ],
)
def test_shorthand_x_reaches_every_overlay(two_col_df, kind, overlay):
    """``x="v:Q"`` with ``kde=True``/``rug=True`` reaches every kind's
    overlay as ``v``, byte-identical to ``x="v"``, instead of raising a
    raw column error."""
    shorthand = fm.displot(two_col_df, x="v:Q", kind=kind, **{overlay: True}).to_svg()
    plain = fm.displot(two_col_df, x="v", kind=kind, **{overlay: True}).to_svg()
    assert shorthand == plain


# ---------------------------------------------------------------------------
# The overlay route, an x=None call, null x values and a non-string x.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("overlay", ["kde", "rug"])
def test_overlay_x_reads_v_despite_a_later_encode_x(two_col_df, overlay):
    """The overlay block resolves inside displot, from the x= keyword,
    before encode= lands: encode={"x": "w"} re-points the primary layer's
    x, but the overlay keeps reading v."""
    chart = fm.displot(two_col_df, x="v", kind="hist", encode={"x": "w"}, **{overlay: True})
    spec = json.loads(chart.to_spec().to_json())
    if overlay == "kde":
        (kde_step,) = [step for step in all_steps(spec) if step["type"] == "kde"]
        assert kde_step["field"] == "v"
    else:
        rug_layer = next(layer for layer in spec["layers"] if layer.get("name") == "rug")
        assert rug_layer["encoding"]["x"]["field"] == "v"


@pytest.mark.parametrize("overlay", ["kde", "rug"])
@pytest.mark.parametrize("door", ["encode", "posthoc"])
def test_an_overlaid_ecdf_reads_the_x_keyword_through_every_later_door(two_col_df, overlay, door):
    """With a kde= or rug= overlay, displot builds the ECDF and its overlays
    before it returns, from the x= keyword alone, so a later x through an
    encode= entry or a post-hoc .encode() reaches none of them: the line
    keeps counting and drawing v, byte-identical to the call without the
    later x."""
    if door == "encode":
        chart = fm.displot(two_col_df, x="v", kind="ecdf", encode={"x": "w"}, **{overlay: True})
    else:
        chart = fm.displot(two_col_df, x="v", kind="ecdf", **{overlay: True}).encode(x="w")
    plain = fm.displot(two_col_df, x="v", kind="ecdf", **{overlay: True})
    assert chart.to_svg() == plain.to_svg()
    spec = json.loads(chart.to_spec().to_json())
    (window,) = [step for step in all_steps(spec) if step["type"] == "data_window"]
    assert window["sort"] == ["v"]
    assert window["ops"][0]["field"] == "v"


def test_encode_x_with_no_x_keyword_is_refused_as_x_none(two_col_df):
    """``displot(kind="ecdf", encode={"x": "v"})`` with no ``x=`` keyword is
    refused exactly as ``displot(kind="ecdf")`` is: ``encode=`` is read only
    after the call-time "needs x=" check, which sees the keyword alone."""
    with pytest.raises(ValueError) as with_encode:
        fm.displot(two_col_df, kind="ecdf", encode={"x": "v"})
    with pytest.raises(ValueError) as without_encode:
        fm.displot(two_col_df, kind="ecdf")
    assert str(with_encode.value) == str(without_encode.value)


def test_null_x_values_use_the_same_window_wire():
    """A null x value adds nothing to the data_window op: nulls are left
    to the transform's own handling, not filtered by displot."""
    df = pl.DataFrame({"v": [1.0, None, 2.0, None, 3.0]})
    chart = fm.displot(df, x="v", kind="ecdf")
    spec = json.loads(chart.to_spec().to_json())
    (window,) = [step for step in all_steps(spec) if step["type"] == "data_window"]
    assert window == {
        "type": "data_window",
        "ops": [{"op": "count", "field": "v", "as": "count"}],
        "sort": ["v"],
        "frame": [None, 0],
    }


def test_a_numeric_x_keeps_its_existing_type_error(two_col_df):
    """A wrong-type ``x=`` (a number) keeps the same ``encode()``-level
    ``TypeError`` every kind raises; the ecdf's own lazy resolution never
    sees it."""
    with pytest.raises(TypeError, match=r"encode\(x=\.\.\.\) expects str, X instance"):
        fm.displot(two_col_df, x=5, kind="ecdf")


# ---------------------------------------------------------------------------
# A fieldless x is refused, not passed raw to the window transform.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda df: fm.displot(df, x=fm.X(title="t"), kind="ecdf"), id="keyword"),
        pytest.param(
            lambda df: fm.displot(df, x="v", kind="ecdf", encode={"x": fm.X(title="t")}),
            id="encode",
        ),
        pytest.param(
            lambda df: fm.displot(df, x="v", kind="ecdf").encode(x=fm.X(title="t")), id="posthoc"
        ),
    ],
)
def test_a_fieldless_x_is_refused(two_col_df, build):
    """A fieldless x= -- one door's own keyword, or a later door replacing
    a real column with one -- is refused in displot's voice, naming the
    remedy, rather than reaching the window transform raw."""
    with pytest.raises(ValueError) as excinfo:
        build(two_col_df).to_svg()
    assert str(excinfo.value) == (
        "displot(kind='ecdf') needs x bound to a column -- the ECDF counts the raw "
        "values of the variable x names, and this x binding names none. Bind x to a "
        "column: x='<column>', or fm.X('<column>', ...) to keep the channel's options."
    )


def test_a_fieldless_x_control_with_a_field_renders(two_col_df):
    """Control: a channel object that DOES carry a field is unaffected."""
    assert fm.displot(two_col_df, x=fm.X("v", title="t"), kind="ecdf").to_svg()


# ---------------------------------------------------------------------------
# The count axis is titled "count" through every door, collision or not.
# ---------------------------------------------------------------------------


@pytest.fixture
def count_collision_df():
    """A source column literally named "count", the case the ecdf's own
    column-name collision avoidance exists for."""
    return pl.DataFrame({"v": list(range(1, 11)), "count": [9] * 10})


@pytest.mark.parametrize(
    "fieldless_y",
    [
        pytest.param(lambda: fm.Y(scale=fm.LinearScale()), id="scale"),
        pytest.param(lambda: fm.Y(format=".1f"), id="format"),
    ],
)
@pytest.mark.parametrize("door", ["encode", "posthoc"])
def test_the_count_title_does_not_depend_on_the_door(count_collision_df, door, fieldless_y):
    """On a frame with a source `"count"` column, a fieldless y= through
    `encode=` or a post-hoc `.encode()` still titles the axis "count" --
    byte-identical to the same call on a frame without the collision --
    instead of showing the internal `"count_1"` column name."""
    plain_df = count_collision_df.drop("count")
    if door == "encode":
        collision = fm.displot(count_collision_df, x="v", kind="ecdf", encode={"y": fieldless_y()})
        plain = fm.displot(plain_df, x="v", kind="ecdf", encode={"y": fieldless_y()})
    else:
        collision = fm.displot(count_collision_df, x="v", kind="ecdf").encode(y=fieldless_y())
        plain = fm.displot(plain_df, x="v", kind="ecdf").encode(y=fieldless_y())
    assert collision.to_svg() == plain.to_svg()


def test_a_post_hoc_fieldless_y_title_control(count_collision_df):
    """Control: the caller's own title= still wins over the "count" default."""
    chart = fm.displot(count_collision_df, x="v", kind="ecdf").encode(y=fm.Y(title="t"))
    assert json.loads(chart.to_spec().to_json())["encoding"]["y"]["title"] == "t"


# ---------------------------------------------------------------------------
# The window groups by the chart's effective encoding: its own bindings,
# then a `+`/`LayerChart` composition's, on every channel it leaves unbound.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build,expected",
    [
        pytest.param(lambda df: fm.displot(df, x="v", kind="ecdf"), None, id="plain_no_groupby"),
        pytest.param(
            lambda df: (
                fm.displot(df, x="v", kind="ecdf") + fm.Chart(df).mark_tick().encode(x="v")
            ).encode(color="g"),
            ["g"],
            id="left_operand",
        ),
        pytest.param(
            lambda df: (
                fm.Chart(df).mark_tick().encode(x="v") + fm.displot(df, x="v", kind="ecdf")
            ).encode(color="g"),
            ["g"],
            id="right_operand",
        ),
        pytest.param(
            lambda df: (
                (fm.displot(df, x="v", kind="ecdf") + fm.Chart(df).mark_tick().encode(x="v"))
                + fm.Chart(df).mark_tick().encode(x="v")
            ).encode(color="g"),
            ["g"],
            id="nested",
        ),
        pytest.param(
            lambda df: (
                fm.displot(df, x="v", kind="ecdf", hue="h") + fm.Chart(df).mark_tick().encode(x="v")
            ).encode(detail="g"),
            ["h", "g"],
            id="own_hue_plus_parent_detail",
        ),
        pytest.param(
            lambda df: fm.displot(df, x="v", kind="ecdf").encode(color="g"),
            ["g"],
            id="posthoc_color_regroup",
        ),
        pytest.param(
            lambda df: (
                fm.displot(df, x="v", kind="ecdf", hue="h") + fm.Chart(df).mark_tick().encode(x="v")
            ).encode(color="g"),
            ["h"],
            id="own_hue_shadows_parent_color",
        ),
        pytest.param(
            lambda df: fm.displot(df, x="v", kind="ecdf", hue="h", col="g"),
            ["h"],
            id="hue_and_facet_no_facet_field",
        ),
        pytest.param(
            lambda df: (
                (fm.displot(df, x="v", kind="ecdf") + fm.Chart(df).mark_tick().encode(x="v"))
                .encode(color="h")
                .facet(col="g")
            ),
            ["h"],
            id="composition_color_plus_facet_no_facet_field",
        ),
    ],
)
def test_the_ecdf_groups_by_its_effective_encoding(grouped_df, build, expected):
    """The window's groupby= reads the chart's own grouping bindings, then
    -- on a +/LayerChart operand, at any depth, and on a post-hoc regroup
    -- each composition binding on a grouping channel the chart leaves
    unbound; an own binding on that channel shadows the parent's. No facet
    field is folded in: a facet panel's own rows already partition the
    window at render time."""
    assert _window_groupby(build(grouped_df)) == expected


def test_a_composition_color_draws_each_groups_own_count(grouped_df):
    """The left-operand composition row, rendered: each color group's step
    heights run its own riser-plus-5 (0..5), never the pooled 0..10 a
    dropped parent grouping would draw."""
    chart = (
        fm.displot(grouped_df, x="v", kind="ecdf") + fm.Chart(grouped_df).mark_tick().encode(x="v")
    ).encode(color="g")
    svg = chart.to_svg()
    y_ladder = value_axis_ladder(svg, axis="y")
    lines = step_after_paths(svg)
    assert len(lines) == 2, lines
    for _stroke, corners in lines:
        heights = sorted(round(value_for_pixel(y_ladder, py)) for _px, py in corners)
        assert heights == [0, 1, 2, 3, 4, 5], heights
