"""A horizontal catplot's messages name the channels its caller wrote.

``catplot(df, y="c", x="w", orient="h")`` lowers to ``x=category, y=value``
plus ``CoordFlip``. The caller bound the value to ``x``, so every warning,
scale message and refusal about that value reads ``x``, on the static and the
interactive route, in a composition and in a facet. A flip the caller wrote
(``.coord(CoordFlip())``) keeps naming the channels the caller wrote, and a
layer the caller attaches to a horizontal catplot, or a chart added to it,
keeps the naming of what its author wrote. The call-applied flip rides the
wire as ``flip_by_call``, set by the catplot lowering on its own layers only
and absent from every spec the lowering did not build.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from ferrum.coord import CoordFlip
from ferrum.diagnostics.sources._compared import ComparedModelSource
from tests.test_bug_hunt_compare_dodge import FakeShapSource, _importance_cms
from tests.test_domain_rows_left_out import _domain_warning, _statistic_domain_warning
from tests.test_null_group_warning import (
    _CATPLOT_KINDS,
    _DRAWN_Z,
    _EMPTY_Z,
    _UNDRAWABLE,
    _faceted_slot_frame,
    _hue_group_frame,
    _log_message,
    _render_capturing,
    _slot_frame,
    _slot_message,
)

_KINDS = [*_CATPLOT_KINDS, "boxen"]

# Categories a and b; ``w`` has two rows outside [0, 10] and one a log axis cannot place.
_VALUE_FRAME = pl.DataFrame(
    {"c": ["a", "a", "b", "b"], "w": [-1.0, 2.0, 3.0, 50.0]},
    schema={"c": pl.Utf8, "w": pl.Float64},
)
_DOMAIN = fm.LinearScale(domain=[0, 10])

# Counts of a (2) and b (3), for the count kind's own value channel.
_COUNT_FRAME = pl.DataFrame({"c": ["a", "a", "b", "b", "b"]})


def _horizontal(df: pl.DataFrame, kind: str, hue: str | None = None, **options: Any) -> fm.Chart:
    """The caller's ``catplot(y="c", x="w", orient="h")``: the value on ``x``."""
    keys: dict[str, Any] = {} if hue is None else {"hue": hue}
    return fm.catplot(df, y="c", x="w", kind=kind, orient="h", **keys, **options)


def _vertical(df: pl.DataFrame, kind: str, hue: str | None = None) -> fm.Chart:
    keys: dict[str, Any] = {} if hue is None else {"hue": hue}
    return fm.catplot(df, x="c", y="w", kind=kind, **keys)


def _spec(chart: Any) -> dict:
    return json.loads(chart.to_spec().to_json())


def _flips(spec: dict) -> list[bool]:
    """Whether each drawn unit of a built spec records a call-applied flip: its layers when it
    has any, else the chart itself."""
    layers = spec.get("layers")
    if layers:
        return [bool(layer.get("flip_by_call", False)) for layer in layers]
    return [bool(spec.get("flip_by_call", False))]


def _every_message(chart: fm.Chart) -> list[str]:
    return _render_capturing(chart.to_svg)[1]


# ---------------------------------------------------------------------------
# The null census names the channel the caller wrote.
# ---------------------------------------------------------------------------


def _as_the_caller_wrote_it(messages: list[str]) -> list[str]:
    """The vertical call's messages as the horizontal call raises them: the value the vertical
    call binds to ``y`` the horizontal one binds to ``x``."""
    return [m.replace("non-finite y,", "non-finite x,") for m in messages]


def _assert_names_the_value_as_x(horizontal: fm.Chart, vertical: fm.Chart) -> str:
    """Render both under ``action="always"`` and require the horizontal call's full message list
    to be the vertical call's with the value's channel renamed; return the horizontal svg."""
    svg, raised = _render_capturing(horizontal.to_svg)
    _, expected = _render_capturing(vertical.to_svg)
    assert expected, "the control raises nothing, so the comparison proves nothing"
    assert raised == _as_the_caller_wrote_it(expected)
    return svg


@pytest.mark.parametrize("hue", [None, "g"], ids=["no_hue", "hue"])
@pytest.mark.parametrize("kind", _KINDS)
def test_a_horizontal_catplots_empty_slot_names_the_value_channel_the_caller_wrote(
    kind: str, hue: str | None
):
    """Every message the horizontal call raises is the vertical call's, whole list and every
    copy, with the census reading ``x`` for the value."""
    frame = _slot_frame(_EMPTY_Z)
    svg = _assert_names_the_value_as_x(_horizontal(frame, kind, hue), _vertical(frame, kind, hue))
    assert ">z<" in svg


@pytest.mark.parametrize("value", _UNDRAWABLE)
@pytest.mark.parametrize("kind", _KINDS)
def test_a_horizontal_catplot_names_a_slot_of_null_nan_or_infinite_values(kind: str, value: Any):
    """A value that places nowhere on a linear axis empties the slot, whichever way it fails to."""
    frame = _slot_frame([value] * 6)
    svg = _assert_names_the_value_as_x(_horizontal(frame, kind), _vertical(frame, kind))
    assert ">z<" in svg


@pytest.mark.parametrize("value", _UNDRAWABLE)
def test_a_horizontal_strips_empty_slot_reads_x_once(value: Any):
    chart = _horizontal(_slot_frame([value] * 6), "strip")
    assert _every_message(chart) == [_slot_message("catplot", "x")]


@pytest.mark.parametrize("kind", ["strip", "box", "point"])
def test_a_horizontal_catplots_wholly_null_hue_group_names_the_value_channel(kind: str):
    """Hue level ``r`` is null throughout, so the group, not a slot, draws nothing."""
    frame = _hue_group_frame()
    horizontal = fm.catplot(frame, y="c", x="w", kind=kind, hue="g", orient="h")
    vertical = fm.catplot(frame, x="c", y="w", kind=kind, hue="g")
    _assert_names_the_value_as_x(horizontal, vertical)
    if kind == "strip":
        assert _every_message(horizontal) == [_slot_message("catplot", "x", "r", "g")]


def test_a_vertical_catplot_and_one_with_the_callers_own_flip_name_y():
    """No flip, and a flip the caller wrote, both keep the channel the caller bound the value to."""
    frame = _slot_frame(_EMPTY_Z)
    expected = [_slot_message("catplot", "y")]
    assert _every_message(_vertical(frame, "strip")) == expected
    assert _every_message(_vertical(frame, "strip").coord(CoordFlip())) == expected


def test_a_boxplot_with_the_callers_own_flip_raises_what_it_raises_unflipped():
    frame = _slot_frame(_EMPTY_Z)
    assert _every_message(_vertical(frame, "box").coord(CoordFlip())) == _every_message(
        _vertical(frame, "box")
    )


def test_a_horizontal_catplot_reaches_the_interactive_route_with_the_same_texts():
    """The interactive route raises the same texts as the static one, the caller's channel in
    each, and raises them as often as it does for the same chart drawn vertically. The count is
    compared, not fixed: the route builds its scene twice, so every text arrives more than once on
    any chart, and Python's default filter cannot state one text per call here because which copy
    it keeps depends on what the process warned before."""
    frame = _slot_frame(_EMPTY_Z)
    horizontal = _horizontal(frame, "strip")
    vertical = _vertical(frame, "strip")
    _, raised = _render_capturing(lambda: horizontal.interactive().to_html())
    _, control = _render_capturing(lambda: vertical.interactive().to_html())
    assert control
    assert raised == [m.replace("non-finite y,", "non-finite x,") for m in control]
    assert raised == [_slot_message("catplot", "x")] * len(raised)


# ---------------------------------------------------------------------------
# The scale warnings and a refusal name it too.
# ---------------------------------------------------------------------------


def test_a_horizontal_catplots_explicit_domain_warning_names_x():
    chart = fm.catplot(_VALUE_FRAME, y="c", x=fm.X("w", scale=_DOMAIN), kind="strip", orient="h")
    assert _every_message(chart) == [_domain_warning("x", "w", 2)]


def test_a_vertical_catplots_explicit_domain_warning_names_y():
    for flip in (False, True):
        chart = fm.catplot(_VALUE_FRAME, x="c", y=fm.Y("w", scale=_DOMAIN), kind="strip")
        if flip:
            chart = chart.coord(CoordFlip())
        assert _every_message(chart) == [_domain_warning("y", "w", 2)]


def test_a_horizontal_catplots_log_scale_warning_names_x():
    chart = fm.catplot(
        _VALUE_FRAME, y="c", x=fm.X("w", scale=fm.LogScale()), kind="strip", orient="h"
    )
    assert _every_message(chart) == [_log_message("x", "w", 1)]


def test_a_vertical_catplots_log_scale_warning_names_y():
    chart = fm.catplot(_VALUE_FRAME, x="c", y=fm.Y("w", scale=fm.LogScale()), kind="strip")
    assert _every_message(chart) == [_log_message("y", "w", 1)]


def _all_negative_catplot(*, horizontal: bool) -> fm.Chart:
    frame = _VALUE_FRAME.with_columns(w=pl.col("w").abs().neg() - 1.0)
    if horizontal:
        return fm.catplot(frame, y="c", x=fm.X("w", scale=fm.LogScale()), kind="strip", orient="h")
    return fm.catplot(frame, x="c", y=fm.Y("w", scale=fm.LogScale()), kind="strip")


def _refusal(chart: fm.Chart) -> str:
    with pytest.raises(ValueError) as raised:
        chart.to_svg()
    return str(raised.value)


def _no_positive_value_refusal(channel: str, rows: int) -> str:
    return (
        f"{channel} scale: every value of 'w' ({rows} rows) is negative, and a LogScale with no "
        "domain maps only positive values; pass an explicit negative domain, e.g. "
        "fm.LogScale(domain=[-51, -2]), to draw a mirrored log scale"
    )


def test_a_horizontal_catplots_positional_refusal_names_the_channel_the_caller_wrote():
    """A refusal about the value's scale reads ``x`` on the horizontal call and ``y`` on the
    vertical one."""
    assert _refusal(_all_negative_catplot(horizontal=True)) == _no_positive_value_refusal("x", 4)
    assert _refusal(_all_negative_catplot(horizontal=False)) == _no_positive_value_refusal("y", 4)
    assert _refusal(_all_negative_catplot(horizontal=False).coord(CoordFlip())) == (
        _no_positive_value_refusal("y", 4)
    )


# ---------------------------------------------------------------------------
# Every message family that names a positional channel, across the routes that raise it.
# ---------------------------------------------------------------------------

# Categories a and b of three rows; ``w`` has one value a log axis cannot place, two outside
# [0, 10] and five that are not negative.
_FAMILY_FRAME = pl.DataFrame(
    {
        "c": ["a", "a", "b", "b", "a", "b"],
        "g": ["p", "q", "p", "q", "q", "p"],
        "w": [-1.0, 2.0, 3.0, 50.0, 4.0, 5.0],
    },
    schema={"c": pl.Utf8, "g": pl.Utf8, "w": pl.Float64},
)


def _value_scale_catplot(scale: Any, *, horizontal: bool, **options: Any) -> fm.Chart:
    """A strip of ``w`` by ``c`` with ``scale`` on the value, written the caller's way."""
    if horizontal:
        return fm.catplot(
            _FAMILY_FRAME,
            y="c",
            x=fm.X("w", scale=scale),
            kind="strip",
            orient="h",
            **options,
        )
    return fm.catplot(_FAMILY_FRAME, x="c", y=fm.Y("w", scale=scale), kind="strip", **options)


def _overlaid(scale: Any, *, horizontal: bool) -> fm.Chart:
    """The strip layered with a point chart of its own, through the overlay route."""
    if horizontal:
        other = fm.Chart(_FAMILY_FRAME).mark_point().encode(x="w", y="c")
    else:
        other = fm.Chart(_FAMILY_FRAME).mark_point().encode(x="c", y="w")
    return fm.LayerChart(_value_scale_catplot(scale, horizontal=horizontal), other)


def _independent(scale: Any, *, horizontal: bool) -> fm.Chart:
    chart = _value_scale_catplot(scale, horizontal=horizontal, col="g")
    return chart.share_scale(**{("x" if horizontal else "y"): "independent"})


def _log_off_positive_side(axis: str, rows: int) -> str:
    return (
        f"{axis} scale: {rows} row(s) of 'w' are zero or positive, which a LogScale cannot place, "
        "so they are not drawn; filter them, or use fm.SymlogScale() for data that reaches zero"
    )


_NEGATIVE_LOG = fm.LogScale(domain=[-5, -1])


@pytest.mark.parametrize(
    "build,expected",
    [
        pytest.param(
            lambda: _overlaid(fm.LogScale(), horizontal=True),
            [_log_message("x", "w", 1)],
            id="overlay_log_horizontal",
        ),
        pytest.param(
            lambda: _overlaid(fm.LogScale(), horizontal=False),
            [_log_message("y", "w", 1)],
            id="overlay_log_vertical",
        ),
        pytest.param(
            lambda: _overlaid(_DOMAIN, horizontal=True),
            [_domain_warning("x", "w", 4)],
            id="overlay_domain_horizontal",
        ),
        pytest.param(
            lambda: _overlaid(_DOMAIN, horizontal=False),
            [_domain_warning("y", "w", 4)],
            id="overlay_domain_vertical",
        ),
        pytest.param(
            lambda: _independent(fm.LogScale(), horizontal=True),
            [_log_message("x", "w", 1)],
            id="independent_axis_log_horizontal",
        ),
        pytest.param(
            lambda: _independent(fm.LogScale(), horizontal=False),
            [_log_message("y", "w", 1)],
            id="independent_axis_log_vertical",
        ),
        pytest.param(
            lambda: _value_scale_catplot(fm.SymlogScale(domain=[0, 10]), horizontal=True),
            [_domain_warning("x", "w", 2)],
            id="symlog_domain_horizontal",
        ),
        pytest.param(
            lambda: _value_scale_catplot(fm.SymlogScale(domain=[0, 10]), horizontal=False),
            [_domain_warning("y", "w", 2)],
            id="symlog_domain_vertical",
        ),
        pytest.param(
            lambda: _value_scale_catplot(_NEGATIVE_LOG, horizontal=True),
            [_log_off_positive_side("x", 5)],
            id="negative_log_horizontal",
        ),
        pytest.param(
            lambda: _value_scale_catplot(_NEGATIVE_LOG, horizontal=False),
            [_log_off_positive_side("y", 5)],
            id="negative_log_vertical",
        ),
    ],
)
def test_a_scale_warning_names_the_callers_channel_on_every_route(
    build: Callable[[], fm.Chart], expected: list[str]
):
    """The overlay route, an independent facet axis, an explicit domain on a symlog scale and a
    log scale reaching the other side of zero each raise the one text the caller reads, the
    horizontal call's value on ``x`` and the vertical call's on ``y``."""
    assert _every_message(build()) == expected


def test_an_attached_layers_scale_warning_keeps_the_channel_its_author_wrote():
    """The catplot's own strip reads ``x``; the text layer attached to it is written in the
    lowering's space, so its warning about the same value reads ``y``."""
    chart = _value_scale_catplot(
        fm.LinearScale(domain=[0, 5]),
        horizontal=True,
        layers=[fm.Layer(mark="text", encoding={"text": "c"})],
    )
    assert _every_message(chart) == [_domain_warning("x", "w", 3), _domain_warning("y", "w", 3)]


def _axis_orient_refusal(axis: str, orient: str, expected: str) -> str:
    return f"axis orient '{orient}' is invalid for the {axis} axis (expected {expected})"


@pytest.mark.parametrize(
    "build,message",
    [
        pytest.param(
            lambda: fm.catplot(
                _FAMILY_FRAME,
                y="c",
                x=fm.X("w", axis=fm.Axis(orient="left")),
                kind="strip",
                orient="h",
            ),
            _axis_orient_refusal("x", "left", "'top' or 'bottom'"),
            id="value_axis_horizontal",
        ),
        pytest.param(
            lambda: fm.catplot(
                _FAMILY_FRAME,
                y=fm.Y("c", axis=fm.Axis(orient="top")),
                x="w",
                kind="strip",
                orient="h",
            ),
            _axis_orient_refusal("y", "top", "'left' or 'right'"),
            id="category_axis_horizontal",
        ),
        pytest.param(
            lambda: fm.catplot(
                _FAMILY_FRAME,
                y="c",
                x=fm.X("w", axis=fm.Axis(orient="left")),
                kind="strip",
                col="g",
                orient="h",
            ).share_scale(x="independent"),
            _axis_orient_refusal("x", "left", "'top' or 'bottom'"),
            id="independent_axis_horizontal",
        ),
        pytest.param(
            lambda: fm.catplot(
                _FAMILY_FRAME,
                x="c",
                y=fm.Y("w", axis=fm.Axis(orient="bottom")),
                kind="strip",
            ),
            _axis_orient_refusal("y", "bottom", "'left' or 'right'"),
            id="value_axis_vertical",
        ),
    ],
)
def test_an_axis_orient_refusal_names_the_axis_the_caller_set_it_on(
    build: Callable[[], fm.Chart], message: str
):
    assert _refusal(build()) == message


def _position_and_stack_refusal(channel: str) -> str:
    return (
        "catplot: position=fm.Dodge(...) and stack= on "
        f"{channel} are both set, but a mark draws one position adjustment. Keep one: remove "
        f"stack= from {channel}, or remove position=."
    )


@pytest.mark.parametrize(
    "orient,keys,channel",
    [
        pytest.param("h", {"y": "c"}, "x", id="horizontal"),
        pytest.param("v", {"x": "c"}, "y", id="vertical"),
    ],
)
def test_a_stack_shorthand_beside_a_position_names_the_channel_the_bars_grow_along(
    orient: str, keys: dict, channel: str
):
    """The count channel is given through ``encode=``, in the lowering's space, and is named by
    the visual channel the bars grow along."""
    chart = fm.catplot(
        _FAMILY_FRAME,
        kind="count",
        hue="g",
        dodge=True,
        orient=orient,
        encode={"y": fm.Y(aggregate="count", stack="zero")},
        **keys,
    )
    assert _refusal(chart) == _position_and_stack_refusal(channel)


# ---------------------------------------------------------------------------
# A binding given through encode= is named by the keyword vocabulary.
# ---------------------------------------------------------------------------


def test_an_encode_binding_on_a_horizontal_catplot_names_the_visual_channel():
    """``encode=`` is written in the lowering's space (the value on ``y``) and its binding is
    named in the keyword vocabulary, the visual channel: the value axis a horizontal catplot
    draws is ``x``."""
    chart = fm.catplot(
        _VALUE_FRAME,
        y="c",
        x="w",
        kind="strip",
        orient="h",
        encode={"y": fm.Y("w", scale=_DOMAIN)},
    )
    assert _every_message(chart) == [_domain_warning("x", "w", 2)]


def test_a_horizontal_count_plots_value_channel_is_named_x():
    """The count kind's bars are an encoding statistic; its caller binds no value channel, and a
    scale given through ``encode=`` on the count channel is named by the visual one."""
    scale = fm.LinearScale(domain=[0, 1])
    chart = fm.catplot(
        _COUNT_FRAME,
        y="c",
        kind="count",
        orient="h",
        encode={"y": fm.Y(aggregate="count", scale=scale)},
    )
    assert _every_message(chart) == [_statistic_domain_warning("catplot", "x", "count_all", 2)]
    vertical = fm.catplot(
        _COUNT_FRAME, x="c", kind="count", encode={"y": fm.Y(aggregate="count", scale=scale)}
    )
    assert _every_message(vertical) == [_statistic_domain_warning("catplot", "y", "count_all", 2)]


# ---------------------------------------------------------------------------
# The count plot's own refusal names the channel its bars grow along.
# ---------------------------------------------------------------------------


def _count_label_refusal(channel: str) -> str:
    return (
        "catplot(kind='count'): layers= binds text='n', but the count plot draws each bar's "
        f"count as {channel}=count(), so there is no column 'n'; label each bar with "
        "text='count()' (or fm.Text(aggregate='count')), which counts the rows the layer reads "
        "as each bar does"
    )


@pytest.mark.parametrize(
    "build,channel",
    [
        pytest.param(
            lambda layers: fm.catplot(_COUNT_FRAME, x="c", kind="count", layers=layers),
            "y",
            id="vertical",
        ),
        pytest.param(
            lambda layers: fm.catplot(_COUNT_FRAME, y="c", kind="count", orient="h", layers=layers),
            "x",
            id="horizontal",
        ),
    ],
)
def test_a_count_plots_label_refusal_names_the_channel_its_bars_grow_along(
    build: Callable[..., fm.Chart], channel: str
):
    with pytest.raises(ValueError) as raised:
        build([fm.Layer(mark="text", encoding={"text": "n"})])
    assert str(raised.value) == _count_label_refusal(channel)


# ---------------------------------------------------------------------------
# Layers the caller attaches, and charts added, keep their own naming.
# ---------------------------------------------------------------------------


def test_a_layer_attached_to_a_horizontal_catplot_keeps_the_naming_of_what_its_author_wrote():
    """The catplot's own layer reads ``x``; the attached layer is written in the lowering's space
    and reads ``y``, and its wire layer carries no ``flip_by_call``."""
    frame = _slot_frame(_EMPTY_Z)
    chart = _horizontal(frame, "strip", layers=[fm.Layer(mark="point", encoding={"y": "w"})])
    assert _every_message(chart) == [_slot_message("catplot", "x"), _slot_message("catplot", "y")]
    assert _flips(_spec(chart)) == [True, False]


def test_a_chart_added_to_a_horizontal_catplot_keeps_its_own_naming():
    frame = _slot_frame(_EMPTY_Z)
    # The added chart is written for the panel the host draws: the value on ``x``, the category on ``y``.
    other = fm.Chart(frame).mark_point().encode(x="w", y="c")
    chart = _horizontal(frame, "strip") + other
    assert _every_message(chart) == [
        _slot_message("catplot", "x"),
        "layer 1: " + _slot_message("mark_point", "x"),
    ]
    assert _flips(_spec(chart)) == [True, False]


# ---------------------------------------------------------------------------
# A mark the caller places over a horizontal catplot is the caller's layer.
# ---------------------------------------------------------------------------


def _hand_built_flip(mark_method: str) -> fm.Chart:
    """The same chart written by hand: the lowering's encoding, the mark, the caller's own flip."""
    chart = getattr(fm.Chart(_slot_frame(_EMPTY_Z)), mark_method)()
    return chart.encode(x="c", y="w").coord(CoordFlip())


@pytest.mark.parametrize("kind", ["strip", "box", "violin", "swarm"])
def test_a_mark_the_caller_places_over_a_horizontal_catplot_names_what_the_caller_wrote(kind: str):
    """``.mark_point()`` replaces the catplot's mark with the caller's own, written against the
    lowering's encoding, so it names the channels as for any flip the caller wrote. A flat kind
    and a kind that holds a pending composite mark agree."""
    chart = _horizontal(_slot_frame(_EMPTY_Z), kind).mark_point()
    raised = _every_message(chart)
    assert raised == _every_message(_hand_built_flip("mark_point"))
    assert raised == [_slot_message("mark_point", "y")]
    assert "flip_by_call" not in chart.to_spec().to_json()


def test_a_composite_mark_the_caller_places_over_a_horizontal_strip_names_what_the_caller_wrote():
    chart = _horizontal(_slot_frame(_EMPTY_Z), "strip").mark_boxplot()
    raised = _every_message(chart)
    assert raised == _every_message(_hand_built_flip("mark_boxplot"))
    assert raised
    assert all("non-finite y," in message for message in raised)
    assert "flip_by_call" not in chart.to_spec().to_json()


# ---------------------------------------------------------------------------
# Compositions and facets.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "compose",
    [
        pytest.param(fm.hconcat, id="hconcat"),
        pytest.param(fm.vconcat, id="vconcat"),
        pytest.param(lambda *charts: fm.concat(*charts, columns=2), id="grid"),
    ],
)
def test_a_horizontal_catplot_in_a_composition_names_x_and_its_neighbour_is_unchanged(
    compose: Callable[..., Any],
):
    frame = _slot_frame(_EMPTY_Z)
    neighbour = fm.Chart(frame).mark_point().encode(x="c", y="w")
    composed = compose(_horizontal(frame, "strip"), neighbour)
    assert _every_message(composed) == [
        _slot_message("catplot", "x"),
        _slot_message("mark_point", "y"),
    ]


def test_a_faceted_horizontal_catplot_names_x_in_the_panel_that_is_empty():
    frame = _faceted_slot_frame(["q"])
    chart = fm.catplot(frame, y="c", x="w", kind="strip", col="f", orient="h")
    assert _every_message(chart) == [_slot_message("catplot", "x", panel="f=q")]
    vertical = fm.catplot(frame, x="c", y="w", kind="strip", col="f")
    assert _every_message(vertical) == [_slot_message("catplot", "y", panel="f=q")]


def test_a_horizontal_catplot_that_draws_every_slot_is_silent():
    chart = _horizontal(_slot_frame(_DRAWN_Z), "strip")
    assert _every_message(chart) == []


# ---------------------------------------------------------------------------
# Only the name a message prints reads the fact: sort and drawing are unchanged.
# ---------------------------------------------------------------------------


def _without_flip_fact(node: Any) -> Any:
    """``node`` with every ``flip_by_call`` key removed, however deep."""
    if isinstance(node, dict):
        return {k: _without_flip_fact(v) for k, v in node.items() if k != "flip_by_call"}
    if isinstance(node, list):
        return [_without_flip_fact(v) for v in node]
    return node


_SORT_TOKENS = [None, "x", "-x", "y", "-y"]


@pytest.mark.filterwarnings("ignore:catplot. inner marks")
@pytest.mark.parametrize("token", _SORT_TOKENS, ids=lambda t: f"sort={t}")
@pytest.mark.parametrize("hue", [None, "g"], ids=["no_hue", "hue"])
@pytest.mark.parametrize("kind", _KINDS)
def test_the_call_applied_fact_changes_neither_the_sort_nor_the_drawing(
    kind: str, hue: str | None, token: str | None
):
    """A horizontal catplot is, but for the one boolean, the vertical call under the caller's own
    flip: its wire spec without ``flip_by_call`` and its svg bytes are that chart's, for every
    ``sort=`` token."""
    frame = _slot_frame(_DRAWN_Z)
    encode = {} if token is None else {"encode": {"x": fm.X("c", sort=token)}}
    horizontal = _horizontal(frame, kind, hue, **encode)
    by_hand = fm.catplot(
        frame, x="c", y="w", kind=kind, **({} if hue is None else {"hue": hue}), **encode
    )
    by_hand = by_hand.coord(CoordFlip())
    assert _without_flip_fact(_spec(horizontal)) == _spec(by_hand)
    assert horizontal.to_svg() == by_hand.to_svg()


# ---------------------------------------------------------------------------
# The fact rides the wire only where the catplot lowering applied the flip.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [*_KINDS, "count"])
def test_a_horizontal_catplot_marks_every_drawn_unit_as_flipped_by_the_call(kind: str):
    frame = _COUNT_FRAME if kind == "count" else _slot_frame(_DRAWN_Z)
    if kind == "count":
        chart = fm.catplot(frame, y="c", kind="count", orient="h")
    else:
        chart = _horizontal(frame, kind)
    flips = _flips(_spec(chart))
    assert flips and all(flips)


def test_a_layer_attached_to_a_horizontal_catplot_has_no_flip_by_call_beside_the_catplots_own():
    for kind, own in [("strip", 1), ("point", 2)]:
        chart = _horizontal(
            _slot_frame(_DRAWN_Z), kind, layers=[fm.Layer(mark="rule", encoding={"y": "w"})]
        )
        assert _flips(_spec(chart)) == [True] * own + [False]


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: fm.Chart(_VALUE_FRAME).mark_point().encode(x="c", y="w"), id="plain"),
        pytest.param(
            lambda: fm.Chart(_VALUE_FRAME).mark_point().encode(x="c", y="w").coord(CoordFlip()),
            id="callers_flip",
        ),
        pytest.param(
            lambda: fm.Chart(_VALUE_FRAME).mark_bar(orient="horizontal").encode(x="c", y="w"),
            id="horizontal_bar",
        ),
        pytest.param(lambda: _vertical(_VALUE_FRAME, "strip"), id="vertical_strip"),
        pytest.param(lambda: _vertical(_VALUE_FRAME, "point"), id="vertical_point"),
        pytest.param(lambda: _vertical(_VALUE_FRAME, "box"), id="vertical_box"),
        pytest.param(
            lambda: _vertical(_VALUE_FRAME, "strip").coord(CoordFlip()),
            id="vertical_with_callers_flip",
        ),
    ],
)
def test_a_spec_the_catplot_lowering_did_not_flip_carries_no_flip_by_call(
    build: Callable[[], fm.Chart],
):
    """No key anywhere on the wire, so the serialized spec is what it was."""
    chart = build()
    assert "flip_by_call" not in chart.to_spec().to_json()


def test_a_horizontal_catplot_followed_by_a_cartesian_coord_names_as_written():
    """The lowering still stamps its layers, but with no flip there is nothing to un-flip: the
    census reads the channel the value is drawn on, as for any unflipped chart."""
    chart = _horizontal(_slot_frame(_EMPTY_Z), "strip").coord(fm.CoordCartesian())
    assert _every_message(chart) == [_slot_message("catplot", "y")]


def test_importance_chart_compare_carries_no_flip_by_call():
    """``importance_chart(compare=...)`` binds no ``x`` or ``y`` for its caller, so its own flip
    keeps the lowering's names and its spec carries no ``flip_by_call``."""
    cms = _importance_cms(
        a={"f1": (0.5, 0.1), "f2": (0.3, 0.1)},
        b={"f1": (0.4, 0.1), "f2": (0.2, 0.1)},
    )
    wire = fm.importance_chart(cms).to_spec().to_json()
    assert '"flip"' in wire
    assert "flip_by_call" not in wire


def test_shap_bar_chart_compare_carries_no_flip_by_call():
    cms = ComparedModelSource(
        {
            "a": FakeShapSource([{"f1": 0.5, "f2": -0.3}, {"f1": 0.4, "f2": 0.2}]),
            "b": FakeShapSource([{"f1": 0.1, "f2": -0.6}, {"f1": 0.2, "f2": 0.3}]),
        }
    )
    wire = fm.shap_bar_chart(cms).to_spec().to_json()
    assert "flip_by_call" not in wire
