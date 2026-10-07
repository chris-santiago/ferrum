"""A mark's style kwargs are held to that mark at every door, in the caller's voice.

The doors are ``fm.Layer(mark_kwargs=)``, a figure's ``mark=`` dict (flat,
named sub-layer, compound), ``lmplot``'s ``line_kws=``/``scatter_kws=``, a
composite's ``layer_kwargs=`` and ``Chart.override(mark_<leaf>=)``. Each door
is checked against three inputs: a key no mark has, a real key the target mark
does not draw, and (the vacuity control) a key it does draw, which must change
the render.
"""

from __future__ import annotations

import inspect
import json
import re
import warnings
from decimal import Decimal
from typing import Any, Callable, NamedTuple

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum._core import (
    ChartSpec,
    blend_accepted_values,
    interpolate_accepted_values,
    mark_bool_kwargs,
    mark_kwargs_fields,
    mark_string_kwargs,
    mark_style_fields,
    render_svg,
    stroke_cap_accepted_values,
    stroke_join_accepted_values,
    style_table_marks,
)
from ferrum._entry import EntryRefusal
from ferrum.exceptions import FerrumOverrideError

_RNG = np.random.default_rng(0)
_DF = pl.DataFrame({"x": _RNG.normal(size=30), "y": _RNG.normal(size=30)})
_CAT = pl.DataFrame({"g": ["a", "b"] * 15, "v": _RNG.normal(size=30)})
_YT = np.array([0, 1, 0, 1, 1, 0] * 5)
_YP = _RNG.random(30)


def _layered(style: dict) -> Any:
    return (
        fm.Chart(_DF)
        .mark_point()
        .encode(x="x", y="y")
        .layer(fm.Layer(mark="line", mark_kwargs=style))
    )


# door -> (build(style), voice, mark the style lands on, a key that mark draws)
_DOORS: dict[str, tuple[Callable[[dict], Any], str, str, dict]] = {
    "Layer": (_layered, "Layer", "line", {"stroke_width": 6}),
    "relplot mark=": (
        lambda st: fm.relplot(_DF, x="x", y="y", mark=st),
        "relplot",
        "point",
        {"opacity": 0.2},
    ),
    "roc_chart mark={line}": (
        lambda st: fm.roc_chart(y_true=_YT, y_pred=_YP, mark={"line": st}),
        "roc_chart",
        "line",
        {"stroke_width": 6},
    ),
    "lmplot line_kws": (
        lambda st: fm.lmplot(_DF, x="x", y="y", line_kws=st),
        "lmplot",
        "line",
        {"stroke_width": 6},
    ),
    "lmplot scatter_kws": (
        lambda st: fm.lmplot(_DF, x="x", y="y", scatter_kws=st),
        "lmplot",
        "point",
        {"opacity": 0.2},
    ),
    "mark_smooth layer_kwargs": (
        lambda st: (
            fm.Chart(_DF).mark_smooth(ci=0.95, layer_kwargs={"line": st}).encode(x="x", y="y")
        ),
        "mark_smooth",
        "line",
        {"stroke_width": 6},
    ),
}


def _not_drawn(mark: str) -> str:
    return "interpolate" if mark == "point" else "filled"


def _render(build: Callable[[dict], Any], style: dict) -> str:
    return build(style).to_svg()


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_key_no_mark_has_refuses_in_the_callers_voice(door: str) -> None:
    build, voice, _, _ = _DOORS[door]
    with pytest.raises(ValueError) as exc_info:
        _render(build, {"bogus_kw": 1})
    assert str(exc_info.value).startswith(
        f"{voice}: unknown keyword argument 'bogus_kw'. Honored: "
    )


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_key_the_mark_does_not_draw_refuses_in_the_callers_voice(door: str) -> None:
    build, voice, mark, _ = _DOORS[door]
    key = _not_drawn(mark)
    with pytest.raises(ValueError) as exc_info:
        _render(build, {key: "step" if key == "interpolate" else True})
    assert str(exc_info.value).startswith(
        f"{voice}: keyword argument {key!r} is not honored by mark_{mark}. Honored: "
    )


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_key_the_mark_draws_reaches_the_render(door: str) -> None:
    """Vacuity control: the same door honors a kwarg its mark draws."""
    build, _, _, drawn = _DOORS[door]
    assert _render(build, drawn) != _render(build, {})


def test_an_alias_resolves_on_a_layer_door() -> None:
    """``color`` reaches a line as its stroke, where it used to reach the
    renderer unread."""
    svg = _layered({"color": "#ff0000"}).to_svg()
    assert 'stroke="#ff0000"' in svg
    assert _layered({"color": "#ff0000"}).to_svg() != _layered({}).to_svg()


def test_override_leaf_is_held_to_the_charts_mark() -> None:
    chart = fm.Chart(_DF).mark_line().encode(x="x", y="y").override(mark_shape="square")
    with pytest.raises(FerrumOverrideError) as exc_info:
        chart.to_svg()
    assert str(exc_info.value).startswith(
        "Chart.override: keyword argument 'shape' is not honored by mark_line. Honored: "
    )


# ── figure mark= sub-layer rules ────────────────────────────────────────────


def test_style_for_an_undrawn_sub_layer_refuses() -> None:
    with pytest.raises(ValueError) as exc_info:
        fm.roc_chart(
            y_true=_YT, y_pred=_YP, annotate_auc=False, mark={"auc_label": {"opacity": 0.1}}
        )
    assert str(exc_info.value) == (
        "roc_chart(mark={'auc_label': ...}): this call draws no 'auc_label' sub-layer, "
        "so nothing would take the style; it draws ['line', 'reference']"
    )


def test_style_for_the_drawn_metric_label_applies() -> None:
    """Control: with the label drawn, the same override reaches it."""
    base = fm.roc_chart(y_true=_YT, y_pred=_YP).to_svg()
    styled = fm.roc_chart(y_true=_YT, y_pred=_YP, mark={"auc_label": {"font_size": 20}}).to_svg()
    assert 'font-size="20"' in styled and 'font-size="20"' not in base


def test_dropping_an_undrawn_sub_layer_is_accepted() -> None:
    plain = fm.roc_chart(y_true=_YT, y_pred=_YP, annotate_auc=False).to_svg()
    dropped = fm.roc_chart(
        y_true=_YT, y_pred=_YP, annotate_auc=False, mark={"auc_label": False}
    ).to_svg()
    assert dropped == plain


def test_a_non_dict_sub_layer_value_refuses_as_value_error() -> None:
    with pytest.raises(ValueError) as exc_info:
        fm.roc_chart(y_true=_YT, y_pred=_YP, mark={"line": 5})
    assert not isinstance(exc_info.value, TypeError)
    assert str(exc_info.value) == (
        "roc_chart(mark={'line': ...}) must be a dict of mark kwargs, or False to "
        "drop the sub-layer; got int"
    )


def test_an_unknown_sub_layer_names_the_figure() -> None:
    with pytest.raises(ValueError) as exc_info:
        fm.roc_chart(y_true=_YT, y_pred=_YP, mark={"bogus": {}})
    assert str(exc_info.value) == (
        "roc_chart(mark={'bogus': ...}): unknown sub-layer 'bogus'; this chart's sub-layers are "
        "['auc_label', 'line', 'reference']"
    )


def test_a_compound_figure_refuses_a_key_no_panel_takes() -> None:
    with pytest.raises(ValueError) as exc_info:
        fm.jointplot(_DF, x="x", y="y", mark={"bogus": 1})
    assert str(exc_info.value) == (
        "jointplot(mark=...): no panel of this chart draws a sub-layer named, or a mark "
        "that honors, ['bogus'], so nothing would take them"
    )


def test_a_regplot_refusal_names_regplot_not_the_figure_it_builds_through() -> None:
    with pytest.raises(ValueError) as exc_info:
        fm.regplot(_DF, x="x", y="y", line_kws={"filled": True})
    assert str(exc_info.value).startswith("regplot: keyword argument 'filled'")


def test_a_visualizer_refuses_a_malformed_mark_at_fit() -> None:
    viz = fm.ClassBalanceVisualizer(mark={"bogus_kw": 1})
    with pytest.raises(ValueError) as exc_info:
        viz.fit([0, 1, 1, 2, 2, 2])
    assert str(exc_info.value).startswith("ClassBalanceVisualizer: unknown keyword argument")


def test_layer_kwargs_for_an_undrawn_layer_refuses() -> None:
    chart = (
        fm.Chart(_CAT)
        .mark_boxplot(outliers=False, layer_kwargs={"outlier": {"fill": "#ff0000"}})
        .encode(x="g", y="v")
    )
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    assert str(exc_info.value) == (
        "mark_boxplot(layer_kwargs={'outlier': ...}): this call draws no 'outlier' sub-layer, "
        "so nothing would take the style; it draws "
        "['box', 'lower_cap', 'median', 'upper_cap', 'whisker']"
    )


# ── decimal numeric columns ─────────────────────────────────────────────────

_DEC = pl.DataFrame(
    {"a": [1.0, 2.0, 3.0], "d": [Decimal("1.5"), Decimal("2.5"), Decimal("0.5")], "b": [3, 1, 2]}
)


@pytest.mark.parametrize("entry", ["heatmap", "clustermap", "pairplot"])
def test_a_decimal_column_refuses_naming_the_entry(entry: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        getattr(fm, entry)(_DEC)
    assert str(exc_info.value) == (
        f"{entry}: column 'd' is a decimal column (Decimal128(38, 1)); a numeric column "
        "must be an integer or float column, so cast it to float first"
    )


def test_the_same_frame_cast_to_float_renders() -> None:
    """Control: the refusal is about the dtype, not the data."""
    fm.heatmap(_DEC.with_columns(pl.col("d").cast(pl.Float64))).to_svg()


# ── the style table: every door accepts exactly what the mark draws ─────────
#
# Rust owns which style fields each primitive mark draws
# (`ferrum._core.mark_style_fields`). The populations below are read from it,
# never hand-listed: every mark Rust knows crossed with every `mark_style`
# field it does not draw, at every door that reaches that mark. `size` is held
# to its own rule (its refusal names the mark's own width/font kwarg) and is
# pinned by `tests/marks/test_mark_honored_kwargs.py`, not here.

#: Python-owned facts the doors speak. Pinned against production below so the
#: messages here cannot drift from what the doors actually refuse.
_INTERNAL_DOORS = {
    "detail": "encode(detail=...)",
    "clip": "mark_function(clip=...)",
    "value_axis": "the orient= of the statistical mark that draws it",
}
_METHOD_ONLY = {"bar": frozenset({"orient"}), "tick": frozenset({"orient"})}

#: One well-typed value per wire field, so a raw-wire cell reaches the style
#: gate rather than a serde type error. A field Rust adds without a sample
#: here fails the population check loudly.
_SAMPLE: dict[str, Any] = {
    "size": 30.0,
    "stroke": "#ff0000",
    "fill": "#ff0000",
    "opacity": 0.5,
    "corner_radius": 3.0,
    "stroke_width": 4.0,
    "stroke_dash": [4.0, 2.0],
    "font_size": 14.0,
    "font_weight": "bold",
    "align": "left",
    "baseline": "top",
    "dx": 3.0,
    "dy": 3.0,
    "angle": 30.0,
    "detail": "x",
    "cmap": "viridis",
    "interpolate": "step",
    "stroke_cap": "round",
    "stroke_join": "round",
    "filled": False,
    "shape": "square",
    "limit": 5,
    "band_size": 0.5,
    "line": True,
    "borders": True,
    "leader_line": True,
    "width": 10.0,
    "height": 10.0,
    "clip": False,
    "blend": "additive",
    "zero": False,
    "arrow": True,
    "arrow_size": 14.0,
    "value_axis": "x",
}


#: Every primitive mark, in Rust's style-table order.
_MARKS = tuple(style_table_marks())
_FIELDS = tuple(mark_kwargs_fields())
#: The marks a `Chart.mark_<name>()` method draws directly or as its one
#: emitted mark (`mark_ribbon` is a composite whose layer is a ribbon).
_METHOD_MARKS = tuple(m for m in _MARKS if hasattr(fm.Chart, f"mark_{m}"))
#: The method marks whose chart stays flat (one mark, no layers), which is
#: what `.override(mark_*)` styles.
_FLAT_METHOD_MARKS = tuple(
    m for m in _METHOD_MARKS if getattr(fm.Chart(_DF), f"mark_{m}")()._pending_stat_mark is None
)


def _drawn(mark: str) -> set[str]:
    return set(mark_style_fields(mark))


def _layer_honored(mark: str) -> list[str]:
    return sorted(_drawn(mark) - set(_INTERNAL_DOORS))


def _method_honored(mark: str) -> list[str]:
    return sorted(set(_layer_honored(mark)) | _METHOD_ONLY.get(mark, frozenset()))


def _undrawn_fields(mark: str) -> list[str]:
    return [
        f for f in _FIELDS if f not in _drawn(mark) and f != "size" and f not in _INTERNAL_DOORS
    ]


def _not_honored(voice: str, mark: str, field: str, honored: list[str]) -> str:
    return f"{voice}: keyword argument {field!r} is not honored by mark_{mark}. Honored: {honored}"


def _internal_refusal(voice: str, field: str) -> str:
    return f"{voice}: keyword argument {field!r} is not a style kwarg; set it through {_INTERNAL_DOORS[field]}"


def _method_only_refusal(voice: str, mark: str, field: str) -> str:
    return (
        f"{voice}: keyword argument {field!r} is an argument of mark_{mark}(), not a layer "
        f"style; set it through mark_{mark}({field}=...)"
    )


def test_the_populations_are_read_from_rust_and_cover_every_field() -> None:
    """Vacuity control for every matrix below: the populations are neither
    empty nor everything, and the Python-owned sets are production's."""
    from ferrum.marks.base import _INTERNAL_STYLE_DOORS, METHOD_ONLY_KWARGS

    assert set(_SAMPLE) == set(_FIELDS)
    assert {"point", "bar", "polygon", "ribbon", "label"} <= set(_MARKS)
    assert "polygon" not in _METHOD_MARKS and "ribbon" not in _FLAT_METHOD_MARKS
    assert set(_FLAT_METHOD_MARKS) < set(_METHOD_MARKS) < set(_MARKS)
    # Every wire field is drawn by at least one mark, so the union of the
    # table is the whole wire vocabulary.
    assert set().union(*(_drawn(m) for m in _MARKS)) == set(_FIELDS)
    assert all(_undrawn_fields(m) for m in _MARKS)
    assert set(_INTERNAL_STYLE_DOORS) == set(_INTERNAL_DOORS)
    assert METHOD_ONLY_KWARGS == _METHOD_ONLY


_METHOD_CELLS = [(m, f) for m in _METHOD_MARKS for f in _undrawn_fields(m)]


@pytest.mark.parametrize(("mark", "field"), _METHOD_CELLS)
def test_a_mark_method_refuses_a_field_its_mark_never_draws(mark: str, field: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        getattr(fm.Chart(_DF), f"mark_{mark}")(**{field: _SAMPLE[field]}).encode(
            x="x", y="y"
        ).to_spec()
    assert str(exc_info.value) == _not_honored(f"mark_{mark}", mark, field, _method_honored(mark))


_LAYER_CELLS = [(m, f) for m in _MARKS for f in _undrawn_fields(m)]


@pytest.mark.parametrize(("mark", "field"), _LAYER_CELLS)
def test_a_layer_refuses_a_field_its_mark_never_draws(mark: str, field: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        fm.Layer(mark=mark, mark_kwargs={field: _SAMPLE[field]})
    assert str(exc_info.value) == _not_honored("Layer", mark, field, _layer_honored(mark))


_OVERRIDE_CELLS = [(m, f) for m in _FLAT_METHOD_MARKS for f in _undrawn_fields(m)]


@pytest.mark.parametrize(("mark", "field"), _OVERRIDE_CELLS)
def test_an_override_refuses_a_field_the_charts_mark_never_draws(mark: str, field: str) -> None:
    chart = (
        getattr(fm.Chart(_DF), f"mark_{mark}")()
        .encode(x="x", y="y")
        .override(**{f"mark_{field}": _SAMPLE[field]})
    )
    with pytest.raises(FerrumOverrideError) as exc_info:
        chart.to_spec()
    assert str(exc_info.value) == _not_honored("Chart.override", mark, field, _layer_honored(mark))


# A figure or composite door, the voice it refuses in, and the mark it styles.
_FIGURE_DOORS: dict[str, tuple[Callable[[dict], Any], str, str]] = {
    "relplot mark=": (lambda st: fm.relplot(_DF, x="x", y="y", mark=st), "relplot", "point"),
    "relplot line mark=": (
        lambda st: fm.relplot(_DF, x="x", y="y", kind="line", mark=st),
        "relplot",
        "line",
    ),
    "catplot count mark=": (
        lambda st: fm.catplot(_CAT, x="g", kind="count", mark=st),
        "catplot",
        "bar",
    ),
    "lmplot line_kws": (lambda st: fm.lmplot(_DF, x="x", y="y", line_kws=st), "lmplot", "line"),
    "lmplot scatter_kws": (
        lambda st: fm.lmplot(_DF, x="x", y="y", scatter_kws=st),
        "lmplot",
        "point",
    ),
    "mark_smooth layer_kwargs": (
        lambda st: (
            fm.Chart(_DF).mark_smooth(ci=0.95, layer_kwargs={"line": st}).encode(x="x", y="y")
        ),
        "mark_smooth",
        "line",
    ),
}
_FIGURE_CELLS = [(d, f) for d, (_, _, m) in _FIGURE_DOORS.items() for f in _undrawn_fields(m)]


@pytest.mark.parametrize(("door", "field"), _FIGURE_CELLS)
def test_a_figure_door_refuses_a_field_its_mark_never_draws(door: str, field: str) -> None:
    build, voice, mark = _FIGURE_DOORS[door]
    with pytest.raises(ValueError) as exc_info:
        build({field: _SAMPLE[field]}).to_spec()
    assert str(exc_info.value) == _not_honored(voice, mark, field, _layer_honored(mark))


def _wire_spec(mark: str, style: dict) -> str:
    spec: dict[str, Any] = {
        "data": {"kind": "named", "name": "default"},
        "mark": mark,
        "encoding": {"x": {"field": "x"}, "y": {"field": "y"}},
        "mark_style": style,
    }
    if mark == "arc":
        spec["coord"] = {"kind": "polar", "theta": "x"}
    return json.dumps(spec)


_WIRE_CELLS = [(m, f) for m in _MARKS for f in _FIELDS if f not in _drawn(m) and f != "size"]


@pytest.mark.parametrize(("mark", "field"), _WIRE_CELLS)
def test_a_raw_wire_spec_refuses_a_field_its_mark_never_draws(mark: str, field: str) -> None:
    """The render gate, for a producer no Python door validates."""
    spec = ChartSpec.from_json(_wire_spec(mark, {field: _SAMPLE[field]}))
    with pytest.raises(ValueError) as exc_info:
        render_svg(spec, _DF.to_arrow(), viewport=(300.0, 200.0))
    assert str(exc_info.value) == (
        f"mark_{mark}: {field} has no effect on a {mark} mark; a {mark} mark draws "
        + ", ".join(mark_style_fields(mark))
    )


def test_a_raw_wire_spec_with_an_unknown_style_key_refuses() -> None:
    with pytest.raises(ValueError, match=r"unknown field `bogus_kw`"):
        ChartSpec.from_json(_wire_spec("point", {"bogus_kw": 1}))


# ── fields only ferrum's own code writes ────────────────────────────────────

_INTERNAL_CELLS = (
    [("Layer", m, f) for m in _MARKS for f in _INTERNAL_DOORS]
    + [("method", m, f) for m in _METHOD_MARKS for f in _INTERNAL_DOORS]
    + [("override", m, f) for m in _FLAT_METHOD_MARKS for f in _INTERNAL_DOORS]
    + [("figure", d, f) for d in _FIGURE_DOORS for f in _INTERNAL_DOORS]
)


@pytest.mark.parametrize(("door", "target", "field"), _INTERNAL_CELLS)
def test_an_internal_field_refuses_at_every_door_naming_its_real_door(
    door: str, target: str, field: str
) -> None:
    style = {field: _SAMPLE[field]}
    error: type[Exception] = ValueError
    if door == "Layer":
        voice = "Layer"
        build: Callable[[], Any] = lambda: fm.Layer(mark=target, mark_kwargs=style)  # noqa: E731
    elif door == "method":
        voice = f"mark_{target}"
        build = lambda: getattr(fm.Chart(_DF), f"mark_{target}")(**style).to_spec()  # noqa: E731
    elif door == "override":
        voice, error = "Chart.override", FerrumOverrideError
        build = lambda: (  # noqa: E731
            getattr(fm.Chart(_DF), f"mark_{target}")()
            .encode(x="x", y="y")
            .override(**{f"mark_{field}": style[field]})
            .to_spec()
        )
    else:
        fig, voice, _ = _FIGURE_DOORS[target]
        build = lambda: fig(style).to_spec()  # noqa: E731
    with pytest.raises(error) as exc_info:
        build()
    assert str(exc_info.value) == _internal_refusal(voice, field)


# ── a layer with no mark draws a point, and is held to the point's row ──────


def test_a_markless_layer_refuses_an_internal_field_in_the_layer_voice() -> None:
    with pytest.raises(ValueError) as exc_info:
        fm.Layer(mark_kwargs={"detail": "g"})
    assert str(exc_info.value) == _internal_refusal("Layer", "detail")


def test_a_markless_layer_refuses_a_field_the_point_does_not_draw() -> None:
    honored = sorted(set(mark_style_fields("point")) - set(_INTERNAL_DOORS))
    with pytest.raises(ValueError) as exc_info:
        fm.Layer(mark_kwargs={"font_size": 9})
    assert str(exc_info.value) == (
        f"Layer: keyword argument 'font_size' is not honored by mark_point. Honored: {honored}"
    )
    with pytest.raises(ValueError) as exc_info:
        fm.Layer(mark_kwargs={"bogus_kw": 1})
    assert str(exc_info.value) == f"Layer: unknown keyword argument 'bogus_kw'. Honored: {honored}"


def test_a_markless_layer_resolves_aliases_as_a_point() -> None:
    from ferrum.marks.base import layer_honors

    assert fm.Layer(mark_kwargs={"color": "red", "alpha": 0.5}).mark_kwargs == {
        "fill": "red",
        "opacity": 0.5,
    }
    assert layer_honors(None, "color")
    assert not layer_honors(None, "font_size")
    # Control: the alias is resolved per mark, so it is a stroke on a line.
    assert fm.Layer(mark="line", mark_kwargs={"color": "red"}).mark_kwargs == {"stroke": "red"}


def test_a_layer_mark_that_is_not_a_name_refuses() -> None:
    with pytest.raises(ValueError) as exc_info:
        fm.Layer(mark=3)
    assert str(exc_info.value) == (
        "Layer(mark=...): a layer's mark is a mark name such as 'point' or 'line'; got 3"
    )


# ── the method-only kwargs ──────────────────────────────────────────────────

_METHOD_ONLY_CELLS = [
    (door, m, f)
    for m, fields in _METHOD_ONLY.items()
    for f in sorted(fields)
    for door in ("Layer", "override")
]


@pytest.mark.parametrize(("door", "mark", "field"), _METHOD_ONLY_CELLS)
def test_a_method_only_kwarg_refuses_off_its_method(door: str, mark: str, field: str) -> None:
    if door == "Layer":
        with pytest.raises(ValueError) as exc_info:
            fm.Layer(mark=mark, mark_kwargs={field: "horizontal"})
        voice = "Layer"
    else:
        chart = (
            getattr(fm.Chart(_CAT), f"mark_{mark}")()
            .encode(x="g", y="v")
            .override(**{f"mark_{field}": "horizontal"})
        )
        with pytest.raises(FerrumOverrideError) as exc_info:
            chart.to_spec()
        voice = "Chart.override"
    assert str(exc_info.value) == _method_only_refusal(voice, mark, field)


def test_a_method_only_kwarg_refuses_through_a_figure_mark_dict() -> None:
    with pytest.raises(ValueError) as exc_info:
        fm.catplot(_CAT, x="g", kind="count", mark={"orient": "horizontal"})
    assert str(exc_info.value) == _method_only_refusal("catplot", "bar", "orient")


@pytest.mark.parametrize("mark", sorted(_METHOD_ONLY))
def test_the_method_honors_its_own_kwarg(mark: str) -> None:
    """Control: on its own method `orient` flips the chart."""
    flat = getattr(fm.Chart(_CAT), f"mark_{mark}")().encode(x="g", y="v").to_svg()
    flipped = (
        getattr(fm.Chart(_CAT), f"mark_{mark}")(orient="horizontal").encode(x="g", y="v").to_svg()
    )
    assert flipped != flat


# ── a key no layer takes: one judgment, every key, every layer's mark ───────

#: Every key a layer door can refuse on a primitive mark: each field it never
#: draws, a constant `size` where it draws none, and its method-only kwargs.
_UNTAKEN_CELLS = [
    (m, f)
    for m in _MARKS
    for f in [*_undrawn_fields(m), "size", *sorted(_METHOD_ONLY.get(m, ()))]
    if f not in _layer_honored(m)
]


def test_the_untaken_population_covers_every_kind_of_refusal() -> None:
    """Vacuity control: the cells reach a method-only key, a size the mark
    never draws, and a field it never draws, and skip a key it takes."""
    assert ("bar", "orient") in _UNTAKEN_CELLS
    assert ("rule", "size") in _UNTAKEN_CELLS
    assert ("point", "interpolate") in _UNTAKEN_CELLS
    assert ("point", "size") not in _UNTAKEN_CELLS


@pytest.mark.parametrize(("mark", "field"), _UNTAKEN_CELLS)
def test_an_untaken_key_on_one_mark_refuses_as_the_layer_door_does(mark: str, field: str) -> None:
    """The spread door and the layer door judge a key one way, so one key on
    a chart of one mark refuses with the layer door's own text."""
    from ferrum.marks.base import refuse_untaken_layer_style, resolve_layer_style

    value = 3.0 if field == "size" else _SAMPLE.get(field, "horizontal")
    with pytest.raises(ValueError) as layer_door:
        resolve_layer_style("relplot", mark, {field: value})
    with pytest.raises(ValueError) as spread_door:
        refuse_untaken_layer_style("relplot", [mark], {field: value})
    assert str(spread_door.value) == str(layer_door.value)


def _point_and_line() -> Any:
    return (fm.Chart(_DF).mark_point() + fm.Chart(_DF).mark_line()).encode(x="x", y="y")


def _spread_refusal(chart: Any, style: dict) -> EntryRefusal:
    from ferrum._overrides import _apply_overrides

    with pytest.raises(EntryRefusal) as exc_info:
        _apply_overrides(chart, entry="relplot", mark=style)
    return exc_info.value


def test_a_key_no_layer_takes_names_every_layers_mark() -> None:
    """On a layered chart each layer's own refusal is named on its own line,
    so the refusal does not imply the other layers' fields are wrong too."""
    refusal = _spread_refusal(_point_and_line(), {"font_size": 3})
    assert str(refusal) == (
        "relplot: no layer of this chart takes ['font_size']:"
        "\n  mark_point: keyword argument 'font_size' is not honored by mark_point. "
        f"Honored: {_layer_honored('point')}"
        "\n  mark_line: keyword argument 'font_size' is not honored by mark_line. "
        f"Honored: {_layer_honored('line')}"
    )


def test_every_key_no_layer_takes_is_named() -> None:
    """Every untaken key is refused, not only the first, each as a sentence of
    its own, and a key the layer takes is not named."""
    refusal = _spread_refusal(
        fm.Chart(_DF).mark_point(), {"font_size": 3, "opacity": 0.5, "interpolate": "step"}
    )
    assert str(refusal) == (
        "relplot: no layer of this chart takes ['font_size', 'interpolate']:"
        "\n  mark_point: keyword argument 'font_size' is not honored by mark_point. "
        "Keyword argument 'interpolate' is not honored by mark_point. "
        f"Honored: {_layer_honored('point')}"
    )


def test_an_unknown_key_names_each_layers_mark_beside_its_honored_list() -> None:
    """An unknown key on a chart of several marks is refused on each layer's
    line, so each Honored list sits beside the mark that takes it."""
    refusal = _spread_refusal(_point_and_line(), {"bogus": 1, "font_size": 3})
    assert str(refusal) == (
        "relplot: no layer of this chart takes ['bogus', 'font_size']:"
        "\n  mark_point: unknown keyword argument 'bogus'. "
        "Keyword argument 'font_size' is not honored by mark_point. "
        f"Honored: {_layer_honored('point')}"
        "\n  mark_line: unknown keyword argument 'bogus'. "
        "Keyword argument 'font_size' is not honored by mark_line. "
        f"Honored: {_layer_honored('line')}"
    )


def test_a_method_only_key_across_layers_lists_honored_only_where_it_is_not_honored() -> None:
    """`orient` is `mark_bar()`'s own argument, so the bar line names that
    method and lists no fields; on the point it is a field the mark does not
    take, so that line lists the point's fields."""
    chart = (fm.Chart(_DF).mark_bar() + fm.Chart(_DF).mark_point()).encode(x="x", y="y")
    refusal = _spread_refusal(chart, {"orient": "horizontal"})
    assert str(refusal) == (
        "relplot: no layer of this chart takes ['orient']:"
        "\n  mark_bar: keyword argument 'orient' is an argument of mark_bar(), not a layer "
        "style; set it through mark_bar(orient=...)"
        "\n  mark_point: keyword argument 'orient' is not honored by mark_point. "
        f"Honored: {_layer_honored('point')}"
    )


def test_a_refusal_spanning_layers_names_its_call_once_in_any_voice() -> None:
    """The call is named once, at the head, in whichever voice speaks it, and
    each layer lists what it takes once."""
    chart = (
        fm.Chart(_DF).mark_point() + fm.Chart(_DF).mark_line() + fm.Chart(_DF).mark_area()
    ).encode(x="x", y="y")
    refusal = _spread_refusal(chart, {"bogus": 1, "font_size": 3})
    for voice in ("relplot", "catplot"):
        text = str(refusal.spoken_as(voice))
        assert text.startswith(f"{voice}: no layer of this chart takes ['bogus', 'font_size']:")
        assert text.count(voice) == 1, text
        assert text.count("Honored:") == text.count("\n") == 3, text


def test_a_size_no_layer_draws_names_each_marks_size_rule_once_per_layer() -> None:
    """A constant size on layers that draw none gets each mark's size-rule
    reason, its remedy kept, with the call named only at the head."""
    chart = (fm.Chart(_DF).mark_area() + fm.Chart(_DF).mark_rule()).encode(x="x", y="y")
    refusal = _spread_refusal(chart, {"size": 3.0})
    assert str(refusal) == (
        "relplot: no layer of this chart takes ['size']:"
        "\n  mark_area: keyword argument 'size' has no effect on mark_area; received 3"
        "\n  mark_rule: keyword argument 'size' has no effect on mark_rule; received 3. "
        "Set stroke_width= instead"
    )


def test_a_size_reason_beside_another_ends_as_a_sentence_of_its_own() -> None:
    """The size rule's remedy is a sentence of its own, so the next reason
    starts a new sentence instead of following it after a semicolon."""
    chart = (fm.Chart(_DF).mark_rule() + fm.Chart(_DF).mark_area()).encode(x="x", y="y")
    refusal = _spread_refusal(chart, {"size": 3.0, "bogus": 1})
    assert str(refusal) == (
        "relplot: no layer of this chart takes ['size', 'bogus']:"
        "\n  mark_rule: keyword argument 'size' has no effect on mark_rule; received 3. "
        "Set stroke_width= instead. Unknown keyword argument 'bogus'. "
        f"Honored: {_layer_honored('rule')}"
        "\n  mark_area: keyword argument 'size' has no effect on mark_area; received 3. "
        "Unknown keyword argument 'bogus'. "
        f"Honored: {_layer_honored('area')}"
    )


def test_a_non_numeric_size_reason_names_the_argument_not_the_call() -> None:
    chart = (fm.Chart(_DF).mark_area() + fm.Chart(_DF).mark_rule()).encode(x="x", y="y")
    refusal = _spread_refusal(chart, {"size": "big"})
    reason = (
        "keyword argument 'size' must be a real number (int, float, numpy scalar, or "
        "Decimal), not str; received 'big'"
    )
    assert str(refusal) == (
        "relplot: no layer of this chart takes ['size']:"
        f"\n  mark_area: {reason}\n  mark_rule: {reason}"
    )


def test_the_size_branchs_reason_and_refusal_report_the_same_outcome() -> None:
    """`_untaken_key`'s size branch decides whether a constant `size=` on
    `mark` refuses by one call, `_size_rule_reason`, and only builds the
    full, re-voiceable refusal once that single decision says to refuse.
    The clause this returns and the refusal's own rendered text can
    therefore never report a different outcome for the same `mark` and
    `size` -- covering both ways a constant size refuses: a mark that
    draws none at all (`area`) and a value outside the positive range
    (`point`, `-1.0`). The third case pins the other side of that single
    gate: a value the size rule accepts is never refused as a size-rule
    violation, even when `honored` (deliberately empty here) omits
    `size` for an unrelated reason -- it falls through to the generic
    not-honored refusal instead.
    """
    from ferrum.marks.base import _untaken_key

    for mark, size in [("area", 3.0), ("point", -1.0)]:
        untaken = _untaken_key("relplot", mark, "size", size, frozenset())
        assert untaken is not None
        clause = untaken.reason.removeprefix("keyword argument 'size' ")
        assert clause in str(untaken.refusal), (mark, size, untaken.reason, str(untaken.refusal))

    accepted = _untaken_key("relplot", "point", "size", 5.0, frozenset())
    assert accepted is not None
    assert "positive number" not in accepted.reason
    assert "not honored by mark_point" in accepted.reason


#: Every primitive mark that draws no constant size.
_SIZELESS_MARKS = [m for m in _MARKS if "size" not in _layer_honored(m)]


@pytest.mark.parametrize("mark", _SIZELESS_MARKS)
def test_an_unset_size_is_unset_at_the_spread_door_as_at_the_layer_door(mark: str) -> None:
    """`size=None` on a mark that draws no size is unset at every door: the
    layer door keeps it unset and the spread door refuses nothing."""
    from ferrum.marks.base import refuse_untaken_layer_style, resolve_layer_style

    assert resolve_layer_style("Layer", mark, {"size": None}) == {"size": None}
    assert refuse_untaken_layer_style("relplot", [mark], {"size": None}) is None
    assert refuse_untaken_layer_style("relplot", [mark, "point"], {"size": None}) is None


def test_an_unset_size_on_a_chart_whose_layers_draw_none_styles_nothing() -> None:
    """Through the figure's `mark=` door: no refusal and no layer restyled.
    Control: a set size on the same chart refuses."""
    from ferrum._overrides import _apply_overrides

    chart = (fm.Chart(_DF).mark_area() + fm.Chart(_DF).mark_rule()).encode(x="x", y="y")
    styled = _apply_overrides(chart, entry="relplot", mark={"size": None})
    assert not any(ly.mark_kwargs for ly in styled._layers)
    assert styled.to_svg() == chart.to_svg()
    _spread_refusal(chart, {"size": 3.0})


def test_an_unset_size_beside_another_untaken_key_refuses_only_that_key() -> None:
    """Only the key that is set is refused, with the layer door's own text."""
    from ferrum.marks.base import refuse_untaken_layer_style, resolve_layer_style

    with pytest.raises(ValueError) as layer_door:
        resolve_layer_style("relplot", "area", {"bogus": 1, "size": None})
    with pytest.raises(ValueError) as spread_door:
        refuse_untaken_layer_style("relplot", ["area"], {"bogus": 1, "size": None})
    assert str(spread_door.value) == str(layer_door.value)
    assert "'size'" not in str(spread_door.value)


# ── None is unset at every door ─────────────────────────────────────────────

#: Every kwarg a mark method takes, each mark it draws.
_METHOD_NONE_CELLS = [(m, f) for m in _METHOD_MARKS for f in _method_honored(m)]
#: Every kwarg a layer takes, and a size it never draws.
_LAYER_NONE_CELLS = [(m, f) for m in _MARKS for f in sorted({*_layer_honored(m), "size"})]


def test_the_none_populations_reach_a_method_only_kwarg_and_an_undrawn_size() -> None:
    assert ("bar", "orient") in _METHOD_NONE_CELLS
    assert ("rule", "size") in _LAYER_NONE_CELLS
    assert ("point", "bogus") not in _METHOD_NONE_CELLS


@pytest.mark.parametrize(("mark", "field"), _METHOD_NONE_CELLS)
def test_none_is_unset_at_a_mark_method(mark: str, field: str) -> None:
    from ferrum.marks.base import resolve_mark_kwargs

    getattr(fm.Chart(_DF), f"mark_{mark}")(**{field: None})
    assert resolve_mark_kwargs(f"mark_{mark}", mark, {field: None}) == {field: None}


@pytest.mark.parametrize(("mark", "field"), _LAYER_NONE_CELLS)
def test_none_is_unset_on_a_layer(mark: str, field: str) -> None:
    assert fm.Layer(mark=mark, mark_kwargs={field: None}).mark_kwargs == {field: None}


@pytest.mark.parametrize("mark", ["bar", "tick"])
def test_orient_none_draws_as_no_orient(mark: str) -> None:
    """`orient=None` is unset, so the mark draws exactly as with no orient;
    control: `orient="horizontal"` flips it."""
    plain = getattr(fm.Chart(_CAT), f"mark_{mark}")().encode(x="g", y="v").to_svg()
    unset = getattr(fm.Chart(_CAT), f"mark_{mark}")(orient=None).encode(x="g", y="v").to_svg()
    flipped = (
        getattr(fm.Chart(_CAT), f"mark_{mark}")(orient="horizontal").encode(x="g", y="v").to_svg()
    )
    assert unset == plain != flipped


def test_a_key_one_layer_takes_reaches_it_on_a_layered_chart() -> None:
    """Control: a key the line takes is not refused because the point does not."""
    from ferrum._overrides import _apply_overrides

    styled = _apply_overrides(_point_and_line(), entry="relplot", mark={"interpolate": "step"})
    assert [ly.mark_kwargs for ly in styled._layers] == [{}, {"interpolate": "step"}]


# ── a flat bar's zero is consumed as mark_bar(zero=) consumes it ────────────

_HIGH = pl.DataFrame({"g": ["a", "b", "c"], "v": [50.0, 70.0, 80.0]})


def _no_warnings(build: Callable[[], str]) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return build()


def test_a_flat_bar_override_zero_draws_like_mark_bar_zero() -> None:
    method = fm.Chart(_HIGH).mark_bar(zero=False).encode(x="g", y="v").to_svg()
    anchored = fm.Chart(_HIGH).mark_bar().encode(x="g", y="v").to_svg()
    override = _no_warnings(
        lambda: fm.Chart(_HIGH).mark_bar().encode(x="g", y="v").override(mark_zero=False).to_svg()
    )
    assert override == method
    assert method != anchored


def test_a_flat_bar_figure_zero_is_consumed_not_put_on_the_wire() -> None:
    chart = fm.catplot(_CAT, x="g", kind="count", mark={"zero": False})
    spec = json.loads(chart.to_spec().to_json())
    assert "zero" not in spec.get("mark_style", {})
    assert chart._mark_kwargs.get("zero") is False
    _no_warnings(chart.to_svg)


def test_a_layered_bar_zero_rides_the_layer() -> None:
    chart = fm.catplot(_CAT, x="g", y="v", kind="bar", mark={"bar": {"zero": False}})
    layers = json.loads(chart.to_spec().to_json())["layers"]
    assert [ly.get("mark_style", {}).get("zero") for ly in layers] == [False, None]
    _no_warnings(chart.to_svg)


# ── a composite spreads style only onto the marks that draw it ──────────────

_COMPOSITES: dict[str, Callable[[dict], Any]] = {
    "mark_smooth": lambda st: fm.Chart(_DF).mark_smooth(ci=0.95, **st).encode(x="x", y="y"),
    "mark_boxplot": lambda st: fm.Chart(_CAT).mark_boxplot(**st).encode(x="g", y="v"),
    "mark_errorbar": lambda st: fm.Chart(_CAT).mark_errorbar(**st).encode(x="g", y="v"),
    "mark_histogram": lambda st: fm.Chart(_DF).mark_histogram(**st).encode(x="x"),
}


def _emitted_marks(build: Callable[[dict], Any]) -> list[str]:
    resolved = build({})._resolve_pending()
    return sorted({ly.mark for ly in resolved._layers} if resolved._layers else {resolved._mark})


def _no_layer_draws(entry: str, fields: list[str], marks: list[str]) -> str:
    return (
        f"{entry}: keyword argument(s) {fields!r} are drawn by no mark this call draws "
        f"({marks}), so nothing would take them"
    )


#: A field that names one of the method's own parameters (`mark_boxplot`'s
#: `width=`) is that parameter, not style, so it is no cell.
_COMPOSITE_CELLS = [
    (entry, f)
    for entry, build in _COMPOSITES.items()
    for f in _FIELDS
    if f not in _INTERNAL_DOORS
    and f != "size"
    and f not in inspect.signature(getattr(fm.Chart, entry)).parameters
    and not any(f in _drawn(m) for m in _emitted_marks(build))
]


@pytest.mark.parametrize(("entry", "field"), _COMPOSITE_CELLS)
def test_a_composite_refuses_a_field_no_emitted_mark_draws(entry: str, field: str) -> None:
    build = _COMPOSITES[entry]
    with pytest.raises(ValueError) as exc_info:
        build({field: _SAMPLE[field]}).to_spec()
    assert str(exc_info.value) == _no_layer_draws(entry, [field], _emitted_marks(build))


def test_a_desugar_called_directly_refuses_a_field_no_layer_draws() -> None:
    from ferrum.marks.composite import desugar_errorbar

    with pytest.raises(ValueError) as exc_info:
        desugar_errorbar("g", "v", shape="square")
    assert str(exc_info.value) == _no_layer_draws("mark_errorbar", ["shape"], ["rule", "tick"])


@pytest.mark.parametrize(("entry", "field"), _COMPOSITE_CELLS)
def test_a_composite_reads_none_as_unset_for_a_field_no_emitted_mark_draws(
    entry: str, field: str
) -> None:
    """The same cells set to `None`: unset, so nothing is refused and the
    chart draws as it does without the field."""
    build = _COMPOSITES[entry]
    assert build({field: None}).to_svg() == build({}).to_svg()


_SIZED_COMPOSITES: dict[str, Callable[[dict], Any]] = {
    **_COMPOSITES,
    "mark_function": lambda st: fm.Chart(None).mark_function(np.sin, domain=[0, 6.0], **st),
}


@pytest.mark.parametrize("entry", sorted(_SIZED_COMPOSITES))
def test_a_composite_size_none_is_unset(entry: str) -> None:
    """`size=None` reaches no size rule, so it neither refuses nor raises a
    raw type error, and the chart draws as it does without it."""
    build = _SIZED_COMPOSITES[entry]
    assert build({"size": None}).to_svg() == build({}).to_svg()


def test_a_composite_spreads_a_field_only_onto_the_layers_that_draw_it() -> None:
    """Control: the field reaches the line, which draws it, and not the
    ribbon, which never would."""
    chart = _COMPOSITES["mark_smooth"]({"stroke_dash": [4.0, 2.0]})
    layers = json.loads(chart.to_spec().to_json())["layers"]
    dashes = {ly["mark"]: ly.get("mark_style", {}).get("stroke_dash") for ly in layers}
    assert dashes == {"ribbon": None, "line": [4.0, 2.0]}
    _no_warnings(chart.to_svg)


def test_a_composite_catch_all_none_leaves_the_desugars_own_default() -> None:
    """`None` on the chart-level catch-all is unset, so it does not
    overwrite a desugar's own built-in default: the ribbon still draws
    its hard-coded `opacity=0.3`, not fully opaque, and a non-`None`
    catch-all still overrides it as before."""
    chart = _COMPOSITES["mark_smooth"]({"opacity": None})
    layers = json.loads(chart.to_spec().to_json())["layers"]
    opacities = {ly["mark"]: ly.get("mark_style", {}).get("opacity") for ly in layers}
    assert opacities == {"ribbon": 0.3, "line": None}
    _no_warnings(chart.to_svg)

    overridden = _COMPOSITES["mark_smooth"]({"opacity": 0.8})
    overridden_layers = json.loads(overridden.to_spec().to_json())["layers"]
    assert {ly["mark"]: ly.get("mark_style", {}).get("opacity") for ly in overridden_layers} == {
        "ribbon": 0.8,
        "line": 0.8,
    }


def test_a_compound_figure_spreads_a_field_only_onto_the_panels_that_draw_it() -> None:
    """The marginal bars round their corners; the joint scatter, whose point
    mark never draws a corner radius, is left as it was."""
    plain = fm.jointplot(_DF, x="x", y="y").to_svg()
    rounded = _no_warnings(
        lambda: fm.jointplot(_DF, x="x", y="y", mark={"corner_radius": 3.0}).to_svg()
    )
    assert 'rx="3"' in rounded and 'rx="3"' not in plain
    assert rounded.count("<circle") == plain.count("<circle") > 0


# ── a field that needs another input warns when that input is absent ───────

_WIDE = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0],
        "x2": [1.5, 2.5, 3.5],
        "y": [1.0, 3.0, 2.0],
        "y2": [2.0, 4.0, 3.0],
        "g": ["a", "b", "c"],
    }
)
#: The encodings each method mark needs to draw anything at all.
_DRAWABLE: dict[str, dict[str, str]] = {
    "point": {"x": "x", "y": "y"},
    "bar": {"x": "g", "y": "y"},
    "area": {"x": "x", "y": "y"},
    "rect": {"x": "x", "x2": "x2", "y": "y", "y2": "y2"},
    "arc": {"theta": "y"},
    "ribbon": {"x": "x", "y": "y", "y2": "y2"},
    "segment": {"x": "x", "y": "y", "x2": "x2", "y2": "y2"},
}
_URL = "data:image/png;base64,iVBORw0KGgo="
#: `_WIDE` with a second band column and an image url per row.
_NEEDS = _WIDE.with_columns(pl.Series("h", ["p", "q", "p"]), pl.Series("u", [_URL] * 3))
#: Points binned by a raster into a 4 x 4 grid of differing counts.
_GRID = pl.DataFrame(
    [(float(i), float(j)) for i in range(4) for j in range(4) for _ in range(i + 4 * j + 1)],
    schema=["x", "y"],
    orient="row",
)
_POLY = pl.DataFrame(
    {"x": [0.0, 1.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0, 1.0], "v": [1.0, 2.0, 3.0, 4.0]}
)
_HEATMAP = {"x": "g", "y": "h", "color": "x"}
_RANGES = {"x": "x", "x2": "x2", "y": "y", "y2": "y2"}


def _polygon(style: dict, *, color: str = "v") -> Any:
    """A polygon layer over its own vertices; no mark method draws one."""
    data = _POLY.with_columns(pl.col("v").cast(pl.Utf8).alias("c"))
    return (
        fm.Chart(data)
        .mark_point()
        .encode(x="x", y="y", color=color)
        .layer(fm.Layer(mark="polygon", mark_kwargs=style))
    )


def _drawable(mark: str, style: dict) -> Any:
    if mark == "polygon":
        return _polygon(style)
    chart = getattr(fm.Chart(_WIDE), f"mark_{mark}")(**style).encode(**_DRAWABLE[mark])
    return chart.coord(fm.CoordPolar()) if mark == "arc" else chart


def _rect(style: dict, encoding: dict[str, str]) -> Any:
    return fm.Chart(_NEEDS).mark_rect(**style).encode(**encoding)


def _label(style: dict) -> Any:
    return fm.Chart(_NEEDS).mark_label(**style).encode(x="x", y="y", text="g")


def _image(style: dict) -> Any:
    return fm.Chart(_NEEDS).mark_image(**style).encode(x="x", y="y", url="u")


def _raster(style: dict) -> Any:
    return fm.Chart(_GRID).mark_raster(resolution=4, **style).encode(x="x", y="y")


def _layered_bar(style: dict) -> Any:
    """A bar layer on a chart of several layers, its values far from zero."""
    data = _CAT.with_columns(pl.col("v") + 50.0)
    return fm.catplot(data, x="g", y="v", kind="bar", mark={"bar": style})


class _Scenes(NamedTuple):
    """Charts drawing a field without (*unmet*) and with (*met*) the input it needs."""

    unmet: Callable[[str, dict], list[Any]]
    met: Callable[[str, dict], Any]


#: For each need Rust's style table names, the charts that lack and supply it.
#: A need set by style is supplied by that style; one set by an encoding, the
#: data or the mark's shape is supplied by a chart that has it.
_NEED_SCENES: dict[str, _Scenes] = {
    "stroke": _Scenes(
        lambda m, st: [_drawable(m, st)],
        lambda m, st: _drawable(m, {"stroke": "#000000", **st}),
    ),
    "arrow": _Scenes(
        lambda m, st: [_drawable(m, st)],
        lambda m, st: _drawable(m, {"arrow": True, **st}),
    ),
    # dx alone or dy alone: the label places itself.
    "leader_placement": _Scenes(
        lambda m, st: [_label(st)],
        lambda m, st: _label({"dx": 7.0, "dy": 7.0, **st}),
    ),
    # A text encoding on a heatmap: a heatmap without one, and text on a
    # rect of two quantitative ranges, both lack it.
    "text_channel": _Scenes(
        lambda m, st: [_rect(st, _HEATMAP), _rect(st, {**_RANGES, "text": "g"})],
        lambda m, st: _rect(st, {**_HEATMAP, "text": "g"}),
    ),
    # A band axis paired with a range.
    "band_scale": _Scenes(
        lambda m, st: [_rect(st, _HEATMAP), _rect(st, _RANGES)],
        lambda m, st: _rect(st, {"x": "g", "y": "y", "y2": "y2"}),
    ),
    "url_channel": _Scenes(lambda m, st: [_raster(st)], lambda m, st: _image(st)),
    "raster_data": _Scenes(lambda m, st: [_image(st)], lambda m, st: _raster(st)),
    "quantitative_color": _Scenes(
        lambda m, st: [_polygon(st, color="c")], lambda m, st: _polygon(st)
    ),
    # A flat bar's zero= is consumed by Python and never reaches the mark
    # (`test_a_flat_bar_figure_zero_is_consumed_not_put_on_the_wire`), so no
    # public chart lacks this need.
    "layered_chart": _Scenes(lambda m, st: [], lambda m, st: _layered_bar(st)),
}
#: Every field Rust's style table conditions on a need, on every mark.
_NEED_CELLS = [
    (m, f, need) for m in _MARKS for f, need in mark_style_fields(m).items() if need is not None
]
_UNMET_CELLS = [
    (m, f, need, i)
    for m, f, need in _NEED_CELLS
    for i in range(len(_NEED_SCENES[need].unmet(m, {}) if need in _NEED_SCENES else []))
]


def _render_warnings(chart: Any) -> tuple[str, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    return svg, [str(w.message) for w in caught]


def _inert(messages: list[str]) -> list[str]:
    return [m for m in messages if " has no effect on this " in m]


def test_the_inert_population_is_every_need_rust_names() -> None:
    """Every need the table names has scenes, every scene a cell, and every
    need but the one no public chart lacks has an unmet scene."""
    needs = {need for _, _, need in _NEED_CELLS}
    assert needs == set(_NEED_SCENES)
    assert {need for _, _, need, _ in _UNMET_CELLS} == needs - {"layered_chart"}
    assert {"leader_placement", "text_channel", "band_scale", "url_channel"} <= needs
    assert ("label", "dx", "leader_placement", 0) in _UNMET_CELLS
    assert ("rect", "font_size", "text_channel", 1) in _UNMET_CELLS


@pytest.mark.parametrize(("mark", "field", "need", "scene"), _UNMET_CELLS)
def test_a_field_without_its_enabling_input_warns_inert(
    mark: str, field: str, need: str, scene: int
) -> None:
    """The field warns once, naming itself and its mark."""
    chart = _NEED_SCENES[need].unmet(mark, {field: _SAMPLE[field]})[scene]
    _, messages = _render_warnings(chart)
    inert = _inert(messages)
    assert len(inert) == 1, messages
    assert re.search(
        rf"mark_\w+: {field} has no effect on this {mark} mark without .+; it was ignored$",
        inert[0],
    ), inert[0]


@pytest.mark.parametrize("scene", range(len(_NEED_SCENES["text_channel"].unmet("rect", {}))))
def test_a_rect_font_size_names_both_inputs_it_needs(scene: int) -> None:
    """A heatmap without a text encoding and text on a rect of two ranges
    both lack the one input pair a rect's font size draws with, so both
    warnings name the text encoding and the heatmap shape."""
    chart = _NEED_SCENES["text_channel"].unmet("rect", {"font_size": 14.0})[scene]
    _, messages = _render_warnings(chart)
    assert _inert(messages) == [
        "mark_rect: font_size has no effect on this rect mark without both a text encoding "
        "and the heatmap shape (one cell per x/y band pair, no x2 or y2 range); it was ignored"
    ]


@pytest.mark.parametrize(("mark", "field", "need"), _NEED_CELLS)
def test_a_field_with_its_enabling_input_draws_without_warning(
    mark: str, field: str, need: str
) -> None:
    met = _NEED_SCENES[need].met
    svg, messages = _render_warnings(met(mark, {field: _SAMPLE[field]}))
    assert not _inert(messages), messages
    assert svg != met(mark, {}).to_svg()


# ── vocabularies ────────────────────────────────────────────────────────────

_VOCABULARIES = {
    "blend": blend_accepted_values(),
    "stroke_cap": stroke_cap_accepted_values(),
    "stroke_join": stroke_join_accepted_values(),
}


@pytest.mark.parametrize("field", sorted(_VOCABULARIES))
def test_a_value_outside_a_rust_vocabulary_refuses_at_the_door(field: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        fm.Chart(_DF).mark_line(**{field: "bogus"})
    assert str(exc_info.value) == (
        f"mark_line: unknown {field} 'bogus'; expected {'|'.join(_VOCABULARIES[field])}"
    )


@pytest.mark.parametrize("field", sorted(_VOCABULARIES))
def test_a_value_outside_a_rust_vocabulary_refuses_on_a_layer(field: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        fm.Layer(mark="line", mark_kwargs={field: "bogus"})
    assert str(exc_info.value) == (
        f"Layer: unknown {field} 'bogus'; expected {'|'.join(_VOCABULARIES[field])}"
    )


@pytest.mark.parametrize("mark", sorted(_METHOD_ONLY))
def test_an_orient_outside_its_vocabulary_refuses(mark: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        getattr(fm.Chart(_CAT), f"mark_{mark}")(orient="diagonal")
    assert str(exc_info.value) == (
        f"mark_{mark}: orient must be one of ['horizontal', 'vertical']; got 'diagonal'"
    )


def test_mark_raster_refuses_a_blend_outside_the_vocabulary() -> None:
    with pytest.raises(ValueError) as exc_info:
        fm.Chart(_DF).mark_raster(blend="multiply")
    assert str(exc_info.value) == (
        f"mark_raster: unknown blend 'multiply'; expected {'|'.join(blend_accepted_values())}"
    )


# ── blend= works on every door ──────────────────────────────────────────────

_SCREEN = "mix-blend-mode:screen"


def _point(**style: Any) -> Any:
    return fm.Chart(_DF).mark_point(**style).encode(x="x", y="y")


_BLEND_DOORS: dict[str, Callable[[dict], Any]] = {
    "mark_point": lambda st: _point(**st),
    "a + b": lambda st: _point(**st) + fm.Chart(_DF).mark_line().encode(x="x", y="y"),
    "relplot mark=": lambda st: fm.relplot(_DF, x="x", y="y", mark=st),
    "Layer": lambda st: (
        fm.Chart(_DF).mark_line().encode(x="x", y="y").layer(fm.Layer(mark="point", mark_kwargs=st))
    ),
    "override": lambda st: _point().override(**{f"mark_{k}": v for k, v in st.items()}),
    "roc_chart mark={line}": lambda st: fm.roc_chart(y_true=_YT, y_pred=_YP, mark={"line": st}),
    "mark_smooth": lambda st: fm.Chart(_DF).mark_smooth(**st).encode(x="x", y="y"),
    "mark_raster": lambda st: fm.Chart(_DF).mark_raster(**st).encode(x="x", y="y"),
}


@pytest.mark.parametrize("door", sorted(_BLEND_DOORS))
def test_blend_additive_blends_exactly_the_layer_that_set_it(door: str) -> None:
    build = _BLEND_DOORS[door]
    plain = build({}).to_svg()
    assert plain.count(_SCREEN) == 0
    assert build({"blend": "additive"}).to_svg().count(_SCREEN) == 1


@pytest.mark.parametrize("door", sorted(_BLEND_DOORS))
def test_blend_alpha_is_the_default_byte_for_byte(door: str) -> None:
    build = _BLEND_DOORS[door]
    assert build({"blend": "alpha"}).to_svg() == build({}).to_svg()


def test_mark_raster_additive_rides_mark_style_not_a_layer_slot() -> None:
    chart = fm.Chart(_DF).mark_raster(blend="additive").encode(x="x", y="y")
    (layer,) = json.loads(chart.to_spec().to_json())["layers"]
    assert "blend" not in layer
    assert layer["mark_style"]["blend"] == "additive"


def test_mark_raster_blend_none_is_unset_on_the_wire() -> None:
    # blend=None is "unset" (mark_raster's vocabulary check passes it), so the
    # image layer serializes exactly as a raster that never named a blend:
    # no mark_style at all, not an empty one.
    def spec(**style: Any) -> str:
        return fm.Chart(_DF).mark_raster(**style).encode(x="x", y="y").to_spec().to_json()

    (layer,) = json.loads(spec(blend=None))["layers"]
    assert "mark_style" not in layer
    assert spec(blend=None) == spec()


def test_mark_raster_writes_the_blend_it_was_given_and_leaves_the_alias_to_rust() -> None:
    # Python reads no spelling of the vocabulary: `alpha` reaches the wire as
    # written, and Rust's `BlendMode::parse` draws it as `normal`.
    def layer(**style: Any) -> dict:
        chart = fm.Chart(_DF).mark_raster(**style).encode(x="x", y="y")
        (only,) = json.loads(chart.to_spec().to_json())["layers"]
        return only

    assert layer(blend="alpha")["mark_style"] == {"blend": "alpha"}
    assert layer(blend="normal")["mark_style"] == {"blend": "normal"}
    assert "mark_style" not in layer()


# ── a value of the wrong type refuses at every door ─────────────────────────
#
# Rust publishes which wire fields are truth-valued and which are strings
# (`mark_bool_kwargs`, `mark_string_kwargs`). Each door holds a field of that
# kind to its type before anything reads it, so `"no"` is never taken as
# True and a non-string `shape`/`fill`/`stroke` never reaches the wire. Each
# cell styles, at one door, a mark that draws the field; a field only
# ferrum's own code writes is refused as such whatever its type, and a string
# field a Rust vocabulary governs refuses a non-string as outside that
# vocabulary.

#: Every style field a Rust vocabulary governs, with its accepted values.
_ALL_VOCABULARIES = {**_VOCABULARIES, "interpolate": interpolate_accepted_values()}


def _string_refusal(voice: str, field: str) -> str:
    if field in _ALL_VOCABULARIES:
        return f"{voice}: unknown {field} '5'; expected {'|'.join(_ALL_VOCABULARIES[field])}"
    return f"{voice}: {field} must be a string; got 5"


#: field kind -> (its fields, a wrong-typed value, the refusal for it)
_KINDS: dict[str, tuple[tuple[str, ...], Any, Callable[[str, str], str]]] = {
    "bool": (
        tuple(mark_bool_kwargs()),
        "no",
        lambda voice, field: f"{voice}: {field} must be True or False; got 'no'",
    ),
    "string": (tuple(mark_string_kwargs()), 5, _string_refusal),
}

_TYPE_DOORS = ("method", "Layer", "override", "figure", "composite")


def _first_drawing(marks: Any, field: str) -> str | None:
    return next((m for m in marks if field in _drawn(m)), None)


def _figure_drawing(field: str) -> str | None:
    return next((d for d, (_, _, m) in _FIGURE_DOORS.items() if field in _drawn(m)), None)


def _typed_door(door: str, field: str) -> tuple[Callable[[Any], Any], str, type[Exception]] | None:
    """The build of one *door* styling a mark that draws *field*, with its
    voice and the refusal's type; ``None`` when no mark that door reaches
    draws the field. An internal field is no mark's, so it takes the door's
    first target."""
    internal = field in _INTERNAL_DOORS
    if door == "method":
        mark = _METHOD_MARKS[0] if internal else _first_drawing(_METHOD_MARKS, field)
        if mark is None:
            return None
        return (
            lambda v: (
                getattr(fm.Chart(_DF), f"mark_{mark}")(**{field: v}).encode(x="x", y="y").to_spec()
            ),
            f"mark_{mark}",
            ValueError,
        )
    if door == "Layer":
        mark = _MARKS[0] if internal else _first_drawing(_MARKS, field)
        if mark is None:
            return None
        return (
            lambda v: fm.Layer(mark=mark, mark_kwargs={field: v}),
            "Layer",
            ValueError,
        )
    if door == "override":
        mark = _FLAT_METHOD_MARKS[0] if internal else _first_drawing(_FLAT_METHOD_MARKS, field)
        if mark is None:
            return None
        return (
            lambda v: (
                getattr(fm.Chart(_DF), f"mark_{mark}")()
                .encode(x="x", y="y")
                .override(**{f"mark_{field}": v})
                .to_spec()
            ),
            "Chart.override",
            FerrumOverrideError,
        )
    if door == "figure":
        name = next(iter(_FIGURE_DOORS)) if internal else _figure_drawing(field)
        if name is None:
            return None
        fig, voice, _ = _FIGURE_DOORS[name]
        return (lambda v: fig({field: v}).to_spec(), voice, ValueError)
    build = _COMPOSITES["mark_smooth"]
    return (lambda v: build({field: v}).to_spec(), "mark_smooth", ValueError)


_TYPE_CELLS = [
    (kind, door, field)
    for kind, (fields, _, _) in _KINDS.items()
    for field in fields
    for door in _TYPE_DOORS
    if _typed_door(door, field) is not None
]


def test_the_wrong_type_populations_are_read_from_rust_and_reach_every_door() -> None:
    """Vacuity control: each kind has fields outside the internal set, every
    door has cells of both kinds, and every field of a kind is refused at the
    two doors that reach every field."""
    bools, strings = _KINDS["bool"][0], _KINDS["string"][0]
    assert not set(bools) & set(strings)
    assert set(bools) | set(strings) < set(_FIELDS)
    assert {"zero", "filled", "clip"} <= set(bools)
    assert {"shape", "fill", "stroke", "value_axis"} <= set(strings)
    # A vocabulary field is never truth-valued, and some are string fields,
    # so the matrix reaches `_string_refusal`'s vocabulary branch as well as
    # its plain one (`shape` above). Which vocabularies are strings is Rust's
    # to say.
    assert not set(_ALL_VOCABULARIES) & set(bools)
    assert set(_ALL_VOCABULARIES) & set(strings)
    for kind, (fields, _, _) in _KINDS.items():
        assert set(fields) - set(_INTERNAL_DOORS)
        for door in _TYPE_DOORS:
            assert any(c[:2] == (kind, door) for c in _TYPE_CELLS), (kind, door)
        for door in ("Layer", "composite"):
            assert {c[2] for c in _TYPE_CELLS if c[:2] == (kind, door)} == set(fields)


@pytest.mark.parametrize(("kind", "door", "field"), _TYPE_CELLS)
def test_a_wrong_typed_value_refuses_at_every_door(kind: str, door: str, field: str) -> None:
    _, wrong, refusal = _KINDS[kind]
    build, voice, error = _typed_door(door, field)  # type: ignore[misc]
    with pytest.raises(error) as exc_info:
        build(wrong)
    expected = (
        _internal_refusal(voice, field) if field in _INTERNAL_DOORS else refusal(voice, field)
    )
    assert str(exc_info.value) == expected


@pytest.mark.parametrize(("kind", "door", "field"), _TYPE_CELLS)
def test_a_correctly_typed_value_passes_the_same_door(kind: str, door: str, field: str) -> None:
    """Control: the cell that refuses a wrong type takes the right one. An
    internal field is refused whatever its type; a composite's catch-all
    field that none of its emitted marks draws is refused as untaken."""
    build, voice, error = _typed_door(door, field)  # type: ignore[misc]
    if field in _INTERNAL_DOORS:
        with pytest.raises(error) as exc_info:
            build(_SAMPLE[field])
        assert str(exc_info.value) == _internal_refusal(voice, field)
        return
    emitted = _emitted_marks(_COMPOSITES["mark_smooth"])
    if door == "composite" and not any(field in _drawn(m) for m in emitted):
        with pytest.raises(ValueError) as exc_info:
            build(_SAMPLE[field])
        assert str(exc_info.value) == _no_layer_draws("mark_smooth", [field], emitted)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build(_SAMPLE[field])


def test_a_flat_bar_zero_is_never_read_for_its_truthiness() -> None:
    """``zero="no"`` used to anchor the bar at zero as if it were True."""
    for build, voice in (
        (lambda: fm.Chart(_HIGH).mark_bar(zero="no"), "mark_bar"),
        (
            lambda: (
                fm.Chart(_HIGH).mark_bar().encode(x="g", y="v").override(mark_zero="no").to_spec()
            ),
            "Chart.override",
        ),
        (lambda: fm.catplot(_CAT, x="g", kind="count", mark={"zero": "no"}), "catplot"),
    ):
        with pytest.raises(ValueError) as exc_info:
            build()
        assert str(exc_info.value) == f"{voice}: zero must be True or False; got 'no'"


#: Uneven category counts, so a count bar left unanchored loses its zero.
_COUNTED = pl.DataFrame({"g": ["a"] * 5 + ["b"] * 7 + ["c"] * 9})


@pytest.mark.parametrize(
    "build",
    [
        lambda zero: fm.Chart(_HIGH).mark_bar(**zero).encode(x="g", y="v"),
        lambda zero: (
            fm.Chart(_HIGH)
            .mark_bar()
            .encode(x="g", y="v")
            .override(**{f"mark_{k}": v for k, v in zero.items()})
        ),
        lambda zero: fm.catplot(_COUNTED, x="g", kind="count", mark=zero),
    ],
    ids=["method", "override", "figure"],
)
def test_a_flat_bar_zero_none_is_unset_and_keeps_the_zero_anchor(build: Callable) -> None:
    """``zero=None`` is the unset value, so the bar stays anchored at zero as
    by default; it used to be read as ``bool(None)``, unanchored."""
    anchored = build({}).to_svg()
    assert build({"zero": None}).to_svg() == anchored
    assert build({"zero": False}).to_svg() != anchored


def test_a_numpy_bool_is_a_bool() -> None:
    """Control: a numpy truth value is taken as the bool it is, and reaches
    the wire as one."""
    unanchored = fm.Chart(_HIGH).mark_bar(zero=False).encode(x="g", y="v").to_svg()
    assert fm.Chart(_HIGH).mark_bar(zero=np.False_).encode(x="g", y="v").to_svg() == unanchored
    layer = fm.Layer(mark="point", mark_kwargs={"filled": np.False_})
    assert layer.mark_kwargs["filled"] is False


#: Repeated x values, so an error band has an extent to draw.
_BANDED = pl.DataFrame({"x": [0.0, 1.0, 2.0] * 4, "y": [float(i % 5) for i in range(12)]})

#: A truth-valued argument a mark method consumes itself:
#: (method, argument, its default, a chart building that method with it).
#: A sample with ties, which dequantization spreads apart.
_TIED = pl.DataFrame({"x": [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0] * 3})

_METHOD_BOOLS = [
    pytest.param(
        "mark_function",
        "clip",
        True,
        lambda v: fm.Chart(None).mark_function(np.sin, domain=[0, 6.28], clip=v),
        id="mark_function",
    ),
    pytest.param(
        "mark_errorband",
        "borders",
        False,
        lambda v: fm.Chart(_BANDED).mark_errorband(borders=v).encode(x="x", y="y"),
        id="mark_errorband",
    ),
    pytest.param(
        "mark_qq",
        "line",
        True,
        lambda v: fm.Chart(_DF).mark_qq(line=v).encode(x="x"),
        id="mark_qq",
    ),
    pytest.param(
        "mark_qq",
        "dequantize",
        False,
        lambda v: fm.Chart(_TIED).mark_qq(dequantize=v).encode(x="x"),
        id="mark_qq-dequantize",
    ),
    pytest.param(
        "mark_contour",
        "fill",
        False,
        lambda v: fm.Chart(_DF).mark_contour(fill=v).encode(x="x", y="y"),
        id="mark_contour-fill",
    ),
    pytest.param(
        "mark_contour",
        "smooth",
        True,
        lambda v: fm.Chart(_DF).mark_contour(smooth=v).encode(x="x", y="y"),
        id="mark_contour-smooth",
    ),
    pytest.param(
        "mark_density",
        "fill",
        True,
        lambda v: fm.Chart(_DF).mark_density(fill=v).encode(x="x"),
        id="mark_density-fill",
    ),
    pytest.param(
        "mark_density",
        "cumulative",
        False,
        lambda v: fm.Chart(_DF).mark_density(cumulative=v).encode(x="x"),
        id="mark_density-cumulative",
    ),
    pytest.param(
        "mark_density",
        "smooth",
        True,
        lambda v: fm.Chart(_DF).mark_density(smooth=v).encode(x="x", y="y"),
        id="mark_density-smooth",
    ),
]


@pytest.mark.parametrize(("voice", "arg", "default", "build"), _METHOD_BOOLS)
@pytest.mark.parametrize("wrong", ["no", 0])
def test_a_method_bool_argument_is_never_read_for_its_truthiness(
    voice: str, arg: str, default: bool, build: Callable, wrong: Any
) -> None:
    """``borders="no"`` and ``mark_contour(fill="no")`` used to draw as if
    True; ``line="no"``, ``dequantize="no"`` and ``smooth="no"`` failed in
    the transform with PyO3's text. Each refuses at the door, in its voice."""
    with pytest.raises(EntryRefusal) as exc_info:
        build(wrong)
    assert str(exc_info.value) == f"{voice}: {arg} must be True or False; got {wrong!r}"


@pytest.mark.parametrize(("voice", "arg", "default", "build"), _METHOD_BOOLS)
def test_a_method_bool_argument_takes_a_truth_value_and_none_as_unset(
    voice: str, arg: str, default: bool, build: Callable
) -> None:
    """Control: a builtin or numpy truth value is the bool it is, ``None`` is
    the default, and the two values draw different charts."""
    svg = {v: build(v).to_svg() for v in (True, False)}
    assert svg[True] != svg[False]
    assert build(None).to_svg() == svg[default]
    assert build(np.True_).to_svg() == svg[True]
    assert build(np.False_).to_svg() == svg[False]


def test_mark_density_fill_is_a_truth_value_not_a_color() -> None:
    """``fill="red"`` used to draw an area, read for its truthiness; a fill
    color is ``color=``, which still reaches the area."""
    with pytest.raises(EntryRefusal) as exc_info:
        fm.Chart(_DF).mark_density(fill="red").encode(x="x")
    assert str(exc_info.value) == "mark_density: fill must be True or False; got 'red'"
    svg = fm.Chart(_DF).mark_density(color="red").encode(x="x").to_svg()
    assert 'fill="rgba(255,0,0,' in svg


def test_mark_function_clip_false_unclips_the_line() -> None:
    """Control: a real ``False`` (numpy or builtin) still escapes the clip."""

    def wire(**kw: Any) -> dict:
        chart = fm.Chart(None).mark_function(np.sin, domain=[0, 6.28], **kw)
        return json.loads(chart.to_spec().to_json()).get("mark_style") or {}

    assert wire(clip=False) == wire(clip=np.False_) == {"clip": False}
    assert wire() == wire(clip=True) == {}


def test_an_override_reaches_the_wire_as_the_gate_resolved_it() -> None:
    """``Chart.override`` styles a mark with the values the style gate
    returns, the same ones the mark method stores, not the raw leaves."""
    method = fm.Chart(_DF).mark_line(stroke_dash="dashed").encode(x="x", y="y")
    override = fm.Chart(_DF).mark_line().encode(x="x", y="y").override(mark_stroke_dash="dashed")
    assert method.to_spec().to_json() == override.to_spec().to_json()
    assert override.to_svg() == method.to_svg()


# ── violin ──────────────────────────────────────────────────────────────────


def test_violin_documents_only_the_style_it_writes() -> None:
    from ferrum.marks.heavy_stat import desugar_violin

    assert "fill_opacity" not in (desugar_violin.__doc__ or "")
    chart = fm.Chart(_CAT).mark_violin().encode(x="g", y="v")
    (body,) = [
        ly for ly in json.loads(chart.to_spec().to_json())["layers"] if ly["mark"] == "polygon"
    ]
    assert set(body["mark_style"]) == {"detail"}
