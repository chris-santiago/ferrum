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
import warnings
from decimal import Decimal
from typing import Any, Callable

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
        "roc_chart(mark=...): unknown sub-layer 'bogus'; this chart's sub-layers are "
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
    with pytest.raises(ValueError, match=r"draws no \['outlier'\] sub-layer"):
        chart.to_svg()


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


def test_a_composite_spreads_a_field_only_onto_the_layers_that_draw_it() -> None:
    """Control: the field reaches the line, which draws it, and not the
    ribbon, which never would."""
    chart = _COMPOSITES["mark_smooth"]({"stroke_dash": [4.0, 2.0]})
    layers = json.loads(chart.to_spec().to_json())["layers"]
    dashes = {ly["mark"]: ly.get("mark_style", {}).get("stroke_dash") for ly in layers}
    assert dashes == {"ribbon": None, "line": [4.0, 2.0]}
    _no_warnings(chart.to_svg)


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
#: The style that supplies each need a style kwarg can supply.
_ENABLES = {"stroke": {"stroke": "#000000"}, "arrow": {"arrow": True}}
_INERT_CELLS = [
    (m, f, need)
    for m in _METHOD_MARKS
    for f, need in mark_style_fields(m).items()
    if need in _ENABLES
]


def _drawable(mark: str, style: dict) -> Any:
    chart = getattr(fm.Chart(_WIDE), f"mark_{mark}")(**style).encode(**_DRAWABLE[mark])
    return chart.coord(fm.CoordPolar()) if mark == "arc" else chart


def _render_warnings(chart: Any) -> tuple[str, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    return svg, [str(w.message) for w in caught]


def test_the_inert_population_is_every_style_supplied_need() -> None:
    assert {need for _, _, need in _INERT_CELLS} == set(_ENABLES)
    assert {"point", "bar", "segment"} <= {m for m, _, _ in _INERT_CELLS}


@pytest.mark.parametrize(("mark", "field", "need"), _INERT_CELLS)
def test_a_field_without_its_enabling_input_warns_inert(mark: str, field: str, need: str) -> None:
    chart = _drawable(mark, {field: _SAMPLE[field]})
    # A method whose mark is a composite's one layer speaks as a layer.
    layer = "layer 0: " if chart._resolve_pending()._layers else ""
    _, messages = _render_warnings(chart)
    inert = [m for m in messages if "has no effect on this" in m]
    assert len(inert) == 1, messages
    assert inert[0].startswith(
        f"{layer}mark_{mark}: {field} has no effect on this {mark} mark without "
    )
    assert inert[0].endswith("; it was ignored")


@pytest.mark.parametrize(("mark", "field", "need"), _INERT_CELLS)
def test_a_field_with_its_enabling_input_draws_without_warning(
    mark: str, field: str, need: str
) -> None:
    enabled = _ENABLES[need]
    svg, messages = _render_warnings(_drawable(mark, {**enabled, field: _SAMPLE[field]}))
    assert not [m for m in messages if "has no effect on this" in m], messages
    assert svg != _drawable(mark, enabled).to_svg()


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
