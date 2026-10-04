"""A mark's style kwargs are held to that mark at every door, in the caller's voice.

The doors are ``fm.Layer(mark_kwargs=)``, a figure's ``mark=`` dict (flat,
named sub-layer, compound), ``lmplot``'s ``line_kws=``/``scatter_kws=``, a
composite's ``layer_kwargs=`` and ``Chart.override(mark_<leaf>=)``. Each door
is checked against three inputs: a key no mark has, a real key the target mark
does not draw, and (the vacuity control) a key it does draw, which must change
the render.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable

import numpy as np
import polars as pl
import pytest

import ferrum as fm
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
    "Layer": (_layered, "mark_line", "line", {"stroke_width": 6}),
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


# ── decimal feature columns ─────────────────────────────────────────────────

_DEC = pl.DataFrame(
    {"a": [1.0, 2.0, 3.0], "d": [Decimal("1.5"), Decimal("2.5"), Decimal("0.5")], "b": [3, 1, 2]}
)


@pytest.mark.parametrize("entry", ["heatmap", "clustermap", "pairplot"])
def test_a_decimal_feature_column_refuses_naming_the_entry(entry: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        getattr(fm, entry)(_DEC)
    assert str(exc_info.value) == (
        f"{entry}: column 'd' is a decimal column (Decimal128(38, 1)); a feature column "
        "must be an integer or float column, so cast it to float first"
    )


def test_the_same_frame_cast_to_float_renders() -> None:
    """Control: the refusal is about the dtype, not the data."""
    fm.heatmap(_DEC.with_columns(pl.col("d").cast(pl.Float64))).to_svg()
