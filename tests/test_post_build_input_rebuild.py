"""A composite mark that resolved early rebuilds on a later binding to another column.

A composite mark (``mark_smooth``, ``mark_histogram``, ``mark_boxplot``,
...) resolves when its chart takes ``.layer(...)``, becomes a ``+`` operand,
or receives a ``mark=`` override: its statistic is built from the x/y it
holds at that moment. A later ``.encode()`` that binds x or y to a column
other than the one the statistic read draws what the same binding written
before the resolution draws: the statistic is built again from the new
column, and the chart's own x/y take the rebuilt build's own, so axis titles
and the drawn layers read as the building door's do.

The check this module repeats is the twin comparison: the chart that took the
binding after the resolution renders byte-identical to the chart that was
given the binding first, on the static and the interactive route. A
negative control disables the one reader of written inputs and requires the
twins to differ, so an equality that held only because both sides ignored
the binding would fail.

What stays as it was: a same-column restatement (the binding names the field
the statistic read), a grouping binding alone, a binding on a composition
(``+``, ``Layer(data=)``), and a pending mark, which reads its final encoding
itself. Those rows are pinned by digest against the unmodified build.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from functools import partial
from typing import Any

import polars as pl
import pytest

import ferrum as fm
import ferrum._compose as _compose


def _frame() -> pl.DataFrame:
    n = 36
    return pl.DataFrame(
        {
            "v": [float(i) for i in range(n)],
            "w": [float((i * 7) % 11) + 0.5 * i for i in range(n)],
            "w2": [float((i * 5) % 13) + 0.25 * i + 3.0 for i in range(n)],
            "u": [float((i * 3) % 17) + 0.1 * i for i in range(n)],
            "g": [("a", "b", "c")[i % 3] for i in range(n)],
            "f": [("p", "q")[i % 2] for i in range(n)],
        }
    )


_DF = _frame()
_OTHER = _DF.head(12)


def _calc():
    return fm.transform_calculate("w3", "datum.w * 2 + 1")


def _rule() -> Any:
    return fm.Layer(mark="tick", encoding={"x": "v"})


def _smooth() -> "fm.Chart":
    return fm.Chart(_DF).mark_smooth().encode(x="v", y="w")


def _histogram() -> "fm.Chart":
    return fm.Chart(_DF).mark_histogram().encode(x="v")


def _boxplot() -> "fm.Chart":
    return fm.Chart(_DF).mark_boxplot().encode(x="g", y="v")


#: (case id, the pending chart's constructor, the binding written later).
_REBINDINGS: list[tuple[str, Callable[[], "fm.Chart"], dict[str, Any]]] = [
    ("smooth-y", _smooth, {"y": "w2"}),
    ("smooth-x", _smooth, {"x": "u"}),
    ("smooth-x-and-y", _smooth, {"x": "u", "y": "w2"}),
    ("histogram-x", _histogram, {"x": "u"}),
    ("boxplot-y", _boxplot, {"y": "w2"}),
    ("smooth-y-with-title", _smooth, {"y": fm.Y("w2", title="T")}),
    ("smooth-x-with-scale", _smooth, {"x": fm.X("u", scale=fm.SymlogScale())}),
    ("smooth-y-and-color", _smooth, {"y": "w2", "color": "g"}),
]
_IDS = [case[0] for case in _REBINDINGS]


def _door(
    make: Callable[[], "fm.Chart"], later: dict[str, Any], *, layer: bool = True
) -> "fm.Chart":
    """The building door: *later* written on the pending chart, then resolved by ``.layer()``."""
    chart = make().encode(**later)
    return chart.layer(_rule()) if layer else chart


def _after_layer(make: Callable[[], "fm.Chart"], later: dict[str, Any]) -> "fm.Chart":
    """The chart that resolved first and takes *later* afterwards."""
    return make().layer(_rule()).encode(**later)


def _svg(chart: "fm.Chart") -> str:
    return chart.to_svg()


@pytest.mark.parametrize(("case", "make", "later"), _REBINDINGS, ids=_IDS)
def test_a_binding_after_layer_equals_the_binding_written_first(case, make, later) -> None:
    assert _svg(_after_layer(make, later)) == _svg(_door(make, later))


@pytest.mark.parametrize(("case", "make", "later"), _REBINDINGS, ids=_IDS)
def test_the_rebuilt_chart_serializes_as_the_building_door_does(case, make, later) -> None:
    assert _after_layer(make, later).to_json() == _door(make, later).to_json()


@pytest.mark.parametrize(("case", "make", "later"), _REBINDINGS, ids=_IDS)
def test_the_interactive_route_equals_the_static_twin(case, make, later) -> None:
    late = _after_layer(make, later)
    assert late.interactive().to_html() == _door(make, later).interactive().to_html()


@pytest.mark.parametrize(("case", "make", "later"), _REBINDINGS, ids=_IDS)
def test_the_negative_control_differs_when_the_rebuild_is_disabled(
    monkeypatch, case, make, later
) -> None:
    """With the reader of written inputs returning nothing the twins differ, so equality above is not vacuous."""
    door = _svg(_door(make, later))
    monkeypatch.setattr(_compose, "written_inputs", lambda chart, record: {})
    assert _svg(_after_layer(make, later)) != door


def test_a_binding_after_layer_on_a_faceted_chart_equals_the_faceted_door() -> None:
    door = _smooth().encode(y="w2").facet(col="f").layer(_rule())
    for late in (
        _smooth().facet(col="f").layer(_rule()).encode(y="w2"),
        _smooth().layer(_rule()).facet(col="f").encode(y="w2"),
    ):
        assert _svg(late) == _svg(door)


@pytest.mark.parametrize(
    "make_mark,later",
    [
        (lambda c: c.mark_smooth().encode(x="v", y="w"), {"y": "w2"}),
        (lambda c: c.mark_histogram().encode(x="v"), {"x": "u"}),
        (lambda c: c.mark_boxplot().encode(x="g", y="v"), {"y": "w2"}),
    ],
    ids=["smooth", "histogram", "boxplot"],
)
def test_a_faceted_mark_with_another_column_equals_the_faceted_pre_resolution_door(
    make_mark, later
) -> None:
    def chart() -> "fm.Chart":
        return make_mark(fm.Chart(_DF).facet(col="f"))

    door = chart().encode(**later).layer(_rule())
    assert _svg(chart().layer(_rule()).encode(**later)) == _svg(door)


def test_a_binding_after_a_transform_chain_reads_the_transformed_rows() -> None:
    door = _smooth().encode(y="w3").transform(_calc()).layer(_rule())
    late = _smooth().transform(_calc()).layer(_rule()).encode(y="w3")
    assert _svg(late) == _svg(door)


@pytest.mark.parametrize(
    "make_mark,later",
    [
        (lambda c: c.mark_smooth().encode(x="v", y="w"), {"y": "w3"}),
        (lambda c: c.mark_boxplot().encode(x="g", y="v"), {"y": "w3"}),
        (lambda c: c.mark_histogram().encode(x="v"), {"x": "w3"}),
    ],
    ids=["smooth", "boxplot", "histogram"],
)
def test_each_composite_mark_over_a_transform_chain_equals_its_door(make_mark, later) -> None:
    chart = make_mark(fm.Chart(_DF)).transform(_calc())
    door = make_mark(fm.Chart(_DF)).encode(**later).transform(_calc()).layer(_rule())
    assert _svg(chart.layer(_rule()).encode(**later)) == _svg(door)


def test_a_resolved_operand_rebound_before_plus_equals_the_binding_written_first() -> None:
    """The operand resolved at ``.layer()``, took ``y`` after it, then joined a ``+``."""
    late = _smooth().layer(_rule()).encode(y="w2") + _rule_chart()
    door = _smooth().encode(y="w2").layer(_rule()) + _rule_chart()
    assert _svg(late) == _svg(door)
    assert late.interactive().to_html() == door.interactive().to_html()


def _rule_chart() -> "fm.Chart":
    return fm.Chart(_DF).mark_tick().encode(x="v")


def test_a_binding_after_a_resolving_override_equals_the_pending_door() -> None:
    """A resolved chart taking a mark override still rebuilds on a later other-column binding."""
    from ferrum._overrides import _apply_mark_overrides

    resolved = _apply_mark_overrides(_smooth(), {}, entry="mark_smooth")
    door = _smooth().encode(y="w2").layer()
    assert _svg(resolved.encode(y="w2")) == _svg(door)


def test_a_binding_replaces_the_whole_channel() -> None:
    late = _after_layer(_smooth, {"y": fm.Y("w2", title="T")})
    door = _door(_smooth, {"y": fm.Y("w2", title="T")})
    assert _svg(late) == _svg(door)
    assert ">T<" in _svg(late)
    assert ">w<" not in _svg(late)


def _refusal(chart: "fm.Chart") -> tuple[type, str]:
    with pytest.raises(Exception) as caught:  # noqa: PT011 - the type is part of the comparison
        chart.to_svg()
    return type(caught.value), str(caught.value)


def test_a_binding_on_a_missing_column_is_the_statistics_refusal() -> None:
    late = _refusal(_after_layer(_smooth, {"y": "nope"}))
    door = _refusal(_door(_smooth, {"y": "nope"}, layer=False))
    assert late == door
    assert "nope" in late[1]


def test_a_binding_naming_the_statistics_output_column_is_refused_as_the_pending_door() -> None:
    later = {"y": fm.Y("y", title="T")}
    late = _refusal(_after_layer(_smooth, later))
    door = _refusal(_door(_smooth, later, layer=False))
    assert late == door


def test_grouping_and_inputs_rebuild_in_one_resolution(monkeypatch) -> None:
    import ferrum._desugar as _desugar

    calls = {"n": 0}
    original = _desugar._resolve_pending_as

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_desugar, "_resolve_pending_as", counted)
    chart = _smooth().layer(_rule())
    assert calls["n"] == 1
    chart.encode(y="w2", color="g").to_svg()
    assert calls["n"] == 2


# ── Unchanged rows, pinned by digest against the unmodified build ──────────


def _digest(chart: "fm.Chart") -> str:
    return hashlib.sha256(chart.to_svg().encode()).hexdigest()[:16]


def _unchanged_charts() -> dict[str, "fm.Chart"]:
    layered = _smooth().layer(_rule())
    return {
        "labs-retitle": layered.labs(x="X title", y="T"),
        "restate-y-title": layered.encode(y=fm.Y("w", title="T")),
        "restate-y-format": layered.encode(y=fm.Y("w", format=".1f")),
        "restate-x-scale": layered.encode(x=fm.X("v", scale=fm.SymlogScale())),
        "color-only": layered.encode(color="g"),
        "plus-composition": (layered + _rule_chart()).encode(y="w2"),
        "layer-with-data": _smooth()
        .layer(fm.Layer(mark="tick", data=_OTHER, encoding={"x": "v"}))
        .encode(y="w2"),
        "pending": _smooth().encode(y="w2"),
        "pending-then-layer-free": _smooth().encode(y="w2").layer(),
    }


_HEAD_DIGESTS: dict[str, str] = {
    "color-only": "c4dffaeaec70e861",
    "labs-retitle": "056510aec1ffce40",
    "layer-with-data": "0b51f24d98230506",
    "pending": "94063103b8babe87",
    "pending-then-layer-free": "94063103b8babe87",
    "plus-composition": "d479280a5c160267",
    "restate-x-scale": "c573c3d60d9e1b96",
    "restate-y-format": "c573c3d60d9e1b96",
    "restate-y-title": "e284c04995a9a904",
}


@pytest.mark.parametrize("name", sorted(_unchanged_charts()))
def test_an_unchanged_row_renders_as_the_unmodified_build_does(name: str) -> None:
    assert _digest(_unchanged_charts()[name]) == _HEAD_DIGESTS[name]


# ── Twin pins over the boundaries a rebinding reaches ──────────────────────
#
# Each compares the chart that took the binding after the resolution with the
# one given it first, by outcome: the SVG and the ordered warnings, or the
# refusal's type and text, so a boundary that warns or refuses does so on both.


def _wide() -> pl.DataFrame:
    n = 24
    return _DF.head(n).with_columns(
        pl.Series("wi", [(i * 7) % 11 + i for i in range(n)], dtype=pl.Int64),
        pl.Series("wf", [float((i * 5) % 13) + 0.25 * i for i in range(n)], dtype=pl.Float32),
        pl.Series("wn", [None if i % 5 == 0 else float((i * 5) % 13) + i for i in range(n)]),
        pl.Series(
            "wnan", [float("nan") if i % 5 == 0 else float((i * 5) % 13) + i for i in range(n)]
        ),
    )


def _outcome(build: Callable[[], "fm.Chart"]) -> tuple:
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result: tuple = ("svg", build().to_svg())
        except Exception as err:  # noqa: BLE001 - the refusal is part of the outcome
            result = ("refused", type(err), str(err))
    return (*result, [str(w.message) for w in caught])


def _twin(make: Callable[["fm.Chart"], "fm.Chart"], data: pl.DataFrame, later: dict) -> tuple:
    return (
        _outcome(lambda: make(fm.Chart(data)).layer(_rule()).encode(**later)),
        _outcome(lambda: make(fm.Chart(data)).encode(**later).layer(_rule())),
    )


def _smooth_of(chart: "fm.Chart") -> "fm.Chart":
    return chart.mark_smooth().encode(x="v", y="w")


@pytest.mark.parametrize("column", ["wi", "wf", "wn", "wnan"])
def test_a_rebinding_to_a_column_of_another_dtype_or_with_gaps_equals_its_door(column) -> None:
    late, door = _twin(_smooth_of, _wide(), {"y": column})
    assert late == door
    assert late[0] == "svg"


def test_a_rebinding_to_a_column_of_nulls_is_the_door_s_outcome() -> None:
    data = _wide().with_columns(pl.lit(None, dtype=pl.Float64).alias("wn"))
    late, door = _twin(_smooth_of, data, {"y": "wn"})
    assert late == door


def test_a_rebinding_over_a_zero_row_frame_is_the_door_s_outcome() -> None:
    late, door = _twin(_smooth_of, _wide().head(0), {"y": "w2"})
    assert late == door
    assert late[-1], "the zero-row door warns, so the twin must warn too"


def test_a_rebinding_inside_a_composite_leaf_equals_its_door() -> None:
    late = _smooth().layer(_rule()).encode(y="w2")
    door = _smooth().encode(y="w2").layer(_rule())
    other = _rule_chart()
    assert fm.hconcat(late, other).to_svg() == fm.hconcat(door, other).to_svg()
    assert fm.vconcat(late, other).to_svg() == fm.vconcat(door, other).to_svg()
    assert fm.LayerChart(late, other).to_svg() == fm.LayerChart(door, other).to_svg()
    assert (
        fm.hconcat(late, other).interactive().to_html()
        == fm.hconcat(door, other).interactive().to_html()
    )


def test_the_interactive_route_equals_the_door_on_a_faceted_chart() -> None:
    door = _smooth().encode(y="w2").facet(col="f").layer(_rule())
    for late in (
        _smooth().facet(col="f").layer(_rule()).encode(y="w2"),
        _smooth().layer(_rule()).facet(col="f").encode(y="w2"),
    ):
        assert late.interactive().to_html() == door.interactive().to_html()


def test_the_interactive_route_equals_the_door_over_a_transform_chain() -> None:
    door = _smooth().encode(y="w3").transform(_calc()).layer(_rule())
    late = _smooth().transform(_calc()).layer(_rule()).encode(y="w3")
    assert late.interactive().to_html() == door.interactive().to_html()


def test_the_interactive_route_equals_the_door_after_a_mark_override() -> None:
    from ferrum._overrides import _apply_mark_overrides

    resolved = _apply_mark_overrides(_smooth(), {}, entry="mark_smooth")
    door = _smooth().encode(y="w2").layer()
    assert resolved.encode(y="w2").interactive().to_html() == door.interactive().to_html()


# ── A channel the pending chart left unbound ────────────────────────────────


@pytest.mark.parametrize(
    ("make", "later"),
    [
        (lambda c: c.mark_density().encode(x="v"), {"y": "w"}),
        (lambda c: c.mark_histogram().encode(x="v"), {"y": "w"}),
        (lambda c: c.mark_qq().encode(x="v"), {"y": "w"}),
    ],
    ids=["density-y", "histogram-y", "qq-y"],
)
def test_a_binding_on_a_channel_the_pending_chart_left_unbound_equals_its_door(make, later) -> None:
    late, door = _twin(make, _wide(), later)
    assert late == door


def test_the_unbound_channel_twin_differs_when_the_rebuild_is_disabled(monkeypatch) -> None:
    def make(chart):
        return chart.mark_density().encode(x="v")

    monkeypatch.setattr(_compose, "written_inputs", lambda chart, record: {})
    late, door = _twin(make, _wide(), {"y": "w"})
    assert late != door


def test_a_binding_on_an_unbound_channel_after_a_mark_override_equals_its_door() -> None:
    from ferrum._overrides import _apply_mark_overrides

    def pending() -> "fm.Chart":
        return fm.Chart(_wide()).mark_density().encode(x="v")

    resolved = _apply_mark_overrides(pending(), {}, entry="mark_density")
    assert _outcome(lambda: resolved.encode(y="w")) == _outcome(
        lambda: pending().encode(y="w").layer()
    )


def test_a_fieldless_binding_on_an_unbound_channel_leaves_the_build_as_it_is() -> None:
    base = fm.Chart(_wide()).mark_density().encode(x="v").layer(_rule())
    assert _outcome(lambda: base.encode(y=fm.Y(title="T"))) == _outcome(lambda: base)


def _styled_flat(style: dict) -> "fm.Chart":
    from ferrum._overrides import _apply_mark_overrides

    return _apply_mark_overrides(
        fm.Chart(_wide()).mark_density().encode(x="v"), style, entry="mark_density"
    )


@pytest.mark.parametrize("style", [{"opacity": 0.3}, {"color": "red"}], ids=["opacity", "color"])
def test_a_flat_chart_rebuilt_into_layers_refuses_the_style_the_door_refuses(style) -> None:
    """The 2-D density's layers honor neither key: the style is refused, not dropped."""
    from ferrum._overrides import _apply_mark_overrides

    pending = fm.Chart(_wide()).mark_density().encode(x="v", y="w")
    with pytest.raises(Exception) as door:  # noqa: PT011 - the text is the comparison
        _apply_mark_overrides(pending, style, entry="mark_density")
    with pytest.raises(Exception) as late:  # noqa: PT011
        _styled_flat(style).encode(y="w").to_svg()
    # The flat chart holds the resolved spelling of the key (`color` is `fill`).
    spelled = str(late.value).replace("'fill'", "'color'").replace("{'fill'", "{'color'")
    assert (type(late.value), spelled) == (type(door.value), str(door.value))
    assert "unknown sub-layer" in spelled


# ═══════════════════════════════════════════════════════════════════════════
# A figure refits when a later binding names another column
# ═══════════════════════════════════════════════════════════════════════════
#
# ``regplot``, ``lmplot``, ``residplot``, ``displot`` and ``catplot`` build their
# statistics when called. A binding written afterwards (``.encode()``, before a
# ``+``, before or after a data-less ``.layer()``) on a column other than the one
# the figure read draws what the call draws with that column: the figure is called
# again with the binding as its own ``x=``/``y=``. The oracle is always the
# building door, and every comparison also holds on the interactive route.


def _fig_frame() -> pl.DataFrame:
    n = 36
    return _DF.with_columns(
        pl.Series("b", [int(((i * 7) % 11) + (i % 5) > 7) for i in range(n)]),
        pl.Series("b2", [int(((i * 5) % 13) + (i % 4) > 8) for i in range(n)]),
        pl.Series("k", [i % 4 for i in range(n)], dtype=pl.Int64),
        pl.Series("kk", [(i % 4) + 1 for i in range(n)], dtype=pl.Int64),
    )


_FIG = _fig_frame()
_FIG_OTHER = _FIG.head(12)


def _door_twin(build: Callable[..., "fm.Chart"], base: dict, later: dict, *, after=None) -> tuple:
    """(the figure that took *later* afterwards, the call that was given it), both as outcomes."""

    def late() -> "fm.Chart":
        chart = build(_FIG, **base).encode(**later)
        return after(chart) if after is not None else chart

    def door() -> "fm.Chart":
        chart = build(_FIG, **{**base, **later})
        return after(chart) if after is not None else chart

    return _outcome(late), _outcome(door)


_XY = {"x": "v", "y": "w"}
_BIN_XY = {"x": "v", "y": "b"}

#: (case id, the figure call, the keywords it was built with, the binding written later).
_FIT_REBINDINGS: list[tuple[str, Callable[..., "fm.Chart"], dict, dict]] = [
    ("regplot-y", fm.regplot, _XY, {"y": "w2"}),
    ("regplot-x", fm.regplot, _XY, {"x": "u"}),
    ("regplot-x-and-y", fm.regplot, _XY, {"x": "u", "y": "w2"}),
    ("lmplot-y", fm.lmplot, _XY, {"y": "w2"}),
    ("lmplot-x", fm.lmplot, _XY, {"x": "u"}),
    ("regplot-robust-y", partial(fm.regplot, method="robust"), _XY, {"y": "w2"}),
    ("regplot-robust-x", partial(fm.regplot, method="robust"), _XY, {"x": "u"}),
    ("regplot-glm-y", partial(fm.regplot, method="glm"), _BIN_XY, {"y": "b2"}),
    ("regplot-glm-x", partial(fm.regplot, method="glm"), _BIN_XY, {"x": "u"}),
    ("regplot-logistic-y", partial(fm.regplot, method="logistic"), _BIN_XY, {"y": "b2"}),
    ("regplot-logistic-x", partial(fm.regplot, method="logistic"), _BIN_XY, {"x": "u"}),
    ("regplot-quadratic-y", partial(fm.regplot, method="quadratic"), _XY, {"y": "w2"}),
    (
        "regplot-robust-no-scatter-y",
        partial(fm.regplot, method="robust", scatter=False),
        _XY,
        {"y": "w2"},
    ),
    (
        "regplot-glm-no-scatter-y",
        partial(fm.regplot, method="glm", scatter=False),
        _BIN_XY,
        {"y": "b2"},
    ),
    ("regplot-x-title", fm.regplot, _XY, {"x": fm.X("u", title="U")}),
    (
        "regplot-both-channels",
        fm.regplot,
        _XY,
        {"x": fm.X("u", scale=fm.LogScale()), "y": fm.Y("w", title="R")},
    ),
    ("regplot-y-title", fm.regplot, _XY, {"y": fm.Y("w2", title="T")}),
    ("regplot-hue-y", partial(fm.regplot, hue="g"), _XY, {"y": "w2"}),
    ("lmplot-hue-x", partial(fm.lmplot, hue="g"), _XY, {"x": "u"}),
    ("lmplot-col-y", partial(fm.lmplot, col="f"), _XY, {"y": "w2"}),
    ("lmplot-col-hue-y", partial(fm.lmplot, col="f", hue="g"), _XY, {"y": "w2"}),
    ("lmplot-jitter-y", partial(fm.lmplot, x_jitter=0.2), _XY, {"y": "w2"}),
    ("regplot-logx-y", partial(fm.regplot, logx=True), {"x": "kk", "y": "w"}, {"y": "w2"}),
    ("residplot-y", fm.residplot, _XY, {"y": "w2"}),
    ("residplot-x", fm.residplot, _XY, {"x": "u"}),
    ("residplot-lowess-y", partial(fm.residplot, lowess=True), _XY, {"y": "w2"}),
    ("residplot-lowess-x", partial(fm.residplot, lowess=True), _XY, {"x": "u"}),
    ("residplot-robust-y", partial(fm.residplot, robust=True), _XY, {"y": "w2"}),
    (
        "residplot-flat-y",
        partial(fm.residplot, zero_line=False, show_metrics=False),
        _XY,
        {"y": "w2"},
    ),
    (
        "residplot-flat-x",
        partial(fm.residplot, zero_line=False, show_metrics=False),
        _XY,
        {"x": "u"},
    ),
    (
        "residplot-flat-x-and-y",
        partial(fm.residplot, zero_line=False, show_metrics=False),
        _XY,
        {"x": "u", "y": "w2"},
    ),
    (
        "residplot-flat-x-title",
        partial(fm.residplot, zero_line=False, show_metrics=False),
        _XY,
        {"x": fm.X("u", title="U")},
    ),
    ("residplot-hue-y", partial(fm.residplot, color="g"), _XY, {"y": "w2"}),
]
_FIT_IDS = [case[0] for case in _FIT_REBINDINGS]


@pytest.mark.parametrize(("case", "build", "base", "later"), _FIT_REBINDINGS, ids=_FIT_IDS)
def test_a_figure_given_another_column_afterwards_equals_the_call_with_that_column(
    case, build, base, later
) -> None:
    late, door = _door_twin(build, base, later)
    assert late == door
    assert late[0] == "svg"


#: The rows whose later binding restates a column the call read (a title on the
#: same field): that restatement reaches the chart's own axis, as it does when no
#: other column is written, so the interactive twin is the call then the restatement.
_RESTATING = {"regplot-both-channels"}


@pytest.mark.parametrize(
    ("case", "build", "base", "later"),
    [case for case in _FIT_REBINDINGS if case[0] not in _RESTATING],
    ids=[case[0] for case in _FIT_REBINDINGS if case[0] not in _RESTATING],
)
def test_the_interactive_route_of_a_refitted_figure_equals_its_door(
    case, build, base, later
) -> None:
    late = build(_FIG, **base).encode(**later)
    door = build(_FIG, **{**base, **later})
    assert late.interactive().to_html() == door.interactive().to_html()


def test_a_rebinding_with_a_same_column_restatement_equals_the_call_then_the_restatement() -> None:
    """``.encode(x=<another column>, y=<same column, retitled>)``: the retitle acts as it does alone."""
    log_x = fm.X("u", scale=fm.LogScale())
    retitled = fm.Y("w", title="R")
    late = fm.regplot(_FIG, x="v", y="w").encode(x=log_x, y=retitled)
    door = fm.regplot(_FIG, x=log_x, y="w").encode(y=retitled)
    assert late.to_svg() == door.to_svg()
    assert late.interactive().to_html() == door.interactive().to_html()


@pytest.mark.parametrize(("case", "build", "base", "later"), _FIT_REBINDINGS, ids=_FIT_IDS)
def test_the_negative_control_of_a_figure_differs_when_the_rebuild_is_disabled(
    monkeypatch, case, build, base, later
) -> None:
    """With no written input read, the later binding is ignored and the twin differs from its door."""
    monkeypatch.setattr(_compose, "written_inputs", lambda chart, record: {})
    late, door = _door_twin(build, base, later)
    assert late != door


def test_a_figure_refit_draws_the_new_column_not_the_old_one() -> None:
    """Scatter, fit and band all read ``w2``: the chart is not the old one under a new title."""
    old = fm.regplot(_FIG, x="v", y="w").to_svg()
    late = fm.regplot(_FIG, x="v", y="w").encode(y="w2").to_svg()
    assert late != old
    assert late == fm.regplot(_FIG, x="v", y="w2").to_svg()


def _calc3() -> Any:
    return fm.transform_calculate("w3", "datum.w * 2 + 1")


def _agg() -> Any:
    return fm.transform_aggregate({"field": "w", "fn": "mean", "as": "w_mean"}, groupby=["k"])


@pytest.mark.parametrize(
    ("build", "base"),
    [
        (fm.regplot, _XY),
        (fm.lmplot, _XY),
        (fm.residplot, _XY),
        (partial(fm.residplot, zero_line=False, show_metrics=False), _XY),
    ],
    ids=["regplot", "lmplot", "residplot", "residplot-flat"],
)
def test_a_binding_to_a_transform_output_equals_the_call_with_the_transform(build, base) -> None:
    late = _outcome(lambda: build(_FIG, **base).transform(_calc3()).encode(y="w3"))
    door = _outcome(lambda: build(_FIG, x="v", y="w3").transform(_calc3()))
    assert late == door
    assert late[0] == "svg"


def test_the_negative_control_over_a_transform_differs_when_the_rebuild_is_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(_compose, "written_inputs", lambda chart, record: {})
    late = _outcome(lambda: fm.regplot(_FIG, **_XY).transform(_calc3()).encode(y="w3"))
    door = _outcome(lambda: fm.regplot(_FIG, x="v", y="w3").transform(_calc3()))
    assert late != door


@pytest.mark.parametrize(
    "build", [fm.regplot, fm.lmplot, fm.residplot], ids=["regplot", "lmplot", "residplot"]
)
def test_a_binding_of_both_channels_to_an_aggregate_equals_the_call_with_the_transform(
    build,
) -> None:
    late = _outcome(lambda: build(_FIG, **_XY).transform(_agg()).encode(x="k", y="w_mean"))
    door = _outcome(lambda: build(_FIG, x="k", y="w_mean").transform(_agg()))
    assert late == door
    assert late[0] == "svg"


@pytest.mark.parametrize(
    "build", [fm.regplot, fm.lmplot, fm.residplot], ids=["regplot", "lmplot", "residplot"]
)
def test_a_binding_naming_only_the_aggregate_keeps_the_dropped_column_text(build) -> None:
    """x is still the dropped ``v``: the refusal names it, and now the sentence is true."""
    late = _outcome(lambda: build(_FIG, **_XY).transform(_agg()).encode(y="w_mean"))
    door = _outcome(lambda: build(_FIG, x="v", y="w_mean").transform(_agg()))
    assert late == door
    assert late[0] == "refused"
    assert "'v'" in late[2]


@pytest.mark.parametrize(
    ("build", "base", "later"),
    [
        (fm.regplot, _XY, {"y": "nope"}),
        (fm.regplot, _XY, {"x": "nope"}),
        (fm.lmplot, _XY, {"y": "nope"}),
        (fm.residplot, _XY, {"y": "nope"}),
        (partial(fm.residplot, zero_line=False, show_metrics=False), _XY, {"x": "zz"}),
        (partial(fm.regplot, method="robust", scatter=False), _XY, {"y": "nope"}),
    ],
    ids=[
        "regplot-y",
        "regplot-x",
        "lmplot-y",
        "residplot-y",
        "residplot-flat-x",
        "robust-no-scatter",
    ],
)
def test_a_binding_on_a_missing_column_is_the_calls_refusal(build, base, later) -> None:
    late, door = _door_twin(build, base, later)
    assert late == door
    assert late[0] == "refused"
    assert next(iter(later.values())) in late[2]


@pytest.mark.parametrize(
    ("build", "base", "later"),
    [
        (fm.regplot, _XY, {"y": fm.Y("w2", type="N")}),
        (fm.regplot, _XY, {"x": fm.X("u", type="O")}),
        (fm.lmplot, _XY, {"y": fm.Y("w2", type="N")}),
        (fm.residplot, _XY, {"x": fm.X("u", type="N")}),
        (partial(fm.regplot, method="logistic"), _BIN_XY, {"y": "w2"}),
    ],
    ids=[
        "regplot-y-nominal",
        "regplot-x-ordinal",
        "lmplot-y-nominal",
        "residplot-x-nominal",
        "logistic-non-binary",
    ],
)
def test_another_column_the_call_refuses_is_refused_in_the_calls_words(build, base, later) -> None:
    late, door = _door_twin(build, base, later)
    assert late == door
    assert late[0] == "refused"


@pytest.mark.parametrize(
    ("build", "base", "later"),
    [
        (fm.regplot, _XY, {"x": fm.X("u", bin=True)}),
        (fm.regplot, _XY, {"y": fm.Y("w2", aggregate="mean")}),
        (fm.lmplot, _XY, {"x": fm.X("u", bin=True)}),
        (fm.residplot, _XY, {"y": fm.Y("w2", aggregate="mean")}),
        (partial(fm.regplot, method="robust", scatter=False), _XY, {"x": fm.X("u", bin=True)}),
    ],
    ids=["regplot-bin", "regplot-aggregate", "lmplot-bin", "residplot-aggregate", "pending-bin"],
)
def test_a_statistic_declared_on_another_column_is_refused_before_any_rebuild(
    build, base, later
) -> None:
    """The field-agnostic backstop runs first: the later binding is refused in the text
    the call door gives the same binding, in full."""
    late, door = _door_twin(build, base, later)
    assert late == door
    assert late[0] == "refused"
    assert issubclass(late[1], ValueError)
    assert "is not supported" in late[2]


@pytest.mark.parametrize("later", [{"x": "u"}, {"y": "w2"}], ids=["x", "y"])
@pytest.mark.parametrize("flat", [False, True], ids=["layered", "flat"])
def test_residplot_over_data_with_its_own_x_and_residual_columns_refits_on_another_column(
    later, flat
) -> None:
    """The data's own ``x`` and ``residual`` columns are not the figure's drawn ones: another column still refits."""
    clash = _FIG.with_columns(pl.col("u").alias("x"), pl.col("w2").alias("residual"))
    extra = {"zero_line": False, "show_metrics": False} if flat else {}
    late = _outcome(lambda: fm.residplot(clash, x="v", y="w", **extra).encode(**later))
    door = _outcome(lambda: fm.residplot(clash, **{**_XY, **later}, **extra))
    assert late == door
    assert late[0] == "svg"


# ── a rebuilt figure joins a composition, or is written again ────────────────


def test_a_refit_figure_regrouped_by_a_composition_keeps_its_written_column() -> None:
    """The figure rebuilt on ``w2`` that a later grouping builds again is still on ``w2``."""
    late = (fm.regplot(_FIG, x="v", y="w").encode(y="w2") + _rule_chart()).encode(color="g")
    door = (fm.regplot(_FIG, x="v", y="w2") + _rule_chart()).encode(color="g")
    assert _outcome(lambda: late) == _outcome(lambda: door)
    assert _outcome(lambda: late)[0] == "svg"


def test_a_refit_figure_written_again_is_read_against_the_column_it_now_draws() -> None:
    """After the refit on ``w2``, a binding naming ``w`` is another column, not the call's own."""
    layer = fm.Layer(mark="tick", encoding={"x": "v"})
    late = fm.regplot(_FIG, x="v", y="w").encode(y="w2").layer(layer).encode(y=fm.Y("w", title="T"))
    door = fm.regplot(_FIG, x="v", y=fm.Y("w", title="T")).layer(layer)
    assert _outcome(lambda: late) == _outcome(lambda: door)
    assert _outcome(lambda: late)[0] == "svg"


def test_a_refit_figure_written_a_second_time_equals_the_call_with_the_last_column() -> None:
    late = fm.regplot(_FIG, x="v", y="w").encode(y="w2").encode(y="w3").transform(_calc3())
    door = fm.regplot(_FIG, x="v", y="w3").transform(_calc3())
    assert _outcome(lambda: late) == _outcome(lambda: door)


# ── beside a layer the caller added ──────────────────────────────────────────


def _beside_doors() -> list[tuple[str, Callable[[], "fm.Chart"], Callable[[], "fm.Chart"]]]:
    layer = fm.Layer(mark="tick", encoding={"x": "v"})

    def plain(**kw: Any) -> "fm.Chart":
        return fm.regplot(_FIG, x="v", y="w", **kw)

    def refit(**kw: Any) -> "fm.Chart":
        return fm.regplot(_FIG, x="v", y="w2", **kw)

    return [
        (
            "plus-after",
            lambda: plain().encode(y="w2") + _rule_chart(),
            lambda: refit() + _rule_chart(),
        ),
        (
            "plus-before-layer",
            lambda: _rule_chart() + plain().encode(y="w2"),
            lambda: _rule_chart() + refit(),
        ),
        (
            "encode-then-layer",
            lambda: plain().encode(y="w2").layer(layer),
            lambda: refit().layer(layer),
        ),
        (
            "layer-then-encode",
            lambda: plain().layer(layer).encode(y="w2"),
            lambda: refit().layer(layer),
        ),
        (
            "no-scatter-layer-then-encode",
            lambda: plain(method="robust", scatter=False).layer(layer).encode(y="w2"),
            lambda: refit(method="robust", scatter=False).layer(layer),
        ),
        (
            "no-scatter-smooth-layer-then-encode",
            lambda: plain(scatter=False).layer(layer).encode(y="w2"),
            lambda: refit(scatter=False).layer(layer),
        ),
    ]


@pytest.mark.parametrize("door", [case[0] for case in _beside_doors()])
def test_a_refit_beside_a_layer_the_caller_added_equals_the_door(door) -> None:
    late, building = {case[0]: case[1:] for case in _beside_doors()}[door]
    assert _outcome(late) == _outcome(building)


@pytest.mark.parametrize("door", [case[0] for case in _beside_doors()])
def test_the_interactive_route_beside_a_caller_layer_equals_the_door(door) -> None:
    late, building = {case[0]: case[1:] for case in _beside_doors()}[door]
    assert late().interactive().to_html() == building().interactive().to_html()


@pytest.mark.parametrize("door", [case[0] for case in _beside_doors()])
def test_the_negative_control_beside_a_caller_layer_differs_when_the_rebuild_is_disabled(
    monkeypatch, door
) -> None:
    monkeypatch.setattr(_compose, "written_inputs", lambda chart, record: {})
    late, building = {case[0]: case[1:] for case in _beside_doors()}[door]
    assert _outcome(late) != _outcome(building)


# ── grouped and faceted, at the written column ───────────────────────────────


def test_a_rebinding_and_a_grouping_in_one_encode_equal_the_call_with_both() -> None:
    late = _outcome(lambda: fm.regplot(_FIG, x="v", y="w").encode(y="w2", color="g"))
    door = _outcome(lambda: fm.regplot(_FIG, x="v", y="w2", hue="g"))
    assert late == door


def test_a_rebinding_after_a_grouping_equals_the_call_with_both() -> None:
    late = _outcome(lambda: fm.regplot(_FIG, x="v", y="w").encode(color="g").encode(y="w2"))
    door = _outcome(lambda: fm.regplot(_FIG, x="v", y="w2", hue="g"))
    assert late == door


# ── residplot's null drop reads the rows its fit reads ───────────────────────


def _nulls() -> pl.DataFrame:
    return _FIG.with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("w")).alias("w"),
        pl.when(pl.int_range(pl.len()) % 5 == 0).then(None).otherwise(pl.col("v")).alias("v"),
    )


def test_residplot_with_the_default_dropna_reads_a_transform_output_column() -> None:
    chart = fm.residplot(_FIG, x="v", y="w3").transform(_calc3())
    assert (
        chart.to_svg()
        == fm.residplot(_FIG, x="v", y="w3", dropna=False).transform(_calc3()).to_svg()
    )


_DROPNA_FORMS: dict[str, Callable[..., "fm.Chart"]] = {
    "layered": lambda **kw: fm.residplot(_nulls(), x="v", y="w", **kw),
    "flat": lambda **kw: fm.residplot(
        _nulls(), x="v", y="w", zero_line=False, show_metrics=False, **kw
    ),
    "lowess": lambda **kw: fm.residplot(_nulls(), x="v", y="w", lowess=True, **kw),
    "robust": lambda **kw: fm.residplot(_nulls(), x="v", y="w", robust=True, **kw),
    "color": lambda **kw: fm.residplot(_nulls(), x="v", y="w", color="g", **kw),
}


@pytest.mark.parametrize("form", sorted(_DROPNA_FORMS))
def test_residplot_dropna_over_nulls_draws_as_the_unmodified_build_does(form) -> None:
    drop = _digest(_DROPNA_FORMS[form](dropna=True))
    keep = _digest(_DROPNA_FORMS[form](dropna=False))
    assert (drop, keep) == _HEAD_DROPNA[form]


def _null_levels() -> pl.DataFrame:
    """Facet level ``q`` is null in ``v`` throughout, ``r`` in ``w`` throughout, ``p`` complete."""
    level = [("p", "q", "r")[i % 3] for i in range(_FIG.height)]
    return _FIG.with_columns(
        pl.Series("lv", level),
        pl.when(pl.Series(level) == "q").then(None).otherwise(pl.col("v")).alias("v"),
        pl.when(pl.Series(level) == "r").then(None).otherwise(pl.col("w")).alias("w"),
    )


_DROPNA_EXTRA = {
    "layered": {},
    "flat": {"zero_line": False, "show_metrics": False},
    "lowess": {"lowess": True},
    "robust": {"robust": True},
    "color": {"color": "g"},
}


@pytest.mark.parametrize("form", sorted(_DROPNA_EXTRA))
def test_residplot_dropna_drops_rows_null_in_either_fit_column(form) -> None:
    """The default draws exactly the complete rows, facets included: a level null in either fit
    column draws no panel and warns nothing, as the chart over the frame without the null rows."""
    import warnings

    extra = _DROPNA_EXTRA[form]
    complete = _null_levels().drop_nulls(subset=["v", "w"])
    door = fm.residplot(complete, x="v", y="w", dropna=False, **extra).facet(col="lv")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = fm.residplot(_null_levels(), x="v", y="w", **extra).facet(col="lv")
        svg, html = chart.to_svg(), chart.interactive().to_html()
    assert [str(w.message) for w in caught] == []
    assert svg == door.to_svg()
    assert html == door.interactive().to_html()
    kept = fm.residplot(_null_levels(), x="v", y="w", dropna=False, **extra).facet(col="lv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert kept.to_svg() != svg


@pytest.mark.parametrize("form", sorted(_DROPNA_EXTRA))
def test_residplot_dropna_keeps_the_complete_rows_in_the_chart_s_own_data(form) -> None:
    chart = fm.residplot(_null_levels(), x="v", y="w", **_DROPNA_EXTRA[form])
    assert chart._data.height == _null_levels().drop_nulls(subset=["v", "w"]).height


@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("column", ["zz", "d"])
def test_a_residplot_column_the_rows_lack_or_cannot_fit_is_the_fits_own_refusal(
    dropna, column
) -> None:
    """The statistic refuses it, in one text for either ``dropna``, never an expression step's."""
    from decimal import Decimal

    frame = _FIG.with_columns(pl.Series("d", [Decimal("1.5")] * _FIG.height))
    late = _outcome(lambda: fm.residplot(frame, x="v", y="w", dropna=dropna).encode(x=column))
    door = _outcome(lambda: fm.residplot(frame, x=column, y="w", dropna=dropna))
    assert late == door
    assert late[0] == "refused"
    assert late[1] is ValueError
    assert late[2].startswith("transform failed: ValueError: stat_smooth: column '")
    assert "data_filter" not in late[2]


# ── displot overlays and mark= ───────────────────────────────────────────────

_DISPLOT_FORMS: list[tuple[str, Callable[..., "fm.Chart"]]] = [
    ("hist-kde", partial(fm.displot, kde=True)),
    ("hist-rug", partial(fm.displot, rug=True)),
    ("hist-kde-rug", partial(fm.displot, kde=True, rug=True)),
    ("kde-rug", partial(fm.displot, kind="kde", rug=True)),
    ("hist-mark", partial(fm.displot, kind="hist", mark={"opacity": 0.5})),
    ("hist-kde-hue", partial(fm.displot, kde=True, hue="g")),
]


@pytest.mark.parametrize(("case", "build"), _DISPLOT_FORMS, ids=[c[0] for c in _DISPLOT_FORMS])
def test_a_displot_overlay_given_another_column_afterwards_equals_the_call(case, build) -> None:
    late = _outcome(lambda: build(_FIG, x="v").encode(x="u"))
    door = _outcome(lambda: build(_FIG, x="u"))
    assert late == door
    assert late[0] == "svg"


@pytest.mark.parametrize(("case", "build"), _DISPLOT_FORMS, ids=[c[0] for c in _DISPLOT_FORMS])
def test_the_interactive_route_of_a_displot_refit_equals_its_door(case, build) -> None:
    late = build(_FIG, x="v").encode(x="u")
    assert late.interactive().to_html() == build(_FIG, x="u").interactive().to_html()


@pytest.mark.parametrize(("case", "build"), _DISPLOT_FORMS, ids=[c[0] for c in _DISPLOT_FORMS])
def test_the_displot_negative_control_differs_when_the_rebuild_is_disabled(
    monkeypatch, case, build
) -> None:
    monkeypatch.setattr(_compose, "written_inputs", lambda chart, record: {})
    assert _outcome(lambda: build(_FIG, x="v").encode(x="u")) != _outcome(
        lambda: build(_FIG, x="u")
    )


def test_a_displot_overlay_bound_to_a_statistics_output_column_is_refused_for_the_missing_column() -> (
    None
):
    chart = fm.displot(_FIG, x="v", kde=True).encode(x="bin_start")
    door = _outcome(lambda: fm.displot(_FIG, x="bin_start", kde=True))
    assert _outcome(lambda: chart) == door
    assert door[0] == "refused"
    assert "bin_start" in door[2]


def test_a_displot_overlay_with_a_transform_chain_reads_the_transformed_rows() -> None:
    late = _outcome(lambda: fm.displot(_FIG, x="v", kde=True).transform(_calc3()).encode(x="w3"))
    door = _outcome(lambda: fm.displot(_FIG, x="w3", kde=True).transform(_calc3()))
    assert late == door
    assert late[0] == "svg"


def test_a_displot_overlay_beside_a_caller_layer_equals_the_door() -> None:
    layer = fm.Layer(mark="tick", encoding={"x": "v"})
    late = fm.displot(_FIG, x="v", kde=True).layer(layer).encode(x="u")
    door = fm.displot(_FIG, x="u", kde=True).layer(layer)
    assert late.to_svg() == door.to_svg()


# ── catplot point and bar ────────────────────────────────────────────────────

_CATPLOT_FORMS: list[tuple[str, Callable[..., "fm.Chart"], dict, dict, dict]] = [
    ("point-y", partial(fm.catplot, kind="point"), {"x": "g", "y": "w"}, {"y": "w2"}, {"y": "w2"}),
    ("bar-y", partial(fm.catplot, kind="bar"), {"x": "g", "y": "w"}, {"y": "w2"}, {"y": "w2"}),
    ("point-x", partial(fm.catplot, kind="point"), {"x": "g", "y": "w"}, {"x": "f"}, {"x": "f"}),
    (
        "point-native",
        partial(fm.catplot, kind="point", native_scale=True),
        {"x": "k", "y": "w"},
        {"y": "w2"},
        {"y": "w2"},
    ),
    (
        "point-native-x",
        partial(fm.catplot, kind="point", native_scale=True),
        {"x": "k", "y": "w"},
        {"x": "kk"},
        {"x": "kk"},
    ),
    (
        "point-hue",
        partial(fm.catplot, kind="point", hue="f"),
        {"x": "g", "y": "w"},
        {"y": "w2"},
        {"y": "w2"},
    ),
    (
        "bar-dodge",
        partial(fm.catplot, kind="bar", hue="f", dodge=True),
        {"x": "g", "y": "w"},
        {"y": "w2"},
        {"y": "w2"},
    ),
    # Layer space, as shipped: on a horizontal catplot the later binding's y is the value.
    (
        "point-horizontal",
        partial(fm.catplot, kind="point", orient="h"),
        {"y": "g", "x": "w"},
        {"y": "w2"},
        {"x": "w2"},
    ),
    (
        "bar-horizontal",
        partial(fm.catplot, kind="bar", orient="h"),
        {"y": "g", "x": "w"},
        {"y": "w2"},
        {"x": "w2"},
    ),
    (
        "point-horizontal-native",
        partial(fm.catplot, kind="point", orient="h", native_scale=True),
        {"y": "k", "x": "w"},
        {"y": "w2"},
        {"x": "w2"},
    ),
]


@pytest.mark.parametrize(
    ("case", "build", "base", "later", "door_kw"),
    _CATPLOT_FORMS,
    ids=[c[0] for c in _CATPLOT_FORMS],
)
def test_a_catplot_given_another_column_afterwards_equals_the_call(
    case, build, base, later, door_kw
) -> None:
    late = _outcome(lambda: build(_FIG, **base).encode(**later))
    door = _outcome(lambda: build(_FIG, **{**base, **door_kw}))
    assert late == door
    assert late[0] == "svg"


@pytest.mark.parametrize(
    ("case", "build", "base", "later", "door_kw"),
    _CATPLOT_FORMS,
    ids=[c[0] for c in _CATPLOT_FORMS],
)
def test_the_interactive_route_of_a_catplot_refit_equals_its_door(
    case, build, base, later, door_kw
) -> None:
    late = build(_FIG, **base).encode(**later)
    assert (
        late.interactive().to_html() == build(_FIG, **{**base, **door_kw}).interactive().to_html()
    )


@pytest.mark.parametrize(
    ("case", "build", "base", "later", "door_kw"),
    _CATPLOT_FORMS,
    ids=[c[0] for c in _CATPLOT_FORMS],
)
def test_the_catplot_negative_control_differs_when_the_rebuild_is_disabled(
    monkeypatch, case, build, base, later, door_kw
) -> None:
    monkeypatch.setattr(_compose, "written_inputs", lambda chart, record: {})
    late = _outcome(lambda: build(_FIG, **base).encode(**later))
    assert late != _outcome(lambda: build(_FIG, **{**base, **door_kw}))


def test_a_horizontal_catplots_later_encode_is_read_in_layer_space() -> None:
    """The returned chart's ``y`` is the value of a horizontal catplot: writing ``y`` rebinds the value."""
    chart = fm.catplot(_FIG, y="g", x="w", kind="point", orient="h").encode(y="w2")
    assert chart.to_svg() == fm.catplot(_FIG, y="g", x="w2", kind="point", orient="h").to_svg()


def test_a_catplot_bound_to_a_missing_column_is_the_calls_refusal() -> None:
    late = _outcome(lambda: fm.catplot(_FIG, x="g", y="w", kind="point").encode(y="nope"))
    door = _outcome(lambda: fm.catplot(_FIG, x="g", y="nope", kind="point"))
    assert late == door
    assert late[0] == "refused"


# ── same-column restatements keep today's output ─────────────────────────────


def _restated() -> dict[str, "fm.Chart"]:
    return {
        "displot-kde-title": fm.displot(_FIG, x="v", kde=True).encode(x=fm.X("v", title="T")),
        "displot-kde-format": fm.displot(_FIG, x="v", kde=True).encode(x=fm.X("v", format=".1f")),
        "displot-kde-scale": fm.displot(_FIG, x="v", kde=True).encode(
            x=fm.X("v", scale=fm.SymlogScale())
        ),
        "displot-rug-title": fm.displot(_FIG, x="v", rug=True).encode(x=fm.X("v", title="T")),
        "catplot-point-title": fm.catplot(_FIG, x="g", y="w", kind="point").encode(
            y=fm.Y("w", title="T")
        ),
        "catplot-point-format": fm.catplot(_FIG, x="g", y="w", kind="point").encode(
            y=fm.Y("w", format=".1f")
        ),
        "catplot-bar-scale": fm.catplot(_FIG, x="g", y="w", kind="bar").encode(
            y=fm.Y("w", scale=fm.SymlogScale())
        ),
        "catplot-horizontal-value-title": fm.catplot(
            _FIG, y="g", x="w", kind="point", orient="h"
        ).encode(y=fm.Y("w", title="T")),
        "catplot-horizontal-category-title": fm.catplot(
            _FIG, y="g", x="w", kind="bar", orient="h"
        ).encode(x=fm.X("g", title="T")),
        "catplot-native-title": fm.catplot(
            _FIG, x="k", y="w", kind="point", native_scale=True
        ).encode(x=fm.X("k", title="T")),
        "lmplot-hue-title": fm.lmplot(_FIG, x="v", y="w", hue="g").encode(x=fm.X("v", title="T")),
        "residplot-flat-x-title": fm.residplot(
            _FIG, x="v", y="w", zero_line=False, show_metrics=False
        ).encode(x=fm.X("v", title="T")),
        "regplot-title": fm.regplot(_FIG, x="v", y="w").encode(y=fm.Y("w", title="T")),
        "regplot-scale": fm.regplot(_FIG, x="v", y="w").encode(x=fm.X("v", scale=fm.SymlogScale())),
        "regplot-format": fm.regplot(_FIG, x="v", y="w").encode(y=fm.Y("w", format=".1f")),
        "lmplot-axis": fm.lmplot(_FIG, x="v", y="w").encode(
            x=fm.X("v", axis=fm.Axis(label_angle=45))
        ),
        "residplot-title": fm.residplot(_FIG, x="v", y="w").encode(x=fm.X("v", title="T")),
        "residplot-residual": fm.residplot(_FIG, x="v", y="w").encode(y=fm.Y("residual")),
        "residplot-flat-y-title": fm.residplot(
            _FIG, x="v", y="w", zero_line=False, show_metrics=False
        ).encode(y=fm.Y(title="R")),
        "regplot-nosc-pending-other": fm.regplot(_FIG, x="v", y="w", scatter=False).encode(y="w2"),
        "regplot-plus-composition": (fm.regplot(_FIG, x="v", y="w") + _rule_chart()).encode(y="w2"),
        "regplot-layer-with-data": fm.regplot(_FIG, x="v", y="w")
        .layer(fm.Layer(mark="tick", data=_FIG_OTHER, encoding={"x": "v"}))
        .encode(y="w2"),
    }


@pytest.mark.parametrize("name", sorted(_restated()))
def test_a_restatement_or_a_composition_binding_renders_as_the_unmodified_build_does(name) -> None:
    assert _digest(_restated()[name]) == _HEAD_RESTATED[name]


def _int_x_forms() -> dict[str, "fm.Chart"]:
    ints = _FIG.with_columns(pl.col("k").alias("ki"))
    return {
        "regplot-jitter-int": fm.regplot(ints, x="k", y="w", x_jitter=0.2),
        "lmplot-jitter-int-col": fm.lmplot(ints, x="k", y="w", x_jitter=0.2, col="f"),
        "catplot-native-int-point": fm.catplot(ints, x="k", y="w", kind="point", native_scale=True),
        "catplot-native-int-dodge": fm.catplot(
            ints, x="k", y="w", kind="bar", native_scale=True, hue="f", dodge=True
        ),
        "catplot-native-int-strip": fm.catplot(
            ints, x="k", y="w", kind="strip", native_scale=True, jitter=True
        ),
        "catplot-native-int-strip-dodge": fm.catplot(
            ints, x="k", y="w", kind="strip", native_scale=True, hue="f", dodge=True
        ),
    }


@pytest.mark.parametrize("name", sorted(_int_x_forms()))
def test_a_jittered_or_native_scale_integer_column_draws_as_the_unmodified_build_does(name) -> None:
    assert _digest(_int_x_forms()[name]) == _HEAD_INT_X[name]


def test_a_jittered_regplot_rebound_to_an_integer_column_equals_the_call() -> None:
    late = _outcome(lambda: fm.regplot(_FIG, x="v", y="w", x_jitter=0.2).encode(x="k"))
    door = _outcome(lambda: fm.regplot(_FIG, x="k", y="w", x_jitter=0.2))
    assert late == door
    assert late[0] == "svg"


def test_a_native_scale_catplot_rebound_to_an_integer_column_equals_the_call() -> None:
    late = _outcome(
        lambda: fm.catplot(_FIG, x="kk", y="w", kind="point", native_scale=True).encode(x="k")
    )
    door = _outcome(lambda: fm.catplot(_FIG, x="k", y="w", kind="point", native_scale=True))
    assert late == door
    assert late[0] == "svg"


# ── the written column through every door ────────────────────────────────────


def test_a_refit_figure_inside_a_composite_leaf_equals_its_door() -> None:
    late = fm.regplot(_FIG, x="v", y="w").encode(y="w2")
    door = fm.regplot(_FIG, x="v", y="w2")
    other = _rule_chart()
    assert fm.hconcat(late, other).to_svg() == fm.hconcat(door, other).to_svg()
    assert fm.vconcat(late, other).to_svg() == fm.vconcat(door, other).to_svg()
    assert (
        fm.hconcat(late, other).interactive().to_html()
        == fm.hconcat(door, other).interactive().to_html()
    )


def test_a_refit_figure_in_a_layer_chart_equals_its_door() -> None:
    late = fm.regplot(_FIG, x="v", y="w").encode(y="w2")
    door = fm.regplot(_FIG, x="v", y="w2")
    other = _rule_chart()
    assert fm.LayerChart(late, other).to_svg() == fm.LayerChart(door, other).to_svg()


def test_a_refit_figure_renders_the_same_twice_and_in_json() -> None:
    chart = fm.regplot(_FIG, x="v", y="w").encode(y="w2")
    first = chart.to_svg()
    assert chart.to_svg() == first
    assert chart.to_json() == fm.regplot(_FIG, x="v", y="w2").to_json()


def test_a_refit_residplot_serializes_as_its_door_does() -> None:
    for later in ({"y": "w2"}, {"x": "u"}):
        late = fm.residplot(_FIG, x="v", y="w").encode(**later)
        door = fm.residplot(_FIG, **{**_XY, **later})
        assert late.to_json() == door.to_json()


# ── a data-less layer attached to residplot with the default dropna ───────────


def _null_rows() -> pl.DataFrame:
    n = 24
    return pl.DataFrame(
        {
            "v": [float(i) for i in range(n)],
            "w": [float((i * 7) % 11) + 0.5 * i for i in range(n)],
        }
    ).with_columns(
        pl.when(pl.int_range(pl.len()) % 5 == 0).then(None).otherwise(pl.col("w")).alias("w"),
        pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("v")).alias("v"),
    )


_RESIDPLOT_FORMS: dict[str, dict] = {
    "layered": {},
    "flat": {"zero_line": False, "show_metrics": False},
    "lowess": {"lowess": True},
}


def _attached(form: str, door: str) -> "fm.Chart":
    rows = _null_rows()
    residual = fm.residplot(rows, x="v", y="w", **_RESIDPLOT_FORMS[form])
    points = fm.Chart(rows).mark_point().encode(x="v", y="w")
    layer = fm.Layer(mark="point", encoding={"x": "v", "y": "w"})
    return {
        "plus-right": lambda: residual + points,
        "plus-left": lambda: points + residual,
        "layer": lambda: residual.layer(layer),
        "layer-tick": lambda: residual.layer(fm.Layer(mark="tick", encoding={"x": "v"})),
        "layer-chart": lambda: fm.LayerChart(residual, points),
    }[door]()


#: Measured on the build before residplot's null drop moved into its statistics run.
_HEAD_ATTACHED: dict[tuple[str, str], str] = {
    ("layered", "plus-right"): "d8f69a2b42b3",
    ("layered", "plus-left"): "b19c1767a00a",
    ("layered", "layer"): "d8f69a2b42b3",
    ("layered", "layer-chart"): "ae3386f37d61",
    ("flat", "plus-right"): "06ab00f30a7f",
    ("flat", "plus-left"): "364a6b341751",
    ("flat", "layer"): "06ab00f30a7f",
    ("flat", "layer-chart"): "bf127a671fac",
    ("lowess", "plus-right"): "633a55d37bd8",
    ("lowess", "plus-left"): "423834637d3a",
    ("lowess", "layer"): "633a55d37bd8",
    ("lowess", "layer-chart"): "415303bbea54",
    # An x-only layer reads the chart's rows: the complete rows, as before.
    ("layered", "layer-tick"): "877b5b22258f",
    ("flat", "layer-tick"): "0e1f72045a68",
    ("lowess", "layer-tick"): "f20c3ce16b74",
}


@pytest.mark.parametrize(("form", "door"), sorted(_HEAD_ATTACHED))
def test_a_layer_attached_to_a_residplot_over_nulls_draws_and_warns_as_before(form, door) -> None:
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = _attached(form, door).to_svg()
    assert hashlib.sha256(svg.encode()).hexdigest()[:12] == _HEAD_ATTACHED[(form, door)]
    assert [str(w.message) for w in caught] == []


# ── a later binding restating the call's column ──────────────────────────────


@pytest.mark.parametrize(
    "build",
    [fm.regplot, fm.lmplot, fm.residplot],
    ids=["regplot", "lmplot", "residplot"],
)
def test_a_binding_back_to_the_keywords_column_after_an_encode_kwarg_draws_the_keywords_column(
    build,
) -> None:
    """``encode={'y': 'w2'}`` is the call's own y; a later ``.encode(y='w')`` is another column than that."""
    late = _outcome(lambda: build(_FIG, x="v", y="w", encode={"y": "w2"}).encode(y="w"))
    door = _outcome(lambda: build(_FIG, x="v", y="w"))
    assert late == door
    assert late[0] == "svg"
    assert late != _outcome(lambda: build(_FIG, x="v", y="w", encode={"y": "w2"}))


# ── a refit whose new column is null on other rows than the old one ──────────


def _other_null_rows() -> pl.DataFrame:
    """``w`` and ``w2`` are null on different rows; group ``z`` and facet level ``q`` are only on some of them."""
    n = 24
    i = pl.int_range(pl.len())
    return (
        pl.DataFrame({"v": [float(k) for k in range(n)]})
        .with_columns(
            (pl.col("v") * 1.5 + (pl.col("v") % 5)).alias("w"),
            (pl.col("v") * 0.7 - (pl.col("v") % 4)).alias("w2"),
            pl.Series("g", ["a" if k % 2 else "b" for k in range(n)]),
            pl.Series("f", ["p" if k < 12 else "q" for k in range(n)]),
        )
        .with_columns(
            pl.when(i % 3 == 0).then(None).otherwise(pl.col("w")).alias("w"),
            pl.when(i % 7 == 1).then(None).otherwise(pl.col("w2")).alias("w2"),
            pl.when(i % 7 == 1).then(pl.lit("z")).otherwise(pl.col("g")).alias("g"),
        )
    )


_NULL_ROWS = _other_null_rows()
_NULL_ROWS_FORMS: dict[str, dict] = {
    "layered": {},
    "flat": {"zero_line": False, "show_metrics": False},
    "lowess": {"lowess": True},
    "color": {"color": "g"},
}


def _route_outcome(build: Callable[[], Any], route: str) -> tuple:
    """(the output of *route*, the ordered warning messages), or the refusal's type and text."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            chart = build()
            out = chart.interactive().to_html() if route == "html" else chart.to_svg()
        except Exception as err:  # noqa: BLE001 - the refusal is part of the outcome
            out = ("refused", type(err), str(err))
    return out, [str(w.message) for w in caught]


def _null_rows_doors(kw: dict, old: str, new: str) -> dict[str, tuple[Callable, Callable]]:
    tick = fm.Chart(_NULL_ROWS).mark_tick().encode(x="v")
    tick_layer = fm.Layer(mark="tick", encoding={"x": "v"})

    def late() -> "fm.Chart":
        return fm.residplot(_NULL_ROWS, x="v", y=old, **kw).encode(y=new)

    def door() -> "fm.Chart":
        return fm.residplot(_NULL_ROWS, x="v", y=new, **kw)

    return {
        "plain": (late, door),
        "facet-then-encode": (
            lambda: fm.residplot(_NULL_ROWS, x="v", y=old, **kw).facet(col="f").encode(y=new),
            lambda: door().facet(col="f"),
        ),
        "encode-then-facet": (lambda: late().facet(col="f"), lambda: door().facet(col="f")),
        "layer-then-encode": (
            lambda: fm.residplot(_NULL_ROWS, x="v", y=old, **kw).layer(tick_layer).encode(y=new),
            lambda: door().layer(tick_layer),
        ),
        "encode-then-layer": (lambda: late().layer(tick_layer), lambda: door().layer(tick_layer)),
        "plus-after": (lambda: late() + tick, lambda: door() + tick),
        "plus-before": (lambda: tick + late(), lambda: tick + door()),
        "layer-chart": (
            lambda: fm.LayerChart(late(), tick),
            lambda: fm.LayerChart(door(), tick),
        ),
        "hconcat": (lambda: fm.hconcat(late(), tick), lambda: fm.hconcat(door(), tick)),
        "regrouped-plus": (
            lambda: (late() + tick).encode(detail="f"),
            lambda: (door() + tick).encode(detail="f"),
        ),
        "regrouped-layer": (
            lambda: late().layer(tick_layer).encode(detail="f"),
            lambda: door().layer(tick_layer).encode(detail="f"),
        ),
    }


_NULL_ROWS_DOORS = sorted(_null_rows_doors({}, "w", "w2"))


@pytest.mark.parametrize("route", ["svg", "html"])
@pytest.mark.parametrize("door", _NULL_ROWS_DOORS)
@pytest.mark.parametrize("direction", [("w", "w2"), ("w2", "w")], ids=["w-to-w2", "w2-to-w"])
@pytest.mark.parametrize("form", sorted(_NULL_ROWS_FORMS))
def test_a_residplot_refit_over_other_null_rows_equals_its_call_on_every_door(
    form, direction, door, route
) -> None:
    """The refit draws the rows complete in the written column, as the call does: output and ordered warnings."""
    old, new = direction
    late, call = _null_rows_doors(_NULL_ROWS_FORMS[form], old, new)[door]
    assert _route_outcome(late, route) == _route_outcome(call, route)
    assert not isinstance(_route_outcome(late, route)[0], tuple)


@pytest.mark.parametrize("route", ["svg", "html"])
@pytest.mark.parametrize("door", ["transform", "transform-facet", "transform-plus"])
@pytest.mark.parametrize("form", sorted(_NULL_ROWS_FORMS))
def test_a_residplot_refit_to_a_transform_output_over_null_rows_equals_its_call(
    form, door, route
) -> None:
    """A ``.transform()`` output column with the default dropna: rows are dropped on ``v`` only."""
    kw = _NULL_ROWS_FORMS[form]
    step = fm.transform_calculate("w3", "datum.v * 2 + 1")
    tick = fm.Chart(_NULL_ROWS).mark_tick().encode(x="v")

    def late() -> "fm.Chart":
        return fm.residplot(_NULL_ROWS, x="v", y="w", **kw).transform(step).encode(y="w3")

    def call() -> "fm.Chart":
        return fm.residplot(_NULL_ROWS, x="v", y="w3", **kw).transform(step)

    wrap = {
        "transform": lambda chart: chart,
        "transform-facet": lambda chart: chart.facet(col="f"),
        "transform-plus": lambda chart: chart + tick,
    }[door]
    assert _route_outcome(lambda: wrap(late()), route) == _route_outcome(
        lambda: wrap(call()), route
    )
    assert not isinstance(_route_outcome(lambda: wrap(late()), route)[0], tuple)


# ── boundaries a refit crosses ───────────────────────────────────────────────


def _boundary_frame() -> pl.DataFrame:
    n = _FIG.height
    return _FIG.with_columns(
        pl.col("w2").cast(pl.Float32).alias("w32"),
        pl.when(pl.int_range(pl.len()) % 5 == 0)
        .then(float("nan"))
        .otherwise(pl.col("w2"))
        .alias("wnan"),
        pl.when(pl.int_range(pl.len()) == 3)
        .then(float("inf"))
        .otherwise(pl.col("w2"))
        .alias("winf"),
        pl.when(pl.int_range(pl.len()) % 4 == 0).then(None).otherwise(pl.col("w2")).alias("wnull"),
        pl.Series("wi", [i % 7 + i for i in range(n)], dtype=pl.Int64),
    )


#: Every figure the refit reaches, as (id, build, base keywords, the later binding's
#: channel in the chart's space, the call keyword that channel is on the call door, the
#: column a clean rebinding names). A horizontal catplot's later ``y`` is the value, the
#: call's ``x`` ("Layer space, as shipped").
_BOUNDARY_FIGURES: list[tuple[str, Callable[..., "fm.Chart"], dict, str, str, str]] = [
    ("regplot", fm.regplot, _XY, "y", "y", "w2"),
    ("lmplot", fm.lmplot, _XY, "y", "y", "w2"),
    ("residplot", fm.residplot, _XY, "y", "y", "w2"),
    (
        "residplot-flat",
        partial(fm.residplot, zero_line=False, show_metrics=False),
        _XY,
        "y",
        "y",
        "w2",
    ),
    (
        "regplot-robust-no-scatter",
        partial(fm.regplot, method="robust", scatter=False),
        _XY,
        "y",
        "y",
        "w2",
    ),
    *((f"displot-{case}", build, {"x": "v"}, "x", "x", "u") for case, build in _DISPLOT_FORMS),
    *(
        (f"catplot-{case}", build, base, next(iter(later)), next(iter(door_kw)), "w2")
        for case, build, base, later, door_kw in _CATPLOT_FORMS
        if next(iter(later)) == "y"
    ),
]
_BOUNDARY_IDS = [figure[0] for figure in _BOUNDARY_FIGURES]
_BOUNDARY_COLUMNS = ["w32", "wnan", "winf", "wnull", "wi"]


def _boundary_pair(figure: tuple, frame: pl.DataFrame, column: str) -> tuple[Callable, Callable]:
    """(the figure given *column* afterwards, the call given it), over *frame*."""
    _, build, base, later, door, _ = figure
    return (
        lambda: build(frame, **base).encode(**{later: column}),
        lambda: build(frame, **{**base, door: column}),
    )


@pytest.mark.parametrize("route", ["svg", "html"])
@pytest.mark.parametrize("zero_rows", [False, True], ids=["rows", "zero-rows"])
@pytest.mark.parametrize("column", _BOUNDARY_COLUMNS)
@pytest.mark.parametrize("figure", _BOUNDARY_FIGURES, ids=_BOUNDARY_IDS)
def test_a_refit_over_a_column_of_every_dtype_and_gap_equals_its_call(
    figure, column, zero_rows, route
) -> None:
    frame = _boundary_frame()
    if zero_rows:
        frame = frame.head(0)
    late, door = _boundary_pair(figure, frame, column)
    assert _route_outcome(late, route) == _route_outcome(door, route)


@pytest.mark.parametrize("route", ["svg", "html"])
@pytest.mark.parametrize("column", _BOUNDARY_COLUMNS)
@pytest.mark.parametrize("figure", _BOUNDARY_FIGURES, ids=_BOUNDARY_IDS)
def test_the_boundary_negative_control_differs_when_the_rebuild_is_disabled(
    monkeypatch, figure, column, route
) -> None:
    """Over rows, each boundary cell differs from its call once the written input is not read."""
    monkeypatch.setattr(_compose, "written_inputs", lambda chart, record: {})
    late, door = _boundary_pair(figure, _boundary_frame(), column)
    assert _route_outcome(late, route) != _route_outcome(door, route)


@pytest.mark.parametrize("route", ["svg", "html"])
@pytest.mark.parametrize("order", ["facet-then-encode", "encode-then-facet"])
@pytest.mark.parametrize("figure", _BOUNDARY_FIGURES, ids=_BOUNDARY_IDS)
def test_a_refit_on_a_faceted_figure_equals_its_call_faceted(figure, order, route) -> None:
    _, build, base, later, door_key, column = figure
    if order == "facet-then-encode":
        late = lambda: build(_FIG, **base).facet(col="f").encode(**{later: column})  # noqa: E731
    else:
        late = lambda: build(_FIG, **base).encode(**{later: column}).facet(col="f")  # noqa: E731
    door = lambda: build(_FIG, **{**base, door_key: column}).facet(col="f")  # noqa: E731
    assert _route_outcome(late, route) == _route_outcome(door, route)
    assert not isinstance(_route_outcome(late, route)[0], tuple)


_HEAD_DROPNA: dict[str, tuple[str, str]] = {
    "color": ("a1c2584d9d20d48a", "554363d34714dd3c"),
    "flat": ("1a8ba64de0c7c71b", "1a8ba64de0c7c71b"),
    "layered": ("9dc6988dcbe5980f", "9dc6988dcbe5980f"),
    "lowess": ("7538ad70e6aac3e5", "7538ad70e6aac3e5"),
    "robust": ("9dc6988dcbe5980f", "9dc6988dcbe5980f"),
}
_HEAD_RESTATED: dict[str, str] = {
    "residplot-flat-x-title": "dc522a5433b23bdf",
    "lmplot-hue-title": "ffb15f5142147d56",
    "catplot-native-title": "70704588a57b7f91",
    "catplot-horizontal-value-title": "d1c1844c443b51c1",
    "catplot-horizontal-category-title": "d88dc8da1874c077",
    "catplot-bar-scale": "66fed91f5c1e7253",
    "catplot-point-format": "60b57b32e21669c4",
    "catplot-point-title": "16e561f222a2a516",
    "displot-kde-format": "9fa8a2d1ae4fcd85",
    "displot-kde-scale": "9fa8a2d1ae4fcd85",
    "displot-kde-title": "62c176ccfca2117a",
    "displot-rug-title": "b34e8679d6c441a7",
    "lmplot-axis": "20ae31144b29b9c1",
    "regplot-format": "f51456f07322e84a",
    "regplot-layer-with-data": "bc2aad290d24ab2d",
    "regplot-nosc-pending-other": "dad5f1976ce78200",
    "regplot-plus-composition": "459b588fa7b038e9",
    "regplot-scale": "b754e8e379d5ec76",
    "regplot-title": "e14b3b5ebcbec640",
    "residplot-flat-y-title": "29f5e69a67ab6cca",
    "residplot-residual": "95f286bf863d4c04",
    "residplot-title": "751f6d4db8a1508d",
}
_HEAD_INT_X: dict[str, str] = {
    "catplot-native-int-dodge": "449648aec5d1f2eb",
    "catplot-native-int-point": "70a4f1eecc2cfd53",
    "catplot-native-int-strip": "efa5b23fab3fbe09",
    "catplot-native-int-strip-dodge": "9c883e046cc1881c",
    "lmplot-jitter-int-col": "2be5bff2d5e0c1c9",
    "regplot-jitter-int": "640ffa406c4d6328",
}
