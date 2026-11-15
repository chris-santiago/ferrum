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
