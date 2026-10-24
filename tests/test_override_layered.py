"""``Chart.override(<channel>_scale_*=...)`` reaches every layer on every composition path.

A composite mark (``mark_boxplot``, ``mark_violin``, ``mark_errorbar``,
``mark_boxen``, the catplot box/violin kinds), a figure helper built on layers
(``regplot``, ``lmplot``), ``+`` and ``Chart.layer(Layer(...))`` all serialize layers that
redeclare the channel with their own copy of its scale. The override must be
merged into each such layer's scale exactly as it is merged into the chart-level
encoding, so the rendered axis is the one the channel's own ``scale=`` gives.

The matrix is composition path x override leaf. Each cell asserts the
overridden chart's rendered y tick *pixel ladder* (value -> pixel position,
not just the label text) equals that of the same chart built with ``scale=``
on the channel itself (the reference), and that the reference's ladder
differs from the un-overridden chart's, so a cell can never pass vacuously.
Comparing pixel positions rather than label text matters for leaf ``type``:
a Log scale's ``breaks_log`` ticks can fall back to the same linear
``nice_ticks`` label text the base scale already draws when the data spans
less than one decade, so two leaf-independent checks alone keep a wrong
reference from passing: the axis covers the data on every cell, and
``reverse`` draws the base ticks in reverse label order.
The leaf population is the override registry's own (``_scale_leaves()``),
partitioned into the leaves tested here and the leaves excluded with a stated
reason (``_EXCLUDED``); the partition is pinned. A gate-refused combination
refuses with identical text on every path.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest

import ferrum as fm
from ferrum import _override_apply as oa
from ferrum.layer import Layer
from tests._svg_extents import axis_tick_labels, value_axis_ladder

# Uneven values so `nice=True` moves the axis ends onto round numbers.
_DF = pl.DataFrame(
    {
        "g": ["a", "a", "b", "b", "a", "b"],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "v": [1.3, 2.1, 3.4, 4.2, 5.5, 6.7],
    }
)


def _chart() -> fm.Chart:
    return fm.Chart(_DF)


# Each path builds the chart with the value channel `y` it is handed: the bare
# field for the overridden/base charts, `fm.Y("v", scale=...)` for the reference.
_PATHS: dict[str, Callable[[object], fm.Chart]] = {
    "flat_bar": lambda y: _chart().mark_bar().encode(x="g", y=y),
    "flat_point": lambda y: _chart().mark_point().encode(x="x", y=y),
    "plus": lambda y: (
        _chart().mark_point().encode(x="x", y=y) + _chart().mark_line().encode(x="x", y=y)
    ),
    "chart_layer": lambda y: (
        _chart()
        .mark_point()
        .encode(x="x", y=y)
        .layer(Layer(mark="line", encoding={"x": fm.X("x"), "y": _as_channel(y)}))
    ),
    "mark_boxplot": lambda y: _chart().mark_boxplot().encode(x="g", y=y),
    "mark_violin": lambda y: _chart().mark_violin().encode(x="g", y=y),
    "mark_errorbar": lambda y: _chart().mark_errorbar().encode(x="g", y=y),
    "mark_boxen": lambda y: _chart().mark_boxen().encode(x="g", y=y),
    "catplot_box": lambda y: fm.catplot(_DF, x="g", y=y, kind="box"),
    "catplot_violin": lambda y: fm.catplot(_DF, x="g", y=y, kind="violin"),
    "regplot": lambda y: fm.regplot(_DF, x="x", y=y),
    "lmplot": lambda y: fm.lmplot(_DF, x="x", y=y),
    # Only the line layer declares y, as a bare string: the chart has none.
    "layer_only_channel": lambda y: (
        _chart().mark_rule().encode(x="x").layer(Layer(mark="line", encoding={"x": "x", "y": y}))
    ),
    # The rule layer declares no y: it inherits the chart-level encoding.
    "layer_partial_channel": lambda y: (
        _chart()
        .mark_point()
        .encode(x="x", y=y)
        .layer(Layer(mark="rule", encoding={"x": fm.X("x")}))
    ),
}


def _as_channel(y: object) -> object:
    return fm.Y(y) if isinstance(y, str) else y


# Registry leaves this matrix does not render, each with the reason. Together
# with `_LEAF_VALUES` they partition `_scale_leaves()` (pinned below), so a new
# registry leaf fails the pin until it is classified.
_EXCLUDED: dict[str, str] = {
    "zero": "flat_bar already anchors at zero, so its reference cannot differ from its base",
    "padding": "moves pixels, not tick labels",
    "range": "moves pixels, not tick labels",
    "clamp": "clamps marks to a stated domain; with no domain the ticks do not move",
    "domainParam": "needs a registered fm.param(); a static render reads its initial value",
    "domain_param": "snake alias of domainParam",
    "base": "log-only: refused on the linear axis every path draws",
    "constant": "symlog-only: refused on the linear axis every path draws",
    "exponent": "pow-only: refused on the linear axis every path draws",
    "align": "band-only: the value axis is continuous",
    "paddingInner": "band-only: the value axis is continuous",
    "padding_inner": "band-only: the value axis is continuous (used by _REFUSED)",
    "paddingOuter": "band-only: the value axis is continuous",
    "padding_outer": "band-only: the value axis is continuous",
    "scheme": "color-scale-only",
    "stops": "color-scale-only",
    "domainMid": "diverging-color-only",
    "domain_mid": "diverging-color-only",
    "bins": "bin-ordinal-color-only",
    "bin": "bin-ordinal-color-only",
}

# A value per rendered leaf that moves a linear value axis visibly.
_LEAF_VALUES: dict[str, object] = {
    "domain": [0, 100],
    "type": "log",
    "reverse": True,
    "nice": True,
}

_LEAVES: dict[str, object] = {
    leaf: _LEAF_VALUES[leaf] for leaf in sorted(oa._scale_leaves()) if leaf not in _EXCLUDED
}

# Refused by the wire gate: `log` accepts no `paddingInner`.
_REFUSED = {"type": "log", "padding_inner": 0.3}


def _override_kwargs(leaves: dict[str, object]) -> dict[str, object]:
    return {f"y_scale_{leaf}": value for leaf, value in leaves.items()}


def _own_scale(leaves: dict[str, object]) -> dict[str, object]:
    """The channel's own ``scale=`` dict equivalent to *leaves* as an override."""
    return {oa._SCALE_LEAF_WIRE_ALIASES.get(k, k): v for k, v in leaves.items()}


def test_leaf_partition_covers_the_registry() -> None:
    registry = oa._scale_leaves()
    assert set(_LEAF_VALUES).isdisjoint(_EXCLUDED)
    assert set(_LEAF_VALUES) | set(_EXCLUDED) == registry
    assert set(_REFUSED) <= registry


def _assert_axis_covers_the_data(ticks: list[str]) -> None:
    """The tick ladder, extended by one step at each end, contains the data.

    Ticks sit inside the scale's domain, so the outermost tick may fall short
    of the data by less than a step; a truncated axis falls short by more.
    """
    values = sorted(float(t) for t in ticks)
    lo_step = values[1] - values[0]
    hi_step = values[-1] - values[-2]
    assert values[0] - lo_step <= _DF["v"].min()
    assert values[-1] + hi_step >= _DF["v"].max()


@pytest.mark.parametrize("leaf", sorted(_LEAVES))
@pytest.mark.parametrize("path", sorted(_PATHS))
def test_override_renders_like_the_channels_own_scale(path: str, leaf: str) -> None:
    build = _PATHS[path]
    leaves = {leaf: _LEAVES[leaf]}
    base_svg = build("v").to_svg()
    reference_svg = build(fm.Y("v", scale=_own_scale(leaves))).to_svg()
    overridden_svg = build("v").override(**_override_kwargs(leaves)).to_svg()

    base = axis_tick_labels(base_svg, axis="y")
    reference = axis_tick_labels(reference_svg, axis="y")
    overridden = axis_tick_labels(overridden_svg, axis="y")

    # The discriminator is where values LAND, not just which labels are
    # drawn: for leaf "type", "v" spans 1.3..6.7 (under one decade), where
    # a Log scale's breaks_log ticks fall back to the same linear
    # nice_ticks TEXT the base scale already uses, so comparing labels
    # alone would pass even if the override never reached the scale type.
    # A log axis places a value at log(v/lo)/log(hi/lo) of the height, a
    # linear one at (v-lo)/(hi-lo): the pixel ladder tells them apart even
    # when the two scales happen to label the same round numbers.
    base_ladder = value_axis_ladder(base_svg, axis="y")
    reference_ladder = value_axis_ladder(reference_svg, axis="y")
    overridden_ladder = value_axis_ladder(overridden_svg, axis="y")

    assert reference_ladder != base_ladder, (
        f"{path}/{leaf}: the own-scale reference must move the axis"
    )
    assert overridden_ladder == reference_ladder
    _assert_axis_covers_the_data(overridden)
    if leaf == "reverse":
        assert overridden == base[::-1]


def _refusal(chart: fm.Chart) -> str:
    with pytest.raises(ValueError) as info:
        chart.to_svg()
    return str(info.value)


def test_refused_override_refuses_identically_on_every_path() -> None:
    expected = _refusal(_PATHS["flat_point"]("v").override(**_override_kwargs(_REFUSED)))
    assert "unknown key 'paddingInner' for type 'log'" in expected
    texts = {
        path: _refusal(build("v").override(**_override_kwargs(_REFUSED)))
        for path, build in _PATHS.items()
    }
    assert texts == dict.fromkeys(_PATHS, expected)
