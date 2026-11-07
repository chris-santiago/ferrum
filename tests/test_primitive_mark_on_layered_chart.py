"""A primitive mark call on a chart drawn from layers is refused at the call.

A chart holds either one mark of its own (``_mark``) or a list of layers
(``_layers``), never both meaningfully at once: a layered chart's own
``_mark`` only mirrors layer 0. A primitive mark call (``mark_point()``,
``mark_bar()``, ...) on any chart holding at least one layer refuses rather
than being silently dropped, naming the chart's own layers -- each by its
own ``name`` where it has one (the same key a figure's
``mark={'<layer>': {...}}`` targets), or by the mark it draws otherwise (a
layer with no mark of its own draws as a point) -- and two remedies: set
the mark before layering, or restyle a layer by name.

Three routes stay exactly as they were, because none of them reaches
``Chart._set_mark`` at all: a composite mark call (``.mark_histogram()``,
...) on a chart already holding layers; a diagnostic restyle of a
composite mark built by a visualizer
(``learning_curve_chart(...).mark_learning_curve(...)``, pinned in
``tests/diagnostics/test_selection.py``); and a fitted-layer mark
(``.mark_smooth()``, ``.mark_function()``). This guard changes nothing
about what any of the three already did to a layered chart's own layers.
A flat chart with no layers is unaffected too, including a figure whose
overlays were never requested, which still replaces via the
missing-positional-channel refusal when the replacement leaves an axis
unbound.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import pytest

import ferrum as fm
from ferrum._layer import _Layer, layer_mark

_DF = pl.DataFrame({"v": [1.0, 2.0, 2.5, 3.0, 4.0, 5.5], "w": [2.0, 1.0, 3.0, 5.0, 4.0, 6.0]})
_EMPTY_DF = pl.DataFrame({"v": pl.Series([], dtype=pl.Float64), "w": pl.Series([], dtype=pl.Float64)})
_ONE_ROW_DF = pl.DataFrame({"v": [1.0], "w": [2.0]})

#: Every public method that is itself one of ``Chart._set_mark``'s 14 call
#: sites (``mark_circle``/``mark_square`` forward to ``mark_point`` rather
#: than calling ``_set_mark`` themselves, so they are covered through it,
#: not listed again here; the 14th site, ``mark_line(point=True)``'s own
#: internal ``self._set_mark("point")``, can never be reached on a layered
#: receiver, since the same call's leading ``_set_mark("line", ...)`` refuses
#: first).
_PRIMITIVE_MARK_NAMES = (
    "point",
    "line",
    "bar",
    "area",
    "rule",
    "text",
    "tick",
    "rect",
    "segment",
    "arc",
    "image",
    "geoshape",
    "label",
)


def _layer_label(layer: _Layer) -> str:
    """How the refusal names *layer*: its own name, or the mark it draws."""
    return layer.name or layer_mark(layer.mark)


def _layered_refusal_message(entry: str, layers: Sequence[_Layer]) -> str:
    """The expected text for *entry* on a chart drawn from *layers*."""
    names = ", ".join(_layer_label(layer) for layer in layers)
    count = len(layers)
    noun = "layer" if count == 1 else "layers"
    return (
        f"{entry}() cannot replace the marks of a chart drawn from {count} {noun} "
        f"({names}): a mark call replaces a chart's one mark, and a layered chart "
        "has none of its own. Set the mark on a chart before layering it "
        f"(fm.Chart(...).{entry}() + ...), or restyle a layer by name "
        "(a figure's mark={'<layer>': {...}})."
    )


def _point(data: pl.DataFrame = _DF) -> fm.Chart:
    return fm.Chart(data).encode(x="v", y="w").mark_point()


def _line(data: pl.DataFrame = _DF) -> fm.Chart:
    return fm.Chart(data).encode(x="v", y="w").mark_line()


def _markless_host(data: pl.DataFrame = _DF) -> fm.Chart:
    """A chart with no mark of its own: ``.encode()`` with no ``mark_*()`` call."""
    return fm.Chart(data).encode(x="v", y="w")


def _assert_layered_refusal(
    chart: fm.Chart, call: str, expected_names: Sequence[str], **kwargs
) -> None:
    """*chart* renders a real drawing, then ``mark_<call>(**kwargs)`` refuses
    with the layered-chart message naming *expected_names* -- proving the
    receiver has the defect's shape (a renderable layered chart whose
    primitive mark call was a silent no-op before this refusal existed),
    not merely a receiver that already failed for an unrelated reason."""
    assert chart.to_svg()
    assert tuple(_layer_label(layer) for layer in chart._layers) == tuple(expected_names)

    with pytest.raises(ValueError) as excinfo:
        getattr(chart, f"mark_{call}")(**kwargs)

    assert str(excinfo.value) == _layered_refusal_message(f"mark_{call}", chart._layers)


@pytest.mark.parametrize(
    "build,call,expected_names",
    [
        pytest.param(
            lambda: fm.displot(_DF, x="v", kind="hist", kde=True),
            "point",
            ("histogram", "kde"),
            id="displot-hist-kde",
        ),
        pytest.param(
            lambda: fm.displot(_DF, x="v", kind="ecdf", rug=True),
            "point",
            ("kde", "rug"),
            id="displot-ecdf-rug",
        ),
        pytest.param(
            lambda: fm.regplot(_DF, x="v", y="w"),
            "bar",
            ("scatter", "ribbon", "line", "metrics"),
            id="regplot",
        ),
        pytest.param(
            lambda: fm.lmplot(_DF, x="v", y="w"),
            "bar",
            ("scatter", "ribbon", "line", "metrics"),
            id="lmplot",
        ),
        pytest.param(
            lambda: fm.residplot(_DF, x="v", y="w"),
            "bar",
            ("point", "reference", "metrics"),
            id="residplot",
        ),
        pytest.param(lambda: _point() + _line(), "bar", ("point", "line"), id="plus"),
        pytest.param(
            lambda: _point() + fm.SecondaryY("w", mark="bar"),
            "line",
            ("point", "bar"),
            id="secondary-y",
        ),
        pytest.param(
            lambda: _point().layer(fm.Layer(mark="line")),
            "bar",
            ("point", "line"),
            id="layer-call",
        ),
        pytest.param(
            lambda: fm.Chart(_DF).encode(x="v", y="w").mark_line(point=True),
            "bar",
            ("line", "point"),
            id="mark-line-point-overlay",
        ),
        pytest.param(
            lambda: _markless_host().layer(fm.Layer(mark="line")),
            "point",
            ("point", "line"),
            id="markless-host-layer-call",
        ),
        pytest.param(
            lambda: _markless_host() + _line(),
            "point",
            ("point", "line"),
            id="markless-host-plus",
        ),
        *[
            pytest.param(
                lambda: _point() + _line(), name, ("point", "line"), id=f"every-method-{name}"
            )
            for name in _PRIMITIVE_MARK_NAMES
        ],
    ],
)
def test_a_primitive_mark_on_a_layered_chart_is_refused_at_the_call(build, call, expected_names) -> None:
    """Every door that produces a layered chart, and every public primitive
    mark method, refuses that method's own call, naming the chart's own
    layers -- each by its name where it has one, else by the mark it draws
    -- in the message's count and list. Each row calls the method it is
    found with, not an arbitrary stand-in."""
    chart = build()

    _assert_layered_refusal(chart, call, expected_names)


def test_a_markless_host_with_neither_name_nor_mark_is_named_by_the_mark_it_draws() -> None:
    """A chart with no ``mark_*()`` call of its own, used as a layering host
    (``.layer(...)`` or ``+``), contributes a layer with neither ``name`` nor
    ``mark`` set -- a plausible Vega-Lite-style idiom, not merely a
    constructed edge case. It renders as a point (the wire default for an
    unmarked layer), and the refusal names it ``"point"``, never an invented
    placeholder."""
    host = _markless_host()
    assert host._layers is None

    via_layer = host.layer(fm.Layer(mark="line"))
    assert via_layer._layers[0].name is None
    assert via_layer._layers[0].mark is None

    with pytest.raises(ValueError) as excinfo:
        via_layer.mark_point()
    assert str(excinfo.value) == _layered_refusal_message("mark_point", via_layer._layers)
    assert "(point, line)" in str(excinfo.value)


def test_the_refusal_fires_before_a_chained_encode_runs() -> None:
    """``.mark_point().encode(y="w")`` on a layered chart never reaches
    ``.encode()``: the refusal raises at the ``mark_point()`` call itself."""
    chart = fm.displot(_DF, x="v", kind="ecdf", rug=True)

    with pytest.raises(ValueError) as excinfo:
        chart.mark_point().encode(y="w")

    assert str(excinfo.value) == _layered_refusal_message("mark_point", chart._layers)


@pytest.mark.parametrize("frame", [_EMPTY_DF, _ONE_ROW_DF], ids=["zero-row", "one-row"])
def test_a_layered_chart_over_zero_or_one_row_data_is_refused_without_reading_a_row(frame) -> None:
    """The refusal is purely structural (the chart already holds layers),
    so it fires the same way regardless of how many rows the data has --
    including a chart that still renders (an empty or near-empty panel)
    rather than one that fails for an unrelated reason."""
    chart = _point(frame) + _line(frame)

    _assert_layered_refusal(chart, "bar", ("point", "line"))


def test_an_unknown_kwarg_on_a_layered_receiver_is_refused_as_the_layering_not_the_kwarg() -> None:
    """``_refuse_fitted_mark_after_pending_composite`` runs before a fitted
    mark's own kwargs are read; the layered-chart refusal runs the same way,
    ahead of ``MarkBase``'s own unknown-kwarg validation -- an unrecognized
    kwarg on a layered receiver never reaches that validator at all."""
    chart = _point() + _line()

    _assert_layered_refusal(chart, "bar", ("point", "line"), bogus=1)


def test_a_wrong_type_kwarg_on_a_layered_receiver_is_refused_as_the_layering_not_the_type() -> None:
    """A kwarg of the wrong type (``opacity="x"``, which ``MarkBase`` would
    otherwise reject with a type error) never reaches that check either."""
    chart = _point() + _line()

    _assert_layered_refusal(chart, "bar", ("point", "line"), opacity="x")


@pytest.mark.parametrize("position", [None, "dodge"], ids=["none", "dodge"])
def test_position_on_a_layered_receiver_is_refused_as_the_layering_even_when_unset(position) -> None:
    """``position=`` is popped from kwargs after the layering check runs, so
    it never reaches eligibility validation either way -- including
    ``position=None``, which is unset and would otherwise be a no-op."""
    chart = _point() + _line()

    _assert_layered_refusal(chart, "bar", ("point", "line"), position=position)


def test_a_chart_layered_with_no_extra_layers_is_also_refused() -> None:
    """``.layer()`` with no arguments still turns a flat chart into a
    one-layer chart (``_take_layers`` promotes the host's own mark into
    ``_layers`` and empties ``_mark``), a case no listed door reaches but
    the same structural rule covers: a chart drawn from exactly one layer
    still has no mark of its own for a later primitive call to replace, so
    the message correctly says "1 layer", singular."""
    chart = _point().layer()
    assert len(chart._layers) == 1

    _assert_layered_refusal(chart, "bar", ("point",))


def test_mark_line_point_overlay_itself_renders_unaffected() -> None:
    """The negative control: ``mark_line(point=True)`` builds the line and
    point marks by calling ``_set_mark`` twice on the ORIGINAL (unlayered)
    receiver, then composes them with ``+`` -- neither call ever sees the
    layers the composition produces, so it renders exactly as the
    equivalent hand-built composition does."""
    built_in = fm.Chart(_DF).encode(x="v", y="w").mark_line(point=True)
    hand_built = fm.Chart(_DF).encode(x="v", y="w").mark_line() + fm.Chart(_DF).encode(
        x="v", y="w"
    ).mark_point()

    assert built_in.to_svg() == hand_built.to_svg()


def test_a_flat_chart_still_replaces_its_mark_with_a_later_primitive_call() -> None:
    """A chart with no layers (``_layers is None``) is unaffected: a later
    primitive mark call keeps replacing the earlier one, as it always did,
    drawing byte-identically to the same mark set directly."""
    replaced = fm.Chart(_DF).encode(x="v", y="v").mark_point().mark_bar()
    direct = fm.Chart(_DF).encode(x="v", y="v").mark_bar()

    assert replaced._mark == "bar"
    assert replaced._layers is None
    assert replaced.to_svg() == direct.to_svg()


@pytest.mark.parametrize("kind", ["hist", "kde", "ecdf"])
def test_a_displot_with_no_overlay_still_replaces_and_refuses_on_an_unbound_axis(kind: str) -> None:
    """A figure call that never requested an overlay (no ``kde=``/``rug=``)
    never becomes layered, so a chained ``mark_point()`` still replaces its
    statistic -- and, with no ``y=`` surviving that replacement, refuses
    with the missing-positional-channel text, not this module's
    layered-chart refusal."""
    chart = fm.displot(_DF, x="v", kind=kind).mark_point()
    assert chart._layers is None

    with pytest.raises(ValueError) as excinfo:
        chart.to_svg()

    assert str(excinfo.value) == (
        "mark_point needs both positional channels, but nothing binds 'y'. Encode "
        "'y' on the chart to draw it, or draw from 'x' alone with mark_tick() or "
        "mark_rule()."
    )


def test_a_composite_mark_call_on_a_layered_chart_never_reaches_this_refusal() -> None:
    """``.mark_density()`` is a composite call (``Chart._set_composite_mark``),
    not one of ``_set_mark``'s 14 primitive sites: called on an already-layered
    chart, it still replaces every layer under the ordinary mark-replacement
    rule, drawing exactly what the same call draws on a chart that was never
    layered at all -- never the refusal a primitive call would raise, and
    never a render that somehow keeps the layers it replaced."""
    layered = (_point() + _line()).mark_density().encode(x="v")
    never_layered = fm.Chart(_DF).encode(x="v", y="w").mark_point().mark_density().encode(x="v")

    assert layered.to_svg() == never_layered.to_svg()


def test_mark_smooth_on_a_layered_chart_keeps_every_layer_unaffected() -> None:
    """A fitted-layer mark (``mark_smooth``/``mark_function``) on an
    already-layered chart is the one designed exception: it keeps every
    layer already drawn and adds the fit as one more, through the same
    ``+`` composition an already-layered chart already took -- so its
    render is byte-identical to composing the fit in by hand."""
    before = _point() + _line()

    via_smooth = before.mark_smooth(method="linear")
    via_plus = before + fm.Chart(_DF).encode(x="v", y="w").mark_smooth(method="linear")

    assert via_smooth.to_svg() == via_plus.to_svg()
    assert len(via_smooth._resolve_pending()._layers) == 3
