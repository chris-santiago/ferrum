"""Wave-7 line and size capabilities hold the same way under every layering verb.

The verbs are ``chart + chart``, ``fm.layer``, independent-y ``LayerChart``,
``Chart.layer()`` on a marked host, on a markless host, and called once per
layer, and the ``hconcat`` composite. The features are a line's conditional
``size`` scope, auxiliary-channel promotion, the composite ``size=``
refusal, and the tapered width, including the refusal that names the
public entry. Each verb places two members, given as plain descriptions,
in both orders.

The one rule that decides a conditional's reach: a conditional declared on
a chart reaches the layers that chart's own field encodings reach. An
operand of ``chart + chart`` (and so every ``fm.layer``/``LayerChart``
member) keeps it on its own layers. A host of ``Chart.layer()`` shares it
with every layer it is given, the way its field ``size`` is inherited, and
a ``Layer``'s own encoding keeps it on that layer.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding import _auxiliary_scale_channels

_DF = pl.DataFrame(
    {
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 1.0, 0.0, 1.0],
        "v": [1.0, 2.0, 3.0, 4.0],
        "w": [4.0, 3.0, 2.0, 1.0],
    }
)
_SEL = fm.selection_point(name="s", clear="false")


@dataclass(frozen=True)
class _Member:
    """One layer the verbs place: a primitive mark plus its own encoding and style."""

    mark: str
    encoding: dict = field(default_factory=dict)
    mark_kwargs: dict = field(default_factory=dict)

    def chart(self) -> fm.Chart:
        return getattr(fm.Chart(_DF), f"mark_{self.mark}")(**self.mark_kwargs).encode(
            x="x", y="y", **self.encoding
        )

    def layer(self) -> fm.Layer:
        return fm.Layer(
            mark=self.mark,
            encoding={"x": fm.X("x"), "y": fm.Y("y"), **self.encoding},
            mark_kwargs=dict(self.mark_kwargs) or None,
        )


def _markless() -> fm.Chart:
    return fm.Chart(_DF).encode(x="x", y="y")


@dataclass(frozen=True)
class _Verb:
    """A public way to put two members in one chart.

    ``host_shares`` is True when the first member is a host whose own
    declarations reach every layer the verb gives it (``Chart.layer()``
    on a marked chart).
    """

    build: Callable[[_Member, _Member], Any]
    host_shares: bool = False


_VERBS: dict[str, _Verb] = {
    "plus": _Verb(lambda a, b: a.chart() + b.chart()),
    "fm_layer": _Verb(lambda a, b: fm.layer(a.chart(), b.chart())),
    "layerchart_independent_y": _Verb(
        lambda a, b: fm.LayerChart(a.chart(), b.chart(), resolve={"y": "independent"})
    ),
    "chart_layer_marked_host": _Verb(lambda a, b: a.chart().layer(b.layer()), host_shares=True),
    "chart_layer_markless": _Verb(lambda a, b: _markless().layer(a.layer(), b.layer())),
    "chart_layer_one_per_call": _Verb(lambda a, b: _markless().layer(a.layer()).layer(b.layer())),
    "hconcat": _Verb(lambda a, b: fm.hconcat(a.chart(), b.chart())),
}

_ORDERS = ("ab", "ba")


def _place(verb: str, order: str, a: _Member, b: _Member) -> Any:
    first, second = (a, b) if order == "ab" else (b, a)
    return _VERBS[verb].build(first, second)


def _line_batches(chart: Any) -> list[dict]:
    scene = json.loads(chart.interactive().scene_json)
    return [batch for p in scene["panels"] for batch in p["marks"] if batch["kind"] == "line"]


def _line_widths(chart: Any) -> list[list[float] | None]:
    """Each line node's per-vertex ``widths`` in the interactive scene, ``None`` when untapered."""
    return [node.get("widths") for batch in _line_batches(chart) for node in batch["nodes"]]


def _cond(otherwise: float) -> Any:
    return fm.when(_SEL).then(100).otherwise(otherwise)


def _with_selection(chart: Any) -> Any:
    return chart.add_selection(_SEL) if isinstance(chart, fm.Chart) else chart


# ---------------------------------------------------------------------------
# A line's conditional size scope.
# ---------------------------------------------------------------------------

_PLAIN_LINE = _Member("line")


def _conditional_point(otherwise: float) -> _Member:
    return _Member("point", {"size": _cond(otherwise)})


def _restyles_line(verb: str, order: str) -> bool:
    """A point's conditional reaches the line only when the point is a sharing host."""
    return _VERBS[verb].host_shares and order == "ab"


def test_the_conditional_scope_matrix_has_both_outcomes() -> None:
    outcomes = {_restyles_line(v, o) for v in _VERBS for o in _ORDERS}
    assert outcomes == {True, False}


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("verb", sorted(_VERBS))
def test_a_points_conditional_size_restyles_a_line_only_where_it_reaches(
    verb: str, order: str
) -> None:
    chart = _with_selection(_place(verb, order, _conditional_point(20), _PLAIN_LINE))
    batches = _line_batches(chart)
    assert len(batches) == 1
    assert ("vertex_rows" in batches[0]) is _restyles_line(verb, order)
    assert re.findall(r'<polyline[^>]*\bstroke-width="([\d.]+)"', chart.to_svg()) == ["1.5"]


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("verb", sorted(_VERBS))
def test_a_points_zero_size_branch_refuses_on_a_line_only_where_it_reaches(
    verb: str, order: str
) -> None:
    chart = _with_selection(_place(verb, order, _conditional_point(0), _PLAIN_LINE))
    if not _restyles_line(verb, order):
        chart.to_svg()
        chart.interactive()
        return
    with pytest.raises(ValueError) as exc:
        chart.to_svg()
    assert str(exc.value) == (
        "layer 1: mark_line: size=fm.when(...) on selection 's' has a branch of size 0, "
        "which gives no positive line width. Every width along a line must be positive; "
        "give every branch a size above 0."
    )


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("verb", sorted(_VERBS))
def test_a_lines_own_conditional_size_restyles_it_under_every_verb(verb: str, order: str) -> None:
    line = _Member("line", {"size": _cond(20)})
    chart = _with_selection(_place(verb, order, _Member("point"), line))
    assert [b.get("vertex_rows") for b in _line_batches(chart)] == [[[0, 1, 2, 3]]]


def _chart_level_routes(size: dict) -> dict[str, Callable[[], fm.Chart]]:
    """Every way to attach layers to a host; *size* is declared on the host first."""
    point = _Member("point")
    return {
        "layer_once": lambda: _markless().encode(**size).layer(point.layer(), _PLAIN_LINE.layer()),
        "layer_per_call": lambda: (
            _markless().encode(**size).layer(point.layer()).layer(_PLAIN_LINE.layer())
        ),
        "marked_host_layer": lambda: point.chart().encode(**size).layer(_PLAIN_LINE.layer()),
        "line_host_layer": lambda: _PLAIN_LINE.chart().encode(**size).layer(point.layer()),
        "secondary_y": lambda: point.chart().encode(**size) + fm.SecondaryY("v", mark="line"),
        "plus": lambda: point.chart().encode(**size) + _PLAIN_LINE.chart(),
    }


def _assert_reaches_every_layer(chart: fm.Chart) -> None:
    wire = json.loads(chart.to_spec().to_json())
    assert [c["layers"] for c in wire["conditionals"]] == [list(range(len(wire["layers"])))]
    batches = _line_batches(chart)
    assert batches and all("vertex_rows" in b for b in batches)


@pytest.mark.parametrize("route", sorted(set(_chart_level_routes({})) - {"plus"}))
def test_a_hosts_conditional_reaches_every_layer_attached_after_it(route: str) -> None:
    _assert_reaches_every_layer(_with_selection(_chart_level_routes({"size": _cond(20)})[route]()))


@pytest.mark.parametrize("route", sorted(_chart_level_routes({})))
def test_a_conditional_declared_after_the_layers_reaches_every_layer(route: str) -> None:
    _assert_reaches_every_layer(
        _with_selection(_chart_level_routes({})[route]().encode(size=_cond(20)))
    )


def test_an_operands_conditional_stays_on_its_own_layers_after_a_later_merge() -> None:
    inner = (_Member("point").chart() + _Member("point").chart()).encode(size=_cond(20))
    chart = _with_selection(inner + _PLAIN_LINE.chart())
    wire = json.loads(chart.to_spec().to_json())
    assert [c["layers"] for c in wire["conditionals"]] == [[0, 1]]
    assert ["vertex_rows" in b for b in _line_batches(chart)] == [False]


def test_a_conditional_is_not_a_constructor_field_of_the_public_spec() -> None:
    with pytest.raises(TypeError):
        fm.ConditionalSpec(selection_name="s", if_selected=1, if_not=2, layers=(0,))


# ---------------------------------------------------------------------------
# Auxiliary-channel promotion.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("verb", sorted(_VERBS))
def test_a_lines_own_size_field_tapers_it_under_every_verb(verb: str, order: str) -> None:
    line = _Member("line", {"size": fm.Size("v")})
    chart = _place(verb, order, _Member("point"), line)
    widths = _line_widths(chart)
    assert len(widths) == 1
    assert widths[0] is not None and len(widths[0]) == 4
    assert widths[0] == sorted(widths[0]) and widths[0][0] < widths[0][-1]


def _channel_class(channel: str) -> type:
    return getattr(fm, "".join(part.title() for part in channel.split("_")))


_PROMOTION_VERBS: dict[str, Callable[[fm.Chart, _Member], fm.Chart]] = {
    "plus": lambda host, m: host + m.chart(),
    "chart_layer": lambda host, m: host.layer(m.layer()),
}


@pytest.mark.parametrize("verb", sorted(_PROMOTION_VERBS))
@pytest.mark.parametrize("channel", sorted(_auxiliary_scale_channels()))
def test_a_later_layers_auxiliary_binding_renders_as_if_declared_chart_level(
    channel: str, verb: str
) -> None:
    binding = {channel: _channel_class(channel)("v")}
    member = _Member("point", binding)
    combine = _PROMOTION_VERBS[verb]
    promoted = combine(_PLAIN_LINE.chart(), member).to_svg()
    chart_level = combine(_PLAIN_LINE.chart().encode(**binding), member).to_svg()
    unbound = combine(_PLAIN_LINE.chart(), _Member("point")).to_svg()
    assert promoted == chart_level
    assert promoted != unbound


# Two binders arriving in ONE merge onto a host that binds nothing, so the
# promotion has to choose between them.
_TWO_BINDER_VERBS: dict[str, Callable[[_Member, _Member], fm.Chart]] = {
    "plus": lambda a, b: _PLAIN_LINE.chart() + (a.chart() + b.chart()),
    "chart_layer": lambda a, b: _PLAIN_LINE.chart().layer(a.layer(), b.layer()),
}


@pytest.mark.parametrize("verb", sorted(_TWO_BINDER_VERBS))
@pytest.mark.parametrize("fields", [("v", "w"), ("w", "v")])
def test_the_first_layer_binding_a_channel_decides_its_scale(
    verb: str, fields: tuple[str, str]
) -> None:
    first, second = (_Member("point", {"size": fm.Size(f)}) for f in fields)
    wire = json.loads(_TWO_BINDER_VERBS[verb](first, second).to_spec().to_json())
    assert wire["encoding"]["size"]["field"] == fields[0]


# ---------------------------------------------------------------------------
# A composite's constant size= refuses where nothing it draws reads it.
# ---------------------------------------------------------------------------

_DENSITY_REFUSAL = "mark_density(size=...) has no effect on mark_density; received 5."


@dataclass(frozen=True)
class _Composite:
    """A composite mark: how to build it, and the x column its own output publishes."""

    build: Callable[..., fm.Chart]
    output_x: str


_DENSITY = _Composite(lambda **kw: fm.Chart(_DF).mark_density(**kw).encode(x="x"), "value")
# With a CI band the fit is a ribbon plus a line, a layered composite.
_SMOOTH = _Composite(
    lambda **kw: fm.Chart(_DF).mark_smooth(ci=0.95, **kw).encode(x="x", y="y"), "x"
)

_COMPOSITE_ROUTES: dict[str, Callable[[_Composite, dict], Any]] = {
    "flat": lambda c, kw: c.build(**kw),
    "plus": lambda c, kw: _Member("point").chart() + c.build(**kw),
    "fm_layer": lambda c, kw: fm.layer(_Member("point").chart(), c.build(**kw)),
    # The attached rule reads the composite's own output.
    "chart_layer_host": lambda c, kw: c.build(**kw).layer(
        fm.Layer(mark="rule", encoding={"x": fm.X(c.output_x)})
    ),
    "hconcat": lambda c, kw: fm.hconcat(_Member("point").chart(), c.build(**kw)),
}


@pytest.mark.parametrize("route", sorted(_COMPOSITE_ROUTES))
def test_a_composite_that_draws_no_size_refuses_it_under_every_route(route: str) -> None:
    _COMPOSITE_ROUTES[route](_DENSITY, {}).to_svg()
    with pytest.raises(ValueError) as exc:
        _COMPOSITE_ROUTES[route](_DENSITY, {"size": 5}).to_svg()
    assert str(exc.value) == _DENSITY_REFUSAL


def test_a_composite_size_refusal_names_the_stamped_entry() -> None:
    """A figure function forwarding size= to a drawn-nothing composite refuses in its own voice.

    `_with_entry` is how a figure function stamps its own name over the
    primitives/composites it builds (see `test_relplot_regplot.py`); here it
    stands in for a figure function that forwards the caller's `size=` to
    `mark_density`, which draws neither point nor line. The refusal must
    name the stamped entry, not the internal `mark_density` it routed
    through.
    """
    stamped = fm.Chart(_DF).mark_density(size=5).encode(x="x")._with_entry("kdeplot")
    with pytest.raises(ValueError) as exc:
        stamped.to_svg()
    assert str(exc.value) == "kdeplot(size=...) has no effect on kdeplot; received 5."


@pytest.mark.parametrize("route", sorted(_COMPOSITE_ROUTES))
def test_a_composite_that_draws_size_honors_it_under_every_route(route: str) -> None:
    chart = _COMPOSITE_ROUTES[route](_SMOOTH, {"size": 40})
    plain = _COMPOSITE_ROUTES[route](_SMOOTH, {})
    widths = sorted(n["style"]["width"] for b in _line_batches(chart) for n in b["nodes"])
    plain_widths = sorted(n["style"]["width"] for b in _line_batches(plain) for n in b["nodes"])
    fit_width = round(2 * math.sqrt(40 / math.pi), 3)
    # Exactly the fit line moved, to a dot of area 40's diameter.
    assert len(widths) == len(plain_widths)
    assert [round(w, 3) for w in widths if w not in plain_widths] == [fit_width]


# ---------------------------------------------------------------------------
# A width refusal on a composite's line names the composite.
# ---------------------------------------------------------------------------

_SMOOTH_CONFLICT = (
    "mark_smooth: size and stroke_width both set the line's width (size=40, "
    "stroke_width=3); bind only one. size= sets a dot-diameter width that can taper "
    "along the line; stroke_width= sets a width in pixels."
)


# Without a CI band the fit is one unlayered line, a flat composite.
_FLAT_SMOOTH = _Composite(lambda **kw: fm.Chart(_DF).mark_smooth(**kw).encode(x="x", y="y"), "x")
_SMOOTHS = {"layered": _SMOOTH, "flat": _FLAT_SMOOTH}


def _line_prefix(chart: fm.Chart) -> str:
    """The ``layer {i}: `` prefix a refusal about *chart*'s one line carries, ``""`` when flat."""
    layers = json.loads(chart.to_spec().to_json()).get("layers")
    if layers is None:
        return ""
    (index,) = [i for i, ly in enumerate(layers) if ly["mark"] == "line"]
    return f"layer {index}: "


def _refusal_prefix(chart: Any, output: str) -> str:
    """Where a refusal about the second member's line is voiced, per render route."""
    leaf = "failed to render composite leaf leaf #1: "
    if isinstance(chart, fm.LayerChart):
        if output == "to_svg":
            # The static overlay renders each member as its own leaf; a flat
            # member is named by its place in the overlay, here layer 1.
            return leaf + (_line_prefix(chart._charts[1]) or "layer 1: ")
        return _line_prefix(chart._build_merged())
    if isinstance(chart, fm.HConcatChart):
        return leaf + _line_prefix(chart.charts[1])
    return _line_prefix(chart)


def test_the_flat_smooth_is_flat_and_the_layered_one_is_not() -> None:
    flat = json.loads(_FLAT_SMOOTH.build().to_spec().to_json())
    layered = json.loads(_SMOOTH.build().to_spec().to_json())
    assert "layers" not in flat and flat["entry"] == "mark_smooth"
    assert len(layered["layers"]) == 2 and "entry" not in layered


@pytest.mark.parametrize("output", ["to_svg", "interactive"])
@pytest.mark.parametrize("route", sorted(_COMPOSITE_ROUTES))
@pytest.mark.parametrize("shape", sorted(_SMOOTHS))
def test_a_composite_lines_width_conflict_names_the_composite(
    shape: str, route: str, output: str
) -> None:
    chart = _COMPOSITE_ROUTES[route](_SMOOTHS[shape], {"size": 40, "stroke_width": 3})
    with pytest.raises(ValueError) as exc:
        getattr(chart, output)()
    assert str(exc.value) == _refusal_prefix(chart, output) + _SMOOTH_CONFLICT


# The merged chart of each merging verb, flat smooth placed second.
_FLAT_MERGES: dict[str, Callable[[], fm.Chart]] = {
    "plus": lambda: _Member("point").chart() + _FLAT_SMOOTH.build(),
    "plus_reversed": lambda: _FLAT_SMOOTH.build() + _Member("point").chart(),
    "fm_layer_merged": lambda: fm.layer(
        _Member("point").chart(), _FLAT_SMOOTH.build()
    )._build_merged(),
    "chart_layer_host": lambda: _FLAT_SMOOTH.build().layer(_Member("point").layer()),
}


@pytest.mark.parametrize("verb", sorted(_FLAT_MERGES))
def test_a_flat_composites_entry_moves_onto_the_layer_it_contributes(verb: str) -> None:
    wire = json.loads(_FLAT_MERGES[verb]().to_spec().to_json())
    assert "entry" not in wire
    entries = {ly["mark"]: ly.get("entry") for ly in wire["layers"]}
    assert entries == {"point": None, "line": "mark_smooth"}


def test_a_desugared_layer_names_its_entry_and_a_declared_one_does_not() -> None:
    layers = json.loads((_Member("point").chart() + _SMOOTH.build()).to_spec().to_json())["layers"]
    assert [ly.get("entry") for ly in layers] == [None, "mark_smooth", "mark_smooth"]


@pytest.mark.parametrize("output", ["to_svg", "interactive"])
@pytest.mark.parametrize("scatter", [True, False], ids=["with_scatter", "fit_only"])
@pytest.mark.parametrize("entry", ["lmplot", "regplot"])
def test_a_figure_functions_width_conflict_names_the_figure_function(
    entry: str, scatter: bool, output: str
) -> None:
    chart = getattr(fm, entry)(
        _DF, x="x", y="y", scatter=scatter, line_kws={"size": 40, "stroke_width": 3}
    )
    layers = json.loads(chart.to_spec().to_json())["layers"]
    assert {ly.get("entry") for ly in layers} == {entry}
    (index,) = [i for i, ly in enumerate(layers) if ly["mark"] == "line"]
    with pytest.raises(ValueError) as exc:
        getattr(chart, output)()
    assert str(exc.value) == f"layer {index}: {_SMOOTH_CONFLICT.replace('mark_smooth', entry)}"


_WIDTH_CONFLICT = {"size": 40, "stroke_width": 3}
_FIGURE_DF = pl.DataFrame(
    {
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [0.0, 1.0, 0.0, 1.0, 3.0, 2.0],
        "g": ["a", "b", "a", "b", "a", "b"],
    }
)

# Figure functions whose own line takes the caller's style, each reaching it
# through its `mark=` override.
_FIGURE_LINES: dict[str, Callable[[], fm.Chart]] = {
    "relplot": lambda: fm.relplot(_FIGURE_DF, x="x", y="y", kind="line", mark=_WIDTH_CONFLICT),
    "residplot": lambda: fm.residplot(
        _FIGURE_DF, x="x", y="y", lowess=True, mark={"lowess": _WIDTH_CONFLICT}
    ),
    "displot": lambda: fm.displot(_FIGURE_DF, x="x", kind="ecdf", mark=_WIDTH_CONFLICT),
}


@pytest.mark.parametrize("output", ["to_svg", "interactive"])
@pytest.mark.parametrize("entry", sorted(_FIGURE_LINES))
def test_a_figure_lines_width_conflict_names_the_figure_function(entry: str, output: str) -> None:
    chart = _FIGURE_LINES[entry]()
    with pytest.raises(ValueError) as exc:
        getattr(chart, output)()
    assert str(exc.value) == _line_prefix(chart) + _SMOOTH_CONFLICT.replace("mark_smooth", entry)


def _leaf_entries(chart: Any) -> set:
    """Every entry a leaf chart of *chart* records, on the chart or on its layers."""
    if isinstance(chart, fm.Chart):
        wire = json.loads(chart.to_spec().to_json())
        layers = wire.get("layers")
        return {ly.get("entry") for ly in layers} if layers is not None else {wire.get("entry")}
    if isinstance(chart, fm.RepeatChart):
        # The panels a repeat draws are its expansion, not its templates.
        return set().union(*(_leaf_entries(panel) for _, _, panel in chart.expand()))
    return set().union(*(_leaf_entries(child) for child in chart.charts))


@pytest.mark.parametrize(
    ("entry", "build"),
    [
        ("jointplot", lambda: fm.jointplot(_FIGURE_DF, x="x", y="y", kind="reg")),
        ("pairplot", lambda: fm.pairplot(_FIGURE_DF.select("x", "y"), kind="reg")),
    ],
)
def test_every_leaf_of_a_composite_figure_names_the_figure_function(
    entry: str, build: Callable[[], Any]
) -> None:
    assert _leaf_entries(build()) == {entry}


# ---------------------------------------------------------------------------
# A composite's own color stays on its own layers.
# ---------------------------------------------------------------------------

_ROC = pl.DataFrame(
    {
        "fpr": [0.0, 0.2, 1.0, 0.0, 0.5, 1.0],
        "tpr": [0.0, 0.7, 1.0, 0.0, 0.6, 1.0],
        "class": ["a", "a", "a", "b", "b", "b"],
        "auc": [0.8, 0.8, 0.8, 0.6, 0.6, 0.6],
    }
)
_CUT = pl.DataFrame({"fpr": [0.3], "tpr": [0.0], "tpr2": [1.0]})


def _importance() -> fm.Chart:
    from tests.diagnostics.test_compare_aggregate_goldens import _dodge_compare_models

    base, alt, X, y = _dodge_compare_models()
    return fm.importance_chart(base, X, y, compare={"alt": alt})


def _importance_rule() -> fm.Chart:
    from tests.diagnostics.test_compare_aggregate_goldens import _dodge_compare_models

    _, _, X, _ = _dodge_compare_models()
    # Drawn in the composite's flipped space: importance across, features down.
    marker = pl.DataFrame({"feature": [X.columns[0]], "importance": [0.1]})
    return fm.Chart(marker).mark_rule().encode(x="importance", y="feature")


@dataclass(frozen=True)
class _ColoredComposite:
    """A composite that colors some of its own layers, and a plain rule to attach to it."""

    build: Callable[[], fm.Chart]
    rule_chart: Callable[[], fm.Chart]
    rule_layer: Callable[[], fm.Layer]


_COLORED_COMPOSITES: dict[str, _ColoredComposite] = {
    "importance_chart_compare": _ColoredComposite(
        _importance,
        _importance_rule,
        lambda: fm.Layer(mark="rule", encoding={"x": fm.X("feature")}),
    ),
    "mark_roc": _ColoredComposite(
        lambda: fm.Chart(_ROC).mark_roc(),
        lambda: fm.Chart(_CUT).mark_rule().encode(x="fpr", y="tpr", y2="tpr2"),
        lambda: fm.Layer(mark="rule", data=_CUT, encoding={"x": fm.X("fpr")}),
    ),
}

_ATTACH: dict[str, Callable[[_ColoredComposite], fm.Chart]] = {
    "plus": lambda c: c.build() + c.rule_chart(),
    "chart_layer": lambda c: c.build().layer(c.rule_layer()),
}


def _legend_titles(svg: str) -> list[str]:
    return re.findall(r'font-weight="600" text-anchor="start">([^<]*)<', svg)


def _line_strokes(svg: str) -> set[str]:
    return set(re.findall(r'<line [^>]*stroke="(#[0-9a-fA-F]+)"', svg))


@pytest.mark.parametrize("verb", sorted(_ATTACH))
@pytest.mark.parametrize("composite", sorted(_COLORED_COMPOSITES))
def test_attaching_a_layer_keeps_a_composites_color_on_its_own_layers(
    composite: str, verb: str
) -> None:
    spec = _COLORED_COMPOSITES[composite]
    alone = spec.build().to_svg()
    attached = _ATTACH[verb](spec).to_svg()
    # No legend title appears for the composite's layer-level color ...
    assert _legend_titles(attached) == _legend_titles(alone)
    # ... and the attached rule draws in its own stroke, not the palette.
    assert _line_strokes(attached) == _line_strokes(alone) | _line_strokes(
        spec.rule_chart().to_svg()
    )


def test_vacuity_control_a_user_authored_color_is_promoted_with_its_title() -> None:
    rule = _COLORED_COMPOSITES["mark_roc"].rule_chart()
    colored = fm.Chart(_ROC).mark_line().encode(x="fpr", y="tpr", color="class")
    svg = (rule + colored).to_svg()
    assert _legend_titles(svg) == ["class"]
