"""Every ferrum warning names the caller's file and line, on every render route.

The auto-raster warnings, the ``override()`` deprecation, every warning the
Rust renderer emits (``NoRowsToDraw``, ``DomainRowsLeftOut`` and the transform
warnings) and every ``warn_once`` family are attributed to the first frame
outside ``ferrum``: the line in the test, or in the user's own helper function
that called ferrum, whichever route (``to_svg``, ``to_png``, ``.interactive()``,
``to_html``, ``save``, a composite leaf, a figure) and door (``+``,
``Chart.layer``, ``LayerChart``, ``vconcat``, a facet) the chart took. Each pin
compares the ordered list of full messages, their category and their location,
so a doubled emission, a changed text or a frame inside ferrum fails it.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from ferrum import Chart
from ferrum._warn import reset_warnings
from ferrum.render_config import RenderConfig
from tests.test_domain_rows_left_out import _domain_warning
from tests.test_null_group_warning import _NO_FACET_ROWS_WARNING, _NO_ROWS_TO_DRAW
from tests.test_null_group_warning import _null_key_dropped_message

_N = 100
_DF = pl.DataFrame(
    {
        "x": [float(i) for i in range(_N)],
        "y": [float(i) for i in range(_N)],
        "z": [float(i) for i in range(_N)],
        "g": ["a", "b"] * (_N // 2),
    }
)

_SMALL_THRESHOLD = RenderConfig(raster_threshold=10)

_SUBSTITUTED = (
    "Auto-raster: substituted mark_raster for point (100 marks > threshold 10). "
    "Pass raster=False to .show()/.save() to disable."
)
_UNTYPED_AXES = (
    "Chart has 100 marks which may produce large output. "
    "Use `.mark_raster()` for efficient rendering, or set `raster=False` to suppress this warning."
)
_CATEGORICAL_COLOR = (
    "Chart has 100 marks with a categorical/ordinal color field 'g', which is permanently "
    "excluded from raster substitution because rasterising would discard categorical "
    "information -- there is no pixel aggregate that preserves a non-quantitative color's "
    "identity. This exclusion does not apply to quantitative color, which substitutes "
    "automatically. To rasterize this chart anyway (accepting that the category collapses to "
    "one summary color per pixel), call `.mark_raster(aggregate=..., field=...)` manually."
)
_CHANNEL_AGGREGATE = (
    "Chart has 100 marks which may produce large output, and its color encoding carries its "
    "own aggregate=, so its marks are drawn per group -- `.mark_raster()` cannot draw it "
    "either. Compute the aggregate with `.transform(fm.transform_aggregate(...))` and bind the "
    "output column with no aggregate= of its own, or set `raster=False` to suppress this "
    "warning."
)
_POSITIONAL_BIN = (
    "Chart has 100 marks which may produce large output, and its x or y encoding carries a "
    "bin= with no aggregate=, so `.mark_raster()` cannot draw it either. Put an aggregate= on "
    "the opposing positional channel (e.g. `y=fm.Y('field', aggregate='mean')` beside "
    "`x=fm.X('field', bin=True)`), or set `raster=False` to suppress this warning."
)
_OVERRIDE = "override(width=...) is deprecated; use .properties(width=...) instead."

_THETA = (
    "Encoding channel 'theta' is accepted but not rendered; the renderer ignores it and it is "
    "omitted from the spec."
)
_DETAIL = (
    "encode(detail=...) is accepted but ignored by mark_bar; only mark_line, mark_area, "
    "mark_polygon, and mark_ribbon read the detail channel for per-group splitting."
)
_STROKE = (
    "encode(stroke=...) is ignored when color= is also encoded; stroke is aliased to color "
    "only when color is absent."
)
_STACK = (
    "stack='zero' is not supported by mark_point and was ignored. Stacking is honored by "
    "mark_bar and mark_area, and by mark_arc and mark_text on the theta channel of a CoordPolar."
)
_DISPLOT_BINS = (
    "displot(kind='kde', bins=5): bins= is inert for kind='kde' -- it evaluates no binned "
    "statistic, so the rule you named cannot act on anything and the render is identical for "
    "every value. Drop bins=, or use kind='hist', which bins."
)
_ROC_AVERAGE = (
    "mark_roc(average='marco') matched no rows in the class column; rendering every class "
    "unfiltered. Check for a typo in average= (expected one of 'macro', 'micro', 'weighted', "
    "or a value actually present in the data)."
)
_DISCRIMINATION_METRICS = (
    "mark_discrimination_threshold(metrics=('nope',)) matched no rows in the metric column; "
    "rendering every metric unfiltered. Check for a typo in metrics=."
)


def _capture(call: Callable[[], Any]) -> list[warnings.WarningMessage]:
    """Every warning *call* raises, with no de-duplication and a fresh warn-once registry."""
    reset_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return caught


def _call_line(call: Callable[[], Any]) -> int:
    """The line a one-line lambda sits on: the frame a warning must be attributed to."""
    return call.__code__.co_firstlineno


def _assert_warned(
    caught: list[warnings.WarningMessage],
    messages: list[str],
    line: int,
    *,
    category: type[Warning] = UserWarning,
) -> None:
    """*caught* is exactly *messages*, in order, each of *category*, each named at *line* here."""
    assert [(str(w.message), w.category, w.filename, w.lineno) for w in caught] == [
        (message, category, __file__, line) for message in messages
    ]


def _raster_point() -> Chart:
    return (
        Chart(_DF).mark_point().encode(x="x:Q", y="y:Q").properties(render_config=_SMALL_THRESHOLD)
    )


def _routes(chart: Chart, tmp_path) -> dict[str, Callable[[], Any]]:
    """Every output route of one chart, each a single-line callable."""
    svg_path = str(tmp_path / "out.svg")
    html_path = str(tmp_path / "out.html")
    return {
        "to_svg": lambda: chart.to_svg(),
        "to_png": lambda: chart.to_png(),
        "interactive": lambda: chart.interactive(),
        "to_html": lambda: chart.to_html(),
        "interactive_to_html": lambda: chart.interactive().to_html(),
        "save_svg": lambda: chart.save(svg_path),
        "save_html": lambda: chart.save(html_path),
    }


_ROUTE_NAMES = [
    "to_svg",
    "to_png",
    "interactive",
    "to_html",
    "interactive_to_html",
    "save_svg",
    "save_html",
]


# ---------------------------------------------------------------------------
# Auto-raster warnings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", _ROUTE_NAMES)
def test_the_auto_raster_substitution_warning_names_the_caller_on_every_route(route, tmp_path):
    call = _routes(_raster_point(), tmp_path)[route]
    _assert_warned(_capture(call), [_SUBSTITUTED], _call_line(call))


def test_the_auto_raster_warning_inside_hconcat_names_the_caller():
    one, two = _raster_point(), _raster_point()
    call = lambda: fm.hconcat(one, two).to_svg()  # noqa: E731
    _assert_warned(_capture(call), [_SUBSTITUTED, _SUBSTITUTED], _call_line(call))


def _ineligible_charts() -> dict[str, tuple[Chart, str]]:
    """Each over-threshold chart an auto-raster gate refuses, with the one warning it raises."""
    base = lambda: Chart(_DF).mark_point().properties(render_config=_SMALL_THRESHOLD)  # noqa: E731
    return {
        "untyped_axes": (base().encode(x="x", y="y"), _UNTYPED_AXES),
        "categorical_color": (base().encode(x="x:Q", y="y:Q", color="g"), _CATEGORICAL_COLOR),
        "channel_aggregate": (
            base().encode(x="x:Q", y="y:Q", color=fm.Color("z:Q", aggregate="sum")),
            _CHANNEL_AGGREGATE,
        ),
        "positional_bin": (base().encode(x=fm.X("x:Q", bin=True), y="y:Q"), _POSITIONAL_BIN),
    }


_GATES = ["untyped_axes", "categorical_color", "channel_aggregate", "positional_bin"]


@pytest.mark.parametrize("gate", _GATES)
@pytest.mark.parametrize("route", _ROUTE_NAMES)
def test_each_auto_raster_guidance_warning_names_the_caller_on_every_route(gate, route, tmp_path):
    chart, message = _ineligible_charts()[gate]
    call = _routes(chart, tmp_path)[route]
    _assert_warned(_capture(call), [message], _call_line(call))


@pytest.mark.parametrize("gate", _GATES)
def test_each_auto_raster_guidance_warning_inside_hconcat_names_the_caller(gate):
    chart, message = _ineligible_charts()[gate]
    call = lambda: fm.hconcat(chart, chart).to_svg()  # noqa: E731
    _assert_warned(_capture(call), [message, message], _call_line(call))


# ---------------------------------------------------------------------------
# The override() deprecation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", _ROUTE_NAMES)
def test_the_override_deprecation_names_the_caller_on_every_route(route, tmp_path):
    chart = Chart(_DF).mark_point().encode(x="x", y="y").override(width=300)
    call = _routes(chart, tmp_path)[route]
    _assert_warned(_capture(call), [_OVERRIDE], _call_line(call), category=DeprecationWarning)


def test_the_override_deprecation_inside_hconcat_names_the_caller():
    chart = Chart(_DF).mark_point().encode(x="x", y="y").override(width=300)
    call = lambda: fm.hconcat(chart, chart).to_svg()  # noqa: E731
    _assert_warned(
        _capture(call), [_OVERRIDE, _OVERRIDE], _call_line(call), category=DeprecationWarning
    )


# ---------------------------------------------------------------------------
# Rust render warnings
# ---------------------------------------------------------------------------


def _no_rows_chart() -> Chart:
    return Chart(_DF.head(0)).mark_point().encode(x="x", y="y")


def _populated_chart() -> Chart:
    return Chart(_DF).mark_point().encode(x="x", y="y")


def _domain_chart() -> Chart:
    return (
        Chart(_DF)
        .mark_point()
        .encode(x=fm.X("x", scale=fm.LinearScale(domain=[10.0, 90.0])), y="y")
    )


_DOMAIN_ONE_LEAF = _domain_warning("x", "x", 19)
_DOMAIN_TWO_LEAVES = _domain_warning("x", "x", 38)


@pytest.mark.parametrize("route", _ROUTE_NAMES)
def test_a_no_rows_warning_names_the_caller_on_every_route(route, tmp_path):
    call = _routes(_no_rows_chart(), tmp_path)[route]
    _assert_warned(_capture(call), [_NO_ROWS_TO_DRAW], _call_line(call))


@pytest.mark.parametrize("route", _ROUTE_NAMES)
def test_a_domain_rows_left_out_warning_names_the_caller_on_every_route(route, tmp_path):
    call = _routes(_domain_chart(), tmp_path)[route]
    _assert_warned(_capture(call), [_DOMAIN_ONE_LEAF], _call_line(call))


@pytest.mark.parametrize("route", ["to_svg", "interactive", "to_html"])
def test_a_composite_leaf_warning_names_the_caller(route):
    composite = fm.hconcat(_domain_chart(), _domain_chart())
    call = {
        "to_svg": lambda: composite.to_svg(),
        "interactive": lambda: composite.interactive(),
        "to_html": lambda: composite.to_html(),
    }[route]
    _assert_warned(_capture(call), [_DOMAIN_TWO_LEAVES], _call_line(call))


def test_a_composite_no_rows_warning_names_the_caller():
    composite = fm.hconcat(_no_rows_chart(), _populated_chart())
    call = lambda: composite.to_svg()  # noqa: E731
    _assert_warned(_capture(call), [_NO_ROWS_TO_DRAW], _call_line(call))


def test_a_figure_warning_names_the_caller():
    figure = fm.relplot(_DF.head(0), x="x", y="y")
    call = lambda: figure.to_svg()  # noqa: E731
    _assert_warned(_capture(call), [_NO_ROWS_TO_DRAW], _call_line(call))


def test_a_transform_warning_before_a_refusal_names_the_caller():
    """The warning a run raised ahead of its refusal is delivered with the refusal's frame."""
    frame = _DF.with_columns(
        t=pl.Series(list("abcde") * (_N // 5)),
        g=pl.Series(["a", "b", None, "b", "a"] * (_N // 5)),
    )
    chart = Chart(frame).mark_point().encode(x="x", y="t", color="g").mark_smooth()

    def call() -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            chart.to_svg()

    # The raising line is inside `call`, two lines below its `def`.
    _assert_warned(
        _capture(call),
        [_null_key_dropped_message("Smooth", 20, "g")],
        call.__code__.co_firstlineno + 2,
    )


# ---------------------------------------------------------------------------
# Every door
# ---------------------------------------------------------------------------

# Per door and family: the ordered messages the static route and the interactive route raise.
# The two routes differ only where `.interactive()` renders the layers as one merged panel.

_ONE = {
    "domain": [_DOMAIN_TWO_LEAVES],
    "raster": [_SUBSTITUTED],
    "no_rows": [_NO_ROWS_TO_DRAW],
}


def _family_chart(family: str) -> Chart:
    return {
        "domain": _domain_chart,
        "raster": _raster_point,
        "no_rows": _no_rows_chart,
    }[family]()


def _door(door: str, family: str) -> Any:
    one = _family_chart
    return {
        "plus": lambda: one(family) + one(family),
        "chart_layer": lambda: one(family).layer(fm.Layer(mark="point")),
        "layer_chart": lambda: fm.LayerChart(one(family), one(family)),
        "independent_y": lambda: fm.LayerChart(
            one(family), one(family), resolve={"y": "independent"}
        ),
        "vconcat": lambda: fm.vconcat(
            one(family), _populated_chart() if family == "no_rows" else one(family)
        ),
        "facet": lambda: one(family).facet(col="g"),
    }[door]()


# (door, family) -> (static messages, interactive messages)
_DOOR_MESSAGES: dict[tuple[str, str], tuple[list[str], list[str]]] = {}


def _expect(door: str, family: str, static: list[str], interactive: list[str] | None = None):
    _DOOR_MESSAGES[(door, family)] = (static, static if interactive is None else interactive)


for _family, _msg in [
    ("domain", _DOMAIN_TWO_LEAVES),
    ("raster", _SUBSTITUTED),
    ("no_rows", _NO_ROWS_TO_DRAW),
]:
    _expect("plus", _family, [_msg])
    _expect("independent_y", _family, [_msg])
_expect("chart_layer", "domain", [_DOMAIN_TWO_LEAVES])
_expect("chart_layer", "raster", [_SUBSTITUTED])
_expect("chart_layer", "no_rows", [_NO_ROWS_TO_DRAW])
_expect("layer_chart", "domain", [_DOMAIN_TWO_LEAVES])
_expect("layer_chart", "raster", [_SUBSTITUTED, _SUBSTITUTED], [_SUBSTITUTED])
_expect("layer_chart", "no_rows", [_NO_ROWS_TO_DRAW])
_expect("vconcat", "domain", [_DOMAIN_TWO_LEAVES])
_expect("vconcat", "raster", [_SUBSTITUTED, _SUBSTITUTED])
_expect("vconcat", "no_rows", [_NO_ROWS_TO_DRAW])
_expect("facet", "domain", [_domain_warning("x", "x", 19)])
_expect("facet", "raster", [_SUBSTITUTED])
_expect("facet", "no_rows", [_NO_FACET_ROWS_WARNING])


@pytest.mark.parametrize("route", ["to_svg", "interactive"])
@pytest.mark.parametrize("family", ["domain", "raster", "no_rows"])
@pytest.mark.parametrize(
    "door", ["plus", "chart_layer", "layer_chart", "independent_y", "vconcat", "facet"]
)
def test_a_warning_names_the_caller_on_every_door(door, family, route):
    static, interactive = _DOOR_MESSAGES[(door, family)]
    call = {
        "to_svg": lambda: _door(door, family).to_svg(),
        "interactive": lambda: _door(door, family).interactive(),
    }[route]
    _assert_warned(_capture(call), static if route == "to_svg" else interactive, _call_line(call))


# ---------------------------------------------------------------------------
# warn_once families
# ---------------------------------------------------------------------------

_ROC = pl.DataFrame(
    {
        "fpr": [0.0, 0.5, 1.0] * 3,
        "tpr": [0.0, 0.6, 1.0, 0.0, 0.7, 1.0, 0.0, 0.65, 1.0],
        "class": ["0"] * 3 + ["1"] * 3 + ["macro"] * 3,
    }
)
_THRESHOLDS = pl.DataFrame(
    {
        "threshold": [0.1, 0.2, 0.3] * 2,
        "metric": ["precision"] * 3 + ["f1"] * 3,
        "value": [0.5, 0.6, 0.7, 0.4, 0.9, 0.3],
    }
)

#: family -> (a builder of the chart, the one message it warns once).
_WARN_ONCE_FAMILIES: dict[str, tuple[Callable[[], Chart], str]] = {
    "unrendered_channel": (lambda: _populated_chart().encode(theta="z"), _THETA),
    "ignored_detail": (
        lambda: Chart(_DF).mark_bar().encode(x="g", y="y", detail="g"),
        _DETAIL,
    ),
    "ignored_stroke": (
        lambda: Chart(_DF).mark_point().encode(x="x", y="y", color="g", stroke="g"),
        _STROKE,
    ),
    "ignored_stack": (
        lambda: Chart(_DF).mark_point().encode(x="x", y=fm.Y("y", stack="zero")),
        _STACK,
    ),
    "roc_average": (
        lambda: Chart(_ROC).mark_roc(average="marco", reference_line=False),
        _ROC_AVERAGE,
    ),
    "discrimination_metrics": (
        lambda: Chart(_THRESHOLDS).mark_discrimination_threshold(metrics=("nope",)),
        _DISCRIMINATION_METRICS,
    ),
}


#: Families raised while the chart is built, so named at the line that builds it.
_RAISED_AT_BUILD = frozenset({"roc_average", "discrimination_metrics"})


def _family_line(family: str, build: Callable[[], Chart], call: Callable[[], Any]) -> int:
    return _call_line(build) if family in _RAISED_AT_BUILD else _call_line(call)


@pytest.mark.parametrize("route", ["to_svg", "interactive", "to_html"])
@pytest.mark.parametrize("family", sorted(_WARN_ONCE_FAMILIES))
def test_a_warn_once_family_names_the_caller_on_every_route(family, route):
    build, message = _WARN_ONCE_FAMILIES[family]
    call = {
        "to_svg": lambda: build().to_svg(),
        "interactive": lambda: build().interactive(),
        "to_html": lambda: build().to_html(),
    }[route]
    _assert_warned(_capture(call), [message], _family_line(family, build, call))


@pytest.mark.parametrize("family", sorted(_WARN_ONCE_FAMILIES))
def test_a_warn_once_family_names_the_caller_inside_hconcat(family):
    build, message = _WARN_ONCE_FAMILIES[family]
    call = lambda: fm.hconcat(build(), _populated_chart()).to_svg()  # noqa: E731
    _assert_warned(_capture(call), [message], _family_line(family, build, call))


def test_a_displot_warning_names_the_caller_at_build_time_and_on_render():
    build = lambda: fm.displot(_DF, x="x", kind="kde", bins=5)  # noqa: E731
    _assert_warned(_capture(build), [_DISPLOT_BINS], _call_line(build))
    render = lambda: build().to_svg()  # noqa: E731
    _assert_warned(_capture(render), [_DISPLOT_BINS], _call_line(build))


def test_an_encoding_channel_kwarg_warning_names_the_caller():
    build = lambda: fm.X("x", bandPosition=3)  # noqa: E731
    _assert_warned(
        _capture(build),
        ["x(bandPosition=...) is accepted but not yet honored; planned for a future Phase."],
        _call_line(build),
    )


# ---------------------------------------------------------------------------
# A ferrum call made inside a user helper
# ---------------------------------------------------------------------------


def _helper_render_svg(chart: Chart) -> str:
    return chart.to_svg()


def _helper_render_interactive(chart: Chart) -> Any:
    return chart.interactive()


def _helper_render_composite(chart: Chart) -> str:
    return fm.hconcat(chart, _populated_chart()).to_svg()


def _helper_render_theta(_chart: Chart) -> str:
    return _populated_chart().encode(theta="z").to_svg()


_HELPER_CHARTS: dict[str, tuple[Callable[[], Chart], str]] = {
    "raster": (_raster_point, _SUBSTITUTED),
    "no_rows": (_no_rows_chart, _NO_ROWS_TO_DRAW),
    "domain": (_domain_chart, _DOMAIN_ONE_LEAF),
}
_HELPERS = {
    "svg": _helper_render_svg,
    "interactive": _helper_render_interactive,
    "composite": _helper_render_composite,
}


@pytest.mark.parametrize("helper_name", sorted(_HELPERS))
@pytest.mark.parametrize("family", sorted(_HELPER_CHARTS))
def test_a_warning_from_a_call_inside_a_user_helper_names_the_helpers_line(helper_name, family):
    helper = _HELPERS[helper_name]
    chart_factory, message = _HELPER_CHARTS[family]
    chart = chart_factory()
    caught = _capture(lambda: helper(chart))
    _assert_warned(caught, [message], helper.__code__.co_firstlineno + 1)


def test_a_warn_once_family_inside_a_user_helper_names_the_helpers_line():
    caught = _capture(lambda: _helper_render_theta(_populated_chart()))
    _assert_warned(caught, [_THETA], _helper_render_theta.__code__.co_firstlineno + 1)
