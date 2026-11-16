"""A figure rebuilds once per distinct grouping, not once per render.

``ferrum.plots._helpers._figure_entry`` records a figure's own call
(``ferrum._compose.figure_recorded``) so a later ``.encode()``, or a
composition the figure joins, can call the figure again under the new
grouping bindings and refit its statistics. That rebuild runs through a
memoized closure (``ferrum._compose._memoized``): rendering the same chart
more than once groups it the same way every time, so the call that builds
the chart -- and reissues any build-time warning -- runs again only the
first time a given grouping is seen, never once per render. The same
memoization covers a composite mark's own pending-mark resolution
(``ferrum._compose.composite_source``), exercised here through
``mark_smooth().layer(...)``.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

import ferrum as fm
import ferrum._desugar as _desugar
import ferrum.plots._helpers as _helpers

_DF = pl.DataFrame(
    {
        "x": list(range(20)),
        "y": list(range(20)),
        "y2": list(range(20)),
        "g": ["a", "b"] * 10,
        "h": ["p", "q"] * 10,
    }
)


def _two_fit_figure():
    """A throwaway figure whose own body composes two statistics-bearing operands with ``+``.

    Both ``mark_smooth()`` operands carry their own statistics, so
    ``figure_recorded`` records a ``_figure_source`` for each -- the case
    where a figure's result is itself a composition, not just one layered
    chart. Both sources share the same memoized ``rebuilt`` closure.
    """
    from ferrum.plots._helpers import _figure_entry

    @_figure_entry
    def two_fit_figure(data, *, encode=None):
        a = fm.Chart(data).mark_smooth(ci=None).encode(x="x", y="y")
        b = fm.Chart(data).mark_smooth(ci=None).encode(x="x", y="y2")
        chart = a + b
        if encode:
            chart = chart.encode(**encode)
        return chart

    return two_fit_figure


@pytest.fixture
def entry_calls(monkeypatch):
    """Every public entry a figure call actually built, in call order."""
    calls: list[str] = []
    original = _helpers.speak_as

    def counted(entry, build, *args, **kwargs):
        calls.append(entry)
        return original(entry, build, *args, **kwargs)

    monkeypatch.setattr(_helpers, "speak_as", counted)
    return calls


@pytest.fixture
def resolve_calls(monkeypatch):
    """The number of times a pending composite mark's desugar actually ran."""
    counted = {"n": 0}
    original = _desugar._resolve_pending_as

    def wrapped(*args, **kwargs):
        counted["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_desugar, "_resolve_pending_as", wrapped)
    return counted


def test_a_figures_fit_runs_once_per_grouping_across_every_render_door(entry_calls) -> None:
    """``regplot(...).encode(color="g")`` fits once, then rebuilds once -- never again, however it renders."""
    chart = fm.regplot(_DF, x="x", y="y").encode(color="g")
    chart.to_svg()
    chart.to_json()
    chart.to_spec()
    chart.interactive().to_html()
    assert entry_calls == ["regplot", "regplot"]


def test_a_second_distinct_grouping_costs_exactly_one_more_call(entry_calls) -> None:
    """A later ``.encode()`` that changes the grouping again rebuilds once more, not once per render."""
    base = fm.regplot(_DF, x="x", y="y")
    grouped = base.encode(color="g")
    grouped.to_svg()
    grouped.to_svg()
    assert entry_calls == ["regplot", "regplot"]

    regrouped = grouped.encode(color="h")
    regrouped.to_svg()
    regrouped.to_svg()
    assert entry_calls == ["regplot", "regplot", "regplot"]


def test_a_plus_composition_of_a_figure_rebuilds_once_per_grouping(entry_calls) -> None:
    """``(figure + point).encode(color=)`` fits once, rebuilds once, however many times it renders."""
    scatter = fm.Chart(_DF).mark_point().encode(x="x", y="y")
    chart = (fm.regplot(_DF, x="x", y="y") + scatter).encode(color="g")
    chart.to_svg()
    chart.to_svg()
    chart.to_json()
    assert entry_calls == ["regplot", "regplot"]


def test_a_figure_with_no_later_grouping_builds_exactly_once(entry_calls) -> None:
    """A figure never regrouped after its own call renders from the one build, any number of times."""
    chart = fm.regplot(_DF, x="x", y="y")
    chart.to_svg()
    chart.to_svg()
    chart.to_json()
    assert entry_calls == ["regplot"]


def test_a_build_time_warning_under_a_grouping_is_not_reissued_on_every_render() -> None:
    """``show_metrics=True`` under a grouping warns once, not once per render."""
    chart = fm.regplot(_DF, x="x", y="y", show_metrics=True).encode(color="g")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart.to_svg()
        chart.to_svg()
        chart.to_json()
    messages = [str(w.message) for w in caught]
    assert messages == [
        "regplot: show_metrics=True has no single fit to describe: the fit is "
        "computed per group of 'g', so the R²/RMSE/MAE corner is not drawn"
    ]


def test_every_render_of_a_regrouped_figure_is_byte_identical() -> None:
    """Rebuilding once does not change what the chart draws on any later render."""
    chart = fm.regplot(_DF, x="x", y="y").encode(color="g")
    first = chart.to_svg()
    assert chart.to_svg() == first
    assert chart.to_svg() == first


def test_a_composite_mark_re_resolved_under_a_later_grouping_resolves_once_per_grouping(
    resolve_calls,
) -> None:
    """``mark_smooth().layer(...).encode(color=)`` resolves its fit once per grouping, not once per render."""
    chart = (
        fm.Chart(_DF)
        .mark_smooth(ci=None)
        .encode(x="x", y="y")
        .layer(fm.Layer(mark="tick", encoding={"x": "x"}))
    )
    # Resolving once here (inside `.layer()`) builds the ungrouped fit.
    assert resolve_calls["n"] == 1

    grouped = chart.encode(color="g")
    first = grouped.to_svg()
    assert resolve_calls["n"] == 2
    assert grouped.to_svg() == first
    assert grouped.to_json()
    assert resolve_calls["n"] == 2


def test_a_figure_composed_of_two_fit_operands_shares_one_memo_across_them(entry_calls) -> None:
    """A figure whose own result is a ``+`` of two statistics-bearing operands rebuilds once, not once per operand.

    ``figure_recorded`` gives each such operand its own replay record, but
    both records close over the same memoized ``rebuilt``: rendering after
    a later grouping calls the figure again exactly once, even though two
    operands each ask for the rebuild.
    """
    two_fit_figure = _two_fit_figure()
    chart = two_fit_figure(_DF)
    assert entry_calls == ["two_fit_figure"]
    assert len(chart._operands) == 2
    assert all(op.statistics for op in chart._operands)

    grouped = chart.encode(color="g")
    assert entry_calls == ["two_fit_figure"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped.to_svg()
    assert entry_calls == ["two_fit_figure", "two_fit_figure"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped.to_svg()
        grouped.to_json()
    assert entry_calls == ["two_fit_figure", "two_fit_figure"]


def _resolved_smooth():
    return (
        fm.Chart(_DF)
        .mark_smooth(ci=None)
        .encode(x="x", y="y")
        .layer(fm.Layer(mark="tick", encoding={"x": "x"}))
    )


def test_a_composite_mark_rebuilt_on_a_later_binding_resolves_once_per_input(resolve_calls) -> None:
    """``.layer(...).encode(y='y2')`` resolves the fit again once, however many times it renders."""
    chart = _resolved_smooth()
    assert resolve_calls["n"] == 1

    rebound = chart.encode(y="y2")
    first = rebound.to_svg()
    assert resolve_calls["n"] == 2
    assert rebound.to_svg() == first
    assert rebound.to_json()
    assert rebound.interactive().to_html()
    assert resolve_calls["n"] == 2


def test_a_second_distinct_input_costs_exactly_one_more_resolution(resolve_calls) -> None:
    chart = _resolved_smooth()
    chart.encode(y="y2").to_svg()
    assert resolve_calls["n"] == 2

    chart.encode(x="y2").to_svg()
    chart.encode(x="y2").to_svg()
    assert resolve_calls["n"] == 3


def test_a_grouping_and_an_input_rebuild_in_one_resolution(resolve_calls) -> None:
    """``.encode(y=, color=)`` after ``.layer()`` is one rebuild, not one per kind of change."""
    chart = _resolved_smooth()
    both = chart.encode(y="y2", color="g")
    both.to_svg()
    both.to_svg()
    assert resolve_calls["n"] == 2


def test_a_same_column_restatement_does_not_resolve_again(resolve_calls) -> None:
    chart = _resolved_smooth()
    chart.encode(y=fm.Y("y", title="T")).to_svg()
    assert resolve_calls["n"] == 1


# ── A figure given another column afterwards ────────────────────────────────


def test_a_figure_given_another_column_calls_the_figure_once_per_input_across_every_render_door(
    entry_calls,
) -> None:
    """``regplot(...).encode(y='y2')`` calls the figure once more, however many doors render it."""
    chart = fm.regplot(_DF, x="x", y="y").encode(y="y2")
    chart.to_svg()
    chart.to_json()
    chart.to_spec()
    chart.interactive().to_html()
    assert entry_calls == ["regplot", "regplot"]


def test_a_second_distinct_input_costs_exactly_one_more_figure_call(entry_calls) -> None:
    base = fm.regplot(_DF, x="x", y="y")
    base.encode(y="y2").to_svg()
    base.encode(y="y2").to_svg()
    assert entry_calls == ["regplot", "regplot"]

    base.encode(x="y2").to_svg()
    base.encode(x="y2").to_svg()
    assert entry_calls == ["regplot", "regplot", "regplot"]


def test_a_grouping_and_an_input_rebuild_in_one_figure_call(entry_calls) -> None:
    """``.encode(y='y2', color='g')`` is one figure call, not one per kind of change."""
    chart = fm.regplot(_DF, x="x", y="y").encode(y="y2", color="g")
    first = chart.to_svg()
    chart.to_svg()
    assert entry_calls == ["regplot", "regplot"]
    door = fm.regplot(_DF, x="x", y="y2", hue="g")
    assert first == door.to_svg()


def test_a_same_column_restatement_calls_the_figure_no_more_than_the_build(entry_calls) -> None:
    fm.regplot(_DF, x="x", y="y").encode(y=fm.Y("y", title="T")).to_svg()
    assert entry_calls == ["regplot"]


def test_a_figure_rebuilt_on_another_column_does_not_reissue_its_build_warnings() -> None:
    """``show_metrics=True`` under a grouping warns once for the build it draws, not once per render."""
    chart = fm.regplot(_DF, x="x", y="y", show_metrics=True, hue="g").encode(y="y2")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart.to_svg()
        chart.to_svg()
        chart.to_json()
    assert [str(w.message) for w in caught] == [
        "regplot: show_metrics=True has no single fit to describe: the fit is "
        "computed per group of 'g', so the R²/RMSE/MAE corner is not drawn"
    ]


@pytest.mark.parametrize(
    ("name", "build", "later", "entry"),
    [
        ("residplot", lambda: fm.residplot(_DF, x="x", y="y"), {"y": "y2"}, "residplot"),
        ("lmplot", lambda: fm.lmplot(_DF, x="x", y="y", col="g"), {"y": "y2"}, "lmplot"),
        ("displot", lambda: fm.displot(_DF, x="x", kde=True), {"x": "y2"}, "displot"),
        ("catplot", lambda: fm.catplot(_DF, x="g", y="y", kind="point"), {"y": "y2"}, "catplot"),
    ],
)
def test_every_refit_figure_is_called_once_more_per_distinct_input(
    entry_calls, name, build, later, entry
) -> None:
    chart = build().encode(**later)
    chart.to_svg()
    chart.to_svg()
    chart.interactive().to_html()
    assert entry_calls == [entry, entry]


def test_a_figure_refit_beside_a_plus_operand_is_one_figure_call(entry_calls) -> None:
    """``figure.encode(y=...) + other`` resolves the refit once when it joins, then reuses it."""
    scatter = fm.Chart(_DF).mark_point().encode(x="x", y="y")
    chart = fm.regplot(_DF, x="x", y="y").encode(y="y2") + scatter
    chart.to_svg()
    chart.to_svg()
    chart.to_json()
    assert entry_calls == ["regplot", "regplot"]


def test_a_figure_operand_with_no_statistics_is_not_drawn_again_for_a_grouping() -> None:
    """A grouping calls the figure again for its fit; the points beside it are not rebuilt by it.

    The figure here reads its own ``encode=`` to style its points, so a points
    operand that was rebuilt for the grouping would draw at the other opacity.
    """
    import json

    from ferrum.plots._helpers import _figure_entry

    @_figure_entry
    def fit_over_points(data, *, x, y, encode=None):
        points = fm.Chart(data).mark_point(opacity=0.25 if encode else 1.0).encode(x=x, y=y)
        fit = fm.Chart(data).mark_smooth(ci=None).encode(x=x, y=y)
        chart = points + fit
        return chart.encode(**encode) if encode else chart

    grouped = fit_over_points(_DF, x="x", y="y").encode(color="g")
    spec = json.loads(grouped.to_spec().to_json())
    points = next(layer for layer in spec["layers"] if layer["mark"] == "point")
    assert points["mark_style"]["opacity"] == 1.0


@pytest.mark.parametrize(
    ("name", "build", "restated", "entry"),
    [
        (
            "catplot-horizontal-category",
            lambda: fm.catplot(_DF, y="g", x="y", kind="point", orient="h"),
            {"x": fm.X("g", title="T")},
            "catplot",
        ),
        (
            "catplot-horizontal-value",
            lambda: fm.catplot(_DF, y="g", x="y", kind="point", orient="h"),
            {"y": fm.Y("y", title="T")},
            "catplot",
        ),
        (
            "residplot-drawn-column-in-encode",
            lambda: fm.residplot(_DF, x="x", y="y", encode={"y": "residual"}),
            {"y": fm.Y("y", title="T")},
            "residplot",
        ),
        (
            "displot-overlay",
            lambda: fm.displot(_DF, x="x", kde=True),
            {"x": fm.X("x", title="T")},
            "displot",
        ),
    ],
    ids=[
        "catplot-horizontal-category",
        "catplot-horizontal-value",
        "residplot-encode-drawn",
        "displot-overlay",
    ],
)
def test_a_binding_restating_the_column_the_call_read_calls_the_figure_no_more_than_the_build(
    entry_calls, name, build, restated, entry
) -> None:
    """The input a figure declares is the column it read, in the chart's own space."""
    build().encode(**restated).to_svg()
    assert entry_calls == [entry]


@pytest.mark.parametrize(
    ("name", "restated"),
    [
        ("x-with-format", lambda: {"x": fm.X("x", format=".1f")}),
        ("y-with-axis", lambda: {"y": fm.Y("residual", axis=fm.Axis(label_angle=45))}),
    ],
    ids=["x-with-format", "y-with-axis"],
)
def test_a_binding_on_the_column_residplot_draws_carries_its_options_without_another_call(
    entry_calls, name, restated
) -> None:
    """The drawn column is the figure's own: an option written on it is read, not refit."""
    frame = _DF.rename({"x": "v"})
    fm.residplot(frame, x="v", y="y").encode(**restated()).to_svg()
    assert entry_calls == ["residplot"]
