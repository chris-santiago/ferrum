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
