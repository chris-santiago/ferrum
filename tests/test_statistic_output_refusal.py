"""A ``.transform()`` step reading a column only a chart's statistics compute is refused
naming the statistics, the columns they compute, and the remedy, on the flat, static
``LayerChart``-member and ``+`` doors. A column matching a statistic's output exactly
keeps its own more specific text, and a chain with no statistics keeps the generic
"not found" text.
"""

import json

import polars as pl
import pytest

import ferrum as fm

#: Shared rows: two groups, five rows, matching the Rust transform tests this pins.
_ROWS = pl.DataFrame({"g": ["a", "a", "b", "b", "b"], "v": [1.0, 2.0, 3.0, 4.0, 5.0]})

#: The composite render's envelope around its first leaf's refusal.
_FIRST_LEAF = "failed to render composite leaf leaf #0: "


def _count_bar() -> fm.Chart:
    """A bar chart counting rows per group; its real output column is ``count_all``."""
    return fm.Chart(_ROWS).mark_bar().encode(x="g", y="count()")


def _hint(statistics: str, outputs: str, remedy: str) -> str:
    """The clause every refusal of a column only a chart's statistics compute shares."""
    return (
        f"The chart's statistics ({statistics}) compute {outputs} after its "
        f".transform() steps, so .transform() cannot read them: {remedy}"
    )


_FLAT_REMEDY = (
    "compute the value in .transform() before this step (e.g. with fm.transform_aggregate) "
    "and bind the encoding to it"
)


def _composed_remedy(outputs: str) -> str:
    """The ``+`` door's own remedy: name one of the readable statistic columns directly."""
    return (
        "a step written on the composition runs after an operand's statistics only where it "
        f"names one of their columns: name one of {outputs}"
    )


def test_a_flat_charts_near_miss_column_is_refused_with_the_remedy():
    """``y=count()``'s real output is ``count_all``, so a filter reading ``count`` is not an
    exact match: the generic not-found text gains the statistics hint naming ``count_all``.
    """
    chart = _count_bar().transform(fm.transform_filter("datum.count > 1"))
    with pytest.raises(ValueError) as refused:
        chart.to_svg()
    assert str(refused.value) == (
        "transform failed: ValueError: data_filter: column 'count' not found; "
        'available: ["g", "v"]. ' + _hint("stat_aggregate", "['count_all']", _FLAT_REMEDY)
    )


def test_a_static_layerchart_members_near_miss_column_carries_the_same_text_in_the_leaf_envelope():
    """The same typed refusal as the flat door, wrapped in the composite leaf envelope."""
    bar = _count_bar().transform(fm.transform_filter("datum.count > 1"))
    other = fm.Chart(_ROWS).mark_point().encode(x="g", y="v")
    with pytest.raises(ValueError) as refused:
        fm.LayerChart(bar, other).to_svg()
    assert str(refused.value) == (
        _FIRST_LEAF + "transform failed: ValueError: data_filter: column 'count' not found; "
        'available: ["g", "v"]. ' + _hint("stat_aggregate", "['count_all']", _FLAT_REMEDY)
    )


def test_a_plus_compositions_step_naming_no_operands_column_names_every_reached_chains_statistics():
    """``refuse_unrun``: no operand's rows ever carry ``count``, so the refusal names the
    union of every reached operand's statistics, with the composed door's own remedy.
    """
    bar = _count_bar()
    other = fm.Chart(_ROWS).mark_point().encode(x="g", y="v")
    combo = (bar + other).transform(fm.transform_filter("datum.count > 1"))
    with pytest.raises(ValueError) as refused:
        combo.to_svg()
    assert str(refused.value) == (
        "transform failed: ValueError: data_filter: column 'count' not found; "
        'available: ["g", "v"]. '
        + _hint("stat_aggregate", "['count_all']", _composed_remedy("['count_all']"))
    )


@pytest.mark.parametrize(
    ("make_other", "available"),
    [
        pytest.param(
            lambda: fm.Chart(_ROWS).mark_point().encode(x="g", y="v"),
            '["g", "v"]',
            id="plain",
        ),
        pytest.param(
            lambda: (
                fm.Chart(_ROWS)
                .mark_point()
                .encode(x="g", y="v")
                .transform(fm.transform_calculate("w", "datum.v * 2"))
            ),
            '["g", "v", "w"]',
            id="with-a-prior-transform-on-the-other-operand",
        ),
    ],
)
def test_a_plus_compositions_hint_is_order_independent(make_other, available):
    """The statistic operand being second still names its statistics: the union does not
    depend on which operand a copy meets first. ``other``, the non-statistic left operand,
    may carry a prior ``.transform()`` of its own, and the ``available`` half then reads
    its output columns, not the statistic operand's.
    """
    combo = (make_other() + _count_bar()).transform(fm.transform_filter("datum.count > 1"))
    with pytest.raises(ValueError) as refused:
        combo.to_svg()
    assert str(refused.value) == (
        "transform failed: ValueError: data_filter: column 'count' not found; "
        f"available: {available}. "
        + _hint("stat_aggregate", "['count_all']", _composed_remedy("['count_all']"))
    )


def test_a_plus_compositions_step_over_a_distinct_frame_operand_keeps_its_plain_text():
    """A step on a ``+`` composition never reaches an operand over a different frame, so a
    statistic that operand carries is never named: naming it would still not let the step
    read it. The refusal stays the plain not-found text, over the reached operand's own rows.
    """
    other = (
        fm.Chart(pl.DataFrame({"g": ["a", "b"], "v": [10.0, 20.0]}))
        .mark_point()
        .encode(x="g", y="v")
    )
    combo = (other + _count_bar()).transform(fm.transform_filter("datum.count > 1"))
    with pytest.raises(ValueError) as refused:
        combo.to_svg()
    assert (
        str(refused.value)
        == 'transform failed: ValueError: data_filter: column \'count\' not found; available: ["g", "v"]'
    )


@pytest.mark.parametrize(
    ("make_chart", "column", "statistics"),
    [
        pytest.param(
            lambda: fm.Chart(_ROWS).mark_bar().encode(x="g", y="mean(v)"),
            "mean_v",
            "stat_aggregate",
            id="mean_v",
        ),
        pytest.param(
            lambda: fm.Chart(_ROWS).mark_histogram().encode(x="v"),
            "count",
            "stat_bin",
            id="histogram_count",
        ),
        pytest.param(
            lambda: fm.Chart(_ROWS).mark_bar().encode(x="g", y="count()"),
            "count_all",
            "stat_aggregate",
            id="count_all",
        ),
    ],
)
def test_a_flat_charts_exact_statistic_output_read_keeps_its_own_text(
    make_chart, column, statistics
):
    """A column matching a statistic's output exactly keeps its specific text: naming the
    statistic and remedy up front, with no generic not-found text or hint appended beside it.
    """
    chart = make_chart().transform(fm.transform_filter(f"datum.{column} > 1"))
    with pytest.raises(ValueError) as refused:
        chart.to_svg()
    assert str(refused.value) == (
        f"transform failed: ValueError: data_filter: column '{column}' is computed by the "
        f"chart's statistics ({statistics}), which run after the chart's .transform() steps: "
        f".transform() reads the chart's rows before its statistics. Compute '{column}' in "
        ".transform() before this step (e.g. with fm.transform_aggregate) and bind the "
        "encoding to it"
    )


def test_a_flat_charts_two_statistic_outputs_name_one_label_and_both_columns():
    """One ``Aggregate`` computing two encoding statistics (``mean(v)`` and ``count()``)
    names its label once, with both output columns listed, in the order the chart builds them.
    """
    chart = (
        fm.Chart(_ROWS)
        .mark_bar()
        .encode(x="g", y="mean(v)", color="count()")
        .transform(fm.transform_filter("datum.zzz > 1"))
    )
    with pytest.raises(ValueError) as refused:
        chart.to_svg()
    assert str(refused.value) == (
        "transform failed: ValueError: data_filter: column 'zzz' not found; "
        'available: ["g", "v"]. ' + _hint("stat_aggregate", "['mean_v', 'count_all']", _FLAT_REMEDY)
    )


def test_a_plus_compositions_hint_unions_two_different_statistics():
    """A histogram operand (``stat_bin``) and a counting bar operand (``stat_aggregate``)
    both contribute to the union: both labels and the deduplicated columns of both are named.
    """
    hist = fm.Chart(_ROWS).mark_histogram().encode(x="v")
    combo = (hist + _count_bar()).transform(fm.transform_filter("datum.zzz > 1"))
    with pytest.raises(ValueError) as refused:
        combo.to_svg()
    outputs = "['bin_start', 'bin_end', 'count', 'density', 'count_all']"
    assert str(refused.value) == (
        "transform failed: ValueError: data_filter: column 'zzz' not found; "
        'available: ["g", "v"]. '
        + _hint("stat_bin, stat_aggregate", outputs, _composed_remedy(outputs))
    )


def _tooltip_records(chart: fm.Chart) -> list[dict[str, object]]:
    """Every mark's tooltip record, read off the interactive scene, across every panel."""
    scene = json.loads(chart.interactive().scene_json)
    records = []
    for panel in scene["panels"]:
        for batch in panel["marks"]:
            for tooltip in batch.get("tooltips") or []:
                records.append({field["name"]: field["value"] for field in tooltip["fields"]})
    return records


def _bar_counts(chart: fm.Chart) -> list[float]:
    """Each bar's ``count``, read off its tooltip record."""
    return sorted(float(record["count"]) for record in _tooltip_records(chart) if "count" in record)


def test_a_plus_compositions_step_naming_a_readable_statistic_column_still_runs():
    """A histogram's real ``count`` column, read by a step on a ``+`` composition, still
    filters the histogram's bars: the refusal never fires for a column that is genuinely
    readable somewhere in the lineage.
    """
    hist = fm.Chart(_ROWS).mark_histogram().encode(x="v")
    other = fm.Chart(_ROWS).mark_point().encode(x="v", y="v")
    combo = (hist + other).transform(fm.transform_filter("datum.count > 1"))
    assert _bar_counts(combo) == [2.0]


def test_an_unknown_column_in_a_chain_with_no_statistics_keeps_the_plain_text():
    """A chart with no statistics at all keeps the step's own plain not-found text: there
    is nothing to hint at, so the refusal gains no statistics clause.
    """
    chart = (
        fm.Chart(_ROWS)
        .mark_point()
        .encode(x="g", y="v")
        .transform(fm.transform_filter("datum.zzz > 1"))
    )
    with pytest.raises(ValueError) as refused:
        chart.to_svg()
    assert (
        str(refused.value)
        == 'transform failed: ValueError: data_filter: column \'zzz\' not found; available: ["g", "v"]'
    )


def test_a_typo_column_in_a_chain_with_statistics_gets_the_plain_text_plus_the_hint():
    """A column matching none of the statistics' own outputs still names them, with the
    remedy, even though it is an arbitrary typo rather than a near-miss of a real output.
    """
    chart = _count_bar().transform(fm.transform_filter("datum.zzz > 1"))
    with pytest.raises(ValueError) as refused:
        chart.to_svg()
    assert str(refused.value) == (
        "transform failed: ValueError: data_filter: column 'zzz' not found; "
        'available: ["g", "v"]. ' + _hint("stat_aggregate", "['count_all']", _FLAT_REMEDY)
    )


def test_a_readable_filter_before_a_count_runs_unchanged():
    """A column the rows do carry (``v``) filters before the statistics run, exactly as a
    step with nothing to do with any statistic's output would.
    """
    chart = _count_bar().transform(fm.transform_filter("datum.v > 1"))
    counts = {record["g"]: float(record["count_all"]) for record in _tooltip_records(chart)}
    assert counts == {"a": 1.0, "b": 3.0}
