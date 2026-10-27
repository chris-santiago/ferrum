"""Encode-path ``bin=`` lowering on the single-chart path (spec §6.6, F-L01-01).

A ``Bin`` transform replaces its input batch's schema outright with its own
``bin_start``/``bin_end``/``count``/``density`` columns (plus ``agg_value`` when
it aggregates).  The layered path has re-pointed a binning layer's encoding at
those columns since FA-4; the single-chart path never did, so *every* standalone
``bin=True`` idiom -- the canonical Vega-Lite histogram included -- raised at
render time against a column the Bin had already replaced.

These tests pin the lowering behaviorally, through ``.to_svg()`` and through the
plotted values, not only through the spec JSON: the spec-shape-only coverage the
FA-4 suite carried is exactly what let a docstring claim the standalone
histogram "produces valid output" while it raised for a year.

The per-bin values are read back with a ``mark_text`` oracle
(``text="agg_value"`` renders each bin's aggregate as an SVG text node), which
is exact and hand-checkable -- unlike bar geometry, which the scale's range
padding makes only approximately proportional to the value.
"""

from __future__ import annotations

import json
import re

import polars as pl
import pytest

import ferrum as fm
from ferrum._registry import registry_output_columns

#: Deterministic bins over ``_df()``'s ``v``: edges 1, 6, 11 -- so bin 0 holds
#: v=1..5 (w mean 30) and bin 1 holds v=6..10 (w mean 80), both hand-computable.
_TWO_BINS = {"bin_count": 2, "extent": (1.0, 11.0), "nice": False}

#: Deterministic bins over ``_null_df()``'s ``v``: edges 1, 5, 9 -- four rows
#: per bin, one of which has a null ``w``.
_TWO_BINS_NULL = {"bin_count": 2, "extent": (1.0, 9.0), "nice": False}

#: A fill nothing else in the SVG uses, so data bars are selectable by it.
_BAR_FILL = "#2f6f4e"


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "w": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        }
    )


def _null_df() -> pl.DataFrame:
    """Four rows per bin, one null ``w`` in each: rows-per-bin 4, non-null 3."""
    return pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "w": [10.0, None, 30.0, 40.0, 50.0, None, 70.0, 80.0],
        }
    )


def _grouped_df() -> pl.DataFrame:
    """``_df()`` twice over: group ``"b"`` carries double group ``"a"``'s ``w``."""
    base = _df()
    return pl.concat(
        [
            base.with_columns(pl.lit("a").alias("g")),
            base.with_columns(pl.lit("b").alias("g"), (pl.col("w") * 2.0).alias("w")),
        ]
    )


def _texts(svg: str) -> list[str]:
    """Every rendered text node, in document order (axis labels included)."""
    return re.findall(r"<text[^>]*>([^<]*)</text>", svg)


def _bars(svg: str) -> list[dict[str, float]]:
    """Every ``_BAR_FILL`` data rect, in document order."""
    out: list[dict[str, float]] = []
    for tag in re.findall(r"<rect[^>]*>", svg):
        if _BAR_FILL not in tag:
            continue
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag))
        out.append({k: float(attrs[k]) for k in ("x", "y", "width", "height")})
    return out


def _spec(chart) -> dict:
    return json.loads(chart.to_spec().to_json())


def _transforms(spec: dict, kind: str) -> list[dict]:
    return [t for t in (spec.get("transforms") or []) if t.get("type") == kind]


# ---------------------------------------------------------------------------
# The three F-L01-01 repros, verbatim: each raised at render time before
# ---------------------------------------------------------------------------


def test_bin_true_with_count_column_renders():
    """Repro 1: the canonical histogram idiom (``bin=True`` + ``y="count"``)."""
    chart = fm.Chart(_df()).mark_bar().encode(x=fm.X("v", bin=True), y="count")
    assert chart.to_svg().lstrip().startswith("<")


def test_bin_true_with_count_shorthand_renders():
    """Repro 2: ``y="count():Q"`` -- the pending count-aggregate is absorbed."""
    chart = fm.Chart(_df()).mark_bar().encode(x=fm.X("v", bin=True), y="count():Q")
    assert chart.to_svg().lstrip().startswith("<")


def test_bin_true_with_opposing_aggregate_renders():
    """Repro 3: ``bin=True`` + ``aggregate=`` on the opposing channel."""
    chart = fm.Chart(_df()).mark_bar().encode(x=fm.X("v", bin=True), y=fm.Y("w", aggregate="mean"))
    assert chart.to_svg().lstrip().startswith("<")


# ---------------------------------------------------------------------------
# What the lowering re-points, and to what
# ---------------------------------------------------------------------------


def test_binned_channel_points_at_bin_start_and_keeps_its_name_as_the_title():
    chart = fm.Chart(_df()).mark_bar().encode(x=fm.X("v", bin=True), y="count")
    enc = _spec(chart)["encoding"]
    assert enc["x"]["field"] == "bin_start"
    # Title recovery: the axis keeps reading "v", not "bin_start".
    assert enc["x"]["title"] == "v"


def test_binned_channel_gains_the_bin_end_span():
    chart = fm.Chart(_df()).mark_bar().encode(x=fm.X("v", bin=True), y="count")
    enc = _spec(chart)["encoding"]
    assert enc["x2"]["field"] == "bin_end"


def test_bars_span_their_bin_edge_to_edge():
    """The span channel is what makes binned bars touch: two bins over the full
    x extent must tile the plot area, not sit at a default width apart."""
    chart = fm.Chart(_df()).mark_bar(color=_BAR_FILL).encode(x=fm.X("v", bin=_TWO_BINS), y="count")
    bars = _bars(chart.to_svg())
    assert len(bars) == 2
    left, right = sorted(bars, key=lambda b: b["x"])
    assert left["width"] == pytest.approx(right["width"], rel=1e-6)
    # Contiguous: bin 0's right edge is bin 1's left edge.
    assert left["x"] + left["width"] == pytest.approx(right["x"], abs=0.01)


def test_explicit_span_channel_is_not_overwritten():
    """A user-bound x2 wins; the lowering only fills an empty span slot."""
    chart = fm.Chart(_df()).mark_bar().encode(x=fm.X("v", bin=True), x2=fm.X2("w"), y="count")
    assert _spec(chart)["encoding"]["x2"]["field"] == "w"


def test_an_explicit_title_beats_title_recovery():
    chart = (
        fm.Chart(_df()).mark_bar().encode(x=fm.X("v", bin=True, title="Value bucket"), y="count")
    )
    assert _spec(chart)["encoding"]["x"]["title"] == "Value bucket"


# ---------------------------------------------------------------------------
# The values actually plotted
# ---------------------------------------------------------------------------


def test_binned_mean_matches_hand_computed_per_bin_means():
    """Bins [1, 6) and [6, 11) hold w = 10..50 and 60..100: means 30 and 80.

    Discriminating: a count would plot 5 and 5, and the raw column would plot
    ten values, so neither passes this assertion.
    """
    chart = (
        fm.Chart(_df())
        .mark_text()
        .encode(
            x=fm.X("v", bin=_TWO_BINS),
            y=fm.Y("w", aggregate="mean"),
            text="agg_value",
        )
    )
    assert _texts(chart.to_svg())[-2:] == ["30", "80"]


@pytest.mark.parametrize(
    ("method", "expected"),
    [("sum", ["150", "400"]), ("max", ["50", "100"]), ("min", ["10", "60"])],
)
def test_every_folded_method_uses_the_same_lowering(method, expected):
    chart = (
        fm.Chart(_df())
        .mark_text()
        .encode(
            x=fm.X("v", bin=_TWO_BINS),
            y=fm.Y("w", aggregate=method),
            text="agg_value",
        )
    )
    assert _texts(chart.to_svg())[-2:] == expected


def test_count_shorthand_plots_the_bins_own_count_column():
    chart = (
        fm.Chart(_df()).mark_text().encode(x=fm.X("v", bin=_TWO_BINS), y="count():Q", text="count")
    )
    assert _texts(chart.to_svg())[-2:] == ["5", "5"]


def test_bin_with_aggregate_lowers_to_exactly_one_bin_and_no_aggregate():
    """The fold is the point: one Bin carrying agg_field/agg, not a Bin whose
    output a second Aggregate transform then tries to group by."""
    chart = fm.Chart(_df()).mark_bar().encode(x=fm.X("v", bin=True), y=fm.Y("w", aggregate="mean"))
    spec = _spec(chart)
    bins = _transforms(spec, "bin")
    assert len(bins) == 1
    assert bins[0]["agg_field"] == "w"
    assert bins[0]["agg"] == "mean"
    assert not _transforms(spec, "aggregate")
    assert spec["encoding"]["y"]["field"] == "agg_value"
    # The aggregated axis is titled by the aggregate's output name, as an
    # unbinned aggregate's is: "mean_w", never "agg_value".
    assert spec["encoding"]["y"]["title"] == "mean_w"


def test_fieldless_count_and_named_count_diverge_on_nulls():
    """``Y(None, aggregate="count")`` and ``Y("w", aggregate="count")`` are NOT
    the same statistic, and the difference is observable.

    The field-less form is rows per bin: it maps to the Bin's own count column,
    which counts rows -- 4 and 4 here.  The named form folds to the Bin's
    aggregate column with ``agg="count"``, which is the *Aggregate* vocabulary's
    count of NON-NULL values of ``w`` -- 3 and 3, because each bin holds one
    null.  Spec §6.6 prescribes exactly this split, and it was documented but
    unpinned; without this test a future "these two are the same thing"
    simplification would be invisible.
    """
    rows = (
        fm.Chart(_null_df())
        .mark_text()
        .encode(
            x=fm.X("v", bin=_TWO_BINS_NULL),
            y=fm.Y(None, aggregate="count"),
            text="count",
        )
    )
    non_null = (
        fm.Chart(_null_df())
        .mark_text()
        .encode(
            x=fm.X("v", bin=_TWO_BINS_NULL),
            y=fm.Y("w", aggregate="count"),
            text="agg_value",
        )
    )

    assert _spec(rows)["encoding"]["y"]["field"] == "count"
    assert not _transforms(_spec(rows), "bin")[0].get("agg")
    assert _texts(rows.to_svg())[-2:] == ["4", "4"]

    non_null_bin = _transforms(_spec(non_null), "bin")[0]
    assert _spec(non_null)["encoding"]["y"]["field"] == "agg_value"
    assert (non_null_bin["agg_field"], non_null_bin["agg"]) == ("w", "count")
    assert _texts(non_null.to_svg())[-2:] == ["3", "3"]


def test_count_shorthand_absorbs_its_pending_aggregate():
    chart = fm.Chart(_df()).mark_bar().encode(x=fm.X("v", bin=True), y="count():Q")
    spec = _spec(chart)
    assert not _transforms(spec, "aggregate")
    assert spec["encoding"]["y"]["field"] == "count"


# ---------------------------------------------------------------------------
# Horizontal variants -- Y(bin=True) mirrors X(bin=True)
# ---------------------------------------------------------------------------


def test_horizontal_bin_lowers_to_y_and_the_y2_span():
    chart = fm.Chart(_df()).mark_bar().encode(y=fm.Y("v", bin=True), x="count")
    enc = _spec(chart)["encoding"]
    assert enc["y"]["field"] == "bin_start"
    assert enc["y"]["title"] == "v"
    assert enc["y2"]["field"] == "bin_end"
    assert "x2" not in enc
    assert chart.to_svg().lstrip().startswith("<")


def test_horizontal_bin_with_opposing_aggregate_matches_the_vertical_values():
    chart = (
        fm.Chart(_df())
        .mark_text()
        .encode(
            y=fm.Y("v", bin=_TWO_BINS),
            x=fm.X("w", aggregate="mean"),
            text="agg_value",
        )
    )
    spec = _spec(chart)
    assert spec["encoding"]["x"]["field"] == "agg_value"
    assert _transforms(spec, "bin")[0]["agg"] == "mean"
    assert sorted(_texts(chart.to_svg())[-2:]) == ["30", "80"]


# ---------------------------------------------------------------------------
# Grouped and faceted
# ---------------------------------------------------------------------------


def test_grouped_bin_with_aggregate_computes_per_group_bin_means():
    """``bin={"groupby": [...]}`` keeps each group's own per-bin mean.

    Group "b" carries double group "a"'s ``w``, so its bin means must be 60 and
    160 against "a"'s 30 and 80 -- a global aggregate would plot one pair.

    Asserted on the trailing MARK nodes, not a set-subset over every text node:
    the y-axis for this chart ticks 30, 40, ... 160, so all four expected values
    appear as tick labels and a subset assertion would survive a chart that
    plots no marks at all.  No ``color=`` here for the same reason -- a legend
    block appends its own text nodes after the marks.
    """
    chart = (
        fm.Chart(_grouped_df())
        .mark_text()
        .encode(
            x=fm.X("v", bin={**_TWO_BINS, "groupby": ["g"]}),
            y=fm.Y("w", aggregate="mean"),
            text="agg_value",
        )
    )
    assert _texts(chart.to_svg())[-4:] == ["30", "80", "60", "160"]


def test_faceted_bin_with_aggregate_computes_each_panels_own_means():
    """Each facet panel plots its own group's per-bin means.

    Anchored on each panel's strip label, which no tick label can equal, so the
    assertion reads the marks rather than the axis.
    """
    chart = (
        fm.Chart(_grouped_df())
        .mark_text()
        .encode(
            x=fm.X("v", bin={**_TWO_BINS, "groupby": ["g"]}),
            y=fm.Y("w", aggregate="mean"),
            text="agg_value",
        )
        .facet(col="g")
    )
    texts = _texts(chart.to_svg())

    def _marks_of(panel: str) -> list[str]:
        assert panel in texts, f"no strip label {panel!r} in {texts}"
        start = texts.index(panel) + 1
        return texts[start : start + 2]

    assert _marks_of("a") == ["30", "80"]
    assert _marks_of("b") == ["60", "160"]


# ---------------------------------------------------------------------------
# Refusals -- every one of these produced an obscure render-time error before
# ---------------------------------------------------------------------------


def test_unknown_aggregate_method_refuses_with_the_bin_vocabulary():
    chart = fm.Chart(_df()).mark_bar().encode(x=fm.X("v", bin=True), y=fm.Y("w", aggregate="bogus"))
    with pytest.raises(ValueError, match=r"unknown agg 'bogus'.*mean\|sum\|count"):
        chart.to_spec()


def test_bin_and_aggregate_on_the_same_channel_refuses():
    chart = fm.Chart(_df()).mark_bar().encode(x=fm.X("v", bin=True, aggregate="mean"), y="count")
    with pytest.raises(ValueError, match="both bin= and aggregate="):
        chart.to_spec()


def test_two_binned_channels_refuse_and_name_the_2d_binning_surface():
    chart = fm.Chart(_df()).mark_bar().encode(x=fm.X("v", bin=True), y=fm.Y("w", bin=True))
    with pytest.raises(ValueError, match=r"more than one bin=.*Bin2D\(x=\.\.\., y=\.\.\.\)"):
        chart.to_spec()


def test_the_multi_bin_refusals_named_remedy_is_a_real_api():
    """The refusal points at 2-D binning; that surface has to exist and work.

    An earlier revision named ``transform_bin_2d()``, which is not an API this
    library has ever exposed.  A refusal that prescribes a phantom remedy is
    worse than one that prescribes nothing, so the remedy is executed here.
    """
    _bin2d = registry_output_columns("bin_2d", ["x_lo", "x_hi", "y_lo", "y_hi", "count"])
    chart = (
        fm.Chart(_df())
        .transform(fm.Bin2D(x="v", y="w"))
        .mark_rect()
        .encode(
            x=_bin2d["x_lo"],
            x2=_bin2d["x_hi"],
            y=_bin2d["y_lo"],
            y2=_bin2d["y_hi"],
            color=_bin2d["count"],
        )
    )
    assert chart.to_svg().lstrip().startswith("<")


def test_a_binned_channel_carrying_an_aggregate_refuses_beside_a_second_aggregate():
    """The binned channel is the bin position, never an aggregated value, whatever aggregates beside it."""
    chart = (
        fm.Chart(_df())
        .mark_bar()
        .encode(x=fm.X("v", bin=True, aggregate="mean"), y=fm.Y("w", aggregate="sum"))
    )
    with pytest.raises(ValueError, match=r"channel 'x' carries both bin= and aggregate="):
        chart.to_spec()


def test_an_aggregate_on_a_channel_that_draws_no_value_is_refused():
    """``_honored_kwargs`` is the contract: Detail draws no aggregated value, so
    its ``aggregate=`` is refused before any Bin or Aggregate could read it."""
    with pytest.raises(ValueError) as exc:
        fm.Detail("w", aggregate="mean")
    assert str(exc.value) == (
        "Detail(aggregate='mean') is not supported: Detail draws no aggregated value. "
        "Compute the aggregate with .transform(fm.transform_aggregate(...)) and bind its "
        "output column, or drop aggregate=."
    )


def _per_bin_means() -> list[float]:
    """``w``'s mean over each of ``_TWO_BINS``' two bins, computed independently."""
    frame = _df()
    return [
        frame.filter(pl.col("v") < 6.0)["w"].mean(),
        frame.filter(pl.col("v") >= 6.0)["w"].mean(),
    ]


def _titles(svg: str) -> list[str]:
    return re.findall(r"<title>([^<]*)</title>", svg)


def test_a_tooltip_aggregate_beside_a_bin_is_computed_per_bin():
    """A value channel's aggregate folds into the Bin, which computes it per bin."""
    chart = (
        fm.Chart(_df())
        .mark_bar()
        .encode(x=fm.X("v", bin=_TWO_BINS), y="count()", tooltip=fm.Tooltip("w", aggregate="mean"))
    )
    spec = _spec(chart)
    bin_xform = _transforms(spec, "bin")[0]
    assert (bin_xform["agg_field"], bin_xform["agg"]) == ("w", "mean")
    assert spec["encoding"]["tooltip"] == {"field": "agg_value"}
    assert spec["encoding"]["y"]["field"] == "count"
    assert _titles(chart.to_svg()) == [f"agg_value: {m:g}" for m in _per_bin_means()]


def test_two_channels_aggregating_the_same_field_share_the_bins_aggregate():
    chart = (
        fm.Chart(_df())
        .mark_text()
        .encode(
            x=fm.X("v", bin=_TWO_BINS),
            y=fm.Y("w", aggregate="mean"),
            text=fm.Text("w", aggregate="mean"),
        )
    )
    spec = _spec(chart)
    assert len(_transforms(spec, "bin")) == 1
    assert spec["encoding"]["y"]["field"] == spec["encoding"]["text"]["field"] == "agg_value"
    texts = re.findall(r"<text[^>]*>([^<]*)</text>", chart.to_svg())
    # The labels are the last text the chart draws, after its axes.
    assert texts[-2:] == [f"{m:g}" for m in _per_bin_means()]


def test_two_different_aggregates_beside_a_bin_are_both_computed_per_bin():
    """Two distinct aggregates fold into the one Bin through ``aggregates``, each drawn from its own column."""
    chart = (
        fm.Chart(_df())
        .mark_text()
        .encode(
            x=fm.X("v", bin=_TWO_BINS),
            y=fm.Y("w", aggregate="mean"),
            text=fm.Text("w", aggregate="sum"),
        )
    )
    spec = _spec(chart)
    (bin_xform,) = _transforms(spec, "bin")
    assert bin_xform["aggregates"] == [
        {"field": "w", "fn": "mean", "as": "mean_w"},
        {"field": "w", "fn": "sum", "as": "sum_w"},
    ]
    assert "agg_field" not in bin_xform
    assert (spec["encoding"]["y"]["field"], spec["encoding"]["text"]["field"]) == (
        "mean_w",
        "sum_w",
    )
    frame = _df()
    sums = [frame.filter(pl.col("v") < 6.0)["w"].sum(), frame.filter(pl.col("v") >= 6.0)["w"].sum()]
    assert _texts(chart.to_svg())[-2:] == [f"{s:g}" for s in sums]


def test_a_grouped_bin_folds_its_aggregate_per_bin_and_group():
    frame = _df().with_columns(pl.Series("g", ["a", "b"] * 5))
    chart = (
        fm.Chart(frame)
        .mark_bar()
        .encode(x=fm.X("v", bin=_TWO_BINS), y=fm.Y("w", aggregate="mean"), color="g")
    )
    bin_xform = _transforms(_spec(chart), "bin")[0]
    assert bin_xform["groupby"] == ["g"]
    assert (bin_xform["agg_field"], bin_xform["agg"]) == ("w", "mean")
    assert chart.to_svg().count("<rect") >= 4


def test_a_binned_color_keeps_every_row_and_adds_no_bin_step():
    """``Color(bin=True)`` maps its field through a binned scale over the unchanged rows.

    Rejects the row-replacing ``Bin`` it used to lower to, which dropped the
    column ``y`` reads and refused with ``unknown column 'w'``.
    """
    chart = fm.Chart(_df()).mark_point().encode(x="v", y="w", color=fm.Color("w", bin=True))
    spec = _spec(chart)
    assert _transforms(spec, "bin") == []
    assert spec["encoding"]["color"]["field"] == "w"
    assert spec["encoding"]["color"]["scale"]["type"] == "bin-ordinal"
    assert chart.to_svg().count("<circle") == _df().height


def test_a_color_aggregate_beside_a_bin_is_computed_per_bin():
    """A ``Color(aggregate=...)`` beside a ``bin=`` on x folds into the Bin's
    aggregate, so each bar is colored by its bin's mean."""
    chart = (
        fm.Chart(_df())
        .mark_bar()
        .encode(x=fm.X("v", bin=_TWO_BINS), y="count", color=fm.Color("w", aggregate="mean"))
    )
    spec = _spec(chart)
    bin_xform = _transforms(spec, "bin")[0]
    assert (bin_xform["agg_field"], bin_xform["agg"]) == ("w", "mean")
    assert spec["encoding"]["color"]["field"] == "agg_value"
    fills = re.findall(r'<rect [^>]*fill="(#[0-9a-f]{6})"', chart.to_svg())
    assert len(set(fills[-2:])) == len(set(_per_bin_means())) == 2


def test_prebuilt_bin_instance_with_an_opposing_aggregate_refuses_and_names_the_fix():
    from ferrum._core import Bin as RustBin

    chart = (
        fm.Chart(_df())
        .mark_bar()
        .encode(x=fm.X("v", bin=RustBin("v", bin_count=2)), y=fm.Y("w", aggregate="mean"))
    )
    with pytest.raises(ValueError, match=r"agg_field='w'.*agg='mean'"):
        chart.to_spec()


def test_prebuilt_bin_instance_carrying_its_own_aggregate_works():
    """The refusal above names a fix that must actually work."""
    from ferrum._core import Bin as RustBin

    chart = (
        fm.Chart(_df())
        .mark_text()
        .encode(
            x=fm.X(
                "v",
                bin=RustBin(
                    "v", bin_count=2, extent=(1.0, 11.0), nice=False, agg_field="w", agg="mean"
                ),
            ),
            y="agg_value",
            text="agg_value",
        )
    )
    assert _texts(chart.to_svg())[-2:] == ["30", "80"]


def test_fieldless_non_count_aggregate_refuses():
    chart = fm.Chart(_df()).mark_bar().encode(x=fm.X("v", bin=True), y=fm.Y(None, aggregate="sum"))
    with pytest.raises(ValueError, match="has no source column"):
        chart.to_spec()


# ---------------------------------------------------------------------------
# Byte-stability guard: a chart that declares no bin is untouched
# ---------------------------------------------------------------------------


def test_chart_without_bin_gains_no_span_and_no_title():
    chart = fm.Chart(_df()).mark_bar().encode(x="v", y="w")
    enc = _spec(chart)["encoding"]
    assert "x2" not in enc
    assert "title" not in enc["x"]
    assert enc["x"]["field"] == "v"


def test_plain_aggregate_without_bin_still_builds_an_aggregate_transform():
    """The bin lowering must not swallow an aggregate on an unbinned chart."""
    chart = fm.Chart(_df()).mark_bar().encode(x="v", y=fm.Y("w", aggregate="mean"))
    spec = _spec(chart)
    assert len(_transforms(spec, "aggregate")) == 1
    assert not _transforms(spec, "bin")
    assert spec["encoding"]["y"]["field"] == "mean_w"


# ---------------------------------------------------------------------------
# A positional bin with no aggregate bins each row in place
# ---------------------------------------------------------------------------


def test_a_prebuilt_bin_without_an_aggregate_places_each_row_at_its_bin():
    """A pre-built ``bin=Bin(...)`` keeps its parameters and bins each row, as ``bin={...}`` does."""
    from ferrum._core import Bin as RustBin

    prebuilt = (
        fm.Chart(_df()).mark_point().encode(x=fm.X("v", bin=RustBin("v", bin_count=2)), y="w")
    )
    spelled = fm.Chart(_df()).mark_point().encode(x=fm.X("v", bin={"bin_count": 2}), y="w")
    (bin_xform,) = _transforms(_spec(prebuilt), "bin")
    assert (bin_xform["bin_count"], bin_xform["per_row"]) == (2, True)
    svg = prebuilt.to_svg()
    assert svg == spelled.to_svg()
    assert svg.count("<circle") == _df().height


def test_a_bin_a_later_encode_replaced_computes_nothing():
    """The encoding decides the statistic: a ``bin=`` channel a later ``encode()`` replaced bins nothing."""
    replaced = fm.Chart(_df()).mark_point().encode(x=fm.X("v", bin=True), y="w").encode(x="v")
    assert not _transforms(_spec(replaced), "bin")
    assert replaced.to_svg() == fm.Chart(_df()).mark_point().encode(x="v", y="w").to_svg()


def test_an_aggregate_a_later_encode_replaced_computes_nothing():
    replaced = fm.Chart(_df()).mark_point().encode(x="v", y="mean(w)").encode(y="w")
    assert not _transforms(_spec(replaced), "aggregate")
    assert replaced.to_svg() == fm.Chart(_df()).mark_point().encode(x="v", y="w").to_svg()


# ---------------------------------------------------------------------------
# The statistic plan's step, and the one reader of a transform's wire
# ---------------------------------------------------------------------------


def test_a_plan_computing_nothing_has_no_step():
    from ferrum._encode_lowering import _statistic_plan

    plan = _statistic_plan({"x": fm.X("v"), "y": fm.Y("w")}, None)
    assert not plan.computes
    with pytest.raises(RuntimeError, match="computes no statistic"):
        plan.step()


def test_a_pre_built_bin_the_plan_does_not_extend_is_sent_as_built():
    """The Bin object itself reaches the transform list when the plan adds no parameter."""
    from ferrum._core import Bin as RustBin
    from ferrum._encode_lowering import _statistic_plan

    prebuilt = RustBin("v", bin_count=2)
    plan = _statistic_plan({"x": fm.X("v", bin=prebuilt), "y": fm.Y("count")}, None)
    assert plan.step() is prebuilt


def test_a_pre_built_bin_the_plan_groups_is_sent_as_its_wire_with_the_groups_added():
    """The plan's own group joins the Bin's declared groupby, which stays marked written.

    ``g`` is the caller's own ``groupby=`` on the ``Bin``; ``h`` is the
    planner's addition from the ``color`` channel. Only ``g`` is
    ``written_groupby``: a line reading this Bin's output splits by the
    group the caller wrote, never by one the planner added.
    """
    from ferrum._core import Bin as RustBin
    from ferrum._encode_lowering import _statistic_plan
    from ferrum._registry import transform_wires

    prebuilt = RustBin("v", bin_count=2, groupby=["g"])
    plan = _statistic_plan({"x": fm.X("v", bin=prebuilt), "y": "count", "color": "h"}, None)
    (wire,) = transform_wires([prebuilt])
    assert plan.step() == {
        **wire,
        "groupby": ["g", "h"],
        "shared_extent": True,
        "written_groupby": ["g"],
    }


def test_the_transform_wire_reader_reads_a_bins_own_parameters():
    from ferrum._registry import transform_wires

    (wire,) = transform_wires(
        [fm.Bin("v", bin_count=3, aggregates=[fm.AggregateOp("w", "mean", "mean_w")])]
    )
    assert wire["bin_count"] == 3
    assert wire["aggregates"] == [{"field": "w", "fn": "mean", "as": "mean_w"}]


@pytest.mark.parametrize(
    "bin_",
    [
        True,
        {"groupby": ["g"]},
        {"agg_field": "w", "agg": "mean"},
        {"aggregates": [fm.AggregateOp("w", "mean", "mean_w")]},
        "pre-built",
    ],
    ids=["true", "dict-groupby", "dict-agg", "dict-aggregates", "pre-built"],
)
def test_a_bins_declared_outputs_are_the_columns_rust_says_it_computes(bin_):
    """One output set, Rust's, drives the refusal text, the per-row test and the groupby."""
    from ferrum._core import Bin as RustBin
    from ferrum._core import bin_output_columns
    from ferrum._encode_lowering import _bin_declaration

    if bin_ == "pre-built":
        bin_ = RustBin("v", aggregates=[fm.AggregateOp("w", "sum", "sum_w")])
        built = bin_
    elif bin_ is True:
        built = RustBin("v")
    else:
        built = RustBin("v", **bin_)
    declaration = _bin_declaration("x", fm.X("v", bin=bin_))
    assert declaration.outputs == frozenset(bin_output_columns(built))


def test_a_raw_column_named_like_an_aggregate_a_bin_does_not_compute_groups_the_bin():
    """A counting Bin computes no ``agg_value``, so a data column of that name is a group key."""
    frame = _df().with_columns(pl.Series("agg_value", ["a", "b"] * 5))
    chart = (
        fm.Chart(frame).mark_bar().encode(x=fm.X("v", bin=_TWO_BINS), y="count", color="agg_value")
    )
    (bin_xform,) = _transforms(_spec(chart), "bin")
    assert bin_xform["groupby"] == ["agg_value"]
