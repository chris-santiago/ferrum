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
import warnings

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
    # Title recovery on the aggregated axis too: it reads "w", not "agg_value".
    assert spec["encoding"]["y"]["title"] == "w"


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


def test_two_aggregating_channels_on_a_binned_encoding_refuse():
    """Triggered through channels that HONOR ``aggregate=`` -- only X and Y do.

    An earlier revision pinned this with ``size=Size(..., aggregate=...)``, a
    kwarg Size does not honor, so the refusal's only proof was an input the
    library documents as warned-and-dropped.  This arm is checked before the
    same-channel refusal precisely so it stays reachable: two honored aggregate
    channels are necessarily X and Y, which necessarily includes the binned one.
    """
    chart = (
        fm.Chart(_df())
        .mark_bar()
        .encode(x=fm.X("v", bin=True, aggregate="mean"), y=fm.Y("w", aggregate="sum"))
    )
    with pytest.raises(ValueError, match=r"aggregates on more than one channel \(x, y\)"):
        chart.to_spec()


def test_an_unhonored_aggregate_does_not_steer_the_bin():
    """``_honored_kwargs`` is the contract: an unhonored ``aggregate=`` is
    warned-and-dropped, so it must not reach in and set the Bin's
    ``agg_field``/``agg`` nor get re-pointed at the aggregate column.

    Tooltip honors only ``type``.  Before the gate, this chart emitted a Bin
    carrying ``agg_field="w", agg="mean"`` and a tooltip bound to ``agg_value``
    -- a kwarg the library says it ignores, silently rewriting the pipeline.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        chart = (
            fm.Chart(_df())
            .mark_bar()
            .encode(
                x=fm.X("v", bin=True),
                y="count",
                tooltip=fm.Tooltip("w", aggregate="mean"),
            )
        )
    spec = _spec(chart)
    bin_xform = _transforms(spec, "bin")[0]
    assert "agg_field" not in bin_xform
    assert "agg" not in bin_xform
    assert spec["encoding"]["tooltip"]["field"] != "agg_value"


def test_unhonored_bin_on_color_fails_with_unknown_column():
    """The honored-kwarg gate is half-enforced (batch-E close remediation,
    design S2): ``to_implicit_transforms()`` emits a ``_PendingBin`` for
    ANY channel's ``bin=``, honored or not, so an unhonored ``Color(bin=True)``
    still puts a Bin transform on the chart's pipeline and re-points the
    color channel at the Bin's own output column (``_lower_bin_encoding``'s
    ``bin_axes`` scan is deliberately not honored-gated either -- see its
    docstring). The Bin then replaces the WHOLE batch schema with its own
    columns, dropping the ``x``/``y`` columns the rest of the chart still
    reads -- the resulting failure names the unrelated `x` column as
    "unknown", not `color=`'s bin=, which is confusing but is documented,
    pre-existing, current behavior (not a regression this batch introduces).
    This pins that CURRENT failure mode so the earmarked sentinel-bypass
    follow-up (closing the gate at ``to_implicit_transforms``) has a visible
    blast radius to fix, rather than an invisible one."""
    chart = fm.Chart(_df()).mark_point().encode(x="v", y="w", color=fm.Color("w", bin=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match=r"unknown column 'v'"):
            chart.to_svg()


def test_unhonored_aggregate_on_color_beside_a_bin_fails_with_column_not_found():
    """The other half of the same gate asymmetry: an unhonored
    ``Color(aggregate=...)`` beside a real ``bin=`` on x is excluded from
    ``_lower_bin_encoding``'s fold (Color does not honor ``aggregate=``), so
    it is never absorbed into the Bin -- but its ``_PendingAggregate``
    sentinel still reaches the transform list unconditionally (same root
    cause as the bin case above: ``to_implicit_transforms()`` does not
    consult the honored set) and becomes a real ``Aggregate`` transform.
    That Aggregate groups by the Bin's own OUTPUT columns (``v``/``count``
    no longer exist post-Bin; the aggregate's groupby is inferred from the
    now-rewritten encoding) and tries to read ``cnum`` from a batch the Bin
    has already replaced, failing with a Rust-side column-not-found error
    rather than the "not honored" message the gate is supposed to produce.
    Pins the CURRENT failure mode (pre-existing, not worsened by this
    batch) so the same earmarked follow-up's blast radius is visible here
    too."""
    chart = (
        fm.Chart(_df())
        .mark_point()
        .encode(x=fm.X("v", bin=True), y="count", color=fm.Color("w", aggregate="mean"))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match=r"stat_aggregate: column 'w' not found"):
            chart.to_svg()


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
