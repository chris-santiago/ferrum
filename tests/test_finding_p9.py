"""Regression tests for design-review finding P9 (accept-and-``del`` mark
parameters).

Each ``mark_*`` method audited in P9 either accepted a documented parameter
and silently dropped it (``del <param>`` with no wiring), or accepted a
parameter that could never do anything (an unreachable/dead desugar
argument).  This file proves the disposition for each site:

* *become functional* -- the parameter now measurably changes the resolved
  chart's data or rendered output.
* *removed* -- passing the parameter now raises ``TypeError`` (it fell
  through to ``_split_style_kwargs``'s style-kwarg path and was rejected by
  ``MarkBase``) instead of silently doing nothing.
* *used* -- ``algorithm`` now appears in the Rank2D chart title.

See ``.claude/output/decisions/2026-08-27-design-review-findings-decision.md``
(P9 section) and ``.claude/output/specs/2026-08-27-findings-remediation-design.md``
(§4-P9) for the full disposition table.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import polars as pl
import pytest

import ferrum
from ferrum._warn import reset_warnings
from tests.fixtures import load_dataset, load_fixture


# ---------------------------------------------------------------------------
# Become functional: average (mark_roc, mark_pr)
# ---------------------------------------------------------------------------


def test_mark_roc_average_filters_to_requested_row():
    df = pl.DataFrame(
        {
            "fpr": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "tpr": [0.0, 0.6, 1.0, 0.0, 0.7, 1.0, 0.0, 0.65, 1.0],
            "class": ["0", "0", "0", "1", "1", "1", "macro", "macro", "macro"],
        }
    )
    chart = ferrum.Chart(df).mark_roc(average="macro", reference_line=False)
    classes = set(chart._data["class"].unique().to_list())
    assert classes == {"macro"}


def test_mark_pr_average_filters_to_requested_row():
    df = pl.DataFrame(
        {
            "recall": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "precision": [1.0, 0.8, 0.0, 1.0, 0.7, 0.0, 1.0, 0.75, 0.0],
            "class": ["0", "0", "0", "1", "1", "1", "macro", "macro", "macro"],
        }
    )
    chart = ferrum.Chart(df).mark_pr(average="macro")
    classes = set(chart._data["class"].unique().to_list())
    assert classes == {"macro"}


def test_mark_roc_average_none_renders_all_classes():
    """Keep-case: the default ``average=None`` still renders every class."""
    df = pl.DataFrame(
        {
            "fpr": [0.0, 1.0, 0.0, 1.0],
            "tpr": [0.0, 1.0, 0.0, 1.0],
            "class": ["0", "0", "1", "1"],
        }
    )
    chart = ferrum.Chart(df).mark_roc(reference_line=False)
    classes = set(chart._data["class"].unique().to_list())
    assert classes == {"0", "1"}


def test_mark_roc_average_on_binary_curve_falls_back_silently():
    """Load-bearing case: a binary curve's ``class`` column carries exactly
    one value, so ``average="macro"`` (the default ``roc_chart`` forwards
    even for binary models) can never match -- must stay unfiltered and
    silent, not warn."""
    reset_warnings()
    df = pl.DataFrame(
        {
            "fpr": [0.0, 0.5, 1.0],
            "tpr": [0.0, 0.6, 1.0],
            "class": ["1", "1", "1"],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = ferrum.Chart(df).mark_roc(average="macro", reference_line=False)
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert set(chart._data["class"].unique().to_list()) == {"1"}


def test_mark_roc_average_typo_on_multiclass_data_warns_once():
    """A multiclass ``class`` column with more than one value present means
    an average row plausibly could have matched -- a request that matches
    nothing is most likely a typo, so this warns instead of silently
    rendering every class."""
    reset_warnings()
    df = pl.DataFrame(
        {
            "fpr": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "tpr": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "class": ["0", "0", "1", "1", "macro", "macro"],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = ferrum.Chart(df).mark_roc(average="marco", reference_line=False)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "average" in str(user_warnings[0].message)
    # Falls back to unfiltered, exactly like the silent binary case.
    assert set(chart._data["class"].unique().to_list()) == {"0", "1", "macro"}


def test_mark_roc_average_integer_class_column_does_not_raise():
    """Mutation-testing gap close: ``_utf8_col`` (``marks/_desugar_helpers.py``)
    casts the discriminator column to ``Utf8`` before comparing it against
    a Python string literal, per its own docstring contract ("an
    out-of-contract numeric/categorical dtype produces a normal empty-match
    miss instead of polars' ComputeError"). No existing test exercised a
    non-Utf8 ``class`` column, so dropping that cast (which
    ``ferrum._filter_class_average``/``mark_roc``'s ``average=`` filter
    depends on) went undetected: without it, polars raises
    ``InvalidOperationError: cannot compare string with numeric type``
    instead of the documented, non-crashing fallback."""
    reset_warnings()
    df = pl.DataFrame(
        {
            "fpr": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "tpr": [0.0, 0.6, 1.0, 0.0, 0.7, 1.0],
            "class": [0, 0, 0, 1, 1, 1],
        }
    )
    chart = ferrum.Chart(df).mark_roc(average="macro", reference_line=False)
    svg = chart.to_svg()
    assert "<svg" in svg
    # No int value ever equals the string "macro", and this is a
    # multiclass column, so this hits the multiclass-mismatch fallback
    # (unfiltered, all 6 rows kept) -- the same documented behavior as the
    # Utf8-typo case above, just reached through a non-Utf8 column.
    assert chart._data.height == 6


def _roc_average_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fpr": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "tpr": [0.0, 0.6, 1.0, 0.0, 0.7, 1.0, 0.0, 0.65, 1.0],
            "class": ["0", "0", "0", "1", "1", "1", "macro", "macro", "macro"],
        }
    )


def test_mark_roc_average_filters_to_requested_row_pandas_backed():
    """S4 close-out regression (RED pre-fix): ``_set_composite_mark`` applied
    ``data_transform`` only when ``isinstance(new._data, pl.DataFrame)``, so
    every P9 ``data_transform``-wired mark parameter -- ``average`` here --
    silently no-op'd on a pandas-backed ``Chart``: the filter never ran and
    all 9 rows stayed unfiltered, with no error and no warning. Now routed
    through ``ferrum._coerce.to_polars`` at that seam, so the filter fires
    for every supported input type, not just polars."""
    pd = pytest.importorskip("pandas")
    df = _roc_average_frame().to_pandas()
    assert isinstance(df, pd.DataFrame)
    chart = ferrum.Chart(df).mark_roc(average="macro", reference_line=False)
    assert isinstance(chart._data, pl.DataFrame)
    classes = set(chart._data["class"].unique().to_list())
    assert classes == {"macro"}


def test_mark_roc_average_filters_to_requested_row_pyarrow_backed():
    """S4 close-out regression (RED pre-fix): same silent no-op as the
    pandas case above, for a pyarrow.Table-backed ``Chart``."""
    pa = pytest.importorskip("pyarrow")
    tbl = _roc_average_frame().to_arrow()
    assert isinstance(tbl, pa.Table)
    chart = ferrum.Chart(tbl).mark_roc(average="macro", reference_line=False)
    assert isinstance(chart._data, pl.DataFrame)
    classes = set(chart._data["class"].unique().to_list())
    assert classes == {"macro"}


def test_mark_roc_average_polars_backed_data_transform_is_byte_identical():
    """Global constraint: routing ``data_transform`` through
    ``to_polars`` must not change already-working polars behavior.
    ``to_polars`` is a pure passthrough for a ``pl.DataFrame`` input (same
    object, not a copy), so the polars path is provably unaffected -- this
    pins that as an executable assertion: the resolved frame equals the
    naive expected filter exactly (not just "same set of classes"), and the
    rendered SVG for a polars-backed chart matches a chart built directly
    from the pre-filtered frame."""
    df = _roc_average_frame()
    chart = ferrum.Chart(df).mark_roc(average="macro", reference_line=False)
    expected = df.filter(pl.col("class") == "macro")
    assert chart._data.equals(expected)

    direct = ferrum.Chart(expected).mark_roc(reference_line=False)
    assert chart.encode(x="fpr", y="tpr").to_svg() == direct.encode(x="fpr", y="tpr").to_svg()


def test_roc_chart_figure_path_unaffected_by_data_transform_seam():
    """Confirm figure-function paths (``roc_chart``, always polars-sourced
    via ``ModelSource.roc_curve``) render unaffected by the
    ``_set_composite_mark`` seam fix."""
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"], random_state=0)
    svg = ferrum.roc_chart(source, per_class=False).to_svg()
    assert "<svg" in svg
    assert "AUC" in svg


# ---------------------------------------------------------------------------
# Become functional: split (mark_cv_scores)
# ---------------------------------------------------------------------------


def test_mark_cv_scores_split_filters_rows():
    df = pl.DataFrame(
        {
            "fold": [0, 1, 2, 0, 1, 2],
            "split": ["train", "train", "train", "test", "test", "test"],
            "score": [0.9, 0.91, 0.89, 0.8, 0.82, 0.79],
        }
    )
    chart = ferrum.Chart(df).mark_cv_scores(kind="strip", split="train")
    splits = set(chart._data["split"].unique().to_list())
    assert splits == {"train"}


def test_mark_cv_scores_split_both_keeps_all_rows():
    df = pl.DataFrame(
        {
            "fold": [0, 1, 0, 1],
            "split": ["train", "train", "test", "test"],
            "score": [0.9, 0.91, 0.8, 0.82],
        }
    )
    chart = ferrum.Chart(df).mark_cv_scores(kind="strip", split="both")
    splits = set(chart._data["split"].unique().to_list())
    assert splits == {"train", "test"}


def test_mark_cv_scores_invalid_split_raises_value_error():
    """A ``split`` typo (e.g. ``"tets"``) must reject loudly, not silently
    filter the DataFrame to zero rows and render an empty chart -- the same
    "works or rejects loudly" standard as every other P9-implemented
    parameter's typo path."""
    df = pl.DataFrame(
        {
            "fold": [0, 1, 0, 1],
            "split": ["train", "train", "test", "test"],
            "score": [0.9, 0.91, 0.8, 0.82],
        }
    )
    with pytest.raises(ValueError, match="split"):
        ferrum.Chart(df).mark_cv_scores(kind="strip", split="tets")


# ---------------------------------------------------------------------------
# Become functional: reference_line=False (mark_gain, mark_lift)
# ---------------------------------------------------------------------------


def test_mark_gain_reference_line_false_drops_baseline_rows():
    df = pl.DataFrame(
        {
            "percent_population": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "gain": [0.0, 0.5, 1.0, 0.0, 0.6, 1.0],
            "class": ["baseline", "baseline", "baseline", "0", "0", "0"],
        }
    )
    chart = ferrum.Chart(df).mark_gain(reference_line=False)
    classes = set(chart._data["class"].unique().to_list())
    assert classes == {"0"}


def test_mark_lift_reference_line_false_drops_baseline_rows():
    df = pl.DataFrame(
        {
            "percent_population": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "lift": [1.0, 1.0, 1.0, 1.0, 1.4, 1.8],
            "class": ["baseline", "baseline", "baseline", "0", "0", "0"],
        }
    )
    chart = ferrum.Chart(df).mark_lift(reference_line=False)
    classes = set(chart._data["class"].unique().to_list())
    assert classes == {"0"}


def test_mark_gain_reference_line_true_keeps_baseline_rows():
    """Keep-case: the default ``reference_line=True`` still keeps the
    baseline diagonal rows."""
    df = pl.DataFrame(
        {
            "percent_population": [0.0, 1.0, 0.0, 1.0],
            "gain": [0.0, 1.0, 0.0, 1.0],
            "class": ["baseline", "baseline", "0", "0"],
        }
    )
    chart = ferrum.Chart(df).mark_gain()
    classes = set(chart._data["class"].unique().to_list())
    assert classes == {"baseline", "0"}


# ---------------------------------------------------------------------------
# Become functional: metrics (mark_discrimination_threshold)
# ---------------------------------------------------------------------------


def test_mark_discrimination_threshold_metrics_filters_rows():
    df = pl.DataFrame(
        {
            "threshold": [0.1, 0.2, 0.3] * 4,
            "metric": ["precision"] * 3 + ["recall"] * 3 + ["f1"] * 3 + ["queue_rate"] * 3,
            "value": [0.5] * 12,
        }
    )
    chart = ferrum.Chart(df).mark_discrimination_threshold(
        metrics=("precision", "recall"), optimum_label=False
    )
    present = set(chart._data["metric"].unique().to_list())
    assert present == {"precision", "recall"}


def test_mark_discrimination_threshold_default_metrics_keeps_all_rows():
    """Keep-case: the default ``metrics`` tuple matches every row present,
    so nothing is filtered."""
    df = pl.DataFrame(
        {
            "threshold": [0.1, 0.2] * 4,
            "metric": ["precision"] * 2 + ["recall"] * 2 + ["f1"] * 2 + ["queue_rate"] * 2,
            "value": [0.5] * 8,
        }
    )
    chart = ferrum.Chart(df).mark_discrimination_threshold(optimum_label=False)
    present = set(chart._data["metric"].unique().to_list())
    assert present == {"precision", "recall", "f1", "queue_rate"}


def test_mark_discrimination_threshold_relabeled_metrics_fall_back_silently():
    """Load-bearing case: ``discrimination_threshold_chart`` relabels the
    ``metric`` column to display names ("precision" -> "Precision", etc.)
    before calling this mark, so the raw-name ``metrics`` tuple it forwards
    never matches exactly -- but the relabeling is a case/spacing rename of
    the same names, so this must stay unfiltered and silent, not warn."""
    reset_warnings()
    df = pl.DataFrame(
        {
            "threshold": [0.1, 0.2] * 4,
            "metric": ["Precision"] * 2 + ["Recall"] * 2 + ["F1"] * 2 + ["Queue rate"] * 2,
            "value": [0.5] * 8,
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = ferrum.Chart(df).mark_discrimination_threshold(optimum_label=False)
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    present = set(chart._data["metric"].unique().to_list())
    assert present == {"Precision", "Recall", "F1", "Queue rate"}


def test_mark_discrimination_threshold_metrics_typo_warns_once():
    """A ``metrics`` tuple with zero overlap by any reading (exact or
    normalized) is most likely a typo, so this warns instead of silently
    rendering every metric."""
    reset_warnings()
    df = pl.DataFrame(
        {
            "threshold": [0.1, 0.2] * 2,
            "metric": ["precision"] * 2 + ["recall"] * 2,
            "value": [0.5] * 4,
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = ferrum.Chart(df).mark_discrimination_threshold(
            metrics=("precisionn", "recalll"), optimum_label=False
        )
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "metrics" in str(user_warnings[0].message)
    # Falls back to unfiltered, exactly like the relabeled-metrics case.
    assert set(chart._data["metric"].unique().to_list()) == {"precision", "recall"}


# ---------------------------------------------------------------------------
# Become functional: order, color_bar (mark_shap_beeswarm)
# ---------------------------------------------------------------------------


def test_mark_shap_beeswarm_order_abs_mean_orders_features_descending():
    df = pl.DataFrame(
        {
            "feature": ["low", "low", "high", "high"],
            "shap_value": [0.1, -0.1, 0.9, -0.9],
            "feature_value_normalized": [0.0, 0.0, 0.0, 0.0],
        }
    )
    chart = ferrum.Chart(df).mark_shap_beeswarm(order="abs_mean", zero_line=False)
    order = chart._data["feature"].unique(maintain_order=True).to_list()
    assert order == ["high", "low"]


def test_mark_shap_beeswarm_order_mean_uses_signed_average():
    """``order="mean"`` ranks by signed mean SHAP, so a large-magnitude
    negative feature sorts *after* a small-magnitude positive one --
    the opposite of ``order="abs_mean"`` on the same data."""
    df = pl.DataFrame(
        {
            "feature": ["neg", "neg", "pos", "pos"],
            "shap_value": [-0.9, -0.9, 0.1, 0.1],
            "feature_value_normalized": [0.0, 0.0, 0.0, 0.0],
        }
    )
    abs_mean_order = (
        ferrum.Chart(df)
        .mark_shap_beeswarm(order="abs_mean", zero_line=False)
        ._data["feature"]
        .unique(maintain_order=True)
        .to_list()
    )
    mean_order = (
        ferrum.Chart(df)
        .mark_shap_beeswarm(order="mean", zero_line=False)
        ._data["feature"]
        .unique(maintain_order=True)
        .to_list()
    )
    assert abs_mean_order == ["neg", "pos"]
    assert mean_order == ["pos", "neg"]


def test_mark_shap_beeswarm_order_max_orders_by_max_abs_shap():
    """``order="max"`` (2026-08-27 close-out: union-vocabulary restoration
    -- ``mark_shap_beeswarm`` gains ``"max"`` for the first time here, on
    the same ``expr.max()`` aggregation branch the figure-side
    ``_shap_order_features`` always had) ranks by descending
    ``max(|shap_value|)``, not ``mean(|shap_value|)``. Feature "spike" has
    a lower mean-|shap| (0.5) than "steady" (0.6) but a higher max-|shap|
    (0.9 vs 0.6, a single outlier sample), so the two orders disagree."""
    df = pl.DataFrame(
        {
            "feature": ["spike", "spike", "steady", "steady"],
            "shap_value": [0.1, 0.9, 0.6, 0.6],
            "feature_value_normalized": [0.0, 0.0, 0.0, 0.0],
        }
    )
    abs_mean_order = (
        ferrum.Chart(df)
        .mark_shap_beeswarm(order="abs_mean", zero_line=False)
        ._data["feature"]
        .unique(maintain_order=True)
        .to_list()
    )
    max_order = (
        ferrum.Chart(df)
        .mark_shap_beeswarm(order="max", zero_line=False)
        ._data["feature"]
        .unique(maintain_order=True)
        .to_list()
    )
    assert abs_mean_order == ["steady", "spike"]
    assert max_order == ["spike", "steady"]


def test_mark_shap_beeswarm_order_none_preserves_row_order():
    """Keep-case: ``order="none"`` leaves the incoming row order untouched
    even though ``abs_mean``/``mean`` would reorder this data."""
    df = pl.DataFrame(
        {
            "feature": ["low", "low", "high", "high"],
            "shap_value": [0.1, -0.1, 0.9, -0.9],
            "feature_value_normalized": [0.0, 0.0, 0.0, 0.0],
        }
    )
    chart = ferrum.Chart(df).mark_shap_beeswarm(order="none", zero_line=False)
    order = chart._data["feature"].unique(maintain_order=True).to_list()
    assert order == ["low", "high"]


def test_mark_shap_beeswarm_invalid_order_raises_value_error():
    df = pl.DataFrame({"feature": ["a"], "shap_value": [0.1], "feature_value_normalized": [0.0]})
    with pytest.raises(ValueError, match="order"):
        ferrum.Chart(df).mark_shap_beeswarm(order="bogus")


def _shap_beeswarm_color_legend(*, color_bar: bool) -> dict | None:
    df = pl.DataFrame(
        {
            "feature": ["a", "a"],
            "shap_value": [0.1, -0.2],
            "feature_value_normalized": [0.3, -0.1],
        }
    )
    resolved = (
        ferrum.Chart(df).mark_shap_beeswarm(color_bar=color_bar, zero_line=False)._resolve_pending()
    )
    for layer in resolved.to_dict()["layers"]:
        color = layer.get("encoding", {}).get("color")
        if color is not None:
            return color.get("legend")
    raise AssertionError("no layer carried a color encoding")


def test_mark_shap_beeswarm_color_bar_true_sets_tick_label_legend():
    legend = _shap_beeswarm_color_legend(color_bar=True)
    assert legend == {"tickLabels": ["Low", "", "", "", "High"]}


def test_mark_shap_beeswarm_color_bar_false_disables_legend():
    legend = _shap_beeswarm_color_legend(color_bar=False)
    assert legend == {"disabled": True}


def _shap_beeswarm_chart(*, color_bar: bool) -> "ferrum.Chart":
    df = pl.DataFrame(
        {
            "feature": ["f0", "f0", "f1", "f1", "f2", "f2"],
            "shap_value": [0.1, -0.2, 0.3, 0.4, -0.1, 0.05],
            "feature_value_normalized": [0.5, -0.5, 1.0, -1.0, 0.2, -0.2],
        }
    )
    return (
        ferrum.Chart(df).mark_shap_beeswarm(color_bar=color_bar).encode(x="shap_value", y="feature")
    )


def test_mark_shap_beeswarm_color_bar_false_actually_disables_rendered_colorbar():
    """The per-layer ``legend={"disabled": True}`` dict asserted by
    ``test_mark_shap_beeswarm_color_bar_false_disables_legend`` above is
    necessary but not sufficient: it only proves the *spec* carries the
    right value, not that the renderer honors it. This is the discriminating
    regression -- pre-fix, ``color_bar=False`` was accepted, validated, and
    silently discarded: the rendered SVG was byte-identical to the default
    and a colorbar still appeared (a design-review / intent-review finding).
    The Rust renderer's colorbar-legend construction for a layered chart
    reads its ``legend=`` config from the *chart-level* ``encoding.color``,
    not from any per-layer color channel -- ``Chart.mark_shap_beeswarm`` now
    mirrors the layer's ``Color(...)`` config onto the chart-level encoding
    for exactly this reason (see that method's docstring)."""
    svg_on = _shap_beeswarm_chart(color_bar=True).to_svg()
    svg_off = _shap_beeswarm_chart(color_bar=False).to_svg()

    assert svg_on != svg_off
    assert "ferrum-colorbar" in svg_on
    assert "ferrum-colorbar" not in svg_off
    # The default (color_bar=True) also renders the mark's documented custom
    # "Low"/"High" tick labels, not raw numeric scale ticks -- proving the
    # chart-level legend mirror carries the *content* of the config, not
    # merely its presence/absence.
    assert ">Low<" in svg_on
    assert ">High<" in svg_on
    assert ">Low<" not in svg_off
    assert ">High<" not in svg_off


def test_mark_shap_bar_max_display_truncates_documented_contract():
    """``mark_shap_bar``'s documented direct-call contract is ``feature`` +
    ``abs_mean_shap`` (see ``desugar_shap_bar``'s data contract), one row
    per feature -- not the long-form ``shap_value`` schema. Pre-fix,
    ``_shap_bar_filter`` guarded on ``"shap_value" in df.columns``, a column
    this contract never carries, so the truncation silently never fired on
    documented input (RED pre-fix: this frame keeps all 5 features instead
    of the requested 2)."""
    df = pl.DataFrame(
        {
            "feature": ["a", "b", "c", "d", "e"],
            "abs_mean_shap": [0.5, 0.1, 0.9, 0.3, 0.2],
        }
    )
    chart = ferrum.Chart(df).mark_shap_bar(max_display=2)
    kept = set(chart._data["feature"].unique().to_list())
    assert kept == {"c", "a"}, f"expected the top-2 abs_mean_shap features, got {kept}"


def test_mark_shap_waterfall_max_display_truncates_documented_contract():
    """``mark_shap_waterfall``'s documented direct-call contract is
    ``feature`` + ``x0``/``x1`` + ``shap_sign`` (see
    ``desugar_shap_waterfall``'s data contract) -- not the long-form
    ``shap_value`` schema. Pre-fix, ``_shap_waterfall_filter`` guarded on
    ``"shap_value" in df.columns``, a column this contract never carries, so
    the truncation silently never fired on documented input (RED pre-fix:
    this frame keeps all 5 features instead of the requested 2)."""
    df = pl.DataFrame(
        {
            "feature": ["a", "b", "c", "d", "e"],
            "x0": [0.0, 0.5, 0.1, 1.0, 0.3],
            "x1": [0.5, 0.6, 1.1, 1.2, 0.5],
            "shap_sign": ["positive"] * 5,
        }
    )
    chart = ferrum.Chart(df).mark_shap_waterfall(sample_idx=0, max_display=2)
    kept = set(chart._data["feature"].unique().to_list())
    assert kept == {"c", "a"}, f"expected the top-2 |x1 - x0| features, got {kept}"


# ---------------------------------------------------------------------------
# Used: algorithm in the Rank2D title
# ---------------------------------------------------------------------------


def test_rank2d_chart_title_includes_algorithm():
    df = pl.DataFrame({"f0": [1.0, 2.0, 3.0], "f1": [2.0, 1.0, 0.5]})
    svg = ferrum.rank2d_chart(df, algorithm="spearman").to_svg()
    assert "Feature Correlation (Spearman)" in svg


def test_rank2d_chart_title_default_algorithm_is_pearson():
    df = pl.DataFrame({"f0": [1.0, 2.0, 3.0], "f1": [2.0, 1.0, 0.5]})
    svg = ferrum.rank2d_chart(df).to_svg()
    assert "Feature Correlation (Pearson)" in svg


# ---------------------------------------------------------------------------
# Removed: ci_style (mark_alpha_selection)
# ---------------------------------------------------------------------------


def test_mark_alpha_selection_ci_style_raises_type_error():
    df = pl.DataFrame({"alpha": [0.1, 1.0], "mean_score": [0.8, 0.75]})
    chart = ferrum.Chart(df).mark_alpha_selection(ci_style="band")
    with pytest.raises(TypeError, match="ci_style"):
        chart._resolve_pending()


# ---------------------------------------------------------------------------
# Removed: n_bins / strategy (mark_calibration)
# ---------------------------------------------------------------------------


def test_mark_calibration_n_bins_raises_type_error():
    df = pl.DataFrame({"mean_predicted": [0.1, 0.5, 0.9], "fraction_positive": [0.05, 0.5, 0.95]})
    chart = ferrum.Chart(df).mark_calibration(n_bins=5)
    with pytest.raises(TypeError, match="n_bins"):
        chart._resolve_pending()


def test_mark_calibration_strategy_raises_type_error():
    df = pl.DataFrame({"mean_predicted": [0.1, 0.5, 0.9], "fraction_positive": [0.05, 0.5, 0.95]})
    chart = ferrum.Chart(df).mark_calibration(strategy="quantile")
    with pytest.raises(TypeError, match="strategy"):
        chart._resolve_pending()


# ---------------------------------------------------------------------------
# Removed: n_components (both pca-scree desugars, consistent behavior)
# ---------------------------------------------------------------------------


def _pca_scree_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "component": [1, 2, 3],
            "explained_variance_ratio": [0.5, 0.3, 0.2],
            "cumulative_variance_ratio": [0.5, 0.8, 1.0],
        }
    )


def test_mark_pca_scree_n_components_raises_type_error():
    chart = ferrum.Chart(_pca_scree_df()).mark_pca_scree(n_components=2)
    with pytest.raises(TypeError, match="n_components"):
        chart._resolve_pending()


def test_mark_pca_scree_with_threshold_n_components_raises_type_error():
    """The threshold-line branch (a different desugar function) rejects
    ``n_components`` the same way as the plain branch above -- the P9 fix
    resolved the drop/raise asymmetry between the two desugars to one
    consistent (raising) behavior."""
    chart = ferrum.Chart(_pca_scree_df()).mark_pca_scree(threshold_line=0.9, n_components=2)
    with pytest.raises(TypeError, match="n_components"):
        chart._resolve_pending()


# ---------------------------------------------------------------------------
# ROC figure title stays in sync with the post-filter (average-restricted)
# curve, not the pre-filter frame that still carries every per-class row.
# ---------------------------------------------------------------------------


def test_roc_chart_multiclass_per_class_false_titles_with_auc():
    """``roc_chart(per_class=False)`` on multiclass data now renders exactly
    one (average) curve via the P9 ``average`` fix. The title must reflect
    that single-curve state -- computed from the post-filter frame -- not
    the pre-filter frame, which still holds every per-class row plus the
    average row and would make the single rendered curve look untitled."""
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])

    chart = ferrum.roc_chart(source, per_class=False)
    assert len(set(chart._data["class"].unique().to_list())) == 1
    svg = chart.to_svg()
    assert "ROC Curve — AUC 0." in svg
    assert ">ROC Curve<" not in svg


# ---------------------------------------------------------------------------
# Task 14 (P9 AST guard follow-up): informational no-ops now warn directly
# at the mark method -- proba (mark_decision_boundary), n_thresholds
# (mark_discrimination_threshold) -- but the figure functions that forward
# neither into the mark call must stay silent, and the parameter must not
# change a single byte of rendered output either way.
# ---------------------------------------------------------------------------


def _decision_boundary_grid_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [0.0, 1.0],
            "x2": [1.0, 2.0],
            "y": [0.0, 0.0],
            "y2": [1.0, 1.0],
            "z": ["0", "1"],
        }
    )


def test_mark_decision_boundary_proba_true_warns_once():
    reset_warnings()
    df = _decision_boundary_grid_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = ferrum.Chart(df).mark_decision_boundary(proba=True).to_svg()
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "proba" in str(user_warnings[0].message)
    assert "<svg" in svg


def test_mark_decision_boundary_proba_default_is_silent():
    reset_warnings()
    df = _decision_boundary_grid_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ferrum.Chart(df).mark_decision_boundary().to_svg()
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []


def test_mark_decision_boundary_proba_has_no_effect_on_output():
    """``proba`` is ``del``eted unread by the desugar -- passing it must
    not change a single byte of the rendered output, only emit the
    warning above."""
    df = _decision_boundary_grid_df()
    reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg_default = ferrum.Chart(df).mark_decision_boundary().to_svg()
        svg_proba_true = ferrum.Chart(df).mark_decision_boundary(proba=True).to_svg()
    assert svg_default == svg_proba_true


def test_decision_boundary_chart_stays_silent_regardless_of_proba():
    """The figure-function path no longer forwards ``proba`` to
    ``mark_decision_boundary`` (Task 14), so it must never trigger the
    mark-level warning -- for either value, and with the scatter overlay
    on or off."""
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    for proba in (False, True):
        for scatter in (False, True):
            reset_warnings()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                chart = ferrum.decision_boundary_chart(
                    model,
                    X,
                    df["y"],
                    features=(0, 1),
                    grid_resolution=20,
                    proba=proba,
                    scatter=scatter,
                )
                chart.to_svg()
            user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
            assert user_warnings == [], (proba, scatter, user_warnings)


def _discrimination_threshold_long_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "threshold": [0.1, 0.2, 0.3] * 4,
            "metric": ["precision"] * 3 + ["recall"] * 3 + ["f1"] * 3 + ["queue_rate"] * 3,
            "value": [0.5] * 12,
        }
    )


def test_mark_discrimination_threshold_n_thresholds_non_default_warns_once():
    reset_warnings()
    df = _discrimination_threshold_long_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = (
            ferrum.Chart(df)
            .mark_discrimination_threshold(n_thresholds=200, optimum_label=False)
            .to_svg()
        )
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "n_thresholds" in str(user_warnings[0].message)
    assert "<svg" in svg


def test_mark_discrimination_threshold_n_thresholds_default_is_silent():
    reset_warnings()
    df = _discrimination_threshold_long_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ferrum.Chart(df).mark_discrimination_threshold(optimum_label=False).to_svg()
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []


def test_mark_discrimination_threshold_n_thresholds_has_no_effect_on_output():
    """``n_thresholds`` is ``del``eted unread by the desugar -- passing a
    non-default value must not change a single byte of the rendered
    output, only emit the warning above."""
    df = _discrimination_threshold_long_df()
    reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg_default = ferrum.Chart(df).mark_discrimination_threshold(optimum_label=False).to_svg()
        svg_non_default = (
            ferrum.Chart(df)
            .mark_discrimination_threshold(n_thresholds=200, optimum_label=False)
            .to_svg()
        )
    assert svg_default == svg_non_default


def test_discrimination_threshold_chart_stays_silent_regardless_of_n_thresholds():
    """The figure-function path no longer forwards ``n_thresholds`` to
    ``mark_discrimination_threshold`` (Task 14), so it must never trigger
    the mark-level warning -- default or not."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0.1, 0.3, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6])
    for n_thresholds in (50, 10, 200):
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart = ferrum.discrimination_threshold_chart(
                y_true=y_true, y_pred=y_pred, n_thresholds=n_thresholds
            )
            chart.to_svg()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings == [], (n_thresholds, user_warnings)


def test_informational_kwarg_warning_points_at_caller_not_ferrum_source():
    """S4 fix: the warning must be attributed to the user's call site, not
    to ``warn_informational_kwarg`` or the mixin method in between (two
    extra ferrum frames sit between this line and ``warnings.warn``)."""
    reset_warnings()
    df = _decision_boundary_grid_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ferrum.Chart(df).mark_decision_boundary(proba=True)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert user_warnings[0].filename == __file__, (
        f"expected the warning attributed to this test file, got "
        f"{user_warnings[0].filename!r} -- stacklevel points into ferrum's "
        f"own source instead of the caller"
    )


# ---------------------------------------------------------------------------
# Quality-review follow-up (2026-08-27, Task 14 extension): the AST guard's
# `del`-only scope missed three parameters that were declared and simply
# never referenced -- the identical P9 defect, invisible to a guard that
# only inspects `ast.Delete`. Two (`normalize` on mark_confusion, `center`
# on mark_pdp) are the same "effect happens upstream" shape as proba /
# n_thresholds above; both are registered and warn -- see
# `ferrum.marks._informational_kwargs` for the disposition writeup. The
# third (`palette` on mark_boxen) was different in kind: no call site
# anywhere honored it, a real, unimplemented feature rather than a value
# computed elsewhere. It has since been implemented (residuals batch,
# #91, 2026-08-27): `desugar_boxen` now colors the depth bands directly
# from `palette`, so it is genuinely used and no longer registered or
# warned -- see the `test_mark_boxen_palette_*` tests below.
# ---------------------------------------------------------------------------


def _confusion_matrix_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "actual": ["0", "0", "1", "1"],
            "predicted": ["0", "1", "0", "1"],
            "value": [5.0, 1.0, 2.0, 6.0],
            "value_fmt": ["5", "1", "2", "6"],
        }
    )


def test_mark_confusion_normalize_non_none_warns_once():
    reset_warnings()
    df = _confusion_matrix_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = ferrum.Chart(df).mark_confusion(normalize="true").to_svg()
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "normalize" in str(user_warnings[0].message)
    assert "<svg" in svg


def test_mark_confusion_normalize_default_is_silent():
    reset_warnings()
    df = _confusion_matrix_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ferrum.Chart(df).mark_confusion().to_svg()
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []


def test_mark_confusion_normalize_has_no_effect_on_output():
    """``normalize`` is never referenced by ``desugar_confusion`` -- passing
    it must not change a single byte of rendered output, only emit the
    warning above."""
    df = _confusion_matrix_df()
    reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg_default = ferrum.Chart(df).mark_confusion().to_svg()
        svg_normalize = ferrum.Chart(df).mark_confusion(normalize="true").to_svg()
    assert svg_default == svg_normalize


def test_confusion_matrix_chart_stays_silent_regardless_of_normalize():
    """The figure-function path no longer forwards ``normalize`` to
    ``mark_confusion`` (Task 14), so it must never trigger the mark-level
    warning -- for any normalization mode."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    for normalize in ("true", "pred", "all", None):
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart = ferrum.confusion_matrix_chart(y_true=y_true, y_pred=y_pred, normalize=normalize)
            chart.to_svg()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings == [], (normalize, user_warnings)


def _pdp_average_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": ["f0"] * 3 + ["f1"] * 3,
            "feature_value": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "pd_value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


def test_mark_pdp_center_true_warns_once():
    reset_warnings()
    df = _pdp_average_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = ferrum.Chart(df).mark_pdp(center=True).to_svg()
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "center" in str(user_warnings[0].message)
    assert "<svg" in svg


def test_mark_pdp_center_default_is_silent():
    reset_warnings()
    df = _pdp_average_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ferrum.Chart(df).mark_pdp().to_svg()
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []


def test_mark_pdp_center_has_no_effect_on_output():
    """``center`` is never referenced by ``desugar_pdp`` -- passing it must
    not change a single byte of rendered output, only emit the warning
    above."""
    df = _pdp_average_df()
    reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg_default = ferrum.Chart(df).mark_pdp().to_svg()
        svg_center_true = ferrum.Chart(df).mark_pdp(center=True).to_svg()
    assert svg_default == svg_center_true


def test_pdp_chart_stays_silent_regardless_of_center():
    """The figure-function path no longer forwards ``center`` to
    ``mark_pdp`` (Task 14), so it must never trigger the mark-level
    warning -- default or not."""
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    for center in (False, True):
        reset_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart = ferrum.pdp_chart(
                model, X, df["y"], features=["f0", "f1"], grid_resolution=10, center=center
            )
            chart.to_svg()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings == [], (center, user_warnings)


def _boxen_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "group": ["a"] * 10 + ["b"] * 10,
            "val": list(range(10)) + list(range(5, 15)),
        }
    )


def _boxen_df_multi_band() -> pl.DataFrame:
    """The standard palette-testing fixture (quality-review cycle-3
    census used n=200): large enough per group that the mark's default
    ``k_depth="tukey"`` selects several real letter-value depths, so
    palette-mapping assertions exercise the mark's actual default
    configuration, not just ``k_depth="full"``."""
    return pl.DataFrame(
        {
            "group": ["a"] * 200 + ["b"] * 200,
            "val": list(range(200)) + list(range(100, 300)),
        }
    )


def _boxen_df_small() -> pl.DataFrame:
    """Small enough to be a meaningfully different regime from
    ``_boxen_df_multi_band``, but still >=32 rows/group -- ``k_depth=
    "tukey"``'s ``floor(log2(n)) - 3`` needs ``n >= 32`` to reach a real
    depth beyond the median (``k=2``); below that, every row is the
    degenerate ``k=1`` band and no color could possibly be visible
    regardless of mapping. Reaches exactly one real band (``k=2``)."""
    return pl.DataFrame(
        {
            "group": ["a"] * 40 + ["b"] * 40,
            "val": list(range(40)) + list(range(20, 60)),
        }
    )


def _rect_fills(svg: str) -> list[str]:
    """Every ``fill="..."`` value on a ``<rect>`` element, in document order."""
    return re.findall(r'<rect\b[^>]*\bfill="([^"]+)"', svg)


def _rect_geoms(svg: str) -> list[tuple[float, float, float, float, str]]:
    """``(x, y, width, height, fill)`` for every ``<rect x=... y=...
    width=... height=... fill=...>`` element, in document order.

    Ties markup to *geometry* rather than fill alone, so a paint-order
    regression (a band emitted with the right color but occluded by a
    later, wider, fully-opaque band) is visible to a test even when the
    fill attributes alone would look correct."""
    pattern = re.compile(
        r'<rect\b[^>]*\bx="([-\d.]+)"[^>]*\by="([-\d.]+)"[^>]*'
        r'\bwidth="([-\d.]+)"[^>]*\bheight="([-\d.]+)"[^>]*\bfill="([^"]+)"'
    )
    return [
        (float(x), float(y), float(w), float(h), fill) for x, y, w, h, fill in pattern.findall(svg)
    ]


def _dedup_consecutive(values: list[str]) -> list[str]:
    """Collapse adjacent repeats (one rect per group, per depth band) while
    preserving depth order -- unlike ``dict.fromkeys``, a color that
    reappears in a later, non-adjacent band (palette cycling) is kept."""
    out: list[str] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def test_mark_boxen_palette_named_yields_distinct_band_fills():
    """A named palette colors the depth bands directly, and never warns
    (spec §4.4/§9.4: ≥2 distinct band fills; no more warn-bridge)."""
    reset_warnings()
    df = _boxen_df_multi_band()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = ferrum.Chart(df).mark_boxen(palette="tableau10").encode(x="group", y="val").to_svg()
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    expected = ferrum.color.palette("tableau10", n=5)
    band_fills = [f for f in _rect_fills(svg) if f in expected]
    assert len(set(band_fills)) >= 2
    # k=1 (the always-degenerate median band) borrows k=2's color slot
    # instead of consuming one of its own. k=2 is the **base band** --
    # spec §4.4's re-amended anchor (quality-review cycle-3) -- so it (and
    # the k=1 band that shares its color) is colors[0] directly, and is
    # painted last/on top, i.e. the last fill in document order.
    assert band_fills[-1] == expected[0]


def test_mark_boxen_palette_list_applies_in_order_and_cycles():
    """An explicit color sequence is applied to bands in order, and a
    shorter-than-the-colorable-band-count list cycles (spec §4.4/§9.4).

    ``k_depth="full"`` on the standard fixture reaches all 6 configured
    depths; the mark's default ``k_depth="tukey"`` on the same data
    reaches only ~3 real depths, and since k=1 always shares k=2's color
    slot (previous test), a 3-depth render only ever shows 2
    dedup-distinct colors -- not enough to prove cycling through a
    2-color list."""
    df = _boxen_df_multi_band()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (
            ferrum.Chart(df)
            .mark_boxen(palette=["#111111", "#222222"], k_depth="full")
            .encode(x="group", y="val")
            .to_svg()
        )
    # Two rows per depth (one rect per group), so dedup consecutive
    # duplicates while preserving depth order.
    band_colors = _dedup_consecutive(_rect_fills(svg))
    band_colors = [c for c in band_colors if c in {"#111111", "#222222"}]
    assert len(band_colors) >= 3, "need >=3 bands to prove cycling"
    # Cycled to 5 slots (colors[i % 2] for i in 0..4): [c0,c1,c0,c1,c0],
    # indexed k=2->0, k=3->1, k=4->2, k=5->3, k=6->4 (base-band anchor).
    # Document order is widest-first (k=6..2, then k=1 merging into k=2):
    # fills = [c0,c1,c0,c1,c0,c0] -> dedup = [c0,c1,c0,c1,c0].
    assert band_colors[0] == "#111111"  # k=6, index 4 -> cycled color c0
    assert band_colors[1] == "#222222"  # k=5, index 3 -> c1
    assert band_colors[-1] == "#111111"  # k=2 (and k=1, merged): index 0 -> c0


def _assert_boxen_palette_visually_correct(svg: str, expected_colors: list[str]) -> None:
    """Shared assertion body for the base-band color-mapping contract
    (spec §4.4, re-amended after quality-review cycle-3): per group,
    (a) band heights strictly decrease in document order (widest-first
    paint order, so nesting stays visible -- S1), (b) the innermost
    *real* band's fill is ``expected_colors[0]`` (the base-band anchor --
    ``k=2``, painted last among real bands, right before the always-
    degenerate ``k=1`` that shares its color -- so it is the
    second-to-last entry in document order), and (c) no color is visible
    *only* on the degenerate (minimum-height) band.

    Verified on rect *geometry*, not fill attributes alone -- a band that
    is emitted with the right color but occluded, or a color assigned
    only to a band that never gets real extent, both look correct in
    markup while being invisible on screen."""
    expected_fills = set(expected_colors)
    band_rects = [g for g in _rect_geoms(svg) if g[4] in expected_fills]
    by_group: dict[float, list[tuple[float, str]]] = {}
    for x, _y, _w, h, fill in band_rects:
        by_group.setdefault(round(x, 1), []).append((h, fill))
    assert len(by_group) >= 2, "need >=2 groups (categorical positions)"
    for group_x, height_fills in by_group.items():
        heights = [h for h, _fill in height_fills]
        assert len(heights) >= 2, (
            f"group at x={group_x}: need at least the degenerate k=1 band plus one real band"
        )
        assert heights == sorted(heights, reverse=True) and len(set(heights)) == len(heights), (
            f"group at x={group_x}: band heights not strictly decreasing "
            f"in document order (widest-first, i.e. nested and visible): "
            f"{heights}"
        )

        # Base-band anchor: colors[0] lands on the innermost *real* band
        # (k=2), which is the second-to-last entry -- the last entry is
        # always the degenerate k=1 band (borrows k=2's color, smaller
        # height).
        innermost_real_height, innermost_real_fill = height_fills[-2]
        assert innermost_real_fill == expected_colors[0], (
            f"group at x={group_x}: innermost real band (height="
            f"{innermost_real_height}) has fill {innermost_real_fill!r}, "
            f"expected colors[0] = {expected_colors[0]!r}"
        )

        # No requested color may be visible *only* on the degenerate
        # (minimum-height) band -- every fill that appears must also
        # appear on a taller (real) band.
        min_height = min(heights)
        degenerate_only_fills = {f for h, f in height_fills if h == min_height} - {
            f for h, f in height_fills if h > min_height
        }
        assert not degenerate_only_fills, (
            f"group at x={group_x}: color(s) {degenerate_only_fills} only "
            f"appear on the degenerate (height={min_height}) band"
        )


def test_mark_boxen_palette_bands_paint_widest_first_per_group():
    """Structural nesting pin (quality-review S1, spec §4.4 "Paint order"
    amendment): under ``palette=``, depth bands must paint widest-first
    (outermost under, innermost on top) within each group, so every band
    stays visibly nested instead of the widest band occluding the rest.
    Also pins the **base-band color mapping** (spec §4.4, re-amended after
    quality-review cycle-3): ``colors[0]`` must be visible -- landing on
    the innermost real band, ``k=2``, which is guaranteed to render
    whenever *any* real depth exists -- at ``k_depth="full"``, the mark's
    default ``k_depth`` on the standard fixture, and at a small ``n`` that
    barely reaches one real band. An earlier anchor (widest *configured*
    band, ``k=_BOXEN_K_MAX``) only rendered ``colors[0]`` when a dataset
    happened to reach full depth -- dead on every typical dataset under
    the default ``k_depth`` (measured by quality review: only the last
    two of six colors ever appeared at n=200) -- so this test deliberately
    covers all three regimes, not just the one configuration
    (``k_depth="full"``) that cannot fail."""
    expected = ferrum.color.palette("tableau10", n=5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg_full_depth = (
            ferrum.Chart(_boxen_df_multi_band())
            .mark_boxen(palette="tableau10", k_depth="full")
            .encode(x="group", y="val")
            .to_svg()
        )
        svg_default_kdepth = (
            ferrum.Chart(_boxen_df_multi_band())
            .mark_boxen(palette="tableau10")
            .encode(x="group", y="val")
            .to_svg()
        )
        svg_small_n = (
            ferrum.Chart(_boxen_df_small())
            .mark_boxen(palette="tableau10")
            .encode(x="group", y="val")
            .to_svg()
        )

    _assert_boxen_palette_visually_correct(svg_full_depth, expected)
    _assert_boxen_palette_visually_correct(svg_default_kdepth, expected)
    _assert_boxen_palette_visually_correct(svg_small_n, expected)


def test_mark_boxen_palette_continuous_scheme_full_depth_reaches_ramp_endpoint():
    """Regression guard for the palette-expansion-count defect (quality-
    review cycle-3 S3, fixed by sizing ``_resolve_boxen_palette``'s
    request to ``_BOXEN_VISIBLE_BANDS`` instead of ``_BOXEN_K_MAX``) --
    proven necessary by quality review's cycle-4 mutation probe: setting
    ``_BOXEN_VISIBLE_BANDS = _BOXEN_K_MAX`` left every existing palette
    test green, because ``_boxen_band_color_index`` always yields exactly
    5 distinct indices regardless of list length, and the categorical
    (``tableau10``) tests used elsewhere only check that *known* colors
    appear, not that the request count matches the consumable slot count.

    A *continuous* palette is the shape that actually discriminates: it
    is resampled evenly across ``[0, 1]`` at however many colors are
    requested, so requesting 6 points instead of 5 shifts *every* sample
    position, not just the extra one -- under the mutation, the last
    *consumed* color (index 4 of 6, at ``t=0.8``) is no longer the
    palette's endpoint (``t=1.0``), so ``viridis``'s final color never
    renders at all. ``k_depth="full"`` on the standard fixture guarantees
    every one of the 5 consumable bands (``k=2..6``) has real data, so
    every one of the 5 expected samples -- including the endpoint --
    must appear on a non-degenerate (height > 1) rect."""
    df = _boxen_df_multi_band()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (
            ferrum.Chart(df)
            .mark_boxen(palette="viridis", k_depth="full")
            .encode(x="group", y="val")
            .to_svg()
        )
    expected = ferrum.color.palette("viridis", n=5)
    rendered_non_degenerate = {fill for _x, _y, _w, h, fill in _rect_geoms(svg) if h > 1}
    missing = set(expected) - rendered_non_degenerate
    assert not missing, (
        f"expected viridis samples {expected} not all rendered at "
        f"non-degenerate height; missing {sorted(missing)} (a missing "
        f"endpoint, {expected[-1]!r}, means the palette expansion count "
        f"no longer matches the consumable slot count)"
    )


def test_mark_boxen_palette_conflicts_with_chart_level_color_encoding():
    """``palette=`` combined with a chart-level ``.encode(color=...)``
    channel raises ``ValueError`` instead of silently rendering a flat
    block (quality-review S2): the color encoding always overrides a
    layer's ``fill=``, so the palette would have no visible effect while
    still forcing opacity to 1.0. Checked both call orders -- desugaring
    is deferred until ``.encode()`` is fully known, regardless of whether
    ``mark_boxen()`` or ``.encode(color=...)`` came first in the chain."""
    df = _boxen_df_multi_band()
    with pytest.raises(ValueError, match="color encoding"):
        ferrum.Chart(df).mark_boxen(palette=["#111111", "#222222"]).encode(
            x="group", y="val", color="group"
        ).to_svg()
    with pytest.raises(ValueError, match="color encoding"):
        ferrum.Chart(df).encode(x="group", y="val", color="group").mark_boxen(
            palette=["#111111", "#222222"]
        ).to_svg()


def test_mark_boxen_palette_and_color_field_still_compose():
    """``color_field=`` (boxen's own per-group grouping kwarg) is a
    different mechanism from a chart-level ``color`` encoding and stays
    unaffected by the new conflict guard."""
    df = _boxen_df_multi_band()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (
            ferrum.Chart(df)
            .mark_boxen(palette=["#111111", "#222222"], color_field="group")
            .encode(x="group", y="val")
            .to_svg()
        )
    fills = set(_rect_fills(svg))
    assert {"#111111", "#222222"} <= fills


def test_mark_boxen_palette_non_iterable_raises_value_error():
    """A non-``str``, non-iterable ``palette=`` value (e.g. an ``int``)
    raises the same named ``ValueError`` shape as the empty-sequence
    guard, not a bare ``TypeError`` leaking from ``list(palette)``
    (quality-review S4)."""
    df = _boxen_df()
    with pytest.raises(ValueError, match=r"mark_boxen\(palette=\.\.\.\)"):
        ferrum.Chart(df).mark_boxen(palette=5).encode(x="group", y="val").to_svg()


def test_mark_boxen_palette_none_is_byte_identical_to_default():
    """``palette=None`` (explicit or omitted) keeps the opacity-ramp
    shading byte-identical -- one of the batch's pinned invariants
    (spec §7, §9.4)."""
    df = _boxen_df_multi_band()
    reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg_default = ferrum.Chart(df).mark_boxen().encode(x="group", y="val").to_svg()
        svg_none = ferrum.Chart(df).mark_boxen(palette=None).encode(x="group", y="val").to_svg()
    assert svg_default == svg_none


def test_mark_boxen_palette_invalid_name_raises_value_error():
    """An unrecognized palette name raises through the same validation
    path ``scheme=`` uses (spec §4.4/§9.4)."""
    df = _boxen_df()
    with pytest.raises(ValueError, match="Unknown palette"):
        ferrum.Chart(df).mark_boxen(palette="not-a-real-palette").encode(
            x="group", y="val"
        ).to_svg()


@pytest.mark.parametrize("empty_palette", [[], ()])
def test_mark_boxen_palette_empty_sequence_raises_value_error(empty_palette):
    """An empty color sequence can't color any band (and would otherwise
    ``ZeroDivisionError`` in the cycling ``i % len(colors)`` arithmetic) --
    hardening around the sequence path, named as a ``mark_boxen(palette=)``
    error so the caller knows which argument is at fault."""
    df = _boxen_df()
    with pytest.raises(ValueError, match=r"mark_boxen\(palette=\.\.\.\)"):
        ferrum.Chart(df).mark_boxen(palette=empty_palette).encode(x="group", y="val").to_svg()
