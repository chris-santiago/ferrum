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
# `del`-only scope missed three parameters that are declared and simply
# never referenced -- the identical P9 defect, invisible to a guard that
# only inspects `ast.Delete`. Two (`normalize` on mark_confusion, `center`
# on mark_pdp) are the same "effect happens upstream" shape as proba /
# n_thresholds above. The third (`palette` on mark_boxen) is different in
# kind: no call site anywhere honors it -- it is a real, unimplemented
# feature, not a value computed elsewhere. All three are now registered
# and warn; see `ferrum.marks._informational_kwargs` for the full
# disposition writeup, including the tracked palette follow-up.
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


def test_mark_boxen_palette_non_none_warns_once():
    reset_warnings()
    df = _boxen_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = ferrum.Chart(df).mark_boxen(palette=["#111111", "#222222"])
        svg = chart.encode(x="group", y="val").to_svg()
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "palette" in str(user_warnings[0].message)
    assert "<svg" in svg


def test_mark_boxen_palette_default_is_silent():
    reset_warnings()
    df = _boxen_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ferrum.Chart(df).mark_boxen().encode(x="group", y="val").to_svg()
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []


def test_mark_boxen_palette_has_no_effect_on_output():
    """``palette`` is a genuinely unimplemented feature (not "effect
    happens elsewhere" like the other four registry entries) -- passing it
    changes nothing about the rendered colors, which is exactly the defect
    this warning surfaces rather than resolves. Pinned here so a future
    palette implementation is a deliberate, visible change to this test,
    not a silent one."""
    df = _boxen_df()
    reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg_default = ferrum.Chart(df).mark_boxen().encode(x="group", y="val").to_svg()
        svg_palette = (
            ferrum.Chart(df)
            .mark_boxen(palette=["#111111", "#222222"])
            .encode(x="group", y="val")
            .to_svg()
        )
    assert svg_default == svg_palette
