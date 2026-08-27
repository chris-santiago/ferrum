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
