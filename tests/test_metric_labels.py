"""Schwabish SB1 — AUCLabel / APLabel / BrierLabel + OutlierLabel + annotate_arrow.

Tests use small synthetic data so the SVG inspection is deterministic.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ferrum import (
    APLabel,
    AUCLabel,
    BrierLabel,
    Chart,
    OutlierLabel,
    annotate_arrow,
)
from tests._svg_extents import text_runs as _svg_texts


# ---- Single-source metric-kind table (T4.7b) ---------------------------------


def test_metric_label_specs_single_source():
    """The metric-kind table is the one source shared by the direct-mark
    ``__radd__`` path and the figure-function explicit-field path.

    Both surfaces must read the same
    ``(value_fn, display_name, column, format)`` per kind: the per-class
    dataclass defaults derive their prefix AND their format from the table,
    and ``_apply_metric_label_explicit`` looks the kind up in the same
    table. A drift here is exactly the two-code-paths hazard MOD-05
    flagged. ``column`` (the broadcast metric column a ferrum diagnostic
    frame carries) joined the tuple at the batch G close, so *which frame
    column a surface reads* is a property of the metric kind rather than of
    the calling surface; ``display_name`` replaced the stored ``prefix`` in
    cycle 4 for the same reason, with ``prefix`` now derived from it, and
    ``format`` joined for the same reason at #219 (2026-09-13).

    Brier's value function is a ``_NoCurveDefinition`` rather than a
    callable: the drawn reliability curve does not define a Brier score, so
    there is no fallback to integrate (#219(3)).
    """
    from ferrum._metric_labels import (
        APLabel,
        AUCLabel,
        BrierLabel,
        _METRIC_LABEL_SPECS,
        _NoCurveDefinition,
        _ap_step,
        _trapezoid_auc,
    )

    assert _METRIC_LABEL_SPECS["auc"] == (_trapezoid_auc, "AUC", "auc", ".3f")
    assert _METRIC_LABEL_SPECS["ap"] == (_ap_step, "AP", "ap", ".3f")
    brier = _METRIC_LABEL_SPECS["brier"]
    assert (brier.display_name, brier.column, brier.format) == ("Brier", "brier", ".3f")
    assert isinstance(brier.value_fn, _NoCurveDefinition)
    assert brier.value_fn.reason  # the refusal quotes it; it must not be blank
    assert set(_METRIC_LABEL_SPECS) == {"auc", "ap", "brier"}

    expected_fns = {"auc": _trapezoid_auc, "ap": _ap_step, "brier": brier.value_fn}
    # The dataclass prefix/format defaults are the table's (no second copy).
    for cls, kind in ((AUCLabel, "auc"), (APLabel, "ap"), (BrierLabel, "brier")):
        assert cls().prefix == _METRIC_LABEL_SPECS[kind].prefix
        assert cls().format == _METRIC_LABEL_SPECS[kind].format
        assert cls._kind == kind
        # __radd__ resolves its value function FROM the table: i.e. the lookup
        # ``_METRIC_LABEL_SPECS[self._kind].value_fn`` returns the expected fn
        # for this class. (Pins the resolution, not just that ``_kind`` matches
        # a key.)
        assert _METRIC_LABEL_SPECS[cls._kind].value_fn is expected_fns[kind]


def test_metric_label_classes_registry_derived_consistent():
    """``_METRIC_LABEL_CLASSES`` is derived from the classes and stays in lockstep
    with ``_METRIC_LABEL_SPECS``: every spec kind has a class, every class's
    ``_kind`` is a spec key, and the explicit-field path constructs the right
    class per kind.
    """
    from ferrum._metric_labels import (
        APLabel,
        AUCLabel,
        BrierLabel,
        _METRIC_LABEL_CLASSES,
        _METRIC_LABEL_SPECS,
        _apply_metric_label_explicit,
    )

    # Registry is keyed by each class's own ``_kind`` (cannot desync).
    assert _METRIC_LABEL_CLASSES == {"auc": AUCLabel, "ap": APLabel, "brier": BrierLabel}
    # Bidirectional key agreement with the spec table.
    assert set(_METRIC_LABEL_CLASSES) == set(_METRIC_LABEL_SPECS)
    for kind, cls in _METRIC_LABEL_CLASSES.items():
        assert cls._kind == kind

    # The explicit-field path constructs the registry's class for each kind.
    # The frame carries every kind's broadcast column so the brier kind, which
    # has no curve fallback, can resolve a value here too.
    base = Chart(_roc_data().with_columns(ap=pl.lit(0.5), brier=pl.lit(0.25)))
    base = base.encode(x="fpr", y="tpr").mark_line()
    constructed: list[type] = []

    import ferrum._metric_labels as ml

    real_apply = ml._apply_metric_label

    def _spy(base_chart, label_obj, **kwargs):
        constructed.append(type(label_obj))
        return real_apply(base_chart, label_obj, **kwargs)

    ml._apply_metric_label = _spy
    try:
        for kind, cls in _METRIC_LABEL_CLASSES.items():
            constructed.clear()
            _apply_metric_label_explicit(base, kind, x_col="fpr", y_col="tpr")
            assert constructed == [cls], (kind, constructed, cls)
    finally:
        ml._apply_metric_label = real_apply


def test_metric_label_kind_is_classvar_not_a_field():
    """``_kind`` is a ``ClassVar`` so the frozen dataclass excludes it from
    ``fields()``/``eq``/``hash``; adding the annotation must not turn it into a
    dataclass field.
    """
    import dataclasses

    from ferrum._metric_labels import APLabel, AUCLabel, BrierLabel

    for cls in (AUCLabel, APLabel, BrierLabel):
        names = [f.name for f in dataclasses.fields(cls)]
        assert names == ["position", "format", "prefix"], (cls.__name__, names)
        assert "_kind" not in names


def test_metric_label_explicit_rejects_unknown_kind():
    """Figure-path entry point validates against the shared table."""
    import pytest

    from ferrum import Chart
    from ferrum._metric_labels import _apply_metric_label_explicit

    base = Chart(_roc_data()).encode(x="fpr", y="tpr").mark_line()
    with pytest.raises(
        ValueError, match=r"_apply_metric_label_explicit: label_kind must be one of"
    ):
        _apply_metric_label_explicit(base, "f1", x_col="fpr", y_col="tpr")


# ---- AUCLabel ----------------------------------------------------------------


def _roc_data(label: str = "c0", scale: float = 1.0):
    """Synthetic ROC: tpr = sqrt(fpr) yields AUC = 2/3 ≈ 0.667."""
    fpr = np.linspace(0, 1, 50)
    tpr = np.sqrt(fpr) * scale
    return pl.DataFrame({"fpr": fpr, "tpr": tpr, "class": [label] * len(fpr)})


def test_auc_label_default_format():
    df = _roc_data()
    chart = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line() + AUCLabel()
    svg = chart.to_svg()
    # tpr = sqrt(fpr) → AUC = 2/3 → 0.667
    assert "AUC = 0.667" in svg or "AUC = 0.666" in svg


def test_auc_label_custom_format_and_prefix():
    df = _roc_data()
    chart = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line() + AUCLabel(
        format=".2f", prefix="auc:"
    )
    svg = chart.to_svg()
    assert "auc:0.67" in svg


def test_auc_label_multi_class_emits_one_per_class():
    df = pl.concat([_roc_data("c0"), _roc_data("c1")])
    chart = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line() + AUCLabel()
    svg = chart.to_svg()
    assert svg.count("AUC =") == 2


# ---- APLabel -----------------------------------------------------------------


def _pr_data():
    """Synthetic PR: precision = 1 - 0.5 * recall yields AP ≈ 0.745."""
    recall = np.linspace(0, 1, 50)
    precision = 1.0 - 0.5 * recall
    return pl.DataFrame({"recall": recall, "precision": precision, "class": ["c0"] * len(recall)})


def test_ap_label_default():
    df = _pr_data()
    chart = Chart(df).encode(x="recall", y="precision", color="class").mark_line() + APLabel()
    svg = chart.to_svg()
    assert "AP = " in svg


# ---- BrierLabel --------------------------------------------------------------


def _calibration_data(*, brier: float | None = 0.042):
    """A reliability curve, with the per-sample Brier the kernel would attach.

    ``brier=None`` drops the column — the hand-built frame #219(3) refuses
    on, because the binned points alone cannot answer what the Brier score
    is.
    """
    p = np.linspace(0.05, 0.95, 19)
    obs = p.copy()
    df = pl.DataFrame({"predicted": p, "observed": obs, "model": ["m0"] * len(p)})
    if brier is None:
        return df
    return df.with_columns(brier=pl.lit(brier))


def test_brier_label_default():
    df = _calibration_data()
    chart = Chart(df).encode(x="predicted", y="observed", color="model").mark_line() + BrierLabel()
    svg = chart.to_svg()
    assert "Brier = 0.042" in svg


def test_brier_label_refuses_a_frame_without_the_brier_column():
    """#219(3): the silent bin-level-MSE fallback is deleted.

    ``+ BrierLabel()`` over a hand-built reliability frame used to render a
    number computed by integrating ``(predicted, observed)`` — a bin-level
    statistic that moves with ``n_bins=``/``strategy=``, labelled "Brier"
    as if it were the per-sample score. There is no honest number to show,
    so the label refuses, naming the column it needs and the frame it got.
    """
    import pytest

    df = _calibration_data(brier=None)
    chart = Chart(df).encode(x="predicted", y="observed", color="model").mark_line()
    with pytest.raises(ValueError) as excinfo:
        chart + BrierLabel()
    message = str(excinfo.value)
    assert message.startswith("BrierLabel: ")
    assert "'brier'" in message  # names the missing column
    assert "n_bins=/strategy=" in message  # names the hazard the fallback had
    # Reports the caller's own columns, not this helper's internal row index.
    assert "['model', 'observed', 'predicted']" in message
    assert "_idx" not in message


def test_brier_label_refuses_a_grouped_frame_with_an_all_null_group():
    """#219(3), cycle 2: ``_broadcast_metric`` returns ``None`` for three
    different facts (column absent / present-but-no-value /
    present-with-multiple-values), and the refusal used to claim the column
    was absent for all three. Here it is present -- one group's ``brier``
    is entirely null (e.g. a per-model concat where one model's Brier
    failed to compute) -- so the message must not send a reader looking for
    a column that its own "Columns present" list shows is right there.
    """
    import pytest

    have_brier = _calibration_data(brier=0.042)
    no_brier = _calibration_data(brier=None).with_columns(
        model=pl.lit("m1"), brier=pl.lit(None, dtype=pl.Float64)
    )
    df = pl.concat([have_brier, no_brier])
    chart = Chart(df).encode(x="predicted", y="observed", color="model").mark_line()
    with pytest.raises(ValueError) as excinfo:
        chart + BrierLabel()
    message = str(excinfo.value)
    assert message.startswith("BrierLabel: ")
    assert "the 'brier' column holds no non-null value for these rows" in message
    assert (
        "the data has no" not in message
    )  # untruthful for this shape -- the column is right there
    assert "'brier'" in message.split("Columns present:")[1]


def test_brier_label_refuses_an_ungrouped_frame_with_multiple_brier_values():
    """The other reachable column-present shape: an ungrouped frame holding
    more than one model's ``calibration_curve()`` rows (concatenated with no
    ``color=`` encoding to separate them), so ``brier`` carries two distinct
    values and no single broadcast value describes the frame.
    """
    import pytest

    two_models = pl.concat([_calibration_data(brier=0.042), _calibration_data(brier=0.099)]).drop(
        "model"
    )
    chart = Chart(two_models).encode(x="predicted", y="observed").mark_line()
    with pytest.raises(ValueError) as excinfo:
        chart + BrierLabel()
    message = str(excinfo.value)
    assert message.startswith("BrierLabel: ")
    assert "the 'brier' column holds more than one distinct value for these rows" in message
    assert "the data has no" not in message
    assert "'brier'" in message.split("Columns present:")[1]


def test_auc_and_ap_labels_still_integrate_a_column_less_frame():
    """The refusal is Brier's alone: for AUC/AP the drawn curve *is* the
    metric, so the composable path over arbitrary user data keeps working.
    """
    auc_chart = Chart(_roc_data()).encode(x="fpr", y="tpr").mark_line() + AUCLabel()
    ap_chart = Chart(_pr_data()).encode(x="recall", y="precision").mark_line() + APLabel()
    assert "auc" not in _roc_data().columns and "ap" not in _pr_data().columns
    assert "AUC = 0.66" in auc_chart.to_svg()
    assert "AP = 0.7" in ap_chart.to_svg()


# ---- OutlierLabel ------------------------------------------------------------


def test_outlier_label_threshold_3_emits_only_high_z():
    rng = np.random.default_rng(0)
    n = 1000
    residuals = rng.normal(0, 1, n)
    residuals[[100, 200, 300]] = 5.0
    residuals[[400, 500]] = -4.5
    obs_ids = [f"id_{i}" for i in range(n)]
    df = pl.DataFrame(
        {
            "fitted": np.linspace(0, 10, n),
            "residual": residuals,
            "obs_id": obs_ids,
        }
    )
    chart = Chart(df).encode(x="fitted", y="residual").mark_point() + OutlierLabel(
        threshold=3.0, field="residual", label_field="obs_id"
    )
    svg = chart.to_svg()
    for idx in (100, 200, 300, 400, 500):
        assert f">id_{idx}<" in svg


def test_outlier_label_max_labels_caps():
    rng = np.random.default_rng(0)
    n = 1000
    residuals = rng.normal(0, 1, n)
    residuals[:20] = 5.0
    df = pl.DataFrame(
        {
            "fitted": np.linspace(0, 10, n),
            "residual": residuals,
            "obs_id": [f"id_{i}" for i in range(n)],
        }
    )
    chart = Chart(df).encode(x="fitted", y="residual").mark_point() + OutlierLabel(
        threshold=3.0, field="residual", label_field="obs_id", max_labels=3
    )
    svg = chart.to_svg()
    count = sum(1 for i in range(20) if f">id_{i}<" in svg)
    assert count == 3


def test_outlier_label_no_outliers_returns_base_chart():
    rng = np.random.default_rng(1)
    n = 100
    df = pl.DataFrame(
        {
            "fitted": np.linspace(0, 10, n),
            "residual": rng.normal(0, 1, n),  # nothing above z=10
        }
    )
    chart = Chart(df).encode(x="fitted", y="residual").mark_point()
    out = chart + OutlierLabel(threshold=10.0)
    assert out is chart


# ---- annotate_arrow ----------------------------------------------------------


def test_annotate_arrow_without_label_uses_mark_segment():
    """Plain annotate_arrow yields a mark_segment chart."""
    chart = annotate_arrow(0.5, 1.5, 1.5, 3.5)
    assert chart._mark == "segment"
    assert "x" in chart._encoding and "y" in chart._encoding
    assert "x2" in chart._encoding and "y2" in chart._encoding


def test_annotate_arrow_with_label_is_plain_chart_with_label_annotation():
    """annotate_arrow(..., label=...) returns a plain Chart (GH #84) whose
    ``mark_segment`` layer draws the arrow standalone and whose
    ``_annotations`` carries the label as a real ``AnnotationText`` entry
    (so it renders standalone too), and whose ``_annotation_primitive`` is
    ``[AnnotationArrow, AnnotationText]`` so a ``+`` overlay carries both.

    Previously this composed via ``&`` and returned a ``VConcatChart``,
    which broke ``chart + annotate_arrow(..., label=...)`` (TypeError) and
    rendered the label in a separate panel below the arrow instead of
    overlaid in one panel.
    """
    from ferrum.annotation.primitives import AnnotationArrow, AnnotationText
    from ferrum.composition import VConcatChart

    chart = annotate_arrow(0.5, 1.5, 1.5, 3.5, label="trend")
    assert isinstance(chart, Chart)
    assert not isinstance(chart, VConcatChart)
    assert chart._mark == "segment"

    # Label renders standalone via a real `_annotations` entry.
    assert len(chart._annotations) == 1
    text_items = [item for item in chart._annotations[0].items if isinstance(item, AnnotationText)]
    assert len(text_items) == 1
    assert text_items[0].text == "trend"

    # `+` overlay carries both the arrow and its label via `_annotation_primitive`.
    assert isinstance(chart._annotation_primitive, list)
    assert len(chart._annotation_primitive) == 2
    assert isinstance(chart._annotation_primitive[0], AnnotationArrow)
    assert isinstance(chart._annotation_primitive[1], AnnotationText)
    assert chart._annotation_primitive[1].text == "trend"


def test_annotate_arrow_stroke_passes_through():
    chart = annotate_arrow(0, 0, 1, 1, stroke="#ff0000")
    assert chart._mark_kwargs.get("stroke") == "#ff0000"


# ---- Absent vs. present-but-NaN metric column (batch G close) ---------------


def test_broadcast_metric_propagates_a_present_nan():
    """A metric column the kernel computed as NaN says "undefined", which is
    a different fact from "this frame carries no such metric". Collapsing
    the two sent a degenerate curve down the integration fallback, which
    turned the one honest signal into a plausible-looking fabricated
    number.
    """
    import math

    from ferrum._metric_labels import _broadcast_metric

    present_nan = pl.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": [float("nan")] * 2})
    value = _broadcast_metric(present_nan, "auc")
    assert value is not None and math.isnan(value)

    absent = pl.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})
    assert _broadcast_metric(absent, "auc") is None

    two_curves = pl.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": [0.5, 0.9]})
    assert _broadcast_metric(two_curves, "auc") is None


def test_curve_metric_shows_nan_instead_of_integrating_a_broken_curve():
    """``_curve_metric`` only integrates when the frame carries no metric
    column at all. A NaN in the column reaches the display surface as NaN.
    """
    import math

    from ferrum._metric_labels import _curve_metric

    broken = pl.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": [float("nan")] * 2})
    assert math.isnan(_curve_metric(broken, kind="auc", x_col="fpr", y_col="tpr", where="test"))

    # Same curve, no column: the integration fallback still works, which is
    # what makes ``chart + AUCLabel()`` usable on arbitrary user data.
    user_data = pl.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})
    assert _curve_metric(user_data, kind="auc", x_col="fpr", y_col="tpr", where="test") == 0.5


# ---- One owner for the metric string (#219, 2026-09-13) ---------------------


def test_render_metric_is_the_only_producer_of_a_metric_string():
    """A metric kind carries one spelling and one rounding of its own value.

    Before this, six sites carried their own ``:.3f`` next to their own
    ``"AUC"``/``"AP"``/``"Brier"`` literal: the chart titles (three), the
    legend rename, the mark annotation and the overlay. Checked
    structurally rather than by grepping for the string, so it stays true
    for a metric kind added next year.
    """
    from ferrum._metric_labels import _METRIC_LABEL_SPECS, _default_format, _render_metric

    for kind, spec in _METRIC_LABEL_SPECS.items():
        # The prefix is derived from the display name, not stored beside it,
        # and the format is a property of the kind, not of a call site.
        assert spec.prefix == f"{spec.display_name} = "
        assert _default_format(kind) == spec.format
        assert _render_metric(kind, 0.6666666) == f"{spec.display_name} = 0.667"

    # The public per-label overrides still reach the rendered string — they
    # are the one reason the parameters exist.
    assert _render_metric("auc", 0.6666666, prefix="auc:", fmt=".2f") == "auc:0.67"


def test_every_metric_surface_renders_byte_identical_text():
    """Byte-parity pin (spec §6): every surface that shows a metric produces
    exactly ``_render_metric``'s bytes for the same value.

    The figure overlay, the ``mark_roc``/``mark_pr`` annotation and the
    composable ``+ AUCLabel()`` path are checked against the owner's output
    on one frame whose AUC is fixed by its broadcast column, so a surface
    that reintroduced its own format spec (``:.2f``, a thousands separator,
    a renamed metric) fails here rather than in a golden diff.
    """
    from ferrum._metric_labels import _render_metric
    from ferrum.marks._desugar_helpers import _inject_curve_annotation

    expected = _render_metric("auc", 0.875)
    assert expected == "AUC = 0.875"

    df = _roc_data().with_columns(auc=pl.lit(0.875))

    # 1. The composable label path.
    composed = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line() + AUCLabel()
    assert expected in _svg_texts(composed.to_svg())

    # 2. The mark annotation, which composes the same string after the group
    #    name rather than formatting its own.
    annotated = _inject_curve_annotation(
        df,
        group_field="class",
        kind="auc",
        prefix="_auc",
        x_col="fpr",
        y_col="tpr",
        where="test",
    )
    assert annotated["_auc_label"].drop_nulls().to_list() == [f"c0: {expected}"]

    # 3. The figure-function overlay path.
    from ferrum._metric_labels import _apply_metric_label_explicit

    figure_like = _apply_metric_label_explicit(
        Chart(df).encode(x="fpr", y="tpr").mark_line(),
        "auc",
        x_col="fpr",
        y_col="tpr",
        color_col="class",
    )
    assert expected in _svg_texts(figure_like.to_svg())


# ---- Re-applying a label replaces it (#219(2), 2026-09-13) ------------------


def _metric_label_layers(chart) -> list:
    from ferrum._metric_labels import _is_metric_label_layer

    return [layer for layer in (chart._layers or []) if _is_metric_label_layer(layer)]


def test_reapplying_a_metric_label_replaces_the_existing_layer():
    """``chart + AUCLabel()`` on a chart that already carries one must not
    stack an exact overstrike: same text, same coordinates, twice.
    """
    df = _roc_data().with_columns(auc=pl.lit(0.875))
    once = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line() + AUCLabel()
    twice = once + AUCLabel()
    thrice = twice + AUCLabel()

    assert len(_metric_label_layers(once)) == 1
    assert len(_metric_label_layers(twice)) == 1
    assert len(_metric_label_layers(thrice)) == 1
    assert len(twice._layers) == len(once._layers)
    assert twice.to_svg().count("AUC = 0.875") == 1
    assert twice.to_svg() == once.to_svg()


def test_reapplying_a_metric_label_adopts_the_new_labels_settings():
    """Replacement, not deduplication: the second label's ``format=`` and
    ``prefix=`` are what renders, and the first label's text is gone.
    """
    df = _roc_data().with_columns(auc=pl.lit(0.875))
    once = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line() + AUCLabel()
    replaced = once + AUCLabel(format=".1f", prefix="auc:")

    texts = _svg_texts(replaced.to_svg())
    assert "auc:0.9" in texts
    assert "AUC = 0.875" not in texts
    assert len(_metric_label_layers(replaced)) == 1


# ---- Null group values skipped (#220) ----------------------------------------


def test_apply_metric_label_skips_null_color_group(recwarn):
    """A null ``color_col`` value (e.g. an injected iso-line sentinel row)
    must not reach ``pl.col(color_col) == cls``, which always evaluates to
    null in polars and warns "Comparisons with None always result in
    null" -- the same guard ``_curve_metrics_by_group`` already applies.
    """
    df = pl.concat(
        [
            _roc_data("c0"),
            _roc_data("c1"),
            pl.DataFrame({"fpr": [0.5], "tpr": [0.5], "class": [None]}),
        ],
        how="diagonal",
    )
    chart = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line() + AUCLabel()
    svg = chart.to_svg()
    assert svg.count("AUC =") == 2
    user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert not user_warnings, [str(w.message) for w in user_warnings]


def test_broadcast_metric_rejects_an_unknown_kind():
    """Every metric lookup validates its kind. `_broadcast_metric` was the
    one direct caller that bypassed `_curve_metric`'s check, so a typo gave
    a bare `KeyError` from the table instead of a named error.
    """
    import pytest

    from ferrum._metric_labels import _broadcast_metric

    df = pl.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": [0.5, 0.5]})
    with pytest.raises(ValueError, match=r"_broadcast_metric.*kind"):
        _broadcast_metric(df, "brier_score")
