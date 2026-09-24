"""Phase 10b tests: classification curve marks + figure functions + visualizers."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

import ferrum
from tests.fixtures import load_dataset, load_fixture


@pytest.fixture(scope="module")
def binary_source():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    return ferrum.ModelSource(model, X, df["y"])


@pytest.fixture(scope="module")
def multi_source():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    return ferrum.ModelSource(model, X, df["y"])


# --- Mark-layer tests (Task 15) -------------------------------------


def test_mark_roc_renders_binary(binary_source):
    roc = binary_source.roc_curve()
    svg = ferrum.Chart(roc).mark_roc().to_svg()
    assert "<svg" in svg


def test_mark_roc_renders_multiclass(multi_source):
    roc = multi_source.roc_curve()
    svg = ferrum.Chart(roc).mark_roc().to_svg()
    assert "<svg" in svg


def test_mark_roc_annotate_auc_renders_text_label(binary_source):
    """annotate_auc=True emits a per-class text layer at fixed (x, y)
    coordinates with the formatted AUC value. The chart builder injects
    `_auc_label_x/_y/text` columns one-non-null-per-class, and Rust's
    mark_text skips null rows so exactly N labels render for N classes.
    """
    # Direct mark_roc(...) on a raw DataFrame doesn't go through the
    # chart builder, so the annotation columns won't be present and the
    # text layer will render zero labels. The figure-function path is
    # the canonical entry: build via ferrum.roc_chart(annotate_auc=True).
    svg = ferrum.roc_chart(
        binary_source,
        annotate_auc=True,
    ).to_svg()
    assert "<svg" in svg
    assert "<text" in svg
    # The formatted "AUC = 0." literal should appear in the SVG text.
    assert "AUC = 0." in svg


def test_mark_pr_renders(binary_source):
    pr = binary_source.pr_curve()
    svg = ferrum.Chart(pr).mark_pr().to_svg()
    assert "<svg" in svg


def test_mark_pr_iso_lines_renders_4_iso_curves(binary_source):
    """iso_lines=True appends F-score iso curves for F = 0.2, 0.4, 0.6, 0.8
    plus a text-label layer with the F-value at each curve's rightmost
    point. The chart builder owns the iso-row injection; the desugar
    emits one extra dashed line layer + one text layer."""
    svg = ferrum.pr_chart(binary_source, iso_lines=True).to_svg()
    assert "<svg" in svg
    for f in ("F=0.2", "F=0.4", "F=0.6", "F=0.8"):
        assert f in svg, f"iso label {f!r} missing from SVG"


def test_mark_pr_annotate_ap_renders_text_label(binary_source):
    """annotate_ap=True emits a per-class text layer at fixed (x, y)
    with the formatted AP value, color-coded to match each curve."""
    svg = ferrum.pr_chart(binary_source, annotate_ap=True).to_svg()
    assert "<svg" in svg
    assert "AP = 0." in svg


def test_mark_pr_annotate_ap_and_iso_lines_compose(binary_source):
    """The two annotation features compose: AP labels + iso curves
    render in the same chart without collision."""
    svg = ferrum.pr_chart(
        binary_source,
        annotate_ap=True,
        iso_lines=True,
    ).to_svg()
    assert "AP = 0." in svg
    assert "F=0.6" in svg


def test_mark_calibration_renders(binary_source):
    cal = binary_source.calibration_curve(n_bins=10)
    svg = ferrum.Chart(cal).mark_calibration().to_svg()
    assert "<svg" in svg


def test_mark_gain_renders(binary_source):
    gain = binary_source.cumulative_gain()
    svg = ferrum.Chart(gain).mark_gain().to_svg()
    assert "<svg" in svg


def test_mark_lift_renders(binary_source):
    lift = binary_source.lift_curve()
    svg = ferrum.Chart(lift).mark_lift().to_svg()
    assert "<svg" in svg


def test_mark_discrimination_threshold_renders(binary_source):
    dt = binary_source.discrimination_threshold(n_thresholds=20)
    long = dt.unpivot(
        index="threshold",
        on=["precision", "recall", "f1", "queue_rate"],
        variable_name="metric",
        value_name="value",
    )
    svg = ferrum.Chart(long).mark_discrimination_threshold().to_svg()
    assert "<svg" in svg


def test_mark_discrimination_threshold_renders_threshold_line(binary_source):
    """threshold_line=True emits exactly one vertical mark_rule at the
    F1-best threshold -- not one horizontal rule per (threshold, metric)
    data row.

    Regression for the discrimination-threshold rule layer inheriting the
    chart-level y="value" encoding through the layered/desugar path: with
    n_thresholds=20 and 4 metrics, an un-fixed render draws 80 horizontal
    dashed rules (one per data row, spanning the full plot width) and zero
    vertical ones. This asserts the dashed threshold-line layer's own
    `<line>` element (identified by its distinctive
    ``stroke-dasharray="4,4"`` style, which no gridline/tick/axis line
    carries) is emitted exactly once, is vertical (x1 == x2), and spans a
    non-zero extent (y1 != y2) rather than being a degenerate point.
    """
    svg = ferrum.discrimination_threshold_chart(
        binary_source,
        threshold_line=True,
        n_thresholds=20,
    ).to_svg()
    assert "<svg" in svg

    dashed_lines = re.findall(r'<line\b[^>]*stroke-dasharray="4,4"[^>]*/>', svg)
    assert len(dashed_lines) == 1, (
        f"expected exactly 1 dashed threshold-line rule, got {len(dashed_lines)}: {dashed_lines}"
    )
    (line,) = dashed_lines
    x1, y1, x2, y2 = (
        float(re.search(rf'{attr}="([^"]+)"', line).group(1)) for attr in ("x1", "y1", "x2", "y2")
    )
    assert x1 == x2, f"threshold line must be vertical (x1 == x2); got x1={x1}, x2={x2}"
    assert y1 != y2, f"threshold line must span a non-zero height; got y1={y1}, y2={y2}"


def test_mark_discrimination_threshold_metrics_filter_matches_relabeled_names():
    """metrics=("f1",) must select the relabeled "F1" row, not silently
    render every metric.

    Regression for the loose-match gap left over from the #96 fix: the F1-
    optimum lookup in `_disc_threshold_prep` was fixed to match loosely via
    `_normalized_col` (case/spacing-insensitive), but the `metrics=` filter
    one block above it still exact-matched a Python `set`. A caller that
    relabels the metric column for display ("f1" -> "F1", exactly what
    `discrimination_threshold_chart` does) got that filter silently
    no-op'd -- `keep` came back empty, the loose-overlap check suppressed
    the warning, and every metric rendered unfiltered instead of just the
    requested one.
    """
    import polars as pl

    long_df = pl.DataFrame(
        {
            "threshold": [0.1, 0.2, 0.3, 0.4],
            "metric": ["Precision", "Recall", "F1", "Queue rate"],
            "value": [0.5, 0.6, 0.7, 0.2],
        }
    )
    chart = ferrum.Chart(long_df).mark_discrimination_threshold(metrics=("f1",))
    assert chart._data["metric"].to_list() == ["F1"], (
        f"metrics=('f1',) must filter down to the relabeled 'F1' row only; "
        f"got {chart._data['metric'].to_list()!r}"
    )


# --- #189: explicit-vs-default optimum_label/threshold_line -----------------
# `mark_discrimination_threshold`'s own `optimum_label`/`threshold_line`
# default to the `_UNSET` sentinel (`ferrum._desugar`) rather than a
# materialized bool, so `_prep_discrimination_threshold_sentinels` can tell
# "the caller explicitly asked for this" from "the caller never touched the
# flag" once a `metrics=` subset excludes `"f1"` and leaves neither sentinel
# column for the layer to read. An explicit ask refuses (self-contradicting:
# there is no F1-optimum row to label/rule at); the unset default keeps the
# pre-#189 silent downgrade.
_NO_F1_LONG_DF = pl.DataFrame(
    {
        "threshold": [0.1, 0.2, 0.3],
        "metric": ["precision"] * 3,
        "value": [0.5, 0.6, 0.7],
    }
)


@pytest.mark.parametrize("flag", ["optimum_label", "threshold_line"])
def test_mark_discrimination_threshold_explicit_flag_without_f1_refuses(flag):
    """Direct-API pin for #189's refusal shape: EXPLICIT ``optimum_label=True``
    or ``threshold_line=True`` on ``mark_discrimination_threshold`` with a
    ``metrics=`` subset that excludes ``"f1"`` raises a typed ``ValueError``
    naming both the flag and ``metrics=`` instead of silently rendering
    without the layer."""
    with pytest.raises(ValueError, match=f"{flag}.*metrics"):
        ferrum.Chart(_NO_F1_LONG_DF).mark_discrimination_threshold(
            metrics=("precision",), **{flag: True}
        ).to_svg()


def test_mark_discrimination_threshold_default_optimum_label_without_f1_stays_silent():
    """Direct-API pin for #189's default-path preservation: leaving
    ``optimum_label`` UNSET (the mark's own default) with a ``metrics=``
    subset that excludes ``"f1"`` must still silently omit the label --
    exactly batch-F T3's crash-shape fix, unaffected by the new explicit
    refusal above."""
    svg = (
        ferrum.Chart(_NO_F1_LONG_DF).mark_discrimination_threshold(metrics=("precision",)).to_svg()
    )
    assert "<svg" in svg
    assert "max F1" not in svg


def test_mark_discrimination_threshold_default_threshold_line_without_f1_stays_silent():
    """Sibling of the above for ``threshold_line`` -- its own default is
    ``False`` (never the value the refusal fires on), so leaving it UNSET
    with an f1-less ``metrics=`` renders without error or a rule."""
    svg = (
        ferrum.Chart(_NO_F1_LONG_DF).mark_discrimination_threshold(metrics=("precision",)).to_svg()
    )
    assert "<svg" in svg
    assert 'stroke-dasharray="4,4"' not in svg


def test_mark_discrimination_threshold_explicit_optimum_label_metrics_includes_f1_but_data_lacks_row():
    """Cycle-2 fix (quality-review REVISE): the refusal message must not
    unconditionally blame ``metrics=`` for excluding ``"f1"``.

    Here ``metrics=`` is left at the mark's own default, which genuinely
    *includes* ``"f1"`` -- the failure is that the caller's hand-built
    long-form data simply has no ``"f1"`` row in its ``metric`` column.
    Pre-fix this raised claiming ``metrics=(...) excludes 'f1'`` (false)
    with the remediation "Include 'f1' in metrics=" (already there,
    actively wrong advice). This is the exact live repro from the quality
    reviewer's REVISE verdict.
    """
    df = pl.DataFrame(
        {
            "threshold": [0.1, 0.2, 0.3],
            "metric": ["precision", "recall", "queue_rate"],
            "value": [0.5, 0.6, 0.2],
        }
    )
    with pytest.raises(ValueError, match="optimum_label") as excinfo:
        ferrum.Chart(df).mark_discrimination_threshold(optimum_label=True).to_svg()
    message = str(excinfo.value)
    assert "includes 'f1'" in message, message
    assert "excludes 'f1'" not in message, message
    assert "no 'f1' row" in message, message


@pytest.mark.parametrize("flag", ["optimum_label", "threshold_line"])
def test_discrimination_threshold_visualizer_explicit_flag_without_f1_refuses(flag):
    """Visualizer-level pin for #189 (spec-review coverage gap): an
    EXPLICIT ``optimum_label=True``/``threshold_line=True`` on
    ``DiscriminationThresholdVisualizer`` with a ``metrics=`` subset that
    excludes ``"f1"`` raises the same typed ``ValueError`` as the mark and
    figure-function paths.

    ``.fit()`` itself does not raise -- ``_materialize`` always computes
    the full metric set (including f1) from ``ModelSource.discrimination_
    threshold`` regardless of ``metrics=``, and ``_build_chart`` only
    assembles a lazy ``Chart``. The refusal fires when the pending mark is
    resolved, at ``.show().to_svg()``.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.DiscriminationThresholdVisualizer(
        model,
        n_thresholds=20,
        metrics=("precision", "recall"),
        **{flag: True},
    ).fit(df.select(["f0", "f1", "f2", "f3"]), df["y"])
    with pytest.raises(ValueError, match=f"{flag}.*metrics"):
        viz.show().to_svg()


def test_discrimination_threshold_visualizer_default_flags_without_f1_stay_silent():
    """Visualizer-level pin for #189's default-path preservation (spec-
    review coverage gap): leaving both ``optimum_label`` and
    ``threshold_line`` UNSET (their own defaults) with an f1-less
    ``metrics=`` still silently omits the label/rule -- the same
    batch-F T3 crash-shape fix the mark and figure-function paths keep,
    now pinned at the visualizer layer too.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.DiscriminationThresholdVisualizer(
        model,
        n_thresholds=20,
        metrics=("precision", "recall"),
    ).fit(df.select(["f0", "f1", "f2", "f3"]), df["y"])
    svg = viz.show().to_svg()
    assert "<svg" in svg
    assert "max F1" not in svg
    assert 'stroke-dasharray="4,4"' not in svg


def test_discrimination_threshold_chart_explicit_optimum_label_without_f1_refuses(binary_source):
    """Figure-level pin for #189's ``optimum_label`` refusal shape (spec-
    review coverage gap): sibling of
    ``test_discrimination_threshold_explicit_threshold_line_without_f1_refuses``
    (``tests/test_bug_hunt_model_diagnostics.py``), which only pins
    ``threshold_line`` at this layer. ``discrimination_threshold_chart``'s
    own ``optimum_label`` default also uses the ``_UNSET`` sentinel (the
    footprint deviation this task made to keep default-path silence
    intact for every caller who never touches ``optimum_label``), so an
    EXPLICIT ``optimum_label=True`` with an f1-less ``metrics=`` must
    raise here too.
    """
    chart = ferrum.discrimination_threshold_chart(
        binary_source,
        optimum_label=True,
        metrics=("precision", "recall"),
        n_thresholds=20,
    )
    with pytest.raises(ValueError, match="optimum_label.*metrics"):
        chart.to_svg()


def test_discrimination_threshold_chart_optimum_label_present_vs_absent(binary_source):
    """optimum_label=True renders the max-F1 annotation text on the
    figure-function path; optimum_label=False renders none of it.

    Regression for the F1-row case-mismatch bug (#96): `discrimination_
    threshold_chart` relabels the metric column to display names ("f1" ->
    "F1") before calling `mark_discrimination_threshold`, so the F1-optimum
    lookup in `_disc_threshold_prep` (`pl.col("metric") == "f1"`) matched
    zero rows and silently skipped the `_optimum_x`/`_optimum_y`/
    `_optimum_text` sentinel-column injection for every
    `discrimination_threshold_chart()` call -- not just `threshold_line=
    True` ones. Asserts on the actual rendered label text (not merely the
    presence of some `<text>` element, which the axis/legend/title already
    guarantee), and that the same text is genuinely absent when
    `optimum_label=False`.
    """
    svg_on = ferrum.discrimination_threshold_chart(
        binary_source,
        optimum_label=True,
        n_thresholds=20,
    ).to_svg()
    svg_off = ferrum.discrimination_threshold_chart(
        binary_source,
        optimum_label=False,
        n_thresholds=20,
    ).to_svg()

    label = re.search(r"max F1 = [\d.]+ @ t=[\d.]+", svg_on)
    assert label is not None, "optimum_label=True must render the max-F1 annotation text"
    assert svg_on.count(label.group(0)) == 1

    assert "max F1" not in svg_off, "optimum_label=False must not render the max-F1 annotation text"


def test_discrimination_threshold_chart_metrics_accepts_display_and_raw_names(binary_source):
    """`metrics=` on the figure-function path accepts both the raw sweep
    column name ("f1") and the display label the function itself produces
    ("F1") -- and filters the render down to just that metric in both
    cases, instead of crashing on one of them.

    Same-family regression as the mark-level `metrics=` loose-match fix
    (`_disc_threshold_prep` / `_normalized_col`): `_discrimination_
    threshold_chart_from_source` used to unpivot the wide sweep frame with
    `on=list(metrics)` *before* relabeling the `metric` column to display
    names, so passing the function's own output vocabulary back in
    (`metrics=("F1",)`) raised `polars.exceptions.ColumnNotFoundError:
    "F1" not found` -- the sweep frame only ever has the raw "f1" column.
    """
    svg_display = ferrum.discrimination_threshold_chart(
        binary_source, metrics=("F1",), n_thresholds=20
    ).to_svg()
    svg_raw = ferrum.discrimination_threshold_chart(
        binary_source, metrics=("f1",), n_thresholds=20
    ).to_svg()
    for svg in (svg_display, svg_raw):
        assert "<svg" in svg
        assert "F1" in svg
        for other in ("Precision", "Recall", "Queue rate"):
            assert other not in svg, f"{other!r} must not render when metrics=('F1',)"


def test_discrimination_threshold_chart_metrics_typo_raises_value_error(binary_source):
    """An unresolvable `metrics=` name on the figure-function path raises a
    typed `ValueError` naming the bad value and the valid vocabulary,
    instead of a bare `polars.exceptions.ColumnNotFoundError` leaking out
    of `DataFrame.unpivot`."""
    with pytest.raises(ValueError) as excinfo:
        ferrum.discrimination_threshold_chart(binary_source, metrics=("bogus",), n_thresholds=20)
    message = str(excinfo.value)
    assert "bogus" in message
    for valid in ("precision", "recall", "f1", "queue_rate"):
        assert valid in message


# --- Figure-function tests (Task 16) --------------------------------


def test_roc_chart_figure_function(binary_source):
    svg = ferrum.roc_chart(binary_source).to_svg()
    assert "<svg" in svg


def test_roc_chart_from_model():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    svg = ferrum.roc_chart(
        model,
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    ).to_svg()
    assert "<svg" in svg


def test_pr_chart_figure_function(binary_source):
    svg = ferrum.pr_chart(binary_source).to_svg()
    assert "<svg" in svg


def test_pr_chart_per_class_false_routes_to_macro(multi_source):
    """``per_class=False`` should render a single averaged curve. The
    routed DataFrame from ``ModelSource.pr_curve(average='macro')``
    carries a single ``class='macro'`` value (no per-class rows), so the
    chart contains exactly one line. Regression test for the silent-
    discard bug fixed in the post-merge cleanup sweep.
    """
    chart = ferrum.pr_chart(multi_source, per_class=False)
    svg = chart.to_svg()
    assert "<svg" in svg
    # Verify the underlying DataFrame holds one summary class (the macro
    # summary) rather than several per-class curves. The legend value is the
    # bare class name: #219/D4 retired the "(AP = x)" rename, leaving the
    # endpoint overlay as the single metric surface.
    classes = set(chart._data["class"].unique().to_list())
    assert classes == {"macro"}


def test_pr_chart_per_class_false_micro(multi_source):
    chart = ferrum.pr_chart(multi_source, per_class=False, average="micro")
    classes = set(chart._data["class"].unique().to_list())
    assert classes == {"micro"}


def test_calibration_chart_figure_function(binary_source):
    svg = ferrum.calibration_chart(binary_source, n_bins=5).to_svg()
    assert "<svg" in svg


def test_calibration_chart_multi_model_compared_source_passthrough():
    """calibration_chart accepts a pre-built ComparedModelSource as the
    sole positional argument, mirroring the sibling figure-function
    passthrough path (see test_compared_source_passthrough_via_figure in
    test_compare.py).
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    cms = ferrum.ModelSource.compare({"a": model, "b": model}, X, df["y"])
    chart = ferrum.calibration_chart(cms)
    assert "<svg" in chart.to_svg()


def test_gain_chart_figure_function(binary_source):
    svg = ferrum.gain_chart(binary_source).to_svg()
    assert "<svg" in svg


def test_lift_chart_figure_function(binary_source):
    svg = ferrum.lift_chart(binary_source).to_svg()
    assert "<svg" in svg


# --- F-L11-08 / G4: gain/lift show real class names, not a binary rename map ---


def _direct_label_texts(svg: str) -> set[str]:
    """Text content of gain/lift chart direct-label + legend `<text>` nodes,
    excluding axis-tick text.

    A naive `">0<" in svg` (or any other bare substring) check does NOT
    discriminate here: axis ticks render digit text too (e.g. the y-axis
    '0'..'1' gridline labels, the x-axis '0'..'1' percentage labels), so a
    chart with *zero* digit class labels anywhere (e.g. `class_names=["A",
    "B", "C"]`) still trips `">0<" in svg` / `">1<" in svg` via its ticks
    alone -- verified live against the spec reviewer's proof. Direct labels
    and axis ticks are visually distinguishable by SVG attributes even
    though both are `<text>`: ticks render in the theme's muted gray
    (`fill="#6b7280"`), while direct labels/legend text use the dark ink
    color (`fill="#1f2937"`) at the same `font-size="11"` the chart title
    and axis titles do NOT share (those are `font-size="13"
    font-weight="600"`). That combination -- dark fill, tick-sized font,
    no bold weight -- uniquely selects the direct-label layer.
    """
    return {
        content
        for attrs, content in re.findall(r"<text([^>]*)>([^<]*)</text>", svg)
        if 'fill="#1f2937"' in attrs and 'font-size="11"' in attrs
    }


@pytest.mark.parametrize("chart_fn", [ferrum.gain_chart, ferrum.lift_chart])
def test_gain_lift_binary_shows_real_names_not_class_map(binary_source, chart_fn):
    """Disclosed behavior change: binary charts now label '0'/'1' (the
    model's real integer class names) instead of the old hardcoded
    'Class 0'/'Class 1' rewrite."""
    svg = chart_fn(binary_source).to_svg()
    assert _direct_label_texts(svg) == {"0", "1", "Baseline"}
    assert "Class 0" not in svg
    assert "Class 1" not in svg


@pytest.mark.parametrize("chart_fn", [ferrum.gain_chart, ferrum.lift_chart])
def test_gain_lift_multiclass_shows_every_real_class_plus_baseline(multi_source, chart_fn):
    """A 3rd (and beyond) class must show its real name, not fall through
    as a bare unlabeled digit alongside a stale binary-only rename map."""
    svg = chart_fn(multi_source).to_svg()
    assert _direct_label_texts(svg) == {"0", "1", "2", "Baseline"}
    assert "Class 0" not in svg
    assert "Class 1" not in svg
    assert "Class 2" not in svg


class _StringClassesModel:
    """Wraps a fitted binary classifier and presents string ``classes_``.

    Exercises the ``model.classes_`` display-resolution arm specifically
    (as opposed to the constructor's ``class_names=`` override arm, covered
    directly by ``test_gain_lift_class_names_kwarg_*`` below), with ``y``
    and ``classes_`` both string-typed so ``y_true == class_value``
    binarizes correctly in ``lift_frame``/``gain_frame``.
    """

    def __init__(self, model, names):
        self._model = model
        self.classes_ = np.array(names)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


@pytest.mark.parametrize("chart_fn", [ferrum.gain_chart, ferrum.lift_chart])
def test_gain_lift_string_class_names_never_rewritten(chart_fn):
    """A model with real string classes_ (e.g. sklearn's classes_=[...])
    must show those strings untouched, with no Class-N leakage."""
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y_str = df["y"].cast(pl.Utf8).replace({"0": "setosa", "1": "versicolor"})
    source = ferrum.ModelSource(_StringClassesModel(model, ["setosa", "versicolor"]), X, y_str)
    svg = chart_fn(source).to_svg()
    assert _direct_label_texts(svg) == {"setosa", "versicolor", "Baseline"}
    assert "Class 0" not in svg
    assert "Class 1" not in svg


# --- G4 (F-L11-08): constructor `class_names=` must not silently drop or
# corrupt classes when the alias diverges from y's own encoding -----------


@pytest.mark.parametrize("chart_fn", [ferrum.gain_chart, ferrum.lift_chart])
def test_gain_lift_class_names_kwarg_shows_every_real_class(chart_fn):
    """Direct regression coverage for the constructor's ``class_names=``
    override path: an alias that doesn't round-trip back to ``y``'s own
    encoding (e.g. ``"negative"`` against an int-coded ``y``) must still
    binarize against the model's real ``classes_``, not the alias
    itself."""
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"], class_names=["negative", "positive"])
    svg = chart_fn(source).to_svg()
    assert _direct_label_texts(svg) == {"negative", "positive", "Baseline"}


@pytest.mark.parametrize(
    "y_transform, class_names",
    [
        pytest.param(lambda y: y, None, id="numeric_y_no_names"),
        pytest.param(lambda y: y.cast(pl.Utf8), None, id="utf8_y_no_names"),
        pytest.param(
            lambda y: y.cast(pl.Utf8).cast(pl.Categorical), None, id="categorical_y_no_names"
        ),
        pytest.param(lambda y: y.cast(pl.Utf8), ["A", "B", "C"], id="utf8_y_with_class_names"),
        pytest.param(lambda y: y, ["A", "B", "C"], id="numeric_y_with_class_names"),
        pytest.param(
            lambda y: y.cast(pl.Utf8).cast(pl.Enum(["2", "0", "1"])),
            None,
            id="enum_y_no_names",
        ),
    ],
)
def test_gain_lift_class_values_land_in_ys_domain_matrix(y_transform, class_names):
    """``class_values[i]`` must be coerced into y's domain regardless of
    which side of the resolution -- the display label, or
    ``real_classes`` (``model.classes_``) -- supplies the raw value.

    Covers every combination of ``y``'s dtype (numeric / Utf8 /
    Categorical / Enum) diverging from ``model.classes_``'s own dtype,
    with and without a ``class_names=`` display alias -- a mismatch on
    either side must not silently drop a class from ``lift_curve()``,
    flatline ``cumulative_gain()``, or produce a NaN AUC from
    ``roc_curve()``/``pr_curve()``.

    ``enum_y_no_names``'s category order (``["2", "0", "1"]``) is
    deliberately scrambled relative to the natural ``["0", "1", "2"]``
    value order: a coercion that reinterprets an int ``classes_``'s
    values as physical dictionary *codes* into that category list
    (rather than casting them) would silently produce a
    wrong-but-plausible-looking curve pinned to the wrong classes,
    instead of an honest failure -- digit-string categories keep
    ``roc_curve()``'s numeric parsing intact so this row isolates the
    coercion defect from the unrelated, pre-existing, out-of-scope
    ``_curve_frames.roc_frame`` binary-branch gap (tracked separately,
    not fixed here).
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = y_transform(df["y"])
    kwargs = {} if class_names is None else {"class_names": class_names}
    source = ferrum.ModelSource(model, X, y, **kwargs)

    gain = source.cumulative_gain()
    lift = source.lift_curve()
    gain_real_classes = set(gain["class"].unique().to_list()) - {"baseline"}
    lift_real_classes = set(lift["class"].unique().to_list()) - {"baseline"}
    assert len(gain_real_classes) == 3, f"cumulative_gain() lost classes: {gain_real_classes}"
    assert lift_real_classes == gain_real_classes, (
        f"lift_curve() {lift_real_classes} disagrees with cumulative_gain() {gain_real_classes}"
    )
    for cls in gain_real_classes:
        assert (gain.filter(pl.col("class") == cls)["gain"] > 0).any(), (
            f"gain for class {cls!r} is flatlined at 0"
        )
        assert (lift.filter(pl.col("class") == cls)["lift"] != 1.0).any(), (
            f"lift for class {cls!r} is flatlined at 1.0"
        )

    # roc_curve()/pr_curve() feed the same _resolve_proba binarization, so
    # a dropped/mismatched class_values entry shows up as NaN AUC there too.
    roc = source.roc_curve()
    assert not roc["auc"].is_nan().any(), "roc_curve() produced NaN AUC"

    if class_names is None:
        # Non-degenerate and non-NaN is not enough: a coercion that
        # reinterprets values as physical codes instead of casting them
        # can silently attach a class's real curve to the WRONG label --
        # still non-degenerate, still non-NaN, just pinned to the wrong
        # class (exactly the failure mode a scrambled-order Enum column
        # is built to expose). Pin per-label AUC against a reference
        # source built from the untransformed int y, whose "0"/"1"/"2"
        # labels are unambiguous stringified class indices -- correct
        # coercion must attribute the identical AUC to the identical
        # label regardless of what dtype y was actually supplied as.
        reference_roc = ferrum.ModelSource(model, X, df["y"]).roc_curve()
        reference_auc = dict(zip(reference_roc["class"].to_list(), reference_roc["auc"].to_list()))
        for cls, auc in zip(roc["class"].to_list(), roc["auc"].to_list()):
            assert cls in reference_auc, f"unexpected class label {cls!r}"
            assert auc == pytest.approx(reference_auc[cls], abs=1e-9), (
                f"class {cls!r} AUC {auc} != reference {reference_auc[cls]} -- "
                "class values were mispaired, not just coerced"
            )

    if class_names is not None:
        # The rendered chart only carries string display labels; string
        # y-dtype rows (utf8/categorical) still render their raw values
        # as the "class" discriminator when class_names is None, which
        # the earlier binary/multiclass/string-classes tests already
        # cover -- this direct-label check is specific to the
        # class_names= override rows, where the display alias must
        # survive to the SVG on both chart functions.
        for chart_fn in (ferrum.gain_chart, ferrum.lift_chart):
            svg = chart_fn(source).to_svg()
            assert _direct_label_texts(svg) == set(class_names) | {"Baseline"}


def test_discrimination_threshold_class_names_alias_does_not_zero_the_sweep():
    """``discrimination_threshold()`` binarizes against the model's real
    positive class, not a ``class_names=`` display alias -- an alias
    that never matches ``y``'s own domain must not silently produce an
    all-zero precision/recall/F1 sweep."""
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"], class_names=["negative", "positive"])
    dt = source.discrimination_threshold(n_thresholds=20)
    assert dt["precision"].max() > 0.0
    assert dt["recall"].max() > 0.0


def test_confusion_matrix_class_names_alias_shows_real_counts():
    """``confusion_matrix()`` encodes against the model's real classes and
    applies ``class_names=`` as a display-only rename afterward -- an
    alias that never matches ``y``'s own domain must not silently zero
    every cell."""
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"], class_names=["negative", "positive"])
    cm = source.confusion_matrix()
    assert cm["value"].sum() == df.height
    assert set(cm["actual"].unique().to_list()) == {"negative", "positive"}
    assert set(cm["predicted"].unique().to_list()) == {"negative", "positive"}


def _fit_fresh_binary_source(y_transform, *, class_names=None):
    """Fit a brand-new ``LogisticRegression`` on the binary_classification
    fixture's X with y cast via ``y_transform``, so ``model.classes_``
    shares y's exact dtype by construction (the pre-fit ``binary_logistic``
    fixture is always int-coded, so it can't exercise this).

    This is the identity case for the class-value coercion mechanism:
    the comparison value already lives in y's domain because the model
    was fit on y itself, so any coercion the source applies must be a
    no-op that preserves it, not a lossy round-trip that corrupts it.
    """
    from sklearn.linear_model import LogisticRegression

    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = y_transform(df["y"])
    model = LogisticRegression(random_state=0, max_iter=500).fit(X.to_numpy(), y.to_numpy())
    kwargs = {} if class_names is None else {"class_names": class_names}
    return ferrum.ModelSource(model, X, y, **kwargs), df.height


@pytest.mark.parametrize(
    "y_transform",
    [
        pytest.param(lambda y: y, id="int"),
        pytest.param(lambda y: y.cast(pl.Float64), id="float"),
        pytest.param(lambda y: y.cast(pl.Utf8), id="utf8"),
        pytest.param(lambda y: y.cast(pl.Utf8).cast(pl.Categorical), id="categorical"),
        pytest.param(lambda y: y.cast(pl.Boolean), id="bool"),
    ],
)
@pytest.mark.parametrize(
    "class_names",
    [
        pytest.param(None, id="no_class_names"),
        pytest.param(["negative", "positive"], id="with_class_names"),
    ],
)
def test_class_values_dtype_matrix_every_dtype_nondegenerate(y_transform, class_names):
    """Exhaustive dtype x class_names= matrix across every method that
    binarizes a class value against y: ``confusion_matrix``,
    ``discrimination_threshold``, ``cumulative_gain``, ``lift_curve``
    (plus ``roc_curve`` as a spot AUC check). Every cell must produce
    non-degenerate output for every dtype y can legally carry -- int,
    float, Utf8, Categorical, Boolean -- with or without a
    ``class_names=`` display alias.

    Every row here fits a fresh model directly on y itself, so
    ``model.classes_`` shares y's domain trivially (the identity case)
    -- this proves the coercion mechanism is a safe no-op for values
    already in y's domain. ``Enum`` is deliberately NOT a row here: an
    Enum y's real class values already round-trip losslessly through
    this identity design regardless of category order (a same-value
    cast into any Enum matches by string value, not list position), so
    it cannot exercise the dictionary-code-reinterpretation defect --
    see ``test_gain_lift_class_values_land_in_ys_domain_matrix``'s
    ``enum_y_no_names`` row, which uses a fixed int-fitted model against
    a differently-typed y precisely to exercise cross-dtype mismatches.

    Guards against a class-value coercion mechanism that only handles a
    subset of dtypes and silently zeroes the rest -- the recurring
    all-zero-output defect this module's ``class_names=`` fix has hit
    at three different call sites (``confusion_matrix``,
    ``discrimination_threshold``, the ``_resolve_proba`` consumers).
    """
    source, n = _fit_fresh_binary_source(y_transform, class_names=class_names)

    cm = source.confusion_matrix()
    assert cm["value"].sum() == n, f"confusion_matrix() degenerate: sum={cm['value'].sum()}"

    dt = source.discrimination_threshold(n_thresholds=20)
    assert dt["precision"].max() > 0.0, "discrimination_threshold() precision flatlined at 0"
    assert dt["recall"].max() > 0.0, "discrimination_threshold() recall flatlined at 0"

    gain = source.cumulative_gain()
    lift = source.lift_curve()
    gain_classes = set(gain["class"].unique().to_list()) - {"baseline"}
    lift_classes = set(lift["class"].unique().to_list()) - {"baseline"}
    assert len(gain_classes) == 2, f"cumulative_gain() lost classes: {gain_classes}"
    assert lift_classes == gain_classes, (
        f"lift_curve() {lift_classes} disagrees with cumulative_gain() {gain_classes}"
    )
    for cls in gain_classes:
        assert (gain.filter(pl.col("class") == cls)["gain"] > 0).any(), (
            f"gain for class {cls!r} is flatlined at 0"
        )
        assert (lift.filter(pl.col("class") == cls)["lift"] != 1.0).any(), (
            f"lift for class {cls!r} is flatlined at 1.0"
        )

    roc = source.roc_curve()
    assert not roc["auc"].is_nan().any(), "roc_curve() produced NaN AUC"


def test_discrimination_threshold_chart_figure_function(binary_source):
    svg = ferrum.discrimination_threshold_chart(
        binary_source,
        n_thresholds=20,
    ).to_svg()
    assert "<svg" in svg


# --- Visualizer tests (Task 17) --------------------------------------


def test_roc_visualizer():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.ROCVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "auc_mean=" in repr(viz)
    assert 0.0 <= viz._metrics["auc_mean"] <= 1.0
    assert "<svg" in viz.show().to_svg()


def test_roc_visualizer_score():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    viz = ferrum.ROCVisualizer(model).fit(X, df["y"])
    auc = viz.score(X.to_numpy(), df["y"].to_numpy())
    assert 0.0 <= auc <= 1.0


def test_pr_visualizer():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.PRVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "ap_mean=" in repr(viz)
    assert 0.0 <= viz._metrics["ap_mean"] <= 1.0
    assert "<svg" in viz.show().to_svg()


def test_calibration_visualizer():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.CalibrationVisualizer(model, n_bins=5).fit(
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "calibration_error=" in repr(viz)
    assert viz._metrics["calibration_error"] >= 0.0
    assert "<svg" in viz.show().to_svg()


def test_calibration_visualizer_multi_model_variadic():
    """Multi-model CalibrationVisualizer (Phase 10h) routes through
    ComparedModelSource, auto-naming each positional model 'model_0',
    'model_1', etc. and rendering an overlay reliability diagram.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    viz = ferrum.CalibrationVisualizer(model, model, n_bins=5).fit(X, df["y"])
    # Two models overlaid -> one keyed calibration_error per model, since
    # no single pooled number describes them both.
    assert "calibration_error[model_0]=" in repr(viz)
    assert "calibration_error[model_1]=" in repr(viz)
    assert "<svg" in viz.show().to_svg()


def test_calibration_visualizer_multi_model_error_is_per_model():
    """Each overlaid model's ``calibration_error`` must be its own -- equal
    to the number that model reports alone -- not a single pooled mean over
    every model's rows. Pre-fix the visualizer computed one scalar over the
    whole compared frame while its own comment claimed per-model values.
    """
    from sklearn.linear_model import LogisticRegression

    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]
    loose = load_fixture("binary_logistic")
    strict = LogisticRegression(C=0.01, max_iter=1000).fit(X.to_numpy(), y.to_numpy())

    solo_loose = ferrum.CalibrationVisualizer(loose, n_bins=5).fit(X, y)
    solo_strict = ferrum.CalibrationVisualizer(strict, n_bins=5).fit(X, y)
    expected_loose = solo_loose._metrics["calibration_error"]
    expected_strict = solo_strict._metrics["calibration_error"]
    # The fixture has to discriminate: a pooled mean of two equal numbers
    # would be indistinguishable from the per-model answer.
    assert expected_loose != pytest.approx(expected_strict)

    overlay = ferrum.CalibrationVisualizer({"loose": loose, "strict": strict}, n_bins=5).fit(X, y)
    assert overlay._metrics["calibration_error[loose]"] == pytest.approx(expected_loose)
    assert overlay._metrics["calibration_error[strict]"] == pytest.approx(expected_strict)
    assert "calibration_error" not in overlay._metrics


def _calibration_base_and_rival():
    """A base model and a rival whose calibration genuinely differs.

    Shared by the ``compare=`` pins below: every one of them is only worth
    making if the base's number and the rival's number are distinguishable,
    so a pooled or rival-sourced answer cannot pass by coincidence.
    """
    from sklearn.linear_model import LogisticRegression

    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]
    base = load_fixture("binary_logistic")
    rival = LogisticRegression(C=0.01, max_iter=1000).fit(X.to_numpy(), y.to_numpy())
    return X, y, base, rival


def test_calibration_visualizer_compare_metrics_describe_the_base_model():
    """``compare=`` (#212) has ONE distinguished base: the positional model.

    Unlike the symmetric ``*models`` overlay, ``_metrics`` carries a single
    ``calibration_error`` -- the base model's own, identical to what the
    base alone reports -- so ``__repr__`` and ``.score()`` describe the same
    model the headline number does, exactly as for
    ``ROCVisualizer``/``PRVisualizer``/``DiscriminationThresholdVisualizer``.
    """
    X, y, base, rival = _calibration_base_and_rival()

    solo_base = ferrum.CalibrationVisualizer(base, n_bins=5).fit(X, y)
    solo_rival = ferrum.CalibrationVisualizer(rival, n_bins=5).fit(X, y)
    expected = solo_base._metrics["calibration_error"]
    # The fixture discriminates: pooling the two models' rows, or reading
    # the rival's, could not produce the base's number.
    assert expected != pytest.approx(solo_rival._metrics["calibration_error"])

    viz = ferrum.CalibrationVisualizer(base, n_bins=5, compare={"rival": rival}).fit(X, y)

    assert viz._source.model_names == ["base", "rival"]
    assert list(viz._metrics) == ["calibration_error"]
    assert viz._metrics["calibration_error"] == pytest.approx(expected)
    assert "calibration_error=" in repr(viz)
    assert "<svg" in viz.show().to_svg()


def test_calibration_visualizer_compare_scores_the_base_model():
    """A ``compare=`` overlay is scoreable; a symmetric ``*models`` one is not.

    The split is the whole point of the additive-parity decision: with a
    distinguished base, ``.score()`` has a well-defined answer (the base
    model's own), so ``has_score`` must be True -- while the symmetric form
    keeps its documented ``0.0`` no-single-score fallback.
    """
    X, y, base, rival = _calibration_base_and_rival()

    compared = ferrum.CalibrationVisualizer(base, n_bins=5, compare={"rival": rival}).fit(X, y)
    assert compared.has_score is True
    assert compared.score(X, y) == pytest.approx(float(base.score(X, y)))

    symmetric = ferrum.CalibrationVisualizer(base, rival, n_bins=5).fit(X, y)
    assert symmetric.has_score is False
    assert symmetric.score(X, y) == 0.0


@pytest.mark.parametrize(
    "models",
    [
        pytest.param(lambda base, rival: (base, rival), id="two_positional"),
        pytest.param(lambda base, rival: ({"a": base, "b": rival},), id="dict_positional"),
    ],
)
def test_calibration_visualizer_refuses_models_and_compare_together(models):
    """The two multi-model idioms are mutually exclusive, and say so.

    ``*models`` is symmetric (no base, per-model keys, no score) while
    ``compare=`` distinguishes a base -- there is no answer that serves both
    contracts, so the contradictory request is refused at construction
    rather than silently resolved one way.
    """
    _X, _y, base, rival = _calibration_base_and_rival()

    with pytest.raises(ValueError, match=r"not both; got .* together with compare="):
        ferrum.CalibrationVisualizer(*models(base, rival), compare={"rival": rival})


def test_calibration_visualizer_compare_refusals_match_the_curve_visualizers():
    """``compare=``'s malformed-input refusals come from the shared owner.

    Routing ``fit`` through ``_build_compare_source``'s delegation to
    ``ModelSource._from_base_and_compare`` (#214) is what buys the
    reserved-``"base"`` guard and the non-dict refusal for free; these two
    pins fail if the class grows its own private resolution instead.
    """
    X, y, base, rival = _calibration_base_and_rival()

    with pytest.raises(ValueError, match=r"reserved for the primary"):
        ferrum.CalibrationVisualizer(base, compare={"base": rival}).fit(X, y)
    with pytest.raises(TypeError, match=r"compare= must be dict\[str, model\] or None"):
        ferrum.CalibrationVisualizer(base, compare=[rival]).fit(X, y)


def test_discrimination_threshold_visualizer():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.DiscriminationThresholdVisualizer(
        model,
        n_thresholds=20,
    ).fit(df.select(["f0", "f1", "f2", "f3"]), df["y"])
    assert "best_threshold=" in repr(viz)
    assert 0.0 <= viz._metrics["best_threshold"] <= 1.0
    assert 0.0 <= viz._metrics["best_f1"] <= 1.0
    assert "<svg" in viz.show().to_svg()


def test_discrimination_threshold_visualizer_threshold_line():
    """threshold_line=True on the visualizer should render the same
    vertical F1-best rule as the figure-level function. Without this
    kwarg sklearn-protocol users could not get the rule at all.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.DiscriminationThresholdVisualizer(
        model,
        n_thresholds=20,
        threshold_line=True,
    ).fit(df.select(["f0", "f1", "f2", "f3"]), df["y"])
    svg = viz.show().to_svg()
    # mark_rule emits an SVG <line> for the vertical span.
    assert "<line " in svg


def test_discrimination_threshold_visualizer_cv_forwards_and_is_deterministic():
    """``cv=`` forwards straight through to
    ``ModelSource.discrimination_threshold`` (#211).

    Comparing two ``cv=5`` visualizer instances against each other only
    pins determinism -- it does not discriminate a severed forwarding path:
    a ``_materialize`` that silently hardcoded ``cv=None`` would still pass
    that check, since two ``cv=None`` runs agree with each other too. This
    instead ties the visualizer's ``_metrics`` to a directly-computed
    ``ModelSource(...).discrimination_threshold(cv=5)`` frame's own
    F1-optimum point, built from the exact same ``(model, X, y,
    random_state)`` the visualizer's ``_build_source`` uses for a plain
    (non-``compare=``) fit, and separately proves ``cv=5`` was genuinely
    used -- not silently substituted with ``cv=None`` -- by first
    confirming the two disagree on this fixture.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X, y = df.select(["f0", "f1", "f2", "f3"]), df["y"]

    shadow = ferrum.ModelSource(model, X, y, random_state=None)

    cv5 = shadow.discrimination_threshold(n_thresholds=20, cv=5)
    idx_cv5 = cv5["f1"].arg_max()
    expected_threshold = float(cv5["threshold"][idx_cv5])
    expected_f1 = float(cv5["f1"][idx_cv5])

    cv_none = shadow.discrimination_threshold(n_thresholds=20, cv=None)
    idx_none = cv_none["f1"].arg_max()
    none_threshold = float(cv_none["threshold"][idx_none])
    none_f1 = float(cv_none["f1"][idx_none])

    # The fixture has to discriminate, or a severed forwarding path that
    # silently fell back to cv=None would be invisible below.
    assert (expected_threshold, expected_f1) != (none_threshold, none_f1)

    viz = ferrum.DiscriminationThresholdVisualizer(model, n_thresholds=20, cv=5).fit(X, y)
    assert viz._metrics["best_threshold"] == expected_threshold
    assert viz._metrics["best_f1"] == expected_f1
    assert (viz._metrics["best_threshold"], viz._metrics["best_f1"]) != (
        none_threshold,
        none_f1,
    )


def test_visualizer_unfit_repr():
    model = load_fixture("binary_logistic")
    viz = ferrum.ROCVisualizer(model)
    assert repr(viz) == "ROCVisualizer(unfit)"


# --- 10c: confusion matrix (Task 18) ----------------------------------


def test_confusion_matrix_schema(binary_source):
    cm = binary_source.confusion_matrix()
    assert set(cm.columns) == {"actual", "predicted", "value", "value_fmt"}
    assert cm.height == 4  # 2x2 for binary


def test_confusion_matrix_normalized(binary_source):
    import polars as pl

    cm = binary_source.confusion_matrix(normalize="true")
    assert cm.height == 4
    row_sums = cm.group_by("actual").agg(pl.col("value").sum()).sort("actual")["value"].to_list()
    for s in row_sums:
        assert abs(s - 1.0) < 1e-9


def test_confusion_matrix_multiclass(multi_source):
    cm = multi_source.confusion_matrix()
    assert cm.height == 9  # 3x3 for 3-class


def test_confusion_matrix_caching(binary_source):
    assert binary_source.confusion_matrix() is binary_source.confusion_matrix()


def test_mark_confusion_renders(binary_source):
    cm = binary_source.confusion_matrix(normalize="true")
    svg = ferrum.Chart(cm).mark_confusion(normalize="true").to_svg()
    assert "<svg" in svg
    # Annotated by default — text labels appear (data labels + axis labels).
    assert svg.count("<text ") >= 4


def test_mark_confusion_annotate_off(binary_source):
    cm = binary_source.confusion_matrix(normalize=None)
    svg = ferrum.Chart(cm).mark_confusion(annotate=False).to_svg()
    assert "<svg" in svg


def test_confusion_matrix_chart_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.confusion_matrix_chart(
        model,
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "<svg" in chart.to_svg()


def test_confusion_matrix_chart_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.confusion_matrix_chart(
        model,
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
        normalize="true",
    )
    assert "<svg" in chart.to_svg()


# --- 10c: class prediction error (Task 19) ----------------------------


def test_mark_class_prediction_error_renders(multi_source):
    cm = multi_source.confusion_matrix(normalize=None)
    svg = ferrum.Chart(cm).mark_class_prediction_error().to_svg()
    assert "<svg" in svg


def test_class_prediction_error_chart_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.class_prediction_error_chart(
        model,
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "<svg" in chart.to_svg()


def test_class_prediction_error_chart_normalized():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.class_prediction_error_chart(
        model,
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
        normalize=True,
    )
    assert "<svg" in chart.to_svg()


# --- 10c: visualizers (Task 20) ---------------------------------------


def test_confusion_matrix_visualizer():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    viz = ferrum.ConfusionMatrixVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "accuracy=" in repr(viz)
    assert 0.0 <= viz._metrics["accuracy"] <= 1.0
    assert "<svg" in viz.show().to_svg()


def test_classification_report_visualizer():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    viz = ferrum.ClassificationReportVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "f1_macro=" in repr(viz)
    assert 0.0 <= viz._metrics["f1_macro"] <= 1.0
    assert "<svg" in viz.show().to_svg()


def test_class_prediction_error_visualizer():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    viz = ferrum.ClassPredictionErrorVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]),
        df["y"],
    )
    assert "accuracy=" in repr(viz)
    assert "<svg" in viz.show().to_svg()


def test_class_balance_visualizer_y_only():
    df = load_dataset("multiclass_classification")
    viz = ferrum.ClassBalanceVisualizer().fit(df["y"])
    assert "n_classes=" in repr(viz)
    assert "imbalance_ratio=" in repr(viz)
    assert viz._metrics["n_classes"] >= 2.0
    assert viz._metrics["imbalance_ratio"] >= 1.0
    assert "<svg" in viz.show().to_svg()


def test_class_balance_visualizer_xy_signature():
    df = load_dataset("multiclass_classification")
    # sklearn-shape .fit(X, y) should also work; X is ignored.
    X = df.select(["f0", "f1", "f2", "f3"])
    viz = ferrum.ClassBalanceVisualizer().fit(X, df["y"])
    assert "<svg" in viz.show().to_svg()


# ---------------------------------------------------------------------------
# Public figure-function smoke tests
# ---------------------------------------------------------------------------


def test_classification_report_chart_returns_chart():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    chart = ferrum.classification_report_chart(model, X, df["y"])
    assert "<svg" in chart.to_svg()


# --- G3: precomputed path + title + X-less refusal (F-L11-05 + F-L11-09) ---


def test_classification_report_chart_has_title():
    """F-L11-09: the chart carries a title like every sibling — pre-fix
    ``chart._title`` was ``None``.
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    chart = ferrum.classification_report_chart(model, X, df["y"])
    assert chart._title is not None
    assert chart._title.text == "Classification Report"


def test_classification_report_chart_precomputed_renders():
    """F-L11-05: ``y_true=``/``y_pred=`` must work — pre-fix this raised
    ``TypeError: classification_report_chart() got an unexpected keyword
    argument 'y_true'`` because the tenth precomputed-family member never
    forwarded these kwargs into ``_resolve_source``.
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y_pred = model.predict(X.to_numpy())
    chart = ferrum.classification_report_chart(y_true=df["y"].to_numpy(), y_pred=y_pred)
    assert "<svg" in chart.to_svg()
    assert chart._title.text == "Classification Report"


def test_classification_report_chart_precomputed_matches_model_path():
    """Precomputed and model-path reports agree to machine precision on
    identical predictions (G3 acceptance criterion 4).

    Sorts both frames by ``(class, metric)`` before comparing so this test's
    job stays value parity, not axis order -- the two paths' class-axis
    render order is pinned separately, and identically since #224a, by
    ``test_classification_report_chart_class_axis_order_now_identical``.
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y_true = df["y"].to_numpy()
    y_pred = model.predict(X.to_numpy())

    model_chart = ferrum.classification_report_chart(model, X, df["y"])
    precomputed_chart = ferrum.classification_report_chart(y_true=y_true, y_pred=y_pred)

    model_rows = model_chart._data.sort(["class", "metric"])[
        ["class", "metric", "value"]
    ].to_dicts()
    precomputed_rows = precomputed_chart._data.sort(["class", "metric"])[
        ["class", "metric", "value"]
    ].to_dicts()
    assert [(r["class"], r["metric"]) for r in model_rows] == [
        (r["class"], r["metric"]) for r in precomputed_rows
    ]
    for model_row, precomputed_row in zip(model_rows, precomputed_rows):
        assert model_row["value"] == pytest.approx(precomputed_row["value"], abs=1e-9)


def test_classification_report_chart_class_axis_order_now_identical():
    """#224a: the model path and the precomputed path now order the class
    axis identically -- both by sklearn's ``unique_labels`` convention (the
    union of the raw values, ascending) -- closing the documented divergence
    this test used to pin. Before the fix, the precomputed path sorted its
    already-stringified ``confusion_matrix`` labels with ``key=str``, which
    orders by digit count before value and so puts ``"10"``/``"11"`` before
    ``"2"``; the model path's sklearn-backed ``classification_report`` never
    stringified before sorting, so the two disagreed. Verified at both the
    frame level and the rendered SVG's y-axis tick order (the user-visible
    surface #224a's verify-then-file text is actually about), on a 12-class
    fixture. Twelve classes is *not* the smallest shape the two rules
    disagree on -- decimal-string order and numeric order part company as
    soon as two labels have unequal digit counts, which needs only two
    classes (``{2, 10}``, pinned by
    ``test_precomputed_classification_report_order_moves_below_ten_classes``).
    The 12-class fixture is used here because it puts a full
    ``10``/``11``-before-``2`` interleave on the rendered axis.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from tests._svg_extents import axis_tick_labels

    n_classes = 12
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = rng.integers(0, n_classes, size=400)
    # Guarantee every class appears at least once in y_true so both paths'
    # label sets (unique_labels / sorted(set(...))) cover all n_classes
    # regardless of what the fitted model actually predicts.
    X = np.vstack([X, np.zeros((n_classes, 3))])
    y = np.concatenate([y, np.arange(n_classes)])
    model = LogisticRegression(max_iter=200).fit(X, y)
    y_pred = model.predict(X)

    model_chart = ferrum.classification_report_chart(model, X, y)
    precomputed_chart = ferrum.classification_report_chart(y_true=y, y_pred=y_pred)

    model_order = list(dict.fromkeys(model_chart._data["class"].to_list()))
    precomputed_order = list(dict.fromkeys(precomputed_chart._data["class"].to_list()))

    expected_order = [str(c) for c in sorted(set(y.tolist()))]
    assert model_order == expected_order
    assert precomputed_order == expected_order, (
        "expected the precomputed path to retire its key=str order and "
        "match the model path's numeric-ascending unique_labels order"
    )

    # The rendered surface, not only the frame: both charts' y-axis tick
    # text must appear in the identical screen order.
    model_ticks = axis_tick_labels(model_chart.to_svg(), axis="y")
    precomputed_ticks = axis_tick_labels(precomputed_chart.to_svg(), axis="y")
    assert model_ticks == expected_order
    assert precomputed_ticks == expected_order
    assert model_ticks == precomputed_ticks


def test_precomputed_source_classification_report_hand_computed():
    """Precision/recall/F1 computed from confusion counts against a
    hand-computed reference (not sklearn) — class ``2`` is never predicted,
    exercising the zero-division-as-0 branch.
    """
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 1]

    chart = ferrum.classification_report_chart(y_true=y_true, y_pred=y_pred)
    rows = {(r["class"], r["metric"]): r["value"] for r in chart._data.to_dicts()}

    expected = {
        ("0", "precision"): 1.0,
        ("0", "recall"): 1.0,
        ("0", "f1-score"): 1.0,
        ("1", "precision"): 0.5,
        ("1", "recall"): 1.0,
        ("1", "f1-score"): 2 * 0.5 * 1.0 / (0.5 + 1.0),
        ("2", "precision"): 0.0,
        ("2", "recall"): 0.0,
        ("2", "f1-score"): 0.0,
    }
    for key, expected_value in expected.items():
        assert rows[key] == pytest.approx(expected_value, abs=1e-9), key


def test_classification_report_chart_model_path_requires_X():
    """G6 sibling refusal: ``classification_report_chart(model, y=...)`` with
    no ``X`` must raise the typed domain message — pre-fix
    ``model.predict(None)`` leaked sklearn's bare "Expected 2D array, got
    scalar array instead: array=None..." error.

    ``y`` is supplied here (unlike a bare ``classification_report_chart(model)``)
    so this test isolates the X guard specifically: the guard order is
    y-then-X (mirroring ``ModelSource.confusion_matrix``'s own order,
    ``sources/_classification.py:333-337``), so an X-and-y-less call would
    hit the y guard first — that case is
    ``test_classification_report_chart_model_path_requires_y`` below.
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    with pytest.raises(ValueError, match=r"classification_report.*requires the training data"):
        ferrum.classification_report_chart(model, y=df["y"])


def test_classification_report_chart_model_path_requires_y():
    """Sibling refusal to the X guard above, mirroring
    ``ModelSource.confusion_matrix``'s y-then-X guard order
    (``sources/_classification.py:333-337``): ``classification_report_chart(model,
    X)`` with no ``y`` must raise the typed domain message — pre-fix
    sklearn's ``classification_report(None, ...)`` leaked a foreign
    ``InvalidParameterError`` naming ``y_true``, a real keyword of this
    same chart function, so following the leaked message led to a dead
    end (``classification_report_chart(y_true=...)`` without ``y_pred=``
    raises a different error).
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    with pytest.raises(ValueError, match=r"classification_report\(\) requires y to be provided"):
        ferrum.classification_report_chart(model, X)


def test_class_balance_chart_returns_chart():
    df = load_dataset("multiclass_classification")
    chart = ferrum.class_balance_chart(df["y"])
    assert "<svg" in chart.to_svg()


def test_class_balance_chart_accepts_list():
    chart = ferrum.class_balance_chart([0, 1, 1, 2, 0, 2, 2])
    assert "<svg" in chart.to_svg()


def test_class_balance_chart_bars_colored_by_class():
    """Regression: bars must use distinct colors per class, not a single fill."""
    import re

    chart = ferrum.class_balance_chart([0, 1, 1, 2, 0, 2, 2])
    svg = chart.to_svg()
    fills = set(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg))
    assert len(fills) >= 3, f"Expected ≥3 distinct fill colors for 3 classes; got {fills}"


# ---------------------------------------------------------------------------
# Batch G close — mutation-gate pins
# ---------------------------------------------------------------------------


# Shared by every confusion-derived dtype matrix below (confusion itself and
# its #206 sibling, class_prediction_error, which renders the same frame
# wholesale) -- five cells covering numeric / Utf8 / Categorical / Enum y and
# a Utf8 y with a `class_names=` alias, the shape a #206 fix must clone
# per the batch-G close's own dtype matrix this mirrors.
_CONFUSION_DTYPE_MATRIX = [
    pytest.param(lambda y: y, None, id="numeric_y_no_names"),
    pytest.param(lambda y: y.cast(pl.Utf8), None, id="utf8_y_no_names"),
    pytest.param(lambda y: y.cast(pl.Utf8).cast(pl.Categorical), None, id="categorical_y_no_names"),
    pytest.param(lambda y: y.cast(pl.Utf8), ["A", "B", "C"], id="utf8_y_with_class_names"),
    pytest.param(
        lambda y: y.cast(pl.Utf8).cast(pl.Enum(["2", "0", "1"])), None, id="enum_y_no_names"
    ),
]


@pytest.mark.parametrize("y_transform, class_names", _CONFUSION_DTYPE_MATRIX)
def test_confusion_matrix_encode_values_land_in_ys_domain_matrix(y_transform, class_names):
    """``confusion_matrix`` is the third call site of the batch's
    display-label-vs-comparison-value rule, and the only one without a
    dtype matrix guarding it -- the gain/lift twin
    (``test_gain_lift_class_values_land_in_ys_domain_matrix``, whose shape
    this mirrors) killed the analogous break while this one survived.

    ``model.classes_`` is int-typed here; when ``y`` is Utf8 / Categorical
    / Enum, an uncoerced ``classes_`` matches nothing, the kernel maps every
    row to code ``-1`` and drops it, and the rendered matrix is all zeros.
    So the assertion is on the counts: every row of ``y`` must land in some
    cell, and the diagonal must be non-empty for a model that is not
    entirely wrong.
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = y_transform(df["y"])
    kwargs = {} if class_names is None else {"class_names": class_names}
    source = ferrum.ModelSource(model, X, y, **kwargs)

    cm = source.confusion_matrix()
    assert cm.height == 9  # 3 x 3 cells, none dropped
    assert float(cm["value"].sum()) == float(len(y)), (
        "counts vanished: a class value did not land in y's domain, so every "
        "(actual, predicted) pair encoded to -1 and the kernel dropped it"
    )
    diagonal = cm.filter(pl.col("actual") == pl.col("predicted"))
    assert float(diagonal["value"].sum()) > 0.0

    # Relabelling y cannot move a count: the matrix must be exactly the one
    # the numeric-y source produces, cell for cell.
    baseline = ferrum.ModelSource(model, X, df["y"]).confusion_matrix()
    assert cm["value"].to_list() == baseline["value"].to_list()

    # And at the user-visible surface. Only the cell texts are compared --
    # a `class_names=` alias legitimately changes the axis tick labels.
    def cells(src):
        texts = re.findall(
            r"<text[^>]*>([^<]*)</text>", ferrum.confusion_matrix_chart(src).to_svg()
        )
        return [t for t in texts if re.fullmatch(r"\d+\.\d+", t)]

    rendered = cells(source)
    assert rendered, "no numeric cells rendered at all"
    assert set(rendered) != {"0.00"}, "the rendered matrix is a grid of zeros"
    assert rendered == cells(ferrum.ModelSource(model, X, df["y"]))


@pytest.mark.parametrize("y_transform, class_names", _CONFUSION_DTYPE_MATRIX)
def test_class_prediction_error_chart_dtype_matrix_already_correct(y_transform, class_names):
    """#206 regression pin, not a fix: ``class_prediction_error_chart``
    renders ``ModelSource.confusion_matrix(normalize=None)`` wholesale
    (``_class_prediction_error_chart_from_source``), so it inherited the
    batch-G confusion fix by transitivity and never touched ``model.predict``
    directly -- there is no second dtype-domain translation for it to need.
    Verified by cloning ``test_confusion_matrix_encode_values_land_in_ys_domain_matrix``'s
    dtype matrix at this figure's own surface: the stacked-bar segment
    values must equal the numeric-y baseline for every dtype cell, and its
    goldens are the negative control -- clean-target goldens do not exercise
    this branch and must not move.
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = y_transform(df["y"])
    kwargs = {} if class_names is None else {"class_names": class_names}
    source = ferrum.ModelSource(model, X, y, **kwargs)
    baseline_source = ferrum.ModelSource(model, X, df["y"])

    segments = ferrum.class_prediction_error_chart(source)._data
    baseline = ferrum.class_prediction_error_chart(baseline_source)._data
    assert segments["value"].to_list() == baseline["value"].to_list()
    assert sum(segments["value"].to_list()) == float(len(y)), (
        "counts vanished: a class value did not land in y's domain"
    )

    svg = ferrum.class_prediction_error_chart(source).to_svg()
    assert "<svg" in svg


@pytest.mark.parametrize("y_transform, class_names", _CONFUSION_DTYPE_MATRIX)
def test_classification_report_chart_dtype_matrix_matches_numeric_baseline(
    y_transform, class_names
):
    """#206: the classification_report model path translates ``model.predict``
    into ``y``'s domain the same way the confusion-matrix fix does, so a
    utf8/Categorical/Enum ``y`` fit against an int-coded ``classes_`` (the
    ordinary shape when a user label-encodes for training and keeps the
    readable target for plotting) renders the SAME per-class values as the
    clean numeric-``y`` baseline, instead of leaking sklearn's own
    ``ValueError: Mix of label input types (string and number)`` before any
    metric is computed.
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = y_transform(df["y"])
    kwargs = {} if class_names is None else {"class_names": class_names}
    source = ferrum.ModelSource(model, X, y, **kwargs)
    baseline_source = ferrum.ModelSource(model, X, df["y"])

    report = ferrum.classification_report_chart(source)._data.sort(["class", "metric"])
    baseline = ferrum.classification_report_chart(baseline_source)._data.sort(["class", "metric"])
    assert report["value"].to_list() == pytest.approx(baseline["value"].to_list(), abs=1e-9)

    assert "<svg" in ferrum.classification_report_chart(source).to_svg()


def test_classification_report_visualizer_f1_macro_matches_numeric_baseline_utf8_y():
    """#206, ``ClassificationReportVisualizer`` sibling: ``_materialize``'s
    ``f1_macro`` must not disagree with the numeric-``y`` baseline just
    because ``y`` is label-encoded differently than ``model.classes_`` --
    the same translation ``_classification_report_chart``'s model path now
    runs, reused rather than re-derived.

    One cell (Utf8 ``y``), not the full ``_CONFUSION_DTYPE_MATRIX`` --
    named accordingly rather than under a "dtype_matrix" name that implied
    the full parametrization above.
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y_str = df["y"].cast(pl.Utf8)

    baseline_viz = ferrum.ClassificationReportVisualizer(model).fit(X, df["y"])
    viz = ferrum.ClassificationReportVisualizer(model).fit(X, y_str)
    assert viz._metrics["f1_macro"] == pytest.approx(baseline_viz._metrics["f1_macro"], abs=1e-9)


def test_classification_report_chart_model_path_null_y_refuses():
    """#206 cross-task note: the model path now routes a null-bearing ``y``
    through the shared target-hygiene guard (T1, #203) instead of leaking
    scikit-learn's own null error -- the model path was the one classification
    surface T1 disclosed as outside the contract; this closes it.

    ``diagnostic`` differs by consumer: the chart function names itself
    (it IS ``classification_report_chart``), while the visualizer names
    itself (``ClassificationReportVisualizer``) rather than the figure
    function it delegates to, since a caller who builds the visualizer
    directly never calls ``classification_report_chart``.
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y_null = df["y"].cast(pl.Utf8).scatter([0, 1], None)

    with pytest.raises(ValueError, match=r"classification_report_chart: y has 2 null row"):
        ferrum.classification_report_chart(model, X, y_null)

    with pytest.raises(ValueError, match=r"ClassificationReportVisualizer: y has 2 null row"):
        ferrum.ClassificationReportVisualizer(model).fit(X, y_null)


def test_classification_report_visualizer_fit_without_y_refuses():
    """#206 cycle-2 fix: ``ClassificationReportVisualizer(model).fit(X)`` --
    ``y`` defaulting to ``None`` per ``FerrumVisualizer.fit``'s own signature
    -- must refuse with a typed domain message, mirroring the chart
    consumer's own ``source.y is None`` guard. Pre-fix, ``np.asarray(None)``
    reached ``require_no_null_target``'s ``_null_mask``, whose ``.tolist()``
    is ``None``, and iterating it leaked ``TypeError: 'NoneType' object is
    not iterable`` from ferrum's own internals.
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])

    with pytest.raises(
        ValueError, match=r"ClassificationReportVisualizer\.fit\(\) requires the target"
    ):
        ferrum.ClassificationReportVisualizer(model).fit(X)


def test_classification_report_chart_model_path_total_mismatch_refuses():
    """#206 cycle 5: a model's ``classes_`` translated into ``y``'s domain
    but matching none of ``y``'s observed values used to render an 18-row,
    every-metric-0.0 report in silence, while ``confusion_matrix_chart`` on
    the identical source warned and dropped the rows. This is the
    total-mismatch shape #203's standing policy already refuses elsewhere
    (``check_resolved_target``'s zero-match case, ``binary_indicator``'s
    positive-class case), so it now refuses here too instead of trading
    #206's own pre-fix loud sklearn ``Mix of label input types`` error for
    a silent, wrong-looking-plausible chart.

    ``model.classes_`` is ``[0, 1, 2]`` (confirmed via the fixture);
    ``y`` is relabelled to ``"c0"``/``"c1"``/``"c2"`` -- disjoint from the
    translated class vocabulary (``"0"``/``"1"``/``"2"``) by construction,
    not merely differently spelled. This is the disjoint-*vocabulary*
    shape, the one #203's matrix adjudicated; the sibling test below pins
    that disjoint *predictions* alone are not this shape.
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y_disjoint = pl.Series("y", [f"c{v}" for v in df["y"].to_list()])

    with pytest.raises(
        ValueError,
        match=(
            r"classification_report_chart: the model's classes_ \[0, 1, 2\], "
            r"translated into y's domain, match none of y's observed values"
        ),
    ):
        ferrum.classification_report_chart(model, X, y_disjoint)

    with pytest.raises(
        ValueError,
        match=(
            r"ClassificationReportVisualizer: the model's classes_ \[0, 1, 2\], "
            r"translated into y's domain, match none of y's observed values"
        ),
    ):
        ferrum.ClassificationReportVisualizer(model).fit(X, y_disjoint)


def test_classification_report_chart_model_path_partial_overlap_still_renders():
    """The total-mismatch refusal above must not fire on a partial stray:
    part of the translated class vocabulary lands in ``y``'s domain, part
    doesn't -- the ordinary out-of-vocabulary case scikit-learn's own report
    already tolerates. Relabels only two of three classes to a disjoint
    spelling, leaving the third (``2``) untouched so the translated
    ``classes_`` and ``y`` overlap on at least one class.

    Asserts the rendered class axis, not a row count: a row count passes on
    a report of the wrong shape (cycle-5 review, finding 18).
    """
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    relabel = {0: "c0", 1: "c1", 2: "2"}
    y_partial = pl.Series("y", [relabel[v] for v in df["y"].to_list()])

    df_report = ferrum.classification_report_chart(model, X, y_partial)._data
    assert sorted(set(df_report["class"].to_list())) == ["0", "1", "2", "c0", "c1"]
    assert df_report.height == 15


def test_classification_report_chart_model_path_predictions_disjoint_from_y_renders():
    """The guard's subject is the class *vocabulary*, not the realized
    predictions -- and this is the shape that separates the two predicates
    (cycle-6 fix, #206 cycle-5 review finding 17).

    Predictions are a *sample* of the vocabulary, so ``predictions & y`` can
    be empty while ``translated classes_ & y`` is not: here ``classes_`` is
    ``[0, 1]``, ``y`` is integer (identity translation, no dtype mismatch
    anywhere), and the slice is the 7 rows where ``y`` holds only ``0`` and
    the model predicts only ``1`` -- "how does the model do on the rows it
    gets entirely wrong?", a legitimate report request. The vocabulary
    ``[0, 1]`` overlaps ``y``'s ``[0]``, so nothing is mistranslated and the
    report must render.

    Pre-fix the prediction-based predicate refused it, with a message whose
    claim ("the model's classes_ ... match none of y's observed values") is
    measurably false on this input. Released baseline (measured 2026-09-13,
    ``classification_report(y_sub, model.predict(X_sub), output_dict=True,
    zero_division=0)``): classes ``'0'``/``'1'``, every precision/recall/F1
    ``0.0``, support ``7.0``/``0.0`` -- so the values below are sklearn's
    own, not a shape this pin invented.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]
    predictions = np.asarray(model.predict(X.to_numpy()))

    rows = np.flatnonzero((y.to_numpy() == 0) & (predictions == 1)).tolist()
    X_sub, y_sub = X[rows], y[rows]
    # The predicate-discriminating premise, asserted rather than assumed:
    # y and the predictions share no value, y and classes_ do.
    assert set(y_sub.to_list()) == {0}
    assert set(predictions[rows].tolist()) == {1}
    assert not set(y_sub.to_list()).isdisjoint(set(model.classes_.tolist()))

    df_report = ferrum.classification_report_chart(model, X_sub, y_sub)._data
    assert sorted(set(df_report["class"].to_list())) == ["0", "1"]
    assert df_report.height == 6
    assert df_report["value"].to_list() == [0.0] * 6

    viz = ferrum.ClassificationReportVisualizer(model).fit(X_sub, y_sub)
    assert viz._metrics["f1_macro"] == 0.0


# --- #224a: the report's class-axis partition (_target_hygiene.report_labels) ---
#
# One cell of the partition per test below, plus a totality property test.
# Every cell is a named contract, so a new input shape lands in exactly one
# of them rather than in whichever hole the previous repro left open: cell 1
# a null in either array, cell 2 a mix of ordering families, cell 3 an
# orderable homogeneous union (the answer), cell 4 an unorderable one.


def test_precomputed_classification_report_mixed_label_types_refuses():
    """Cell 2 of ``report_labels``' partition: values from two ordering
    families refuse as a *mix of label types*, the shape scikit-learn's own
    ``unique_labels`` refuses too.

    Released baseline (measured 2026-09-13 against ``HEAD``): this input
    **rendered** -- ``sorted(..., key=str)`` stringified first, so classes
    ``'0'``/``'1'`` came out with every metric 0.5, silently scoring string
    predictions against integer targets. #224a's first cycle replaced that
    with an unowned ``TypeError: '<' not supported between instances of
    'str' and 'int'`` (never released). The whole message is asserted, not
    only its first clause: the remedy sentence is the part that was wrong
    for a null-bearing ``y_pred`` in cycle 2, so it is pinned here and the
    null shapes are pinned separately below.
    """
    y_true = [0, 1, 0, 1]
    y_pred = np.array(["0", "1", "1", "0"], dtype=object)

    with pytest.raises(ValueError) as excinfo:
        ferrum.classification_report_chart(y_true=y_true, y_pred=y_pred)

    assert str(excinfo.value) == (
        "classification_report_chart: y_true and y_pred hold a mix of label "
        "types that cannot be ordered together, the same shape scikit-learn's "
        "own unique_labels refuses. y_true holds [0, 1] and y_pred holds "
        "['0', '1']. Use one consistent value type for y_true and y_pred "
        "before calling classification_report_chart."
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred", "observed"),
    [
        pytest.param(
            np.array(["a", "b", "a", "b"], dtype=object),
            np.array(["a", None, "a", "b"], dtype=object),
            "['a', 'b', None]",
            id="object-None",
        ),
        pytest.param(
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([0.0, np.nan, 0.0, 1.0]),
            "[0.0, 1.0, nan]",
            id="float-NaN",
        ),
        pytest.param(
            np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
            np.array(["2020-01-01", "NaT"], dtype="datetime64[ns]"),
            None,
            id="temporal-NaT",
        ),
    ],
)
def test_precomputed_classification_report_null_y_pred_refuses(y_true, y_pred, observed):
    """Cell 1 of the partition, ``y_pred`` half: a null *prediction* refuses
    in the null guard's own vocabulary, naming ``y_pred`` -- in every dtype
    family a null can arrive in, because the cell tests through
    ``_null_mask`` rather than through whichever spelling a repro used.

    Released baseline (measured 2026-09-13 against ``HEAD``): all three
    rendered a phantom class of their own -- ``y_pred=["a",None,"a","b"]``
    produced a ``'None'`` row at 0/0/0 with class ``'b'``'s recall at 0.5,
    and ``y_pred=[0.0,nan,0.0,1.0]`` a ``'nan'`` row the same way. Cycle 2
    of #224a diagnosed only the object-``None`` one, and diagnosed it
    *wrong* -- as a type mix, prescribing "use one consistent value type"
    for a caller whose defect was a null -- while the float-``NaN`` twin
    still rendered silently, so the answer still depended on the target's
    dtype.
    """
    with pytest.raises(ValueError) as excinfo:
        ferrum.classification_report_chart(y_true=y_true, y_pred=y_pred)

    message = str(excinfo.value)
    assert message.startswith(
        "classification_report_chart: y_pred has 1 null row(s) out of "
        f"{len(y_pred)}, and a null is not a predicted class"
    )
    assert message.endswith(
        "Drop or impute the null rows before calling classification_report_chart."
    )
    # The remedy names the null, not a type mix (cycle 2's misdiagnosis).
    assert "consistent value type" not in message
    if observed is not None:
        assert f"y_pred holds {observed}" in message


@pytest.mark.parametrize(
    ("value_a", "value_b", "family"),
    [
        pytest.param(1 + 1j, 2 + 0j, "complex", id="complex"),
        pytest.param({"k": 1}, {"k": 2}, "dict", id="dict"),
    ],
)
def test_precomputed_classification_report_unorderable_values_refuse(value_a, value_b, family):
    """Cell 4 of the partition, the *unorderable* half: values of *one*
    family that do not order at all refuse by name -- not as a "mix of
    label types", which would send the caller after a defect their input
    does not have.

    Released baseline (measured 2026-09-13 against ``HEAD``): the complex
    pair rendered classes ``'(1+1j)'``/``'(2+0j)'`` through
    ``sorted(..., key=str)``, and the dict pair leaked ``TypeError:
    unhashable type: 'dict'`` from the ``set()`` the union was built with.
    ``dict`` is unorderable (no ``<``) as well as unhashable, so it takes
    this branch rather than the sibling *unkeyable* one below (which needs
    a value that DOES order).
    """
    y_true = np.array([value_a, value_b], dtype=object)
    y_pred = np.array([value_a, value_a], dtype=object)

    with pytest.raises(ValueError) as excinfo:
        ferrum.classification_report_chart(y_true=y_true, y_pred=y_pred)

    message = str(excinfo.value)
    assert message.startswith(
        f"classification_report_chart: y_true and y_pred hold {family} values "
        f"that cannot be used as class labels. They do not order against "
        f"each other"
    )
    assert "mix of label types" not in message


@pytest.mark.parametrize(
    ("value_a", "value_b", "family"),
    [
        pytest.param([1], [2], "list", id="list"),
        pytest.param({1}, {2}, "set", id="set"),
        pytest.param(bytearray(b"a"), bytearray(b"b"), "bytes", id="bytearray"),
    ],
)
def test_precomputed_classification_report_unkeyable_values_refuse(value_a, value_b, family):
    """Cell 4 of the partition, the *unkeyable* half: values that DO order
    (``list``, ``set`` and ``bytearray`` all support ``<``) but cannot be
    hashed refuse by name too -- the confusion matrix this report is built
    from keys each class by its label value, so an orderable-but-unhashable
    label is exactly as unusable as an unorderable one.

    Released baseline / cycle-3 measurement: ``report_labels`` itself never
    raised on these (its sort over a concatenated list, not a ``set``,
    succeeds), so each one answered here and then leaked
    ``TypeError: unhashable type: 'list'``/``'set'``/``'bytearray'`` one
    statement later, out of ``_PrecomputedSource.confusion_matrix``'s own
    ``set()``-based label derivation.
    """
    y_true = _object_array([value_a, value_b])
    y_pred = _object_array([value_a, value_a])

    with pytest.raises(ValueError) as excinfo:
        ferrum.classification_report_chart(y_true=y_true, y_pred=y_pred)

    message = str(excinfo.value)
    assert message.startswith(
        f"classification_report_chart: y_true and y_pred hold {family} values "
        f"that cannot be used as class labels. They order, but cannot be "
        f"hashed"
    )
    assert "mix of label types" not in message


def test_precomputed_classification_report_frozenset_labels_dedupe_by_equality():
    """Cell 3's answer, on the one shape that is hashable but only
    partially ordered (``frozenset``, ordered by subset): the former
    adjacency-only dedupe assumed equal values sort next to each other,
    which a partial order does not guarantee.

    Released baseline / cycle-3 measurement: ``report_labels`` on this
    exact pair (``true=[{1}, {2}]``, ``pred=[{2}, {1}]``, all frozensets)
    returned ``[frozenset({1}), frozenset({2}), frozenset({1})]`` -- three
    labels for two classes. Deduplicating by hash equality
    (``dict.fromkeys``) instead of adjacency closes it: exactly two
    distinct classes render, regardless of dtype.
    """
    a, b = frozenset({1}), frozenset({2})
    y_true = _object_array([a, b, a, b])
    y_pred = _object_array([b, a, a, b])

    frame = ferrum.classification_report_chart(y_true=y_true, y_pred=y_pred)._data
    classes = list(dict.fromkeys(frame["class"].to_list()))
    assert len(classes) == 2
    assert set(classes) == {str(a), str(b)}


def test_precomputed_classification_report_order_moves_below_ten_classes():
    """Cell 3, and the true blast radius of #224a's order change: the two
    rules disagree whenever the class labels' decimal strings sort
    differently from their values, which needs only unequal digit counts --
    **two** classes are enough, not ten.

    Measured 2026-09-13: ``{2, 10}`` rendered ``['10', '2']`` on the
    released code and renders ``['2', '10']`` now. The disclosure that
    "fewer than 10 classes are unaffected" was false, and this pins the
    corrected claim.

    ``confusion_matrix_chart`` moved with it on 2026-09-17 (cycle-2 design
    review finding A). It rendered ``['10', '2']`` through two cycles while
    the report rendered ``['2', '10']`` -- the documented asymmetry, and
    real rather than a fixture artifact -- until the precomputed path's
    ``sorted(..., key=str)`` rule was retired for the same partition the
    report uses. Both figures now put one class axis in one order, which is
    what the family-totality ruling asked for; this asserts the pair, so a
    future change that moves one without the other fails here.
    """
    y_true = np.array([2, 10, 2, 10])
    y_pred = np.array([2, 10, 10, 2])

    report = ferrum.classification_report_chart(y_true=y_true, y_pred=y_pred)._data
    assert list(dict.fromkeys(report["class"].to_list())) == ["2", "10"]

    confusion = ferrum.confusion_matrix_chart(y_true=y_true, y_pred=y_pred)._data
    assert list(dict.fromkeys(confusion["actual"].to_list())) == ["2", "10"]


def test_report_labels_numeric_unions_are_not_a_type_mix():
    """Cell 2's boundary: ``int``/``float``/``bool`` are one ordering
    family, so a union of them is *not* a type mix. A partition keyed on
    the exact Python type would refuse all three of these, which the
    released code accepted -- the mixed cell has to mean "cannot be
    ordered together", not "not spelled the same way".
    """
    from ferrum.diagnostics._internal._target_hygiene import report_labels

    def labels(true_values, pred_values):
        return report_labels(
            np.array(true_values, dtype=object),
            np.array(pred_values, dtype=object),
            diagnostic="probe",
        )

    assert labels([0, 1], [0.0, 1.0]) == [0, 1]
    assert labels([True, False], [1, 0]) == [False, True]
    assert labels([Decimal("1"), Decimal("2")], [1, 2]) == [Decimal("1"), Decimal("2")]


class _Orderable:
    """A custom label value that defines ``<`` (and so has a class axis)."""

    def __init__(self, rank: int) -> None:
        self.rank = rank

    def __lt__(self, other: object) -> bool:
        return self.rank < other.rank  # type: ignore[attr-defined]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Orderable) and self.rank == other.rank

    def __hash__(self) -> int:
        return hash(self.rank)


class _Unorderable:
    """A custom label value with no ordering at all."""


_REPORT_LABEL_SHAPES = {
    "empty": ([], []),
    "one class": ([1], [1]),
    "ints": ([0, 1, 2], [1, 1, 2]),
    "floats": ([0.5, 1.5], [1.5, 0.5]),
    "bools": ([True, False], [False, True]),
    "decimals": ([Decimal("1"), Decimal("10")], [Decimal("10"), Decimal("1")]),
    "strings": (["a", "b"], ["b", "a"]),
    "bytes": ([b"a", b"b"], [b"b", b"a"]),
    "dates": ([date(2020, 1, 1), date(2020, 1, 2)], [date(2020, 1, 2), date(2020, 1, 1)]),
    "datetimes": ([datetime(2020, 1, 1), datetime(2020, 1, 2)], [datetime(2020, 1, 1)] * 2),
    "times": ([time(1), time(2)], [time(2), time(1)]),
    "timedeltas": ([timedelta(1), timedelta(2)], [timedelta(2), timedelta(1)]),
    "tuples": ([(1, 2), (1, 3)], [(1, 3), (1, 2)]),
    "frozensets": ([frozenset({1}), frozenset({2})], [frozenset({2}), frozenset({1})]),
    "sets (unhashable, orderable)": ([{1}, {2}], [{2}, {1}]),
    "lists (unhashable, orderable)": ([[1], [2]], [[2], [1]]),
    "custom orderable": ([_Orderable(1), _Orderable(2)], [_Orderable(2), _Orderable(1)]),
    "custom unorderable": ([_Unorderable(), _Unorderable()], [_Unorderable()]),
    "dicts": ([{"k": 1}, {"k": 2}], [{"k": 1}]),
    "complex": ([1 + 1j, 2 + 0j], [2 + 0j, 1 + 1j]),
    "ndarray values": ([np.array([1, 2]), np.array([3, 4])], [np.array([1, 2])]),
    "int vs str": ([0, 1], ["0", "1"]),
    "str vs date": (["a"], [date(2020, 1, 1)]),
    "bytes vs str": ([b"a"], ["a"]),
    "date vs datetime": ([date(2020, 1, 1)], [datetime(2020, 1, 1)]),
    "None in y_true": ([None, "a"], ["a", "a"]),
    "None in y_pred": (["a", "b"], ["a", None]),
    "NaN in y_pred": ([0.0, 1.0], [0.0, float("nan")]),
    "null and mixed at once": ([0, 1], ["0", None]),
    "y_true empty, y_pred not": ([], ["a"]),
    "y_pred empty, y_true not": (["a"], []),
}


def _object_array(values: list) -> np.ndarray:
    """An object array holding ``values`` element for element.

    Assigned one slot at a time rather than with ``np.array(values,
    dtype=object)``, which would broadcast a list-of-lists into a 2-D array
    instead of holding the lists as label values.
    """
    arr = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        arr[i] = value
    return arr


_NAMED_REFUSALS = (
    # cell 1, either array
    r"^probe: (y|y_pred) has \d+ null row\(s\) out of \d+, and a null is not ",
    # cell 2
    r"^probe: y_true and y_pred hold a mix of label types that cannot be ordered ",
    # cell 4 -- one stem, two reasons (does not order / orders but unhashable)
    r"^probe: y_true and y_pred hold \S+ values that cannot be used as class labels\. ",
)


@pytest.mark.parametrize("shape", list(_REPORT_LABEL_SHAPES), ids=list(_REPORT_LABEL_SHAPES))
def test_report_labels_answers_or_refuses_by_name(shape):
    """Totality, as a property rather than as a list of repros: for *any*
    input, ``report_labels`` either returns a sorted label list or raises
    one of its three named refusals. Nothing may leak an unowned
    ``TypeError``/``ValueError`` out of a sort, a hash or a comparison --
    the failure class that recurred across three fix cycles, each time
    through a shape the previous repro had not thought of.

    The table spans every ordering family the partition knows, values that
    are orderable but unhashable, values that are hashable but unorderable,
    each null spelling in each array, a null and a type mix at once, and
    the empty cases. It is a property over the *values* a 1-D pair can
    hold; array shape is a caller precondition enforced upstream
    (``_require_1d_array``, and the source constructor for a scalar), not
    part of this contract.

    The answer branch (a returned list, not a refusal) is checked against
    the contract cell 3 actually promises: pairwise distinct by equality,
    and non-decreasing for every family that totally orders. ``frozenset``
    is the one family here that only *partially* orders (by subset), so
    ``sorted()`` does not itself produce an ascending sequence for it --
    only distinctness is asserted there. This is the branch a duplicated,
    arbitrarily-ordered answer would previously have passed silently
    (measured: the ``frozensets`` row returned a duplicate label before
    the dedup fix, and ``isinstance``/``len`` alone could not see it).
    """
    from ferrum.diagnostics._internal._target_hygiene import _value_family, report_labels

    true_values, pred_values = _REPORT_LABEL_SHAPES[shape]
    try:
        labels = report_labels(
            _object_array(true_values), _object_array(pred_values), diagnostic="probe"
        )
    except Exception as exc:  # noqa: BLE001 - totality is the property under test
        assert isinstance(exc, ValueError) and any(
            re.match(pattern, str(exc)) for pattern in _NAMED_REFUSALS
        ), f"{shape}: leaked an unnamed {type(exc).__name__}: {exc}"
        return
    assert isinstance(labels, list)
    assert len(labels) <= len(true_values) + len(pred_values)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            assert a != b, f"{shape}: duplicate label {a!r} survived dedup: {labels!r}"
    if labels and _value_family(labels[0]) != "frozenset":
        for i in range(len(labels) - 1):
            assert not (labels[i + 1] < labels[i]), f"{shape}: labels are not ascending: {labels!r}"


def test_classification_report_cells_render_two_decimal_places():
    """``value_fmt`` is the entire visible content of a classification
    report chart, and nothing asserted its text -- widening the format to
    four decimals survived the suite. Hand-computed on an asymmetric
    fixture so each cell is a distinguishable number.

    ``y_true = [0,0,0,0,1,1]``, ``y_pred = [0,0,0,1,1,1]``. Class 0: three
    of three predicted-0 are right (precision 1.00) out of four actual-0
    (recall 0.75), f1 = 2(1)(0.75)/1.75 = 0.857. Class 1: two of three
    predicted-1 are right (precision 0.667), both actual-1 found (recall
    1.00), f1 = 0.80.
    """
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])

    frame = ferrum.classification_report_chart(y_true=y_true, y_pred=y_pred)._data
    fmt = {
        (row["class"], row["metric"]): row["value_fmt"]
        for row in frame.select(["class", "metric", "value_fmt"]).to_dicts()
    }
    assert fmt[("0", "precision")] == "1.00"
    assert fmt[("0", "recall")] == "0.75"
    assert fmt[("0", "f1-score")] == "0.86"
    assert fmt[("1", "precision")] == "0.67"
    assert fmt[("1", "f1-score")] == "0.80"

    texts = re.findall(
        r"<text[^>]*>([^<]*)</text>",
        ferrum.classification_report_chart(y_true=y_true, y_pred=y_pred).to_svg(),
    )
    for cell in ("1.00", "0.75", "0.86", "0.67", "0.80"):
        assert cell in texts, f"cell {cell!r} is not rendered; got {texts}"
    for widened in ("1.0000", "0.7500", "0.8571", "0.6667", "0.8000"):
        assert widened not in texts, f"cells rendered four decimals: {widened!r} in {texts}"


def test_classification_report_chart_refuses_a_y_less_source_before_an_x_less_one():
    """Both guards raise a typed domain error, but the order is a contract:
    ``y`` is checked first, mirroring ``ModelSource.confusion_matrix``'s
    own y-then-X order, so a source missing both is told about ``y``. Only
    observable when both are absent, which nothing constructed.
    """
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")

    both_missing = ferrum.ModelSource(model)
    with pytest.raises(ValueError, match=r"requires y to be provided"):
        ferrum.classification_report_chart(both_missing)

    # X alone missing still names X, so the y-first order is not just
    # "y's message always wins".
    y_only = ferrum.ModelSource(model, None, df["y"])
    with pytest.raises(ValueError, match=r"requires the training data"):
        ferrum.classification_report_chart(y_only)


def test_calibration_error_is_the_mean_squared_deviation_not_the_mean():
    """``calibration_error``'s *magnitude*, against arithmetic done here.

    The batch verified which model each number belonged to and never what
    the number was: one test asserted only ``>= 0.0``, one only that the
    keys existed, and one compared the per-model value against the *solo*
    visualizer's value -- i.e. against itself, which any consistent formula
    change satisfies. So swapping the documented mean *squared* deviation
    for a plain mean survived the whole pin set.

    The strict model is the discriminating fixture: its bins sit mostly
    above the diagonal, so the signed mean is **negative** while the mean
    square is positive. A formula swap cannot hide behind a sign here.
    """
    from sklearn.linear_model import LogisticRegression

    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]
    strict = LogisticRegression(C=0.01, max_iter=1000).fit(X.to_numpy(), y.to_numpy())

    # Recomputed from the frame the visualizer charted, by a different
    # expression than the one under test.
    bins = ferrum.ModelSource(strict, X, y).calibration_curve(n_bins=5)
    deviation = (bins["fraction_positive"] - bins["mean_predicted"]).to_numpy()
    expected = float(np.mean(deviation**2))
    signed_mean = float(np.mean(deviation))

    # The fixture really discriminates: the two candidate formulas do not
    # even agree on sign.
    assert signed_mean < 0.0 < expected

    viz = ferrum.CalibrationVisualizer(strict, n_bins=5).fit(X, y)
    assert viz._metrics["calibration_error"] == pytest.approx(expected)
    # And the absolute magnitude, pinned outright.
    assert viz._metrics["calibration_error"] == pytest.approx(0.06288996582434439)

    # A better-calibrated model must score lower -- the metric has to be
    # oriented, not merely reproducible.
    loose = load_fixture("binary_logistic")
    loose_viz = ferrum.CalibrationVisualizer(loose, n_bins=5).fit(X, y)
    assert loose_viz._metrics["calibration_error"] == pytest.approx(0.00498048724996414)
    assert loose_viz._metrics["calibration_error"] < viz._metrics["calibration_error"]


# --- the dtype fast path answers exactly what the per-value scan answers ----


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.array([True, False]), id="bool"),
        pytest.param(np.array([1, 2], dtype=np.int64), id="int64"),
        pytest.param(np.array([1, 2], dtype=np.uint8), id="uint8"),
        pytest.param(np.array([1.0, 2.0], dtype=np.float64), id="float64"),
        pytest.param(np.array([1.0, 2.0], dtype=np.float16), id="float16"),
        pytest.param(np.array([1.0, 2.0], dtype=np.longdouble), id="longdouble"),
        pytest.param(np.array(["a", "b"]), id="unicode"),
        pytest.param(np.array([b"a", b"b"]), id="bytes"),
        pytest.param(np.array([1 + 1j, 2 + 0j]), id="complex"),
        pytest.param(np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"), id="date64"),
        pytest.param(
            np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"), id="datetime64"
        ),
        pytest.param(np.array([1, 2], dtype="timedelta64[Y]"), id="timedelta64-year"),
        pytest.param(np.array([1, 2], dtype="timedelta64[ns]"), id="timedelta64-ns"),
        pytest.param(np.array(["a", 1], dtype=object), id="object-mixed"),
        pytest.param(np.array([], dtype="U1"), id="empty-unicode"),
        pytest.param(np.array([], dtype=np.int64), id="empty-int"),
    ],
)
def test_the_dtype_family_fast_path_agrees_with_the_per_value_scan(values):
    """The optimization's own pin (cycle-2 design review, missing-test 3).

    ``_families_of`` reads the family set off the numpy dtype when the
    dtype determines one, instead of running ``_value_family`` over every
    row -- 318ms of ``report_labels``' 435ms at 1e6 rows. The map's
    exclusions are load-bearing, not conservatism: ``datetime64[D]``'s
    ``.tolist()`` yields ``datetime.date`` (family ``"date"``, not
    ``"datetime"``), ``timedelta64[Y]``'s yields ``int`` (family
    ``"number"``, not ``"timedelta"``), and ``complex`` must reach
    ``_value_family``'s own exclusion. A contributor who "completed" the
    map with ``"M"``, ``"m"`` or ``"c"`` would break all three, and only a
    partition test at exactly the right dtype would have noticed.
    """
    from ferrum.diagnostics._internal._target_hygiene import _families_of, _value_family

    listed = values.tolist()
    assert _families_of(values, listed) == {_value_family(v) for v in listed}
