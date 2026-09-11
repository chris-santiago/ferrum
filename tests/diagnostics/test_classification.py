"""Phase 10b tests: classification curve marks + figure functions + visualizers."""

from __future__ import annotations

import re

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
    # summary) rather than several per-class curves.
    # After legend-label renaming, the single class value carries the AP score.
    classes = set(chart._data["class"].unique().to_list())
    assert len(classes) == 1
    assert next(iter(classes)).startswith("macro (AP =")


def test_pr_chart_per_class_false_micro(multi_source):
    chart = ferrum.pr_chart(multi_source, per_class=False, average="micro")
    classes = set(chart._data["class"].unique().to_list())
    assert len(classes) == 1
    assert next(iter(classes)).startswith("micro (AP =")


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

    Deliberately order-insensitive: both frames are sorted by
    ``(class, metric)`` before comparing, because the two paths render
    their class axis in different (but each individually documented and
    separately pinned -- see
    ``test_classification_report_chart_class_axis_order_documented_divergence``)
    orders for >= 10 numeric classes. This test's job is value parity,
    not axis order.
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


def test_classification_report_chart_class_axis_order_documented_divergence():
    """The model path orders classes as sklearn's ``classification_report``
    returns them (numeric-sorted, via ``unique_labels``); the precomputed
    path orders them ``sorted(..., key=str)`` (the ``confusion_matrix``
    precedent -- see ``_PrecomputedSource.classification_report``'s
    docstring). For >= 10 numeric classes these disagree: string order
    interleaves ``"10"``/``"11"`` before ``"2"``. The parity test above
    sorts both frames before comparing, so it cannot see a regression in
    either path's own render order -- this test pins each path's ORDER
    independently instead of the row multiset.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

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

    assert model_order == [str(c) for c in sorted(set(y.tolist()))]
    assert precomputed_order == sorted({str(c) for c in y.tolist()}, key=str)
    assert model_order != precomputed_order, (
        "expected the documented >= 10-class order divergence between the "
        "model path (numeric) and precomputed path (string-sorted)"
    )


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


@pytest.mark.parametrize(
    "y_transform, class_names",
    [
        pytest.param(lambda y: y, None, id="numeric_y_no_names"),
        pytest.param(lambda y: y.cast(pl.Utf8), None, id="utf8_y_no_names"),
        pytest.param(
            lambda y: y.cast(pl.Utf8).cast(pl.Categorical), None, id="categorical_y_no_names"
        ),
        pytest.param(lambda y: y.cast(pl.Utf8), ["A", "B", "C"], id="utf8_y_with_class_names"),
        pytest.param(
            lambda y: y.cast(pl.Utf8).cast(pl.Enum(["2", "0", "1"])), None, id="enum_y_no_names"
        ),
    ],
)
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
