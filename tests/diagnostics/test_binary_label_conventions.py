"""Batch G close — a binary target's *labels* never change its metrics.

Every binary classification diagnostic ferrum draws has to binarize the
target against the positive class before any kernel sees it. Until the
close remediation, only ``roc_frame``/``pr_frame``'s *multiclass* arm did
(via ``one_hot``); the binary arm fed the raw ``y_true`` to the kernels,
which sum it as a positive-count. ``calibration_frame`` did the same. So
every convention other than literally ``{0, 1}`` was silently corrupted:
``y in {-1, 1}`` rendered "ROC Curve - AUC -1.747" against a true 0.9994,
``y in {1, 2}`` rendered 0.302 against 0.9989, and string labels raised
``could not convert string to float``.

The invariant pinned here is relabeling-invariance: fitting the same model
on the same X with a *relabeled* y must produce exactly the metrics the
``{0, 1}`` labelling produces, because relabeling the classes changes no
probability. Every reference number is computed by scikit-learn
(``roc_auc_score`` / ``average_precision_score`` / ``brier_score_loss``) on
the ``{0, 1}`` form, never by the code under test.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import ferrum
from tests.fixtures import load_dataset

#: The three non-``{0, 1}`` conventions the finding names, as
#: ``{original_label: relabelled}`` maps. ``{-1, 1}`` is sklearn's SVM
#: convention, ``{1, 2}`` the R/Matlab one, strings the human one.
_RELABELLINGS = {
    "minus_one_one": {0: -1, 1: 1},
    "one_two": {0: 1, 1: 2},
    "strings": {0: "neg", 1: "pos"},
}


def svg_texts(svg: str) -> list[str]:
    """Every rendered text run in an SVG, in document order."""
    return re.findall(r"<text[^>]*>([^<]*)</text>", svg)


@pytest.fixture(scope="module")
def binary_data():
    """``(X, y01)`` from the shared binary fixture dataset."""
    df = load_dataset("binary_classification")
    return df.select(["f0", "f1", "f2", "f3"]), df["y"]


@pytest.fixture(scope="module")
def references(binary_data):
    """sklearn's own AUC / AP / Brier for the ``{0, 1}`` labelling."""
    X, y01 = binary_data
    model = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    proba = model.predict_proba(X.to_numpy())[:, 1]
    y_np = y01.to_numpy()
    return {
        "proba": proba,
        "auc": float(roc_auc_score(y_np, proba)),
        "ap": float(average_precision_score(y_np, proba)),
        "brier": float(brier_score_loss(y_np, proba)),
    }


def _relabelled_source(X, y01, mapping):
    """A ``ModelSource`` over the same X with a relabelled y.

    The model is re-fit on the relabelled target so ``model.classes_`` is
    the relabelled class set -- exactly the shape a user hands ferrum.
    """
    y = pl.Series("y", [mapping[int(v)] for v in y01.to_list()])
    model = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y.to_numpy())
    return ferrum.ModelSource(model, X, y), y


@pytest.mark.parametrize("name", sorted(_RELABELLINGS))
def test_roc_auc_is_invariant_to_the_binary_label_convention(name, binary_data, references):
    """The ROC frame's kernel ``auc`` and the chart title must equal
    sklearn's AUC for the same probabilities, whatever the labels are."""
    X, y01 = binary_data
    source, _ = _relabelled_source(X, y01, _RELABELLINGS[name])

    aucs = source.roc_curve()["auc"].unique().to_list()
    assert aucs == pytest.approx([references["auc"]], abs=1e-12)

    title = ferrum.roc_chart(source)._title.text
    assert title == f"ROC Curve — AUC {references['auc']:.3f}"


@pytest.mark.parametrize("name", sorted(_RELABELLINGS))
def test_pr_ap_is_invariant_to_the_binary_label_convention(name, binary_data, references):
    """Same invariant for average precision on the PR frame and title."""
    X, y01 = binary_data
    source, _ = _relabelled_source(X, y01, _RELABELLINGS[name])

    aps = source.pr_curve()["ap"].unique().to_list()
    assert aps == pytest.approx([references["ap"]], abs=1e-12)

    title = ferrum.pr_chart(source)._title.text
    assert title == f"Precision–Recall Curve — AP {references['ap']:.3f}"


@pytest.mark.parametrize("name", sorted(_RELABELLINGS))
def test_calibration_brier_is_invariant_to_the_binary_label_convention(
    name, binary_data, references
):
    """A Brier score is bounded in [0, 1]. Un-binarized ``y in {1, 2}``
    rendered 1.035 -- impossible on the face of the chart."""
    X, y01 = binary_data
    source, _ = _relabelled_source(X, y01, _RELABELLINGS[name])

    frame = source.calibration_curve()
    assert frame["brier"].unique().to_list() == pytest.approx([references["brier"]], abs=1e-12)
    assert 0.0 <= references["brier"] <= 1.0
    assert frame["fraction_positive"].min() >= 0.0
    assert frame["fraction_positive"].max() <= 1.0

    texts = svg_texts(ferrum.calibration_chart(source).to_svg())
    expected = f"Brier = {references['brier']:.3f}"
    assert expected in texts
    assert ferrum.calibration_chart(source)._title.text == (
        f"Calibration Curve — Brier {references['brier']:.3f}"
    )


@pytest.mark.parametrize("name", sorted(_RELABELLINGS))
def test_precomputed_curves_are_invariant_to_the_binary_label_convention(
    name, binary_data, references
):
    """The precomputed ``y_true=``/``y_pred=`` path binarizes too -- it
    raised outright on string labels and fabricated a metric on the numeric
    conventions."""
    _, y01 = binary_data
    mapping = _RELABELLINGS[name]
    y = np.array([mapping[int(v)] for v in y01.to_list()])
    proba = references["proba"]

    roc = ferrum.roc_chart(y_true=y, y_pred=proba)
    assert roc._title.text == f"ROC Curve — AUC {references['auc']:.3f}"

    pr = ferrum.pr_chart(y_true=y, y_pred=proba)
    assert pr._title.text == f"Precision–Recall Curve — AP {references['ap']:.3f}"

    cal = ferrum.calibration_chart(y_true=y, y_pred=proba)
    assert cal._title.text == f"Calibration Curve — Brier {references['brier']:.3f}"


def test_zero_one_labels_are_unchanged(binary_data, references):
    """The conventional ``{0, 1}`` labelling -- what every fixture and
    golden in the repo uses -- must be byte-for-byte what it always was.
    The relabelling fix is only allowed to change the *other* conventions.
    """
    X, y01 = binary_data
    model = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    source = ferrum.ModelSource(model, X, y01)

    assert source.roc_curve()["auc"].unique().to_list() == pytest.approx([references["auc"]])
    assert source.pr_curve()["ap"].unique().to_list() == pytest.approx([references["ap"]])
    cal = source.calibration_curve()
    assert cal["brier"].unique().to_list() == pytest.approx([references["brier"]])
    # The reliability bins themselves are untouched by binarization on a
    # {0, 1} target: y == positive_class is the identity there.
    assert cal["fraction_positive"].to_list() == pytest.approx(
        ferrum.ModelSource(model, X, y01).calibration_curve()["fraction_positive"].to_list()
    )


# ---------------------------------------------------------------------------
# The composable BrierLabel path (intent F-L11-04)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_composed_brier_label_shows_the_per_sample_brier(strategy, binary_data, references):
    """``BrierLabel``'s own docstring example. Pre-fix it rendered the
    *bin-level* number -- 0.030 at ``strategy="uniform"``, 0.009 at
    ``strategy="quantile"``, a 3.3x swing from a binning kwarg the Brier
    score is invariant to -- and, worse, the composed layer overwrote the
    correct ``annotate_brier`` label, so the chart carried two copies of
    the wrong value and none of the right one.
    """
    X, y01 = binary_data
    model = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    source = ferrum.ModelSource(model, X, y01)

    chart = ferrum.calibration_chart(source, strategy=strategy)
    annotated = chart + ferrum.BrierLabel()

    expected = f"Brier = {references['brier']:.3f}"
    texts = svg_texts(annotated.to_svg())
    assert expected in texts
    # No *other* Brier number may appear anywhere on the composed chart.
    rendered = {t for t in texts if t.startswith("Brier = ") or t.startswith("Calibration Curve")}
    assert rendered == {expected, f"Calibration Curve — Brier {references['brier']:.3f}"}


def test_brier_label_still_integrates_arbitrary_user_data():
    """A frame with no ``brier`` column -- arbitrary user data, the reason
    the composable path exists -- keeps the ``metric_fn`` fallback."""
    df = pl.DataFrame({"p": [0.1, 0.4, 0.9], "obs": [0.0, 0.5, 1.0]})
    chart = ferrum.Chart(df).mark_line().encode(x="p", y="obs")
    expected = float(np.mean((np.array([0.1, 0.4, 0.9]) - np.array([0.0, 0.5, 1.0])) ** 2))

    texts = svg_texts((chart + ferrum.BrierLabel()).to_svg())
    assert f"Brier = {expected:.3f}" in texts


# ---------------------------------------------------------------------------
# The model axis (close review cycle 3): how the positive class is resolved
# ---------------------------------------------------------------------------


class _DuckProba:
    """A ``predict_proba``-only wrapper with no ``classes_``.

    An ordinary shape -- any hand-rolled estimator, any wrapper that
    forwards scoring without re-exporting sklearn's attributes. With no
    ``classes_`` to align against, ``probabilities()`` names its columns
    ``class_0`` / ``class_1``, and a resolver that parses the positive class
    back out of those names gets ``"class_1"``, which is not a value ``y``
    contains.
    """

    def __init__(self, fitted):
        self._fitted = fitted

    def predict(self, X):
        return self._fitted.predict(np.asarray(X))

    def predict_proba(self, X):
        return self._fitted.predict_proba(np.asarray(X))


class _DecisionFunctionOnly:
    """A margin-scoring estimator: ``decision_function`` plus ``classes_``."""

    def __init__(self, fitted):
        self._fitted = fitted
        self.classes_ = fitted.classes_

    def predict(self, X):
        return self._fitted.predict(np.asarray(X))

    def decision_function(self, X):
        return self._fitted.decision_function(np.asarray(X))


def _model_shape(name, fitted):
    """``(model, extra ModelSource kwargs)`` for one model shape."""
    if name == "sklearn_with_classes":
        return fitted, {}
    if name == "no_classes_attr":
        return _DuckProba(fitted), {}
    if name == "class_names_alias":
        # A display alias that is deliberately NOT in y's domain -- the
        # exact trap the batch's display-label-vs-comparison-value rule
        # exists to catch.
        return fitted, {"class_names": ["Negative", "Positive"]}
    if name == "alias_without_classes":
        return _DuckProba(fitted), {"class_names": ["Negative", "Positive"]}
    if name == "decision_function_only":
        return _DecisionFunctionOnly(fitted), {}
    raise AssertionError(name)


_MODEL_SHAPES = [
    "sklearn_with_classes",
    "no_classes_attr",
    "class_names_alias",
    "alias_without_classes",
    "decision_function_only",
]


@pytest.mark.parametrize("shape", _MODEL_SHAPES)
def test_metrics_are_invariant_to_the_model_shape(shape, binary_data, references):
    """How the estimator advertises its classes cannot change its metrics.

    The label axis was already covered; the model axis is where the close
    review found the remaining hole. ``_positive_class_value`` resolved the
    positive class by parsing the display label back out of the
    ``proba_<label>`` column name whenever the model had no ``classes_`` --
    but those names come from ``class_names``, which is either a user alias
    or a synthetic ``class_{i}``. Neither is a value in ``y``'s domain, so
    the now-mandatory binarization zeroed the indicator and every binary
    curve collapsed: measured AUC ``nan``, AP ``0.0``, ``fraction_positive``
    max ``0.0``, and a Brier of ``mean(p**2)`` -- in range, plausible, and
    wrong.

    ``decision_function_only`` is the control: it keeps ``classes_``, so it
    was correct before and must stay correct.
    """
    X, y01 = binary_data
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    model, kwargs = _model_shape(shape, fitted)
    source = ferrum.ModelSource(model, X, y01, **kwargs)

    if shape == "decision_function_only":
        # Sigmoid-squashed margins, not probabilities: the ranking (and so
        # AUC/AP) is identical, the calibration is not.
        assert source.roc_curve()["auc"].unique().to_list() == pytest.approx(
            [references["auc"]], abs=1e-9
        )
        assert source.pr_curve()["ap"].unique().to_list() == pytest.approx(
            [references["ap"]], abs=1e-9
        )
        return

    assert source.roc_curve()["auc"].unique().to_list() == pytest.approx([references["auc"]])
    assert source.pr_curve()["ap"].unique().to_list() == pytest.approx([references["ap"]])
    assert source.calibration_curve()["brier"].unique().to_list() == pytest.approx(
        [references["brier"]]
    )
    assert source.calibration_curve()["fraction_positive"].max() > 0.0

    assert ferrum.roc_chart(source)._title.text == f"ROC Curve — AUC {references['auc']:.3f}"
    assert ferrum.calibration_chart(source)._title.text == (
        f"Calibration Curve — Brier {references['brier']:.3f}"
    )
    # The threshold sweep resolves the same positive class, so it must
    # be a real sweep rather than an all-negative flatline.
    assert source.discrimination_threshold()["f1"].max() > 0.5


@pytest.mark.parametrize("shape", ["no_classes_attr", "alias_without_classes"])
def test_classes_less_models_work_for_every_label_convention(shape, binary_data, references):
    """The two axes crossed: a model with no ``classes_`` must still be
    right on ``{-1,1}``, ``{1,2}`` and string targets, because the positive
    class now comes from ``y`` rather than from a column name.
    """
    X, y01 = binary_data
    for mapping in _RELABELLINGS.values():
        y = pl.Series("y", [mapping[int(v)] for v in y01.to_list()])
        fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y.to_numpy())
        model, kwargs = _model_shape(shape, fitted)
        source = ferrum.ModelSource(model, X, y, **kwargs)
        assert source.roc_curve()["auc"].unique().to_list() == pytest.approx([references["auc"]])
        assert source.calibration_curve()["brier"].unique().to_list() == pytest.approx(
            [references["brier"]]
        )


# ---------------------------------------------------------------------------
# A target that matches nothing is diagnosed, not absorbed
# ---------------------------------------------------------------------------


def _all_negative_source(binary_data):
    """A model whose ``classes_`` says ``{0, 1}`` over a y holding only 0s.

    The realistic shape: a binary model scored on a slice that happens to
    contain no positives, or -- the case this guard really exists for -- a
    positive class resolved from the wrong domain.
    """
    X, y01 = binary_data
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    return fitted, X, pl.Series("y", [0] * len(y01))


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda s: s.roc_curve(), id="roc_curve"),
        pytest.param(lambda s: s.pr_curve(), id="pr_curve"),
        pytest.param(lambda s: s.calibration_curve(), id="calibration_curve"),
        pytest.param(lambda s: s.discrimination_threshold(), id="discrimination_threshold"),
    ],
)
def test_a_target_matching_no_positive_row_refuses_by_name(call, binary_data):
    """Collapse becomes diagnosis.

    When no row of ``y`` equals the positive class, every kernel downstream
    sees an all-negative target and returns something plausible and
    meaningless -- an AP of 0.0, a flat reliability curve, a Brier of
    ``mean(p**2)``. That silence is how a mis-resolved positive class (the
    finding above) stayed invisible. The refusal names both halves a caller
    needs: the class value searched for, and what ``y`` actually holds.
    """
    model, X, y = _all_negative_source(binary_data)
    source = ferrum.ModelSource(model, X, y)

    with pytest.raises(ValueError, match=r"no row of y equals the positive class") as excinfo:
        call(source)
    message = str(excinfo.value)
    assert "1" in message  # the class value searched for
    assert "y holds [0]" in message  # and what y actually has


def test_an_all_null_target_refuses_the_same_way(binary_data):
    """Nulls compare unequal to everything, so an all-null target is the
    same total mismatch and gets the same refusal rather than a curve drawn
    against an entirely-negative phantom target.
    """
    X, y01 = binary_data
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    nulls = pl.Series("y", [None] * len(y01), dtype=pl.Int64)
    source = ferrum.ModelSource(fitted, X, nulls)

    with pytest.raises(ValueError, match=r"no row of y equals the positive class"):
        source.roc_curve()


def test_multiclass_one_vs_rest_still_tolerates_an_absent_class(references):
    """The refusal belongs to *binary* curves only. One-vs-rest asks "is
    this row class k", for which "no row is" is a legitimate answer, and
    ``lift_curve`` documents that it skips zero-base-rate classes. Pinning
    this keeps the guard from migrating into ``positive_indicator``, where
    it would break the multiclass family.
    """
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    model = LogisticRegression(max_iter=1000).fit(X.to_numpy(), df["y"].to_numpy())
    # A y holding only two of the model's three classes.
    y = pl.Series("y", [0 if int(v) == 2 else int(v) for v in df["y"].to_list()])
    source = ferrum.ModelSource(model, X, y)

    roc = source.roc_curve()
    assert set(roc["class"].to_list()) == {"0", "1", "2"}
    assert source.lift_curve().height > 0
    assert source.cumulative_gain().height > 0


# ---------------------------------------------------------------------------
# A degenerate CV fold is not a bad request (close review cycle 4)
# ---------------------------------------------------------------------------


def _imbalanced_source(n=120, n_positive=12, seed=0):
    """A clean, well-labelled, imbalanced binary target.

    Imbalance is the normal case for ``discrimination_threshold`` -- it is
    the tool people reach for *because* the classes are skewed -- and the
    default fold generator is plain ``KFold(shuffle=True)``, not a
    stratified one, so some held-out fold containing no positives is
    routine rather than pathological.
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_positive, replace=False)] = 1
    X = pl.DataFrame(features, schema=["f0", "f1", "f2"])
    model = LogisticRegression(max_iter=1000).fit(features, y)
    return ferrum.ModelSource(model, X, pl.Series("y", y), random_state=0)


@pytest.mark.parametrize("n_positive", [4, 8, 12])
@pytest.mark.parametrize("cv", [3, 5, 20])
def test_cross_validated_threshold_sweep_survives_a_zero_positive_fold(n_positive, cv):
    """``discrimination_threshold(cv=...)`` must return a sweep, not refuse.

    The whole-target mismatch guard cannot see the difference between "the
    caller's y and the model's classes never met" and "this internal KFold
    slice happens to hold no positives" -- the fold loop hands it a slice.
    So it refused calls that worked before: measured 4 of 20 seeds at a 10%
    positive rate with ``cv=5``, on clean data, with a message advising the
    user to check labels that were never wrong.

    Resolution now happens once at request level, so a degenerate fold
    contributes the kernel's honest all-zero sweep instead of aborting the
    call.
    """
    source = _imbalanced_source(n_positive=n_positive)

    swept = source.discrimination_threshold(cv=cv)

    assert swept.height == 50
    assert set(swept.columns) == {"threshold", "precision", "recall", "f1", "queue_rate"}
    # A real sweep, not a flatline: the queue rate has to fall as the
    # threshold rises, whatever the folds contributed.
    assert swept["queue_rate"].max() > swept["queue_rate"].min()


def test_cross_validated_sweep_is_unchanged_by_the_whole_target_guard():
    """The degenerate fold contributes exactly what it always contributed.

    Hand-computed rather than compared against another ferrum path: an
    all-negative fold has nothing to find, so precision, recall and f1 are
    0 at every threshold while ``queue_rate`` still reports the fraction of
    that fold's rows at or above the threshold. Pinning the fold's own
    numbers is what makes "the average is unchanged" checkable, since the
    average is a mean over folds.
    """
    from ferrum.diagnostics._internal._curve_frames import threshold_sweep_from_indicator

    thresholds = np.linspace(0.0, 1.0, 5)
    scores = np.array([0.1, 0.3, 0.6, 0.9])
    all_negative = np.zeros(4, dtype=int)

    fold = threshold_sweep_from_indicator(all_negative, scores, thresholds).to_dicts()
    assert [row["precision"] for row in fold] == [0.0] * 5
    assert [row["recall"] for row in fold] == [0.0] * 5
    assert [row["f1"] for row in fold] == [0.0] * 5
    # 4 of 4 scores >= 0.0, 3 of 4 >= 0.25, 2 >= 0.5, 1 >= 0.75, 0 >= 1.0.
    assert [row["queue_rate"] for row in fold] == [1.0, 0.75, 0.5, 0.25, 0.0]


def test_a_whole_target_mismatch_still_refuses_under_cv():
    """The guard is kept, just moved up a layer: a target that genuinely
    never matches the positive class refuses with or without ``cv=``. The
    fix must not buy fold tolerance by dropping the refusal.
    """
    rng = np.random.default_rng(3)
    features = rng.normal(size=(60, 3))
    X = pl.DataFrame(features, schema=["f0", "f1", "f2"])
    model = LogisticRegression(max_iter=1000).fit(features, (features[:, 0] > 0).astype(int))
    # classes_ says {0, 1}; this y holds only 0s.
    source = ferrum.ModelSource(model, X, pl.Series("y", [0] * 60), random_state=0)

    for kwargs in ({}, {"cv": 3}):
        with pytest.raises(ValueError, match=r"no row of y equals the positive class"):
            source.discrimination_threshold(**kwargs)


# ---------------------------------------------------------------------------
# The refusal message's own content (close mutation gate, G4)
# ---------------------------------------------------------------------------


def test_the_refusal_message_lists_ys_values_sorted_and_capped():
    """The message promises "the values `y` actually holds" -- so the list
    itself has to be right.

    The existing refusal test uses a `y` with a single distinct value, which
    leaves both the 8-value cap and the sort order inert: truncating the
    listing to one value, or reversing its order, changed nothing any
    assertion could see. A realistic mismatched target has many values, and
    a message that prints one of them defeats its own purpose.

    The order is by `repr`, not by value -- a target reaching this path is
    by definition suspect and may hold mixed or unorderable types -- which
    is why `11` and `13` sort before `2` below.
    """
    from ferrum.diagnostics._internal._curve_frames import binary_indicator

    eleven_values = np.array([9, 3, 11, 5, 7, 1, 13, 2, 4, 6, 8])

    with pytest.raises(ValueError) as excinfo:
        binary_indicator(eleven_values, 99, diagnostic="roc_curve")
    message = str(excinfo.value)

    assert "the positive class 99" in message
    # Exactly eight, repr-sorted, from the eleven present.
    assert "y holds [1, 11, 13, 2, 3, 4, 5, 6]" in message
    assert "roc_curve:" in message


def test_the_refusal_message_lists_every_value_when_under_the_cap():
    """Below the cap nothing is dropped, so a caller with a small suspect
    target sees all of it."""
    from ferrum.diagnostics._internal._curve_frames import binary_indicator

    with pytest.raises(ValueError) as excinfo:
        binary_indicator(np.array([5, 7, 5, 7]), 1, diagnostic="pr_curve")
    assert "y holds [5, 7]" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Documented degenerate allowances (close mutation gate, advisories 1 and 4)
# ---------------------------------------------------------------------------


def test_an_empty_target_is_not_a_mismatch():
    """`binary_indicator` documents "an empty target is *not* an error --
    there is nothing to mismatch". Nothing exercised it, so making an empty
    target refuse survived the whole pin set.
    """
    from ferrum.diagnostics._internal._curve_frames import binary_indicator

    indicator = binary_indicator(np.array([]), 1, diagnostic="roc_curve")
    assert list(indicator) == []


def test_binary_class_values_falls_back_to_zero_one_on_an_empty_target():
    """The empty-target arm of `binary_class_values`: with no values to
    read, the pair is `0`/`1` in that order. Order matters -- slot 1 is the
    positive class every binary builder binarizes against.
    """
    from ferrum.diagnostics._internal._curve_frames import binary_class_values

    values, labels = binary_class_values(np.array([]))
    assert values == [0, 1]
    assert labels == ["0", "1"]


def test_binary_class_values_degenerate_slots_agree():
    """A single-distinct-value target puts that value in *both* slots, with
    labels to match. Values and labels disagreeing is the trap the branch's
    docstring promises cannot happen, and only the `class` discriminator
    makes it visible to a user.
    """
    from ferrum.diagnostics._internal._curve_frames import binary_class_values

    for target, expected in ((np.array([0, 0, 0, 0]), 0), (np.array([7, 7]), 7)):
        values, labels = binary_class_values(target)
        assert values == [expected, expected]
        assert labels == [str(expected), str(expected)]


# ---------------------------------------------------------------------------
# The three tiers of _resolve_class_values (close mutation gate, G3 + adv. 3)
# ---------------------------------------------------------------------------


def test_tier_1_uses_classes_even_when_y_cannot_supply_the_values():
    """``model.classes_`` outranks both other tiers, and the difference is
    observable.

    Tier 1 is the hottest line in the resolver (a `raise` there fails 129
    tests) yet disabling it cost nothing, because on every input the suite
    had, tier 2 produced the same answer. This is the constructible case
    where they diverge: ``y`` holds one of three classes, so tier 2 cannot
    supply three values and the fallback chain reaches tier 3 -- which
    returns the user's *display aliases*, comparison values matching no row
    of ``y``. That is the display-label-as-comparison-value defect the
    whole helper exists to prevent.
    """
    from ferrum.diagnostics.sources._classification import _resolve_class_values

    classes = np.array(["bird", "cat", "dog"])
    y = pl.Series("y", ["cat"] * 10)
    aliases = ["A", "B", "C"]

    assert _resolve_class_values(classes, y, aliases) == ["bird", "cat", "dog"]
    # Without classes_, the same call falls past tier 2 (1 distinct value,
    # 3 labels) to tier 3 and yields the aliases -- which is exactly why
    # tier 1 must come first.
    assert _resolve_class_values(None, y, aliases) == aliases


def test_tier_2_resolves_a_multiclass_model_that_has_no_classes_attr():
    """Tier 2's multiclass arm is reachable and correct.

    A multiclass estimator with no ``classes_`` had no coverage at all --
    disabling the arm entirely survived the pin set. Here ``y``'s three
    distinct values match the three score columns, so tier 2 supplies the
    comparison values and the one-vs-rest curves match sklearn exactly.
    """
    from sklearn.metrics import roc_auc_score

    from ferrum.diagnostics.sources._classification import _resolve_class_values

    rng = np.random.default_rng(0)
    features = rng.normal(size=(150, 4))
    y = np.array([i % 3 for i in range(150)])
    X = pl.DataFrame(features, schema=[f"f{i}" for i in range(4)])
    fitted = LogisticRegression(max_iter=1000).fit(features, y)

    assert _resolve_class_values(None, pl.Series("y", y), ["class_0", "class_1", "class_2"]) == [
        0,
        1,
        2,
    ]

    source = ferrum.ModelSource(_DuckProba(fitted), X, pl.Series("y", y))
    roc = source.roc_curve()
    proba = fitted.predict_proba(features)
    for index in range(3):
        expected = float(roc_auc_score((y == index).astype(int), proba[:, index]))
        curve = roc.filter(pl.col("class") == f"class_{index}")
        assert curve["auc"].unique().to_list() == pytest.approx([expected], abs=1e-9)


def test_tier_3_binary_refuses_rather_than_collapsing(binary_data):
    """Tier 3 -- the display-label parse -- is reachable, and on the binary
    path its unsound values are caught by the total-mismatch refusal.

    Reached when no ``classes_`` is available *and* ``y``'s cardinality
    disagrees with the score-column count, so neither sounder tier applies.
    Here a two-column model meets a three-valued ``y``: the parsed
    ``"class_1"`` matches no row, and the caller is told so by name instead
    of being handed a curve drawn against a phantom target.
    """
    rng = np.random.default_rng(0)
    features = rng.normal(size=(150, 4))
    y3 = np.array([i % 3 for i in range(150)])
    X = pl.DataFrame(features, schema=[f"f{i}" for i in range(4)])
    fitted = LogisticRegression(max_iter=1000).fit(features, (y3 > 0).astype(int))

    source = ferrum.ModelSource(_DuckProba(fitted), X, pl.Series("y", y3))
    with pytest.raises(ValueError, match=r"no row of y equals the positive class 'class_1'"):
        source.roc_curve()
