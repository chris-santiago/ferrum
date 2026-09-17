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

import datetime as dt
import re
import warnings

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
    """The ROC frame's kernel ``auc`` and the rendered AUC annotation must
    equal sklearn's AUC for the same probabilities, whatever the labels are.

    The chart *title* carried this pin until #219/D4 (2026-09-13) retired
    the title's metric suffix; the endpoint overlay is now the one metric
    surface, so the rendered assertion moved there."""
    X, y01 = binary_data
    source, _ = _relabelled_source(X, y01, _RELABELLINGS[name])

    aucs = source.roc_curve()["auc"].unique().to_list()
    assert aucs == pytest.approx([references["auc"]], abs=1e-12)

    chart = ferrum.roc_chart(source)
    assert chart._title.text == "ROC Curve"
    assert f"AUC = {references['auc']:.3f}" in svg_texts(chart.to_svg())


@pytest.mark.parametrize("name", sorted(_RELABELLINGS))
def test_pr_ap_is_invariant_to_the_binary_label_convention(name, binary_data, references):
    """Same invariant for average precision on the PR frame and overlay."""
    X, y01 = binary_data
    source, _ = _relabelled_source(X, y01, _RELABELLINGS[name])

    aps = source.pr_curve()["ap"].unique().to_list()
    assert aps == pytest.approx([references["ap"]], abs=1e-12)

    chart = ferrum.pr_chart(source)
    assert chart._title.text == "Precision–Recall Curve"
    assert f"AP = {references['ap']:.3f}" in svg_texts(chart.to_svg())


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

    chart = ferrum.calibration_chart(source)
    expected = f"Brier = {references['brier']:.3f}"
    assert expected in svg_texts(chart.to_svg())
    assert chart._title.text == "Calibration Curve"


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
    assert f"AUC = {references['auc']:.3f}" in svg_texts(roc.to_svg())

    pr = ferrum.pr_chart(y_true=y, y_pred=proba)
    assert f"AP = {references['ap']:.3f}" in svg_texts(pr.to_svg())

    cal = ferrum.calibration_chart(y_true=y, y_pred=proba)
    assert f"Brier = {references['brier']:.3f}" in svg_texts(cal.to_svg())


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

    Since #219(2) the composed label *replaces* the figure's rather than
    overstriking it, so the chart carries exactly one Brier string either
    way -- which is what the equality below now asserts.
    """
    X, y01 = binary_data
    model = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    source = ferrum.ModelSource(model, X, y01)

    chart = ferrum.calibration_chart(source, strategy=strategy)
    annotated = chart + ferrum.BrierLabel()

    expected = f"Brier = {references['brier']:.3f}"
    texts = svg_texts(annotated.to_svg())
    # Exactly one Brier string, and it is the per-sample one. No other Brier
    # number may appear anywhere on the composed chart, and the title -- which
    # used to carry a second copy -- is the plain name.
    assert [t for t in texts if t.startswith("Brier")] == [expected]
    assert "Calibration Curve" in texts


def test_brier_label_refuses_arbitrary_user_data_without_the_column():
    """A frame with no ``brier`` column cannot answer what the Brier score
    is: the binned points it draws define a different statistic. Since
    #219(3) the label refuses instead of showing that statistic under the
    Brier name -- the integrating fallback this test used to pin is gone."""
    df = pl.DataFrame({"p": [0.1, 0.4, 0.9], "obs": [0.0, 0.5, 1.0]})
    chart = ferrum.Chart(df).mark_line().encode(x="p", y="obs")

    with pytest.raises(ValueError, match=r"BrierLabel: the data has no 'brier' column"):
        chart + ferrum.BrierLabel()


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


class _WrongClasses(_DuckProba):
    """A model whose ``classes_`` only half-agrees with ``y``.

    The shape that separates the stray rule from the total-mismatch rule:
    with ``classes_ = [0, 2]`` against a ``{0, 1}`` target, the negative
    class matches real rows while the positive class matches none.
    """

    def __init__(self, fitted, classes):
        super().__init__(fitted)
        self.classes_ = np.asarray(classes)


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

    # Rendered surface (the overlay, sole metric surface since #219/D4).
    assert f"AUC = {references['auc']:.3f}" in svg_texts(ferrum.roc_chart(source).to_svg())
    assert f"Brier = {references['brier']:.3f}" in svg_texts(
        ferrum.calibration_chart(source).to_svg()
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


def test_an_all_null_target_refuses_as_a_null_target(binary_data):
    """An all-null target still refuses -- by the null rule, not the
    mismatch rule (#203, 2026-09-13).

    Batch G caught this case through the total-mismatch guard, because
    nulls compare unequal to everything and the target therefore looked
    entirely negative. The advice it gave ("check that y and the model's
    classes_ describe the same labels") sent the caller to inspect a
    ``classes_`` that was never wrong. The null guard now runs first and
    names the actual problem, so the refusal is kept and the diagnosis is
    the true one.
    """
    X, y01 = binary_data
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    nulls = pl.Series("y", [None] * len(y01), dtype=pl.Int64)
    source = ferrum.ModelSource(fitted, X, nulls)

    with pytest.raises(ValueError, match=r"roc_curve: y has 200 null row\(s\) out of 200"):
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
    the tool people reach for *because* the classes are skewed. The default
    fold generator is an unshuffled ``StratifiedKFold`` (#211, 2026-09-13),
    which keeps each fold's positive rate close to the whole target's --
    stratification makes a zero-positive *test* fold rare rather than the
    routine occurrence it was under the old shuffled plain ``KFold``, but
    "rare" is not "impossible" at these small positive counts, which is
    exactly what this test still exercises.
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


def test_discrimination_threshold_cv_is_deterministic_with_no_random_state():
    """Integer ``cv=`` is deterministic by construction (#211, 2026-09-13).

    The default splitter is now an unshuffled ``StratifiedKFold``, which has
    no seed to thread. Two independently-constructed sources over the same
    data with no ``random_state`` supplied at all must sweep to byte-
    identical frames -- restoring the spec's "ignores random_state" claim
    (``ferrum-spec.md:245``) to truth. Before this fix the splitter was
    ``KFold(shuffle=True, random_state=self._random_state or 0)``, which was
    deterministic only by accident of the ``or 0`` default, not because the
    method genuinely ignored the seed.
    """
    swept_a = _imbalanced_source().discrimination_threshold(cv=5)
    swept_b = _imbalanced_source().discrimination_threshold(cv=5)
    assert swept_a.equals(swept_b)


def test_discrimination_threshold_cv_ignores_random_state():
    """N3: two sources built from the identical data but DIFFERENT
    ``random_state`` values must sweep to the identical frame under
    ``cv=``, proving the spec's claim true rather than merely unbroken by
    coincidence.
    """
    rng = np.random.default_rng(0)
    features = rng.normal(size=(120, 3))
    y = np.zeros(120, dtype=int)
    y[rng.choice(120, 12, replace=False)] = 1
    X = pl.DataFrame(features, schema=["f0", "f1", "f2"])
    model = LogisticRegression(max_iter=1000).fit(features, y)

    seeded_low = ferrum.ModelSource(model, X, pl.Series("y", y), random_state=1)
    seeded_high = ferrum.ModelSource(model, X, pl.Series("y", y), random_state=99)
    assert seeded_low.discrimination_threshold(cv=5).equals(
        seeded_high.discrimination_threshold(cv=5)
    )


def test_discrimination_threshold_cv_accepts_a_user_supplied_stratified_splitter():
    """N2: a user-supplied splitter's ``.split()`` used to be called with no
    ``y`` (``splitter.split(X_np)``), so a stratified splitter -- which
    requires ``y`` -- raised ``TypeError: split() missing 1 required
    positional argument: 'y'``. ``.split(X, y)`` now supplies it.
    """
    from sklearn.model_selection import StratifiedKFold

    source = _imbalanced_source()
    swept = source.discrimination_threshold(cv=StratifiedKFold(n_splits=5))
    assert swept.height == 50
    assert set(swept.columns) == {"threshold", "precision", "recall", "f1", "queue_rate"}


class _FixedFoldSplitter:
    """A single hard-coded ``(train, test)`` split.

    Forces a train fold missing a class deterministically, rather than
    relying on ``StratifiedKFold``'s probabilistic behavior at small class
    counts to happen to produce one.
    """

    def __init__(self, train: np.ndarray, test: np.ndarray):
        self._train = train
        self._test = test

    def split(self, X, y=None, groups=None):
        yield self._train, self._test


def test_discrimination_threshold_cv_warns_on_a_train_fold_missing_the_positive_class():
    """A TRAIN fold missing a class is #211's unhandled sibling to the
    routine all-negative TEST fold: it leaves the cloned estimator nothing
    to fit two classes from, so sklearn's own ``fit`` raises. The warning
    must name the fold and the missing class *before* that raise, not
    replace it.
    """
    n = 20
    y = np.zeros(n, dtype=int)
    y[:4] = 1  # positives only at indices 0-3
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n, 3))
    X = pl.DataFrame(features, schema=["f0", "f1", "f2"])
    model = LogisticRegression(max_iter=1000).fit(features, y)
    source = ferrum.ModelSource(model, X, pl.Series("y", y), random_state=0)

    train = np.arange(4, n)  # excludes every positive row
    test = np.arange(0, 4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="2 classes"):
            source.discrimination_threshold(cv=_FixedFoldSplitter(train, test))

    messages = [str(w.message) for w in caught if w.category is UserWarning]
    assert any("fold 0" in m and "no rows of the positive class" in m for m in messages)


def test_discrimination_threshold_cv_warns_on_a_train_fold_missing_the_negative_class():
    """The symmetric case: a train fold holding only the positive class."""
    n = 20
    y = np.zeros(n, dtype=int)
    y[16:] = 1  # negatives only at indices 0-15; positives at 16-19
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n, 3))
    X = pl.DataFrame(features, schema=["f0", "f1", "f2"])
    model = LogisticRegression(max_iter=1000).fit(features, y)
    source = ferrum.ModelSource(model, X, pl.Series("y", y), random_state=0)

    train = np.arange(16, n)  # every row is the positive class
    test = np.arange(0, 16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="2 classes"):
            source.discrimination_threshold(cv=_FixedFoldSplitter(train, test))

    messages = [str(w.message) for w in caught if w.category is UserWarning]
    assert any("fold 0" in m and "no rows of the negative class" in m for m in messages)


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


# ---------------------------------------------------------------------------
# The target-hygiene matrix (#203)
# ---------------------------------------------------------------------------
#
# One contract for a target that disagrees with the model, one cell per
# shape, pinned on both source paths:
#
#   nulls                      -> refuse (any dtype, any diagnostic)
#   stray values among valid   -> warn + absorb as negative
#   nothing matches (binary)   -> refuse, by the positive class's name
#   nothing matches (multi)    -> refuse, by the resolved class set
#   some classes absent (multi)-> absorb in silence (one-vs-rest's own answer)
#
# Before this, the answer depended on the target's dtype and on which
# resolution tier the model happened to land in; the sections above pin the
# parts batch G closed, this one pins the rest.


#: The six binary diagnostics that binarize ``y`` against a positive class,
#: as ``(source method call, the name its refusals use)``. ``cumulative_gain``
#: and ``lift_curve`` are the two the batch-G guard never reached.
_BINARY_DIAGNOSTICS = [
    pytest.param(lambda s: s.roc_curve(), "roc_curve", id="roc_curve"),
    pytest.param(lambda s: s.pr_curve(), "pr_curve", id="pr_curve"),
    pytest.param(lambda s: s.calibration_curve(), "calibration_curve", id="calibration_curve"),
    pytest.param(lambda s: s.cumulative_gain(), "cumulative_gain", id="cumulative_gain"),
    pytest.param(lambda s: s.lift_curve(), "lift_curve", id="lift_curve"),
    pytest.param(
        lambda s: s.discrimination_threshold(),
        "discrimination_threshold",
        id="discrimination_threshold",
    ),
]

#: The same six on the precomputed path, as figure functions taking
#: ``y_true=``/``y_pred=``, plus the two label-consuming ones.
_PRECOMPUTED_SCORE_CHARTS = [
    pytest.param(ferrum.roc_chart, "roc_chart", id="roc_chart"),
    pytest.param(ferrum.pr_chart, "pr_chart", id="pr_chart"),
    pytest.param(ferrum.calibration_chart, "calibration_chart", id="calibration_chart"),
    pytest.param(ferrum.gain_chart, "gain_chart", id="gain_chart"),
    pytest.param(ferrum.lift_chart, "lift_chart", id="lift_chart"),
    pytest.param(
        ferrum.discrimination_threshold_chart,
        "discrimination_threshold_chart",
        id="discrimination_threshold_chart",
    ),
]

_PRECOMPUTED_LABEL_CHARTS = [
    pytest.param(ferrum.confusion_matrix_chart, "confusion_matrix_chart", id="confusion_matrix"),
    pytest.param(
        ferrum.classification_report_chart,
        "classification_report_chart",
        id="classification_report",
    ),
]


def _corrupted_binary_source(binary_data, replacement, *, dtype=pl.Int64, hide_classes=False):
    """A ``ModelSource`` whose y's first ten rows carry ``replacement``.

    The model is fitted on the clean ``{0, 1}`` target, so ``classes_`` is
    right and only the target is corrupt -- the shape a user actually hands
    ferrum when ten rows of a join came back empty or mis-coded.
    ``hide_classes`` wraps it so resolution has to fall past tier 1.
    """
    X, y01 = binary_data
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    values = [replacement if i < 10 else v for i, v in enumerate(y01.to_list())]
    y = pl.Series("y", values, dtype=dtype)
    model = _DuckProba(fitted) if hide_classes else fitted
    return ferrum.ModelSource(model, X, y)


def _string_binary_source(binary_data, replacement):
    """The utf8 twin of :func:`_corrupted_binary_source`.

    Dtype is the axis that decided the old behavior: a string target with
    nulls crashed with an unowned ``TypeError``, a numeric one degraded in
    silence. Both must now answer identically.
    """
    X, y01 = binary_data
    labels = ["pos" if int(v) == 1 else "neg" for v in y01.to_list()]
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), np.array(labels))
    values = [replacement if i < 10 else v for i, v in enumerate(labels)]
    return ferrum.ModelSource(fitted, X, pl.Series("y", values, dtype=pl.Utf8))


# --- cell: a null-bearing target refuses -----------------------------------


@pytest.mark.parametrize(("call", "name"), _BINARY_DIAGNOSTICS)
@pytest.mark.parametrize("dtype", ["numeric", "utf8"])
def test_a_null_bearing_target_refuses_on_every_binary_diagnostic(call, name, dtype, binary_data):
    """Ten null rows out of two hundred, and every diagnostic says so.

    This is the cell that ends the dtype fork. On ``Utf8`` the call raised
    ``TypeError: '<' not supported between instances of 'str' and 'NoneType'``
    from whichever of four ``np.unique``/``sorted`` sites it reached first,
    naming neither the target nor the diagnostic. On ``Int64`` nothing
    raised at all: the nulls arrived as ``NaN``, became a third distinct
    class, and the metric silently shifted (measured AUC 0.9395 against a
    clean 0.9843) -- or, on a ``classes_``-less model, the extra class
    pushed resolution from tier 2 to tier 3 and turned the same input into
    a refusal about the wrong thing.
    """
    source = (
        _corrupted_binary_source(binary_data, None)
        if dtype == "numeric"
        else _string_binary_source(binary_data, None)
    )

    with pytest.raises(ValueError) as excinfo:
        call(source)
    message = str(excinfo.value)
    assert message.startswith(f"{name}: y has 10 null row(s) out of 200")
    assert "Drop or impute the null rows" in message
    # The observed values still come back, nulls included, so the caller can
    # see what the target holds without re-deriving it.
    assert ("None" in message) if dtype == "utf8" else ("nan" in message)


@pytest.mark.parametrize("hide_classes", [False, True], ids=["with_classes_", "no_classes_"])
def test_a_null_bearing_target_refuses_on_confusion_matrix(hide_classes, binary_data):
    """Crash site four: ``confusion_matrix``'s no-``classes_`` branch.

    ``sorted(set(y_true) | set(y_pred))`` is a plain sort with no key, so a
    ``None`` among string labels raised the identical unowned ``TypeError``
    from a completely different owner. Both branches route through the same
    guard now.
    """
    X, y01 = binary_data
    labels = ["pos" if int(v) == 1 else "neg" for v in y01.to_list()]
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), np.array(labels))
    values = [None if i < 10 else v for i, v in enumerate(labels)]
    model = _DuckProba(fitted) if hide_classes else fitted
    source = ferrum.ModelSource(model, X, pl.Series("y", values, dtype=pl.Utf8))

    with pytest.raises(ValueError, match=r"confusion_matrix: y has 10 null row\(s\) out of 200"):
        source.confusion_matrix()


def test_a_null_bearing_multiclass_target_refuses():
    """The multiclass tier guard is the third crash site, and it answers
    the same way the binary ones do."""
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    labels = [f"c{int(v)}" for v in df["y"].to_list()]
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), np.array(labels))
    values = [None if i < 10 else v for i, v in enumerate(labels)]
    source = ferrum.ModelSource(_DuckProba(fitted), X, pl.Series("y", values, dtype=pl.Utf8))

    with pytest.raises(ValueError, match=r"roc_curve: y has 10 null row\(s\) out of 300"):
        source.roc_curve()


@pytest.mark.parametrize(("chart_fn", "name"), _PRECOMPUTED_SCORE_CHARTS)
@pytest.mark.parametrize("dtype", ["numeric", "utf8"])
def test_a_null_bearing_target_refuses_on_the_precomputed_path(chart_fn, name, dtype):
    """The precomputed twins had no null coverage at all, and three
    different behaviors between them: ``roc_chart`` refused for the wrong
    reason ("got 3 unique classes", counting ``NaN`` as one),
    ``calibration_chart`` refused for another wrong reason ("no row of y
    equals the positive class nan"), and ``gain_chart`` raised the unowned
    ``TypeError``. One answer now.
    """
    if dtype == "numeric":
        y_true = np.array([1.0, np.nan, 0.0, 1.0, 0.0, 1.0])
    else:
        y_true = np.array(["pos", None, "neg", "pos", "neg", "pos"], dtype=object)
    y_pred = np.array([0.9, 0.5, 0.1, 0.8, 0.2, 0.7])

    with pytest.raises(ValueError) as excinfo:
        chart_fn(y_true=y_true, y_pred=y_pred)
    assert str(excinfo.value).startswith(f"{name}: y has 1 null row(s) out of 6")


@pytest.mark.parametrize(("chart_fn", "name"), _PRECOMPUTED_LABEL_CHARTS)
def test_a_null_bearing_target_refuses_on_the_precomputed_label_charts(chart_fn, name):
    """``sorted(..., key=str)`` made these two accidentally immune to the
    crash -- they stringified ``None`` to ``"None"`` and rendered it as a
    class of its own, which is worse than raising. They refuse now."""
    y_true = np.array(["a", None, "b", "a"], dtype=object)
    y_pred = np.array(["a", "a", "b", "b"], dtype=object)

    with pytest.raises(ValueError) as excinfo:
        chart_fn(y_true=y_true, y_pred=y_pred)
    assert str(excinfo.value).startswith(f"{name}: y has 1 null row(s) out of 4")


# --- cell: stray out-of-domain values warn and absorb ----------------------


@pytest.mark.parametrize(("call", "name"), _BINARY_DIAGNOSTICS)
def test_a_stray_value_absorbs_as_negative_and_says_so(call, name, binary_data):
    """Absorb is right; the silence was the defect.

    Counting an out-of-domain row as "not the positive class" is
    scikit-learn's own ``pos_label`` semantics, so the answer stays. What
    was missing is any sign that ten of the two hundred rows were not the
    target the caller thought they were passing -- the metric just moved.
    """
    source = _corrupted_binary_source(binary_data, 7)

    with pytest.warns(UserWarning) as record:
        call(source)

    hygiene = [str(w.message) for w in record if str(w.message).startswith(f"{name}:")]
    assert len(hygiene) == 1, hygiene
    message = hygiene[0]
    assert "10 row(s) of y hold values outside the resolved class set [0, 1]" in message
    assert "were counted as negative" in message
    assert "out-of-domain values are [7]" in message


def test_the_absorbed_metric_is_the_absorbed_metric(binary_data, references):
    """The number the warning warns about, checked against sklearn.

    The absorb has to be exactly "those rows are negatives" -- not a drop,
    not a re-fit -- so sklearn scoring the same probabilities against the
    same absorbed indicator is the oracle. Pinning the *gap* as well keeps
    the cell honest: a warning about a metric that had not moved would be
    noise.
    """
    source = _corrupted_binary_source(binary_data, 7)
    with pytest.warns(UserWarning, match=r"roc_curve: 10 row\(s\)"):
        auc = source.roc_curve()["auc"].unique().to_list()

    y_absorbed = np.array([0 if i < 10 else int(v) for i, v in enumerate(binary_data[1].to_list())])
    expected = float(roc_auc_score(y_absorbed, references["proba"]))
    assert auc == pytest.approx([expected], abs=1e-12)
    assert abs(expected - references["auc"]) > 0.01


def test_a_stray_value_absorbs_on_the_multiclass_path_too(binary_data):
    """Uniform across arities, not only across the six binary entries."""
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), df["y"].to_numpy())
    values = [9 if i < 10 else int(v) for i, v in enumerate(df["y"].to_list())]
    source = ferrum.ModelSource(fitted, X, pl.Series("y", values))

    with pytest.warns(UserWarning, match=r"roc_curve: 10 row\(s\) of y hold values outside"):
        roc = source.roc_curve()
    assert set(roc["class"].to_list()) == {"0", "1", "2"}


def test_the_precomputed_calibration_pair_warns_about_what_it_absorbs():
    """The one precomputed diagnostic that can absorb a stray.

    roc/pr/gain/lift read their class set off ``y_true`` itself, so every
    row is in it by construction; ``calibration_chart`` reduces a 3+-valued
    target to its extremes (``binary_class_values``) and counts everything
    between as negative. It did that in silence.
    """
    y_true = np.array([0, 1, 7, 0, 1, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.5, 0.2, 0.8, 0.7, 0.3, 0.6])

    with pytest.warns(UserWarning) as record:
        ferrum.calibration_chart(y_true=y_true, y_pred=y_pred)

    hygiene = [str(w.message) for w in record if str(w.message).startswith("calibration_chart:")]
    assert len(hygiene) == 1, hygiene
    assert "resolved class set [0, 7]" in hygiene[0]
    assert "out-of-domain values are [1]" in hygiene[0]


@pytest.mark.parametrize(("call", "name"), _BINARY_DIAGNOSTICS)
def test_a_clean_target_emits_no_hygiene_warning(call, name, binary_data):
    """The guard's own false-positive check: a clean ``{0, 1}`` target --
    what every fixture and golden in the repo uses -- must stay silent, or
    the whole warning becomes noise users learn to filter."""
    X, y01 = binary_data
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    source = ferrum.ModelSource(fitted, X, y01)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call(source)
    assert [str(w.message) for w in record if str(w.message).startswith(f"{name}:")] == []


def test_multiclass_absent_classes_absorb_without_a_warning():
    """The one-vs-rest allowance is not a stray.

    ``test_multiclass_one_vs_rest_still_tolerates_an_absent_class`` pins
    that the *curves* survive; this pins that they survive quietly. Every
    row of ``y`` is inside the resolved class set -- it is the class set
    that has a member no row holds -- so there is nothing absorbed to warn
    about, and warning here would fire on the routine case.
    """
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    model = LogisticRegression(max_iter=1000).fit(X.to_numpy(), df["y"].to_numpy())
    y = pl.Series("y", [0 if int(v) == 2 else int(v) for v in df["y"].to_list()])
    source = ferrum.ModelSource(model, X, y)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        source.roc_curve()
    assert [str(w.message) for w in record if str(w.message).startswith("roc_curve:")] == []


# --- cell: nothing matches -> refuse, on the two reaches that lacked it ----


@pytest.mark.parametrize(
    ("call", "name"),
    [
        pytest.param(lambda s: s.cumulative_gain(), "cumulative_gain", id="cumulative_gain"),
        pytest.param(lambda s: s.lift_curve(), "lift_curve", id="lift_curve"),
    ],
)
def test_gain_and_lift_join_the_total_mismatch_refusal(call, name, binary_data):
    """``binary_indicator``'s docstring has always said "every binary curve
    builder"; gain and lift were the two that weren't.

    On a target holding no positive row, ``cumulative_gain`` drew a curve
    reaching gain 1.0 (its denominator floors at 1) and ``lift_curve``
    dropped the positive class through its zero-base-rate ``continue`` and
    returned the baseline alone -- both plausible, neither meaningful,
    while ``roc_curve`` on the identical source refused by name.
    """
    X, y01 = binary_data
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    source = ferrum.ModelSource(fitted, X, pl.Series("y", [0] * len(y01)))

    with pytest.raises(ValueError) as excinfo:
        call(source)
    message = str(excinfo.value)
    assert message.startswith(f"{name}: no row of y equals the positive class 1")
    assert "y holds [0]" in message


def test_tier_3_multiclass_refuses_when_nothing_matches():
    """The silence the resolver's own docstring called a known gap.

    A ``classes_``-less multiclass model whose ``y`` cardinality disagrees
    with its score-column count resolves to parsed ``class_{i}`` labels,
    which match no row. Binary caught that through the positive class's
    guard; multiclass had none and returned three all-NaN curves, where
    sklearn's one-vs-rest AUCs over the same probabilities are 0.872,
    0.942 and 0.930 -- the refusal is what stops that from rendering as a
    blank chart with no explanation.
    """
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), df["y"].to_numpy())
    y = pl.Series("y", [9 if i < 10 else int(v) for i, v in enumerate(df["y"].to_list())])
    source = ferrum.ModelSource(_DuckProba(fitted), X, y)

    with pytest.raises(ValueError) as excinfo:
        source.roc_curve()
    message = str(excinfo.value)
    assert message.startswith(
        "roc_curve: no row of y equals any of the resolved class values "
        "['class_0', 'class_1', 'class_2']"
    )
    assert "y holds [0, 1, 2, 9]" in message


def test_a_stray_value_still_refuses_when_resolution_cannot_survive_it(binary_data):
    """The one cell where a stray refuses instead of absorbing, pinned so
    the asymmetry is deliberate rather than discovered.

    Absorb needs a class set to absorb *into*. On a ``classes_``-less
    binary model the stray inflates ``y``'s cardinality past the score
    column count, resolution falls to tier 3's parsed ``class_{i}`` labels,
    and those match no row at all -- a total mismatch, not a stray among
    valid rows. It refuses through ``binary_indicator``, exactly as it did
    before #203, and the hygiene guard defers to that message rather than
    producing a second one.
    """
    source = _corrupted_binary_source(binary_data, 7, hide_classes=True)

    with pytest.raises(ValueError) as excinfo:
        source.roc_curve()
    assert "no row of y equals the positive class 'class_1'" in str(excinfo.value)


def test_a_partly_wrong_class_set_refuses_without_first_claiming_an_absorb(binary_data):
    """A refusal cannot be preceded by a warning about what it absorbed.

    ``classes_ = [0, 2]`` against a ``{0, 1}`` target resolves a positive
    class no row holds while the negative class still matches half the
    rows, so it is a *partial* mismatch to the stray rule and a total one
    to ``binary_indicator``. It used to be both: the call emitted
    ``roc_curve: 96 row(s) of y hold values outside the resolved class set
    [0, 2] and were counted as negative`` and *then* raised. Nothing was
    counted as negative -- the call returned no number at all -- so the
    warning asserted something that never happened. The refusal is the
    whole answer now.
    """
    X, y01 = binary_data
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    source = ferrum.ModelSource(_WrongClasses(fitted, [0, 2]), X, y01)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as excinfo:
            source.roc_curve()
    assert str(excinfo.value).startswith("roc_curve: no row of y equals the positive class 2")
    assert [str(w.message) for w in record if str(w.message).startswith("roc_curve:")] == []


# --- cell: null beats stray when the target holds both ---------------------


def test_a_null_beats_a_stray_when_the_target_holds_both(binary_data):
    """Both rules claim this target, and the refusal has to win.

    A target carrying five nulls *and* ten stray rows satisfies the null
    rule (refuse) and the stray rule (warn and absorb) at once. Refusing is
    right -- the absorb would report a number measured partly against rows
    the target never recorded -- but the ordering was unpinned, so swapping
    the two guards would have turned this into a warning plus that number
    with nothing failing.
    """
    X, y01 = binary_data
    values = [None if i < 5 else (7 if i < 15 else int(v)) for i, v in enumerate(y01.to_list())]
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    source = ferrum.ModelSource(fitted, X, pl.Series("y", values))

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as excinfo:
            source.roc_curve()
    assert str(excinfo.value).startswith("roc_curve: y has 5 null row(s) out of 200")
    assert [str(w.message) for w in record if str(w.message).startswith("roc_curve:")] == []


# --- cell: a temporal null is a null too -----------------------------------


@pytest.mark.parametrize(
    ("kind", "stamps"),
    [
        pytest.param("date", [dt.date(2020, 1, 1), dt.date(2020, 1, 2)], id="date"),
        pytest.param(
            "datetime",
            [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2)],
            id="datetime",
        ),
    ],
)
def test_a_temporal_null_target_refuses_like_every_other_dtype(kind, stamps):
    """The dtype family the null mask did not inspect at all.

    A ``pl.Date``/``pl.Datetime`` column reaches numpy as ``datetime64``
    carrying ``NaT``, which is neither a float ``NaN`` nor an object
    ``None``, so the mask came back all-``False`` and the guard passed a
    null-bearing target straight through. Measured before this cell:
    ``confusion_matrix_chart`` rendered a ``None`` row *and* column as a
    class of its own -- exactly the pre-#203 behavior the null rule
    replaced on every other dtype. The rule is the target's, not the
    dtype's.
    """
    y_true = np.asarray(pl.Series("y", [stamps[0], None, stamps[1], stamps[0]]))
    y_pred = np.asarray(pl.Series("p", [stamps[0], stamps[0], stamps[1], stamps[0]]))
    assert y_true.dtype.kind == "M", y_true.dtype

    with pytest.raises(ValueError) as excinfo:
        ferrum.confusion_matrix_chart(y_true=y_true, y_pred=y_pred)
    assert str(excinfo.value).startswith("confusion_matrix_chart: y has 1 null row(s) out of 4")


# --- cell: the confusion family drops its strays, and says dropped ---------


def test_a_stray_value_is_dropped_from_the_confusion_matrix_and_says_so(binary_data):
    """The one diagnostic whose strays do not absorb.

    ``confusion_frame`` encodes each value to its index in the model's
    class set and maps everything else to the kernel's out-of-vocabulary
    code, which drops the row: it is counted in no cell rather than
    absorbed into a negative. Measured before this: the cells summed to
    190.0 of 200 rows in complete silence, while ``roc_curve`` on the
    identical source warned. The message has to say *dropped*, because a
    message borrowed from the absorb cell would describe arithmetic the
    kernel does not do.
    """
    source = _corrupted_binary_source(binary_data, 7)

    with pytest.warns(UserWarning) as record:
        cm = source.confusion_matrix()

    hygiene = [str(w.message) for w in record if str(w.message).startswith("confusion_matrix:")]
    assert len(hygiene) == 1, hygiene
    message = hygiene[0]
    assert "10 row(s) of y out of 200 hold values outside the class set [0, 1]" in message
    assert "dropped from it entirely" in message
    assert "counted in no cell, not absorbed into one" in message
    assert "out-of-domain values are [7]" in message
    # The loss the warning is about, in the frame itself.
    assert cm["value"].sum() == pytest.approx(190.0)


def test_the_confusion_drop_warning_covers_the_multiclass_matrix_too():
    """Uniform across arities, like the absorb cell above it."""
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), df["y"].to_numpy())
    values = [9 if i < 10 else int(v) for i, v in enumerate(df["y"].to_list())]
    source = ferrum.ModelSource(fitted, X, pl.Series("y", values))

    with pytest.warns(UserWarning) as record:
        cm = source.confusion_matrix()

    hygiene = [str(w.message) for w in record if str(w.message).startswith("confusion_matrix:")]
    assert len(hygiene) == 1, hygiene
    assert "10 row(s) of y out of 300 hold values outside the class set [0, 1, 2]" in hygiene[0]
    assert cm["value"].sum() == pytest.approx(290.0)


def test_class_prediction_error_inherits_the_confusion_drop_warning(binary_data):
    """It renders the confusion frame, so it loses the same rows.

    A *fresh* source deliberately: every diagnostic result is cached on
    ``(method, kwargs)``, so a chart built from a source whose
    ``confusion_matrix()`` was already computed returns the cached frame
    and does not re-warn.
    """
    source = _corrupted_binary_source(binary_data, 7)

    with pytest.warns(UserWarning, match=r"confusion_matrix: 10 row\(s\) of y out of 200"):
        ferrum.class_prediction_error_chart(source)


def test_a_clean_target_leaves_the_confusion_matrix_silent(binary_data):
    """The drop warning's own false-positive check.

    Every confusion fixture and golden in the repo passes a target whose
    values are all in ``classes_``; none of them may start warning.
    """
    X, y01 = binary_data
    fitted = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y01.to_numpy())
    source = ferrum.ModelSource(fitted, X, y01)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        source.confusion_matrix()
    assert [str(w.message) for w in record if str(w.message).startswith("confusion_matrix:")] == []
