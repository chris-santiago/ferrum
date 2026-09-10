"""Batch G / F-L11-01 — every displayed AUC/AP is the correct metric, once.

Three claims are pinned here, all at the user-observable surface:

1. Every one of the four swept sort sites is **stable**, so a curve with tied
   values (routine on ROC/PR/gain/lift curves) yields the correct answer
   rather than whichever within-tie permutation numpy's quicksort produced:
   ``_trapezoid_auc`` and ``_ap_step`` integrating, ``gain_frame`` and
   ``lift_frame`` ranking by score, and ``_pr_average`` sorting by recall
   before interpolating.
2. A diagnostic chart shows **one** AUC/AP, from the frame's kernel-computed
   ``auc``/``ap`` column: chart title, legend text and label overlay agree
   with the column and with each other.
3. ``mark_roc(annotate_auc=True)`` / ``mark_pr(annotate_ap=True)`` render
   their label instead of raising ``layer N: unknown column '_auc_label_x'``.

Every numeric expectation is hand-computed here (Python's ``sorted`` is a
guaranteed-stable sort; the trapezoid and step-integration loops are written
out inline) or read from a *different* code path than the one asserted --
never from the function under test.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import ferrum
from ferrum._metric_labels import _ap_step, _trapezoid_auc
from ferrum.diagnostics._internal._curve_frames import (
    _AVG_GRID_POINTS,
    gain_frame,
    lift_frame,
    pr_frame,
)
from tests.fixtures import load_dataset, load_fixture

# Two tie-heavy curves as ``(x, y)`` points, in threshold order. Eight of the
# seventeen x values repeat on each, so the within-tie ordering alone decides
# which y bounds each trapezoid -- the failure mode F-L11-01 describes. Fixed
# values, reproducible.
TIED_ROC_POINTS = [
    (0.44, 1.0), (0.0, 0.16), (0.7, 1.0), (0.47, 1.0), (0.48, 1.0), (0.0, 0.56),
    (0.89, 1.0), (0.0, 0.26), (0.14, 1.0), (0.0, 0.53), (0.0, 0.76), (0.0, 0.45),
    (0.0, 0.88), (0.61, 1.0), (0.0, 0.49), (0.46, 1.0), (0.73, 1.0),
]  # fmt: skip
TIED_PR_POINTS = [
    (0.84, 0.52), (0.0, 0.16), (0.0, 0.39), (0.87, 0.73), (0.8, 0.78), (0.0, 0.07),
    (0.13, 0.99), (0.74, 0.9), (0.39, 0.5), (0.0, 0.26), (0.14, 0.56), (0.0, 0.39),
    (0.0, 0.52), (0.0, 0.3), (0.14, 0.71), (0.0, 0.49), (0.53, 0.86),
]  # fmt: skip

TIED_FPR = [x for x, _ in TIED_ROC_POINTS]
TIED_TPR = [y for _, y in TIED_ROC_POINTS]
TIED_RECALL = [x for x, _ in TIED_PR_POINTS]
TIED_PRECISION = [y for _, y in TIED_PR_POINTS]


def hand_trapezoid(x: list[float], y: list[float]) -> float:
    """Trapezoidal integral over a stable-sorted curve, written out by hand."""
    order = sorted(range(len(x)), key=lambda i: x[i])  # sorted() is stable
    xs = [x[i] for i in order]
    ys = [y[i] for i in order]
    return sum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0 for i in range(1, len(xs)))


def hand_ap_step(recall: list[float], precision: list[float]) -> float:
    """Step-integrated AP over a stable-sorted curve, written out by hand."""
    order = sorted(range(len(recall)), key=lambda i: recall[i])
    total = 0.0
    previous = 0.0
    for i in order:
        total += (recall[i] - previous) * precision[i]
        previous = recall[i]
    return total


def svg_texts(svg: str) -> list[str]:
    """Every rendered text run in an SVG, in document order."""
    return [t for t in re.findall(r"<text[^>]*>([^<]*)</text>", svg)]


@pytest.fixture(scope="module")
def binary_source():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    return ferrum.ModelSource(model, df.select(["f0", "f1", "f2", "f3"]), df["y"])


@pytest.fixture(scope="module")
def multi_source():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    return ferrum.ModelSource(model, df.select(["f0", "f1", "f2", "f3"]), df["y"])


# ---------------------------------------------------------------------------
# 1. Stable integration
# ---------------------------------------------------------------------------


def test_trapezoid_auc_matches_hand_stable_integral_on_tied_x():
    """Tied x values must keep their threshold order through the sort."""
    x = np.array(TIED_FPR)
    y = np.array(TIED_TPR)
    assert _trapezoid_auc(x, y) == pytest.approx(hand_trapezoid(TIED_FPR, TIED_TPR), abs=1e-12)


def test_ap_step_matches_hand_stable_integral_on_tied_recall():
    x = np.array(TIED_RECALL)
    y = np.array(TIED_PRECISION)
    assert _ap_step(x, y) == pytest.approx(hand_ap_step(TIED_RECALL, TIED_PRECISION), abs=1e-12)


def test_auc_label_overlay_on_tied_user_data_is_correct_and_repeatable():
    """The composable ``+ AUCLabel()`` path has no metric column to read, so
    it integrates in Python -- on tie-heavy user data it must still land on
    the hand-computed value, and two builds of the same chart must agree."""
    df = pl.DataFrame({"fpr": TIED_FPR, "tpr": TIED_TPR})
    expected = f"AUC = {hand_trapezoid(TIED_FPR, TIED_TPR):.3f}"

    def build() -> str:
        chart = ferrum.Chart(df).mark_line().encode(x="fpr", y="tpr")
        return (chart + ferrum.AUCLabel()).to_svg()

    first, second = build(), build()
    assert expected in svg_texts(first)
    assert first == second


# ---------------------------------------------------------------------------
# 2. One metric, one source: title == legend == overlay == frame column
# ---------------------------------------------------------------------------


def test_roc_chart_title_legend_and_overlay_all_equal_the_auc_column(binary_source):
    column_auc = binary_source.roc_curve(average=None)["auc"].unique().to_list()
    assert len(column_auc) == 1
    expected = f"{column_auc[0]:.3f}"

    chart = ferrum.roc_chart(binary_source, annotate_auc=True)
    texts = svg_texts(chart.to_svg())

    assert chart._title.text == f"ROC Curve — AUC {expected}"
    assert f"AUC = {expected}" in texts  # label overlay
    assert any(t.endswith(f"(AUC = {expected})") for t in texts)  # legend entry


def test_pr_chart_title_legend_and_overlay_all_equal_the_ap_column(binary_source):
    column_ap = binary_source.pr_curve(average=None)["ap"].unique().to_list()
    assert len(column_ap) == 1
    expected = f"{column_ap[0]:.3f}"

    chart = ferrum.pr_chart(binary_source, annotate_ap=True)
    texts = svg_texts(chart.to_svg())

    assert chart._title.text == f"Precision–Recall Curve — AP {expected}"
    assert f"AP = {expected}" in texts
    assert any(t.endswith(f"(AP = {expected})") for t in texts)


def test_roc_chart_per_class_legend_entries_equal_their_own_auc_column(multi_source):
    """Multiclass: each legend entry carries its own class's column AUC."""
    frame = multi_source.roc_curve(average=None)
    per_class = {
        cls: frame.filter(pl.col("class") == cls)["auc"].unique().to_list()[0]
        for cls in frame["class"].unique(maintain_order=True).to_list()
    }
    texts = svg_texts(ferrum.roc_chart(multi_source, annotate_auc=True).to_svg())
    for cls, auc in per_class.items():
        assert f"{cls} (AUC = {auc:.3f})" in texts


def test_roc_chart_averaged_title_is_the_macro_mean_of_per_class_aucs(multi_source):
    """``per_class=False`` titles the weighted mean the frame carries -- for
    macro that is the plain mean of the per-class AUCs (sklearn parity), not
    a trapezoid over the interpolated average curve."""
    frame = multi_source.roc_curve(average=None)
    per_class = [
        frame.filter(pl.col("class") == cls)["auc"].unique().to_list()[0]
        for cls in frame["class"].unique(maintain_order=True).to_list()
    ]
    expected = sum(per_class) / len(per_class)

    chart = ferrum.roc_chart(multi_source, per_class=False, average="macro")
    assert chart._title.text == f"ROC Curve — AUC {expected:.3f}"


def test_pr_chart_averaged_title_is_the_macro_mean_of_per_class_aps(multi_source):
    """Same semantics for PR, on a fixture where the two candidate readings
    visibly disagree: the mean of the per-class APs rounds to 0.872, while
    a step integral over the interpolated macro curve rounds to 0.868."""
    frame = multi_source.pr_curve(average=None)
    per_class = [
        frame.filter(pl.col("class") == cls)["ap"].unique().to_list()[0]
        for cls in frame["class"].unique(maintain_order=True).to_list()
    ]
    expected = sum(per_class) / len(per_class)

    macro = multi_source.pr_curve(average="macro")
    interpolated = _ap_step(
        np.asarray(macro["recall"].to_list(), dtype=float),
        np.asarray(macro["precision"].to_list(), dtype=float),
    )
    assert f"{expected:.3f}" != f"{interpolated:.3f}"  # the readings really differ

    chart = ferrum.pr_chart(multi_source, per_class=False, average="macro")
    assert chart._title.text == f"Precision–Recall Curve — AP {expected:.3f}"


# ---------------------------------------------------------------------------
# 3. mark_roc / mark_pr annotate wiring
# ---------------------------------------------------------------------------


def test_mark_roc_annotate_auc_renders_the_frame_column_value(binary_source):
    """The mark used to raise ``layer N: unknown column '_auc_label_x'``:
    the desugar emitted a text layer bound to columns nobody injected."""
    frame = binary_source.roc_curve(average=None)
    auc = frame["auc"].unique().to_list()[0]
    cls = frame["class"].unique().to_list()[0]

    svg = ferrum.Chart(frame).mark_roc(annotate_auc=True).encode(x="fpr", y="tpr").to_svg()
    assert f"{cls}: AUC = {auc:.3f}" in svg_texts(svg)


def test_mark_pr_annotate_ap_renders_the_frame_column_value(binary_source):
    frame = binary_source.pr_curve(average=None)
    ap = frame["ap"].unique().to_list()[0]
    cls = frame["class"].unique().to_list()[0]

    svg = ferrum.Chart(frame).mark_pr(annotate_ap=True).encode(x="recall", y="precision").to_svg()
    assert f"{cls}: AP = {ap:.3f}" in svg_texts(svg)


def test_mark_roc_annotate_auc_integrates_when_the_frame_has_no_auc_column():
    """A hand-built curve frame carries no kernel metric, so the label falls
    back to the stable integration -- and still renders."""
    df = pl.DataFrame(
        {
            "fpr": TIED_FPR,
            "tpr": TIED_TPR,
            "class": ["a"] * len(TIED_FPR),
        }
    )
    svg = ferrum.Chart(df).mark_roc(annotate_auc=True).encode(x="fpr", y="tpr").to_svg()
    assert f"a: AUC = {hand_trapezoid(TIED_FPR, TIED_TPR):.3f}" in svg_texts(svg)


def test_mark_roc_annotate_auc_keeps_caller_injected_label_columns(binary_source):
    """A caller that pre-injects the label columns keeps its own values."""
    frame = binary_source.roc_curve(average=None).with_columns(
        pl.Series("_auc_label_x", [0.5] + [None] * (binary_source.roc_curve()["auc"].len() - 1)),
        pl.Series("_auc_label_y", [0.5] + [None] * (binary_source.roc_curve()["auc"].len() - 1)),
        pl.Series(
            "_auc_label",
            ["caller supplied"] + [None] * (binary_source.roc_curve()["auc"].len() - 1),
        ),
    )
    svg = ferrum.Chart(frame).mark_roc(annotate_auc=True).encode(x="fpr", y="tpr").to_svg()
    assert "caller supplied" in svg_texts(svg)


def test_mark_roc_annotate_auc_false_emits_no_label(binary_source):
    frame = binary_source.roc_curve(average=None)
    svg = ferrum.Chart(frame).mark_roc(annotate_auc=False).encode(x="fpr", y="tpr").to_svg()
    assert not any("AUC" in t for t in svg_texts(svg))


# ---------------------------------------------------------------------------
# 4. Gain / lift tie ordering
# ---------------------------------------------------------------------------

# 40 rows over three distinct scores, so every score is tied 13-14 ways: the
# ranking of tied rows is entirely the sort's choice.
_TIED_SCORES = [0.9 if i % 3 == 0 else (0.5 if i % 3 == 1 else 0.1) for i in range(40)]
_TIED_Y = [1 if (i % 5) in (0, 3) else 0 for i in range(40)]


def _hand_stable_ranking() -> list[int]:
    """Row order for a descending-score sweep, ties in input order."""
    return sorted(range(len(_TIED_SCORES)), key=lambda i: -_TIED_SCORES[i])


def test_gain_frame_orders_tied_scores_by_input_order():
    order = _hand_stable_ranking()
    cumulative = np.cumsum([_TIED_Y[i] for i in order])
    expected = [0.0] + [c / cumulative[-1] for c in cumulative]

    frame = gain_frame(
        np.array(_TIED_Y),
        np.array(_TIED_SCORES, dtype=float).reshape(-1, 1),
        [1],
        ["1"],
    )
    got = frame.filter(pl.col("class") == "1")["gain"].to_list()
    assert got == pytest.approx(expected)


def test_lift_frame_orders_tied_scores_by_input_order():
    order = _hand_stable_ranking()
    cumulative = np.cumsum([_TIED_Y[i] for i in order])
    base_rate = sum(_TIED_Y) / len(_TIED_Y)
    expected = [(c / (i + 1)) / base_rate for i, c in enumerate(cumulative)]

    frame = lift_frame(
        np.array(_TIED_Y),
        np.array(_TIED_SCORES, dtype=float).reshape(-1, 1),
        [1],
        ["1"],
    )
    got = frame.filter(pl.col("class") == "1")["lift"].to_list()
    assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5. PR macro/weighted-average recall sort
# ---------------------------------------------------------------------------

# The fourth stable-sort site: ``_pr_average`` sorts each per-class curve by
# recall before interpolating it onto the shared grid. A PR curve's recall
# stays *flat* across every threshold step that admits only false positives,
# so tied recall values are the norm, not an edge case -- and which precision
# a tied recall carries into ``np.interp`` is entirely the sort's choice.
#
# The fixture below makes that dense on purpose: 60 rows, three classes, and a
# score column per class that is a deterministic permutation of 60 distinct
# levels (23 is coprime to 60). Each class is negative on two rows in three,
# so each one-vs-rest curve spends long runs at a flat recall while precision
# falls -- 40 of its recall values repeat. Under the unstable sort the
# interpolated macro curve moves by up to 1.0 against the values below.
_PR_AVG_N = 60
_PR_AVG_CLASSES = ["0", "1", "2"]
_PR_AVG_Y = np.array([(i * 7) % 3 for i in range(_PR_AVG_N)])
_PR_AVG_SCORES = np.array(
    [
        [(((i * 23 + k * 13) % _PR_AVG_N) + 1) / (_PR_AVG_N + 1.0) for k in range(3)]
        for i in range(_PR_AVG_N)
    ]
)
#: Floor on repeated recall values per per-class curve. If the kernel ever
#: stops emitting flat-recall runs this fixture silently stops testing the
#: sort, so the test asserts the tie density it depends on.
_PR_AVG_MIN_TIES = 20


def test_pr_average_interpolates_over_a_stable_recall_sort():
    """The macro summary curve equals a hand stable-sorted interpolation.

    The per-class curves come from ``pr_frame(average=None)``, which never
    touches the sort under test, and the ordering reference is Python's
    guaranteed-stable ``sorted`` -- so the only thing being asserted is which
    precision each tied recall hands to ``np.interp``.
    """
    class_values = list(range(len(_PR_AVG_CLASSES)))
    per_class = pr_frame(_PR_AVG_Y, _PR_AVG_SCORES, class_values, _PR_AVG_CLASSES, average=None)
    grid = np.linspace(0.0, 1.0, _AVG_GRID_POINTS)

    curves: list[np.ndarray] = []
    for label in _PR_AVG_CLASSES:
        curve = per_class.filter(pl.col("class") == label)
        recall = curve["recall"].to_list()
        precision = curve["precision"].to_list()
        assert len(recall) - len(set(recall)) >= _PR_AVG_MIN_TIES
        order = sorted(range(len(recall)), key=lambda i: recall[i])  # sorted() is stable
        curves.append(np.interp(grid, [recall[i] for i in order], [precision[i] for i in order]))
    # macro weights are uniform, so the summary is the plain mean per grid point.
    expected = [sum(curve[j] for curve in curves) / len(curves) for j in range(len(grid))]

    macro = pr_frame(_PR_AVG_Y, _PR_AVG_SCORES, class_values, _PR_AVG_CLASSES, average="macro")
    assert macro["class"].unique().to_list() == ["macro"]
    assert macro["recall"].to_list() == pytest.approx(list(grid))
    assert macro["precision"].to_list() == pytest.approx(expected)
