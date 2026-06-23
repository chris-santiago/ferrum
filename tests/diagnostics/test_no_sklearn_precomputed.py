"""Dependency-isolation test: precomputed curve path must never import sklearn.

Following the pattern in test_no_sklearn_at_import.py: drop sklearn from
sys.modules, then exercise all five precomputed curve methods end-to-end and
assert no re-import occurs.

sklearn is NOT imported at module level here — doing so would break the
guarantee we are asserting.  All reference computations use numpy only.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Drop-sklearn helper (mirrors test_no_sklearn_at_import.py)
# ---------------------------------------------------------------------------


def _drop_sklearn() -> None:
    for mod in list(sys.modules):
        if mod == "sklearn" or mod.startswith("sklearn."):
            del sys.modules[mod]


def _assert_sklearn_absent(label: str) -> None:
    offenders = [m for m in sys.modules if m == "sklearn" or m.startswith("sklearn.")]
    assert not offenders, (
        f"sklearn was imported as a side-effect of {label}. Offending modules: {offenders}"
    )


# ---------------------------------------------------------------------------
# Inline test arrays (no sklearn)
# ---------------------------------------------------------------------------


def _binary_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """y_true, y_score (soft), y_hard — deterministic binary data."""
    rng = np.random.default_rng(0)
    n = 60
    y_true = rng.integers(0, 2, size=n)
    y_score = np.clip(rng.standard_normal(n) * 0.4 + y_true * 0.8, 0.0, 1.0)
    # Hard labels: threshold at 0.5
    y_hard = (y_score >= 0.5).astype(int)
    return y_true, y_score, y_hard


def _multiclass_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """y_true, y_prob 2-D (soft), y_hard — 3-class deterministic data."""
    rng = np.random.default_rng(99)
    n = 75
    y_true = rng.integers(0, 3, size=n)
    logits = rng.standard_normal((n, 3))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    y_prob = exp / exp.sum(axis=1, keepdims=True)
    y_hard = y_prob.argmax(axis=1)
    return y_true, y_prob, y_hard


# ---------------------------------------------------------------------------
# roc_curve
# ---------------------------------------------------------------------------


def test_roc_curve_precomputed_binary_no_sklearn():
    _drop_sklearn()
    from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

    y_true, y_score, _ = _binary_arrays()
    src = _PrecomputedSource(y_true, y_score)
    df = src.roc_curve()

    _assert_sklearn_absent("_PrecomputedSource.roc_curve() [binary]")
    assert "fpr" in df.columns
    assert "tpr" in df.columns
    assert "threshold" in df.columns
    assert len(df) > 2


def test_roc_curve_precomputed_multiclass_no_sklearn():
    _drop_sklearn()
    from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

    y_true, y_prob, _ = _multiclass_arrays()
    src = _PrecomputedSource(y_true, y_prob)
    df = src.roc_curve()

    _assert_sklearn_absent("_PrecomputedSource.roc_curve() [multiclass]")
    classes = sorted(str(c) for c in np.unique(y_true).tolist())
    assert set(df["class"].unique().to_list()) == set(classes)


# ---------------------------------------------------------------------------
# pr_curve
# ---------------------------------------------------------------------------


def test_pr_curve_precomputed_binary_no_sklearn():
    _drop_sklearn()
    from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

    y_true, y_score, _ = _binary_arrays()
    src = _PrecomputedSource(y_true, y_score)
    df = src.pr_curve()

    _assert_sklearn_absent("_PrecomputedSource.pr_curve() [binary]")
    assert "precision" in df.columns
    assert "recall" in df.columns
    assert len(df) > 2


def test_pr_curve_precomputed_multiclass_no_sklearn():
    _drop_sklearn()
    from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

    y_true, y_prob, _ = _multiclass_arrays()
    src = _PrecomputedSource(y_true, y_prob)
    df = src.pr_curve()

    _assert_sklearn_absent("_PrecomputedSource.pr_curve() [multiclass]")
    classes = sorted(str(c) for c in np.unique(y_true).tolist())
    assert set(df["class"].unique().to_list()) == set(classes)


# ---------------------------------------------------------------------------
# calibration_curve
# ---------------------------------------------------------------------------


def test_calibration_curve_precomputed_binary_no_sklearn():
    _drop_sklearn()
    from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

    y_true, y_score, _ = _binary_arrays()
    src = _PrecomputedSource(y_true, y_score)
    df = src.calibration_curve()

    _assert_sklearn_absent("_PrecomputedSource.calibration_curve() [binary]")
    assert "mean_predicted" in df.columns
    assert "fraction_positive" in df.columns
    # At least one bin must be non-empty
    assert len(df) >= 1


def test_calibration_curve_precomputed_quantile_no_sklearn():
    _drop_sklearn()
    from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

    y_true, y_score, _ = _binary_arrays()
    src = _PrecomputedSource(y_true, y_score)
    df = src.calibration_curve(n_bins=5, strategy="quantile")

    _assert_sklearn_absent("_PrecomputedSource.calibration_curve(strategy='quantile')")
    # Quantile strategy may merge duplicate bin-edge values, so result can have
    # fewer than n_bins rows.  Assert at least one non-empty bin is returned.
    assert len(df) >= 1
    assert len(df) <= 5


# ---------------------------------------------------------------------------
# confusion_matrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_precomputed_binary_no_sklearn():
    _drop_sklearn()
    from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

    y_true, _, y_hard = _binary_arrays()
    src = _PrecomputedSource(y_true, y_hard)
    df = src.confusion_matrix()

    _assert_sklearn_absent("_PrecomputedSource.confusion_matrix() [binary]")
    assert "actual" in df.columns
    assert "predicted" in df.columns
    assert "value" in df.columns
    assert "value_fmt" in df.columns
    # 2x2 matrix = 4 cells
    assert len(df) == 4


def test_confusion_matrix_precomputed_multiclass_no_sklearn():
    _drop_sklearn()
    from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

    y_true, _, y_hard = _multiclass_arrays()
    src = _PrecomputedSource(y_true, y_hard)
    df = src.confusion_matrix()

    _assert_sklearn_absent("_PrecomputedSource.confusion_matrix() [multiclass]")
    n_classes = len(np.unique(y_true))
    # 3x3 matrix = 9 cells
    assert len(df) == n_classes * n_classes


# ---------------------------------------------------------------------------
# discrimination_threshold (cv=None)
# ---------------------------------------------------------------------------


def test_discrimination_threshold_precomputed_binary_no_sklearn():
    _drop_sklearn()
    from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

    y_true, y_score, _ = _binary_arrays()
    src = _PrecomputedSource(y_true, y_score)
    df = src.discrimination_threshold(n_thresholds=20)

    _assert_sklearn_absent("_PrecomputedSource.discrimination_threshold() [binary]")
    assert "threshold" in df.columns
    assert "precision" in df.columns
    assert "recall" in df.columns
    assert "f1" in df.columns
    assert "queue_rate" in df.columns
    assert len(df) == 20


def test_discrimination_threshold_cv_none_no_sklearn():
    """Explicit cv=None must not trigger sklearn import."""
    _drop_sklearn()
    from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

    y_true, y_score, _ = _binary_arrays()
    src = _PrecomputedSource(y_true, y_score)
    df = src.discrimination_threshold(cv=None, n_thresholds=10)

    _assert_sklearn_absent("_PrecomputedSource.discrimination_threshold(cv=None)")
    assert len(df) == 10


def test_discrimination_threshold_cv_raises():
    """cv=<int> must raise ValueError, not import sklearn."""
    _drop_sklearn()
    from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

    y_true, y_score, _ = _binary_arrays()
    src = _PrecomputedSource(y_true, y_score)
    with pytest.raises(ValueError, match="cv="):
        src.discrimination_threshold(cv=3)
    # ValueError is raised before any sklearn import could happen
    _assert_sklearn_absent("_PrecomputedSource.discrimination_threshold(cv=3) [ValueError path]")
