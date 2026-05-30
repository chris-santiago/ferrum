"""Byte-parity tests: Rust diagnostic kernels vs scikit-learn.

Each kernel is verified against the corresponding sklearn function under the
conventions documented in the design spec (§7):

  - roc_curve_kernel    vs sklearn.metrics.roc_curve
  - roc_auc             vs sklearn.metrics.roc_auc_score
  - pr_curve_kernel     vs sklearn.metrics.precision_recall_curve
  - average_precision   vs sklearn.metrics.average_precision_score
  - calibration_kernel  vs sklearn.calibration.calibration_curve
  - confusion_kernel    vs sklearn.metrics.confusion_matrix
  - prf_at_thresholds   vs per-threshold precision_recall_fscore_support

A second section asserts model-path vs precomputed-path equivalence (spec §10).
"""

from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from sklearn.calibration import calibration_curve as skl_calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

import ferrum
from ferrum._core import (
    average_precision,
    calibration_kernel,
    confusion_kernel,
    pr_curve_kernel,
    prf_at_thresholds,
    roc_auc,
    roc_curve_kernel,
)
from ferrum._diagnostics.precomputed import _PrecomputedSource
from tests.fixtures import load_dataset, load_fixture

_ATOL = 1e-9
_RTOL = 1e-9


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_arrays():
    """Small, deterministic binary classification arrays."""
    rng = np.random.default_rng(42)
    n = 80
    y_true = rng.integers(0, 2, size=n).astype(np.float64)
    y_score = np.clip(rng.standard_normal(n) * 0.4 + y_true * 0.8, 0.0, 1.0)
    return y_true, y_score


@pytest.fixture(scope="module")
def multiclass_arrays():
    """Small 3-class OvR arrays."""
    rng = np.random.default_rng(7)
    n = 90
    y_true = rng.integers(0, 3, size=n)  # int labels
    # (n, 3) probability matrix — softmax-shaped
    logits = rng.standard_normal((n, 3))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    y_prob = exp / exp.sum(axis=1, keepdims=True)
    return y_true, y_prob


@pytest.fixture(scope="module")
def model_binary():
    """Pinned fitted binary LogisticRegression + data."""
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]
    return model, X, y


# ---------------------------------------------------------------------------
# Arrow conversion helpers
# ---------------------------------------------------------------------------


def _f64(arr: np.ndarray) -> pa.Array:
    return pa.array(arr.astype(np.float64), type=pa.float64())


def _i64(arr: np.ndarray) -> pa.Array:
    return pa.array(arr.astype(np.int64), type=pa.int64())


# ---------------------------------------------------------------------------
# 1. roc_curve_kernel — binary, drop_intermediate on/off, degenerate cases
# ---------------------------------------------------------------------------


class TestRocCurveKernel:
    def _check(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        drop_intermediate: bool,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skl_fpr, skl_tpr, skl_thr = roc_curve(
                y_true.astype(int), y_score, drop_intermediate=drop_intermediate
            )
        rb = roc_curve_kernel(_f64(y_true), _f64(y_score), drop_intermediate)
        df = cast("pl.DataFrame", pl.from_arrow(rb))

        np.testing.assert_allclose(df["fpr"].to_numpy(), skl_fpr, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(df["tpr"].to_numpy(), skl_tpr, rtol=_RTOL, atol=_ATOL)
        # sklearn leading threshold is +inf; Rust must match
        np.testing.assert_allclose(df["threshold"].to_numpy(), skl_thr, rtol=_RTOL, atol=_ATOL)

    def test_binary_drop_intermediate_true(self, binary_arrays):
        y_true, y_score = binary_arrays
        self._check(y_true, y_score, drop_intermediate=True)

    def test_binary_drop_intermediate_false(self, binary_arrays):
        y_true, y_score = binary_arrays
        self._check(y_true, y_score, drop_intermediate=False)

    def test_multiclass_ovr(self, multiclass_arrays):
        y_true, y_prob = multiclass_arrays
        classes = np.unique(y_true)
        for i, cls in enumerate(classes):
            y_bin = (y_true == cls).astype(float)
            self._check(y_bin, y_prob[:, i], drop_intermediate=True)

    def test_all_correct(self):
        """All predictions on the correct side of the boundary."""
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=float)
        self._check(y_true, y_score, drop_intermediate=True)

    def test_all_positive_labels(self):
        """Single class present — sklearn emits NaN FPR; kernel must match exactly."""
        y_true = np.array([1, 1, 1], dtype=float)
        y_score = np.array([0.1, 0.5, 0.9], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skl_fpr, skl_tpr, skl_thr = roc_curve(y_true.astype(int), y_score)
        rb = roc_curve_kernel(_f64(y_true), _f64(y_score), True)
        df = cast("pl.DataFrame", pl.from_arrow(rb))
        np.testing.assert_allclose(
            df["fpr"].to_numpy(), skl_fpr, equal_nan=True, rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(
            df["tpr"].to_numpy(), skl_tpr, equal_nan=True, rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(
            df["threshold"].to_numpy(), skl_thr, equal_nan=True, rtol=1e-9, atol=1e-12
        )

    def test_all_negative_labels(self):
        """Single negative class — sklearn emits NaN for all TPR values; kernel must match exactly."""
        y_true = np.array([0, 0, 0], dtype=float)
        y_score = np.array([0.1, 0.5, 0.9], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skl_fpr, skl_tpr, skl_thr = roc_curve(y_true.astype(int), y_score)
        rb = roc_curve_kernel(_f64(y_true), _f64(y_score), True)
        df = cast("pl.DataFrame", pl.from_arrow(rb))
        np.testing.assert_allclose(
            df["fpr"].to_numpy(), skl_fpr, equal_nan=True, rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(
            df["tpr"].to_numpy(), skl_tpr, equal_nan=True, rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(
            df["threshold"].to_numpy(), skl_thr, equal_nan=True, rtol=1e-9, atol=1e-12
        )

    def test_tie_heavy_scores(self):
        """Many identical scores; Rust tie aggregation must match sklearn."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        y_score = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7], dtype=float
        )
        self._check(y_true, y_score, drop_intermediate=True)
        self._check(y_true, y_score, drop_intermediate=False)


# ---------------------------------------------------------------------------
# 2. roc_auc
# ---------------------------------------------------------------------------


class TestRocAuc:
    def _check(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skl_val = roc_auc_score(y_true.astype(int), y_score)
        rust_val = float(roc_auc(_f64(y_true), _f64(y_score)))
        np.testing.assert_allclose(rust_val, skl_val, rtol=_RTOL, atol=_ATOL)

    def test_binary(self, binary_arrays):
        y_true, y_score = binary_arrays
        self._check(y_true, y_score)

    def test_multiclass_ovr(self, multiclass_arrays):
        y_true, y_prob = multiclass_arrays
        classes = np.unique(y_true)
        for i, cls in enumerate(classes):
            y_bin = (y_true == cls).astype(float)
            self._check(y_bin, y_prob[:, i])

    def test_all_correct(self):
        y_true = np.array([0, 0, 1, 1], dtype=float)
        y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
        self._check(y_true, y_score)

    def test_tie_heavy(self):
        y_true = np.array([0, 1, 0, 1, 0, 1], dtype=float)
        y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
        self._check(y_true, y_score)


# ---------------------------------------------------------------------------
# 3. pr_curve_kernel
# ---------------------------------------------------------------------------


class TestPrCurveKernel:
    def _check(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skl_prec, skl_rec, skl_thr = precision_recall_curve(y_true.astype(int), y_score)
        rb = pr_curve_kernel(_f64(y_true), _f64(y_score))
        df = pl.from_arrow(rb)

        # precision and recall lengths must match sklearn (includes trailing sentinel)
        assert len(df) == len(skl_prec)
        np.testing.assert_allclose(df["precision"].to_numpy(), skl_prec, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(df["recall"].to_numpy(), skl_rec, rtol=_RTOL, atol=_ATOL)
        # kernel threshold array is one longer than sklearn's (NaN-padded final point)
        rust_thr = df["threshold"].to_numpy()
        assert len(rust_thr) == len(skl_prec), "threshold length must equal precision length"
        np.testing.assert_allclose(rust_thr[:-1], skl_thr, rtol=_RTOL, atol=_ATOL)
        assert np.isnan(rust_thr[-1]), "last threshold slot must be NaN"

    def test_binary(self, binary_arrays):
        y_true, y_score = binary_arrays
        self._check(y_true, y_score)

    def test_multiclass_ovr(self, multiclass_arrays):
        y_true, y_prob = multiclass_arrays
        classes = np.unique(y_true)
        for i, cls in enumerate(classes):
            y_bin = (y_true == cls).astype(float)
            self._check(y_bin, y_prob[:, i])

    def test_all_positive(self):
        """All true positives — sklearn fills precision=1 for all recall points."""
        y_true = np.ones(6, dtype=float)
        y_score = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._check(y_true, y_score)

    def test_all_negative_labels(self):
        """No positive class — sklearn warning; kernel must match output."""
        y_true = np.zeros(6, dtype=float)
        y_score = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._check(y_true, y_score)

    def test_tie_heavy_scores(self):
        y_true = np.array([0, 1, 0, 1, 0, 1], dtype=float)
        y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._check(y_true, y_score)


# ---------------------------------------------------------------------------
# 4. average_precision
# ---------------------------------------------------------------------------


class TestAveragePrecision:
    def _check(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skl_val = average_precision_score(y_true.astype(int), y_score)
        rust_val = float(average_precision(_f64(y_true), _f64(y_score)))
        np.testing.assert_allclose(rust_val, skl_val, rtol=_RTOL, atol=_ATOL)

    def test_binary(self, binary_arrays):
        y_true, y_score = binary_arrays
        self._check(y_true, y_score)

    def test_multiclass_ovr(self, multiclass_arrays):
        y_true, y_prob = multiclass_arrays
        classes = np.unique(y_true)
        for i, cls in enumerate(classes):
            y_bin = (y_true == cls).astype(float)
            self._check(y_bin, y_prob[:, i])

    def test_all_correct(self):
        y_true = np.array([0, 0, 1, 1], dtype=float)
        y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
        self._check(y_true, y_score)

    def test_all_negative_returns_zero(self):
        """sklearn returns 0.0 for all-negative y_true; kernel must match."""
        y_true = np.zeros(5, dtype=float)
        y_score = np.array([0.1, 0.5, 0.9, 0.3, 0.7], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._check(y_true, y_score)


# ---------------------------------------------------------------------------
# 5. calibration_kernel
# ---------------------------------------------------------------------------


class TestCalibrationKernel:
    def _check(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int,
        strategy: str,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skl_frac, skl_mpred = skl_calibration_curve(
                y_true.astype(int), y_prob, n_bins=n_bins, strategy=strategy
            )
        rb = calibration_kernel(_f64(y_true), _f64(y_prob), n_bins, strategy)
        df = pl.from_arrow(rb)

        # Both drop empty bins; lengths must agree
        assert len(df) == len(skl_frac), (
            f"bin count mismatch: rust={len(df)}, sklearn={len(skl_frac)} "
            f"(strategy={strategy!r}, n_bins={n_bins})"
        )
        np.testing.assert_allclose(
            df["fraction_positive"].to_numpy(), skl_frac, rtol=_RTOL, atol=_ATOL
        )
        np.testing.assert_allclose(
            df["mean_predicted"].to_numpy(), skl_mpred, rtol=_RTOL, atol=_ATOL
        )

    def test_uniform_binary(self, binary_arrays):
        y_true, y_score = binary_arrays
        self._check(y_true, y_score, n_bins=10, strategy="uniform")

    def test_quantile_binary(self, binary_arrays):
        y_true, y_score = binary_arrays
        self._check(y_true, y_score, n_bins=10, strategy="quantile")

    def test_uniform_5_bins(self, binary_arrays):
        y_true, y_score = binary_arrays
        self._check(y_true, y_score, n_bins=5, strategy="uniform")

    def test_quantile_5_bins(self, binary_arrays):
        y_true, y_score = binary_arrays
        self._check(y_true, y_score, n_bins=5, strategy="quantile")

    def test_empty_bins_dropped(self):
        """Sparse data with many bins — empty bins must be dropped on both sides."""
        y_true = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.1, 0.9, 0.15, 0.85, 0.2])
        self._check(y_true, y_prob, n_bins=10, strategy="uniform")

    def test_multiclass_ovr(self, multiclass_arrays):
        """Calibration is binary; verify per-class OvR."""
        y_true, y_prob = multiclass_arrays
        classes = np.unique(y_true)
        for i, cls in enumerate(classes):
            y_bin = (y_true == cls).astype(float)
            self._check(y_bin, y_prob[:, i], n_bins=5, strategy="uniform")


# ---------------------------------------------------------------------------
# 6. confusion_kernel
# ---------------------------------------------------------------------------


class TestConfusionKernel:
    def _reconstruct(self, df: pl.DataFrame, n_labels: int) -> np.ndarray:
        m = np.zeros((n_labels, n_labels))
        for row_dict in df.to_dicts():
            m[row_dict["row"], row_dict["col"]] = row_dict["value"]
        return m

    def _check(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: np.ndarray,
        normalize: str | None,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skl_cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
        n = len(labels)
        label_to_code = {lbl: i for i, lbl in enumerate(labels.tolist())}
        yt_codes = np.array([label_to_code[v] for v in y_true.tolist()], dtype=np.int64)
        yp_codes = np.array([label_to_code[v] for v in y_pred.tolist()], dtype=np.int64)
        norm_str = normalize if normalize is not None else ""
        rb = confusion_kernel(
            _i64(yt_codes),
            _i64(yp_codes),
            pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
            norm_str,
        )
        rust_cm = self._reconstruct(pl.from_arrow(rb), n)
        np.testing.assert_allclose(rust_cm, skl_cm, rtol=_RTOL, atol=_ATOL)

    def test_binary_no_normalize(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        labels = np.array([0, 1])
        self._check(y_true, y_pred, labels, normalize=None)

    def test_binary_normalize_true(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        labels = np.array([0, 1])
        self._check(y_true, y_pred, labels, normalize="true")

    def test_binary_normalize_pred(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        labels = np.array([0, 1])
        self._check(y_true, y_pred, labels, normalize="pred")

    def test_binary_normalize_all(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        labels = np.array([0, 1])
        self._check(y_true, y_pred, labels, normalize="all")

    def test_multiclass_no_normalize(self):
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2, 0, 0, 1])
        labels = np.array([0, 1, 2])
        self._check(y_true, y_pred, labels, normalize=None)

    def test_multiclass_normalize_true(self):
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2, 0, 0, 1])
        labels = np.array([0, 1, 2])
        self._check(y_true, y_pred, labels, normalize="true")

    def test_all_correct(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        labels = np.array([0, 1, 2])
        self._check(y_true, y_pred, labels, normalize=None)


# ---------------------------------------------------------------------------
# 7. prf_at_thresholds
# ---------------------------------------------------------------------------


class TestPrfAtThresholds:
    def _check(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        thresholds: np.ndarray,
    ) -> None:
        rb = prf_at_thresholds(_f64(y_true), _f64(y_score), _f64(thresholds))
        df = pl.from_arrow(rb)

        for i, t in enumerate(thresholds):
            y_pred_bin = (y_score >= t).astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skl_p, skl_r, skl_f, _ = precision_recall_fscore_support(
                    y_true.astype(int), y_pred_bin, average="binary", zero_division=0
                )
            np.testing.assert_allclose(df["precision"][i], skl_p, rtol=_RTOL, atol=_ATOL)
            np.testing.assert_allclose(df["recall"][i], skl_r, rtol=_RTOL, atol=_ATOL)
            np.testing.assert_allclose(df["f1"][i], skl_f, rtol=_RTOL, atol=_ATOL)
            # queue_rate = fraction of samples predicted positive
            expected_qr = float((y_score >= t).sum()) / len(y_score)
            np.testing.assert_allclose(df["queue_rate"][i], expected_qr, rtol=_RTOL, atol=_ATOL)

    def test_binary(self, binary_arrays):
        y_true, y_score = binary_arrays
        thresholds = np.linspace(0.1, 0.9, 9)
        self._check(y_true, y_score, thresholds)

    def test_single_threshold(self, binary_arrays):
        y_true, y_score = binary_arrays
        self._check(y_true, y_score, np.array([0.5]))

    def test_all_positive_above_threshold(self, binary_arrays):
        y_true, _ = binary_arrays
        y_score = np.ones_like(y_true) * 0.9
        self._check(y_true, y_score, np.array([0.5, 0.8, 0.95]))

    def test_threshold_zero_positives(self, binary_arrays):
        """All scores below threshold — zero_division=0 path."""
        y_true, y_score = binary_arrays
        self._check(y_true, y_score, np.array([1.1]))


# ---------------------------------------------------------------------------
# 8. Model-path vs precomputed-path equivalence (spec §10)
# ---------------------------------------------------------------------------


class TestModelVsPrecomputedEquivalence:
    """Single source-of-truth check: both paths must emit identical frames."""

    @pytest.fixture(scope="class")
    def sources(self, model_binary):
        model, X, y = model_binary
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_true = np.asarray(y)
            y_score_1d = model.predict_proba(X.to_pandas())[:, 1]
            y_hard = model.predict(X.to_pandas())
        model_source = ferrum.ModelSource(model, X, y)
        precomp_source = _PrecomputedSource(y_true, y_score_1d)
        precomp_hard = _PrecomputedSource(y_true, y_hard)
        return model_source, precomp_source, precomp_hard, y_true, y_score_1d

    def test_roc_curve_equivalence(self, sources):
        model_src, precomp_src, _, _, _ = sources
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_df = model_src.roc_curve()
        precomp_df = precomp_src.roc_curve()
        np.testing.assert_allclose(
            model_df["fpr"].to_numpy(), precomp_df["fpr"].to_numpy(), rtol=_RTOL, atol=_ATOL
        )
        np.testing.assert_allclose(
            model_df["tpr"].to_numpy(), precomp_df["tpr"].to_numpy(), rtol=_RTOL, atol=_ATOL
        )
        np.testing.assert_allclose(
            model_df["auc"].to_numpy(), precomp_df["auc"].to_numpy(), rtol=_RTOL, atol=_ATOL
        )

    def test_pr_curve_equivalence(self, sources):
        model_src, precomp_src, _, _, _ = sources
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_df = model_src.pr_curve()
        precomp_df = precomp_src.pr_curve()
        np.testing.assert_allclose(
            model_df["precision"].to_numpy(),
            precomp_df["precision"].to_numpy(),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            model_df["recall"].to_numpy(), precomp_df["recall"].to_numpy(), rtol=_RTOL, atol=_ATOL
        )
        np.testing.assert_allclose(
            model_df["ap"].to_numpy(), precomp_df["ap"].to_numpy(), rtol=_RTOL, atol=_ATOL
        )

    def test_calibration_curve_equivalence(self, sources):
        model_src, precomp_src, _, _, _ = sources
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_df = model_src.calibration_curve()
        precomp_df = precomp_src.calibration_curve()
        assert model_df.shape == precomp_df.shape
        np.testing.assert_allclose(
            model_df["mean_predicted"].to_numpy(),
            precomp_df["mean_predicted"].to_numpy(),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            model_df["fraction_positive"].to_numpy(),
            precomp_df["fraction_positive"].to_numpy(),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_confusion_matrix_equivalence(self, sources):
        model_src, _, precomp_hard, _, _ = sources
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_df = model_src.confusion_matrix()
        precomp_df = precomp_hard.confusion_matrix()
        assert model_df.shape == precomp_df.shape
        assert model_df["actual"].to_list() == precomp_df["actual"].to_list()
        assert model_df["predicted"].to_list() == precomp_df["predicted"].to_list()
        np.testing.assert_allclose(
            model_df["value"].to_numpy(), precomp_df["value"].to_numpy(), rtol=_RTOL, atol=_ATOL
        )
