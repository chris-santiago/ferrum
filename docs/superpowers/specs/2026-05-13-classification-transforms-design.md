# Classification Curve Transforms — Rust Pipeline Integration

**Date:** 2026-05-13
**Status:** Proposed

## Problem

Classification diagnostic curves (ROC, PR, calibration, gain, lift, discrimination threshold, confusion matrix) are computed in Python numpy inside `_classification.py`, then handed to Rust for rendering. This is the only remaining diagnostic subsystem where the stat computation happens outside the Rust pipeline. Every other stat mark (KDE, LOESS, histogram, contour, boxplot, violin, etc.) runs its transform in Rust during the render pass.

The Python implementations depend on sklearn's `roc_curve`, `precision_recall_curve`, `calibration_curve`, plus manual numpy operations for gain/lift/threshold sweep/confusion. At 500K+ rows, this means a full copy from Arrow to numpy, Python-side sort+cumsum, then Arrow back to Rust — an unnecessary round-trip.

## Decision

Add 7 new `TransformSpec` variants to the Rust pipeline so classification curves are computed inline during the render pass, identical to how every other stat transform works.

## What moves to Rust

| Transform | Input columns | Output columns | Algorithm |
|---|---|---|---|
| `RocCurve` | `y_true: Float64`, `y_score: Float64` | `fpr`, `tpr`, `threshold`, `auc` (all Float64) | Sort by y_score descending, walk to compute FPR/TPR at each threshold, trapezoidal AUC |
| `PrCurve` | `y_true: Float64`, `y_score: Float64` | `precision`, `recall`, `threshold` (all Float64) | Sort by y_score descending, walk to compute precision/recall at each threshold |
| `CalibrationCurve` | `y_true: Float64`, `y_score: Float64` | `mean_predicted`, `fraction_positive`, `count` (Float64, Float64, Int64) | Bin y_score into n_bins, compute per-bin mean predicted and fraction of positives |
| `CumulativeGain` | `y_true: Float64`, `y_score: Float64` | `percent_population`, `gain` (all Float64) | Sort by y_score descending, cumsum of positives / total positives |
| `LiftCurve` | `y_true: Float64`, `y_score: Float64` | `percent_population`, `lift` (all Float64) | Gain / baseline rate |
| `DiscriminationThreshold` | `y_true: Float64`, `y_score: Float64` | `threshold`, `precision`, `recall`, `f1`, `queue_rate` (all Float64) | Sweep thresholds, compute metrics at each |
| `ConfusionMatrix` | `y_true: Utf8`, `y_pred: Utf8` | `actual`, `predicted` (Utf8), `value` (Float64), `value_fmt` (Utf8) | Cross-tabulate, optionally normalize |

## What stays in Python

| Operation | Why |
|---|---|
| Multi-class one-vs-rest binarization | The Python layer expands one multiclass problem into multiple binary problems by iterating over `classes` from `model.classes_`. Each binary sub-problem is then a standard `(y_true_binary, y_score)` pair that the Rust transform handles. |
| Multi-class average curves (micro/macro/weighted) | These aggregate across per-class curves using interpolation and class-weight averaging. The per-class curves are Rust transforms; the averaging is Python orchestration. |
| Cross-validated threshold sweep | `_discrimination_threshold_cv` clones and refits the sklearn model per fold. The model fitting is inherently sklearn; the per-fold threshold sweep can use the Rust transform. |
| `ModelSource.probabilities()` | Calls `model.predict_proba()` / `model.decision_function()` — sklearn API boundary. |

## Rust implementation

### New transform modules

Each transform gets its own file following the existing pattern (e.g. `bin.rs`, `kde.rs`):

```
crates/ferrum-core/src/transform/
├── roc_curve.rs
├── pr_curve.rs
├── calibration.rs
├── gain_lift.rs          # CumulativeGain + LiftCurve share a module (lift = gain / baseline)
├── threshold_sweep.rs    # DiscriminationThreshold
└── confusion.rs
```

### TransformSpec variants

Add to `core.rs` enum and macro table:

```rust
// In TransformSpec enum:
RocCurve(RocCurveSpec),
PrCurve(PrCurveSpec),
Calibration(CalibrationSpec),
CumulativeGain(CumulativeGainSpec),
LiftCurve(LiftCurveSpec),
DiscriminationThreshold(DiscriminationThresholdSpec),
ConfusionMatrix(ConfusionMatrixSpec),

// In for_each_transform! macro:
RocCurve              => roc_curve            : PyRocCurve,
PrCurve               => pr_curve             : PyPrCurve,
Calibration           => calibration          : PyCalibration,
CumulativeGain        => gain_lift            : PyCumulativeGain,
LiftCurve             => gain_lift            : PyLiftCurve,
DiscriminationThreshold => threshold_sweep    : PyDiscriminationThreshold,
ConfusionMatrix       => confusion            : PyConfusionMatrix,
```

### Spec structs

```rust
// roc_curve.rs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct RocCurveSpec {
    pub y_true: String,        // column name
    pub y_score: String,       // column name
    #[serde(default = "default_true")]
    pub drop_intermediate: bool,
}

// pr_curve.rs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct PrCurveSpec {
    pub y_true: String,
    pub y_score: String,
}

// calibration.rs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct CalibrationSpec {
    pub y_true: String,
    pub y_score: String,
    #[serde(default = "default_10")]
    pub n_bins: usize,
    #[serde(default = "default_uniform")]
    pub strategy: String,      // "uniform" or "quantile"
}

// gain_lift.rs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct CumulativeGainSpec {
    pub y_true: String,
    pub y_score: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct LiftCurveSpec {
    pub y_true: String,
    pub y_score: String,
}

// threshold_sweep.rs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct DiscriminationThresholdSpec {
    pub y_true: String,
    pub y_score: String,
    #[serde(default = "default_50")]
    pub n_thresholds: usize,
}

// confusion.rs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ConfusionMatrixSpec {
    pub y_true: String,
    pub y_pred: String,
    pub normalize: Option<String>, // None, "true", "pred", "all"
    pub labels: Option<Vec<String>>,
}
```

### Algorithm implementations

All are sort-and-scan or binning operations — no linear algebra, no external deps beyond Arrow.

**ROC curve** (`roc_curve.rs`):
1. Extract `y_true` (binary 0/1) and `y_score` columns.
2. Sort by `y_score` descending (stable sort).
3. Walk sorted array: at each distinct threshold, compute `FP / (FP + TN)` (FPR) and `TP / (TP + FN)` (TPR).
4. If `drop_intermediate`: remove points where TPR doesn't change.
5. AUC via trapezoidal rule on the emitted `(fpr, tpr)` pairs.
6. Emit RecordBatch `{fpr, tpr, threshold, auc}`.

**PR curve** (`pr_curve.rs`):
1. Sort by `y_score` descending.
2. Walk: at each threshold, precision = TP / (TP + FP), recall = TP / (TP + FN).
3. Emit RecordBatch `{precision, recall, threshold}`.

**Calibration curve** (`calibration.rs`):
1. Compute bin edges: uniform → `linspace(0, 1, n_bins+1)`, quantile → quantiles of `y_score`.
2. Assign each sample to a bin.
3. Per bin: `mean_predicted` = mean of `y_score`, `fraction_positive` = mean of `y_true`, `count` = number of samples.
4. Drop empty bins.
5. Emit RecordBatch `{mean_predicted, fraction_positive, count}`.

**Cumulative gain** (`gain_lift.rs`):
1. Sort by `y_score` descending.
2. `cum_pos = cumsum(y_true[sorted])`.
3. `gain = cum_pos / total_positives`.
4. `percent_population = [1..n] / n`.
5. Prepend `(0, 0)`.
6. Emit RecordBatch `{percent_population, gain}`.

**Lift curve** (`gain_lift.rs`):
1. Compute gain curve as above.
2. `lift = (cum_pos / sample_count) / base_rate` where `base_rate = total_positives / n`.
3. Emit RecordBatch `{percent_population, lift}`.

**Discrimination threshold** (`threshold_sweep.rs`):
1. Generate `n_thresholds` evenly spaced in `[0, 1]`.
2. Sort `y_score`.
3. For each threshold t: binary-search to find split point, compute TP/FP/TN/FN from the split, derive precision/recall/F1/queue_rate.
4. Emit RecordBatch `{threshold, precision, recall, f1, queue_rate}`.

**Confusion matrix** (`confusion.rs`):
1. Extract `y_true` and `y_pred` as string arrays.
2. Determine label set (from `labels` if provided, else union of unique values).
3. Cross-tabulate into n_labels × n_labels counts.
4. Optionally normalize by row (`"true"`), column (`"pred"`), or total (`"all"`).
5. Emit long-form RecordBatch `{actual, predicted, value, value_fmt}`.

## Python rewiring

### `_classification.py` data source methods

Each method changes from "compute the curve in Python" to "prepare the raw data columns and let the transform handle it."

**`roc_curve()` → emit transform spec:**

Before:
```python
fpr, tpr, thr = sklearn.metrics.roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)
# ... build rows dict
```

After (binary case):
```python
# Prepare a DataFrame with y_true (binary) and y_score columns.
# The Rust RocCurve transform computes fpr/tpr/threshold/auc.
df = pl.DataFrame({"y_true": y_bin, "y_score": y_score_series})
# Apply transform via the chart pipeline (or call _core directly for the source method).
```

For the `ModelSource.roc_curve()` method specifically, which returns a DataFrame (not a chart), we call the Rust transform as a standalone function — same pattern as the existing `hat_matrix_stats`:

```python
from ferrum._core import PyRocCurve
spec = PyRocCurve(y_true="y_true", y_score="y_score", drop_intermediate=True)
result_batch = spec.apply(input_batch)  # Arrow in, Arrow out
df = pl.from_arrow(result_batch)
```

The multi-class path loops over classes in Python (binarizing y_true for each class), calls the Rust transform per class, and concatenates the per-class DataFrames.

**Same pattern for:** `pr_curve()`, `calibration_curve()`, `cumulative_gain()`, `lift_curve()`, `discrimination_threshold()`, `confusion_matrix()`.

### Mark desugaring

`desugar_roc`, `desugar_pr`, etc. in `marks/diagnostic.py` currently expect precomputed columns (`fpr`, `tpr`). With transforms in the pipeline, two modes:

1. **Source method path** (existing: `Chart(src.roc_curve()).mark_roc()`): Data already has `fpr`/`tpr` columns. No transform needed. Desugarer works as-is.

2. **Raw data path** (new: `Chart(raw_df).mark_roc(y_true="label", y_score="prob")`): Desugarer emits a `RocCurve` transform spec that the renderer applies before drawing. The mark builder attaches the transform to the chart spec's transform list.

This dual mode is how `mark_smooth` already works — it can accept precomputed smooth data or apply the LOESS transform inline.

### Figure functions

`roc_chart()`, `pr_chart()`, `calibration_chart()`, `gain_chart()`, `lift_chart()`, etc. in `figures.py` and `charts.py` currently call `source.roc_curve()` to get precomputed data. After the change, they can either:

- Continue calling `source.roc_curve()` (which now internally uses the Rust transform) — no change needed at this level.
- Or construct a chart with raw data + transform spec for the "no ModelSource" path.

The figure functions don't need to change unless we want to add raw-data overloads (e.g. `fm.roc_chart(y_true, y_score)` without a fitted model). That's a nice-to-have but not required for this spec.

## Verification plan

### Parity tests

Each Rust transform must match sklearn within tolerance:

| Transform | Reference | Tolerance | Test cases |
|---|---|---|---|
| `RocCurve` | `sklearn.metrics.roc_curve` + `roc_auc_score` | 1e-12 for fpr/tpr, 1e-10 for AUC | n in {100, 1000, 10000}, balanced + imbalanced |
| `PrCurve` | `sklearn.metrics.precision_recall_curve` | 1e-12 | same sweep |
| `CalibrationCurve` | `sklearn.calibration.calibration_curve` | 1e-10 | n_bins in {5, 10, 20}, strategy in {uniform, quantile} |
| `CumulativeGain` | manual numpy reference (no sklearn equivalent) | 1e-12 | binary + multiclass |
| `LiftCurve` | manual numpy reference | 1e-12 | same |
| `DiscriminationThreshold` | manual numpy reference | 1e-10 | n_thresholds in {10, 50, 100} |
| `ConfusionMatrix` | `sklearn.metrics.confusion_matrix` | exact (integer counts) | 2-class, 3-class, with normalization |

### Integration tests

- Full test suite passes (`uv run pytest -x -q`).
- Existing diagnostic chart goldens remain byte-identical (the transforms produce the same data as the Python implementations did).
- New raw-data path: `Chart(df).mark_roc(y_true="label", y_score="prob")` renders correctly.

### Rust unit tests

- ROC of perfect classifier: AUC = 1.0, FPR = [0, 0, 1], TPR = [0, 1, 1].
- ROC of random classifier: AUC ≈ 0.5.
- PR of perfect classifier: all precisions = 1.0.
- Calibration with perfectly calibrated scores: `fraction_positive ≈ mean_predicted` per bin.
- Confusion matrix with perfect predictions: diagonal only.
- Gain curve starts at (0, 0) and ends at (1, 1).
- Lift curve at 100% population = 1.0.

## Risk

**Low.** ROC/PR/gain/lift are sort+cumsum; calibration is binning; confusion is cross-tabulation; threshold sweep is binary search. All are well-understood O(n log n) or O(n) algorithms. The only subtlety is matching sklearn's exact conventions for edge cases (e.g. the extra threshold point sklearn appends, the `drop_intermediate` deduplication, the precision-recall convention at recall=0).

## Non-goal

- Multi-class curve averaging (micro/macro/weighted) stays in Python. These are orchestration over per-class results, not per-sample transforms.
- Cross-validated threshold sweep stays in Python (model fitting is sklearn).
- `ModelSource.probabilities()` stays in Python (sklearn API boundary).
