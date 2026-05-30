# Diagnostic Curve Kernels in Rust — Design Spec

## 1. Scope

Move the five scikit-learn metric computations used by the classification
diagnostics — ROC curve, precision-recall curve, calibration curve, confusion
matrix, and the precision/recall/F1 threshold sweep — out of `sklearn.metrics`
and into Rust kernels in `crates/ferrum-core/src/diagnostics.rs`. The kernels
consume Arrow arrays and return Arrow `RecordBatch`es directly, eliminating the
`list[dict]` row-materialization glue in the Python source adapters. Both the
precomputed array path (`_PrecomputedSource`) and the model-backed path
(`ModelSource`) route their curve math through these kernels. After this change,
scikit-learn is required **only** when a user passes a fitted estimator;
computing any of these diagnostics from raw `(y_true, y_pred)` arrays needs no
scikit-learn at all.

## 2. Goals

- `roc_chart`, `pr_chart`, `calibration_chart`, `confusion_matrix_chart`, and
  `discrimination_threshold_chart` (precomputed inputs) run with scikit-learn
  **not installed**.
- One implementation of each curve's math, shared by the precomputed and
  model-backed paths.
- Curve adapters return Arrow tables built columnarly; no per-row Python `dict`
  construction in the ported functions.
- Byte-for-byte identical chart output to today — existing goldens and tests
  pass unchanged.
- Dependency extras make the boundary obvious: `scikit-learn` belongs to the
  model-backed feature set, not to core diagnostics on arrays.

## 3. Non-goals

- **Category A is untouched.** Anything that re-fits or re-scores the user's
  estimator stays on scikit-learn: `learning_curve`, `validation_curve`,
  `cv_scores`, `alpha_selection`, permutation importance, `partial_dependence`,
  `probabilities` (`predict_proba`/`decision_function`), and the `cv=` branch of
  `discrimination_threshold`. These are structurally bound to the foreign model
  object and gain nothing from a port.
- **Gain / lift / cumulative-gain math is not ported to Rust.** It already uses
  no scikit-learn. It receives only the columnar-Python cleanup described below.
- No change to any public chart-function signature, parameter, or output schema.
- No performance target. The motivation is dependency removal plus glue
  elimination, not beating scikit-learn's vectorized core.

## 4. System behavior

A user with only ferrum's base dependencies installed (no `models` extra) can
call:

```python
ferrum.roc_chart(y_true, y_pred)
ferrum.pr_chart(y_true, y_pred)
ferrum.calibration_chart(y_true, y_pred)
ferrum.confusion_matrix_chart(y_true, y_pred)
ferrum.discrimination_threshold_chart(y_true, y_pred)   # cv=None
```

and get correct charts. Importing scikit-learn is never attempted on this path.

A user passing a fitted estimator (`ferrum.roc_chart(model, X, y)`, the
`ModelSource` path) still requires the `models` extra — but only because the
estimator itself is a scikit-learn object whose `predict_proba` /
`decision_function` must be called. The **curve math** that consumes those
scores runs through the same Rust kernels as the precomputed path.

`discrimination_threshold_chart(..., cv=<n>)` continues to require scikit-learn
(it re-fits across folds) and raises the existing `ValueError` on the precomputed
path. No behavior change there.

Error messaging: when scikit-learn is genuinely needed (model-backed methods,
`cv=`), the existing `require_sklearn(...)` hint is unchanged. The precomputed
curve methods no longer call `require_sklearn`.

## 5. Architecture

**Computation ownership after this change:**

| Stage | Owner |
|---|---|
| Curve/matrix math (sort, cumsum, bincount, integration) | Rust `diagnostics.rs` kernels |
| Per-class fan-out, micro/macro/weighted averaging, grid interpolation | Python adapter, columnar (numpy → polars) |
| Score extraction from a fitted model (`predict_proba`, etc.) | Python `ModelSource`, scikit-learn |
| Long-form assembly into the chart-contract DataFrame | Arrow `RecordBatch` from Rust, or `pl.concat` of columnar frames in Python |

**Data flow (precomputed path, ROC example):** raw arrays → `pyarrow` arrays →
Rust `roc_curve_kernel` → Arrow `RecordBatch` with the per-curve columns →
`pl.from_arrow`. For multiclass with averaging, the adapter calls the kernel
once per class, then computes the averaged summary curve in columnar Python and
`pl.concat`s the result. The chart layer's contract — a long DataFrame keyed by
a `class` discriminator column with the scalar metric (`auc`/`ap`) broadcast per
row — is preserved exactly; only its construction moves from row-dicts to
columnar/Arrow.

**Averaging stays in Python by design.** The micro path is a single kernel call
on raveled binarized arrays. The macro/weighted paths interpolate per-class
curves onto a shared grid (`np.interp`) and take a weighted mean — cheap,
vectorized, and not worth a second Rust surface. Only the per-curve kernels and
the scalar metrics live in Rust.

The kernels mirror the established Arrow-boundary pattern already used by
`hat_matrix_stats` and `studentized_residual_no_x` (pyo3-arrow `PyArray` /
`PyRecordBatch` in and out), and register in `lib.rs` alongside the existing
diagnostics functions.

## 6. Canonical interfaces / data contracts

Rust kernels exposed to Python (pyo3 signatures, illustrative — names and
columns are the contract, argument plumbing is not):

```text
roc_curve_kernel(y_true: f64[], y_score: f64[], drop_intermediate: bool)
    -> RecordBatch{ fpr: f64, tpr: f64, threshold: f64 }
roc_auc(y_true: f64[], y_score: f64[]) -> f64

pr_curve_kernel(y_true: f64[], y_score: f64[])
    -> RecordBatch{ precision: f64, recall: f64, threshold: f64 }   # threshold padded with NaN on the final point
average_precision(y_true: f64[], y_score: f64[]) -> f64

calibration_kernel(y_true: f64[], y_prob: f64[], n_bins: u32, strategy: "uniform"|"quantile")
    -> RecordBatch{ mean_predicted: f64, fraction_positive: f64, count: i64 }   # empty bins dropped

confusion_kernel(y_true: i64[], y_pred: i64[], labels: i64[], normalize: ""|"true"|"pred"|"all")
    -> RecordBatch{ row: i64, col: i64, value: f64 }   # dense L×L in row-major order

prf_at_thresholds(y_true: f64[], y_score: f64[], thresholds: f64[])
    -> RecordBatch{ precision: f64, recall: f64, f1: f64, queue_rate: f64 }
```

The **Python-facing DataFrame schemas are unchanged** from today and remain the
contract the chart builders consume:

- ROC: `fpr, tpr, threshold, class, auc`
- PR: `precision, recall, threshold, class, ap`
- Calibration: `mean_predicted, fraction_positive, count`
- Confusion: `actual, predicted, value, value_fmt`
- Discrimination threshold: `threshold, precision, recall, f1, queue_rate`

The kernels emit the numeric core; the Python adapter attaches the `class`
string column, broadcasts the scalar metric, maps integer label indices back to
original label values, and formats `value_fmt`. Label-to-index mapping
(`confusion_kernel` takes integer-encoded labels and a sorted `labels` array)
stays in Python so arbitrary string/categorical labels are handled where the
dtype is known.

## 7. Invariants and constraints

- **Byte-parity with scikit-learn.** Chart output (and therefore every existing
  golden SVG) must not change. The kernels must reproduce scikit-learn's
  conventions, not merely "a correct curve." Specifically:
  - `roc_curve`: descending-score sort with tie aggregation; the leading
    threshold sentinel scikit-learn emits; `drop_intermediate` collinear-point
    pruning that keeps only points changing the curve's slope.
  - `precision_recall_curve`: distinct ascending thresholds; reversed-cumsum
    precision/recall; the trailing `(precision=1, recall=0)` endpoint; threshold
    array one shorter than precision/recall (the adapter pads with `NaN`).
  - `average_precision_score`: step-function sum `Σ (Rₙ − Rₙ₋₁)·Pₙ`, **not**
    trapezoidal.
  - `roc_auc_score`: trapezoidal area.
  - `calibration_curve`: empty bins dropped; `uniform` vs `quantile` edges per
    scikit-learn; bin counts aligned to the surviving bins.
  - `confusion_matrix`: sorted labels; `normalize ∈ {true, pred, all, None}`
    row/column/total normalization.
- **No scikit-learn import on the precomputed curve path** — not at module
  import, not inside the method bodies for the five functions.
- **No `list[dict]` in the ported functions or in gain/lift/cumulative.** Output
  frames are built columnarly or handed back as Arrow.
- **The `models` / `shap` / `all` extras still pin `scikit-learn`** — the
  model-backed path genuinely needs it. The change is that the *precomputed*
  path no longer transitively needs it.
- The existing import-time sklearn-free guarantee (asserted by
  `test_no_sklearn_at_import.py`) is preserved and extended to the precomputed
  compute path.

## 8. Key decisions and tradeoffs

- **Exact byte-parity over numeric-equivalence (locked).** Reproduce
  scikit-learn's exact point sets and conventions so goldens and tests pass
  untouched. Rejected: "equivalent within tolerance, regenerate goldens" — it
  trades a one-time Rust effort for permanent golden churn and a manual
  rasterize-and-reinspect pass on every affected chart. Parity is verified in
  tests by comparing kernel output against scikit-learn (a dev dependency),
  pinned to the version used to generate the goldens.
- **Both paths route through the Rust kernels (locked).** Single source of truth
  for curve math. The model-backed path still imports scikit-learn for the
  estimator, but its metric computation no longer duplicates the precomputed
  path's. Rejected: de-sklearn only the precomputed path — it would leave two
  parallel metric implementations to keep bit-identical, exactly the drift risk
  this consolidation removes.
- **Averaging and interpolation stay in Python, columnar (locked).** micro =
  one kernel call on raveled arrays; macro/weighted = `np.interp` onto a shared
  grid + weighted mean. Porting this adds Rust surface for cheap vectorized work
  with no dependency payoff.
- **Gain / lift / cumulative-gain get a columnar-Python cleanup, not a port
  (locked).** They use no scikit-learn, so they do not block the dependency
  goal. They keep their math in Python but replace `rows.append(...)` /
  `pl.DataFrame(rows)` with vectorized numpy → `pl.DataFrame`/`pl.concat`,
  removing the row-dict cost. Output schema unchanged.
- **Confusion-matrix label handling stays in Python.** The kernel works on
  integer-encoded labels plus a sorted `labels` array; the adapter owns
  encode/decode so arbitrary string and categorical labels work where their
  dtype is known. Avoids pushing dtype polymorphism across the FFI boundary.
- **Kernels return the numeric core only; the adapter attaches `class`, the
  broadcast scalar, and `value_fmt`.** Keeps the Rust surface minimal and the
  long-form/grouping concerns where the chart contract already lives.

## 9. Acceptance criteria

- With scikit-learn uninstalled, the five precomputed-path charts render and
  match their committed goldens byte-for-byte.
- The model-backed equivalents produce identical output to today, with curve
  math flowing through the Rust kernels (no `sklearn.metrics` import on the
  metric step).
- A parity test suite compares each kernel against the corresponding
  `sklearn.metrics` / `sklearn.calibration` function across binary, multiclass,
  degenerate (single-class, all-correct, empty-bin), and tie-heavy inputs, and
  asserts equality under the byte-parity conventions in §7.
- No ported function and none of gain/lift/cumulative constructs a `list[dict]`.
- `cargo test` passes (kernel unit tests included); `cargo clippy -D warnings`
  is clean on `ferrum-core`.
- `test_no_sklearn_at_import.py` still passes, and a new test asserts the five
  precomputed curve methods run with `sklearn` absent from `sys.modules`.

## 10. Validation strategy

- **Parity harness:** for randomized and adversarial inputs (ties, single class,
  all-positive/all-negative, empty quantile bins, `drop_intermediate` on/off,
  micro/macro/weighted), assert kernel output equals scikit-learn's under the
  documented conventions. This is the primary correctness gate and the guard for
  byte-parity.
- **Golden inspection:** any golden that *does* shift (it should not) is
  rasterized to PNG via `scripts/snapshot-goldens.py` and visually inspected
  before acceptance, per the project golden-blessing rule. Expectation is zero
  shifted goldens.
- **Dependency isolation test:** drop `sklearn` from `sys.modules`, then exercise
  each precomputed curve method end-to-end; assert no re-import occurs.
- **Model-path equivalence:** assert the model-backed and precomputed paths emit
  identical curve frames for the same `(y_true, scores)` so the single-source-of-
  truth claim is enforced by test, not convention.

## 11. Open questions

None blocking. The one detail requiring care during implementation — the exact
leading-threshold sentinel and `drop_intermediate` pruning rule for the installed
scikit-learn version — is resolved empirically by the parity harness against that
version, not by spec.
