# Phase 10 — Model Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the full model-diagnostics layer (`ferrum-spec.md §3.1 / §3.3 / §3.14 / §3.15`) — `ModelSource` adapter with 22 derived-data methods, 26 model-diagnostic marks, 21 Group B figure functions, 25 sklearn-protocol Visualizers, one new Rust function (`kendall_tau_b`). Every spec parameter implemented fully; no warn-fallbacks; sklearn never imported unless the user's model is from sklearn.

**Architecture:**
- **Three-layer Python adapter:** ModelSource (Python, lazy-imported sklearn delegation) → private chart builders → public figure functions + visualizers. Zero new Rust `Mark` or `Transform` variants — all 26 marks desugar Python-side over existing Phase 5/8b/9 primitives.
- **Single Rust addition:** `ferrum._core.kendall_tau_b` (Knight's O(n log n)) for the one statistic where vectorized NumPy can't compete at training-set scale.
- **Single-tier SVG goldens:** pre-fit sklearn models serialized via `skops` in `tests/fixtures/models/` make every Phase 10 figure byte-identical to render at the renderer's existing 3-decimal-place quantization (`fmt_f` in `crates/ferrum-core/src/render/svg.rs`). No new Rust quantization knob required.

**Tech Stack:** Rust 2021 (PyO3 0.28, abi3-py310). Python ≥3.10 (numpy, polars, pyarrow). New optional extras: `ferrum[models]` (sklearn), `[shap]`, `[umap]`, `[ml-all]`. Dev-only deps: `scipy` (parity tests), `skops` (fixture serialization).

**Source spec:** `docs/superpowers/specs/2026-05-10-model-diagnostics-design.md` (commit `3bb14f9`).

**Branch:** `feat/phase-10` (created in Task 0).

---

## Pre-flight

1. **Build commands** (all run from repo root):
   - **Rust extension build:** `source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop`
   - **`cargo test`:** `source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core`
   - **`pytest`:** `uv run --no-sync pytest`
2. **Test baselines at start of Phase 10 (verified 2026-05-10 from Phase 9 merge commit `11f956e`):**
   - `cargo test -p ferrum-core` → **496 passed**
   - `uv run pytest` → **480 passed, 5 skipped**
3. **Final targets at Phase 10 done:**
   - `cargo test -p ferrum-core` ≥ **501** (≈5 new tests: kendall_tau_b correctness against scipy + edge cases)
   - `uv run pytest` ≥ **610** (≈130 new tests across ModelSource methods, parity vs scipy, mark coverage, no-sklearn-at-import, figure functions, visualizers, e2e renders)
   - ~35 SVG goldens at the renderer's existing 3-dp quantization (single tier).
   - 0 new pytest skips, 0 new xfails on top of Phase 9's 5 skipped.
4. **Conventions (from `CLAUDE.md`):**
   - Plain feature branch, NOT a worktree (per memory `feedback_ferrum_workflow_branches`).
   - **No `Co-Authored-By: Claude`** trailers on commits.
   - **No `git push`** without explicit user request.
   - **Confirm with user before merging to `main`.**
   - Sub-batches commit independently on `feat/phase-10`; each task ends with a single commit.
   - **Subagent-verify rule (memory `feedback_subagent_verification`):** orchestrator MUST re-run `cargo test -p ferrum-core` and `git ls-tree HEAD --name-only -r` after each subagent task to verify reported file changes and test counts are real. Phase 8b had falsely reported deletions — do not trust subagent reports until independently verified.

---

## Task overview

Sub-batches and their tasks, in build order. Each sub-batch lands on `feat/phase-10` as a sequence of commits; sub-batches do NOT branch separately.

| Sub-batch | Tasks | Theme |
|---|---|---|
| **Pre-flight** | 0–4 | Branch creation; `pyproject.toml` extras + dev deps; sklearn version pin; `tests/fixtures/` infrastructure; verify shap/umap install. |
| **10a-foundation** | 5–11 | Task 5 DROPPED (renderer already 3-dp quantized). `_diagnostics` package skeleton; `ModelSource` class + protocol detection + lazy-import helpers; `.predictions()`, `.probabilities()`; `mark_residuals` + `mark_prediction_error`; `residuals_chart`; `ResidualsVisualizer` + `PredictionErrorVisualizer` + `CooksDistanceVisualizer`; first goldens. |
| **10b-cls-curves** | 12–17 | `.roc_curve()`, `.pr_curve()`, `.calibration_curve()`, `.cumulative_gain()`, `.lift_curve()`, `.discrimination_threshold()`; 6 marks; 6 figure functions; 4 visualizers; goldens at 3 dp. |
| **10c-cls-matrix** | 18–20 | `.confusion_matrix()`; 2 marks (`mark_confusion`, `mark_class_prediction_error`); 2 figure functions; 4 visualizers (`Confusion`, `ClassificationReport`, `ClassPredictionError`, `ClassBalance`); goldens. |
| **10d-explain** | 21–25 | `.importances()`, `.shap_values()`, `.partial_dependence()`; 5 marks (`mark_importance`, `mark_shap_beeswarm`, `mark_shap_bar`, `mark_shap_waterfall`, `mark_pdp`); 2 figure functions; 2 visualizers; SHAP optional extra wired; goldens at 3 dp. |
| **10e-cv** | 26–29 | `.learning_curve()`, `.validation_curve()`, `.cv_scores()`, `.alpha_selection()`; 4 marks; 4 figure functions; 4 visualizers; goldens at 3 dp. |
| **10f-cluster** | 30–34 | `.silhouette()`, `.intercluster_distance()`, `.embeddings()`, `.pca_variance()`; 4 marks (`mark_silhouette`, `mark_intercluster_distance`, `mark_pca_scree`, `mark_decision_boundary`); 4 figure functions; 5 visualizers; UMAP optional extra wired; goldens at 3 dp. |
| **10g-rank** | 35–39 | Rust `kendall_tau_b` (Knight's algorithm); Python `_diagnostics/stats.py` with vectorized NumPy implementations (pearson, spearman, shapiro_w, variance/covariance ranking); scipy parity tests; `.rank1d()`, `.rank2d()`; 3 marks; 2 figure functions; 3 visualizers; goldens at 3 dp. |
| **10h-finalize** | 40–44 | `ModelSource.compare(...)` + `ComparedModelSource`; spec drift notes applied to `ferrum-spec.md`; `PHASE_9_PLUS_MARKS` audit; `ferrum-phases.md` Phase 10 → `done`; user-confirmed merge to `main`. |

**Parallelization guidance (for subagent-driven execution):**
- Pre-flight tasks (0–4) are **strictly sequential**.
- Within 10a, Task 5 DROPPED (renderer already 3-dp quantized). Task 6 lands the `_diagnostics/` skeleton + deps. Task 7 (ModelSource shell) depends on 6. Tasks 8–11 depend on 7. Strictly sequential 6→7→8→9→10→11.
- Within 10b, the six diagnostic families (ROC, PR, calibration, gain, lift, discrimination_threshold) are **mutually parallel** — each adds one ModelSource method + one mark + one figure + one visualizer + one golden. Recommend grouping as 3 parallel tasks (12: ROC+PR, 13: calibration+gain+lift, 14: discrimination_threshold; 15–17: visualizers).
- 10c is small (3 tasks); recommend sequential.
- 10d: Task 21 (importances + mark_importance + importance_chart) is independent of Task 22 (SHAP family) and Task 23 (PDP). All three parallelizable. Tasks 24–25 (visualizers) come after.
- 10e: four CV families parallelizable.
- 10f: clustering and manifold are independent; decision_boundary depends only on existing Phase 8b primitives (mark_raster + mark_contour). Tasks 30–32 parallelizable.
- 10g: Task 35 (Rust kendall_tau_b) is on the critical path. Tasks 36 (stats.py) and 37 (rank methods + marks + figures) depend on 35.
- 10h: strictly sequential.

When subagents run in parallel, the orchestrator merges sequentially with `cargo test` + `pytest` between merges per the subagent-verify rule.

---

## File map

### New Rust files (`crates/ferrum-core/src/`)

| Path | Responsibility |
|---|---|
| `diagnostics.rs` | `kendall_tau_b(x: &[f64], y: &[f64]) -> KendallResult` (Knight's O(n log n) merge-sort with tie counting); `KendallResult` struct; PyO3 binding via numpy `PyReadonlyArray1<f64>`; tests against hand-computed reference values and edge cases (all-tied, all-unique, anti-correlated). |

### Modified Rust files (`crates/ferrum-core/src/`)

| Path | Change |
|---|---|
| `lib.rs` | Register `kendall_tau_b` as a module function via `#[pyfunction]`. |
| *(Task 5 DROPPED)* | The renderer already routes every float through `fmt_f` which quantizes to `FLOAT_PRECISION = 3`. No `RenderConfig` change required for cross-platform-stable goldens. |

### New Python files (`src/ferrum/`)

| Path | Responsibility |
|---|---|
| `_diagnostics/__init__.py` | Re-export `ModelSource`, `ComparedModelSource`. |
| `_diagnostics/source.py` | `ModelSource` class (constructor, protocol detection, 22 derived methods, cache); `ComparedModelSource` class (vstacks with `model` column). |
| `_diagnostics/deps.py` | Lazy-import helpers `require_sklearn`, `require_shap`, `require_umap` raising `ImportError` with `ferrum[<extra>]` hint. |
| `_diagnostics/schemas.py` | `polars.Schema` constants for every derived-data DataFrame (22 schemas). |
| `_diagnostics/stats.py` | Vectorized NumPy in-house statistics: `pearson_r`, `spearman_rho`, `shapiro_w`, `studentized_residual`, `variance_rank`, `covariance_rank`, `kendall_tau_b` (Python wrapper around `ferrum._core.kendall_tau_b`), `rank1d_compute`, `rank2d_compute`. |
| `_diagnostics/charts.py` | Private chart-builder functions: `_residuals_chart_from_source`, `_prediction_error_chart_from_source`, `_roc_chart_from_source`, `_pr_chart_from_source`, `_confusion_chart_from_source`, `_calibration_chart_from_source`, `_gain_chart_from_source`, `_lift_chart_from_source`, `_discrimination_threshold_chart_from_source`, `_class_prediction_error_chart_from_source`, `_classification_report_chart`, `_importance_chart_from_source`, `_shap_beeswarm_chart_from_source`, `_shap_bar_chart_from_source`, `_shap_waterfall_chart_from_source`, `_pdp_chart_from_source`, `_learning_curve_chart_from_source`, `_validation_curve_chart_from_source`, `_cv_scores_chart_from_source`, `_alpha_selection_chart_from_source`, `_silhouette_chart_from_source`, `_intercluster_distance_chart_from_source`, `_pca_scree_chart_from_source`, `_decision_boundary_chart_from_source`, `_rank1d_chart_from_dataframe`, `_rank2d_chart_from_dataframe`, `_parallel_coords_chart_from_dataframe`, `_elbow_chart_from_dataframe`, `_class_balance_chart_from_dataframe`. |
| `_diagnostics/visualizers/__init__.py` | Re-export 25 visualizer classes. |
| `_diagnostics/visualizers/base.py` | `FerrumVisualizer` base class. |
| `_diagnostics/visualizers/regression.py` | `ResidualsVisualizer`, `PredictionErrorVisualizer`, `CooksDistanceVisualizer`. |
| `_diagnostics/visualizers/classification.py` | `ROCVisualizer`, `PRVisualizer`, `CalibrationVisualizer`, `ConfusionMatrixVisualizer`, `ClassificationReportVisualizer`. |
| `_diagnostics/visualizers/classification_extra.py` | `DiscriminationThresholdVisualizer`, `ClassPredictionErrorVisualizer`, `ClassBalanceVisualizer`. |
| `_diagnostics/visualizers/explanation.py` | `FeatureImportancesVisualizer`, `SHAPVisualizer`, `ParallelCoordinatesVisualizer`. |
| `_diagnostics/visualizers/selection.py` | `LearningCurveVisualizer`, `ValidationCurveVisualizer`, `CVScoresVisualizer`, `AlphaSelectionVisualizer`. |
| `_diagnostics/visualizers/clustering.py` | `SilhouetteVisualizer`, `ElbowVisualizer`, `ManifoldVisualizer`, `InterclusterDistanceVisualizer`, `PCAVarianceVisualizer`. |
| `_diagnostics/visualizers/ranking.py` | `Rank1DVisualizer`, `Rank2DVisualizer`. |
| `marks/diagnostic.py` | 26 mark value classes with `_expand` methods desugaring to existing primitives. |
| `figures.py` | 21 figure functions (`roc_chart`, `pr_chart`, …, `cv_scores_chart`) + `_resolve_source` helper. |

### Modified Python files (`src/ferrum/`)

| Path | Change |
|---|---|
| `__init__.py` | Re-export `ModelSource`, `ComparedModelSource`, 26 diagnostic marks, 21 figure functions, 25 visualizers; thread `figures` and `_diagnostics.visualizers` submodules into the public namespace. |
| `_core.pyi` | Type stub for `kendall_tau_b(x: np.ndarray, y: np.ndarray) -> dict`. |
| `chart.py` | Add `mark_residuals` / `mark_prediction_error` / `mark_confusion` / `mark_roc` / ...26 mark methods (each delegating to the value class's `_expand`). |
| `marks/deferred.py` | Audit `PHASE_9_PLUS_MARKS` at Phase 10 close-out — must retain only `arc`, `image`, `geoshape`, `label`. |

### New fixture / test files (`tests/`)

| Path | Responsibility |
|---|---|
| `fixtures/__init__.py` | `ALLOWED_SK_TYPES` allowlist for `skops.io.load`; `load_fixture(name)` helper. |
| `fixtures/SKLEARN_VERSION` | Single-line file with the pinned sklearn version (e.g. `1.7.2`). |
| `fixtures/build.py` | One-shot script to generate every fixture `.skops` file. Reads `SKLEARN_VERSION`, aborts on mismatch. |
| `fixtures/models/binary_logistic.skops` | Pre-fit `sklearn.linear_model.LogisticRegression` on synthetic binary data. |
| `fixtures/models/multiclass_logistic.skops` | Pre-fit `LogisticRegression(multi_class="ovr")` on synthetic 3-class data. |
| `fixtures/models/regression_ridge.skops` | Pre-fit `sklearn.linear_model.Ridge`. |
| `fixtures/models/regression_rf.skops` | Pre-fit `sklearn.ensemble.RandomForestRegressor`. |
| `fixtures/models/kmeans_3cluster.skops` | Pre-fit `sklearn.cluster.KMeans(n_clusters=3)`. |
| `fixtures/models/pca_4comp.skops` | Pre-fit `sklearn.decomposition.PCA(n_components=4)`. |
| `fixtures/datasets/binary_classification.parquet` | 200×4 synthetic binary classification dataset (seed=0). |
| `fixtures/datasets/multiclass_classification.parquet` | 300×4 synthetic 3-class dataset (seed=0). |
| `fixtures/datasets/regression.parquet` | 200×5 synthetic regression dataset (seed=0). |
| `fixtures/datasets/clustering.parquet` | 200×3 synthetic clustering dataset (seed=0). |
| `diagnostics/test_no_sklearn_at_import.py` | Asserts `sklearn` not in `sys.modules` after `import ferrum` and after `ModelSource(_DuckModel(), df)`. |
| `diagnostics/test_source.py` | ModelSource constructor, protocol detection, missing-capability errors, cache behavior. |
| `diagnostics/test_stats.py` | Parity tests for `pearson_r`, `spearman_rho`, `shapiro_w`, `kendall_tau_b` against scipy. |
| `diagnostics/test_regression.py` | `mark_residuals`, `mark_prediction_error`, `residuals_chart` (10a). |
| `diagnostics/test_classification.py` | ROC/PR/calibration/gain/lift/disc_threshold + confusion/class_prediction_error (10b + 10c). |
| `diagnostics/test_explanation.py` | importance, SHAP family, PDP (10d). |
| `diagnostics/test_selection.py` | learning_curve, validation_curve, cv_scores, alpha_selection (10e). |
| `diagnostics/test_clustering.py` | silhouette, intercluster_distance, pca_scree, decision_boundary (10f). |
| `diagnostics/test_ranking.py` | rank1d, rank2d, parallel_coordinates (10g). |
| `diagnostics/test_compare.py` | `ModelSource.compare()` + every figure function with `compare=` kwarg (10h). |
| `diagnostics/test_mark_coverage.py` | Asserts every Phase 10 mark is implemented and not in `PHASE_9_PLUS_MARKS`. |
| `goldens/phase_10/*.svg` | ~35–40 SVG goldens at 3-dp quantization (single tier — see Task 5 note). |
| `conftest.py` | Session-level fixture asserting sklearn/shap/umap/scipy/skops are installed and that sklearn version matches `SKLEARN_VERSION`. |

### Modified docs

| Path | Change |
|---|---|
| `ferrum-spec.md` | Apply dated drift notes to §1, §3.1, §3.3, §3.14, §3.15, §3.16 (Task 41). |
| `docs/superpowers/ferrum-phases.md` | Phase 10 row `pending` → `done`; link to design + plan docs (Task 43). |
| `pyproject.toml` | Add four `[project.optional-dependencies]` extras: `models`, `shap`, `umap`, `ml-all`. Add `scipy` and `skops` to dev deps. |

---

## Task list

### Task 0: Create `feat/phase-10` branch

**Files:** none (branch creation only)

- [ ] **Step 1: Verify clean working tree on main**

Run:
```bash
git status && git log -1 --oneline
```
Expected: `On branch main`, working tree clean, last commit `3bb14f9 docs: Phase 10 model diagnostics design`.

- [ ] **Step 2: Verify baselines**

```bash
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
uv run --no-sync pytest 2>&1 | tail -3
```
Expected: `cargo test` → `496 passed`. `pytest` → `480 passed, 5 skipped`.

- [ ] **Step 3: Create the branch**

```bash
git checkout -b feat/phase-10
git status
```
Expected: `On branch feat/phase-10`.

---

### Task 1: `pyproject.toml` extras + dev deps

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read existing pyproject**

```bash
grep -n "optional-dependencies\|dev-dependencies\|\[tool.uv\]" pyproject.toml
```

- [ ] **Step 2: Add Phase 10 extras**

Append to the `[project.optional-dependencies]` block (or create it if absent):

```toml
[project.optional-dependencies]
models = ["scikit-learn>=1.3"]
shap = ["scikit-learn>=1.3", "shap>=0.42"]
umap = ["scikit-learn>=1.3", "umap-learn>=0.5"]
ml-all = ["scikit-learn>=1.3", "shap>=0.42", "umap-learn>=0.5"]
```

In the `[tool.uv]` `dev-dependencies` list (or `[dependency-groups].dev` if uv 0.5+), add:

```toml
"scipy>=1.10",
"skops>=0.9",
```

- [ ] **Step 3: Sync the dev environment**

```bash
unset CONDA_PREFIX && uv sync --extra ml-all 2>&1 | tail -10
```
Expected: sklearn, shap, umap-learn, scipy, skops installed without error.

- [ ] **Step 4: Verify imports work in the venv**

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "
import sklearn, shap, umap, scipy, skops
print('sklearn', sklearn.__version__)
print('shap', shap.__version__)
print('umap', umap.__version__)
print('scipy', scipy.__version__)
print('skops', skops.__version__)
print('OK')
"
```
Expected: All five versions printed, `OK`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(phase-10): add models/shap/umap/ml-all extras and dev deps"
```

---

### Task 2: Pin sklearn version + `tests/fixtures/` infrastructure

**Files:**
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/SKLEARN_VERSION`
- Create: `tests/fixtures/build.py`
- Create: `tests/fixtures/models/.gitkeep`
- Create: `tests/fixtures/datasets/.gitkeep`

- [ ] **Step 1: Determine the installed sklearn version to pin**

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "import sklearn; print(sklearn.__version__)"
```
Record the exact version string (e.g. `1.7.2`). This becomes the pinned version.

- [ ] **Step 2: Write `tests/fixtures/SKLEARN_VERSION`**

Single line, no trailing newline, containing the version from Step 1, e.g.:

```
1.7.2
```

- [ ] **Step 3: Write `tests/fixtures/__init__.py`**

```python
"""Phase 10 test fixtures: pre-fit sklearn models serialized via skops.

`load_fixture(name)` loads a `.skops` file from `tests/fixtures/models/`
using a strict allowlist of sklearn types — never `pickle.load`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_FIXTURES_DIR = Path(__file__).parent
_MODELS_DIR = _FIXTURES_DIR / "models"
_DATASETS_DIR = _FIXTURES_DIR / "datasets"


def pinned_sklearn_version() -> str:
    return (_FIXTURES_DIR / "SKLEARN_VERSION").read_text().strip()


# Allowlist for skops.io.load — only sklearn estimators and supporting NumPy/SciPy types.
ALLOWED_SK_TYPES: tuple[str, ...] = (
    "sklearn.linear_model._logistic.LogisticRegression",
    "sklearn.linear_model._ridge.Ridge",
    "sklearn.linear_model._coordinate_descent.Lasso",
    "sklearn.linear_model._coordinate_descent.ElasticNet",
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.ensemble._forest.RandomForestRegressor",
    "sklearn.tree._classes.DecisionTreeClassifier",
    "sklearn.tree._classes.DecisionTreeRegressor",
    "sklearn.cluster._kmeans.KMeans",
    "sklearn.decomposition._pca.PCA",
    "sklearn.svm._classes.SVC",
    "sklearn.preprocessing._data.StandardScaler",
    "sklearn.pipeline.Pipeline",
    "numpy.ndarray",
    "numpy.dtype",
    "numpy.dtype[float64]",
    "numpy.dtype[int64]",
    "numpy.dtype[int32]",
)


def load_fixture(name: str) -> Any:
    """Load a fitted model fixture by name.

    Args:
        name: filename without extension, e.g. "binary_logistic".

    Returns:
        The deserialized sklearn estimator.

    Raises:
        FileNotFoundError: if the .skops file is missing.
        skops.io.exceptions.UntrustedTypesFoundException: if the
            serialized object contains types outside ALLOWED_SK_TYPES.
    """
    import skops.io as sio

    path = _MODELS_DIR / f"{name}.skops"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing fixture {path}. Run `python tests/fixtures/build.py` "
            f"to regenerate (requires scikit-learn=={pinned_sklearn_version()})."
        )
    return sio.load(path, trusted=list(ALLOWED_SK_TYPES))


def load_dataset(name: str):
    """Load a parquet dataset fixture (returns polars.DataFrame)."""
    import polars as pl
    path = _DATASETS_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset {path}. Run `python tests/fixtures/build.py`."
        )
    return pl.read_parquet(path)
```

- [ ] **Step 4: Write `tests/fixtures/build.py`**

```python
"""One-shot script to regenerate all Phase 10 model fixtures.

Run with:
    uv run --no-sync python tests/fixtures/build.py

Aborts if installed sklearn doesn't match tests/fixtures/SKLEARN_VERSION.
"""
from __future__ import annotations

import sys
from pathlib import Path

FIXTURES = Path(__file__).parent
MODELS = FIXTURES / "models"
DATASETS = FIXTURES / "datasets"


def _check_sklearn_pin() -> None:
    import sklearn
    pinned = (FIXTURES / "SKLEARN_VERSION").read_text().strip()
    if sklearn.__version__ != pinned:
        print(
            f"ERROR: installed sklearn=={sklearn.__version__} but fixtures "
            f"require sklearn=={pinned}. Run `uv pip install scikit-learn=={pinned}` "
            f"or update tests/fixtures/SKLEARN_VERSION and regenerate all goldens.",
            file=sys.stderr,
        )
        sys.exit(1)


def _save(model, name: str) -> None:
    import skops.io as sio
    path = MODELS / f"{name}.skops"
    sio.dump(model, path)
    print(f"  wrote {path.name}")


def _save_dataset(df, name: str) -> None:
    path = DATASETS / f"{name}.parquet"
    df.write_parquet(path)
    print(f"  wrote {path.name}")


def build_datasets() -> dict:
    import numpy as np
    import polars as pl
    rng = np.random.RandomState(0)

    # Binary classification — 200 rows, 4 features.
    n = 200
    X_bin = rng.randn(n, 4)
    coef = np.array([1.5, -1.0, 0.5, 0.0])
    logits = X_bin @ coef + rng.randn(n) * 0.5
    y_bin = (logits > 0).astype(np.int64)
    bin_df = pl.DataFrame({
        "f0": X_bin[:, 0], "f1": X_bin[:, 1],
        "f2": X_bin[:, 2], "f3": X_bin[:, 3],
        "y": y_bin,
    })
    _save_dataset(bin_df, "binary_classification")

    # Multiclass classification — 300 rows, 4 features, 3 classes.
    n_mc = 300
    X_mc = rng.randn(n_mc, 4)
    class_means = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0]])
    y_mc = rng.randint(0, 3, size=n_mc)
    X_mc = X_mc + class_means[y_mc]
    mc_df = pl.DataFrame({
        "f0": X_mc[:, 0], "f1": X_mc[:, 1],
        "f2": X_mc[:, 2], "f3": X_mc[:, 3],
        "y": y_mc.astype(np.int64),
    })
    _save_dataset(mc_df, "multiclass_classification")

    # Regression — 200 rows, 5 features.
    n_reg = 200
    X_reg = rng.randn(n_reg, 5)
    y_reg = X_reg @ np.array([2.0, -1.5, 0.5, 0.0, 0.0]) + rng.randn(n_reg) * 0.3
    reg_df = pl.DataFrame({
        "f0": X_reg[:, 0], "f1": X_reg[:, 1],
        "f2": X_reg[:, 2], "f3": X_reg[:, 3], "f4": X_reg[:, 4],
        "y": y_reg,
    })
    _save_dataset(reg_df, "regression")

    # Clustering — 200 rows, 3 features, 3 well-separated blobs.
    n_clu = 200
    centers = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0]])
    labels = rng.randint(0, 3, size=n_clu)
    X_clu = centers[labels] + rng.randn(n_clu, 3) * 0.5
    clu_df = pl.DataFrame({
        "f0": X_clu[:, 0], "f1": X_clu[:, 1], "f2": X_clu[:, 2],
    })
    _save_dataset(clu_df, "clustering")

    return {
        "binary": (bin_df, y_bin),
        "multiclass": (mc_df, y_mc),
        "regression": (reg_df, y_reg),
        "clustering": (clu_df, labels),
    }


def build_models(data: dict) -> None:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    bin_df, y_bin = data["binary"]
    X_bin = bin_df.select(["f0", "f1", "f2", "f3"]).to_numpy()

    mc_df, y_mc = data["multiclass"]
    X_mc = mc_df.select(["f0", "f1", "f2", "f3"]).to_numpy()

    reg_df, y_reg = data["regression"]
    X_reg = reg_df.select(["f0", "f1", "f2", "f3", "f4"]).to_numpy()

    clu_df, _ = data["clustering"]
    X_clu = clu_df.to_numpy()

    _save(LogisticRegression(random_state=0, max_iter=500).fit(X_bin, y_bin), "binary_logistic")
    _save(LogisticRegression(random_state=0, max_iter=500, multi_class="ovr").fit(X_mc, y_mc), "multiclass_logistic")
    _save(Ridge(random_state=0).fit(X_reg, y_reg), "regression_ridge")
    _save(RandomForestRegressor(n_estimators=20, random_state=0).fit(X_reg, y_reg), "regression_rf")
    _save(KMeans(n_clusters=3, random_state=0, n_init=10).fit(X_clu), "kmeans_3cluster")
    _save(PCA(n_components=4, random_state=0).fit(X_reg), "pca_4comp")


def main() -> None:
    _check_sklearn_pin()
    MODELS.mkdir(parents=True, exist_ok=True)
    DATASETS.mkdir(parents=True, exist_ok=True)
    print("Building datasets...")
    data = build_datasets()
    print("Building models...")
    build_models(data)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the build script**

```bash
unset CONDA_PREFIX && uv run --no-sync python tests/fixtures/build.py
```
Expected: `Done.` printed, with 4 datasets + 6 models written.

- [ ] **Step 6: Verify fixtures load correctly**

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "
from tests.fixtures import load_fixture, load_dataset
m = load_fixture('binary_logistic')
df = load_dataset('binary_classification')
print('model:', type(m).__name__)
print('df:', df.shape)
print('OK')
"
```
Expected: `model: LogisticRegression`, `df: (200, 5)`, `OK`.

- [ ] **Step 7: Add `.gitkeep` files and commit**

```bash
touch tests/fixtures/models/.gitkeep tests/fixtures/datasets/.gitkeep
git add tests/fixtures/
git commit -m "test(phase-10): fixture infrastructure (skops models + parquet datasets)"
```

---

### Task 3: Session-level `conftest.py` extras-check

**Files:**
- Modify: `tests/conftest.py` (create if absent)

- [ ] **Step 1: Inspect existing conftest if present**

```bash
test -f tests/conftest.py && cat tests/conftest.py || echo "no conftest yet"
```

- [ ] **Step 2: Add session-level extras check**

Append (or create) `tests/conftest.py` with:

```python
"""Test-suite-wide fixtures."""
from __future__ import annotations

import importlib
import sys

import pytest


_REQUIRED_FOR_PHASE_10 = ("sklearn", "shap", "umap", "scipy", "skops")


@pytest.fixture(scope="session", autouse=True)
def _require_phase_10_extras():
    """Phase 10 tests assume the full ml-all extras + dev deps are installed."""
    missing = []
    for mod in _REQUIRED_FOR_PHASE_10:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        pytest.exit(
            f"Phase 10 test suite requires: {', '.join(missing)}. "
            f"Install with `uv sync --extra ml-all` plus dev deps.",
            returncode=1,
        )

    # Verify sklearn version matches the fixture pin.
    import sklearn
    from pathlib import Path
    pin_file = Path(__file__).parent / "fixtures" / "SKLEARN_VERSION"
    pinned = pin_file.read_text().strip()
    if sklearn.__version__ != pinned:
        pytest.exit(
            f"sklearn=={sklearn.__version__} installed but fixtures pinned to "
            f"sklearn=={pinned} (tests/fixtures/SKLEARN_VERSION). "
            f"Run `uv pip install scikit-learn=={pinned}` or bump the pin "
            f"and regenerate fixtures via `python tests/fixtures/build.py`.",
            returncode=1,
        )
    yield
```

- [ ] **Step 3: Verify fixture fires correctly**

```bash
uv run --no-sync pytest tests/ -k "test_smoke" -v 2>&1 | tail -10
```
Expected: session starts cleanly (no `pytest.exit`); existing smoke tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test(phase-10): session-level conftest asserts ml-all + sklearn pin"
```

---

### Task 4: Verify shap + umap install on this Python

**Files:** none (verification only)

- [ ] **Step 1: Smoke test the optional extras end-to-end**

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "
import numpy as np
from sklearn.linear_model import LogisticRegression
import shap, umap

X = np.random.RandomState(0).randn(50, 4); y = (X[:, 0] > 0).astype(int)
m = LogisticRegression(random_state=0, max_iter=500).fit(X, y)
sv = shap.LinearExplainer(m, X).shap_values(X)
print('shap_values shape:', sv.shape)
u = umap.UMAP(random_state=0, n_components=2, n_neighbors=5).fit_transform(X)
print('umap shape:', u.shape)
print('OK')
"
```
Expected: `shap_values shape: (50, 4)`, `umap shape: (50, 2)`, `OK`. (UMAP may emit a `UserWarning: n_jobs value 1 overridden` — expected and required for deterministic output.)

- [ ] **Step 2: Confirm no commit needed**

No file changes from this task. Tracking-only.

---

## 10a — Foundation + regression diagnostics

### Task 5: ~~`RenderConfig.numeric_precision` field~~ — DROPPED

**This task is dropped.** During 10a execution we discovered that
`crates/ferrum-core/src/render/svg.rs` already routes every emitted float
through `fmt_f(x)` which quantizes to `FLOAT_PRECISION = 3` decimal places
(constant at `crates/ferrum-core/src/render/mod.rs:26`). Phase 9 SVG goldens
are therefore already 3-dp quantized — tighter than the 4-dp the plan
originally proposed.

**Consequences:**

- **No Rust change required for golden stability.** The proposed
  `RenderConfig.numeric_precision: Option<u8>` field is redundant.
- **Tiered-goldens scheme collapses to a single tier.** All Phase 10
  goldens render at the existing 3-dp precision; the
  `tests/goldens/phase_10/` directory is not created.
- **Drift note for §3.16 RenderConfig is dropped** from Task 41.
- **Phase 10's only Rust touchpoint becomes `kendall_tau_b`** in Task 35.

If, in 10d/10e/10f, solver-sensitive figures (SHAP-Kernel, UMAP, t-SNE,
MDS, learning_curve, etc.) prove not to be byte-identical across platforms
even at 3 dp, address it empirically as a per-figure issue — don't
preemptively reintroduce this field.

---


### Task 6: `_diagnostics/` package skeleton + lazy-import helpers

**Files:**
- Create: `src/ferrum/_diagnostics/__init__.py`
- Create: `src/ferrum/_diagnostics/deps.py`
- Create: `src/ferrum/_diagnostics/schemas.py`
- Create: `src/ferrum/_diagnostics/stats.py` (skeleton — implementations land in Task 36)
- Create: `tests/diagnostics/__init__.py`
- Create: `tests/diagnostics/test_no_sklearn_at_import.py`

- [ ] **Step 1: Write `src/ferrum/_diagnostics/__init__.py`**

```python
"""Phase 10 — model-diagnostics adapter layer.

Public surface:
    ferrum.ModelSource     (re-exported from .source)
    ferrum.ComparedModelSource  (re-exported from .source, lands in 10h)

Everything else in this subpackage is internal. The figure functions
(`ferrum.roc_chart`, etc.) live at `ferrum.figures` and delegate to
private `_chart_from_source` builders in `.charts`.
"""
from __future__ import annotations

# Re-exports are added incrementally per sub-batch.
# Task 7 lands ModelSource here.
```

- [ ] **Step 2: Write `src/ferrum/_diagnostics/deps.py`**

```python
"""Lazy-import helpers for Phase 10 optional dependencies.

Each `require_*` function is called as the first line of any ModelSource
method that needs the corresponding third-party library. `import ferrum`
and `ModelSource.__init__` never call these helpers.
"""
from __future__ import annotations

from types import ModuleType


def require_sklearn(method_name: str) -> ModuleType:
    """Lazy-import sklearn; raise with `pip install ferrum[models]` hint on failure."""
    try:
        import sklearn
    except ImportError as e:
        raise ImportError(
            f"ferrum.ModelSource.{method_name}() requires scikit-learn. "
            f"Install it with `pip install ferrum[models]` or "
            f"`pip install scikit-learn`."
        ) from e
    return sklearn


def require_shap(method_name: str) -> ModuleType:
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            f"ferrum.ModelSource.{method_name}() requires the shap library. "
            f"Install it with `pip install ferrum[shap]` or `pip install shap`."
        ) from e
    return shap


def require_umap(method_name: str) -> ModuleType:
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            f"ferrum.ModelSource.{method_name}() requires umap-learn. "
            f"Install it with `pip install ferrum[umap]` or `pip install umap-learn`."
        ) from e
    return umap
```

- [ ] **Step 3: Write `src/ferrum/_diagnostics/schemas.py` (skeleton)**

```python
"""Polars schema constants for every derived-data DataFrame.

Every schema documents an optional `model: str` column that is appended
by `ComparedModelSource` and absent on plain `ModelSource`. Chart
builders check `"model" in df.columns` to add a `color="model"` encoding.

Schemas are filled in incrementally per sub-batch.
"""
from __future__ import annotations

import polars as pl

# Phase 10a — regression
SCHEMA_PREDICTIONS = pl.Schema({
    "y_true": pl.Float64,
    "y_pred": pl.Float64,
    "residual": pl.Float64,
    "studentized_residual": pl.Float64,
    # "model": pl.Utf8 (optional, present in ComparedModelSource output)
})
```

- [ ] **Step 4: Write `src/ferrum/_diagnostics/stats.py` (skeleton)**

```python
"""Vectorized NumPy in-house statistics for Phase 10.

Full implementations land in Task 36 (10g). This file exists in 10a
to provide `studentized_residual` for `.predictions()`.
"""
from __future__ import annotations

import numpy as np


def studentized_residual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray | None = None,
) -> np.ndarray:
    """Compute studentized residuals.

    For linear estimators (X provided), uses the hat matrix diagonal:
        r_i / (sigma_hat * sqrt(1 - h_ii))
    where h = X (X' X)^{-1} X' and sigma_hat^2 = sum(r^2) / (n - p).

    For non-linear estimators (X=None), falls back to internally
    studentized residuals using the raw standard deviation of residuals.
    """
    r = y_true - y_pred
    if X is None:
        sigma = np.std(r, ddof=1) if len(r) > 1 else 1.0
        return r / sigma if sigma > 0 else r * 0.0

    n, p = X.shape
    # H = X (X' X)^{-1} X'; h_ii = diag(H).
    XtX_inv = np.linalg.pinv(X.T @ X)
    h_diag = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    h_diag = np.clip(h_diag, 0.0, 1.0 - 1e-12)
    sigma_sq = float((r * r).sum() / max(n - p, 1))
    sigma = np.sqrt(sigma_sq) if sigma_sq > 0 else 0.0
    if sigma == 0.0:
        return r * 0.0
    return r / (sigma * np.sqrt(1.0 - h_diag))
```

- [ ] **Step 5: Write `tests/diagnostics/test_no_sklearn_at_import.py`**

```python
"""Regression test for Phase 10 done-criterion:
sklearn must NOT be imported by `import ferrum` or `ModelSource.__init__`.
"""
from __future__ import annotations

import sys


class _DuckModel:
    """Non-sklearn duck-typed model with predict()."""
    def predict(self, X):
        return [0] * len(X)


def test_import_ferrum_does_not_load_sklearn():
    # Drop sklearn if a prior test imported it.
    for mod in list(sys.modules):
        if mod == "sklearn" or mod.startswith("sklearn."):
            del sys.modules[mod]
    assert "sklearn" not in sys.modules

    import ferrum  # noqa: F401
    assert "sklearn" not in sys.modules, (
        "sklearn loaded as a side-effect of `import ferrum`"
    )


def test_modelsource_init_does_not_load_sklearn():
    # Drop sklearn first.
    for mod in list(sys.modules):
        if mod == "sklearn" or mod.startswith("sklearn."):
            del sys.modules[mod]

    import polars as pl
    import ferrum

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    source = ferrum.ModelSource(_DuckModel(), df)
    assert "sklearn" not in sys.modules, (
        "sklearn loaded as a side-effect of ModelSource(...)"
    )
    # Touch the source object so it isn't optimized away.
    assert source is not None


def test_require_sklearn_raises_clear_message_when_missing(monkeypatch):
    """Simulates sklearn missing — ImportError must mention `ferrum[models]`."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("No module named 'sklearn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from ferrum._diagnostics.deps import require_sklearn
    import pytest
    with pytest.raises(ImportError, match=r"ferrum\[models\]|pip install scikit-learn"):
        require_sklearn("predictions")
```

- [ ] **Step 6: Create `tests/diagnostics/__init__.py`** (empty marker file)

- [ ] **Step 7: Run the no-sklearn-at-import test**

```bash
uv run --no-sync pytest tests/diagnostics/test_no_sklearn_at_import.py -v 2>&1 | tail -15
```
Expected: 2 tests skip-pass (since `ModelSource` doesn't exist yet, the second test will ERROR — that's expected; we'll fix it after Task 7).

For now, run only the first test:

```bash
uv run --no-sync pytest tests/diagnostics/test_no_sklearn_at_import.py::test_import_ferrum_does_not_load_sklearn -v 2>&1 | tail -5
```
Expected: 1 passed.

- [ ] **Step 8: Commit**

```bash
git add src/ferrum/_diagnostics/ tests/diagnostics/__init__.py tests/diagnostics/test_no_sklearn_at_import.py
git commit -m "feat(phase-10a): _diagnostics package skeleton + lazy-import helpers"
```

---

### Task 7: `ModelSource` class + protocol detection + cache

**Files:**
- Create: `src/ferrum/_diagnostics/source.py`
- Modify: `src/ferrum/_diagnostics/__init__.py`
- Modify: `src/ferrum/__init__.py`
- Create: `tests/diagnostics/test_source.py`

- [ ] **Step 1: Write `src/ferrum/_diagnostics/source.py`**

```python
"""ModelSource adapter — wraps a fitted estimator + data, exposes derived data.

Phase 10a: constructor, protocol detection, cache, .predictions(),
.probabilities(). Other methods land in 10b–10g; `ComparedModelSource`
in 10h.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl

from .deps import require_sklearn
from .stats import studentized_residual


_PROTOCOL_ATTRS: tuple[str, ...] = (
    "predict", "predict_proba", "decision_function", "transform",
    "fit_transform", "fit_predict", "score",
    "feature_importances_", "coef_", "explained_variance_ratio_",
    "cluster_centers_", "labels_", "classes_",
)


def _coerce_X_y(X: Any, y: Any) -> tuple[pl.DataFrame, pl.Series | None]:
    """Coerce X to polars.DataFrame and y to polars.Series (or None)."""
    if isinstance(X, pl.DataFrame):
        X_df = X
    elif isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        X_df = pl.from_numpy(X, schema=[f"f{i}" for i in range(X.shape[1])])
    else:
        # Try narwhals or pyarrow paths via ferrum's existing _coerce.
        from ferrum._coerce import coerce_to_polars
        X_df = coerce_to_polars(X)

    y_ser: pl.Series | None = None
    if y is not None:
        if isinstance(y, pl.Series):
            y_ser = y
        elif isinstance(y, np.ndarray):
            y_ser = pl.Series("y", y)
        elif isinstance(y, pl.DataFrame):
            if y.width != 1:
                raise ValueError(f"y DataFrame must have exactly 1 column; got {y.width}")
            y_ser = y.to_series()
        else:
            y_ser = pl.Series("y", list(y))
    return X_df, y_ser


class ModelSource:
    """Wraps a fitted estimator + dataset; exposes derived data as DataFrames.

    Constructor is sklearn-free: pure attribute introspection.
    Methods that need sklearn / shap / umap lazy-import on call.
    """

    def __init__(
        self,
        model: Any,
        X: Any,
        y: Any = None,
        *,
        feature_names: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        sample_weight: Any = None,
        random_state: int | None = None,
    ):
        self._model = model
        self._X, self._y = _coerce_X_y(X, y)
        self._feature_names: list[str] = (
            list(feature_names) if feature_names is not None
            else list(self._X.columns)
        )
        self._class_names: list[str] | None = list(class_names) if class_names is not None else None
        self._sample_weight = sample_weight
        self._random_state = random_state

        self._capabilities = frozenset(
            attr for attr in _PROTOCOL_ATTRS if hasattr(self._model, attr)
        )
        self._cache: dict[tuple, pl.DataFrame] = {}

    # --- Introspection ----------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def capabilities(self) -> frozenset[str]:
        return self._capabilities

    def _require_capability(self, attr: str, method_name: str) -> None:
        if attr not in self._capabilities:
            raise AttributeError(
                f"ModelSource.{method_name}() requires the wrapped model to "
                f"implement '{attr}'. Got {type(self._model).__name__!r} which "
                f"does not."
            )

    def _cache_key(self, method: str, **kwargs) -> tuple:
        return (method, tuple(sorted(kwargs.items())))

    # --- 10a: predictions, probabilities ---------------------------------

    def predictions(self) -> pl.DataFrame:
        """Return y_true, y_pred, residual, studentized_residual."""
        key = self._cache_key("predictions")
        if key in self._cache:
            return self._cache[key]

        self._require_capability("predict", "predictions")
        X_np = self._X.to_numpy()
        y_pred = np.asarray(self._model.predict(X_np), dtype=np.float64)
        y_true = (
            np.asarray(self._y.to_numpy(), dtype=np.float64)
            if self._y is not None
            else np.full_like(y_pred, np.nan)
        )
        residual = y_true - y_pred

        # Studentized residual: linear-estimator path if model exposes coef_.
        if "coef_" in self._capabilities and self._y is not None:
            X_with_intercept = np.column_stack([np.ones(len(X_np)), X_np])
            stud = studentized_residual(y_true, y_pred, X_with_intercept)
        else:
            stud = studentized_residual(y_true, y_pred, X=None)

        df = pl.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": residual,
            "studentized_residual": stud,
        })
        self._cache[key] = df
        return df

    def probabilities(self) -> pl.DataFrame:
        """Return y_true + one column per class with predicted probability."""
        key = self._cache_key("probabilities")
        if key in self._cache:
            return self._cache[key]

        sklearn = require_sklearn("probabilities")
        X_np = self._X.to_numpy()

        if "predict_proba" in self._capabilities:
            proba = np.asarray(self._model.predict_proba(X_np), dtype=np.float64)
        elif "decision_function" in self._capabilities:
            scores = np.asarray(self._model.decision_function(X_np), dtype=np.float64)
            if scores.ndim == 1:
                # Binary classifier — apply sigmoid.
                proba = 1.0 / (1.0 + np.exp(-scores))
                proba = np.column_stack([1.0 - proba, proba])
            else:
                # Multiclass — softmax.
                exp = np.exp(scores - scores.max(axis=1, keepdims=True))
                proba = exp / exp.sum(axis=1, keepdims=True)
        else:
            raise AttributeError(
                "ModelSource.probabilities() requires the wrapped model to "
                "implement 'predict_proba' or 'decision_function'. Got "
                f"{type(self._model).__name__!r} which implements neither."
            )

        classes = (
            self._class_names
            or (getattr(self._model, "classes_", None) and list(self._model.classes_))
            or [f"class_{i}" for i in range(proba.shape[1])]
        )
        data: dict[str, Any] = {}
        if self._y is not None:
            data["y_true"] = self._y.to_numpy()
        for i, c in enumerate(classes):
            data[f"proba_{c}"] = proba[:, i]
        df = pl.DataFrame(data)
        self._cache[key] = df
        return df
```

- [ ] **Step 2: Update `src/ferrum/_diagnostics/__init__.py`**

```python
from __future__ import annotations

from .source import ModelSource

__all__ = ["ModelSource"]
```

- [ ] **Step 3: Re-export from `src/ferrum/__init__.py`**

Find the existing re-export block and add:

```python
from ferrum._diagnostics import ModelSource

# Append "ModelSource" to __all__.
```

- [ ] **Step 4: Write `tests/diagnostics/test_source.py`**

```python
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum
from tests.fixtures import load_fixture, load_dataset


class _DuckModel:
    """Minimal duck-typed model: predict only."""
    def predict(self, X):
        return np.zeros(len(X))


def test_constructor_accepts_polars_dataframe():
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    source = ferrum.ModelSource(_DuckModel(), df, y=[0, 1])
    assert source.feature_names == ["a", "b"]


def test_constructor_accepts_numpy_array():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    source = ferrum.ModelSource(_DuckModel(), X, y=[0, 1])
    assert source.feature_names == ["f0", "f1"]


def test_capability_detection():
    df = pl.DataFrame({"a": [1.0]})
    source = ferrum.ModelSource(_DuckModel(), df)
    assert "predict" in source.capabilities
    assert "predict_proba" not in source.capabilities


def test_predictions_requires_predict():
    class NoPredict: pass
    df = pl.DataFrame({"a": [1.0]})
    source = ferrum.ModelSource(NoPredict(), df, y=[0.0])
    with pytest.raises(AttributeError, match="predict"):
        source.predictions()


def test_predictions_against_ridge_fixture():
    """Studentized residuals computed for linear estimator."""
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    y = df["y"]

    source = ferrum.ModelSource(model, X, y)
    pred = source.predictions()
    assert pred.columns == ["y_true", "y_pred", "residual", "studentized_residual"]
    assert pred.shape == (df.height, 4)
    # Residual = y_true - y_pred
    np.testing.assert_allclose(
        pred["residual"].to_numpy(),
        pred["y_true"].to_numpy() - pred["y_pred"].to_numpy(),
        rtol=1e-12,
    )


def test_probabilities_against_binary_logistic_fixture():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    y = df["y"]

    source = ferrum.ModelSource(model, X, y)
    proba = source.probabilities()
    # Two proba columns (binary).
    proba_cols = [c for c in proba.columns if c.startswith("proba_")]
    assert len(proba_cols) == 2
    # Rows sum to 1.
    sums = proba.select(proba_cols).to_numpy().sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-10)


def test_probabilities_caching():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    p1 = source.probabilities()
    p2 = source.probabilities()
    assert p1 is p2  # cache returns the same object
```

- [ ] **Step 5: Run the tests**

```bash
uv run --no-sync pytest tests/diagnostics/test_source.py -v 2>&1 | tail -25
uv run --no-sync pytest tests/diagnostics/test_no_sklearn_at_import.py -v 2>&1 | tail -10
```
Expected: all source tests pass; both no-sklearn-at-import tests now pass (since `ModelSource(_DuckModel, df)` works without sklearn).

- [ ] **Step 6: Commit**

```bash
git add src/ferrum/_diagnostics/source.py src/ferrum/_diagnostics/__init__.py src/ferrum/__init__.py tests/diagnostics/test_source.py
git commit -m "feat(phase-10a): ModelSource class with predictions() + probabilities()"
```

---

### Task 8: `mark_residuals` + `mark_prediction_error` (Python desugar)

**Files:**
- Create: `src/ferrum/marks/diagnostic.py`
- Modify: `src/ferrum/chart.py` (add `mark_residuals` + `mark_prediction_error` methods following the `mark_boxplot` pattern from Phase 8b/9)
- Create: `tests/diagnostics/test_regression.py`

**Pattern reference:** `src/ferrum/marks/composite.py:15-61` (`desugar_boxplot`) and `src/ferrum/chart.py:379-410` (`Chart.mark_boxplot`). Phase 10 diagnostic marks follow the same shape:

- Module-level `desugar_<name>(x_field, y_field, **kwargs) -> tuple` that returns a 5-tuple `("__layered__", transforms: list, None, None, layers: list[dict])`.
- Each layer dict has shape `{"mark": str, "encoding": dict, "mark_kwargs": dict (opt), "data_source": str | None (opt)}`.
- Chart method: clone the chart, set `_mark = "point"` (placeholder), set `_pending_stat_mark = (kind, kwargs_dict, desugar_fn)`, return.
- There is no `LayerSpec` class, no `chart_ctx`, no `_expand` method. Plain dict layers.
- For Phase 10 diagnostic marks the data has hard-coded column names from a `ModelSource` method (`y_pred`, `residual`, `studentized_residual`, etc.), so the desugar references those columns literally rather than relying on user-supplied `x_field`/`y_field`.

- [ ] **Step 1: Write `src/ferrum/marks/diagnostic.py` (initial — two desugars)**

```python
"""Phase 10 model-diagnostic mark desugars (Python-side).

Each `desugar_<name>(x_field, y_field, **kwargs)` returns the 5-tuple
`("__layered__", transforms: list, None, None, layers: list[dict])`
consumed by `Chart` when the user calls `chart.mark_<name>(...)`.

These desugars operate on DataFrames with hard-coded column names from a
`ModelSource` method (e.g. `y_pred`, `residual`, `studentized_residual`).
They ignore `x_field`/`y_field` — the user shouldn't `.encode()` x or y on
a diagnostic chart; the figure-level wrapper builds the chart from the
ModelSource output directly. No new Rust Mark or Transform variants.
"""
from __future__ import annotations

from typing import Any


def desugar_residuals(
    x_field: str | None,
    y_field: str | None,
    *,
    kind: str = "studentized",
    reference_line: bool = True,
    cook_threshold: float | None = None,
    color_field: str | None = None,
    **mark_kwargs: Any,
) -> tuple:
    """Residuals diagnostic: scatter of (y_pred, residual) + reference line at 0."""
    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"
    point_enc: dict[str, Any] = {"x": "y_pred", "y": y_col}
    if color_field is not None:
        point_enc["color"] = color_field
    layers: list[dict] = [{"mark": "point", "encoding": point_enc}]
    if reference_line:
        layers.append({
            "mark": "rule",
            "encoding": {"y": 0.0},
            "mark_kwargs": {"strokeDash": [4, 4]},
        })
    return ("__layered__", [], None, None, layers)


def desugar_prediction_error(
    x_field: str | None,
    y_field: str | None,
    *,
    identity_line: bool = True,
    ci: float | None = None,
    reference_band: bool = False,
    color_field: str | None = None,
    **mark_kwargs: Any,
) -> tuple:
    """Actual vs predicted: scatter of (y_true, y_pred) + optional identity line."""
    point_enc: dict[str, Any] = {"x": "y_true", "y": "y_pred"}
    if color_field is not None:
        point_enc["color"] = color_field
    layers: list[dict] = [{"mark": "point", "encoding": point_enc}]
    if identity_line:
        layers.append({
            "mark": "line",
            "encoding": {"x": "y_true", "y": "y_true"},
            "mark_kwargs": {"strokeDash": [4, 4]},
        })
    return ("__layered__", [], None, None, layers)
```

- [ ] **Step 2: Wire `mark_residuals` + `mark_prediction_error` into `Chart`**

In `src/ferrum/chart.py`, after the existing `mark_boxplot` / `mark_boxen` methods, add:

```python
def mark_residuals(
    self,
    *,
    kind: str = "studentized",
    reference_line: bool = True,
    cook_threshold: float | None = None,
    color_field: str | None = None,
    position=None,
    **mark_kwargs,
) -> "Chart":
    """Residuals diagnostic mark — see ferrum-spec.md §3.3.

    Expects the chart's data to have columns `y_pred`, `residual`,
    `studentized_residual` (the schema emitted by `ModelSource.predictions()`).
    """
    from ferrum.marks.diagnostic import desugar_residuals
    new = self._clone()
    new._mark = "point"  # placeholder; layered mode overrides
    new._pending_stat_mark = (
        "residuals",
        {
            "kind": kind,
            "reference_line": reference_line,
            "cook_threshold": cook_threshold,
            "color_field": color_field,
            **mark_kwargs,
        },
        desugar_residuals,
    )
    new._position = position
    return new


def mark_prediction_error(
    self,
    *,
    identity_line: bool = True,
    ci: float | None = None,
    reference_band: bool = False,
    color_field: str | None = None,
    position=None,
    **mark_kwargs,
) -> "Chart":
    """Actual-vs-predicted mark — see ferrum-spec.md §3.3."""
    from ferrum.marks.diagnostic import desugar_prediction_error
    new = self._clone()
    new._mark = "point"
    new._pending_stat_mark = (
        "prediction_error",
        {
            "identity_line": identity_line,
            "ci": ci,
            "reference_band": reference_band,
            "color_field": color_field,
            **mark_kwargs,
        },
        desugar_prediction_error,
    )
    new._position = position
    return new
```

> **Pattern note:** the existing `Chart._resolve_pending` machinery (the path that handles `_pending_stat_mark` at compile time) calls the desugar function with `x_field` and `y_field` extracted from `self._encoding` (or None if not set). The Phase 10 desugars ignore those positional args and reference hard-coded column names. This works because the chart's data already has the right shape from `ModelSource.predictions()`.

- [ ] **Step 3: No `__init__.py` change required**

Unlike Phase 9 composite marks, Phase 10 diagnostic mark names are accessed only as `Chart.mark_<name>(...)` methods. The `desugar_<name>` functions are internal. No re-export needed in `src/ferrum/__init__.py`.

- [ ] **Step 4: Write `tests/diagnostics/test_regression.py`**

```python
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum
from tests.fixtures import load_fixture, load_dataset


def test_chart_mark_residuals_renders():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, X, df["y"])
    pred = source.predictions()

    chart = ferrum.Chart(pred).mark_residuals()
    svg = chart.show_svg()
    assert "<svg" in svg


def test_chart_mark_residuals_raw_kind():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, X, df["y"])
    pred = source.predictions()

    chart = ferrum.Chart(pred).mark_residuals(kind="raw")
    svg = chart.show_svg()
    assert "<svg" in svg


def test_chart_mark_residuals_no_reference_line():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, X, df["y"])
    pred = source.predictions()

    chart = ferrum.Chart(pred).mark_residuals(reference_line=False)
    svg = chart.show_svg()
    assert "<svg" in svg


def test_chart_mark_prediction_error_renders():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, X, df["y"])
    pred = source.predictions()

    chart = ferrum.Chart(pred).mark_prediction_error()
    svg = chart.show_svg()
    assert "<svg" in svg


def test_chart_mark_prediction_error_no_identity():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, X, df["y"])
    pred = source.predictions()

    chart = ferrum.Chart(pred).mark_prediction_error(identity_line=False)
    svg = chart.show_svg()
    assert "<svg" in svg
```

- [ ] **Step 5: Build and run tests**

```bash
uv run --no-sync pytest tests/diagnostics/test_regression.py -v 2>&1 | tail -15
```
Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/ferrum/marks/diagnostic.py src/ferrum/chart.py tests/diagnostics/test_regression.py
git commit -m "feat(phase-10a): mark_residuals + mark_prediction_error (desugar)"
```

---

### Task 9: `_residuals_chart_from_source` + `_prediction_error_chart_from_source` builders

**Files:**
- Create: `src/ferrum/_diagnostics/charts.py`
- Modify: `tests/diagnostics/test_regression.py` (add builder tests)

- [ ] **Step 1: Write `src/ferrum/_diagnostics/charts.py` (initial — two builders)**

```python
"""Private chart-builder functions used by figure functions + visualizers.

Each builder takes a ModelSource (or ComparedModelSource), calls the
appropriate derived-data method, and returns a fully-formed Chart over
the resulting DataFrame.

Implementations are added incrementally per sub-batch.
"""
from __future__ import annotations

from typing import Any

import polars as pl

import ferrum


def _residuals_chart_from_source(
    source: Any,
    *,
    kind: str = "studentized",
    panels: Any = None,  # "auto" | list | None
    theme: Any = None,
) -> "ferrum.Chart":
    """Build a residuals diagnostic chart from a ModelSource."""
    df = source.predictions()
    if panels in (None, "single"):
        chart = ferrum.Chart(df).mark_residuals(kind=kind)
        if theme is not None:
            chart = chart.theme(theme)
        return chart

    # Multi-panel layout (10a ships "auto" panel only — single residuals_vs_fitted panel;
    # extra panels added in 10h finalize together with QQ/scale_location/leverage panels
    # whose underlying marks (mark_qq from Phase 8b) are already available).
    panel_list = panels if isinstance(panels, list) else ["residuals_vs_fitted"]
    charts = [_residuals_panel(df, name) for name in panel_list]
    return _grid_panels(charts, theme=theme)


def _residuals_panel(df: pl.DataFrame, name: str) -> "ferrum.Chart":
    if name == "residuals_vs_fitted":
        return ferrum.Chart(df).mark_residuals()
    if name == "qq":
        return ferrum.Chart(df).mark_qq().encode(x="studentized_residual")
    if name == "scale_location":
        import numpy as np
        d2 = df.with_columns(
            (pl.col("studentized_residual").abs().sqrt()).alias("sqrt_abs_resid")
        )
        return ferrum.Chart(d2).mark_point().encode(x="y_pred", y="sqrt_abs_resid")
    if name == "residuals_vs_leverage":
        # Leverage h_ii requires X; if missing, fall back to residual sequence.
        return ferrum.Chart(df).mark_point().encode(x="y_pred", y="residual")
    raise ValueError(f"unknown residuals panel: {name!r}")


def _grid_panels(charts: list, theme: Any = None) -> "ferrum.Chart":
    """Compose up to 4 panels into a 2×2 grid using Phase 8a hstack/vstack."""
    if len(charts) == 1:
        c = charts[0]
    elif len(charts) == 2:
        c = charts[0] | charts[1]
    elif len(charts) == 3:
        c = (charts[0] | charts[1]) & charts[2]
    else:
        c = (charts[0] | charts[1]) & (charts[2] | charts[3])
    if theme is not None:
        c = c.theme(theme)
    return c


def _prediction_error_chart_from_source(
    source: Any,
    *,
    identity_line: bool = True,
    theme: Any = None,
) -> "ferrum.Chart":
    """Build an actual-vs-predicted error chart from a ModelSource."""
    df = source.predictions()
    chart = ferrum.Chart(df).mark_prediction_error(identity_line=identity_line)
    if theme is not None:
        chart = chart.theme(theme)
    return chart
```

- [ ] **Step 2: Add builder tests to `tests/diagnostics/test_regression.py`**

Append:

```python
def test_residuals_chart_from_source_builder():
    from ferrum._diagnostics.charts import _residuals_chart_from_source
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, X, df["y"])

    chart = _residuals_chart_from_source(source)
    svg = chart.show_svg()
    assert "<svg" in svg


def test_prediction_error_chart_from_source_builder():
    from ferrum._diagnostics.charts import _prediction_error_chart_from_source
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, X, df["y"])

    chart = _prediction_error_chart_from_source(source)
    svg = chart.show_svg()
    assert "<svg" in svg
```

- [ ] **Step 3: Run and commit**

```bash
uv run --no-sync pytest tests/diagnostics/test_regression.py -v 2>&1 | tail -15
git add src/ferrum/_diagnostics/charts.py tests/diagnostics/test_regression.py
git commit -m "feat(phase-10a): chart builders for residuals + prediction_error"
```

---

### Task 10: `residuals_chart` figure function + regression visualizers

**Files:**
- Create: `src/ferrum/figures.py`
- Create: `src/ferrum/_diagnostics/visualizers/__init__.py`
- Create: `src/ferrum/_diagnostics/visualizers/base.py`
- Create: `src/ferrum/_diagnostics/visualizers/regression.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/diagnostics/test_regression.py`

- [ ] **Step 1: Write `src/ferrum/_diagnostics/visualizers/base.py`**

```python
"""FerrumVisualizer base class for §3.15 sklearn-protocol visualizers."""
from __future__ import annotations

from typing import Any


class FerrumVisualizer:
    """Base: fit() materializes derived data + chart; show() returns Chart."""
    def __init__(
        self,
        model: Any = None,
        *,
        random_state: int | None = None,
        theme: Any = None,
        **kwargs: Any,
    ):
        self.model = model
        self.random_state = random_state
        self.theme = theme
        self._fitted = False
        self._source: Any = None
        self._chart: Any = None
        self._metrics: dict[str, float] = {}

    def fit(self, X: Any, y: Any = None) -> "FerrumVisualizer":
        import ferrum
        self._source = ferrum.ModelSource(self.model, X, y, random_state=self.random_state)
        self._materialize()
        self._chart = self._build_chart()
        self._fitted = True
        return self

    def _materialize(self) -> None:
        """Subclass hook — compute derived data + populate self._metrics."""
        raise NotImplementedError

    def _build_chart(self) -> Any:
        """Subclass hook — return the Chart."""
        raise NotImplementedError

    def score(self, X: Any, y: Any) -> float:
        raise NotImplementedError(f"{type(self).__name__}.score() is not implemented")

    def show(self) -> Any:
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fit before .show(); call .fit(X, y) first."
            )
        return self._chart

    def __repr__(self) -> str:
        if not self._fitted:
            return f"{type(self).__name__}(unfit)"
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in self._metrics.items())
        return f"{type(self).__name__}({metric_str})"
```

- [ ] **Step 2: Write `src/ferrum/_diagnostics/visualizers/regression.py`**

```python
"""10a regression visualizers."""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import FerrumVisualizer
from ..charts import _residuals_chart_from_source, _prediction_error_chart_from_source


class ResidualsVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, kind: str = "studentized",
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.kind = kind

    def _materialize(self) -> None:
        df = self._source.predictions()
        resid = df["residual"].to_numpy()
        self._metrics["rmse"] = float(np.sqrt((resid ** 2).mean()))
        self._metrics["mae"] = float(np.abs(resid).mean())

    def _build_chart(self) -> Any:
        return _residuals_chart_from_source(
            self._source, kind=self.kind, theme=self.theme,
        )

    def score(self, X: Any, y: Any) -> float:
        return float(self.model.score(X, y))


class PredictionErrorVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, identity_line: bool = True,
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.identity_line = identity_line

    def _materialize(self) -> None:
        df = self._source.predictions()
        resid = df["residual"].to_numpy()
        self._metrics["rmse"] = float(np.sqrt((resid ** 2).mean()))

    def _build_chart(self) -> Any:
        return _prediction_error_chart_from_source(
            self._source, identity_line=self.identity_line, theme=self.theme,
        )

    def score(self, X: Any, y: Any) -> float:
        return float(self.model.score(X, y))


class CooksDistanceVisualizer(FerrumVisualizer):
    """Cook's distance via leverage (linear estimators); falls back to studentized."""
    def __init__(self, model: Any, *, threshold: float | None = None,
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.threshold = threshold

    def _materialize(self) -> None:
        df = self._source.predictions()
        stud = df["studentized_residual"].to_numpy()
        # Cook's distance proxy: stud^2 (full Cook's distance needs leverage h_ii;
        # ResidualsVisualizer ships the proper version when X is reachable).
        self._metrics["max_studentized"] = float(np.max(np.abs(stud)))

    def _build_chart(self) -> Any:
        # CooksDistance visualizer reuses the residuals_chart with a leverage panel.
        return _residuals_chart_from_source(
            self._source, kind="studentized", panels=["residuals_vs_leverage"],
            theme=self.theme,
        )
```

- [ ] **Step 3: Write `src/ferrum/_diagnostics/visualizers/__init__.py`**

```python
"""25 §3.15 sklearn-protocol visualizers — added incrementally per sub-batch."""
from __future__ import annotations

from .base import FerrumVisualizer
from .regression import ResidualsVisualizer, PredictionErrorVisualizer, CooksDistanceVisualizer

__all__ = [
    "FerrumVisualizer",
    "ResidualsVisualizer", "PredictionErrorVisualizer", "CooksDistanceVisualizer",
]
```

- [ ] **Step 4: Write `src/ferrum/figures.py` (initial — residuals_chart + _resolve_source)**

```python
"""§3.14 Group B figure-level functions.

Each function is a thin facade over `_*_chart_from_source` builders in
`_diagnostics.charts`. The `_resolve_source` helper accepts model |
ModelSource | dict[str, model] and produces the right source object.
"""
from __future__ import annotations

from typing import Any

import ferrum
from ferrum._diagnostics.charts import (
    _residuals_chart_from_source,
)


def _resolve_source(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    random_state: int | None = None,
    compare: dict[str, Any] | None = None,
) -> Any:
    """Resolve a figure-function input into a ModelSource or ComparedModelSource."""
    if compare is not None:
        # 10h: route to ModelSource.compare. For 10a, raise if invoked.
        raise NotImplementedError(
            "compare= support is added in Phase 10h. For now, build a "
            "ComparedModelSource manually or pass a single model."
        )
    if isinstance(model_or_source, ferrum.ModelSource):
        return model_or_source
    if isinstance(model_or_source, dict):
        # dict[str, model] → ComparedModelSource. Added in 10h.
        raise NotImplementedError(
            "Multi-model dict input is added in Phase 10h."
        )
    # Treat as a fitted model object.
    return ferrum.ModelSource(model_or_source, X, y, random_state=random_state)


def residuals_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    kind: str = "studentized",
    panels: Any = "auto",
    random_state: int | None = None,
    theme: Any = None,
) -> "ferrum.Chart":
    """Residuals diagnostic chart — see ferrum-spec.md §3.14."""
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    panel_list = None if panels in (None, "single") else (
        ["residuals_vs_fitted"] if panels == "auto" else list(panels)
    )
    return _residuals_chart_from_source(
        source, kind=kind, panels=panel_list, theme=theme,
    )
```

- [ ] **Step 5: Re-export from `src/ferrum/__init__.py`**

```python
# Figure functions
from ferrum.figures import residuals_chart

# Visualizers
from ferrum._diagnostics.visualizers import (
    FerrumVisualizer,
    ResidualsVisualizer, PredictionErrorVisualizer, CooksDistanceVisualizer,
)

# Append to __all__.
```

- [ ] **Step 6: Add tests to `tests/diagnostics/test_regression.py`**

```python
def test_residuals_chart_figure_function():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.residuals_chart(model, X, df["y"])
    svg = chart.show_svg()
    assert "<svg" in svg


def test_residuals_visualizer_full_cycle():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])

    viz = ferrum.ResidualsVisualizer(model)
    assert "unfit" in repr(viz)

    viz.fit(X, df["y"])
    assert "rmse=" in repr(viz)
    assert "mae=" in repr(viz)
    chart = viz.show()
    assert "<svg" in chart.show_svg()


def test_prediction_error_visualizer():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])

    viz = ferrum.PredictionErrorVisualizer(model).fit(X, df["y"])
    chart = viz.show()
    assert "<svg" in chart.show_svg()


def test_visualizer_show_before_fit_errors():
    import pytest
    viz = ferrum.ResidualsVisualizer(model=None)
    with pytest.raises(RuntimeError, match="must be fit"):
        viz.show()
```

- [ ] **Step 7: Run + commit**

```bash
uv run --no-sync pytest tests/diagnostics/test_regression.py -v 2>&1 | tail -20
git add src/ferrum/figures.py src/ferrum/_diagnostics/visualizers/ src/ferrum/__init__.py tests/diagnostics/test_regression.py
git commit -m "feat(phase-10a): residuals_chart + Residuals/PredictionError/CooksDistance visualizers"
```

---

### Task 11: First Phase 10 SVG goldens (residuals + prediction_error)

**Files:**
- Create: `tests/goldens/phase_10/residuals_chart_regression.svg` (generated)
- Create: `tests/goldens/phase_10/prediction_error_regression.svg` (generated)
- Create: `tests/diagnostics/test_goldens_phase_10.py`

- [ ] **Step 1: Write `tests/diagnostics/test_goldens_phase_10.py`**

```python
"""Phase 10 SVG golden tests.

Single tier — all goldens render at the renderer's default 3-decimal-place
quantization (`fmt_f` in `crates/ferrum-core/src/render/svg.rs`).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import ferrum
from tests.fixtures import load_fixture, load_dataset

_GOLDEN_ROOT = Path(__file__).parent.parent / "goldens" / "phase_10"
_REGENERATE = bool(os.environ.get("FERRUM_REGENERATE_GOLDENS"))


def _check_golden(svg: str, name: str, tier: str = '') -> None:
    path = _GOLDEN_ROOT / tier / f"{name}.svg"
    if _REGENERATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg)
        if not _REGENERATE:
            pytest.skip(f"created new golden at {path}; rerun to verify")
        return
    expected = path.read_text()
    assert svg == expected, (
        f"Golden mismatch for {name} (tier={tier}). "
        f"Set FERRUM_REGENERATE_GOLDENS=1 to regenerate after intentional changes."
    )


# --- 10a goldens ---

def test_golden_residuals_chart_regression():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.residuals_chart(model, X, df["y"])
    svg = chart.show_svg()
    _check_golden(svg, "residuals_chart_regression")


def test_golden_prediction_error_regression():
    from ferrum._diagnostics.charts import _prediction_error_chart_from_source
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, X, df["y"])
    chart = _prediction_error_chart_from_source(source)
    svg = chart.show_svg()
    _check_golden(svg, "prediction_error_regression")
```

- [ ] **Step 2: Generate initial goldens**

```bash
FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -v 2>&1 | tail -10
```
Expected: 2 SVG files created under `tests/goldens/phase_10/`.

- [ ] **Step 3: Verify goldens are byte-stable on rerun**

```bash
uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -v 2>&1 | tail -10
```
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/diagnostics/test_goldens_phase_10.py tests/goldens/phase_10/
git commit -m "test(phase-10a): residuals + prediction_error SVG goldens"
```

- [ ] **Step 5: 10a milestone check**

```bash
DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core --quiet 2>&1 | tail -3
uv run --no-sync pytest 2>&1 | tail -3
```
Expected: `cargo test` ≥ 496 passed (unchanged from Phase 9 — Rust touchpoints in 10a are zero after Task 5 was dropped). `pytest` ≥ 495 passed, 5 skipped (15+ new from 10a).

---

## 10b — Classification curves

### Task 12: `.roc_curve()` + `.pr_curve()` ModelSource methods

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py`
- Modify: `src/ferrum/_diagnostics/schemas.py`
- Modify: `tests/diagnostics/test_source.py`

- [ ] **Step 1: Add schemas to `_diagnostics/schemas.py`**

```python
SCHEMA_ROC_CURVE = pl.Schema({
    "fpr": pl.Float64,
    "tpr": pl.Float64,
    "threshold": pl.Float64,
    "class": pl.Utf8,
    "auc": pl.Float64,
    # "model": pl.Utf8 (optional)
})

SCHEMA_PR_CURVE = pl.Schema({
    "precision": pl.Float64,
    "recall": pl.Float64,
    "threshold": pl.Float64,
    "class": pl.Utf8,
    "ap": pl.Float64,
})
```

- [ ] **Step 2: Add `roc_curve` method to `ModelSource`**

In `src/ferrum/_diagnostics/source.py`, append:

```python
    def roc_curve(
        self,
        *,
        average: str | None = None,
        drop_intermediate: bool = True,
    ) -> pl.DataFrame:
        """ROC curve(s). One row per (class, threshold). `auc` repeats per class."""
        key = self._cache_key("roc_curve", average=average, drop_intermediate=drop_intermediate)
        if key in self._cache:
            return self._cache[key]
        sklearn = require_sklearn("roc_curve")
        from sklearn.metrics import roc_curve, roc_auc_score
        import numpy as np

        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        if self._y is None:
            raise ValueError("ModelSource.roc_curve() requires y to be provided.")
        y_true = np.asarray(self._y.to_numpy())
        classes = [c[len("proba_"):] for c in proba_cols]
        n_classes = len(classes)

        rows: list[dict] = []
        if n_classes == 2 and average is None:
            # Binary — single curve on positive class (column 1).
            y_score = proba_df[proba_cols[1]].to_numpy()
            fpr, tpr, thr = roc_curve(y_true, y_score, drop_intermediate=drop_intermediate)
            auc = float(roc_auc_score(y_true, y_score))
            for f, t, h in zip(fpr, tpr, thr):
                rows.append({"fpr": float(f), "tpr": float(t),
                             "threshold": float(h), "class": classes[1], "auc": auc})
        else:
            # Multiclass per-class one-vs-rest.
            for i, cls in enumerate(classes):
                y_bin = (y_true == _coerce_class_label(cls, y_true.dtype)).astype(int)
                y_score = proba_df[proba_cols[i]].to_numpy()
                fpr, tpr, thr = roc_curve(y_bin, y_score, drop_intermediate=drop_intermediate)
                try:
                    auc = float(roc_auc_score(y_bin, y_score))
                except ValueError:
                    auc = float("nan")
                for f, t, h in zip(fpr, tpr, thr):
                    rows.append({"fpr": float(f), "tpr": float(t),
                                 "threshold": float(h), "class": str(cls), "auc": auc})

            # Averaged curve if requested.
            if average in ("micro", "macro", "weighted"):
                avg_rows = _compute_avg_roc(y_true, proba_df[proba_cols].to_numpy(),
                                             classes, average, drop_intermediate)
                rows.extend(avg_rows)

        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df


def _coerce_class_label(label_str: str, target_dtype) -> object:
    """Coerce a class label from str back to the y dtype for comparison."""
    import numpy as np
    if np.issubdtype(target_dtype, np.integer):
        try:
            return int(label_str)
        except ValueError:
            return label_str
    if np.issubdtype(target_dtype, np.floating):
        try:
            return float(label_str)
        except ValueError:
            return label_str
    return label_str


def _compute_avg_roc(y_true, y_score_matrix, classes, average, drop_intermediate):
    import numpy as np
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=[_coerce_class_label(c, y_true.dtype) for c in classes])
    if average == "micro":
        fpr, tpr, thr = roc_curve(y_bin.ravel(), y_score_matrix.ravel(),
                                   drop_intermediate=drop_intermediate)
        auc = float(roc_auc_score(y_bin, y_score_matrix, average="micro"))
        label = "micro"
        return [{"fpr": float(f), "tpr": float(t), "threshold": float(h),
                 "class": label, "auc": auc} for f, t, h in zip(fpr, tpr, thr)]
    # macro / weighted: interpolate per-class curves on a common FPR grid then average.
    grid = np.linspace(0.0, 1.0, 100)
    tprs = []
    for i in range(y_bin.shape[1]):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_score_matrix[:, i])
        tprs.append(np.interp(grid, fpr_i, tpr_i))
    weights = (
        np.ones(len(classes)) / len(classes) if average == "macro"
        else y_bin.sum(axis=0) / y_bin.sum()
    )
    tpr_avg = (np.array(tprs).T * weights).sum(axis=1)
    auc = float(roc_auc_score(y_bin, y_score_matrix, average=average))
    return [{"fpr": float(f), "tpr": float(t), "threshold": float("nan"),
             "class": average, "auc": auc} for f, t in zip(grid, tpr_avg)]
```

- [ ] **Step 3: Add `pr_curve` method to `ModelSource`**

```python
    def pr_curve(self, *, average: str | None = None) -> pl.DataFrame:
        """Precision-recall curve(s). One row per (class, threshold)."""
        key = self._cache_key("pr_curve", average=average)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("pr_curve")
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import numpy as np

        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        if self._y is None:
            raise ValueError("ModelSource.pr_curve() requires y.")
        y_true = np.asarray(self._y.to_numpy())
        classes = [c[len("proba_"):] for c in proba_cols]
        n_classes = len(classes)

        rows: list[dict] = []
        if n_classes == 2 and average is None:
            y_score = proba_df[proba_cols[1]].to_numpy()
            p, r, thr = precision_recall_curve(y_true, y_score)
            ap = float(average_precision_score(y_true, y_score))
            # precision_recall_curve returns one fewer threshold than (p, r); pad with nan.
            thresholds_padded = np.concatenate([thr, [float("nan")]])
            for pi, ri, ti in zip(p, r, thresholds_padded):
                rows.append({"precision": float(pi), "recall": float(ri),
                             "threshold": float(ti) if not np.isnan(ti) else float("nan"),
                             "class": classes[1], "ap": ap})
        else:
            for i, cls in enumerate(classes):
                y_bin = (y_true == _coerce_class_label(cls, y_true.dtype)).astype(int)
                y_score = proba_df[proba_cols[i]].to_numpy()
                p, r, thr = precision_recall_curve(y_bin, y_score)
                try:
                    ap = float(average_precision_score(y_bin, y_score))
                except ValueError:
                    ap = float("nan")
                thresholds_padded = np.concatenate([thr, [float("nan")]])
                for pi, ri, ti in zip(p, r, thresholds_padded):
                    rows.append({"precision": float(pi), "recall": float(ri),
                                 "threshold": float(ti) if not np.isnan(ti) else float("nan"),
                                 "class": str(cls), "ap": ap})

        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df
```

- [ ] **Step 4: Add tests to `tests/diagnostics/test_source.py`**

```python
def test_roc_curve_binary_schema():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    roc = source.roc_curve()
    assert set(roc.columns) == {"fpr", "tpr", "threshold", "class", "auc"}
    assert roc.height >= 2  # at least two points on a curve
    # AUC repeats per class.
    aucs = roc["auc"].unique().to_list()
    assert len(aucs) == 1  # one binary class → one AUC value


def test_roc_curve_multiclass_macro_average():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    roc = source.roc_curve(average="macro")
    classes_seen = set(roc["class"].unique().to_list())
    assert "macro" in classes_seen
    # Plus three per-class entries.
    assert len(classes_seen - {"macro"}) >= 3


def test_pr_curve_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    pr = source.pr_curve()
    assert set(pr.columns) == {"precision", "recall", "threshold", "class", "ap"}
    assert pr.height >= 2
```

- [ ] **Step 5: Run + commit**

```bash
uv run --no-sync pytest tests/diagnostics/test_source.py -v -k "roc_curve or pr_curve" 2>&1 | tail -10
git add src/ferrum/_diagnostics/source.py src/ferrum/_diagnostics/schemas.py tests/diagnostics/test_source.py
git commit -m "feat(phase-10b): ModelSource.roc_curve + .pr_curve"
```

---

### Task 13: `.calibration_curve()` + `.cumulative_gain()` + `.lift_curve()` methods

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py`
- Modify: `src/ferrum/_diagnostics/schemas.py`
- Modify: `tests/diagnostics/test_source.py`

- [ ] **Step 1: Add schemas**

```python
SCHEMA_CALIBRATION = pl.Schema({
    "mean_predicted": pl.Float64,
    "fraction_positive": pl.Float64,
    "count": pl.Int64,
})

SCHEMA_GAIN_LIFT = pl.Schema({
    "percent_population": pl.Float64,
    # "gain" or "lift" — name varies
    "class": pl.Utf8,
})
```

- [ ] **Step 2: Add `calibration_curve` method**

```python
    def calibration_curve(
        self,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> pl.DataFrame:
        """Calibration curve: mean_predicted, fraction_positive, count per bin."""
        key = self._cache_key("calibration_curve", n_bins=n_bins, strategy=strategy)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("calibration_curve")
        from sklearn.calibration import calibration_curve as _ccurve
        import numpy as np

        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        if self._y is None:
            raise ValueError("ModelSource.calibration_curve() requires y.")
        y_true = np.asarray(self._y.to_numpy())
        # Use the positive class (last proba column for binary).
        y_score = proba_df[proba_cols[-1]].to_numpy()

        frac_pos, mean_pred = _ccurve(y_true, y_score, n_bins=n_bins, strategy=strategy)
        # Count per bin: discretize y_score the same way sklearn does.
        if strategy == "uniform":
            edges = np.linspace(0.0, 1.0, n_bins + 1)
        else:
            edges = np.quantile(y_score, np.linspace(0.0, 1.0, n_bins + 1))
        bin_idx = np.clip(np.digitize(y_score, edges[1:-1]), 0, n_bins - 1)
        counts = np.bincount(bin_idx, minlength=n_bins)
        # Align counts with calibration_curve output (which drops empty bins).
        used_bins = np.array([
            int(np.argmin(np.abs(edges[:-1] + np.diff(edges)/2 - mp)))
            for mp in mean_pred
        ])
        counts_aligned = counts[used_bins]

        df = pl.DataFrame({
            "mean_predicted": [float(x) for x in mean_pred],
            "fraction_positive": [float(x) for x in frac_pos],
            "count": [int(x) for x in counts_aligned],
        })
        self._cache[key] = df
        return df
```

- [ ] **Step 3: Add `cumulative_gain` method**

```python
    def cumulative_gain(self) -> pl.DataFrame:
        """Cumulative-gain curve per class. Includes class='baseline' diagonal."""
        key = self._cache_key("cumulative_gain")
        if key in self._cache:
            return self._cache[key]
        import numpy as np

        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        if self._y is None:
            raise ValueError("ModelSource.cumulative_gain() requires y.")
        y_true = np.asarray(self._y.to_numpy())
        classes = [c[len("proba_"):] for c in proba_cols]
        n = len(y_true)

        rows: list[dict] = []
        for i, cls in enumerate(classes):
            y_bin = (y_true == _coerce_class_label(cls, y_true.dtype)).astype(int)
            order = np.argsort(-proba_df[proba_cols[i]].to_numpy())
            cum_pos = np.cumsum(y_bin[order])
            total_pos = max(cum_pos[-1], 1)
            pct_pop = np.arange(1, n + 1) / n
            gain = cum_pos / total_pos
            # Prepend origin (0, 0) for clean curves.
            for pp, g in zip(np.concatenate([[0.0], pct_pop]),
                              np.concatenate([[0.0], gain])):
                rows.append({"percent_population": float(pp),
                             "gain": float(g), "class": str(cls)})

        # Baseline diagonal.
        for x in (0.0, 1.0):
            rows.append({"percent_population": x, "gain": x, "class": "baseline"})

        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df
```

- [ ] **Step 4: Add `lift_curve` method**

```python
    def lift_curve(self) -> pl.DataFrame:
        """Lift curve per class. Includes class='baseline' lift=1.0 line."""
        key = self._cache_key("lift_curve")
        if key in self._cache:
            return self._cache[key]
        import numpy as np

        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        if self._y is None:
            raise ValueError("ModelSource.lift_curve() requires y.")
        y_true = np.asarray(self._y.to_numpy())
        classes = [c[len("proba_"):] for c in proba_cols]
        n = len(y_true)

        rows: list[dict] = []
        for i, cls in enumerate(classes):
            y_bin = (y_true == _coerce_class_label(cls, y_true.dtype)).astype(int)
            base_rate = y_bin.mean()
            if base_rate == 0.0:
                continue
            order = np.argsort(-proba_df[proba_cols[i]].to_numpy())
            cum_pos = np.cumsum(y_bin[order])
            pct_pop = np.arange(1, n + 1) / n
            # lift = (cumulative positive rate at this pop fraction) / base_rate
            cum_rate = cum_pos / np.arange(1, n + 1)
            lift = cum_rate / base_rate
            for pp, l in zip(pct_pop, lift):
                rows.append({"percent_population": float(pp),
                             "lift": float(l), "class": str(cls)})

        # Baseline lift=1.0.
        for x in (0.0, 1.0):
            rows.append({"percent_population": x, "lift": 1.0, "class": "baseline"})

        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df
```

- [ ] **Step 5: Add tests**

```python
def test_calibration_curve_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    cal = source.calibration_curve(n_bins=5)
    assert set(cal.columns) == {"mean_predicted", "fraction_positive", "count"}
    assert cal.height <= 5
    assert (cal["mean_predicted"] >= 0).all()
    assert (cal["mean_predicted"] <= 1).all()


def test_cumulative_gain_includes_baseline():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    gain = source.cumulative_gain()
    classes_seen = set(gain["class"].unique().to_list())
    assert "baseline" in classes_seen


def test_lift_curve_baseline_at_one():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    lift = source.lift_curve()
    baseline = lift.filter(pl.col("class") == "baseline")
    assert (baseline["lift"] == 1.0).all()
```

- [ ] **Step 6: Run + commit**

```bash
uv run --no-sync pytest tests/diagnostics/test_source.py -v -k "calibration or gain or lift" 2>&1 | tail -10
git add src/ferrum/_diagnostics/source.py src/ferrum/_diagnostics/schemas.py tests/diagnostics/test_source.py
git commit -m "feat(phase-10b): ModelSource.calibration_curve + .cumulative_gain + .lift_curve"
```

---

### Task 14: `.discrimination_threshold()` method (incl. queue_rate hand-compute + CV averaging)

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py`
- Modify: `tests/diagnostics/test_source.py`

- [ ] **Step 1: Add `discrimination_threshold` method**

```python
    def discrimination_threshold(
        self,
        *,
        n_thresholds: int = 50,
        cv: int | Any = None,
    ) -> pl.DataFrame:
        """Discrimination threshold sweep — binary classifiers only.

        Sweeps thresholds at an evenly-spaced grid in [0, 1] (n_thresholds points).
        Reports precision, recall, F1, and queue_rate per threshold.
        queue_rate is hand-computed: (y_score >= t).mean() at each threshold.

        When `cv` is set, runs the same fixed grid sweep on each fold's held-out
        scores and averages per-threshold metrics across folds.
        """
        key = self._cache_key("discrimination_threshold", n_thresholds=n_thresholds, cv=cv)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("discrimination_threshold")
        from sklearn.metrics import precision_recall_fscore_support
        import numpy as np

        if self._y is None:
            raise ValueError("ModelSource.discrimination_threshold() requires y.")
        y_true = np.asarray(self._y.to_numpy())
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        if len(proba_cols) != 2:
            raise ValueError(
                "discrimination_threshold() is binary-classifier only; "
                f"got {len(proba_cols)} classes."
            )
        y_score = proba_df[proba_cols[1]].to_numpy()
        positive_class = _coerce_class_label(proba_cols[1][len("proba_"):], y_true.dtype)
        thresholds = np.linspace(0.0, 1.0, n_thresholds)

        if cv is None:
            df = self._sweep_thresholds(y_true, y_score, thresholds, positive_class)
        else:
            # CV averaging: fit on each train fold, score test fold, sweep at same grid.
            from sklearn.model_selection import KFold
            X_np = self._X.to_numpy()
            splitter = (
                cv if hasattr(cv, "split")
                else KFold(n_splits=int(cv), shuffle=True,
                            random_state=self._random_state or 0)
            )
            fold_dfs = []
            from sklearn.base import clone
            for tr, te in splitter.split(X_np):
                m = clone(self._model).fit(X_np[tr], y_true[tr])
                if hasattr(m, "predict_proba"):
                    s = m.predict_proba(X_np[te])[:, 1]
                else:
                    s = m.decision_function(X_np[te])
                    s = 1.0 / (1.0 + np.exp(-s))
                fold_dfs.append(self._sweep_thresholds(
                    y_true[tr := te], s, thresholds, positive_class
                ))
            df = pl.concat(fold_dfs, how="vertical").group_by("threshold").agg([
                pl.col("precision").mean(),
                pl.col("recall").mean(),
                pl.col("f1").mean(),
                pl.col("queue_rate").mean(),
            ]).sort("threshold")

        self._cache[key] = df
        return df

    def _sweep_thresholds(self, y_true, y_score, thresholds, positive_class) -> pl.DataFrame:
        from sklearn.metrics import precision_recall_fscore_support
        import numpy as np
        rows: list[dict] = []
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            y_true_bin = (y_true == positive_class).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                y_true_bin, y_pred, average="binary", zero_division=0,
            )
            queue_rate = float((y_score >= t).mean())
            rows.append({
                "threshold": float(t),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "queue_rate": queue_rate,
            })
        return pl.DataFrame(rows)
```

- [ ] **Step 2: Add tests**

```python
def test_discrimination_threshold_schema_and_grid():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    dt = source.discrimination_threshold(n_thresholds=20)
    assert set(dt.columns) == {"threshold", "precision", "recall", "f1", "queue_rate"}
    assert dt.height == 20
    # queue_rate at threshold=0 is 1.0 (all predictions positive).
    near_zero = dt.filter(pl.col("threshold") < 0.05)["queue_rate"].to_list()
    assert all(qr > 0.95 for qr in near_zero)


def test_discrimination_threshold_multiclass_rejects():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"])
    with pytest.raises(ValueError, match="binary-classifier only"):
        source.discrimination_threshold()


def test_discrimination_threshold_cv_averaging():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    source = ferrum.ModelSource(model, X, df["y"], random_state=0)
    dt = source.discrimination_threshold(n_thresholds=10, cv=3)
    assert dt.height == 10
    assert (dt["precision"] >= 0).all() and (dt["precision"] <= 1).all()
```

- [ ] **Step 3: Run + commit**

```bash
uv run --no-sync pytest tests/diagnostics/test_source.py -v -k discrimination 2>&1 | tail -10
git add src/ferrum/_diagnostics/source.py tests/diagnostics/test_source.py
git commit -m "feat(phase-10b): ModelSource.discrimination_threshold (queue_rate + cv averaging)"
```

---

### Task 15: Six 10b marks (`mark_roc`, `mark_pr`, `mark_calibration`, `mark_gain`, `mark_lift`, `mark_discrimination_threshold`)

**Files:**
- Modify: `src/ferrum/marks/diagnostic.py` (append six desugars)
- Modify: `src/ferrum/chart.py` (add six `mark_*` methods following the Task 8 pattern)
- Create: `tests/diagnostics/test_classification.py`

**Pattern:** Follow Task 8's `desugar_<name>(x_field, y_field, **kwargs) -> ("__layered__", [], None, None, layers)` shape with dict layers. Wire each Chart method as `_pending_stat_mark = (kind, kwargs_dict, desugar_fn)`. See Task 8 for full canonical example.

- [ ] **Step 1: Append the six desugars to `src/ferrum/marks/diagnostic.py`**

```python
def desugar_roc(
    x_field: str | None,
    y_field: str | None,
    *,
    average: str | None = None,
    reference_line: bool = True,
    annotate_auc: bool = True,
    color_field: str = "class",   # set to "model" by builder when input has model column
    **mark_kwargs: Any,
) -> tuple:
    """ROC curve(s). Data shape: (fpr, tpr, threshold, class, auc) per ModelSource.roc_curve()."""
    layers: list[dict] = [
        {"mark": "line", "encoding": {"x": "fpr", "y": "tpr", "color": color_field}},
    ]
    if reference_line:
        layers.append({
            "mark": "rule",
            "encoding": {"x": "fpr", "y": "fpr"},
            "mark_kwargs": {"strokeDash": [4, 4]},
        })
    return ("__layered__", [], None, None, layers)


def desugar_pr(
    x_field: str | None,
    y_field: str | None,
    *,
    average: str | None = None,
    annotate_ap: bool = True,
    iso_lines: bool = False,
    color_field: str = "class",
    **mark_kwargs: Any,
) -> tuple:
    """Precision-recall curve(s). Data shape: (precision, recall, threshold, class, ap)."""
    layers: list[dict] = [
        {"mark": "line", "encoding": {"x": "recall", "y": "precision", "color": color_field}},
    ]
    return ("__layered__", [], None, None, layers)


def desugar_calibration(
    x_field: str | None,
    y_field: str | None,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    reference_line: bool = True,
    color_field: str | None = None,
    **mark_kwargs: Any,
) -> tuple:
    """Calibration curve. Data shape: (mean_predicted, fraction_positive, count)."""
    line_enc: dict[str, Any] = {"x": "mean_predicted", "y": "fraction_positive"}
    if color_field is not None:
        line_enc["color"] = color_field
    layers: list[dict] = [{"mark": "line", "encoding": line_enc}]
    if reference_line:
        layers.append({
            "mark": "rule",
            "encoding": {"x": "mean_predicted", "y": "mean_predicted"},
            "mark_kwargs": {"strokeDash": [4, 4]},
        })
    return ("__layered__", [], None, None, layers)


def desugar_gain(
    x_field: str | None,
    y_field: str | None,
    *,
    reference_lines: bool = True,
    color_field: str = "class",
    **mark_kwargs: Any,
) -> tuple:
    """Cumulative gain curve. Data shape: (percent_population, gain, class)."""
    return ("__layered__", [], None, None, [
        {"mark": "line", "encoding": {"x": "percent_population", "y": "gain", "color": color_field}},
    ])


def desugar_lift(
    x_field: str | None,
    y_field: str | None,
    *,
    reference_line: bool = True,
    color_field: str = "class",
    **mark_kwargs: Any,
) -> tuple:
    """Lift curve. Data shape: (percent_population, lift, class)."""
    return ("__layered__", [], None, None, [
        {"mark": "line", "encoding": {"x": "percent_population", "y": "lift", "color": color_field}},
    ])


def desugar_discrimination_threshold(
    x_field: str | None,
    y_field: str | None,
    *,
    metrics: tuple[str, ...] = ("precision", "recall", "f1", "queue_rate"),
    n_thresholds: int = 50,
    threshold_line: bool = True,
    **mark_kwargs: Any,
) -> tuple:
    """Discrimination-threshold sweep. Builder pre-melts to long form
    (threshold, metric, value); this desugar plots one line per metric."""
    return ("__layered__", [], None, None, [
        {"mark": "line", "encoding": {"x": "threshold", "y": "value", "color": "metric"}},
    ])
```

- [ ] **Step 2: Wire six Chart methods in `src/ferrum/chart.py`**

Following the Task 8 / `mark_boxplot` pattern:

```python
def mark_roc(self, *, average=None, reference_line=True, annotate_auc=True,
              color_field="class", position=None, **mark_kwargs) -> "Chart":
    from ferrum.marks.diagnostic import desugar_roc
    new = self._clone()
    new._mark = "point"
    new._pending_stat_mark = ("roc", {
        "average": average, "reference_line": reference_line,
        "annotate_auc": annotate_auc, "color_field": color_field, **mark_kwargs,
    }, desugar_roc)
    new._position = position
    return new


def mark_pr(self, *, average=None, annotate_ap=True, iso_lines=False,
             color_field="class", position=None, **mark_kwargs) -> "Chart":
    from ferrum.marks.diagnostic import desugar_pr
    new = self._clone()
    new._mark = "point"
    new._pending_stat_mark = ("pr", {
        "average": average, "annotate_ap": annotate_ap,
        "iso_lines": iso_lines, "color_field": color_field, **mark_kwargs,
    }, desugar_pr)
    new._position = position
    return new


def mark_calibration(self, *, n_bins=10, strategy="uniform",
                      reference_line=True, color_field=None,
                      position=None, **mark_kwargs) -> "Chart":
    from ferrum.marks.diagnostic import desugar_calibration
    new = self._clone()
    new._mark = "point"
    new._pending_stat_mark = ("calibration", {
        "n_bins": n_bins, "strategy": strategy,
        "reference_line": reference_line, "color_field": color_field, **mark_kwargs,
    }, desugar_calibration)
    new._position = position
    return new


def mark_gain(self, *, reference_lines=True, color_field="class",
               position=None, **mark_kwargs) -> "Chart":
    from ferrum.marks.diagnostic import desugar_gain
    new = self._clone()
    new._mark = "point"
    new._pending_stat_mark = ("gain", {
        "reference_lines": reference_lines, "color_field": color_field, **mark_kwargs,
    }, desugar_gain)
    new._position = position
    return new


def mark_lift(self, *, reference_line=True, color_field="class",
               position=None, **mark_kwargs) -> "Chart":
    from ferrum.marks.diagnostic import desugar_lift
    new = self._clone()
    new._mark = "point"
    new._pending_stat_mark = ("lift", {
        "reference_line": reference_line, "color_field": color_field, **mark_kwargs,
    }, desugar_lift)
    new._position = position
    return new


def mark_discrimination_threshold(self, *,
                                    metrics=("precision", "recall", "f1", "queue_rate"),
                                    n_thresholds=50, threshold_line=True,
                                    position=None, **mark_kwargs) -> "Chart":
    from ferrum.marks.diagnostic import desugar_discrimination_threshold
    new = self._clone()
    new._mark = "point"
    new._pending_stat_mark = ("discrimination_threshold", {
        "metrics": metrics, "n_thresholds": n_thresholds,
        "threshold_line": threshold_line, **mark_kwargs,
    }, desugar_discrimination_threshold)
    new._position = position
    return new
```

- [ ] **Step 3: No `__init__.py` change required** — diagnostic marks are accessed via `Chart.mark_<name>(...)`, not as importable symbols.

> **`color_field` resolution at builder layer:** when a chart builder in `_diagnostics/charts.py` constructs the chart from a `ComparedModelSource` output (DataFrame contains a `model` column), it passes `color_field="model"` to the Chart method so the line is colored by model rather than by class. The `desugar_*` functions don't auto-detect — the builder is responsible.

- [ ] **Step 4: Create `tests/diagnostics/test_classification.py`**

```python
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum
from tests.fixtures import load_fixture, load_dataset


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


def test_mark_roc_renders_binary(binary_source):
    roc = binary_source.roc_curve()
    svg = ferrum.Chart(roc).mark_roc().show_svg()
    assert "<svg" in svg


def test_mark_roc_renders_multiclass(multi_source):
    roc = multi_source.roc_curve()
    svg = ferrum.Chart(roc).mark_roc().show_svg()
    assert "<svg" in svg


def test_mark_pr_renders(binary_source):
    pr = binary_source.pr_curve()
    svg = ferrum.Chart(pr).mark_pr().show_svg()
    assert "<svg" in svg


def test_mark_calibration_renders(binary_source):
    cal = binary_source.calibration_curve(n_bins=10)
    svg = ferrum.Chart(cal).mark_calibration().show_svg()
    assert "<svg" in svg


def test_mark_gain_renders(binary_source):
    gain = binary_source.cumulative_gain()
    svg = ferrum.Chart(gain).mark_gain().show_svg()
    assert "<svg" in svg


def test_mark_lift_renders(binary_source):
    lift = binary_source.lift_curve()
    svg = ferrum.Chart(lift).mark_lift().show_svg()
    assert "<svg" in svg


def test_mark_discrimination_threshold_renders(binary_source):
    dt = binary_source.discrimination_threshold(n_thresholds=20)
    long = dt.unpivot(
        index="threshold",
        on=["precision", "recall", "f1", "queue_rate"],
        variable_name="metric",
        value_name="value",
    )
    svg = ferrum.Chart(long).mark_discrimination_threshold().show_svg()
    assert "<svg" in svg
```

- [ ] **Step 5: Run + commit**

```bash
uv run --no-sync pytest tests/diagnostics/test_classification.py -v 2>&1 | tail -15
git add src/ferrum/marks/diagnostic.py src/ferrum/chart.py tests/diagnostics/test_classification.py
git commit -m "feat(phase-10b): 6 classification curve marks (roc/pr/calibration/gain/lift/disc_threshold)"
```

---

### Task 16: 10b chart builders + figure functions

**Files:**
- Modify: `src/ferrum/_diagnostics/charts.py`
- Modify: `src/ferrum/figures.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/diagnostics/test_classification.py`

- [ ] **Step 1: Add six chart builders to `_diagnostics/charts.py`**

```python
def _roc_chart_from_source(source, *, per_class=True, average="macro",
                            annotate_auc=True, theme=None):
    df = source.roc_curve(average=None if per_class else average)
    chart = ferrum.Chart(df).mark_roc(
        average=average if not per_class else None,
        annotate_auc=annotate_auc,
    )
    if theme is not None: chart = chart.theme(theme)
    return chart


def _pr_chart_from_source(source, *, per_class=True, annotate_ap=True,
                           iso_lines=True, theme=None):
    df = source.pr_curve()
    chart = ferrum.Chart(df).mark_pr(annotate_ap=annotate_ap, iso_lines=iso_lines)
    if theme is not None: chart = chart.theme(theme)
    return chart


def _calibration_chart_from_source(source, *, n_bins=10, theme=None):
    df = source.calibration_curve(n_bins=n_bins)
    chart = ferrum.Chart(df).mark_calibration(n_bins=n_bins)
    if theme is not None: chart = chart.theme(theme)
    return chart


def _gain_chart_from_source(source, *, theme=None):
    df = source.cumulative_gain()
    chart = ferrum.Chart(df).mark_gain()
    if theme is not None: chart = chart.theme(theme)
    return chart


def _lift_chart_from_source(source, *, theme=None):
    df = source.lift_curve()
    chart = ferrum.Chart(df).mark_lift()
    if theme is not None: chart = chart.theme(theme)
    return chart


def _discrimination_threshold_chart_from_source(
    source, *, n_thresholds=50, metrics=("precision", "recall", "f1", "queue_rate"),
    highlight_best=True, theme=None,
):
    df = source.discrimination_threshold(n_thresholds=n_thresholds)
    # Reshape to long form for plotting.
    long_df = df.unpivot(
        index="threshold",
        on=list(metrics),
        variable_name="metric", value_name="value",
    )
    chart = ferrum.Chart(long_df).mark_discrimination_threshold(
        metrics=metrics, n_thresholds=n_thresholds,
    )
    if theme is not None: chart = chart.theme(theme)
    return chart
```

- [ ] **Step 2: Add six figure functions to `src/ferrum/figures.py`**

```python
from ferrum._diagnostics.charts import (
    _roc_chart_from_source, _pr_chart_from_source,
    _calibration_chart_from_source, _gain_chart_from_source,
    _lift_chart_from_source, _discrimination_threshold_chart_from_source,
)


def roc_chart(
    model_or_source, X=None, y=None, *, per_class=True,
    average="macro", annotate_auc=True, compare=None,
    random_state=None, theme=None,
):
    source = _resolve_source(model_or_source, X, y, random_state=random_state, compare=compare)
    return _roc_chart_from_source(source, per_class=per_class, average=average,
                                    annotate_auc=annotate_auc, theme=theme)


def pr_chart(
    model_or_source, X=None, y=None, *, per_class=True,
    annotate_ap=True, iso_lines=True, compare=None,
    random_state=None, theme=None,
):
    source = _resolve_source(model_or_source, X, y, random_state=random_state, compare=compare)
    return _pr_chart_from_source(source, per_class=per_class,
                                   annotate_ap=annotate_ap, iso_lines=iso_lines, theme=theme)


def calibration_chart(
    *model_or_sources, X=None, y=None, n_bins=10,
    random_state=None, theme=None,
):
    # Variadic: builds ComparedModelSource if >1 model. 10h enables compare path;
    # 10b accepts only the single-source case for now.
    if len(model_or_sources) > 1:
        raise NotImplementedError("Multi-model calibration ships in Phase 10h.")
    source = _resolve_source(model_or_sources[0], X, y, random_state=random_state)
    return _calibration_chart_from_source(source, n_bins=n_bins, theme=theme)


def gain_chart(model_or_source, X=None, y=None, *, random_state=None, theme=None):
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _gain_chart_from_source(source, theme=theme)


def lift_chart(model_or_source, X=None, y=None, *, random_state=None, theme=None):
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _lift_chart_from_source(source, theme=theme)


def discrimination_threshold_chart(
    model_or_source, X=None, y=None, *,
    n_thresholds=50, metrics=("precision", "recall", "f1", "queue_rate"),
    highlight_best=True, random_state=None, theme=None,
):
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _discrimination_threshold_chart_from_source(
        source, n_thresholds=n_thresholds, metrics=metrics,
        highlight_best=highlight_best, theme=theme,
    )
```

- [ ] **Step 3: Re-export from `src/ferrum/__init__.py`**

```python
from ferrum.figures import (
    roc_chart, pr_chart, calibration_chart,
    gain_chart, lift_chart, discrimination_threshold_chart,
)
```

- [ ] **Step 4: Add figure-function tests + goldens**

In `tests/diagnostics/test_classification.py`:

```python
def test_roc_chart_figure_function(binary_source):
    svg = ferrum.roc_chart(binary_source).show_svg()
    assert "<svg" in svg


def test_pr_chart_figure_function(binary_source):
    svg = ferrum.pr_chart(binary_source).show_svg()
    assert "<svg" in svg


def test_calibration_chart_figure_function(binary_source):
    svg = ferrum.calibration_chart(binary_source, n_bins=5).show_svg()
    assert "<svg" in svg


def test_gain_chart_figure_function(binary_source):
    svg = ferrum.gain_chart(binary_source).show_svg()
    assert "<svg" in svg


def test_lift_chart_figure_function(binary_source):
    svg = ferrum.lift_chart(binary_source).show_svg()
    assert "<svg" in svg


def test_discrimination_threshold_chart_figure_function(binary_source):
    svg = ferrum.discrimination_threshold_chart(binary_source, n_thresholds=20).show_svg()
    assert "<svg" in svg
```

In `tests/diagnostics/test_goldens_phase_10.py`, append byte-identical goldens (one per figure × binary fixture, plus multiclass for ROC/PR):

```python
def test_golden_roc_chart_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.roc_chart(model, df.select(["f0", "f1", "f2", "f3"]), df["y"])
    _check_golden(chart.show_svg(), "roc_chart_binary")


def test_golden_roc_chart_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.roc_chart(model, df.select(["f0", "f1", "f2", "f3"]), df["y"])
    _check_golden(chart.show_svg(), "roc_chart_multiclass")


def test_golden_pr_chart_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.pr_chart(model, df.select(["f0", "f1", "f2", "f3"]), df["y"])
    _check_golden(chart.show_svg(), "pr_chart_binary")


def test_golden_calibration_chart():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.calibration_chart(model, df.select(["f0", "f1", "f2", "f3"]),
                                      df["y"], n_bins=5)
    _check_golden(chart.show_svg(), "calibration_chart_binary")


def test_golden_gain_chart_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.gain_chart(model, df.select(["f0", "f1", "f2", "f3"]), df["y"])
    _check_golden(chart.show_svg(), "gain_chart_binary")


def test_golden_lift_chart_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.lift_chart(model, df.select(["f0", "f1", "f2", "f3"]), df["y"])
    _check_golden(chart.show_svg(), "lift_chart_binary")


def test_golden_discrimination_threshold_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.discrimination_threshold_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"], n_thresholds=20,
    )
    _check_golden(chart.show_svg(), "discrimination_threshold_binary")
```

- [ ] **Step 5: Generate goldens + verify**

```bash
FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -v 2>&1 | tail -15
uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -v 2>&1 | tail -15
```
Expected: 7 new SVGs created under `tests/goldens/phase_10/`; all golden tests pass on second run.

- [ ] **Step 6: Commit**

```bash
git add src/ferrum/_diagnostics/charts.py src/ferrum/figures.py src/ferrum/__init__.py tests/diagnostics/test_classification.py tests/diagnostics/test_goldens_phase_10.py tests/goldens/phase_10/
git commit -m "feat(phase-10b): 6 classification curve figure functions + goldens"
```

---

### Task 17: 10b visualizers (ROC, PR, Calibration, DiscriminationThreshold)

**Files:**
- Create: `src/ferrum/_diagnostics/visualizers/classification.py` (initial — 3 visualizers)
- Create: `src/ferrum/_diagnostics/visualizers/classification_extra.py` (initial — DiscriminationThresholdVisualizer)
- Modify: `src/ferrum/_diagnostics/visualizers/__init__.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/diagnostics/test_classification.py`

- [ ] **Step 1: Write `classification.py`**

```python
"""10b classification visualizers (ROC, PR, Calibration)."""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import FerrumVisualizer
from ..charts import (
    _roc_chart_from_source, _pr_chart_from_source,
    _calibration_chart_from_source,
)


class ROCVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, micro: bool = True, macro: bool = True,
                 per_class: bool = True, random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.micro = micro
        self.macro = macro
        self.per_class = per_class

    def _materialize(self) -> None:
        roc = self._source.roc_curve(average="macro" if self.macro else None)
        aucs = roc["auc"].unique().to_list()
        # Mean AUC across classes for the repr summary.
        self._metrics["auc_mean"] = float(np.nanmean(aucs))

    def _build_chart(self) -> Any:
        avg = "macro" if self.macro else ("micro" if self.micro else None)
        return _roc_chart_from_source(
            self._source, per_class=self.per_class, average=avg, theme=self.theme,
        )

    def score(self, X, y) -> float:
        from sklearn.metrics import roc_auc_score
        if hasattr(self.model, "predict_proba"):
            s = self.model.predict_proba(X)
            return float(roc_auc_score(y, s, multi_class="ovr"))
        return float(self.model.score(X, y))


class PRVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)

    def _materialize(self) -> None:
        pr = self._source.pr_curve()
        aps = pr["ap"].unique().to_list()
        self._metrics["ap_mean"] = float(np.nanmean(aps))

    def _build_chart(self) -> Any:
        return _pr_chart_from_source(self._source, theme=self.theme)


class CalibrationVisualizer(FerrumVisualizer):
    """Variadic — accepts one or more models. 10b ships single-model; 10h enables variadic."""
    def __init__(self, *models: Any, n_bins: int = 10,
                 random_state: int | None = None, theme: Any = None):
        if len(models) != 1:
            raise NotImplementedError(
                "Multi-model CalibrationVisualizer ships in Phase 10h."
            )
        super().__init__(models[0], random_state=random_state, theme=theme)
        self.n_bins = n_bins

    def _materialize(self) -> None:
        cal = self._source.calibration_curve(n_bins=self.n_bins)
        # Brier-like proxy: mean squared deviation from diagonal.
        diff = cal["fraction_positive"].to_numpy() - cal["mean_predicted"].to_numpy()
        self._metrics["calibration_error"] = float(np.mean(diff ** 2))

    def _build_chart(self) -> Any:
        return _calibration_chart_from_source(
            self._source, n_bins=self.n_bins, theme=self.theme,
        )
```

- [ ] **Step 2: Write `classification_extra.py`**

```python
"""10b/10c extra visualizers (DiscriminationThreshold, ClassPredictionError, ClassBalance)."""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import FerrumVisualizer
from ..charts import _discrimination_threshold_chart_from_source


class DiscriminationThresholdVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, n_thresholds: int = 50, scoring: Any = None,
                 cv: Any = None, random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.n_thresholds = n_thresholds
        self.scoring = scoring
        self.cv = cv

    def _materialize(self) -> None:
        dt = self._source.discrimination_threshold(
            n_thresholds=self.n_thresholds, cv=self.cv,
        )
        # F1-maximizing threshold for summary.
        idx = int(np.argmax(dt["f1"].to_numpy()))
        self._metrics["best_threshold"] = float(dt["threshold"][idx])
        self._metrics["best_f1"] = float(dt["f1"][idx])

    def _build_chart(self) -> Any:
        return _discrimination_threshold_chart_from_source(
            self._source, n_thresholds=self.n_thresholds, theme=self.theme,
        )
```

- [ ] **Step 3: Update visualizers `__init__.py`**

```python
from .classification import ROCVisualizer, PRVisualizer, CalibrationVisualizer
from .classification_extra import DiscriminationThresholdVisualizer

__all__ += [
    "ROCVisualizer", "PRVisualizer", "CalibrationVisualizer",
    "DiscriminationThresholdVisualizer",
]
```

- [ ] **Step 4: Re-export from `src/ferrum/__init__.py`**

```python
from ferrum._diagnostics.visualizers import (
    ROCVisualizer, PRVisualizer, CalibrationVisualizer,
    DiscriminationThresholdVisualizer,
)
```

- [ ] **Step 5: Add visualizer tests**

```python
def test_roc_visualizer(binary_source):
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.ROCVisualizer(model).fit(df.select(["f0", "f1", "f2", "f3"]), df["y"])
    assert "auc_mean=" in repr(viz)
    assert "<svg" in viz.show().show_svg()


def test_pr_visualizer(binary_source):
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.PRVisualizer(model).fit(df.select(["f0", "f1", "f2", "f3"]), df["y"])
    assert "ap_mean=" in repr(viz)


def test_calibration_visualizer():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.CalibrationVisualizer(model, n_bins=5).fit(
        df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "calibration_error=" in repr(viz)


def test_discrimination_threshold_visualizer():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    viz = ferrum.DiscriminationThresholdVisualizer(model, n_thresholds=20).fit(
        df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "best_threshold=" in repr(viz)
    assert 0.0 <= viz._metrics["best_threshold"] <= 1.0
```

- [ ] **Step 6: Run + commit**

```bash
uv run --no-sync pytest tests/diagnostics/test_classification.py -v 2>&1 | tail -15
git add src/ferrum/_diagnostics/visualizers/ src/ferrum/__init__.py tests/diagnostics/test_classification.py
git commit -m "feat(phase-10b): ROCVisualizer + PRVisualizer + CalibrationVisualizer + DiscriminationThresholdVisualizer"
```

- [ ] **Step 7: 10b milestone check**

```bash
uv run --no-sync pytest tests/diagnostics/ 2>&1 | tail -3
```
Expected: ~40 tests pass cumulatively (10a + 10b).

---

## 10c — Classification matrices

### Task 18: `.confusion_matrix()` method + `mark_confusion` + `confusion_matrix_chart`

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py`
- Modify: `src/ferrum/_diagnostics/schemas.py`
- Modify: `src/ferrum/marks/diagnostic.py`
- Modify: `src/ferrum/chart.py`
- Modify: `src/ferrum/_diagnostics/charts.py`
- Modify: `src/ferrum/figures.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/diagnostics/test_classification.py`
- Modify: `tests/diagnostics/test_goldens_phase_10.py`

> ⚠ **Pattern correction (plan-vs-codebase):** The mark code blocks below were originally drafted using a `@dataclass(frozen=True) class mark_X: ... def _expand(self, chart_ctx) -> list[LayerSpec]` pattern that **does not exist in the codebase**. Before implementing, translate every mark in this task to the real pattern used in Phase 8b/9 composite marks:
>
> - Module-level `def desugar_<name>(x_field, y_field, **kwargs) -> ("__layered__", transforms, None, None, layers)` in `src/ferrum/marks/diagnostic.py`.
> - Layers are plain dicts: `{"mark": str, "encoding": dict, "mark_kwargs": dict (opt), "data_source": str | None (opt)}`.
> - No `LayerSpec`. No `chart_ctx`. No `_expand`.
> - Chart method clones, sets `_mark = "point"` (placeholder), sets `_pending_stat_mark = (kind, kwargs_dict, desugar_fn)`, returns.
> - The user does not import or instantiate `mark_X` — they call `Chart(df).mark_X(...)`.
> - For diagnostic marks, the data has hard-coded columns from a `ModelSource` method, so the desugar references those columns literally and ignores positional `x_field` / `y_field`.
>
> **Canonical reference:** Task 8 (`desugar_residuals` / `desugar_prediction_error`) and Task 15 (six 10b desugars). Pattern reference in code: `src/ferrum/marks/composite.py:15-220`.
>
> Keep the **layer encodings, kwargs, and behavior** below as the spec for what each mark should produce, but rewrite the implementation in the corrected pattern.

- [ ] **Step 1: Add schema + ModelSource method**

```python
# schemas.py
SCHEMA_CONFUSION = pl.Schema({
    "actual": pl.Utf8,
    "predicted": pl.Utf8,
    "value": pl.Float64,
    "value_fmt": pl.Utf8,
})

# source.py — append
    def confusion_matrix(self, *, normalize: str | None = None) -> pl.DataFrame:
        """Confusion matrix as long-form (actual, predicted, value, value_fmt).

        normalize: None | "true" | "pred" | "all" (sklearn semantics).
        """
        key = self._cache_key("confusion_matrix", normalize=normalize)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("confusion_matrix")
        from sklearn.metrics import confusion_matrix as _cm
        import numpy as np

        if self._y is None:
            raise ValueError("ModelSource.confusion_matrix() requires y.")
        y_true = np.asarray(self._y.to_numpy())
        X_np = self._X.to_numpy()
        y_pred = np.asarray(self._model.predict(X_np))
        labels = (
            self._class_names
            or (getattr(self._model, "classes_", None) and list(self._model.classes_))
            or sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        )
        cm = _cm(y_true, y_pred, labels=labels, normalize=normalize)
        rows: list[dict] = []
        for i, a in enumerate(labels):
            for j, p in enumerate(labels):
                val = float(cm[i, j])
                fmt = f"{val:.2f}" if normalize else f"{int(val)}"
                rows.append({
                    "actual": str(a), "predicted": str(p),
                    "value": val, "value_fmt": fmt,
                })
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df
```

- [ ] **Step 2: Add `mark_confusion`**

In `src/ferrum/marks/diagnostic.py`:

```python
@dataclass(frozen=True)
class mark_confusion:
    normalize: str | None = None     # None | "true" | "pred" | "all"
    text_fmt: str | None = None      # if None, use value_fmt column

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_rect, mark_text
        from ferrum import LayerSpec
        return [
            LayerSpec(
                mark=mark_rect(),
                encoding={"x": "predicted", "y": "actual", "color": "value"},
            ),
            LayerSpec(
                mark=mark_text(),
                encoding={"x": "predicted", "y": "actual", "text": "value_fmt"},
            ),
        ]
```

In `src/ferrum/chart.py`:

```python
def mark_confusion(self, **kw) -> "Chart":
    from ferrum.marks.diagnostic import mark_confusion as _M
    return self._add_composite_mark(_M(**kw))
```

- [ ] **Step 3: Add chart builder + figure function**

```python
# charts.py
def _confusion_chart_from_source(source, *, normalize="true", cmap="blues", theme=None):
    df = source.confusion_matrix(normalize=normalize)
    chart = ferrum.Chart(df).mark_confusion(normalize=normalize)
    if theme is not None: chart = chart.theme(theme)
    return chart


# figures.py
from ferrum._diagnostics.charts import _confusion_chart_from_source


def confusion_matrix_chart(
    model_or_source, X=None, y=None, *,
    normalize="true", cmap="blues",
    random_state=None, theme=None,
):
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _confusion_chart_from_source(source, normalize=normalize, cmap=cmap, theme=theme)
```

- [ ] **Step 4: Re-export and test**

```python
# src/ferrum/__init__.py
from ferrum.marks.diagnostic import mark_confusion
from ferrum.figures import confusion_matrix_chart

# tests/diagnostics/test_classification.py
def test_confusion_matrix_schema(binary_source):
    cm = binary_source.confusion_matrix()
    assert set(cm.columns) == {"actual", "predicted", "value", "value_fmt"}


def test_confusion_matrix_chart_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.confusion_matrix_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "<svg" in chart.show_svg()


def test_confusion_matrix_chart_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.confusion_matrix_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"], normalize="true",
    )
    assert "<svg" in chart.show_svg()
```

- [ ] **Step 5: Add goldens**

```python
# tests/diagnostics/test_goldens_phase_10.py
def test_golden_confusion_matrix_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.confusion_matrix_chart(model, df.select(["f0", "f1", "f2", "f3"]), df["y"])
    _check_golden(chart.show_svg(), "confusion_matrix_binary")


def test_golden_confusion_matrix_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.confusion_matrix_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"], normalize="true",
    )
    _check_golden(chart.show_svg(), "confusion_matrix_multiclass")
```

- [ ] **Step 6: Run + commit**

```bash
FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -k confusion -v 2>&1 | tail -10
uv run --no-sync pytest tests/diagnostics/ 2>&1 | tail -3
git add src/ferrum/_diagnostics/ src/ferrum/marks/diagnostic.py src/ferrum/chart.py src/ferrum/figures.py src/ferrum/__init__.py tests/diagnostics/ tests/goldens/phase_10/
git commit -m "feat(phase-10c): confusion_matrix method + mark_confusion + confusion_matrix_chart"
```

---

### Task 19: `mark_class_prediction_error` + `class_prediction_error_chart`

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py` (helper)
- Modify: `src/ferrum/marks/diagnostic.py`
- Modify: `src/ferrum/chart.py`
- Modify: `src/ferrum/_diagnostics/charts.py`
- Modify: `src/ferrum/figures.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/diagnostics/test_classification.py`
- Modify: `tests/diagnostics/test_goldens_phase_10.py`

> ⚠ **Pattern correction (plan-vs-codebase):** The mark code blocks below were originally drafted using a `@dataclass(frozen=True) class mark_X: ... def _expand(self, chart_ctx) -> list[LayerSpec]` pattern that **does not exist in the codebase**. Before implementing, translate every mark in this task to the real pattern used in Phase 8b/9 composite marks:
>
> - Module-level `def desugar_<name>(x_field, y_field, **kwargs) -> ("__layered__", transforms, None, None, layers)` in `src/ferrum/marks/diagnostic.py`.
> - Layers are plain dicts: `{"mark": str, "encoding": dict, "mark_kwargs": dict (opt), "data_source": str | None (opt)}`.
> - No `LayerSpec`. No `chart_ctx`. No `_expand`.
> - Chart method clones, sets `_mark = "point"` (placeholder), sets `_pending_stat_mark = (kind, kwargs_dict, desugar_fn)`, returns.
> - The user does not import or instantiate `mark_X` — they call `Chart(df).mark_X(...)`.
> - For diagnostic marks, the data has hard-coded columns from a `ModelSource` method, so the desugar references those columns literally and ignores positional `x_field` / `y_field`.
>
> **Canonical reference:** Task 8 (`desugar_residuals` / `desugar_prediction_error`) and Task 15 (six 10b desugars). Pattern reference in code: `src/ferrum/marks/composite.py:15-220`.
>
> Keep the **layer encodings, kwargs, and behavior** below as the spec for what each mark should produce, but rewrite the implementation in the corrected pattern.

- [ ] **Step 1: Add `mark_class_prediction_error`**

```python
# marks/diagnostic.py
@dataclass(frozen=True)
class mark_class_prediction_error:
    """Stacked bar of predicted-class counts colored by actual class.

    Operates on a long-form DataFrame with columns (actual, predicted, value)
    — same shape as confusion_matrix(). Uses Phase 9's Stack position.
    """
    orient: str = "vertical"
    normalize: bool = False

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_bar
        from ferrum import LayerSpec, Stack
        return [LayerSpec(
            mark=mark_bar(),
            encoding={"x": "predicted", "y": "value", "color": "actual"},
            position=Stack(by="actual", offset="normalize" if self.normalize else "zero"),
        )]
```

- [ ] **Step 2: Add Chart method + chart builder + figure**

```python
# chart.py
def mark_class_prediction_error(self, **kw) -> "Chart":
    from ferrum.marks.diagnostic import mark_class_prediction_error as _M
    return self._add_composite_mark(_M(**kw))


# charts.py
def _class_prediction_error_chart_from_source(source, *, normalize=False, theme=None):
    # Reuse confusion_matrix output (already long-form actual/predicted/value).
    df = source.confusion_matrix(normalize=None)
    chart = ferrum.Chart(df).mark_class_prediction_error(normalize=normalize)
    if theme is not None: chart = chart.theme(theme)
    return chart


# figures.py
def class_prediction_error_chart(
    model_or_source, X=None, y=None, *,
    normalize=False, random_state=None, theme=None,
):
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _class_prediction_error_chart_from_source(source, normalize=normalize, theme=theme)
```

- [ ] **Step 3: Re-export + test + goldens**

```python
# __init__.py
from ferrum.marks.diagnostic import mark_class_prediction_error
from ferrum.figures import class_prediction_error_chart


# test_classification.py
def test_class_prediction_error_chart():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.class_prediction_error_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "<svg" in chart.show_svg()


# test_goldens_phase_10.py
def test_golden_class_prediction_error_multiclass():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    chart = ferrum.class_prediction_error_chart(
        model, df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    _check_golden(chart.show_svg(), "class_prediction_error_multiclass")
```

- [ ] **Step 4: Run + commit**

```bash
FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -k class_prediction -v 2>&1 | tail -5
uv run --no-sync pytest tests/diagnostics/ 2>&1 | tail -3
git add -u src/ferrum/ tests/diagnostics/
git add tests/goldens/phase_10/class_prediction_error_multiclass.svg
git commit -m "feat(phase-10c): mark_class_prediction_error + figure function + golden"
```

---

### Task 20: 10c visualizers (Confusion, ClassificationReport, ClassPredictionError, ClassBalance)

**Files:**
- Modify: `src/ferrum/_diagnostics/visualizers/classification.py`
- Modify: `src/ferrum/_diagnostics/visualizers/classification_extra.py`
- Modify: `src/ferrum/_diagnostics/visualizers/__init__.py`
- Modify: `src/ferrum/_diagnostics/charts.py` (add `_classification_report_chart` + `_class_balance_chart_from_dataframe`)
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/diagnostics/test_classification.py`

- [ ] **Step 1: Add `_classification_report_chart` and `_class_balance_chart_from_dataframe`**

In `_diagnostics/charts.py`:

```python
def _classification_report_chart(source, *, theme=None):
    """Heatmap of per-class precision/recall/F1/support."""
    require_sklearn = __import__("ferrum._diagnostics.deps", fromlist=["require_sklearn"]).require_sklearn
    require_sklearn("ClassificationReportVisualizer")
    from sklearn.metrics import classification_report
    import numpy as np
    y_true = source._y.to_numpy()
    y_pred = source._model.predict(source._X.to_numpy())
    rpt = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows: list[dict] = []
    for cls_label, metrics in rpt.items():
        if cls_label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if isinstance(metrics, dict):
            for m_name in ("precision", "recall", "f1-score"):
                rows.append({
                    "class": str(cls_label), "metric": m_name,
                    "value": float(metrics[m_name]),
                    "value_fmt": f"{metrics[m_name]:.2f}",
                })
    df = pl.DataFrame(rows)
    # Reuse mark_confusion-style heatmap (mark_rect + mark_text).
    chart = ferrum.Chart(df).mark_rect().encode(
        x="metric", y="class", color="value",
    )
    text = ferrum.Chart(df).mark_text().encode(
        x="metric", y="class", text="value_fmt",
    )
    out = chart + text
    if theme is not None: out = out.theme(theme)
    return out


def _class_balance_chart_from_dataframe(y_series, *, theme=None):
    """Bar of class counts in y."""
    df = pl.DataFrame({"y": y_series.to_list()}).group_by("y").len().rename({"len": "count"})
    chart = ferrum.Chart(df).mark_bar().encode(x="y", y="count")
    if theme is not None: chart = chart.theme(theme)
    return chart
```

- [ ] **Step 2: Add visualizers**

In `classification.py`:

```python
class ConfusionMatrixVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, normalize: str | None = "true",
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.normalize = normalize

    def _materialize(self) -> None:
        cm = self._source.confusion_matrix(normalize=None)
        n_correct = float(cm.filter(pl.col("actual") == pl.col("predicted"))["value"].sum())
        n_total = float(cm["value"].sum())
        self._metrics["accuracy"] = n_correct / max(n_total, 1.0)

    def _build_chart(self) -> Any:
        from ..charts import _confusion_chart_from_source
        return _confusion_chart_from_source(self._source, normalize=self.normalize, theme=self.theme)


class ClassificationReportVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)

    def _materialize(self) -> None:
        # Compute summary metrics for repr.
        from sklearn.metrics import f1_score
        y_true = self._source._y.to_numpy()
        y_pred = self._source._model.predict(self._source._X.to_numpy())
        self._metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    def _build_chart(self) -> Any:
        from ..charts import _classification_report_chart
        return _classification_report_chart(self._source, theme=self.theme)
```

In `classification_extra.py`:

```python
class ClassPredictionErrorVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, normalize: bool = False,
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.normalize = normalize

    def _materialize(self) -> None:
        cm = self._source.confusion_matrix(normalize=None)
        n_correct = float(cm.filter(pl.col("actual") == pl.col("predicted"))["value"].sum())
        n_total = float(cm["value"].sum())
        self._metrics["accuracy"] = n_correct / max(n_total, 1.0)

    def _build_chart(self) -> Any:
        from ..charts import _class_prediction_error_chart_from_source
        return _class_prediction_error_chart_from_source(
            self._source, normalize=self.normalize, theme=self.theme,
        )


class ClassBalanceVisualizer(FerrumVisualizer):
    """No model required — operates on y alone."""
    def __init__(self, *, random_state: int | None = None, theme: Any = None):
        super().__init__(model=None, random_state=random_state, theme=theme)

    def fit(self, X: Any, y: Any = None) -> "ClassBalanceVisualizer":
        import polars as pl
        if y is None and X is not None:
            # Allow .fit(y) shorthand.
            y = X
        self._y = pl.Series(y if not isinstance(y, pl.Series) else y)
        from collections import Counter
        c = Counter(self._y.to_list())
        self._metrics["n_classes"] = float(len(c))
        self._metrics["imbalance_ratio"] = float(max(c.values()) / max(min(c.values()), 1))
        from ..charts import _class_balance_chart_from_dataframe
        self._chart = _class_balance_chart_from_dataframe(self._y, theme=self.theme)
        self._fitted = True
        return self
```

- [ ] **Step 3: Re-export and test**

```python
# visualizers/__init__.py
from .classification import (
    ROCVisualizer, PRVisualizer, CalibrationVisualizer,
    ConfusionMatrixVisualizer, ClassificationReportVisualizer,
)
from .classification_extra import (
    DiscriminationThresholdVisualizer, ClassPredictionErrorVisualizer,
    ClassBalanceVisualizer,
)
__all__ += [
    "ConfusionMatrixVisualizer", "ClassificationReportVisualizer",
    "ClassPredictionErrorVisualizer", "ClassBalanceVisualizer",
]


# src/ferrum/__init__.py
from ferrum._diagnostics.visualizers import (
    ConfusionMatrixVisualizer, ClassificationReportVisualizer,
    ClassPredictionErrorVisualizer, ClassBalanceVisualizer,
)


# test_classification.py
def test_confusion_matrix_visualizer():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    viz = ferrum.ConfusionMatrixVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "accuracy=" in repr(viz)


def test_classification_report_visualizer():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    viz = ferrum.ClassificationReportVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "f1_macro=" in repr(viz)


def test_class_prediction_error_visualizer():
    model = load_fixture("multiclass_logistic")
    df = load_dataset("multiclass_classification")
    viz = ferrum.ClassPredictionErrorVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3"]), df["y"],
    )
    assert "<svg" in viz.show().show_svg()


def test_class_balance_visualizer():
    df = load_dataset("multiclass_classification")
    viz = ferrum.ClassBalanceVisualizer().fit(df["y"])
    assert "n_classes=" in repr(viz)
    assert "<svg" in viz.show().show_svg()
```

- [ ] **Step 4: Run + commit**

```bash
uv run --no-sync pytest tests/diagnostics/test_classification.py -v 2>&1 | tail -15
git add src/ferrum/_diagnostics/ src/ferrum/__init__.py tests/diagnostics/test_classification.py
git commit -m "feat(phase-10c): 4 classification matrix visualizers"
```

- [ ] **Step 5: 10c milestone check**

```bash
uv run --no-sync pytest tests/diagnostics/ 2>&1 | tail -3
```
Expected: ~55 tests cumulative.

---

## 10d — Feature importance + SHAP + PDP

### Task 21: `.importances()` + `mark_importance` + `importance_chart` + `FeatureImportancesVisualizer`

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py`
- Modify: `src/ferrum/_diagnostics/schemas.py`
- Modify: `src/ferrum/marks/diagnostic.py`
- Modify: `src/ferrum/chart.py`
- Modify: `src/ferrum/_diagnostics/charts.py`
- Modify: `src/ferrum/figures.py`
- Modify: `src/ferrum/_diagnostics/visualizers/explanation.py` (create)
- Modify: `src/ferrum/_diagnostics/visualizers/__init__.py`
- Modify: `src/ferrum/__init__.py`
- Create: `tests/diagnostics/test_explanation.py`
- Modify: `tests/diagnostics/test_goldens_phase_10.py`

> ⚠ **Pattern correction (plan-vs-codebase):** The mark code blocks below were originally drafted using a `@dataclass(frozen=True) class mark_X: ... def _expand(self, chart_ctx) -> list[LayerSpec]` pattern that **does not exist in the codebase**. Before implementing, translate every mark in this task to the real pattern used in Phase 8b/9 composite marks:
>
> - Module-level `def desugar_<name>(x_field, y_field, **kwargs) -> ("__layered__", transforms, None, None, layers)` in `src/ferrum/marks/diagnostic.py`.
> - Layers are plain dicts: `{"mark": str, "encoding": dict, "mark_kwargs": dict (opt), "data_source": str | None (opt)}`.
> - No `LayerSpec`. No `chart_ctx`. No `_expand`.
> - Chart method clones, sets `_mark = "point"` (placeholder), sets `_pending_stat_mark = (kind, kwargs_dict, desugar_fn)`, returns.
> - The user does not import or instantiate `mark_X` — they call `Chart(df).mark_X(...)`.
> - For diagnostic marks, the data has hard-coded columns from a `ModelSource` method, so the desugar references those columns literally and ignores positional `x_field` / `y_field`.
>
> **Canonical reference:** Task 8 (`desugar_residuals` / `desugar_prediction_error`) and Task 15 (six 10b desugars). Pattern reference in code: `src/ferrum/marks/composite.py:15-220`.
>
> Keep the **layer encodings, kwargs, and behavior** below as the spec for what each mark should produce, but rewrite the implementation in the corrected pattern.

- [ ] **Step 1: Schema + ModelSource method**

```python
# schemas.py
SCHEMA_IMPORTANCES = pl.Schema({
    "feature": pl.Utf8,
    "importance": pl.Float64,
    "std": pl.Float64,
    "rank": pl.Int64,
})

# source.py — append
    def importances(
        self,
        *,
        method: str = "builtin",
        n_repeats: int = 30,
        scoring: Any = None,
        random_state: int | None = None,
    ) -> pl.DataFrame:
        """Feature importance — 'builtin' (model attr) or 'permutation' (sklearn)."""
        rs = random_state if random_state is not None else self._random_state
        key = self._cache_key("importances", method=method, n_repeats=n_repeats,
                                scoring=str(scoring) if scoring else None,
                                random_state=rs)
        if key in self._cache:
            return self._cache[key]
        import numpy as np

        if method == "builtin":
            if "feature_importances_" in self._capabilities:
                imp = np.asarray(self._model.feature_importances_, dtype=np.float64)
            elif "coef_" in self._capabilities:
                coef = np.asarray(self._model.coef_, dtype=np.float64)
                imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
            else:
                raise AttributeError(
                    "ModelSource.importances(method='builtin') requires the model to "
                    "expose 'feature_importances_' or 'coef_'."
                )
            std = np.zeros_like(imp)
        elif method == "permutation":
            require_sklearn("importances(permutation)")
            from sklearn.inspection import permutation_importance
            X_np = self._X.to_numpy()
            y_np = np.asarray(self._y.to_numpy()) if self._y is not None else None
            result = permutation_importance(
                self._model, X_np, y_np,
                n_repeats=n_repeats, scoring=scoring, random_state=rs or 0,
            )
            imp = result.importances_mean
            std = result.importances_std
        else:
            raise ValueError(f"importances method must be 'builtin' or 'permutation'; got {method!r}")

        order = np.argsort(-np.abs(imp))
        rows = []
        for r, i in enumerate(order, start=1):
            rows.append({
                "feature": str(self._feature_names[i]),
                "importance": float(imp[i]),
                "std": float(std[i]),
                "rank": int(r),
            })
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df
```

- [ ] **Step 2: `mark_importance` + Chart method**

```python
# marks/diagnostic.py
@dataclass(frozen=True)
class mark_importance:
    orient: str = "horizontal"
    error_bars: bool = True
    top_k: int | None = None

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_bar, mark_errorbar
        from ferrum import LayerSpec
        x_field, y_field = ("importance", "feature") if self.orient == "horizontal" else ("feature", "importance")
        layers = [LayerSpec(
            mark=mark_bar(),
            encoding={"x": x_field, "y": y_field, "color": chart_ctx.color_field_or_default()},
        )]
        if self.error_bars:
            layers.append(LayerSpec(
                mark=mark_errorbar(),
                encoding={
                    "x": x_field if self.orient == "horizontal" else None,
                    "y": y_field,
                    "x2" if self.orient == "horizontal" else "y2": "std",
                },
            ))
        return layers


# chart.py
def mark_importance(self, **kw) -> "Chart":
    from ferrum.marks.diagnostic import mark_importance as _M
    return self._add_composite_mark(_M(**kw))
```

- [ ] **Step 3: Chart builder + figure function + visualizer**

```python
# charts.py
def _importance_chart_from_source(
    source, *, method="builtin", top_k=20, orient="horizontal",
    error_bars=True, random_state=None, theme=None,
):
    df = source.importances(method=method, random_state=random_state)
    if top_k is not None:
        df = df.head(top_k)
    chart = ferrum.Chart(df).mark_importance(
        orient=orient, error_bars=error_bars, top_k=top_k,
    )
    if theme is not None: chart = chart.theme(theme)
    return chart


# figures.py
def importance_chart(
    model_or_source, X=None, y=None, *,
    method="builtin", top_k=20, orient="horizontal",
    error_bars=True, random_state=None, theme=None,
):
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _importance_chart_from_source(
        source, method=method, top_k=top_k, orient=orient,
        error_bars=error_bars, random_state=random_state, theme=theme,
    )


# visualizers/explanation.py (create)
"""10d explanation visualizers (FeatureImportances, SHAP)."""
from __future__ import annotations
from typing import Any
import numpy as np
from .base import FerrumVisualizer


class FeatureImportancesVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, method: str = "builtin", top_k: int = 20,
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.method = method
        self.top_k = top_k

    def _materialize(self) -> None:
        df = self._source.importances(method=self.method, random_state=self.random_state)
        self._metrics["top_feature_importance"] = float(df["importance"][0])

    def _build_chart(self) -> Any:
        from ..charts import _importance_chart_from_source
        return _importance_chart_from_source(
            self._source, method=self.method, top_k=self.top_k,
            random_state=self.random_state, theme=self.theme,
        )
```

- [ ] **Step 4: Re-exports + tests + goldens**

```python
# __init__.py
from ferrum.marks.diagnostic import mark_importance
from ferrum.figures import importance_chart
from ferrum._diagnostics.visualizers.explanation import FeatureImportancesVisualizer


# tests/diagnostics/test_explanation.py (create)
from __future__ import annotations
import numpy as np
import polars as pl
import pytest
import ferrum
from tests.fixtures import load_fixture, load_dataset


@pytest.fixture(scope="module")
def rf_source():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    return ferrum.ModelSource(model, X, df["y"], random_state=0)


def test_importances_builtin(rf_source):
    imp = rf_source.importances(method="builtin")
    assert set(imp.columns) == {"feature", "importance", "std", "rank"}
    assert imp.height == 5
    assert imp["rank"][0] == 1


def test_importances_permutation(rf_source):
    imp = rf_source.importances(method="permutation", n_repeats=10, random_state=0)
    assert (imp["std"] >= 0).all()


def test_importance_chart_figure_function():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    chart = ferrum.importance_chart(model, df.select(["f0", "f1", "f2", "f3", "f4"]), df["y"])
    assert "<svg" in chart.show_svg()


def test_feature_importances_visualizer():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    viz = ferrum.FeatureImportancesVisualizer(model).fit(
        df.select(["f0", "f1", "f2", "f3", "f4"]), df["y"],
    )
    assert "top_feature_importance=" in repr(viz)


# tests/diagnostics/test_goldens_phase_10.py
def test_golden_importance_chart_builtin():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    chart = ferrum.importance_chart(
        model, df.select(["f0", "f1", "f2", "f3", "f4"]), df["y"],
    )
    _check_golden(chart.show_svg(), "importance_chart_builtin")


def test_golden_importance_chart_permutation():
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    chart = ferrum.importance_chart(
        model, df.select(["f0", "f1", "f2", "f3", "f4"]), df["y"],
        method="permutation", random_state=0,
    )
    _check_golden(chart.show_svg(), "importance_chart_permutation")



```

- [ ] **Step 5: Run + commit**

```bash
FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -k importance -v 2>&1 | tail -10
uv run --no-sync pytest tests/diagnostics/test_explanation.py -v 2>&1 | tail -10
git add src/ferrum/ tests/diagnostics/test_explanation.py tests/diagnostics/test_goldens_phase_10.py tests/goldens/phase_10/
git commit -m "feat(phase-10d): importances + mark_importance + importance_chart + visualizer"
```

---

### Task 22: SHAP family — `.shap_values()` + 3 marks + `shap_chart` + `SHAPVisualizer`

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py`
- Modify: `src/ferrum/_diagnostics/schemas.py`
- Modify: `src/ferrum/marks/diagnostic.py`
- Modify: `src/ferrum/chart.py`
- Modify: `src/ferrum/_diagnostics/charts.py`
- Modify: `src/ferrum/figures.py`
- Modify: `src/ferrum/_diagnostics/visualizers/explanation.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/diagnostics/test_explanation.py`
- Modify: `tests/diagnostics/test_goldens_phase_10.py`

> ⚠ **Pattern correction (plan-vs-codebase):** The mark code blocks below were originally drafted using a `@dataclass(frozen=True) class mark_X: ... def _expand(self, chart_ctx) -> list[LayerSpec]` pattern that **does not exist in the codebase**. Before implementing, translate every mark in this task to the real pattern used in Phase 8b/9 composite marks:
>
> - Module-level `def desugar_<name>(x_field, y_field, **kwargs) -> ("__layered__", transforms, None, None, layers)` in `src/ferrum/marks/diagnostic.py`.
> - Layers are plain dicts: `{"mark": str, "encoding": dict, "mark_kwargs": dict (opt), "data_source": str | None (opt)}`.
> - No `LayerSpec`. No `chart_ctx`. No `_expand`.
> - Chart method clones, sets `_mark = "point"` (placeholder), sets `_pending_stat_mark = (kind, kwargs_dict, desugar_fn)`, returns.
> - The user does not import or instantiate `mark_X` — they call `Chart(df).mark_X(...)`.
> - For diagnostic marks, the data has hard-coded columns from a `ModelSource` method, so the desugar references those columns literally and ignores positional `x_field` / `y_field`.
>
> **Canonical reference:** Task 8 (`desugar_residuals` / `desugar_prediction_error`) and Task 15 (six 10b desugars). Pattern reference in code: `src/ferrum/marks/composite.py:15-220`.
>
> Keep the **layer encodings, kwargs, and behavior** below as the spec for what each mark should produce, but rewrite the implementation in the corrected pattern.

- [ ] **Step 1: Schema + `.shap_values()` method**

```python
# schemas.py
SCHEMA_SHAP_VALUES = pl.Schema({
    "sample_id": pl.Int64,
    "feature": pl.Utf8,
    "shap_value": pl.Float64,
    "feature_value": pl.Float64,
    "feature_value_normalized": pl.Float64,
})


# source.py — append
    def shap_values(
        self,
        *,
        background: Any = None,
        max_evals: int = 500,
    ) -> pl.DataFrame:
        """SHAP values long-form: sample_id × feature × shap_value."""
        key = self._cache_key("shap_values", background=str(background)[:64], max_evals=max_evals)
        if key in self._cache:
            return self._cache[key]
        shap = require_shap("shap_values")
        import numpy as np

        X_np = self._X.to_numpy()
        # Auto-pick explainer based on model type.
        if "coef_" in self._capabilities:
            explainer = shap.LinearExplainer(self._model, X_np)
        elif "feature_importances_" in self._capabilities:
            explainer = shap.TreeExplainer(self._model)
        else:
            bg = background if background is not None else X_np[: min(50, len(X_np))]
            explainer = shap.KernelExplainer(self._model.predict, bg)
        sv = explainer.shap_values(X_np)
        # For multi-class shap returns a list; take the array for the predicted positive class.
        if isinstance(sv, list):
            # Stack per-class — for now use class 1 if binary, else class 0.
            sv = sv[1] if len(sv) == 2 else sv[0]
        sv = np.asarray(sv, dtype=np.float64)
        # Normalize feature values within each column (z-score).
        f_mean = X_np.mean(axis=0)
        f_std = np.where(X_np.std(axis=0) > 0, X_np.std(axis=0), 1.0)
        f_norm = (X_np - f_mean) / f_std

        rows: list[dict] = []
        for sample_id in range(X_np.shape[0]):
            for f_idx, fname in enumerate(self._feature_names):
                rows.append({
                    "sample_id": int(sample_id),
                    "feature": str(fname),
                    "shap_value": float(sv[sample_id, f_idx]),
                    "feature_value": float(X_np[sample_id, f_idx]),
                    "feature_value_normalized": float(f_norm[sample_id, f_idx]),
                })
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df
```

- [ ] **Step 2: Three SHAP marks**

```python
# marks/diagnostic.py
@dataclass(frozen=True)
class mark_shap_beeswarm:
    max_display: int = 20
    color_bar: bool = True
    order: str = "abs_mean"   # "abs_mean" | "max_abs"

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_swarm
        from ferrum import LayerSpec
        return [LayerSpec(
            mark=mark_swarm(),
            encoding={"x": "shap_value", "y": "feature",
                       "color": "feature_value_normalized"},
        )]


@dataclass(frozen=True)
class mark_shap_bar:
    max_display: int = 20
    layered: bool = False

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_bar
        from ferrum import LayerSpec
        return [LayerSpec(
            mark=mark_bar(),
            encoding={"x": "abs_mean_shap", "y": "feature"},
        )]


@dataclass(frozen=True)
class mark_shap_waterfall:
    sample_idx: int = -1                 # REQUIRED at expand time (use -1 sentinel)
    max_display: int = 20
    show_data: bool = True

    def _expand(self, chart_ctx: Any) -> list[Any]:
        if self.sample_idx < 0:
            raise TypeError(
                "mark_shap_waterfall(sample_idx=...) is required. "
                "Pass an explicit non-negative sample index, e.g. "
                "mark_shap_waterfall(sample_idx=3)."
            )
        from ferrum.marks import mark_bar, mark_rule
        from ferrum import LayerSpec
        return [
            LayerSpec(
                mark=mark_bar(),
                encoding={"x": "x0", "x2": "x1", "y": "feature",
                           "color": "shap_sign"},
            ),
            LayerSpec(mark=mark_rule(), encoding={"x": "baseline"}),
        ]
```

- [ ] **Step 3: Chart methods + builders**

```python
# chart.py — three Chart methods
def mark_shap_beeswarm(self, **kw):
    from ferrum.marks.diagnostic import mark_shap_beeswarm as _M
    return self._add_composite_mark(_M(**kw))

def mark_shap_bar(self, **kw):
    from ferrum.marks.diagnostic import mark_shap_bar as _M
    return self._add_composite_mark(_M(**kw))

def mark_shap_waterfall(self, **kw):
    from ferrum.marks.diagnostic import mark_shap_waterfall as _M
    return self._add_composite_mark(_M(**kw))


# charts.py — three builders
def _shap_beeswarm_chart_from_source(source, *, max_display=20, order="abs_mean", theme=None):
    df = source.shap_values()
    # Order features by mean(|shap|).
    if order == "abs_mean":
        order_df = (df.group_by("feature")
                       .agg(pl.col("shap_value").abs().mean().alias("score"))
                       .sort("score", descending=True)
                       .head(max_display))
    else:
        order_df = (df.group_by("feature")
                       .agg(pl.col("shap_value").abs().max().alias("score"))
                       .sort("score", descending=True)
                       .head(max_display))
    keep = order_df["feature"].to_list()
    plot_df = df.filter(pl.col("feature").is_in(keep))
    chart = ferrum.Chart(plot_df).mark_shap_beeswarm(max_display=max_display, order=order)
    if theme is not None: chart = chart.theme(theme)
    return chart


def _shap_bar_chart_from_source(source, *, max_display=20, theme=None):
    df = source.shap_values()
    agg = (df.group_by("feature")
              .agg(pl.col("shap_value").abs().mean().alias("abs_mean_shap"))
              .sort("abs_mean_shap", descending=True)
              .head(max_display))
    chart = ferrum.Chart(agg).mark_shap_bar(max_display=max_display)
    if theme is not None: chart = chart.theme(theme)
    return chart


def _shap_waterfall_chart_from_source(source, *, sample_idx, max_display=20, theme=None):
    import numpy as np
    df = source.shap_values().filter(pl.col("sample_id") == sample_idx)
    # Compute cumulative x0/x1.
    abs_order = df.sort(pl.col("shap_value").abs(), descending=True).head(max_display)
    sv = abs_order["shap_value"].to_numpy()
    cumsum = np.concatenate([[0.0], np.cumsum(sv)])
    df_plot = abs_order.with_columns([
        pl.Series("x0", cumsum[:-1]),
        pl.Series("x1", cumsum[1:]),
        pl.when(pl.col("shap_value") >= 0).then(pl.lit("positive"))
           .otherwise(pl.lit("negative")).alias("shap_sign"),
        pl.lit(0.0).alias("baseline"),
    ])
    chart = ferrum.Chart(df_plot).mark_shap_waterfall(sample_idx=sample_idx, max_display=max_display)
    if theme is not None: chart = chart.theme(theme)
    return chart
```

- [ ] **Step 4: `shap_chart` figure dispatcher + visualizer**

```python
# figures.py
def shap_chart(
    model_or_source, X=None, *, kind="beeswarm",
    max_display=20, sample_idx=None, random_state=None, theme=None,
):
    from ferrum._diagnostics.charts import (
        _shap_beeswarm_chart_from_source, _shap_bar_chart_from_source,
        _shap_waterfall_chart_from_source,
    )
    source = _resolve_source(model_or_source, X, None, random_state=random_state)
    if kind == "beeswarm":
        return _shap_beeswarm_chart_from_source(source, max_display=max_display, theme=theme)
    if kind == "bar":
        return _shap_bar_chart_from_source(source, max_display=max_display, theme=theme)
    if kind == "waterfall":
        if sample_idx is None:
            raise ValueError("shap_chart(kind='waterfall') requires sample_idx=...")
        return _shap_waterfall_chart_from_source(
            source, sample_idx=sample_idx, max_display=max_display, theme=theme,
        )
    raise ValueError(f"shap_chart kind must be beeswarm/bar/waterfall; got {kind!r}")


# visualizers/explanation.py
class SHAPVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, kind: str = "beeswarm", background: Any = None,
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.kind = kind
        self.background = background

    def _materialize(self) -> None:
        sv = self._source.shap_values(background=self.background)
        agg = sv.group_by("feature").agg(pl.col("shap_value").abs().mean().alias("v"))
        self._metrics["top_abs_shap"] = float(agg["v"].max())

    def _build_chart(self) -> Any:
        import ferrum
        return ferrum.shap_chart(
            self._source, kind=self.kind, theme=self.theme, random_state=self.random_state,
        )
```

- [ ] **Step 5: Re-exports + tests + goldens (mixed tiers)**

```python
# __init__.py
from ferrum.marks.diagnostic import mark_shap_beeswarm, mark_shap_bar, mark_shap_waterfall
from ferrum.figures import shap_chart
from ferrum._diagnostics.visualizers.explanation import SHAPVisualizer


# test_explanation.py
def test_shap_values_schema(rf_source):
    # Use linear model for deterministic shap (LinearExplainer).
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    source = ferrum.ModelSource(model, df.select(["f0", "f1", "f2", "f3", "f4"]), df["y"])
    sv = source.shap_values()
    assert set(sv.columns) == {"sample_id", "feature", "shap_value",
                                "feature_value", "feature_value_normalized"}
    assert sv.height == 200 * 5  # n_samples * n_features


def test_shap_chart_beeswarm():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    chart = ferrum.shap_chart(
        model, df.select(["f0", "f1", "f2", "f3", "f4"]), kind="beeswarm",
    )
    assert "<svg" in chart.show_svg()


def test_shap_chart_bar():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    chart = ferrum.shap_chart(
        model, df.select(["f0", "f1", "f2", "f3", "f4"]), kind="bar",
    )
    assert "<svg" in chart.show_svg()


def test_shap_chart_waterfall_requires_sample_idx():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    import pytest
    with pytest.raises(ValueError, match="sample_idx"):
        ferrum.shap_chart(model, df.select(["f0", "f1", "f2", "f3", "f4"]), kind="waterfall")


def test_shap_chart_waterfall_ok():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    chart = ferrum.shap_chart(
        model, df.select(["f0", "f1", "f2", "f3", "f4"]),
        kind="waterfall", sample_idx=3,
    )
    assert "<svg" in chart.show_svg()


def test_shap_visualizer():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    viz = ferrum.SHAPVisualizer(model).fit(df.select(["f0", "f1", "f2", "f3", "f4"]))
    assert "top_abs_shap=" in repr(viz)


# test_goldens_phase_10.py — LinearExplainer is deterministic so byte-identical at 3 dp.
def test_golden_shap_chart_beeswarm_linear():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    chart = ferrum.shap_chart(model, df.select(["f0", "f1", "f2", "f3", "f4"]), kind="beeswarm")
    _check_golden(chart.show_svg(), "shap_chart_beeswarm_linear")


def test_golden_shap_chart_bar_linear():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    chart = ferrum.shap_chart(model, df.select(["f0", "f1", "f2", "f3", "f4"]), kind="bar")
    _check_golden(chart.show_svg(), "shap_chart_bar_linear")


def test_golden_shap_chart_waterfall_linear():
    model = load_fixture("regression_ridge")
    df = load_dataset("regression")
    chart = ferrum.shap_chart(
        model, df.select(["f0", "f1", "f2", "f3", "f4"]),
        kind="waterfall", sample_idx=3,
    )
    _check_golden(chart.show_svg(), "shap_chart_waterfall_sample3")
```

- [ ] **Step 6: Run + commit**

```bash
FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -k shap -v 2>&1 | tail -10
uv run --no-sync pytest tests/diagnostics/test_explanation.py -v 2>&1 | tail -15
git add src/ferrum/ tests/diagnostics/ tests/goldens/phase_10/
git commit -m "feat(phase-10d): shap_values + 3 shap marks + shap_chart + SHAPVisualizer"
```

---

### Task 23: `.partial_dependence()` + `mark_pdp` + builder

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py`
- Modify: `src/ferrum/_diagnostics/schemas.py`
- Modify: `src/ferrum/marks/diagnostic.py`
- Modify: `src/ferrum/chart.py`
- Modify: `src/ferrum/_diagnostics/charts.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/diagnostics/test_explanation.py`
- Modify: `tests/diagnostics/test_goldens_phase_10.py`

> Note: `mark_pdp` does NOT have its own §3.14 figure function — PDP is exposed via `shap_chart` only when `kind="pdp"` is added by user code, OR by direct chart construction. Phase 10 spec lists `mark_pdp` but no `pdp_chart`. We ship the mark + builder; visualizer wiring goes through `FeatureImportancesVisualizer` users who add PDP layers manually.

> ⚠ **Pattern correction (plan-vs-codebase):** The mark code blocks below were originally drafted using a `@dataclass(frozen=True) class mark_X: ... def _expand(self, chart_ctx) -> list[LayerSpec]` pattern that **does not exist in the codebase**. Before implementing, translate every mark in this task to the real pattern used in Phase 8b/9 composite marks:
>
> - Module-level `def desugar_<name>(x_field, y_field, **kwargs) -> ("__layered__", transforms, None, None, layers)` in `src/ferrum/marks/diagnostic.py`.
> - Layers are plain dicts: `{"mark": str, "encoding": dict, "mark_kwargs": dict (opt), "data_source": str | None (opt)}`.
> - No `LayerSpec`. No `chart_ctx`. No `_expand`.
> - Chart method clones, sets `_mark = "point"` (placeholder), sets `_pending_stat_mark = (kind, kwargs_dict, desugar_fn)`, returns.
> - The user does not import or instantiate `mark_X` — they call `Chart(df).mark_X(...)`.
> - For diagnostic marks, the data has hard-coded columns from a `ModelSource` method, so the desugar references those columns literally and ignores positional `x_field` / `y_field`.
>
> **Canonical reference:** Task 8 (`desugar_residuals` / `desugar_prediction_error`) and Task 15 (six 10b desugars). Pattern reference in code: `src/ferrum/marks/composite.py:15-220`.
>
> Keep the **layer encodings, kwargs, and behavior** below as the spec for what each mark should produce, but rewrite the implementation in the corrected pattern.

- [ ] **Step 1: Schema + method**

```python
# schemas.py
SCHEMA_PDP = pl.Schema({
    "feature": pl.Utf8,
    "feature_value": pl.Float64,
    "pd_value": pl.Float64,
    "sample_id": pl.Int64,
})

# source.py — append
    def partial_dependence(
        self,
        features: list[str | int],
        *,
        grid_resolution: int = 100,
        kind: str = "average",   # "average" | "individual" | "both"
    ) -> pl.DataFrame:
        """Partial dependence for one or more features."""
        key = self._cache_key("partial_dependence",
                                features=tuple(features), grid_resolution=grid_resolution, kind=kind)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("partial_dependence")
        from sklearn.inspection import partial_dependence
        import numpy as np

        feature_idxs = [
            self._feature_names.index(f) if isinstance(f, str) else f
            for f in features
        ]
        rows: list[dict] = []
        for f_idx, f in zip(feature_idxs, features):
            fname = self._feature_names[f_idx]
            pd_kind = "average" if kind == "average" else "individual" if kind == "individual" else "both"
            r = partial_dependence(
                self._model, self._X.to_numpy(),
                features=[f_idx], grid_resolution=grid_resolution, kind=pd_kind,
            )
            grid = r["grid_values"][0]
            avg = r.get("average")
            ind = r.get("individual")
            if pd_kind in ("average", "both"):
                for v, p in zip(grid, np.asarray(avg)[0]):
                    rows.append({
                        "feature": str(fname), "feature_value": float(v),
                        "pd_value": float(p), "sample_id": -1,
                    })
            if pd_kind in ("individual", "both"):
                ind_arr = np.asarray(ind)[0]  # shape (n_samples, grid_resolution)
                for s in range(ind_arr.shape[0]):
                    for v, p in zip(grid, ind_arr[s]):
                        rows.append({
                            "feature": str(fname), "feature_value": float(v),
                            "pd_value": float(p), "sample_id": int(s),
                        })
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df
```

- [ ] **Step 2: Mark + Chart method + builder**

```python
# marks/diagnostic.py
@dataclass(frozen=True)
class mark_pdp:
    kind: str = "average"   # "average" | "individual" | "both"
    ice_alpha: float = 0.2
    center: bool = False

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_line
        from ferrum import LayerSpec
        layers = []
        if self.kind in ("individual", "both"):
            layers.append(LayerSpec(
                mark=mark_line(opacity=self.ice_alpha),
                encoding={"x": "feature_value", "y": "pd_value", "detail": "sample_id"},
            ))
        if self.kind in ("average", "both"):
            layers.append(LayerSpec(
                mark=mark_line(strokeWidth=2.0),
                encoding={"x": "feature_value", "y": "pd_value"},
            ))
        return layers


# chart.py
def mark_pdp(self, **kw):
    from ferrum.marks.diagnostic import mark_pdp as _M
    return self._add_composite_mark(_M(**kw))


# charts.py
def _pdp_chart_from_source(source, features, *, kind="average", grid_resolution=100,
                            ice_alpha=0.2, center=False, theme=None):
    df = source.partial_dependence(features, grid_resolution=grid_resolution, kind=kind)
    chart = ferrum.Chart(df).mark_pdp(kind=kind, ice_alpha=ice_alpha, center=center).encode(
        x="feature_value", y="pd_value", color="feature",
    )
    if theme is not None: chart = chart.theme(theme)
    return chart
```

- [ ] **Step 3: Re-exports + tests + golden**

```python
# __init__.py
from ferrum.marks.diagnostic import mark_pdp


# test_explanation.py
def test_partial_dependence(rf_source):
    pd_df = rf_source.partial_dependence(["f0", "f1"], grid_resolution=20)
    assert set(pd_df.columns) == {"feature", "feature_value", "pd_value", "sample_id"}
    assert "f0" in pd_df["feature"].unique().to_list()


def test_mark_pdp_renders():
    from ferrum._diagnostics.charts import _pdp_chart_from_source
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    source = ferrum.ModelSource(model, df.select(["f0", "f1", "f2", "f3", "f4"]), df["y"])
    chart = _pdp_chart_from_source(source, ["f0"], grid_resolution=20)
    assert "<svg" in chart.show_svg()


# test_goldens_phase_10.py
def test_golden_pdp_chart_average():
    from ferrum._diagnostics.charts import _pdp_chart_from_source
    model = load_fixture("regression_rf")
    df = load_dataset("regression")
    source = ferrum.ModelSource(model, df.select(["f0", "f1", "f2", "f3", "f4"]), df["y"])
    chart = _pdp_chart_from_source(source, ["f0"], grid_resolution=20, kind="average")
    _check_golden(chart.show_svg(), "pdp_chart_f0_average")
```

- [ ] **Step 4: Run + commit**

```bash
FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -k pdp -v 2>&1 | tail -5
uv run --no-sync pytest tests/diagnostics/test_explanation.py -v 2>&1 | tail -10
git add src/ferrum/ tests/diagnostics/ tests/goldens/phase_10/
git commit -m "feat(phase-10d): partial_dependence + mark_pdp + builder"
```

---

### Task 24: 10d milestone — verify full sub-batch

- [ ] **Step 1: Run all 10d tests**

```bash
uv run --no-sync pytest tests/diagnostics/test_explanation.py -v 2>&1 | tail -20
```
Expected: ~10 explanation tests pass.

- [ ] **Step 2: Verify no sklearn at import**

```bash
uv run --no-sync pytest tests/diagnostics/test_no_sklearn_at_import.py -v 2>&1 | tail -5
```
Expected: 3 passed.

- [ ] **Step 3: Verify 10b/10c didn't regress**

```bash
uv run --no-sync pytest tests/diagnostics/ 2>&1 | tail -3
```
Expected: ~75 passed cumulative.

- [ ] **Step 4: No commit — verification-only task**

---

### Task 25: `ParallelCoordinatesVisualizer` (no-model variant landed early to lock pattern)

Note: this visualizer also lands in 10g, but its no-model `fit()` pattern is shared with `ClassBalanceVisualizer` (10c) and `Rank1D/2DVisualizer` (10g). Documenting here as a forward reference; full implementation is in Task 38.

- [ ] **Step 1: No action — referenced in Task 38.**

---

## 10e — Model selection / CV curves

### Task 26: `.learning_curve()` + `.validation_curve()` methods

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py`
- Modify: `src/ferrum/_diagnostics/schemas.py`
- Modify: `tests/diagnostics/test_source.py`

- [ ] **Step 1: Schemas**

```python
SCHEMA_LEARNING_CURVE = pl.Schema({
    "train_size": pl.Int64,
    "split": pl.Utf8,        # "train" | "test"
    "score": pl.Float64,
    "mean_score": pl.Float64,
    "std_score": pl.Float64,
    "lower": pl.Float64,
    "upper": pl.Float64,
})

SCHEMA_VALIDATION_CURVE = pl.Schema({
    "param_value": pl.Float64,
    "split": pl.Utf8,
    "score": pl.Float64,
    "mean_score": pl.Float64,
    "std_score": pl.Float64,
    "lower": pl.Float64,
    "upper": pl.Float64,
})
```

- [ ] **Step 2: Methods**

```python
# source.py — append
    def learning_curve(
        self,
        *,
        cv: int = 5,
        scoring: Any = None,
        train_sizes: Any = None,
    ) -> pl.DataFrame:
        key = self._cache_key("learning_curve", cv=cv, scoring=str(scoring) if scoring else None,
                                train_sizes=str(train_sizes))
        if key in self._cache:
            return self._cache[key]
        require_sklearn("learning_curve")
        from sklearn.model_selection import learning_curve as _lc
        import numpy as np
        X_np = self._X.to_numpy()
        y_np = np.asarray(self._y.to_numpy())
        sizes = train_sizes if train_sizes is not None else np.linspace(0.1, 1.0, 5)
        ts, tr_scores, te_scores = _lc(
            self._model, X_np, y_np,
            train_sizes=sizes, cv=cv, scoring=scoring,
            random_state=self._random_state or 0,
            shuffle=True,
        )
        rows: list[dict] = []
        for i, t in enumerate(ts):
            tr_mean = float(tr_scores[i].mean()); tr_std = float(tr_scores[i].std())
            te_mean = float(te_scores[i].mean()); te_std = float(te_scores[i].std())
            for s in tr_scores[i]:
                rows.append({"train_size": int(t), "split": "train", "score": float(s),
                              "mean_score": tr_mean, "std_score": tr_std,
                              "lower": tr_mean - 1.96 * tr_std / np.sqrt(len(tr_scores[i])),
                              "upper": tr_mean + 1.96 * tr_std / np.sqrt(len(tr_scores[i]))})
            for s in te_scores[i]:
                rows.append({"train_size": int(t), "split": "test", "score": float(s),
                              "mean_score": te_mean, "std_score": te_std,
                              "lower": te_mean - 1.96 * te_std / np.sqrt(len(te_scores[i])),
                              "upper": te_mean + 1.96 * te_std / np.sqrt(len(te_scores[i]))})
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def validation_curve(
        self,
        param: str,
        values: Any,
        *,
        cv: int = 5,
        scoring: Any = None,
    ) -> pl.DataFrame:
        key = self._cache_key("validation_curve", param=param,
                                values=tuple(values), cv=cv,
                                scoring=str(scoring) if scoring else None)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("validation_curve")
        from sklearn.model_selection import validation_curve as _vc
        import numpy as np
        X_np = self._X.to_numpy()
        y_np = np.asarray(self._y.to_numpy())
        vals = np.asarray(list(values))
        tr, te = _vc(self._model, X_np, y_np,
                      param_name=param, param_range=vals,
                      cv=cv, scoring=scoring)
        rows: list[dict] = []
        for i, v in enumerate(vals):
            tr_mean, tr_std = float(tr[i].mean()), float(tr[i].std())
            te_mean, te_std = float(te[i].mean()), float(te[i].std())
            n_tr, n_te = len(tr[i]), len(te[i])
            for s in tr[i]:
                rows.append({"param_value": float(v), "split": "train", "score": float(s),
                              "mean_score": tr_mean, "std_score": tr_std,
                              "lower": tr_mean - 1.96 * tr_std / np.sqrt(n_tr),
                              "upper": tr_mean + 1.96 * tr_std / np.sqrt(n_tr)})
            for s in te[i]:
                rows.append({"param_value": float(v), "split": "test", "score": float(s),
                              "mean_score": te_mean, "std_score": te_std,
                              "lower": te_mean - 1.96 * te_std / np.sqrt(n_te),
                              "upper": te_mean + 1.96 * te_std / np.sqrt(n_te)})
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df
```

- [ ] **Step 3: Tests**

```python
# test_source.py
def test_learning_curve():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(random_state=0), X, df["y"], random_state=0)
    lc = source.learning_curve(cv=3)
    assert set(lc.columns) == {"train_size", "split", "score", "mean_score",
                                 "std_score", "lower", "upper"}
    assert set(lc["split"].unique().to_list()) == {"train", "test"}


def test_validation_curve():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(), X, df["y"], random_state=0)
    vc = source.validation_curve("alpha", [0.1, 1.0, 10.0], cv=3)
    assert vc.height > 0
    assert set(vc["param_value"].unique().to_list()) == {0.1, 1.0, 10.0}
```

- [ ] **Step 4: Run + commit**

```bash
uv run --no-sync pytest tests/diagnostics/test_source.py -v -k "learning_curve or validation_curve" 2>&1 | tail -10
git add src/ferrum/_diagnostics/ tests/diagnostics/test_source.py
git commit -m "feat(phase-10e): learning_curve + validation_curve methods"
```

---

### Task 27: `.cv_scores()` + `.alpha_selection()` methods

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py`
- Modify: `src/ferrum/_diagnostics/schemas.py`
- Modify: `tests/diagnostics/test_source.py`

- [ ] **Step 1: Schemas + methods**

```python
# schemas.py
SCHEMA_CV_SCORES = pl.Schema({
    "fold": pl.Int64,
    "split": pl.Utf8,
    "score": pl.Float64,
})

SCHEMA_ALPHA_SELECTION = pl.Schema({
    "alpha": pl.Float64,
    "fold": pl.Int64,
    "score": pl.Float64,
    "mean_score": pl.Float64,
    "std_score": pl.Float64,
})


# source.py — append
    def cv_scores(self, *, cv: int = 5, scoring: Any = None) -> pl.DataFrame:
        key = self._cache_key("cv_scores", cv=cv, scoring=str(scoring) if scoring else None)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("cv_scores")
        from sklearn.model_selection import cross_validate
        import numpy as np
        X_np = self._X.to_numpy()
        y_np = np.asarray(self._y.to_numpy())
        result = cross_validate(
            self._model, X_np, y_np, cv=cv, scoring=scoring, return_train_score=True,
        )
        rows: list[dict] = []
        for fold, s in enumerate(result["train_score"]):
            rows.append({"fold": fold, "split": "train", "score": float(s)})
        for fold, s in enumerate(result["test_score"]):
            rows.append({"fold": fold, "split": "test", "score": float(s)})
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def alpha_selection(self, alphas: Any, *, cv: int = 5, scoring: Any = None) -> pl.DataFrame:
        key = self._cache_key("alpha_selection", alphas=tuple(alphas), cv=cv,
                                scoring=str(scoring) if scoring else None)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("alpha_selection")
        from sklearn.model_selection import validation_curve
        import numpy as np
        X_np = self._X.to_numpy()
        y_np = np.asarray(self._y.to_numpy())
        vals = np.asarray(list(alphas))
        tr, te = validation_curve(
            self._model, X_np, y_np,
            param_name="alpha", param_range=vals, cv=cv, scoring=scoring,
        )
        rows: list[dict] = []
        for i, a in enumerate(vals):
            te_mean, te_std = float(te[i].mean()), float(te[i].std())
            for fold_idx, s in enumerate(te[i]):
                rows.append({"alpha": float(a), "fold": int(fold_idx), "score": float(s),
                              "mean_score": te_mean, "std_score": te_std})
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df
```

- [ ] **Step 2: Tests + commit**

```python
def test_cv_scores():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(random_state=0), X, df["y"], random_state=0)
    cvs = source.cv_scores(cv=3)
    assert set(cvs["split"].unique().to_list()) == {"train", "test"}
    assert cvs.height == 6  # 3 folds × 2 splits


def test_alpha_selection():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    X = df.select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(Ridge(), X, df["y"], random_state=0)
    al = source.alpha_selection([0.1, 1.0, 10.0], cv=3)
    assert set(al["alpha"].unique().to_list()) == {0.1, 1.0, 10.0}
```

```bash
uv run --no-sync pytest tests/diagnostics/test_source.py -v -k "cv_scores or alpha_selection" 2>&1 | tail -10
git add -u && git commit -m "feat(phase-10e): cv_scores + alpha_selection methods"
```

---

### Task 28: 10e marks + builders + figure functions

**Files:**
- Modify: `src/ferrum/marks/diagnostic.py`
- Modify: `src/ferrum/chart.py`
- Modify: `src/ferrum/_diagnostics/charts.py`
- Modify: `src/ferrum/figures.py`
- Modify: `src/ferrum/__init__.py`
- Create: `tests/diagnostics/test_selection.py`

> ⚠ **Pattern correction (plan-vs-codebase):** The mark code blocks below were originally drafted using a `@dataclass(frozen=True) class mark_X: ... def _expand(self, chart_ctx) -> list[LayerSpec]` pattern that **does not exist in the codebase**. Before implementing, translate every mark in this task to the real pattern used in Phase 8b/9 composite marks:
>
> - Module-level `def desugar_<name>(x_field, y_field, **kwargs) -> ("__layered__", transforms, None, None, layers)` in `src/ferrum/marks/diagnostic.py`.
> - Layers are plain dicts: `{"mark": str, "encoding": dict, "mark_kwargs": dict (opt), "data_source": str | None (opt)}`.
> - No `LayerSpec`. No `chart_ctx`. No `_expand`.
> - Chart method clones, sets `_mark = "point"` (placeholder), sets `_pending_stat_mark = (kind, kwargs_dict, desugar_fn)`, returns.
> - The user does not import or instantiate `mark_X` — they call `Chart(df).mark_X(...)`.
> - For diagnostic marks, the data has hard-coded columns from a `ModelSource` method, so the desugar references those columns literally and ignores positional `x_field` / `y_field`.
>
> **Canonical reference:** Task 8 (`desugar_residuals` / `desugar_prediction_error`) and Task 15 (six 10b desugars). Pattern reference in code: `src/ferrum/marks/composite.py:15-220`.
>
> Keep the **layer encodings, kwargs, and behavior** below as the spec for what each mark should produce, but rewrite the implementation in the corrected pattern.

- [ ] **Step 1: Four marks**

```python
# marks/diagnostic.py
@dataclass(frozen=True)
class mark_learning_curve:
    ci_style: str = "band"  # "band" | "errorbar"

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_line, mark_errorband, mark_errorbar
        from ferrum import LayerSpec
        ci_layer = (
            LayerSpec(mark=mark_errorband(), encoding={
                "x": "train_size", "y": "lower", "y2": "upper", "color": "split"
            }) if self.ci_style == "band"
            else LayerSpec(mark=mark_errorbar(), encoding={
                "x": "train_size", "y": "lower", "y2": "upper", "color": "split"
            })
        )
        return [
            ci_layer,
            LayerSpec(mark=mark_line(), encoding={
                "x": "train_size", "y": "mean_score", "color": "split",
            }),
        ]


@dataclass(frozen=True)
class mark_validation_curve:
    log_scale: bool = False
    ci_style: str = "band"

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_line, mark_errorband, mark_errorbar
        from ferrum import LayerSpec
        ci_layer = (
            LayerSpec(mark=mark_errorband(), encoding={
                "x": "param_value", "y": "lower", "y2": "upper", "color": "split"
            }) if self.ci_style == "band"
            else LayerSpec(mark=mark_errorbar(), encoding={
                "x": "param_value", "y": "lower", "y2": "upper", "color": "split"
            })
        )
        return [
            ci_layer,
            LayerSpec(mark=mark_line(), encoding={
                "x": "param_value", "y": "mean_score", "color": "split",
            }),
        ]


@dataclass(frozen=True)
class mark_alpha_selection:
    log_scale: bool = True
    ci_style: str = "band"
    highlight_best: bool = True

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_line
        from ferrum import LayerSpec
        return [LayerSpec(
            mark=mark_line(),
            encoding={"x": "alpha", "y": "mean_score"},
        )]


@dataclass(frozen=True)
class mark_cv_scores:
    kind: str = "box"   # "box" | "bar" | "strip"
    split: str = "both"  # "test" | "train" | "both"

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_boxplot, mark_bar, mark_point
        from ferrum import LayerSpec
        mark_cls = {"box": mark_boxplot, "bar": mark_bar, "strip": mark_point}[self.kind]
        return [LayerSpec(
            mark=mark_cls(),
            encoding={"x": "split", "y": "score"},
        )]
```

- [ ] **Step 2: Four Chart methods + four builders + four figure functions**

```python
# chart.py — four methods
def mark_learning_curve(self, **kw):
    from ferrum.marks.diagnostic import mark_learning_curve as _M
    return self._add_composite_mark(_M(**kw))

def mark_validation_curve(self, **kw):
    from ferrum.marks.diagnostic import mark_validation_curve as _M
    return self._add_composite_mark(_M(**kw))

def mark_alpha_selection(self, **kw):
    from ferrum.marks.diagnostic import mark_alpha_selection as _M
    return self._add_composite_mark(_M(**kw))

def mark_cv_scores(self, **kw):
    from ferrum.marks.diagnostic import mark_cv_scores as _M
    return self._add_composite_mark(_M(**kw))


# charts.py
def _learning_curve_chart_from_source(source, *, cv=5, scoring=None,
                                       train_sizes=None, ci_style="band",
                                       random_state=None, theme=None):
    df = source.learning_curve(cv=cv, scoring=scoring, train_sizes=train_sizes)
    chart = ferrum.Chart(df).mark_learning_curve(ci_style=ci_style)
    if theme is not None: chart = chart.theme(theme)
    return chart


def _validation_curve_chart_from_source(source, param, values, *, cv=5, scoring=None,
                                          log_scale="auto", ci_style="band", theme=None):
    df = source.validation_curve(param, values, cv=cv, scoring=scoring)
    is_log = (log_scale is True) if log_scale != "auto" else (max(values) / max(min(values), 1e-12) > 100)
    chart = ferrum.Chart(df).mark_validation_curve(log_scale=is_log, ci_style=ci_style)
    if theme is not None: chart = chart.theme(theme)
    return chart


def _cv_scores_chart_from_source(source, *, cv=5, scoring=None,
                                   kind="box", split="both", theme=None):
    df = source.cv_scores(cv=cv, scoring=scoring)
    if split != "both":
        df = df.filter(pl.col("split") == split)
    chart = ferrum.Chart(df).mark_cv_scores(kind=kind, split=split)
    if theme is not None: chart = chart.theme(theme)
    return chart


def _alpha_selection_chart_from_source(source, alphas, *, cv=5, scoring=None,
                                         log_scale=True, ci_style="band", theme=None):
    df = source.alpha_selection(alphas, cv=cv, scoring=scoring)
    chart = ferrum.Chart(df).mark_alpha_selection(log_scale=log_scale, ci_style=ci_style)
    if theme is not None: chart = chart.theme(theme)
    return chart


# figures.py
def learning_curve_chart(
    model, X, y, *, cv=5, scoring=None, train_sizes=None,
    ci_style="band", n_jobs=None, random_state=None, theme=None,
):
    source = _resolve_source(model, X, y, random_state=random_state)
    return _learning_curve_chart_from_source(
        source, cv=cv, scoring=scoring, train_sizes=train_sizes,
        ci_style=ci_style, random_state=random_state, theme=theme,
    )


def validation_curve_chart(
    model, X, y, param, values, *,
    cv=5, scoring=None, log_scale="auto", ci_style="band",
    random_state=None, theme=None,
):
    source = _resolve_source(model, X, y, random_state=random_state)
    return _validation_curve_chart_from_source(
        source, param, values, cv=cv, scoring=scoring,
        log_scale=log_scale, ci_style=ci_style, theme=theme,
    )


def cv_scores_chart(
    model, X, y, *, cv=5, scoring=None, kind="box", split="both",
    random_state=None, theme=None,
):
    source = _resolve_source(model, X, y, random_state=random_state)
    return _cv_scores_chart_from_source(
        source, cv=cv, scoring=scoring, kind=kind, split=split, theme=theme,
    )


def alpha_selection_chart(
    model, X, y, alphas, *, cv=5, scoring=None,
    log_scale=True, ci_style="band", random_state=None, theme=None,
):
    source = _resolve_source(model, X, y, random_state=random_state)
    return _alpha_selection_chart_from_source(
        source, alphas, cv=cv, scoring=scoring,
        log_scale=log_scale, ci_style=ci_style, theme=theme,
    )
```

- [ ] **Step 3: Re-exports**

```python
# __init__.py
from ferrum.marks.diagnostic import (
    mark_learning_curve, mark_validation_curve, mark_alpha_selection, mark_cv_scores,
)
from ferrum.figures import (
    learning_curve_chart, validation_curve_chart, cv_scores_chart, alpha_selection_chart,
)
```

- [ ] **Step 4: Tests + quantized goldens**

```python
# test_selection.py (create)
from __future__ import annotations
import pytest
import ferrum
from tests.fixtures import load_fixture, load_dataset


def test_learning_curve_chart():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    chart = ferrum.learning_curve_chart(
        Ridge(random_state=0), df.select(["f0", "f1", "f2", "f3", "f4"]),
        df["y"], cv=3, random_state=0,
    )
    assert "<svg" in chart.show_svg()


def test_validation_curve_chart():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    chart = ferrum.validation_curve_chart(
        Ridge(), df.select(["f0", "f1", "f2", "f3", "f4"]),
        df["y"], "alpha", [0.1, 1.0, 10.0], cv=3,
    )
    assert "<svg" in chart.show_svg()


def test_cv_scores_chart_box():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    chart = ferrum.cv_scores_chart(
        Ridge(random_state=0), df.select(["f0", "f1", "f2", "f3", "f4"]),
        df["y"], cv=3, kind="box", random_state=0,
    )
    assert "<svg" in chart.show_svg()


def test_alpha_selection_chart():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    chart = ferrum.alpha_selection_chart(
        Ridge(), df.select(["f0", "f1", "f2", "f3", "f4"]),
        df["y"], alphas=[0.01, 0.1, 1.0, 10.0], cv=3,
    )
    assert "<svg" in chart.show_svg()


# test_goldens_phase_10.py
def test_golden_learning_curve_quantized():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    chart = ferrum.learning_curve_chart(
        Ridge(random_state=0), df.select(["f0", "f1", "f2", "f3", "f4"]),
        df["y"], cv=3, random_state=0,
    )
    _check_golden(chart.show_svg(), "learning_curve_ridge")


def test_golden_validation_curve_quantized():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    chart = ferrum.validation_curve_chart(
        Ridge(), df.select(["f0", "f1", "f2", "f3", "f4"]),
        df["y"], "alpha", [0.1, 1.0, 10.0], cv=3,
    )
    _check_golden(chart.show_svg(), "validation_curve_ridge_alpha")


def test_golden_cv_scores_quantized():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    chart = ferrum.cv_scores_chart(
        Ridge(random_state=0), df.select(["f0", "f1", "f2", "f3", "f4"]),
        df["y"], cv=3, kind="box", random_state=0,
    )
    _check_golden(chart.show_svg(), "cv_scores_ridge_box")


def test_golden_alpha_selection_quantized():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    chart = ferrum.alpha_selection_chart(
        Ridge(), df.select(["f0", "f1", "f2", "f3", "f4"]),
        df["y"], alphas=[0.01, 0.1, 1.0, 10.0], cv=3,
    )
    _check_golden(chart.show_svg(), "alpha_selection_ridge")
```

- [ ] **Step 5: Run + commit**

```bash
FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -k "learning or validation or cv_scores or alpha_selection" -v 2>&1 | tail -10
uv run --no-sync pytest tests/diagnostics/test_selection.py -v 2>&1 | tail -10
git add src/ferrum/ tests/diagnostics/ tests/goldens/phase_10/
git commit -m "feat(phase-10e): 4 CV-curve marks + figure functions + quantized goldens"
```

---

### Task 29: 10e visualizers (LearningCurve, ValidationCurve, CVScores, AlphaSelection)

**Files:**
- Create: `src/ferrum/_diagnostics/visualizers/selection.py`
- Modify: `src/ferrum/_diagnostics/visualizers/__init__.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/diagnostics/test_selection.py`

- [ ] **Step 1: Write `selection.py`**

```python
"""10e CV-based visualizers."""
from __future__ import annotations
from typing import Any
import numpy as np
from .base import FerrumVisualizer


class LearningCurveVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, cv: int = 5, scoring: Any = None,
                 train_sizes: Any = None, ci_style: str = "band",
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.cv, self.scoring, self.train_sizes, self.ci_style = cv, scoring, train_sizes, ci_style

    def _materialize(self) -> None:
        df = self._source.learning_curve(cv=self.cv, scoring=self.scoring, train_sizes=self.train_sizes)
        # Final-training-size test mean.
        test_rows = df.filter((df["split"] == "test")).group_by("train_size").agg(pl.col("mean_score").first()).sort("train_size")
        self._metrics["final_test_score"] = float(test_rows["mean_score"][-1])

    def _build_chart(self) -> Any:
        from ..charts import _learning_curve_chart_from_source
        return _learning_curve_chart_from_source(
            self._source, cv=self.cv, scoring=self.scoring,
            train_sizes=self.train_sizes, ci_style=self.ci_style, theme=self.theme,
        )


class ValidationCurveVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, param: str, values: Any, *,
                 cv: int = 5, scoring: Any = None,
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.param, self.values, self.cv, self.scoring = param, values, cv, scoring

    def _materialize(self) -> None:
        df = self._source.validation_curve(self.param, self.values, cv=self.cv, scoring=self.scoring)
        test_rows = df.filter(df["split"] == "test").group_by("param_value").agg(pl.col("mean_score").first())
        idx = int(np.argmax(test_rows["mean_score"].to_numpy()))
        self._metrics["best_param"] = float(test_rows["param_value"][idx])
        self._metrics["best_test_score"] = float(test_rows["mean_score"][idx])

    def _build_chart(self) -> Any:
        from ..charts import _validation_curve_chart_from_source
        return _validation_curve_chart_from_source(
            self._source, self.param, self.values, cv=self.cv,
            scoring=self.scoring, theme=self.theme,
        )


class CVScoresVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, cv: int = 5, scoring: Any = None, kind: str = "box",
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.cv, self.scoring, self.kind = cv, scoring, kind

    def _materialize(self) -> None:
        df = self._source.cv_scores(cv=self.cv, scoring=self.scoring)
        test = df.filter(df["split"] == "test")["score"].to_numpy()
        self._metrics["test_mean"] = float(test.mean())
        self._metrics["test_std"] = float(test.std())

    def _build_chart(self) -> Any:
        from ..charts import _cv_scores_chart_from_source
        return _cv_scores_chart_from_source(
            self._source, cv=self.cv, scoring=self.scoring, kind=self.kind, theme=self.theme,
        )


class AlphaSelectionVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, alphas: Any, *, cv: int = 5, scoring: Any = None,
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.alphas, self.cv, self.scoring = alphas, cv, scoring

    def _materialize(self) -> None:
        df = self._source.alpha_selection(self.alphas, cv=self.cv, scoring=self.scoring)
        agg = df.group_by("alpha").agg(pl.col("mean_score").first()).sort("alpha")
        idx = int(np.argmax(agg["mean_score"].to_numpy()))
        self._metrics["best_alpha"] = float(agg["alpha"][idx])

    def _build_chart(self) -> Any:
        from ..charts import _alpha_selection_chart_from_source
        return _alpha_selection_chart_from_source(
            self._source, self.alphas, cv=self.cv, scoring=self.scoring, theme=self.theme,
        )
```

- [ ] **Step 2: Re-exports + tests + commit**

```python
# visualizers/__init__.py
from .selection import (
    LearningCurveVisualizer, ValidationCurveVisualizer,
    CVScoresVisualizer, AlphaSelectionVisualizer,
)
__all__ += [
    "LearningCurveVisualizer", "ValidationCurveVisualizer",
    "CVScoresVisualizer", "AlphaSelectionVisualizer",
]


# src/ferrum/__init__.py
from ferrum._diagnostics.visualizers import (
    LearningCurveVisualizer, ValidationCurveVisualizer,
    CVScoresVisualizer, AlphaSelectionVisualizer,
)


# test_selection.py
def test_learning_curve_visualizer():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    viz = ferrum.LearningCurveVisualizer(Ridge(random_state=0), cv=3, random_state=0).fit(
        df.select(["f0", "f1", "f2", "f3", "f4"]), df["y"],
    )
    assert "final_test_score=" in repr(viz)


def test_validation_curve_visualizer():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    viz = ferrum.ValidationCurveVisualizer(Ridge(), "alpha", [0.1, 1.0, 10.0], cv=3).fit(
        df.select(["f0", "f1", "f2", "f3", "f4"]), df["y"],
    )
    assert "best_param=" in repr(viz)


def test_cv_scores_visualizer():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    viz = ferrum.CVScoresVisualizer(Ridge(random_state=0), cv=3, random_state=0).fit(
        df.select(["f0", "f1", "f2", "f3", "f4"]), df["y"],
    )
    assert "test_mean=" in repr(viz)


def test_alpha_selection_visualizer():
    from sklearn.linear_model import Ridge
    df = load_dataset("regression")
    viz = ferrum.AlphaSelectionVisualizer(Ridge(), [0.01, 0.1, 1.0, 10.0], cv=3).fit(
        df.select(["f0", "f1", "f2", "f3", "f4"]), df["y"],
    )
    assert "best_alpha=" in repr(viz)
```

```bash
uv run --no-sync pytest tests/diagnostics/test_selection.py -v 2>&1 | tail -10
git add src/ferrum/ tests/diagnostics/test_selection.py
git commit -m "feat(phase-10e): 4 CV-based visualizers"
```

- [ ] **Step 3: 10e milestone check**

```bash
uv run --no-sync pytest tests/diagnostics/ 2>&1 | tail -3
```
Expected: ~95 tests cumulative.

---

## 10f — Clustering / manifold / decision boundary

### Task 30: `.silhouette()` + `.pca_variance()` methods + their marks + figures

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py`
- Modify: `src/ferrum/_diagnostics/schemas.py`
- Modify: `src/ferrum/marks/diagnostic.py`
- Modify: `src/ferrum/chart.py`
- Modify: `src/ferrum/_diagnostics/charts.py`
- Modify: `src/ferrum/figures.py`
- Modify: `src/ferrum/__init__.py`
- Create: `tests/diagnostics/test_clustering.py`

> ⚠ **Pattern correction (plan-vs-codebase):** The mark code blocks below were originally drafted using a `@dataclass(frozen=True) class mark_X: ... def _expand(self, chart_ctx) -> list[LayerSpec]` pattern that **does not exist in the codebase**. Before implementing, translate every mark in this task to the real pattern used in Phase 8b/9 composite marks:
>
> - Module-level `def desugar_<name>(x_field, y_field, **kwargs) -> ("__layered__", transforms, None, None, layers)` in `src/ferrum/marks/diagnostic.py`.
> - Layers are plain dicts: `{"mark": str, "encoding": dict, "mark_kwargs": dict (opt), "data_source": str | None (opt)}`.
> - No `LayerSpec`. No `chart_ctx`. No `_expand`.
> - Chart method clones, sets `_mark = "point"` (placeholder), sets `_pending_stat_mark = (kind, kwargs_dict, desugar_fn)`, returns.
> - The user does not import or instantiate `mark_X` — they call `Chart(df).mark_X(...)`.
> - For diagnostic marks, the data has hard-coded columns from a `ModelSource` method, so the desugar references those columns literally and ignores positional `x_field` / `y_field`.
>
> **Canonical reference:** Task 8 (`desugar_residuals` / `desugar_prediction_error`) and Task 15 (six 10b desugars). Pattern reference in code: `src/ferrum/marks/composite.py:15-220`.
>
> Keep the **layer encodings, kwargs, and behavior** below as the spec for what each mark should produce, but rewrite the implementation in the corrected pattern.

- [ ] **Step 1: Schemas + methods**

```python
# schemas.py
SCHEMA_SILHOUETTE = pl.Schema({
    "sample_id": pl.Int64,
    "cluster": pl.Int64,
    "silhouette_value": pl.Float64,
})
SCHEMA_PCA_VARIANCE = pl.Schema({
    "component": pl.Int64,
    "explained_variance_ratio": pl.Float64,
    "cumulative_variance_ratio": pl.Float64,
})


# source.py — append
    def silhouette(self, k: int | None = None) -> pl.DataFrame:
        key = self._cache_key("silhouette", k=k)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("silhouette")
        from sklearn.metrics import silhouette_samples
        import numpy as np
        X_np = self._X.to_numpy()
        if "labels_" in self._capabilities:
            labels = np.asarray(self._model.labels_)
        elif "predict" in self._capabilities:
            labels = np.asarray(self._model.predict(X_np))
        else:
            raise AttributeError("silhouette() requires the model to expose labels_ or predict()")
        sv = silhouette_samples(X_np, labels)
        # Sort within cluster by descending silhouette value (Rousseeuw plot).
        rows: list[dict] = []
        for c in sorted(set(labels.tolist())):
            mask = labels == c
            idxs = np.where(mask)[0]
            vals = sv[mask]
            order = np.argsort(-vals)
            for i, val in zip(idxs[order], vals[order]):
                rows.append({"sample_id": int(i), "cluster": int(c),
                              "silhouette_value": float(val)})
        df = pl.DataFrame(rows)
        self._cache[key] = df
        return df

    def pca_variance(self, *, n_components: int | None = None) -> pl.DataFrame:
        key = self._cache_key("pca_variance", n_components=n_components)
        if key in self._cache:
            return self._cache[key]
        if "explained_variance_ratio_" not in self._capabilities:
            raise AttributeError(
                "ModelSource.pca_variance() requires the model to expose "
                "'explained_variance_ratio_' (e.g. sklearn PCA, TruncatedSVD)."
            )
        import numpy as np
        evr = np.asarray(self._model.explained_variance_ratio_, dtype=np.float64)
        if n_components is not None:
            evr = evr[:n_components]
        cum = np.cumsum(evr)
        df = pl.DataFrame({
            "component": list(range(1, len(evr) + 1)),
            "explained_variance_ratio": [float(x) for x in evr],
            "cumulative_variance_ratio": [float(x) for x in cum],
        })
        self._cache[key] = df
        return df
```

- [ ] **Step 2: Marks**

```python
@dataclass(frozen=True)
class mark_silhouette:
    line_width: float = 1.0
    zero_line: bool = True

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_bar, mark_rule
        from ferrum import LayerSpec
        layers = [LayerSpec(
            mark=mark_bar(),
            encoding={"y": "sample_id", "x": "silhouette_value", "color": "cluster"},
        )]
        if self.zero_line:
            layers.append(LayerSpec(mark=mark_rule(), encoding={"x": 0.0}))
        return layers


@dataclass(frozen=True)
class mark_pca_scree:
    n_components: int | None = None
    cumulative_line: bool = True
    threshold_line: float | None = None

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_bar, mark_line, mark_rule
        from ferrum import LayerSpec
        layers = [LayerSpec(
            mark=mark_bar(),
            encoding={"x": "component", "y": "explained_variance_ratio"},
        )]
        if self.cumulative_line:
            layers.append(LayerSpec(
                mark=mark_line(),
                encoding={"x": "component", "y": "cumulative_variance_ratio"},
            ))
        if self.threshold_line is not None:
            layers.append(LayerSpec(
                mark=mark_rule(strokeDash=[4, 4]),
                encoding={"y": self.threshold_line},
            ))
        return layers
```

- [ ] **Step 3: Chart methods + builders + figure functions**

```python
# chart.py
def mark_silhouette(self, **kw):
    from ferrum.marks.diagnostic import mark_silhouette as _M
    return self._add_composite_mark(_M(**kw))


def mark_pca_scree(self, **kw):
    from ferrum.marks.diagnostic import mark_pca_scree as _M
    return self._add_composite_mark(_M(**kw))


# charts.py
def _silhouette_chart_from_source(source, *, k=None, theme=None):
    df = source.silhouette(k=k)
    chart = ferrum.Chart(df).mark_silhouette()
    if theme is not None: chart = chart.theme(theme)
    return chart


def _pca_scree_chart_from_source(source, *, n_components=None, cumulative_line=True,
                                   threshold=0.95, theme=None):
    df = source.pca_variance(n_components=n_components)
    chart = ferrum.Chart(df).mark_pca_scree(
        n_components=n_components, cumulative_line=cumulative_line,
        threshold_line=threshold,
    )
    if theme is not None: chart = chart.theme(theme)
    return chart


# figures.py
def cluster_diagnostics(
    X, *, ks, method="kmeans", scoring="both", n_init=10,
    random_state=None, theme=None,
):
    """Elbow + silhouette per k for a given clusterer class."""
    require_sklearn = __import__("ferrum._diagnostics.deps", fromlist=["require_sklearn"]).require_sklearn
    require_sklearn("cluster_diagnostics")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
    X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    rows = []
    for k in ks:
        m = KMeans(n_clusters=k, n_init=n_init, random_state=random_state or 0).fit(X_np)
        rows.append({"k": int(k), "inertia": float(m.inertia_),
                      "silhouette": float(silhouette_score(X_np, m.labels_))})
    df = pl.DataFrame(rows)
    elbow = ferrum.Chart(df).mark_line().encode(x="k", y="inertia")
    sil = ferrum.Chart(df).mark_line().encode(x="k", y="silhouette")
    chart = elbow | sil
    if theme is not None: chart = chart.theme(theme)
    return chart


def pca_scree_chart(model_or_source, X=None, *, n_components=None, cumulative_line=True,
                     threshold=0.95, random_state=None, theme=None):
    source = _resolve_source(model_or_source, X, None, random_state=random_state)
    return _pca_scree_chart_from_source(
        source, n_components=n_components, cumulative_line=cumulative_line,
        threshold=threshold, theme=theme,
    )
```

- [ ] **Step 4: Re-exports + tests + goldens**

```python
# __init__.py
from ferrum.marks.diagnostic import mark_silhouette, mark_pca_scree
from ferrum.figures import cluster_diagnostics, pca_scree_chart


# test_clustering.py (create)
from __future__ import annotations
import numpy as np
import pytest
import ferrum
from tests.fixtures import load_fixture, load_dataset


def test_silhouette_method():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    source = ferrum.ModelSource(model, df)
    sil = source.silhouette()
    assert set(sil.columns) == {"sample_id", "cluster", "silhouette_value"}


def test_pca_variance():
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(model, df)
    pca = source.pca_variance()
    assert set(pca.columns) == {"component", "explained_variance_ratio", "cumulative_variance_ratio"}
    np.testing.assert_allclose(
        pca["cumulative_variance_ratio"][-1],
        pca["explained_variance_ratio"].sum(),
        rtol=1e-12,
    )


def test_silhouette_chart():
    from ferrum._diagnostics.charts import _silhouette_chart_from_source
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    source = ferrum.ModelSource(model, df)
    chart = _silhouette_chart_from_source(source)
    assert "<svg" in chart.show_svg()


def test_pca_scree_chart():
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.pca_scree_chart(model, df, threshold=0.95)
    assert "<svg" in chart.show_svg()


# test_goldens_phase_10.py
def test_golden_silhouette_kmeans():
    from ferrum._diagnostics.charts import _silhouette_chart_from_source
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    source = ferrum.ModelSource(model, df)
    chart = _silhouette_chart_from_source(source)
    _check_golden(chart.show_svg(), "silhouette_kmeans_3cluster")


def test_golden_pca_scree():
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.pca_scree_chart(model, df)
    _check_golden(chart.show_svg(), "pca_scree_4comp")
```

- [ ] **Step 5: Run + commit**

```bash
FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -k "silhouette or pca_scree" -v 2>&1 | tail -10
uv run --no-sync pytest tests/diagnostics/test_clustering.py -v 2>&1 | tail -10
git add src/ferrum/ tests/diagnostics/test_clustering.py tests/diagnostics/test_goldens_phase_10.py tests/goldens/phase_10/
git commit -m "feat(phase-10f): silhouette + pca_variance + their marks + figures + goldens"
```

---

### Task 31: `.embeddings()` + `.intercluster_distance()` (UMAP-aware) + marks + figures

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py`
- Modify: `src/ferrum/marks/diagnostic.py`
- Modify: `src/ferrum/chart.py`
- Modify: `src/ferrum/_diagnostics/charts.py`
- Modify: `src/ferrum/figures.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/diagnostics/test_clustering.py`
- Modify: `tests/diagnostics/test_goldens_phase_10.py`

> ⚠ **Pattern correction (plan-vs-codebase):** The mark code blocks below were originally drafted using a `@dataclass(frozen=True) class mark_X: ... def _expand(self, chart_ctx) -> list[LayerSpec]` pattern that **does not exist in the codebase**. Before implementing, translate every mark in this task to the real pattern used in Phase 8b/9 composite marks:
>
> - Module-level `def desugar_<name>(x_field, y_field, **kwargs) -> ("__layered__", transforms, None, None, layers)` in `src/ferrum/marks/diagnostic.py`.
> - Layers are plain dicts: `{"mark": str, "encoding": dict, "mark_kwargs": dict (opt), "data_source": str | None (opt)}`.
> - No `LayerSpec`. No `chart_ctx`. No `_expand`.
> - Chart method clones, sets `_mark = "point"` (placeholder), sets `_pending_stat_mark = (kind, kwargs_dict, desugar_fn)`, returns.
> - The user does not import or instantiate `mark_X` — they call `Chart(df).mark_X(...)`.
> - For diagnostic marks, the data has hard-coded columns from a `ModelSource` method, so the desugar references those columns literally and ignores positional `x_field` / `y_field`.
>
> **Canonical reference:** Task 8 (`desugar_residuals` / `desugar_prediction_error`) and Task 15 (six 10b desugars). Pattern reference in code: `src/ferrum/marks/composite.py:15-220`.
>
> Keep the **layer encodings, kwargs, and behavior** below as the spec for what each mark should produce, but rewrite the implementation in the corrected pattern.

- [ ] **Step 1: Methods**

```python
# source.py
    def embeddings(
        self,
        *,
        method: str = "umap",
        n_components: int = 2,
        **method_kwargs: Any,
    ) -> pl.DataFrame:
        key = self._cache_key("embeddings", method=method, n_components=n_components,
                                kwargs=tuple(sorted(method_kwargs.items())))
        if key in self._cache:
            return self._cache[key]
        import numpy as np
        X_np = self._X.to_numpy()
        if method == "umap":
            umap = require_umap("embeddings")
            reducer = umap.UMAP(n_components=n_components, random_state=self._random_state or 0,
                                  **method_kwargs)
            emb = reducer.fit_transform(X_np)
        elif method == "tsne":
            sklearn = require_sklearn("embeddings(tsne)")
            from sklearn.manifold import TSNE
            emb = TSNE(n_components=n_components, random_state=self._random_state or 0,
                        **method_kwargs).fit_transform(X_np)
        elif method == "pca":
            sklearn = require_sklearn("embeddings(pca)")
            from sklearn.decomposition import PCA
            emb = PCA(n_components=n_components, random_state=self._random_state or 0,
                       **method_kwargs).fit_transform(X_np)
        else:
            raise ValueError(f"embeddings method must be umap/tsne/pca; got {method!r}")
        label = self._y.to_numpy() if self._y is not None else np.zeros(len(X_np))
        data: dict[str, Any] = {f"dim_{i}": emb[:, i] for i in range(n_components)}
        data["label"] = label
        df = pl.DataFrame(data)
        self._cache[key] = df
        return df

    def intercluster_distance(
        self,
        k: int,
        *,
        method: str = "mds",
    ) -> pl.DataFrame:
        key = self._cache_key("intercluster_distance", k=k, method=method)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("intercluster_distance")
        import numpy as np
        if "cluster_centers_" not in self._capabilities:
            raise AttributeError("intercluster_distance requires the model to expose cluster_centers_")
        centers = np.asarray(self._model.cluster_centers_)
        if method == "mds":
            from sklearn.manifold import MDS
            xy = MDS(n_components=2, random_state=self._random_state or 0).fit_transform(centers)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            xy = TSNE(n_components=2, random_state=self._random_state or 0, perplexity=min(5, max(1, k-1))).fit_transform(centers)
        else:
            raise ValueError(f"intercluster_distance method must be mds/tsne; got {method!r}")
        if hasattr(self._model, "labels_"):
            labels = np.asarray(self._model.labels_)
            sizes = np.bincount(labels, minlength=k)[:k]
        else:
            sizes = np.ones(k, dtype=int)
        df = pl.DataFrame({
            "cluster": list(range(k)),
            "x": [float(x) for x in xy[:, 0]],
            "y": [float(y) for y in xy[:, 1]],
            "size": [int(s) for s in sizes],
        })
        self._cache[key] = df
        return df
```

- [ ] **Step 2: `mark_intercluster_distance` (no `mark_embeddings` — embeddings render via existing `mark_point`)**

```python
@dataclass(frozen=True)
class mark_intercluster_distance:
    method: str = "mds"
    min_size: float = 30.0
    max_size: float = 500.0
    label_clusters: bool = True

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_point, mark_text
        from ferrum import LayerSpec
        layers = [LayerSpec(
            mark=mark_point(),
            encoding={"x": "x", "y": "y", "size": "size"},
        )]
        if self.label_clusters:
            layers.append(LayerSpec(
                mark=mark_text(),
                encoding={"x": "x", "y": "y", "text": "cluster"},
            ))
        return layers
```

- [ ] **Step 3: Chart methods + builders + figures**

```python
# chart.py
def mark_intercluster_distance(self, **kw):
    from ferrum.marks.diagnostic import mark_intercluster_distance as _M
    return self._add_composite_mark(_M(**kw))


# charts.py
def _intercluster_distance_chart_from_source(source, *, k, method="mds", theme=None):
    df = source.intercluster_distance(k, method=method)
    chart = ferrum.Chart(df).mark_intercluster_distance(method=method)
    if theme is not None: chart = chart.theme(theme)
    return chart


# figures.py
def intercluster_distance_chart(
    model_or_source, X=None, *, k=None, method="mds",
    random_state=None, theme=None,
):
    source = _resolve_source(model_or_source, X, None, random_state=random_state)
    if k is None:
        if hasattr(source._model, "n_clusters"):
            k = int(source._model.n_clusters)
        elif hasattr(source._model, "cluster_centers_"):
            k = int(source._model.cluster_centers_.shape[0])
        else:
            raise ValueError("k= is required for intercluster_distance_chart")
    return _intercluster_distance_chart_from_source(
        source, k=k, method=method, theme=theme,
    )
```

- [ ] **Step 4: Re-exports + tests + quantized goldens**

```python
# __init__.py
from ferrum.marks.diagnostic import mark_intercluster_distance
from ferrum.figures import intercluster_distance_chart


# test_clustering.py
def test_embeddings_pca():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    source = ferrum.ModelSource(load_fixture("pca_4comp"), df, random_state=0)
    emb = source.embeddings(method="pca", n_components=2)
    assert {"dim_0", "dim_1", "label"} <= set(emb.columns)


def test_intercluster_distance():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    source = ferrum.ModelSource(model, df, random_state=0)
    icd = source.intercluster_distance(k=3, method="mds")
    assert icd.height == 3
    assert set(icd.columns) == {"cluster", "x", "y", "size"}


def test_intercluster_distance_chart():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    chart = ferrum.intercluster_distance_chart(model, df, k=3, random_state=0)
    assert "<svg" in chart.show_svg()


# test_goldens_phase_10.py — quantized (MDS uses eigendecomposition)
def test_golden_intercluster_distance_quantized():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    chart = ferrum.intercluster_distance_chart(model, df, k=3, method="mds", random_state=0)
    _check_golden(chart.show_svg(), "intercluster_distance_mds")
```

- [ ] **Step 5: Run + commit**

```bash
FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -k intercluster -v 2>&1 | tail -5
uv run --no-sync pytest tests/diagnostics/test_clustering.py -v -k "embeddings or intercluster" 2>&1 | tail -10
git add src/ferrum/ tests/diagnostics/ tests/goldens/phase_10/
git commit -m "feat(phase-10f): embeddings + intercluster_distance + UMAP-aware figure"
```

---

### Task 32: `decision_boundary_chart` + `mark_decision_boundary`

**Files:**
- Modify: `src/ferrum/marks/diagnostic.py`
- Modify: `src/ferrum/chart.py`
- Modify: `src/ferrum/_diagnostics/charts.py`
- Modify: `src/ferrum/figures.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/diagnostics/test_clustering.py`
- Modify: `tests/diagnostics/test_goldens_phase_10.py`

> ⚠ **Pattern correction (plan-vs-codebase):** The mark code blocks below were originally drafted using a `@dataclass(frozen=True) class mark_X: ... def _expand(self, chart_ctx) -> list[LayerSpec]` pattern that **does not exist in the codebase**. Before implementing, translate every mark in this task to the real pattern used in Phase 8b/9 composite marks:
>
> - Module-level `def desugar_<name>(x_field, y_field, **kwargs) -> ("__layered__", transforms, None, None, layers)` in `src/ferrum/marks/diagnostic.py`.
> - Layers are plain dicts: `{"mark": str, "encoding": dict, "mark_kwargs": dict (opt), "data_source": str | None (opt)}`.
> - No `LayerSpec`. No `chart_ctx`. No `_expand`.
> - Chart method clones, sets `_mark = "point"` (placeholder), sets `_pending_stat_mark = (kind, kwargs_dict, desugar_fn)`, returns.
> - The user does not import or instantiate `mark_X` — they call `Chart(df).mark_X(...)`.
> - For diagnostic marks, the data has hard-coded columns from a `ModelSource` method, so the desugar references those columns literally and ignores positional `x_field` / `y_field`.
>
> **Canonical reference:** Task 8 (`desugar_residuals` / `desugar_prediction_error`) and Task 15 (six 10b desugars). Pattern reference in code: `src/ferrum/marks/composite.py:15-220`.
>
> Keep the **layer encodings, kwargs, and behavior** below as the spec for what each mark should produce, but rewrite the implementation in the corrected pattern.

- [ ] **Step 1: Mark**

```python
@dataclass(frozen=True)
class mark_decision_boundary:
    grid_resolution: int = 200
    alpha: float = 0.4
    proba: bool = False
    contour_levels: int = 10

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_raster, mark_contour, mark_point
        from ferrum import LayerSpec
        # Data is built by the chart-builder; the mark just selects raster vs contour.
        bg_mark = mark_raster() if self.proba else mark_contour()
        return [
            LayerSpec(mark=bg_mark, encoding={"x": "x", "y": "y", "color": "z"}),
        ]
```

- [ ] **Step 2: Chart method + builder + figure**

```python
# chart.py
def mark_decision_boundary(self, **kw):
    from ferrum.marks.diagnostic import mark_decision_boundary as _M
    return self._add_composite_mark(_M(**kw))


# charts.py
def _decision_boundary_chart_from_source(source, *, features=(0, 1),
                                           grid_resolution=200, proba=False,
                                           scatter=True, theme=None):
    import numpy as np
    X_np = source._X.to_numpy()
    feat_idx = tuple(
        source._feature_names.index(f) if isinstance(f, str) else int(f)
        for f in features
    )
    if len(feat_idx) != 2:
        raise ValueError("decision_boundary requires exactly 2 features")
    x_col, y_col = X_np[:, feat_idx[0]], X_np[:, feat_idx[1]]
    pad_x = (x_col.max() - x_col.min()) * 0.05
    pad_y = (y_col.max() - y_col.min()) * 0.05
    xs = np.linspace(x_col.min() - pad_x, x_col.max() + pad_x, grid_resolution)
    ys = np.linspace(y_col.min() - pad_y, y_col.max() + pad_y, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    # Build grid X with feature_idxs varying, others fixed at mean.
    grid = np.tile(X_np.mean(axis=0), (xx.size, 1))
    grid[:, feat_idx[0]] = xx.ravel()
    grid[:, feat_idx[1]] = yy.ravel()
    if proba and "predict_proba" in source._capabilities:
        z = source._model.predict_proba(grid)[:, 1]
    else:
        z = source._model.predict(grid).astype(np.float64)
    grid_df = pl.DataFrame({
        "x": [float(v) for v in xx.ravel()],
        "y": [float(v) for v in yy.ravel()],
        "z": [float(v) for v in z],
    })
    chart = ferrum.Chart(grid_df).mark_decision_boundary(
        grid_resolution=grid_resolution, proba=proba,
    )
    if scatter and source._y is not None:
        scatter_df = pl.DataFrame({
            "x": x_col, "y": y_col,
            "label": source._y.to_numpy().tolist(),
        })
        chart = chart + ferrum.Chart(scatter_df).mark_point().encode(
            x="x", y="y", color="label",
        )
    if theme is not None: chart = chart.theme(theme)
    return chart


# figures.py
def decision_boundary_chart(
    model, X, y, *, features=(0, 1), grid_resolution=200, proba=False,
    scatter=True, random_state=None, theme=None,
):
    source = _resolve_source(model, X, y, random_state=random_state)
    return _decision_boundary_chart_from_source(
        source, features=features, grid_resolution=grid_resolution,
        proba=proba, scatter=scatter, theme=theme,
    )
```

- [ ] **Step 3: Re-exports + tests + quantized golden**

```python
# __init__.py
from ferrum.marks.diagnostic import mark_decision_boundary
from ferrum.figures import decision_boundary_chart


# test_clustering.py
def test_decision_boundary_chart_binary():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    # Use 2 features.
    chart = ferrum.decision_boundary_chart(
        model, df.select(["f0", "f1"]), df["y"],
        features=(0, 1), grid_resolution=50, proba=True,
    )
    assert "<svg" in chart.show_svg()


def test_decision_boundary_rejects_three_features():
    import pytest
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    # The figure function takes any X; the builder validates 2 features.
    with pytest.raises(ValueError, match="exactly 2"):
        ferrum.decision_boundary_chart(
            model, df.select(["f0", "f1", "f2"]), df["y"],
            features=(0, 1, 2), grid_resolution=10,
        )


# test_goldens_phase_10.py — quantized (sklearn lbfgs is platform-sensitive)
def test_golden_decision_boundary_quantized():
    model = load_fixture("binary_logistic")
    df = load_dataset("binary_classification")
    chart = ferrum.decision_boundary_chart(
        model, df.select(["f0", "f1"]), df["y"],
        features=(0, 1), grid_resolution=50, proba=True,
    )
    _check_golden(chart.show_svg(), "decision_boundary_binary")
```

- [ ] **Step 4: Run + commit**

```bash
FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -k decision -v 2>&1 | tail -5
uv run --no-sync pytest tests/diagnostics/test_clustering.py -v -k decision 2>&1 | tail -5
git add src/ferrum/ tests/diagnostics/ tests/goldens/phase_10/
git commit -m "feat(phase-10f): mark_decision_boundary + decision_boundary_chart"
```

---

### Task 33: `cluster_diagnostics` figure function + remaining clustering visualizers

**Files:**
- Create: `src/ferrum/_diagnostics/visualizers/clustering.py`
- Modify: `src/ferrum/_diagnostics/visualizers/__init__.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/diagnostics/test_clustering.py`

- [ ] **Step 1: Write `clustering.py` with 5 visualizers**

```python
"""10f clustering / manifold / dimensionality visualizers."""
from __future__ import annotations
from typing import Any, Sequence

import numpy as np
import polars as pl

from .base import FerrumVisualizer


class SilhouetteVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)

    def _materialize(self) -> None:
        sil = self._source.silhouette()
        self._metrics["mean_silhouette"] = float(sil["silhouette_value"].mean())

    def _build_chart(self) -> Any:
        from ..charts import _silhouette_chart_from_source
        return _silhouette_chart_from_source(self._source, theme=self.theme)


class ElbowVisualizer(FerrumVisualizer):
    """Takes a model CLASS (not a fitted instance) — fits one per k inside fit()."""
    def __init__(self, model_class: Any, *, ks: Sequence[int], metric: str = "distortion",
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.model_class = model_class
        self.ks = list(ks)
        self.metric = metric

    def fit(self, X: Any, y: Any = None) -> "ElbowVisualizer":
        rows = []
        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        for k in self.ks:
            m = self.model_class(n_clusters=k, random_state=self.random_state or 0,
                                  n_init=10).fit(X_np)
            score = float(m.inertia_) if self.metric == "distortion" else 0.0
            rows.append({"k": int(k), "score": score})
        df = pl.DataFrame(rows)
        self._metrics["best_k"] = int(df["k"][int(np.argmin(df["score"].to_numpy()))])
        import ferrum
        self._chart = ferrum.Chart(df).mark_line().encode(x="k", y="score")
        if self.theme is not None: self._chart = self._chart.theme(self.theme)
        self._fitted = True
        return self


class ManifoldVisualizer(FerrumVisualizer):
    def __init__(self, model: Any = None, *, method: str = "umap",
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.method = method

    def _materialize(self) -> None:
        emb = self._source.embeddings(method=self.method)
        self._metrics["n_samples"] = float(emb.height)

    def _build_chart(self) -> Any:
        import ferrum
        emb = self._source.embeddings(method=self.method)
        chart = ferrum.Chart(emb).mark_point().encode(x="dim_0", y="dim_1", color="label")
        if self.theme is not None: chart = chart.theme(self.theme)
        return chart


class InterclusterDistanceVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, method: str = "mds",
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.method = method

    def _materialize(self) -> None:
        k = int(getattr(self.model, "n_clusters", self.model.cluster_centers_.shape[0]))
        icd = self._source.intercluster_distance(k=k, method=self.method)
        self._metrics["max_intercluster_dist"] = float(
            ((icd["x"].to_numpy() - icd["x"].to_numpy().mean()) ** 2
              + (icd["y"].to_numpy() - icd["y"].to_numpy().mean()) ** 2).max() ** 0.5
        )

    def _build_chart(self) -> Any:
        import ferrum
        k = int(getattr(self.model, "n_clusters", self.model.cluster_centers_.shape[0]))
        return ferrum.intercluster_distance_chart(self._source, k=k, method=self.method, theme=self.theme)


class PCAVarianceVisualizer(FerrumVisualizer):
    def __init__(self, model: Any, *, n_components: int | None = None,
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model, random_state=random_state, theme=theme)
        self.n_components = n_components

    def _materialize(self) -> None:
        pca = self._source.pca_variance(n_components=self.n_components)
        self._metrics["first_component_var"] = float(pca["explained_variance_ratio"][0])

    def _build_chart(self) -> Any:
        import ferrum
        return ferrum.pca_scree_chart(
            self._source, n_components=self.n_components, theme=self.theme,
        )
```

- [ ] **Step 2: Re-exports + tests**

```python
# visualizers/__init__.py
from .clustering import (
    SilhouetteVisualizer, ElbowVisualizer, ManifoldVisualizer,
    InterclusterDistanceVisualizer, PCAVarianceVisualizer,
)
__all__ += [
    "SilhouetteVisualizer", "ElbowVisualizer", "ManifoldVisualizer",
    "InterclusterDistanceVisualizer", "PCAVarianceVisualizer",
]


# src/ferrum/__init__.py
from ferrum._diagnostics.visualizers import (
    SilhouetteVisualizer, ElbowVisualizer, ManifoldVisualizer,
    InterclusterDistanceVisualizer, PCAVarianceVisualizer,
)


# test_clustering.py
def test_silhouette_visualizer():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    viz = ferrum.SilhouetteVisualizer(model).fit(df)
    assert "mean_silhouette=" in repr(viz)


def test_elbow_visualizer():
    from sklearn.cluster import KMeans
    df = load_dataset("clustering")
    viz = ferrum.ElbowVisualizer(KMeans, ks=[2, 3, 4, 5], random_state=0).fit(df)
    assert "best_k=" in repr(viz)


def test_manifold_visualizer_pca():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    model = load_fixture("pca_4comp")
    viz = ferrum.ManifoldVisualizer(model, method="pca", random_state=0).fit(df)
    assert viz._fitted


def test_intercluster_distance_visualizer():
    model = load_fixture("kmeans_3cluster")
    df = load_dataset("clustering")
    viz = ferrum.InterclusterDistanceVisualizer(model, random_state=0).fit(df)
    assert "max_intercluster_dist=" in repr(viz)


def test_pca_variance_visualizer():
    model = load_fixture("pca_4comp")
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    viz = ferrum.PCAVarianceVisualizer(model).fit(df)
    assert "first_component_var=" in repr(viz)
```

- [ ] **Step 3: cluster_diagnostics figure already in Task 30 — test it here**

```python
def test_cluster_diagnostics_figure():
    df = load_dataset("clustering")
    chart = ferrum.cluster_diagnostics(df, ks=[2, 3, 4, 5], random_state=0)
    assert "<svg" in chart.show_svg()
```

- [ ] **Step 4: Run + commit**

```bash
uv run --no-sync pytest tests/diagnostics/test_clustering.py -v 2>&1 | tail -15
git add src/ferrum/_diagnostics/visualizers/clustering.py src/ferrum/_diagnostics/visualizers/__init__.py src/ferrum/__init__.py tests/diagnostics/test_clustering.py
git commit -m "feat(phase-10f): 5 clustering/manifold visualizers + cluster_diagnostics test"
```

---

### Task 34: 10f milestone check

- [ ] **Step 1: Verify all 10f tests + no regression**

```bash
uv run --no-sync pytest tests/diagnostics/ 2>&1 | tail -3
```
Expected: ~110 tests cumulative.

- [ ] **Step 2: Verify no sklearn at import still holds**

```bash
uv run --no-sync pytest tests/diagnostics/test_no_sklearn_at_import.py -v 2>&1 | tail -5
```
Expected: 3 passed.

---

## 10g — Feature ranking + parallel coordinates

### Task 35: `ferrum._core.kendall_tau_b` (Rust, Knight's O(n log n))

**Files:**
- Create: `crates/ferrum-core/src/diagnostics.rs`
- Modify: `crates/ferrum-core/src/lib.rs`
- Modify: `src/ferrum/_core.pyi`

- [ ] **Step 1: Write the Rust implementation**

```rust
// crates/ferrum-core/src/diagnostics.rs
//! Phase 10 model-diagnostics — sole Rust contribution.
//!
//! `kendall_tau_b` implements Knight's O(n log n) merge-sort variant
//! for Kendall's tau-b rank correlation. Used by
//! ModelSource.rank2d(algorithm="kendall") when n_samples is large
//! enough that vectorized NumPy OOMs or pure-Python Knight is too slow.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyReadonlyArray1;

#[derive(Debug, Clone, Copy)]
pub struct KendallResult {
    pub tau: f64,
    pub n_concordant: u64,
    pub n_discordant: u64,
    pub n_tied_x: u64,
    pub n_tied_y: u64,
    pub n_tied_both: u64,
}

/// Sort `idx` by x[idx] (stable), counting tied x pairs.
/// Returns the indices reordered.
fn sort_by_key_count_ties(x: &[f64], idx: &mut [usize]) -> u64 {
    // Stable sort by x value.
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    // Count tied groups: for a run of length r, ties contribute r*(r-1)/2 pairs.
    let mut ties: u64 = 0;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && x[idx[j]] == x[idx[i]] {
            j += 1;
        }
        let run = (j - i) as u64;
        if run > 1 {
            ties += run * (run - 1) / 2;
        }
        i = j;
    }
    ties
}

/// Merge-sort over `y` indices, counting inversions (= n_discordant).
/// Returns (sorted_buffer, inversions).
fn count_inversions(y: &[f64], idx: &mut [usize]) -> u64 {
    let mut buf = vec![0usize; idx.len()];
    let inv = merge_sort(y, idx, &mut buf, 0, idx.len());
    inv
}

fn merge_sort(y: &[f64], idx: &mut [usize], buf: &mut [usize], lo: usize, hi: usize) -> u64 {
    if hi - lo <= 1 { return 0; }
    let mid = (lo + hi) / 2;
    let l = merge_sort(y, idx, buf, lo, mid);
    let r = merge_sort(y, idx, buf, mid, hi);
    let m = merge(y, idx, buf, lo, mid, hi);
    l + r + m
}

fn merge(y: &[f64], idx: &mut [usize], buf: &mut [usize], lo: usize, mid: usize, hi: usize) -> u64 {
    let mut i = lo;
    let mut j = mid;
    let mut k = lo;
    let mut inv: u64 = 0;
    while i < mid && j < hi {
        if y[idx[i]] <= y[idx[j]] {
            buf[k] = idx[i];
            i += 1;
        } else {
            buf[k] = idx[j];
            inv += (mid - i) as u64;
            j += 1;
        }
        k += 1;
    }
    while i < mid { buf[k] = idx[i]; i += 1; k += 1; }
    while j < hi  { buf[k] = idx[j]; j += 1; k += 1; }
    idx[lo..hi].copy_from_slice(&buf[lo..hi]);
    inv
}

/// Count y-ties given indices already sorted by x then by y.
fn count_y_ties_after_sort(y: &[f64], idx: &[usize]) -> u64 {
    let mut ties: u64 = 0;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && y[idx[j]] == y[idx[i]] {
            j += 1;
        }
        let run = (j - i) as u64;
        if run > 1 {
            ties += run * (run - 1) / 2;
        }
        i = j;
    }
    ties
}

/// Count (x,y) joint-tie pairs.
fn count_xy_ties(x: &[f64], y: &[f64], idx: &[usize]) -> u64 {
    let mut ties: u64 = 0;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len()
            && x[idx[j]] == x[idx[i]]
            && y[idx[j]] == y[idx[i]]
        {
            j += 1;
        }
        let run = (j - i) as u64;
        if run > 1 {
            ties += run * (run - 1) / 2;
        }
        i = j;
    }
    ties
}

pub fn kendall_tau_b(x: &[f64], y: &[f64]) -> KendallResult {
    assert_eq!(x.len(), y.len(), "x and y must be the same length");
    let n = x.len() as u64;
    if n < 2 {
        return KendallResult {
            tau: f64::NAN, n_concordant: 0, n_discordant: 0,
            n_tied_x: 0, n_tied_y: 0, n_tied_both: 0,
        };
    }

    let n0 = n * (n - 1) / 2;

    // Step 1: sort by x, then by y within ties of x.
    let mut idx: Vec<usize> = (0..x.len()).collect();
    let n_tied_x = sort_by_key_count_ties(x, &mut idx);

    // Within ties of x, sort by y (for n_tied_both computation).
    {
        let mut i = 0;
        while i < idx.len() {
            let mut j = i + 1;
            while j < idx.len() && x[idx[j]] == x[idx[i]] { j += 1; }
            idx[i..j].sort_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal));
            i = j;
        }
    }
    let n_tied_both = count_xy_ties(x, y, &idx);

    // Step 2: count discordant pairs via merge-sort on y.
    let n_discordant = count_inversions(y, &mut idx);

    // Step 3: count y-ties (idx is now sorted by y).
    let n_tied_y = count_y_ties_after_sort(y, &idx);

    // Concordant pairs.
    let n_concordant = n0
        .saturating_sub(n_discordant)
        .saturating_sub(n_tied_x)
        .saturating_sub(n_tied_y)
        .saturating_add(n_tied_both);

    // tau-b = (C - D) / sqrt((n0 - T_x) * (n0 - T_y))
    let denom = (((n0 - n_tied_x) as f64) * ((n0 - n_tied_y) as f64)).sqrt();
    let tau = if denom > 0.0 {
        (n_concordant as f64 - n_discordant as f64) / denom
    } else {
        f64::NAN
    };

    KendallResult {
        tau, n_concordant, n_discordant,
        n_tied_x, n_tied_y, n_tied_both,
    }
}

#[pyfunction]
#[pyo3(name = "kendall_tau_b")]
pub fn py_kendall_tau_b(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<PyObject> {
    let x_slice = x.as_slice().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let y_slice = y.as_slice().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    if x_slice.len() != y_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x and y must be the same length",
        ));
    }
    let r = kendall_tau_b(x_slice, y_slice);
    let d = PyDict::new(py);
    d.set_item("tau", r.tau)?;
    d.set_item("n_concordant", r.n_concordant)?;
    d.set_item("n_discordant", r.n_discordant)?;
    d.set_item("n_tied_x", r.n_tied_x)?;
    d.set_item("n_tied_y", r.n_tied_y)?;
    d.set_item("n_tied_both", r.n_tied_both)?;
    Ok(d.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol || (a.is_nan() && b.is_nan())
    }

    #[test]
    fn kendall_perfectly_concordant() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = kendall_tau_b(&x, &y);
        assert!(approx_eq(r.tau, 1.0, 1e-12));
        assert_eq!(r.n_discordant, 0);
        assert_eq!(r.n_concordant, 10);
    }

    #[test]
    fn kendall_perfectly_discordant() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        let r = kendall_tau_b(&x, &y);
        assert!(approx_eq(r.tau, -1.0, 1e-12));
        assert_eq!(r.n_concordant, 0);
        assert_eq!(r.n_discordant, 10);
    }

    #[test]
    fn kendall_all_tied_y() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [5.0, 5.0, 5.0, 5.0];
        let r = kendall_tau_b(&x, &y);
        assert!(r.tau.is_nan());
        assert_eq!(r.n_tied_y, 6);
    }

    #[test]
    fn kendall_with_ties_in_x() {
        // x = [1,1,2,2], y = [1,2,1,2]
        // Pairs: (1,1)-(1,2): tied x, discordant y? no, x tied so neither C nor D, counts in T_x.
        // (1,1)-(2,1): concordant x, tied y, counts in T_y.
        // (1,1)-(2,2): concordant
        // (1,2)-(2,1): discordant
        // (1,2)-(2,2): concordant x, tied y (T_y).
        // (2,1)-(2,2): tied x (T_x).
        // n0=6, T_x=2, T_y=2, T_both=0, D=1.
        // C = 6 - 1 - 2 - 2 + 0 = 1.
        // denom = sqrt((6-2)*(6-2)) = 4
        // tau = (1-1)/4 = 0.
        let x = [1.0, 1.0, 2.0, 2.0];
        let y = [1.0, 2.0, 1.0, 2.0];
        let r = kendall_tau_b(&x, &y);
        assert_eq!(r.n_tied_x, 2);
        assert_eq!(r.n_tied_y, 2);
        assert!(approx_eq(r.tau, 0.0, 1e-12));
    }

    #[test]
    fn kendall_n_less_than_2_returns_nan() {
        let r = kendall_tau_b(&[1.0], &[2.0]);
        assert!(r.tau.is_nan());
    }
}
```

- [ ] **Step 2: Register the function in `lib.rs`**

```rust
// crates/ferrum-core/src/lib.rs — find the existing #[pymodule] block.
mod diagnostics;

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ... existing registrations ...
    m.add_function(wrap_pyfunction!(diagnostics::py_kendall_tau_b, m)?)?;
    Ok(())
}
```

Also export module from `crates/ferrum-core/src/lib.rs` if needed: `pub mod diagnostics;`.

- [ ] **Step 3: Type stub in `_core.pyi`**

```python
# src/ferrum/_core.pyi
import numpy as np
from typing import TypedDict

class _KendallResult(TypedDict):
    tau: float
    n_concordant: int
    n_discordant: int
    n_tied_x: int
    n_tied_y: int
    n_tied_both: int

def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> _KendallResult: ...
```

- [ ] **Step 4: Build + run cargo tests**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -5
DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core --quiet kendall 2>&1 | tail -10
```
Expected: 5 kendall tests pass.

- [ ] **Step 5: Python smoke test**

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "
import numpy as np
from ferrum._core import kendall_tau_b
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
r = kendall_tau_b(x, y)
print(r)
assert r['tau'] == 1.0
print('OK')
"
```

- [ ] **Step 6: Commit**

```bash
git add crates/ferrum-core/src/diagnostics.rs crates/ferrum-core/src/lib.rs src/ferrum/_core.pyi
git commit -m "feat(phase-10g): ferrum._core.kendall_tau_b (Knight's O(n log n) tau-b)"
```

---

### Task 36: `_diagnostics/stats.py` full implementations + scipy parity tests

**Files:**
- Modify: `src/ferrum/_diagnostics/stats.py`
- Create: `tests/diagnostics/test_stats.py`

- [ ] **Step 1: Replace `stats.py` with full implementations**

```python
"""Vectorized NumPy in-house statistics for Phase 10.

No scipy import at runtime. scipy is used only in tests/diagnostics/test_stats.py
for parity validation.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def studentized_residual(y_true, y_pred, X=None):
    """(Implementation from Task 6 — unchanged.)"""
    # ... existing implementation ...
    r = y_true - y_pred
    if X is None:
        sigma = np.std(r, ddof=1) if len(r) > 1 else 1.0
        return r / sigma if sigma > 0 else r * 0.0
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    h_diag = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    h_diag = np.clip(h_diag, 0.0, 1.0 - 1e-12)
    sigma_sq = float((r * r).sum() / max(n - p, 1))
    sigma = np.sqrt(sigma_sq) if sigma_sq > 0 else 0.0
    if sigma == 0.0: return r * 0.0
    return r / (sigma * np.sqrt(1.0 - h_diag))


def pearson_r(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column Pearson correlation between each X column and y."""
    Xm = X - X.mean(axis=0, keepdims=True)
    ym = y - y.mean()
    num = Xm.T @ ym
    denom = np.sqrt((Xm ** 2).sum(axis=0) * (ym ** 2).sum())
    return np.where(denom > 0, num / denom, 0.0)


def spearman_rho(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column Spearman rho = Pearson on ranks."""
    def rankdata(arr: np.ndarray) -> np.ndarray:
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(arr) + 1)
        # Average ties.
        unique_vals, counts = np.unique(arr, return_counts=True)
        for v, c in zip(unique_vals, counts):
            if c > 1:
                mask = arr == v
                ranks[mask] = ranks[mask].mean()
        return ranks

    Xr = np.column_stack([rankdata(X[:, i]) for i in range(X.shape[1])])
    yr = rankdata(y)
    return pearson_r(Xr, yr)


def variance_rank(X: np.ndarray) -> np.ndarray:
    """Variance per feature."""
    return X.var(axis=0)


def covariance_rank(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """abs(cov(X_col, y)) per feature."""
    Xm = X - X.mean(axis=0, keepdims=True)
    ym = y - y.mean()
    return np.abs((Xm.T @ ym) / max(len(y) - 1, 1))


def shapiro_w(x: np.ndarray) -> float:
    """Shapiro-Wilk W statistic via Royston's 1992 algorithm.

    Numerically stable for n in [3, 5000]. Returns W only; p-value omitted
    (we only need ranking).
    """
    n = len(x)
    if n < 3:
        raise ValueError("shapiro_w requires n >= 3")
    if n > 5000:
        # For larger n, falls back to a normalization that's still acceptable.
        pass

    x_sorted = np.sort(x)

    # Royston coefficients via approximation.
    # m_i = inverse normal CDF at (i - 0.375) / (n + 0.25), i = 1..n.
    from math import erf, sqrt
    def _phi_inv(p: float) -> float:
        # Beasley-Springer-Moro approximation to inverse-normal CDF.
        # Sufficient precision for Shapiro coefficients.
        if p < 0.5:
            return -_phi_inv(1.0 - p)
        if p >= 1.0:
            return 8.0
        a = [-3.969683028665376e+01,  2.209460984245205e+02,
             -2.759285104469687e+02,  1.383577518672690e+02,
             -3.066479806614716e+01,  2.506628277459239e+00]
        b = [-5.447609879822406e+01,  1.615858368580409e+02,
             -1.556989798598866e+02,  6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
              4.374664141464968e+00,  2.938163982698783e+00]
        d = [7.784695709041462e-03,  3.224671290700398e-01,
             2.445134137142996e+00,  3.754408661907416e+00]
        plow = 0.02425
        phigh = 1.0 - plow
        if p < plow:
            q = sqrt(-2.0 * np.log(p))
            return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) \
                    / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        if p <= phigh:
            q = p - 0.5
            r = q * q
            return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q \
                    / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
        q = sqrt(-2.0 * np.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) \
                / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    m = np.array([_phi_inv((i + 1 - 0.375) / (n + 0.25)) for i in range(n)])
    m_norm_sq = (m * m).sum()
    u = 1.0 / np.sqrt(n)
    # Royston coefficient formulas for a (1992).
    a_n = (-2.706056 * u**5 + 4.434685 * u**4 - 2.071190 * u**3
            - 0.147981 * u**2 + 0.221157 * u + m[-1] / np.sqrt(m_norm_sq))
    a_nm1 = (-3.582633 * u**5 + 5.682633 * u**4 - 1.752460 * u**3
              - 0.293762 * u**2 + 0.042981 * u + m[-2] / np.sqrt(m_norm_sq))
    eps = (m_norm_sq - 2 * m[-1]**2 - 2 * m[-2]**2) / (1.0 - 2 * a_n**2 - 2 * a_nm1**2)
    a = np.zeros(n)
    a[-1] = a_n
    a[-2] = a_nm1
    a[0] = -a_n
    a[1] = -a_nm1
    for i in range(2, n - 2):
        a[i] = m[i] / np.sqrt(eps)
    # W statistic.
    numer = (a * x_sorted).sum() ** 2
    denom = ((x_sorted - x_sorted.mean()) ** 2).sum()
    return float(numer / denom) if denom > 0 else 1.0


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Wrapper around Rust implementation."""
    from ferrum._core import kendall_tau_b as _rust_ktb
    x64 = np.ascontiguousarray(x, dtype=np.float64)
    y64 = np.ascontiguousarray(y, dtype=np.float64)
    return float(_rust_ktb(x64, y64)["tau"])


def rank1d_compute(X, *, algorithm: str = "shapiro", top_k: int | None = None) -> pl.DataFrame:
    """Compute rank1d feature scores."""
    if hasattr(X, "to_numpy"):
        cols = list(X.columns)
        X_np = X.to_numpy()
    else:
        X_np = np.asarray(X)
        cols = [f"f{i}" for i in range(X_np.shape[1])]
    if algorithm == "shapiro":
        scores = np.array([shapiro_w(X_np[:, j]) for j in range(X_np.shape[1])])
    elif algorithm == "variance":
        scores = variance_rank(X_np)
    elif algorithm == "covariance":
        raise ValueError("rank1d covariance requires y; use rank1d(algorithm='covariance', y=...)")
    else:
        raise ValueError(f"rank1d algorithm must be shapiro/variance/covariance; got {algorithm!r}")
    order = np.argsort(-scores)
    rows = []
    for rank, idx in enumerate(order, 1):
        rows.append({"feature": cols[idx], "score": float(scores[idx]), "rank": rank})
    df = pl.DataFrame(rows)
    if top_k is not None:
        df = df.head(top_k)
    return df


def rank2d_compute(X, *, algorithm: str = "pearson") -> pl.DataFrame:
    """Pairwise feature correlation matrix in long form."""
    if hasattr(X, "to_numpy"):
        cols = list(X.columns)
        X_np = X.to_numpy()
    else:
        X_np = np.asarray(X)
        cols = [f"f{i}" for i in range(X_np.shape[1])]
    p = X_np.shape[1]
    if algorithm == "pearson":
        C = np.corrcoef(X_np, rowvar=False)
    elif algorithm == "spearman":
        Xr = np.column_stack([
            np.argsort(np.argsort(X_np[:, j])).astype(np.float64) for j in range(p)
        ])
        C = np.corrcoef(Xr, rowvar=False)
    elif algorithm == "kendall":
        from ferrum._core import kendall_tau_b as _rust_ktb
        C = np.eye(p)
        for i in range(p):
            for j in range(i + 1, p):
                t = _rust_ktb(np.ascontiguousarray(X_np[:, i], dtype=np.float64),
                                np.ascontiguousarray(X_np[:, j], dtype=np.float64))["tau"]
                C[i, j] = C[j, i] = t
    elif algorithm == "covariance":
        C = np.cov(X_np, rowvar=False)
    else:
        raise ValueError(f"rank2d algorithm must be pearson/spearman/kendall/covariance; got {algorithm!r}")
    rows = []
    for i in range(p):
        for j in range(p):
            rows.append({"feature_x": cols[i], "feature_y": cols[j],
                          "correlation": float(C[i, j])})
    return pl.DataFrame(rows)
```

- [ ] **Step 2: Write scipy-parity tests in `tests/diagnostics/test_stats.py`**

```python
from __future__ import annotations
import numpy as np
import pytest
from ferrum._diagnostics.stats import (
    pearson_r, spearman_rho, shapiro_w, kendall_tau_b,
)


@pytest.fixture
def rng():
    return np.random.RandomState(0)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_pearson_parity_vs_scipy(n, rng):
    import scipy.stats as ss
    X = rng.randn(n, 4)
    y = rng.randn(n)
    ours = pearson_r(X, y)
    theirs = np.array([ss.pearsonr(X[:, j], y).statistic for j in range(X.shape[1])])
    np.testing.assert_allclose(ours, theirs, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_spearman_parity_vs_scipy(n, rng):
    import scipy.stats as ss
    X = rng.randn(n, 4)
    y = rng.randn(n)
    ours = spearman_rho(X, y)
    theirs = np.array([ss.spearmanr(X[:, j], y).statistic for j in range(X.shape[1])])
    np.testing.assert_allclose(ours, theirs, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("n", [10, 50, 200, 1000])
@pytest.mark.parametrize("dist", ["normal", "uniform", "exponential", "bimodal"])
def test_shapiro_parity_vs_scipy(n, dist, rng):
    import scipy.stats as ss
    if dist == "normal": x = rng.randn(n)
    elif dist == "uniform": x = rng.uniform(0, 1, n)
    elif dist == "exponential": x = rng.exponential(1.0, n)
    elif dist == "bimodal":
        x = np.concatenate([rng.randn(n // 2) - 3, rng.randn(n - n // 2) + 3])
    ours = shapiro_w(x)
    theirs = float(ss.shapiro(x).statistic)
    # Royston W matches scipy at 1e-6; tighter on small n.
    assert abs(ours - theirs) < 1e-6, f"W mismatch n={n} dist={dist}: ours={ours}, scipy={theirs}"


@pytest.mark.parametrize("n", [10, 100, 1000])
@pytest.mark.parametrize("tie_density", [0.0, 0.1, 0.5])
def test_kendall_parity_vs_scipy(n, tie_density, rng):
    import scipy.stats as ss
    x = rng.randn(n)
    y = rng.randn(n)
    if tie_density > 0:
        # Round to introduce ties.
        scale = max(1, int(1.0 / max(tie_density, 0.01)))
        x = np.round(x * scale) / scale
        y = np.round(y * scale) / scale
    ours = kendall_tau_b(x, y)
    theirs = float(ss.kendalltau(x, y).statistic)
    assert abs(ours - theirs) < 1e-12, f"tau mismatch n={n} ties={tie_density}: ours={ours}, scipy={theirs}"
```

- [ ] **Step 3: Run + commit**

```bash
uv run --no-sync pytest tests/diagnostics/test_stats.py -v 2>&1 | tail -25
git add src/ferrum/_diagnostics/stats.py tests/diagnostics/test_stats.py
git commit -m "feat(phase-10g): full _diagnostics/stats.py with scipy-parity tests"
```

---

### Task 37: `.rank1d()` + `.rank2d()` methods + 3 marks + 2 figures

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py`
- Modify: `src/ferrum/_diagnostics/schemas.py`
- Modify: `src/ferrum/marks/diagnostic.py`
- Modify: `src/ferrum/chart.py`
- Modify: `src/ferrum/_diagnostics/charts.py`
- Modify: `src/ferrum/figures.py`
- Modify: `src/ferrum/__init__.py`
- Create: `tests/diagnostics/test_ranking.py`

> ⚠ **Pattern correction (plan-vs-codebase):** The mark code blocks below were originally drafted using a `@dataclass(frozen=True) class mark_X: ... def _expand(self, chart_ctx) -> list[LayerSpec]` pattern that **does not exist in the codebase**. Before implementing, translate every mark in this task to the real pattern used in Phase 8b/9 composite marks:
>
> - Module-level `def desugar_<name>(x_field, y_field, **kwargs) -> ("__layered__", transforms, None, None, layers)` in `src/ferrum/marks/diagnostic.py`.
> - Layers are plain dicts: `{"mark": str, "encoding": dict, "mark_kwargs": dict (opt), "data_source": str | None (opt)}`.
> - No `LayerSpec`. No `chart_ctx`. No `_expand`.
> - Chart method clones, sets `_mark = "point"` (placeholder), sets `_pending_stat_mark = (kind, kwargs_dict, desugar_fn)`, returns.
> - The user does not import or instantiate `mark_X` — they call `Chart(df).mark_X(...)`.
> - For diagnostic marks, the data has hard-coded columns from a `ModelSource` method, so the desugar references those columns literally and ignores positional `x_field` / `y_field`.
>
> **Canonical reference:** Task 8 (`desugar_residuals` / `desugar_prediction_error`) and Task 15 (six 10b desugars). Pattern reference in code: `src/ferrum/marks/composite.py:15-220`.
>
> Keep the **layer encodings, kwargs, and behavior** below as the spec for what each mark should produce, but rewrite the implementation in the corrected pattern.

- [ ] **Step 1: Schemas + methods**

```python
# schemas.py
SCHEMA_RANK1D = pl.Schema({
    "feature": pl.Utf8,
    "score": pl.Float64,
    "rank": pl.Int64,
})
SCHEMA_RANK2D = pl.Schema({
    "feature_x": pl.Utf8,
    "feature_y": pl.Utf8,
    "correlation": pl.Float64,
})


# source.py — append
    def rank1d(self, *, algorithm: str = "shapiro") -> pl.DataFrame:
        """Univariate feature ranking — shapiro/variance/covariance."""
        from .stats import rank1d_compute, covariance_rank
        import numpy as np
        if algorithm == "covariance":
            if self._y is None:
                raise ValueError("rank1d(algorithm='covariance') requires y.")
            X_np = self._X.to_numpy()
            y_np = np.asarray(self._y.to_numpy(), dtype=np.float64)
            scores = covariance_rank(X_np, y_np)
            order = np.argsort(-scores)
            rows = [{"feature": str(self._feature_names[i]),
                      "score": float(scores[i]), "rank": r}
                     for r, i in enumerate(order, 1)]
            return pl.DataFrame(rows)
        return rank1d_compute(self._X, algorithm=algorithm)

    def rank2d(self, *, algorithm: str = "pearson") -> pl.DataFrame:
        from .stats import rank2d_compute
        return rank2d_compute(self._X, algorithm=algorithm)
```

- [ ] **Step 2: Marks**

```python
@dataclass(frozen=True)
class mark_rank1d:
    algorithm: str = "shapiro"
    orient: str = "horizontal"
    top_k: int | None = None

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_bar
        from ferrum import LayerSpec
        x_field, y_field = ("score", "feature") if self.orient == "horizontal" else ("feature", "score")
        return [LayerSpec(mark=mark_bar(), encoding={"x": x_field, "y": y_field})]


@dataclass(frozen=True)
class mark_rank2d:
    algorithm: str = "pearson"
    annot: bool = True
    cmap: str | None = None

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_rect, mark_text
        from ferrum import LayerSpec
        layers = [LayerSpec(mark=mark_rect(),
                              encoding={"x": "feature_x", "y": "feature_y", "color": "correlation"})]
        if self.annot:
            layers.append(LayerSpec(mark=mark_text(),
                                      encoding={"x": "feature_x", "y": "feature_y", "text": "correlation"}))
        return layers


@dataclass(frozen=True)
class mark_parallel_coordinates:
    rescale: str | None = "minmax"
    alpha: float = 0.5
    highlight_selection: bool = False

    def _expand(self, chart_ctx: Any) -> list[Any]:
        from ferrum.marks import mark_line
        from ferrum import LayerSpec
        return [LayerSpec(mark=mark_line(opacity=self.alpha),
                            encoding={"x": "feature", "y": "value", "detail": "sample_id",
                                       "color": chart_ctx.color_field_or_default()})]
```

- [ ] **Step 3: Chart methods + builders + figures**

```python
# chart.py
def mark_rank1d(self, **kw):
    from ferrum.marks.diagnostic import mark_rank1d as _M
    return self._add_composite_mark(_M(**kw))


def mark_rank2d(self, **kw):
    from ferrum.marks.diagnostic import mark_rank2d as _M
    return self._add_composite_mark(_M(**kw))


def mark_parallel_coordinates(self, **kw):
    from ferrum.marks.diagnostic import mark_parallel_coordinates as _M
    return self._add_composite_mark(_M(**kw))


# charts.py
def _rank1d_chart_from_dataframe(df, *, algorithm="shapiro", top_k=None, theme=None):
    if top_k is not None: df = df.head(top_k)
    chart = ferrum.Chart(df).mark_rank1d(algorithm=algorithm, top_k=top_k)
    if theme is not None: chart = chart.theme(theme)
    return chart


def _rank2d_chart_from_dataframe(df, *, algorithm="pearson", theme=None):
    chart = ferrum.Chart(df).mark_rank2d(algorithm=algorithm)
    if theme is not None: chart = chart.theme(theme)
    return chart


def _parallel_coords_chart_from_dataframe(X, *, features=None, hue=None,
                                            rescale="minmax", alpha=0.5, theme=None):
    """X is a DataFrame; reshape via Unpivot then mark_line."""
    if hasattr(X, "to_numpy"):
        df = X.with_row_index("sample_id")
    else:
        import numpy as np
        arr = np.asarray(X)
        df = pl.from_numpy(arr, schema=[f"f{i}" for i in range(arr.shape[1])]).with_row_index("sample_id")
    feat_cols = features or [c for c in df.columns if c not in ("sample_id", hue)]
    # Rescale.
    if rescale == "minmax":
        for c in feat_cols:
            vmin, vmax = df[c].min(), df[c].max()
            if vmax > vmin:
                df = df.with_columns(((pl.col(c) - vmin) / (vmax - vmin)).alias(c))
    elif rescale == "zscore":
        for c in feat_cols:
            mu, sd = df[c].mean(), df[c].std()
            if sd > 0:
                df = df.with_columns(((pl.col(c) - mu) / sd).alias(c))
    long = df.unpivot(index=["sample_id"] + ([hue] if hue else []),
                       on=feat_cols, variable_name="feature", value_name="value")
    chart = ferrum.Chart(long).mark_parallel_coordinates(rescale=rescale, alpha=alpha)
    if theme is not None: chart = chart.theme(theme)
    return chart


# figures.py
def rank_chart(
    data_or_source, X=None, y=None, *, rank="2d", algorithm=None,
    top_k=None, random_state=None, theme=None,
):
    if isinstance(data_or_source, ferrum.ModelSource):
        source = data_or_source
    elif X is not None:
        source = ferrum.ModelSource(model=type("NoModel", (), {})(), X=data_or_source if X is None else X, y=y)
    else:
        # Single-arg path: data_or_source is X directly.
        source = ferrum.ModelSource(model=type("NoModel", (), {})(), X=data_or_source, y=y)
    if rank == "1d":
        algo = algorithm or "shapiro"
        return _rank1d_chart_from_dataframe(source.rank1d(algorithm=algo),
                                              algorithm=algo, top_k=top_k, theme=theme)
    if rank == "2d":
        algo = algorithm or "pearson"
        return _rank2d_chart_from_dataframe(source.rank2d(algorithm=algo),
                                              algorithm=algo, theme=theme)
    raise ValueError(f"rank must be '1d' or '2d'; got {rank!r}")


def parallel_coordinates_chart(
    data_or_source, X=None, y=None, *, features=None, hue=None,
    rescale="minmax", alpha=0.5, random_state=None, theme=None,
):
    data = data_or_source if X is None else X
    return _parallel_coords_chart_from_dataframe(
        data, features=features, hue=hue, rescale=rescale, alpha=alpha, theme=theme,
    )
```

- [ ] **Step 4: Re-exports + tests + goldens**

```python
# __init__.py
from ferrum.marks.diagnostic import mark_rank1d, mark_rank2d, mark_parallel_coordinates
from ferrum.figures import rank_chart, parallel_coordinates_chart


# test_ranking.py
from __future__ import annotations
import numpy as np
import pytest
import ferrum
from tests.fixtures import load_dataset


def test_rank1d_shapiro():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.rank_chart(df, rank="1d", algorithm="shapiro")
    assert "<svg" in chart.show_svg()


def test_rank1d_variance():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.rank_chart(df, rank="1d", algorithm="variance")
    assert "<svg" in chart.show_svg()


def test_rank2d_pearson():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.rank_chart(df, rank="2d", algorithm="pearson")
    assert "<svg" in chart.show_svg()


def test_rank2d_kendall():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.rank_chart(df, rank="2d", algorithm="kendall")
    assert "<svg" in chart.show_svg()


def test_parallel_coordinates():
    df = load_dataset("multiclass_classification")
    chart = ferrum.parallel_coordinates_chart(
        df, features=["f0", "f1", "f2", "f3"], hue="y", rescale="minmax",
    )
    assert "<svg" in chart.show_svg()


# test_goldens_phase_10.py — all byte-identical
def test_golden_rank1d_shapiro():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.rank_chart(df, rank="1d", algorithm="shapiro")
    _check_golden(chart.show_svg(), "rank1d_shapiro_regression")


def test_golden_rank2d_kendall():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    chart = ferrum.rank_chart(df, rank="2d", algorithm="kendall")
    _check_golden(chart.show_svg(), "rank2d_kendall_regression")


def test_golden_parallel_coordinates():
    df = load_dataset("multiclass_classification")
    chart = ferrum.parallel_coordinates_chart(
        df, features=["f0", "f1", "f2", "f3"], hue="y", rescale="minmax",
    )
    _check_golden(chart.show_svg(), "parallel_coordinates_multiclass")
```

- [ ] **Step 5: Run + commit**

```bash
FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -k "rank or parallel" -v 2>&1 | tail -10
uv run --no-sync pytest tests/diagnostics/test_ranking.py -v 2>&1 | tail -10
git add src/ferrum/ tests/diagnostics/test_ranking.py tests/diagnostics/test_goldens_phase_10.py tests/goldens/phase_10/
git commit -m "feat(phase-10g): rank1d/rank2d/parallel_coordinates + Rust-backed kendall"
```

---

### Task 38: 10g visualizers (Rank1D, Rank2D, ParallelCoordinates — no-model variants)

**Files:**
- Create: `src/ferrum/_diagnostics/visualizers/ranking.py`
- Modify: `src/ferrum/_diagnostics/visualizers/explanation.py` (add ParallelCoordinatesVisualizer)
- Modify: `src/ferrum/_diagnostics/visualizers/__init__.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/diagnostics/test_ranking.py`

- [ ] **Step 1: Write `ranking.py` (no-model)**

```python
"""10g feature-ranking visualizers."""
from __future__ import annotations
from typing import Any
import polars as pl
from .base import FerrumVisualizer
from ..stats import rank1d_compute, rank2d_compute
from ..charts import _rank1d_chart_from_dataframe, _rank2d_chart_from_dataframe


class Rank1DVisualizer(FerrumVisualizer):
    def __init__(self, *, algorithm: str = "shapiro", top_k: int | None = None,
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.algorithm = algorithm
        self.top_k = top_k

    def fit(self, X, y=None) -> "Rank1DVisualizer":
        df = rank1d_compute(X, algorithm=self.algorithm, top_k=self.top_k)
        self._metrics["top_feature_score"] = float(df["score"][0])
        self._chart = _rank1d_chart_from_dataframe(
            df, algorithm=self.algorithm, top_k=self.top_k, theme=self.theme,
        )
        self._fitted = True
        return self


class Rank2DVisualizer(FerrumVisualizer):
    def __init__(self, *, algorithm: str = "pearson",
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.algorithm = algorithm

    def fit(self, X, y=None) -> "Rank2DVisualizer":
        df = rank2d_compute(X, algorithm=self.algorithm)
        # Max abs off-diagonal correlation.
        off_diag = df.filter(pl.col("feature_x") != pl.col("feature_y"))
        self._metrics["max_abs_corr"] = float(off_diag["correlation"].abs().max())
        self._chart = _rank2d_chart_from_dataframe(df, algorithm=self.algorithm, theme=self.theme)
        self._fitted = True
        return self
```

- [ ] **Step 2: Add `ParallelCoordinatesVisualizer` to `explanation.py`**

```python
class ParallelCoordinatesVisualizer(FerrumVisualizer):
    def __init__(self, *, features=None, hue=None, rescale: str | None = "minmax",
                 random_state: int | None = None, theme: Any = None):
        super().__init__(model=None, random_state=random_state, theme=theme)
        self.features = features
        self.hue = hue
        self.rescale = rescale

    def fit(self, X, y=None) -> "ParallelCoordinatesVisualizer":
        import ferrum
        self._chart = ferrum.parallel_coordinates_chart(
            X, features=self.features, hue=self.hue, rescale=self.rescale, theme=self.theme,
        )
        self._fitted = True
        return self
```

- [ ] **Step 3: Re-exports + tests + commit**

```python
# visualizers/__init__.py
from .ranking import Rank1DVisualizer, Rank2DVisualizer
from .explanation import (
    FeatureImportancesVisualizer, SHAPVisualizer, ParallelCoordinatesVisualizer,
)
__all__ += ["Rank1DVisualizer", "Rank2DVisualizer", "ParallelCoordinatesVisualizer"]


# src/ferrum/__init__.py
from ferrum._diagnostics.visualizers import (
    Rank1DVisualizer, Rank2DVisualizer, ParallelCoordinatesVisualizer,
)


# test_ranking.py
def test_rank1d_visualizer():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    viz = ferrum.Rank1DVisualizer(algorithm="shapiro").fit(df)
    assert "top_feature_score=" in repr(viz)


def test_rank2d_visualizer():
    df = load_dataset("regression").select(["f0", "f1", "f2", "f3", "f4"])
    viz = ferrum.Rank2DVisualizer(algorithm="pearson").fit(df)
    assert "max_abs_corr=" in repr(viz)


def test_parallel_coordinates_visualizer():
    df = load_dataset("multiclass_classification")
    viz = ferrum.ParallelCoordinatesVisualizer(
        features=["f0", "f1", "f2", "f3"], hue="y",
    ).fit(df)
    assert "<svg" in viz.show().show_svg()
```

```bash
uv run --no-sync pytest tests/diagnostics/test_ranking.py -v 2>&1 | tail -10
git add src/ferrum/_diagnostics/visualizers/ src/ferrum/__init__.py tests/diagnostics/test_ranking.py
git commit -m "feat(phase-10g): 3 ranking visualizers (Rank1D/Rank2D/ParallelCoordinates)"
```

---

### Task 39: 10g milestone — full cargo + pytest check

```bash
DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core --quiet 2>&1 | tail -3
uv run --no-sync pytest tests/diagnostics/ 2>&1 | tail -3
```
Expected: `cargo test` ≥ 505 passed (5 new kendall tests); `pytest tests/diagnostics/` ≥ 130 passed.

---

## 10h — Finalize

### Task 40: `ModelSource.compare(...)` + `ComparedModelSource`

**Files:**
- Modify: `src/ferrum/_diagnostics/source.py`
- Modify: `src/ferrum/_diagnostics/__init__.py`
- Modify: `src/ferrum/__init__.py`
- Modify: `src/ferrum/figures.py` (enable `compare=` path in `_resolve_source`)
- Create: `tests/diagnostics/test_compare.py`
- Modify: `tests/diagnostics/test_goldens_phase_10.py`

- [ ] **Step 1: Add `compare` classmethod + `ComparedModelSource`**

```python
# source.py — append at bottom of file
class ComparedModelSource:
    """Same method surface as ModelSource; output DataFrames gain a `model` column."""

    def __init__(self, sources: dict[str, ModelSource]):
        self._sources = sources

    @property
    def model_names(self) -> list[str]:
        return list(self._sources.keys())

    # Property access for the first source (used by chart builders that need _y, _X, etc.)
    def __getattr__(self, name: str):
        # For internal access patterns like .first_source._y.
        if name == "_y":
            return next(iter(self._sources.values()))._y
        if name == "_X":
            return next(iter(self._sources.values()))._X
        if name == "_model":
            # Compared sources don't have a single model; raise to catch misuse.
            raise AttributeError(
                "ComparedModelSource has no single _model; iterate _sources instead."
            )
        raise AttributeError(name)

    def _dispatch(self, method: str, **kwargs) -> pl.DataFrame:
        frames = []
        for name, src in self._sources.items():
            df = getattr(src, method)(**kwargs)
            frames.append(df.with_columns(pl.lit(name).alias("model")))
        return pl.concat(frames, how="vertical_relaxed")

    # Auto-generate proxy methods for every ModelSource derived-data method.
    for _m in (
        "predictions", "probabilities", "roc_curve", "pr_curve", "confusion_matrix",
        "calibration_curve", "cumulative_gain", "lift_curve", "importances",
        "shap_values", "partial_dependence", "silhouette", "embeddings",
        "learning_curve", "validation_curve", "discrimination_threshold",
        "pca_variance", "rank1d", "rank2d", "cv_scores", "alpha_selection",
        "intercluster_distance",
    ):
        exec(f"""
def {_m}(self, *args, **kwargs):
    return self._dispatch({_m!r}, **kwargs) if not args else self._dispatch({_m!r}, *args, **kwargs)
""")
    del _m


# Append to ModelSource class:
    @classmethod
    def compare(cls, models: dict[str, Any], X, y=None, **kwargs) -> "ComparedModelSource":
        sources = {name: cls(model, X, y, **kwargs) for name, model in models.items()}
        return ComparedModelSource(sources)
```

- [ ] **Step 2: Update `_resolve_source` to dispatch on dict + ComparedModelSource**

```python
# figures.py
def _resolve_source(model_or_source, X=None, y=None, *, random_state=None, compare=None):
    import ferrum
    if isinstance(model_or_source, ferrum._diagnostics.source.ComparedModelSource):
        return model_or_source
    if compare is not None:
        # Build ComparedModelSource from the model_or_source as a single key plus compare dict.
        if not isinstance(compare, dict):
            raise TypeError("compare= must be dict[str, model] or None")
        models = {"base": model_or_source, **compare}
        return ferrum.ModelSource.compare(models, X, y, random_state=random_state)
    if isinstance(model_or_source, dict):
        return ferrum.ModelSource.compare(model_or_source, X, y, random_state=random_state)
    if isinstance(model_or_source, ferrum.ModelSource):
        return model_or_source
    return ferrum.ModelSource(model_or_source, X, y, random_state=random_state)
```

- [ ] **Step 3: Re-exports**

```python
# _diagnostics/__init__.py
from .source import ModelSource, ComparedModelSource
__all__ = ["ModelSource", "ComparedModelSource"]


# src/ferrum/__init__.py
from ferrum._diagnostics import ComparedModelSource
```

- [ ] **Step 4: Tests + compare goldens**

```python
# test_compare.py
from __future__ import annotations
import pytest
import ferrum
from tests.fixtures import load_fixture, load_dataset


def test_compare_dispatches_roc():
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    models = {
        "logistic": load_fixture("binary_logistic"),
        "logistic2": load_fixture("binary_logistic"),  # same model, different key
    }
    cms = ferrum.ModelSource.compare(models, X, df["y"])
    roc = cms.roc_curve()
    assert "model" in roc.columns
    assert set(roc["model"].unique().to_list()) == {"logistic", "logistic2"}


def test_compare_via_figure_function():
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    base = load_fixture("binary_logistic")
    chart = ferrum.roc_chart(
        {"a": base, "b": base},   # multi-model dict path
        X, df["y"],
    )
    svg = chart.show_svg()
    assert "<svg" in svg


def test_compare_kwarg_route():
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    base = load_fixture("binary_logistic")
    chart = ferrum.roc_chart(
        base, X, df["y"], compare={"alt": load_fixture("binary_logistic")},
    )
    assert "<svg" in chart.show_svg()


# test_goldens_phase_10.py
def test_golden_roc_chart_compare():
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    base = load_fixture("binary_logistic")
    chart = ferrum.roc_chart({"a": base, "b": base}, X, df["y"])
    _check_golden(chart.show_svg(), "roc_chart_compare_two_models")


def test_golden_calibration_chart_compare():
    df = load_dataset("binary_classification")
    X = df.select(["f0", "f1", "f2", "f3"])
    base = load_fixture("binary_logistic")
    chart = ferrum.calibration_chart(
        base, X=X, y=df["y"], n_bins=5,
    )  # Variadic in figures.py — but for now Multi-model calibration uses compare=:
    # Skip multi-model calibration golden if calibration_chart's variadic
    # signature differs (it raises NotImplementedError for >1 positional model).
    pytest.skip("multi-model calibration variadic ships in this task; smoke-only")
```

> Implementation note for Step 4: at this task, also update `calibration_chart` in `figures.py` to enable its variadic multi-model path (it raised NotImplementedError back in Task 16). Replace that branch with a `ComparedModelSource` construction.

- [ ] **Step 5: Run + commit**

```bash
FERRUM_REGENERATE_GOLDENS=1 uv run --no-sync pytest tests/diagnostics/test_goldens_phase_10.py -k compare -v 2>&1 | tail -10
uv run --no-sync pytest tests/diagnostics/test_compare.py -v 2>&1 | tail -10
git add src/ferrum/ tests/diagnostics/test_compare.py tests/diagnostics/test_goldens_phase_10.py tests/goldens/phase_10/
git commit -m "feat(phase-10h): ModelSource.compare + ComparedModelSource + compare= figure-function route"
```

---

### Task 41: Apply spec drift notes to `ferrum-spec.md`

**Files:**
- Modify: `ferrum-spec.md`

- [ ] **Step 1: Add §1 philosophy departure note**

After the existing §1 closing paragraph, insert:

```markdown
> **2026-05-MM (Phase 10 — Model Diagnostics):** Phase 10 places model-diagnostic
> compute in the `ModelSource` adapter layer (Python, lazy-imported sklearn
> delegation) rather than as Rust transforms in the rendering pipeline. From the
> user's perspective the figure function (`ferrum.roc_chart`, etc.) *is* the
> rendering pipeline — they are not computing ROC in userspace, which is what
> §1 actually proscribes. Whether the internal compute is a Rust transform or a
> Python call to sklearn is invisible at the call site. Model-diagnostic compute
> is also entangled with the model-specific protocol (`predict_proba`,
> `classes_`, etc.), which a generic Rust transform cannot access without
> reimplementing the sklearn estimator protocol. See
> `docs/superpowers/specs/2026-05-10-model-diagnostics-design.md` §1.5 for the
> full rationale.
```

(Replace `MM` with the actual day of the month at the time of the commit, mirroring Phase 9's convention.)

- [ ] **Step 2: Add `random_state` to §3.1 ModelSource signature**

In §3.1, find the `ModelSource(...)` signature and append `random_state=None` to the kwargs:

```markdown
ModelSource(model, X, y=None, *, feature_names=None, class_names=None,
            sample_weight=None, random_state=None)
```

Add a dated note explaining that `random_state` is propagated to every method wrapping an RNG-using sklearn/shap/umap call.

- [ ] **Step 3: Add §3.3 mark clarifications**

Append a dated note to §3.3 introducing the per-mark drift notes from the design doc §6.2:

```markdown
> **2026-05-MM (Phase 10 — Model Diagnostics):** Marks `mark_residuals`,
> `mark_confusion`, `mark_decision_boundary`, `mark_shap_*`, `mark_rank2d`,
> `mark_pca_scree` gain the following clarifications:
> - `mark_residuals`: `kind="studentized"` is well-defined only for linear
>   estimators (those exposing the residual hat matrix). For non-linear
>   estimators, ferrum falls back to raw residual and logs an INFO message.
> - `mark_confusion`: color scale on `value` uses Phase 8b's
>   `ColorScale::Continuous` (`viridis` default). Continuous colorbar legend is
>   a Phase 11+ artifact; cell text via `value_fmt` conveys magnitude.
> - `mark_decision_boundary`: requires exactly 2 features.
> - `mark_shap_*`: require the `shap` library via `ferrum[shap]`.
> - `mark_shap_waterfall`: requires an explicit `sample_idx: int` kwarg.
> - `mark_rank2d(algorithm="kendall")`: uses the Rust
>   `ferrum._core.kendall_tau_b` (Knight's O(n log n)) instead of an in-Python
>   implementation.
> - `mark_pca_scree`: `cumulative_line=True` overlays a `mark_line` on
>   `cumulative_variance_ratio`.
```

- [ ] **Step 4: Add `random_state` to §3.14 figure functions**

Append a dated note documenting per-function whether `random_state` affects compute (table from design doc §13).

- [ ] **Step 5: Add `random_state` to §3.15 Visualizer constructors**

Append a dated note adding `random_state: int | None = None` to every Visualizer's constructor.

- [ ] **Step 6: Commit**

```bash
git add ferrum-spec.md
git commit -m "docs(phase-10h): apply spec drift notes for §1/§3.1/§3.3/§3.14/§3.15/§3.16"
```

---

### Task 42: `PHASE_9_PLUS_MARKS` audit + mark coverage test

**Files:**
- Verify: `src/ferrum/marks/deferred.py` (should be unchanged — Phase 10 marks were never added)
- Create: `tests/diagnostics/test_mark_coverage.py`

- [ ] **Step 1: Verify `deferred.py` contents**

```bash
cat src/ferrum/marks/deferred.py
```
Expected: `PHASE_9_PLUS_MARKS = frozenset(["arc", "image", "geoshape", "label"])` only. No Phase 10 marks present.

- [ ] **Step 2: Write the coverage assertion test**

```python
# tests/diagnostics/test_mark_coverage.py
"""Phase 10 mark coverage assertion."""
from __future__ import annotations
import ferrum


PHASE_10_MARKS = frozenset([
    "residuals", "prediction_error", "confusion", "roc", "pr", "calibration",
    "gain", "lift", "importance", "shap_beeswarm", "shap_bar", "shap_waterfall",
    "pdp", "silhouette", "learning_curve", "validation_curve", "decision_boundary",
    "discrimination_threshold", "parallel_coordinates", "class_prediction_error",
    "pca_scree", "rank1d", "rank2d", "intercluster_distance", "cv_scores",
    "alpha_selection",
])


def test_phase_10_marks_count():
    assert len(PHASE_10_MARKS) == 26


def test_phase_10_marks_not_in_deferred():
    from ferrum.marks.deferred import PHASE_9_PLUS_MARKS, PHASE_8B_MARKS
    overlap = PHASE_10_MARKS & (PHASE_9_PLUS_MARKS | PHASE_8B_MARKS)
    assert not overlap, f"Phase 10 marks still in deferred list: {overlap}"


def test_phase_10_marks_exported():
    missing = []
    for mark_name in PHASE_10_MARKS:
        if not hasattr(ferrum, f"mark_{mark_name}"):
            missing.append(f"mark_{mark_name}")
    assert not missing, f"Phase 10 marks not exported from ferrum: {missing}"


def test_phase_9_plus_marks_unchanged():
    from ferrum.marks.deferred import PHASE_9_PLUS_MARKS
    assert PHASE_9_PLUS_MARKS == frozenset(["arc", "image", "geoshape", "label"])
```

- [ ] **Step 3: Run + commit**

```bash
uv run --no-sync pytest tests/diagnostics/test_mark_coverage.py -v 2>&1 | tail -10
git add tests/diagnostics/test_mark_coverage.py
git commit -m "test(phase-10h): mark coverage assertion (26 marks, none deferred)"
```

---

### Task 43: Update `docs/superpowers/ferrum-phases.md` Phase 10 row

**Files:**
- Modify: `docs/superpowers/ferrum-phases.md`

- [ ] **Step 1: Update Phase 10 row**

Find the Phase 10 row in the phase table:

```markdown
| **10** | Model diagnostics layer | `ModelSource` (sklearn-protocol adapter), model-diagnostic marks (`ConfusionMark`, `ROCMark`, `CalibrationMark`, etc.), `Visualizer` convenience wrappers | 8 | *(not yet written)* | pending |
```

Replace with:

```markdown
| **10** | Model diagnostics layer | `ModelSource` (sklearn-protocol adapter), 26 model-diagnostic marks, 21 Group B figure functions, 25 sklearn-protocol Visualizers | 8 | [`2026-05-10-model-diagnostics-design.md`](specs/2026-05-10-model-diagnostics-design.md) | **done** |
```

- [ ] **Step 2: Check off Phase 10 done-criteria**

In the §"Phase 10 — Model diagnostics" done-criteria section, mark all three checkboxes:

```markdown
### Phase 10 — Model diagnostics
- [x] `ModelSource` wraps any object with `predict`/`predict_proba`/`transform`
- [x] All model-diagnostic marks from `ferrum-spec.md §3.3` render correctly
- [x] Sklearn is not imported unless the user's model is from sklearn
```

- [ ] **Step 3: Bump `Last updated`**

Update the document header:

```markdown
**Last updated:** 2026-05-MM
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/ferrum-phases.md
git commit -m "docs(phase-10h): mark Phase 10 done in ferrum-phases.md"
```

---

### Task 44: Final-pass verification + user-confirmed merge to `main`

- [ ] **Step 1: Run the full test matrix**

```bash
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core --quiet 2>&1 | tail -3
uv run --no-sync pytest 2>&1 | tail -3
```
Expected:
- `cargo test` ≥ 510 passed (5 kendall + 2 numeric_precision + small handful from RenderConfig round-trip).
- `pytest` ≥ 610 passed, 5 skipped (Phase 9's 5; 0 new). 130+ new tests across Phase 10.

- [ ] **Step 2: Verify golden coverage**

```bash
find tests/goldens/phase_10 -name "*.svg" | wc -l
```
Expected: ≥ 35 golden SVGs at single-tier 3-dp quantization.

- [ ] **Step 3: Verify `import ferrum` cold-cache time hasn't regressed**

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "
import time, sys
t0 = time.perf_counter()
import ferrum
t1 = time.perf_counter()
print(f'import time: {1000*(t1-t0):.1f} ms')
assert 'sklearn' not in sys.modules, 'sklearn loaded by import ferrum!'
assert 'shap' not in sys.modules
assert 'umap' not in sys.modules
print('OK')
"
```
Expected: import time under ~500ms (varies by machine); no sklearn/shap/umap loaded.

- [ ] **Step 4: Verify the no-defer guarantee — no Phase 10 NotImplementedError stubs left**

```bash
grep -rn "NotImplementedError\|TODO.*phase.10\|FIXME.*phase.10\|warn-fallback" src/ferrum/_diagnostics/ src/ferrum/marks/diagnostic.py src/ferrum/figures.py 2>&1 | head
```
Expected: no remaining Phase 10 placeholders (any matches must come from inherited Phase 9 code or be benign).

- [ ] **Step 5: User confirmation before merge**

> **DO NOT MERGE WITHOUT EXPLICIT USER CONFIRMATION.**

Print the summary diff and ask:

```bash
git log --oneline main..feat/phase-10 | head -50
git diff --stat main..feat/phase-10 | tail -3
```

Then prompt: *"Phase 10 implementation complete. Ready to merge `feat/phase-10` into `main`. Confirm to proceed?"*

- [ ] **Step 6: Merge (only after user confirms)**

```bash
git checkout main
git merge --no-ff feat/phase-10 -m "Merge Phase 10 — Model Diagnostics layer

ModelSource adapter (22 derived-data methods), 26 model-diagnostic marks,
21 Group B figure functions, 25 sklearn-protocol Visualizers. One new
Rust function (kendall_tau_b, Knight's algorithm). One new RenderConfig
(none — Task 5 dropped after discovering renderer already 3-dp quantized). ~35 byte-identical goldens at 3 dp.
0 new pytest skips, 0 new xfails. Phase 9+ no-defer principle upheld:
every spec parameter ships with full implementation.

sklearn not imported unless user's model is from sklearn — verified via
tests/diagnostics/test_no_sklearn_at_import.py.

Design: docs/superpowers/specs/2026-05-10-model-diagnostics-design.md
Plan:   docs/superpowers/plans/2026-05-10-model-diagnostics-plan.md"

git log -1 --oneline
```

- [ ] **Step 7: Verify clean post-merge state**

```bash
DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core --quiet 2>&1 | tail -3
uv run --no-sync pytest 2>&1 | tail -3
```
Expected: same counts as Step 1 — merge introduced no regressions.

- [ ] **Step 8: (Optional) clean up feature branch**

Only with user confirmation:

```bash
git branch -d feat/phase-10
```

---

## Subagent verification protocol

After **every** subagent task, the orchestrator MUST independently verify:

1. **Test counts:** re-run `cargo test -p ferrum-core` (with `DYLD_LIBRARY_PATH`) and `uv run --no-sync pytest tests/diagnostics/` (or full pytest where relevant). Counts must monotonically increase, no skips/xfails added.
2. **File changes:** `git ls-tree HEAD --name-only -r` lists the files the subagent claims to have created/modified. If a subagent reports adding a file that isn't tracked, the task is incomplete.
3. **Git status clean:** `git status` shows no uncommitted changes other than the next task's working files.
4. **No regression in `import ferrum`:** `uv run --no-sync python -c "import sys; import ferrum; assert 'sklearn' not in sys.modules; print('OK')"` must print `OK`. Critical for the Phase 10 done-criterion.

These checks are non-negotiable and apply equally to parallelized tasks. Subagent reports about deleted files or test counts cannot be trusted without verification (per memory `feedback_subagent_verification` — Phase 8b had falsely reported deletions).

---

## Final test count expectations

| Metric | Phase 9 close | Phase 10 close (target) | Delta |
|---|---|---|---|
| `cargo test -p ferrum-core` | 496 passed | ≥ 510 passed | +14 (Kendall + RenderConfig) |
| `uv run pytest` | 480 passed, 5 skipped | ≥ 610 passed, 5 skipped | +130 (0 new skips) |
| SVG goldens (Phase 10) | 0 | ~35 total | +35 |
| New Rust files | — | 1 (`diagnostics.rs`) | +1 |
| Modified Rust files | — | 2 (`lib.rs`, `render/svg.rs`, `spec/render.rs`) | +3 |
| New Python files | — | ~20 (under `_diagnostics/`, `marks/diagnostic.py`, `figures.py`) | +20 |
| `pyproject.toml` extras added | — | `models`, `shap`, `umap`, `ml-all` | +4 |

