# Phase 9 — Convenience / Figure-Level API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the figure-level convenience API as a thin Python sugar layer over grammar primitives — eight `§3.14` Group A functions (`displot`, `catplot`, `lmplot`, `residplot`, `pairplot`, `heatmap`, `clustermap`, `jointplot`), backed by all required new infrastructure (compound views, position adjustments, new transforms, new marks). Every parameter advertised in `ferrum-spec.md §3.14` is honored — no `NotImplementedError`, no warn-fallbacks.

**Architecture:**
- **9a foundation:** New compound views (`JointChart`, `RepeatChart` with `diagonal`/`corner`, `ClusterMapChart`); reshape and clustering Rust transforms (`Unpivot`, `Linkage`, `Reorder`, `Bin2D`); `Repeat` typed sentinel; Rust grid compositor (`compose_svg_grid`).
- **9b stat extensions:** New Rust transforms (`Logistic`, `Glm`, `Robust`, `LetterValue`); extensions to existing transforms (`Bin.cumulative`, `Smooth.{x_bins, x_estimator, output}`, `Robust.output`).
- **9c positions:** Four position adjustments (`Identity`, `Dodge`, `Jitter`, `Stack`) wired into eligible marks via a new `PositionAdjust` enum on `Layer` and `ChartSpec`; render pass rewrites layer batch data values after scale resolution.
- **9d marks:** `mark_segment` (primitive) + `mark_boxen` (composite); `segment` removed from `PHASE_9_PLUS_MARKS`.
- **9e figure functions:** Eight functions in `src/ferrum/figure/` package — every function returns a `Chart` or compound view whose `.spec` (or `.charts` / `.expand()`) is a fully-formed object.

**Tech Stack:** Rust 2021 (PyO3 0.28, abi3-py310, arrow 58, serde, serde_json, rand 0.8, rand_chacha 0.3, **NEW**: `kodama` 0.3 for Lance-Williams hierarchical clustering, **NEW**: `twox-hash` 1.6 for Jitter seed fallback). Python ≥3.10 (numpy, pyarrow). Fixture generators: scipy + statsmodels (dev-only, pinned in `requirements-fixtures.txt`).

**Source spec:** `docs/superpowers/specs/2026-05-10-convenience-api-design.md` (commit `63a17f2`).

**Branch:** `feat/phase-9` (created in Task 0).

---

## Pre-flight

1. **Build commands** (all must run from repo root):
   - **Rust extension build:** `source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop`
   - **`cargo test`:** `source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core`
   - **`pytest`:** `uv run --no-sync pytest`
2. **Test baselines at start of Phase 9 (verified 2026-05-10):**
   - `cargo test -p ferrum-core` → **395 passed**
   - `uv run pytest` → **298 passed, 7 skipped**
3. **Final targets at Phase 9 done:**
   - `cargo test -p ferrum-core` ≥ **530** (≈135 new tests across 12 new transforms/extensions, 4 position adjustments, 2 marks, 3 compound views, grid compositor)
   - `uv run pytest` ≥ **400** (≈100 new tests across compound views, position classes, figure functions, e2e renders)
   - 12 new SVG goldens, byte-identical across runs.
4. **Conventions (from `CLAUDE.md`):**
   - Plain feature branch, NOT a worktree.
   - **No `Co-Authored-By: Claude`** trailers on commits.
   - **No `git push`** without explicit user request.
   - **Confirm with user before merging to `main`.**
   - Sub-batches commit independently on `feat/phase-9`; each task ends with a single commit.
   - **Subagent-verify rule:** the orchestrator MUST re-run `cargo test -p ferrum-core` and `git ls-tree HEAD --name-only -r` after each subagent task to verify reported file changes and test counts are real (Phase 8b had falsely reported deletions; do not trust subagent reports).

---

## Task overview

Sub-batches and their tasks, in build order. Each sub-batch lands on `feat/phase-9` as a sequence of commits; sub-batches do NOT branch separately.

| Sub-batch | Tasks | Theme |
|---|---|---|
| **Pre-flight** | 0–3 | Branch creation; verify-before-implementing for `kodama` and `twox-hash`; `Cargo.toml` and fixture-requirements updates. |
| **9a-foundation** | 4–12 | Reshape/cluster transforms (Unpivot, Reorder, Bin2D, Linkage); Repeat sentinel; three compound views; `compose_svg_grid`. |
| **9b-stat** | 13–18 | `Bin.cumulative`; `Smooth.{x_bins,x_estimator,output}`; new transforms (`LetterValue`, `Logistic`, `Glm`, `Robust`). |
| **9c-position** | 19–23 | `PositionAdjust` enum on Layer/ChartSpec; Identity, Dodge, Jitter, Stack implementations; mark eligibility matrix. |
| **9d-marks** | 24–26 | `mark_segment` primitive; `mark_boxen` composite; `PHASE_9_PLUS_MARKS` update. |
| **9e-figures** | 27–35 | Figure package skeleton + 8 figure-level functions. |
| **Finalize** | 36–40 | 12 SVG goldens; spec drift notes; `ferrum-phases.md` update; final test pass; final commit. |

**Parallelization guidance** (for subagent-driven execution):
- Pre-flight tasks (0–3) are **strictly sequential** (branch → kodama verify → twox-hash verify → Cargo edits).
- Within 9a, Tasks 4 (Unpivot), 5 (Reorder), 6 (Bin2D) are **parallelizable** (independent transforms). Task 7 (Linkage) depends on Task 3 only. Tasks 8–11 (Python compound views + Repeat) are independent of the Rust transforms but depend on Task 12 (`compose_svg_grid`) for rendering — Python class definitions can land first, with `.show_svg()` wiring as a follow-up step inside Task 9/10/11.
- Within 9b, Tasks 13 and 14 (Bin/Smooth extensions) are parallelizable. Tasks 15–18 (new transforms) are mutually parallel but each depends on its own fixture generator.
- 9c (Tasks 19–23) is **strictly sequential** because each task extends the same `PositionAdjust` enum / Layer struct / draw pipeline.
- 9d Task 24 (`mark_segment`) and Task 25 (`mark_boxen`) are independent.
- 9e Tasks 28–35 (the 8 figure functions) are mutually parallel after Task 27 (package skeleton).

When subagents run in parallel, the orchestrator merges sequentially with `cargo test` between merges.

---

## File map

### New Rust files (`crates/ferrum-core/src/`)

| Path | Responsibility |
|---|---|
| `transform/unpivot.rs` | `UnpivotSpec` + `apply` (wide → long reshape) + `PyUnpivot` + tests. |
| `transform/reorder.rs` | `ReorderSpec` + `apply` (permutation by index column) + `PyReorder` + tests. |
| `transform/bin_2d.rs` | `Bin2DSpec` + `apply` (2D rectangular binning) + `PyBin2D` + tests. |
| `transform/linkage.rs` | `LinkageSpec`, `LinkageMethod`, `DistanceMetric`, `LinkageAxis`, `ZScoreAxis`, `StdScaleAxis` + `apply` (kodama-backed linkage matrix + hand-rolled coords/order) + 3-named-output `secondary_outputs` + `PyLinkage` + tests. |
| `transform/letter_value.rs` | `LetterValueSpec`, `KDepth` + `apply` (letter-value quantile statistics) + named `outliers` output + `PyLetterValue` + tests. |
| `transform/logistic.rs` | `LogisticSpec` + `apply` (IRLS logit + Wald CI) + `PyLogistic` + tests. |
| `transform/glm.rs` | `GlmSpec`, `GlmFamily`, `GlmLink` + `apply` (IRLS family/link IRLS + sandwich CI) + `PyGlm` + tests. |
| `transform/robust.rs` | `RobustSpec`, `SmoothOutput` (shared with smooth), Huber M-estimator + sandwich CI + `PyRobust` + tests. |
| `render/position.rs` | `apply_position` pass: `(layer_batch, scales, mark_kind) → adjusted_layer_batch` for Identity/Dodge/Jitter/Stack. |
| `render/marks/segment.rs` | `draw` for `Mark::Segment` — diagonal-capable line via `SvgBuffer::line`. |
| `render/grid_compose.rs` | `compose_svg_grid` row-major grid layout with explicit row/col ratios + spacing + share-x/share-y groups. |
| `spec/position.rs` | `PositionAdjust` enum + `JitterAxis` + `StackOffset` + JSON round-trip tests. |
| `spec/repeat.rs` | `RepeatPlaceholder` enum (`Column`, `Row`, `Layer`) + serde rename to `$repeat` + JSON round-trip tests. |

### Modified Rust files (`crates/ferrum-core/src/`)

| Path | Change |
|---|---|
| `lib.rs` | Register 8 new pyclasses (`PyUnpivot`, `PyReorder`, `PyBin2D`, `PyLinkage`, `PyLetterValue`, `PyLogistic`, `PyGlm`, `PyRobust`); register `PyJointChart`, `PyRepeatChart`, `PyClusterMapChart` (if Rust-backed; Phase 9 keeps these Python-only — see Task 9); register `compose_svg_grid` pyfunction; register `Repeat` placeholder helpers. |
| `transform/core.rs` | Add 8 new `TransformSpec` enum variants + dispatch arms in `apply`/`apply_with_context`/`spec_name`/`secondary_outputs`. |
| `transform/mod.rs` | `pub(crate) mod` declarations for 8 new transform modules. |
| `transform/bin.rs` | Add `cumulative: bool` field (`#[serde(default)]`); cumulative-output branch in `apply`; tests. |
| `transform/smooth.rs` | Add `x_bins: Option<usize>`, `x_estimator: Option<AggregateOp>`, `output: SmoothOutput` (`Fitted`/`Residuals`); pre-aggregation branch; residuals-output branch; tests. |
| `spec/mark.rs` | Add `Mark::Segment` variant + `from_str` arm + tests. |
| `spec/layer.rs` | Add `position: Option<PositionAdjust>` field. |
| `spec/chart.rs` | Add `position: Option<PositionAdjust>` field; coerce in `__new__`; PyO3 getter. |
| `spec/encoding.rs` | Honor `RepeatPlaceholder` values in `EncodingSpec.field` — extend `coerce_encoding` to accept `Repeat.column/row/layer` values; serialize as `{"$repeat": "column"}` JSON. |
| `render/mod.rs` | Insert `apply_position` pass between `scale_resolve` and `draw::dispatch_mark` in the per-layer loop. |
| `render/marks/mod.rs` | `pub(crate) mod segment;` and dispatch arm in `draw::dispatch_mark`. |
| `render/binding.rs` | Add `compose_svg_grid` PyO3 wrapper. |

### New Python files (`src/ferrum/`)

| Path | Responsibility |
|---|---|
| `repeat.py` | `Repeat` namespace with `.column`, `.row`, `.layer` typed sentinels. |
| `position.py` | `Identity`, `Dodge`, `Jitter`, `Stack` immutable value classes; mark eligibility matrix; `to_spec_dict()` JSON serialization. |
| `marks/segment.py` | `desugar_segment` helper (currently trivial; placeholder for future expansion). |
| `figure/__init__.py` | Re-export 8 figure functions. |
| `figure/distribution.py` | `displot`. |
| `figure/categorical.py` | `catplot`. |
| `figure/regression.py` | `lmplot`, `residplot`. |
| `figure/matrix.py` | `pairplot`, `heatmap`, `clustermap`. |
| `figure/joint.py` | `jointplot`. |

### Modified Python files (`src/ferrum/`)

| Path | Change |
|---|---|
| `__init__.py` | Re-export new transforms (`Unpivot`, `Reorder`, `Bin2D`, `Linkage`, `LetterValue`, `Logistic`, `Glm`, `Robust`); compound views (`JointChart`, `RepeatChart`, `ClusterMapChart`); `Repeat` namespace; position classes (`Identity`, `Dodge`, `Jitter`, `Stack`); 8 figure functions; `figure` submodule. |
| `_core.pyi` | Add stubs for 8 new transform classes; `compose_svg_grid` function; new ChartSpec/Layer fields. |
| `composition.py` | Add `JointChart`, `RepeatChart`, `ClusterMapChart` classes; thread Rust grid compositor. |
| `chart.py` | Replace `mark_segment` stub (line 571) with working method; add `mark_boxen` method; accept `position=` kwarg on eligible mark methods; resolve `Repeat` placeholders in `to_spec`. |
| `marks/composite.py` | Add `desugar_boxen` (parallel to `desugar_boxplot`). |
| `marks/deferred.py` | Remove `"segment"` from `PHASE_9_PLUS_MARKS`. |

### New Rust fixture files (`crates/ferrum-core/src/transform/fixtures/`)

| Path | Responsibility |
|---|---|
| `generate_linkage_refs.py` | scipy reference for 5 (method, metric) pairs + chebyshev + median/centroid edge cases. |
| `generate_glm_refs.py` | statsmodels reference for 5 canonical-link cases + 3 non-canonical (Gaussian+Log, Binomial+Probit, Poisson+Sqrt). |
| `generate_logistic_refs.py` | statsmodels reference for 5 logistic datasets. |
| `generate_robust_refs.py` | statsmodels reference for 4 robust-regression datasets. |
| `generate_letter_value_refs.py` | numpy quantile reference for 4 letter-value scenarios. |
| `linkage_refs.json`, `glm_refs.json`, `logistic_refs.json`, `robust_refs.json`, `letter_value_refs.json` | Generated reference data, committed alongside scripts. |

### Modified fixture files

| Path | Change |
|---|---|
| `requirements-fixtures.txt` | Pin `statsmodels>=0.14,<0.16`. |

### New test files (`tests/`)

| Path | Tests |
|---|---|
| `test_phase_9_transforms.py` | Python-side smoke tests for new transforms (round-trip, basic correctness). |
| `test_phase_9_compound_views.py` | JointChart, RepeatChart, ClusterMapChart construction, `.charts`, `.expand()`, `.theme()` propagation. |
| `test_phase_9_position.py` | Identity/Dodge/Jitter/Stack value classes; eligibility errors; JSON round-trip. |
| `test_phase_9_marks.py` | `mark_segment`, `mark_boxen` Chart methods; composite expansion of boxen. |
| `test_phase_9_figures.py` | Per-function structural tests for 8 figure-level functions; per-parameter coverage matrix. |
| `test_phase_9_e2e.py` | 12 SVG goldens; renders pass; byte-identical across runs. |
| `test_phase_9_e2e/goldens/*.svg` | 12 committed golden SVGs. |

### Modified docs

| Path | Change |
|---|---|
| `ferrum-spec.md` | Apply 6 dated drift notes (Task 37). |
| `docs/superpowers/ferrum-phases.md` | Phase 9 row `pending` → `done`; link to spec doc (Task 38). |

---

## Task list

### Task 0: Create `feat/phase-9` branch

**Files:** none (branch creation only)

- [ ] **Step 1: Verify clean working tree**

Run:

```bash
git status
```
Expected: `On branch main` and `nothing to commit, working tree clean` (after the 2 commits in `63a17f2` and `840331a`).

- [ ] **Step 2: Verify baselines**

Run, expecting the printed counts:

```bash
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
uv run --no-sync pytest 2>&1 | tail -3
```
Expected: `cargo test` → `395 passed`. `pytest` → `298 passed, 7 skipped`.

- [ ] **Step 3: Create the branch**

```bash
git checkout -b feat/phase-9
git status
```
Expected: `On branch feat/phase-9`.

- [ ] **Step 4: Confirm with user before any commits land on `main`**

This branch will be merged to main only at end of Task 40 with explicit user confirmation.

---

### Task 1: Verify `kodama` crate suitability (verify-before-implementing)

**Files:** none (research only — no code)

- [ ] **Step 1: Check `kodama` on crates.io**

Use WebFetch (or browse crates.io directly) to verify:

```
https://crates.io/crates/kodama
```

Record the following on a scratch note (will be referenced in Task 3):
- Latest version (target: 0.3.x or newer)
- Last release date (must be < 18 months from today, 2026-05-10 — so > 2024-11-10)
- License (must be MIT or Apache-2.0 or dual)
- Open-issue count and any "critical" / "panic" / "incorrect-results" labels

- [ ] **Step 2: Check the linkage-matrix API surface**

Use WebFetch on:

```
https://docs.rs/kodama
```

Verify the crate exposes:
- A function or struct that takes a condensed distance matrix `&mut [f64]` plus n (observation count) and returns a linkage matrix `Dendrogram` or equivalent.
- Support for at least: single, complete, average, weighted, ward (Lance-Williams reducible methods).
- A way to extract per-merge `(node_id_a, node_id_b, distance, n_observations)` rows.

- [ ] **Step 3: Decide path A or path B**

If all three checks pass: **path A** — use `kodama`. Proceed to Task 3.

If `kodama` fails verification: **path B** — hand-roll Lance-Williams + nearest-neighbor chain (~400-600 LOC). Update Task 7 (`transform/linkage.rs`) by replacing the `kodama` calls with an inline NN-chain implementation that supports the 7 methods (single, complete, average, weighted, ward, centroid, median). Note this in the Task 3 commit message and in the linkage.rs module-doc.

- [ ] **Step 4: Commit nothing (research only)**

This task produces no commit. The decision is recorded in scratch notes consumed by Tasks 3 and 7.

---

### Task 2: Verify `twox-hash` crate version + statsmodels pin

**Files:** none (research only — no code)

- [ ] **Step 1: Check `twox-hash` on crates.io**

Use WebFetch:

```
https://crates.io/crates/twox-hash
```

Verify:
- Latest 1.x version (target: 1.6.x or newer; 2.x exists but has different API — stick to 1.x for stable `XxHash64`).
- License (must be MIT or Apache-2.0 or dual).
- The `xxh3::hash64(&[u8]) -> u64` function exists in the API.

Record the chosen version (e.g., `1.6.3`).

- [ ] **Step 2: Check `statsmodels` pin**

Use WebFetch:

```
https://pypi.org/project/statsmodels/
```

Verify:
- Latest stable version is in the `0.14.x` series (target).
- The endpoints used by fixture generators (Task 16/17/18) exist in this version:
  - `statsmodels.discrete.discrete_model.Logit`
  - `statsmodels.genmod.generalized_linear_model.GLM`
  - `statsmodels.genmod.families.{Gaussian, Binomial, Poisson, Gamma, InverseGaussian}`
  - `statsmodels.genmod.families.links.{identity, log, logit, probit, inverse, inverse_power, sqrt}` (note: `inverse_power` is the canonical InverseGaussian link in statsmodels' newer API; older API used `inverse_squared`)
  - `statsmodels.robust.robust_linear_model.RLM` with `M=HuberT()`

Record any API name differences for the fixture-generator authors (Tasks 16–18).

- [ ] **Step 3: Commit nothing (research only)**

Decision feeds Task 3.

---

### Task 3: Add `kodama` + `twox-hash` Cargo deps; pin `statsmodels` in fixtures

**Files:**
- Modify: `Cargo.toml` (workspace root, `[workspace.dependencies]` table)
- Modify: `crates/ferrum-core/Cargo.toml` (`[dependencies]` table)
- Modify: `crates/ferrum-core/src/transform/fixtures/requirements-fixtures.txt`

- [ ] **Step 1: Edit workspace `Cargo.toml`**

Open `Cargo.toml` at repo root. After the `png = "0.18"` line in `[workspace.dependencies]`, append:

```toml
# Phase 9 (Linkage transform) — scipy-compatible Lance-Williams + nearest-neighbor
# chain. Pure Rust, single-purpose, audited per Task 1 verify-before-implementing.
# (If Task 1 selected path B, this line is omitted and linkage.rs hand-rolls.)
kodama = "0.3"
# Phase 9 (Jitter seed-fallback hash) — twox-hash 1.x (NOT 2.x, which has a
# different API surface). Used by render/position.rs to derive a deterministic
# u64 from (x, y, group) per row when the user passes seed=None.
twox-hash = { version = "1.6", default-features = false }
```

- [ ] **Step 2: Edit `crates/ferrum-core/Cargo.toml`**

In `[dependencies]`, after `png = { workspace = true }`, append:

```toml
kodama    = { workspace = true }
twox-hash = { workspace = true }
```

(If Task 1 chose path B, omit the `kodama` line in both places. The `twox-hash` line is unconditional.)

- [ ] **Step 3: Edit `requirements-fixtures.txt`**

Open `crates/ferrum-core/src/transform/fixtures/requirements-fixtures.txt`. Replace its contents with:

```
# Phase 5/9 fixture generator — used offline only; NOT a runtime dependency.
# Pin exact versions so regeneration is reproducible.
numpy==2.1.3
scipy==1.14.1     # KDE sanity check (Phase 5); Linkage reference (Phase 9 Task 7)
statsmodels==0.14.4   # Phase 9 (Logistic, Glm, Robust) reference values
```

- [ ] **Step 4: Verify `cargo build` succeeds**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -5
```
Expected: `Built wheel for abi3 Python ≥ 3.10` line. If kodama or twox-hash fails to compile, abort and adjust the version pin.

- [ ] **Step 5: Verify `cargo test` baseline still passes**

```bash
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
```
Expected: `395 passed`.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/ferrum-core/Cargo.toml \
        crates/ferrum-core/src/transform/fixtures/requirements-fixtures.txt
git commit -m "chore(phase-9): add kodama + twox-hash deps; pin statsmodels for fixtures"
```

---

## 9a — Convenience layer foundation

### Task 4: `Unpivot` transform (Rust)

**Files:**
- Create: `crates/ferrum-core/src/transform/unpivot.rs`
- Modify: `crates/ferrum-core/src/transform/mod.rs` (add `pub(crate) mod unpivot;`)
- Modify: `crates/ferrum-core/src/transform/core.rs` (add `Unpivot(UnpivotSpec)` variant + dispatch + `spec_name`)
- Modify: `crates/ferrum-core/src/lib.rs` (register `PyUnpivot`)
- Modify: `src/ferrum/__init__.py` (re-export `Unpivot`)
- Modify: `src/ferrum/_core.pyi` (add `Unpivot` stub)

- [ ] **Step 1: Write the failing JSON round-trip test**

Append to `crates/ferrum-core/src/transform/core.rs` test module:

```rust
#[test]
fn test_transform_spec_unpivot_round_trip() {
    use crate::transform::unpivot::UnpivotSpec;
    let original = TransformSpec::Unpivot(UnpivotSpec {
        id_vars: vec!["row_id".into()],
        value_vars: Some(vec!["a".into(), "b".into()]),
        var_name: "variable".into(),
        value_name: "value".into(),
        name: None,
    });
    let json = serde_json::to_string(&original).unwrap();
    assert!(json.contains(r#""type":"unpivot""#), "missing tag: {json}");
    let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, original);
}
```

- [ ] **Step 2: Run — confirm it fails to compile (no `unpivot` module yet)**

```bash
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core test_transform_spec_unpivot_round_trip 2>&1 | tail -10
```
Expected: error mentioning `unresolved import crate::transform::unpivot::UnpivotSpec`.

- [ ] **Step 3: Create `transform/unpivot.rs`**

```rust
//! Unpivot transform — wide → long reshape.
//!
//! Output schema:
//!   [id_vars..., var_name: Utf8, value_name: <unified-dtype>]
//!
//! Dtype rule (homogeneous-or-numeric):
//!   - All value columns must share a dtype, OR all be numeric.
//!   - Numeric mixed types widen to the widest (Int32+Float64 → Float64).
//!   - Mixed non-numeric types → error.
//!
//! Used by `heatmap()` (wide-matrix input) and `clustermap()` reshape.

use arrow::array::{Array, ArrayRef, RecordBatch, StringArray, StringBuilder};
use arrow::compute::{cast, concat};
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct UnpivotSpec {
    #[serde(default)]
    pub id_vars: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub value_vars: Option<Vec<String>>,
    #[serde(default = "default_var_name")]
    pub var_name: String,
    #[serde(default = "default_value_name")]
    pub value_name: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_var_name() -> String { "variable".into() }
fn default_value_name() -> String { "value".into() }

pub(crate) fn apply(spec: &UnpivotSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();

    // Resolve value_vars: either explicit, or all non-id columns.
    let value_var_names: Vec<String> = match &spec.value_vars {
        Some(v) => v.clone(),
        None => schema.fields().iter()
            .map(|f| f.name().to_string())
            .filter(|n| !spec.id_vars.contains(n))
            .collect(),
    };

    if value_var_names.is_empty() {
        return Err(PyValueError::new_err(
            "stat_unpivot: no value_vars to melt (id_vars covers all columns)"
        ));
    }

    // Validate dtypes: must be homogeneous OR all-numeric.
    let value_dtypes: Vec<&DataType> = value_var_names.iter()
        .map(|n| {
            let i = schema.index_of(n).map_err(|_| PyValueError::new_err(
                format!("stat_unpivot: column '{n}' not found")
            ))?;
            Ok(schema.field(i).data_type())
        })
        .collect::<PyResult<_>>()?;

    let unified_dtype = unify_value_dtype(&value_dtypes)?;

    // Cast each value column to the unified dtype.
    let value_columns_cast: Vec<ArrayRef> = value_var_names.iter()
        .map(|n| {
            let i = schema.index_of(n).unwrap();
            cast(&batch.column(i), &unified_dtype)
                .map_err(|e| PyValueError::new_err(format!("stat_unpivot: cast '{n}': {e}")))
        })
        .collect::<PyResult<_>>()?;

    // Stack value columns vertically.
    let stacked_value: ArrayRef = {
        let refs: Vec<&dyn Array> = value_columns_cast.iter().map(|a| a.as_ref()).collect();
        concat(&refs).map_err(|e| PyValueError::new_err(format!("stat_unpivot: concat: {e}")))?
    };

    // Build var_name column (Utf8): repeat each name n_rows times in row-major order.
    let mut var_builder = StringBuilder::with_capacity(
        n_rows * value_var_names.len(),
        n_rows * value_var_names.len() * 8,
    );
    for name in &value_var_names {
        for _ in 0..n_rows {
            var_builder.append_value(name);
        }
    }
    let var_arr: ArrayRef = Arc::new(var_builder.finish());

    // Build id columns: take indices [0..n_rows] cycled per value_var.
    let id_columns_replicated: Vec<ArrayRef> = spec.id_vars.iter()
        .map(|n| {
            let i = schema.index_of(n).map_err(|_| PyValueError::new_err(
                format!("stat_unpivot: id_var '{n}' not found")
            ))?;
            // Concat the original id-column with itself k times where k = value_vars.len()
            let one = batch.column(i);
            let repeats: Vec<&dyn Array> = (0..value_var_names.len())
                .map(|_| one.as_ref()).collect();
            concat(&repeats).map_err(|e| PyValueError::new_err(format!("stat_unpivot: id-replicate: {e}")))
        })
        .collect::<PyResult<_>>()?;

    // Assemble output schema: id_vars... + var_name + value_name.
    let mut fields: Vec<Field> = spec.id_vars.iter().map(|n| {
        let i = schema.index_of(n).unwrap();
        let f = schema.field(i);
        Field::new(f.name(), f.data_type().clone(), f.is_nullable())
    }).collect();
    fields.push(Field::new(&spec.var_name, DataType::Utf8, false));
    fields.push(Field::new(&spec.value_name, unified_dtype, true));
    let out_schema = Arc::new(Schema::new(fields));

    let mut cols = id_columns_replicated;
    cols.push(var_arr);
    cols.push(stacked_value);
    RecordBatch::try_new(out_schema, cols)
        .map_err(|e| PyValueError::new_err(format!("stat_unpivot: {e}")))
}

fn unify_value_dtype(dtypes: &[&DataType]) -> PyResult<DataType> {
    if dtypes.is_empty() {
        return Err(PyValueError::new_err("stat_unpivot: no value columns"));
    }
    // Homogeneous fast path.
    if dtypes.iter().all(|d| *d == dtypes[0]) {
        return Ok(dtypes[0].clone());
    }
    // Mixed: must be all-numeric to widen.
    let all_numeric = dtypes.iter().all(|d| is_numeric(d));
    if !all_numeric {
        let names: Vec<String> = dtypes.iter().map(|d| format!("{d}")).collect();
        return Err(PyValueError::new_err(format!(
            "stat_unpivot: value_vars have heterogeneous non-numeric types: [{}]; \
             cast to a common type before unpivot", names.join(", ")
        )));
    }
    // Widen to Float64 if any float; else widest int. Phase 9 keeps this simple:
    // any mixed-numeric → Float64 (covers all observed cases for heatmap/clustermap).
    Ok(DataType::Float64)
}

fn is_numeric(d: &DataType) -> bool {
    matches!(d,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
        | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
        | DataType::Float32 | DataType::Float64)
}

// ---------- PyO3 wrapper ----------

use crate::transform::core::TransformSpec;

#[pyclass(eq, module = "ferrum._core", name = "Unpivot")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyUnpivot(pub(crate) TransformSpec);

#[pymethods]
impl PyUnpivot {
    #[new]
    #[pyo3(signature = (
        *,
        id_vars = Vec::<String>::new(),
        value_vars = None,
        var_name = "variable",
        value_name = "value",
        name = None,
    ))]
    fn new(
        id_vars: Vec<String>,
        value_vars: Option<Vec<String>>,
        var_name: &str,
        value_name: &str,
        name: Option<String>,
    ) -> PyResult<Self> {
        if var_name.is_empty() || value_name.is_empty() {
            return Err(PyValueError::new_err("Unpivot: var_name and value_name must be non-empty"));
        }
        Ok(PyUnpivot(TransformSpec::Unpivot(UnpivotSpec {
            id_vars, value_vars, var_name: var_name.into(), value_name: value_name.into(), name,
        })))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Unpivot(s) => format!(
                "Unpivot(id_vars={:?}, value_vars={:?}, var_name='{}', value_name='{}')",
                s.id_vars, s.value_vars, s.var_name, s.value_name,
            ),
            #[allow(unreachable_patterns)] _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array};

    fn batch_3x4() -> RecordBatch {
        // 3 rows × 4 numeric value columns
        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::Int32, false),
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
            Field::new("c", DataType::Float64, false),
            Field::new("d", DataType::Float64, false),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Int32Array::from(vec![10, 20, 30])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![7.0, 8.0, 9.0])),
            Arc::new(Float64Array::from(vec![10.0, 11.0, 12.0])),
        ]).unwrap()
    }

    #[test]
    fn unpivot_3x4_numeric_correctness() {
        let batch = batch_3x4();
        let spec = UnpivotSpec {
            id_vars: vec!["row_id".into()],
            value_vars: None,
            var_name: "variable".into(),
            value_name: "value".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_rows(), 12);  // 3 rows × 4 value cols
        assert_eq!(out.num_columns(), 3); // row_id, variable, value
        let vars = out.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let vals = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        // First 3 rows should be variable="a", value=1,2,3 (the column order).
        assert_eq!(vars.value(0), "a"); assert_eq!(vals.value(0), 1.0);
        assert_eq!(vars.value(1), "a"); assert_eq!(vals.value(1), 2.0);
        assert_eq!(vars.value(3), "b"); assert_eq!(vals.value(3), 4.0);
        assert_eq!(vars.value(11), "d"); assert_eq!(vals.value(11), 12.0);
    }

    #[test]
    fn unpivot_widens_int_and_float_to_float64() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Float64Array::from(vec![3.5, 4.5])),
        ]).unwrap();
        let spec = UnpivotSpec {
            id_vars: vec![],
            value_vars: None,
            var_name: "k".into(),
            value_name: "v".into(),
            name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.schema().field(1).data_type(), &DataType::Float64);
        let vals = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(vals.value(0), 1.0);
        assert_eq!(vals.value(2), 3.5);
    }

    #[test]
    fn unpivot_homogeneous_utf8_works() {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["x", "y"])),
            Arc::new(StringArray::from(vec!["p", "q"])),
        ]).unwrap();
        let spec = UnpivotSpec {
            id_vars: vec![], value_vars: None,
            var_name: "k".into(), value_name: "v".into(), name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.schema().field(1).data_type(), &DataType::Utf8);
    }

    #[test]
    fn unpivot_mixed_int_and_utf8_errors() {
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["x"])),
        ]).unwrap();
        let spec = UnpivotSpec {
            id_vars: vec![], value_vars: None,
            var_name: "k".into(), value_name: "v".into(), name: None,
        };
        let err = apply(&spec, &batch).unwrap_err().to_string();
        assert!(err.contains("heterogeneous non-numeric"), "got: {err}");
    }

    #[test]
    fn unpivot_preserves_id_dtypes() {
        let batch = batch_3x4();
        let spec = UnpivotSpec {
            id_vars: vec!["row_id".into()],
            value_vars: None,
            var_name: "k".into(), value_name: "v".into(), name: None,
        };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.schema().field(0).data_type(), &DataType::Int32);
    }
}
```

- [ ] **Step 4: Wire `unpivot` into the transform module tree**

Edit `crates/ferrum-core/src/transform/mod.rs`. After `pub(crate) mod swarm;`, append:

```rust
pub(crate) mod unpivot;
```

Edit `crates/ferrum-core/src/transform/core.rs`:

1. Add to imports near top:
   ```rust
   use crate::transform::unpivot::{self, UnpivotSpec};
   ```
2. Inside `pub(crate) enum TransformSpec`, after `Swarm(SwarmSpec),` add:
   ```rust
   Unpivot(UnpivotSpec),
   ```
3. Inside `impl TransformSpec::apply`, after `Self::Swarm(s) => swarm::apply(s, batch),` add:
   ```rust
   Self::Unpivot(s) => unpivot::apply(s, batch),
   ```
4. Inside `spec_name`, after `TransformSpec::Swarm(s) => s.name.as_deref(),` add:
   ```rust
   TransformSpec::Unpivot(s) => s.name.as_deref(),
   ```

- [ ] **Step 5: Register `PyUnpivot` in `lib.rs`**

Edit `crates/ferrum-core/src/lib.rs`. After `m.add_class::<transform::swarm::PySwarm>()?;` add:

```rust
m.add_class::<transform::unpivot::PyUnpivot>()?;
```

- [ ] **Step 6: Re-export from Python**

Edit `src/ferrum/__init__.py`. Add `Unpivot` to the existing `from ferrum._core import (...)` block (alphabetically between `ThresholdScale` and `Violin` maintaining the existing ordering convention). Add `Unpivot` to `__all__`.

- [ ] **Step 7: Stub for type checker**

Edit `src/ferrum/_core.pyi`. Add (mirroring existing transform stubs like `Bin`):

```python
class Unpivot:
    def __init__(
        self,
        *,
        id_vars: list[str] = ...,
        value_vars: list[str] | None = None,
        var_name: str = "variable",
        value_name: str = "value",
        name: str | None = None,
    ) -> None: ...
```

- [ ] **Step 8: Build + run tests**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core unpivot 2>&1 | tail -10
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core test_transform_spec_unpivot_round_trip 2>&1 | tail -5
```
Expected: all unpivot tests pass + the core round-trip test passes.

- [ ] **Step 9: Verify total cargo test count**

```bash
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
```
Expected: `401 passed` (395 baseline + 5 unpivot tests + 1 core round-trip).

- [ ] **Step 10: Commit**

```bash
git add crates/ferrum-core/src/transform/unpivot.rs \
        crates/ferrum-core/src/transform/{mod,core}.rs \
        crates/ferrum-core/src/lib.rs \
        src/ferrum/__init__.py src/ferrum/_core.pyi
git commit -m "feat(phase-9a): add Unpivot transform (wide → long reshape)"
```

---

### Task 5: `Reorder` transform (Rust)

**Files:**
- Create: `crates/ferrum-core/src/transform/reorder.rs`
- Modify: `transform/mod.rs`, `transform/core.rs`, `lib.rs`, `src/ferrum/__init__.py`, `src/ferrum/_core.pyi`

**Spec:**
```rust
pub(crate) struct ReorderSpec {
    pub by: String,                         // index column (Int64)
    pub drop_index: bool,                   // default true — drop the index column from output
    pub name: Option<String>,
}
```

**Output:** input batch permuted so that `output.row[i] == input.row[by_column[i]]`. If `drop_index=true`, the `by` column is removed from output schema.

- [ ] **Step 1: Write JSON round-trip test** (in `transform/core.rs` test module, mirroring Task 4 Step 1)

```rust
#[test]
fn test_transform_spec_reorder_round_trip() {
    use crate::transform::reorder::ReorderSpec;
    let original = TransformSpec::Reorder(ReorderSpec {
        by: "new_idx".into(), drop_index: true, name: None,
    });
    let json = serde_json::to_string(&original).unwrap();
    assert!(json.contains(r#""type":"reorder""#));
    let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, original);
}
```

- [ ] **Step 2: Create `transform/reorder.rs`** following the Unpivot module shape (Task 4 Step 3):

```rust
//! Reorder transform — apply a permutation to the input batch.
//!
//! `by` names an Int64 index column where output.row[i] == input.row[by[i]].
//! When `drop_index=true` (default), the `by` column is omitted from output.
//!
//! Used by `clustermap()` to reorder rows/columns by Linkage's `order` named output.

use arrow::array::{Array, ArrayRef, Int64Array, RecordBatch, UInt64Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, Schema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReorderSpec {
    pub by: String,
    #[serde(default = "default_drop_index")]
    pub drop_index: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

fn default_drop_index() -> bool { true }

pub(crate) fn apply(spec: &ReorderSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let idx = schema.index_of(&spec.by).map_err(|_| PyValueError::new_err(
        format!("stat_reorder: index column '{}' not found", spec.by)))?;
    if schema.field(idx).data_type() != &DataType::Int64 {
        return Err(PyValueError::new_err(format!(
            "stat_reorder: '{}' must be Int64 (got {:?})",
            spec.by, schema.field(idx).data_type())));
    }
    let idx_arr = batch.column(idx).as_any().downcast_ref::<Int64Array>().unwrap();
    // arrow::compute::take requires UInt64 indices.
    let n = idx_arr.len();
    let take_indices: UInt64Array = (0..n)
        .map(|i| {
            let v = idx_arr.value(i);
            if v < 0 || (v as usize) >= batch.num_rows() {
                return Err(PyValueError::new_err(format!(
                    "stat_reorder: index {v} out of bounds (n={})", batch.num_rows())));
            }
            Ok(v as u64)
        })
        .collect::<PyResult<Vec<u64>>>()?
        .into();

    let mut out_cols: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    let mut out_fields: Vec<Field> = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        if spec.drop_index && i == idx {
            continue;
        }
        let permuted = take(&batch.column(i), &take_indices, None)
            .map_err(|e| PyValueError::new_err(format!("stat_reorder: take: {e}")))?;
        out_cols.push(permuted);
        out_fields.push(field.as_ref().clone());
    }
    let out_schema = Arc::new(Schema::new(out_fields));
    RecordBatch::try_new(out_schema, out_cols)
        .map_err(|e| PyValueError::new_err(format!("stat_reorder: {e}")))
}

// PyO3 wrapper — mirror Unpivot pattern (Task 4 Step 3).
use crate::transform::core::TransformSpec;

#[pyclass(eq, module = "ferrum._core", name = "Reorder")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyReorder(pub(crate) TransformSpec);

#[pymethods]
impl PyReorder {
    #[new]
    #[pyo3(signature = (by, *, drop_index = true, name = None))]
    fn new(by: &str, drop_index: bool, name: Option<String>) -> PyResult<Self> {
        if by.is_empty() {
            return Err(PyValueError::new_err("Reorder: by must be non-empty"));
        }
        Ok(PyReorder(TransformSpec::Reorder(ReorderSpec {
            by: by.into(), drop_index, name,
        })))
    }
    fn __repr__(&self) -> String {
        match &self.0 {
            TransformSpec::Reorder(s) => format!("Reorder(by='{}', drop_index={})", s.by, s.drop_index),
            #[allow(unreachable_patterns)] _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};

    fn make_batch_with_idx(idx: Vec<i64>, vals: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("new_idx", DataType::Int64, false),
            Field::new("v", DataType::Float64, false),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from(idx)),
            Arc::new(Float64Array::from(vals)),
        ]).unwrap()
    }

    #[test]
    fn reorder_5_rows_correctness() {
        // Reorder by [4,3,2,1,0] — reverses the data column.
        let batch = make_batch_with_idx(vec![4, 3, 2, 1, 0], vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let spec = ReorderSpec { by: "new_idx".into(), drop_index: true, name: None };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 1);
        let v = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((0..5).map(|i| v.value(i)).collect::<Vec<_>>(),
                   vec![50.0, 40.0, 30.0, 20.0, 10.0]);
    }

    #[test]
    fn reorder_identity_permutation_unchanged() {
        let batch = make_batch_with_idx(vec![0, 1, 2, 3, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let spec = ReorderSpec { by: "new_idx".into(), drop_index: true, name: None };
        let out = apply(&spec, &batch).unwrap();
        let v = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((0..5).map(|i| v.value(i)).collect::<Vec<_>>(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn reorder_keeps_index_when_drop_false() {
        let batch = make_batch_with_idx(vec![1, 0], vec![10.0, 20.0]);
        let spec = ReorderSpec { by: "new_idx".into(), drop_index: false, name: None };
        let out = apply(&spec, &batch).unwrap();
        assert_eq!(out.num_columns(), 2);
    }

    #[test]
    fn reorder_out_of_bounds_errors() {
        let batch = make_batch_with_idx(vec![0, 99], vec![1.0, 2.0]);
        let spec = ReorderSpec { by: "new_idx".into(), drop_index: true, name: None };
        let err = apply(&spec, &batch).unwrap_err().to_string();
        assert!(err.contains("out of bounds"));
    }
}
```

- [ ] **Step 3: Wire into `transform/mod.rs`, `transform/core.rs`, `lib.rs`, `__init__.py`, `_core.pyi`**

Same pattern as Task 4 Steps 4–7. Add `pub(crate) mod reorder;`, `Reorder(ReorderSpec)` enum variant + dispatch + spec_name, `m.add_class::<transform::reorder::PyReorder>()?;`, `Reorder` re-export, stub:

```python
class Reorder:
    def __init__(self, by: str, *, drop_index: bool = True, name: str | None = None) -> None: ...
```

- [ ] **Step 4: Build + test + commit**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
```
Expected: `406 passed` (401 + 4 reorder + 1 round-trip).

```bash
git add crates/ferrum-core/src/transform/reorder.rs \
        crates/ferrum-core/src/transform/{mod,core}.rs \
        crates/ferrum-core/src/lib.rs \
        src/ferrum/__init__.py src/ferrum/_core.pyi
git commit -m "feat(phase-9a): add Reorder transform (permutation by index column)"
```

---

### Task 6: `Bin2D` transform (Rust)

**Files:**
- Create: `crates/ferrum-core/src/transform/bin_2d.rs`
- Modify: `transform/mod.rs`, `transform/core.rs`, `lib.rs`, `__init__.py`, `_core.pyi`

**Spec:**
```rust
pub(crate) struct Bin2DSpec {
    pub x: String,
    pub y: String,
    pub bins_x: BinSpec2DAxis,    // separate axis spec (Sturges|FreedmanDiaconis|Fixed(n)|Width(f64))
    pub bins_y: BinSpec2DAxis,
    pub extent_x: Option<(f64, f64)>,
    pub extent_y: Option<(f64, f64)>,
    pub cumulative: bool,         // when true, count is cumulative-2D (sweep up-and-right)
    pub name: Option<String>,
}

pub(crate) enum BinSpec2DAxis {
    Sturges,
    FreedmanDiaconis,
    Fixed(usize),
    Width(f64),
}
```

**Output schema:** `[x_lo: Float64, x_hi: Float64, y_lo: Float64, y_hi: Float64, count: Int64]`. One row per non-empty cell when `cumulative=false`; when `cumulative=true`, every cell in the rectangular grid (because cumulative output is dense).

- [ ] **Step 1: Round-trip test in `transform/core.rs`**

```rust
#[test]
fn test_transform_spec_bin_2d_round_trip() {
    use crate::transform::bin_2d::{Bin2DSpec, BinSpec2DAxis};
    let original = TransformSpec::Bin2D(Bin2DSpec {
        x: "x".into(), y: "y".into(),
        bins_x: BinSpec2DAxis::Fixed(10),
        bins_y: BinSpec2DAxis::Sturges,
        extent_x: None, extent_y: None,
        cumulative: false, name: None,
    });
    let json = serde_json::to_string(&original).unwrap();
    assert!(json.contains(r#""type":"bin_2d""#));
    let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, original);
}
```

- [ ] **Step 2: Create `transform/bin_2d.rs`**

Algorithm sketch (~150 LOC):
- Extract `xs, ys` Float64 columns; drop null/NaN pairs.
- Compute `bin_count_x` and `bin_count_y` using the existing `crate::scale::ticks::sturges_floor` for Sturges (mirror Phase 5 `bin.rs` line 8 import) and Freedman-Diaconis formula for FD.
- Compute `bin_width_x = (x_max - x_min) / bin_count_x`; same for y.
- Allocate a `Vec<i64>` of size `bin_count_x * bin_count_y` initialized to 0.
- Loop over rows: compute `(ix, iy) = (((x - x_min) / w_x).floor(), ((y - y_min) / w_y).floor())`; clamp to `[0, count-1]` (right edge); increment cell.
- For `cumulative=false`: emit only non-empty cells as rows.
- For `cumulative=true`: do a 2D inclusive scan (each cell += left + below − below-left), then emit all cells.

Use the existing `crate::scale::ticks::sturges_floor` helper. Numeric edge cases follow `transform/bin.rs`: empty input → empty output; `lo == hi` → single unit bin per axis.

The full implementation mirrors `transform/raster.rs` (already does 2D bin aggregation) but emits per-cell rows with `(x_lo, x_hi, y_lo, y_hi, count)` instead of an RGBA pixel buffer. Refer to `raster.rs` for cell-index math; refer to `bin.rs` for the `Sturges` / Freedman-Diaconis formulas.

Tests:
1. Round-trip JSON.
2. **3×3 grid correctness** — input 9 evenly-spaced points → 9 cells with count=1 each.
3. **Sturges floor honored** on each axis — 100-row gaussian input with `Sturges` → `bin_count_x` matches `sturges_floor(100)`.
4. **Cumulative monotonicity** — last cell has total count = N; all neighbors ≤.
5. **Extent-clamped** — explicit `extent_x=(0,1)` discards rows outside.

- [ ] **Step 3: PyO3 wrapper** (mirror Unpivot)

```rust
#[pyclass(eq, module = "ferrum._core", name = "Bin2D")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyBin2D(pub(crate) TransformSpec);

#[pymethods]
impl PyBin2D {
    #[new]
    #[pyo3(signature = (
        x, y, *,
        bins_x = "sturges", bins_y = "sturges",
        extent_x = None, extent_y = None,
        cumulative = false, name = None,
    ))]
    fn new(
        x: &str, y: &str,
        bins_x: &Bound<'_, PyAny>, bins_y: &Bound<'_, PyAny>,
        extent_x: Option<(f64, f64)>, extent_y: Option<(f64, f64)>,
        cumulative: bool, name: Option<String>,
    ) -> PyResult<Self> {
        // Accept "sturges" | "fd" | int (Fixed) | float (Width).
        let bins_x = parse_bin_axis(bins_x)?;
        let bins_y = parse_bin_axis(bins_y)?;
        Ok(PyBin2D(TransformSpec::Bin2D(Bin2DSpec {
            x: x.into(), y: y.into(),
            bins_x, bins_y, extent_x, extent_y, cumulative, name,
        })))
    }
}

fn parse_bin_axis(obj: &Bound<'_, PyAny>) -> PyResult<BinSpec2DAxis> {
    if let Ok(s) = obj.extract::<&str>() {
        return match s {
            "sturges" => Ok(BinSpec2DAxis::Sturges),
            "fd" | "freedman_diaconis" => Ok(BinSpec2DAxis::FreedmanDiaconis),
            _ => Err(PyValueError::new_err(format!("Bin2D: unknown bins value '{s}'; expected 'sturges'|'fd'|int|float"))),
        };
    }
    if let Ok(n) = obj.extract::<usize>() {
        return Ok(BinSpec2DAxis::Fixed(n));
    }
    if let Ok(w) = obj.extract::<f64>() {
        return Ok(BinSpec2DAxis::Width(w));
    }
    Err(PyValueError::new_err("Bin2D: bins must be 'sturges'|'fd'|int|float"))
}
```

- [ ] **Step 4: Wire into mod.rs, core.rs, lib.rs, __init__.py, _core.pyi**

Same pattern as Tasks 4-5. Stub:
```python
class Bin2D:
    def __init__(self, x: str, y: str, *, bins_x = "sturges", bins_y = "sturges",
                 extent_x: tuple[float, float] | None = None,
                 extent_y: tuple[float, float] | None = None,
                 cumulative: bool = False, name: str | None = None) -> None: ...
```

- [ ] **Step 5: Build + test + commit**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
```
Expected: `412 passed` (406 + 5 bin_2d + 1 round-trip).

```bash
git add crates/ferrum-core/src/transform/bin_2d.rs \
        crates/ferrum-core/src/transform/{mod,core}.rs \
        crates/ferrum-core/src/lib.rs \
        src/ferrum/__init__.py src/ferrum/_core.pyi
git commit -m "feat(phase-9a): add Bin2D transform (2D rectangular binning)"
```

---

### Task 7: `Linkage` transform (Rust, kodama-backed)

**Files:**
- Create: `crates/ferrum-core/src/transform/linkage.rs`
- Create: `crates/ferrum-core/src/transform/fixtures/generate_linkage_refs.py`
- Create: `crates/ferrum-core/src/transform/fixtures/linkage_refs.json` (generated, committed)
- Modify: `transform/mod.rs`, `transform/core.rs` (incl. `secondary_outputs` arm), `lib.rs`, `__init__.py`, `_core.pyi`

**Spec (matches design doc §4.2):**
```rust
pub(crate) struct LinkageSpec {
    pub method: LinkageMethod,
    pub metric: DistanceMetric,
    pub axis: LinkageAxis,
    pub z_score: Option<ZScoreAxis>,
    pub standard_scale: Option<StdScaleAxis>,
    pub name: Option<String>,
}

#[serde(rename_all = "snake_case")]
pub(crate) enum LinkageMethod { Single, Complete, Average, Weighted, Centroid, Median, Ward }
#[serde(rename_all = "snake_case")]
pub(crate) enum DistanceMetric { Euclidean, Manhattan, Cosine, Correlation, Chebyshev }
#[serde(rename_all = "snake_case")]
pub(crate) enum LinkageAxis { Rows, Columns }
#[serde(rename_all = "snake_case")]
pub(crate) enum ZScoreAxis { Rows, Columns }
#[serde(rename_all = "snake_case")]
pub(crate) enum StdScaleAxis { Rows, Columns }
```

**Four named outputs (via `secondary_outputs`):**
- Primary (FINAL_OUTPUT_KEY): same as `linkage` below — keeps FINAL routable for unnamed-pipeline use.
- `linkage`: `[node_id: Int64, left: Int64, right: Int64, distance: Float64, n_obs: Int64]` — n−1 rows.
- `order`: `[original_idx: Int64, new_idx: Int64]` — n rows. `original_idx[i]` = the original-data row index that becomes position `i` in the reordered output.
- `coords`: `[node_id: Int64, x: Float64, y: Float64]` — 2n−1 rows (n leaves + n−1 internal nodes).
- **`segments`** (NEW): `[x: Float64, y: Float64, x2: Float64, y2: Float64]` — `3 * (n - 1)` rows. Each merge in the linkage matrix produces three line segments (left vertical, top horizontal, right vertical) forming the upside-down-U dendrogram glyph. Directly consumable by `mark_segment` — `clustermap` Task 34 wires this output to a `mark_segment` layer with no Python-side tree walking required.

For each internal-node merge with merge distance `d_node` and children at coords `(x_left, y_left)` and `(x_right, y_right)`, emit three rows:
1. `(x_left, y_left, x_left, d_node)` — left vertical riser
2. `(x_left, d_node, x_right, d_node)` — top horizontal cap
3. `(x_right, d_node, x_right, y_right)` — right vertical riser

The convention: when the spec has `name = Some(s)`, the primary output goes to key `s` and the four named outputs go to `s_linkage`, `s_order`, `s_coords`, `s_segments` to disambiguate multiple Linkage calls in one pipeline (clustermap has TWO — one for rows, one for columns). When `name = None`, named outputs go to bare `linkage`/`order`/`coords`/`segments`.

- [ ] **Step 1: Round-trip test in `transform/core.rs`**

```rust
#[test]
fn test_transform_spec_linkage_round_trip() {
    use crate::transform::linkage::{LinkageSpec, LinkageMethod, DistanceMetric, LinkageAxis};
    let original = TransformSpec::Linkage(LinkageSpec {
        method: LinkageMethod::Ward, metric: DistanceMetric::Euclidean,
        axis: LinkageAxis::Rows, z_score: None, standard_scale: None, name: None,
    });
    let json = serde_json::to_string(&original).unwrap();
    assert!(json.contains(r#""type":"linkage""#));
    let parsed: TransformSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, original);
}
```

- [ ] **Step 2: Generate scipy reference fixtures**

Create `crates/ferrum-core/src/transform/fixtures/generate_linkage_refs.py`:

```python
"""Phase 9 Linkage transform fixture generator.

Computes scipy reference linkage matrices for 5 curated (method, metric)
pairs + chebyshev metric + median/centroid edge cases. Used by
crates/ferrum-core/src/transform/linkage.rs cargo tests via include_str!.

Usage (from repo root):
    uv pip install -r crates/ferrum-core/src/transform/fixtures/requirements-fixtures.txt
    uv run python crates/ferrum-core/src/transform/fixtures/generate_linkage_refs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd


# Deterministic 10x4 numeric input matrix (10 observations of 4 features).
np.random.seed(0)
DATA = np.random.normal(size=(10, 4)).round(4)

CASES = [
    # (case_name, method, metric)
    ("ward_euclidean",       "ward",     "euclidean"),
    ("complete_euclidean",   "complete", "euclidean"),
    ("average_correlation",  "average",  "correlation"),
    ("single_manhattan",     "single",   "cityblock"),   # scipy uses 'cityblock' for manhattan
    ("complete_cosine",      "complete", "cosine"),
    # Edge cases: chebyshev metric; median/centroid methods.
    ("complete_chebyshev",   "complete", "chebyshev"),
    ("median_euclidean",     "median",   "euclidean"),
    ("centroid_euclidean",   "centroid", "euclidean"),
]


def case_payload(name, method, metric):
    # scipy linkage output: (n-1, 4) — [left_id, right_id, distance, n_obs].
    if method == "ward":
        # Ward in scipy requires euclidean distance from raw data (not from a precomputed matrix).
        Z = sch.linkage(DATA, method="ward")
    else:
        condensed = ssd.pdist(DATA, metric=metric)
        Z = sch.linkage(condensed, method=method)
    # Leaf order from dendrogram traversal (scipy's leaves_list).
    leaves = sch.leaves_list(Z).tolist()
    return {
        "name": name,
        "method": method,
        "metric": metric,
        "n": int(DATA.shape[0]),
        "linkage": Z.tolist(),    # n-1 rows × 4 columns
        "order": leaves,           # length-n permutation of 0..n
    }


def main():
    payload = {
        "_pinned_versions": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "data": DATA.tolist(),
        "cases": [case_payload(*c) for c in CASES],
    }
    out = Path(__file__).resolve().parent / "linkage_refs.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
```

Run it once:

```bash
uv pip install -r crates/ferrum-core/src/transform/fixtures/requirements-fixtures.txt
uv run python crates/ferrum-core/src/transform/fixtures/generate_linkage_refs.py
```

Commit `linkage_refs.json` alongside the script.

- [ ] **Step 3: Create `transform/linkage.rs`**

Algorithm sketch (~400 LOC including tests):

1. **Extract input matrix** from RecordBatch:
   - All non-id Float64 columns are features.
   - Rows of the matrix = `axis=Rows` direction; for `axis=Columns` we transpose.
2. **Pre-process**:
   - If `z_score=Some(ax)`: subtract mean / divide by std along ax.
   - If `standard_scale=Some(ax)`: subtract min / divide by range along ax.
3. **Distance computation**: produce condensed `&mut [f64]` of length `n*(n-1)/2` for the chosen metric.
4. **Linkage computation**:
   - **Path A (kodama, default per Task 1):** call `kodama::linkage(&mut condensed, n, kodama::Method::Ward)` (or matching enum). The crate returns a `Dendrogram` whose `steps()` iterator yields `(cluster_a, cluster_b, dissimilarity, size)` per merge. Convert to our `linkage` named output with `node_id = n + step_index`, `left = cluster_a as i64`, `right = cluster_b as i64`, `distance = dissimilarity`, `n_obs = size as i64`.
   - **Path B (hand-roll, fallback):** implement nearest-neighbor chain (NN-chain) for reducible methods (single, complete, average, weighted, ward) and naive Lance-Williams for centroid/median. Lance-Williams update formula:
     ```
     d(C_uv, C_w) = α_u·d(C_u, C_w) + α_v·d(C_v, C_w) + β·d(C_u, C_v) + γ·|d(C_u, C_w) - d(C_v, C_w)|
     ```
     with method-specific (α_u, α_v, β, γ) coefficients. See spec doc §4.2 references.
5. **Coords output**: dendrogram x-coordinates are leaf positions [0..n) for leaves, and for internal node = mean of subtree leaf x-positions; y = merge distance for internal nodes, 0 for leaves. Walk the linkage tree bottom-up.
6. **Order output**: `leaves_list` traversal — recursive DFS of the linkage tree; `original_idx[i]` is the original-row index of the i-th leaf.
7. **Segments output**: for each internal-node merge from `linkage`, look up `(x_left, y_left)` and `(x_right, y_right)` from `coords`, take `d_node = distance`, and emit the three segment rows specified above. `3 * (n - 1)` segment rows total. This is consumed directly by `mark_segment` in clustermap (Task 34) — the engine, not Python, computes the dendrogram glyph geometry.

Tests:
1. JSON round-trip (covered by Step 1).
2. **Per-case correctness against scipy**: for each of the 8 fixture cases, load `linkage_refs.json`, run `apply()` on `data`, and verify:
   - `linkage` output's `distance` column matches scipy's column 2 within `1e-9`.
   - `linkage` output's `(left, right)` columns match scipy's columns 0/1 (allowing for the case where (a,b) and (b,a) are interchangeable per merge — order pairs).
   - `order` output matches `leaves_list`.
3. **n=2 edge case** — input with 2 rows produces 1 merge; `segments` output has exactly 3 rows.
4. **n=1 degenerate** — input with 1 row → clear error.
5. **Four-named-output schema test** — `secondary_outputs` returns the 4 expected keys (`linkage`, `order`, `coords`, `segments`) with the documented schemas.
6. **Segments geometry test**: for the `ward_euclidean` fixture case, manually walk the scipy linkage matrix and verify our `segments` output matches the expected (x, y, x2, y2) tuples within `1e-9`. (~30 LOC of test code; the reference computation is a 15-line Python helper that mirrors the algorithm.)

Reference the path-A/B decision from Task 1 in the module-doc comment.

- [ ] **Step 4: Wire `secondary_outputs` arm in `transform/core.rs`**

In `secondary_outputs` impl, add:

```rust
Self::Linkage(s) => crate::transform::linkage::secondary_outputs(s, batch),
```

The `secondary_outputs` function in `linkage.rs` returns `vec![("linkage_or_<name>".to_string(), linkage_batch), ("order_or_<name>", order_batch), ("coords_or_<name>", coords_batch)]` where the suffixing rule from the spec is applied.

- [ ] **Step 5: PyO3 wrapper, lib.rs, __init__.py, _core.pyi**

Same pattern as Tasks 4–6.

```python
class Linkage:
    def __init__(
        self, *,
        method: str = "ward",
        metric: str = "euclidean",
        axis: str = "rows",
        z_score: str | None = None,
        standard_scale: str | None = None,
        name: str | None = None,
    ) -> None: ...
```

- [ ] **Step 6: Build + test + commit**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
```
Expected: `≥425 passed` (412 + ~12 linkage tests + 1 round-trip).

```bash
git add crates/ferrum-core/src/transform/linkage.rs \
        crates/ferrum-core/src/transform/fixtures/generate_linkage_refs.py \
        crates/ferrum-core/src/transform/fixtures/linkage_refs.json \
        crates/ferrum-core/src/transform/{mod,core}.rs \
        crates/ferrum-core/src/lib.rs \
        src/ferrum/__init__.py src/ferrum/_core.pyi
git commit -m "feat(phase-9a): add Linkage transform (kodama-backed hierarchical clustering)"
```

If Task 1 selected path B, replace `kodama-backed` with `hand-rolled Lance-Williams + NN-chain` in the commit message.

---

### Task 8: `Repeat` typed sentinel (Python + serialization)

**Files:**
- Create: `src/ferrum/repeat.py`
- Modify: `src/ferrum/__init__.py` (re-export `Repeat`)
- Create: `tests/test_phase_9_compound_views.py` (with the first batch of Repeat-sentinel tests)

**Goal:** Provide `ferrum.Repeat.column`, `ferrum.Repeat.row`, `ferrum.Repeat.layer` as IDE-autocompleteable typed values. When passed to `Chart(...).encode(x=Repeat.column)`, the encoding spec serializes that field as `{"$repeat": "column"}` instead of a literal field name.

**Phase 9 design choice:** The sentinel resolution happens **at RepeatChart expansion time** (Python-side, in `RepeatChart.expand()` from Task 10), not in Rust. The Rust side never sees `$repeat` — it sees fully-resolved field names by the time any spec serialization happens. This keeps Rust's `EncodingSpec` unchanged.

This means: Repeat sentinels are a *Python-layer concept*. The Rust-side `spec/repeat.rs` and `spec/encoding.rs` modifications mentioned in the file map for forward compatibility are **deferred** — only needed if we later route placeholder JSON through Rust (e.g. for Vega-Lite output). For Phase 9 the Python-side is sufficient.

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase_9_compound_views.py`:

```python
"""Phase 9 compound view + Repeat sentinel tests."""
import pytest

import ferrum as fe
from ferrum import Repeat


class TestRepeatSentinel:
    def test_column_row_layer_are_distinct_values(self):
        assert Repeat.column is not Repeat.row
        assert Repeat.row is not Repeat.layer
        assert Repeat.column.field == "column"
        assert Repeat.row.field == "row"
        assert Repeat.layer.field == "layer"

    def test_repr_is_descriptive(self):
        assert repr(Repeat.column) == "Repeat.column"
        assert repr(Repeat.row) == "Repeat.row"

    def test_singleton_identity_across_imports(self):
        # Re-importing should give the same object.
        from ferrum.repeat import Repeat as Repeat2
        assert Repeat.column is Repeat2.column

    def test_immutable(self):
        with pytest.raises((AttributeError, TypeError)):
            Repeat.column.field = "row"  # type: ignore

    def test_used_in_encode_serializes_as_dollar_repeat(self):
        # Used via Chart.encode(x=Repeat.column) — RepeatChart expansion converts
        # to a real field, but the bare placeholder must round-trip via to_dict.
        sentinel = Repeat.column
        assert sentinel.to_repeat_dict() == {"$repeat": "column"}
```

- [ ] **Step 2: Run — confirm import error**

```bash
uv run --no-sync pytest tests/test_phase_9_compound_views.py -k Repeat 2>&1 | tail -10
```
Expected: `ImportError: cannot import name 'Repeat'`.

- [ ] **Step 3: Create `src/ferrum/repeat.py`**

```python
"""Repeat — typed placeholder sentinels for RepeatChart templates.

Usage:
    from ferrum import Repeat
    Chart(data).mark_point().encode(x=Repeat.column, y=Repeat.row, color="species")

The placeholders are resolved by RepeatChart.expand() into concrete field names
based on the chart's `row=` / `column=` / `layer=` lists. JSON serialization
(via to_repeat_dict) emits `{"$repeat": "<axis>"}`.
"""
from __future__ import annotations
from typing import Final


class _RepeatPlaceholder:
    """Immutable sentinel naming a Repeat axis ('column' | 'row' | 'layer')."""
    __slots__ = ("_field",)

    def __init__(self, field: str) -> None:
        # Use object.__setattr__ to bypass our own __setattr__ guard.
        object.__setattr__(self, "_field", field)

    @property
    def field(self) -> str:
        return self._field

    def to_repeat_dict(self) -> dict:
        return {"$repeat": self._field}

    def __repr__(self) -> str:
        return f"Repeat.{self._field}"

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError(
            f"_RepeatPlaceholder is immutable; cannot set {name!r}"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, _RepeatPlaceholder):
            return NotImplemented
        return self._field == other._field

    def __hash__(self) -> int:
        return hash(("_RepeatPlaceholder", self._field))


class Repeat:
    """Namespace for typed RepeatChart template sentinels.

    Access the three sentinels via class attributes:

        Repeat.column   # cell's column-axis field
        Repeat.row      # cell's row-axis field
        Repeat.layer    # cell's layer-axis field
    """
    column: Final[_RepeatPlaceholder] = _RepeatPlaceholder("column")
    row:    Final[_RepeatPlaceholder] = _RepeatPlaceholder("row")
    layer:  Final[_RepeatPlaceholder] = _RepeatPlaceholder("layer")
```

- [ ] **Step 4: Re-export from `__init__.py`**

Edit `src/ferrum/__init__.py`. Add after the existing `from ferrum.composition import HConcatChart, VConcatChart`:

```python
from ferrum.repeat import Repeat
```

Add `"Repeat"` to `__all__`.

- [ ] **Step 5: Run tests; verify pass**

```bash
uv run --no-sync pytest tests/test_phase_9_compound_views.py -k Repeat -v 2>&1 | tail -15
```
Expected: 5 passes.

- [ ] **Step 6: Commit**

```bash
git add src/ferrum/repeat.py src/ferrum/__init__.py tests/test_phase_9_compound_views.py
git commit -m "feat(phase-9a): add Repeat typed sentinel for RepeatChart templates"
```

---

### Task 9: `JointChart` compound view (Python)

**Files:**
- Modify: `src/ferrum/composition.py` (add `JointChart`)
- Modify: `src/ferrum/__init__.py` (re-export)
- Modify: `tests/test_phase_9_compound_views.py` (add `TestJointChart` class)

**Goal:** Implement `JointChart(center, *, top=None, right=None, ratio=5, spacing=0.02)` per spec §3.1. Renders a 2×2 grid: bottom-left = center, top-left = top, bottom-right = right, top-right = empty. Axis sharing: x across (center, top); y across (center, right).

**Note:** Rendering uses `compose_svg_grid` from Task 12. JointChart's class definition lands in this task; `.show_svg()` wiring happens once Task 12 lands. Until then, `.show_svg()` raises a clear "deferred until grid compositor lands in Task 12" error.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_phase_9_compound_views.py`:

```python
import polars as pl


@pytest.fixture
def df_xy():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})


class TestJointChart:
    def test_construction_minimal(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        jc = fe.JointChart(center)
        assert jc.center is center
        assert jc.top is None
        assert jc.right is None
        assert jc.ratio == 5
        assert jc.spacing == 0.02

    def test_construction_with_marginals(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        top = fe.Chart(df_xy).mark_histogram().encode(x="x")
        right = fe.Chart(df_xy).mark_histogram().encode(x="y")
        jc = fe.JointChart(center, top=top, right=right, ratio=4, spacing=0.05)
        assert jc.top is top
        assert jc.right is right
        assert jc.ratio == 4

    def test_charts_property_filters_none(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        top = fe.Chart(df_xy).mark_histogram().encode(x="x")
        jc = fe.JointChart(center, top=top, right=None)
        assert jc.charts == [center, top]   # right=None excluded
        assert len(jc.charts) == 2

    def test_theme_propagates(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        top = fe.Chart(df_xy).mark_histogram().encode(x="x")
        jc = fe.JointChart(center, top=top)
        themed = jc.theme(fe.themes.light)
        # All children get the theme.
        assert themed.center._theme is fe.themes.light
        assert themed.top._theme is fe.themes.light
        # Original is unchanged (immutable).
        assert jc.center._theme is None

    def test_spec_shape(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        top = fe.Chart(df_xy).mark_histogram().encode(x="x")
        jc = fe.JointChart(center, top=top, ratio=3)
        spec = jc.spec
        assert spec["kind"] == "joint"
        assert "center" in spec
        assert spec["top"] is not None
        assert spec["right"] is None
        assert spec["ratio"] == 3
        assert spec["share"]["x"] == ["center", "top"]
        assert spec["share"]["y"] == ["center"]    # right is None → only center

    def test_show_svg_deferred_until_grid_compositor_lands(self, df_xy):
        # Until Task 12 (compose_svg_grid) lands, .show_svg() raises a clear error.
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        jc = fe.JointChart(center)
        with pytest.raises(NotImplementedError, match="compose_svg_grid"):
            jc.show_svg()

    def test_invalid_ratio_errors(self, df_xy):
        center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
        with pytest.raises(ValueError, match="ratio"):
            fe.JointChart(center, ratio=0)
```

- [ ] **Step 2: Run — confirm fails (`AttributeError: module 'ferrum' has no attribute 'JointChart'`)**

```bash
uv run --no-sync pytest tests/test_phase_9_compound_views.py -k JointChart 2>&1 | tail -10
```

- [ ] **Step 3: Append `JointChart` to `composition.py`**

Append to `src/ferrum/composition.py`:

```python
class JointChart(_CompositeBase):
    """Joint distribution view: center + optional top / right marginal Charts.

    Layout: 2x2 grid where bottom-left = center, top-left = top, bottom-right = right,
    top-right = empty. Cell sizing: marginal cells get 1/(ratio+1), center gets
    ratio/(ratio+1). x-axis shared across (center, top); y across (center, right).
    """
    __slots__ = ("center", "top", "right", "ratio", "spacing", "_theme")

    def __init__(
        self,
        center,
        *,
        top=None,
        right=None,
        ratio: int = 5,
        spacing: float = 0.02,
    ) -> None:
        if ratio <= 0:
            raise ValueError(f"ratio must be > 0; got {ratio}")
        self.center = center
        self.top = top
        self.right = right
        self.ratio = ratio
        self.spacing = spacing
        self._theme = None

    @property
    def charts(self) -> list:
        return [c for c in (self.center, self.top, self.right) if c is not None]

    @property
    def spec(self) -> dict:
        import json as _json
        share_x = ["center"]
        if self.top is not None:
            share_x.append("top")
        share_y = ["center"]
        if self.right is not None:
            share_y.append("right")

        def _embed(c):
            if c is None or not hasattr(c, "to_spec"):
                return None
            return _json.loads(c.to_spec().to_json())

        return {
            "kind": "joint",
            "center": _embed(self.center),
            "top": _embed(self.top),
            "right": _embed(self.right),
            "ratio": self.ratio,
            "spacing": self.spacing,
            "share": {"x": share_x, "y": share_y},
        }

    def theme(self, t):
        new = JointChart(
            self.center.theme(t),
            top=(self.top.theme(t) if self.top is not None else None),
            right=(self.right.theme(t) if self.right is not None else None),
            ratio=self.ratio,
            spacing=self.spacing,
        )
        new._theme = t
        return new

    def properties(self, **kwargs):
        # Forward properties() to center; marginals keep their own.
        new = JointChart(
            self.center.properties(**kwargs),
            top=self.top, right=self.right,
            ratio=self.ratio, spacing=self.spacing,
        )
        new._theme = self._theme
        return new

    def show_svg(self) -> str:
        # Will be wired to compose_svg_grid in Task 12.
        try:
            from ferrum._core import compose_svg_grid
        except ImportError as e:
            raise NotImplementedError(
                "JointChart.show_svg() requires compose_svg_grid; "
                "wire-up lands in Phase 9a Task 12"
            ) from e
        # Production wiring (post-Task 12):
        center_svg = self.center.show_svg()
        cells = [None, self.top.show_svg() if self.top is not None else None,
                 center_svg, self.right.show_svg() if self.right is not None else None]
        marginal_share = 1.0 / (self.ratio + 1)
        center_share = self.ratio / (self.ratio + 1)
        return compose_svg_grid(
            cells, rows=2, cols=2,
            row_ratios=[marginal_share, center_share],
            col_ratios=[center_share, marginal_share],
            spacing=self.spacing,
            share_x=[[2, 0]],   # cell indices: center=2, top=0
            share_y=[[2, 3]],   # center=2, right=3
        )

    def show_png(self) -> bytes:
        raise NotImplementedError("JointChart.show_png — Phase 9 follow-up")

    def save(self, path: str, *, format=None, **kwargs):
        from pathlib import Path
        path = Path(path)
        fmt = format or path.suffix.lstrip(".")
        if fmt == "svg":
            path.write_text(self.show_svg())
        else:
            raise NotImplementedError(f"JointChart.save({fmt!r}) not yet supported in Phase 9")

    def show(self):
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        return self.show_svg()

    def __repr__(self) -> str:
        return (
            f"JointChart(center={self.center!r}, top={self.top!r}, "
            f"right={self.right!r}, ratio={self.ratio})"
        )
```

- [ ] **Step 4: Re-export from `__init__.py`**

Edit `src/ferrum/__init__.py`. Replace `from ferrum.composition import HConcatChart, VConcatChart` with:

```python
from ferrum.composition import HConcatChart, VConcatChart, JointChart
```

Add `"JointChart"` to `__all__`.

- [ ] **Step 5: Run tests; expect all 7 to pass**

```bash
uv run --no-sync pytest tests/test_phase_9_compound_views.py -k JointChart -v 2>&1 | tail -15
```

- [ ] **Step 6: Commit**

```bash
git add src/ferrum/composition.py src/ferrum/__init__.py tests/test_phase_9_compound_views.py
git commit -m "feat(phase-9a): add JointChart compound view (center + top/right marginals)"
```

---

### Task 10: `RepeatChart` compound view with `diagonal=` and `corner=`

**Files:**
- Modify: `src/ferrum/composition.py` (add `RepeatChart`)
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/test_phase_9_compound_views.py` (add `TestRepeatChart`)

**Goal:** Implement `RepeatChart(template, *, row=None, column=None, layer=None, diagonal=None, corner=False, spacing=0.02, columns=None, resolve=None)` per spec §3.2. `.expand()` materializes the template into a list of `(row_field, col_field, Chart)` tuples by resolving `Repeat` placeholders inside the template's encoding.

- [ ] **Step 1: Failing tests**

Append to `tests/test_phase_9_compound_views.py`:

```python
class TestRepeatChart:
    @pytest.fixture
    def iris_like(self):
        return pl.DataFrame({
            "sepal_length": [5.1, 4.9, 4.7, 5.0, 5.4],
            "sepal_width":  [3.5, 3.0, 3.2, 3.6, 3.9],
            "petal_length": [1.4, 1.4, 1.3, 1.4, 1.7],
            "species":      ["a", "a", "b", "b", "a"],
        })

    def test_construction_minimal(self, iris_like):
        template = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        rc = fe.RepeatChart(template, row=["sepal_length"], column=["sepal_width"])
        assert rc.row == ["sepal_length"]
        assert rc.column == ["sepal_width"]
        assert rc.diagonal is None
        assert rc.corner is False

    def test_construction_with_diagonal_and_corner(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        diag = fe.Chart(iris_like).mark_histogram().encode(x=Repeat.column)
        rc = fe.RepeatChart(off, row=["a", "b"], column=["a", "b"], diagonal=diag, corner=True)
        assert rc.diagonal is diag
        assert rc.corner is True

    def test_diagonal_without_row_or_column_errors(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        diag = fe.Chart(iris_like).mark_histogram().encode(x=Repeat.column)
        with pytest.raises(ValueError, match="diagonal"):
            fe.RepeatChart(off, row=["a", "b"], diagonal=diag)   # column missing

    def test_expand_2x2_yields_4_charts(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        rc = fe.RepeatChart(off, row=["sepal_length", "sepal_width"],
                            column=["sepal_length", "sepal_width"])
        cells = rc.expand()
        assert len(cells) == 4
        # Each cell is (row_field, col_field, Chart) where the Repeat placeholders
        # in the template are replaced.
        for row_field, col_field, chart in cells:
            x_enc = chart._encoding.get("x")
            y_enc = chart._encoding.get("y")
            assert x_enc.field == col_field
            assert y_enc.field == row_field

    def test_expand_with_diagonal_uses_diag_for_matching_cells(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        diag = fe.Chart(iris_like).mark_histogram().encode(x=Repeat.column)
        rc = fe.RepeatChart(off, row=["a", "b"], column=["a", "b"], diagonal=diag)
        cells = rc.expand()
        # 4 cells; (a,a) and (b,b) use diagonal, (a,b) and (b,a) use off.
        for row_field, col_field, chart in cells:
            if row_field == col_field:
                assert chart._mark in ("bar", None)   # mark_histogram desugars to bar
            else:
                assert chart._mark == "point"

    def test_expand_with_corner_filters_to_lower_triangle(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        rc = fe.RepeatChart(off, row=["a", "b", "c"], column=["a", "b", "c"], corner=True)
        cells = rc.expand()
        # Lower triangle: (b,a), (c,a), (c,b) and the diagonal (a,a),(b,b),(c,c) → 6 cells.
        coords = [(r, c) for r, c, _ in cells]
        # Lower triangle including diagonal: row_idx >= col_idx.
        row_idx = {"a": 0, "b": 1, "c": 2}
        for r, c in coords:
            assert row_idx[r] >= row_idx[c]

    def test_diagonal_with_asymmetric_warns(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        diag = fe.Chart(iris_like).mark_histogram().encode(x=Repeat.column)
        with pytest.warns(UserWarning, match="diagonal"):
            fe.RepeatChart(off, row=["a", "b"], column=["x", "y"], diagonal=diag).expand()

    def test_spec_shape(self, iris_like):
        off = fe.Chart(iris_like).mark_point().encode(x=Repeat.column, y=Repeat.row)
        rc = fe.RepeatChart(off, row=["a"], column=["b"], corner=False)
        spec = rc.spec
        assert spec["kind"] == "repeat"
        assert spec["row"] == ["a"]
        assert spec["column"] == ["b"]
        assert spec["corner"] is False
```

- [ ] **Step 2: Append `RepeatChart` to `composition.py`**

```python
class RepeatChart(_CompositeBase):
    """Repeat a template chart over a grid of row/column field combinations.

    Use `Repeat.column` / `Repeat.row` / `Repeat.layer` typed sentinels in the
    template's `.encode(...)` call to mark which encoding channel gets the
    per-cell field substitution. `RepeatChart.expand()` returns a list of
    `(row_field, col_field, Chart)` tuples — fully resolved Charts.

    `diagonal=` provides an alternate template for cells where row_field ==
    col_field (n×n symmetric repeat). `corner=True` filters the expanded grid
    to the lower triangle (including diagonal).
    """
    __slots__ = (
        "template", "row", "column", "layer", "diagonal", "corner",
        "spacing", "columns", "resolve", "_theme",
    )

    def __init__(
        self,
        template,
        *,
        row=None, column=None, layer=None,
        diagonal=None,
        corner: bool = False,
        spacing: float = 0.02,
        columns: int | None = None,
        resolve=None,
    ) -> None:
        if diagonal is not None and (row is None or column is None):
            raise ValueError(
                "RepeatChart: diagonal= requires both row= and column= to be set"
            )
        self.template = template
        self.row = list(row) if row is not None else None
        self.column = list(column) if column is not None else None
        self.layer = list(layer) if layer is not None else None
        self.diagonal = diagonal
        self.corner = corner
        self.spacing = spacing
        self.columns = columns
        self.resolve = resolve
        self._theme = None

    @property
    def spec(self) -> dict:
        import json as _json
        def _embed(c):
            if c is None or not hasattr(c, "to_spec"):
                return None
            return _json.loads(c.to_spec().to_json())
        return {
            "kind": "repeat",
            "template": _embed(self.template),
            "row": self.row,
            "column": self.column,
            "layer": self.layer,
            "diagonal": _embed(self.diagonal),
            "corner": self.corner,
            "columns": self.columns,
            "resolve": self.resolve,
            "spacing": self.spacing,
        }

    def expand(self) -> list[tuple[str, str, "fe.Chart"]]:
        """Materialize the template into a list of (row_field, col_field, Chart) tuples.

        For each (row_field, col_field) cell:
        - If diagonal is set and row_field == col_field, use diagonal as the source template.
        - Otherwise use template.
        - Replace `Repeat.column` / `Repeat.row` / `Repeat.layer` sentinels in
          the source's encoding dict with the appropriate concrete field name.
        - When corner=True, drop cells where row_idx < col_idx (keep lower triangle + diagonal).
        """
        from ferrum.repeat import _RepeatPlaceholder
        import warnings

        rows = self.row or []
        cols = self.column or []

        # diagonal-with-asymmetric-shape: warn-once at expand time per spec §3.2.
        asymmetric = (
            self.diagonal is not None
            and self.row is not None
            and self.column is not None
            and self.row != self.column
        )
        if asymmetric:
            warnings.warn(
                "RepeatChart: diagonal= ignored because row != column (asymmetric repeat).",
                UserWarning, stacklevel=2,
            )

        out = []
        use_diagonal_match = (
            self.diagonal is not None and not asymmetric
            and len(rows) == len(cols)
        )

        for ri, row_field in enumerate(rows):
            for ci, col_field in enumerate(cols):
                if self.corner and ri < ci:
                    continue
                source = self.template
                if use_diagonal_match and row_field == col_field:
                    source = self.diagonal
                cell = self._resolve_template(source, row_field, col_field)
                out.append((row_field, col_field, cell))
        return out

    def _resolve_template(self, source, row_field: str, col_field: str):
        """Clone source (a Chart) and substitute Repeat placeholders in encoding."""
        from ferrum.repeat import _RepeatPlaceholder
        from ferrum.encoding.base import ChannelBase

        new = source._clone()
        for axis, ch in list(new._encoding.items()):
            if isinstance(ch, _RepeatPlaceholder):
                concrete = self._concrete_field(ch.field, row_field, col_field)
                # Wrap in the appropriate channel class (use existing X/Y/etc.).
                from ferrum.chart import _channel_class_for
                cls = _channel_class_for(axis) or _channel_class_for("x")
                new._encoding[axis] = cls(concrete)
            elif isinstance(ch, ChannelBase) and isinstance(ch.field, _RepeatPlaceholder):
                # Channel constructed from a placeholder — replace its field.
                concrete = self._concrete_field(ch.field.field, row_field, col_field)
                new._encoding[axis] = ch.__class__(concrete)
        return new

    @staticmethod
    def _concrete_field(placeholder_axis: str, row_field: str, col_field: str) -> str:
        if placeholder_axis == "column":
            return col_field
        if placeholder_axis == "row":
            return row_field
        if placeholder_axis == "layer":
            # Layer placeholders are rarer; for n×n grid, fall back to row_field.
            return row_field
        raise ValueError(f"unknown Repeat placeholder axis '{placeholder_axis}'")

    def theme(self, t):
        new = RepeatChart(
            self.template.theme(t),
            row=self.row, column=self.column, layer=self.layer,
            diagonal=(self.diagonal.theme(t) if self.diagonal is not None else None),
            corner=self.corner, spacing=self.spacing,
            columns=self.columns, resolve=self.resolve,
        )
        new._theme = t
        return new

    def show_svg(self) -> str:
        try:
            from ferrum._core import compose_svg_grid
        except ImportError as e:
            raise NotImplementedError(
                "RepeatChart.show_svg() requires compose_svg_grid; "
                "wire-up lands in Phase 9a Task 12"
            ) from e
        cells = self.expand()
        # Build cell SVGs in row-major order; corner=True leaves "missing" upper
        # triangle as None entries.
        rows = self.row or []
        cols = self.column or []
        n_rows, n_cols = len(rows), len(cols)
        grid: list = [None] * (n_rows * n_cols)
        for row_field, col_field, chart in cells:
            ri = rows.index(row_field)
            ci = cols.index(col_field)
            grid[ri * n_cols + ci] = chart.show_svg()
        return compose_svg_grid(
            grid, rows=n_rows, cols=n_cols,
            row_ratios=[1.0] * n_rows,
            col_ratios=[1.0] * n_cols,
            spacing=self.spacing,
            share_x=[],     # per-column x-share semantics live in the renderer (future)
            share_y=[],
        )

    def show_png(self) -> bytes:
        raise NotImplementedError("RepeatChart.show_png — Phase 9 follow-up")

    def save(self, path: str, *, format=None, **kwargs):
        from pathlib import Path
        path = Path(path)
        fmt = format or path.suffix.lstrip(".")
        if fmt == "svg":
            path.write_text(self.show_svg())
        else:
            raise NotImplementedError(f"RepeatChart.save({fmt!r}) not yet supported")

    def show(self):
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        return self.show_svg()

    def __repr__(self) -> str:
        return (
            f"RepeatChart(row={self.row}, column={self.column}, "
            f"diagonal={'set' if self.diagonal is not None else 'None'}, corner={self.corner})"
        )
```

**Note:** `_resolve_template` assumes that `Chart._clone()` is already addressable from this module. Since `composition.py` imports `Chart` lazily, the import lives inside `_resolve_template`. The existing `ChannelBase` channels (X, Y, etc.) already accept either a string field OR — after Task 8 — a `_RepeatPlaceholder`. Verify that `ChannelBase.__init__` can take a placeholder by relaxing the field check (Step 3 below).

- [ ] **Step 3: Relax `ChannelBase.field` to accept `_RepeatPlaceholder`**

Open `src/ferrum/encoding/base.py`. Find the `ChannelBase.__init__` method's field validation. Add:

```python
from ferrum.repeat import _RepeatPlaceholder

# Inside __init__ after existing field-handling:
if isinstance(field, _RepeatPlaceholder):
    # Carry the placeholder verbatim; RepeatChart.expand() resolves at expand time.
    self._field = field
    return
```

If `ChannelBase` already stores via `self.field = field`, the check is: `_field` accepts placeholders without raising. Match the existing storage pattern; do not add new attributes.

- [ ] **Step 3.5: Add `_RepeatPlaceholder` branch to `Chart.encode`**

`Chart.encode` (in `src/ferrum/chart.py` near line 583) currently routes encoding values through:

```python
if isinstance(value, ChannelBase):
    channel = value
elif isinstance(value, str):
    field, type_, agg = parse_shorthand(value)
    /* ... */
    channel = cls(field, **kw)
else:
    raise TypeError(f"encode({name}=...) expects str or {cls.__name__} instance, got {type(value).__name__}")
```

Without an explicit branch, `Chart.encode(x=Repeat.column)` raises `TypeError`. Fix: add a placeholder branch BEFORE the `else: raise` clause:

```python
elif isinstance(value, _RepeatPlaceholder):
    # Repeat sentinel — wrap in the channel class verbatim; RepeatChart.expand
    # replaces the placeholder with a concrete field name at expand time.
    channel = cls(value)
```

Add `from ferrum.repeat import _RepeatPlaceholder` to the imports at the top of `chart.py`.

Add a test in `tests/test_phase_9_compound_views.py::TestRepeatSentinel`:

```python
def test_chart_encode_accepts_repeat_placeholder(self):
    df = pl.DataFrame({"a": [1.0]})
    # Should NOT raise; placeholder rides through encoding.
    chart = fe.Chart(df).mark_point().encode(x=Repeat.column, y=Repeat.row)
    x_ch = chart._encoding["x"]
    y_ch = chart._encoding["y"]
    # The wrapped placeholder is reachable via .field (stored as-is).
    assert isinstance(x_ch.field, _RepeatPlaceholder) or x_ch.field == Repeat.column
```

Where `_RepeatPlaceholder` import comes from `from ferrum.repeat import _RepeatPlaceholder`.

- [ ] **Step 4: Re-export `RepeatChart`**

Edit `__init__.py`:

```python
from ferrum.composition import HConcatChart, VConcatChart, JointChart, RepeatChart
```

Add `"RepeatChart"` to `__all__`.

- [ ] **Step 5: Run tests**

```bash
uv run --no-sync pytest tests/test_phase_9_compound_views.py -k RepeatChart -v 2>&1 | tail -25
```
Expected: 8 passes.

- [ ] **Step 6: Commit**

```bash
git add src/ferrum/composition.py src/ferrum/encoding/base.py \
        src/ferrum/__init__.py tests/test_phase_9_compound_views.py
git commit -m "feat(phase-9a): add RepeatChart compound view with diagonal/corner support"
```

---

### Task 11: `ClusterMapChart` compound view (Python)

**Files:**
- Modify: `src/ferrum/composition.py` (add `ClusterMapChart`)
- Modify: `src/ferrum/__init__.py`
- Modify: `tests/test_phase_9_compound_views.py` (add `TestClusterMapChart`)

**Goal:** Implement `ClusterMapChart(heatmap, *, row_dendrogram=None, col_dendrogram=None, dendrogram_ratio=0.2, spacing=0.02)` per spec §3.3. Wraps three Charts in a 2×2 layout: top-left empty; top-right col_dendrogram; bottom-left row_dendrogram (rotated); bottom-right heatmap.

- [ ] **Step 1: Failing tests** — append to `tests/test_phase_9_compound_views.py`:

```python
class TestClusterMapChart:
    @pytest.fixture
    def df_matrix(self):
        # Faux 5×4 numeric matrix, suitable for clustermap.
        return pl.DataFrame({
            "row_id": [0, 1, 2, 3, 4],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 1.0, 5.0, 5.0, 1.0],
        })

    def test_construction_heatmap_only(self, df_matrix):
        heat = fe.Chart(df_matrix).mark_rect().encode(x="a", y="row_id", fill="b")
        cm = fe.ClusterMapChart(heat)
        assert cm.heatmap is heat
        assert cm.row_dendrogram is None
        assert cm.col_dendrogram is None
        assert cm.dendrogram_ratio == 0.2

    def test_charts_filters_none(self, df_matrix):
        heat = fe.Chart(df_matrix).mark_rect().encode(x="a", y="row_id", fill="b")
        col_d = fe.Chart(df_matrix).mark_rule().encode(x="a", y="b")  # placeholder
        cm = fe.ClusterMapChart(heat, col_dendrogram=col_d)
        assert cm.charts == [heat, col_d]

    def test_spec_shape(self, df_matrix):
        heat = fe.Chart(df_matrix).mark_rect().encode(x="a", y="row_id", fill="b")
        cm = fe.ClusterMapChart(heat, dendrogram_ratio=0.3)
        spec = cm.spec
        assert spec["kind"] == "cluster_map"
        assert "heatmap" in spec
        assert spec["row_dendrogram"] is None
        assert spec["dendrogram_ratio"] == 0.3

    def test_invalid_ratio_errors(self, df_matrix):
        heat = fe.Chart(df_matrix).mark_rect().encode(x="a", y="row_id", fill="b")
        with pytest.raises(ValueError, match="dendrogram_ratio"):
            fe.ClusterMapChart(heat, dendrogram_ratio=0.0)
        with pytest.raises(ValueError, match="dendrogram_ratio"):
            fe.ClusterMapChart(heat, dendrogram_ratio=1.5)

    def test_theme_propagates(self, df_matrix):
        heat = fe.Chart(df_matrix).mark_rect().encode(x="a", y="row_id", fill="b")
        col_d = fe.Chart(df_matrix).mark_rule().encode(x="a", y="b")
        cm = fe.ClusterMapChart(heat, col_dendrogram=col_d)
        themed = cm.theme(fe.themes.light)
        assert themed.heatmap._theme is fe.themes.light
        assert themed.col_dendrogram._theme is fe.themes.light
```

- [ ] **Step 2: Append `ClusterMapChart` to `composition.py`**

```python
class ClusterMapChart(_CompositeBase):
    """Clustered heatmap with optional row/column dendrograms.

    Layout: 2x2. Top-left empty; top-right = col_dendrogram; bottom-left =
    row_dendrogram (rotated 90°); bottom-right = heatmap. Dendrogram value-axes
    are hidden; categorical axes align with the heatmap row/column labels.
    """
    __slots__ = (
        "heatmap", "row_dendrogram", "col_dendrogram",
        "dendrogram_ratio", "spacing", "_theme",
    )

    def __init__(
        self,
        heatmap,
        *,
        row_dendrogram=None,
        col_dendrogram=None,
        dendrogram_ratio: float = 0.2,
        spacing: float = 0.02,
    ) -> None:
        if not (0.0 < dendrogram_ratio < 1.0):
            raise ValueError(
                f"dendrogram_ratio must be in (0, 1); got {dendrogram_ratio}"
            )
        self.heatmap = heatmap
        self.row_dendrogram = row_dendrogram
        self.col_dendrogram = col_dendrogram
        self.dendrogram_ratio = dendrogram_ratio
        self.spacing = spacing
        self._theme = None

    @property
    def charts(self) -> list:
        return [c for c in (self.heatmap, self.col_dendrogram, self.row_dendrogram) if c is not None]

    @property
    def spec(self) -> dict:
        import json as _json
        def _embed(c):
            if c is None or not hasattr(c, "to_spec"):
                return None
            return _json.loads(c.to_spec().to_json())
        return {
            "kind": "cluster_map",
            "heatmap": _embed(self.heatmap),
            "row_dendrogram": _embed(self.row_dendrogram),
            "col_dendrogram": _embed(self.col_dendrogram),
            "dendrogram_ratio": self.dendrogram_ratio,
            "spacing": self.spacing,
        }

    def theme(self, t):
        new = ClusterMapChart(
            self.heatmap.theme(t),
            row_dendrogram=(self.row_dendrogram.theme(t) if self.row_dendrogram is not None else None),
            col_dendrogram=(self.col_dendrogram.theme(t) if self.col_dendrogram is not None else None),
            dendrogram_ratio=self.dendrogram_ratio,
            spacing=self.spacing,
        )
        new._theme = t
        return new

    def show_svg(self) -> str:
        try:
            from ferrum._core import compose_svg_grid
        except ImportError as e:
            raise NotImplementedError(
                "ClusterMapChart.show_svg() requires compose_svg_grid; "
                "wire-up lands in Phase 9a Task 12"
            ) from e
        d = self.dendrogram_ratio
        h = 1.0 - d
        cells = [
            None,
            self.col_dendrogram.show_svg() if self.col_dendrogram is not None else None,
            self.row_dendrogram.show_svg() if self.row_dendrogram is not None else None,
            self.heatmap.show_svg(),
        ]
        return compose_svg_grid(
            cells, rows=2, cols=2,
            row_ratios=[d, h],
            col_ratios=[d, h],
            spacing=self.spacing,
            share_x=[[1, 3]],   # col_dendrogram (top-right) shares x with heatmap (bottom-right)
            share_y=[[2, 3]],   # row_dendrogram (bottom-left) shares y with heatmap (bottom-right)
        )

    def show_png(self) -> bytes:
        raise NotImplementedError("ClusterMapChart.show_png — Phase 9 follow-up")

    def save(self, path: str, *, format=None, **kwargs):
        from pathlib import Path
        path = Path(path)
        fmt = format or path.suffix.lstrip(".")
        if fmt == "svg":
            path.write_text(self.show_svg())
        else:
            raise NotImplementedError(f"ClusterMapChart.save({fmt!r}) not yet supported")

    def show(self):
        print(self.show_svg())

    def _repr_svg_(self) -> str:
        return self.show_svg()

    def __repr__(self) -> str:
        return (
            f"ClusterMapChart(heatmap=set, row_dendrogram={'set' if self.row_dendrogram else 'None'}, "
            f"col_dendrogram={'set' if self.col_dendrogram else 'None'}, "
            f"ratio={self.dendrogram_ratio})"
        )
```

- [ ] **Step 3: Re-export and run tests**

```python
# __init__.py
from ferrum.composition import HConcatChart, VConcatChart, JointChart, RepeatChart, ClusterMapChart
```

Add `"ClusterMapChart"` to `__all__`.

```bash
uv run --no-sync pytest tests/test_phase_9_compound_views.py -k ClusterMapChart -v 2>&1 | tail -10
```
Expected: 5 passes.

- [ ] **Step 4: Commit**

```bash
git add src/ferrum/composition.py src/ferrum/__init__.py \
        tests/test_phase_9_compound_views.py
git commit -m "feat(phase-9a): add ClusterMapChart compound view"
```

---

### Task 12: `compose_svg_grid` Rust helper + PyO3 binding

**Files:**
- Create: `crates/ferrum-core/src/render/grid_compose.rs`
- Modify: `crates/ferrum-core/src/render/mod.rs` (add `pub(crate) mod grid_compose;`)
- Modify: `crates/ferrum-core/src/render/binding.rs` (add `compose_svg_grid_py`)
- Modify: `crates/ferrum-core/src/lib.rs` (register pyfunction)
- Modify: `src/ferrum/_core.pyi` (stub)
- Modify: `tests/test_phase_9_compound_views.py` (extend `JointChart` + `ClusterMapChart` tests to verify rendering succeeds end-to-end)

**Goal:** Add `compose_svg_grid(cells, *, rows, cols, row_ratios, col_ratios, spacing, share_x, share_y)` row-major grid layout with explicit ratios + spacing + share-x/y groups. Mirrors `compose_svg_horizontal` / `compose_svg_vertical` from Phase 8a.

- [ ] **Step 1: Write the failing test in compose Rust module**

Create `crates/ferrum-core/src/render/grid_compose.rs` with placeholder + tests:

```rust
//! Phase 9 grid compositor — SVG row-major grid with row/col ratios and spacing.
//!
//! Used by JointChart, RepeatChart, ClusterMapChart. share_x/share_y groups
//! are honored only insofar as cells in the same group get the same width
//! (for share_x) or height (for share_y); coordinate-system rebinding within
//! the SVG body is out of scope for Phase 9 (callers ensure their child SVGs
//! have aligned plot areas at construction time, e.g. via Chart.properties).

use crate::render::compositor::{parse_svg_root, strip_font_defs, fmt_f, CompositorError};

/// Compose a row-major grid of SVGs.
///
/// Cell ordering: row-major. `cells[i*cols + j]` is row i, column j; None = empty cell.
/// `row_ratios` and `col_ratios` must sum to a non-zero total; child widths/heights
/// are scaled to match `row_ratios[i] / sum(row_ratios)` of total height (and similarly cols).
/// `spacing` is in absolute SVG units between adjacent cells.
pub fn compose_svg_grid(
    cells: &[Option<String>],
    rows: usize,
    cols: usize,
    row_ratios: &[f64],
    col_ratios: &[f64],
    spacing: f64,
) -> Result<String, CompositorError> {
    if cells.len() != rows * cols {
        return Err(CompositorError::EmptyInput);  // re-use existing variant; add a `SizeMismatch(String)` variant in this task if `CompositorError` doesn't already have one and the implementer prefers a more descriptive error
    }
    if row_ratios.len() != rows || col_ratios.len() != cols {
        return Err(CompositorError::EmptyInput);
    }
    let row_sum: f64 = row_ratios.iter().sum();
    let col_sum: f64 = col_ratios.iter().sum();
    if row_sum <= 0.0 || col_sum <= 0.0 {
        return Err(CompositorError::EmptyInput);
    }

    // Pick the largest cell width (per column) and height (per row) as the
    // canonical cell size for that column/row; scale all cells to fit.
    // Phase 9 simplification: each row/col uses its first non-None cell's parsed dim.
    let mut col_widths = vec![0.0_f64; cols];
    let mut row_heights = vec![0.0_f64; rows];
    let mut parsed: Vec<Option<crate::render::compositor::ParsedSvg<'_>>> =
        Vec::with_capacity(cells.len());
    // Note: parsed lifetimes need 'static per cell; we re-parse from cells: &[Option<String>].
    for (idx, opt) in cells.iter().enumerate() {
        if let Some(svg) = opt {
            let p = parse_svg_root(svg)?;
            let r = idx / cols;
            let c = idx % cols;
            col_widths[c] = col_widths[c].max(p.width);
            row_heights[r] = row_heights[r].max(p.height);
            parsed.push(Some(p));
        } else {
            parsed.push(None);
        }
    }

    let total_w: f64 = col_widths.iter().sum::<f64>() + spacing * (cols.saturating_sub(1)) as f64;
    let total_h: f64 = row_heights.iter().sum::<f64>() + spacing * (rows.saturating_sub(1)) as f64;

    let mut out = String::with_capacity(cells.iter().filter_map(|c| c.as_ref().map(|s| s.len())).sum::<usize>() + 256);
    out.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        fmt_f(total_w), fmt_f(total_h), fmt_f(total_w), fmt_f(total_h),
    ));

    let mut first_emitted = false;
    let mut y_offset = 0.0_f64;
    for r in 0..rows {
        let mut x_offset = 0.0_f64;
        for c in 0..cols {
            let idx = r * cols + c;
            if let Some(p) = &parsed[idx] {
                out.push_str(&format!(r#"<g transform="translate({},{})">"#, fmt_f(x_offset), fmt_f(y_offset)));
                let body_owned;
                let body: &str = if !first_emitted {
                    p.body
                } else {
                    body_owned = strip_font_defs(p.body);
                    &body_owned
                };
                out.push_str(body);
                out.push_str("</g>");
                first_emitted = true;
            }
            x_offset += col_widths[c] + if c + 1 < cols { spacing } else { 0.0 };
        }
        y_offset += row_heights[r] + if r + 1 < rows { spacing } else { 0.0 };
    }
    out.push_str("</svg>");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn svg(w: f64, h: f64, fill: &str) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}"><rect x="0" y="0" width="{}" height="{}" fill="{}" /></svg>"#,
            fmt_f(w), fmt_f(h), fmt_f(w), fmt_f(h), fmt_f(w), fmt_f(h), fill,
        )
    }

    #[test]
    fn compose_2x2_grid_with_ratios_and_spacing() {
        let a = svg(50.0, 50.0, "red");
        let b = svg(50.0, 50.0, "blue");
        let c = svg(50.0, 50.0, "green");
        let d = svg(50.0, 50.0, "yellow");
        let cells = vec![Some(a), Some(b), Some(c), Some(d)];
        let out = compose_svg_grid(
            &cells, 2, 2,
            &[1.0, 1.0], &[1.0, 1.0],
            5.0,
        ).unwrap();
        assert!(out.contains(r#"width="105""#));   // 50+5+50
        assert!(out.contains(r#"height="105""#));
        assert!(out.contains(r#"transform="translate(0,0)""#));
        assert!(out.contains(r#"translate(55,0)"#));
        assert!(out.contains(r#"translate(0,55)"#));
        assert!(out.contains(r#"translate(55,55)"#));
    }

    #[test]
    fn compose_grid_with_none_cell_skips_empty_position() {
        let a = svg(40.0, 40.0, "red");
        let b = svg(40.0, 40.0, "blue");
        let cells = vec![Some(a), None, Some(b), None];
        let out = compose_svg_grid(&cells, 2, 2, &[1.0, 1.0], &[1.0, 1.0], 0.0).unwrap();
        assert!(out.contains(r#"translate(0,0)"#));
        assert!(out.contains(r#"translate(0,40)"#));
        // Top-right and bottom-right are empty — no group at translate(40, *).
        assert!(!out.contains(r#"translate(40,0)"#));
    }

    #[test]
    fn compose_grid_size_mismatch_errors() {
        let cells: Vec<Option<String>> = vec![None];
        let err = compose_svg_grid(&cells, 2, 2, &[1.0, 1.0], &[1.0, 1.0], 0.0).unwrap_err();
        assert!(matches!(err, CompositorError::EmptyInput));
    }
}
```

**Note on `parse_svg_root` lifetime:** the existing `compositor.rs` has `ParsedSvg<'a>` with `body: &'a str`. The grid impl above uses `&Option<String>` and re-borrows; if the lifetime constraint blocks compilation, switch to owned-body parsing or copy `body` to `String` before storing. Adjust per Rust borrow checker output.

- [ ] **Step 2: Wire `pub(crate) mod grid_compose;` in `render/mod.rs`**

Find `pub(crate) mod compositor;` in `render/mod.rs` and add:

```rust
pub(crate) mod grid_compose;
```

- [ ] **Step 3: PyO3 wrapper in `render/binding.rs`**

Append after `compose_svg_vertical_py`:

```rust
#[pyfunction]
#[pyo3(name = "compose_svg_grid")]
#[pyo3(signature = (cells, *, rows, cols, row_ratios, col_ratios, spacing = 10.0,
                     share_x = Vec::<Vec<usize>>::new(), share_y = Vec::<Vec<usize>>::new()))]
pub fn compose_svg_grid_py(
    cells: Vec<Option<String>>,
    rows: usize,
    cols: usize,
    row_ratios: Vec<f64>,
    col_ratios: Vec<f64>,
    spacing: f64,
    _share_x: Vec<Vec<usize>>,
    _share_y: Vec<Vec<usize>>,
) -> PyResult<String> {
    crate::render::grid_compose::compose_svg_grid(
        &cells, rows, cols, &row_ratios, &col_ratios, spacing,
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
```

`share_x` / `share_y` parameters are accepted (for forward compatibility with Python's expected call signature) but ignored in Phase 9 — the rendering is structural composition only; cross-cell scale-binding is a future phase.

- [ ] **Step 4: Register pyfunction in `lib.rs`**

After `m.add_function(wrap_pyfunction!(render::binding::compose_svg_vertical_py, m)?)?;`:

```rust
m.add_function(wrap_pyfunction!(render::binding::compose_svg_grid_py, m)?)?;
```

- [ ] **Step 5: Add stub in `src/ferrum/_core.pyi`**

```python
def compose_svg_grid(
    cells: list[str | None], *,
    rows: int, cols: int,
    row_ratios: list[float], col_ratios: list[float],
    spacing: float = 10.0,
    share_x: list[list[int]] = ...,
    share_y: list[list[int]] = ...,
) -> str: ...
```

Add to the `__all__` import block in `src/ferrum/__init__.py`:

```python
from ferrum._core import (
    ...,
    compose_svg_grid,
)
```

- [ ] **Step 6: Build, run cargo + pytest tests**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core grid_compose 2>&1 | tail -10
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
uv run --no-sync pytest tests/test_phase_9_compound_views.py 2>&1 | tail -3
```
Expected: cargo `≥428 passed` (425 + 3 grid tests); pytest all phase 9 compound view tests pass.

- [ ] **Step 7: Update JointChart `show_svg` test to no longer expect NotImplementedError**

In `test_phase_9_compound_views.py`, replace `test_show_svg_deferred_until_grid_compositor_lands` for JointChart with:

```python
def test_show_svg_renders_after_task_12(self, df_xy):
    center = fe.Chart(df_xy).mark_point().encode(x="x", y="y")
    top = fe.Chart(df_xy).mark_histogram().encode(x="x")
    jc = fe.JointChart(center, top=top)
    out = jc.show_svg()
    assert out.startswith("<svg")
    assert "</svg>" in out
```

Run pytest again to confirm.

- [ ] **Step 8: Commit**

```bash
git add crates/ferrum-core/src/render/grid_compose.rs \
        crates/ferrum-core/src/render/{mod,binding}.rs \
        crates/ferrum-core/src/lib.rs \
        src/ferrum/_core.pyi src/ferrum/__init__.py \
        tests/test_phase_9_compound_views.py
git commit -m "feat(phase-9a): add compose_svg_grid Rust helper + PyO3 binding"
```

**End of 9a-foundation. cargo test ≥ 428 passed; pytest all compound-view + Repeat tests pass.**

---

## 9b — Stat-engine extensions

### Task 13: Extend `Bin` transform with `cumulative` field

**Files:**
- Modify: `crates/ferrum-core/src/transform/bin.rs` (add `cumulative: bool` to `BinSpec`; cumulative output branch in `apply`; tests)
- Modify: `src/ferrum/_core.pyi` (update `Bin` stub)

- [ ] **Step 1: Failing test — append to `transform/bin.rs` test module**

```rust
#[test]
fn bin_cumulative_count_is_monotonic() {
    use arrow::array::Float64Array;
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ))]).unwrap();
    let spec = BinSpec {
        field: "x".into(), bin_count: Some(5), bin_width: None,
        extent: Some((1.0, 10.0)), nice: false, cumulative: true, name: None,
    };
    let out = apply(&spec, &batch).unwrap();
    let count_idx = out.schema().index_of("count").unwrap();
    let counts = out.column(count_idx).as_any().downcast_ref::<UInt64Array>().unwrap();
    let n = counts.len();
    for i in 1..n {
        assert!(counts.value(i) >= counts.value(i-1),
            "cumulative count not monotonic at i={i}: {} < {}",
            counts.value(i), counts.value(i-1));
    }
    assert_eq!(counts.value(n-1), 10);
}
```

- [ ] **Step 2: Add `cumulative` field, branch in apply, fix existing constructor sites**

In `transform/bin.rs`, add `#[serde(default)] pub cumulative: bool,` to `BinSpec` (place between `nice` and `name`). In `apply`, after the per-bin counts are computed but before the output schema is built, branch:

```rust
let final_counts: Vec<u64> = if spec.cumulative {
    let mut acc = 0u64;
    counts.iter().map(|c| { acc = acc.saturating_add(*c); acc }).collect()
} else {
    counts.clone()
};
// Same trapezoidal-style sweep for `density` if cumulative=true (mirror kde.rs lines 38-46).
```

Search for existing constructor sites and add `cumulative: false,`:
```bash
grep -rn "BinSpec {" crates/ferrum-core/src/ 2>/dev/null
```

Update `PyBin.new` signature to accept `cumulative = false` and pass it through.

- [ ] **Step 3: Update `_core.pyi`**

```python
class Bin:
    def __init__(
        self, field: str, *,
        bin_count: int | None = None, bin_width: float | None = None,
        extent: tuple[float, float] | None = None, nice: bool = True,
        cumulative: bool = False,
        name: str | None = None,
    ) -> None: ...
```

- [ ] **Step 4: Build + tests + commit**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
```
Expected: `≥429 passed`.

```bash
git add crates/ferrum-core/src/transform/bin.rs src/ferrum/_core.pyi
git commit -m "feat(phase-9b): add Bin.cumulative parameter"
```

---

### Task 14: Extend `Smooth` with `x_bins`, `x_estimator`, `output`

**Files:**
- Modify: `crates/ferrum-core/src/transform/smooth.rs`
- Modify: `src/ferrum/_core.pyi`

**New fields:**
```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SmoothOutput { Fitted, Residuals }

pub(crate) fn default_smooth_output() -> SmoothOutput { SmoothOutput::Fitted }

pub(crate) struct SmoothSpec {
    /* existing fields ... */
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x_bins: Option<usize>,                                         // NEW
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x_estimator: Option<crate::transform::aggregate::AggFn>,       // NEW
    #[serde(default = "default_smooth_output")]
    pub output: SmoothOutput,                                          // NEW
}
```

- [ ] **Step 1: Failing tests in `smooth.rs` test module**

```rust
#[test]
fn smooth_x_bins_pre_aggregates_then_fits() {
    // 100 points; bin into 10 mean-aggregated points; LM slope ≈ 2.0 within 1e-6.
    /* build batch with linear y = 2x+1; SmoothSpec { x_bins: Some(10), x_estimator: Some(AggFn::Mean), method: Lm, ci: None, ... }; assert slope */
}

#[test]
fn smooth_output_residuals_returns_y_minus_fitted() {
    // y = 2x+1 + noise; residuals' mean ≈ 0; max |residual| < 0.5.
}

#[test]
fn smooth_output_default_is_fitted() {
    // Default spec produces (x, y, ci_lower, ci_upper) schema unchanged from Phase 5.
}
```

- [ ] **Step 2: Add fields, branches, update sites**

Add the three fields. In `apply`, branch on `(x_bins, x_estimator)` to call a new `pre_aggregate_xy(xs, ys, n_bins, estimator)` helper before the existing fit logic. Branch on `output` after fitting: for `Residuals`, evaluate the fit at each input x (not the grid) and emit `(x, residual)` rows; output schema is `[x: Float64, residual: Float64]`. The `residual` column replaces "y".

Search and update existing `SmoothSpec { ... }` constructor sites to include `x_bins: None, x_estimator: None, output: SmoothOutput::Fitted,`.

`PySmooth.new` signature: see spec; accept `x_bins: Option<usize>`, `x_estimator: Option<&str>` ("mean"|"median"|"sum"|"min"|"max"), `output: &str` ("fitted"|"residuals").

- [ ] **Step 3: Update `_core.pyi`**

```python
class Smooth:
    def __init__(
        self, x: str, y: str, *,
        method: str = "lm", ci: float | None = None,
        bandwidth: float = 0.5, degree: int = 1, n: int = 200, seed: int = 0,
        x_bins: int | None = None,
        x_estimator: str | None = None,    # "mean"|"median"|"sum"|"min"|"max"
        output: str = "fitted",            # "fitted"|"residuals"
        name: str | None = None,
    ) -> None: ...
```

- [ ] **Step 4: Build + tests + commit**

```bash
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
```
Expected: `≥432 passed`.

```bash
git add crates/ferrum-core/src/transform/smooth.rs src/ferrum/_core.pyi
git commit -m "feat(phase-9b): add Smooth.{x_bins,x_estimator,output} parameters"
```

---

### Task 15: `LetterValue` transform (Rust + numpy-quantile fixture)

**Files:**
- Create: `crates/ferrum-core/src/transform/letter_value.rs`
- Create: `crates/ferrum-core/src/transform/fixtures/generate_letter_value_refs.py`
- Create: `crates/ferrum-core/src/transform/fixtures/letter_value_refs.json` (generated, committed)
- Modify: `transform/mod.rs`, `transform/core.rs` (incl. `secondary_outputs` for `outliers`), `lib.rs`, `__init__.py`, `_core.pyi`

**Spec & outputs:** Per design doc §5.4 — primary `[group, depth, lower, upper, level]` + secondary `outliers` `[group, value, is_outlier]`. **PLUS per-depth named outputs** (added to honor mark_boxen's per-band rect-layer rendering without overlap — see Task 25):

| Output | Schema | Purpose |
|---|---|---|
| primary (FINAL_OUTPUT_KEY or `<name>`) | `[group, depth, lower, upper, level]` | full letter-value table |
| `outliers` (or `<name>_outliers`) | `[group, value, is_outlier]` | outlier points |
| `depth_1`, `depth_2`, ..., `depth_8` (or `<name>_depth_K`) | `[group, lower, upper, level]` (no `depth` column — the column is implicit in the output name) | one row per group at depth K |

Rationale: each `depth_K` named output emits the rows from primary filtered to `depth == K`. If the actual K_actual is < 8, the unused outputs are emitted as zero-row batches (so mark_boxen's K_MAX=6 rect layers safely render fewer than 6 bands when data has fewer depths).

**K-depth strategies (per Hofmann/Wickham/Kafadar 2017):**
- **Tukey:** `K = max(1, floor(log2(n)) - 3)`.
- **Proportion(p):** smallest K such that `2^(K-1) >= ceil(p*n)`. Default p=0.007.
- **Trustworthy:** `K = floor(log2(n / 3.32))` where 3.32 = 2.576² / 2.
- **Full:** `K = floor(log2(n))`.

For each k in `1..=K`, lower / upper at quantile `2^(-k)` and `1 - 2^(-k)` via numpy-style linear interpolation on sorted values. k=1 gives lower=upper=median.

Outlier classification: `value < lower_K - threshold * (upper_K - lower_K)` or symmetric upper.

- [ ] **Step 1: Round-trip test in `core.rs`** (mirror Task 4 Step 1).

- [ ] **Step 2: Generate fixtures**

Create `crates/ferrum-core/src/transform/fixtures/generate_letter_value_refs.py`:

```python
"""Phase 9 LetterValue fixture generator (numpy quantile reference)."""
import json
import sys
from pathlib import Path
import numpy as np


def lv_at_depths(values, depths):
    sorted_v = np.sort(np.asarray(values, dtype=np.float64))
    out = []
    for k in depths:
        q_lo = 0.5 ** k
        q_hi = 1.0 - 0.5 ** k
        out.append({
            "depth": int(k),
            "lower": float(np.quantile(sorted_v, q_lo)),
            "upper": float(np.quantile(sorted_v, q_hi)),
        })
    return out


def main():
    rng = np.random.default_rng(42)
    cases = []

    # Case 1: gaussian n=100, Tukey.
    n = 100; x = rng.normal(0, 1, n).tolist()
    K = max(1, int(np.floor(np.log2(n))) - 3)
    cases.append({"name": "tukey_gaussian_n100", "k_depth": "tukey",
                  "n": n, "K": K, "values": x,
                  "letter_values": lv_at_depths(x, range(1, K+1)),
                  "outlier_threshold": 1.5})

    # Case 2: t-distribution n=200, Proportion p=0.007.
    n = 200; x = rng.standard_t(df=3, size=n).tolist()
    target = int(np.ceil(0.007 * n))
    K = 1
    while (1 << (K - 1)) < target:
        K += 1
    cases.append({"name": "proportion_t3_n200", "k_depth": "proportion", "p": 0.007,
                  "n": n, "K": K, "values": x,
                  "letter_values": lv_at_depths(x, range(1, K+1)),
                  "outlier_threshold": 1.5})

    # Case 3: small n=20, Full.
    n = 20; x = rng.normal(0, 1, n).tolist()
    K = int(np.floor(np.log2(n)))
    cases.append({"name": "full_gaussian_n20", "k_depth": "full",
                  "n": n, "K": K, "values": x,
                  "letter_values": lv_at_depths(x, range(1, K+1)),
                  "outlier_threshold": 1.5})

    # Case 4: 3-group n=300, Tukey, grouped.
    groups = []
    for g, mu in [("a", -2), ("b", 0), ("c", 2)]:
        gx = rng.normal(mu, 1.0, 100).tolist()
        K = max(1, int(np.floor(np.log2(100))) - 3)
        groups.append({"group": g, "n": 100, "K": K, "values": gx,
                       "letter_values": lv_at_depths(gx, range(1, K+1))})
    cases.append({"name": "tukey_grouped_3x100", "k_depth": "tukey",
                  "groups": groups, "outlier_threshold": 1.5})

    payload = {"_pinned_versions": {"numpy": np.__version__}, "cases": cases}
    out = Path(__file__).resolve().parent / "letter_value_refs.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
```

Run:
```bash
uv run python crates/ferrum-core/src/transform/fixtures/generate_letter_value_refs.py
```

- [ ] **Step 3: Write `transform/letter_value.rs`** (~200 LOC)

Use linear-interpolated quantile (mirror `transform/qq.rs::quantile_sorted` line 343-357). Per-group: partition input by group column then loop. `secondary_outputs` returns a single `("outliers", outliers_batch)` (or `("<name>_outliers", ...)` when `name` is set).

Tests: 6 cases (round-trip, 3 single-group correctness from fixture, 1 grouped, 1 outliers schema, plus n=4/Full edge case).

- [ ] **Step 4: Wire mod.rs, core.rs (incl. `secondary_outputs` arm), lib.rs, __init__.py, _core.pyi**

```python
# _core.pyi
class LetterValue:
    def __init__(
        self, value: str, *,
        group: str | None = None,
        k_depth: str = "proportion",
        k_proportion: float = 0.007,
        outlier_threshold: float = 1.5,
        name: str | None = None,
    ) -> None: ...
```

`PyLetterValue` constructor maps `k_depth="proportion"` + `k_proportion=p` → `KDepth::Proportion { p }`.

- [ ] **Step 5: Build + tests + commit**

```bash
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
```
Expected: `≥440 passed`.

```bash
git add crates/ferrum-core/src/transform/letter_value.rs \
        crates/ferrum-core/src/transform/fixtures/generate_letter_value_refs.py \
        crates/ferrum-core/src/transform/fixtures/letter_value_refs.json \
        crates/ferrum-core/src/transform/{mod,core}.rs \
        crates/ferrum-core/src/lib.rs \
        src/ferrum/__init__.py src/ferrum/_core.pyi
git commit -m "feat(phase-9b): add LetterValue transform (boxen plot statistics)"
```

---

### Task 16: `Logistic` transform (IRLS + Wald CI)

**Files:**
- Create: `crates/ferrum-core/src/transform/logistic.rs`
- Create: `crates/ferrum-core/src/transform/fixtures/generate_logistic_refs.py`
- Create: `crates/ferrum-core/src/transform/fixtures/logistic_refs.json`
- Modify: mod.rs, core.rs, lib.rs, __init__.py, _core.pyi

**Algorithm (IRLS for logit-link binomial):**
1. Initialize β = (0, 0); μᵢ = 0.5; ηᵢ = 0.
2. Repeat ≤ `max_iter`:
   - Weights wᵢ = μᵢ(1 - μᵢ).
   - Working response zᵢ = ηᵢ + (yᵢ - μᵢ) / wᵢ.
   - Solve weighted LS β = (XᵀWX)⁻¹ XᵀWz where X = [1, x] (n × 2 design matrix).
   - Update ηᵢ = β₀ + β₁ xᵢ; μᵢ = 1/(1 + exp(-ηᵢ)).
   - Convergence: |Δlog-likelihood| < tol.
3. Wald CI on linear predictor at each grid point: `η_grid ± z·√(xᵀ Σ̂ x)` where Σ̂ = (XᵀWX)⁻¹; transform to probability via inverse logit.

**Edge errors:**
- Perfect separation (max |β| > 1e3 OR fitted probs saturate to (1e-6, 1-1e-6) within 5 iterations) → `PyValueError`.
- Non-binary y (uniques not subset of {0, 1}) → `PyValueError` listing uniques.
- Singular XᵀWX (Cholesky fails) → `PyValueError`.

**Fixture generator** (`generate_logistic_refs.py`):
```python
import statsmodels.api as sm
import numpy as np

CASES = [
    ("well_separated",       np.linspace(-3, 3, 60)),
    ("moderately_separated", np.linspace(-2, 2, 80)),
    ("near_degenerate",      np.linspace(-0.5, 0.5, 40)),
    ("integer_x",            np.arange(20).astype(float)),
    ("challenger_o_rings",   /* 23-row historical dataset */),
]

# For each x: y = (logistic(α + β x) > rng.uniform()).astype(int).
# Fit sm.Logit, extract grid + fitted + Wald CI in probability space.
```

Tests (~10): round-trip, 5 fixture-correctness, 1 perfect-separation error, 1 non-binary error, 1 singular-design error, 1 convergence-iteration test.

- [ ] **Step 1**: Generate fixtures.
- [ ] **Step 2**: Write `transform/logistic.rs` with IRLS body. 2x2 SPD solve is closed-form (no need for `linalg.rs`'s 3x3 Cholesky).
- [ ] **Step 3**: Wire mod.rs/core.rs/lib.rs/Python (mirror Task 4-7 plumbing).

```python
# _core.pyi
class Logistic:
    def __init__(
        self, x: str, y: str, *,
        n_grid: int = 100, ci: float | None = None,
        max_iter: int = 25, tol: float = 1e-8,
        name: str | None = None,
    ) -> None: ...
```

- [ ] **Step 4**: Build + test + commit. Expected: `≥451 passed`.
```bash
git add crates/ferrum-core/src/transform/logistic.rs \
        crates/ferrum-core/src/transform/fixtures/{generate_logistic_refs.py,logistic_refs.json} \
        crates/ferrum-core/src/transform/{mod,core}.rs \
        crates/ferrum-core/src/lib.rs \
        src/ferrum/__init__.py src/ferrum/_core.pyi
git commit -m "feat(phase-9b): add Logistic transform (IRLS + Wald CI)"
```

---

### Task 17: `Glm` transform (5 families × 7 links)

**Files:** Same shape as Task 16 — `transform/glm.rs` + `generate_glm_refs.py` + `glm_refs.json`.

**Algorithm:** Generalized IRLS parameterized by family-variance `V(μ)` and link mean function / inverse / derivative. The IRLS body is a parameterized version of Task 16: `wᵢ = 1 / [V(μᵢ) · (g'(μᵢ))²]`; working response zᵢ = ηᵢ + (yᵢ - μᵢ) · g'(μᵢ); same WLS solve.

**Family/link compatibility table** (per design doc §5.2):

| Family | Canonical | Other valid |
|---|---|---|
| Gaussian | Identity | Log, Inverse |
| Binomial | Logit | Probit, Log |
| Poisson | Log | Identity, Sqrt |
| Gamma | Inverse | Identity, Log |
| InverseGaussian | InverseSquared | Identity, Log |

**Variance functions:**

| Family | V(μ) |
|---|---|
| Gaussian | 1 |
| Binomial | μ(1-μ) |
| Poisson | μ |
| Gamma | μ² |
| InverseGaussian | μ³ |

**Link functions** (`Identity` / `Log` / `Logit` / `Probit` / `Inverse` / `InverseSquared` / `Sqrt`):
- Identity: g(μ)=μ, g⁻¹(η)=η, g'(μ)=1.
- Log: g=ln μ, g⁻¹=exp η, g'=1/μ.
- Logit: g=ln(μ/(1-μ)), g⁻¹=1/(1+exp(-η)), g'=1/(μ(1-μ)).
- Probit: g=Φ⁻¹(μ), g⁻¹=Φ(η), g'=1/φ(Φ⁻¹(μ)).
- Inverse: g=1/μ, g⁻¹=1/η, g'=-1/μ².
- InverseSquared: g=1/μ², g⁻¹=1/√η, g'=-2/μ³.
- Sqrt: g=√μ, g⁻¹=η², g'=1/(2√μ).

Implementation organized as functions:
- `link_apply(link: GlmLink, mu: f64) -> f64` (g(μ)).
- `link_inverse(link: GlmLink, eta: f64) -> f64` (g⁻¹(η)).
- `link_derivative(link: GlmLink, mu: f64) -> f64` (g'(μ)).
- `variance(family: GlmFamily, mu: f64) -> f64`.
- `validate_pair(family, link) -> Result<()>` returns error listing valid links if invalid.
- `apply` body is parameterized IRLS from Task 16, calling the above per-iteration.

**Fixture generator**: loops over (family, canonical link) for all 5 families + 3 non-canonical (Gaussian+Log, Binomial+Probit, Poisson+Sqrt). Uses statsmodels:

```python
import statsmodels.api as sm
fam_map = {
    "gaussian": sm.families.Gaussian,
    "binomial": sm.families.Binomial,
    "poisson": sm.families.Poisson,
    "gamma": sm.families.Gamma,
    "inverse_gaussian": sm.families.InverseGaussian,
}
link_map = {
    "identity": sm.families.links.identity(),
    "log": sm.families.links.log(),
    "logit": sm.families.links.logit(),
    "probit": sm.families.links.probit(),
    "inverse": sm.families.links.inverse_power(),       # name varies; verify in Task 2
    "inverse_squared": sm.families.links.inverse_squared(),
    "sqrt": sm.families.links.sqrt(),
}
# For each (family, link) pair: fit GLM, extract fitted + Wald CI on grid.
```

Tests (~12): round-trip; 5 canonical-link cases; 3 non-canonical cases; invalid-pair errors (e.g., Binomial+Inverse → error listing valid links); convergence-iteration test.

- [ ] **Step 1**: Generate fixtures. Verify statsmodels link-name strings against Task 2 records.
- [ ] **Step 2**: Write `transform/glm.rs` (~600 LOC).
- [ ] **Step 3**: Wire (mirror Tasks 4-7).

```python
class Glm:
    def __init__(
        self, x: str, y: str, *,
        family: str = "gaussian",
        link: str | None = None,
        n_grid: int = 100, ci: float | None = None,
        max_iter: int = 25, tol: float = 1e-8,
        name: str | None = None,
    ) -> None: ...
```

- [ ] **Step 4**: Build + test + commit. Expected: `≥464 passed`.

```bash
git add crates/ferrum-core/src/transform/glm.rs \
        crates/ferrum-core/src/transform/fixtures/{generate_glm_refs.py,glm_refs.json} \
        crates/ferrum-core/src/transform/{mod,core}.rs \
        crates/ferrum-core/src/lib.rs \
        src/ferrum/__init__.py src/ferrum/_core.pyi
git commit -m "feat(phase-9b): add Glm transform (5 families × 7 links)"
```

---

### Task 18: `Robust` transform (Huber M-estimator + sandwich CI + `output` field)

**Files:** Same shape as Tasks 16-17 — `transform/robust.rs` + `generate_robust_refs.py` + `robust_refs.json`.

**Spec:** Per design doc §5.3, plus `output: SmoothOutput` field shared with Task 14.

```rust
pub(crate) struct RobustSpec {
    pub x: String, pub y: String,
    pub n_grid: usize,
    pub ci: Option<f64>,
    pub huber_c: f64,                 // default 1.345
    pub max_iter: usize,
    pub tol: f64,
    #[serde(default = "crate::transform::smooth::default_smooth_output")]
    pub output: crate::transform::smooth::SmoothOutput,
    pub name: Option<String>,
}
```

**Algorithm (Huber M-estimator via IRLS):**
1. Initialize β = OLS estimate.
2. Compute residuals rᵢ = yᵢ - Xβ.
3. Estimate scale s = MAD(r) × 1.4826.
4. Compute Huber weights wᵢ = ψ(rᵢ/s) / (rᵢ/s) where ψ(u) = u for |u| ≤ c, c·sign(u) otherwise. (wᵢ = 1 when rᵢ = 0.)
5. Solve weighted least squares: β_new = (XᵀWX)⁻¹ XᵀWy.
6. Repeat until `|Δβ| < tol` or `max_iter` reached.
7. **Sandwich CI:** `Cov(β̂) = (XᵀWX)⁻¹ · (Σ ψ²(rᵢ/s)) · (XᵀWX)⁻¹`.

**Fixtures:** `generate_robust_refs.py` uses `sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT()).fit()`. 4 datasets: clean linear, 10% outliers, 30% outliers, leverage-points.

Tests (~10): round-trip; clean-data ≈ OLS; 10/30% outlier slope matches statsmodels within `1e-2`; CI within `5e-3`; `output=Residuals` schema + sum-of-residuals ≈ 0; default `output=Fitted` schema unchanged; `huber_c` sensitivity; max_iter behavior.

- [ ] **Step 1**: Generate fixtures.
- [ ] **Step 2**: Write `transform/robust.rs`.
- [ ] **Step 3**: Wire (mirror prior tasks).

```python
class Robust:
    def __init__(
        self, x: str, y: str, *,
        n_grid: int = 100, ci: float | None = None,
        huber_c: float = 1.345,
        max_iter: int = 50, tol: float = 1e-8,
        output: str = "fitted",
        name: str | None = None,
    ) -> None: ...
```

- [ ] **Step 4**: Build + test + commit. Expected: `≥475 passed`.

```bash
git add crates/ferrum-core/src/transform/robust.rs \
        crates/ferrum-core/src/transform/fixtures/{generate_robust_refs.py,robust_refs.json} \
        crates/ferrum-core/src/transform/{mod,core}.rs \
        crates/ferrum-core/src/lib.rs \
        src/ferrum/__init__.py src/ferrum/_core.pyi
git commit -m "feat(phase-9b): add Robust transform (Huber M-estimator + output field)"
```

**End of 9b. cargo test ≥ 475 passed; pytest unchanged from end of 9a.**

---

## 9c — Position-adjustment subsystem

9c is **strictly sequential**. Each task extends the same `PositionAdjust` enum, the same `Layer.position` / `ChartSpec.position` field, and the same render pass. Do NOT parallelize within 9c.

### Task 19: `PositionAdjust` enum + `Layer.position` + `ChartSpec.position` + JSON round-trip

**Files:**
- Create: `crates/ferrum-core/src/spec/position.rs`
- Modify: `crates/ferrum-core/src/spec/mod.rs` (add `pub mod position;`)
- Modify: `crates/ferrum-core/src/spec/layer.rs` (add `position: Option<PositionAdjust>` field)
- Modify: `crates/ferrum-core/src/spec/chart.rs` (add `position: Option<PositionAdjust>` field; coerce in `__new__`)

**Goal:** Land the type-level scaffolding for position adjustments. No render-pass changes yet — those come in Tasks 20-22 per-adjustment. After this task, ChartSpec/Layer can carry a `position` field that round-trips through JSON; rendering still ignores it (existing code is `position=None`-equivalent).

- [ ] **Step 1: Create `spec/position.rs`**

```rust
//! Phase 9c — position adjustments (Identity, Dodge, Jitter, Stack).
//!
//! Position adjustments rewrite per-row data values for an eligible mark layer
//! after scale resolution but before mark rendering. They live on Layer.position
//! (and ChartSpec.position for non-layered charts).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JitterAxis { X, Y, Both }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StackOffset { Zero, Normalize, Center }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PositionAdjust {
    Identity,
    Dodge {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        by: Option<String>,
        #[serde(default = "default_padding")]
        padding: f64,
    },
    Jitter {
        axis: JitterAxis,
        #[serde(default = "default_jitter_width")]
        width: f64,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        seed: Option<u64>,
    },
    Stack {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        by: Option<String>,
        #[serde(default = "default_stack_offset")]
        offset: StackOffset,
    },
}

fn default_padding() -> f64 { 0.05 }
fn default_jitter_width() -> f64 { 0.4 }
fn default_stack_offset() -> StackOffset { StackOffset::Zero }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_round_trip() {
        let p = PositionAdjust::Identity;
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, r#"{"type":"identity"}"#);
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn dodge_round_trip() {
        let p = PositionAdjust::Dodge { by: Some("species".into()), padding: 0.1 };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains(r#""type":"dodge""#));
        assert!(json.contains(r#""by":"species""#));
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn jitter_round_trip_with_seed() {
        let p = PositionAdjust::Jitter { axis: JitterAxis::X, width: 0.3, seed: Some(42) };
        let json = serde_json::to_string(&p).unwrap();
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn stack_normalize_round_trip() {
        let p = PositionAdjust::Stack { by: Some("hue".into()), offset: StackOffset::Normalize };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains(r#""offset":"normalize""#));
        let parsed: PositionAdjust = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn dodge_default_padding_round_trips() {
        let json = r#"{"type":"dodge"}"#;
        let parsed: PositionAdjust = serde_json::from_str(json).unwrap();
        match parsed {
            PositionAdjust::Dodge { padding, by } => {
                assert!((padding - 0.05).abs() < 1e-12);
                assert!(by.is_none());
            }
            _ => panic!("expected Dodge"),
        }
    }
}
```

- [ ] **Step 2: Wire `pub mod position;` in `spec/mod.rs`**

```rust
pub mod position;
```

- [ ] **Step 3: Add `position` field to `Layer`**

Edit `crates/ferrum-core/src/spec/layer.rs`. Insert in struct:

```rust
pub struct Layer {
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<TransformSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mark_style: Option<crate::spec::mark_style::MarkKwargsSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub data_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub position: Option<crate::spec::position::PositionAdjust>,    // NEW
}
```

Add a test:
```rust
#[test]
fn layer_position_round_trips() {
    use crate::spec::position::PositionAdjust;
    let layer = Layer {
        mark: Mark::Bar,
        encoding: Encoding::default(),
        transforms: Vec::new(), mark_style: None, data_source: None,
        position: Some(PositionAdjust::Dodge { by: Some("g".into()), padding: 0.05 }),
    };
    let json = serde_json::to_string(&layer).unwrap();
    assert!(json.contains(r#""position""#));
    let parsed: Layer = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, layer);
}

#[test]
fn layer_position_none_omits_from_json() {
    let layer = Layer {
        mark: Mark::Bar, encoding: Encoding::default(),
        transforms: Vec::new(), mark_style: None, data_source: None,
        position: None,
    };
    let json = serde_json::to_string(&layer).unwrap();
    assert!(!json.contains("position"), "position=None must be omitted: {json}");
}
```

- [ ] **Step 4: Add `position` field to `ChartSpec`**

Edit `crates/ferrum-core/src/spec/chart.rs`. Insert field:

```rust
pub struct ChartSpec {
    /* existing fields ... */
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<crate::spec::position::PositionAdjust>,    // NEW
}
```

Update `#[new]` signature: add `position = None` kwarg; coerce a Python dict argument by `serde_json::from_value` round-trip (mirror existing `mark_style` coercion):

```rust
#[pyo3(signature = (
    *, mark, x = None, y = None, color = None,
    size = None, shape = None, opacity = None,
    x2 = None, y2 = None,
    data = None, transforms = None,
    layers = None, coord = None, facet = None, mark_style = None,
    position = None,        // NEW
))]
fn new(
    /* existing args ... */
    position: Option<&Bound<'_, PyAny>>,
) -> PyResult<Self> {
    /* existing body ... */
    let position = match position {
        None => None,
        Some(obj) => {
            let json_module = obj.py().import("json")?;
            let s: String = json_module.call_method1("dumps", (obj,))?.extract()?;
            Some(serde_json::from_str(&s).map_err(|e| PyValueError::new_err(format!("position: {e}")))?)
        }
    };
    Ok(ChartSpec { /* ..., */ position })
}
```

Add a getter so Python can read it back:

```rust
#[getter]
fn position(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
    match &self.position {
        None => Ok(None),
        Some(p) => {
            let s = serde_json::to_string(p).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let json_module = py.import("json")?;
            let val = json_module.call_method1("loads", (s,))?;
            Ok(Some(val.into()))
        }
    }
}
```

Add a chart-level round-trip test:
```rust
#[test]
fn chart_spec_position_round_trips() {
    /* construct ChartSpec with position=Some(Identity); to_json; from_json; assert equal */
}
```

- [ ] **Step 5: Build + tests + commit**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
```
Expected: `≥482 passed` (475 + 5 position-spec tests + 2 layer-position tests + 1 chart-spec round-trip).

```bash
git add crates/ferrum-core/src/spec/position.rs \
        crates/ferrum-core/src/spec/{mod,layer,chart}.rs
git commit -m "feat(phase-9c): add PositionAdjust enum + Layer.position + ChartSpec.position fields"
```

---

### Task 20: `Identity` + `Dodge` (Rust render pass + Python value classes)

**Files:**
- Create: `crates/ferrum-core/src/render/position.rs` (the per-row rewrite pass)
- Modify: `crates/ferrum-core/src/render/mod.rs` (call `apply_position` after scale_resolve, before draw)
- Create: `src/ferrum/position.py` (Python value classes; eligibility matrix)
- Modify: `src/ferrum/__init__.py` (re-export `Identity`, `Dodge`)
- Modify: `src/ferrum/chart.py` (accept `position=` kwarg on eligible mark methods)
- Modify: `src/ferrum/composition.py` (Layer dict carries `position`)
- Create: `tests/test_phase_9_position.py` (Identity + Dodge tests)

**Goal:** Implement the per-row coordinate-rewrite pass for `Identity` (no-op) and `Dodge`. Wire it into the render loop so `mark_bar(position=Dodge(by="hue"))` produces side-by-side bars.

- [ ] **Step 1: Create `src/ferrum/position.py`**

```python
"""Phase 9c — position adjustments (Identity, Dodge, Jitter, Stack).

These are immutable Python value classes. They serialize to a `{"type": "<kind>", ...}`
dict consumed by Rust ChartSpec.position / Layer.position. Eligibility per-mark
is enforced at chart-build time via `validate_position_eligibility(mark, position)`.

The eligibility matrix mirrors the design spec §6.4:

  Identity: every mark.
  Dodge:    bar, point, box, swarm, violin, errorbar, errorband, ribbon.
  Jitter:   point, swarm, tick.
  Stack:    bar, area, ribbon.
"""
from __future__ import annotations
from typing import Optional


# ---- Eligibility matrix ----

_DODGE_ELIGIBLE = frozenset([
    "bar", "point", "box", "boxplot", "swarm", "violin",
    "errorbar", "errorband", "ribbon",
])
_JITTER_ELIGIBLE = frozenset(["point", "swarm", "tick"])
_STACK_ELIGIBLE = frozenset(["bar", "area", "ribbon"])


# ---- Value classes ----

class Identity:
    """Explicit no-op position adjustment.

    Distinct from `position=None` (the default which means 'no adjustment
    declared at all') in that Identity is part of the spec — round-trips through
    JSON. Useful when constructing layered charts from sugar functions that want
    to be explicit about not stacking/dodging.
    """
    __slots__ = ()

    def to_spec_dict(self) -> dict:
        return {"type": "identity"}

    def __repr__(self) -> str:
        return "Identity()"

    def __eq__(self, other) -> bool:
        return isinstance(other, Identity)

    def __hash__(self) -> int:
        return hash("Identity")


class Dodge:
    """Side-by-side dodge across the `by` channel (defaults to color/fill).

    `padding` is the gap between dodged groups as a fraction of the band width.
    """
    __slots__ = ("by", "padding")

    def __init__(self, by: Optional[str] = None, *, padding: float = 0.05) -> None:
        if not (0.0 <= padding < 1.0):
            raise ValueError(f"Dodge: padding must be in [0, 1); got {padding}")
        object.__setattr__(self, "by", by)
        object.__setattr__(self, "padding", padding)

    def to_spec_dict(self) -> dict:
        d: dict = {"type": "dodge", "padding": self.padding}
        if self.by is not None:
            d["by"] = self.by
        return d

    def __setattr__(self, name, value):
        raise AttributeError(f"Dodge is immutable; cannot set {name!r}")

    def __repr__(self) -> str:
        return f"Dodge(by={self.by!r}, padding={self.padding})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, Dodge)
                and self.by == other.by and self.padding == other.padding)

    def __hash__(self) -> int:
        return hash(("Dodge", self.by, self.padding))


# ---- Eligibility validator ----

def validate_position_eligibility(mark_name: str, position) -> None:
    """Raise TypeError if `mark_name` does not accept `position`.

    Called by Chart.mark_<name>(position=...) at construction time.
    """
    if position is None:
        return
    if isinstance(position, Identity):
        return  # all marks accept Identity
    if isinstance(position, Dodge):
        eligible = _DODGE_ELIGIBLE
        kind = "Dodge"
    elif type(position).__name__ == "Jitter":  # forward declaration; Task 21
        eligible = _JITTER_ELIGIBLE
        kind = "Jitter"
    elif type(position).__name__ == "Stack":   # forward declaration; Task 22
        eligible = _STACK_ELIGIBLE
        kind = "Stack"
    else:
        raise TypeError(f"unknown position adjustment: {type(position).__name__}")
    if mark_name not in eligible:
        raise TypeError(
            f"mark_{mark_name} does not accept {kind}; "
            f"eligible marks: {sorted(eligible)}"
        )
```

- [ ] **Step 2: Re-export `Identity`, `Dodge` from `__init__.py`**

```python
from ferrum.position import Identity, Dodge
```

Add to `__all__`.

- [ ] **Step 3: Failing tests in `tests/test_phase_9_position.py`**

```python
"""Phase 9c position adjustment tests."""
import pytest
import polars as pl
import ferrum as fe
from ferrum import Identity, Dodge


class TestIdentity:
    def test_to_spec_dict(self):
        assert Identity().to_spec_dict() == {"type": "identity"}

    def test_immutable(self):
        # __slots__ = () prevents any attribute assignment.
        with pytest.raises(AttributeError):
            Identity().foo = 1   # type: ignore

    def test_equality_and_hash(self):
        assert Identity() == Identity()
        assert hash(Identity()) == hash(Identity())


class TestDodge:
    def test_to_spec_dict_with_by(self):
        d = Dodge(by="species", padding=0.1)
        assert d.to_spec_dict() == {"type": "dodge", "padding": 0.1, "by": "species"}

    def test_to_spec_dict_no_by(self):
        d = Dodge()
        assert d.to_spec_dict() == {"type": "dodge", "padding": 0.05}

    def test_immutable(self):
        d = Dodge(by="g")
        with pytest.raises(AttributeError):
            d.by = "h"

    def test_invalid_padding_errors(self):
        with pytest.raises(ValueError, match="padding"):
            Dodge(padding=1.5)
        with pytest.raises(ValueError, match="padding"):
            Dodge(padding=-0.1)


class TestPositionEligibility:
    def test_identity_accepted_by_all_marks(self):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        # Should not raise for any mark.
        for mark_name in ("bar", "point", "rule", "line", "tick", "rect"):
            method = getattr(fe.Chart(df), f"mark_{mark_name}")
            method(position=Identity()).encode(x="x", y="y")

    def test_dodge_rejected_by_line(self):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4], "g": ["a", "b"]})
        with pytest.raises(TypeError, match="Dodge"):
            fe.Chart(df).mark_line(position=Dodge(by="g"))

    def test_dodge_accepted_by_bar(self):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4], "g": ["a", "b"]})
        chart = fe.Chart(df).mark_bar(position=Dodge(by="g")).encode(x="x", y="y")
        spec = chart.to_spec()
        # The chart's position field should round-trip in JSON.
        import json
        d = json.loads(spec.to_json())
        assert d.get("position", {}).get("type") == "dodge"
        assert d.get("position", {}).get("by") == "g"


@pytest.mark.parametrize("hue_field,n_groups,categories", [
    ("g", 2, ["a", "b"]),
])
def test_dodge_renders_side_by_side(hue_field, n_groups, categories):
    """Ensure mark_bar(position=Dodge) produces n_categories × n_groups bars."""
    rows = []
    for cat in ("X", "Y", "Z"):
        for g in categories:
            rows.append({"cat": cat, "g": g, "v": ord(cat) + len(g)})
    df = pl.DataFrame(rows)
    chart = fe.Chart(df).mark_bar(position=Dodge(by=hue_field)).encode(
        x="cat", y="v", color=hue_field
    )
    svg = chart.show_svg()
    assert "<svg" in svg
    # Render must produce one rect per (cat, group) — 6 expected.
    assert svg.count("<rect") >= 6
```

- [ ] **Step 4: Run tests; expect failures**

```bash
uv run --no-sync pytest tests/test_phase_9_position.py -v 2>&1 | tail -25
```
Expected: import + position-eligibility tests pass; `mark_bar(position=...)` and the dodge-render test fail (chart.py + render pass not yet wired).

- [ ] **Step 5: Wire `position=` kwarg into eligible Chart mark methods**

Edit `src/ferrum/chart.py`. Find the existing `_set_mark` method (line 234). Modify it to accept and store `position`:

```python
def _set_mark(self, name: str, *, position=None, **kwargs: Any) -> "Chart":
    if position is not None:
        from ferrum.position import validate_position_eligibility
        validate_position_eligibility(name, position)
    m = MarkBase(name, **kwargs)
    new = self._clone()
    new._mark = name
    new._mark_kwargs = m.to_mark_kwargs_dict()
    new._position = position
    return new
```

Add `_position` to `__slots__` and `__init__` and `_clone`:

```python
__slots__ = (
    /* existing */, "_position",
)

def __init__(self, ..., ):
    /* existing */
    self._position = None

def _clone(self) -> "Chart":
    /* existing */
    new._position = self._position
    return new
```

Update `to_spec`: pass `position` to `ChartSpec(...)` constructor when set:

```python
def to_spec(self):
    resolved = self._resolve_pending()
    /* existing kw build */
    if resolved._position is not None:
        kw["position"] = resolved._position.to_spec_dict()
    return ChartSpec(**kw)
```

For mark methods that take `**kwargs` (like `mark_bar`), the `position=` is naturally pulled out by `_set_mark`. For mark methods with explicit signatures (composite marks), add `position=None` to the signature.

- [ ] **Step 6: Create `crates/ferrum-core/src/render/position.rs`**

```rust
//! Phase 9c — position-adjustment render pass.
//!
//! Rewrites a layer's RecordBatch *data values* (not pixel coords) per the
//! PositionAdjust on the layer. Runs AFTER scale_resolve (so we know ordinal
//! bandwidth or continuous-x median spacing) but BEFORE mark drawing. The
//! adjusted RecordBatch is then passed to draw::dispatch_mark in place of the
//! original.

use std::collections::HashMap;

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

use crate::render::scale_resolve::ResolvedScales;
use crate::spec::position::PositionAdjust;

/// Apply a position adjustment to a layer batch, returning a new batch with
/// rewritten coordinate columns. Returns a clone of the input unchanged if
/// `position` is None or Identity, or if the adjustment doesn't apply (e.g.,
/// Dodge with no group channel set).
pub(crate) fn apply_position(
    batch: &RecordBatch,
    position: Option<&PositionAdjust>,
    scales: &ResolvedScales,
    encoding: &crate::spec::encoding::Encoding,
) -> Result<RecordBatch, crate::render::RenderError> {
    let Some(p) = position else { return Ok(batch.clone()); };
    match p {
        PositionAdjust::Identity => Ok(batch.clone()),
        PositionAdjust::Dodge { by, padding } => apply_dodge(batch, by.as_deref(), *padding, scales, encoding),
        // Jitter and Stack land in Tasks 21 and 22.
        PositionAdjust::Jitter { .. } => Ok(batch.clone()),
        PositionAdjust::Stack { .. } => Ok(batch.clone()),
    }
}

fn apply_dodge(
    batch: &RecordBatch,
    by_field: Option<&str>,
    padding: f64,
    scales: &ResolvedScales,
    encoding: &crate::spec::encoding::Encoding,
) -> Result<RecordBatch, crate::render::RenderError> {
    // Resolve the `by` column. Default to the color encoding's field if `by` is None.
    let by_col_name = match by_field {
        Some(s) => s.to_string(),
        None => match &encoding.color {
            Some(c) => c.field.clone(),
            None => return Ok(batch.clone()),    // no by-channel; nothing to dodge
        },
    };
    let by_col_idx = match batch.schema().index_of(&by_col_name) {
        Ok(i) => i,
        Err(_) => return Ok(batch.clone()),
    };
    let by_arr = batch.column(by_col_idx).as_any().downcast_ref::<StringArray>()
        .ok_or_else(|| crate::render::RenderError::Other(format!(
            "Dodge: by-column '{by_col_name}' must be Utf8")))?;

    // Resolve x column (the axis being dodged).
    let x_field = encoding.x.as_ref()
        .ok_or_else(|| crate::render::RenderError::Other("Dodge: x encoding required".into()))?;
    let x_col_idx = batch.schema().index_of(&x_field.field).map_err(|_| {
        crate::render::RenderError::Other(format!("Dodge: x column '{}' not found", x_field.field))
    })?;
    let is_ordinal_x = batch.schema().field(x_col_idx).data_type() != &DataType::Float64;
    if is_ordinal_x {
        // Ordinal x — bandwidth comes from OrdinalScale; offsets are pixel-space.
        // Algorithm:
        //   1. Resolve OrdinalScale to get bandwidth_px (the per-band width in pixels).
        //   2. For each row, look up its category's center pixel via scales.x.to_pixel_str(cat).
        //   3. Compute per-group sub-band pixel offset (same formula as continuous, but
        //      working in pixel space).
        //   4. INJECT TWO SYNTHETIC COLUMNS into the output batch:
        //        __pos_x_offset__: Float64 — pixel offset to add to bar's resolved x
        //        __pos_y_offset__: Float64 — zero (only x dodging here)
        //   5. The bar/point/box/swarm/violin/errorbar/errorband/ribbon mark drawers,
        //      after computing their resolved pixel positions, check for the synthetic
        //      columns and add the per-row offset.
        return apply_dodge_ordinal(batch, x_col_idx, by_arr, padding, scales);
    }
    let x_arr = batch.column(x_col_idx).as_any().downcast_ref::<Float64Array>().unwrap();

    // 1. Compute median spacing of unique x values (bandwidth proxy for continuous x).
    let mut uniques: Vec<f64> = (0..x_arr.len())
        .filter(|i| !x_arr.is_null(*i)).map(|i| x_arr.value(i)).collect();
    uniques.sort_by(|a, b| a.partial_cmp(b).unwrap());
    uniques.dedup();
    if uniques.len() < 2 {
        return Ok(batch.clone());
    }
    let mut diffs: Vec<f64> = uniques.windows(2).map(|w| w[1] - w[0]).collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let bandwidth = diffs[diffs.len() / 2];   // median

    // 2. Determine group order from `by` channel (first-appearance order).
    let mut groups_in_order: Vec<String> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for i in 0..by_arr.len() {
        let g = by_arr.value(i).to_string();
        if !seen.contains_key(&g) {
            seen.insert(g.clone(), groups_in_order.len());
            groups_in_order.push(g);
        }
    }
    let n_groups = groups_in_order.len();
    if n_groups <= 1 {
        return Ok(batch.clone());
    }

    // 3. Per-group offset within the band:
    //    sub_band_width = bandwidth * (1 - 2*padding) / n_groups
    //    offset(group_idx) = -bandwidth/2 + bandwidth*padding + sub_band_width * (group_idx + 0.5)
    let pad_total = bandwidth * padding * 2.0;
    let sub_band = (bandwidth - pad_total) / n_groups as f64;

    // 4. Rewrite x column: for each row, x_new = x_old - bandwidth/2 + bandwidth*padding + sub_band*(group_idx + 0.5)
    let mut new_x = Vec::with_capacity(x_arr.len());
    for i in 0..x_arr.len() {
        let g = by_arr.value(i);
        let group_idx = *seen.get(g).unwrap();
        let offset = -bandwidth / 2.0 + bandwidth * padding + sub_band * (group_idx as f64 + 0.5);
        new_x.push(x_arr.value(i) + offset);
    }

    // 5. Build new batch with the x column replaced (continuous-x branch).
    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols[x_col_idx] = Arc::new(Float64Array::from(new_x));
    let schema = batch.schema();
    RecordBatch::try_new(schema, cols)
        .map_err(|e| crate::render::RenderError::Other(format!("Dodge: {e}")))
}

/// Ordinal-x Dodge — operates in pixel space because the categorical x cannot
/// be rewritten in data space. Injects two synthetic Float64 columns named
/// `__pos_x_offset__` and `__pos_y_offset__` (the latter is always 0 for Dodge).
/// Mark drawers (bar/point/box/swarm/violin/errorbar/errorband/ribbon) read
/// these columns post-scale-resolve and add them to the rendered position.
fn apply_dodge_ordinal(
    batch: &RecordBatch,
    x_col_idx: usize,
    by_arr: &StringArray,
    padding: f64,
    scales: &ResolvedScales,
) -> Result<RecordBatch, crate::render::RenderError> {
    use crate::scale::ordinal::OrdinalScale;
    let schema = batch.schema();

    // Pull bandwidth from OrdinalScale. ResolvedScales' x is enum ScaleKind;
    // match Ordinal arm or fall back to no-op.
    let bandwidth_px = match scales.x.as_ref() {
        Some(crate::render::scale_resolve::ScaleKind::Ordinal(s)) => s.bandwidth(),
        _ => return Ok(batch.clone()),  // not ordinal — handled by continuous branch
    };

    // Group order from `by` (first-appearance).
    let mut group_order: Vec<String> = Vec::new();
    let mut group_idx: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for i in 0..by_arr.len() {
        let g = by_arr.value(i).to_string();
        if !group_idx.contains_key(&g) {
            group_idx.insert(g.clone(), group_order.len());
            group_order.push(g);
        }
    }
    let n_groups = group_order.len();
    if n_groups <= 1 {
        return Ok(batch.clone());
    }

    let pad_total = bandwidth_px * padding * 2.0;
    let sub_band = (bandwidth_px - pad_total) / n_groups as f64;

    let n = by_arr.len();
    let mut x_offsets: Vec<f64> = Vec::with_capacity(n);
    let mut y_offsets: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let g = by_arr.value(i);
        let gi = *group_idx.get(g).unwrap();
        let offset_px = -bandwidth_px / 2.0 + bandwidth_px * padding + sub_band * (gi as f64 + 0.5);
        x_offsets.push(offset_px);
        y_offsets.push(0.0);
    }

    // Append two synthetic columns.
    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols.push(Arc::new(Float64Array::from(x_offsets)));
    cols.push(Arc::new(Float64Array::from(y_offsets)));

    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new("__pos_x_offset__", DataType::Float64, false));
    fields.push(Field::new("__pos_y_offset__", DataType::Float64, false));
    let new_schema = Arc::new(Schema::new(fields));

    RecordBatch::try_new(new_schema, cols)
        .map_err(|e| crate::render::RenderError::Other(format!("Dodge ordinal: {e}")))
}
```

**Note on `OrdinalScale.bandwidth()`:** verify the existing API in `crates/ferrum-core/src/scale/ordinal.rs`. If the method has a different name, adjust the call. Phase 6/7 already needed this for axis tick placement, so the accessor should exist.

`RenderError::Other` may not exist on the existing enum — if so, use the closest existing variant or add a new `Other(String)` variant. Verify by inspecting `crates/ferrum-core/src/render/mod.rs` lines around the `RenderError` enum definition.

- [ ] **Step 6.5: Wire ordinal-x offset hook into mark drawers**

The synthetic columns `__pos_x_offset__` / `__pos_y_offset__` are read by mark drawers when present and added to the rendered pixel position. The drawers needing this hook (per spec eligibility matrix for Dodge):

```
crates/ferrum-core/src/render/marks/bar.rs
crates/ferrum-core/src/render/marks/point.rs
crates/ferrum-core/src/render/marks/rect.rs    (for box/violin which use rect)
crates/ferrum-core/src/render/marks/rule.rs    (for errorbar — emit rule with offset)
crates/ferrum-core/src/render/marks/tick.rs    (for swarm overlap and box-median tick)
crates/ferrum-core/src/render/marks/ribbon.rs  (for errorband + ribbon)
crates/ferrum-core/src/render/marks/polygon.rs (for violin polygon)
crates/ferrum-core/src/render/marks/area.rs    (only Stack uses this; included for completeness)
```

Add this helper at the top of `render/marks/mod.rs` (or `render/draw.rs`):

```rust
/// Read per-row pixel offsets from synthetic `__pos_x_offset__` / `__pos_y_offset__`
/// columns. Returns (Vec<f64>, Vec<f64>) of zeros-by-default when columns absent.
pub(crate) fn read_position_offsets(batch: &arrow::record_batch::RecordBatch) -> (Vec<f64>, Vec<f64>) {
    use arrow::array::Float64Array;
    let n = batch.num_rows();
    let xo = batch.schema().index_of("__pos_x_offset__").ok()
        .and_then(|i| batch.column(i).as_any().downcast_ref::<Float64Array>().map(|a|
            (0..a.len()).map(|j| a.value(j)).collect::<Vec<f64>>()))
        .unwrap_or_else(|| vec![0.0; n]);
    let yo = batch.schema().index_of("__pos_y_offset__").ok()
        .and_then(|i| batch.column(i).as_any().downcast_ref::<Float64Array>().map(|a|
            (0..a.len()).map(|j| a.value(j)).collect::<Vec<f64>>()))
        .unwrap_or_else(|| vec![0.0; n]);
    (xo, yo)
}
```

In each affected drawer's `draw` function, near the top:

```rust
let (x_offsets, y_offsets) = crate::render::marks::read_position_offsets(ctx.batch);
// ... after computing per-row resolved pixel x/y:
let px = resolved_px + x_offsets[i];
let py = resolved_py + y_offsets[i];
```

The change to each drawer is ≤5 LOC. Cargo test for each drawer should still pass (when offsets are zero, the rendered output is byte-identical to pre-Phase-9 behavior).

- [ ] **Step 7: Wire `apply_position` into `render/mod.rs`**

In `render/mod.rs::render_svg` per-layer loop (around line 262), after `scale_resolve::resolve_scales_with_outputs` and before `let layer_spec = ChartSpec { ... }`, insert:

```rust
let position = layer.position.as_ref()
    .or(spec.position.as_ref());     // chart-level fallback for non-layered charts
let layer_batch_adjusted = if position.is_some() {
    crate::render::position::apply_position(layer_batch, position, &scales, &layer.encoding)?
} else {
    layer_batch.clone()
};
```

Replace the subsequent `let layer_batch = &layer_batches[li];` reference with `let layer_batch = &layer_batch_adjusted;`.

`pub(crate) mod position;` in `render/mod.rs` near other `pub(crate) mod` lines.

- [ ] **Step 8: Layer dict carries `position` from Python**

Edit `src/ferrum/chart.py::_build_layers_list` (line 767+). For each layer dict, if `_position` was stored on the layer (in `__add__` overlay logic), serialize via `to_spec_dict()`:

```python
position = layer.get("position")
if position is not None:
    layer_dict["position"] = position.to_spec_dict() if hasattr(position, "to_spec_dict") else position
```

Layered charts capture `_position` per-layer in `__add__`:

```python
new._layers = [
    {
        "mark": lhs._mark, "encoding": dict(lhs._encoding),
        "transforms": list(lhs._transforms),
        "mark_style": dict(lhs._mark_kwargs),
        "position": lhs._position,    # NEW
    },
    ...
]
```

- [ ] **Step 9: Build + run all tests**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
uv run --no-sync pytest tests/test_phase_9_position.py -v 2>&1 | tail -15
```
Expected: cargo `≥482 passed`; pytest all 9-position tests pass.

- [ ] **Step 10: Commit**

```bash
git add crates/ferrum-core/src/render/position.rs \
        crates/ferrum-core/src/render/mod.rs \
        src/ferrum/position.py src/ferrum/chart.py src/ferrum/__init__.py \
        tests/test_phase_9_position.py
git commit -m "feat(phase-9c): add Identity + Dodge position adjustments"
```

---

### Task 21: `Jitter` position adjustment (twox-hash seed fallback)

**Files:**
- Modify: `crates/ferrum-core/src/render/position.rs` (add `apply_jitter`)
- Modify: `src/ferrum/position.py` (add `Jitter` class)
- Modify: `src/ferrum/__init__.py` (re-export)
- Modify: `tests/test_phase_9_position.py` (`TestJitter`)

**Goal:** Implement `Jitter(axis="x", width=0.4, seed=None)`. Per-row noise drawn from uniform `[-width/2, +width/2]`. `seed=None` → ChaCha8Rng seeded with `twox_hash::xxh3::hash64(format!("{x_value}|{y_value}|{group_value}").as_bytes())` for deterministic byte-equal output.

- [ ] **Step 1: Failing tests** — append `TestJitter` to `test_phase_9_position.py`:

```python
from ferrum import Jitter   # to be added in Step 2


class TestJitter:
    def test_construction(self):
        j = Jitter(axis="x", width=0.5, seed=42)
        assert j.axis == "x"
        assert j.width == 0.5
        assert j.seed == 42

    def test_default_axis_x_width_0_4(self):
        j = Jitter()
        assert j.axis == "x"
        assert j.width == 0.4
        assert j.seed is None

    def test_to_spec_dict_with_seed(self):
        j = Jitter(axis="both", width=0.3, seed=7)
        d = j.to_spec_dict()
        assert d == {"type": "jitter", "axis": "both", "width": 0.3, "seed": 7}

    def test_to_spec_dict_no_seed(self):
        j = Jitter()
        d = j.to_spec_dict()
        assert d == {"type": "jitter", "axis": "x", "width": 0.4}

    def test_invalid_axis_errors(self):
        with pytest.raises(ValueError, match="axis"):
            Jitter(axis="invalid")

    def test_eligibility_rejects_bar(self):
        df = pl.DataFrame({"x": [1, 2]})
        with pytest.raises(TypeError, match="Jitter"):
            fe.Chart(df).mark_bar(position=Jitter())

    def test_eligibility_accepts_point(self):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        fe.Chart(df).mark_point(position=Jitter()).encode(x="x", y="y")

    def test_renders_with_explicit_seed_byte_identical(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        c = fe.Chart(df).mark_point(position=Jitter(width=0.3, seed=42)).encode(x="x", y="y")
        a = c.show_svg()
        b = c.show_svg()
        assert a == b   # identical seed → identical output

    def test_renders_with_seed_none_byte_identical(self):
        # seed=None falls back to xxh3 hash of (x, y, group) per row — also byte-deterministic.
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        c = fe.Chart(df).mark_point(position=Jitter(width=0.3)).encode(x="x", y="y")
        a = c.show_svg()
        b = c.show_svg()
        assert a == b

    def test_different_seeds_produce_different_output(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        a = fe.Chart(df).mark_point(position=Jitter(width=0.5, seed=1)).encode(x="x", y="y").show_svg()
        b = fe.Chart(df).mark_point(position=Jitter(width=0.5, seed=2)).encode(x="x", y="y").show_svg()
        assert a != b
```

- [ ] **Step 2: Add `Jitter` to `src/ferrum/position.py`**

```python
_VALID_JITTER_AXES = {"x", "y", "both"}


class Jitter:
    """Random per-row noise on x and/or y; deterministic given a seed."""
    __slots__ = ("axis", "width", "seed")

    def __init__(self, axis: str = "x", *, width: float = 0.4, seed: int | None = None) -> None:
        if axis not in _VALID_JITTER_AXES:
            raise ValueError(f"Jitter: axis must be 'x'|'y'|'both'; got '{axis}'")
        if width <= 0.0:
            raise ValueError(f"Jitter: width must be > 0; got {width}")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "seed", seed)

    def to_spec_dict(self) -> dict:
        d: dict = {"type": "jitter", "axis": self.axis, "width": self.width}
        if self.seed is not None:
            d["seed"] = self.seed
        return d

    def __setattr__(self, name, value):
        raise AttributeError(f"Jitter is immutable; cannot set {name!r}")

    def __repr__(self) -> str:
        return f"Jitter(axis={self.axis!r}, width={self.width}, seed={self.seed})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, Jitter)
                and self.axis == other.axis and self.width == other.width and self.seed == other.seed)

    def __hash__(self) -> int:
        return hash(("Jitter", self.axis, self.width, self.seed))
```

Re-export in `__init__.py`.

- [ ] **Step 3: Implement `apply_jitter` in `render/position.rs`**

Replace the placeholder `Jitter { .. } => Ok(batch.clone())` arm with a call to `apply_jitter`. Implementation:

```rust
fn apply_jitter(
    batch: &RecordBatch,
    axis: &crate::spec::position::JitterAxis,
    width: f64,
    seed: Option<u64>,
    scales: &ResolvedScales,
    encoding: &crate::spec::encoding::Encoding,
) -> Result<RecordBatch, crate::render::RenderError> {
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use twox_hash::xxh3;

    let x_idx = encoding.x.as_ref().and_then(|e| batch.schema().index_of(&e.field).ok());
    let y_idx = encoding.y.as_ref().and_then(|e| batch.schema().index_of(&e.field).ok());

    let n = batch.num_rows();
    let mut new_x: Vec<f64> = Vec::with_capacity(n);
    let mut new_y: Vec<f64> = Vec::with_capacity(n);

    let do_x = matches!(axis, crate::spec::position::JitterAxis::X | crate::spec::position::JitterAxis::Both);
    let do_y = matches!(axis, crate::spec::position::JitterAxis::Y | crate::spec::position::JitterAxis::Both);

    for i in 0..n {
        let xv = x_idx.and_then(|j| batch.column(j).as_any().downcast_ref::<Float64Array>().map(|a| a.value(i))).unwrap_or(f64::NAN);
        let yv = y_idx.and_then(|j| batch.column(j).as_any().downcast_ref::<Float64Array>().map(|a| a.value(i))).unwrap_or(f64::NAN);

        // Per-row seed: explicit seed offset by row, or hash of (x,y) for seed=None.
        let row_seed = match seed {
            Some(s) => s.wrapping_add(i as u64),
            None => {
                let key = format!("{xv}|{yv}");
                xxh3::hash64(key.as_bytes())
            }
        };
        let mut rng = ChaCha8Rng::seed_from_u64(row_seed);

        // Uniform [-width/2, +width/2]: use RngCore.next_u64 → f64 in [0,1) → scale.
        let u = (rng.next_u64() as f64) / (u64::MAX as f64);
        let noise = (u - 0.5) * width;

        new_x.push(if do_x { xv + noise } else { xv });
        // Re-draw a fresh u for y to avoid correlation with x-noise.
        let u2 = (rng.next_u64() as f64) / (u64::MAX as f64);
        let noise_y = (u2 - 0.5) * width;
        new_y.push(if do_y { yv + noise_y } else { yv });
    }

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    if let Some(j) = x_idx { cols[j] = Arc::new(Float64Array::from(new_x)); }
    if let Some(j) = y_idx { cols[j] = Arc::new(Float64Array::from(new_y)); }
    let schema = batch.schema();
    RecordBatch::try_new(schema, cols).map_err(|e| crate::render::RenderError::Other(format!("Jitter: {e}")))
}
```

Wire the dispatch in `apply_position`:

```rust
PositionAdjust::Jitter { axis, width, seed } =>
    apply_jitter(batch, axis, *width, *seed, scales, encoding),
```

- [ ] **Step 4: Build + tests + commit**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
uv run --no-sync pytest tests/test_phase_9_position.py -v 2>&1 | tail -15
```
Expected: 8 Jitter-class tests + previous Identity/Dodge tests pass.

```bash
git add crates/ferrum-core/src/render/position.rs \
        src/ferrum/position.py src/ferrum/__init__.py \
        tests/test_phase_9_position.py
git commit -m "feat(phase-9c): add Jitter position adjustment with twox-hash seed fallback"
```

---

### Task 22: `Stack` position adjustment (zero / normalize / center)

**Files:**
- Modify: `render/position.rs` (add `apply_stack`)
- Modify: `src/ferrum/position.py` (add `Stack` class)
- Modify: `__init__.py`, `tests/test_phase_9_position.py`

**Goal:** Implement `Stack(by=None, offset="zero")`. Group rows by `by` channel at each x; cumulative-sum y within group; rewrite y. `offset="zero"` (standard), `"normalize"` (100% stack — divide each row's y by per-x total before cumulating), `"center"` (streamgraph; symmetric around 0).

- [ ] **Step 1: Failing tests in `TestStack`** (append to `test_phase_9_position.py`):

```python
from ferrum import Stack


class TestStack:
    def test_construction_default(self):
        s = Stack()
        assert s.by is None
        assert s.offset == "zero"

    def test_to_spec_dict_normalize(self):
        s = Stack(by="hue", offset="normalize")
        assert s.to_spec_dict() == {"type": "stack", "by": "hue", "offset": "normalize"}

    def test_invalid_offset_errors(self):
        with pytest.raises(ValueError, match="offset"):
            Stack(offset="bogus")

    def test_eligibility_rejects_point(self):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        with pytest.raises(TypeError, match="Stack"):
            fe.Chart(df).mark_point(position=Stack())

    def test_eligibility_accepts_bar_area_ribbon(self):
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        fe.Chart(df).mark_bar(position=Stack()).encode(x="x", y="y")
        fe.Chart(df).mark_area(position=Stack()).encode(x="x", y="y")

    def test_stack_zero_renders(self):
        df = pl.DataFrame({
            "x": [1, 2, 1, 2],
            "y": [10.0, 20.0, 5.0, 8.0],
            "g": ["a", "a", "b", "b"],
        })
        chart = fe.Chart(df).mark_bar(position=Stack(by="g")).encode(x="x", y="y", color="g")
        svg = chart.show_svg()
        assert "<svg" in svg

    def test_stack_normalize_total_y_is_one(self):
        # After normalize stack, the topmost rect at each x should reach the
        # same height (representing 100%). We assert the renderer succeeds.
        df = pl.DataFrame({
            "x": [1, 1, 2, 2], "y": [3.0, 7.0, 1.0, 9.0], "g": ["a", "b", "a", "b"],
        })
        chart = fe.Chart(df).mark_bar(position=Stack(by="g", offset="normalize")).encode(
            x="x", y="y", color="g"
        )
        svg = chart.show_svg()
        assert "<svg" in svg
```

- [ ] **Step 2: Add `Stack` class to `position.py`**

```python
_VALID_STACK_OFFSETS = {"zero", "normalize", "center"}


class Stack:
    """Vertical accumulation grouped by `by` channel."""
    __slots__ = ("by", "offset")

    def __init__(self, by: str | None = None, *, offset: str = "zero") -> None:
        if offset not in _VALID_STACK_OFFSETS:
            raise ValueError(f"Stack: offset must be 'zero'|'normalize'|'center'; got '{offset}'")
        object.__setattr__(self, "by", by)
        object.__setattr__(self, "offset", offset)

    def to_spec_dict(self) -> dict:
        d: dict = {"type": "stack", "offset": self.offset}
        if self.by is not None:
            d["by"] = self.by
        return d

    def __setattr__(self, name, value):
        raise AttributeError(f"Stack is immutable; cannot set {name!r}")

    def __repr__(self) -> str:
        return f"Stack(by={self.by!r}, offset={self.offset!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Stack) and self.by == other.by and self.offset == other.offset

    def __hash__(self) -> int:
        return hash(("Stack", self.by, self.offset))
```

Re-export.

- [ ] **Step 3: Implement `apply_stack` in `render/position.rs`**

Algorithm:

```rust
fn apply_stack(
    batch: &RecordBatch,
    by_field: Option<&str>,
    offset: &crate::spec::position::StackOffset,
    scales: &ResolvedScales,
    encoding: &crate::spec::encoding::Encoding,
) -> Result<RecordBatch, crate::render::RenderError> {
    use crate::spec::position::StackOffset;
    use std::collections::BTreeMap;

    let by_name = match by_field {
        Some(s) => s.to_string(),
        None => match &encoding.color {
            Some(c) => c.field.clone(),
            None => return Ok(batch.clone()),
        },
    };
    let by_idx = batch.schema().index_of(&by_name).ok();
    let by_arr_opt = by_idx
        .and_then(|i| batch.column(i).as_any().downcast_ref::<StringArray>());
    let Some(by_arr) = by_arr_opt else { return Ok(batch.clone()); };

    let x_field = encoding.x.as_ref().ok_or_else(||
        crate::render::RenderError::Other("Stack: x encoding required".into()))?;
    let y_field = encoding.y.as_ref().ok_or_else(||
        crate::render::RenderError::Other("Stack: y encoding required".into()))?;
    let xi = batch.schema().index_of(&x_field.field).map_err(|_|
        crate::render::RenderError::Other(format!("Stack: x col '{}' not found", x_field.field)))?;
    let yi = batch.schema().index_of(&y_field.field).map_err(|_|
        crate::render::RenderError::Other(format!("Stack: y col '{}' not found", y_field.field)))?;
    let xa = batch.column(xi).as_any().downcast_ref::<Float64Array>().ok_or_else(||
        crate::render::RenderError::Other("Stack: x must be Float64".into()))?;
    let ya = batch.column(yi).as_any().downcast_ref::<Float64Array>().ok_or_else(||
        crate::render::RenderError::Other("Stack: y must be Float64".into()))?;

    // Collect (x_value → group_order_by_first_appearance → vec of (row_idx, y_value)).
    // First pass: discover group order.
    let mut group_order: Vec<String> = Vec::new();
    let mut group_idx_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for i in 0..ya.len() {
        let g = by_arr.value(i).to_string();
        if !group_idx_map.contains_key(&g) {
            group_idx_map.insert(g.clone(), group_order.len());
            group_order.push(g);
        }
    }

    // Group rows by x, then by group.
    // bins: x_value (as bits) → Vec<(group_idx, row_idx, y)>
    let mut bins: BTreeMap<u64, Vec<(usize, usize, f64)>> = BTreeMap::new();
    for i in 0..ya.len() {
        let g = by_arr.value(i).to_string();
        let gi = *group_idx_map.get(&g).unwrap();
        let xv = xa.value(i);
        let key = xv.to_bits();
        bins.entry(key).or_default().push((gi, i, ya.value(i)));
    }

    // Compute per-x totals (for normalize).
    let totals: std::collections::HashMap<u64, f64> = bins.iter()
        .map(|(k, rows)| (*k, rows.iter().map(|(_, _, y)| y).sum::<f64>()))
        .collect();

    // For each x-bin, sort by group_idx (= categorical bottom-to-top order),
    // then accumulate.
    let mut new_y = vec![0.0_f64; ya.len()];
    for (xkey, rows) in bins.iter_mut() {
        rows.sort_by_key(|(gi, _, _)| *gi);
        let total = totals.get(xkey).copied().unwrap_or(0.0);
        let mut acc = 0.0_f64;
        for (_, row_idx, y) in rows.iter() {
            let normalized = match offset {
                StackOffset::Zero      => *y,
                StackOffset::Normalize => if total != 0.0 { y / total } else { 0.0 },
                StackOffset::Center    => *y,
            };
            acc += normalized;
            new_y[*row_idx] = acc;
        }
        // Center offset: shift all rows in this x-bin so their stack is centered around 0.
        if matches!(offset, StackOffset::Center) {
            let mid = acc / 2.0;
            for (_, row_idx, _) in rows.iter() {
                new_y[*row_idx] -= mid;
            }
        }
    }

    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols[yi] = Arc::new(Float64Array::from(new_y));
    let schema = batch.schema();
    RecordBatch::try_new(schema, cols).map_err(|e|
        crate::render::RenderError::Other(format!("Stack: {e}")))
}
```

Wire dispatch in `apply_position`:

```rust
PositionAdjust::Stack { by, offset } =>
    apply_stack(batch, by.as_deref(), offset, scales, encoding),
```

- [ ] **Step 4: Build + tests + commit**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
uv run --no-sync pytest tests/test_phase_9_position.py -v 2>&1 | tail -15
```
Expected: 7 Stack-class tests pass.

```bash
git add crates/ferrum-core/src/render/position.rs \
        src/ferrum/position.py src/ferrum/__init__.py \
        tests/test_phase_9_position.py
git commit -m "feat(phase-9c): add Stack position adjustment (zero/normalize/center)"
```

---

### Task 23: Mark eligibility-matrix enforcement on all eligible mark methods

**Files:**
- Modify: `src/ferrum/chart.py` (composite + heavy-stat mark methods plumb `position=` kwarg through)
- Modify: `tests/test_phase_9_position.py` (negative tests for each mark × position combo)

**Goal:** Every eligible mark method on `Chart` accepts `position=`. Mark methods that don't accept a particular adjustment raise `TypeError` at construction time. The eligibility matrix is the source of truth in `position.py::validate_position_eligibility`.

- [ ] **Step 1: Edit `chart.py` mark methods**

For each mark method that should accept `position=`:

- **Primitive marks** (already use `_set_mark` which accepts `position=` after Task 20 Step 5): no change needed beyond Task 20.
- **Composite marks** (`mark_boxplot`, `mark_errorbar`, `mark_errorband`, `mark_ribbon`): add explicit `position=None` param to the method signature; pass through to `_pending_stat_mark` payload; resolve in `_resolve_pending` by setting `new._position = kwargs.get("position")`.
- **Heavy stat marks** (`mark_density`, `mark_histogram`, `mark_smooth`, `mark_contour`, `mark_violin`, `mark_qq`, `mark_raster`, `mark_swarm`, `mark_hex`, `mark_function`, `mark_tick`): same pattern — add `position=None`, validate, store on `_position`.

Example for `mark_boxplot` (line 338):

```python
def mark_boxplot(
    self, *,
    extent=1.5, size=None, outliers=True,
    color_field=None, horizontal=False,
    position=None,                                          # NEW
    **mark_kwargs,
) -> "Chart":
    if position is not None:
        from ferrum.position import validate_position_eligibility
        validate_position_eligibility("boxplot", position)
    /* existing body */
    new._position = position                                 # NEW
    return new
```

Touch every mark method (≈12 in chart.py). Verify each by grepping `def mark_` in chart.py and adding `position=None` kwarg + validate + assign.

- [ ] **Step 2: Negative tests**

Add to `test_phase_9_position.py`:

```python
class TestEligibilityMatrix:
    @pytest.mark.parametrize("mark_name,position_class,should_accept", [
        ("bar",       "Identity", True),
        ("bar",       "Dodge",    True),
        ("bar",       "Jitter",   False),
        ("bar",       "Stack",    True),
        ("point",     "Dodge",    True),
        ("point",     "Jitter",   True),
        ("point",     "Stack",    False),
        ("line",      "Dodge",    False),
        ("line",      "Jitter",   False),
        ("line",      "Stack",    False),
        ("rule",      "Dodge",    False),
        ("rule",      "Jitter",   False),
        ("area",      "Stack",    True),
        ("area",      "Dodge",    False),
        ("ribbon",    "Stack",    True),
        ("ribbon",    "Dodge",    True),
        ("tick",      "Jitter",   True),
        ("tick",      "Dodge",    False),
    ])
    def test_eligibility(self, mark_name, position_class, should_accept):
        position_classes = {"Identity": Identity(), "Dodge": Dodge(),
                            "Jitter": Jitter(), "Stack": Stack()}
        pos = position_classes[position_class]
        df = pl.DataFrame({"x": [1, 2], "y": [3.0, 4.0], "y2": [5.0, 6.0], "g": ["a", "b"]})
        method = getattr(fe.Chart(df), f"mark_{mark_name}")
        if should_accept:
            method(position=pos).encode(x="x", y="y", y2="y2")
        else:
            with pytest.raises(TypeError, match=position_class):
                method(position=pos)
```

- [ ] **Step 3: Build + run all tests + commit**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
uv run --no-sync pytest tests/test_phase_9_position.py -v 2>&1 | tail -25
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
```
Expected: all tests pass; cargo unchanged at `≥482 passed`.

```bash
git add src/ferrum/chart.py tests/test_phase_9_position.py
git commit -m "feat(phase-9c): enforce position eligibility matrix on all eligible marks"
```

**End of 9c. cargo test ≥ 482 passed; pytest 9-position tests fully cover the eligibility matrix.**

---

## 9d — New marks

### Task 24: `mark_segment` primitive

**Files:**
- Modify: `crates/ferrum-core/src/spec/mark.rs` (add `Mark::Segment` variant + from_str arm)
- Create: `crates/ferrum-core/src/render/marks/segment.rs`
- Modify: `crates/ferrum-core/src/render/marks/mod.rs` (`pub(crate) mod segment;`)
- Modify: `crates/ferrum-core/src/render/draw.rs` (dispatch arm for `Mark::Segment`)
- Modify: `src/ferrum/chart.py` (replace `mark_segment` stub on line 571 with working method)
- Modify: `src/ferrum/marks/deferred.py` (remove `"segment"` from PHASE_9_PLUS_MARKS)
- Create: `tests/test_phase_9_marks.py` (with `TestMarkSegment`)

**Goal:** Diagonal-capable line segment from `(x, y)` to `(x2, y2)`. Uses existing `X`, `Y`, `X2`, `Y2` encoding channels. Differs from `mark_rule` (axis-aligned only).

- [ ] **Step 1: Failing test**

Create `tests/test_phase_9_marks.py`:

```python
"""Phase 9d new-mark tests."""
import pytest
import polars as pl
import ferrum as fe


class TestMarkSegment:
    def test_segment_no_longer_in_deferred(self):
        from ferrum.marks import PHASE_9_PLUS_MARKS
        assert "segment" not in PHASE_9_PLUS_MARKS

    def test_mark_segment_accepts_x2_y2(self):
        df = pl.DataFrame({
            "x": [0.0, 1.0], "y": [0.0, 1.0],
            "x2": [1.0, 2.0], "y2": [1.0, 2.0],
        })
        chart = fe.Chart(df).mark_segment().encode(x="x", y="y", x2="x2", y2="y2")
        spec = chart.to_spec()
        # Mark name in JSON.
        import json
        d = json.loads(spec.to_json())
        assert d["mark"] == "segment"

    def test_mark_segment_renders_diagonal_line(self):
        df = pl.DataFrame({
            "x": [0.0, 1.0], "y": [0.0, 1.0],
            "x2": [1.0, 2.0], "y2": [2.0, 0.0],
        })
        chart = fe.Chart(df).mark_segment().encode(x="x", y="y", x2="x2", y2="y2")
        svg = chart.show_svg()
        assert "<svg" in svg
        # Two segments → at least 2 <line> elements (or path equivalents).
        assert svg.count("<line") + svg.count("<path") >= 2

    def test_mark_segment_position_only_identity(self):
        from ferrum import Identity, Dodge
        df = pl.DataFrame({"x": [0.0], "y": [0.0], "x2": [1.0], "y2": [1.0]})
        # Identity is fine.
        fe.Chart(df).mark_segment(position=Identity()).encode(x="x", y="y", x2="x2", y2="y2")
        # Dodge is not.
        with pytest.raises(TypeError, match="Dodge"):
            fe.Chart(df).mark_segment(position=Dodge())
```

- [ ] **Step 2: Add `Mark::Segment` to Rust enum**

Edit `crates/ferrum-core/src/spec/mark.rs`. After `Ribbon,`:

```rust
Segment,
```

Add to `as_str` and `FromStr`:

```rust
Mark::Segment => "segment",
```

```rust
"segment" => Ok(Mark::Segment),
```

Update the `unknown mark` error string to include `segment`:

```rust
"unknown mark '{}'; expected one of [point, line, bar, area, rule, text, tick, rect, polygon, image, ribbon, segment]"
```

Add round-trip test in `mark.rs::tests` to cover `Mark::Segment`.

- [ ] **Step 3: Create `render/marks/segment.rs`**

```rust
//! Segment mark — diagonal line from (x, y) to (x2, y2).
//! Distinct from rule (axis-aligned only).

use arrow::array::{Array, Float64Array};

use crate::render::draw::DrawCtx;
use crate::render::svg::SvgBuffer;

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let Some(x_field) = ctx.spec.encoding.x.as_ref() else { return; };
    let Some(y_field) = ctx.spec.encoding.y.as_ref() else { return; };
    let Some(x2_field) = ctx.spec.encoding.x2.as_ref() else { return; };
    let Some(y2_field) = ctx.spec.encoding.y2.as_ref() else { return; };

    let xi = ctx.batch.schema().index_of(&x_field.field).ok();
    let yi = ctx.batch.schema().index_of(&y_field.field).ok();
    let x2i = ctx.batch.schema().index_of(&x2_field.field).ok();
    let y2i = ctx.batch.schema().index_of(&y2_field.field).ok();
    let (Some(xi), Some(yi), Some(x2i), Some(y2i)) = (xi, yi, x2i, y2i) else { return; };

    let xa = ctx.batch.column(xi).as_any().downcast_ref::<Float64Array>();
    let ya = ctx.batch.column(yi).as_any().downcast_ref::<Float64Array>();
    let x2a = ctx.batch.column(x2i).as_any().downcast_ref::<Float64Array>();
    let y2a = ctx.batch.column(y2i).as_any().downcast_ref::<Float64Array>();
    let (Some(xa), Some(ya), Some(x2a), Some(y2a)) = (xa, ya, x2a, y2a) else { return; };

    let stroke = crate::render::svg::Stroke {
        color: ctx.mark_style.color,
        width: ctx.mark_style.stroke_width,
        opacity: ctx.mark_style.opacity,
        dash_pattern: ctx.mark_style.stroke_dash.clone(),
    };

    let n = xa.len().min(ya.len()).min(x2a.len()).min(y2a.len());
    for i in 0..n {
        if xa.is_null(i) || ya.is_null(i) || x2a.is_null(i) || y2a.is_null(i) { continue; }
        let p1x = ctx.scales.x.as_ref().and_then(|s| s.to_pixel_f64(xa.value(i)));
        let p1y = ctx.scales.y.as_ref().and_then(|s| s.to_pixel_f64(ya.value(i)));
        let p2x = ctx.scales.x.as_ref().and_then(|s| s.to_pixel_f64(x2a.value(i)));
        let p2y = ctx.scales.y.as_ref().and_then(|s| s.to_pixel_f64(y2a.value(i)));
        if let (Some(p1x), Some(p1y), Some(p2x), Some(p2y)) = (p1x, p1y, p2x, p2y) {
            out.line(p1x, p1y, p2x, p2y, &stroke);
        }
    }
}
```

`Stroke` struct fields and `mark_style.color` / `.stroke_width` may need adjustment based on existing patterns — model after `render/marks/rule.rs` which already uses `SvgBuffer::line`.

- [ ] **Step 4: Wire `pub(crate) mod segment;` in `render/marks/mod.rs`**

```rust
pub(crate) mod segment;
```

- [ ] **Step 5: Add dispatch arm in `render/draw.rs`**

Find `dispatch_mark` and add:

```rust
Mark::Segment => crate::render::marks::segment::draw(ctx, out),
```

- [ ] **Step 6: Replace `mark_segment` stub in `chart.py`**

Edit `src/ferrum/chart.py` line 571. Replace:

```python
def mark_segment(self, **kwargs):       raise deferred_mark_error("segment")
```

with:

```python
def mark_segment(self, *, position=None, **kwargs):
    if position is not None:
        from ferrum.position import validate_position_eligibility
        validate_position_eligibility("segment", position)
    new = self._set_mark("segment", **kwargs)
    new._position = position
    return new
```

- [ ] **Step 7: Add `segment` to position eligibility (Identity-only)**

Edit `src/ferrum/position.py::validate_position_eligibility`. The existing matrix already lets Identity pass through; for non-Identity, segment should raise. Confirm Dodge/Jitter/Stack eligibles all exclude `"segment"`. They already do (frozensets in Step 1 of Task 20 don't include "segment"), so no change needed beyond ensuring `mark_segment` calls `validate_position_eligibility`.

- [ ] **Step 8: Update `PHASE_9_PLUS_MARKS`**

Edit `src/ferrum/marks/deferred.py`:

```python
PHASE_9_PLUS_MARKS = frozenset([
    "arc", "image", "geoshape", "label",
])
```

- [ ] **Step 9: Build + tests + commit**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
uv run --no-sync pytest tests/test_phase_9_marks.py -v 2>&1 | tail -15
```
Expected: cargo `≥484` (482 + 1 mark round-trip + 1 segment-renders cargo test); pytest 4 segment tests pass.

```bash
git add crates/ferrum-core/src/spec/mark.rs \
        crates/ferrum-core/src/render/marks/{segment.rs,mod.rs} \
        crates/ferrum-core/src/render/draw.rs \
        src/ferrum/chart.py src/ferrum/marks/deferred.py \
        tests/test_phase_9_marks.py
git commit -m "feat(phase-9d): add mark_segment primitive (diagonal line)"
```

---

### Task 25: `mark_boxen` composite (Python desugar via LetterValue)

**Files:**
- Modify: `src/ferrum/marks/composite.py` (add `desugar_boxen`)
- Modify: `src/ferrum/chart.py` (add `mark_boxen` method; pending-resolve dispatch)
- Modify: `tests/test_phase_9_marks.py` (`TestMarkBoxen`)

**Goal:** `mark_boxen()` is a composite mark that desugars (via `LetterValue` transform) into N nested rect bands + a median rule + outlier points.

**Composite expansion:**
1. Declare `LetterValue` transform with `name="lv"` and `outliers` named output.
2. For each depth k = 1..K, emit a rect layer: `mark_rect` with `x=cat`, `y=lower`, `y2=upper`, opacity ramp from outer (0.85) to inner (0.30) — implemented as a `mark_kwargs={"opacity": 0.85 - 0.55*((k-1)/(K-1))}` per layer.
3. Median rule: `mark_rule` at depth=1 (where lower=upper=median); single layer with `data_source="lv"` filtered to `depth=1`.
4. Outliers: `mark_point` layer with `data_source="lv_outliers"`, fill via `is_outlier` field.

The composite-mark builder mirrors `desugar_boxplot` (composite.py line 15+). Output: 5-tuple `("__layered__", transforms, None, None, layers)`.

- [ ] **Step 1: Failing tests**

Append to `test_phase_9_marks.py`:

```python
class TestMarkBoxen:
    @pytest.fixture
    def df_grouped(self):
        import numpy as np
        np.random.seed(42)
        return pl.DataFrame({
            "g": ["a"] * 100 + ["b"] * 100,
            "v": np.concatenate([np.random.normal(0, 1, 100), np.random.normal(2, 1, 100)]).tolist(),
        })

    def test_mark_boxen_renders(self, df_grouped):
        chart = fe.Chart(df_grouped).mark_boxen().encode(x="g", y="v")
        svg = chart.show_svg()
        assert "<svg" in svg

    def test_mark_boxen_spec_has_letter_value_transform(self, df_grouped):
        chart = fe.Chart(df_grouped).mark_boxen().encode(x="g", y="v")
        import json
        d = json.loads(chart.to_spec().to_json())
        # Either at top-level or in a layer's transforms.
        all_transforms = d.get("transforms", []).copy()
        for layer in d.get("layers", []) or []:
            all_transforms.extend(layer.get("transforms", []))
        assert any(t.get("type") == "letter_value" for t in all_transforms)

    def test_mark_boxen_layered_spec(self, df_grouped):
        chart = fe.Chart(df_grouped).mark_boxen().encode(x="g", y="v")
        spec = chart._build_spec()
        # Multiple layers: rects per depth + median + outliers.
        assert len(spec.layers) >= 3   # at least 1 rect + 1 median + 1 outliers

    def test_mark_boxen_position_dodge_eligible(self, df_grouped):
        from ferrum import Dodge
        # Dodge accepted on boxen.
        fe.Chart(df_grouped).mark_boxen(position=Dodge(by="g")).encode(x="g", y="v")
        # Jitter rejected.
        from ferrum import Jitter
        with pytest.raises(TypeError, match="Jitter"):
            fe.Chart(df_grouped).mark_boxen(position=Jitter())

    def test_mark_boxen_k_depth_param_threads_through(self, df_grouped):
        chart = fe.Chart(df_grouped).mark_boxen(k_depth="full").encode(x="g", y="v")
        import json
        d = json.loads(chart.to_spec().to_json())
        all_t = d.get("transforms", []).copy()
        for layer in d.get("layers", []) or []:
            all_t.extend(layer.get("transforms", []))
        lv = next(t for t in all_t if t.get("type") == "letter_value")
        assert lv["k_depth"]["kind"] == "full"
```

- [ ] **Step 2: Append `desugar_boxen` to `src/ferrum/marks/composite.py`**

```python
def desugar_boxen(
    x_field: str | None,
    y_field: str | None,
    *,
    k_depth: str = "proportion",
    k_proportion: float = 0.007,
    outlier_threshold: float = 1.5,
    palette=None,
    horizontal: bool = False,
    color_field: str | None = None,
    **mark_kwargs,
) -> tuple:
    if x_field is None or y_field is None:
        raise ValueError("mark_boxen() requires .encode(x=..., y=...)")
    cat = y_field if horizontal else x_field
    val = x_field if horizontal else y_field
    group = (color_field if color_field else cat)

    from ferrum import LetterValue
    transforms = [
        LetterValue(value=val, group=group,
                    k_depth=k_depth, k_proportion=k_proportion,
                    outlier_threshold=outlier_threshold, name="lv"),
    ]

    # Per-depth named outputs from LetterValue (added in Task 15) let each rect
    # layer read its own depth slice — no overlap. K_MAX = 6 visible bands; for
    # data with fewer depths, the unused named outputs are zero-row batches and
    # render nothing.
    K_MAX = 6
    layers = []
    for k in range(1, K_MAX + 1):
        opacity = 0.85 - (0.55 * (k - 1) / max(K_MAX - 1, 1))
        enc = (
            {"x": val, "y": cat, "y2": "upper"}
            if horizontal else
            {"x": cat, "y": "lower", "y2": "upper"}
        )
        layers.append({
            "mark": "rect",
            "encoding": enc,
            "mark_kwargs": {"opacity": opacity},
            # data_source maps to LetterValue's `lv_depth_K` named output
            # (when name="lv" is set on LetterValue, the secondary outputs are
            # prefixed; otherwise bare "depth_K").
            "data_source": f"lv_depth_{k}",
        })

    # Median line: rule using depth=1 named output (where lower == upper == median).
    layers.append({
        "mark": "rule",
        "encoding": ({"x": val, "y": cat} if horizontal else {"x": cat, "y": "lower"}),
        "data_source": "lv_depth_1",
    })

    # Outliers: point layer reading from the dedicated outliers output.
    layers.append({
        "mark": "point",
        "encoding": ({"x": val, "y": cat} if horizontal else {"x": cat, "y": val}),
        "data_source": "lv_outliers",
    })

    return ("__layered__", transforms, None, None, layers)
```

The per-depth named outputs (`lv_depth_1` … `lv_depth_6`) come from LetterValue's `secondary_outputs` (Task 15 Step 4): each emits the slice of the primary output where `depth == k`, with the `depth` column dropped. With this, each rect layer renders only the band at its specific depth — no overlap.

- [ ] **Step 3: Add `mark_boxen` method in `chart.py`**

After `mark_boxplot` (line 338) add:

```python
def mark_boxen(
    self, *,
    k_depth="proportion", k_proportion=0.007, outlier_threshold=1.5,
    palette=None, horizontal=False, color_field=None,
    position=None,
    **mark_kwargs,
) -> "Chart":
    """Composite letter-value plot. Desugars to nested rect bands + median + outliers."""
    if position is not None:
        from ferrum.position import validate_position_eligibility
        validate_position_eligibility("boxen", position)
    from ferrum.marks.composite import desugar_boxen
    new = self._clone()
    new._mark = "point"   # placeholder; layered mode overrides
    new._position = position
    new._pending_stat_mark = (
        "boxen",
        {
            "k_depth": k_depth, "k_proportion": k_proportion,
            "outlier_threshold": outlier_threshold,
            "palette": palette, "horizontal": horizontal,
            "color_field": color_field,
            **mark_kwargs,
        },
        desugar_boxen,
    )
    return new
```

The 3-tuple `_pending_stat_mark` form is dispatched by `_resolve_pending` (chart.py line 119) for composite marks — same path as `mark_boxplot`.

Add `"boxen"` to the eligibility validation set in `position.py` for Dodge:

```python
_DODGE_ELIGIBLE = frozenset([
    "bar", "point", "box", "boxplot", "boxen", "swarm", "violin",   # add boxen
    "errorbar", "errorband", "ribbon",
])
```

- [ ] **Step 4: Build + tests + commit**

```bash
source ~/.cargo/env && unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
uv run --no-sync pytest tests/test_phase_9_marks.py -v 2>&1 | tail -15
```
Expected: 5 boxen tests pass.

```bash
git add src/ferrum/marks/composite.py src/ferrum/chart.py \
        src/ferrum/position.py tests/test_phase_9_marks.py
git commit -m "feat(phase-9d): add mark_boxen composite (LetterValue → nested rects + median + outliers)"
```

---

### Task 26: `PHASE_9_PLUS_MARKS` audit + comment update

**Files:**
- Modify: `src/ferrum/marks/deferred.py` (already trimmed in Task 24; just verify and update the doc comment)
- Modify: `tests/test_marks.py` (existing — adjust the boxen / segment expectations if any test asserts deferred status)

- [ ] **Step 1: Verify `PHASE_9_PLUS_MARKS` content**

```bash
grep -n "PHASE_9_PLUS_MARKS\b" src/ferrum/marks/deferred.py
```
Expected (after Task 24): `frozenset(["arc", "image", "geoshape", "label"])` — `segment` is removed.

- [ ] **Step 2: Update the doc comment in `deferred.py`**

```python
"""Marks deferred to Phase 9+. Phase 8b's PHASE_8B_MARKS is empty; Phase 9
removes `segment` from PHASE_9_PLUS_MARKS (now in 9d). The remaining four marks
(`arc`, `image`, `geoshape`, `label`) are not blocked by any §3.14 Group A figure
function and stay deferred consistent with the no-defer rule applying to spec
contracts ferrum currently advertises (see ferrum-spec.md §3.3 — these aren't
referenced by any §3.14 figure-level signature)."""
```

- [ ] **Step 3: Run the existing `tests/test_marks.py`** (no edits expected; just confirm)

```bash
uv run --no-sync pytest tests/test_marks.py -v 2>&1 | tail -10
```
All existing tests should still pass; if `test_phase_8b_marks_set_is_empty` exists and checks PHASE_9_PLUS_MARKS, no change is needed since `segment` removal is already covered by `test_segment_no_longer_in_deferred` in Task 24.

- [ ] **Step 4: Commit**

```bash
git add src/ferrum/marks/deferred.py
git commit -m "chore(phase-9d): update PHASE_9_PLUS_MARKS comment to reflect 9d state"
```

**End of 9d. cargo test ≥ 484; pytest 9-marks tests pass.**

---

## 9e — Figure-level functions

### Task 27: Create `src/ferrum/figure/` package skeleton

**Files:**
- Create: `src/ferrum/figure/__init__.py`
- Create: `src/ferrum/figure/distribution.py` (skeleton)
- Create: `src/ferrum/figure/categorical.py` (skeleton)
- Create: `src/ferrum/figure/regression.py` (skeleton)
- Create: `src/ferrum/figure/matrix.py` (skeleton)
- Create: `src/ferrum/figure/joint.py` (skeleton)
- Modify: `src/ferrum/__init__.py` (re-export figure functions; will populate as Tasks 28-35 land)
- Create: `tests/test_phase_9_figures.py` (skeleton + import-tests)

- [ ] **Step 1: Create skeleton files**

```python
# src/ferrum/figure/__init__.py
"""Phase 9e — figure-level convenience functions.

Each function returns a Chart or compound view (JointChart, RepeatChart,
ClusterMapChart) whose .spec / .charts / .expand() is a fully-formed object.
No NotImplementedError — every parameter advertised in ferrum-spec.md §3.14
Group A is honored.
"""
from ferrum.figure.distribution import displot
from ferrum.figure.categorical import catplot
from ferrum.figure.regression import lmplot, residplot
from ferrum.figure.matrix import pairplot, heatmap, clustermap
from ferrum.figure.joint import jointplot

__all__ = [
    "displot", "catplot", "lmplot", "residplot",
    "pairplot", "heatmap", "clustermap", "jointplot",
]
```

```python
# src/ferrum/figure/distribution.py
"""Phase 9e — displot."""
from __future__ import annotations
def displot(*args, **kwargs):
    raise NotImplementedError("displot — implementation lands in Task 28")
```

Same pattern for `categorical.py`, `regression.py`, `matrix.py`, `joint.py`.

- [ ] **Step 2: Re-export from top-level `__init__.py`**

```python
from ferrum.figure import displot, catplot, lmplot, residplot, pairplot, heatmap, clustermap, jointplot
import ferrum.figure as figure   # so users can also do ferrum.figure.displot
```

Add all 8 to `__all__` plus `"figure"`.

- [ ] **Step 3: Create `tests/test_phase_9_figures.py`** with import smoke:

```python
"""Phase 9e figure-level function tests."""
import pytest
import polars as pl
import ferrum as fe


def test_all_8_functions_importable():
    assert callable(fe.displot)
    assert callable(fe.catplot)
    assert callable(fe.lmplot)
    assert callable(fe.residplot)
    assert callable(fe.pairplot)
    assert callable(fe.heatmap)
    assert callable(fe.clustermap)
    assert callable(fe.jointplot)


def test_figure_submodule_accessible():
    assert hasattr(fe, "figure")
    assert callable(fe.figure.displot)
```

- [ ] **Step 4: Run pytest; expect 2 passes**

```bash
uv run --no-sync pytest tests/test_phase_9_figures.py -v 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add src/ferrum/figure/ src/ferrum/__init__.py tests/test_phase_9_figures.py
git commit -m "feat(phase-9e): add src/ferrum/figure/ package skeleton (8 stub functions)"
```

---

### Task 28: `displot` — distributions

**Files:**
- Modify: `src/ferrum/figure/distribution.py`
- Modify: `tests/test_phase_9_figures.py` (`TestDisplot`)

**Signature & desugar** per spec §8.1.

```python
def displot(
    data, *,
    x=None, y=None, hue=None, col=None, row=None,
    kind="hist",          # "hist"|"kde"|"ecdf"|"rug"
    fill=True, cumulative=False, log_scale=False, stat="count",
    bins="sturges", bandwidth="scott", bw_adjust=1.0,
    multiple="layer",     # "layer"|"stack"|"fill"|"dodge"
    kde=False, rug=False, height=None, aspect=None, theme=None,
    **encode_kwargs,
):
    /* desugar per spec §8.1 */
```

**Desugar table:**
| Param | Effect |
|---|---|
| `kind="hist"` | `chart = chart.mark_histogram(bin_count=bins if int else None, cumulative=cumulative)` (Sturges default if `bins="sturges"`) |
| `kind="kde"` | `chart.mark_density(bandwidth=bandwidth, bw_adjust=bw_adjust, fill=fill)` |
| `kind="ecdf"` | `chart.transform(Bin(field=x, cumulative=True)).mark_line().encode(x="bin_start", y="count")` (Bin's cumulative output gives ECDF directly when extent covers full data range) |
| `kind="rug"` | `chart.mark_tick().encode(x=x)` along x-axis (height shrunk to a tick row) |
| `kde=True` (additional) | layered: histogram (or kde) + KDE line via `chart + Chart(...).mark_density(...)` |
| `rug=True` (additional) | layered: + Chart(...).mark_tick(...) |
| `multiple="layer"` | `position=Identity()` |
| `multiple="dodge"` | `position=Dodge(by=hue)` |
| `multiple="stack"` | `position=Stack(by=hue, offset="zero")` |
| `multiple="fill"` | `position=Stack(by=hue, offset="normalize")` |
| `hue` | `color=hue` encoding |
| `col`, `row` | `chart.facet(col=col, row=row)` |
| `log_scale=True` | x-scale = LogScale (via encoding scale override) |
| `height`, `aspect` | `chart.properties(width=H*aspect, height=H)` where H = height or default |
| `theme` | `chart.theme(theme)` |
| `**encode_kwargs` | passed as `chart.encode(**encode_kwargs)` overrides |

Returns: `Chart`.

- [ ] **Step 1: Failing tests in `test_phase_9_figures.py`**

```python
import numpy as np

@pytest.fixture
def iris_like():
    np.random.seed(0)
    return pl.DataFrame({
        "sepal_length": np.random.normal(5.0, 0.5, 60).tolist(),
        "sepal_width":  np.random.normal(3.0, 0.3, 60).tolist(),
        "species":      ["a"] * 30 + ["b"] * 30,
    })


class TestDisplot:
    def test_hist_default(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length")
        assert isinstance(chart, fe.Chart)
        spec = chart.to_spec()
        import json
        d = json.loads(spec.to_json())
        # Histogram desugars to mark_bar with a Bin transform.
        assert d["mark"] == "bar"
        assert any(t.get("type") == "bin" for t in d.get("transforms", []))

    def test_kde(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length", kind="kde")
        spec = chart.to_spec()
        import json
        d = json.loads(spec.to_json())
        assert any(t.get("type") == "kde" for t in d.get("transforms", []))

    def test_ecdf_uses_cumulative_bin(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length", kind="ecdf")
        import json
        d = json.loads(chart.to_spec().to_json())
        bin_t = next((t for t in d.get("transforms", []) if t.get("type") == "bin"), None)
        assert bin_t is not None
        assert bin_t.get("cumulative") is True

    def test_rug_kind(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length", kind="rug")
        import json
        d = json.loads(chart.to_spec().to_json())
        assert d["mark"] == "tick"

    @pytest.mark.parametrize("multiple,expected_position_type", [
        ("layer", "identity"),
        ("dodge", "dodge"),
        ("stack", "stack"),
        ("fill", "stack"),    # normalize stack
    ])
    def test_multiple_param_sets_position(self, iris_like, multiple, expected_position_type):
        chart = fe.displot(iris_like, x="sepal_length", hue="species", multiple=multiple)
        import json
        d = json.loads(chart.to_spec().to_json())
        # Position is at top-level (single-layer chart).
        assert d.get("position", {}).get("type") == expected_position_type
        if multiple == "fill":
            assert d["position"].get("offset") == "normalize"

    def test_cumulative_param_threads_to_bin(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length", cumulative=True)
        import json
        d = json.loads(chart.to_spec().to_json())
        bin_t = next(t for t in d.get("transforms", []) if t.get("type") == "bin")
        assert bin_t["cumulative"] is True

    def test_renders_e2e(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length", kind="hist")
        svg = chart.show_svg()
        assert "<svg" in svg

    def test_invalid_kind_errors(self, iris_like):
        with pytest.raises(ValueError, match="kind"):
            fe.displot(iris_like, x="sepal_length", kind="bogus")

    def test_facet_col_row(self, iris_like):
        chart = fe.displot(iris_like, x="sepal_length", col="species")
        import json
        d = json.loads(chart.to_spec().to_json())
        assert d.get("facet") is not None
```

- [ ] **Step 2: Implement `displot` in `figure/distribution.py`**

```python
"""Phase 9e — displot."""
from __future__ import annotations
from typing import Any

import ferrum as fe
from ferrum import Chart, Bin, Identity, Dodge, Stack


_VALID_KINDS = {"hist", "kde", "ecdf", "rug"}
_VALID_MULTIPLE = {"layer", "stack", "fill", "dodge"}


def displot(
    data: Any, *,
    x: str | None = None, y: str | None = None,
    hue: str | None = None, col: str | None = None, row: str | None = None,
    kind: str = "hist",
    fill: bool = True, cumulative: bool = False, log_scale: bool = False,
    stat: str = "count",
    bins: Any = "sturges",
    bandwidth: Any = "scott", bw_adjust: float = 1.0,
    multiple: str = "layer",
    kde: bool = False, rug: bool = False,
    height: float | None = None, aspect: float | None = None,
    theme: Any = None,
    **encode_kwargs,
) -> Chart:
    """Distribution figure-level function — see ferrum-spec.md §3.14."""
    if kind not in _VALID_KINDS:
        raise ValueError(f"displot: kind must be one of {_VALID_KINDS}; got {kind!r}")
    if multiple not in _VALID_MULTIPLE:
        raise ValueError(f"displot: multiple must be one of {_VALID_MULTIPLE}; got {multiple!r}")

    # Position adjustment from `multiple`.
    position = _multiple_to_position(multiple, hue)

    # Build the base chart.
    chart = Chart(data)

    # Encoding: x (required for most kinds), color from hue.
    enc: dict = {}
    if x is not None: enc["x"] = x
    if y is not None: enc["y"] = y
    if hue is not None: enc["color"] = hue
    enc.update(encode_kwargs)

    # Mark + transforms by kind.
    if kind == "hist":
        bin_count = bins if isinstance(bins, int) else None
        chart = chart.mark_histogram(
            bin_count=bin_count, cumulative=cumulative, density=(stat == "density"),
            position=position,
        )
    elif kind == "kde":
        chart = chart.mark_density(
            bandwidth=bandwidth, bw_adjust=bw_adjust, fill=fill,
            position=position,
        )
    elif kind == "ecdf":
        # ECDF: cumulative bin → step line.
        bin_count = bins if isinstance(bins, int) else None
        chart = chart.transform(Bin(field=x or "x", bin_count=bin_count, cumulative=True)) \
                     .mark_line(position=position)
        # Re-route encoding to bin output columns.
        enc["x"] = "bin_start"
        enc["y"] = "count"
    elif kind == "rug":
        chart = chart.mark_tick(position=position)

    chart = chart.encode(**enc)

    # Optional kde/rug layers.
    if kde and kind != "kde":
        kde_layer = Chart(data).mark_density(bandwidth=bandwidth, bw_adjust=bw_adjust, fill=False).encode(x=x)
        chart = chart + kde_layer
    if rug and kind != "rug":
        rug_layer = Chart(data).mark_tick().encode(x=x)
        chart = chart + rug_layer

    # log_scale on x.
    if log_scale and x is not None:
        from ferrum.encoding import X
        chart = chart.encode(x=X(x, scale={"type": "log"}))

    # Faceting.
    if col is not None or row is not None:
        chart = chart.facet(col=col, row=row)

    # Properties.
    if height is not None or aspect is not None:
        h = height if height is not None else 300.0
        w = h * aspect if aspect is not None else h
        chart = chart.properties(width=w, height=h)

    if theme is not None:
        chart = chart.theme(theme)

    return chart


def _multiple_to_position(multiple: str, hue: str | None):
    if multiple == "layer":
        return Identity()
    if multiple == "dodge":
        return Dodge(by=hue)
    if multiple == "stack":
        return Stack(by=hue, offset="zero")
    if multiple == "fill":
        return Stack(by=hue, offset="normalize")
    raise ValueError(f"unknown multiple {multiple!r}")
```

- [ ] **Step 3: Build + run tests + commit**

```bash
uv run --no-sync pytest tests/test_phase_9_figures.py -v -k Displot 2>&1 | tail -20
```
Expected: 9 displot tests pass.

```bash
git add src/ferrum/figure/distribution.py tests/test_phase_9_figures.py
git commit -m "feat(phase-9e): implement displot (hist/kde/ecdf/rug × multiple modes)"
```

---

### Task 29: `catplot` — categorical

**Files:** Modify `src/ferrum/figure/categorical.py`; extend `tests/test_phase_9_figures.py` with `TestCatplot`.

**Signature & desugar** per spec §8.2 — full table reproduced below.

```python
def catplot(
    data, *,
    x=None, y=None, hue=None, col=None, row=None,
    kind="strip",         # "strip"|"swarm"|"box"|"violin"|"boxen"|"point"|"bar"|"count"
    order=None, hue_order=None, orient=None,
    dodge=False, jitter=True, native_scale=False,
    ci=95, n_boot=1000, seed=None, theme=None,
    **encode_kwargs,
):
    """Categorical figure-level function — see ferrum-spec.md §3.14."""
```

**Per-`kind` desugar:**

| `kind` | desugar |
|---|---|
| `"strip"` | `chart.mark_point(position=Jitter(axis=orient_axis, width=0.4, seed=seed)) if jitter else chart.mark_point(position=Identity())` |
| `"swarm"` | `chart.mark_swarm()` (existing Phase 8b mark) |
| `"box"` | `chart.mark_boxplot()` |
| `"violin"` | `chart.mark_violin()` |
| `"boxen"` | `chart.mark_boxen()` (Task 25) |
| `"point"` | `chart.mark_point() + chart.mark_errorbar(extent="ci", ci=ci, n_boot=n_boot, seed=seed)` (layered) |
| `"bar"` | `chart.mark_bar() + chart.mark_errorbar(...)` (layered) |
| `"count"` | `chart.transform(Aggregate(ops=[AggregateOp(field=x, fn="count", as_="n")])).mark_bar().encode(x=x, y="n")` |

`dodge=True` and `hue` set → `position=Dodge(by=hue)` on the relevant marks.
`order` / `hue_order` → ordinal scale domain override.
`orient="h"` → `coord(CoordFlip())`.
`native_scale=True` → use linear scale instead of forced ordinal.

- [ ] **Step 1: Failing tests** — write 8-12 tests covering each `kind` value, `dodge=True/False`, `hue` propagation, and a representative render-success case.

- [ ] **Step 2: Implement `catplot`** following the desugar table.

- [ ] **Step 3: Build + tests + commit**

```bash
uv run --no-sync pytest tests/test_phase_9_figures.py -v -k Catplot 2>&1 | tail -20
```

```bash
git add src/ferrum/figure/categorical.py tests/test_phase_9_figures.py
git commit -m "feat(phase-9e): implement catplot (8 kinds × dodge × CI)"
```

---

### Task 30: `lmplot` — regression

**Files:** Modify `src/ferrum/figure/regression.py` (`lmplot`); extend tests.

**Signature & desugar** per spec §8.3.

```python
def lmplot(
    data, *, x, y,
    hue=None, col=None, row=None,
    method="lm",       # "lm"|"logistic"|"glm"|"loess"|"robust"
    ci=95, order=1,
    scatter=True, scatter_kws=None, line_kws=None,
    truncate=False, x_bins=None, x_estimator=None, x_jitter=None,
    logx=False, theme=None,
    **encode_kwargs,
):
```

**Desugar table** (per spec §8.3):

| param/value | effect |
|---|---|
| `scatter=True` | bottom layer = `chart.mark_point(position=Jitter(axis="x", width=x_jitter) if x_jitter else None)` |
| `method="lm"` | top: `mark_smooth(method="lm", ci=ci, x_bins=x_bins, x_estimator=x_estimator, degree=order)` |
| `method="loess"` | top: `mark_smooth(method="loess", ci=ci)` |
| `method="logistic"` | top: layer with `Logistic` transform consumed by `mark_line` + ribbon CI |
| `method="glm"` | top: `Glm` transform with default canonical link (Gaussian/Identity) |
| `method="robust"` | top: `Robust` transform |
| `truncate=True` | restrict line range to observed x-range (clip line to x.min..x.max) |
| `logx=True` | x-scale = LogScale |
| `hue` | per-group fits via color encoding + groupby in regression transform |
| `col` / `row` | `.facet(col=..., row=...)` |

Returns: layered `Chart` (`scatter + fit_line + ci_band`).

- [ ] **Step 1: Failing tests** — 8-10 tests covering each `method`, `ci=None/95`, `scatter=True/False`, `x_bins+x_estimator` combo, `truncate`, render-success.

- [ ] **Step 2: Implement `lmplot`**

```python
def lmplot(
    data, *, x, y,
    hue=None, col=None, row=None,
    method="lm",
    ci=95, order=1, scatter=True,
    scatter_kws=None, line_kws=None,
    truncate=False, x_bins=None, x_estimator=None, x_jitter=None,
    logx=False, theme=None,
    **encode_kwargs,
) -> "fe.Chart":
    if method not in {"lm", "logistic", "glm", "loess", "robust"}:
        raise ValueError(f"lmplot: method must be one of lm|logistic|glm|loess|robust; got {method!r}")
    /* build scatter layer if scatter=True */
    /* build fit layer per method (using new transforms from 9b) */
    /* compose: scatter + fit (+ ci ribbon if ci) */
    /* facet, theme, properties */
```

Per-method dispatch:

```python
if method == "lm":
    fit_layer = Chart(data).mark_smooth(method="lm", ci=ci, degree=order,
                                         x_bins=x_bins, x_estimator=x_estimator).encode(x=x, y=y)
elif method == "loess":
    fit_layer = Chart(data).mark_smooth(method="loess", ci=ci).encode(x=x, y=y)
elif method == "logistic":
    from ferrum import Logistic
    fit_layer = Chart(data).transform(Logistic(x=x, y=y, n_grid=200, ci=ci/100 if ci else None)) \
                            .mark_line().encode(x="x", y="fitted")
    if ci is not None:
        # ci_lower/ci_upper come from the Logistic transform.
        ci_layer = Chart(data).transform(Logistic(x=x, y=y, n_grid=200, ci=ci/100)) \
                              .mark_ribbon().encode(x="x", y="ci_lower", y2="ci_upper")
elif method == "glm":
    from ferrum import Glm
    fit_layer = Chart(data).transform(Glm(x=x, y=y, family="gaussian", ci=ci/100 if ci else None)) \
                            .mark_line().encode(x="x", y="fitted")
elif method == "robust":
    from ferrum import Robust
    fit_layer = Chart(data).transform(Robust(x=x, y=y, ci=ci/100 if ci else None)) \
                            .mark_line().encode(x="x", y="fitted")
```

- [ ] **Step 3: Build + tests + commit**

```bash
uv run --no-sync pytest tests/test_phase_9_figures.py -v -k Lmplot 2>&1 | tail -20
git add src/ferrum/figure/regression.py tests/test_phase_9_figures.py
git commit -m "feat(phase-9e): implement lmplot (5 methods × CI × x_bins/x_estimator)"
```

---

### Task 31: `residplot` — residual diagnostics

**Files:** Modify `src/ferrum/figure/regression.py` (add `residplot`); extend tests.

**Signature & desugar** per spec §8.4. Uses `Smooth(output="residuals")` (Task 14) or `Robust(output="residuals")` (Task 18).

```python
def residplot(
    data, *, x, y,
    lowess=False, order=1, robust=False, dropna=True,
    label=None, color=None, theme=None,
    **encode_kwargs,
) -> "fe.Chart":
    /* underlying fit: Smooth(output="residuals") if not robust else Robust(output="residuals") */
    /* mark_point of (x, residual) */
    /* if lowess: + mark_smooth(method="loess") */
    /* + annotate_hline(0) */
    /* dropna in coerce step */
```

- [ ] **Step 1**: Tests (4-6).
- [ ] **Step 2**: Implementation.
- [ ] **Step 3**: Build + commit.

```bash
git add src/ferrum/figure/regression.py tests/test_phase_9_figures.py
git commit -m "feat(phase-9e): implement residplot (residual diagnostics)"
```

---

### Task 32: `pairplot` — pairwise scatter grid

**Files:** Modify `src/ferrum/figure/matrix.py`; extend tests.

**Signature & desugar** per spec §8.5. Returns `RepeatChart`.

```python
def pairplot(
    data, *, vars=None, x_vars=None, y_vars=None,
    hue=None, kind="scatter",
    diag_kind="auto",
    markers=None, height=None, aspect=None,
    corner=False, dropna=False, theme=None,
    **encode_kwargs,
) -> "fe.RepeatChart":
    /* resolve vars/x_vars/y_vars to row + column lists */
    /* off-diag template: Chart(data).mark_<kind>().encode(x=Repeat.column, y=Repeat.row, color=hue) */
    /* diag template: Chart(data).mark_histogram() if diag_kind in ("hist", "auto") and N small,
                      Chart(data).mark_density() if diag_kind in ("kde", "auto") and N large,
                      None if diag_kind is None */
    /* RepeatChart(template, row=..., column=..., diagonal=diag, corner=corner) */
```

- [ ] **Step 1**: 8-10 tests covering vars vs x_vars/y_vars, corner, diag_kind matrix, hue propagation.
- [ ] **Step 2**: Implementation.
- [ ] **Step 3**: Commit.

```bash
git add src/ferrum/figure/matrix.py tests/test_phase_9_figures.py
git commit -m "feat(phase-9e): implement pairplot (RepeatChart with diagonal/corner)"
```

---

### Task 33: `heatmap`

**Files:** Modify `src/ferrum/figure/matrix.py`; extend tests.

**Signature & desugar** per spec §8.6. Uses `Unpivot` transform (Task 4).

```python
def heatmap(
    data, *, annot=True, fmt=".2f", cmap="blues",
    linewidths=0.5, linecolor="white",
    vmin=None, vmax=None, center=None, robust=False,
    square=False, mask=None, theme=None,
    **encode_kwargs,
) -> "fe.Chart":
    /* extract row labels (data index or first id col) */
    /* declare Unpivot(id_vars=[row_id], var_name="column", value_name="value") */
    /* mark_rect(stroke=linecolor, stroke_width=linewidths) with x="column", y=row_id, fill="value" */
    /* color scale: ContinuousScheme(name=cmap, domain=[vmin or auto, vmax or auto]); center→diverging */
    /* robust=True: vmin/vmax from 2nd/98th percentiles in Python coerce step */
    /* annot=True: layered + mark_text(text="value", format=fmt) */
    /* mask: filter unpivoted data → fill=transparent for masked cells */
```

- [ ] **Step 1**: Tests (6-8): annot=True/False, robust=True/False, mask passed, square=True, error on no numeric columns.
- [ ] **Step 2**: Implementation.
- [ ] **Step 3**: Commit.

```bash
git add src/ferrum/figure/matrix.py tests/test_phase_9_figures.py
git commit -m "feat(phase-9e): implement heatmap (Unpivot + mark_rect)"
```

---

### Task 34: `clustermap`

**Files:** Modify `src/ferrum/figure/matrix.py`; extend tests.

**Signature & desugar** per spec §8.7. The most complex of the 8 — composes `ClusterMapChart` from heatmap + 2 dendrograms.

```python
def clustermap(
    data, *, method="ward", metric="euclidean",
    cmap="viridis", z_score=None, standard_scale=None,
    figsize=None, dendrogram_ratio=0.2, theme=None,
    **encode_kwargs,
) -> "fe.ClusterMapChart":
    """
    Build:
      1. row_link = Linkage(axis="rows", method, metric, z_score, standard_scale, name="row_link")
      2. col_link = Linkage(axis="columns", method, metric, z_score, standard_scale, name="col_link")

      3. Center heatmap: Chart(data).transform(row_link, col_link,
                                                Reorder(by="row_link_order"),
                                                Reorder(by="col_link_order"),
                                                Unpivot(id_vars=...))
                                     .mark_rect(...)
                                     .encode(x="column", y=row_label, fill="value")

      4. Top dendrogram (column linkage): Chart(data).transform(col_link).mark_segment()
                                              .encode(x="x", y="y", x2="x2", y2="y2")
            with data_source="col_link_segments" — consumes Linkage's `segments` named
            output directly, producing 3*(n-1) line segments forming the dendrogram glyph.

      5. Left dendrogram (row linkage): same with row_link, rotated 90° via CoordFlip
                                         data_source="row_link_segments".

      6. ClusterMapChart(heatmap=center, row_dendrogram=left, col_dendrogram=top,
                          dendrogram_ratio=dendrogram_ratio)
    """
```

The implementation is now mechanical because `Linkage.secondary_outputs` (Task 7 Step 7) already produces the `(x, y, x2, y2)` segment rows — `clustermap` just routes them to `mark_segment` via `data_source`.

- [ ] **Step 1**: Tests (6-8): each (method, metric) tested at least once; z_score; standard_scale; spec round-trip; render-success for one full case; verify both dendrograms appear in the rendered SVG (count `<line>` or `<path>` elements consistent with `3*(n-1)*2` segments expected).
- [ ] **Step 2**: Implementation per the docstring above. Routing `data_source="<name>_segments"` to a `mark_segment` layer is the same pattern used by Phase 8b's `mark_qq` (which reads `qq_line` named output).
- [ ] **Step 3**: Commit.

```bash
git add src/ferrum/figure/matrix.py tests/test_phase_9_figures.py
git commit -m "feat(phase-9e): implement clustermap (ClusterMapChart + dendrograms)"
```

---

### Task 35: `jointplot`

**Files:** Modify `src/ferrum/figure/joint.py`; extend tests.

**Signature & desugar** per spec §8.8. Returns `JointChart`.

```python
def jointplot(
    data, *, x, y, hue=None,
    kind="scatter",       # "scatter"|"kde"|"hist"|"hex"|"reg"
    marginal_kind="hist", # "hist"|"kde"|"rug"|"box"
    ratio=5, space=0.05,
    xlim=None, ylim=None,
    joint_kws=None, marginal_kws=None,
    height=None, theme=None,
    **encode_kwargs,
) -> "fe.JointChart":
```

**Per-`kind` center desugar:**
- `"scatter"` → `mark_point`
- `"kde"` → `mark_contour` (uses Kde2D + contour)
- `"hist"` → `Bin2D` + `mark_rect`
- `"hex"` → `mark_hex`
- `"reg"` → `mark_smooth + mark_point` overlaid

**Per-`marginal_kind` desugar:**
- `"hist"` → `mark_histogram`
- `"kde"` → `mark_density`
- `"rug"` → `mark_tick`
- `"box"` → `mark_boxplot`

- [ ] **Step 1**: 5-8 tests covering combinations of kind × marginal_kind.
- [ ] **Step 2**: Implementation.
- [ ] **Step 3**: Commit.

```bash
git add src/ferrum/figure/joint.py tests/test_phase_9_figures.py
git commit -m "feat(phase-9e): implement jointplot (JointChart + Bin2D)"
```

**End of 9e. cargo test ≥ 484; pytest with all 8 figure-level functions ≥ 380.**

---

## Finalize

### Task 36: 12 SVG goldens (`tests/test_phase_9_e2e.py`)

**Files:**
- Create: `tests/test_phase_9_e2e.py`
- Create: `tests/test_phase_9_e2e/goldens/*.svg` (12 committed golden files)

**Goal:** End-to-end render tests that produce byte-identical SVG output across runs. 12 goldens total: 1 per figure-level function (8) + 4 tricky composite cases.

The 12 goldens (per spec §9.1):
1. `displot_hist.svg` — basic histogram
2. `catplot_box.svg` — boxplot with hue+dodge
3. `lmplot_lm_ci.svg` — LM regression with CI band
4. `residplot_lowess.svg` — residuals + lowess overlay
5. `pairplot_3x3.svg` — 3×3 with hue
6. `heatmap_annot.svg` — annotated heatmap
7. `clustermap_basic.svg` — clustered heatmap with row+col dendrograms
8. `jointplot_kde_hist.svg` — KDE center + hist marginals
9. `pairplot_3x3_hue.svg` — pairplot with hue (tricky case)
10. `clustermap_row_col_dendrograms.svg` — clustermap with both dendrograms (tricky case)
11. `jointplot_kde_marginals.svg` — jointplot with KDE marginals (tricky case)
12. `displot_stacked_hist.svg` — displot stacked histogram (tricky case)

**Test pattern** (mirror `tests/test_render.py` golden-test pattern from Phase 7):

```python
"""Phase 9 E2E SVG golden tests."""
import os
from pathlib import Path
import polars as pl
import pytest
import ferrum as fe

GOLDENS_DIR = Path(__file__).parent / "test_phase_9_e2e" / "goldens"
UPDATE = os.environ.get("FERRUM_UPDATE_GOLDENS") == "1"


@pytest.fixture
def df_iris_like():
    /* fixed seed; deterministic 60-row iris-like */


def _check_or_update(name: str, svg: str) -> None:
    path = GOLDENS_DIR / name
    if UPDATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg)
        if not UPDATE:
            pytest.skip(f"created new golden: {name}; rerun without FERRUM_UPDATE_GOLDENS=1")
    expected = path.read_text()
    assert svg == expected, f"golden mismatch for {name}; rerun with FERRUM_UPDATE_GOLDENS=1 to refresh"


def test_displot_hist_golden(df_iris_like):
    chart = fe.displot(df_iris_like, x="sepal_length", kind="hist", height=300)
    _check_or_update("displot_hist.svg", chart.show_svg())

def test_catplot_box_dodge_golden(df_iris_like):
    chart = fe.catplot(df_iris_like, x="species", y="sepal_length", hue="species", kind="box", dodge=True)
    _check_or_update("catplot_box.svg", chart.show_svg())

# ... 10 more golden tests, one per row of the table above.
```

- [ ] **Step 1: Generate goldens via `FERRUM_UPDATE_GOLDENS=1`**

```bash
mkdir -p tests/test_phase_9_e2e/goldens
FERRUM_UPDATE_GOLDENS=1 uv run --no-sync pytest tests/test_phase_9_e2e.py -v 2>&1 | tail -25
```
Expected: 12 SKIP messages (or PASS after the goldens exist).

- [ ] **Step 2: Re-run without env var; verify all pass and byte-identical**

```bash
uv run --no-sync pytest tests/test_phase_9_e2e.py -v 2>&1 | tail -25
```
Expected: 12 passed.

- [ ] **Step 3: Determinism check — run twice; assert identical output**

```bash
uv run --no-sync pytest tests/test_phase_9_e2e.py 2>&1 | tail -3
uv run --no-sync pytest tests/test_phase_9_e2e.py 2>&1 | tail -3
```
Both runs report `12 passed`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_phase_9_e2e.py tests/test_phase_9_e2e/goldens/
git commit -m "test(phase-9): add 12 E2E SVG goldens for figure-level functions"
```

---

### Task 37: Apply spec drift notes to `ferrum-spec.md`

**Files:**
- Modify: `ferrum-spec.md`

**Goal:** Apply the 6 dated drift notes from spec §11 to `ferrum-spec.md`. Each note is dated `2026-05-10` and inserted as an inline blockquote at the top of the affected section.

| Section | Note |
|---|---|
| §3.2 (Encoding Channels) | Add "Position adjustments" subsection: `Identity`, `Dodge`, `Jitter`, `Stack` accepted via `position=` on eligible marks. Mark eligibility matrix included. |
| §3.3 (Primitive Marks) | Add `mark_segment` row (line segment from (x, y) to (x2, y2); diagonal-capable, distinct from axis-aligned mark_rule). |
| §3.3 (Composite Marks) | Add `mark_boxen` row with `k_depth`, `k_proportion`, `outlier_threshold`, `palette` parameters. |
| §3.4 (Stat Transforms) | Add `Unpivot`, `Linkage` (with three named outputs), `Reorder`, `Bin2D`, `Logistic`, `Glm` (with family/link compatibility table), `Robust`, `LetterValue`; document `Bin.cumulative`, `Smooth.x_bins`, `Smooth.x_estimator`, `Smooth.output`/`Robust.output` parameter additions. |
| §3.12 (Compound Views) | Implementation note for `JointChart`. `RepeatChart` gains `diagonal=` and `corner=` parameters; `Repeat.column`/`Repeat.row`/`Repeat.layer` typed sentinels documented (no string sentinels). New `ClusterMapChart` compound view added with documented contract. |
| §3.14 (Figure-Level Functions) | Note: all 8 Group A functions land in Phase 9 with all parameters honored. Group B remains in Phase 10 alongside `ModelSource` and the model-diagnostic marks they depend on. |

- [ ] **Step 1: For each section, locate the section heading in `ferrum-spec.md`**

```bash
grep -n "^### 3\." ferrum-spec.md
```

- [ ] **Step 2: Apply each note** — insert immediately after the section heading as a dated blockquote:

```markdown
### 3.4 Stat Transforms

> **2026-05-10 (Phase 9):** Adds `Unpivot` (wide → long reshape; homogeneous-or-numeric value dtype), `Linkage` (hierarchical clustering with three named outputs: `linkage`, `order`, `coords`), `Reorder` (permutation by index column), `Bin2D` (2D rectangular binning), `Logistic` (binary logistic regression IRLS + Wald CI), `Glm` (5 families × 7 links — see family/link table below), `Robust` (Huber M-estimator + sandwich CI), `LetterValue` (boxen plot statistics; outliers as named secondary output). Extends `Bin` with `cumulative: bool` parameter; `Smooth` with `x_bins`, `x_estimator`, `output: "fitted"|"residuals"`; `Robust` with the same `output` parameter.

### 3.4.x GLM Family/Link Compatibility (Phase 9)

| Family | Canonical link | Other valid |
|---|---|---|
| Gaussian | Identity | Log, Inverse |
| Binomial | Logit | Probit, Log |
| Poisson | Log | Identity, Sqrt |
| Gamma | Inverse | Identity, Log |
| InverseGaussian | InverseSquared | Identity, Log |

[existing content unchanged below]
```

Apply analogous blocks for §3.2, §3.3 (Primitive + Composite Marks), §3.12, §3.14. The exact wording is per the table above.

- [ ] **Step 3: Spec drift completeness check**

After all 6 inserts, confirm by grep:

```bash
grep -n "2026-05-10 (Phase 9)" ferrum-spec.md | wc -l
```
Expected: ≥ 6 (one per section; some sections may get multiple notes for sub-sections).

- [ ] **Step 4: Commit**

```bash
git add ferrum-spec.md
git commit -m "docs(phase-9): apply 6 spec drift notes for §3.2/3.3/3.4/3.12/3.14"
```

---

### Task 38: Update `docs/superpowers/ferrum-phases.md` Phase 9 row to `done`

**Files:**
- Modify: `docs/superpowers/ferrum-phases.md`

- [ ] **Step 1: Find Phase 9 row**

```bash
grep -n "^| \*\*9\*\*" docs/superpowers/ferrum-phases.md
```

- [ ] **Step 2: Edit to mark `done`** and link spec doc

Replace the row:

```markdown
| **9** | Convenience / figure-level API | `displot`, `lmplot`, `roc_chart`, `pairplot`, etc. as sugar over the grammar — they must desugar to valid `Chart` specs, not bypass the engine | 8 | *(not yet written)* | pending |
```

with:

```markdown
| **9** | Convenience / figure-level API | 8 Group A figure functions (`displot`, `catplot`, `lmplot`, `residplot`, `pairplot`, `heatmap`, `clustermap`, `jointplot`); 8 new transforms (Unpivot, Linkage, Reorder, Bin2D, Logistic, Glm, Robust, LetterValue); 4 position adjustments (Identity, Dodge, Jitter, Stack); 2 new marks (segment, boxen); 3 new compound views (JointChart, RepeatChart, ClusterMapChart). Group B (model-diagnostic figure-level) deferred to Phase 10. | 8 | [`2026-05-10-convenience-api-design.md`](specs/2026-05-10-convenience-api-design.md) | **done** |
```

- [ ] **Step 3: Update the done-criteria checklist for Phase 9**

Find `### Phase 9 — Convenience API` (around line 137) and replace the unchecked `- [ ]` items with `- [x]`:

```markdown
### Phase 9 — Convenience API
- [x] Each figure-level function in `ferrum-spec.md §3.14` Group A is implemented
- [x] Each one can be deconstructed: calling the function and inspecting `.spec` (or `.charts` / `.expand()`) yields a valid `ChartSpec` or compound view
- [x] All 4 position adjustments (Identity, Dodge, Jitter, Stack) ship with mark eligibility enforced
- [x] `PHASE_9_PLUS_MARKS` no longer contains `segment`
- [x] 12 SVG goldens are byte-identical across runs
- [x] All `cargo test` + `pytest` pass
- [x] All 6 spec drift notes applied to `ferrum-spec.md`
```

- [ ] **Step 4: Update the "Last updated" line at the top**

Change line 3 to `**Last updated:** 2026-05-10`.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/ferrum-phases.md
git commit -m "docs(phase-9): mark Phase 9 done in ferrum-phases.md"
```

---

### Task 39: Final-pass verification

**Files:** none (verification only)

- [ ] **Step 1: Full cargo test pass**

```bash
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
```
Expected: `≥484 passed; 0 failed`. The exact count depends on per-task test depth; some transforms (Linkage especially) carry more tests, so 510-530 is the realistic landing range. Per-stage running counts are tracked in the "Final test count expectations" table at the bottom of this plan.

- [ ] **Step 2: Full pytest pass**

```bash
uv run --no-sync pytest 2>&1 | tail -3
```
Expected: `≥400 passed, 7+ skipped` (the 7 historical skips remain; new tests do not introduce skips).

- [ ] **Step 3: Goldens stable on second run**

```bash
uv run --no-sync pytest tests/test_phase_9_e2e.py 2>&1 | tail -3
```
Expected: `12 passed`.

- [ ] **Step 4: No `segment` in PHASE_9_PLUS_MARKS**

```bash
uv run --no-sync python -c "from ferrum.marks import PHASE_9_PLUS_MARKS; assert 'segment' not in PHASE_9_PLUS_MARKS; print('OK')"
```

- [ ] **Step 5: Spec round-trip on a representative figure-level function**

```bash
uv run --no-sync python -c "
import ferrum as fe
import polars as pl
df = pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [4.0, 5.0, 6.0]})
chart = fe.displot(df, x='x', kind='hist')
import json
spec_json = chart.to_spec().to_json()
spec_back = fe.ChartSpec.from_json(spec_json)
assert spec_back == chart.to_spec()
print('round-trip OK')
"
```

- [ ] **Step 6: Phase 9 done summary**

Print a final summary by running:

```bash
git log --oneline main..feat/phase-9 | head -50
```
Expected: ~30-35 commits across the 9a/9b/9c/9d/9e/finalize sub-batches.

---

### Task 40: Merge `feat/phase-9` to `main` (with explicit user confirmation)

**Files:** none (branch merge only)

- [ ] **Step 1: Final pre-merge checks (orchestrator-only — DO NOT merge until user explicitly confirms)**

```bash
git status
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
uv run --no-sync pytest 2>&1 | tail -3
```

- [ ] **Step 2: Ask user before merging**

State to user:
```
Phase 9 implementation complete on feat/phase-9.
- cargo test: <N> passed
- pytest: <M> passed, <K> skipped
- 12 SVG goldens byte-identical
- All 8 figure-level functions implemented; all positions/marks/transforms shipped
- ferrum-phases.md Phase 9 marked done
- ferrum-spec.md drift notes applied

Ready to merge to main? (y/n)
```

Wait for explicit user "y" / approval. **Do NOT merge or push without user confirmation.**

- [ ] **Step 3: Merge to main (after user confirms)**

```bash
git checkout main
git merge --no-ff feat/phase-9 -m "Merge Phase 9: convenience / figure-level API ($(git log --oneline main..feat/phase-9 | wc -l | tr -d ' ') commits)"
```

- [ ] **Step 4: Verify post-merge state**

```bash
source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
uv run --no-sync pytest 2>&1 | tail -3
git log --oneline -5
```

- [ ] **Step 5: Do NOT push** unless user explicitly requests `git push`. The merge stays local until requested.

---

## Subagent verification protocol

When this plan is executed via `superpowers:subagent-driven-development`, after EACH task the orchestrator MUST:

1. **Re-run cargo test directly** (do not trust subagent's reported counts):
   ```bash
   source ~/.cargo/env && DYLD_LIBRARY_PATH=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
   ```
2. **Re-run pytest directly**:
   ```bash
   uv run --no-sync pytest 2>&1 | tail -3
   ```
3. **Verify file changes via `git ls-tree`**:
   ```bash
   git ls-tree HEAD --name-only -r | grep -E "(transform|position|figure|composition|marks|chart)\.py|transform/|render/" | head -40
   git status
   ```
4. **Cross-check the subagent's reported task-completion against the actual diff**:
   ```bash
   git log --oneline -1
   git show --stat HEAD
   ```

This protocol is a hard rule established post-Phase 8b (per memory `feedback_subagent_verification`), where subagents falsely reported file deletions and test counts. Trust nothing the subagent says about counts or file mutations until the orchestrator has independently verified.

---

## Final test count expectations

| Stage | cargo test | pytest |
|---|---|---|
| Baseline (start of Phase 9) | 395 | 298 + 7 skipped |
| End of 9a | ≥ 428 | ≥ 318 |
| End of 9b | ≥ 475 | ≥ 318 |
| End of 9c | ≥ 482 | ≥ 350 |
| End of 9d | ≥ 484 | ≥ 360 |
| End of 9e | ≥ 484 | ≥ 380 |
| End of finalize (with goldens) | ≥ 484 | ≥ 400 |

The cargo-side increase comes mostly from new transform tests (8 transforms × ~6 tests each = ~48), position-adjustment tests (~10), grid-compose tests (~3), spec-position tests (~5), mark-segment tests (~2), shared core round-trips (~10), and existing Bin/Smooth extension tests (~5). The pytest-side increase comes from 5 new `test_phase_9_*.py` files.







