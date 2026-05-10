# Phase 9 — Convenience / Figure-Level API — Design

**Status:** design (pre-implementation)
**Phase:** 9 (per `docs/superpowers/ferrum-phases.md`)
**Authors:** brainstorming session 2026-05-10
**Concept spec:** `ferrum-spec.md` §3.14 (Figure-Level Functions)
**Predecessor phase:** 8b (composite + heavy statistical marks; shipped 2026-05-10, commit `05a3333`)

---

## 1. Goal and scope

Phase 9 ships the **figure-level convenience API** as a thin Python sugar layer over the grammar primitives built in Phases 1–8b. Users call a single function (`displot`, `lmplot`, `pairplot`, …) and get a fully-formed `Chart` (or compound view) whose `.spec` round-trips through the engine. The convenience layer must **desugar to grammar primitives, not bypass the engine** — this is the load-bearing design constraint.

### Scope

`ferrum-spec.md §3.14` lists two groups of figure-level functions:

**Group A — grammar-sugar (in scope for Phase 9):**
`displot`, `catplot`, `lmplot`, `residplot`, `pairplot`, `heatmap`, `clustermap`, `jointplot`.

**Group B — model-diagnostic figure-level functions (DEFERRED to Phase 10):**
`roc_chart`, `pr_chart`, `confusion_matrix_chart`, `calibration_chart`, `gain_chart`, `lift_chart`, `residuals_chart`, `importance_chart`, `shap_chart`, `learning_curve_chart`, `validation_curve_chart`, `cluster_diagnostics`, `decision_boundary_chart`, `discrimination_threshold_chart`, `parallel_coordinates_chart`, `class_prediction_error_chart`, `pca_scree_chart`, `rank_chart`, `alpha_selection_chart`, `intercluster_distance_chart`, `cv_scores_chart`.

Group B depends on Phase 10's `ModelSource` (sklearn-protocol adapter) and on the model-diagnostic marks (`mark_confusion`, `mark_roc`, `mark_calibration`, …) listed in `§3.3 → Model Diagnostic Marks`. None of those exist yet. Building Group B here would invert the phase ordering and require pulling all of Phase 10's infrastructure forward. Group B stays in Phase 10.

### Sub-batch decomposition

Phase 9 is large because earlier phases left load-bearing infrastructure deferred (logistic regression, GLM, robust regression, letter-value statistics, position adjustment, `mark_segment`, `mark_boxen`). Per the **no-defer principle in the project root `CLAUDE.md`**, these are NOT pushed forward again. Phase 9 owns them and ships them right.

Build order (single `feat/phase-9` branch, sub-batches commit independently):

- **9a — Convenience layer foundation.** New compound views (`JointChart`, `RepeatChart`, `ClusterMapChart`); reshape and clustering Rust transforms (`Unpivot`, `Linkage`, `Reorder`, `Bin2D`); `Repeat` typed sentinel.
- **9b — Stat-engine extensions.** New Rust transforms (`Logistic`, `Glm`, `Robust`, `LetterValue`); extensions to existing transforms (`Bin.cumulative`, `Smooth.x_bins`, `Smooth.x_estimator`, `Smooth.output`/`Robust.output` for residuals).
- **9c — Position-adjustment subsystem.** Four position adjustments (`Identity`, `Dodge`, `Jitter`, `Stack`) wired into eligible marks.
- **9d — New marks.** `mark_segment` (primitive), `mark_boxen` (composite). `segment` removed from `PHASE_9_PLUS_MARKS`.
- **9a-finalize — Figure-level functions.** The 8 Group A functions in `src/ferrum/figure/` desugar to specs using everything built in 9a–9d.

Sub-phases manage build order, not scope reduction. All eight figure-level functions land with all `§3.14` parameters honored faithfully — no `NotImplementedError`, no warn-fallbacks for advertised parameter values.

---

## 2. Architecture overview

```
        ┌──────────────────────────────────────────────────────────┐
        │ Figure-level functions (src/ferrum/figure/)              │
        │   displot, catplot, lmplot, residplot,                   │
        │   pairplot, heatmap, clustermap, jointplot               │
        └────────────┬─────────────────────────────────────────────┘
                     │ desugars to ChartSpec / compound-view spec
   ┌─────────────────┼────────────────────────────────────────────┐
   │ Compound views (9a)              Marks (9d)                  │
   │   JointChart, RepeatChart,       mark_boxen, mark_segment    │
   │   ClusterMapChart                                            │
   └─────────────────┬────────────────────────┬───────────────────┘
                     │                        │
   ┌─────────────────┴────────┐      ┌────────┴───────────────────┐
   │ Reshape/cluster (9a)     │      │ Position adjustments (9c)  │
   │   Unpivot, Linkage,      │      │   Identity, Dodge,         │
   │   Reorder, Bin2D         │      │   Jitter, Stack            │
   └──────────────────────────┘      └────────────────────────────┘
                     │
        ┌────────────┴───────────────────────────────────────────┐
        │ Stat-engine extensions (9b)                            │
        │   Logistic, GLM, Robust, LetterValue,                  │
        │   Bin.cumulative, Smooth.x_bins/x_estimator/output     │
        └────────────────────────────────────────────────────────┘
                     │
        ┌────────────┴───────────────────────────────────────────┐
        │ Existing engine (Phases 1–8b, unchanged)               │
        └────────────────────────────────────────────────────────┘
```

### File layout

| Layer | Paths |
|---|---|
| Rust transforms (new) | `crates/ferrum-core/src/transform/{unpivot,linkage,reorder,bin_2d,logistic,glm,robust,letter_value}.rs` |
| Rust transforms (extended) | `crates/ferrum-core/src/transform/{bin,smooth}.rs` (cumulative; x_bins/x_estimator/output) |
| Rust ChartSpec extensions | `crates/ferrum-core/src/spec/{transform,mark,position}.rs` (new variants) |
| Rust render — grid composer | `crates/ferrum-core/src/render/compose.rs` (new `compose_svg_grid`) |
| Rust render — position pass | `crates/ferrum-core/src/render/position.rs` |
| PyO3 bindings | `crates/ferrum-core/src/python/{transforms,marks,position}.rs` (new wrappers) |
| Python compound views | `src/ferrum/composition.py` (add `JointChart`, `RepeatChart`, `ClusterMapChart`); `src/ferrum/repeat.py` (typed sentinel) |
| Python position | `src/ferrum/position.py` (`Identity`, `Dodge`, `Jitter`, `Stack`) |
| Python marks | `src/ferrum/marks/composite.py` (`boxen`); `src/ferrum/marks/base.py` (`segment` routing); `src/ferrum/chart.py` (add `mark_boxen`, `mark_segment`); `src/ferrum/marks/deferred.py` (remove `segment`) |
| Python figure functions | `src/ferrum/figure/{__init__,distribution,categorical,regression,matrix,joint}.py` |
| Public API | `src/ferrum/__init__.py` re-exports |
| Tests | `tests/test_phase_9_compound_views.py`, `test_phase_9_transforms.py`, `test_phase_9_position.py`, `test_phase_9_marks.py`, `test_phase_9_figures.py`, `test_phase_9_e2e.py` |
| Fixture generators | `crates/ferrum-core/src/transform/fixtures/generate_{linkage,glm,logistic,robust,letter_value}_refs.py` |

### Key architectural commitments

- **Figure-level functions live under `src/ferrum/figure/`** as a new package. They are plain functions, not `Chart` methods. `chart.py` stays focused on grammar primitives.
- **Deconstructable by structure, not by test.** Every figure function returns either a `Chart` or a compound view. There is no path from a figure function to rendering that bypasses the spec.
- **Stat-engine extensions follow the existing transform protocol.** Declared in `ChartSpec.transforms`, executed in Rust before layout. No new Python-level computation in figure-level functions; they only build specs.
- **Position adjustment is a new orthogonal axis** between encoding resolution and mark rendering — it is part of the `Mark` payload in the spec, not a new transform.
- **One `ChartSpec` per `RecordBatch`** is preserved (the load-bearing 8a invariant). Mixed-data figure functions (none of the 8 Group A functions are mixed-data) would route through the SVG compositor, not multi-batch logic.

---

## 3. Compound views (9a)

Three new compound view classes in `src/ferrum/composition.py`. All are immutable Python value classes that hold a tree of `Chart` objects (or sub-compound views) and produce a deconstructable `.spec`. All accept `.theme(t)`, `.properties(...)`, `.save(path)`, `.show()`.

### 3.1 `JointChart`

Public API matches `ferrum-spec.md §3.12` verbatim:

```python
JointChart(center, *, top=None, right=None, ratio=5, spacing=0.02)
```

- `center` — any `Chart` (the joint plot, typically `mark_point`, `mark_kde`, `mark_hex`, …)
- `top` — optional `Chart`; rendered above center; x-axis shared with center
- `right` — optional `Chart`; rendered to the right; y-axis shared with center
- `ratio` — center is `ratio` times wider/taller than each marginal
- `spacing` — gap between panels as a fraction of total figure size

**Layout:** 2×2 grid: bottom-left = center, top-left = `top`, bottom-right = `right`, top-right = empty. Cell sizing computed from `ratio`: marginal cells get `1 / (ratio + 1)`, center gets `ratio / (ratio + 1)`.

**Axis sharing:** the layout engine resolves x-scale across (center, top) and y-scale across (center, right). Reuses existing Phase 8a scale-resolve infrastructure; JointChart declares its share-set in the compound-view spec.

**Spec shape:**

```python
{
  "kind": "joint",
  "center": <Chart.spec>,
  "top":    <Chart.spec> | None,
  "right":  <Chart.spec> | None,
  "ratio": 5,
  "spacing": 0.02,
  "share": {"x": ["center", "top"], "y": ["center", "right"]},
}
```

`.charts` returns `[center, top, right]` filtered for `None`. `.theme(t)` propagates to all child charts.

### 3.2 `RepeatChart` with `diagonal=` and typed sentinels

Public API extends `§3.12`:

```python
RepeatChart(template, *, row=None, column=None, layer=None,
            diagonal=None,
            corner=False,
            spacing=0.02, columns=None, resolve=None)
```

`diagonal` is new — applied to cells where `row[i] == column[i]` for an n×n repeat. Used by `pairplot`. `corner=True` filters to the lower triangle only (also for `pairplot(corner=True)`).

**Typed placeholder sentinels.** Replace string sentinels (`"<col>"`) with a typed value object:

```python
from ferrum import Repeat

RepeatChart(
    Chart(data).mark_point().encode(
        x=Repeat.column,        # typed sentinel — IDE-autocompleteable, no string magic
        y=Repeat.row,
        color="species",        # plain field reference, unambiguous
    ),
    row=["mpg", "weight", "hp"],
    column=["mpg", "weight", "hp"],
    diagonal=Chart(data).mark_histogram().encode(x=Repeat.column),
)
```

`Repeat.column`, `Repeat.row`, `Repeat.layer` are class-level attributes returning a small `RepeatPlaceholder(field=…)` value object. The encoding-spec serializer detects these and emits `{"$repeat": "column"}` JSON nodes.

**Spec shape:**

```python
{
  "kind": "repeat",
  "template": <Chart.spec with $repeat sentinels>,
  "row": ["mpg", "weight", "hp"],
  "column": ["mpg", "weight", "hp"],
  "diagonal": <Chart.spec> | None,
  "layer": [...] | None,
  "columns": int | None,
  "corner": false,
  "resolve": {...} | None,
  "spacing": 0.02,
}
```

**Diagonal override semantics:**
- If `row` and `column` are both provided AND equal in length AND `diagonal` is set → renderer uses `diagonal` for cells where `row[i] == column[i]`.
- If `row != column` (asymmetric pairplot via `x_vars`/`y_vars`) → `diagonal` is ignored; `warn_once`.
- If `diagonal` is set without both `row` and `column` set → `ValueError` at construction.

**Methods:** `.theme(t)`, `.properties(...)`, `.save(path)`, `.show()`, `.expand()`. `.expand()` materializes the template by replacing `$repeat` placeholders and returns `[(row_field, col_field, Chart), …]` — fully concrete `Chart` objects.

### 3.3 `ClusterMapChart`

Dedicated compound view for `clustermap()`. Tighter contract than overloading `JointChart`.

```python
ClusterMapChart(
    heatmap,           # the central heatmap Chart (already row/col reordered)
    *,
    row_dendrogram=None,    # left-side Chart of mark_segment over Linkage coords
    col_dendrogram=None,    # top Chart of mark_segment over Linkage coords
    dendrogram_ratio=0.2,   # fraction of total dim taken by each dendrogram
    spacing=0.02,
)
```

**Layout:** 2×2 grid. Top-left: empty. Top-right: `col_dendrogram`. Bottom-left: `row_dendrogram` (rotated 90°). Bottom-right: `heatmap`. Dendrogram axes (the value/distance axis) are hidden by a `axis=None` flag on those Charts; the categorical axis ticks align with the heatmap's row/column labels.

**Spec shape:**

```python
{
  "kind": "cluster_map",
  "heatmap": <Chart.spec>,
  "row_dendrogram": <Chart.spec> | None,
  "col_dendrogram": <Chart.spec> | None,
  "dendrogram_ratio": 0.2,
  "spacing": 0.02,
}
```

`.charts` returns the three sub-charts (filtered for `None`).

### 3.4 Renderer integration

A new Rust grid compositor handles all three compound views:

```rust
pub fn compose_svg_grid(
    cells: &[Option<SvgPanel>],   // row-major, with None for empty cells
    *,
    rows: usize,
    cols: usize,
    row_ratios: Vec<f64>,
    col_ratios: Vec<f64>,
    spacing: f64,
    share_x: Vec<Vec<usize>>,     // groups of cell indices sharing x
    share_y: Vec<Vec<usize>>,     // groups of cell indices sharing y
) -> SvgBuffer
```

Lives in `crates/ferrum-core/src/render/compose.rs`, exposed via PyO3 as `ferrum._core.compose_svg_grid`. JointChart, RepeatChart, ClusterMapChart all call into it with their respective layout configurations. Existing `compose_svg_horizontal` and `compose_svg_vertical` remain (HConcat/VConcat).

---

## 4. Reshape and clustering transforms (9a — Rust)

Four new Rust transforms. All follow the existing TransformSpec protocol from Phase 5/8b. All emit named outputs where useful (Phase 8b protocol).

### 4.1 `Unpivot` — wide → long reshape

```rust
TransformSpec::Unpivot {
    name: Option<String>,
    id_vars: Vec<String>,
    value_vars: Option<Vec<String>>,   // None → all non-id columns
    var_name: String,                  // default "variable"
    value_name: String,                // default "value"
}
```

**Output schema:** `[id_vars..., var_name: Utf8, value_name: <unified_dtype>]`.

**Dtype rule (homogeneous-or-numeric):**
- Value columns must share a dtype OR all be numeric.
- Numeric mixed types widen to the widest (Int32 + Float64 → Float64).
- Mixed non-numeric types (e.g., Int32 + Utf8) → error: `"value_vars have heterogeneous non-numeric types: [Int32, Utf8]; cast to a common type before unpivot"`.

**Implementation:** `crates/ferrum-core/src/transform/unpivot.rs`, ~120 LOC. Hand-rolled on `arrow::compute::concat` and `arrow::compute::take` for id-column replication. No external clustering deps.

**Tests:**
- JSON round-trip
- 3×4 numeric matrix correctness
- Numeric widening (Int32 + Float64 → Float64)
- Homogeneous Utf8 unpivot (e.g., a wide table of categorical flags)
- Error path: mixed non-numeric types → clear error
- Schema: id-column dtypes preserved; var_name is Utf8

### 4.2 `Linkage` — hierarchical clustering

```rust
TransformSpec::Linkage {
    name: Option<String>,
    method: LinkageMethod,            // Single|Complete|Average|Weighted|Centroid|Median|Ward
    metric: DistanceMetric,           // Euclidean|Manhattan|Cosine|Correlation|Chebyshev
    axis: LinkageAxis,                // Rows|Columns
    z_score: Option<ZScoreAxis>,
    standard_scale: Option<StdScaleAxis>,
}
```

**Three named outputs:**

| Output | Schema | Used by |
|---|---|---|
| `linkage` | `[node_id: Int64, left: Int64, right: Int64, distance: Float64, n_obs: Int64]` | dendrogram segments |
| `order` | `[original_idx: Int64, new_idx: Int64]` | row/column reordering for the heatmap |
| `coords` | `[node_id: Int64, x: Float64, y: Float64]` | dendrogram x-y positions for segment endpoints |

This three-output design lets `clustermap` use `order` for reordering AND `coords + linkage` for dendrogram rendering without recomputing.

**Implementation strategy: `kodama` for the linkage matrix; hand-roll `coords` and `order`.** `kodama` is a small pure-Rust hierarchical-clustering crate implementing scipy-compatible Lance-Williams + nearest-neighbor chain. The bug-prone part (Lance-Williams coefficients, NN-chain applicability per method, numerical stability) lives in audited code; the dendrogram coordinate layout and row reordering are simple tree traversals.

**Verify-before-implementing task in plan:** before adding `kodama` to Cargo.toml, verify last release < 18 months, license is MIT/Apache, no critical open issues. If verification fails, fall back to hand-roll (~400-600 LOC).

**Correctness fixtures:** generated against scipy via `crates/ferrum-core/src/transform/fixtures/generate_linkage_refs.py`. Curated coverage (5 (method, metric) pairs hitting each method and each metric at least once):
- ward + euclidean
- complete + euclidean
- average + correlation
- single + manhattan
- complete + cosine

Plus standalone tests for chebyshev metric and median/centroid methods using ward+euclidean as the reference structure.

**Edge cases:** n=2 (single merge), n=1 (degenerate; error), all-zero distances (any method should produce a valid tree).

### 4.3 `Reorder` — index-based row reordering

```rust
TransformSpec::Reorder {
    name: Option<String>,
    by: String,           // index column (typically Linkage's `order` output)
}
```

Applies a permutation to the input batch, ordered by an integer index column. Output schema = input schema (without the index column). ~50 LOC.

**Tests:** JSON round-trip, permutation correctness on 5-row × 4-col input, idempotence with identity permutation.

### 4.4 `Bin2D` — 2D rectangular binning

```rust
TransformSpec::Bin2D {
    name: Option<String>,
    x: String,
    y: String,
    bins_x: BinSpec,            // existing BinSpec from Bin (Sturges, FreedmanDiaconis, Fixed(n))
    bins_y: BinSpec,
    cumulative: bool,
}
```

**Output schema:** `[x_lo: Float64, x_hi: Float64, y_lo: Float64, y_hi: Float64, count: Int64]`.

Used by `jointplot(kind="hist")`. Mirrors the existing `Bin` transform structure but in two dimensions. ~100 LOC.

**Tests:** JSON round-trip, correctness on a 3×3 grid input, Sturges floor honored on each axis independently, cumulative=True monotonicity.

### 4.5 Spec drift notes (consolidated in §11)

Adds `Unpivot`, `Linkage`, `Reorder`, `Bin2D` rows to `ferrum-spec.md §3.4`.

---

## 5. Stat-engine extensions (9b — Rust)

Six new or extended transforms. All follow the existing TransformSpec protocol. All have fixture-backed correctness tests against scipy/statsmodels references.

### 5.1 `Logistic` — binary logistic regression

```rust
TransformSpec::Logistic {
    name: Option<String>,
    x: String,
    y: String,
    n_grid: usize,            // default 100
    ci: Option<f64>,          // confidence level for Wald-based CI band
    max_iter: usize,          // default 25
    tol: f64,                 // default 1e-8
}
```

**Output schema:** `[x: Float64, fitted: Float64, ci_lower: Float64, ci_upper: Float64]`. CI columns null if `ci` is None.

**Algorithm:** IRLS with logit link. Wald CI via the inverse Fisher information matrix at the MLE. Convergence test on log-likelihood delta.

**Edge cases:** perfect separation detected (fitted probabilities saturating to 0/1 in early iterations) → clear error. Non-binary y → error naming the column and unique values. Singular design matrix → error.

**Fixture source:** `statsmodels.discrete.discrete_model.Logit`. Five datasets: well-separated, moderately separated, near-degenerate, integer-valued x, challenger O-rings.

### 5.2 `Glm` — generalized linear model

```rust
TransformSpec::Glm {
    name: Option<String>,
    x: String,
    y: String,
    family: GlmFamily,         // Gaussian|Binomial|Poisson|Gamma|InverseGaussian
    link: Option<GlmLink>,     // Identity|Log|Logit|Probit|Inverse|InverseSquared|Sqrt;
                               //   None → canonical link for the family
    n_grid: usize,
    ci: Option<f64>,
    max_iter: usize,
    tol: f64,
}
```

**Output schema:** identical to Logistic.

**Family/link compatibility:**

| Family | Canonical link | Other valid links |
|---|---|---|
| Gaussian | Identity | Log, Inverse |
| Binomial | Logit | Probit, Log |
| Poisson | Log | Identity, Sqrt |
| Gamma | Inverse | Identity, Log |
| InverseGaussian | InverseSquared | Identity, Log |

Invalid (family, link) pairs error at construction with the valid-link list.

**Fixture source:** `statsmodels.genmod.generalized_linear_model.GLM`. Coverage: each (family, canonical link) tested + 3 non-canonical (Gaussian+Log, Binomial+Probit, Poisson+Sqrt). Other valid combinations are implementation-supported but not exhaustively fixture-tested.

### 5.3 `Robust` — Huber M-estimator

```rust
TransformSpec::Robust {
    name: Option<String>,
    x: String,
    y: String,
    n_grid: usize,
    ci: Option<f64>,
    huber_c: f64,              // default 1.345 (95% Gaussian efficiency)
    max_iter: usize,           // default 50
    tol: f64,
}
```

**Algorithm:** Huber M-estimator via IRLS. MAD scale estimate (× 1.4826). CI via Huber sandwich estimator.

**Why Huber only:** seaborn's `robust=True` uses Huber. RANSAC/LTS/MM/S-estimators are valid alternatives but additional methods, not the default. Documented in spec; future expansion tracked separately.

**Fixture source:** `statsmodels.robust.robust_linear_model.RLM` with `M=HuberT()`. Datasets: clean linear data (matches OLS within tolerance), 10% outliers, 30% outliers, leverage-point dataset.

### 5.4 `LetterValue` — letter-value statistics

```rust
TransformSpec::LetterValue {
    name: Option<String>,
    value: String,
    group: Option<String>,
    k_depth: KDepth,           // Tukey|Proportion(f64)|Trustworthy|Full
    outlier_threshold: f64,    // default 1.5
}
```

**Two named outputs:**

| Output | Schema |
|---|---|
| (default) | `[group: Utf8|Null, depth: Int32, lower: Float64, upper: Float64, level: Utf8]` |
| `outliers` | `[group: Utf8|Null, value: Float64, is_outlier: Bool]` |

**k_depth strategies (Hofmann/Wickham/Kafadar 2017):**
- `Tukey`: `k = ⌊log₂(n)⌋ - 3`
- `Proportion(p)` (default p=0.007): smallest k such that the outermost letter-value contains ≤ p × n observations
- `Trustworthy`: k chosen so estimated CI for outermost letter-value width is below a threshold
- `Full`: depth 1 (single observation)

**Fixture source:** numpy quantile reference (no third-party package). Letter-values are deterministic quantile computations.

### 5.5 `Bin` — `cumulative` parameter (extension)

```rust
TransformSpec::Bin {
    // ... existing fields ...
    cumulative: bool,                 // new; default false
}
```

When `cumulative=true`, the count (or density / probability) column is replaced with its cumulative sum. Tests cover each `stat` value with `cumulative=true` and verify monotonicity. ECDF-style output for `displot(kind="ecdf", cumulative=True)`.

### 5.6 `Smooth` — `x_bins`, `x_estimator`, `output` (extensions)

```rust
TransformSpec::Smooth {
    // ... existing fields ...
    x_bins: Option<usize>,
    x_estimator: Option<AggregateOp>,   // Mean|Median|Sum|Min|Max
    output: SmoothOutput,               // Fitted (default) | Residuals
}
```

When `x_bins` and `x_estimator` are set, x is binned into N equal-width bins, y is aggregated per bin, and the regression is fit on aggregated points. CI computed from aggregated points. Used by `lmplot(x_bins=..., x_estimator=...)`.

`output="residuals"` switches the output column to `y - fitted` instead of `fitted`. Same parameter added to `Robust`. Used by `residplot`.

### 5.7 LOC budget

| Transform | Rust LOC |
|---|---|
| Logistic | ~250 |
| Glm | ~500 |
| Robust | ~300 |
| LetterValue | ~150 |
| Bin.cumulative | ~30 |
| Smooth.x_bins/x_estimator | ~80 |
| Smooth.output / Robust.output | ~40 |
| **Total 9b Rust** | **~1350 LOC** |

Plus ~200 LOC of fixture generators and tests. Consistent with Phase 5's footprint scaled for additional transforms.

---

## 6. Position-adjustment subsystem (9c)

A new orthogonal axis between encoding resolution and mark rendering. Phase 9c ships **all four** adjustments (`Identity`, `Dodge`, `Jitter`, `Stack`). All are required by `§3.14` figure functions.

### 6.1 The model

Position adjustment is a function `(resolved_coords, group_channel) → rewritten_coords` applied per-mark. Lives between the scale-resolve step and mark rendering.

### 6.2 Public API

```python
# src/ferrum/position.py

class Identity: ...                                            # explicit no-op
class Dodge:    def __init__(self, by=None, padding=0.05): ... # by defaults to color/fill
class Jitter:   def __init__(self, axis="x", width=0.4, seed=None): ...
class Stack:    def __init__(self, by=None, offset="zero"): ...  # offset: "zero"|"normalize"|"center"
```

Marks accept `position=` kwarg:

```python
Chart(data).mark_bar(position=Dodge(by="subcategory"))
Chart(data).mark_point(position=Jitter(axis="x", width=0.4))
Chart(data).mark_area(position=Stack(by="hue", offset="normalize"))
```

`position=None` (default) preserves Phase 8a/8b behavior.

### 6.3 ChartSpec / Rust shape

```rust
pub enum PositionAdjust {
    Identity,
    Dodge { by: Option<String>, padding: f64 },
    Jitter { axis: JitterAxis, width: f64, seed: Option<u64> },
    Stack { by: Option<String>, offset: StackOffset },
}

pub enum JitterAxis { X, Y, Both }
pub enum StackOffset { Zero, Normalize, Center }

pub struct Mark {
    // ... existing fields ...
    pub position: Option<PositionAdjust>,
}
```

JSON examples:

```json
{"type": "identity"}
{"type": "dodge", "by": "subcategory", "padding": 0.05}
{"type": "jitter", "axis": "x", "width": 0.4, "seed": null}
{"type": "stack", "by": "subcategory", "offset": "normalize"}
```

### 6.4 Mark eligibility matrix

| Adjustment | bar | point | box | swarm | violin | errorbar | errorband | ribbon | area | tick | rule | line | segment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Identity | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Dodge | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — | — | — |
| Jitter | — | ✓ | — | ✓ | — | — | — | — | — | ✓ | — | — | — |
| Stack | ✓ | — | — | — | — | — | — | ✓ | ✓ | — | — | — | — |

Marks where an adjustment doesn't apply raise `TypeError` at construction with the eligibility list shown.

### 6.5 Resolution semantics

**Dodge:**
- Group rows by the `by` channel value at each x-position; assign offsets within the band; rewrite x.
- Group ordering: `by` channel's categorical order; first-appearance fallback for unordered.
- Bandwidth: ordinal scales use scale bandwidth; continuous scales use median spacing between unique x-values (computed once per pass).

**Jitter:**
- Per-row noise drawn from uniform `[-width/2, +width/2]`, scaled by bandwidth (ordinal) or in absolute units (continuous).
- `seed` explicit → ChaCha8Rng seeded with that value.
- `seed=None` → ChaCha8Rng seeded with `twox_hash::xxh3::hash64` over `(x_value, y_value, group_value)` per row, byte-deterministic across runs of the same spec on the same data.

**Stack:**
- Group rows by `by` channel at each x; cumulative-sum y within group; rewrite y.
- `offset="zero"` — standard stack from y=0.
- `offset="normalize"` — divide each row's y by per-x total before cumulating (100% stacks).
- `offset="center"` — center-stacked (streamgraph).
- Bottom-to-top stacking follows the `by` channel's categorical order.

### 6.6 Implementation

| Adjustment | Rust LOC | Python LOC |
|---|---|---|
| Identity | ~5 | ~10 |
| Dodge | ~150 | ~30 |
| Jitter | ~80 | ~30 |
| Stack | ~150 | ~30 |
| Plumbing (Mark.position field, eligibility checks, PyO3 wrappers) | ~80 | ~80 |
| **Total** | **~465 Rust + ~180 Python** |

Rust pass lives in `crates/ferrum-core/src/render/position.rs`. PyO3 wrappers in `crates/ferrum-core/src/python/position.rs`. Python value classes in `src/ferrum/position.py`.

---

## 7. New marks (9d)

### 7.1 `mark_segment` (primitive)

```python
chart.mark_segment(...)
```

Line segment from `(x, y)` to `(x2, y2)`. Uses existing `X`, `Y`, `X2`, `Y2` encoding channels (already in the Phase 8b encoding system). Differs from `mark_rule` which is axis-aligned.

**Renderer:** routes to existing `SvgBuffer::line(x1, y1, x2, y2, …)` primitive without rule's axis-snap. ~30 LOC of routing.

**Position adjustment eligibility:** `Identity` only.

**Tests:** spec round-trip, 1 golden SVG (4-segment dendrogram-shape figure), eligibility errors for non-Identity position.

### 7.2 `mark_boxen` (composite)

```python
chart.mark_boxen(
    *,
    k_depth="proportion",     # "tukey"|"proportion"|"trustworthy"|"full"
    k_proportion=0.007,
    outlier_threshold=1.5,
    palette=None,             # continuous scheme override
)
```

**Composite expansion** (in `src/ferrum/marks/composite.py`, mirroring 8b's boxplot expansion):

1. **Nested rect bands** — one `mark_rect` layer per letter-value depth. Rect spans the letter-value range (lower → upper). Opacity decreases outward (default linear 0.85 → 0.30, multiplied against the resolved fill color).
2. **Median line** — `mark_rule` at the median value.
3. **Outliers** — `mark_point` for observations classified by `outlier_threshold` × outermost letter-value range.

The composite-mark builder declares the `LetterValue` transform on the spec; the engine runs it before rect layout. Resulting Chart spec contains layered rects + rule + points — fully deconstructable.

**Position adjustment eligibility:** `Dodge`, `Identity`. (Multi-group boxen across categories supports dodge.)

**Color gradient default:** opacity-based (0.85 outer → 0.30 inner), multiplied against resolved fill. `palette=` overrides to a continuous scheme from `ferrum.schemes`.

**Tests:** composite expansion produces N rect layers + 1 rule + 1 point layer; spec round-trip after expansion; goldens for single-group and multi-group dodge; E2E equivalence between `catplot(kind="boxen")` and direct `Chart.mark_boxen()`.

### 7.3 `PHASE_9_PLUS_MARKS` update

Before:
```python
PHASE_9_PLUS_MARKS = frozenset(["arc", "image", "geoshape", "segment", "label"])
```

After:
```python
PHASE_9_PLUS_MARKS = frozenset(["arc", "image", "geoshape", "label"])
```

`segment` removed. The remaining four are not blocked by any `§3.14` Group A function:
- `arc` — pie/donut, not in §3.14
- `geoshape` — maps, Phase 11+
- `image` — raster overlays; `mark_raster` (Phase 8b) covers heatmap-style raster
- `label` — text-with-callout decoration; not in §3.14

These stay deferred consistent with the no-defer rule applying to spec contracts ferrum currently advertises.

---

## 8. Figure-level functions (9a culmination)

Eight functions in `src/ferrum/figure/`. Each desugars to a `Chart` or compound view. All accept `theme=None` and `**encode_kwargs` (passed through to `.encode()` as overrides). Errors raised at construction time with valid-value lists.

### 8.1 `displot` — distributions

```python
ferrum.displot(data, *, x=None, y=None, hue=None, col=None, row=None,
               kind="hist",          # "hist"|"kde"|"ecdf"|"rug"
               fill=True, cumulative=False, log_scale=False, stat="count",
               bins="sturges", bandwidth="scott", bw_adjust=1.0,
               multiple="layer",     # "layer"|"stack"|"fill"|"dodge"
               kde=False, rug=False, height=None, aspect=None, theme=None)
```

**Desugaring:**
- `kind="hist"` → `mark_histogram(bins, cumulative, stat)`
- `kind="kde"` → `mark_density(bandwidth, bw_adjust, fill)`
- `kind="ecdf"` → `Bin(cumulative=True)` + line mark
- `kind="rug"` → `mark_tick` along x-axis
- `kde=True` / `rug=True` → layered marks
- `hue=` → `color=` encoding
- `multiple="layer"` → `position=Identity()`
- `multiple="dodge"` → `position=Dodge(by=hue)`
- `multiple="stack"` → `position=Stack(by=hue, offset="zero")`
- `multiple="fill"` → `position=Stack(by=hue, offset="normalize")`
- `col` / `row` → `.facet(col=col, row=row)`
- `log_scale=True` → x-scale = `LogScale`

Returns: `Chart`.

### 8.2 `catplot` — categorical

```python
ferrum.catplot(data, *, x=None, y=None, hue=None, col=None, row=None,
               kind="strip",         # "strip"|"swarm"|"box"|"violin"|"boxen"|"point"|"bar"|"count"
               order=None, hue_order=None, orient=None,
               dodge=False, jitter=True, native_scale=False,
               ci=95, n_boot=1000, seed=None, theme=None)
```

**Desugaring (by `kind`):**
- `"strip"` → `mark_point(position=Jitter(axis=orient_axis, width=0.4, seed=seed) if jitter else Identity())`
- `"swarm"` → `mark_swarm` (Phase 8b)
- `"box"` → `mark_boxplot`
- `"violin"` → `mark_violin`
- `"boxen"` → `mark_boxen` (9d)
- `"point"` → `mark_point` + `mark_errorbar` with bootstrap CI
- `"bar"` → `mark_bar` + `mark_errorbar` with bootstrap CI
- `"count"` → `mark_bar` with `Aggregate(op="count")`
- `dodge=True` and `hue` set → `position=Dodge(by=hue)`
- `order` / `hue_order` → ordinal scale domain override
- `native_scale=True` → use data's native scale instead of forced ordinal

Returns: `Chart` (layered when CI is present).

### 8.3 `lmplot` — linear and generalized regression

```python
ferrum.lmplot(data, *, x, y, hue=None, col=None, row=None,
              method="lm",           # "lm"|"logistic"|"glm"|"loess"|"robust"
              ci=95, order=1, scatter=True, scatter_kws=None, line_kws=None,
              truncate=False, x_bins=None, x_estimator=None, x_jitter=None,
              logx=False, theme=None)
```

**Desugaring:**
- `scatter=True` → bottom layer `mark_point` (`position=Jitter` if `x_jitter`)
- Top layer by `method`:
  - `"lm"` → `mark_smooth(method="lm", order, ci, x_bins, x_estimator)`
  - `"loess"` → `mark_smooth(method="loess", ci)`
  - `"logistic"` → new `Logistic` transform → line mark + ribbon CI
  - `"glm"` → new `Glm` transform → line mark + ribbon CI
  - `"robust"` → new `Robust` transform → line mark + ribbon CI
- `truncate=True` → restrict line range to observed x-range
- `logx=True` → x-scale = LogScale
- `hue` → per-group fits (color encoding + group-by in regression transform)
- `col` / `row` → faceting

Returns: layered `Chart` (`[scatter, fit]` or `[scatter, fit, ci_band]`).

### 8.4 `residplot` — residual diagnostics

```python
ferrum.residplot(data, *, x, y, lowess=False, order=1, robust=False,
                 dropna=True, label=None, color=None, theme=None)
```

**Desugaring:**
- Underlying fit: `Smooth(output="residuals")` if not robust, else `Robust(output="residuals")`
- Bottom layer: `mark_point` of (x, residual)
- Optional top layer: `mark_smooth(method="loess")` if `lowess=True`
- Reference line at y=0 via `annotate_hline(0)`
- `dropna=True` → filter NaN rows in `_coerce.py` ahead of spec construction

Returns: layered `Chart`.

### 8.5 `pairplot` — pairwise scatter grid

```python
ferrum.pairplot(data, *, vars=None, x_vars=None, y_vars=None,
                hue=None, kind="scatter",     # off-diagonal mark
                diag_kind="auto",             # "hist"|"kde"|None|"auto"
                markers=None, height=None, aspect=None,
                corner=False, dropna=False, theme=None)
```

**Desugaring:**
- Resolves `vars` / `x_vars` / `y_vars` to row + column field lists. Default: numeric columns.
- Builds a `RepeatChart`:
  - `template` = off-diagonal mark per `kind` with `x=Repeat.column, y=Repeat.row, color=hue`
  - `diagonal` = diag mark per `diag_kind` (`mark_histogram` or `mark_density`); `"auto"` = kde when n>1000 else hist; `None` = blank diagonal
  - `row=y_vars or vars`, `column=x_vars or vars`
  - `corner=corner`
- `markers` → shape encoding override per hue group

Returns: `RepeatChart`. `.expand()` yields all cells as concrete `Chart` objects.

### 8.6 `heatmap` — 2D matrix heatmap

```python
ferrum.heatmap(data, *, annot=True, fmt=".2f", cmap="blues",
               linewidths=0.5, linecolor="white",
               vmin=None, vmax=None, center=None, robust=False,
               square=False, mask=None, theme=None)
```

**Desugaring:**
- `Chart(data)` → `Unpivot` transform (id_vars = row index, var_name="column", value_name="value")
- Mark: `mark_rect(stroke=linecolor, stroke_width=linewidths)` with `x="column", y="row", fill="value"`
- Color scale: `ContinuousScheme(name=cmap, domain=[vmin or auto, vmax or auto])`; `center=` shifts to a diverging scale centered on the value
- `robust=True` → vmin/vmax from 2nd/98th percentiles in Python-side coercion (data inspection, not transformation)
- `annot=True` → layered `mark_text(text="value", format=fmt)` on top
- `mask` → boolean matrix; masked cells get fill=transparent (handled via a Filter transform on unpivoted data)
- `square=True` → `Chart.properties(width=H, height=H)` shared

Returns: `Chart` (2-layer if `annot=True`).

### 8.7 `clustermap` — clustered heatmap with dendrograms

```python
ferrum.clustermap(data, *, method="ward", metric="euclidean",
                  cmap="viridis", z_score=None, standard_scale=None,
                  figsize=None, dendrogram_ratio=0.2, theme=None)
```

**Desugaring (most complex of the 8):**

1. Two `Linkage` transforms — one per axis (rows, columns).
2. **Top dendrogram** (column): `Chart(data).transform_linkage(name="col_link", axis="columns", method, metric, z_score, standard_scale).mark_segment()` consuming `coords` + `linkage` named outputs.
3. **Left dendrogram** (row): same with `axis="rows"`, rotated 90°.
4. **Center heatmap**: `Chart(data)` with `Reorder` (using row_link.order) + `Reorder` (using col_link.order) + `Unpivot` + `mark_rect` (heatmap encoding).
5. Compose as `ClusterMapChart(heatmap, row_dendrogram=..., col_dendrogram=..., dendrogram_ratio=dendrogram_ratio)`.

Returns: `ClusterMapChart`.

### 8.8 `jointplot` — joint distribution

```python
ferrum.jointplot(data, *, x, y, hue=None,
                 kind="scatter",       # "scatter"|"kde"|"hist"|"hex"|"reg"
                 marginal_kind="hist", # "hist"|"kde"|"rug"|"box"
                 ratio=5, space=0.05,
                 xlim=None, ylim=None,
                 joint_kws=None, marginal_kws=None,
                 height=None, theme=None)
```

**Desugaring:**
- `center` chart by `kind`:
  - `"scatter"` → `mark_point`
  - `"kde"` → `mark_contour` (Phase 8b) with Kde2D
  - `"hist"` → `Bin2D` + `mark_rect`
  - `"hex"` → `mark_hex` (Phase 8b)
  - `"reg"` → `mark_smooth` overlaid on `mark_point`
- `top` marginal: `Chart(data).<marginal_mark>(x=x)` per `marginal_kind`
- `right` marginal: `Chart(data).<marginal_mark>(y=y)`, oriented vertically
- `JointChart(center, top, right, ratio=ratio, spacing=space)`
- `xlim` / `ylim` → x/y scale domain overrides on center
- `joint_kws`, `marginal_kws` → passed to respective Chart constructors

Returns: `JointChart`.

### 8.9 Module layout

```
src/ferrum/figure/
  __init__.py              # re-exports the 8 functions
  distribution.py          # displot
  categorical.py           # catplot
  regression.py            # lmplot, residplot
  matrix.py                # pairplot, heatmap, clustermap
  joint.py                 # jointplot
```

`src/ferrum/__init__.py` re-exports all 8 functions plus `Repeat`, `JointChart`, `RepeatChart`, `ClusterMapChart`, `Identity`, `Dodge`, `Jitter`, `Stack`, the new transforms, and the new marks.

---

## 9. Testing strategy

### 9.1 Test surface

**Rust unit tests (`cargo test`):**
- New transforms (Unpivot, Linkage, Reorder, Bin2D, Logistic, Glm, Robust, LetterValue) — round-trip JSON + numeric correctness against fixtures
- Bin/Smooth extensions (`cumulative`, `x_bins`, `x_estimator`, `output="residuals"`) — round-trip + correctness
- Position adjustments (Identity, Dodge, Jitter, Stack) — JSON round-trip, applied-to-coords correctness, Jitter determinism with explicit seed AND with seed=None hash fallback
- New marks (segment, boxen) — spec round-trip, eligibility errors
- Compound view specs (JointChart, RepeatChart with diagonal, ClusterMapChart) — JSON round-trip, expansion semantics for RepeatChart
- Grid compositor (`compose_svg_grid`) — geometry math for given ratios + spacing

**Python unit tests:**
- New compound view classes — construction, `.theme()` propagation, `.charts`, `.expand()`, `.spec`
- `Repeat` typed sentinel — placeholder serialization, encoding-spec parsing
- Position adjustment Python value classes — `Dodge`/`Jitter`/`Stack`/`Identity` immutability, JSON round-trip
- `mark_segment` and `mark_boxen` — Chart method APIs, composite expansion of boxen

**Figure-level function tests (`tests/test_phase_9_figures.py`):**

For each of the 8 functions: representative call with all common parameters, asserting `.spec` (or `.charts`) is well-formed and JSON-round-trips. Per-parameter coverage:

| Function | Coverage |
|---|---|
| `displot` | 4 `kind` × `multiple` ∈ {layer, stack, fill, dodge} × cumulative ∈ {True, False} |
| `catplot` | 8 `kind` × dodge ∈ {True, False} |
| `lmplot` | 5 `method` × ci ∈ {None, 95} |
| `pairplot` | vars vs x_vars/y_vars, corner ∈ {True, False}, diag_kind ∈ {hist, kde, None, auto} |
| `heatmap` | annot ∈ {True, False}, robust ∈ {True, False}, mask passed |
| `clustermap` | each (method, metric) tested at least once |
| `jointplot` | 5 `kind` × 4 `marginal_kind` |
| `residplot` | lowess ∈ {True, False}, robust ∈ {True, False} |

**E2E rendering tests (`tests/test_phase_9_e2e.py`):**

12 new SVG goldens total:
- 1 per figure-level function (8 goldens)
- 4 additional tricky cases:
  - pairplot 3×3 with hue
  - clustermap with row+column dendrograms
  - jointplot with kde marginals
  - displot stacked histogram

### 9.2 Fixture generators

`crates/ferrum-core/src/transform/fixtures/`:
- `generate_linkage_refs.py` — scipy reference for the curated 5-pair coverage subset
- `generate_glm_refs.py` — statsmodels reference for canonical-link cases + 3 non-canonical
- `generate_logistic_refs.py` — statsmodels reference for 5 logistic test datasets
- `generate_robust_refs.py` — statsmodels reference for 4 robust-regression test datasets
- `generate_letter_value_refs.py` — numpy quantile reference (no third-party lib)

`requirements-fixtures.txt` updated to add `statsmodels>=0.14` (dev-only). `scipy` already pinned.

### 9.3 Verification step in plan

Before `kodama` is added to `Cargo.toml`: verify last release < 18 months, license MIT/Apache, no critical open issues. If verification fails, fall back to hand-roll Lance-Williams + nearest-neighbor chain (~400-600 LOC).

---

## 10. Done criteria

Phase 9 is `done` when all of:

- [ ] `cargo test` passes (transform, mark, position, compound-view round-trip + correctness)
- [ ] `uv run pytest` passes (Python-side tests including figure-level function tests)
- [ ] All 12 golden SVGs match byte-identically across runs
- [ ] All 8 figure-level functions in `ferrum-spec.md §3.14` Group A are implemented
- [ ] Each figure-level function's `.spec` (or `.charts` / `.expand()`) returns a valid object — verified by a structural test for each
- [ ] All four position adjustments (`Identity`, `Dodge`, `Jitter`, `Stack`) ship with mark eligibility enforced
- [ ] `PHASE_9_PLUS_MARKS` no longer contains `segment`
- [ ] `ferrum-spec.md` has all dated drift notes from §11 below applied
- [ ] `docs/superpowers/ferrum-phases.md` Phase 9 row marked `done`

---

## 11. Spec drift notes (consolidated, all dated 2026-05-10)

To be applied as inline notes to `ferrum-spec.md`:

| Section | Note |
|---|---|
| §3.2 (Encoding Channels) | Add "Position adjustments" subsection: `Identity`, `Dodge`, `Jitter`, `Stack` accepted via `position=` on eligible marks. Mark eligibility matrix included. |
| §3.3 (Primitive Marks) | Add `mark_segment` row (line segment from (x, y) to (x2, y2); diagonal-capable, distinct from axis-aligned mark_rule). |
| §3.3 (Composite Marks) | Add `mark_boxen` row with `k_depth`, `k_proportion`, `outlier_threshold`, `palette` parameters. |
| §3.4 (Stat Transforms) | Add `Unpivot`, `Linkage` (with three named outputs), `Reorder`, `Bin2D`, `Logistic`, `Glm` (with family/link compatibility table), `Robust`, `LetterValue`; document `Bin.cumulative`, `Smooth.x_bins`, `Smooth.x_estimator`, `Smooth.output`/`Robust.output` parameter additions. |
| §3.12 (Compound Views) | Implementation note for `JointChart` (lands in Phase 9 honoring §3.12 contract). `RepeatChart` gains `diagonal=` and `corner=` parameters; `Repeat.column`/`Repeat.row`/`Repeat.layer` typed sentinels documented (no string sentinels). New `ClusterMapChart` compound view added with documented contract (used by `clustermap()`). |
| §3.14 (Figure-Level Functions) | Note: all 8 Group A functions land in Phase 9 with all parameters honored. Group B (21 model-diagnostic figure-level functions) remains in Phase 10 alongside `ModelSource` and the model-diagnostic marks they depend on. |

---

## 12. Open verification tasks (resolved before implementation)

These tasks land in the writing-plans output and must complete before their dependent code is written:

1. **Verify `kodama` crate suitability** — last release date < 18 months, license MIT/Apache-2.0, no critical open issues. If fails: fall back to hand-rolled Lance-Williams (~400-600 LOC).
2. **Verify `statsmodels` API stability** for IRLS / GLM / RLM endpoints used in fixture generation. Pin `statsmodels>=0.14,<0.16`.
3. **Confirm `twox-hash` crate version** for Jitter's seed-fallback hash. Pin to a 1.x release at the time of implementation; document chosen version in the plan.

---

## 13. Cross-cutting principles (recorded for the implementation phase)

- **No further deferrals.** Per `CLAUDE.md` Implementation philosophy section, sub-phases manage build order, not scope.
- **Deconstructable by structure.** Every figure-level function returns a `Chart` or compound view; no direct-to-render path.
- **No matplotlib.** Reaffirmed; statsmodels and scipy are dev-only fixture generators, never imported at runtime.
- **No new global state.** Position adjustment, compound views, transforms — all immutable values. The only sanctioned mutator remains `set_default_theme`.
- **`ferrum-spec.md` is the contract.** Drift notes in §11 land alongside the implementation; spec is updated, not silently diverged.
