# Phase 5 — Stat Engine — Design Spec

**Date:** 2026-05-09
**Phase:** 5 (depends on 3; sibling of 4)
**Status:** approved → ready for implementation plan
**Concept spec:** `ferrum-spec.md` §3.4 (Stat Transforms), §3.5 (Data Transforms)
**Phases doc:** `docs/superpowers/ferrum-phases.md`

---

## 1. Goal

Implement the stat-transform layer in Rust: a small set of statistical primitives declared in `ChartSpec` and executed by the engine before layout. Phase 5 is the "data prep" stage of the future render pipeline (`ChartSpec.data → transforms → scales → layout → render`).

The phases-doc done criteria are binding:

- KDE, bootstrap CI, linear regression, LOESS, binning, and basic aggregation implemented in Rust
- Each transform declared in a `ChartSpec` and executed by the engine before layout
- `cargo test` covers numeric correctness against a reference (scipy/numpy values computed offline)

---

## 2. Scope

Phase 5 ships **5 transforms** that cover the 6 done-criteria capabilities. The mapping:

| Done criterion | Phase 5 transform | Notes |
|---|---|---|
| Binning (Sturges floor) | `stat_bin` | Reuses `crate::scale::ticks::sturges_floor` |
| KDE | `stat_kde` | Gaussian kernel only; Scott / Silverman / Fixed bandwidth |
| Linear regression | `stat_smooth { method = "lm" }` | One transform, two methods (matches §3.4) |
| LOESS | `stat_smooth { method = "loess" }` | Degree configurable `{1, 2}`, default `2` |
| Basic aggregation | `stat_aggregate` | Group-by with mean / sum / count / min / max / median |
| Bootstrap CI | `stat_summary` | Mean + bootstrap CI per group |

### Naming choice (named decision)

Phase 5 follows §3.4's `stat_*` names rather than §3.5's `transform_*` aliases. §3.5's `transform_regression` and `transform_loess` are duplicate spellings of `stat_smooth(method=...)`. Phase 5 implements the canonical `stat_*` names; the `transform_*` aliases can be added later as Python-side sugar in Phase 8 (grammar API) without touching the engine.

### Out of scope (explicit, deferred)

- Other §3.4 stats: `stat_ecdf`, `stat_qq`, `stat_contour`, `stat_kde_2d`, `stat_bin_2d`, `stat_identity`
- All model stats (`stat_roc`, `stat_pr`, `stat_confusion`, `stat_calibration`, `stat_lift`, `stat_importance`, `stat_shap`, `stat_pdp`, `stat_residuals`, `stat_learning_curve`, `stat_validation_curve`) — Phase 10
- All §3.5 data-reshape transforms (`transform_filter`, `transform_calculate`, `transform_fold`, `transform_pivot`, `transform_window`, `transform_join_aggregate`, `transform_impute`, `transform_flatten`, `transform_sample`, `transform_top_k`, `transform_stack`, `transform_timeunit`)
- Statistical mark expansion (e.g., `mark_density` auto-inserting `stat_kde`) — Phase 7
- KDE kernels other than gaussian — Phase 12 (extension points)

---

## 3. Architecture

### 3.1 Module layout

```
crates/ferrum-core/src/
├── lib.rs                       # +mod transform; +pyclass exports
├── spec/
│   └── chart.rs                 # +pub transforms: Vec<TransformSpec>
└── transform/                   # NEW
    ├── mod.rs                   # pub mod core; pub mod {bin,kde,smooth,aggregate,summary,linalg};
    ├── core.rs                  # TransformSpec enum, apply_transforms pipeline driver
    ├── bin.rs                   # stat_bin: histogram + Sturges floor
    ├── kde.rs                   # stat_kde: gaussian KDE
    ├── smooth.rs                # stat_smooth: linear regression + LOESS
    ├── aggregate.rs             # stat_aggregate: group-by aggregations
    ├── summary.rs               # stat_summary: bootstrap CI
    └── linalg.rs                # solve_3x3_spd helper for LOESS degree=2
```

This mirrors Phase 4's `scale/` layout one-for-one (`mod.rs` + `core.rs` + per-variant files + a small shared utility module).

### 3.2 Sealed-enum shape

In `transform/core.rs`:

```rust
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TransformSpec {
    Bin(BinSpec),
    Kde(KdeSpec),
    Smooth(SmoothSpec),
    Aggregate(AggregateSpec),
    Summary(SummarySpec),
}

impl TransformSpec {
    pub fn apply(&self, batch: &RecordBatch) -> PyResult<RecordBatch> {
        match self {
            Self::Bin(s)       => bin::apply(s, batch),
            Self::Kde(s)       => kde::apply(s, batch),
            Self::Smooth(s)    => smooth::apply(s, batch),
            Self::Aggregate(s) => aggregate::apply(s, batch),
            Self::Summary(s)   => summary::apply(s, batch),
        }
    }
}

pub fn apply_transforms(specs: &[TransformSpec], batch: &RecordBatch)
    -> PyResult<RecordBatch>
{
    let mut current = batch.clone();   // Arrow Arc-clone; cheap
    for spec in specs {
        current = spec.apply(&current)?;
    }
    Ok(current)
}
```

JSON round-trips trivially via serde's tagged-enum representation; Python construction is straightforward; no `dyn` dispatch.

### 3.3 ChartSpec extension (backward-compatible)

In `spec/chart.rs`:

```rust
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChartSpec {
    pub mark: String,
    pub x: Option<EncodingSpec>,
    pub y: Option<EncodingSpec>,
    pub data: Option<DataSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<TransformSpec>,
}
```

- `#[serde(default)]` means existing Phase 3 round-trip JSON (which doesn't include `transforms`) deserializes cleanly.
- `skip_serializing_if = "Vec::is_empty"` keeps existing JSON outputs byte-identical when no transforms are declared.
- All Phase 3 cargo tests for `ChartSpec` continue to pass without modification.

### 3.4 Python-facing surface

Each variant gets a thin `#[pyclass]` constructor returning a `TransformSpec`:

- `Bin(field, *, bin_count=None, bin_width=None, extent=None, nice=True)`
- `Kde(field, *, bandwidth="scott", n=512, extent=None, cumulative=False)`
- `Smooth(x, y, *, method="loess", ci=0.95, bandwidth=0.75, degree=2, n=200, seed=0)`
- `Aggregate(ops, *, groupby=None)` where `ops` is a list of `AggregateOp(field, fn, as_)`
- `Summary(field, *, groupby=None, error_fn="ci", ci=0.95, n_boot=1000, seed=0)`

`ChartSpec.__init__` gains a `transforms: list[Transform] = []` parameter. `_core.pyi` stubs are updated for all five constructors plus `AggregateOp`.

`bandwidth`, `method`, `error_fn` are accepted as Python strings/floats and parsed into typed Rust enums at construct time (Phase 4 precedent).

---

## 4. Per-transform contracts

These output schemas are commitments that Phase 7 will encode against.

### 4.1 `stat_bin`

```rust
pub struct BinSpec {
    pub field: String,
    pub bin_count: Option<usize>,
    pub bin_width: Option<f64>,
    pub extent: Option<(f64, f64)>,
    pub nice: bool,
}
```

- **Input:** batch must contain `field` as `Float64`. Nulls and NaN are dropped before binning.
- **Output:** `bin_start: Float64`, `bin_end: Float64`, `count: UInt64`, `density: Float64`. One row per bin. Input column **not** carried through.
- **Defaults:** if neither `bin_count` nor `bin_width` set, `bin_count = sturges_floor(n_non_null)` (imported from `crate::scale::ticks`). `nice=true` rounds the extent to "nice" boundaries.
- **Numeric edge:** all-equal `field` → single bin spanning `[v - 0.5, v + 0.5]` with full count.

### 4.2 `stat_kde`

```rust
pub struct KdeSpec {
    pub field: String,
    pub bandwidth: BandwidthSpec,    // Scott | Silverman | Fixed(f64)
    pub n: usize,
    pub extent: Option<(f64, f64)>,
    pub cumulative: bool,
}
```

- **Input:** `field` as `Float64`; nulls/NaN dropped.
- **Output:** `value: Float64`, `density: Float64`. `n` rows (default 512 per spec §3.4). Input column **not** carried through.
- **Bandwidth:**
  - Scott: `sigma * n^(-1/5)`
  - Silverman: `0.9 * min(sigma, IQR/1.34) * n^(-1/5)`
  - Validated against `scipy.stats.gaussian_kde(bw_method="scott" | "silverman")` in fixtures (scipy version pinned in script header).
- **Cumulative:** when `true`, returns the running integral via trapezoidal rule on the same `value` grid.
- **Numeric edge:** n<2 or zero variance → `density` column all NaN; `value` grid still emitted.

### 4.3 `stat_smooth`

```rust
pub struct SmoothSpec {
    pub x: String,
    pub y: String,
    pub method: SmoothMethod,        // Lm | Loess
    pub ci: Option<f64>,
    pub bandwidth: f64,              // LOESS only; fraction of data per local window, in (0, 1]
    pub degree: u8,                  // LOESS only; 1 or 2
    pub n: usize,
    pub seed: u64,                   // LOESS bootstrap CI; ignored when method=Lm
}
```

- **Input:** `x`, `y` as `Float64`; rows with null/NaN in either dropped.
- **Output:** `x: Float64`, `y: Float64`, `ci_lower: Float64`, `ci_upper: Float64` (latter two NaN when `ci=None`). `n` evaluation points (default 200 per spec).
- **Lm:** OLS on (x, y); evaluate at `n` equally-spaced points across observed extent. CI is the analytic **confidence interval for the conditional mean** at level `ci` (e.g., 0.95 → mean ± `t(α/2, n−2) × SE_fit(x)`); not a prediction interval for individual observations.
- **Loess:** locally-weighted polynomial of `degree` ∈ {1, 2}. Tricube weights. CI band computed via bootstrap with `ChaCha8Rng::seed_from_u64(seed)`; default `seed = 0`. Used only when `method=Loess`; ignored for `Lm`.
  - degree=1 uses a closed-form 2×2 weighted normal-equations solve.
  - degree=2 uses the `solve_3x3_spd` helper in `transform/linalg.rs` (Cholesky on weighted normal equations).
- **Numeric edge:** n<2 → all-NaN output; zero variance in x → all-NaN line; LOESS local window with n<degree+1 → that point's `y`/`ci_*` = NaN.

### 4.4 `stat_aggregate`

```rust
pub struct AggregateSpec {
    pub ops: Vec<AggregateOp>,
    pub groupby: Vec<String>,
}

pub struct AggregateOp {
    pub field: String,
    pub fn_: AggFn,                  // Mean | Sum | Count | Min | Max | Median
    pub as_: String,
}
```

- **Input:** all `field` columns referenced in `ops` must exist as `Float64`; groupby columns must exist as `Utf8` or `Float64`.
- **Output:** one column per groupby key (preserving its dtype) + one `Float64` column per op named by `as_`. One row per unique group. Empty `groupby` produces a single global row.
- **Numeric edge:** all-null group on a field → that op outputs NaN. `Count` always returns `UInt64` cast to `Float64` for schema uniformity across ops.

### 4.5 `stat_summary`

```rust
pub struct SummarySpec {
    pub field: String,
    pub groupby: Vec<String>,
    pub error_fn: ErrorFn,           // Ci | Stderr | Stdev
    pub ci: f64,
    pub n_boot: usize,
    pub seed: u64,
}
```

- **Input:** `field` as `Float64`; groupby columns mandatory by name (empty `groupby` → single global row).
- **Output:** groupby columns + `mean: Float64`, `lower: Float64`, `upper: Float64`. One row per group.
- **Bootstrap CI:** percentile method on `n_boot` resamples with `ChaCha8Rng::seed_from_u64(seed)`. Defaults: `seed=0`, `n_boot=1000`, `ci=0.95`.
- **Stderr:** `mean ± stderr`. **Stdev:** `mean ± stdev`. Both analytic, no RNG.
- **Numeric edge:** group with n<2 → `lower`, `upper` = NaN.

---

## 5. Composition pipeline

`ChartSpec.transforms: Vec<TransformSpec>` is a **sequential pipeline**. `apply_transforms` iterates in declaration order and pipes the output of step `i` as input to step `i+1`. Schema mismatch between adjacent transforms (e.g., `stat_kde` followed by `stat_aggregate(field="missing_col")`) raises `PyValueError` at apply time.

Most natural compositions in Phase 5:

- `[stat_aggregate]` alone → grouped summary table
- `[stat_bin, stat_aggregate]` → histogram with secondary aggregation per bin
- `[stat_summary]` alone → mean ± CI per group
- `[stat_kde]` or `[stat_smooth]` standalone (their fresh-table outputs don't chain naturally with grouped reductions)

DAG composition with named outputs is explicitly deferred. Phase 5's sequential semantics match Vega-Lite's transform array convention.

---

## 6. Error policy (hybrid)

### Construction-time (`PyValueError` from `__new__`)

- `bin_count <= 0` or `bin_width <= 0`
- `n <= 0` (KDE / smooth grid resolution)
- KDE `bandwidth <= 0` or NaN when given as `Fixed(f64)`
- Smooth `bandwidth <= 0 || bandwidth > 1.0` or NaN — validated only when `method = Loess`; ignored for `Lm`
- `n_boot <= 0`
- `ci <= 0 || ci >= 1`
- LOESS `degree ∉ {1, 2}`
- duplicate field name within `groupby`
- empty `ops` for `stat_aggregate`
- unknown enum string for `bandwidth`, `method`, `error_fn`, `fn_`

### Runtime at `apply` (`PyResult::Err(PyValueError::new_err(...))`)

- referenced column missing from the input batch
- referenced column has wrong dtype (Float64 expected, anything else; groupby keys accept Utf8 or Float64)
- input batch has zero rows when the transform requires at least one observation (`stat_bin` is the exception — empty input → empty output, no error)

### Numeric edges (NaN propagation, no error)

- KDE on n<2 or zero variance → density column all NaN
- LOESS local window with n<degree+1 → that point's y/ci_* = NaN
- LM on zero-variance x → all-NaN output line
- Bootstrap CI on group with n<2 → lower, upper = NaN
- Aggregate of all-null group on a field → NaN for that op

This mirrors Phase 4's d3-aligned NaN-at-runtime convention for genuinely undefined math while keeping structural mismatches loud.

---

## 7. New Rust dependencies (named decision)

CLAUDE.md's locked decisions say "no external deps beyond what Phase 1–4 introduced." Phase 5 introduces two named exceptions:

| Crate | Version | Why |
|---|---|---|
| `rand` | `0.8` | Trait abstractions for RNG; `SliceRandom::choose` for bootstrap resampling |
| `rand_chacha` | `0.3` | `ChaCha8Rng` — deterministic, seeded, reproducible across platforms (matters for committed numeric-reference fixtures) |

Pinned in `[workspace.dependencies]`. No new system requirements, no FFI, no transitive native deps. Compile-time impact is a few seconds.

**No** new dependencies for: linear algebra (`linalg.rs` is hand-rolled 3×3 Cholesky), interpolation (closed-form trapezoidal in `kde.rs`), or sorting (`slice::sort_unstable_by`).

---

## 8. Numeric reference & test plan

### 8.1 Reference generator (committed)

- `crates/ferrum-core/tests/fixtures/generate_stat_refs.py` — script that uses scipy and numpy to compute expected outputs for fixed input arrays.
- Header pins exact scipy + numpy versions used to generate the fixtures.
- Output: `crates/ferrum-core/tests/fixtures/stat_refs.json`, consumed via `serde_json::from_str(include_str!("..."))`.
- Companion `crates/ferrum-core/tests/fixtures/requirements-fixtures.txt` for reproducible regeneration via `uv pip install -r`.
- Re-run when scipy is bumped; commit the regenerated JSON in the same commit as the version bump.

This keeps `cargo test` fully hermetic — no Python at test time, no scipy in the dev environment for Rust tests.

### 8.2 Test layout

```
crates/ferrum-core/tests/
├── fixtures/
│   ├── generate_stat_refs.py
│   ├── requirements-fixtures.txt
│   └── stat_refs.json
└── stat/
    ├── bin.rs
    ├── kde.rs
    ├── smooth.rs
    ├── aggregate.rs
    ├── summary.rs
    └── pipeline.rs        # composition + schema-mismatch tests
```

### 8.3 Per-transform test minima

| Transform | Tests |
|---|---|
| `stat_bin` | (a) hardcoded histogram counts vs numpy, (b) Sturges floor default, (c) all-equal data edge, (d) extent override, (e) round-trip `TransformSpec` ↔ JSON |
| `stat_kde` | (a) Scott/Silverman/Fixed bandwidths vs scipy `gaussian_kde` to 1e-6 absolute, (b) cumulative integrates to ~1.0, (c) zero-variance NaN edge, (d) round-trip JSON |
| `stat_smooth` | (a) LM coefficients vs `numpy.polyfit(deg=1)` exactly, (b) LM 95% confidence band for the conditional mean vs numpy reference (`SE_fit(x) = σ̂·√(1/n + (x−x̄)²/Σ(xᵢ−x̄)²)`), (c) LOESS degree=1 vs hand-computed reference, (d) LOESS degree=2 vs hand-computed reference, (e) zero-variance NaN edge, (f) bootstrap LOESS CI deterministic under fixed seed, (g) round-trip JSON |
| `stat_aggregate` | (a) each `AggFn` on grouped data vs numpy, (b) multiple ops on same field, (c) two-key groupby, (d) all-null group → NaN, (e) round-trip JSON |
| `stat_summary` | (a) bootstrap CI with `seed=42` vs numpy bootstrap with same seed (deterministic, exact), (b) stderr/stdev analytic, (c) n=1 group → NaN, (d) round-trip JSON |
| Pipeline | `[stat_bin, stat_aggregate]` end-to-end against numpy reference; `apply_transforms` on empty `Vec` returns input unchanged |
| Schema mismatch | `stat_aggregate { field: "ghost" }` after a transform that drops the column → `PyValueError` |

### 8.4 Tolerances

- `1e-9` absolute for analytic operations (LM coefficients, exact arithmetic on integers, sums)
- `1e-6` absolute for KDE (scipy normalization differences are bounded but nonzero)
- Exact bit-for-bit match for bootstrap CI under fixed seed

### 8.5 Python-side smoke tests

`tests/test_stat_engine.py` — one happy-path test per transform, constructed from a polars DataFrame, plus one `ChartSpec` round-trip with transforms attached.

---

## 9. Done-criteria gate (Phase 5 complete when all pass)

- [ ] `cargo test -p ferrum-core` passes — target ~25 new tests on top of the existing 73
- [ ] `uv run pytest` passes — target ~10 new tests on top of 46
- [ ] Smoke verification:
  ```
  unset CONDA_PREFIX && uv run --no-sync python -c "from ferrum._core import ChartSpec, Bin; \
    spec = ChartSpec(mark='bar', x='x', transforms=[Bin(field='x')]); \
    assert ChartSpec.from_json(spec.to_json()) == spec; print('OK')"
  ```
- [ ] Phase 3's existing JSON round-trip tests still pass without modification (validates the `#[serde(default)]` extension)

---

## 10. Decisions locked during brainstorming (2026-05-09)

| Decision | Choice | Reasoning |
|---|---|---|
| Scope | 6 done-criteria capabilities → 5 transforms (`stat_bin`, `stat_kde`, `stat_smooth`, `stat_aggregate`, `stat_summary`) | Matches phase done-criteria exactly; stat_smooth unifies linear regression + LOESS per §3.4 |
| ChartSpec wiring | `transforms: Vec<TransformSpec>` field + `apply_transforms` engine | Required by done criterion ("declared in a ChartSpec and executed by the engine") |
| Internal Rust shape | Tagged enum + per-module impl methods (Phase 4 precedent) | JSON round-trips trivially; no dyn dispatch; matches existing pattern |
| Numeric references | Committed generator script + hardcoded constants | Hermetic `cargo test`; no scipy at test time; reproducible |
| Error policy | Hybrid: structural → `PyValueError`, numeric → NaN | Matches Phase 4's d3 alignment for math while keeping bugs loud |
| Composition | Sequential pipeline (declaration order) | Matches Vega-Lite convention; predictable; small surface |
| LOESS degree | Configurable {1, 2}, default 2 | Matches §3.4 spec default; degree=1 is a one-branch fast path |
| Naming | `stat_*` from §3.4 (not `transform_*` from §3.5) | Canonical form per the spec; aliases can land in Phase 8 |
| New deps | `rand 0.8` + `rand_chacha 0.3` | Required for seeded reproducible bootstrap; pinned, widely-used, no FFI |

---

## 11. Cross-phase notes

- Phase 7 (renderer) consumes Phase 5's output: the pipeline becomes `ChartSpec.data → transforms (Phase 5) → scales (Phase 4) → layout (Phase 6) → render (Phase 7)`.
- Phase 8 (grammar API) can add `transform_*` aliases as Python sugar over `stat_*`.
- Phase 12 (extension points) will need a public `Transform` trait. Phase 5 keeps the enum private to `ferrum-core`; the trait is a backward-compatible addition later (variants can `impl Transform`, and `Box<dyn Transform>` becomes a separate "extension" path that doesn't touch the sealed enum).
- Statistical mark expansion (e.g., `mark_density` auto-inserting `stat_kde`) is Phase 7's responsibility.

---

## 12. Test count baseline (at start of Phase 5)

- `cargo test -p ferrum-core`: 73 passing
- `uv run pytest`: 46 passing (1 smoke + 4 transport + 13 chart_spec + 28 scales)

Phase 5 target: +25 cargo tests, +10 pytest tests. Both gates must pass before the phase is marked done.
