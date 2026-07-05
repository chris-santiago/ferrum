# Ferrum — Key Architectural Decisions

Decisions extracted from `CLAUDE.md` on 2026-05-15 to keep the project instructions lean.
Read this when making changes that cross subsystem boundaries or introduce new dependencies.

---

## Build & packaging

- **Build backend:** `maturin >= 1.7`
- **Workspace:** Cargo workspace at repo root; `ferrum-core` is the computation engine; `ferrum-wasm` is the WASM renderer (Phase 11, active); `ferrum-shared` is reserved for future phases
- **ABI:** `abi3-py310` — one wheel per platform-arch, works for Python ≥ 3.10
- **`extension-module` feature:** feature-gated (not unconditional) so `cargo test` can link libpython
- **PyO3 version:** pinned in `[workspace.dependencies]`; re-verify against crates.io at the start of any session that adds a new PyO3 API

---

## Data transport

**Arrow C Data Interface via `pyo3-arrow`** (phase 2+); NOT Arrow IPC bytes.

Polars DataFrames implement `__arrow_c_stream__` natively — CDI hands off the buffer pointer directly with zero copies. Before phase 2, no DataFrames cross the boundary.

Decision made 2026-05-09: spec originally said "Arrow IPC" but CDI was chosen for zero-copy polars support.

---

## Chart spec serialization

**JSON via `serde` + `serde_json`** (phase 3+); NOT Arrow schema metadata, NOT a binary codec.

The `ChartSpec` IR is a tree-structured config (mark, encodings, scales, transforms, layers) — JSON matches the public `chart.to_json()` API in `ferrum-spec.md §3.1` and `§3.16`, evolves cleanly across phases as new optional fields are added, and stays human-readable for debugging and test fixtures. Spec size is small (KB), so binary-codec performance gains are irrelevant. Vega-Lite interop (phase 7+ `engine="vega-lite"` output) stays open without translation.

Decision made 2026-05-09: phases doc said "JSON or Arrow schema" — Arrow schema rejected because it describes columns, not config trees, and would require flattening to dotted keys or embedding JSON inside metadata.

---

## DataFrame compatibility layer

**`narwhals` (~1.x)** added (phase 8a+) for non-polars DataFrame inputs (pandas, modin, cuDF, dask, ibis).

Direct CDI path preserved for `polars.DataFrame` and pyarrow `Table`/`RecordBatch`; everything else flows through `narwhals.from_native(data, eager_only=True).to_arrow()`. Dict-of-arrays, list-of-records, and 2D numpy handled by direct `pyarrow.Table.from_*` branches in `src/ferrum/_coerce.py`.

Decision made 2026-05-10: alternative was ~250 LOC of in-house pandas dtype normalization; narwhals owns those bugs, ships modin/cuDF/dask/ibis support for free, and is the same compatibility layer altair adopted in 2024 for an identical problem.

---

## Multi-layer ChartSpec

`layers: Option<Vec<Layer>>` additive field on `ChartSpec` (phase 8a+).

When `layers.is_none()`, the renderer uses single-layer `mark` + `encoding` and the JSON shape is byte-identical to phases 3–7 — existing goldens stay valid. When `layers.is_some()`, the renderer iterates layers within each panel, sharing x/y/color scales by default.

**One `ChartSpec` = one `RecordBatch`** is load-bearing: mixed-data layered charts (`Chart(df1) + Chart(df2)`) merge via null-padded diagonal concat into one batch rather than growing multi-batch logic in the renderer.

Decision made 2026-05-10. (Updated 2026-07-05: the SVG string compositor this
section originally cited was retired by the composite render unification —
see "Composition rendering" below.)

---

## Composition rendering

**One Rust composite entry per output kind; no Python-side scene or SVG merging.**

Every composition form (HConcat/VConcat/Concat wrap grids, JointChart,
ClusterMapChart, RepeatChart, LayerChart overlays) lowers in Python to a
composite spec tree — `{"kind": "leaf"|"composite"|"hole", "layout":
"hconcat|vconcat|grid|wrap|overlay", children, resolve, spacing, ratios,
root-only title/subtitle/caption/config, per-child label}` — and renders
through `render_composite_svg` / `render_composite_interactive` in one call.
Rust owns all three passes: scale resolution across leaves (`resolve=` sharing
via congruent tree-path pairing; x/y/color/size), layout planning (ratio cells
emit native-size content plus a per-panel `LayoutScale` that the WASM loader
bakes at load), and scene assembly (flat pre-order panel namespace, D4c).
Sized holes reserve blank space for empty-data children on linear layouts;
grid/wrap holes are cell-positional. The flat single-chart caption band uses
`wrap_svg_with_chrome` (the extracted single-cell chrome wrap); the N-ary
string compositor (`compose_svg_*`, `compositor.rs`, `grid_compose.rs`) and
the Python scene-merge/scale-share modules no longer exist. The one
interactive exception: LayerChart renders the merged single-panel Chart
because the selection/hit-testing contract requires overlays to be one panel.

Decision made 2026-07-02 (Phase B spec); landed 2026-07-05.

---

## Transform pipeline

Chart-level `spec.transforms` are executed by the Rust renderer before any layer renders; all layers share the final transform output via `FINAL_OUTPUT_KEY`. Named transforms publish their output under a key that individual layers can reference via `Layer.data_source`. Unnamed transforms chain sequentially; named transforms run on the current chained batch without advancing the chain pointer.

`_merge_top_transforms` deduplicates by both identity and value equality (`PyO3 __eq__`) to prevent the same logical transform from running twice when two sides of `+` carry identical transform objects.

**Layer-level transforms (`Layer.transforms`) are stored but NOT executed** by the current renderer. All executable transforms must be at chart level. When wrapping a single-mark chart for `+` composition, `_expand_layers` routes the chart's transforms to the top level (not into the `_Layer`), matching the pre-layered path.

---

## Composite marks

**Composite marks desugar Python-side; no Rust `Composite` Mark variant.**

Composite marks (`mark_boxplot`, `mark_errorbar`, `mark_errorband`, `mark_ribbon`, `mark_violin`, `mark_boxen`, …) build multi-layer `ChartSpec`s in Python via `chart.layer(...)` over primitive marks (rect, rule, point, line, ribbon). Rust has no awareness that a layered spec came from a composite — it just renders layers. Multi-output transforms (`BoxStats`, `Outliers`, `Violin`, `Hex`, …) feed individual layers via `Layer.data_source` matched against `TransformSpec.name`; when both are `None`, behavior is byte-identical to single-layer 8a.

Decision made 2026-05-10: alternative was a Rust `MarkSpec::Composite { layers: Vec<Mark> }` variant — rejected because it duplicates the multi-layer machinery already in `ChartSpec.layers`, and would force every composite-mark expansion to cross the PyO3 boundary as opaque payloads. Same pattern applies to Phase 9's `mark_boxen` and any future composite.

---

## Themes

**Themes are values; one documented contextvar exception.**

`Theme` is an immutable Python value class (phase 8a+). `Chart.theme(t)` per-chart override always wins. `ferrum.set_default_theme(t)` returns a context manager backed by a per-thread `contextvars.ContextVar` for notebook ergonomics — the only sanctioned process-scoped theme state.

Decision made 2026-05-10.

---

## Linear algebra

**Pure-Rust linear algebra via `faer`** (0.24+): Cholesky, SVD, eigendecomposition with zero external dependencies — no LAPACK, no OpenBLAS, no platform-specific linking. Used by `transform/linalg.rs` (hat-matrix diagonal, correlation matrices) and `transform/stats.rs` (PCA, classical MDS). Chosen over `ndarray-linalg` to avoid the LAPACK build-system cost; polars uses the same approach.

Decision made 2026-05-13.

---

## Diagnostic statistics

**Diagnostic statistics run in Rust, not Python.** `src/ferrum/_diagnostics/stats.py` was eliminated; all statistical functions live in `transform/stats.rs` and accept/return Arrow via `pyo3-arrow`.

The only numpy usage remaining in the diagnostics subsystem is at sklearn API boundaries (`model.predict()`, `model.predict_proba()` return numpy) and for genuinely 2D operations (CV split indexing, decision-boundary mesh grids).

t-SNE and UMAP run in Rust via `manifolds-rs` (0.2.4+, pure Rust, MIT). `umap-learn` is no longer a runtime dependency. `manifolds-rs` depends on faer 0.23; a `faer-compat` renamed dependency bridges the version gap (remove when manifolds-rs supports faer ≥0.24).

Decision made 2026-05-13.

---

## Randomness

**Byte-deterministic randomness via seeded `rand_chacha`.**

Every transform that uses randomness seeds `ChaCha8Rng` from a `u64` (transform's `seed` field or `spec.seed`, default `0`). Never `rand::thread_rng()`, never `SystemRandom`, never platform RNG. This makes SVG goldens byte-identical across macOS / Linux / CI and across Rust toolchain versions.

The same rule applies to any future transform or mark that introduces randomness — pick a seed field, document the default, plumb it through `ChaCha8Rng`.

Decision made 2026-05-10.
