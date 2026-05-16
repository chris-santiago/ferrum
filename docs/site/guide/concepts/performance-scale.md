# Performance & scale

A plotting library cannot honestly claim a coherent user model if that model collapses under real data volume. Performance is not only an engineering concern in Ferrum; it shapes what the public API can promise.

The commitment is **continuity**: you should not have to change libraries, rewrite your plots, or adopt a second API just because your dataset stopped being toy-sized. The same chart spec should work at 100 rows and at 10,000,000 rows.

This page explains the architecture behind that commitment and the choices it forces on the public surface.

## The architecture in one paragraph

Python is the declaration layer. Rust is the computation layer. Data crosses the boundary once, through the Arrow C Data Interface, with no row-level copying. The Rust engine runs scale resolution, statistical transforms, and layout against columnar Arrow batches, then produces a renderer-agnostic intermediate form (the SceneGraph). Different backends — SVG, CPU raster, and GPU/WASM — consume that SceneGraph without changing the chart's conceptual identity.

That single pipeline is what lets the same `Chart` spec produce a vector SVG for a publication figure and a rasterized GPU-backed plot for a 10-million-row scatter, both inside one library.

## Python declares, Rust computes

Ferrum treats Python as a specification language. When you build a chart, the Python code is constructing a value — a `ChartSpec` — that describes what you want. The Python side is small, allocation-light, and side-effect free. It does not loop over your data, it does not compute statistics, and it does not generate geometry.

The Rust core does the work. Scale fitting, statistical transforms, mark resolution, layout, and rendering all run in compiled code against columnar data. The Python layer is responsible only for declaration and orchestration.

This split is deliberate. Python is excellent for the expressive grammar that the chart system is built on, but the per-row work — binning a million points, computing a kernel density, fitting a regression line, laying out a faceted compound — is not where Python shines. Putting all of that in Rust is what makes it honest for the same spec to scale.

## Arrow CDI is the boundary

The single point where Python and Rust meet is the Arrow C Data Interface. Ferrum accepts your data — Polars, pandas, modin, cuDF, dask, ibis, Arrow tables, NumPy arrays, or anything Narwhals can interpret — and passes columnar buffers across the boundary by pointer rather than by copy.

For Polars specifically, that handoff is zero-copy: the Rust engine reads the same columnar buffers Polars already owns. For other dataframe sources, Narwhals normalizes the interface and the engine reads through the Arrow representation that results.

This is a structural choice, not a tuning knob. The library is built around the assumption that data already lives in a columnar layout, and the boundary is designed to preserve that layout end-to-end. The same `Chart(data)` constructor accepts every supported dataframe API; the multi-framework story is explored in detail in [Dataframe pluralism](dataframe-pluralism.md).

## Rendering: SVG, raster, and GPU/WASM

Ferrum produces three classes of output, all from the same chart spec:

- **SVG** for static vector output. Useful for publication graphics, exact reproducibility, and small-to-medium mark counts where vector quality matters.
- **CPU raster** for static raster output. Used both as a final format and as the underlying mark technique for `mark_raster` and high-cardinality plots that would overwhelm a vector backend.
- **GPU/WASM** for interactive output. Selections, zoom, pan, and linked views run on a backend that can keep up with millions of marks without forcing you to subsample first.

The chart spec does not change when you switch outputs. The renderer changes. This is the same principle as [statistics in the pipeline](stats-pipeline.md): the structural choice — *where does the work happen?* — is fixed by the library so the user-facing grammar can stay invariant.

## Auto-raster: scale as part of the API

The headline scale problem in visualization is mark count. Every existing library breaks at some mark threshold — Altair around 5,000 rows, seaborn or matplotlib around 100,000 marks, plotly around 500,000. The usual symptoms are slow renders, browser hangs, and eventually crashes.

Ferrum's response is to make the choice between vector and raster part of the chart system rather than the user's problem. You can declare a raster mark explicitly (`mark_raster`, `mark_hex`, `mark_contour`), or rely on auto-raster policies that detect when a vector backend would degrade and switch to a rasterized representation transparently.

The semantics of the chart stay identical. A scatter at 1,000 rows and a scatter at 10,000,000 rows are the same Ferrum spec. The only thing that changes is how the engine draws the marks underneath.

!!! tip "Auto-raster in practice"
    A 1M-point scatter that would produce a **57 MB** SVG with one `<circle>` per mark becomes a **606 KB** SVG when auto-raster kicks in — same chart, same spec, two orders of magnitude smaller output.

### 1M-point scatter: Ferrum vs. Altair vs. seaborn

| Metric | Ferrum | Altair | seaborn |
|---|---|---|---|
| SVG render time | 741 ms | OOM crash | 8.80 s |
| SVG file size | 607 KB | OOM crash | 162.9 MB |
| PNG render time | — | — | 467 ms |
| PNG file size | — | — | 163 KB |
| Interactive HTML | 1.50 s / 5.0 MB | OOM crash | N/A |

Ferrum's SVG is **275x smaller** than seaborn's. Seaborn emits 1M individual `<circle>` path elements (162.9 MB); ferrum's auto-raster substitutes an embedded PNG within the SVG (607 KB). Ferrum renders SVG **12x faster** (741 ms vs. 8.8 s).

Seaborn's PNG path (467 ms, 163 KB) is its strong suit — matplotlib's Agg backend rasterizes efficiently. Ferrum's auto-raster is doing essentially the same thing but wrapped in an SVG container, landing at a comparable size.

Altair can't participate at 1M points — vl-convert's embedded V8 engine hits the heap limit trying to serialize 1M rows through the Vega-Lite runtime (exit code 133 = SIGKILL from the OOM handler). Altair has no auto-raster equivalent.

Interactive HTML output (`.interactive().save()`) stayed at 5.0 MB — the WASM GPU renderer and packed scene data are largely size-invariant.

## SHAP and ICE at full sample size

The plots that matter most for understanding models at scale — SHAP summaries, ICE curves, partial dependence views — are also the plots that existing tools sample or crash on. They are dense by construction: one row per training point, often many marks per row, often interactive.

Ferrum's commitment is that those plots remain part of the same chart language even when scale requires rasterization or GPU-backed interaction. You do not switch to a different visualization library for explainability at scale; you keep using Ferrum and the rendering backend adapts.

## Operational simplicity

Performance is not only about speed in isolation. A library that is fast but requires a fragile system stack — Cairo, X11, a display server, a JavaScript runtime — is harder to deploy where real work happens.

Ferrum favors operational simplicity as part of the same commitment. The rendering stack is pure Rust. There is no matplotlib dependency. There is no display server requirement. Charts render identically in notebooks, scripts, CI pipelines, containers, SSH sessions, and Kubernetes jobs. `pip install ferrum` is the entire setup; the compiled core ships in the wheel.

This is part of why the library can promise that the same plotting code works in development and in production, not only that it runs fast in isolated benchmarks.

## What this does not promise

Performance commitments come with scope limits. Ferrum is built for the common case of statistical plotting, model evaluation, and exploratory analysis on tabular data of all sizes. It is not a streaming visualization system, not a real-time dashboarding framework, and not a graph-rendering library. Animation as a first-class encoding, geographic tile layers, and 3-D coordinate systems are outside the 1.0 scope.

Inside the scope, the bet is that one chart system, with the Python/Rust/Arrow architecture above, can carry you from exploratory analysis through model diagnostics through publication-quality output without changing tools — and that the same code works at every data size you are likely to throw at it.

## Where to go next

- [Stats in the rendering pipeline](stats-pipeline.md) explains why statistical computation lives in the engine alongside layout and rendering.
- [Dataframe pluralism](dataframe-pluralism.md) explains how the Arrow boundary supports pandas, Polars, modin, cuDF, dask, and ibis through one ingestion path.
- [One chart model](one-chart-model.md) covers the grammar that the performance architecture is built to preserve.
- [Why Ferrum](../../getting-started/why-ferrum.md) frames the same architecture as a comparison to existing Python plotting libraries.
