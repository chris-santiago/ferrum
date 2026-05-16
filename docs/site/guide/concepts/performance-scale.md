# Performance & scale

A plotting library cannot honestly claim a coherent user model if that model collapses under real data volume. Performance is not only an engineering concern in Ferrum; it shapes what the public API can promise.

The commitment is **continuity**: you should not have to change libraries, rewrite your plots, or adopt a second API just because your dataset stopped being toy-sized. The same chart spec should work at 100 rows and at 10,000,000 rows.

This page explains the architecture behind that commitment and the choices it forces on the public surface.

## The architecture in one paragraph

Python is the declaration layer. Rust is the computation layer. Data crosses the boundary once, through the Arrow C Data Interface, with no row-level copying. The Rust engine runs scale resolution, statistical transforms, and layout against columnar Arrow batches, then produces a renderer-agnostic intermediate form (the SceneGraph). Different backends — SVG, CPU raster, and GPU/WASM — consume that SceneGraph without changing the chart's conceptual identity.

That single pipeline is what lets the same [`Chart`][ferrum.Chart] spec produce a vector SVG for a publication figure and a rasterized GPU-backed plot for a 10-million-row scatter, both inside one library.

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

Ferrum's response is to make the choice between vector and raster part of the chart system rather than the user's problem. You can declare a raster mark explicitly ([`mark_raster`][ferrum.Chart.mark_raster], [`mark_hex`][ferrum.Chart.mark_hex], [`mark_contour`][ferrum.Chart.mark_contour]), or rely on auto-raster policies that detect when a vector backend would degrade and switch to a rasterized representation transparently.

The semantics of the chart stay identical. A scatter at 1,000 rows and a scatter at 10,000,000 rows are the same Ferrum spec. The only thing that changes is how the engine draws the marks underneath.

!!! tip "Auto-raster in practice"
    A 1M-point scatter that would produce a **57 MB** SVG with one `<circle>` per mark becomes a **606 KB** SVG when auto-raster kicks in — same chart, same spec, two orders of magnitude smaller output.

### Scatter benchmark: Ferrum vs. Altair vs. seaborn

Median of 3 runs on Apple M-series, macOS 24.6.0, Python 3.10. All libraries render the same bivariate-normal data with equivalent chart specifications. Ferrum runs with auto-raster on at both scales.

#### 200,000 points

| Metric | Ferrum | Altair | seaborn |
|---|---|---|---|
| SVG render time | **297 ms** | 2.54 s | 1.75 s |
| SVG file size | **590 KB** | 57.8 MB | 32.6 MB |
| PNG render time | 1.63 s | — | **116 ms** |
| PNG file size | 383 KB | — | **141 KB** |
| Interactive HTML render + save | 606 ms | **462 ms** | — |
| Interactive HTML file size | **4.9 MB** | 14.3 MB | — |

Ferrum SVG is **8.5x faster** than Altair and **5.9x faster** than seaborn, and **98x smaller** than Altair / **55x smaller** than seaborn. Auto-raster collapses 200k circles into one embedded PNG; the other libraries emit individual SVG elements.

Seaborn's PNG path is **14x faster** (116 ms vs. 1.63 s) — matplotlib's Agg backend is a C extension that draws directly to a pixel buffer with no intermediate representation. Ferrum's PNG path pays for the SVG → resvg hop.

Altair's interactive HTML is slightly faster to save (462 ms vs. 606 ms) because it serializes the Vega-Lite JSON spec and defers rendering to the browser. Ferrum pre-renders the scene graph and embeds WASM, producing **2.9x smaller** output (4.9 MB vs. 14.3 MB).

#### 1,000,000 points

| Metric | Ferrum | Altair | seaborn |
|---|---|---|---|
| SVG render time | **755 ms** | OOM crash | 8.55 s |
| SVG file size | **607 KB** | OOM crash | 162.9 MB |
| PNG render time | 2.18 s | — | **466 ms** |
| PNG file size | 386 KB | — | **163 KB** |
| Interactive HTML render + save | **1.53 s** | OOM crash | — |
| Interactive HTML file size | **5.0 MB** | OOM crash | — |

**Altair cannot participate at 1M points.** vl-convert's embedded V8 hits the heap limit (exit 133 / SIGKILL) trying to serialize 1M rows through the Vega-Lite runtime.

Ferrum SVG is **11x faster** than seaborn (755 ms vs. 8.55 s) and **269x smaller** (607 KB vs. 162.9 MB). Seaborn still emits individual SVG path elements at 1M.

Seaborn wins again on raw rasterization speed (466 ms vs. 2.18 s, **4.7x faster**) and PNG file size (163 KB vs. 386 KB).

Ferrum is the **only library that can produce interactive HTML at this scale**. The 5.0 MB output is size-stable regardless of point count (4.9 MB at 200k vs. 5.0 MB at 1M).

#### Key takeaways

1. **Ferrum dominates SVG** — fastest render and smallest file at both scales by large margins (6–11x faster, 55–269x smaller). Auto-raster collapses N individual elements into one embedded raster image.
2. **Seaborn dominates PNG** — matplotlib's Agg rasterizer is purpose-built for direct pixel output and beats ferrum's SVG → resvg pipeline 5–14x.
3. **Altair hits a hard ceiling** — the V8/Vega-Lite architecture OOMs at 1M points. At 200k it works but produces the largest files.
4. **Interactive HTML is Ferrum-only at scale** — neither Altair (OOM) nor seaborn (no interactive output) can produce interactive charts at 1M points. Ferrum's WASM output is size-stable across point counts.
5. **Auto-raster changes the game for SVG** — without it, ferrum's 200k SVG was 20.9 MB / 1.20 s. With it: 590 KB / 297 ms. The default threshold (500k marks) means users get this automatically at high counts.

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
