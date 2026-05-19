# Scatter Benchmark: ferrum vs Altair vs seaborn vs Plotly vs plotnine

Benchmark of a scatter plot with tooltips at two scales (200k and 1M points).
Each measurement is the median of 3 runs. All libraries render the same data
(bivariate normal, seed 42) with equivalent chart specifications.

Ferrum runs with auto-raster forced on (`raster_threshold=1`) at both scales
so the comparison is apples-to-apples: ferrum rasterizes the scatter into an
embedded PNG within the SVG, rather than emitting individual SVG elements.

- **ferrum** — Rust-backed SVG/PNG renderer + WASM interactive HTML (auto-raster on)
- **Altair** — Vega-Lite spec generation + vl-convert (embedded V8) for SVG/HTML
- **seaborn** — matplotlib Agg backend for PNG, matplotlib SVG backend for SVG
- **Plotly** — ScatterGL (WebGL) + kaleido for static export, plotly.js for HTML
- **plotnine** — ggplot2-style grammar layer over matplotlib (same Agg/SVG backends)

Script: `scripts/profile_scatter.py`
Machine: Apple M-series, macOS 24.6.0, Python 3.10

---

## 200,000 points

| Metric | Ferrum | Altair | seaborn | Plotly | plotnine |
|---|---|---|---|---|---|
| SVG render time | **27 ms** | 2.86 s | 1.95 s | 2.51 s | 7.56 s |
| SVG file size | 590.3 KB | 57.8 MB | 32.6 MB | **267.4 KB** | 137.0 MB |
| PNG render time | **78 ms** | — | 119 ms | 2.50 s | 2.35 s |
| PNG file size | 382.9 KB | — | **140.6 KB** | 59.4 KB | 98.5 KB |
| HTML render+save | 67 ms | 482 ms | — | **43 ms** | — |
| HTML file size | **4.9 MB** | 14.3 MB | — | 9.8 MB | — |

### Analysis — 200k

- **SVG:** Ferrum is 93x faster than Plotly, 106x faster than Altair, 72x
  faster than seaborn, and **280x faster than plotnine**. plotnine's ggplot2
  grammar layer adds ~4x overhead on top of matplotlib's own SVG backend
  (7.56 s vs seaborn's 1.95 s). plotnine produces the largest SVG (137 MB)
  because every mark is an individual SVG element, plus the grammar layer
  adds additional grouping/clipping elements.

- **PNG:** Ferrum is fastest (78 ms), seaborn close behind (119 ms). plotnine
  (2.35 s) is ~20x slower than seaborn despite using the same matplotlib Agg
  backend — the grammar overhead dominates. Plotly is similarly slow (2.50 s)
  due to kaleido's Chromium startup.

- **HTML:** Plotly is slightly faster to save (43 ms vs 67 ms) — it serializes
  the plotly.js JSON spec without pre-rendering. Ferrum pre-renders the scene
  graph and embeds WASM. Ferrum's output is 2x smaller (4.9 MB vs 9.8 MB).
  plotnine has no interactive output.

---

## 1,000,000 points

| Metric | Ferrum | Altair | seaborn | Plotly | plotnine |
|---|---|---|---|---|---|
| SVG render time | **57 ms** | OOM crash | 8.55 s | 3.56 s | 38.82 s |
| SVG file size | 606.8 KB | OOM crash | 162.9 MB | **252.8 KB** | 685.0 MB |
| PNG render time | **112 ms** | — | 451 ms | 3.69 s | 11.42 s |
| PNG file size | 386.3 KB | — | **163.0 KB** | 56.3 KB | 93.1 KB |
| HTML render+save | **125 ms** | OOM crash | — | 149 ms | — |
| HTML file size | **5.0 MB** | OOM crash | — | 30.6 MB | — |

### Analysis — 1M

- **Altair cannot participate at 1M points.** vl-convert's embedded V8
  hits the heap limit (exit 133 / SIGKILL) trying to serialize 1M rows.

- **SVG:** Ferrum is 62x faster than Plotly, 150x faster than seaborn, and
  **681x faster than plotnine** (57 ms vs 38.82 s). plotnine's 685 MB SVG
  is the largest output in the benchmark — over 4x larger than seaborn's
  already-massive 163 MB. The ggplot2 grammar layer's per-mark SVG grouping
  compounds with matplotlib's element-per-mark approach.

- **PNG:** Ferrum is fastest (112 ms), seaborn 4x slower (451 ms), plotnine
  102x slower (11.42 s), Plotly 33x slower (3.69 s).

- **HTML:** Ferrum and Plotly both survive at 1M. Ferrum is slightly faster
  (125 ms vs 149 ms) and 6x smaller (5.0 MB vs 30.6 MB). plotnine has no
  interactive output.

---

## Key takeaways

1. **Ferrum dominates SVG render speed** — fastest at both scales by 60–681x
   margins. Auto-raster collapses N elements into one embedded raster image.

2. **plotnine is the slowest library tested** — the ggplot2 grammar layer adds
   3–5x overhead on top of matplotlib at every scale and format. Despite being
   the closest grammar-of-graphics peer to ferrum, it inherits matplotlib's
   worst scaling characteristics and amplifies them.

3. **Plotly produces the smallest static files** — ScatterGL's WebGL canvas
   approach yields tiny SVGs (253–267 KB) and PNGs (56–59 KB), but at the cost
   of 2.5–3.7 s kaleido overhead per export.

4. **Seaborn is the fastest matplotlib-based option** — raw matplotlib Agg
   (119 ms at 200k, 451 ms at 1M) beats plotnine by 20–25x on PNG.

5. **Altair hits a hard ceiling** — the V8/Vega-Lite architecture OOMs at
   1M points. At 200k it works but produces the largest files after plotnine.

6. **Interactive HTML: ferrum wins at scale** — both ferrum and Plotly produce
   interactive HTML at 1M, but ferrum's binary-buffer approach keeps output at
   5.0 MB vs Plotly's 30.6 MB (6x smaller). Neither Altair, seaborn, nor
   plotnine can produce interactive output at this scale.

7. **Auto-raster changes the game for SVG** — without it, ferrum's 200k SVG
   was 20.9 MB / 1.20 s (see prior run). With it: 590 KB / 27 ms. The default
   threshold (500k) means users get this automatically at high counts; forcing
   it lower gives the benefit at any scale.
