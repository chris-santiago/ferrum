# Scatter Benchmark: ferrum vs Altair vs seaborn vs Plotly

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

Script: `scripts/profile_scatter.py`
Machine: Apple M-series, macOS 24.6.0, Python 3.10

---

## 200,000 points

| Metric | Ferrum | Altair | Seaborn | Plotly |
|---|---|---|---|---|
| SVG render time | **27 ms** | 2.86 s | 1.95 s | 2.51 s |
| SVG file size | 590.3 KB | 57.8 MB | 32.6 MB | **267.4 KB** |
| PNG render time | **78 ms** | — | 119 ms | 2.50 s |
| PNG file size | 382.9 KB | — | **140.6 KB** | 59.4 KB |
| HTML render+save | 67 ms | 482 ms | — | **43 ms** |
| HTML file size | **4.9 MB** | 14.3 MB | — | 9.8 MB |

### Analysis — 200k

- **SVG:** Ferrum is 93x faster than Plotly, 106x faster than Altair, and
  72x faster than seaborn. Plotly's SVG is the smallest (267 KB) because
  ScatterGL emits a single canvas-like element; ferrum's auto-raster produces
  a comparable 590 KB. Altair and seaborn emit individual SVG elements (57–32 MB).

- **PNG:** Ferrum is fastest (78ms), seaborn close behind (119ms). Plotly is
  very slow (2.5s) because kaleido spins up headless Chromium for each export.
  Plotly produces the smallest PNG (59 KB) due to WebGL's native rasterization.

- **HTML:** Plotly is slightly faster to save (43ms vs 67ms) — it serializes
  the plotly.js JSON spec without pre-rendering. Ferrum pre-renders the scene
  graph and embeds WASM. Ferrum's output is 2x smaller (4.9 MB vs 9.8 MB).

---

## 1,000,000 points

| Metric | Ferrum | Altair | Seaborn | Plotly |
|---|---|---|---|---|
| SVG render time | **57 ms** | OOM crash | 8.55 s | 3.56 s |
| SVG file size | 606.8 KB | OOM crash | 162.9 MB | **252.8 KB** |
| PNG render time | **112 ms** | — | 451 ms | 3.69 s |
| PNG file size | 386.3 KB | — | **163.0 KB** | 56.3 KB |
| HTML render+save | **125 ms** | OOM crash | — | 149 ms |
| HTML file size | **5.0 MB** | OOM crash | — | 30.6 MB |

### Analysis — 1M

- **Altair cannot participate at 1M points.** vl-convert's embedded V8
  hits the heap limit (exit 133 / SIGKILL) trying to serialize 1M rows.

- **SVG:** Ferrum is 62x faster than Plotly (57ms vs 3.56s) and 150x faster
  than seaborn (57ms vs 8.55s). Plotly's SVG is smallest (253 KB) but takes
  3.5s to produce via kaleido. Ferrum's auto-raster gives 607 KB in 57ms.

- **PNG:** Ferrum is fastest (112ms), seaborn 4x slower (451ms), Plotly 33x
  slower (3.69s). Kaleido's Chromium overhead dominates at every scale.

- **HTML:** Ferrum and Plotly both survive at 1M. Ferrum is slightly faster
  (125ms vs 149ms) and 6x smaller (5.0 MB vs 30.6 MB). Plotly's HTML balloons
  because it embeds all 1M data points as JSON; ferrum uses a binary buffer.

---

## Key takeaways

1. **Ferrum dominates SVG render speed** — fastest at both scales by 60-150x
   margins. Auto-raster collapses N elements into one embedded raster image.

2. **Plotly produces the smallest static files** — ScatterGL's WebGL canvas
   approach yields tiny SVGs (253-267 KB) and PNGs (56-59 KB), but at the cost
   of 2.5-3.7s kaleido overhead per export.

3. **Seaborn dominates PNG speed at scale** — matplotlib's Agg rasterizer
   (119ms at 200k, 451ms at 1M) beats ferrum's SVG→resvg pipeline only when
   ferrum isn't using auto-raster. With auto-raster, ferrum is competitive
   (78ms at 200k, 112ms at 1M).

4. **Altair hits a hard ceiling** — the V8/Vega-Lite architecture OOMs at
   1M points. At 200k it works but produces the largest files.

5. **Interactive HTML: ferrum wins at scale** — both ferrum and Plotly produce
   interactive HTML at 1M, but ferrum's binary-buffer approach keeps output at
   5.0 MB vs Plotly's 30.6 MB (6x smaller). Altair OOMs; seaborn has no
   interactive output.

6. **Kaleido is Plotly's bottleneck** — spinning up headless Chromium makes
   every static export (SVG, PNG) take 2.5-3.7s regardless of data size. For
   interactive HTML (no kaleido), Plotly is competitive with ferrum on speed.

7. **Auto-raster changes the game for SVG** — without it, ferrum's 200k SVG
   was 20.9 MB / 1.20s (see prior run). With it: 590 KB / 27ms. The default
   threshold (500k) means users get this automatically at high counts; forcing
   it lower gives the benefit at any scale.
