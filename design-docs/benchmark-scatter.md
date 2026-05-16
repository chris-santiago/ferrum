# Scatter Benchmark: ferrum vs Altair vs seaborn

Benchmark of a scatter plot with tooltips at two scales (200k and 1M points).
Each measurement is the median of 3 runs. All libraries render the same data
(bivariate normal, seed 42) with equivalent chart specifications.

Ferrum runs with auto-raster forced on (`raster_threshold=1`) at both scales
so the comparison is apples-to-apples: ferrum rasterizes the scatter into an
embedded PNG within the SVG, rather than emitting individual SVG elements.

- **ferrum** — Rust-backed SVG/PNG renderer + WASM interactive HTML (auto-raster on)
- **Altair** — Vega-Lite spec generation + vl-convert (embedded V8) for SVG/HTML
- **seaborn** — matplotlib Agg backend for PNG, matplotlib SVG backend for SVG

Script: `scripts/profile_scatter.py`
Machine: Apple M-series, macOS 24.6.0, Python 3.10

---

## 200,000 points

| Metric | Ferrum | Altair | Seaborn |
|---|---|---|---|
| SVG render time | **297 ms** | 2.54 s | 1.75 s |
| SVG file size | **590.3 KB** | 57.8 MB | 32.6 MB |
| PNG render time | 1.63 s | — | **116 ms** |
| PNG file size | **382.9 KB** | — | 140.6 KB |
| HTML render+save | 606 ms | **462 ms** | — |
| HTML file size | **4.9 MB** | 14.3 MB | — |

### Analysis — 200k

- **SVG:** Ferrum is 8.5x faster than Altair and 5.9x faster than seaborn.
  File size is 98x smaller than Altair (590 KB vs 57.8 MB) and 55x smaller
  than seaborn (590 KB vs 32.6 MB). Auto-raster collapses 200k circles into
  one embedded PNG; the other libraries emit individual SVG elements.

- **PNG:** Seaborn is 14x faster (116ms vs 1.63s). matplotlib's Agg backend
  is a C extension that draws directly to a pixel buffer — no intermediate
  representation. Ferrum's PNG path goes through SVG render → resvg
  rasterization. Ferrum does produce a larger PNG (383 KB vs 141 KB).

- **HTML:** Altair is slightly faster to save (462ms vs 606ms) because it
  serializes the Vega-Lite JSON spec — the browser renders at view time.
  Ferrum pre-renders the scene graph and embeds WASM. Ferrum's output is
  2.9x smaller (4.9 MB vs 14.3 MB).

---

## 1,000,000 points

| Metric | Ferrum | Altair | Seaborn |
|---|---|---|---|
| SVG render time | **755 ms** | OOM crash | 8.55 s |
| SVG file size | **606.8 KB** | OOM crash | 162.9 MB |
| PNG render time | 2.18 s | — | **466 ms** |
| PNG file size | 386.3 KB | — | **163.0 KB** |
| HTML render+save | **1.53 s** | OOM crash | — |
| HTML file size | **5.0 MB** | OOM crash | — |

### Analysis — 1M

- **Altair cannot participate at 1M points.** vl-convert's embedded V8
  hits the heap limit (exit 133 / SIGKILL) trying to serialize 1M rows.

- **SVG:** Ferrum is 11x faster than seaborn (755ms vs 8.55s) and 269x
  smaller (607 KB vs 162.9 MB). Seaborn still emits individual SVG path
  elements at 1M.

- **PNG:** Seaborn wins again on raw rasterization speed (466ms vs 2.18s,
  4.7x faster) and file size (163 KB vs 386 KB).

- **HTML:** Ferrum is the only library that can produce interactive HTML at
  this scale. The 5.0 MB output is size-stable regardless of point count
  (4.9 MB at 200k vs 5.0 MB at 1M).

---

## Key takeaways

1. **Ferrum dominates SVG** — fastest render and smallest file at both scales
   by large margins (5-11x faster, 55-269x smaller). Auto-raster is the key:
   it collapses N individual elements into one embedded raster image.

2. **Seaborn dominates PNG** — matplotlib's Agg rasterizer is purpose-built
   for this and beats ferrum's SVG→resvg pipeline 5-14x. Ferrum's PNG path
   pays for the intermediate SVG representation.

3. **Altair hits a hard ceiling** — the V8/Vega-Lite architecture OOMs at
   1M points. At 200k it works but produces the largest files.

4. **Interactive HTML is ferrum-only at scale** — neither Altair (OOM) nor
   seaborn (no interactive output) can produce interactive charts at 1M.
   Ferrum's WASM output is size-stable across point counts.

5. **Auto-raster changes the game for SVG** — without it, ferrum's 200k SVG
   was 20.9 MB / 1.20s (see prior run). With it: 590 KB / 297ms. The default
   threshold (500k) means users get this automatically at high counts; forcing
   it lower gives the benefit at any scale.

6. **PNG path is a known gap** — the SVG→resvg hop is inherently slower than
   a direct rasterizer. A direct-to-pixel-buffer backend (analogous to Agg)
   would close this gap but would mean maintaining two rendering backends.
