# Saving & export

Ferrum charts render to SVG, PNG, HTML, and JSON — no system dependencies, no display server, no matplotlib. Every chart object (base charts, compound views, helper output, diagnostic charts) supports the same export surface.

## Output methods

The export surface splits into three roles. The `to_*` converters **return an in-memory value**, [`.save()`][ferrum.Chart.save] **writes to disk**, and [`.show()`][ferrum.Chart.show] **displays** the chart.

| Method | Returns | Use case |
|---|---|---|
| [`.to_svg()`][ferrum.Chart.to_svg] | `str` | Get SVG markup as a string |
| [`.to_png()`][ferrum.Chart.to_png] | `bytes` | Get PNG as raw bytes |
| [`.to_html()`][ferrum.Chart.to_html] | `str` | Get the interactive HTML page as a string |
| [`.save(path)`][ferrum.Chart.save] | `None` | Write SVG, PNG, HTML, JSON, or PDF to a file (format inferred from extension) |
| [`.show()`][ferrum.Chart.show] | `None` | Display inline in Jupyter or open in browser |

All of these methods are available on every chart object — base `Chart`, compound views (`HConcatChart`, `VConcatChart`, `JointChart`, `RepeatChart`), and diagnostic helper output.

!!! note "Renamed from `show_svg` / `show_png`"
    `.show_svg()` and `.show_png()` still work as deprecated aliases of [`.to_svg()`][ferrum.Chart.to_svg] and [`.to_png()`][ferrum.Chart.to_png]. They emit a `DeprecationWarning` and will be removed in a future release. The `to_*` names make the convention explicit: `to_*` returns a value, `save` writes to disk, `show` displays.

## Output formats

| Format | Method | File extension | Notes |
|---|---|---|---|
| SVG | [`.to_svg()`][ferrum.Chart.to_svg] | `.svg` | Vector graphics. Default render path. |
| PNG | [`.to_png()`][ferrum.Chart.to_png] | `.png` | Rasterized via `resvg` in Rust. No Cairo/Pillow needed. |
| HTML | [`.to_html()`][ferrum.Chart.to_html] | `.html` | Self-contained interactive page with inlined WASM renderer. |
| JSON | [`.save("out.json")`][ferrum.Chart.save] | `.json` | Chart spec as JSON — the same format as `.to_json()`. |
| PDF | [`.save("out.pdf")`][ferrum.Chart.save] | `.pdf` | Rasterized PNG (Rust `resvg`) wrapped in a PDF page. No Ghostscript or Cairo needed. |

## Saving to disk

[`.save()`][ferrum.Chart.save] infers the format from the file extension:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 3.0]})
chart = fm.Chart(df).mark_point().encode(x="x", y="y")

chart.save("scatter.svg")   # vector
chart.save("scatter.png")   # raster
chart.save("scatter.html")  # interactive (WASM inlined)
chart.save("scatter.json")  # spec
```

Pass `format=` explicitly to override the extension:

```python
chart.save("output", format="svg")
```

## Controlling auto-raster

At high mark counts (default threshold: 500,000), Ferrum transparently substitutes a raster image for per-element SVG marks. Override this per-call with `raster=`:

```python
chart.to_svg(raster=False)   # force vector even at high counts
chart.save("out.svg", raster=False)
chart.to_png(raster=True)    # force raster even at low counts
```

For persistent control, attach a [`RenderConfig`][ferrum.RenderConfig] to the chart:

<!--pytest.mark.skip-->
```python
from ferrum import RenderConfig

config = RenderConfig(raster_threshold=1_000_000, raster_behavior="silent")
chart = chart.properties(render_config=config)
chart.save("out.svg")  # auto-raster fires at 1M marks, silently
```

[`RenderConfig`][ferrum.RenderConfig] parameters: `raster_threshold` (mark count or `None` to disable), `raster_behavior` (`"warn"`, `"silent"`, `"error"`), `raster_aggregate` (`"count"`, `"density"`, `"mean"`, `"sum"`, or `"any"` — `"mean"`/`"sum"` also require `raster_field` naming the column to aggregate), and `raster_scheme` (the canonical colormap keyword; `raster_cmap` is a back-compat alias).

## Getting raw bytes

For programmatic use (embedding in notebooks, serving from a web app, writing to S3):

```python
svg_str = chart.to_svg()      # str — complete <svg>…</svg> document
png_bytes = chart.to_png()    # bytes — raw PNG data
html_str = chart.to_html()    # str — self-contained interactive HTML page
```

[`.to_html()`][ferrum.Chart.to_html] returns the same bytes that [`.save("out.html")`][ferrum.Chart.save] would write, as a string. It accepts the same `embed_wasm=`, `toolbar=`, and `raster=` keywords as the HTML save path.

## PDF export

`chart.save("chart.pdf")` embeds a rasterized PNG (produced by the same Rust `resvg` pipeline as `to_png()`) inside a minimal, dependency-free PDF page written by a pure-Python codec — no Ghostscript, Cairo, or other system tool required. The same zero-system-dependency guarantee as PNG rasterization applies. Because the page holds a raster image, `scale=` controls its resolution just as it does for PNG; PDF output is not resolution-independent vector art.

```python
chart.save("chart.pdf")
```

## High-DPI PNG output

Pass `scale=` to `to_png()` or `save()` to produce higher-resolution PNG output. The default scale is `2.0` (2× pixel density, suitable for standard retina displays). Increase it for print-quality output:

```python
chart.to_png(scale=3.0)          # 3× pixel density
chart.save("chart.png", scale=3.0)
```

The scale factor multiplies the chart's pixel dimensions: a 600 × 400 chart at `scale=3.0` produces a 1800 × 1200 PNG.

!!! note "PNG resolution"
    The `scale=` parameter is the recommended way to control PNG resolution. You can also increase `width` and `height` via [`.properties(width=1200, height=800)`][ferrum.Chart.properties], but `scale=` is simpler for DPI-scaling existing charts without changing their logical dimensions.

## Displaying in Jupyter

In a Jupyter notebook, charts render automatically via `_repr_svg_` — just put the chart as the last expression in a cell:

```python
chart  # renders inline as SVG
```

For interactive rendering (selections, zoom/pan), call [`.interactive()`][ferrum.Chart.interactive] instead — see [Interactive rendering](interactive.md).

Outside of a notebook, [`.show()`][ferrum.Chart.show] writes a temporary SVG and opens it in the system browser.

## HTML export

`.save("file.html")` produces a self-contained HTML file with the WASM GPU renderer and scene data inlined. No server, no CDN, no external dependencies — the file works offline in any modern browser.

When you need the page as a string instead of a file (templating, serving from a web app, embedding in another document), [`.to_html()`][ferrum.Chart.to_html] returns the byte-identical HTML in memory.

This is the right format for sharing interactive charts via email, Slack, or static hosting.

### Toolbar in exported HTML

By default, exported HTML files include the interactive toolbar (Pan, Box Zoom, Box Select, Reset, Save PNG). Pass `toolbar=False` to suppress it:

```python
chart.save("out.html", toolbar=False)
```

The default is `toolbar=True`. The `toolbar=` parameter has no effect on SVG, PNG, or JSON output — it is only meaningful for the HTML format.

`toolbar=` also works on composed views:

```python
(chart1 | chart2).save("out.html", toolbar=False)
```

## Compound views

All composition operators produce objects with the same export surface. A four-panel report saves exactly like a single chart:

<!--pytest.mark.skip-->
```python
report = (roc | calibration) & (confusion | residuals)
report.save("model_report.svg")
report.save("model_report.png")
```

## No system dependencies

Ferrum's rendering pipeline is pure Rust. SVG rendering, PNG rasterization (`resvg`), and WASM compilation all happen inside the wheel. There is no dependency on Cairo, X11, Ghostscript, or any display server. `pip install ferrum-viz` is the entire setup — charts render in Kubernetes, CI, SSH sessions, and headless containers.

## Where to go next

- [First plot](../getting-started/first-plot.md) for a quick start with rendering.
- [Themes](themes.md) for controlling the visual style of exported charts.
- [Interactive rendering](interactive.md) for WASM-based interactive output.
- [Composition](composition.md) for building multi-panel views before export.
