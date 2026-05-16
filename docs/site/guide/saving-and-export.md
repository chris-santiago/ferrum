# Saving & export

Ferrum charts render to SVG, PNG, HTML, and JSON — no system dependencies, no display server, no matplotlib. Every chart object (base charts, compound views, helper output, diagnostic charts) supports the same export surface.

## Output methods

| Method | Returns | Use case |
|---|---|---|
| [`.save(path)`][ferrum.Chart.save] | `None` | Write SVG, PNG, or HTML to a file (format inferred from extension) |
| [`.show_svg()`][ferrum.Chart.show_svg] | `str` | Get SVG markup as a string |
| [`.show_png()`][ferrum.Chart.show_png] | `bytes` | Get PNG as raw bytes |
| [`.show()`][ferrum.Chart.show] | `None` | Display inline in Jupyter or open in browser |

All four methods are available on every chart object — base `Chart`, compound views (`HConcatChart`, `VConcatChart`, `JointChart`, `RepeatChart`), and diagnostic helper output.

## Output formats

| Format | Method | File extension | Notes |
|---|---|---|---|
| SVG | [`.show_svg()`][ferrum.Chart.show_svg] | `.svg` | Vector graphics. Default render path. |
| PNG | [`.show_png()`][ferrum.Chart.show_png] | `.png` | Rasterized via `resvg` in Rust. No Cairo/Pillow needed. |
| HTML | [`.save("out.html")`][ferrum.Chart.save] | `.html` | Self-contained interactive page with inlined WASM renderer. |
| JSON | [`.save("out.json")`][ferrum.Chart.save] | `.json` | Chart spec as JSON — the same format as `.to_json()`. |
| PDF | — | `.pdf` | Not natively supported. Convert from SVG using CairoSVG (`cairosvg chart.svg -o chart.pdf`), Inkscape, or browser print-to-PDF. |

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
chart.show_svg(raster=False)   # force vector even at high counts
chart.save("out.svg", raster=False)
chart.show_png(raster=True)    # force raster even at low counts
```

For persistent control, attach a [`RenderConfig`][ferrum.RenderConfig] to the chart:

<!--pytest.mark.skip-->
```python
from ferrum import RenderConfig

config = RenderConfig(raster_threshold=1_000_000, raster_behavior="silent")
chart = chart.properties(render_config=config)
chart.save("out.svg")  # auto-raster fires at 1M marks, silently
```

[`RenderConfig`][ferrum.RenderConfig] parameters: `raster_threshold` (mark count or `None` to disable), `raster_behavior` (`"warn"`, `"silent"`, `"error"`), `raster_aggregate`, and `raster_cmap`.

## Getting raw bytes

For programmatic use (embedding in notebooks, serving from a web app, writing to S3):

```python
svg_str = chart.show_svg()   # str — complete <svg>…</svg> document
png_bytes = chart.show_png()  # bytes — raw PNG data
```

!!! note "PNG resolution"
    PNG resolution is not separately configurable (there is no DPI parameter). To produce higher-resolution output, increase `width` and `height` via [`.properties(width=1200, height=800)`][ferrum.Chart.properties] before calling `.show_png()` or `.save("out.png")`.

## Displaying in Jupyter

In a Jupyter notebook, charts render automatically via `_repr_svg_` — just put the chart as the last expression in a cell:

```python
chart  # renders inline as SVG
```

For interactive rendering (selections, zoom/pan), call [`.interactive()`][ferrum.Chart.interactive] instead — see [Interactive rendering](interactive.md).

Outside of a notebook, [`.show()`][ferrum.Chart.show] writes a temporary SVG and opens it in the system browser.

## HTML export

`.save("file.html")` produces a self-contained HTML file with the WASM GPU renderer and scene data inlined. No server, no CDN, no external dependencies — the file works offline in any modern browser.

This is the right format for sharing interactive charts via email, Slack, or static hosting.

## Compound views

All composition operators produce objects with the same export surface. A four-panel report saves exactly like a single chart:

<!--pytest.mark.skip-->
```python
report = (roc | calibration) & (confusion | residuals)
report.save("model_report.svg")
report.save("model_report.png")
```

## No system dependencies

Ferrum's rendering pipeline is pure Rust. SVG rendering, PNG rasterization (`resvg`), and WASM compilation all happen inside the wheel. There is no dependency on Cairo, X11, Ghostscript, or any display server. `pip install ferrum` is the entire setup — charts render in Kubernetes, CI, SSH sessions, and headless containers.

## Where to go next

- [First plot](../getting-started/first-plot.md) for a quick start with rendering.
- [Themes](themes.md) for controlling the visual style of exported charts.
- [Interactive rendering](interactive.md) for WASM-based interactive output.
- [Composition](composition.md) for building multi-panel views before export.
