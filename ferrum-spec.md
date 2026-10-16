# Ferrum
## A Statistical Visualization Library — Concept & API Specification

> *Fast. Composable. Statistically honest.*

---

## Part I: Philosophy

### 1. Core Beliefs

#### Grammar first, convenience second
Every visualization, from a scatter plot to a SHAP beeswarm, is a composition of the same primitives: data, encodings, marks, scales, and coordinate systems. Ferrum commits to this without compromise. High-level convenience functions (`displot`, `lmplot`, `roc_chart`) exist, but they are sugar over the grammar, not parallel universes with different rules.

#### Model artifacts are data
A confusion matrix is a cross-tabulation. A ROC curve is a sorted transformation of predicted probabilities. A SHAP value is a column. There is no reason these should require a separate API or a different mental model. Ferrum treats model outputs as data sources that feed into the standard grammar. This eliminates the Yellowbrick problem: diagnostic plots are composable, themeable, and interactive because they are charts, not special objects.

#### Statistics belong in the rendering pipeline, not in userspace
Computing a KDE, bootstrapping a confidence interval, or fitting a LOESS curve should not require the user to call SciPy before plotting. These are stat transforms: first-class operations declared in the chart spec and executed in the Rust engine before rendering. The user declares intent; Ferrum computes.

Model diagnostics follow the same rule from the user's side: the figure function (`ferrum.roc_chart`, ...) is the pipeline, and the user never computes a ROC curve in userspace. Inside it, the `ModelSource` adapter (Python) owns everything that needs the estimator protocol (`predict`, `predict_proba`, `classes_`, ...), delegating to scikit-learn lazily where a diagnostic needs it, because a generic Rust transform cannot drive an arbitrary estimator. The numeric work over the resulting arrays runs on Rust kernels in `ferrum._core`: ROC and precision-recall curves, AUC and average precision, calibration bins, the confusion matrix, the threshold sweep, Kendall's tau-b, the feature rankings, PCA variance, classical MDS and the hat-matrix statistics.

#### Interactivity is a renderer, not a rewrite
You should not need to learn a different API to make a chart interactive. `.interactive()` switches the render target from SVG to a WASM canvas. Selections, zoom, pan, and linked views are declared in the chart spec and handled by the renderer. Plotly's fatal flaw is that interactive charts and static charts are different objects. Ferrum has one chart object.

#### Zero unnecessary copies
Python is the declaration layer. Rust is the computation layer. Data moves between them once, over the Arrow C Data Interface (CDI), mediated in PyO3 by the `pyo3-arrow` crate. A polars `DataFrame` exports `__arrow_c_stream__` natively, so CDI hands Rust the buffer pointer with no copy; other inputs are first normalized to a `pyarrow.Table` (§3.1). Stat transforms, layout, binning, and aggregation happen in Rust, on the batch that crossed.

#### Defaults should be correct, not just pretty
Defaults are chosen as correct starting points, not aesthetic opinions. The default theme uses the `paper_ink` categorical palette, the `cool_blue` sequential ramp and the `blue_to_red` diverging ramp; the colorblind-safe `okabe_ito` palette and perceptually uniform ramps such as `viridis` are one theme key away (§3.13). Default bin counts follow Sturges' rule as a floor.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#1-core-beliefs).

---

### 2. Design Constraints

- **No matplotlib dependency.** Ever. Not as a fallback, not for "legacy support."
- **No mutable global state.** No `rcParams`, no module-level theme object to rebind. Themes are values passed to charts, and a per-chart `.theme()` always wins. The one scoped default is `ferrum.set_default_theme()`, which sets a per-thread `contextvars.ContextVar` and reverts automatically when used as a context manager.
- **No magic inference that silently fails.** If Ferrum infers a scale or encoding type incorrectly, it raises a descriptive error with a suggested fix.
- **Sklearn-compatible, not sklearn-dependent.** The model diagnostics layer works with any object that implements `predict`, `predict_proba`, or `transform`. Sklearn is not imported unless the user passes a fitted sklearn model or calls a diagnostic that needs it (cross-validation, partial dependence, permutation importance), or uses an sklearn-backed extra such as SHAP. The raw-array path (`y_true=` / `y_pred=`) computes ROC, precision-recall, calibration, confusion-matrix, and threshold-sweep metrics on native Rust kernels with no sklearn dependency.
- **One output format per render call.** A chart produces SVG, PNG, PDF, HTML (with the WASM bundle), or its own JSON chart spec. Producing several from the same spec is supported; producing ambiguous mixed output is not.

---

### 3. Relationship to Prior Art

| Library | What Ferrum Inherits | What Ferrum Rejects |
|---|---|---|
| **plotnine / ggplot2** | Grammar of Graphics layering, explicit scales, faceting primitives | matplotlib backend, R-centric defaults |
| **Altair** | Typed encoding channels, selection API, `\|` / `&` composition operators | Vega-Lite JSON as the primary data format (too slow at scale), 5000-row default limit |
| **Seaborn** | Statistical vocabulary, figure-level functions, automatic facet label handling | matplotlib coupling, inconsistent axes-level vs figure-level API surface |
| **Plotly** | Interactivity as a first-class feature, hover tooltips, WASM/HTML output | Separate static and interactive APIs, heavy JavaScript bundle for simple charts |
| **Yellowbrick** | sklearn-protocol visualizers, diagnostic vocabulary | matplotlib coupling, non-composable outputs, parallel API surface |
| **scikit-plot** | Lift/gain curves, calibration curve conventions | Function-level API with no composability |

---

### 4. Naming

The library is named **Ferrum** (Latin: iron). Iron is the substrate of Rust. It is also load-bearing, structural, and unglamorous, which is what a visualization engine should be.

The Python package imports as `ferrum` (PyPI name `ferrum-viz`). Rust crates: `ferrum-core` (the engine, compiled to `ferrum._core`), `ferrum-scene` (the shared SceneGraph types) and `ferrum-wasm` (the interactive renderer).

---

---

## Part II: Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Python Layer                        │
│  Chart spec builder · Encoding DSL · Convenience API    │
│  ModelSource adapters · composition lowering            │
└────────────────────┬────────────────────────────────────┘
                     │  PyO3 + Arrow CDI (pyo3-arrow)
┌────────────────────▼────────────────────────────────────┐
│                 Rust Core (ferrum-core)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Transforms  │  │    Layout    │  │    Scales    │   │
│  │  KDE · CI    │  │ Axes · legend│  │ Domain/Range │   │
│  │  Regression  │  │ Facet grid · │  │ mapping ·    │   │
│  │  Binning     │  │ composites   │  │ Tick gen     │   │
│  │  Aggregation │  │              │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
  ┌──────────────────▼───────────────────────────────┐
  │               Render Pipeline                    │
  │                                                  │
  │   Geometry pass → SceneGraph (ferrum-scene)      │
  │                       │                          │
  │          ┌────────────┴────────────┐             │
  │          ▼                         ▼             │
  │      SVG writer               ferrum-wasm        │
  │      (vector)                 (wgpu on WebGL)    │
  │          │                                       │
  │          ▼                                       │
  │      resvg / tiny-skia raster → PNG (→ PDF)      │
  └──────────────────────────────────────────────────┘
                     │
       ┌─────────────┼──────────────┐
       ▼             ▼              ▼
  SVG/PNG/PDF    HTML+WASM       JSON spec
   (static)    (interactive)  (ChartSpec, interop)
```

SceneGraph is the renderer-agnostic intermediate representation, defined in the `ferrum-scene` crate: a canvas size and background, a title, a list of panels (each with its plot area, clip, coordinate kind, and its grid, axis, mark-batch and annotation layers), the legend, decorations, and the selection and interaction configuration. Its nodes are geometric primitives (rects, circles, lines, paths, polygons, polylines, text, images, groups, and raw SVG). The SVG writer and the WASM renderer both consume it. PNG output is the SVG rasterized by `resvg`/`tiny-skia` in the Rust core, and PDF output wraps that PNG in a single-page PDF. The JSON output is the chart's own serialized `ChartSpec` (`Chart.to_json()`), not a Vega-Lite spec.

---

---

## Part III: API Specification

---

### 3.1 Top-Level Entry Points

#### `ferrum.Chart`
The primary chart constructor. All single-view charts begin here.

```
Chart(data=None, *, width=None, height=None, title=None, description=None)
```

**Parameters.**

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `data` | data frame or `None` | `None` | The chart's data, or `None` when every layer brings its own. Accepted: `polars.DataFrame`, `pyarrow.Table`, `pyarrow.RecordBatch`, `pandas.DataFrame`, modin, cuDF, dask and ibis frames (through narwhals), `dict[str, list]`, `list[dict]`, and a 2-D `numpy.ndarray` (columns named `col_0`, `col_1`, ...). See **Data** below. |
| `width` / `height` | `int`, `float`, `"container"` or `None` | `None` | Canvas size in pixels. `None` leaves the theme's size (640 × 480 by default). See **Size** below. |
| `title` | `str` or `Title` | `None` | Chart title. |
| `description` | `str` | `None` | Accessible alt text, written as the root SVG's `<desc>` and carried into HTML output. |

**Data.** Every accepted input is normalized once to a `pyarrow.Table` (`ferrum._coerce.to_arrow_table`) and then to the dtypes the Rust renderer accepts (`normalize_for_rust`: dates and every timestamp or duration unit become `timestamp[ms]`, dictionary columns are decoded, all-null columns become `float64`) before it crosses the CDI boundary. A polars frame crosses zero-copy. An unsupported type raises `TypeError` naming the supported ones. Two inputs the coercer knows are not chart data today: a `polars.LazyFrame` is accepted by `Chart(...)` but raises `TypeError` at render, and a `ModelSource` is not chart data at all (pass one of its derived frames, or use a figure function).

**Size.** `Chart(width=, height=)`, `Chart.properties(width=, height=)`, the `.properties(width=, height=)` of every composite (`HConcatChart`, `VConcatChart`, `ConcatChart`, `JointChart`, `RepeatChart`, `ClusterMapChart`, `LayerChart`) and `Chart.override(width=, height=)` share one rule (`ferrum._validate`): `None` and `"container"` pass unchanged, and anything else must be a finite number above 0 (an `int`, a `float` or a NumPy number; a `bool` is not a number here), stored as a `float`. Anything else raises a `ValueError` naming the call, e.g. `Chart.properties(width=...) must be positive; received 0`, `... must be a finite numeric pixel value; received nan`, `... must be a numeric pixel value (int or float), not str; received 'abc'`. `Chart(...)` refuses as `Chart(width=...)`, a composite as `<class>.properties(width=...)` (a nested composite names the outermost call), and `.override` as a `FerrumOverrideError` reading `Chart.override(width=...) ...` when overrides are applied; `.override(width=None)` is no override. `fm.compute_layout` keeps its own render-time `invalid viewport: ...` refusal. A numeric size renders identically as an `int` or a `float`, except a chart with no rows, whose SVG header is written from the stored value (`width="300.0"`). `"container"` is accepted and stored, but no render path consumes it yet: `to_svg`, `to_png`, `to_html` and `.interactive()` raise `TypeError: argument 'viewport': must be real number, not str`.

**Methods.** Every method returns a new chart and leaves the receiver unchanged, unless noted.
- `.encode(**channels)`: Bind encoding channels (§3.2). Channels are inherited by all layers unless a layer overrides them.
- `.layer(*layers)`: Add one or more `Layer` objects.
- `.mark_point()`, `.mark_line()`, ... (§3.3): Set the chart's mark, clearing any pending composite mark the chart was still carrying; a later primitive or composite mark call replaces the earlier one and carries its own style, whatever was drawn before. A fitted-layer mark (`.mark_smooth()`, `.mark_function()`) is the one exception: it keeps every existing layer and any primitive mark already set *over its own data*, each with its own data and style, and the fit joins them as one more layer. A primitive mark with no data of its own (`Chart(None).mark_line(stroke_width=7)`, a style-only placeholder) has nothing to keep as a layer: the fitted mark takes the single-chart shape, and the primitive's style merges onto the fitted line's, since there is no separate layer for it to keep. A fitted-layer mark called when a composite mark is still pending on the same chart refuses (`ValueError`, naming both calls and the `+` spelling that composes them instead): a chart holds one pending composite mark, so a second one cannot be kept there.
- `.transform(*transforms)`: Append data transforms to the chart's pipeline (§3.5). Stat transforms are triggered by statistical marks and run in the same pipeline (§3.4).
- `.facet(field=None, *, row=None, col=None, ncols=None, nrows=None, title=None, row_title=None, sort=None, row_sort=None)`: Wrap the chart in a facet grid (§3.9).
- `.share_scale(*, x=None, y=None)`: Facet scale resolution per channel, `"shared"` (default) or `"independent"`; `None` leaves the setting unchanged. Compositions take `resolve=` / `fm.Resolve(scale=, legend=)` instead (§3.9).
- `.coord(coord)`: Set the coordinate system (§3.8).
- `.axis(*, x=None, y=None, show=None)`: Spec-level axis suppression (§3.7). `.configure` and the `configure_*` family set axis, legend, title, grid, padding and color configuration (§3.7).
- `.add_selection(*selections)`: Bind `Selection` objects (§3.10).
- `.add_params(*params)`: Attach reactive parameters (§3.10).
- `.conditional(spec)`: Apply a `ConditionalSpec` built by `sel.when(...).otherwise(...)`; equivalent to `.add_selection(sel).encode(<channel>=spec)`.
- `.properties(*, width=None, height=None, title=None, subtitle=None, caption=None, description=None, render_config=None)`: Set chart-level properties; only the arguments given are updated.
- `.labs(**kwargs)`: Plotnine-style labels: `title`, `subtitle`, and axis titles `x` / `y` (wrapping the existing channel with `title=`).
- `.xlim(lo, hi)` / `.ylim(lo, hi)`: Set the x or y domain.
- `.theme(theme)`: Attach a `Theme`, which wins over the context default (§3.13).
- `.override(**kwargs)`: Store low-level spec-path overrides applied at render time; later calls win key by key.
- `.pipe(fn, *args, **kwargs)`: Return `fn(chart, *args, **kwargs)`.
- `.interactive(*, toolbar=True)`: Switch the render target to WASM. Returns `InteractiveChart`. See **Interactive output**.
- `.to_svg(*, raster=None)` → `str`, `.to_png(*, raster=None, scale=2.0)` → `bytes`, `.to_html(*, embed_wasm=True, toolbar=True, raster=None, csp_nonce=None)` → `str`: Render. `raster` overrides the chart's `RenderConfig` auto-raster policy for this render only (`True` forces raster aggregation, `False` per-element output).
- `.save(path, *, format=None, embed_wasm=True, raster=None, scale=2.0, toolbar=True, csp_nonce=None)`: Render and write to disk. `format` is one of `"svg"`, `"png"`, `"html"`, `"json"`, `"pdf"`, inferred from the extension when `None`. `scale` is the pixel density of PNG and PDF output only. `embed_wasm` (base64-inline the WASM binary), `toolbar` and `csp_nonce` (a `nonce=` on the inline `<style>` and `<script>` tags) apply to HTML only. PDF output is the PNG render wrapped in a single-page PDF.
- `.show(*, raster=None)`: Display inline in Jupyter (as SVG), else open in a browser. `.show_svg(*, raster=None)` and `.show_png(*, raster=None, scale=2.0)` are deprecated aliases that forward to `.to_svg` and `.to_png`.
- `.to_spec()`: Return the Rust `ChartSpec` this chart lowers to.
- `.to_json(*, indent=None)`: Return the `ChartSpec` as JSON. `.to_dict()` returns `json.loads(chart.to_json())`.
- `.layer_names`: Property: the named sub-layers after composite-mark desugaring.
- `__add__(other)` → `Chart` (a layered chart; see **Layered charts** under `ferrum.Layer`)
- `__or__(other)` → `HConcatChart`
- `__and__(other)` → `VConcatChart`

**Interactive output.** With `toolbar=True` (the default), the interactive canvas carries a toolbar of five tools: Pan (P), Box Zoom (Z), Box Select (S), Reset (R) and Save PNG. Box Select is active on load when the chart declares an interval selection or a domain-rescale parameter binding; otherwise Pan is. `toolbar=False` hides it, and `.save(..., toolbar=)` / `.to_html(toolbar=)` control it in exported HTML; it has no effect on SVG, PNG, PDF or JSON. Interactive renders inject hover tooltips from the encoded channels (x, y, color, size, ...) when the chart has no explicit `tooltip=` / `tooltip_fields=` encoding (`_spec_build._inject_auto_tooltips`); an explicit chart-level tooltip wins for every layer. On a layered chart, each layer's tooltip reports that layer's own encoded fields. A field-based linked selection on a layered chart supplies the selection-driven tooltip content at chart level, so per-layer injection does not run for it. Static SVG and PNG output carries no auto-tooltips.

---

#### `ferrum.Layer`

Explicit layer constructor. Used when multiple layers need independent data, marks, or encodings.

```
Layer(data=None, mark=None, *, name=None, encoding=None, transforms=None, mark_kwargs=None)
```

Most users use `mark_*()` methods on `Chart`, and `+`, instead of constructing `Layer` directly.

**Parameters.**

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `data` | data frame or `None` | `None` | The layer's own data; `None` reads the chart's. |
| `mark` | `str` or `None` | `None` | A mark name such as `"point"` or `"line"`. `None` draws a point. Anything that is not a string raises `ValueError: Layer(mark=...): a layer's mark is a mark name such as 'point' or 'line'; got 3`. |
| `name` | `str` or `None` | `None` | The layer's name. |
| `encoding` | `dict` or `None` | `None` | The layer's own encoding channels. |
| `transforms` | `None` or empty | `None` | Must be `None` or empty. A non-empty value raises `ValueError`: "Layer(transforms=[...]): a layer does not run its own transforms. Give the layer its own chart with .transform(...) and compose it with +, which routes the transform to that layer alone." |
| `mark_kwargs` | `dict` or `None` | `None` | The layer's mark style, checked at construction against the mark's honored style fields (the point's when `mark` is `None`), through the same Rust style table as `mark_*()` (§3.3). An unhonored key raises, e.g. `Layer: keyword argument 'font_size' is not honored by mark_point. Honored: [...]`. |

**Wire form.** A serialized layer refuses unknown keys when parsed. It has no `blend` key: a layer's blend is its style field `mark_style.blend`, so a saved spec whose layer carries `"blend"` raises ``unknown field `blend`, expected one of `mark`, `encoding`, ...``. A hand-built `ChartSpec` whose `layers[i].transforms` is non-empty is refused at render with a typed error naming layer `i`.

**Layered charts.** These rules hold for every layered chart, however it is built: `+`, `fm.layer(...)` / `LayerChart`, `SecondaryY`, independent-y, and the desugaring marks that add a layer. Static and interactive output follow them alike.

- **A layer's style is its own.** Every layer renders from its own `mark_style` alone; a kwarg-less sibling renders identically to its flat self (`mark_point(opacity=0.3, stroke_width=5) + mark_line()` draws the default line). A layered `ChartSpec` carries no chart-level `mark_style`, and a hand-built one that sets it is refused at render: `ChartSpec.mark_style is set on a layered chart (2 layers), where it has no effect: a layered chart is styled per layer. Move the style onto each layer's own mark_style.` A flat (no-layers) chart's `mark_style` applies in full. A figure function's `mark={...}` override on a layered chart applies to every layer's own style. `ferrum._core.mark_kwargs_fields()` lists every style wire key in declaration order.
- **Chart-level encodings are shared.** A combined chart's encoding, `color` and `detail` included, reaches every layer that does not set the channel itself, whenever it was declared and whatever layers are added later: `(a + b).encode(detail="g") + c` splits `c` by `g` too. When no layer's mark reads `detail`, the "accepted but ignored" warning fires once, at serialization. An interactive `LayerChart` merges its members with `+`, so a layered member's chart-level encoding reaches the members after it, while the static overlay renders each member alone.
- **Layer 0 establishes the shared axes.** The first chart (`fm.layer(...)`'s first argument, `+`'s left-hand side) decides whether a shared positional axis is continuous or ordinal. A later layer whose column on a continuous shared axis is neither numeric nor temporal is refused: `layer {i}: column '{field}' ({dtype}) cannot be drawn on the shared {channel} axis, which is continuous — a layered chart (fm.layer(...) / chart + chart) merges its members onto one panel. Cast the column, or give the layer its own scale with resolve={"{channel}": "independent"}.` (`fm.layer(...)` wraps it in `composite scale resolution failed: failed to resolve shared {channel} domain for field '{field}': ...`; `+` and `.interactive()` raise it bare.) The reverse order takes the ordinal union and renders. A member whose own `CoordFlip` differs from the panel's is judged by the coord-exchange gate instead.
- **An axis binds from the chart, else from the first eligible layer.** A positional axis takes the chart encoding's binding (merged with layer 0's) when there is one, otherwise the first layer that binds it (`independent_y` layers excluded for `y`). Whether a chart may render against a single axis (the `mark_tick` / `mark_rule` exemption) is decided by every layer that paints, not by the composed spec's root mark. A drawing layer that needs both positional channels must reach each from its own encoding or the chart's; reaching one only through a sibling's binding raises `ValueError`: `layer 0: mark_point needs both positional channels, but no 'y' encoding reaches it — a sibling layer binds 'y', which resolves the axis but leaves mark_point with nothing to plot. Encode 'y' on this layer, or on the chart, to draw it.` A channel bound nowhere raises the channel-level encoding error instead. `tests/test_single_axis_eligibility.py` pins that every ordered two-layer pair over `{point, line, tick, rule}` × `{x}, {y}, {x, y}` either refuses or draws all its marks.
- **Axis titles come only from a binding the caller wrote.** The sibling-binding fallback resolves a scale, not a title: `Chart(df).mark_tick(...) + Chart(df).mark_line(...)` renders on the line's scale with no y-axis title. To title it, put the title where a binding is read: on the composition (`(a + b).encode(y=fm.Y("density", title="Density"))`) or on the first chart. A `title=` on a later layer is ignored, so `tick + line_titled` is untitled and `line_plain + tick_titled` shows layer 0's field name.
- **`+` keeps each layer's data routing.** `a + b` preserves a layer's own `data_source` (a multi-output composite mark such as `mark_qq`, `mark_violin` or `mark_boxplot` points its layers at its own named transform), substituting the merge's name only when a layer has none, so such marks compose in either order. Two different composite charts whose named transforms share a name (every `mark_qq()` names its transform `"qq_main"`) are refused before the merge with a `ValueError` naming the layer and the transform output both sides would read; a self-add and two structurally identical transforms are allowed.
- **`+` over different data reads each operand's own rows.** `a + b` over two frames concatenates them, renaming the right frame's columns that collide with the left's. An operand with a transform chain (its own `.transform(...)`, a composite mark such as `mark_boxplot`, `mark_violin` or `mark_contour`, a per-layer aggregate or bin, or an earlier `+`) runs that chain over exactly the rows it brought, so the other frame's padded rows never reach its statistics: a boxplot, violin or mean bar beside another frame draws what `fm.LayerChart(a, b)` draws, with no empty `null` category on its axis. An operand with no chain draws straight from the combined rows. A chart-level `x`/`y` reaches every layer that binds none, on either side, whether it was written on the left operand or bound on the composition after the merge; a right layer reads it in the right frame's own columns, so `P(L).encode(x="a", y="b") + P(R)` and `(P(L) + P(R)).encode(x="a", y="b")` both draw both frames' `a`/`b`, as `fm.LayerChart` draws them. A right layer gets the left operand's binding as the caller wrote it, even where a composite mark rewrote it for its own layers (a histogram's binned `x`, `mark_smooth`'s fitted columns): `mark_smooth().encode(x="a", y="b") + P(R)` draws `R`'s `a`/`b`. A channel the caller never wrote (a histogram's `count` `y`) does not reach it until the composition binds that channel with `.encode(...)` (a retitle such as `.labs(x=...)` does not bind it). A drawing layer left without an `x` or `y` it needs is refused with the missing-positional-channel `ValueError` above, as `fm.LayerChart` refuses the same pair, rather than drawing nothing: `mark_histogram().encode(x="a") + P(R)` raises for `y`. A composed chart used as an operand keeps its own `x`/`y` for its layers. A step added to the composition after the merge (`(a + b).transform(...)`) runs over the left operand's rows, including those it reads through its own branch and those of a left operand nested in an earlier `+`; it does not reach a right operand's own chain.
- **Encoding aggregates and bins are per layer.** An aggregate or bin on a chart's encoding (`y="mean(y)"`, `x=fm.X("v", bin=True)`) is computed by each layer that binds or inherits it, over the rows that layer reads (the chart's rows after its `.transform(...)` steps, or an operand's own rows), grouped by the layer's channels merged over the ones it inherits, plus the facet fields. So `Chart(df).mark_bar().encode(x="g", y="mean(y)").layer(fm.Layer(mark="text", encoding={"text": fm.Text("y", aggregate="mean")}))` labels each bar with the mean it draws, and a point layer binding `color="c"` with `y=fm.Y("y", aggregate="mean")` draws the mean per (`g`, `c`). An encoding statistic is not a column other layers read: naming its output (`text="mean_y"`) is refused at render with the statistic remedy, and `+ fm.SecondaryY("mean_y")` is refused when it is added. To share aggregated rows, aggregate them with `.transform(fm.transform_aggregate(...))`.

**Bars and dodge.** Bars start at zero: an unstacked bar's base sits on its value axis's 0 pixel whenever 0 is in the value domain, vertical or horizontal, reversed or not, so a negative bar hangs below (or left of) the 0 tick; when 0 is outside the domain (a log scale, `zero=False`, an explicit domain such as `[40, 80]`) the base is the plot edge on zero's side, the edge nearer the domain endpoint with the smaller absolute value (`render/scale_resolve::ScaleKind::value_baseline_px`). A bar's span runs from its base to its value, signed, and is never clamped. A bar whose value extent is 0 (an empty histogram bin, a zero value, a stack whose values sum to 0) is a zero-height (or zero-width) rect at its base, so every row keeps its rect. A bar whose band or bin width is 0 emits no rect (`render/marks/bar.rs::is_zero_position_extent`); when that empties a batch that had rows to draw, the render warns once, e.g. `bar layer 0 drew nothing: mark_bar() computed a zero band or bin width for all 4 of its rects (...)`; a batch empty for another reason (no rows, null categories, non-finite values) stays quiet. Under `Dodge` on a band axis that carries a paired span channel (`x`/`x2`, `bin_start`/`bin_end`), both ends of each row's span move into the group's own sub-slot of that row's span (`render/position.rs::apply_dodge`), so a dodged histogram, including one with a single bin, draws side-by-side sub-bars rather than offset full-width bars, and a null endpoint stays null.

---

#### `ferrum.ModelSource`

Wraps a fitted estimator and a dataset; exposes derived data sources as polars frames.

```
ModelSource(model, X=None, y=None, *, feature_names=None, class_names=None, sample_weight=None, random_state=None)
```

**Parameters.**

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `model` | fitted estimator | required | Any object implementing the protocol a diagnostic needs (`predict`, `predict_proba`, `coef_`, `feature_importances_`, `cluster_centers_`, ...). The detected set is the read-only `.capabilities` property. |
| `X` | array-like or `None` | `None` | Feature matrix. Three methods run without it: `pca_variance` when the model exposes `explained_variance_ratio_`, `intercluster_distance`, and `importances` with its default `method="builtin"` when the source has `feature_names=` (without them it raises `ModelSource.importances() requires feature names: construct with ModelSource(model, X) or ModelSource(model, feature_names=[...]).`). Every other method on an X-less source raises `ValueError: ModelSource.<method>() requires the training data: ModelSource(model, X).`, where `<method>` is the method called, except for the methods derived from probabilities (`roc_curve`, `pr_curve`, `calibration_curve`, `cumulative_gain`, `lift_curve`, `discrimination_threshold`), which name `ModelSource.probabilities()`. A method that also needs `y` checks `y` first, so on a source with neither `confusion_matrix`, `calibration_curve`, `discrimination_threshold`, `cv_scores`, `learning_curve`, `validation_curve` and `alpha_selection` raise `ModelSource.<method>() requires y to be provided.` |
| `y` | array-like or `None` | `None` | Targets. |
| `feature_names` | sequence of `str` | `None` | Feature names. |
| `class_names` | sequence of `str` | `None` | Class names. |
| `sample_weight` | array-like | `None` | Per-sample weights. |
| `random_state` | `int` or `None` | `None` | Seed for every derived method with a random step: permutation importances, UMAP and t-SNE embeddings, the t-SNE projection of cluster centers, and the shuffled folds of `learning_curve`. Unset means seed `0`, so those methods are reproducible either way. The other methods are deterministic and ignore it. |

**Properties.** `.X`, `.y`, `.feature_names`, `.capabilities`, and `.class_names -> list[str] | None`: the constructor's `class_names=` if set, else the stringified `model.classes_` if present, else `None` (proxied on `ComparedModelSource`).

**Methods** (all return `polars.DataFrame`)
- `.predictions()`: `y_true`, `y_pred`, `residual`, `studentized_residual`, `cooks_distance`, `leverage`
- `.probabilities()`: `y_true`, then one `proba_<class>` column per class
- `.roc_curve(*, average=None, drop_intermediate=True, diagnostic=None)`: `fpr`, `tpr`, `threshold`, `class`, `auc`
- `.pr_curve(*, average=None, diagnostic=None)`: `precision`, `recall`, `threshold`, `class`, `ap`
- `.confusion_matrix(*, normalize=None, diagnostic=None)`: `actual`, `predicted`, `value`, `value_fmt`
- `.calibration_curve(*, n_bins=10, strategy="uniform", diagnostic=None)`: `mean_predicted`, `fraction_positive`, `count`, `brier`
- `.cumulative_gain(*, diagnostic=None)`: `percent_population`, `gain`, `class`
- `.lift_curve(*, diagnostic=None)`: `percent_population`, `lift`, `class`
- `.importances(*, method="builtin", n_repeats=30, scoring=None, random_state=None)`: `feature`, `importance`, `std`, `rank`. `method`: `"builtin"` (`feature_importances_` or `coef_`; `std` is 0) or `"permutation"` (sklearn's `permutation_importance`).
- `.shap_values(*, background=None, max_evals=500)`: `sample_id`, `feature`, `shap_value`, `feature_value`, `feature_value_normalized`, `class_label`. `background` defaults to the first 50 rows of `X`.
- `.partial_dependence(features, *, grid_resolution=100, kind="average")`: `feature`, `feature_value`, `pd_value`, `class_label`, `sample_id`
- `.silhouette(*, k=None)`: `sample_id`, `y_position` (stack order), `cluster`, `silhouette_value`, sorted within cluster descending. `k` filters to clusters in `range(k)`.
- `.embeddings(*, method="umap", n_components=2, **method_kwargs)`: `dim_0`, `dim_1` (`dim_2`), `label`. See **Embeddings**.
- `.learning_curve(*, cv=5, scoring=None, train_sizes=None)`: `train_size`, `split`, `score`, `mean_score`, `std_score`, `lower`, `upper`
- `.validation_curve(param, values, *, cv=5, scoring=None)`: `param_value`, `split`, `score`, `mean_score`, `std_score`, `lower`, `upper`
- `.discrimination_threshold(*, n_thresholds=50, cv=None, diagnostic=None)`: `threshold`, `precision`, `recall`, `f1`, `queue_rate`. Binary classifiers only. See **Discrimination threshold**.
- `.pca_variance(*, n_components=None)`: `component`, `explained_variance_ratio`, `cumulative_variance_ratio`. The model must expose `explained_variance_ratio_` (e.g. sklearn PCA, TruncatedSVD). `n_components` limits output to the first n components.
- `.rank1d(*, algorithm="shapiro")`: `feature`, `score`, `rank`. `algorithm`: `"shapiro"` (Shapiro-Wilk normality score), `"variance"`, or `"covariance"` (covariance with `y`).
- `.rank2d(*, algorithm="pearson")`: `feature_x`, `feature_y`, `correlation`. `algorithm`: `"pearson"`, `"spearman"`, `"kendall"` (Knight's O(n log n) tau-b, `ferrum._core.kendall_tau_b`) or `"covariance"`.
- `.cv_scores(*, cv=5, scoring=None)`: `fold`, `split`, `score`. `split` is `"train"` or `"test"`. `scoring` is an sklearn scorer string or callable, defaulting to the model's own scorer.
- `.alpha_selection(alphas, *, cv=5, scoring=None)`: `alpha`, `fold`, `score`, `mean_score`, `std_score`. CV score at each regularization value; intended for Ridge, Lasso, ElasticNet and other regularized estimators.
- `.intercluster_distance(k, *, method="mds")`: `cluster`, `x`, `y`, `size`. The first `k` cluster centers (capped at the model's center count) projected to 2-D, with `size` the membership count. `method`: `"mds"` (classical MDS, `ferrum._core.mds_classical`) or `"tsne"`, which needs at least 4 centers (fewer raises a `ValueError` pointing to `"mds"`) and clamps perplexity to `min(5, k - 2)`.

`diagnostic=` names the public entry a refusal or warning from that call carries; it defaults to the method's own name.

**Class method**
- `ModelSource.compare(models, X, y=None, **kwargs)` → `ComparedModelSource`. `models` is a `dict[str, estimator]`. Every derived method returns the same schema with an additional `model` column.

**`class_label`.** `shap_values()` and `partial_dependence()` label rows from one shared vocabulary (`sources/_importance.py::_output_class_labels`): a single-output model without `classes_` emits `"target"`; a multi-output model without `classes_` (e.g. a `RandomForestRegressor` fit on a 2-column `y`) emits `target_0 ... target_{n-1}`; a binary classifier emits the positive-class name; a K-class classifier emits one row group per class. The shap figure functions select and facet on `class_label`, so `per_class=False` selects the first output's rows, and `pdp_chart` binds `class_label` to color when a frame carries more than one class (grouping keys for `center=`, `kind="both"` and the ICE `detail` pairing all carry it, so curves never chain across classes).

**Feature names.** When the model exposes `feature_names_in_` and pandas is importable, `.partial_dependence()` and `.importances(method="permutation")` hand sklearn a pandas frame carrying those names, so sklearn sees what it was fit on and warns nothing; the numbers equal the numpy path's. Whether `X` was nameless is decided at construction: a bare `numpy.ndarray` and a pandas frame with a default `RangeIndex` for columns are nameless; a pandas frame with a real column index and any polars frame are named. A polars frame built from a bare array without `schema=` (auto-named `column_0, ...`) therefore counts as named, and a model fit under other names raises sklearn's feature-name-mismatch `ValueError`.

**Embeddings.** `method` is `"umap"`, `"tsne"` or `"pca"`, computed in Rust (manifolds-rs for UMAP and Barnes-Hut t-SNE). `method_kwargs` is checked per method: `"umap"` accepts `n_neighbors`, `min_dist`, `n_epochs`; `"tsne"` accepts `perplexity`, `learning_rate`, `n_iter`; `"pca"` accepts none. Any other key raises `ValueError` naming the accepted keys. t-SNE `perplexity` defaults to `min(30, n_samples - 2)`, floored at 1. An explicit value is validated, never reduced: non-numeric, non-finite, non-positive, or above `n_samples - 2` raises a `ValueError` naming `perplexity`, `n_samples` and the bound. UMAP and t-SNE output is reproducible for a given `random_state` within one platform and toolchain, but is not byte-stable across platforms (manifolds-rs does not use the ChaCha8 RNG the transforms use), so it has no goldens.

**Discrimination threshold.** `cv=None` sweeps on the source's own data. An int `cv` resolves through sklearn's `check_cv(cv, y, classifier=True)`, an unshuffled `StratifiedKFold` with no seed to thread; a splitter object is called as `.split(X, y)`. A train fold left with a single class warns, naming the fold and the missing class, before the fit fails. A test fold with a single class contributes `precision = recall = f1 = 0` with its real `queue_rate`.

**Ranking.** `rank1d_chart` / `rank2d_chart` and the `Rank1DVisualizer` / `Rank2DVisualizer` route raw tabular input (`polars.DataFrame`, `pyarrow.Table` of any chunk count, `pandas.DataFrame`, 2-D `numpy.ndarray`) to the Rust rank kernels through `ferrum._coerce.to_polars`, so every chunk is read and an empty frame keeps its columns; a fitted estimator goes through `ModelSource`, and an X-less one raises the typed `requires the training data` refusal for every algorithm. `algorithm="covariance"` requires `y`. A `y` whose length differs from `X`'s row count raises a typed `ValueError` before dispatch. A null `y` slot is dropped from the computation, not read as zero, with a warning naming the algorithm and the dropped row count; an all-null `y` degrades to n = 0 without error.

**Cache.** Each derived method caches its frame under a key of the method name and its kwargs (`sources/_base.py::BaseSource._cache_key`). Array-like kwargs (`ndarray`, `list`, `tuple`, nested) are normalized to nested tuples of their own values, with each numeric leaf keeping its kind, so equal content keys equally only when the types match too (`train_sizes=[1]`, an absolute count, and `train_sizes=[1.0]`, a fraction, key apart). A value still unhashable after normalization (a dict, a CV splitter, an object with no usable `__hash__`) is never cached: the call always recomputes, and nothing is stored. Two logically different calls therefore never share a cached frame.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#31-top-level-entry-points).

---

### 3.2 Encoding Channels

Encoding channels are typed objects passed as keyword arguments to `.encode()` (or held in a `Layer`'s `encoding=` dict). Every channel also accepts a bare field-name string as shorthand: `encode(x="hp")` is `encode(x=fm.X("hp"))`.

**Constructor contract** (`ferrum.encoding.base.ChannelBase`). Every channel class is `Channel(field=None, **kwargs)`. `field` is a column name (a `str`, or a `RepeatSpec` placeholder); any other type raises `TypeError`. A field string carrying an inline type (`"hp:Q"`) is split into field and `type`, and `type_=` is an accepted spelling of `type=` (`type` wins when both are given). `type` takes `"Q"`/`"N"`/`"O"`/`"T"` or the long spellings (`"quantitative"`, ...), normalized to the one-letter code; when omitted, the type is inferred from the column dtype at render. Each class honors a fixed set of keyword arguments, its `_honored_kwargs`, which is the single source of truth for what a channel accepts:

- A kwarg in the honored set is validated and serialized.
- `title=` on a channel that declares no title of its own is refused with `ValueError` (`<Channel>(title='T') is not honored: <Channel> draws no legend, and title= is not supported on this channel.`).
- A kwarg the channel permanently refuses (`columns=` on `FacetRow`/`FacetCol`) raises `ValueError` stating why.
- Any other kwarg warns once (`ferrum._warn.warn_once`) and is dropped.

**Legend surfaces** (`ChannelBase._legend_surface`). `Color`, `Fill`, `Stroke`, `Size`, `Shape` and `StrokeDash` own a legend: `legend=None`/`False` suppresses it and a `Legend(...)` or dict configures it. `X` and `Y` have no legend of their own: a `Legend(...)` there contributes style overrides to the shared color-legend cascade (precedence `color` > `x` > `y`, field by field), and a bare `None`/`False` is a no-op. Every other channel has no legend: `legend=None`/`False` is accepted silently and puts nothing on the wire, and any other legend value raises `ValueError`. A `Size`, `Shape` or `StrokeDash` legend value whose `orient=` is anything but `"none"` raises `ValueError`, because auxiliary legends always share the chart's legend block (place the block with `Color(legend=Legend(orient=...))` or `configure_legend(orient=...)`); `orient="none"` disables that channel's own block.

**Legend titles.** A legend-owning channel's `title=` is the title its legend prints, on every legend kind (swatch legend, continuous and binned colorbars, conditional legend, facet band, auxiliary blocks); with no `title=` the field name is the title. A legend reads its binding from the first layer's resolved encoding, so a channel bound only on a layer still draws and titles its legend, and a chart-level binding of the same field merges into the layer's binding property by property, the layer's own properties winning.

**Shorthand strings.** Positional and appearance channels accept:

- `"fieldname"`: a bare field.
- `"aggregate(fieldname)"`: an aggregate applied before mapping, e.g. `"mean(price)"`, `"count()"`. The aggregate vocabulary is `mean`, `sum`, `count`, `min`, `max`, `median`, `variance`/`var`, `stdev`/`std`, `q1`, `q3`, `distinct`; any other name raises `ValueError` listing it.
- `"fieldname:Q"` / `":N"` / `":O"` / `":T"`: inline type annotation (quantitative, nominal, ordinal, temporal).

**Which channels render** (`ferrum.encoding._channel_class_map()`; `encode()` accepts every channel and sorts each into exactly one bucket):

| Bucket | Channels | Behavior |
|---|---|---|
| Honored | `x`, `y`, `x2`, `y2`, `color`, `size`, `shape`, `opacity`, `fill_opacity`, `stroke_opacity`, `stroke_width`, `stroke_dash`, `angle`, `text`, `tooltip`, `href`, `url`, `description`, `key` | Serialized to its own `EncodingSpec`, chart-level and per layer. |
| Alias | `fill`, `stroke`, `detail` | `fill` and `stroke` alias to `color`. When `color` is also bound, `fill` is dropped silently and `stroke` is dropped with a once-per-call warning. `detail` becomes the mark's group-by key; only `mark_line`, `mark_area` and `mark_polygon` read it, and any other mark warns once that it is ignored. |
| Warn | `x_error`, `y_error`, `x_error2`, `y_error2`, and `tooltip_field` used as a top-level channel | Warned once and omitted from the spec. `mark_errorbar` computes its own extents; `TooltipField` is valid only inside `Tooltip(...)`. |
| Polar | `theta`, `radius`, `theta2`, `radius2` | Remapped to `x`/`y` and rendered when the encoding's effective coordinate system is `CoordPolar`; otherwise warned once and omitted. The effective coord is the chart's for the chart-level encoding, and for a layer's own encoding it is the layer's own `coord` if it declares one, else the chart's. |
| Facet | `facet`, `facet_row`, `facet_col` | Resolved into the chart's facet state (see Facet below), never through the warn path. |

**Warning deduplication.** `warn_once` deduplicates on `(channel, kwarg, message)` per process: a repeat of the same message is silent, and two different messages for the same pair both emit. Tests reset the registry with `ferrum._warn.reset_warnings()`.

#### Positional

| Class | Shorthand alias | Description |
|---|---|---|
| `X(field=None, *, type, bin, aggregate, scale, axis, title, stack, sort, impute, format, format_type, legend)` | `"fieldname"` | Horizontal position |
| `Y(field=None, *, type, bin, aggregate, scale, axis, title, stack, sort, impute, format, format_type, legend)` | `"fieldname"` | Vertical position |
| `X2(field=None, *, type)` | none | Horizontal span end (for `mark_rect`, `mark_rule`, ...) |
| `Y2(field=None, *, type)` | none | Vertical span end |
| `XError(field=None, *, type)` | none | Symmetric error extent on X (warn bucket) |
| `YError(field=None, *, type)` | none | Symmetric error extent on Y (warn bucket) |
| `XError2(field=None, *, type)` | none | Asymmetric error upper bound on X (warn bucket) |
| `YError2(field=None, *, type)` | none | Asymmetric error upper bound on Y (warn bucket) |
| `Theta(field=None, *, type, stack)` | none | Angular position (arc/pie marks, under `CoordPolar`) |
| `Radius(field=None, *, type, stack)` | none | Radial position (stacked radius requires `mark_bar` under `CoordPolar`) |
| `Theta2(field=None, *, type)` | none | Angular span end |
| `Radius2(field=None, *, type)` | none | Radial span end |

Positional keyword arguments (every default is unset):

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `type` | `str` | inferred from dtype | `"Q"`, `"N"`, `"O"`, `"T"` or the long spellings. |
| `bin` | `bool` or `Bin` | unset | Bin the field before mapping (see Encode-level binning). |
| `aggregate` | `str` | unset | Aggregate applied before mapping, from the shorthand vocabulary above. |
| `scale` | scale object | unset | Explicit scale (§3.6). |
| `axis` | `Axis`, dict or `False` | unset | Axis configuration (§3.7); `False` suppresses the axis. |
| `title` | `str` | the field name | Axis title. |
| `stack` | `bool` or `str` | unset | `True`, `"zero"`, `"normalize"`, `"center"`, or a falsy spelling (`False`, `"false"`, `"null"`, `"none"`) to disable stacking. |
| `sort` | `str`, list or dict | unset | Ordinal domain order: `"ascending"`/`"descending"`, an explicit list of values, a channel shorthand such as `"-y"` (sort by the per-category aggregate of another channel), or a sort-field object `{"field", "op", "order"}`. |
| `impute` | dict | unset | Imputation applied before mapping. |
| `format` / `format_type` | `str` | unset | Tick-label format and its kind (`"number"` or `"time"`); `formatType` is an accepted spelling of `format_type`. |
| `legend` | `Legend` or dict | unset | Style overrides for the shared color legend (cascade surface, above). |

**Sort-field validation.** A sort-field object without a string `"field"`, or with an `"op"`/`"order"` outside the accepted vocabulary (read from `ferrum._core.sort_accepted_vocabulary()`), raises `ValueError` at construction naming the channel, on every channel that honors `sort` (`X`, `Y`, `Color`, `Shape`, `StrokeDash`, the three facet channels) and on `EncodingSpec`. An absent `"op"`/`"order"` key takes its default (`sum`, `ascending`); a key present with `None` is refused. A hand-built spec that reaches render with a malformed sort object refuses with `RenderError::SortSpecMalformed` (`"{location}: sort={value} is malformed ({reason})"`). A well-formed sort that cannot be applied (an unresolvable field, an unknown channel shorthand, a channel-reference sort on an unbound axis) warns and falls back to insertion order.

**Encode-level binning.** `X(field, bin=True)` (or `Y`) is lowered once, by one helper shared by `Chart.encode()` and layer resolution (`_lower_bin_encoding`), into a `Bin` transform: the binned channel is re-pointed at `bin_start` and, unless a span channel is already bound, `x2`/`y2` is added at `bin_end`, so each bar spans its own bin. An aggregate on the opposing channel folds into the same `Bin`: a fieldless `Y(aggregate="count")` reads the bin's `count` column (rows per bin), and `Y("v", aggregate=...)` with any method (including `"count"` on a named column, which counts non-null `v` values) sets the `Bin`'s `agg_field`/`agg` and reads its `agg_value` column. Axis titles keep the user's field names (`"v"`, not `bin_start`/`agg_value`). The fold refuses with `ValueError`: more than one `bin=` in an encoding (use `mark_hex()` or `fm.Bin2D` for two-dimensional binning); one channel carrying both `bin=` and `aggregate=`; more than one channel carrying `aggregate=`; a fieldless aggregate other than `count`; a pre-built `bin=Bin(...)` asked to absorb an opposing aggregate (name `agg_field`/`agg` on the `Bin(...)` instead); and an unknown aggregate name.

**Remap channel rule.** A composite mark that re-points a positional channel at its transform's output (`mark_histogram`, 1-D `mark_density`, `mark_smooth` without `ci`, `mark_function`) keeps every option the caller wrote on that channel; an explicit `title=` wins, else the caller's field name is stamped when the field changed. `bin`/`aggregate` are consumed, `type` is forced to quantitative, and an explicit `scale=` carries through directly. A channel the caller never bound gets a fresh quantitative channel; `x2`/`y2` are untouched.

**Position adjustments** (`ferrum.position`). A sibling concept to channels: an immutable value passed as `position=` on an eligible mark, at chart or layer level, and serialized as `{"type": "<kind>", ...}`.

```
Identity()
Dodge(by=None, padding=0.05)
Jitter(axis="x", width=0.4, seed=None)
Stack(by=None, offset="zero", anchor="top", value_axis=None)
```

| Position | Eligible marks |
|---|---|
| `Identity` | every mark (the default; a no-op) |
| `Dodge` | `bar`, `point`, `box`, `boxplot`, `boxen`, `swarm`, `violin`, `errorbar`, `errorband`, `ribbon`, `histogram`, `density`, `importance`, `shap_bar`, `cv_scores`, `text` |
| `Jitter` | `point`, `swarm`, `tick` |
| `Stack` | `bar`, `area`, `ribbon`, `histogram`, `density` (segment top), and `text`, `point`, `rule`, `tick` (labels placed on a stacked layer) |

- `Dodge.by` and `Stack.by` name the grouping field; when omitted, the color/fill field is used. `Dodge.padding` is a fraction of the band width in `[0, 1)`.
- `Jitter.axis` is `"x"`, `"y"` or `"both"`; jitter uses a ChaCha8 RNG seeded from `seed`, or from each row's data when `seed=None`, so output is deterministic.
- `Stack.offset` is `"zero"`, `"normalize"` (percent fill) or `"center"` (streamgraph). `anchor="top"` places each row at its segment top, `"mid"` at its midpoint. `value_axis` names the cumulated channel; `None` infers it from `CoordFlip`.
- An ineligible (mark, position) pair raises `TypeError` at chart-build time, naming the eligible marks.
- **Column dtypes.** All three adjustments read the value column through one numeric reader, so the same numeric dtypes are accepted by `Stack`, `Dodge` and `Jitter`. `Stack` keys its category column through the ordinal-category reader the band ladder uses, so two rows share a stack bin exactly when they share a band; where no band ladder exists (a continuous or temporal axis) rows stack by exact value, so `mark_area(position=Stack())` over a date or timestamp `x` stacks per instant. A temporal column declared ordinal is refused.

#### Appearance

| Class | Description |
|---|---|
| `Color(field=None, *, type, scheme, scale, legend, title, sort, condition)` | Fill and stroke color |
| `Fill(field=None, *, type, scheme, scale, legend, title, condition)` | Alias of `color` (its conditional also targets `color`) |
| `Stroke(field=None, *, type, scheme, scale, legend, title)` | Alias of `color`; not a conditional target |
| `Opacity(field=None, *, type, scale, condition)` | Overall opacity |
| `FillOpacity(field=None, *, type, scale, condition)` | Fill opacity |
| `StrokeOpacity(field=None, *, type, scale, condition)` | Stroke opacity |
| `StrokeWidth(field=None, *, type, condition)` | Stroke width, a per-row constant (no scale) |
| `StrokeDash(field=None, *, type, scale, legend, title, sort, condition)` | Dash pattern |
| `Size(field=None, *, type, scale, legend, title, condition)` | Point area, or line width |
| `Shape(field=None, *, type, scale, legend, title, sort)` | Point shape |
| `Angle(field=None, *, type, condition)` | Point rotation in degrees, a per-row constant (no scale) |

Every keyword argument above is optional and defaults to unset, and an unset keyword is not written to the encoding: `type` is inferred from the column dtype, `scheme` and `scale` come from the theme and scale resolution (§3.6), a channel with its own legend draws it titled with the field name, `sort` keeps the default domain order, and no `condition` applies. `field=None` binds no column.

`scheme=` names a palette and is validated at construction against the palette registry. `StrokeWidth` and `Angle` take no `scale=`; a `scale=` on either warns once and is dropped.

**Color vocabulary.** Every color string accepts one vocabulary: the 148 CSS Color 4 named colors (case-insensitive, trimmed), `#rgb`/`#rgba`/`#rrggbb`/`#rrggbbaa` hex, and `rgb(r,g,b)`/`rgba(r,g,b,a)` with integer 0-255 channels and a float 0-1 alpha. An unparseable string raises a typed error naming the accepted forms at mark construction (`fill=`/`stroke=`/`color=`), at render-time style resolution, in `ferrum.color.to_hex`, in selection styling, in the `background=` override, and at construction of the chart-config color fields (`AxisConfig`, `LegendConfig`, `TitleConfig`, `GridAxisConfig`, `GridConfig`; `"transparent"` stays legal in `GridConfig.band_colors`). An explicit `range=` or gradient entry on a `Color`/`Fill`/`Stroke` scale or in `configure_color(range=)` that does not parse refuses with a typed `RenderError` naming the first bad entry; parsing is all-or-nothing because the range is indexed by position. `ferrum.color.to_hex` reads a tuple's fourth component as alpha (emitting `#rrggbbaa`) and raises `ValueError` on a non-numeric component.

**Clearing paint.** `"none"` and `"transparent"` (case-insensitive, trimmed) clear a mark's fill or stroke, and an explicit clear outranks a bound color channel. A cleared fill writes SVG `fill="none"`; a cleared stroke writes no stroke attribute; a cleared text fill writes `fill="none"` on each `<text>`, and the interactive runtime draws nothing for it. A paint that merely evaluates to transparent (`"#00000000"`) is not a clear.

**Continuous color on lines.** `mark_line` and `mark_ribbon` cannot vary color along a stroke: a continuous or discretizing `Color` scale on either emits a `RenderWarning` naming the mark and the scale kind and suppresses the colorbar.

**Opacity-family scales.** `Opacity`, `FillOpacity` and `StrokeOpacity` honor `scale=`. With no explicit scale, a quantitative field maps its data extent to `[opacity_min, opacity_max]` (theme default `[0.1, 1.0]`); explicit range endpoints clamp into `[0, 1]`. For `fill_opacity`/`stroke_opacity`, a constant column with neither an explicit domain nor an explicit range keeps each row's own value as a literal alpha; with an explicit domain or range it resolves the degenerate-domain midpoint `(lo + hi) / 2` of the clamped range. `Opacity` always repaints a constant column to that midpoint. A non-numeric column on `fill_opacity`/`stroke_opacity` is a typed render error naming the channel and dtype. A non-linear scale (`Log`/`Pow`/`Sqrt`) on any opacity channel warns (`RenderWarning::UnsupportedOpacityScale`, naming the fallback actually taken: linear resolution, or literal alphas for a constant column) and is not honored.

**Auxiliary-scale warnings.** `ShapePaletteOverflowed`, `StrokeDashPaletteOverflowed` and `UnsupportedOpacityScale` are recorded once per render whatever the panel count. `ColorPaletteOverflowed` is recorded once per panel, because an independently resolved color scale can overflow differently per panel.

**Size.** On a point, `size` is an area in px² on the size scale (theme default range 4 to 36). On a line, `size` is a width that varies vertex by vertex within each series (a taper): `w(v) = 2*sqrt(size_scale(v)/pi)`, the diameter of a point of the same size value, so points and lines layered on one size field share one scale and one legend.

- Each `color`/`detail`/categorical `stroke_dash` group tapers separately; a null or non-finite size row takes the group's constant width; a step `interpolate=` holds a row's width across its step, a curved one gives each command endpoint its row's width; a `BreakAxis` cut interpolates the width in arc length. A dash on a tapered line is laid along the centerline's arc length; a zero-length dash under a round or square cap draws a dot one width across, and nothing under a butt cap. A one-vertex tapered piece draws nothing.
- Static output draws one filled outline per tapered node; `.interactive()` draws the width perpendicular to the on-screen centerline at any zoom, and hover tolerance is the widest vertex. Unsized lines carry no `widths` and serialize unchanged.
- A line layer that binds both `size` and `stroke_width` itself (fields or constant kwargs) refuses at render (`to_svg()`, `to_png()`, `save()`, `.interactive()`) with `ValueError`: `mark_line: size and stroke_width both set the line's width (size={s}, stroke_width={w}); bind only one. size= sets a dot-diameter width that can taper along the line; stroke_width= sets a width in pixels.` A size value that maps to no positive width also refuses.
- A layer's own width outranks an inherited one: a line's own `stroke_width` outranks a `size` it only inherited, and its own constant `size=`/`stroke_width=` outranks an inherited field of either channel. A line that inherits both keeps the inherited `stroke_width`, untapered.
- When every layer that draws the size field is a line, each size-legend entry is a line swatch whose stroke width is the entry's diameter; a mixed point-and-line legend keeps circles. A size legend is drawn only when some layer actually draws the size field (point, line or text, after width precedence).
- A conditional `size` on a line (`size=fm.when(sel).then(a).otherwise(b)`) is honored interactively, re-tessellating the line on each selection change; static export draws the base width. A non-positive branch on a line's own conditional refuses.

**Constant `size=`.** Which marks draw a constant `size=` (point and line) and which values are legal (finite and above zero) is one Rust rule, published as `ferrum._core.marks_drawing_constant_size()` and `check_mark_size(entry, mark, size)`, and applied at render and at construction by every door (`mark_*` kwargs, `fm.Layer(mark_kwargs=)`, a figure's `mark=` and per-layer kwargs, a visualizer's `mark=`, `mark_function(**mark_kwargs)`). `mark_line(size=a)` draws a constant width `2*sqrt(a/pi)`; an encoded `size` outranks it. A non-positive value raises `ValueError` (`mark_line(size=...) must be a positive number; received 0`); `size=` on a mark that draws no size raises `{call}(size=...) has no effect on mark_{mark}; received {value}.`, adding the mark's own width kwarg where one exists (`Set stroke_width= instead.`). A composite's catch-all `size=` meets the rule of the first emitted mark that draws it and is spread only onto emitted marks that draw it (`mark_smooth(size=40)` sizes its fit line, not its band); a composite that emits no point or line refuses it, naming the mark it emits.

**StrokeDash and the dash palette.** The dash palette has one home, `ferrum_scene::DASH_PALETTE`, read by the SVG path, the GPU packed-instance path and the WASM shaders: solid plus seven patterns, `[6,3]`, `[2,3]`, `[6,3,2,3]` and the four named linetypes `"dashed"` `[4,2]`, `"dotted"` `[1,3]`, `"dashdot"` `[4,2,1,2]`, `"longdash"` `[8,4]`. A mark's own `stroke_dash=` value (a named linetype, a `"4,2"` string or a list) or a `StrokeDash` conditional literal can select any of them, and an interactive dash alternates along the stroke as the static `stroke-dasharray` does. A pattern outside the palette renders solid interactively. A bound `stroke_dash` column is honored by `mark_line`, `mark_rule`, `mark_point`, `mark_bar` and `mark_rect`: a numeric column is a palette index clamped to the first four slots (solid plus the first three patterns), and a categorical column resolves through `StrokeDashScale`, cycling its categories through those same four slots (`StrokeDashPaletteOverflowed` past four categories). A categorical `stroke_dash` splits `mark_line`/`mark_ribbon` series by (color, detail, stroke_dash). A dash-swatch legend draws beside the shape legend. `StrokeDash(scale=...)` honors the domain (it reorders the category-to-pattern assignment); an explicit `range=` raises `ValueError` at construction, because the categorical scale's output is a fixed set of palette patterns, and a hand-built spec with one refuses at render (`StrokeDash(scale=...) does not honor an explicit range (received [...]): the categorical scale's output is a fixed set of 4 palette patterns, not a value a scale can redeclare`).

**Shape.** `Shape(scale=...)` honors domain (category-to-glyph order) and range (the glyphs). Range entries are parsed all-or-nothing against the strict point-shape table (`circle`, `square`, `cross`, `diamond`, `triangle-up`, `triangle-down`, `hline`, `vline` and their aliases); an unknown entry raises `ValueError` at construction naming `Shape(scale=...)`, and a typed `RenderError` for a hand-built spec. The shape reader takes the same column dtypes as `color` (string, categorical, integer, float, boolean).

**Layer promotion.** Under `chart + chart` and `Chart.layer()`, a later layer's own `color`, `size`, `opacity`, `fill_opacity`, `stroke_opacity`, `shape` or `stroke_dash` binding that the host leaves unbound is promoted to the chart level, the first binding in layer order winning, so `point + sized_line` draws exactly as `sized_line + point`. Promotion reads only what the caller authored: a binding a composite mark or figure function placed on some of its own layers is never promoted. A metric label (`AUCLabel`, `APLabel`, `BrierLabel`) promotes its base's own bindings so it draws in its curve's hue.

**Conditionals.** A selection conditional, `encode(<channel>=sel.when(...).otherwise(...))` or `<Channel>(field, condition=...)` (the two spellings are equivalent), may target exactly `color`, `size`, `opacity`, `stroke_width`, `stroke_opacity`, `stroke_dash`, `fill_opacity` and `angle` (`ferrum.encoding._channel_policy._CONDITIONAL_CHANNELS`, mirrored from Rust's `EncodingValue` variants and test-enforced against them). `fill` is accepted and carries its conditional onto `color`. `stroke`, `shape`, `x` and `y` are refused by name (`_CONDITIONAL_REFUSALS`), as is every other channel, with `ValueError` at `encode()` naming the supported set.

- **One conditional per wire channel.** Two declarations that resolve to the same wire channel raise `ValueError` at the call that creates the conflict, at every declaration door: `encode()` (both spellings), `Chart.conditional()`, `Chart.layer(Layer(encoding=...))` (both spellings, lifted into the chart's conditionals there) and `chart + chart` (checked cumulatively across a chain). The message names the door, both declarations and the wire channel: `"{context}: {first} and {second} both declare a conditional on the '{wire}' wire channel{detail}; a channel can carry only one conditional. Drop one of the two declarations, or move one to a different channel."`, where a declaration is named by its parameter spelling or by its role and selection (`the left chart's conditional on selection 's'`).
- **An alias's own field against its own conditional.** `encode(color=fm.Color("d"), fill=fm.Fill("c", condition=cond))`, in either order or in a later `encode(fill=...)` call, raises, because the `fill -> color` alias would drop `fill`'s field while its conditional survived. A fieldless `fill` conditional builds. The reverse cross-call order (`encode(fill=fm.Fill("c", condition=cond))`, then a later `encode(color=...)`) builds and drops `fill`'s field silently.
- **Scope.** A conditional is scoped, once when the spec is built, to the layers of the chart or `fm.Layer` that declared it: an operand of `chart + chart` and a `fm.Layer`'s own encoding keep their conditionals on their own layers, while the host of `Chart.layer()` and of `chart + fm.SecondaryY(...)` shares its conditionals with every layer attached to it. On the wire each `ChartSpec.conditionals` entry carries an optional `layers` list (absent means the whole chart); naming a layer the chart lacks refuses. A figure's `mark={name: False}` that drops every layer a conditional was declared on drops the conditional too.

**Related mark rules.** These rules travel with the channels above:

- **Style kwargs.** The style-door contract (which doors take mark style, how a refusal is voiced, `None` as unset, and how a figure's `mark=` dict reaches its layers) is stated once in §3.3 Marks.
- **Figures title their legends** with real names, never an internal column name (for example `confusion_matrix_chart` `"Proportion"`, `roc_chart`/`pr_chart` `"Class"`, or `"Model"` under `compare=`; `heatmap`/`clustermap` the input's single numeric column name, else `"Value"`); a user's own `color=`/`encode=` override wins.
- **Decimal columns.** `Linkage`, `Reorder(axis="columns")`, `heatmap`, `clustermap` and `pairplot` read integer and float columns (`ferrum._core.numeric_columns`); a `Decimal` column raises `ValueError`: `<entry>: column '<name>' is a decimal column (<dtype>); a numeric column must be an integer or float column, so cast it to float first`.
- **Truth-value arguments.** `mark_contour(fill=, smooth=)`, `mark_density(fill=, cumulative=, smooth=)` and `mark_qq(dequantize=, line=)` take only `True`, `False` or `None` (`mark_density: fill must be True or False; got 'red'`). `mark_bar(zero=None)` keeps the zero anchor; `mark_raster(blend=None)` draws as `normal`.

#### Text / Detail / Tooltip

| Class | Description |
|---|---|
| `Text(field=None, *, type, format, format_type)` | Text label content |
| `Detail(field=None, *, type)` | Group-by without a visual channel (alias bucket) |
| `Tooltip(*fields, type)` | Hover tooltip fields. Two or more fields may mix strings and `TooltipField` objects; a single field must be a string (`Tooltip(TooltipField(...))` raises `TypeError`), and a lone `TooltipField` is passed directly as `tooltip=` |
| `TooltipField(field=None, *, type, title, format, format_type)` | Tooltip field with a label and formatting |
| `Href(field=None, *, type)` | URL for click-through on interactive charts |
| `Url(field=None, *, type)` | Image URL (`mark_image`) |
| `Description(field=None, *, type)` | Accessible description per mark |
| `Key(field=None, *, type)` | Identity key for animated transitions |

Every keyword argument above is optional and defaults to unset, and an unset keyword is not written to the encoding: `type` is inferred from the column dtype, an unset `format`/`format_type` uses the default formatter for the value, and an unset `TooltipField(title=)` labels the field with its name.

- **Formats.** `Text(format=)` and `TooltipField(format=)` resolve named format presets (`"currency"`, `"date_iso"`, `"ordinal"`, ...) through the same resolver as `Axis`/`Legend` (§3.7). `formatType` is an accepted spelling of `format_type`.
- A `text` field that names a column absent from the layer's batch, or one neither the string nor the numeric reader can read, refuses on `mark_text`, `mark_label` and the heatmap cell-annotation text, rather than rendering blank. A `color` column absent from one layer's own batch, but present on a sibling layer sharing the encoding, renders that layer with no color.
- **Key.** Static SVG output is identical with and without `key=`. In the interactive runtime, marks in a paired batch (batches pair by panel and batch index) are matched by key during a scene transition: matched keys interpolate geometry and color, a key present only on the new side enters (opacity ramping up), and a key present only on the old side exits (opacity ramping down). A batch falls back to index pairing as a whole when either side lacks usable keys: absent keys, a duplicate key, a key/instance count mismatch, a mixed-kind batch, or a key column whose values are not distinct per row (`Boolean` above two rows, nulls, saturated floats) or whose dtype has no key reader (`Duration`, `List`, `Struct`). Packed batches (1000 marks or more) carry keys in a `HAS_KEYS` sidecar. Mesh-backed marks (line, area, path) have no per-mark identity and do not transition. Key-based selection membership and key-addressed hit-testing do not exist.

#### Facet (used with `.facet()`)

| Class | Description |
|---|---|
| `Facet(field=None, *, type, sort, title, columns)` | Wrap facet dimension |
| `FacetRow(field=None, *, type, sort, title)` | Row facet dimension |
| `FacetCol(field=None, *, type, sort, title)` | Column facet dimension |

Every keyword argument above is optional and defaults to unset: `type` is inferred from the column dtype, `sort` keeps the default domain order, an unset `title` gives each strip the bare value (see Titles), and an unset `columns` leaves the wrap count to `Chart.facet(ncols=)`, else a single column (a `FacetCol` dimension lays its panels side by side).

- **Titles.** With a `title=`, a strip reads `"{title}: {value}"`; without one, or with an empty or whitespace-only title, it reads the bare value (never the field name). A non-blank title renders verbatim, untrimmed. `Chart.facet(title=)` titles the column strip in grid mode and the only strip in wrap mode; `row_title=` titles the grid row strip. `FacetRow` used alone lowers to wrap mode, so its title and sort route to the wrap strip's `title`/`sort`; in grid mode `FacetCol` routes to `title`/`sort` and `FacetRow` to `row_title`/`row_sort`.
- **Wrap count.** `Facet(columns=N)` writes the same `ncols` that `Chart.facet(ncols=N)` writes, last writer winning. `FacetRow(columns=)`/`FacetCol(columns=)` raise `ValueError`, since a grid has no wrap count.
- **Sort.** The three facet channels and `Chart.facet(sort=, row_sort=)` take the positional `sort` vocabulary and order panels through the ordinal domain's sort resolution. `Chart.facet(row_sort=)` on a single-dimension declaration raises `ValueError`. A well-formed facet sort that cannot be applied warns `RenderWarning::FacetSortSpecIgnored` (`"facet sort spec could not be applied ({reason}); panels fall back to insertion order"`); a hand-built `row_sort` without a row dimension refuses with `SortSpecMalformed`.
- **One writer of facet state.** `Chart.encode()`'s facet channels, every arity of `Chart.facet()`, and `Chart.share_scale()` all write through `_Facet.merged_with` (`src/ferrum/_facet.py`). A field a call states wins; a field it does not restate (`title`, `row_title`, `sort`, `row_sort`, `resolve`, `ncols`/`nrows`, `wrap_orient`, and the grid's other dimension) survives. A row or column declaration onto a grid replaces only that dimension. Onto a single-dimension facet, a declaration of the same orientation replaces the field and stays a wrap, while one of the other orientation (or an orientation-less `Facet`/`.facet(field=...)`) promotes to a grid that keeps the prior field on its own axis, its title and sort moving with it. `encode(facet=...)` onto an existing grid raises `ValueError` naming the grid and pointing at `facet_row=`/`facet_col=` or `.facet(...)`; `.facet(field=...)` onto a grid redeclares a wrap, carrying the fields it does not restate and dropping `row_title`/`row_sort`. `encode()` decides the declaration kind from the facet channels stated in that call only.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#32-encoding-channels).

---

### 3.3 Marks

Marks are `Chart.mark_*()` methods that accept visual property overrides as keyword arguments. Primitive marks draw one layer; composite and statistical marks desugar into one or more primitive layers, adding the stat transforms they need.

**Style kwargs: one table per mark.** Which style fields a primitive mark draws is defined once, in Rust (`Mark::style_fields` in `crates/ferrum-core/src/spec/mark.rs`), and published as `ferrum._core.mark_style_fields(mark)` (a dict of field to the extra input it needs to show, `None` when it always shows) over `ferrum._core.style_table_marks()`. Every door that takes mark style derives from that table: the `mark_*()` methods, `fm.Layer(mark_kwargs=)`, `Chart.override(mark_<field>=)`, a figure function's `mark=` and `*_kws`, `layer_kwargs=`, and every composite's catch-all kwargs. A door accepts exactly the fields the table lists for the mark. A composite accepts a field when some mark it emits draws it, and the field lands only on the emitted layers that draw it.

**Refusals.** A style field the mark does not draw is refused at the call with `ferrum._entry.EntryRefusal` (a `ValueError`) naming the mark and its honored set, for example `mark_point: keyword argument 'font_size' is not honored by mark_point. Honored: [...]`. A composite field no emitted mark draws refuses as `mark_boxplot: keyword argument(s) ['cmap'] are drawn by no mark this call draws (['point', 'rect', 'rule', 'tick']), so nothing would take them`. A statistical mark's own named parameters (`bandwidth`, `kernel`, `bin_count`, ...) are parameters of that method, not style fields, so passing one to a different mark is refused. On the wire, `MarkKwargsSpec` refuses an unknown key when parsed, and render refuses a set field the mark never draws (`RenderError::MarkStyleNotDrawn`, voiced by the layer's entry). Every door resolves style through one function, `ferrum.marks.base.resolve_layer_style`: aliases resolve, and an unknown or unhonored kwarg, a constant `size` the mark never draws, or a malformed value refuses naming the public call that received it: `mark_<mark>` for a mark method, `Layer` for `fm.Layer` (`Layer: keyword argument 'font_size' is not honored by mark_point. Honored: [...]`), the figure for `mark=`/`*_kws`, the visualizer class at `fit()`, and `Chart.override` for an override, which raises `FerrumOverrideError` (also a `ValueError`). Under `compare=`, a refusal is prefixed with the model whose panel refused. Render checks every layer's style once, before any panel is drawn.

**`None` is unset.** A style key set to `None` is a key not set: no door that takes the key refuses it, and a door that merges a caller's style over a style already set leaves that value in place (`mark_point(opacity=0.3).override(mark_opacity=None)` draws at 0.3). A key that is not a style of the door refuses whatever its value.

**A figure's `mark=` dict** reaches every layer (or panel) whose mark honors a key and refuses when none does, listing each layer's reasons; a `mark={name: {...}}` style for a sub-layer the call did not draw refuses (`{name: False}` and `{name: {}}` are accepted), and a `mark=` that drops every sub-layer refuses. The metric labels a figure draws are the sub-layers `auc_label`, `ap_label` and `brier_label`.

**Method-only and internal fields.** `orient` is an argument of `mark_bar()` and `mark_tick()` only (a coordinate flip, not a style field): every other door refuses it, and a value outside `['horizontal', 'vertical']` refuses. `zero` is a bar style field taken at every door. `detail`, `clip` and `value_axis` are written only by ferrum itself: the style doors refuse them, naming the call that sets each (`encode(detail=...)`, `mark_function(clip=...)`, a statistical mark's `orient=`).

**Vocabularies.** `stroke_cap` (`butt|round|square`), `stroke_join` (`miter|round|bevel`), `blend` (`normal|alpha|additive`) and `interpolate` (`linear|step|step-before|step-after|monotone|basis|cardinal`) are Rust vocabularies, published as `stroke_cap_accepted_values()`, `stroke_join_accepted_values()`, `blend_accepted_values()` and `interpolate_accepted_values()` on `ferrum._core`. A value outside one refuses at the door, e.g. `mark_line: unknown stroke_cap 'flat'; expected butt|round|square`.

**Inert fields warn.** A drawn field whose need is unmet once scales resolve warns once per layer (`RenderWarning::MarkStyleInert`), e.g. `mark_bar: stroke_width has no effect on this bar mark without a stroke color (stroke=) or a numeric stroke_width encoding (which stroke='none' turns off); it was ignored`. The needs, per mark, are those `mark_style_fields` reports: a stroke color (`stroke=`), a hollow outline (`filled=False`) or a numeric `stroke_width` encoding for a point's stroke fields; a stroke color or border lines (`line=True` or `borders=True`) for an area's stroke fields; a stroke color or a numeric `stroke_width` encoding for a bar's or rect's stroke fields; a stroke color for a polygon's, ribbon's or arc's stroke fields; `arrow=True` for a segment's `arrow_size`; a quantitative color encoding for a polygon's `cmap`; raster data from `mark_raster()` for an image's `cmap`; a `url` encoding for an image's `width`/`height`; a band axis paired with a `y2`/`x2` range for a rect's `band_size`; a `text` encoding for a rect's `font_size`; both `dx` and `dy` for a label's offsets; and a layered chart for a bar's `zero`.

**`blend`.** Every mark takes `blend`, resolved per layer onto the scene batch. `normal` is the default and `alpha` is an older name for it; `additive` adds each mark's color, scaled by its opacity, to what is beneath. Static SVG wraps each additive batch, text and labels included, in its own `mix-blend-mode:screen` group, the closest SVG mode; the interactive view adds mark over mark within the batch and draws blended text and labels in a screen-blended overlay that Save PNG composites the same way. A chart with no `blend` renders as `normal`.

**`stroke_dash`.** One normalizer (`ferrum.marks.base._normalize_stroke_dash`) serves every surface that takes `stroke_dash=`: mark kwargs, `annotate_hline`/`annotate_vline`/`annotate_abline`, `fm.SelectionMark(stroke_dash=)`, and the selection conditional's `stroke_dash` literal (§3.10). It accepts a named linetype (`"solid"`, `"dashed"`, `"dotted"`, `"dashdot"`, `"longdash"`), a comma-separated string (`"4,2"`, spaces allowed), a bare number (one dash length, so `stroke_dash=5` renders `stroke-dasharray="5"`), or a list or tuple of lengths. Each length is a finite, non-negative real number (`int`, `float`, numpy integer and floating scalars, `Decimal`; never `bool`). Anything else (a negative or non-finite length, a dict, set, bytes, an array, `np.bool_`, an unknown name) is refused at construction with a `ValueError` naming the door and the value. Every accepted form reaches the wire as a list of floats. `mark_line(linetype=)` remains an accepted alias of the named forms.

**`interpolate`.** `mark_line`, `mark_area` and `mark_ribbon` evaluate every `interpolate` value through one Rust curve module (`render::marks::curve`): `monotone` is Fritsch–Carlson, `basis` a uniform cubic B-spline, `cardinal` Catmull–Rom with tension 0. A series of two points or fewer always draws as a straight line. A banded or stacked area or ribbon steps both its top and bottom edges under a step value. A curve is parameterized along the series' true independent axis, so a horizontal stack (`Stack(value_axis="x")`) or a chart under `CoordFlip` is monotone along the axis it varies on. An unknown value refuses with `RenderError::UnknownInterpolate`, listing the accepted set, on every mark that takes the kwarg.

**Layered domains include ranges.** When layers share an axis, the shared numeric domain unions each layer's `x2`/`y2` fields as well as its `x`/`y`, so a layer whose range channels extend past its anchors (bands, error bars, a rule diagonal) stays inside the visible domain.

#### Primitive Marks

| Mark | Description | Key Parameters |
|---|---|---|
| `mark_point(...)` | Scatter / dot plot | `size`, `shape`, `filled`, `stroke`, `stroke_width` |
| `mark_line(...)` | Line chart | `interpolate`, `stroke_width`, `stroke_dash`, `stroke_cap`, `stroke_join`, `size` |
| `mark_area(...)` | Area chart | `interpolate`, `line`, `borders`, `opacity` |
| `mark_bar(...)` | Bar chart | `orient`, `corner_radius`, `zero`, `stroke`, `stroke_width` |
| `mark_rect(...)` | Filled rectangle; heatmaps | `corner_radius`, `band_size`, `font_size` |
| `mark_rule(...)` | Horizontal, vertical or diagonal reference line | `stroke_dash`, `stroke_width` |
| `mark_text(...)` | Text labels at the encoded coordinate | `align`, `baseline`, `dx`, `dy`, `font_size`, `font_weight`, `angle`, `limit` |
| `mark_tick(...)` | Rug / tick marks | `band_size`, `orient` |
| `mark_arc(...)` | Arc / pie / donut wedges; requires `CoordPolar`, whose `inner_radius`, `outer_radius` and `pad_angle` shape the ring | `corner_radius` |
| `mark_image(...)` | Image at point, or the raster image `mark_raster()` emits | `width`, `height` (with a `url` encoding), `cmap` (raster data) |
| `mark_geoshape(...)` | Choropleth / geographic shape; the projection is set by `CoordGeo`'s `projection` | `fill`, `stroke`, `stroke_width` |
| `mark_segment(...)` | Line segment from (x, y) to (x2, y2), diagonal-capable. Needs X, Y, X2 and Y2. Used for dumbbell charts, slope graphs, network edges and free-form annotations. | `stroke`, `stroke_width`, `stroke_dash`, `stroke_cap`, `arrow` (bool, default `False`), `arrow_size` (px, default `8.0`) |
| `mark_label(...)` | Point-anchored text placed by greedy collision avoidance, with an optional leader line | `font_size`, `font_weight`, `dx`, `dy`, `leader_line` |

**Tapered lines.** A line layer that encodes `size=` carries per-vertex widths, and static SVG, PNG and PDF export draw it as one filled outline `<path>` (fill is the stroke color with its opacity; `stroke="none"`) whose half-thickness at each vertex is `w/2` along the perpendicular offset from the centerline. Curve families are flattened to a fixed subdivision before outlining. With no `stroke_join=` set, a tapered line joins with bevel; a constant `size=N` or `stroke_width=` keeps a plain stroked line with SVG's miter default. Caps and joins otherwise follow `stroke_cap`/`stroke_join`, baked into the outline geometry. A `stroke_dash` on a tapered line is laid along the centerline's arc length, each on-piece its own filled ring with its widths interpolated at the cut points, so the outline never carries `stroke-width`, `stroke-dasharray` or `stroke-opacity`. Self-overlapping geometry paints its overlap once, at the drawn opacity.

**`mark_segment`.** A chart missing `x2` or `y2` draws an empty layer with a warning naming the missing channel. `arrow=True` draws a filled triangle head at `(x2, y2)`, sized by `arrow_size`, from the same triangle helper (`draw::arrowhead_triangle`) the `annotate_arrow` path uses, so the two are geometrically identical for the same endpoints and size. The head takes the row's own resolved opacity, stroke opacity and fill opacity.

**`mark_rule` shapes.** A rule draws one of six channel shapes: `y` alone (a horizontal span), `x` alone (a vertical span), `x` and `y` with no `x2`/`y2` (a horizontal span at `y`, the documented tie-break of `RuleShape::resolve`), `x`+`y`+`y2` (a ranged vertical segment), `y`+`x`+`x2` (a ranged horizontal segment), or all four of `x`, `y`, `x2`, `y2` numeric (a diagonal segment). Any other combination raises `RenderError::UnsupportedChannelCombination` (a `ValueError`) listing the supported shapes; none falls through to an empty render. The anchor reading follows the resolved scale kind, not the column dtype, so an ordinal scale on a numeric or string column bands, and a null ordinal anchor bands at the null category as `mark_point`/`mark_bar` do.

**`mark_arc` and polar bars.** `mark_arc` raises `ValueError` unless the chart sets `Chart.coord(fm.CoordPolar(...))`. `corner_radius` rounds each wedge corner with an exact tangent circle, clamped per corner to the largest radius the wedge's span and ring thickness allow. The same wedge path draws a `mark_bar` under `CoordPolar`, so `corner_radius` rounds polar bars too. In the interactive view each wedge hit-tests on its true angular span, so a full-circle wedge is hittable at every angle.

**`mark_tick` length.** On an ordinal tick (an ordinal x or y present, whether or not the other axis is declared) `band_size` is the tick's length as a fraction of its category band, default `0.6`. On a rug (one quantitative axis, no categorical channel) a set `band_size` is the tick's length as a fraction of the panel's cross-axis extent (panel height for an x-rug, width for a y-rug); an unset one keeps the fixed rug length (`theme.sizes.tick_size * 2`), not the ordinal `0.6`. A tick whose x resolves to a quantitative scale while y is also declared (the boxplot and errorbar cap idiom) keeps a fixed length and ignores `band_size`.

**`mark_tick` paint.** Ticks draw per-row `color`, `fill`, `stroke_opacity` and `opacity` encodings. A constant `stroke=` wins over an encoded `color=`, while an encoded `opacity=` replaces a constant `opacity=` outright. An encoded `stroke_dash` column has no effect on ticks (unlike `mark_rule`, `mark_line` and `mark_ribbon`); only its legend draws.

#### Composite Marks (expand to multiple primitive layers)

| Mark | Expands To | Key Parameters |
|---|---|---|
| `mark_boxplot(...)` | box + whisker + caps + median + outlier points | `whisker_mult` (IQR multiple, default `1.5`, or `"min-max"`, which also drops the outlier layer), `size` (box width as a band fraction, default `0.6`; alias `width`), `outliers`, `color_field`, `orient` |
| `mark_boxen(...)` | nested letter-value rectangles + outlier points (`LetterValue` transform) | `k_depth` (`"tukey"`\|`"proportion"`\|`"trustworthy"`\|`"full"`, default `"tukey"`, capped at depth 12), `k_proportion` (default `0.007`), `outlier_threshold` (default `0.0`, a multiple of the outermost band's span), `palette` (`str`\|`Sequence[str]`\|`None`) |
| `mark_errorbar(...)` | rule + tick caps | `method` (`"ci"`, `"stderr"`, `"stdev"`, `"iqr"`; default `"ci"`), `ticks` |
| `mark_errorband(...)` | ribbon (+ optional border lines) | `method` (as `mark_errorbar`), `borders` |
| `mark_ribbon(...)` | area between Y and Y2 | `opacity` (default `0.3`), `interpolate` (default `"linear"`) |

**Deprecated `extent=`.** `mark_boxplot(extent=)` is accepted for one release as an alias of `whisker_mult=`, and `mark_errorbar(extent=)` / `mark_errorband(extent=)` as an alias of `method=`, each with a deprecation warning: `extent` is reserved for data-domain bounds.

**`mark_boxen` depth bands and `palette=`.** `LetterValue` names each depth by the Tukey letter-value sequence, `M` (depth 1, the median), `F`, `E`, `D`, `C`, `B`, `A`, `Z`, then `Y`, `X`, `W`, `V` for depths 9 to 12. Depth is capped at 12 for every `k_depth` strategy (Rust `letter_value::K_MAX_NAMED`), and `desugar_boxen` builds one layer per depth from the transform's declared secondary outputs. The median is a band-confined `tick`, not a full-panel rule. Each layer's positional encoding carries the user's own field names as axis titles. The outlier layer draws only rows the transform flags as outliers, against a fence of `[lower - outlier_threshold * span, upper + outlier_threshold * span]` beyond the deepest materialized band; at the default `outlier_threshold=0.0` that fence sits at the band's own outermost values, so every observation strictly beyond them is an outlier and a tie at the fence is not.

With `palette=None` the bands are shaded by an opacity ramp normalized against the depth this chart's data actually reaches (estimated from the bound data, at least 2 bands; the full cap when the data cannot be read), in ascending-depth layer order. A `palette=` paints flat fills at opacity 1.0 anchored on the base band: `colors[0]` goes to depth 2, the innermost real interval, and later colors go outward; depth 1 (zero-pixel by construction) borrows depth 2's color. A named palette is sampled at the 11 colorable depths through `ferrum.color.palette(name, n=11)`, raising the `scheme=` `ValueError` for an unknown name; an explicit color sequence is applied inside-out and cycled. An empty sequence, or a value that is neither a string nor iterable, raises `ValueError`. With a palette the bands paint widest-first so the nesting stays visible. Boxen has no hue channel: `palette` is depth-band coloring, `color_field` only changes the `LetterValue` grouping, and combining `palette=` with a chart-level `.encode(color=...)` raises `ValueError` in either call order. An explicit `fill=`/`opacity=` mark keyword still wins over either shading.

**Category axes are nominal by default.** `mark_boxplot`, `mark_boxen`, `mark_errorbar`, `mark_violin` and `mark_swarm` type their category channel nominal when the caller declares no type, whatever the column dtype, so an integer, float or boolean category column draws band geometry with one tick per category (`ferrum.marks._desugar_helpers.typed_positional_channel`). A declared type always wins: `catplot(native_scale=True)` declares quantitative and stays quantitative. `mark_errorband` keeps an undeclared `x` untyped, since its x is a value axis. `mark_tick` is a primitive and takes no such default: it bands when its channel is declared nominal, and draws a rug when the channel is an undeclared numeric.

**`layer_kwargs=`.** Every composite and statistical mark takes `layer_kwargs={<sub-layer>: {...}}`, keyed by the sub-layer names that call emits (for `mark_smooth`: `line`, `ribbon`, `metrics`; for `mark_boxplot`: `box`, `whisker`, `lower_cap`, `upper_cap`, `median`, `outlier`), the same vocabulary a figure function's `mark={...}` override uses. A bare style kwarg reaches every emitted layer that draws it; a per-layer value outranks it on the layer it names. An unknown sub-layer name raises, listing the chart's sub-layers.

**`mark_ribbon(interpolate=)`.** The full interpolate vocabulary is accepted and draws on both the top and the reversed bottom edge; an unknown value refuses as `mark_ribbon: unknown interpolate ...`.

#### Statistical Marks (trigger stat engine)

| Mark | Stat Computed | Key Parameters |
|---|---|---|
| `mark_smooth(...)` | LOESS or regression fit + CI band | `method` (`"loess"` default, `"lm"`/`"linear"`, `"quadratic"`, `"cubic"`, `"log"`, `"sqrt"`, `"glm"`, `"gam"`), `ci`, `bandwidth` (LOESS span; any other method raises), `degree` (accepted per method; see the method's docstring), `k` (`"gam"` basis dimension; any other method raises), `n`, `groupby`, `layer_kwargs` |
| `mark_density(...)` | KDE | `bandwidth` (`"scott"` default, `"silverman"`, or a float), `kernel` (default `"gaussian"`, vocabulary `transform_registry("kde")["methods"]`), `extent`, `n` (default `512`), `cumulative`, `fill` (bool: area or line, default `True`), `smooth` (bool, bivariate), `multiple` (`"layer"`\|`"stack"`\|`"fill"`\|`"dodge"`, default `"layer"`), `orient` |
| `mark_histogram(...)` | Frequency / density bins | `bins` (`"sturges"`\|`"fd"`\|`"freedman_diaconis"`\|`int` bin count\|`float` bin width; mutually exclusive with `bin_count`/`bin_width`), `bin_count`, `bin_width`, `density`, `cumulative`, `right` (only `False`; `True` raises, bins are `[lo, hi)`), `multiple` (`"layer"`\|`"stack"`\|`"fill"`\|`"dodge"`, default `"layer"`), `position_by`, `orient` |
| `mark_contour(...)` | 2D density contours | `bandwidth` (default `"scott"`), `thresholds` (default `6`, or a list of levels), `smooth` (default `True`), `fill` (default `False`), `scheme` (alias `cmap`), `groupby` |
| `mark_violin(...)` | KDE + optional inner mark | `bandwidth` (default `"scott"`), `inner` (`"box"` default, `"quartile"`, `"point"`, `"none"`; `None` means `"none"`), `shared_extent`, `orient` |
| `mark_qq(...)` | Quantile–quantile | `distribution` (`"normal"` default, `"uniform"`, `"exponential"`, `"lognormal"`), `dequantize`, `line` (default `True`) |
| `mark_raster(...)` | Aggregates data to a pixel grid and renders a colored image | `aggregate` (`"count"` default, `"density"`, `"any"`, `"mean"`, `"sum"`), `field` (required for `"mean"`/`"sum"`; numeric), `scheme` (alias `cmap`), `resolution` (`"screen"` default, `int`, or `(nx, ny)`), `blend`, `min_count`, `log_scale` |
| `mark_swarm(...)` | Beeswarm for categorical data: points spread along the category axis to avoid overplotting while preserving the value distribution. General-purpose; distinct from `mark_shap_beeswarm`. | `size` (dot diameter in px, default `4`), `spacing` (edge gap in px, default `1.0`), `orient`, `side` (`"both"`\|`"left"`\|`"right"`), `dodge` (a column name) |
| `mark_hex(...)` | Hexagonal binning: each hex colored by its aggregate. Preferred over `mark_raster` when topology and shape perception matter. | `bin_size` (hex radius in data units, `None` chooses one), `aggregate` (default `"count"`, vocabulary `transform_registry("hex")["methods"]`), `field`, `scheme` (alias `cmap`), `stroke`, `stroke_width` (default `0`) |
| `mark_function(fn, ...)` | Plot a Python callable f(x) → y as a line. `fn` receives a numpy array of x values and returns a numpy array of y values. | `domain` (`[x_min, x_max]`), `n` (evaluation points, default `200`), `clip` (default `True`) |

**Aggregates that need a field.** `mark_raster` and `mark_hex` validate `aggregate` and its field requirement at the call, against `transform_registry("raster")` / `transform_registry("hex")` (`methods`, `methods_needing_field`). An unknown aggregate raises `ValueError` listing the vocabulary; an aggregate that needs a field and has none raises naming the field-free alternatives. `mark_hex` aggregates `mean`, `sum`, `count`, `min`, `max`, `median`, `variance`/`var` and `stdev`/`std`; an `.encode(color=...)` naming a column supplies the field when it is omitted. A raster `field` must be numeric: integer columns are coerced to `Float64`, a non-numeric column raises at render.

**`mark_raster(resolution="screen")`.** The grid size follows a pre-layout estimate of the panel (the viewport minus the theme's padding, divided by the facet grid's columns and rows), so a larger chart gets a larger grid. The estimate excludes axis titles, tick labels, legends and strip bands, so it approximates rather than equals the drawn plot area. An explicit `int` or `(nx, ny)` ignores the estimate. A direct `apply()` of the `Raster` transform outside the render pipeline falls back to 256×256. `blend` is the shared style field: `"additive"` blends in both static SVG and the interactive view.

**`mark_smooth` method knobs.** `bandwidth` is the LOESS span and `k` is the GAM basis dimension; each raises `ValueError` on any other method, naming the knob that method has. `degree` is accepted per method: any `degree >= 1` on `"lm"` (`>= 2` promotes to a polynomial fit), only the fixed degree on `"quadratic"`/`"cubic"`, `1` or `2` on `"loess"`, and none on `"log"`, `"sqrt"`, `"glm"` or `"gam"`.

**`mark_density` modes.** With only `x` encoded, `mark_density` is a 1-D KDE drawn as an area (`fill=True`) or a line. `multiple="stack"`/`"fill"` stack the group curves on one shared grid; `"dodge"` draws them overlapping, like `"layer"`. With both `x` and `y` encoded it switches to bivariate KDE and renders filled contours, equivalent to `mark_contour(fill=True)`; `multiple` applies to univariate mode only.

**`mark_contour` levels.** Under `smooth=True` a 3×3 Gaussian pass (center weight 4, edges 2, corners 1, over 16) runs before level computation, so the levels and the contoured surface are the same grid. Fill mode draws `thresholds` bands: the interior bands between consecutive levels plus a top band from the highest level to the surface maximum, so the density peak is always filled and `thresholds=1` draws one band. Each band's `level_value` is its lower bound. With `groupby` set, one surface is computed per group and drawn in isoline mode, ignoring `fill` and `scheme`. Bivariate `mark_density` shares this path.

**`mark_violin(inner=)`.** `"box"` draws an IQR box and median; `"quartile"` draws q1, median and q3 as ticks (`band_size=0.8` of the category band, q1/q3 dashed, median solid) confined to the violin body; `"point"` draws the data points. Under a hue split (`color=` distinct from the category axis) no inner variant dodges per hue: the inner mark sits at the category center, and any `inner` other than `"none"` warns each time the chart resolves (every `to_json()` and render), naming the door called. `catplot(kind="violin", hue=...)` voices that warning as `catplot(kind='violin')` and points to `Chart.mark_violin(inner="none")`, since `catplot` has no `inner=` of its own. An unknown `inner` raises listing `['box', 'none', 'point', 'quartile']`.

**`mark_function`.** `fn` is evaluated over `domain`, or over the resolved x-domain of the chart's other data layers when `domain` is omitted; with neither, it raises `ValueError`. `clip=True` confines the line to the panel like every mark; `clip=False` draws the batch outside the panel's clip in both SVG and the interactive view, so the curve extends past the panel edge. `clip` changes rendering only, never the evaluated domain; a non-boolean `clip` is refused.

**`mark_swarm` placement.** `size` is the dot diameter in pixels (unlike `mark_point(size=)`, an area in px²) and `spacing` the gap between dot edges; a non-positive, non-finite, `bool` or non-numeric `size` is refused when the chart resolves. Placement happens at render time against the panel's resolved value-axis scale, so adjacent placed dot centers are exactly `size + spacing` px apart under titles, facets, explicit domains, log scales and dodge bands; a direct `apply()` of the `Swarm` transform returns placeholder offsets. The swarm's point layer draws at area `π·(size/2)²` on every swarm route (`mark_swarm`, `catplot(kind="swarm")`, the boxplot-plus-swarm composite), and its own size outranks `Theme(point_size=)`. An encoded `size=` channel or a point layer's own `size` still outranks it for drawing, and the swarm then packs at the largest diameter any point layer it feeds can draw: the size range's maximum for a size encoding (widened to the null-size fallback when a null or unmapped size row is wider), `2·sqrt(a/π)` for a constant `size=a`, else the swarm's own `size`. The bound is scale-derived, so it matches the size legend's largest circle and every facet packs at the same diameter (`crates/ferrum-core/src/render/position.rs`).

**`fm.Swarm` transform.** `fm.Swarm` passes its input columns through unchanged (minus rows with a null `dodge` value) and places dots at render, packing whichever rows the layer draws. `fm.Swarm(point_size=)` and `mark_swarm(size=)` are diameters in pixels, both defaulting to `4`; a `Swarm`-fed point layer with no size draws at that diameter. A swarm whose dots reach past half its category slot warns once.

**`mark_hex` color.** `mark_hex` colors its cells by the per-cell aggregate. A `color=` naming the aggregated column (or Hex's `value` output) supplies `field` and styles the cells; any other `color=` raises `ValueError` (`mark_hex: color='cat' is drawn nowhere; ...`), and a `scheme=` beside a different `cmap=` raises.

**Auto-raster (`mark_raster` as a policy).** `mark_raster` is the explicit form of density rasterization. Auto-raster substitutes it for the chart's mark when the chart's row count reaches `RenderConfig(raster_threshold=)` (default `500_000`; `None` disables it; `raster=False` on `.show()`/`.save()` disables it per call). It applies only to a single-mark `point`, `bar`, `rect`, `tick`, `rule` or `segment` chart (never a composite or statistical mark) whose `x` and `y` are both explicitly annotated quantitative. An over-threshold chart that is not eligible renders as vector and warns. `raster_behavior` sets what happens on substitution: `"warn"` (default) substitutes and warns, `"silent"` substitutes quietly, `"error"` raises `ValueError` instead (`ferrum._render.Chart._apply_auto_raster`).

**Auto-raster and color.** The color channel's type is its annotation when it has one and the column dtype otherwise, so `color="z"` on a numeric column is quantitative. A quantitative color substitutes `mark_raster(aggregate="mean", field=<the color field>)`, colored by the color channel's own `scheme=` if declared, else `raster_scheme`, and the original color channel is dropped. A categorical, ordinal or temporal color, or one whose type cannot be determined (a missing column, a non-polars source), blocks substitution permanently, since no pixel aggregate preserves a category; the warning names the field, the reason and the manual `mark_raster(aggregate=..., field=...)` route. An explicit `RenderConfig(raster_aggregate=..., raster_field=...)` that differs from the defaults (`"count"`, `None`) wins over the color-driven aggregate.

**Color channel typing.** A color scale is continuous when `Color(type=)` is quantitative or temporal, or when no type is given and the column dtype is numeric (any integer, unsigned or float width) or temporal (date, timestamp); every other case is a categorical color scale. A quantitative or temporal type declared on a non-numeric column raises `ValueError` (`encoding 'color' expected numeric column for quantitative/temporal type, got Utf8`) rather than inferring silently.

#### Model Diagnostic Marks

A model diagnostic mark renders the long-form frame that a `ModelSource` method (or the precomputed `y_true=`/`y_pred=` path) emits, and desugars into primitive layers. It computes nothing about the model itself: binning, sweeping, normalizing and ranking happen upstream, in the source method or the figure function that calls the mark. Every mark below also takes `position=None` and `**mark_kwargs`, the mark-style fields of its primary layer. A keyword the mark does not honor is refused at render with `EntryRefusal` (a `ValueError`) naming the keyword and listing the honored fields. The signatures are the shipped ones, defaults included.

| Mark | Input frame (source method) | Parameters |
|---|---|---|
| `mark_residuals(*, kind="studentized", reference_line=True, cook_threshold=None, color_field=None)` | `y_true`, `y_pred`, `residual`, `studentized_residual`, `cooks_distance` (`predictions`) | `kind`: `"studentized"` (alias `"scaled"`) plots `studentized_residual`; any other value plots `residual` (`"raw"` is the documented spelling; the value is not validated). `reference_line`: a rule at y = 0. `cook_threshold`: a float, or `"auto"` for the `4 / n` rule; points above it are overlaid in red with a black outline. `None` disables it. |
| `mark_prediction_error(*, reference_line=True, ci=None, reference_band=False, color_field=None)` | `y_true`, `y_pred` (`predictions`) | `reference_line`: the y = x identity line. `ci`: confidence level of a prediction-interval band, `None` for none. `reference_band`: a shaded band around the identity line. |
| `mark_confusion(*, normalize=None, annotate=True, color_field="value", cmap=None)` | `actual`, `predicted`, `value`, `value_fmt` (`confusion_matrix`) | Ordinal heatmap plus a text layer reading `value_fmt` when `annotate=True`. `cmap=None` uses the active theme's sequential scheme. `normalize` is informational at this layer (see below). |
| `mark_roc(*, average=None, reference_line=True, annotate_auc=False, color_field="class")` | `fpr`, `tpr`, `threshold`, `class`, `auc` (`roc_curve`) | `average` keeps only the matching `class` row (`"macro"`, `"micro"`, ...); see the average filter below. `reference_line`: the chance diagonal. `annotate_auc`: the per-curve AUC label. `color_field` is a column name or a channel such as `fm.Color("class", title="Class")`. |
| `mark_pr(*, average=None, annotate_ap=False, iso_lines=False, color_field="class")` | `precision`, `recall`, `threshold`, `class`, `ap` (`pr_curve`) | `average` and `color_field` as for `mark_roc`. `annotate_ap`: the per-curve AP label. `iso_lines`: labelled iso-F1 reference curves behind the data. |
| `mark_calibration(*, reference_line=True, color_field=None)` | `mean_predicted`, `fraction_positive`, `count` (`calibration_curve`) | `reference_line`: the perfect-calibration diagonal. No binning parameters: `n_bins` and `strategy` belong to `ModelSource.calibration_curve` and `calibration_chart`. |
| `mark_gain(*, reference_line=True, color_field="class")` | `percent_population`, `gain`, `class` (`cumulative_gain`) | `reference_line=False` drops the `class="baseline"` rows. |
| `mark_lift(*, reference_line=True, color_field="class")` | `percent_population`, `lift`, `class` (`lift_curve`) | `reference_line=False` drops the `class="baseline"` (lift = 1) rows. |
| `mark_importance(*, orient="horizontal", error_bars=True, top_k=None, color_field=None)` | `feature`, `importance`, `std` (`importances`) | `orient`: `"horizontal"` or `"vertical"`. `error_bars=True` draws an errorbar layer from `imp_lower`/`imp_upper` columns, which `importance_chart` computes; a frame without them raises at render (pass `error_bars=False`). `top_k` keeps the `top_k` largest importances. `color_field` colours the bars, never the error bars. |
| `mark_shap_beeswarm(*, max_display=20, color_bar=True, order="abs_mean", zero_line=True)` | long-form SHAP frame (`shap_values`) | `order`: `"abs_mean"` (descending mean absolute SHAP), `"max"` (descending maximum absolute SHAP), `"mean"` (descending signed mean) or `"none"`; anything else raises `ValueError`. `color_bar=True` draws the feature-value colour bar with `"Low"`/`"High"` tick labels; `False` suppresses it. `zero_line`: a dashed rule at SHAP = 0. |
| `mark_shap_bar(*, max_display=20, orient="horizontal", color_field=None)` | `feature`, `abs_mean_shap` | `max_display` keeps the top features by descending `abs_mean_shap`. |
| `mark_shap_waterfall(*, sample_idx=-1, max_display=20)` | `feature`, `x0`, `x1`, `shap_sign` | `sample_idx` must be passed explicitly: the default `-1` raises `ValueError` at the call. `max_display` keeps the features with the largest `abs(x1 - x0)`. |
| `mark_pdp(*, kind="average", ice_alpha=0.2, center=False, color_field="feature")` | `feature`, `feature_value`, `pd_value`, `class_label`, `sample_id` (`partial_dependence`) | `kind`: `"average"`, `"individual"` (ICE lines) or `"both"`. `ice_alpha`: ICE line opacity. `center` is informational at this layer (see below). |
| `mark_silhouette(*, zero_line=True, color_field="cluster")` | `sample_id`, `y_position`, `cluster`, `silhouette_value` (`silhouette`) | One bar per sample, stacked per cluster in descending coefficient order. `zero_line`: a rule at silhouette = 0. |
| `mark_learning_curve(*, ci_style="band", color_field="split")` | `learning_curve` frame | `ci_style`: `"band"` (shaded ribbon) or `"errorbar"`. |
| `mark_validation_curve(*, log_scale=False, ci_style="band", color_field="split", param_label=None)` | `validation_curve` frame | `log_scale`: log x axis. `ci_style` as above. `param_label`: the x-axis title. |
| `mark_decision_boundary(*, proba=False, color_field="z")` | `x`, `x2`, `y`, `y2`, `z` | Colours the grid by `z`. `proba` is informational at this layer (see below). |
| `mark_discrimination_threshold(*, metrics=("precision", "recall", "f1", "queue_rate"), n_thresholds=50, threshold_line, optimum_label)` | long-form `threshold`, `metric`, `value` (melted from `discrimination_threshold`) | `metrics`: the lines drawn. `threshold_line` (effective default `False`): a rule at the F1-optimal threshold. `optimum_label` (effective default `True`): the text `"max F1 = 0.xxx @ t=0.xx"` at the optimum. `n_thresholds` is informational at this layer (see below). |
| `mark_parallel_coordinates(*, alpha=0.5, color_field=None)` | `feature`, `value`, `sample_id` | One polyline per `sample_id` (routed through `detail`). `alpha`: line opacity. |
| `mark_class_prediction_error(*, normalize=False, color_field="predicted", show_counts=True)` | `actual`, `predicted`, `value` (`confusion_matrix`) | One stacked bar per actual class, segments coloured by predicted class. `normalize=True` scales each bar to 100%. `show_counts`: per-segment count text, skipping empty segments. |
| `mark_pca_scree(*, cumulative_line=True, threshold_line=None)` | `component`, `explained_variance_ratio`, `cumulative_variance_ratio` (`pca_variance`) | `cumulative_line`: the cumulative-variance line, drawn as the first layer so its wider range drives the y scale. `threshold_line`: a horizontal rule at that cumulative level (e.g. `0.95`). The component count belongs to `pca_variance` and `pca_scree_chart`. |
| `mark_rank1d(*, orient="horizontal", color_field=None)` | `feature`, `score`, `rank` (`rank1d`) | `orient`: `"horizontal"` or `"vertical"`. The ranking algorithm belongs to `ModelSource.rank1d`. |
| `mark_rank2d(*, annot=True, color_field="correlation", text_field="correlation_fmt", cmap=None)` | `feature_x`, `feature_y`, `correlation` (`rank2d`) | `annot`: per-cell value text read from `text_field`. `cmap=None` uses the active theme's diverging scheme. The algorithm (`"pearson"`, `"spearman"`, `"kendall"`, `"covariance"`) belongs to `ModelSource.rank2d`, which computes all four in Rust. |
| `mark_intercluster_distance(*, label_clusters=True, color_field="cluster")` | `cluster`, `x`, `y`, `size` (`intercluster_distance`) | One point per cluster centre, area encoding membership through the `size` channel. `label_clusters`: cluster-id text. |
| `mark_cv_scores(*, kind="box", split="both", color_field=None)` | `cv_scores` frame | `kind`: `"box"`, `"bar"` or `"strip"`. `split`: `"train"`, `"test"` or `"both"`. A distinct `color_field` groups box statistics by `(split, color_field)` and drops the strip jitter so a `position=Dodge(...)` can offset points. |
| `mark_alpha_selection(*, log_scale=True, highlight_best=True)` | `alpha`, `fold`, `score`, `mean_score`, `std_score` (`alpha_selection`) | A single mean-score line, one point per alpha, with no CI band and no `ci_style`. `log_scale`: log x axis. `highlight_best`: a vertical rule at the alpha maximizing `mean_score`. |

**Parameters that live upstream.** A mark receives data already shaped by its source method, so a parameter that changes that shaping is not a mark parameter. `mark_calibration` takes no `n_bins`/`strategy`, `mark_pca_scree` no `n_components` and `mark_alpha_selection` no `ci_style`; passing one is refused as an unknown keyword. Four mark parameters are kept but informational, registered in `ferrum.marks._informational_kwargs.INFORMATIONAL_KWARGS`: `mark_confusion(normalize=)`, `mark_pdp(center=)`, `mark_decision_boundary(proba=)` and `mark_discrimination_threshold(n_thresholds=)`. A non-default value emits a one-time `UserWarning` naming the no-op and the real home (`ModelSource.confusion_matrix`/`confusion_matrix_chart`, `pdp_chart`, `decision_boundary_chart`, `ModelSource.discrimination_threshold`/`discrimination_threshold_chart`), and never changes the render. The figure functions do not forward these values to the mark.

**Data preparation runs on every input type.** The frame-level preparation a mark applies (the `average` filter, `split`, `reference_line` sorting and baseline filtering, `metrics`, `order`, `top_k`) first coerces the chart's data through `ferrum._coerce.to_polars`, so a pandas or pyarrow input is prepared the same way as a polars one.

**The `average` filter** (`mark_roc`, `mark_pr`, `ferrum.marks._desugar_helpers._filter_class_average`) keeps the rows whose `class` value equals `average` exactly. When nothing matches, a frame with a single `class` value (a binary curve, which carries no average row) is left unfiltered, and a multiclass frame warns once and renders every class.

**Metric annotations.** `mark_roc(annotate_auc=True)` and `mark_pr(annotate_ap=True)` place one label per curve at its endpoint, reading `"{group}: AUC = 0.987"` (or `AP`). The value comes from the frame's `auc`/`ap` column when present; a present-but-NaN value renders as `nan` rather than being re-derived. Without the column the value is integrated from the drawn curve after a stable sort, so tied x values cannot change it. The raw marks default to no label; the figure functions `roc_chart`, `pr_chart` and `calibration_chart` default `annotate_auc`/`annotate_ap`/`annotate_brier` to `True`, and own their overlay.

In the figure functions the flag drives exactly one surface, the per-curve label. Titles carry no metric: `roc_chart` is titled `"ROC Curve"`, `pr_chart` `"Precision–Recall Curve"` and `calibration_chart` `"Calibration Curve"` in every shape, and a legend entry is the bare class or model name. So `annotate_*=False` removes the number from the figure. Every metric string is composed by `ferrum._metric_labels._render_metric` with the kind's display name and `.3f` format from `_METRIC_LABEL_SPECS`. Adding `AUCLabel()`, `APLabel()` or `BrierLabel()` to a chart that already has that label replaces it, so a re-application changes `position=`, `format=` or `prefix=`. `BrierLabel` on a frame without a `brier` column raises `ValueError` naming the column: binned reliability points do not define a Brier score. `AUCLabel`/`APLabel` integrate the drawn curve when no metric column exists.

**Per-mark rendering details.**
- Studentized residuals use the leverage-aware hat matrix (`ferrum._core.hat_matrix_stats`) when the estimator exposes `coef_`; otherwise they are internally studentized (`ferrum._core.studentized_residual_no_x`), and `cooks_distance` is NaN.
- `decision_boundary_chart` builds a two-feature grid and fixes any other feature at its column mean.
- The SHAP frames come from `ModelSource.shap_values`, which needs the `shap` library (`pip install ferrum[shap]`).
- `intercluster_distance_chart` pads the x and y domains by 15% (at least 0.5) so large bubbles do not clip.
- `mark_discrimination_threshold`'s rule and label sit at the F1 optimum, so they need `"f1"` in `metrics=`. Without it, an explicit `threshold_line=True` or `optimum_label=True` raises `ValueError` naming the flag and `metrics=`, while the unset defaults quietly draw neither. `discrimination_threshold_chart` and `DiscriminationThresholdVisualizer` forward `optimum_label` unset, so the same split holds through them.

**Classification target contract.** One contract governs how the target `y` meets the model, on the seven `ModelSource` classification methods (`roc_curve`, `pr_curve`, `calibration_curve`, `cumulative_gain`, `lift_curve`, `discrimination_threshold`, `confusion_matrix`), on the precomputed `y_true=`/`y_pred=` path of `roc_chart`, `pr_chart`, `calibration_chart`, `gain_chart`, `lift_chart`, `discrimination_threshold_chart`, `confusion_matrix_chart` and `classification_report_chart`, and on every figure function and visualizer built on them (`class_prediction_error_chart` included). `class_balance_chart` is outside it: it counts `y`'s own values and resolves no classes.

- *Binarization.* Every binary curve compares `y` against the positive class before the Rust kernels see it, so a target's labels do not change its metrics: `{-1, 1}`, `{1, 2}` and string labels give the same numbers as `{0, 1}` for the same probabilities.
- *Positive-class resolution* (`_resolve_class_values`) always yields values from `y`'s own domain, in three tiers: the model's `classes_` coerced into `y`'s dtype; else `y`'s distinct values, ascending, when their count equals the score-column count; else the score-column labels parsed into `y`'s dtype. On the precomputed path the binary pair is read from `y_true` in its own dtype, ascending, the larger value positive, and a single-class `y_true` labels its curve with the class present.
- *Nulls refuse.* A target with any null row (`NaN`, `None`, `NaT`, in every dtype family) raises `ValueError` naming the entry, the null count out of the row count and `y`'s observed values. A null prediction refuses the same way on both `confusion_matrix` implementations and on both `classification_report_chart` paths, rather than becoming a class of its own. A pandas target holding `pd.NA` is refused earlier, by ingestion, with an untyped `TypeError`.
- *Stray values warn.* Rows holding a value outside an otherwise matching class set emit one `UserWarning` per call naming the entry, the row count and the values. A curve counts those rows as negative and says so; the confusion matrix (`confusion_matrix`, `confusion_matrix_chart`, `class_prediction_error_chart`) drops them from every cell and says so.
- *Total mismatch refuses.* A binary target with no row equal to the positive class raises `ValueError` naming the class and `y`'s values (gain and lift included), with no extra stray warning ahead of it. A multiclass resolution where no resolved class matches any row refuses. When the model's `classes_`, translated into `y`'s domain, match none of `y`'s values, `confusion_matrix_chart`, `class_prediction_error_chart`, `classification_report_chart` and `ClassificationReportVisualizer` refuse; a partial overlap renders. The check tests the class vocabulary, not the predictions a model happened to make. A stray value on a `classes_`-less binary model refuses, since the inflated cardinality leaves only parsed labels that match no row.
- *Tolerated.* Multiclass one-vs-rest tolerates some absent classes, and `lift_curve` skips a zero-base-rate class. An empty target is not an error; an all-positive target yields the kernels' NaN. The positive class is validated once against the whole request: `discrimination_threshold(cv=...)` sweeps each fold from that validated indicator, so a held-out fold with no positives contributes `precision = recall = f1 = 0` with its real `queue_rate`.
- *Predictions in `y`'s domain.* When the model exposes `classes_`, `model.predict`'s output is translated into `y`'s domain through the positional `classes_` to `y`-values correspondence before the confusion matrix or the classification report is computed, so an int-coded `classes_` against a string, `Categorical` or `Enum` `y` scores correctly. On the classification report's model path, a model without `classes_` hands its raw predictions through, so a non-numeric `y` against numeric predictions surfaces scikit-learn's own `ValueError: Mix of label input types`.
- *One class axis.* `classification_report_chart` (both paths), the precomputed `confusion_matrix` and the `classes_`-less branch of `ModelSource.confusion_matrix` derive the class axis from `ferrum.diagnostics._internal._target_hygiene.report_labels`: the union of `y_true` and `y_pred` in natural ascending order (so `{1, 2, 10}` orders `1, 2, 10`). A null in either array refuses naming the array; a mix of ordering families refuses as a mix of label types (`int` with `float`, or `bool` with `int`, is one family); a value that does not order (`complex`) or cannot be hashed (`list`, `set`, `bytearray`, `dict`) refuses naming the reason; otherwise labels are deduplicated by hash equality. `classification_report_chart` requires 1-D `y_true` and `y_pred`, refusing a 2-D array with the array's name and shape.
- *The refusal names your entry.* Every classification method takes `diagnostic=` (default: the method's own name) and each public entry passes its own, so a message opens with what the caller invoked: `roc_chart:`, `ROCVisualizer:`, `roc_curve:` for a direct source call, and `confusion_matrix_chart:` or `class_prediction_error_chart:` for the shared confusion frame. On the model path, a refusal's remedy clause names the model rather than an array the caller never passed. The name labels messages only and is not part of the cache key.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#33-marks).

---

### 3.4 Stat Transforms

Stat transforms run in the Rust engine. Each receives the layer's column-oriented Arrow table and returns a new table, plus, for some, named secondary tables. Every stat transform is a `ferrum._core` class re-exported on `ferrum`.

**Applying a stat transform.** A statistical or model diagnostic mark (§3.3) adds its own stat transform, so most charts never name one. To run one directly, pass an instance to `Chart.transform(...)`, which appends it to the chart's one transform pipeline, run in declaration order before any mark's own stat. A `Layer` does not run transforms of its own: `Layer(transforms=[...])` raises `ValueError` and names the route, a separate chart with `.transform(...)` composed with `+`, which routes the transform to that layer alone.

**Stat transform classes.** Every class except `AggregateOp` also takes `name=None`, which labels the transform's output (and prefixes its secondary-table keys) so a sibling transform such as `Reorder(from_=...)` can read it.

| Class | Output | Meaning |
|---|---|---|
| `Bin(field, *, bins=None, bin_count=None, bin_width=None, extent=None, nice=True, cumulative=False, shared_extent=False, groupby=None, agg_field=None, agg=None)` | `bin_start`, `bin_end`, `count`, `density`, `agg_value` | 1D binning. `cumulative=True` appends a `cumulative` running-count column (the ECDF form). `agg_field`/`agg` aggregate a second column per bin. |
| `Bin2D(x, y, *, bins_x=None, bins_y=None, extent_x=None, extent_y=None, cumulative=False)` | `x_lo`, `x_hi`, `y_lo`, `y_hi`, `count` | Rectangular 2D binning over `(x, y)`. |
| `Kde(field, *, bandwidth=None, bw_adjust=1.0, n=512, extent=None, cumulative=False, shared_extent=False, kernel="gaussian", groupby=None, count_scale=None)` | `value`, `density` | 1D kernel density estimate on an `n`-point grid. |
| `Kde2D(x, y, *, bandwidth=None, n=128, extent=None, groupby=None)` | `grid_x`, `grid_y`, `density` (list columns), `nx`, `ny`, `extent` | 2D kernel density estimate: one row (one per group when grouped) carrying the whole grid. |
| `Contour(*, thresholds=6, fill=False, smooth=True)` | isolines: `level_id`, `level_value`, `contour_x`, `contour_y`, `contour_x2`, `contour_y2` (one row per segment) | Marching-squares isolines over a `Kde2D`-shaped grid; `fill=True` emits isoband polygons instead, one row per vertex. |
| `Hex(x, y, *, bin_size=None, aggregate="count", field=None)` | `hex_x`, `hex_y`, `hex_id`, `value` | Hexagonal binning; `mark_hex` adds it implicitly. |
| `Raster(x, y, *, aggregate="count", field=None, resolution=None, min_count=None, log_scale=False)` | `x_min`, `x_max`, `y_min`, `y_max`, `width`, `height`, `pixel_data` | Rasterized 2D aggregation for dense scatter (`mark_raster`). |
| `Smooth(x, y, *, method="loess", ci=0.95, bandwidth=None, degree=None, k=None, n=200, seed=0, x_bins=None, x_estimator=None, output="fitted", inject_zero_ref=False, inject_metrics=False, groupby=None, x_range=None)` | `x`, `y`, `ci_lower`, `ci_upper` | Fitted curve with a confidence band; see Smoothing. |
| `Logistic(x, y, *, n_grid=100, ci=None, max_iter=25, tol=1e-08)` | `x`, `y`, `ci_lower`, `ci_upper` | Binary logistic regression by IRLS, with a Wald interval. |
| `Glm(x, y, *, family="gaussian", link=None, n_grid=100, ci=None, max_iter=25, tol=1e-08)` | `x`, `y`, `ci_lower`, `ci_upper` | Generalized linear model by IRLS; `link=None` takes the family's canonical link (table below). |
| `Robust(x, y, *, n_grid=100, ci=None, huber_c=1.345, max_iter=25, tol=1e-08, output="fitted", inject_zero_ref=False, inject_metrics=False, groupby=None, x_range=None)` | `x`, `y`, `ci_lower`, `ci_upper` | Huber M-estimator regression with a sandwich interval. With `groupby`, one refit per group. |
| `Summary(field, *, groupby=None, error_fn="ci", ci=0.95, n_boot=1000, seed=0)` | `mean`, `lower`, `upper` | Mean with an uncertainty interval; `error_fn` is `"ci"` (seeded bootstrap), `"stderr"` or `"stdev"`. |
| `ErrorExtent(field, *, method="ci", seed=0, n_boot=1000)` | `mean`, `lower`, `upper` | Error-bar extent; `method` is `ci`, `stderr`, `stdev` or `iqr`. Also takes `groupby` (default ungrouped). |
| `Aggregate(ops, *, groupby=None)` | one column per op | Group-by aggregation. `ops` is a list of `AggregateOp(field, fn_, as_)`. |
| `BoxStats(field, *, whisker_extent=1.5)` | `q1`, `median`, `q3`, `lower_whisker`, `upper_whisker` | Box-plot statistics. Also takes `groupby` (default ungrouped). |
| `Outliers(field, *, extent=1.5)` | the outlying input rows | Rows beyond `extent` IQRs of the quartiles. Also takes `groupby` (default ungrouped). |
| `Violin(field, *, bandwidth=None, bw_adjust=1.0, n=256, width=0.4, shared_extent=False, horizontal=False)` | `group_id`, `violin_x`, `violin_y` | Violin outlines from a per-group KDE. Also takes `groupby` (default ungrouped). |
| `LetterValue(value, *, group=None, k_depth="proportion", k_proportion=0.007, outlier_threshold=0.0, sort=None)` | `group`, `depth`, `lower`, `upper`, `level`; secondary `outliers` and one `depth_N` table per depth | Letter-value (boxen) statistics. `k_depth` is `"tukey"`, `"proportion"`, `"trustworthy"` or `"full"`. `outlier_threshold` fences at `[lower - t*span, upper + t*span]` of the group's outermost band; at the default `0.0` that is the outermost letter values themselves, so every observation strictly beyond them is an outlier and a tie at the fence is not. |
| `QQ(field, *, distribution="normal", dequantize=False, emit_line=True, seed=0)` | `theoretical`, `sample`; secondary `qq_line` | Quantile-quantile pairs against `normal`, `uniform`, `exponential` or `lognormal`, with a reference-line table when `emit_line=True`. |
| `Swarm(category, value, *, point_size=4.0, spacing=1.0, side="both", orient="vertical", dodge=None)` | the input rows, unchanged | Beeswarm placement (`mark_swarm`): the transform validates the columns, and the dot offsets are packed in pixel space at render time against the resolved scale. |
| `Linkage(*, method="ward", metric="euclidean", axis="rows", z_score=None, standard_scale=None)` | `node_id`, `left`, `right`, `distance`, `n_obs`; secondary `linkage`, `order`, `coords`, `segments` | Hierarchical clustering. `method` is one of `single`, `complete`, `average`, `weighted`, `centroid`, `median`, `ward`. |
| `Reorder(by, *, drop_index=True, from_=None, axis="rows")` | the input rows, permuted | Permutes rows (or columns) by an index column, typically a `Linkage` `order` table named through `from_`. |
| `Unpivot(*, value_vars=None, var_name="variable", value_name="value")` | long-form rows | Wide to long reshape; also takes `id_vars` (default empty; an unset `value_vars` melts every column not in `id_vars`). Value columns must share a dtype or all be numeric. |
| `ReferenceLine(x_field, y_field, *, x=(0.0, 1.0), y=(0.0, 1.0))` | two rows, `x_field` and `y_field` | The two endpoints of a fixed reference line, for a layer that reads it by name (as `calibration_chart`'s diagonal does). |

**Binning.** `Bin(bins=)` takes `"sturges"`, `"fd"` (Freedman-Diaconis), a positive `int` (a bin count), a positive finite `float` (a bin width) or `None`. It is mutually exclusive with `bin_count=` and `bin_width=`; naming both raises `ValueError` (`Bin: bins= is mutually exclusive with bin_count=/bin_width=; pass only one`). An unknown rule name raises `Bin: unknown bins value '<v>'; expected 'sturges'|'fd'|int|float`.

**Density bandwidth.** `Kde`, `Kde2D` and `Violin` take `bandwidth=None`, `"scott"`, `"silverman"` or a positive finite float. `None` means Scott's rule (`σ · n^(-1/5)`, with the sample standard deviation); `"silverman"` is `0.9 · min(σ, IQR/1.34) · n^(-1/5)`; a float is a fixed bandwidth. On `Kde` and `Violin`, `bw_adjust` multiplies the chosen bandwidth. Any other string raises (`Kde: unknown bandwidth 'foo'; expected 'scott' | 'silverman' | float`). The kernel vocabulary, shared by `Kde` and `transform_density`, is `gaussian`, `epanechnikov` (alias `epan`), `tophat`, `uniform`, `cosine`, `exponential` and `linear`.

**Smoothing.** `Smooth(method=)` accepts `lm` (alias `linear`), `loess`, `quadratic`, `cubic`, `log`, `sqrt`, `glm` and `gam`; another value raises at construction.
- `lm` fits a straight line. With `degree>=2` it fits a polynomial of that degree; `degree=0` raises. `quadratic` and `cubic` fit a fixed degree 2 or 3: an agreeing `degree=` is accepted, a conflicting one raises and names `method="lm", degree=d` as the way to get degree `d`.
- `loess` takes `bandwidth` (the span, a finite value in `(0, 1]`, default `0.75`) and `degree` 1 or 2 (default 2).
- `glm` fits a Gaussian/identity GLM by IRLS.
- `gam` fits a penalized cubic B-spline: basis dimension `k` (default 10, clamped to `[4, distinct x count]`, `k<4` raises), second-order difference penalty, smoothing parameter chosen by GCV over a fixed 33-point grid, and a Bayesian posterior band. It uses no randomness, so `seed=` does not affect it, and outside the observed x range it holds its boundary value.
- `degree=` on `log`, `sqrt`, `glm` or `gam` raises. An explicit `bandwidth=` on any method but `loess` raises unless it equals the `0.75` default. `k=` on any method but `gam` raises.
- A whole-frame polynomial fit whose degree is at least its usable point count raises. A fit the data cannot support otherwise (a `gam` with fewer than 4 distinct x values, a zero-variance x) degrades to a NaN range and warns, as does a degenerate group of a grouped fit or a degenerate facet panel.
- `output="residuals"` returns `y_obs - y_hat` in `y`; `inject_zero_ref` requires it. `x_bins` and `x_estimator` (`mean`, `sum`, `min`, `max`, `median`) bin x and aggregate y per bin before fitting.
- `ci` must lie in `(0, 1)`.

**GLM family and link compatibility.** `Glm` refuses a link outside its family's row (`Glm: family 'poisson' incompatible with link 'logit'; valid links: [...]`).

| `family` | Canonical link | Other valid links |
|---|---|---|
| `gaussian` | `identity` | `log`, `inverse` |
| `binomial` | `logit` | `probit`, `log` |
| `poisson` | `log` | `identity`, `sqrt` |
| `gamma` | `inverse` | `identity`, `log` |
| `inverse_gaussian` | `inverse_squared` | `identity`, `log` |

**Grouping.** Every transform that takes `groupby` (stat and data alike) partitions through one shared group-key toolkit, so the rules are the same everywhere:
- A group column may be any string, integer, unsigned, float or boolean dtype (including `Categorical`), and appended group columns keep their original dtype. A temporal group column is refused, naming the column and the supported dtypes.
- Groups come out in first-appearance order, not sorted. Category-axis order, legend order and palette assignment follow from that order.
- A row whose group key is null is left out of every group. If every row's key is null, the transform warns (`<transform>: every row has a null in its groupby key (<cols>), so the data could not be split into groups; the grouped output is empty`).
- Three kinds of transform keep null-keyed rows instead, and warn with their own disposition when every key is null. `Impute` falls back to the whole column's statistic. `Smooth` and `Robust` fit the whole batch as one pooled fit. `transform_stack`, `transform_join_aggregate`, `transform_window` and `LetterValue` pool the null-keyed rows into one group, because each must give every input row a computed value.
- A group column named twice in `groupby`, or one that collides with the transform's own output column, raises and names the transform's real output columns.

**Degrade warnings.** A transform that degrades rather than fails names it in a `UserWarning` raised when the chart renders: a fit that could not be computed (`Smooth`, `Robust`, `transform_regression`, `transform_loess`, `Kde` on an all-NaN group or panel, naming the group), a parameter that could not be honored (such as `gam`'s `k` clamp or `LetterValue`'s depth cap), an `Impute` fallback, and the all-null group key above. The rendered output is the documented degrade.

**Model statistics.** Model diagnostics have no stat transform classes. `ModelSource` computes their data (`ModelSource.roc_curve`, `ModelSource.pr_curve`, `ModelSource.confusion_matrix`, `ModelSource.calibration_curve`, `ModelSource.lift_curve`, `ModelSource.cumulative_gain`, `ModelSource.importances`, `ModelSource.shap_values`, `ModelSource.partial_dependence`, `ModelSource.learning_curve`, `ModelSource.validation_curve` and the rest of §3.1), and the model diagnostic marks of §3.3 consume it.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#34-stat-transforms).

---

### 3.5 Data Transforms

Data transform builders are plain functions on `ferrum` (defined in `ferrum.transforms`). Each returns a transform-spec `dict` for `Chart.transform(...)`, which runs it in the Rust engine in declaration order, before any mark's own stat. They reshape, filter or derive columns.

| Builder | Meaning | Key parameters |
|---|---|---|
| `transform_filter(predicate)` | Row filter | An expression string, a dict filter, or a `Parameter` (a selection or variable parameter). A `Parameter` keeps every row in the static render and crossfilters live in the interactive runtime. |
| `transform_calculate(as_, expr)` | Derived column | `expr` is an expression string (see Expressions). |
| `transform_aggregate(*aggregates, groupby=None)` | Group-by aggregation, collapsing rows | Each aggregate is a dict `{"field": ..., "fn": ..., "as": ...}`. |
| `transform_join_aggregate(*aggregates, groupby=None)` | Aggregate columns added without collapsing rows | Same aggregate dicts. |
| `transform_bin(field, *, as_=None, maxbins=None, step=None, nice=True)` | Bin a field | `maxbins` caps the bin count; `step` fixes the bin width. |
| `transform_fold(fields, *, as_=("key", "value"))` | Wide to long (melt) | |
| `transform_pivot(field, value, *, groupby=None, limit=None, op="sum")` | Long to wide | `op` is `sum`, `mean`, `count`, `min`, `max`, `first` or `last`. |
| `transform_window(*window_transforms, sort=None, groupby=None, frame=None, ignore_peers=False)` | Window functions | Each op is a dict `{"op": ..., "as": ...}` with optional `field` and `param`. See Window functions. |
| `transform_density(field, *, bandwidth=None, kernel="gaussian", groupby=None, extent=None, steps=None, cumulative=False, as_=("value", "density"))` | KDE | `bandwidth=None` uses `1.06 · σ · n^(-1/5)` (population σ). `kernel` takes the §3.4 kernel vocabulary. Every `groupby` column is honored and appended to the output. |
| `transform_regression(x, y, *, method="linear", order=1, groupby=None, as_=("x", "y"))` | Regression curve | `method` is `linear`, `poly`, `exp`, `log` or `pow`; `order` is the polynomial order for `poly`. |
| `transform_loess(x, y, *, bandwidth=0.3, groupby=None, as_=("x", "y"))` | LOESS curve | |
| `transform_impute(field, *, method="value", value=None, groupby=None, key=None, keyvals=None)` | Fill missing values | `method` is `value`, `mean`, `median`, `min` or `max`. See Imputation. |
| `transform_flatten(fields, *, as_=None)` | Expand list-typed columns | |
| `transform_sample(n, *, seed=42)` | Seeded random row sample | Byte-deterministic for a given `seed`. |
| `transform_top_k(n, *, field, op="sum", sort="descending", groupby=None)` | Keep the top `n` | `op` is `sum`, `mean`, `count`, `min` or `max`; with `groupby`, groups are ranked by their aggregate. |
| `transform_stack(field, *, groupby, sort=None, as_=("y0", "y1"), offset="zero")` | Stacked extents for area and bar | `offset` is `zero`, `normalize` or `center`. |
| `transform_timeunit(field, unit, *, utc=False, as_=None)` | Temporal unit extraction | `unit` is `year`, `month`, `day`, `hour`, `minute`, `second`, `day_of_week`, `week` or `quarter`; `as_` defaults to `"{unit}_{field}"`. |

**Validation at the builder.** A bare string where a list of column names belongs (`groupby="g"`, `keyvals="abc"`) raises `ValueError`, since it would iterate into characters. A `groupby` that names the same column twice raises at the builder (`transform_aggregate: groupby column 'g' is repeated in `groupby`; ...`).

**Expressions.** `transform_filter` and `transform_calculate` share one expression grammar: field access (`datum.x`, `datum["a b"]`), arithmetic, comparison and logical operators, the ternary `cond ? a : b`, string, boolean and `null` literals, scientific notation, array literals, and function calls resolved against a builtin registry. The one builtin is `indexof(array, value)`, the index of the first element equal to `value`, else `-1`, so membership is `indexof([1, 2, 3], datum.id) >= 0`. An unknown function raises naming the known ones. A dict filter compiles to this grammar: a list value to the `indexof(...) >= 0` form, an operator dict to comparisons, a scalar to equality, joined with `&&`. Its values are encoded by the grammar's own literal rules (`true`, `false`, `null`), numpy and pandas scalars are normalized to Python values, and non-finite floats or unsupported types raise by name. An expression whose overall result is an array raises at render, naming the expression and the row: arrays are valid only as an `indexof` argument.

**Aggregation vocabulary.** `transform_aggregate`, `transform_join_aggregate`, `Aggregate`/`AggregateOp(fn_=)` and `Bin(agg=)` share one vocabulary: `mean`, `sum`, `count`, `min`, `max`, `median`, `variance` (alias `var`), `stdev` (alias `std`), `q1`, `q3`, `distinct`. `Hex(aggregate=)` takes the subset `count`, `mean`, `sum`, `min`, `max`, `median`, `variance`/`var`, `stdev`/`std`; `Raster(aggregate=)` has its own `count`, `density`, `mean`, `sum`, `any`; `Smooth(x_estimator=)` takes `mean`, `sum`, `min`, `max`, `median`. An alias is stored and written to the wire in its canonical spelling (`Hex(aggregate="std")` holds `stdev`). An aggregate spelling is parsed when the class is constructed or the spec is deserialized, so a bad value in a hand-built spec refuses when the spec is read.

**Refusal wording.** A Rust-side door (a class constructor, an apply-time door, the wire) refuses a declared vocabulary value (a method, kernel, distribution, aggregate, offset or interpolation) as `<door>: unknown <noun> '<value>'; expected a|b|c`, listing the vocabulary bare, `|`-joined, in declared order. Python declaration doors (`mark_hex`, `mark_raster` and `RenderConfig`'s `aggregate=`, `mark_smooth`/`lmplot`'s `x_estimator=`) use the package-wide `<door>: <param> must be one of [<sorted list>]; got <value>` shape. `interpolate=` on `mark_line`, `mark_area` and `mark_ribbon` refuses at the call, a non-string included, in the Rust shape under the calling door's name: `mark_line: unknown interpolate 'cubic'; expected linear|step|step-before|step-after|monotone|basis|cardinal`. `mark_hex` and `mark_raster` refuse an unknown `aggregate=` at the call; an aggregate that needs a field but has none refuses at resolution, since `.encode(color=...)` may still supply it.

**Window functions.** `transform_window` ops are `row_number`, `rank`, `dense_rank`, `lag`, `lead`, `first_value`, `last_value`, `sum`, `mean`, `count`, `min` and `max`; all but the first three need a `field`. `rank` and `dense_rank` without a field take their ties from the `sort` columns: rows equal on every sort column share a rank (competition ranks for `rank`, dense ranks for `dense_rank`). Sort and field columns coerce every integer, unsigned, float and temporal dtype; another dtype raises naming the transform, column and dtype. Null sort values order last and tie with each other. A `frame` of `(preceding, following)` offsets (`None` is unbounded) that lies entirely before its row selects no rows. With `ignore_peers=False`, a rolling frame is anchored on the current row's run of tied peers and each end grows to whole peer runs, so tied rows always share one frame and one value; `ignore_peers=True` uses row offsets alone. This differs from Vega, which anchors on the current row: for sort key `t=[1,1,2,3,3]`, `x=[10,20,30,40,50]` and `frame=(-1, 0)`, ferrum's rolling sum is `[30,30,60,120,120]` where Vega's is `[30,30,60,120,90]`.

**Grouped fits and fills.** `transform_regression` and `transform_loess` fit each group independently over that group's own x range and append the group columns at their original dtypes. `transform_impute(groupby=)` computes the fill from each group's own rows; a group with no usable values falls back to the whole column's statistic. With `key=`, the key universe is `keyvals=` when given, else the distinct key values across the whole batch, and each group (the whole batch with no `groupby`) missing a key gets one appended row: its group columns, the key, the group's fill value, and every other column null. `keyvals=` without `key=` raises, as does a `keyvals` element outside the key column's dtype range. The general grouping rules are in §3.4 Grouping.

**Introspection.** `ferrum._core.transform_registry(kind)` is the Rust-owned description of each transform kind: a dict with `methods` (the method vocabulary), `methods_needing_field` (the spellings the kind refuses without a field), `outputs` (`(name, dtype)` pairs), `secondary_outputs` (`(key, columns)` pairs for named secondary tables under default construction) and `params` (the accepted spec fields). An unknown kind raises `ValueError` listing the known kinds. Python reads its vocabularies and output column names from this registry. Every transform class's `repr()` shows every argument it holds, with canonical vocabulary spellings; `eval(repr(x)) == x` is not promised.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#35-data-transforms).

---

### 3.6 Scales

Scales map data domain values to visual range values. A scale is attached to an encoding channel through its `scale=` keyword, as a scale-class instance (`fm.X("v", scale=fm.LogScale())`) or as a raw dict (`fm.X("v", scale={"type": "log"})`). The two spellings reach the same wire `ScaleSpec` and render identically.

#### Scale Classes

```
LinearScale(*, domain=None, range=None, clamp=False, nice=False, padding=None, reverse=False)
LogScale(*, domain=None, range=None, base=10.0, clamp=False, nice=False, padding=None, reverse=False)
PowScale(*, domain=None, range=None, exponent=2.0, clamp=False, nice=False, padding=None, reverse=False)
SqrtScale(*, domain=None, range=None, clamp=False, nice=False, padding=None, reverse=False)
SymlogScale(*, domain=None, range=None, constant=1.0, clamp=False, nice=False, padding=None, reverse=False)
TimeScale(*, domain=None, range=None, clamp=False, nice=False, padding=None, utc=False, reverse=False)
OrdinalScale(*, domain, range=None, padding=0.0)
PointScale(*, domain=None, padding=None, align=0.5, reverse=False, range=None)
BandScale(*, domain=None, padding=None, padding_inner=None, padding_outer=None, align=0.5, range=None)
SequentialScale(*, scheme=None, domain=None, reverse=False)
DivergingScale(*, scheme=None, domain=None, domain_mid=None)
ThresholdScale(*, domain, range)
QuantileScale(*, domain, range)
QuantizeScale(*, domain, range)
BinOrdinalScale(*, bins, scheme=None)
```

Every scale class is a `ferrum._core` class exported on `ferrum`, and every argument is keyword-only.

| Class | Wire `type` | Kind | Notes |
|---|---|---|---|
| `LinearScale` | `"linear"` | Continuous | The default for a quantitative field. A raw dict with no `"type"` key is a linear scale. |
| `LogScale` | `"log"` | Continuous | `base` is the logarithm base. |
| `PowScale` | `"pow"` | Continuous | `exponent` is the power. |
| `SqrtScale` | `"sqrt"` | Continuous | A power scale with exponent 0.5. |
| `SymlogScale` | `"symlog"` | Continuous | Symmetric log, defined through zero; `constant` sets the linear region. |
| `TimeScale` | `"time"`, or `"utc"` with `utc=True` | Continuous, temporal | Domain values are epoch milliseconds (see TimeScale below). |
| `OrdinalScale` | `"ordinal"` | Discrete | `domain` is required; `range` may be pixel positions or color strings. |
| `PointScale` | `"point"` | Discrete | Zero-width positions, for dot plots. |
| `BandScale` | `"band"` | Discrete | Bands with inner and outer padding, for bars. |
| `SequentialScale` | `"sequential"` | Color | A continuous scheme over a numeric domain. |
| `DivergingScale` | `"diverging"` | Color | A diverging scheme; `domain_mid` (wire `domainMid`) is the neutral midpoint. |
| `ThresholdScale` | `"threshold"` | Discretizing | `k` ascending thresholds split the domain into `k + 1` buckets; `range` holds the `k + 1` numeric outputs. |
| `QuantileScale` | `"quantile"` | Discretizing | Buckets at data quantiles, one bucket per `range` entry; the `quantiles` getter reports the computed thresholds. |
| `QuantizeScale` | `"quantize"` | Discretizing | `domain=[lo, hi]` cut into one uniform bucket per `range` entry; `range` holds color strings. |
| `BinOrdinalScale` | `"bin-ordinal"` | Discretizing | Explicit bin boundaries, colored from `scheme`. |

The sixteen wire tags are published by `ferrum._core.scale_type_tags()` (and the temporal pair `"time"`, `"utc"` by `ferrum._core.scale_temporal_type_tags()`).

**Raw-dict scales and the key gate.** A raw-dict scale is checked at the wire boundary against the accepted-key set of its type, published as `ferrum._core.scale_accepted_keys(scale_type)` (which raises `ValueError` for an unknown type). Wire keys are camelCase (`paddingInner`, `paddingOuter`, `domainMid`, `domainParam`). An unknown key is refused with `ValueError: scale: unknown key '<key>' for type '<type>'; accepted: <sorted list>`. A dict whose `domain` is a `fm.param(...)` (or its `.ref()` marker) emits `domainParam` with the parameter's name in place of a literal domain, for a reactive domain. A `"time"`/`"utc"` dict converts each domain element under the same rule as `TimeScale(domain=...)` before serialization.

The gate is keyed by scale type, not by channel. A key valid for the type is accepted on every channel even where that channel does not read it: `reverse` on a `color`, `size` or `opacity` scale is accepted and has no effect, since only positional resolution reads it. An opacity-family channel (`opacity`, `fill_opacity`, `stroke_opacity`) resolves only a linear scale; any other scale type there warns (`<channel>: <type> scale is not supported on opacity-family channels; ...`, naming the wire tag) and falls back to the default linear resolution.

**Continuous domain and range.** With no `domain=`, a continuous scale infers its domain from the data extent. An explicit `domain=` or `range=` must have exactly two elements, on the class and the dict spellings alike; a dict scale of the wrong length is refused at render with `<channel> scale: domain must have length 2; got <n>` (or the range equivalent). The constructors also refuse equal endpoints (`domain endpoints must differ`); a dict-spelled or parameter-driven domain with equal endpoints is not refused, and a reactive `domainParam` domain that collapses to one value renders around that midpoint. A caller scale over a column with no non-null value resolves to the `(0, 1)` fallback the automatic path uses. `nice=True` extends the domain to round tick values: with an explicit domain the constructor nices it eagerly (so `.domain` and `.ticks()` reflect it), and with an inferred domain the renderer nices the data extent; the class and dict spellings render identically. `round=` is not a parameter of any scale class.

**Padding (continuous scales).** `padding` is a fraction of the plot dimension reserved inside each end of the axis, so marks do not touch the axis line or the plot edge. Resolution, in the renderer (`resolve_padding_fraction`):

- an explicit `padding=p` is used as given, `0.0` disabling it;
- with `padding` unset and an explicit `domain=`, the padding is `0.0` (the stated domain is authoritative);
- otherwise the padding is `0.05`, capped at 8 px per side.

An explicit pixel `range=` bypasses the inset and is used as the final range.

**`reverse=`.** The keyword names three mechanisms, depending on the class and the channel:

- On a continuous positional scale (`LinearScale`, `LogScale`, `PowScale`, `SqrtScale`, `SymlogScale`, `TimeScale`, or a dict of those types) on `x`/`y`, `reverse=True` swaps the resolved domain pair at render. On an explicit domain it is exactly `domain=[b, a]` written by hand; on an inferred domain the swap follows padding inference. Marks and tick labels flip together. An explicit reversed `range=[hi, lo]` composes with it as stated, and `.xlim()`/`.ylim()` (or `CoordCartesian(xlim=, ylim=)`) replace the domain wholesale and win over `reverse`. The swap happens at render, so the compute facade ignores it: `fm.LinearScale(domain=[0, 10], range=[0, 100], reverse=True).scale(0)` is `0.0`.
- On `PointScale`, `reverse=True` reverses the sorted category order before layout, and the compute facade reflects it: `fm.PointScale(domain=list("abc"), range=[0, 100], reverse=True).scale("a")` is `83.33`.
- On a non-positional channel it is accepted and inert (see the key gate above). `BandScale` has no `reverse` keyword and refuses it.

**TimeScale.** Each domain element may be a `float` or `int` (epoch milliseconds), a `datetime.date` (midnight UTC), a `datetime.datetime` (naive means UTC; an aware value converts through its own offset), or an ISO-8601 date or datetime string (parsed by `datetime.date.fromisoformat` / `datetime.datetime.fromisoformat`). `bool` is refused, an unparseable string raises `ValueError` naming the accepted formats, and any other type raises `TypeError` naming the accepted forms. All temporal rendering (tick values, calendar nicing, time formatting) is UTC, so `TimeScale(utc=True)` and `TimeScale(utc=False)` render byte-identical output; `utc=True` selects the wire tag `"utc"`, which survives resolution for downstream consumers but selects no rendering behavior.

#### Discrete scales: band, point and ordinal

**BandScale padding.** `padding`, `padding_inner` and `padding_outer` are each `None` until set. Resolution has one owner, in Rust: `inner = padding_inner ?? padding ?? 0` and `outer = padding_outer ?? padding ?? 0`. A band scale named with no padding (`fm.BandScale()`, `{"type": "band"}`, or `.override(x_scale_type="band")`) therefore renders exactly like the auto-inferred categorical axis (inner 0, outer 0, `align` 0.5). Getters: `padding` returns the set value or `None`; `padding_inner`/`padding_outer` return `side ?? padding`, or `None` when neither is set. `BandScale() == BandScale(padding=None)` but `BandScale() != BandScale(padding=0.0)`, though both draw the same pixels; `repr(fm.BandScale())` is `BandScale(domain=[], padding=None, padding_inner=None, padding_outer=None, align=0.5)`. Only user-set padding keys reach the wire.

**PointScale padding.** An unset `padding` resolves in the renderer to 0.5 (half a step at each end); the getter returns `None` when unset, so `PointScale() != PointScale(padding=0.5)` and `repr(PointScale())` prints `padding=None`, though both render identically. `OrdinalScale` is the exception among the discrete classes: its `padding` defaults to `0.0` and its getter returns that.

**Validation.** On the class door, `padding` and `padding_inner` must be numbers in `[0, 1]` inclusive and `padding_outer` a finite number `>= 0`, each refused as `ValueError: BandScale(padding_inner=...) must be a number in [0, 1]; received 1.5` (or `... must be a finite number >= 0; received -1`); a malformed `padding` is refused even when explicit inner and outer values would mask it. `align` must be in `[0, 1]` (`BandScale(align=...) must be a number in [0, 1]; received 2`), and `PointScale(padding=...)` must be a finite number `>= 0`. Type coercion is PyO3's: `BandScale(padding="a")` raises `TypeError: argument 'padding': must be real number, not str`, and `padding=True` is read as `1.0`. The dict door refuses a malformed band padding at render, naming the scale and the channel the caller wrote (`ValueError: band scale on x: paddingInner must be a number in [0, 1]; received 1.5`), even under `CoordFlip`; it does not check `PointScale`'s `padding >= 0`, either class's `align` bounds, or `OrdinalScale`'s `padding` bound.

**Band geometry.** Bands follow d3's `scaleBand`/`scalePoint` layout model. Mark widths and axis tick-label centers read the same resolved scale. Every band consumer follows the resolved layout: bar, boxplot, tick, heatmap cell, dodge sub-bands, swarm overflow and categorical tick placement. A mark's width is the scale's padding-aware `bandwidth()` times a band factor: a fixed 0.8 for a bar (`mark_bar` takes no `band_size`), and the mark's `band_size` (default 0.6) for a box or tick. The category count is the scale's domain, not the data batch's, so a layer or facet cell missing categories sizes its marks against the full domain. A heatmap cell is its drawn band on each axis: a padded `BandScale` opens a gutter between cells, and with zero padding cells tile exactly. Against an explicit `PointScale`, which has no inner gap, a mark takes `|step|` times its band factor. `Dodge` subdivides the drawn band and applies its own `padding` within it. `BandScale.align` is visible only when the band denominator `n - padding_inner + 2 * padding_outer` falls below 1 and is clamped to 1 (in practice a single category with large inner padding); `PointScale.align` has no effect for any domain size.

**Violin exception.** `mark_violin` sizes each polygon before scale resolution, as panel pixels divided by the batch's group count. It agrees with its band siblings only in the automatic case (no explicit scale): under `BandScale(padding_inner=p)` violins keep the full unpadded slot while siblings narrow to `(1 - p)` of it, and under an explicit `domain=` naming an absent category, or an explicit `range=` other than the plot area, the siblings' band moves while the violin's does not.

**Compute facade.** `BandScale.scale(value)` returns the band's leading edge and `bandwidth()` returns `|step| * (1 - padding_inner)`: `BandScale(domain=list("abcd"), range=[0, 400]).bandwidth()` is `100.0`, and `BandScale(domain=list("abcd"), range=[40, 260]).scale("a")` is `40.0`. Two divergences from d3 are documented in `crates/ferrum-core/src/scale/discrete.rs`: outer padding is applied directly to the start and `align` distributes only the denominator clamp's leftover (the two agree when `align = 0.5` or `padding_outer = 0`), and a negative-extent range keeps its sign, so `lead(i)` is the leading edge in domain order.

**Explicit tick values on a categorical axis.** When an explicit tick-value override (`configure_axis(tick_values=...)`) supplies exactly one value per category, each relabeled tick lands on its category's center, the same center the mark sits on, with or without an explicit `range=`. When its length differs from the category count, the ticks fall on evenly spaced slots across the plot area.

**Ordinal color ranges.** `OrdinalScale(range=...)` accepts color strings (hex `#rrggbb`/`#rgb` or CSS named colors) as well as pixel positions. Paired with `domain`, this maps categories to explicit colors: `OrdinalScale(domain=["A", "B", "C"], range=["#ccc", "#ccc", "#e4572e"])`.

#### Discretizing color scales

`Color(scale=...)` honors `QuantizeScale`, `QuantileScale`, `ThresholdScale` and `BinOrdinalScale` as bucketed scales. Quantize buckets `domain=[lo, hi]` (the data extent when a dict omits it) into one uniform bucket per `range` entry; Quantile buckets at data quantiles; Threshold maps `k` thresholds to `k + 1` buckets; BinOrdinal uses explicit bin boundaries.

**Bucket colors**, in precedence order (`bucket_colors`):

1. An explicit color-string range: `QuantizeScale`'s `range`, or an `OrdinalScale` string range. It is all-or-nothing: one unparseable entry raises `ValueError: could not parse color '<entry>' in an explicit color range`. `ThresholdScale` and `QuantileScale` ranges are numeric and only set the bucket count.
2. A categorical scheme name (on `Color(scheme=)` or the scale's own `scheme`): its entries in declaration order, cycling with a `ColorPaletteOverflowed` warning when the buckets outnumber them.
3. Otherwise a continuous scheme (the named one, else the theme's), sampled at evenly spaced points.

A descending Quantize `domain=[hi, lo]`, or a fully descending threshold or bin-boundary list, normalizes to ascending with the swatch order reversed; a non-monotonic boundary list (reachable only through a raw dict) is a typed render error. A `configure_color(range=...)` whose length differs from the bucket count warns (`color range names <n> color(s) but the binned color scale has <m> bucket(s); the range was not applied`). The color legend draws discrete labeled swatches for a discretizing scale instead of a gradient.

**Continuous color scales.** `fm.continuous_palette(name)` returns a `ContinuousScheme` usable as `Color(scale=...)`, equivalent to `SequentialScale(scheme=name)`. `fm.Gradient(stops)` builds one from explicit `(position, color)` stops, which reach the render path and compose with `reverse`.

#### Color Scheme Constants (`ferrum.schemes`)

Scheme names are strings, accepted by `Color(scheme=)`, the scale classes' `scheme=`, and the theme's `color_scheme`, `sequential_scheme` and `diverging_scheme`. The one registry is `crates/ferrum-core/src/render/color/palette.rs`; `ferrum._core.list_palettes()` lists every name, `ferrum.color.palette(name, n)` returns hex colors, and `ferrum.schemes` holds `continuous_palette` and `Gradient`.

**Categorical:** `paper_ink` (the default theme's), `okabe_ito`, `tableau10`, `set1`, `set2`, `paired`, `pastel`, `dark2`, `slate_citrus`, `arctic_signal`

**Sequential:** `cool_blue` (the default theme's), `viridis`, `plasma`, `magma`, `inferno`, `cividis`, `blues`, `reds`, `greens`, `oranges`, `purples`, `warm_ochre`, `night_blue`, `electric_lime`, `signal_blue`, `ember_orange`

**Diverging:** `blue_to_red` (the default theme's), `rdbu`, `cyan_to_amber`, `blue_to_violet`

There are no cyclical schemes. A theme's `color_scheme` must name a categorical or continuous scheme, and an unknown name raises `ValueError` at render entry (§3.13). Categorical color resolves the scheme from the encoding's own `scheme` first, then the theme's `color_scheme`; when that name is a continuous scheme, a nominal encoding uses `tableau10` instead, so categories never collapse to a single hue.

#### Scales on layered charts

**Positional axes.** On a layered chart (`+`, `Chart.layer(...)` or `fm.layer(...)`), the first caller-written continuous scale in layer order establishes the channel's axis, whichever layer carries it. A layer that declares no scale on the channel (or carries only a bar's injected zero anchor), or that carries an equal copy of the establishing scale, resolves on that axis; its data, together with any paired `y2`/`x2`, widens the extent only when the establishing scale states no `domain` (a stated `domain` is used as written). A layer's data is the batch it draws: its own named output when it reads one (a stat layer's result, a filtered operand's rows), else the chart's data, plus the facet-shared output on a shared faceted axis. Its fields widen the axis from that batch only, never from a same-named column in another batch, so `mark_smooth(ci=0.95)` spans its fit and band rather than the raw minimum, a user column named `lower` does not stretch `mark_errorbar`, and a filtered operand of `+` spans only the rows it keeps. Layer 0's field, which types the axis, is read from layer 0's batch, and an empty batch contributes nothing. The same rule holds for the ordinal category union and for a caller-written scale. A layer with a different scale, or with its own `domain`, stays out of the axis and warns once per layer and channel (`RenderWarning::LayerScaleExcluded`: `layer <i>: its <channel> scale differs from the one layer <j> establishes the <channel> axis with, so it stays out of that axis; ...`). On `fm.layer` such a layer draws on its own unlabelled scale; on `+` there is one scale per axis, so its marks fall outside the shared axis and are clipped. Both routes resolve the same axis for the same layers in any order (`tests/test_layer_union_provenance.py`). A categorical first scale keeps the ordinal band union and never refuses. When layer 0 (on y, the first layer that is not a secondary-y layer) binds a categorical column, the panel axis is layer 0's categorical axis, and a later layer's continuous caller scale, including one a per-layer `CoordFlip` moves onto that channel, stays out with the same warning. A flipped layer that binds layer 0's own categorical column is refused with the `LayerCoordUnplaceable` reason on `+` and `fm.layer`, in static and interactive output. One route difference remains: a bar's injected zero anchor merges into the chart-level scale copy only, so a reversed-scale bar beside a scale-less point spans to zero on `fm.layer` but not on `+`.

On an unlayered chart, a caller scale stating no `domain` spans `y` and its paired `y2` (or `x` and `x2`) together.

**The bar's zero anchor.** A bar chart's value scale is injected by ferrum to anchor the axis at zero, and the injection is marked with the wire bit `systemInjected` on the linear scale. A scale the caller writes carries no such bit and wins over the cross-layer union; the injected one yields to it, so a bar overlaid with a mark whose values extend beyond it shares one value axis with that mark. The key is accepted on a hand-written linear dict (where it makes that scale yield as an injected one would) but has no override path. An explicit `domain=` on a bar's value scale suppresses the zero anchor.

**Color domains.** On `fm.layer(...)` the categorical color domain is the union of every layer's categories, in layer order. On `+`, the chart-level color domain is the first color-bearing layer's: a later layer's categories outside it draw in the first palette color and are absent from the legend, so `a + b` and `b + a` color differently when their categories differ. When only one layer encodes color, its colors survive in either order. Axis titles are taken from the first data-bearing layer; annotation-only layers (e.g. `annotate_rect`) never supply or rename axis titles.

#### Scale overrides

`Chart.override(<channel>_scale_<leaf>=...)` sets one scale key. Its leaves are the gate's own accepted wire keys (`scale_accepted_keys`), so `nice`, `zero`, `stops`, `domainParam`, `paddingInner`, `paddingOuter` and `domainMid` are all reachable, and each camelCase key also has a snake_case spelling (`x_scale_padding_inner` and `x_scale_paddingInner` both emit `paddingInner`). `systemInjected` has no override path, and `utc` and `quantiles` are not leaves (the type switch to `"utc"` is `<channel>_scale_type="utc"`): an unknown leaf is refused with `Chart.override: Unknown override path '<name>'`. A leaf the effective type does not accept is refused naming that type's accepted keys, e.g. `.override(x_scale_padding_inner=0.3)` on a linear x; it works once the effective type accepts it (`.override(x_scale_type="band", x_scale_padding_inner=0.3)`).

- **Every value is checked before conversion**, naming the path the caller wrote: `Chart.override(x_scale_padding=...) must be a number in [0, 1]; received 1.5` on a band x, `Chart.override(x_scale_align=...) must be a number in [0, 1]; received 5`, `Chart.override(x_scale_padding=...) must be a finite number >= 0; received -1` on a point x, and `Chart.override(x_scale_padding=...) must be in [0, 1]; got 1.5` on an ordinal x. A malformed `TimeScale` domain element refuses the same way.
- **`None` is no override** on every leaf: the chart's own scale value stands, and a later `None` undoes an earlier override.
- **Every layer.** The leaves merge into every layer that declares the channel as well as the chart-level encoding, so composite marks, figure helpers built on layers, `+` and `Chart.layer(...)` render the override the way the channel's own `scale=` does. A refusal from the scale gate surfaces from the chart-level encoding when the chart declares the channel and from the declaring layer otherwise.
- **Bar zero anchor.** `mark_bar().override(y_scale_domain=[50, 200])` renders identically to `Y("val", scale={"domain": [50, 200], "zero": False})`: the override merges before the zero anchor is injected.
- **Base scale first.** On a channel with an explicit `scale=`, a type switch first validates the base scale's keys against its own type, so an already-invalid base scale refuses as the bare gate would.
- **Type switches translate padding by meaning** (`ferrum._core.translate_scale_switch(old_type, new_type, base)`, one table in `scale/discrete.rs`). Only user-set values are carried. Ordinal to band: `padding` becomes `paddingInner`. Ordinal to point: `padding` is dropped. Point to band: `padding` becomes `paddingOuter`, `align` stays. Point to ordinal: both dropped. Band to ordinal: the inner side (`paddingInner`, else `padding`) becomes `padding`; outer and `align` are dropped. Band to point: the outer side (`paddingOuter`, else `padding`) becomes `padding`, `align` stays. Continuous to discrete and back: `padding` is dropped. Between continuous types: unchanged. Every other key is kept when the new type accepts it and dropped otherwise. A dropped value is discarded silently. A malformed value is refused under the old type, naming the key the caller wrote, whether carried or dropped (`ValueError: scale: paddingInner must be a number in [0, 1]; received 1.5`). A key stated in the same call as the new type is applied after the translation, under the new type.
- **`inject_scale` is not a type switch.** A composite's `inject_scale` replaces a scale whole and does not translate.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#36-scales).

---

### 3.7 Axes and Legends

Axes and legends are configured at three levels: per channel (`fm.X(..., axis=fm.Axis(...))`, `fm.Color(..., legend=fm.Legend(...))`), per chart (`configure_axis(...)`, `configure(axis_x=AxisConfig(...))`, `configure_legend(...)`), and by the theme. For one field the most specific setting wins: a mark literal, then the per-channel value, then the chart-level value, then the theme.

#### `Axis`

```
Axis(title, *, orient=None, ticks, tick_count=None, tick_extra, tick_min_step=None,
     grid, grid_dash=None, grid_width=None, grid_color=None, grid_opacity=None,
     labels, label_angle=None, label_flush, label_overlap,
     label_format=None, label_format_type=None, label_font_size=None, label_color=None,
     domain, domain_width=None, domain_color=None,
     offset=None, translate=None, min_band=None, max_band=None,
     title_orient=None, title_font_size=None, title_color=None, title_padding=None,
     values=None, zindex=None, label_map=None)
```

`Axis` is a frozen dataclass; its fields can also be passed positionally in declaration order. Parameters without a default in the signature are unset when omitted:

| Parameter | Effective default | Meaning |
|---|---|---|
| `title` | the field name | Axis title. `None` suppresses the title. |
| `ticks` | `True` | Draw tick marks. |
| `tick_extra` | `False` | Include an extra tick at the domain boundary. |
| `grid` | the theme's `grid` (on) | Draw gridlines for this axis. |
| `labels` | `True` | Draw tick labels. |
| `label_flush` | `False` | Flush the first and last labels to the axis ends. |
| `label_overlap` | `"greedy"` | Label collision strategy: `"greedy"` (the collision cascade), `"parity"` (keep every other label), `"rotate"` (force the rotate stage), `"true"` (keep every label), `"false"` (same as `"greedy"`). |
| `domain` | `True` | Draw the axis line. |

The remaining fields default to `None`, which leaves the renderer's or theme's value in place. `orient` places the axis (`"top"`, `"bottom"`, `"left"`, `"right"`). `title_orient` sets the title's rotation (a horizontal value lays a y-axis title flat). `offset` and `translate` shift the axis in pixels. `min_band`/`max_band` clamp the pixel band reserved for the axis. `values` lists explicit tick values. `label_map` renames tick labels: it maps original label text to display text and is applied to the data before the scale is computed, so keys absent from the data are ignored. The style fields (`grid_*`, `label_*`, `domain_*`, `title_*`, `tick_count`, `tick_min_step`, `zindex`) set what their names say.

**Unset versus explicit.** For `ticks`, `tick_extra`, `grid`, `labels`, `label_flush`, `label_overlap` and `domain`, omitting the field or passing `None` means "not specified" and the key is not serialized. Any other value, including one equal to the effective default (`Axis(grid=True)`), is serialized, so it beats a conflicting chart-level value for that field.

**Validation.** `label_overlap` refuses any other token at construction with `ValueError` (`Axis.label_overlap: label_overlap must be one of ['false', 'greedy', 'parity', 'rotate', 'true']; got 'hide'`); a raw axis dict gets the same check. `min_extent`/`max_extent` are deprecated aliases of `min_band`/`max_band` on `Axis`, `AxisConfig` and `configure_axis`: each emits a `DeprecationWarning` attributed to the caller's line (also through `dataclasses.replace`), and passing an alias with its canonical name raises `TypeError`. On an instance, `min_extent`/`max_extent` read back the unset sentinel, and `inspect.signature` shows the two aliases as `InitVar[object]` parameters.

**Hiding an axis from its channel.** `fm.X(field, axis=None)` (or `Y`) hides that axis entirely, through the same switch as `Chart.axis(x=False)`. `axis=False` serializes a suppression dict that turns off the axis line, ticks, labels and grid; the axis title is still drawn. `Chart.axis()` wins over either spelling.

**Tick values.** `values=` takes numbers (and temporal values on a temporal axis). On a categorical axis a list of strings is refused at render with the wire deserializer's message, for example `ValueError: axis: invalid type: string "a", expected f64 at line 1 column 15`. A tick format composes with `values=`: an untyped `".1%"` on a numeric axis formats the explicit values too.

**Gutters follow drawn chrome.** Layout reserves an axis title gutter and a tick-label band only for chrome that is drawn. `.axis(show=False)` on a 640x480 chart with 16 px padding leaves the plot rect at exactly `(16, 16, 608, 448)`; hiding one axis frees only that side; `Axis(labels=False)` frees the label band and keeps the title gutter; a `CoordPolar` chart reserves no Cartesian gutters.

**Tick-label formats.** Formats are applied in Rust, identically in SVG, PNG and the labels baked into interactive exports. Numbers use the d3-format grammar (`[[fill]align][sign][symbol][0][width][,][.precision][~][type]`, including `s`, `%`, `p`, `r`, `g` and the `~` trim flag); time uses strftime directives (`%b %Y`, `%Y-%m-%d`, `%H:%M`). A named preset from `ferrum.format_presets.ALL_PRESETS` (for example `"currency"`, `"percent"`, `"date_iso"`, `"ordinal"`) resolves to its spec on every surface that accepts a format: `Axis(label_format=)`, `AxisConfig(label_format=)`/`configure_axis(label_format=)`, `Legend(format=)`, an encoding's `format=` (`X`, `Y`, `Text`, `TooltipField` and the other channels that accept it), and raw axis or legend dicts (including the `labelFormat` spelling). A string that is not a preset name passes through as a raw spec. `AxisConfig.label_format` accepts preset names only and raises `ValueError` for any other name; a raw spec goes in `label_format_raw`, and the two are mutually exclusive. `"ordinal"` renders `1st`, `2nd`, `3rd`, `4th`, with `11th`–`13th`. An explicit per-channel format beats a chart-level one.

How a spec is read:

- Every surface reads a format the same way: the shared axis, an independent facet axis (`share_scale(x="independent")`), the colorbar, the size, shape and stroke-dash legends, and the text channel. An independent facet axis applies `Axis(label_format="%Y/%m")` exactly as the shared axis does.
- A spec is strftime only when it contains a `%` escape. An undeclared `%`-bearing spec is read as time where the formatted value is temporal (an axis by its scale; a legend or the text channel by its field) and as d3 otherwise. On a temporal field `"%Y"` renders years; on a non-temporal field it is refused as a d3 spec (`invalid format spec "%Y": unrecognized token "Y" at position 1 ...`). On a temporal field a `%` spec that is not a valid date pattern (`".1%"`) is refused (`invalid format spec ".1%": invalid strftime pattern ".1%": bad or unsupported format string`).
- A time-typed format (`format_type="time"`, a strftime spec, or a time preset) on a non-temporal value is refused, naming the surface: `format "%Y" is typed "time", which on the 'x' axis requires a temporal scale, but the resolved scale is not temporal`. Legends and the text channel check the field instead (`... which on the 'color' legend requires a temporal field, but the field is not temporal`; also `'size'`, `'shape'` and `'stroke_dash'` legends and the `'text'` channel). A bare `format_type="time"` with no spec reads `format type "time" on the 'x' axis requires a temporal scale, ...`. On a temporal value a time-typed spec with no `%` pattern (`".2f"` typed `"time"`) falls back to the scale's own date labels.
- The ordinal preset formats numbers. `"ordinal"` resolves to the one wire token `"__ordinal__"`, and a declared ordinal preset wins over a stray `format_type="time"`: on a non-temporal value every surface renders `1st`, `2nd`, … and no axis refuses it as a time format. On a temporal value it is refused (`the "ordinal" format on the 'x' axis formats numbers and requires a non-temporal scale, but the resolved scale is temporal`).
- A d3 number format (`.2f`, `,.0f`, or a number preset such as `"percent"`) on a temporal value is refused (`the number format ".2f" on the 'x' axis requires a non-temporal scale, but the resolved scale is temporal; a date pattern (e.g. "%Y") formats a temporal value`).
- An `Axis(label_format_type=)` types the shorthand `X(format=)` beside it, and the merged pair meets the same checks as its one-place spellings: `fm.X("d:T", format="%Y", axis=fm.Axis(label_format_type="number"))` refuses with `invalid format spec "%Y": unrecognized token "Y" ...`, and `fm.X("n:Q", format=".1f", format_type="time", axis=fm.Axis(label_format_type="number"))` still meets the time-typed refusal. Only a time-typed shorthand is checked as written, so `fm.X("d:T", format=".2f", axis=fm.Axis(label_format_type="time"))` renders the scale's own date labels.
- A malformed spec raises `ValueError` naming the problem (`invalid format spec "curency%": invalid strftime pattern "curency%": bad or unsupported format string`).
- A legend whose rendering the caller disabled (`legend=None`, `Legend(orient="none")`, `configure_legend(orient="none")`) is not checked. A legend the renderer itself suppresses with a warning (a colorbar on a `mark_line` or `mark_ribbon` layer) is still checked, and so is a color legend on a categorical scale. The shape and stroke-dash legends grammar-check `Legend(format=)` as the color legend does. An axis is always checked, even with `axis=None`.
- A chart-level `configure_axis(label_format=...)` reaches both axes, so a time preset there is refused when one axis is not temporal; target one axis with `configure(axis_x=AxisConfig(label_format=...))` or the per-channel `Axis`.
- A field is temporal when its declared type is temporal or, undeclared, its column is a date or datetime. The tooltip column reads only the declared `format_type`, which no Python spelling of `TooltipField` puts on the wire.

**Chart-level axis configuration.** `configure_axis(...)` (both axes) and `configure(axis_x=, axis_y=, axis_y2=AxisConfig(...))` (one axis, or the secondary y axis of a `SecondaryY` chart) carry the chart-level fields. Besides the style fields they include `domain_min`/`domain_max`, `nice`, `zero`, `tick_count`, `tick_size`, `tick_values`, `labels`, `ticks`, `title`, `label_overlap` (honored on both axes), `offset` and `label_flush`. A per-channel and a chart-level setting compose by precedence, and an `axis_x` setting never reaches the y axis. `domain_min`/`domain_max` feed positional scale resolution: an encoding's own `scale=` domain wins over them silently; on an ordinal axis they are not applied and a warning says so (`domain_min, domain_max apply to continuous scales; the x axis is ordinal, so they were not applied`); a domain a log scale cannot take refuses with the log scale's own sentence (`x axis scale-domain config: log scale domain endpoints must have the same sign`); equal endpoints refuse at construction (`AxisConfig: domain_min/domain_max: domain endpoints must differ (lo != hi)`). `nice=True` rounds with the scale kind's own `nice()`, including `log` and `time`. `AxisConfig.x`/`.y` (and `configure_axis(x=, y=)`) are deprecated and have no effect: they warn, attributed to the caller, and are not serialized. Per-axis gridlines take `GridConfig(x=, y=)`, each a bare `bool`, a `GridAxisConfig(enabled=None, color=None, width=None, dash=None, opacity=None)` or its dict form; anything else is refused naming the accepted shapes.

Every chart-level config section (`axis`, `axis_x`, `axis_y`, `axis_y2`, `legend`, `title`, `grid`, `padding`, `color`, and each item of the `annotations` and `structural` arrays) refuses an unknown key at render, before deserialization: `chart config: unknown key '<k>' in <section>; accepted: <sorted list>`, or for an array item `chart config: unknown key '<k>' in annotations[0] (text); accepted: ...`. Every field the wire schema parses is recorded as honored or rejected in a completeness-tested manifest.

**Axis overrides.** `.override(x_axis_tick_size=...)` beats `configure_axis(tick_size=...)`. `.override(axis_label_format=...)` validates a preset name at the call; `axis_label_format_raw=` carries a raw spec. `.override(axis_x=...)`, `.override(axis_y=...)` and the `x_axis_x`, `x_axis_y`, `y_axis_x`, `y_axis_y` spellings raise `FerrumOverrideError` naming the replacement (`Chart.override: Unknown override path 'axis_x' (deprecated). Use Chart.axis(x=False) instead.`).

#### `Legend`

```
Legend(title, *, orient, direction, type=None,
       tick_count=None, tick_min_step=None, values=None,
       format=None, format_type=None,
       label_font_size=None, label_color=None, label_limit=None,
       symbol_size=None, symbol_stroke_width=None, symbol_type=None,
       gradient_length=None, gradient_thickness=None,
       columns=None, column_padding=None, row_padding=None,
       clip_height=None, title_font_size=None, title_padding=None,
       offset=None, padding=None, zindex=None)
```

| Parameter | Effective default | Meaning |
|---|---|---|
| `title` | the field name | Legend title. `None` suppresses the title. |
| `orient` | the theme's `legend_orient` (`"right"`) | Edge the legend sits on: `"right"`, `"left"`, `"top"`, `"bottom"`, or `"none"` to suppress it. |
| `direction` | implied by `orient` | Entry arrangement: `"vertical"` or `"horizontal"`. Side legends stack, top and bottom strips run across. |

The other fields default to `None` (renderer or theme value). `symbol_size` is the swatch area in px², on the mark `size` mapping (a circle swatch has radius `sqrt(symbol_size / pi)`); unset, zero or negative keeps the default swatch (radius 4 px). `format`/`format_type` follow the tick-label format rules under `Axis`.

`orient` and `direction` follow the same unset-versus-explicit rule as the `Axis` boolean fields: omitted or `None` is not serialized, any other value is. Both validate at construction on `Legend`, `LegendConfig`/`configure_legend` and raw legend dicts (`Legend.orient: orient must be one of ['bottom', 'left', 'none', 'right', 'top']; got 'middle'`). The two are independent: all eight combinations draw every entry, and `direction` also sizes the reserved block. A colorbar honors `direction`: `"horizontal"` draws a left-to-right gradient with tick labels beneath it, and a `"top"`/`"bottom"` colorbar with no explicit direction is horizontal.

**Suppressing a legend.** `legend=None` (or `False`) on a channel that owns a legend surface (`Color`, `Size`, `Shape` and the other appearance channels) suppresses that channel's legend, as does `Legend(orient="none")` or chart-level `configure_legend(orient="none")`.

**Entry values.** On a categorical legend `Legend(values=[...])` filters and orders the entries; a value that names no category is skipped with a warning (`legend values [z] match no legend entry and were skipped`). On a colorbar `values` replaces the tick labels.

**Positional channels.** `X` and `Y` have no legend of their own. An explicit `Legend(...)` or dict on `x`/`y` addresses the chart's legend and fills any field the `color` channel's `legend=` left unset; precedence is `color` > `x` > `y`. A dict that sets `{"disabled": True}` suppresses. Bare `legend=None` or `legend=False` on `x`/`y` is a no-op and leaves every legend as it is.

**Precedence.** Per-channel `Legend` fields (including `orient`, `columns` and `title_font_size`) beat `configure_legend(...)`. `configure_legend(label_font_size=)` sizes legend labels only; axis tick labels take `configure_axis(label_font_size=)`. `configure_legend()` has no `format` field; colorbar formatting is per channel.

**Same-field legends.** When `shape` or `stroke_dash` encodes the same field as a categorical `color`, one merged legend is drawn, and its entries show the folded channel: a dash-carrying entry draws a dashed line swatch, a shape-carrying entry draws its glyph. Identities are matched to entries by label, so sort order cannot transpose them. If only the folded channel states an explicit title, the legend uses it; if both state different titles, the color title stands and a warning names the discarded one. When `color`, `shape` and `stroke_dash` all share a field, the swatches show the shape, the dash patterns are not drawn, and a warning says so. When the color legend does not enumerate the field (it is disabled, or it is a colorbar), the shape or dash block renders on its own. `size` never folds into a categorical color legend, because `size` on a string column is refused (`size: column 'g' has unsupported dtype: Utf8`).

#### `Chart.axis()`

```
Chart.axis(*, x=None, y=None, show=None) -> Chart
```

Hides or shows the chart's x and/or y axis line, ticks, tick labels and title at layout time, and returns a new `Chart`. `x=False`/`y=False` hides one axis; `show=False` hides both and is exclusive with `x`/`y` (`ValueError: Chart.axis: pass either show=… OR x=/y=, not both`). `None` leaves the current setting. A hidden axis reserves no gutter (see **Gutters follow drawn chrome**). `Chart.axis()` wins over every per-channel axis setting, including `axis=None` and an `Axis(...)` that asks for labels or a title. `clustermap()` dendrogram panels and `JointChart` marginals use it. It serializes to the optional booleans `axis_x`/`axis_y` on `ChartSpec`, absent when unset (visible).

#### Chart-level Padding: `configure_padding()` / `PaddingConfig`

```
Chart.configure_padding(*, top=None, right=None, bottom=None, left=None, auto: bool | None = None) -> Chart
PaddingConfig(top=None, right=None, bottom=None, left=None, auto=None)
```

Sets the minimum pixel padding on each of the plot's four sides, independent of the theme's flat `padding` (16 px by default, §3.13). Each side is optional and independent: an unset side falls back to the theme value, and a set side always wins on that side, whether or not `auto` is on.

| Parameter | Effective default | Contract |
|---|---|---|
| `top`, `right`, `bottom`, `left` | unset (the theme's `padding`) | A finite, non-negative pixel value (`int` or `float`). |
| `auto` | unset (built-in default: on) | `True` or `False`; `None` means unset. |

**Validation.** Every padding value is checked when the `PaddingConfig` is constructed (`PaddingConfig.__post_init__`), the one validator; `configure_padding(...)` builds a `PaddingConfig`, and `.override(padding_top=..., ...)` builds one from the `padding_*` keys named in that call. `PaddingConfig.to_dict()` only serializes, omitting unset fields (`PaddingConfig().to_dict() == {}`), so an unset `auto` never reaches the wire.

- A side that is negative, non-finite, a `bool`, or any non-`Real` value (including a `NormCoord`: these are pixel slots) raises `ValueError` naming the side, for example `padding.top must be non-negative; received -1`, `padding.top must be a finite numeric pixel value; received nan`, `padding.top must be a numeric pixel value (int or float), not bool; received True`.
- A set `auto` must be `True` or `False` (`numpy.bool_` is accepted). Anything else, including `1` and `0`, is refused, never read by truthiness. `configure_padding(auto=)` and `PaddingConfig(auto=)` raise `ValueError: padding.auto must be True or False; received 'yes'` at the call. `.override(padding_auto="yes")` raises `FerrumOverrideError: Chart.override(padding_auto=...) must be True or False; received 'yes'` at render, when overrides are applied, and a bad side through the override raises in the same voice (`Chart.override(padding_top=...) must be non-negative; received -1`). `FerrumOverrideError` is a `ValueError`. `.override(padding_auto=None)` is no override.

**Layering.** `configure_padding` calls merge field by field: a later call that sets only sides keeps the `auto` an earlier call set, so `configure_padding(auto=False).configure_padding(left=5)` stays off and `configure_padding(auto=True).configure_padding(left=5)` stays on. A composite parent's `auto` reaches a leaf whose own `auto` is unset. When no layer sets `auto`, the built-in layout default applies, which is on; it lives only in the Rust layout default (`ThemePadding::default().padding_auto`) and has no theme key. `fm.compute_layout` takes no `auto` argument and runs with the built-in default. The render path and the chrome path share one configure-layer merge.

**`PaddingExceedsViewport`.** A padding (explicit or the theme default) that leaves no usable plot area on an axis is refused at render with the caller's value and the side that overflowed: `padding {side}={padding} exceeds viewport dimension {viewport_dim}`, `side` one of `"top"`, `"right"`, `"bottom"`, `"left"` (for example `configure_padding(top=1e9)` on a 200 px tall chart reports `padding top=1000000000 exceeds viewport dimension 200`). A non-finite viewport is refused before layout, with `auto` on or off: `properties(width=float("nan"))` is refused at the call, and `compute_layout` refuses `invalid viewport: width=NaN, height=200 (both must be finite and > 0)`.

**What `auto` does.**

When `auto` is on, a side that is not set explicitly may be widened beyond the theme padding so that two things stay inside the canvas:

1. **The drawn x tick labels of a continuous x axis.** `auto` measures the labels the axis actually draws, after its collision cascade has chosen the angle, font size and thinning (or at an explicit `label_angle`, on a bottom or top axis), at each label's drawn angle and anchor and at its tick's real pixel (the scale insets end ticks from the plot edge). A label that overhangs a side's canvas edge by `o` px asks for `o / lever` px on that side, where `lever` is how far its tick moves per pixel of that margin, measured by laying the plot out with the margin 1 px wider; on a faceted chart a tick moves by its panel's share of the region. Each side takes the largest charge over every drawn label of every panel. A label whose lever is below 1e-9 (a tick pinned at the far end) charges nothing, and so does a label whose charge exceeds the room its panel has before reaching its floor (below); such a label neither caps its own side nor reduces the other side's extra. The top side is also widened by half a label line when a flat, continuous y axis's top label would reach past the canvas.
2. **A clipped axis title.** Titles are centered on the plot, so `auto` recenters them with an asymmetric margin, solved after the tick-label pass against the plot that pass leaves.

Both passes credit what the chart already reserves on a side: the legend gutter, the chart title band and the secondary-y bands. Because a legend's reservation depends on the padding, the extra is solved as a bounded fixed point: each pass lays the chart out at its candidate padding and re-measures the reservations and the drawn labels, accepting once both agree within 1e-6 px, for at most 64 passes. A plain chart settles in two passes. When the measurement never settles, the last pass whose padding covered what it measured ships. When no pass covered, the per-side maximum of the extra each pass needed ships, unless that leaves the plot unusable, in which case `auto` adds nothing.

`auto` never acts on: an explicitly set side; annotations; the tick labels of a band, point or ordinal axis (they sit inside their bands); a hidden axis (`Chart.axis(x=False)` / `(y=False)`); an axis whose title is wider than the viewport, on which `auto` adds nothing at all, tick-label rescue included. A y axis capped with `max_band` keeps its cap: `auto` counts the capped gutter as the cushion and never widens it, so a tight cap can still leave that axis's label or title clipped. Charts whose labels and titles already fit lay out exactly as with `auto=False`.

**What `auto` may cost.**

`auto` judges every candidate padding against the `auto=False` layout of the same chart, and declines a rescue rather than trade the chart for it:

- **Half the plot.** Every panel keeps at least half of its `auto=False` width and half of its height. A single label or title whose rescue needs more adds nothing on that side. Rescues on one axis that each fit but together need more are not scaled down to a partial rescue: that axis adds nothing.
- **No more lost x labels.** An x axis loses a tick when the tick has a generated label and the axis does not draw it in full (culled, or cut short with an ellipsis); an axis with hidden labels loses nothing. A padding is unusable when it lays out a different number of x axes than `auto=False`, or when any x axis loses more of its own ticks than it loses at `auto=False`. Only the count matters, not which ticks are labelled. One exception: on a continuous x axis, an even thinning (the drawn labels are exactly every `s`-th labelled tick from the first, none cut short, at least three drawn) is not a loss. Thinning to two labels, an irregular cull, an ellipsis or tick marks only remain losses.
- **No collapse.** A padding that leaves a panel at or below the 1 px minimum is never accepted, so `auto` never blanks a chart that `auto=False` draws, and never turns a chart that renders into a `PaddingExceedsViewport` refusal. When the fixed padding alone already leaves no room on an axis, `auto` adds nothing there.
- **Each axis on its own.** When no candidate padding is usable, each axis's rescue is judged alone, with the other axis's extra removed. An axis whose rescue alone is unusable is dropped and the other axis's valid rescue still ships. When both fail, or each is valid alone but together they fail, `auto` adds nothing and the chart lays out exactly as with `auto=False`.

**The x label band.**

The x axis reserves the band its labels are drawn in, from the same label layout the axis draws (tick pixels, per-tick budget, cascade or explicit angle), at the width the axis is drawn at: after the y and secondary-y bands, at the facet cell width, inside a composite's imposed region, and, for independent-axis facets, from each panel's own ticks. Rotated labels reserve the widest drawn label at its drawn angle; flat or wrapped labels reserve the standoff plus 1.2 times the font size per further line. Under `CoordFixed`, the band is solved as a bounded fixed point at the panel the aspect leaves. The x title clears the last drawn label line by the axis title padding, on bottom and top axes alike, and a top axis stacks its wrapped label lines away from the plot.

The band has a height budget: when the labels the cascade would draw leave a panel at or below the 1 px minimum, the axis culls labels by stride until the band fits, and when no stride fits it draws tick marks only. The budget applies under `Theme(cull_labels=False)` too; `label_overlap="true"` keeps every label and is never budgeted. This guarantees a drawable plot, not a roomy one.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#37-axes-and-legends).

---

### 3.8 Coordinate Systems

A coordinate system is set with `Chart.coord(coord)`, which returns a new chart. `mark_*(orient="horizontal")` sets `CoordFlip` the same way. The classes live in `ferrum.coord` and are exported from `ferrum`.

| Class | Description | Key Parameters |
|---|---|---|
| `CoordCartesian(xlim=None, ylim=None, expand=True, clip=True)` | Default Cartesian coordinates. `xlim`/`ylim` set an explicit `(min, max)` domain for that axis; `expand` pads the domain around the data; `clip` clips marks to the plot area. | `xlim`, `ylim` |
| `CoordFlip()` | Exchanges the x and y roles: the column on `x` is drawn on the vertical axis and the column on `y` on the horizontal one. | |
| `CoordPolar(theta="x", start=0.0, direction=1, inner_radius=0.0, outer_radius=None, pad_angle=0.0)` | Polar coordinates (pie, donut, radial charts). | `theta`, `inner_radius` |
| `CoordGeo(projection="mercator")` | Geographic map projection. | `projection` |
| `CoordFixed(ratio=1.0, xlim=None, ylim=None, expand=True, clip=True)` | Cartesian coordinates with a fixed panel aspect ratio. | `ratio` |

- **`CoordCartesian` and `CoordFixed`.** `clip=False` widens the clip region from the plot area to the whole viewport, so a mark that extends past the plot area is drawn there. A point whose position lies outside the domain is skipped whether or not `clip` is set.
- **`CoordFixed(ratio=)`** is the plot area's width divided by its height in pixels. The layout shrinks the binding dimension of the plot area until `width / height == ratio` and centers it, so `ratio=1.0` draws a square plot area whatever the data ranges are. It does not equalize data units per pixel. A non-positive ratio leaves the plot area unchanged.
- **`CoordPolar`.** `theta` names the channel (`"x"` or `"y"`) mapped to the angle; the other channel is the radius. `start` is the starting angle in radians, with `0` at 12 o'clock. `direction` is `1` for clockwise or `-1` for counter-clockwise; any other value, including the strings `"clockwise"` and `"counter_clockwise"`, raises `ValueError`. `inner_radius` and `outer_radius` are pixels: `inner_radius > 0` cuts a donut hole, and `outer_radius=None` fills to half the smaller plot-area dimension. `pad_angle` is the gap between adjacent slices in radians.
- **`CoordGeo(projection=)`** takes one of `"mercator"`, `"albers_usa"`, `"equal_earth"`, `"natural_earth"`, `"orthographic"` or `"equirectangular"`; any other name raises `ValueError` listing the six.
- **Projected coords draw no Cartesian axes.** Under `CoordPolar` the panel draws a circular boundary with radial ticks, and under `CoordGeo` it draws no axes at all. Neither reserves a Cartesian tick-label band or title gutter.
- **Marks that need a specific coord.** `mark_arc` under any coord other than `CoordPolar` raises `ValueError: mark_arc requires a Polar coord`. `mark_geoshape` with a `__geometry__` column under any coord other than `CoordGeo` raises `ValueError: mark_geoshape requires a Geo coord`; without a geometry column it renders an empty scene under any coord. `mark_image` under `CoordPolar` or `CoordGeo` draws nothing and warns `image layer N drew nothing: mark_image() does not support CoordPolar; ...` (or `CoordGeo`).

**`CoordFlip` semantics.** Every channel is authored in unflipped space, and the flip is applied once:

- A sort shorthand names the channel it was written against: `fm.X("c", sort="-y")` orders the categories by descending `y` with or without `CoordFlip`, as `sort=fm.SortField(...)` does. The same holds for the `shape`, `stroke_dash` and `color` categorical domains. A sort spec that cannot be applied warns once.
- A chart-level encoding is moved into flipped space before it is merged with each layer's, so the marks and the axis read one scale.
- An area or density series closes its fill on its own value baseline. `mark_density(orientation="horizontal")` declares its value axis, and the declaration composes with `CoordFlip` (the two cancel), so a flipped horizontal density draws as a vertical one.
- A flipped `mark_ribbon` (and everything built on it: `mark_errorband`, `mark_smooth(ci=)`, the learning and validation curve bands, the QQ confidence region) draws its band along the flipped axis.
- A declared stack value axis, `fm.Stack(value_axis=)`, is authored in unflipped space and moves with the flip: `Stack(value_axis="x")` under `CoordFlip` stacks on y, and `Stack(value_axis="y")` stacks on x, as the undeclared default does.

**Coords on layered charts.** Each member of `chart + chart`, `fm.layer(...)` and `LayerChart` keeps its own coord, and that coord places the member's marks. The panel's chrome (Cartesian axes or the polar boundary), its aspect ratio and its scale domains come from the first member's coord. `fm.layer(bar, arc.coord(fm.CoordPolar(theta="x")))` therefore draws the arc member's wedges on both `to_svg()` and `.interactive()`, and a member with `CoordFlip` draws flipped. A per-layer coord may declare placement only (`CoordFlip`, `CoordPolar`, `CoordGeo`, or a bare `CoordCartesian()`). A layered chart raises one typed `ValueError` naming the layer and the coord when a member's coord cannot be placed on the shared panel:

- it declares a panel property: `xlim=`/`ylim=`, `expand=False`, `clip=False`, or `CoordFixed(ratio=)`. This refuses on `chart + chart`, `.interactive()` and `resolve={"y": "independent"}`.
- its `CoordFlip` lands a column on a shared axis that cannot draw it, for example a string column on the panel's continuous y axis: `layer 1: fm.CoordFlip leaves this layer's positional channels in the opposite space from the panel's axes, and the panel's y axis is continuous while the column that lands on it ('c': Utf8) cannot be drawn against a continuous axis — ...`. This refuses on `chart + chart`, `.interactive()` and a static `LayerChart` with shared y (whose message is prefixed `composite scale resolution failed: failed to resolve shared y domain for field 'c': `). It renders on `resolve={"y": "independent"}`, whose member resolves its own y axis. The same composition over an integer category column renders, since a continuous axis can draw it. A member's flipped ordinal channel widens the band axis it lands on.

The `chart + chart` merge warns `Layered chart `+`: secondary layer's theme/facet is ignored; primary layer wins.` when a secondary member carries its own theme or facet; it does not drop the member's coord.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#38-coordinate-systems).

---

### 3.9 Faceting

#### `Chart.facet()`

A faceted chart repeats one chart across the values of one or two fields. It is declared on the chart itself with `Chart.facet()` or with the facet encoding channels (`Facet`, `FacetRow`, `FacetCol`, §3.2); there is no separate facet-spec object.

```
Chart.facet(field=None, *, row=None, col=None, ncols=None, nrows=None, title=None, row_title=None, sort=None, row_sort=None)
```

| Parameter | Default | Meaning |
|---|---|---|
| `field` | unset | Wrap mode: one field, its panels wrapped into rows. |
| `row`, `col` | unset | Grid mode when both are given: `row` values down, `col` values across. One alone is wrap mode on that field, unless it merges onto a prior facet of the other orientation (§3.2, One writer of facet state). |
| `ncols` | unset | Columns in wrap mode; rows follow from the panel count. Unset lays a `col=` wrap out as one row and a `field=` or `row=` wrap as one column. In grid mode it caps the columns, and unset uses the `col` field's distinct-value count. |
| `nrows` | unset | Grid mode only: caps the rows, and unset uses the `row` field's distinct-value count. Panels beyond `nrows * ncols` are dropped with a warning. A wrap facet ignores `nrows`. |
| `title` | unset | Title of the wrap strip, or of the column strip in grid mode. A strip reads `"{title}: {value}"`, or the bare value when unset. |
| `row_title` | unset | Title of the row strip in grid mode. On a single-dimension declaration it raises `ValueError`. |
| `sort`, `row_sort` | unset | Panel order for the strip `title` / `row_title` names, in the positional `sort` vocabulary. |

Strip titles, wrap count, sort and the merge rules for repeated facet declarations are stated with the facet channels in §3.2.

**Panel axes.** Every panel draws the x and y axes the layout lays out for it, on shared and independent facets alike:

- The x axis title is drawn only under the bottom row, and the y title only beside the left column. A panel in a short wrap with no panel below it draws its x axis but no x title. A 2x2 facet draws two of each title.
- Hiding an axis with `Chart.axis(x=False)` or `axis(y=False)` hides it on every panel, shared or independent.
- An explicit `BandScale(range=...)` or `PointScale(range=...)` axis is offset per panel, so its ticks and labels sit on that panel's marks, including under `CoordFlip` and on a secondary y axis.
- A collapsed panel draws no axis.

**Scale sharing.** A facet's positional scales are shared by default: every panel uses the union domain. `Chart.share_scale(x=, y=)` (§3.1) sets `"shared"` or `"independent"` per channel, and raises `ValueError` (`share_scale() requires a faceted chart; call facet() first`) on an unfaceted chart.

#### Repeated charts

A chart repeated across a list of fields (a pairwise matrix, for example) is a `RepeatChart` (§3.12): the template's `.encode()` marks the substituted channel with the `fm.Repeat.row`, `fm.Repeat.column` or `fm.Repeat.layer` sentinel, and `RepeatChart(template, row=[...], column=[...])` expands one panel per field combination. `ferrum.pairplot` builds one. There is no `Chart.repeat()` method.

#### Resolution

Compositions (§3.12) take `resolve=`, either a flat `{channel: mode}` dict or a `Resolve` value. A facet uses `Chart.share_scale()` instead.

```
Resolve(scale=None, legend=None)
```

| Parameter | Default | Meaning |
|---|---|---|
| `scale` | unset | `{channel: mode}`, each mode `"shared"` or `"independent"`, for `"x"`, `"y"`, `"color"`, `"size"`. `"shared"` unions the channel's domain across the composition's panels. A flat dict passed as `resolve=` means the same as a `Resolve` whose `scale` is that dict. |
| `legend` | unset | `{channel: mode}` for `"color"` and `"size"` only. An unlisted channel follows its scale mode. |

- **Validation.** A mode other than `"shared"`/`"independent"`, or a legend channel other than `color`/`size`, raises `ValueError` when the composition is built, e.g. `HConcatChart: resolve.scale['x'] must be one of ['independent', 'shared']; got 'sharedd'`. A scale channel outside `x`/`y`/`color`/`size` marked `"shared"` raises at render: `resolve= marks 'opacity' 'shared', which the composite resolve pass does not support (supported: ('x', 'y', 'color', 'size')); set it 'independent' or drop it from resolve=`. `legend={ch: "shared"}` over a scale that is not shared raises at render: `resolve.legend['color']='shared' requires resolve.scale['color']='shared' (got scale='independent', legend='shared'); a shared legend needs a unioned domain to dedup from`.
- **Shared legends.** A composition whose `color` or `size` scale resolves shared draws one figure-level legend outside the panel grid and suppresses the per-panel legends. `legend={"color": "independent"}` keeps the per-panel legends over the shared scale. `pairplot(hue=)` and `jointplot(hue=)` draw the shared legend by default.
- **Nested compositions (`color`, `size`).** A node's effective mode for a channel is its own `resolve=` setting when it states one, and otherwise the nearest ancestor's. A node whose effective mode is `"shared"` unions the domain across every leaf below it, through nested composites and layered children. An explicit `"independent"` opts its whole subtree out, and a node re-shared beneath it starts a separate union. The figure legend attaches at the outermost shared node, so an outer-shared tree, a nested-and-outer-shared tree and a layered chart inside a shared concat each draw exactly one legend.
- **Nested compositions (`x`, `y`).** Positional sharing does not inherit: it pairs congruent direct children by tree position (a grid pairs panels by column).
- **Turning legends off.** `configure_legend(orient="none")` suppresses legends through the same mechanism as `Color(legend=None)`: on a single chart it removes the legend and its gutter, and on a composition it disables every legend the node covers, including the figure-level legend.
- `Resolve` has no `axis=` member.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#39-faceting).

---

### 3.10 Selections (Interactivity)

Selections are interactive state. A chart declares them, and the WASM renderer (`.interactive()`, `to_html()` on an interactive chart) resolves them. Static output (`to_svg`, `to_png`, PDF) ignores selections and conditionals: marks draw their base encoding, never a conditional branch.

| Class | Description | Key Parameters |
|---|---|---|
| `selection_point(*, fields=None, encodings=None, nearest=False, toggle="event.shiftKey", on="click", clear="mouseout", resolve="global", name=None, bind=None)` | Click selection of one or more marks | `bind="legend"` makes legend entries toggle the selection |
| `selection_interval(*, fields=None, encodings=None, translate=True, zoom=True, mark=None, resolve="global", name=None)` | Drag (brush) selection of a rectangular region | `mark=` styles the brush |
| `selection_single(*, fields=None, encodings=None, nearest=False, toggle="false", on="click", clear="mouseout", resolve="global", name=None, bind=None)` | `selection_point` whose click always replaces the selection | |
| `selection_multi(*, fields=None, encodings=None, nearest=False, toggle="event.shiftKey", on="click", clear="mouseout", resolve="global", name=None, bind=None)` | `selection_point` with shift-click adding to the selection | |
| `SelectionMark(fill=None, stroke=None, fill_opacity=0.3, stroke_opacity=1.0, stroke_width=1.0, stroke_dash=None)` | Style of the interval brush rectangle | |

**Declaring a selection.**

- `name` is the selection's identity; when omitted, a random `sel_<8 hex>` name is generated. `encodings` entries are lowercased. `fields`, `encodings` and the event arguments are stored as given.
- `on`, `clear` and `toggle` take `"click"`, `"mouseover"`, `"mouseout"`, `"dblclick"` or `"event.shiftKey"`; `"false"` serializes as `"click"`, and any other string serializes as a custom event expression.
- Only a shift-click on a selection whose `toggle` is `"event.shiftKey"` adds to or removes from the selection. Every other click replaces it, and a click that hits no mark empties a point selection.
- A point selection with `fields` selects every mark whose values in those fields equal the clicked mark's (numerically equal values match).
- `resolve` must be `"global"`, `"union"` or `"intersect"`, and each `encodings` entry must be one of `x`, `y`, `color`, `size`, `shape`, `opacity`, `stroke_width`, `stroke_opacity`, `stroke_dash`, `fill_opacity` or `angle`. Neither is checked at construction: another value is refused when the chart renders, with the wire's message (`selections: unknown variant 'bogus', expected one of 'global', 'union', 'intersect'`).
- On a point selection, `bind="legend"` makes each legend entry a click target that toggles its category in the selection. No other `bind` value has an effect.
- The brush style passed as `selection_interval(mark=...)` is serialized with the selection. `fill` and `stroke` take the color vocabulary of mark construction (a CSS name, hex, `rgb()`/`rgba()`); an unparseable string raises `ValueError` (`SelectionMark: fill='nonsense' is not a valid color (...)`), and `"none"`/`"transparent"` are refused because the selection wire has no cleared paint (`SelectionMark: fill='none': selection styling cannot express a cleared paint ('none'/'transparent'); provide a color`). `stroke_dash` takes the `stroke_dash` vocabulary of marks (a list, a comma string or a named linetype) and refuses anything else at construction. The interactive runtime draws the brush in a default translucent blue (`rgba(51, 136, 204, 0.2)` fill, `rgba(51, 136, 204, 0.6)` stroke) and does not yet apply `SelectionMark`: its `fill`/`stroke` reach the page as color objects the brush overlay cannot use, and its opacities, width and dash are not read.

**Registering a selection.** `Chart.add_selection(*selections)` registers selections; it refuses anything that is not a `Selection` with `TypeError` (`add_selection() expects Selection instances ... Did you mean add_params()?`). `Chart.add_params()` registers `fm.param()` variables and also registers any selection passed to it. A conditional's selection is registered automatically by `encode()`, `Chart.conditional()` and `Chart.layer()`, so no separate `add_selection()` is needed.

**One name is one selection.** Two different definitions under one name are refused, naming the differing fields. Definitions are compared by their wire form, so `fields=["x"]` and `fields=("x",)` are one definition.

- Within one chart, at the declaring call (`add_selection()`, `add_params()`, `encode()`, `Chart.layer()`, `Chart.conditional()`, `chart + chart`): `add_selection(): selection 's' is declared twice with different definitions: first declares type='point', second declares type='interval'. One name is one selection on this chart: pass the same Selection object to every call that declares 's', or give the second one its own name.`
- Across a composition, at render: `hconcat: selection 's' is declared twice with different definitions: chart 1 of hconcat declares nearest=False, chart 2 of hconcat declares nearest=True. One name is one selection across the composition: share one selection between the charts, or give each its own name.`
- A selection declared on several charts under one name with one definition (linked views) is one scene selection that owns the panels of all of them.

**Input scope.** A selection reacts to clicks and brushes only in the panels of the chart that declared it: every facet panel of that chart, and an overlay group as one owner. On the scene wire each selection carries `panels`, its owner panel ids, omitted on a one-panel scene.

**Using selections in encodings.**

A conditional is built with `sel.when(if_selected).otherwise(if_not)` or `fm.when(sel).then(if_selected).otherwise(if_not)`. `fm.when()` takes a selection only; an `fm.param()` variable raises `TypeError`. There are four entry points, and all resolve through one conditional vocabulary:

```
chart.encode(opacity=fm.when(sel).then(1.0).otherwise(0.2))
chart.encode(color=Color("category", type="N", condition=sel.when(value("red")).otherwise(value("#aaa"))))
chart.conditional(sel.when(value("red")).otherwise(value("#aaa")))
chart.layer(Layer(mark="point", encoding={"opacity": fm.when(sel).then(1.0).otherwise(0.2)}))
```

- The encode form takes the wire channel from the encode key.
- The `condition=` form keeps the base field encoding and adds the conditional on the encode key's channel. A `condition=` that is not a `ConditionalSpec` raises `TypeError` naming both forms.
- `Chart.conditional()` takes one pre-built `ConditionalSpec`. It reads the channel from the branches: an encoding-channel branch on either side wins; otherwise the first literal names it, a number meaning `opacity` and a string `color`.
- The `Layer` form scopes the conditional to that layer.
- A conditional may target `color` (and `fill`, which resolves to `color`), `size`, `opacity`, `stroke_width`, `stroke_opacity`, `stroke_dash`, `fill_opacity` and `angle`. Any other channel is refused with `ValueError` at the declaring call (`condition= is not supported on the 'x' encoding channel; a selection conditional can target only: ...`). A wire channel carries at most one conditional; a second is refused naming both declarations.

**Branch values.**

- A branch is an encoding channel object, a bare string, `fm.value(literal)`, or `fm.param(...).ref()`. A bare string is a field in shorthand grammar (`"category:N"`), never a literal.
- A field branch (a channel object or a bare string) keeps the mark's own encoded value for that channel; it does not encode the field. To color marks by a field and grey out the rest, encode the field on the channel itself (`Color("category", condition=...)`) or as the base encoding. A field branch on `encode(<channel>=cond)` alone draws a legend for the field while every mark keeps the default color.
- A literal is typed by the conditional's channel. `color` takes a color string only (a number is refused). `stroke_dash` takes a named linetype, a comma string, a number or a list, so `fm.value("dashed")`, `fm.value("4,2")` and `fm.value([4, 2])` serialize the same pattern. `size`, `opacity`, `stroke_width`, `stroke_opacity`, `fill_opacity` and `angle` take a finite number only, so a string (including `"0.5"`), `bool`, `nan` and `inf` are refused. Every refusal is one `ValueError` shape naming the call and the channel: `fm.value('red') for channel='size' needs a number, got str: 'red'`. A color string that does not parse, and `"none"`/`"transparent"`, are refused as in `SelectionMark`.
- A `stroke_dash` pattern outside the interactive dash palette is accepted and draws solid interactively.
- A `.ref()` branch resolves to the named `fm.param()`'s current static value when the chart serializes. The parameter must be registered with `add_params()`. An unregistered name, a name registered as a selection, or a parameter with no `value` raises `ValueError` naming the parameter. A ref is resolved once, statically; the runtime does not re-resolve it when the parameter changes.
- A ref behaves like the parameter itself at the other two sites that take one: a scale `domain=` (it emits `domainParam`, and needs `add_params()` to reach the wire) and `fm.transform_filter()` (`transform_filter(q.ref()) == transform_filter(q)`).

**Conditional scope.** A conditional restyles only the marks of the chart or `fm.Layer` that declared it; a chart-level conditional (a `.layer()` host's, `fm.SecondaryY`'s) reaches every layer attached to it. Linked views declare the conditional on each chart over one shared selection. On the scene wire each `MarkBatch` lists the conditionals it answers to as `conditionals` (indices into `interaction.conditionals`), omitted when empty. An empty selection applies no conditional: every mark keeps its base encoding until something is selected.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#310-selections-interactivity).

---

### 3.11 Annotations

Annotations are overlays placed in data, normalized or pixel coordinates. Except for the unlabelled `annotate_arrow` (below), they do not take part in scale domain calculation: an annotation outside the data's domain is not drawn into view.

**Coordinates.** Every positional argument accepts a number, a `datetime.date`/`datetime.datetime` or ISO-8601 string (converted to epoch milliseconds, UTC, to match a temporal axis), a category string on an ordinal axis, or a wrapper: `fm.NormCoord(v)` (also `annotation.norm(v)`) is a fraction of the plot area with `0.0` at the left/top edge and `1.0` at the right/bottom edge, and `fm.PixelCoord(v)` (also `annotation.px(v)`) is a pixel offset.

#### `annotate_*` helpers

Each `annotate_*` function in `ferrum` returns a small `Chart` that renders on its own and composes with `|`/`&`. Added to a chart with `+`, it contributes an annotation overlay instead of its mark layer, so the base chart's layers, data and domains are unchanged.

| Function | Description | Key Parameters |
|---|---|---|
| `annotate_hline(y, *, label=None, stroke=None, stroke_dash=None)` | Horizontal reference line across the full plot width. | `label` |
| `annotate_vline(x, *, label=None, stroke=None, stroke_dash=None)` | Vertical reference line across the full plot height. | `label` |
| `annotate_rect(x1, x2, y1, y2, *, fill=None, opacity=0.1, label=None)` | Shaded rectangle. | `fill`, `opacity` |
| `annotate_text(x, y, text, *, dx=0, dy=0, anchor=None, align=None, baseline="middle", font_size=None, color=None, angle=None)` | Free text. | `anchor` |
| `annotate_arrow(x1, y1, x2, y2, *, label=None, label_side="start", stroke=None)` | Arrow from `(x1, y1)` to `(x2, y2)`. | `label`, `label_side` |
| `annotate_abline(slope, intercept, *, stroke="black", stroke_width=1.0, stroke_dash=None, opacity=1.0)` | The line `y = slope * x + intercept`, clipped to the plot area. | `slope`, `intercept` |

- **Reference-line labels.** `annotate_hline(label=)` draws the text at the line's right end, inset 6 px from the right edge and 6 px above the rule, right-aligned. `annotate_vline(label=)` draws it at the line's top, 6 px below the top edge and 6 px right of the rule, left-aligned. The position is fixed; neither function takes a position argument. The label renders standalone and under `+`.
- **Line style.** `stroke` defaults to the mark's themed color standalone and to `#333` in the `+` overlay. `stroke_dash` takes a list of numbers, a comma-separated string such as `"4,4"`, or a named linetype (`"solid"`, `"dashed"`, `"dotted"`, `"dashdot"`, `"longdash"`); `annotate_hline`, `annotate_vline` and `annotate_abline` normalize it once, so the standalone mark and the overlay draw the same dash.
- **`annotate_rect`.** The overlay fill defaults to `#cccccc`. `label` is accepted and draws nothing.
- **`annotate_text` anchor.** `anchor` is the canonical keyword in the SVG vocabulary (`"start"`, `"middle"`, `"end"`, also accepting `"left"`, `"right"` and `"center"`). `align` is a deprecated alias in the `"left"`/`"center"`/`"right"` vocabulary, mapped `{left: start, center: middle, right: end}`. With neither, the text is centered. Passing both raises `ValueError: annotate_text() received both 'anchor' and 'align'; pass only one ('anchor' is the canonical SVG vocabulary, 'align' is the alias).` An out-of-vocabulary value raises `ValueError` naming the keyword the caller wrote. `baseline` is `"top"`, `"middle"` or `"bottom"`. Overlay defaults: `font_size` 12, `color` `#333`.
- **`annotate_arrow`.** A filled 8 px triangle head is drawn at `(x2, y2)` on every path. With `label=`, the text sits at `(x1, y1)` for `label_side="start"` (6 px left, right-aligned) and at `(x2, y2)` otherwise (6 px right, left-aligned); under `+` the arrow and its label are an annotation overlay with a `#333` default stroke and width 1.5. Without a label, `+` merges the arrow's `mark_segment` layer into the chart, so it keeps the themed stroke and its endpoints widen the chart's domains.
- **`annotate_abline`** is a two-point line spanning x in `[-1e6, 1e6]`, clipped by the renderer.

#### Annotation primitives

`ferrum.annotation` holds the primitive factories, each returning a frozen value that `chart + primitive` (or `chart + fm.Annotate([...])`) appends to the chart's annotations:

| Factory | Draws |
|---|---|
| `annotation.text(x, y, text, *, font_size=12, color="#333", anchor="start", baseline="middle", angle=0, dx=0, dy=0, z="above_marks")` | Text. `z` is `"above_marks"` or `"below_marks"`. |
| `annotation.line(x1, y1, x2, y2, *, stroke="#333", stroke_width=1, dash=None)` | A line segment. |
| `annotation.rect(x1, y1, x2, y2, *, fill, opacity=0.1, stroke=None, corner_radius=0)` | A rectangle. |
| `annotation.span(axis, start, end, *, fill, opacity=0.3, label=None, label_position="top")` | A full-height (`axis="x"`) or full-width (`axis="y"`) band between `start` and `end`. |
| `annotation.arrow(x, y, x2, y2, *, stroke="#333", stroke_width=1.5, head_size=8)` | An arrow with a triangle head at `(x2, y2)`. |
| `annotation.bracket(x1, y1, x2, y2, *, label, direction="above", stroke="#333", tip_length=6)` | A labelled bracket. |
| `annotation.callout(x, y, text, *, text_x=None, text_y=None, arrow="curved", padding=4, background="#fff", border_color="#ccc", border_radius=3)` | Boxed text with a leader to `(x, y)`. |
| `annotation.image(x, y, src, *, width=50, height=50, anchor="center")` | An image. |

**Span labels.** A span's `label` is centered horizontally on the span. Its vertical place is the span's `label_position` keyword: `"top"` (the default) hangs the text 6 px below the span's top edge, `"bottom"` sits it 6 px above the bottom edge, and `"middle"` centers it. For an `axis="x"` span, which runs the full plot height, `"top"` is the top of the plot area. Any other value raises `ValueError` at construction.

#### Metric and outlier labels

These are added with `chart + Label(...)` and read the chart's own `x`/`y` (and `color`) encoding. Each raises `ValueError` when the base chart lacks an `x` or `y` encoding or the named column is missing.

| Class | Description | Key Parameters |
|---|---|---|
| `AUCLabel(position="end", format=".3f", prefix="AUC = ")` | AUC per series on a ROC chart. | `position` |
| `APLabel(position="end", format=".3f", prefix="AP = ")` | Average precision per series on a PR chart. | `position` |
| `BrierLabel(position="corner", format=".3f", prefix="Brier = ")` | Brier score per series on a calibration chart. | `position` |
| `OutlierLabel(threshold=3.0, field=None, label_field=None, max_labels=10)` | Text labels on points whose z-score exceeds `threshold`. | `threshold` |

- **The metric.** The label reads the frame's broadcast metric column (`auc`, `ap` or `brier`) when the series carries exactly one value for it, which is the diagnostic kernel's number, so the label, title and legend agree. Otherwise `AUCLabel` integrates the drawn curve (trapezoidal) and `APLabel` takes the step-function area. `BrierLabel` has no curve definition and raises `ValueError` without the column; `ModelSource.calibration_curve()` frames and `calibration_chart()` carry it. A NaN metric renders as `nan`.
- **Placement.** One label per `color` group (null groups skipped), stacked downward from the top of the y range by 6% of the range. `position="end"` anchors each at its series' rightmost point; `"corner"` anchors every label at the overall maximum x. Any other `position`, or a `format` that cannot format a float, raises `ValueError` at construction naming the class.
- **Re-applying replaces.** Adding a metric label to a chart that already carries one, including a figure function's default label, replaces it, so `position=`/`format=` can be changed.
- **`OutlierLabel`** computes z-scores of `field` (default: the `y` column) with the sample standard deviation, keeps points whose absolute z-score exceeds `threshold`, labels at most `max_labels` of them in descending order, and takes the text from `label_field` when that column exists, otherwise from `field`. With no outliers it returns the chart unchanged.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#311-annotations).

---

### 3.12 Compound Views

A compound view arranges several charts into one figure. Every compound view renders through one Rust composite entry per output kind (`render_composite_svg` for static output, `render_composite_interactive` for interactive output). Python lowers the composition to a composite spec tree. Rust resolves the shared scales that `resolve=` asks for across the tree's leaves, plans the layout (ratio cells, and holes for empty corner, trailing or empty-data cells) and emits one scene. Static and interactive output share that layout. There is no fallback renderer: a composition that cannot lower raises a typed `ValueError`.

| Class / Operator | Description | Key Parameters |
|---|---|---|
| `HConcatChart(charts, *, spacing=10.0, resolve=None)` / `chart1 \| chart2` | Horizontal concatenation of a list of charts | `spacing`, `resolve` |
| `VConcatChart(charts, *, spacing=10.0, resolve=None)` / `chart1 & chart2` | Vertical concatenation of a list of charts | `spacing`, `resolve` |
| `ConcatChart(*charts, columns=None, spacing=10.0, resolve=None)` | Wrapping concatenation, `columns` charts per row (one row when `None`) | `columns`, `spacing`, `resolve` |
| `LayerChart(*charts, resolve=None, title=None)` | Overlay on one shared x coordinate space | `resolve`, `title` |
| `RepeatChart(template, *, row=None, column=None, layer=None, diagonal=None, corner=False, spacing=10.0, columns=None, resolve=None)` | Repeat a template across field lists | `row`, `column`, `layer`, `diagonal`, `corner`, `columns`, `resolve` |
| `JointChart(center, *, top=None, right=None, ratio=5, spacing=10.0)` | Center plot with top and right marginals | `ratio`, `spacing` |
| `ClusterMapChart(heatmap, *, row_dendrogram=None, col_dendrogram=None, dendrogram_ratio=0.2, spacing=10.0)` | Clustered heatmap with row and column dendrograms | `dendrogram_ratio`, `spacing` |

The functions `fm.hconcat(*charts, spacing=10.0, resolve=None)`, `fm.vconcat(*charts, spacing=10.0, resolve=None)`, `fm.concat(*charts, columns=None, spacing=10.0, resolve=None)` and `fm.layer(*charts, resolve=None, title=None)` build the same objects from positional charts. `chart1 | chart2` and `chart1 & chart2` work on charts and compositions alike; with a non-chart operand (`chart | 3`) Python raises its own operator `TypeError`. `Chart + Chart` returns a layered `Chart` (§3.1), not a `LayerChart`.

**Construction.** Every child must be a `Chart` or a composition; anything else raises `TypeError: <Class> children must be Chart or composition objects, got <type>` at construction. The optional slots (`top`, `right`, `diagonal`, `row_dendrogram`, `col_dendrogram`) accept `None`; the required ones do not. `ConcatChart()` and `LayerChart()` with no charts raise `ValueError` (`"ConcatChart requires at least one chart"`, `"LayerChart requires at least one chart"`), and `columns` must be greater than 0.

**Rendering an empty figure.** A compound view whose every child has zero rows raises `ValueError: <Class>: every child's data is empty; nothing to render`. A figure with at least one non-empty child renders, and an empty child becomes a hole.

**Figure chrome.** `.properties(title=, subtitle=, caption=)` on a concatenation, `RepeatChart`, `JointChart` or `ClusterMapChart` stores the text at the figure level, where it renders once around the whole figure and is never copied into a panel. Every other `properties` keyword (for example `width`, `height`) is forwarded to each child, except on `JointChart` and `ClusterMapChart`, which forward it to the center chart or the heatmap only. A `fm.Title` carrying its own `subtitle` routes it to the figure subtitle, unless `subtitle=` is also given in the same call, which is refused. A `fm.Title` carrying a style field the figure title cannot express (`anchor`, `offset`, `font_size`, `font_weight`, `color`, `subtitle_font_size`, `subtitle_color`) is refused. `LayerChart` takes its title as the `title=` constructor argument; it has no figure-subtitle slot, so a `fm.Title` with a `subtitle` or any style field is refused there. All compound views also accept `.theme()`, `.save()` and `.show()`.

**Spacing and grid sizing.** `spacing` is a gap in pixels between adjacent panels, default `10.0` on every compound view. A grid allocates each column a width of `K_w * col_ratio[c]`, where `K_w = min_c(native_width[c] / col_ratio[c])`, and rows the same way. The dominant cell renders at its native size; a cell whose native size differs from its slot is scaled into the slot, non-uniformly when the two aspect ratios differ. `JointChart(ratio=5)` gets its 5:1 center-to-marginal proportions this way. A `JointChart`'s marginals and a `ClusterMapChart`'s dendrograms take their size from the center chart or the heatmap (through `ratio` or `dendrogram_ratio`) at render time.

**Shared scales.** `HConcatChart`, `VConcatChart`, `ConcatChart`, `RepeatChart` and `LayerChart` take `resolve=` (a `{channel: "shared" | "independent"}` dict or a `fm.Resolve`), and `share_scale()` updates it; §3.9 Resolution states the rule. `JointChart` and `ClusterMapChart` have no `resolve=`: their panel alignment is fixed layout geometry, and `share_scale()` on them raises `ValueError: <Class>: share_scale requires a resolve= field, which <Class> does not carry (its panel alignment is fixed layout geometry, not a resolve= channel); construct a composition that supports resolve= (HConcat/VConcat/Concat/Repeat/Layer) instead`.

**Panels that annotate another panel.** A `JointChart` marginal and a `ClusterMapChart` dendrogram span exactly the extent of the panel they annotate on the axis they share, and keep their own extent on the other axis. The lowering declares which cells align (`extent_alignments` on the composite wire node, each `{follower, leader, axis}`), and the alignment is scaled by the follower's native size over the leader's because the grid scales each cell into its slot. A declaration naming a nested-composite cell (for example a `LayerChart` or `HConcatChart` as a jointplot's center) cannot be honored: it warns `grid cell <f> cannot span cell <l>'s <axis> extent: one of them is a nested composite with no single plot rect; that panel lays out on its own terms` and that panel lays out on its own. An alignment that would push the follower's marks over its own chrome (for example its own legend) is dropped, and the follower renders unaligned.

**Warnings.** A compound view reports every leaf's warnings in leaf order. A warning that reports a chart-level declaration rather than a data condition (`ConfigSurfaceNotPresent`, `BreakAxisMergeConflict`) is reported once for the figure, however many leaves repeat the declaration (for example a `RepeatChart` cloning one template). Every other warning fans out per leaf, because two leaves meeting the same data condition are two separate facts.

**Layered charts.** `LayerChart`, `fm.layer` and `Chart + Chart` overlay their layers on one panel:

- Every layer of a `LayerChart` must be a plain `Chart`; a composition there raises `ValueError: LayerChart: every layer must be a plain Chart, got <Class>`.
- `x` is always shared. `resolve={"x": "independent"}` raises `ValueError: LayerChart: layers share one coordinate space (overlay contract); per-layer independent x scales are not supported (see GH #55 dual-x-axis)`.
- `y` is shared by default. `resolve={"y": "independent"}` draws a dual-axis chart: layer 0's y scale drives the left axis and the gridlines, and each later layer resolves its own y scale and gets its own right-side axis, stacked outward, with no limit on the count. `SecondaryY` (below) is sugar for this.
- Render routes: static output with shared `y` renders through the composite overlay tree; static output with independent `y`, and all interactive output, render the merged single-panel chart, because selections and hit-testing need the overlay to be one scene panel.
- Chrome is drawn once. In the overlay tree every layer lays out against one shared plot rect, the per-side intersection of the layers' own gutters (title, legend and axis bands), and only the first layer draws grid, axes, above-marks axis chrome and the title. A later layer's legend, a `zindex >= 1` axis and a `z="below_marks"` annotation still render, once. When the layers' gutters leave no common plot area, each layer keeps its own rect and chrome and the render warns `overlay gutters diverged; <n> layers render with independent chrome`.
- A layer at index 1 or later whose positional encoding (`x`, `y`, `x2`, `y2`) names a column missing from that layer's own data raises `layer <i>: encoding '<channel>' references unknown column '<name>' (not present in that layer's data)`. A `data_source=` layer is checked against its own data. An inherited color channel absent from a layer's data still renders, and a zero-row layer is skipped.
- A shared ordinal axis covers every layer's categories, in first-appearance order across layers in layer order: `Chart({cat: [a, b, c]}) + Chart({cat: [x, y, z]})` draws six bands and six marks. An explicit ordinal `scale=` domain is not unioned across layers.
- A null in a positional ordinal column claims a null-category band only when some null-category row carries a non-null value on the opposite positional channel; a non-finite value counts as occupied. A genuine null category with a value keeps its own band, distinct from `0`, `false` or `""`. The rule is judged per layer against the positional field that layer binds itself, so a layer that binds only one positional channel opens no null band. The null padding a merge of two DataFrames introduces therefore never adds a band.

**Repeat.** The template marks the substituted channel with the typed sentinels `fm.Repeat.column`, `fm.Repeat.row` and `fm.Repeat.layer` in `.encode(...)`; a string such as `"repeat:column"` is parsed as an ordinary shorthand and refused. With both `row=` and `column=`, the grid has `len(row) × len(column)` panels. `diagonal=` is an alternate template for panels where the row field equals the column field, and `corner=True` keeps only the lower triangle including the diagonal. With only one of `row=` or `column=`, `columns=` wraps the panels (one row for a column repeat, one column for a row repeat when unset). `layer=` overlays one template copy per layer field in each panel, combined with `+`; diagonal panels are not layered. `resolve=` "shared" unions a channel's domain across every panel and every layer at render time; `expand()` returns the panels as `(row_field, col_field, Chart)` tuples without applying it. Refusals, all `ValueError` prefixed `RepeatChart:`: none of `row=`, `column=`, `layer=` set; `diagonal=` or `corner=True` without both `row=` and `column=`; `diagonal=` with `row != column` (raised when the grid expands); a template using a sentinel whose axis is unset (`template uses Repeat.row but row= was not set`). `fm.pairplot` builds a `RepeatChart`.

**Cluster map.** `ClusterMapChart` is a 2×2 grid: the heatmap bottom-right, the column dendrogram top-right, the row dendrogram bottom-left (drawn rotated) and an empty top-left corner. `dendrogram_ratio` is the fraction of the width and height each dendrogram takes, and must lie in the open interval (0, 1) (`ValueError: dendrogram_ratio must be in (0, 1); got <v>`). Through `fm.clustermap`, each dendrogram's leaf `k` sits on the center of heatmap cell `k`, the row dendrogram's leaves face the heatmap with leaf 0 beside the top row, and the heatmap's columns follow the column clustering's leaf order.

#### `JointChart`

Compound view with a center plot and optional marginal plots: the top marginal shares the center's x axis and the right marginal shares its y axis. The marginals are ordinary `Chart` objects, typically `mark_histogram` or `mark_density`.

```
JointChart(center, *, top=None, right=None, ratio=5, spacing=10.0)
```

| Parameter | Default | Meaning |
|---|---|---|
| `center` | required | Any `Chart`, in the bottom-left cell |
| `top` | `None` | Marginal above the center, spanning the center's x extent |
| `right` | `None` | Marginal right of the center, spanning the center's y extent |
| `ratio` | `5` | Size ratio of the center panel to each marginal; must be > 0 (`ValueError: ratio must be > 0; got <v>`) |
| `spacing` | `10.0` | Pixel gap between panels |

The top-right cell is empty. Most users construct `JointChart` via `ferrum.jointplot(...)`.

#### `SecondaryY`

```
SecondaryY(field, mark="line", axis=None, color=None, opacity=None, scale=None)
```

| Parameter | Default | Meaning |
|---|---|---|
| `field` | required | Data field mapped to the secondary y axis |
| `mark` | `"line"` | Mark type for the secondary series |
| `axis` | `None` | `Axis` config for the right-side axis |
| `color` | `None` | Literal mark color; unset means the theme default, like any other mark |
| `opacity` | `None` | Literal mark opacity |
| `scale` | `None` | `Scale` config for the secondary y axis |

`chart + SecondaryY(...)` appends one layer to `chart`: mark `mark`, `y` on `field` (carrying `axis` and `scale`), `x` inherited from the base chart's x encoding, and the literal `color` and `opacity`. The layer is an independent-y layer, the same mechanism as `LayerChart(resolve={"y": "independent"})`, so it renders in static and interactive output alike, reserves a right-side margin band and is interactive like any other layer. The base chart's own layers keep sharing the left axis. Adding several `SecondaryY` instances stacks several right axes outward. A base chart with no `x` encoding raises `ValueError: SecondaryY: the base chart must have an x encoding to inherit -- call .encode(x=...) before adding SecondaryY(...).`

A secondary axis draws gridlines only when asked. Show: the secondary layer's own `Axis(grid=)` wins over `configure(axis_y2=AxisConfig(grid=))`, and both default off. The theme's grid default, a shared `configure_axis(grid=)` and `configure_grid(y=)` never turn it on, because `axis_y2` does not fall back to shared axis config. A chart-level `axis_y2` applies to every stacked secondary, and a per-layer `Axis(grid=False)` overrides it for one. Style: the secondary layer's own `grid_*` keys win over `axis_y2.grid_*`, which win over the theme's major grid style (including a flat `configure_grid(color=, width=, dash=, opacity=)`); style without show draws nothing. When lit, horizontal gridlines draw across the plot at that secondary's own tick positions, after the primary's, in every panel. Grid bands stay primary-only, minor gridlines draw only when `Theme(grid_minor=True)` and the secondary's own show are both on, a primary-y `break_axis` never remaps them, and they stay fixed under interactive zoom like the primary's. There is no `configure_grid(y2=)`.

#### `BreakAxis`

```
BreakAxis(axis, gap, break_size=12, break_style="slash")
```

`chart + BreakAxis(...)` removes one or more ranges from a continuous positional scale, so a distant outlier cannot compress the rest of the data into an unreadable band, and draws a break indicator where each removed range was. `BreakAxis` is a frozen dataclass in `ferrum.structural`, exported as `ferrum.BreakAxis`.

**Parameters.**

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `axis` | `str` | required | Which **physical** axis to break: `"x"` or `"y"`. Physical, not encoding-relative: under `CoordFlip()`, `axis="y"` still breaks the physical y axis, which then carries the flipped chart's x encoding. |
| `gap` | `tuple` or `list` | required | One `(start, end)` pair, or a sequence of pairs. Each bound is numeric (a real number: `int`, `float`, a numpy integer or floating scalar, or `Decimal`) or temporal (an ISO-8601 date/datetime `str`, `datetime.date`, `datetime.datetime`, `numpy.datetime64`, `pandas.Timestamp`). |
| `break_size` | `float` | `12` | Pixels of axis reserved for each drawn break indicator. |
| `break_style` | `str` | `"slash"` | Indicator style: `"slash"`, `"zigzag"`, `"wave"`, or `"gap"` (whitespace, no indicator drawn). |

**Construction and validation** (`BreakAxis.__post_init__`). Every refusal below is a `ValueError` at construction time that names the offending value and, for a bound, its pair as `BreakAxis.gap[i]`.

- `axis` must be `"x"` or `"y"`: `BreakAxis.axis: axis must be one of ['x', 'y']; got 'z'`. `break_style` must be one of the four named styles.
- `gap` must be a tuple or list (`BreakAxis.gap must be a (start, end) tuple or a list of such tuples, not int; got 5`), must hold at least one pair, and every pair must have exactly two bounds.
- Both bounds of one pair must be the same kind: a pair mixing a numeric and a temporal bound is refused (`BreakAxis.gap[i]: mixes a numeric bound (...) with a temporal bound (...); both bounds of one gap must be the same kind`).
- A temporal bound is coerced to epoch-milliseconds UTC by `ferrum.annotation.coords.temporal_coord_to_epoch_ms`, the same function the encoding layer's `TimeScale`-domain coercion and annotation coordinates use, so those three Python surfaces accept the same types. A naive bound is treated as UTC; an aware one is converted to UTC first. A string that is not ISO-8601 is refused as a date (`cannot parse 'nope' as an ISO-8601 date or datetime (use 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS')`), not as "not numeric". A not-a-time bound (`pandas.NaT`, `numpy.datetime64("NaT")`) is refused by name, as is a `numpy.datetime64` too large to represent as a microsecond-precision epoch offset.
- Every bound then passes the package's finite-real coercion (`ferrum._validate.coerce_finite_real`, through `coerce_finite_ordered_pair`): a non-number such as `bool` or `numpy.timedelta64` is refused as `BreakAxis.gap[0]: start must be a real number (int, float, numpy scalar, or Decimal), not bool; received True`, and a value whose float is not finite (NaN, ±inf, `10**400`, `Decimal("sNaN")`) as `... must be a finite number a float can hold; received nan`.
- Each pair must be ordered, `start < end`; equal bounds are a zero-width gap and are refused (`BreakAxis.gap[0]: start (2.0) must be less than end (2.0); got (2.0, 2.0)`).
- Two cases are not construction errors, because the constructor cannot see the data: a gap outside the resolved domain, and a temporal gap on a non-temporal axis. Both render as if the break had not been declared, and warn (see Warnings).

One temporal surface is narrower: the Rust-backed `TimeScale(domain=[...])` constructor refuses `numpy.datetime64` with a `TypeError`, which the three Python surfaces above accept.

The stored `gap` keeps the container type it was given (a tuple of pairs stays a tuple, so a tuple-spelled `BreakAxis` stays hashable), with each pair canonicalized to a 2-tuple of floats. The wire form is `{"type": "break_axis", "axis", "gaps": [[start, end], ...], "break_size", "break_style"}`. The Rust `chart_config` gate accepts `f64` bounds only, so a hand-built config dict must carry already-converted epoch-ms numbers, not date strings. Its `axis` key accepts exactly `"x"` and `"y"`, in either wire spelling of a unit variant (the string, or the single-key map `{"y": null}`), and refuses any other token with `chart config: structural[0] (break_axis).axis: axis must be one of ['x', 'y']; got 'z'`. The same gate value-checks each `gaps[j]` pair of 2 numbers: a non-finite bound is refused (`chart config: structural[0] (break_axis).gaps[0]: start must be finite; received NaN`, and `end must be finite; received inf` for the end), and so is a pair whose start is not less than its end (`chart config: structural[0] (break_axis).gaps[0]: start (200.0) must be less than end (100.0); got (200.0, 100.0)`; equal bounds are refused the same way). A gap that is not a list of exactly 2 numbers is left to deserialization's own type error. These refusals are reachable only through a hand-built config dict passed to private render entries, since `BreakAxis` itself already refuses the same values at construction.

**Render model.** Gaps are normalized once per axis before any geometry is built. Every gap that reaches this step has already been refused if it was non-finite or reversed, by `BreakAxis` or by the wire gate; the normalizer still drops a non-finite gap as a guard for a `BreakAxisSpec` built directly in Rust. Each gap is then clipped to the resolved domain, a gap that clips to nothing is dropped (when no gap survives, the axis renders unbroken and warns, see Warnings), and the rest are sorted and merged, so overlapping, nested, adjacent and unsorted gaps reconcile into the same union and an indicator reserves pixels only for a gap it draws. Declaration order does not matter. The surviving gaps become a piecewise scale whose segments tile the broken scale's own padded pixel range, not the raw plot rect, so a broken axis's ticks, marks, gridlines and indicator sit on the same pixels as the unbroken chart's, and a value past the domain clamps to the domain's own end pixel. Each surviving gap consumes `break_size` pixels. A segment includes both of its bounds: a value exactly on a gap edge belongs to the visible segment beside it. A break on a reversed (descending) scale applies the same way. A `SecondaryY` or other independent-y layer maps through its own y scale, so a y break leaves its marks and its axis untouched, while an x break applies to it like any other layer.

**Per-mark-kind behavior**, decided by what the geometry encodes rather than by node shape (`BreakPlacement::for_kind`, exhaustive over mark batch kinds):

| Mark batch kind | Behavior under a gap |
|---|---|
| `bar`, `area`, `rect`, `ribbon`, `polygon` (incl. `mark_violin` and `mark_geoshape`) | Clipped: one sub-node per visible segment it intersects, nothing when it lies wholly inside a gap. A cut edge that the gap produced is squared on a rounded (`corner_radius`) piece; corners inherited from the source rect keep their radius. The outline an area draws over its fill when its `line` or `borders` option is on is split like a `line`, so it follows the fill's pieces. |
| `point`, `tick`, `text` | Translated whole to the anchor's compressed position, keeping drawn size and shape, and hidden whole when the anchor is inside a gap. The anchor is the row's own scale-mapped position, so an asymmetric shape and a `dx`/`dy`-offset label hide on their data position, not on their drawn extent. |
| `image` | Left at its unbroken position, neither moved nor hidden: placement of an image node under a break is not defined. |
| `label` | The text moves as a glyph; the leader line's two ends are two data positions and are placed independently. |
| `line`, `rule`, `arc` | Split: one sub-node per visible piece, cut at each gap edge with the cross-axis coordinate interpolated at the crossing, and the in-gap portion not drawn. A step line (`interpolate="step"`, `"step-before"`, `"step-after"`) splits the same way. A stroke with no visible piece is dropped. |
| `segment` | The shaft splits as above; an `arrow=True` head is rebuilt from the endpoints of the piece reaching its tip, keeping its drawn size and pointing along that piece. A tip inside a gap drops the head. |

Split pieces carry their row metadata through 1→N bookkeeping, so `data_indices`, tooltips, hrefs, descriptions and polar spans stay one entry per node. A `key=` sidecar keeps the original key on the first piece and suffixes every later piece as `{key}#{i}`, choosing the smallest free `i` so a second axis's pass cannot collide with the first's.

**Axis chrome.** Only nodes a tick value positioned are re-placed, and only on the axis they were built along: a tick dash and its label translate rigidly on that tick's own anchor, so a label keeps its drawn baseline offset exactly and hides if and only if its own tick's anchor is in a gap. A tick on a gap edge belongs to the visible segment and is drawn. The axis title and the domain line are exempt (their positions come from the axis rect, not from tick values), as is all cross-axis chrome, so a y break moves horizontal gridlines and leaves vertical ones spanning the full plot rect. A gridline moves with its tick and is hidden when its tick is in a gap. A grid band fill is extent geometry: it is clipped into one piece per visible segment, and its outermost boundaries stay on the plot rect's edges, so the padding strip at each end of the axis keeps its stripe.

**Annotations** (§3.11) are routed by what their geometry means, mirroring the mark side. `rect` and `span` are clipped into one piece per visible segment, silently (a span's label re-anchors on the largest surviving piece, ties keeping the first). `line` and `arrow` split through the same vertex splitter marks use. `text`, `callout` and `image` keep a plain membership test on their coordinate; a callout's background box is drawn size, never data, and is never clipped. A surviving data-space coordinate is placed through the break, on the pixel a mark at the same value takes. The primitives one `annotate_*()` call expands to are routed one by one, so an `annotate_hline(label=...)` line can clip while its label is suppressed. `bracket` is suppressed whole whenever a gap reaches it, by design: its two endpoint tips say where the bracketed range begins and ends, so a cut would either strand them or invent new ones. An annotation is suppressed only when nothing of it survives, or when its kind admits no honest split. Suppression rolls up into one warning per axis per mechanism (a coordinate inside a gap, versus an interval reaching across one), counting annotation elements, not `annotate_*()` calls: four `annotate_hline(label=...)` calls inside a y gap give one warning, `8 annotation elements (line, text) on the y axis fell inside a BreakAxis gap and were suppressed`. Pixel- and norm-space annotation coordinates are never touched by a break.

**Facets and composites.** A break applies to every facet panel and is built per panel, against that panel's own plot rect and scale pixel range, so identical data places identically in every panel and every panel draws its own indicator. Diagnosis is per chart, not per panel: a fact every reporting panel agrees on is reported once, with the single-panel text. `BreakAxisIneffective` alone is qualified when panels disagree. When only some panels report it, the text names the first and the share, `BreakAxis on the y axis had no effect: no gap overlaps the axis domain (0, 20) (facet panel 1); this holds on 1 of 2 panels, and the break applies on the rest`; when every panel reports it with different domains, `... the axis domain (0, 900) (measured on facet panel 0); other panels reported this fact with a different value, so this is panel 0's, not every panel's`. Panel domains can differ under the default shared resolve too (a bar's zero-based domain resolves per panel). `BreakAxisOnNonlinearScale` and `BreakAxisMergeConflict` describe the spec and are reported once, unqualified. Chart-level annotations are built once, against the reference panel's scales, so an in-gap annotation is suppressed in every panel and reported once. An `Inset` is chart-level too: it is built once against the reference panel and painted after that panel's break indicators, so an inset covers an indicator it overlaps. Composite leaves are separate scenes with their own break build and their own warnings.

**Multiple specs on one axis** merge into a single break result before any geometry is built: their gap lists union through the same normalization, so N same-axis specs render byte-identically to one spec declaring the union. A conflicting `break_style` or `break_size` among merged specs keeps the first declaration's value and warns once per axis per field.

**Coordinate systems.** `CoordPolar` and `CoordGeo` combined with any `BreakAxis` refuse at the render funnel with a `ValueError` naming the axis and the coord, `BreakAxis(axis="y") is not supported under Polar coordinates; remove the break or render this chart under Cartesian coords` (`RenderError::BreakAxisIncompatibleCoord`). A composite leaf that combines them refuses the same way. Every other coord renders normally.

**Warnings** (all `RenderWarning`, surfaced as Python `UserWarning`s):

- `BreakAxisIneffective`: no gap overlaps the axis domain, or the axis is ordinal. The chart then renders exactly as if the break had not been declared, padding included. The domain is stated low to high whatever the scale's declared order: `BreakAxis on the y axis had no effect: no gap overlaps the axis domain (10, 100)` for a reversed `[100, 10]` domain, and `... the axis is ordinal, so its gaps have no numeric data range to remove`.
- `BreakAxisOnNonlinearScale`: a `log`, `pow` (which includes `sqrt`) or `symlog` axis. Placement uses a linear pixel-to-data map, so the removed range and the indicator position can differ from the gap asked for.
- `BreakAxisMergeConflict`: same-axis specs disagree on `break_style` or `break_size`, e.g. `multiple BreakAxis specs on the y axis disagree on break_style; the first declaration's value was kept`.
- `BreakClipSidecarMisaligned`: a split batch whose per-node metadata length disagreed, so tooltips, links or keys on it may name the wrong marks. This is a builder defect that a debug build asserts on; the warning is the release build's report of it.

**Limits.** A curved line (`mark_line(interpolate="basis"/"monotone"/"cardinal")`) and a curved closed ring keep a per-vertex placement and can still draw a stroke through the hidden band, because cutting a placed curve faithfully needs subdividing it. A piece that is at once rotated (`angle != 0`), rounded and cut keeps its original corner radius, because converting it to a path would move its rotation anchor. Non-linear-scale placement is linear (warned, above). Interactive (WASM) output builds its scene through the same break-aware pass, so marks are placed identically, but break-aware interaction (hit-testing, keyed transition pairing across a split) is not provided.

#### `Inset`

```
Inset(chart, bounds, border=True, border_color="#999", border_dash=None, background="#fff", shadow=False, connect_to=None, connect_style="lines")
```

`chart + Inset(...)` embeds another chart inside the parent's plot area. The inset is independent of the parent: its scales, axes, marks and configuration are its own, the parent's `configure_*()` settings do not cascade into it, and it does not change the parent's layout. `Inset` is a frozen dataclass in `ferrum.structural`, exported as `ferrum.Inset`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `chart` | `Chart` | required | The chart to embed. |
| `bounds` | `tuple` | required | `(left, top, right, bottom)` within the parent plot area. Each entry resolves on its own: a bare real number or `fm.norm(value)` is a normalized `[0, 1]` fraction of the plot area (never a data coordinate), and `fm.px(value)` is an absolute pixel offset from the plot area's origin, resolved in Rust against the computed plot rect at layout time. Kinds may be mixed in one tuple. |
| `border` | `bool` | `True` | Draw a 1 px border around the inset. |
| `border_color` | `str` | `"#999"` | Border color. |
| `border_dash` | `list[float]` or `None` | `None` | Border dash pattern; `None` is solid. |
| `background` | `str` or `None` | `"#fff"` | Fill drawn behind the inset; `None` draws no fill. |
| `shadow` | `bool` | `False` | Draw a drop shadow behind the inset. |
| `connect_to` | `tuple` or `None` | `None` | Data coordinates `(x, y)` in the parent chart that the inset zooms into, mapped through the parent's primary scales. |
| `connect_style` | `str` | `"lines"` | Connector from `connect_to` to the inset: One of `"lines"`, `"bracket"`, `"none"`. `"lines"` draws two lines from the mapped point to the inset's two nearest corners; `"bracket"` and `"none"` draw no connector. |

**Bounds validation** (`Inset.__post_init__`). `bounds` must have exactly four entries. Each entry is classified once (`norm`, `px`, or a bare number read as `norm`) and validated through the package's shared fraction and pixel validators with the entry's label; every refusal is a `ValueError`:

- A bare entry must be a real number: `int`, `float`, any other non-`bool` `numbers.Real`, a numpy integer or floating scalar, or `Decimal`. Anything else is refused by type name, including `bool`, numeric strings, `None` and `complex`: `Inset.bounds[0] (left) must be a number between 0 and 1, not str; received 'a'`. A wrapped bound accepts the same values a bare one does.
- A normalized entry must lie in `[0, 1]` (`Inset.bounds[2] (right) must be between 0 and 1; received 1.5`); a pixel entry must be non-negative (`Inset.bounds[0] (left) must be non-negative; received -1.0`).
- The box must not be empty: `left < right` and `top < bottom`, compared only when both entries of the pair have the same kind, so a mixed pair such as `left=px(50), right=norm(0.1)` is never compared (`Inset.bounds: empty box — left (0.5) must be less than right (0.5); ...`).
- `connect_style` must be one of `"bracket"`, `"lines"`, `"none"`.

Every accepted bound reaches the wire as a float: a `px` bound as `{"px": f}`, a normalized one as a bare number.

**Rendering.** The inset chart is rendered on its own and embedded as a positioned, clipped nested `<svg>` at its bounds, painted back to front as shadow, background, content, border, then connector. Its clip and legend ids are namespaced per inset so they cannot collide with the parent's or another inset's. The inset chart is laid out at its own width and height (640x480 unless its `properties()` set them) and the nested `<svg>` scales that drawing uniformly into the bounds box, so text, marks and tick labels shrink by the same factor. An inset is fixed chrome: it does not follow pan or zoom.

#### Mark builds that draw nothing

A mark builder that receives rows it cannot draw names why, through `RenderWarning::MarkBuildEmpty`, read as `{mark} layer {layer} drew nothing: {reason}`. The cases are a `mark_segment` bound without `x2` or `y2`, a `mark_image` with no `url=` encoding and no raster-transform data, a `mark_image` under `CoordPolar` or `CoordGeo`, a `mark_geoshape` batch with no geometry column, a `mark_rect` box body whose x and y both resolved to non-ordinal scales, and zero-extent bars. Every other empty result is silent by design: a zero-row batch, a documented per-row null, non-finite or out-of-domain skip, a present but all-null geometry column, and a facet panel left empty by a filter. An ordinal positional scale whose bandwidth resolves below one pixel warns once per panel with `RenderWarning::GroupCardinalityDegenerate`, `{n} categories were placed on an axis with only {bandwidth_px:.2}px per category; some may not render visibly` (singular for one category).

#### Numeric arguments

Every numeric door shares one predicate and one coercion in `ferrum._validate`: `is_real_number` (any `numbers.Real` or `Decimal`, never `bool`, never a timedelta), its integer sibling `is_integer_number`, and `coerce_finite_real`. A door refuses a non-number with `"{label} must be a real number (int, float, numpy scalar, or Decimal), not {type}; received {value!r}"` and a value whose float is not finite with `"{label} must be a finite number a float can hold; received {value!r}"`, and otherwise stores a builtin `int` or `float`, so numpy scalars and `Decimal` serialize as plain numbers.

- The doors that use it include `BreakAxis.gap` and `Inset.bounds` (above), `heatmap(vmin=, vmax=, center=)` (`heatmap(vmax=True)` is refused), `AxisConfig(domain_min=, domain_max=)` and the `override(<axis>_domain_min/max=)` leaves, `PaddingConfig`, `GridConfig`/`GridAxisConfig` `width`/`opacity`, `override(padding_*=)` (`PaddingConfig(top=5).to_dict()` gives `{"top": 5.0, ...}`), `ModelSource.embeddings(method="tsne", perplexity=)`, the numeric `mark_*` style kwargs (`size`, `opacity`/`alpha`, `stroke_width`, `corner_radius`, `font_size`, `dx`, `dy`, `angle`, `band_size`, `width`, `height`, `arrow_size`; the list `ferrum._core.mark_numeric_kwargs()` publishes), `fm.transform_impute(value=)`, and the number literals of `fm.transform_filter` and `fm.transform_impute(keyvals=)`. A dash length and a conditional numeric literal refuse `10**400` and `Decimal("sNaN")` in their own refusal shapes.
- The unsigned-integer mark fields that `ferrum._core.mark_integer_kwargs()` publishes (`limit`) go through `coerce_non_negative_int` instead: `mark_text(limit=2.5)` refuses with `mark_text(limit=...) must be a non-negative int, not float; received 2.5`, and `limit=-1` with `mark_text(limit=...) must be >= 0; received -1`.
- A style kwarg the mark does not honor is refused as not honored before its value is checked, so `mark_point(corner_radius="x")` raises the mark's `EntryRefusal` naming its honored kwargs.
- Counts go through `coerce_positive_int` (`Chart.facet(ncols=, nrows=)`, `Facet(columns=)`, `catplot(n_boot=)`), which accepts a numpy integer and stores a builtin `int`. It refuses a value below 1 (`Chart.facet(ncols=...) must be >= 1; received 0`), a `bool` or any non-integer (`Chart.facet(ncols=...) must be a positive int, not bool; received True`, `... not float; received 2.5`), and a value above `2**53`: `Chart.facet(ncols=...) must be at most 2**53, the largest integer a float holds exactly; received 9007199254740993`. A `Chart.facet(ncols=)` value between `2**32` and `2**53` passes Python and is refused by Rust's `u32` field at render, naming `facet:`.
- The `ci=` and `n_boot=` arguments of `catplot` are validated: `ci` is a percent in the open interval `(0, 100)`, and a non-default `ci=` or `n_boot=` on a kind that draws no interval (`strip`, `swarm`, `box`, `violin`, `boxen`, `count`) refuses: `catplot(ci=90) is not supported with kind='box' -- that kind draws no confidence interval, so ci= would be ignored. Drop ci=, or use one of the estimating kinds ('bar', 'point').` The defaults (`ci=95`, `n_boot=1000`) stay silent on every kind.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#312-compound-views).

---

### 3.13 Themes

A `Theme` is an immutable value object holding a set of optional styling keys. A key that is unset (or passed as `None`) falls back to the built-in default listed below. `fm.Theme` and `fm.themes.Theme` are the same class.

```
Theme(
    # Canvas and spacing
    background=None, background_color=None, padding=None,
    axis_title_padding=None, column_padding=None, row_padding=None,

    # Typography
    font_family=None, font_size=None, font_weight=None, font_color=None,
    title_font_family=None, title_font_size=None, title_font_weight=None,
    title_color=None, title_anchor=None, title_offset=None,
    label_font_family=None, label_color=None,

    # Grid
    grid=None, grid_color=None, grid_width=None, grid_dash=None, grid_opacity=None,
    major_grid_color=None, major_grid_width=None, major_grid_dash=None, major_grid_opacity=None,
    minor=None, minor_grid_color=None, minor_grid_width=None, minor_grid_dash=None, minor_grid_opacity=None,

    # Axes and tick labels
    axis_line=None, axis_line_width=None, axis_line_color=None,
    tick_size=None, tick_width=None, tick_color=None,
    cull_threshold=None, cull_labels=None,

    # Marks
    mark_color=None, point_size=None, point_opacity=None,
    line_stroke_width=None, bar_corner_radius=None,
    area_opacity=None, opacity=None,

    # Palettes
    color_scheme=None, sequential_scheme=None, diverging_scheme=None,

    # Legend
    legend_orient=None, legend_direction=None, legend_title_font_size=None,

    # Facet strips and reference lines
    strip_background_color=None, strip_text_size=None, strip_padding=None,
    reference_line_color=None, reference_line_dash=None,
)
Theme.update(**kwargs)
```

The accepted key set is published by the Rust binding (`ferrum._core.theme_known_keys()`); `Theme` derives its key list from it, so Python and Rust accept exactly the same keys. A key outside that set raises `ValueError` at construction: `Unknown Theme key(s): ['width']. See ferrum-spec.md §3.13 for the supported key list.` Chart size is not a theme key; it is set on the chart (`Chart(df, width=, height=)` or `.properties()`). Values are not checked at construction; the Rust binding checks them when the chart renders (see Validation).

Effective defaults (Paper Ink, the built-in identity):

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `background` | color | `"#FAF7F2"` | Chart background. `background_color` is an alias for the same field; if both are given, `background` wins. |
| `padding` | `float` | `16` | Outer padding on every side, in px. Per-side padding and `padding.auto` are chart-level (§3.7 `configure_padding()`). |
| `axis_title_padding` | `float` | `8` | Gap between tick labels and the axis title, in px. |
| `column_padding`, `row_padding` | `float` | `12` | Gap between composite and facet cells, in px. |
| `font_family` | `str` | `"Inter"` | Body-text family. Inter is bundled with the renderer, so text measures and renders the same on every host. |
| `font_size` | `float` | `11` | Body and tick-label font size. There is no separate `label_font_size` key: `font_size` is the label size. |
| `font_weight` | `str` | `"normal"` | Body-text weight. |
| `font_color` | color | `"#1F2937"` | Body-text color. |
| `title_font_family` | `str` | `"Inter"` | Chart-title family. |
| `title_font_size` | `float` | `13` | Chart-title size. |
| `title_font_weight` | `str` | `"600"` | Chart-title weight. |
| `title_color` | color | `"#1F2937"` | Chart-title color. |
| `title_anchor` | `str` | `"start"` | Title alignment: `"start"`, `"middle"` or `"end"`. |
| `title_offset` | `float` | `6` | Gap below the title, in px. |
| `label_font_family` | `str` | `"Inter"` | Tick-label family. |
| `label_color` | color | `"#6B7280"` | Tick-label color. |
| `grid` | `bool` or `fm.Grid` | `True` | Draw major gridlines. An `fm.Grid` value (§3.19) expands into the `grid`, `minor` and per-level keys below. |
| `grid_color`, `grid_width`, `grid_dash`, `grid_opacity` | color, `float`, `list[float]`, `float` | `"#D6D3D1"`, `0.5`, solid, `1.0` | Major-gridline style. These flat keys style major gridlines only. |
| `major_grid_color`, `major_grid_width`, `major_grid_dash`, `major_grid_opacity` | as above | the `grid_*` value | Major-gridline style; each wins over its `grid_*` counterpart. |
| `minor` | `bool` | `False` | Draw minor gridlines on continuous scales. |
| `minor_grid_color`, `minor_grid_width`, `minor_grid_dash`, `minor_grid_opacity` | color, `float`, `list[float]`, `float` | `"#E8E6E4"`, `0.3`, solid, `0.6` | Minor-gridline style. They do not inherit from `grid_*`. |
| `axis_line` | `bool` | `True` | Draw the axis domain line. |
| `axis_line_width` | `float` | `1` | Axis-line stroke width. |
| `axis_line_color` | color | `"#6B7280"` | Axis-line color. |
| `tick_size`, `tick_width` | `float` | `4`, `1` | Tick length and stroke width, in px. |
| `tick_color` | color | `"#6B7280"` | Tick color. |
| `cull_threshold` | `int` (non-negative) | `8` | Minimum pixel gap between adjacent x tick labels (see Tick-label gap and culling). |
| `cull_labels` | `bool` | `True` | Whether the layout may drop x tick labels to reach that gap. Unset, it follows `cull_threshold`. |
| `mark_color` | color | `"#2563EB"` | Color of marks with no color encoding. |
| `point_size` | `float` | `36` | Default point area. |
| `point_opacity` | `float` | `1.0` | Default point opacity. |
| `line_stroke_width` | `float` | `1.5` | Default line width. |
| `bar_corner_radius` | `float` | `0` | Default bar corner radius. |
| `area_opacity` | `float` | `0.35` | Default area fill opacity. |
| `opacity` | `float` | `1.0` | Default mark opacity. |
| `color_scheme` | `str` | `"paper_ink"` | Categorical palette. Any categorical or continuous scheme name is accepted (§3.6 lists them and how a continuous name resolves on a nominal encoding). |
| `sequential_scheme` | `str` | `"cool_blue"` | Default ramp for continuous color (heatmaps, densities) when no explicit scheme is given. |
| `diverging_scheme` | `str` | `"blue_to_red"` | Default ramp for diverging color (correlation matrices and similar) when no explicit scheme is given. |
| `legend_orient` | `str` | `"right"` | Legend side: `"left"`, `"right"`, `"top"` or `"bottom"`. |
| `legend_direction` | `str` | implied by orient | `"horizontal"` or `"vertical"`. |
| `legend_title_font_size` | `float` | `11` | Legend-title size. |
| `strip_background_color` | color | `"#EDE9E3"` | Facet strip-title background. |
| `strip_text_size` | `float` | `12` | Facet strip-title font size. |
| `strip_padding` | `float` | `6` | Vertical padding around a facet strip title, in px. |
| `reference_line_color` | color | `"#9CA3AF"` | Color of reference and annotation guide lines. |
| `reference_line_dash` | `list[float]` | `[4, 4]` | Dash pattern of reference and annotation guide lines. |

**Fallbacks.** An unset `title_*` or `label_*` key that has a body-text counterpart takes that counterpart's value when the counterpart is set, before the default applies: `title_color` and `label_color` fall back to `font_color`, `title_font_family` and `label_font_family` to `font_family`, `title_font_weight` to `font_weight`, and `title_font_size` to `font_size`. So `Theme(font_size=20)` sets the title to 20 px as well. `title_anchor` and `title_offset` have no counterpart. `Theme.to_spec_dict()` resolves the fallbacks and renames `background` to `background_color`, so the Rust binding receives a fully resolved dict. The binding consumes every accepted key.

**Validation** (Rust, at render, all `ValueError`):

- An enumerated key outside its vocabulary is refused by name: `title_anchor must be one of 'start'|'middle'|'end', got 'left'`, `legend_orient must be one of 'left'|'right'|'top'|'bottom', got 'inside'`, `legend_direction must be one of 'horizontal'|'vertical', got 'diag'`.
- A color key accepts a CSS color name, `#rrggbb`/`#rrggbbaa`, the shorthand `#rgb`/`#rgba`, or `rgb()`/`rgba()`; anything else is refused: `invalid color 'notacolor': expected a CSS color name, #rrggbb[aa]/#rgb[a] hex, or rgb()/rgba()`.
- An unknown scheme name is refused with the supported list: `Unknown color_scheme: 'nope'. Supported categorical: ... Supported sequential: ...`; `sequential_scheme` and `diverging_scheme` must name a continuous scheme (`Unknown sequential_scheme: 'tableau10'. Supported: ...`).
- A value of the wrong type is refused with the deserializer's message, for example `theme: invalid type: string "x", expected f64 ...` for `padding="x"`, and ``theme: invalid value: integer `-1`, expected u32 ...`` for `cull_threshold=-1` (a float `cull_threshold` is refused the same way).

**Tick-label gap and culling.** `cull_threshold` and `cull_labels` govern the x tick-label collision cascade. `cull_threshold` is the minimum pixel gap the cascade keeps between adjacent visible labels. It shrinks the fit budget of every presentation stage the cascade tries before dropping labels (flat, wrapped, smaller font, rotated), so a stage resolves only when it leaves that gap; at `0` the test is plain non-overlap. `cull_labels` says whether the cascade may then drop labels by stride to reach the gap. The two resolve as follows:

| `cull_threshold` | `cull_labels` | Result |
|---|---|---|
| unset | unset | Gap 8 px, dropping permitted (the default). |
| `N > 0` | unset | Gap `N` px, dropping permitted. |
| `0` | unset | No gap requirement, never drop a label. |
| any | `False` | That gap drives the presentation stages, and every label is kept. Which stage the labels land in depends on density: sparse labels satisfy the gap early (flat or a shallow rotation), dense ones fall through to the last resort, rotated -90 degrees and possibly overlapping. |
| `0` | `True` | No extra gap; labels are dropped only where they genuinely overlap. |

`cull_labels`, when given, always wins over the permission `cull_threshold` implies. When labels are dropped, the survivors choose their own presentation at the new, sparser density, so a survivor that no longer needs rotation renders flat. Mid-label elision (truncation with an ellipsis) is the step after culling; it is exempt from the gap and fires only where truncating a label frees enough width to help, so an infeasible truncation leaves the label intact. `fm.Axis(label_overlap=...)` (§3.7) overrides the cascade: `"true"` keeps every label regardless of either key. The axis band-height budget of §3.7 can still drop labels by stride under `cull_labels=False`. The policy applies to x tick labels only; y tick labels are never dropped by it. It applies identically to an un-faceted chart, a shared-x facet, and a facet whose x scale resolves independently (`.facet(col=...).share_scale(x="independent")`): each panel's x axis reads the theme's policy, and a shared-x and an independent-x facet over the same data keep the same labels.

**Built-in themes** (`fm.themes`):

| Name | Description |
|---|---|
| `default` | `Theme()`: every key at its built-in default (Paper Ink). |
| `paper_ink` | Paper Ink with its identity keys set explicitly (cream `#FAF7F2` background, blue `#2563EB` marks, warm grid); renders the same as `default` and serves as a derivation base. |
| `slate_citrus` | Dark navy background (`#111827`), light text, `slate_citrus` palette with neon accents. |
| `arctic_signal` | Cool white background (`#F8FAFC`), sky-blue lead mark, `arctic_signal` palette. |
| `observable` | White background, tableau blue `#4C78A8` marks, neutral gray grid, `tableau10` palette. |
| `minimal` | No grid, no axis lines, zero-length ticks, 24 px padding. |
| `dark` | Dark navy background (`#1a1a2e`), light text, `dark2` palette. |
| `publication` | White background, no grid, black axes and text, bold middle-anchored title, black marks, `tableau10` palette. |
| `economist` | Light blue background (`#d3e0e6`), red bold title, no axis lines, `set1` palette. |
| `fivethirtyeight` | Grey background (`#f0f0f0`), no axis lines, bold start-anchored title, `set1` palette. |
| `solarized_light` | Solarized cream background (`#fdf6e3`), `set2` palette. |
| `solarized_dark` | Solarized dark background (`#002b36`), `set2` palette. |

A built-in is an ordinary `Theme`, so `.update()` derives from it. Apart from the `default`/`paper_ink` pair, no two built-ins render the same chart identically.

**Theme composition:**

```python
my_theme = fm.themes.minimal.update(
    font_family="Berkeley Mono",
    grid=fm.Grid(major=True, minor=False, color="#f5f5f5"),
    title_font_weight="bold",
)
```

`Theme.update(**kwargs)` returns a new `Theme` with the given keys replaced; passing `None` for a key removes it, so it falls back to its default. The source theme is unchanged. Two themes are equal when they hold the same keys and values, and a `Theme` is hashable (list values such as `grid_dash` included).

**Applying a theme.** `Chart.theme(t)` attaches `t` to a chart and replaces the whole active theme; it does not merge with the context default, so `.theme(fm.Theme())` renders at the built-in defaults even inside a dark context. A composition's `.theme(t)` applies `t` to every sub-chart. Layering with `+` keeps the primary layer's theme and warns (``Layered chart `+`: secondary layer's theme/facet is ignored; primary layer wins.``) when the right-hand layer carries a different one. Every other presentation source wins over the theme: `Chart.override()`, then per-channel `axis=`/`legend=`, then `configure_*()` (§3.7), then the theme, then the context default, then the built-in defaults.

**Context default:**

```
set_default_theme(theme)
theme_context(theme)
get_default_theme()
```

`ferrum.set_default_theme(theme)` sets the default theme used by any chart that has no `.theme()` of its own, and returns a context manager. Used with `with`, the previous default is restored on exit, nested blocks included; called without `with`, the default stays set for the rest of the current context. `theme_context(theme)` is the same function under a context-manager name, and `get_default_theme()` returns the current default (initially `fm.themes.default`). A non-`Theme` argument raises `TypeError` (`theme must be a Theme instance, got str`).

The default lives in a `contextvars.ContextVar`, not in module state. It is per thread: a new thread starts at `fm.themes.default` whatever its creator set, and a default set inside a thread does not leak out of it. An asyncio task starts with a copy of the default current when it was created. The default is read when a chart renders, not when it is built, so a chart built inside a `with` block and rendered after it uses the restored default. A per-chart `.theme()` always wins. This is the one sanctioned process-wide styling mutator.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#313-themes).

---

### 3.14 Figure-Level Functions

Figure-level functions return a `Chart` or a compound view. They handle data reshaping, faceting, axis labeling and legend placement themselves, and every result renders, saves and exports like any chart.

| Function | Returns |
|---|---|
| `displot`, `catplot`, `relplot`, `lmplot`, `regplot`, `residplot`, `heatmap` | `Chart` |
| `pairplot` | `RepeatChart` |
| `jointplot` | `JointChart` |
| `clustermap` | `ClusterMapChart` |
| the model-diagnostic functions (Model Diagnostics, below) | `Chart`, or a composite (`ConcatChart`, `HConcatChart`, `VConcatChart`) for multi-panel and `compare=` layouts |

Every leaf chart of a result lowers to a `ChartSpec` (`chart.to_spec()`) that round-trips through `ChartSpec.from_json(spec.to_json())`; a compound view reaches its leaves through `.charts` (and `RepeatChart.expand()`).

**Shared keyword arguments.** Every figure function, including every model diagnostic, takes keyword-only `mark=None`, `encode=None`, `properties=None`, `layers=None` and `theme=None`. They apply in the order mark, encode, properties, layers, then the theme: `mark=` is a dict of mark overrides (keyed by layer name on a layered chart, flat on a single-mark chart), `encode=` is merged through `Chart.encode(**encode)`, `properties=` through `Chart.properties(**properties)`, `layers=` is appended through `Chart.layer(*layers)`, and `theme=` through `Chart.theme()`. On a composite result the overrides fan out to every child: a `mark=` key applies to each child that draws the layer it names and refuses when no child draws it, and `title`/`subtitle`/`caption` in `properties=` land at the figure level. A malformed `mark=` refuses before anything is computed. The distribution, categorical, relational, regression and matrix functions also take `**encode_kwargs`, forwarded to `Chart.encode()`. A warning or refusal about a layer the figure draws, and a refusal from its override or style doors, names the public function the caller invoked (`displot(...)`), not the mark method it used.

**Size arguments.** `displot`, `catplot`, `relplot` and `pairplot` check `height=` (a finite pixel value above 0) and `aspect=` (a finite number above 0), `jointplot` checks `height=`, and `clustermap` checks that `figsize=` is a `(width, height)` pair of finite pixel values above 0. The check runs at the figure's entry, before any size is derived, and `None` leaves the size unset. A malformed value (0, negative, NaN, infinity, a `bool`, a non-number) raises `EntryRefusal` (a `ValueError` subclass) naming the argument the caller passed: `displot(height=...) must be positive; received 0`, `displot(aspect=...) must be a finite number a float can hold; received nan`, `displot(height=...) must be a numeric pixel value (int or float), not bool; received True`, `clustermap(figsize=...) must be a (width, height) pair; received (300,)`. A size that is 0 only because the data is empty (row facets over an empty frame) is left unset, so the empty chart still renders.

**Multi-model comparison (`compare=`).** A model-diagnostic function that takes `compare=` compares several models in one figure. There are three equivalent spellings: a dict as the first argument (`roc_chart({"a": model_a, "b": model_b}, X, y)`), the `compare=` keyword (`roc_chart(model_a, X, y, compare={"alt": model_b})`, where the first model is named `"base"`), or a `ComparedModelSource` built with `ModelSource.compare(models, X, y)` and passed first. Each model's frame carries a `model` column, and the layout depends on the chart:

| Layout | Charts |
|---|---|
| One panel, curves or points colored by `model` | `roc_chart`, `pr_chart`, `calibration_chart`, `gain_chart`, `lift_chart`, `residuals_chart` with a single panel (`panels="single"` or `None`), `prediction_error_chart` without `ci=` or `reference_band=` |
| One shared-axis panel, marks dodged by `model` (color `model`, a model legend) | `importance_chart`, `shap_bar_chart` with `per_class=False` (the default), `cv_scores_chart` with `kind="box"` (the default) or `"strip"` |
| Small multiples: a `ConcatChart` with one panel per model, in one row, each panel titled with its model's name; x and y scales shared | `discrimination_threshold_chart`, `shap_bar_chart` with `per_class=True`, `shap_beeswarm_chart`, `shap_waterfall_chart`, `shap_chart`, `learning_curve_chart`, `validation_curve_chart`, `alpha_selection_chart`, `cv_scores_chart` with `kind="bar"`, `cooks_distance_chart`, `residuals_chart` with several panels (the default `panels="auto"`), `prediction_error_chart` with `ci=` or `reference_band=` |
| Small multiples, x independent and y shared | `pdp_chart` |
| Small multiples, x and y independent | `pca_scree_chart` (fitted PCA estimators), `intercluster_distance_chart`, `silhouette_chart`, `manifold_chart` |

Each small-multiples panel is built by calling the single-model chart on that model alone, so everything it computes (a confidence band, a leverage panel, an F1 optimum) comes from that model's data only, never pooled. When the single-model chart is itself a composite (multi-panel `residuals_chart`, `pdp_chart`), each panel is that composite, titled with the model name. A refusal raised while a panel builds is prefixed with the model that refused: `compare={'alt': ...}: ...`.

`discrimination_threshold_chart` composes rather than overlays because its color channel already carries the metric legend. Each panel draws its own model's sweep, F1-optimum annotation and `threshold_line` rule, and `DiscriminationThresholdVisualizer(compare=)` renders through the same builder, so its `"base"` panel annotates the threshold and F1 that its repr reports.

The dodged layouts share one categorical axis across models, with each band's bars in model registration order (`"base"` first):

- `importance_chart` ranks features once across all models by mean absolute importance (descending; a NaN-scored feature sorts last), keeps that one top-`top_k` set for every model, and computes the value-axis domain over the combined frame. Error rules and value labels dodge with their bars. `orient="horizontal"` (the default) draws the vertical form under `CoordFlip`; `orient="vertical"` draws it directly.
- `shap_bar_chart` pools every model's SHAP values, ranks features once with the `order=` rule, keeps one top-`max_display` set, and aggregates each model's bars per feature. It is always horizontal, drawn as the vertical form under `CoordFlip`.
- `cv_scores_chart` keeps `split` (`train`/`test`, filtered by `split=`) as the categorical axis and draws one box or strip group per model in each split band. Under `compare=`, `kind="strip"` draws no jitter, since one layer takes one position adjustment and the dodge uses it.

`parallel_coordinates_chart`, `rank1d_chart` and `rank2d_chart` need no model: each accepts a DataFrame or 2-D array directly, and none takes `compare=`. `confusion_matrix_chart`, `class_prediction_error_chart`, `classification_report_chart`, `class_balance_chart` and `decision_boundary_chart` have no `compare=` parameter. `cluster_diagnostics` and `elbow_chart` take `compare=` for signature uniformity but refuse any value other than `None` (`compare= is not supported for cluster_diagnostics: this takes the feature matrix and sweeps one clusterer class over k internally; ...`), because they sweep one clusterer class over `k` and wrap no per-model source. `pca_scree_chart` refuses `compare=` on its raw-matrix path (`pca_scree_chart: compare= requires a fitted PCA estimator; the raw-DataFrame / array path computes Rust SVD on a single matrix and has no per-model source to compare.`). `random_state=` seeds the randomized steps of the chart's source as described for `ModelSource` (§3.1).

**Explicit positional scales through composite marks.** A composite mark (box, violin, cv_scores, letter-value and the rest) rebuilds each layer's encoding when it desugars, and an explicit `scale=` on the chart-level `x` or `y` channel carries onto those layers by one rule: a layer channel with no scale takes the chart-level scale whole; a layer scale with no `domain` (validation_curve's log x) takes only the chart-level domain and keeps its own `type`, `range` and other keys; a layer scale that already has a `domain` (SHAP's mark-computed domain) is left alone. Only `x` and `y` carry; `x2`/`y2` share their axis's scale through the primary channel. So a `compare=` composition resolves shared scales over composite-mark panels too: `cv_scores_chart(kind="bar")` panels under `compare=` draw one union y axis.

#### Distribution

```
ferrum.displot(data, *, x=None, y=None, hue=None, col=None, row=None,
               kind="hist",          # "hist" | "kde" | "ecdf" | "rug"
               fill=True, cumulative=False, log_scale=False, stat="count",
               bins="sturges", bandwidth="scott", bw_adjust=1.0,
               multiple="layer",     # "layer" | "stack" | "fill" | "dodge"
               kde=False, rug=False, height=None, aspect=None,
               mark=None, encode=None, properties=None, layers=None,
               theme=None, **encode_kwargs)
```

`displot` returns a `Chart` that draws one univariate distribution per `hue` level.

| `kind` | Draws | Axes |
|---|---|---|
| `"hist"` (default) | `mark_histogram` bars | x: the variable; y: `count`, or `density` under `stat="density"` |
| `"kde"` | `mark_density`, filled unless `fill=False` | x: the variable; y: `density` |
| `"ecdf"` | a cumulative `Bin` drawn as `mark_line`, one line per `hue` level on bins the levels share | x: titled with the variable; y: `count` |
| `"rug"` | `mark_tick` ticks, against `x=` or `y=` | the bound axis |

**Validation.** `kind=`, `multiple=` and `stat=` are closed vocabularies, checked for every `kind` before anything is built: `displot: stat must be one of ['count', 'density']; got 'probability'` (and the same shape for `kind` and `multiple`). The kinds `"hist"`, `"kde"` and `"ecdf"` draw the variable `x=` names and compute the other axis, so a call without `x=` refuses, in the caller's own column name: `displot(kind='hist', y='v') is not supported -- kind='hist' draws the distribution of the variable x= names and computes the other axis itself (a count, a density, a cumulative frequency), so a y= variable has nowhere to go. Pass x='v', or kind='rug', which draws its ticks against either axis.` `kind="rug"` draws from `x=` or `y=`. The `kde=` and `rug=` overlays are drawn against the `x=` variable too, so `displot(kind='rug', y='v', kde=True)` refuses: `... is not supported -- the kde= overlay is drawn against the variable x= names, and this call binds y= instead, so the overlay has no variable to read. Pass x='v' (kind='rug' draws the same ticks from x= or y=), or drop kde=.`

**`bins=`.** The vocabulary is closed, and is the one `Bin(bins=)` parses (§3.4):

| Value | Meaning |
|---|---|
| `"sturges"` | Sturges' rule (the default) |
| `"fd"` / `"freedman_diaconis"` | Freedman-Diaconis rule |
| `int` ≥ 1 (a numpy integer too) | an exact bin count |
| positive finite `float` | a bin width in data units |
| `None` | Sturges |

Anything else raises `ValueError` in `displot`'s voice, with no case or whitespace coercion: `displot: unknown bins value 'auto'; expected 'sturges'|'fd'|int|float` (also for `"scott"`, `"FD"`, `"fd "`); `displot: bins must not be a bool (True); expected 'sturges'|'fd'|int|float`, checked before the integer arm; `displot: bins as an int must be a bin count >= 1 within machine range; got 0` (also for negatives and integers too large for the machine); `displot: bins as a float must be a positive finite bin width; got 0` (also for NaN and infinity); and `displot: bins must be 'sturges'|'fd'|int|float` for a list, tuple, dict or other object. `bins=` is validated for every `kind`, but only `"hist"` and `"ecdf"` bin. A rule stated on `kind="kde"` or `kind="rug"` warns once and renders identically for every value: `displot(kind='kde', bins=7): bins= is inert for kind='kde' -- it evaluates no binned statistic, so the rule you named cannot act on anything and the render is identical for every value. Drop bins=, or use kind='hist'/'ecdf', which bin.` `bins=None` and `bins="sturges"` do not warn. `Bin2D(bins_x=, bins_y=)` and `Kde(count_scale=)` use the same parser and refuse the same values at construction in their own voice (`Bin2D: ...`, `stat_kde: ...`). On the wire a Freedman-Diaconis `Bin` carries `"bin_rule": "freedman_diaconis"`, Sturges carries no mode key, and a payload naming `bin_rule` together with `bin_count` or `bin_width` refuses (`stat_bin: bin_rule '<rule>' cannot be combined with bin_count/bin_width`). A Freedman-Diaconis `Bin`'s repr ends in `bins='freedman_diaconis'` (the repr rule is §3.5's).

**Density parameters.** `bandwidth=` (default `"scott"`) and `bw_adjust=` (default `1.0`) reach the `Kde` transform of `kind="kde"` and of the `kde=True` overlay. `bandwidth` takes `"scott"`, `"silverman"`, a positive finite float, or `None` (Scott); any other string raises `Kde: unknown bandwidth 'foo'; expected 'scott' | 'silverman' | float` (§3.4). `fill=` applies to `kind="kde"` only.

**`cumulative=`.** Under `kind="hist"` the bars are a cumulative histogram; under `kind="kde"` the curve is the CDF, grouped fits included. The `kde=True` overlay takes the same `cumulative` value, so a cumulative histogram's overlay is a CDF.

**`multiple=`.** One owner (`marks/_multiple.py`) decides both the position adjustment and the binning extent of each mode, so bars that must line up are binned on the same edges:

| `multiple` | Position adjustment (grouped by `hue`) | Histogram bin edges |
|---|---|---|
| `"layer"` (default) | `Identity()`: groups overlap | each `hue` group bins over its own range |
| `"stack"` | `Stack` with `offset="zero"` | one set of edges shared by every group |
| `"fill"` | `Stack` with `offset="normalize"`: bars fill `[0, 1]` | shared |
| `"dodge"` | `Dodge`: each group takes one slot of the bin's band | shared |

Under `"stack"` and `"fill"` every group's bar in a bin sits on the same x and width. Under `"dodge"` every group's bars have one width on one grid, side by side and non-overlapping, one slot per group within each bin. Without `hue=` every mode draws one plain histogram. `mark_histogram(position_by=)` is the public door this forwarding uses (§3.3): the column the `multiple=` adjustment groups by, a column name or `None`; anything else refuses, e.g. `mark_histogram: position_by must be a column name (str) or None, not Color; received Color('g'). Pass the column name itself: position_by='g'.` Limit: `kind="kde"` does not support `"stack"` or `"fill"`. The mode's position adjustment is applied, but the densities are evaluated per group on grids the groups do not share, so the result is not a stacked or normalized density.

**Overlays.** `kde=True` (on a kind other than `"kde"`) adds a `line` layer named `kde` that reads the original rows through a named `Kde` transform, grouped by `hue` and colored like the primary mark, one curve per level. Under `kind="hist"`, `stat="count"` and any `multiple` other than `"fill"`, where the value axis is in counts, the curve is count-scaled: density × group size × bin width (the CDF × group size when cumulative), with the bin width taken from the same `bins=` rule and binning extent as the bars, so the curve tracks the bars it is drawn against. Under `stat="density"`, `multiple="fill"`, `kind="ecdf"` or `kind="rug"` the curve is an unscaled density. Under `multiple="stack"` the count-scaled curves are drawn from zero, not stacked on the groups below. `rug=True` (on a kind other than `"rug"`) adds a `tick` layer named `rug` that reads the original `x=` values. Both overlays carry an explicit `Identity()` position, so neither is ever stacked, dodged or normalized by `multiple=`. On a layered chart the primary layer is named `histogram`, `kde` or `rug` after its mark, the names `mark=` overrides target. `displot(x=..., kind="rug", kde=True)` binds the overlay's density column as the chart's y axis, titled `density`, the same title `kind="kde"` and `kind="kde", rug=True` give that axis; the rug ticks sit at its base, and with `hue=` it draws one curve per level.

**Encoding.** `hue=` is a column name or a channel object (`fm.Color("g")`) and is always encoded as a nominal color, so a numeric group column gets a swatch legend, not a colorbar. `log_scale=True` puts a log scale on x. `col=`/`row=` facet the chart.

**Sizing.** With neither `height=` nor `aspect=` (and no `row=`), the chart keeps the default canvas size. Otherwise `height` (default 300 when only `aspect` is given) is the canvas height and the width is `height × aspect` (`height` when `aspect` is unset). Under `row=` faceting, `height` is the height of one row panel (default 150 px), the canvas height is that times the number of row levels, and the width is `height × aspect` when `aspect` is given, not growing with the panel count. Under `col=` faceting alone, `height` and `aspect` size the whole canvas.

#### Categorical

```
ferrum.catplot(data, *, x=None, y=None, hue=None, col=None, row=None,
               kind="strip",         # "strip" | "swarm" | "box" | "violin" |
                                     # "boxen" | "point" | "bar" | "count"
               order=None, hue_order=None, orient=None,
               dodge=False, jitter=True, native_scale=False,
               ci=95, n_boot=1000, seed=None,
               height=None, aspect=None,
               mark=None, encode=None, properties=None, layers=None,
               theme=None, **encode_kwargs)
```

`kind` picks the mark: `"strip"` is `mark_point` with `Jitter` (or `Dodge` when `dodge=True` and `hue=` is set, which replaces the jitter), `"swarm"` is `mark_swarm`, `"box"` `mark_boxplot`, `"violin"` `mark_violin`, `"boxen"` `mark_boxen`, `"count"` an `Aggregate(count)` drawn with `mark_bar`, and `"point"`/`"bar"` are the two estimating kinds below. An unknown `kind` raises `ValueError` listing the eight. `height`/`aspect` have the same meaning as on `displot`: with `row=` set, `height` is the per-panel height.

**Axes and orientation.** `x` is the category axis and `y` the value axis. `orient="h"` swaps them (`y` names the category, `x` the value) and adds `CoordFlip`; any other value, including `None` and `"v"`, draws vertically. `x`, `y` and `hue` each take a column name or an already-built channel object (`fm.X("g")`, `fm.Color("h")`); every option stated on the caller's channel (`sort=`, `scale=`, `title=`, `type_=`, ...) reaches all eight kinds.

**The category channel is always typed.** `native_scale=False` (the default) types it `"nominal"`, so even a numeric column (an integer year) renders as equal-spaced categorical bands. `native_scale=True` types it `"quantitative"` and places each category at its numeric position; a non-numeric column raises `catplot(native_scale=True): the category column 'g' is not numeric (dtype String), so it has no native scale to place values on. Pass native_scale=False (the default) to render it as categorical bands instead.` An explicit `type_=` on the caller's own channel is honored verbatim and bypasses `native_scale`, including its dtype check. On a native scale, `kind="strip"`'s jitter width is `0.4` times the column's median spacing between unique values (`0.4` data units when there are fewer than two unique finite values), and whenever a strip jitters or any kind dodges, the category domain is padded by half the adjustment's maximum offset plus 5% so no adjusted mark falls outside it. The padding never overrides a scale the caller set on the category channel. `kind="boxen"`'s letter-value transform requires a string category column whatever `native_scale` says.

**Hue.** A bare-string `hue=` is typed nominal on every kind. A channel object with no `type_` is typed nominal too; one that states a continuous `type_` keeps it (and its colorbar). `hue_order=` declares the hue categorical: with a channel object it rebuilds the channel with `sort=hue_order` (and nominal typing when the channel states no type), keeping the channel's other options. `hue_order=` beside a quantitative or temporal `type_` raises `catplot(hue=fm.Color('h', type_=...), hue_order=...): hue_order= cannot order a quantitative hue channel -- there is no discrete legend order for a continuous scale to apply. Drop hue_order=, or type the channel nominal/ordinal (e.g. type_='nominal') so hue_order= has levels to order.` With `dodge=False` (the default) the hue levels of one category overlay at the same position; `dodge=True` places them side by side.

**Ordering precedence.** `order=` and `x=fm.X(..., sort=[...])` order the category axis identically, and `hue_order=` and `hue=fm.Color(..., sort=[...])` the hue levels and legend, on all eight kinds and with byte-identical output for the same order. When both spellings are given, the keyword wins. `encode={"x": ...}` is a raw override that replaces the builder's category channel on every kind, discarding its typing and its native-scale padding; prefer `x=fm.X(...)` to keep both.

**Estimating kinds.** `kind="point"` and `kind="bar"` draw one estimate per (category, `hue`) group: a `Summary` transform computes the mean and a bootstrap confidence interval, drawn as a capless whisker under the point marker or over the bar. Both are layered charts whose layers are named `"whisker"` and `"point"`/`"bar"`, so `mark=` takes per-layer keys (`mark={"bar": {...}}`). `ci` is a percent in the open interval `(0, 100)` and `n_boot` an integer `>= 1`; `bool` and numeric strings are refused for both, and the refusal reads `catplot(ci=100) is not supported -- ci is a PERCENT and must be a number in the open interval (0, 100), exclusive at both ends (...). Pass ci=95 (the default) or any percent strictly between 0 and 100.` `seed=` seeds the bootstrap and the strip jitter; `seed=None` maps to a fixed seed, so output is byte-deterministic. Both kinds need a category and a value channel and raise `catplot(kind='point') summarises a value column per category, so it requires both a category and a value channel: ...` without one. The other six kinds draw no interval: a non-default `ci=` or `n_boot=` on them raises `catplot(ci=90) is not supported with kind='box' -- that kind draws no confidence interval, so ci= would be ignored. Drop ci=, or use one of the estimating kinds ('bar', 'point').`

- **Per-observation channels are refused on the estimating kinds.** `size`, `shape`, `opacity`, `fill_opacity`, `stroke_opacity`, `stroke_width`, `stroke_dash`, `angle`, `color`, `fill` and `stroke`, through a bare keyword or `encode=`, raise `catplot(kind='point', size=...) is not supported -- those channels map a visual property per OBSERVATION, and kind='point' draws one estimate per (category, hue) group from a Summary aggregate, ...`, naming `hue=` (which joins the `Summary` groupby, so its legend is true) and the per-row kinds as the routes. The six per-row kinds accept all of these.
- **A value-axis `scale=` needs a `domain`.** An explicit value-axis scale with a `domain` governs both layers; a domain-less one (a bare `{"type": "log"}`) raises `catplot(kind='point'): a value-axis scale= without an explicit domain cannot be honored, because 'point' draws an estimate and its CI whisker from different columns of one Summary output, ... Add a domain (e.g. scale={'type': 'log', 'domain': [lo, hi]}), or use a kind that draws one mark per row.` A scale on the category channel is always honored.
- **Bars start at zero.** A layered chart's auto-inferred value-axis domain extends to zero whenever a bar layer binds that channel, the same zero anchor a single-chart `mark_bar` gets from its `scale.zero`, unless the bar binds an explicit span channel (`y2`/`x2`). This rule applies to every layered spec with a bar layer, not only `catplot`. `mark_bar(zero=False)` opts a layer out and travels with the mark as `mark_style.zero`, so it holds on the merged single-panel path (`resolve={"y": "independent"}`, `SecondaryY`, interactive `LayerChart`); on `catplot(kind="bar")` the same opt-out is `mark={"bar": {"zero": False}}`.

**`kind="count"`** counts rows of the resolved category channel per category (and per `hue` level when `hue=` is set, unless `hue` names the category column), so a channel object works and an `encode={"x": fm.X("other")}` override counts the column the axis actually binds. It requires a category channel: `catplot(kind='count') requires x= (or y= when orient='h')`. The count lands on the value axis as column `n`.

#### Relational

```
ferrum.relplot(data, *, x, y, hue=None, size=None, style=None,
               col=None, row=None,
               kind="scatter",        # "scatter" | "line"
               height=None, aspect=None,
               mark=None, encode=None, properties=None, layers=None,
               theme=None, **encode_kwargs)
```

`size` maps to the `Size` channel (point area or line width). `style` maps to `Shape` for `kind="scatter"` or `StrokeDash` for `kind="line"`. `hue` keeps dtype inference on `kind="scatter"` (a numeric column renders a gradient) and is typed nominal on `kind="line"`, which groups lines by it.

#### Regression

```
ferrum.lmplot(data, *, x, y, hue=None, col=None, row=None,
              method="lm",           # "lm" | "loess" | "quadratic" | "cubic" | "log" | "sqrt" | "gam" | "logistic" | "glm" | "robust"
              ci=95, order=None, scatter=True, scatter_kws=None, line_kws=None,
              truncate=True, x_bins=None, x_estimator=None, x_jitter=None,
              logx=False, show_metrics=True,
              mark=None, encode=None, properties=None, layers=None,
              theme=None, **encode_kwargs)

ferrum.regplot(data, *, x, y, hue=None,
               method="lm",          # same as lmplot
               ci=95, order=None, scatter=True, scatter_kws=None, line_kws=None,
               truncate=True, x_bins=None, x_estimator=None, x_jitter=None,
               logx=False, show_metrics=True,
               mark=None, encode=None, properties=None, layers=None,
               theme=None, **encode_kwargs)

ferrum.residplot(data, *, x, y, lowess=False, order=1, robust=False,
                 dropna=True, show_metrics=True, zero_line=True,
                 label=None, color=None,
                 mark=None, encode=None, properties=None, layers=None,
                 theme=None, **encode_kwargs)
```

**Methods.** `"lm"`, `"loess"`, `"quadratic"`, `"cubic"`, `"log"`, `"sqrt"` and `"gam"` route through `mark_smooth`; this set derives from the `Smooth` registry's methods minus `"linear"` and `"glm"`. `"logistic"`, `"glm"` and `"robust"` use their own `Logistic`, `Glm` (gaussian family) and `Robust` transforms. `mark_smooth`'s alias spellings (`"linear"`, `"lowess"`) are not accepted: an unknown method raises `lmplot: method must be one of ['cubic', 'gam', 'glm', 'lm', 'loess', 'log', 'logistic', 'quadratic', 'robust', 'sqrt']; got 'linear'`.

**Parameters.**

- `order=None` uses the method's own default degree. An explicit `order` is forwarded as `degree=` to every `mark_smooth`-routed method, and `Smooth`'s degree matrix decides: `"lm"` accepts any degree `>= 1` (`>= 2` fits a polynomial); `"quadratic"`/`"cubic"` accept only their own degree; for `"loess"` it is the local-fit degree, `1` or `2`; `"log"`/`"sqrt"`/`"gam"` refuse any degree. A negative `order` raises naming `lmplot`. `order` is ignored for the three dedicated methods.
- `ci` is a percent (`ci=95`), or `None` for no band.
- `truncate=True` clips the fit to the observed x range; `truncate=False` extends it over the data range padded 5% per side, for every `mark_smooth`-routed method and `"robust"`. `"logistic"`/`"glm"` refuse it: `lmplot(method='glm', truncate=False) is not supported -- the 'glm' transform has no x_range parameter ...`.
- `x_bins=`/`x_estimator=` pre-aggregate the fit input on every `mark_smooth`-routed method (`x_estimator` defaults to the transform's `"mean"` and is checked against `fm.Smooth`'s vocabulary). The three dedicated methods refuse either one alone: `lmplot(method='robust', x_bins=...) is not supported -- the 'robust' transform has no x_bins/x_estimator binning parameters ...`.
- `x_jitter=` jitters the scatter along x; an integer predictor is widened to float first, so it works on any numeric dtype.
- `logx=True` puts a log scale on the x channel every layer shares. A channel that already carries its own `scale=` refuses: `lmplot(logx=True, x=<channel with its own scale=>): two instructions for one x scale. Keep your scale= and drop logx=True, or pass x as a column name and let logx set the log scale.`
- `show_metrics=True` adds a top-right corner annotation with `R²` / `RMSE` / `MAE`, computed in Rust, for every `mark_smooth`-routed method and `"robust"`. It is skipped for `"logistic"`/`"glm"` and whenever `hue=` is set.
- `x=` and `y=` take a column name or an encoding object wrapping one, on every method; the channel's `title=`/`scale=` survive. `None` raises `lmplot(x=None): x names the column to fit and has no default. Pass a column name, or an encoding object wrapping one.`

**Hue and pooled fits.** For the `mark_smooth`-routed methods, `hue=` (typed nominal) refits per level, each fit and band in its level's color. `"logistic"`, `"glm"` and `"robust"` fit one curve pooled across all rows; when any channel that resolves to `color` is active (`hue=`, or `color=`/`fill=`/`stroke=` as a keyword or in `encode=`), the scatter is colored per level and the pooled fit and band draw in a neutral color.

**Layers and styling.** With a scatter, the chart's layers are `"scatter"`, then `"ribbon"` (when `ci` is set), `"line"` and `"metrics"` (when metrics apply); `mark={...}` keys use these names. The scatter always draws the raw observations: the fit binds to a named transform its own layers read. `scatter_kws` styles the scatter and `line_kws` the fit line alone, on every method and shape. Every fit path titles its axes with the caller's column names (or the channel's own `title=`), with or without a scatter.

**`regplot`** is `lmplot` without faceting: every other parameter is forwarded to `lmplot` unchanged, and its refusals read as `lmplot`'s. `col=` or `row=` raises `TypeError: regplot(col=...) is not supported -- regplot is the axes-level regression plot and does not facet. Use lmplot(..., col=...) instead.` The grammar-level `facet=` encoding channel stays available, as on any `Chart`.

**`residplot`** fits `Smooth(method="lm", degree=order)` (or `Robust` with `robust=True`) in residual mode and plots `(x, residual)` with `mark_point`. Its axes are titled with the caller's x column and `"<y> residual"`. `dropna=True` drops rows null in `x` or `y` first. `zero_line=True` adds a dashed rule at 0; `lowess=True` overlays a LOESS curve of the residuals; `show_metrics=True` adds the corner `R²` / `RMSE` / `MAE`. Layer names are `"point"`, `"reference"`, `"metrics"` and `"lowess"`. `x=`/`y=` take a column name or an encoding object and refuse `None` as `lmplot` does, voiced `residplot`.

- **`color=`** is checked against the input frame's columns first. A column name groups the residuals: the fit is refit per group, the points and the lowess overlay (one curve per group) color by it, and the corner metrics are suppressed. Otherwise the value is parsed as a literal color (CSS name, hex, `rgb()`/`rgba()`, RGB tuple) and painted as the points' `fill`, never as a color encoding. A value that is neither raises `residplot(color='x'): not a column of the data and not a parseable color. As a column name it would group the data, ...; as a literal color it would paint the marks, but the color parser refused it (...)`.
- **`label=`** adds a single legend entry through a private `_label` column. It composes with `robust=True`, with a grouping `color=`, and with a literal `color=` (the swatch and the points take that color). It keeps the corner metrics. A frame that already has a `_label` column refuses: `residplot(label=...) needs a private '_label' column, but the data already has one -- the label would overwrite it silently. ...`

#### Matrix / Pairwise

```
ferrum.pairplot(data, *, vars=None, x_vars=None, y_vars=None,
                hue=None, kind="scatter",     # off-diagonal: "scatter" | "kde" | "hist" | "reg"
                diag_kind="auto",             # "auto" (= "kde") | "hist" | "kde" | None | "none"
                markers=None, height=None, aspect=None,
                corner=False, dropna=False,
                mark=None, encode=None, properties=None, layers=None,
                theme=None, **encode_kwargs)

ferrum.heatmap(data, *, annot=True, fmt=".2f", scheme=None, cmap=None,
               linewidths=None, linecolor=None,   # None: 0.5 px, "white"
               vmin=None, vmax=None, center=None, robust=False,
               square=False, mask=None,
               mark=None, encode=None, properties=None, layers=None,
               theme=None, **encode_kwargs)

ferrum.clustermap(data, *, method="ward", metric="euclidean",
                  scheme=None, cmap=None,
                  linewidths=None, linecolor=None,   # None: 0.5 px, "white"
                  z_score=None, standard_scale=None,
                  figsize=None, dendrogram_ratio=0.2,
                  mark=None, encode=None, properties=None, layers=None,
                  theme=None, **encode_kwargs)
```

`pairplot` returns a `RepeatChart`; `heatmap` returns a `Chart` (layers `"cells"` and, with `annot=True`, `"label"`, titled `"Heatmap"` by default); `clustermap` returns a `ClusterMapChart` (center heatmap plus row and column dendrograms, titled `"Clustermap"`).

**Colormap.** `scheme=` is the canonical colormap kwarg on `heatmap` and `clustermap`; `cmap=` is its alias. Passing both with different values raises `heatmap: pass either scheme= or cmap= (its alias), not both with different values; got scheme='blues', cmap='rdbu'`. A name that is not a registered continuous colormap raises at the call, before rendering: `heatmap: color scheme 'nope' is not recognized; available: viridis, plasma, magma, ...`. Unset, the color scale takes the theme's `diverging_scheme` when its domain spans zero and the theme's `sequential_scheme` otherwise.

**Heatmap color domain.** `vmin`, `vmax` and `center` must be finite real numbers (`bool` refused); a non-finite value raises `heatmap(vmin=...) must be a finite number a float can hold; received inf` before the chart is built. They are honored end to end through a Diverging scale (when `center=` is set, as its `domainMid`) or a Linear scale with an explicit `domain`, `robust=True` sets each endpoint the caller left unset to the 2nd or 98th percentile of the finite cell values. When only one of `vmin`/`vmax` is given, the other is filled from the finite pre-mask data extent; with no finite data to fill from, the override is dropped with a `UserWarning` naming the reason. A filled domain that comes out inverted (`vmin=50` above the data max), or an explicit `vmin > vmax`, renders deterministically as one flat color, not a reversed ramp; the filled case warns, naming both endpoints. A `center=` outside the effective domain renders a one-sided compressed ramp and warns. The colorbar shows the same scale the cells use, so under `center=` its midpoint color sits at `center`. These `heatmap` warnings point at the caller's own line.

**Masking and shape.** `mask="upper"` keeps the upper triangle and the diagonal, `"lower"` the lower triangle and the diagonal, and a boolean array shaped like the numeric block hides its `True` cells; any other string, or a mis-shaped array, raises. `square=True` sets width and height to 30 px times the larger of the row and column counts. `annot=True` labels each cell with its value formatted by `fmt`.

**Cell border.** `heatmap` and `clustermap` share one cell-border rule: `linewidths=None` means 0.5 px and `linecolor=None` means `"white"`, so both draw a white 0.5 px stroke per cell by default, and `linewidths=0` draws none. `linewidths` must be a finite, non-negative real number (`bool` excluded); `linecolor` must be a color `ferrum.color.to_hex` accepts or a paint sentinel. Refusals name the figure and the argument:

- `linewidths=-1`: `heatmap(linewidths=...) must be non-negative; received -1`.
- `linewidths=float("nan")`: `heatmap(linewidths=...) must be a finite numeric pixel value; received nan`.
- `linewidths=True`: `heatmap(linewidths=...) must be a numeric pixel value (int or float), not bool; received True`.
- `linewidths="a"`: `heatmap(linewidths=...) must be a numeric pixel value (int or float), not str; received 'a'`.
- `linecolor="notacolor"`: `heatmap(linecolor=...)='notacolor' is not a valid color (invalid color 'notacolor': expected a CSS color name, #rrggbb[aa]/#rgb[a] hex, or rgb()/rgba())`.
- `linecolor=3`: `heatmap(linecolor=...)=3 must be a color string, not int`.

`clustermap` refuses the same values with the same text, voiced `clustermap(...)`, and a malformed `linecolor` is refused even beside `linewidths=0`. `linecolor="none"` or `"transparent"` draws no border, like `linewidths=0`; `"theme:label"` draws the theme's label color. An explicit value made inert by the other argument warns, naming both: `heatmap: linewidths=2.0 has no effect when linecolor='none' draws no cell border; it was ignored` and `heatmap: linecolor='red' has no effect when linewidths=0.0 draws no cell border; it was ignored`; `None` on either side never warns. A `mark={"stroke": ..., "stroke_width": ...}` override still wins on the cells. Other figures that draw cells (the confusion-matrix and classification-report charts, `rank2d_chart`, `jointplot(kind="hist")`) draw no border by default; `mark=` reaches one.

**Clustermap.** `method` and `metric` go to `Linkage`. `z_score` and `standard_scale` take `0` (rows), `1` (columns), `"rows"`, `"columns"` or `None`; anything else raises `clustermap: z_score=2 is invalid; expected 0 (rows), 1 (columns), 'rows', 'columns', or None`. `figsize=(width, height)` sizes the center heatmap in pixels. `dendrogram_ratio` is the share of the width and height each dendrogram panel takes. The dendrogram panels hide their axes and reserve no gutter for them, so their whole share is branch area; a ratio too small for the panel's padding raises a layout error (`padding top=16 exceeds viewport dimension ...`) rather than drawing a blank panel.

#### Joint Distribution

```
ferrum.jointplot(data, *, x, y, hue=None,
                 kind="scatter",       # "scatter" | "kde" | "hist" | "hex" | "reg"
                 marginal_kind="hist", # "hist" | "kde" | "rug" | "box"
                 ratio=5, space=0.05,
                 xlim=None, ylim=None,
                 joint_kws=None, marginal_kws=None,
                 height=None,
                 mark=None, encode=None, properties=None, layers=None,
                 theme=None, **encode_kwargs)
```

Returns a `JointChart`. `kind` controls the center panel mark; `marginal_kind` controls both marginal panels. `joint_kws` and `marginal_kws` are passed as keyword arguments to the center and marginal mark calls respectively.

#### Model Diagnostics (figure-level)

The model-diagnostic figure functions take a fitted estimator, a `ModelSource`, or (where `compare=` is accepted) a multi-model source as their first argument `model`. The exceptions are `parallel_coordinates_chart(data)`, `rank1d_chart`/`rank2d_chart`/`rank_chart(data_or_source)`, `class_balance_chart(y)`, and `cluster_diagnostics(model)`, whose `model` is the feature matrix it sweeps clusterers over. Every function takes the keyword-only override arguments `mark=`, `encode=`, `properties=`, `layers=` and `theme=`; the model-backed ones also take `random_state=`, which is forwarded to the `ModelSource`. Multi-model `compare=` is described at the start of §3.14; `cluster_diagnostics` and `elbow_chart` accept the keyword but refuse any non-`None` value with a `ValueError`, because they wrap no per-model source.

```
ferrum.roc_chart(model=None, X=None, y=None, *,
                 y_true=None, y_pred=None,   # precomputed path (1-D binary or 2-D multiclass scores)
                 per_class=True, average="macro", annotate_auc=True,
                 subtitle=None, compare=None, random_state=None,
                 mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.pr_chart(model=None, X=None, y=None, *,
                y_true=None, y_pred=None,    # precomputed path (1-D binary or 2-D multiclass scores)
                per_class=True, average="macro", annotate_ap=True, iso_lines=False,
                subtitle=None, compare=None, random_state=None,
                mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.confusion_matrix_chart(model=None, X=None, y=None, *,
                               y_true=None, y_pred=None,   # precomputed path (1-D hard labels)
                               normalize="true",           # "true" | "pred" | "all" | None (counts)
                               annotate=True, random_state=None,
                               mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.classification_report_chart(model=None, X=None, y=None, *,
                                    y_true=None, y_pred=None,   # precomputed path (1-D hard labels)
                                    random_state=None,
                                    mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.calibration_chart(model=None, X=None, y=None, *,
                          y_true=None, y_pred=None,  # precomputed path (1-D positive-class proba)
                          n_bins=10, strategy="uniform",   # "uniform" | "quantile"
                          annotate_brier=True, subtitle=None, compare=None, random_state=None,
                          mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.gain_chart(model=None, X=None, y=None, *,
                  y_true=None, y_pred=None,          # precomputed path (soft scores)
                  compare=None, subtitle=None, random_state=None,
                  mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.lift_chart(model=None, X=None, y=None, *,
                  y_true=None, y_pred=None,          # precomputed path (soft scores)
                  compare=None, subtitle=None, random_state=None,
                  mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.residuals_chart(model=None, X=None, y=None, *,
                        y_true=None, y_pred=None,    # precomputed path (fitted values); residuals = y_true - y_pred
                        kind="studentized",          # "studentized" | "raw"
                        cook_threshold=None,         # float | "auto" (4/n) | None
                        panels="auto",
                        # panels: "auto" | "single" | None | list of
                        #         "residuals_vs_fitted" | "qq" |
                        #         "scale_location" | "residuals_vs_leverage"
                        annotate_metrics=True, subtitle=None, compare=None, random_state=None,
                        mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.prediction_error_chart(model=None, X=None, y=None, *,
                               y_true=None, y_pred=None,   # precomputed path (fitted values)
                               reference_line=True, ci=None, reference_band=False,
                               compare=None, random_state=None,
                               mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.cooks_distance_chart(model=None, X=None, y=None, *,
                             threshold=None,             # float | "auto" | None
                             compare=None, random_state=None,
                             mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.importance_chart(model, X=None, y=None, *,
                         method="builtin", top_k=20, orient="horizontal",
                         error_bars=True, show_values=True,
                         subtitle=None, compare=None, random_state=None,
                         mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.shap_beeswarm_chart(model, X=None, y=None, *,
                            max_display=20, order="abs_mean", background=None,
                            per_class=False, zero_line=True, compare=None, random_state=None,
                            mark=None, encode=None, properties=None, layers=None, theme=None)
ferrum.shap_bar_chart(model, X=None, y=None, *,
                       max_display=20, order="abs_mean", background=None,
                       per_class=False, compare=None, random_state=None,
                       mark=None, encode=None, properties=None, layers=None, theme=None)
ferrum.shap_waterfall_chart(model, X=None, y=None, *,
                             sample_idx, max_display=20, order="abs_mean", background=None,
                             per_class=False, compare=None, random_state=None,
                             mark=None, encode=None, properties=None, layers=None, theme=None)
ferrum.shap_chart(model, X=None, y=None, *,
                   kind="beeswarm",   # "beeswarm" | "bar" | "waterfall"; deprecated dispatcher
                   max_display=20, sample_idx=None, order="abs_mean", background=None,
                   compare=None, random_state=None,
                   mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.pdp_chart(model, X=None, y=None, *,
                  features=None, grid_resolution=100, kind="average",
                  ice_alpha=0.2, center=False, compare=None, random_state=None,
                  mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.learning_curve_chart(model, X=None, y=None, *,
                              cv=5, scoring=None, train_sizes=None, ci_style="band",
                              subtitle=None, compare=None, random_state=None,
                              mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.validation_curve_chart(model, X=None, y=None, *,
                                param="alpha", values=None,   # values= is required
                                cv=5, scoring=None, log_scale="auto", ci_style="band",
                                subtitle=None, compare=None, random_state=None,
                                mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.alpha_selection_chart(model, X=None, y=None, *,
                               alphas=None, cv=5, scoring=None,
                               log_scale=True, highlight_best=True, compare=None, random_state=None,
                               mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.cv_scores_chart(model, X=None, y=None, *,
                        cv=5, scoring=None, kind="box", split="both",
                        compare=None, random_state=None,
                        mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.cluster_diagnostics(model, *, ks,   # model: the feature matrix
                             method="kmeans",   # "kmeans" | "hierarchical"
                             scoring="both",    # "elbow" | "silhouette" | "both"
                             n_init=10, compare=None, random_state=None,
                             mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.elbow_chart(model, X, *, ks, metric="distortion", compare=None, random_state=None,
                    mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.silhouette_chart(model, X=None, *, subtitle=None, compare=None, random_state=None,
                         mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.intercluster_distance_chart(model, X=None, *, k=None, method="mds",
                                    compare=None, random_state=None,
                                    mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.pca_scree_chart(model, X=None, *, n_components=None,
                        cumulative_line=True, threshold=0.95,
                        subtitle=None, compare=None, random_state=None,
                        mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.manifold_chart(model, X=None, *, method="umap", compare=None, random_state=None,
                       mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.decision_boundary_chart(model, X, y=None, *,
                                 features=(0, 1),    # column indices or names
                                 grid_resolution=200,
                                 proba=False,         # plot the probability surface if True
                                 scatter=True, random_state=None,
                                 mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.discrimination_threshold_chart(model=None, X=None, y=None, *,
                                       y_true=None, y_pred=None,  # precomputed path (1-D positive-class scores); cv= not supported
                                       n_thresholds=50,
                                       metrics=("precision", "recall", "f1", "queue_rate"),
                                       cv=None, threshold_line=False, optimum_label,
                                       compare=None, subtitle=None, random_state=None,
                                       mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.class_prediction_error_chart(model=None, X=None, y=None, *,
                                     y_true=None, y_pred=None,  # precomputed path (1-D hard labels)
                                     normalize=False, show_counts=True, random_state=None,
                                     mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.class_balance_chart(y, *, mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.parallel_coordinates_chart(data, *,
                                   features=None, hue=None,
                                   rescale="minmax",   # "minmax" | "zscore" | None
                                   alpha=0.5,
                                   mark=None, encode=None, properties=None, layers=None, theme=None)

ferrum.rank1d_chart(data_or_source, X=None, y=None, *,
                     algorithm=None,   # None selects "shapiro"
                     top_k=None, orient="horizontal", color_field=None, random_state=None,
                     mark=None, encode=None, properties=None, layers=None, theme=None)
ferrum.rank2d_chart(data_or_source, X=None, y=None, *,
                     algorithm=None,   # None selects "pearson"
                     annot=True, random_state=None,
                     mark=None, encode=None, properties=None, layers=None, theme=None)
ferrum.rank_chart(data_or_source, X=None, y=None, *,
                   rank="2d",       # "1d" | "2d"; deprecated dispatcher
                   algorithm=None, top_k=None, annot=True, orient="horizontal",
                   color_field=None, random_state=None,
                   mark=None, encode=None, properties=None, layers=None, theme=None)
```

`discrimination_threshold_chart`'s `optimum_label` is keyword-only with an unset default that behaves as `True`; see that function below.

**Precomputed input path.** Eleven prediction-evaluation functions accept `y_true=` and `y_pred=` as keyword-only arguments that bypass the fitted model entirely: `roc_chart`, `pr_chart`, `calibration_chart`, `gain_chart`, `lift_chart`, `discrimination_threshold_chart`, `confusion_matrix_chart`, `class_prediction_error_chart`, `classification_report_chart`, `residuals_chart` and `prediction_error_chart`. The chart is the same one the model path draws from the same predictions. Exactly one path must be active:

- neither: `Supply either a fitted model/source (model=) or precomputed arrays (y_true=, y_pred=).`
- both: `Supply either a model/source (model=) or precomputed arrays (y_true=, y_pred=), not both.`
- one array only: `Precomputed path requires both y_true= and y_pred=; y_pred= is missing.`
- `compare=` with arrays: `compare= is not supported with precomputed y_true/y_pred inputs.  Multi-model comparison requires a fitted model on each path.` (`confusion_matrix_chart`, `class_prediction_error_chart` and `classification_report_chart` take no `compare=` at all.)

All of these are `ValueError`. What `y_pred` holds depends on the function: soft scores for `roc_chart`, `pr_chart`, `gain_chart` and `lift_chart` (1-D positive-class scores for a binary target, or a 2-D `(n_samples, n_classes)` matrix); 1-D positive-class probabilities for `calibration_chart` and 1-D positive-class scores for `discrimination_threshold_chart` (which also refuses `cv=` on this path: `cross-validation requires a fitted model`); 1-D hard labels for `confusion_matrix_chart`, `class_prediction_error_chart` and `classification_report_chart`; 1-D fitted values for `residuals_chart` and `prediction_error_chart`.

A `y_pred` of the wrong shape is refused with a `ValueError` naming the figure function that was called and the observed shape, for example `confusion_matrix_chart requires 1-D hard class labels; got an array with shape (200, 3).` and `prediction_error_chart requires 1-D fitted values; got an array with shape (200, 2).` A 2-D score matrix whose column count differs from the class count is refused (`pr_chart: y_pred has 2 score column(s) but y_true has 3 class(es); ...`), and so is a 1-D `y_pred` against a `y_true` with more than two distinct classes on `roc_chart`, `pr_chart`, `gain_chart` and `lift_chart` (`roc_chart with a 1-D y_pred requires binary y_true; got 3 unique classes.`). A `y_true` holding a single class is not multiclass: the curve still renders, its AUC or AP is NaN, and no metric label is drawn for it. A null in the target is refused before any metric is computed (`pr_chart: y has 50 null row(s) out of 200, and a null is not a class observation ...`).

**Titles and metric labels.** `roc_chart` is titled `"ROC Curve"`, `pr_chart` `"Precision–Recall Curve"` and `calibration_chart` `"Calibration Curve"` in every shape (single curve, per class, averaged, compared, precomputed); the title never carries a metric. `annotate_auc`, `annotate_ap` and `annotate_brier` (all default `True`) each control one surface, a text label per curve placed at the curve's end and stacked so labels do not overlap one another: `AUC = 0.984`, `AP = 0.973`, `Brier = 0.060`, three decimals. Turning the flag off removes the number from the figure. Every function that takes `subtitle=` (`roc_chart`, `pr_chart`, `calibration_chart`, `gain_chart`, `lift_chart`, `residuals_chart`, `importance_chart`, `learning_curve_chart`, `validation_curve_chart`, `discrimination_threshold_chart`, `silhouette_chart`, `pca_scree_chart`) renders it beneath the title through `Title(text, subtitle=...)`. `roc_chart` and `pr_chart` title their legend `"Class"`, or `"Model"` under `compare=`.

**`roc_chart`, `pr_chart`.** With `per_class=True` (default) one one-vs-rest curve is drawn per class, and a binary target draws the positive class's curve; `average` (`"macro"`, `"micro"`, `"weighted"` or `None`) applies only when `per_class=False`, which draws a single averaged curve. A binary `pr_chart` also draws a dashed grey horizontal baseline at the positive-class prevalence, the chance-level floor. Turning `iso_lines` on overlays dashed grey (`#AAAAAA` at opacity 0.6) F-score iso-curves at F = 0.2, 0.4, 0.6 and 0.8, each labelled `F=0.2` .. `F=0.8` at its right end; the curves group on the `detail` channel, so they never take a class color.

**`confusion_matrix_chart`.** `normalize="true"` (default) normalizes rows, `"pred"` columns, `"all"` the whole matrix, and `None` shows raw counts; any other value is refused. The chart is titled `"Confusion Matrix"`, with axes `"True label"` and `"Predicted label"` and a color legend titled `"Proportion"` (normalized) or `"Count"`. `annotate=True` prints each cell's value, two decimals when normalized and an integer for counts. There is no `cmap` parameter.

**`classification_report_chart`.** A rect-plus-text heatmap titled `"Classification Report"` (title only, no `subtitle=`): rows are class labels, columns are `precision`, `recall` and `f1-score`, color encodes the value, and each cell prints it to two decimals. There is no `support` row, a deliberate difference from sklearn's `classification_report`. On the precomputed path the metrics come from confusion counts with no sklearn import. Both paths order classes the same way, the sklearn `unique_labels` order: the distinct values of `y_true` and `y_pred` in their own ascending order, raw values, not stringified (`[10, 2, 3]` orders `2, 3, 10`). `y_true` and `y_pred` must each be 1-D. Class labels are derived by partition over the values the two arrays hold, and every value family that cannot be a class label refuses with a `ValueError` stating what the arrays hold: a null, a mix of ordering families, a family that does not order (`complex`, `dict`), and one that orders but cannot be hashed (`list`, `set`, `bytearray`), for example `classification_report_chart: y_true and y_pred hold complex values that cannot be used as class labels. ...`. A homogeneous, orderable, hashable family is deduplicated by hash equality. On the model path both `X` and `y` are required; a missing `y` raises `ModelSource.classification_report() requires y to be provided.` When the model exposes `classes_`, they are coerced into `y`'s dtype and the call refuses if they share no value with `y` (`classification_report_chart: the model's classes_ ['a', 'b'], translated into y's domain, match none of y's observed values, ...`); a partial overlap, or a model that predicts none of `y`'s classes on the reported slice, still renders.

**`calibration_chart`.** The `BrierLabel` shows the per-sample Brier score of the predictions, the same number on every path and invariant to `n_bins=` and `strategy=`. The score is a broadcast `brier` column on the calibration frame (`mean_predicted`, `fraction_positive`, `count`, `brier`), computed when the frame is assembled from the per-sample predictions and the binarized target, as `auc` and `ap` are broadcast onto ROC and PR frames. The figure's label and a composed `calibration_chart(...) + fm.BrierLabel()` read that one column, and a re-applied label replaces the figure's own rather than stacking a second; `roc_chart(...) + fm.AUCLabel()` and `pr_chart(...) + fm.APLabel()` read the `auc` and `ap` columns the same way (metric labels on other frames are described in §3.15). Under `compare=` each model's label shows that model's own Brier score.

**`gain_chart`, `lift_chart`.** Titled `"Cumulative Gains Curve"` and `"Lift Curve"`, with one curve per class labelled at its end by the class's real name, and a reference curve labelled `"Baseline"`, in place of a legend. Class names resolve through `ModelSource.class_names` (§3.1): `class_names=`, then `model.classes_`, then the stringified class value, so an integer binary target labels its curves `"0"` and `"1"` and a string target `"neg"` and `"pos"`; every class of a multiclass target gets its own name. Under `compare=` the curves are labelled by model name instead.

**`residuals_chart`, `cooks_distance_chart`, `prediction_error_chart`.** `panels="auto"` (default) composes the canonical four panels, `residuals_vs_fitted`, `qq`, `scale_location` and `residuals_vs_leverage`, as a 2x2 grid; `panels="single"` or `None` returns only the residuals-vs-fitted panel; a list selects panels by name, and an unknown name is refused (`unknown residuals panel: 'bogus'`). Leverage and Cook's distance need the hat matrix of a linear estimator exposing `coef_`. For a non-linear estimator or a precomputed source the leverage panel is dropped from `panels="auto"`, which then renders the other three. When dropping it would leave no panel (`panels=["residuals_vs_leverage"]`, or `cooks_distance_chart`, which always draws that one panel), the call raises a `ValueError` naming the requirement and, when a model is present, its type: ``Cook's distance and residuals-vs-leverage are hat-matrix quantities that require a linear estimator exposing `coef_`. Got 'RandomForestRegressor'. Pass a linear estimator (e.g. LinearRegression, Ridge, Lasso), or drop residuals_vs_leverage from panels= and use the other diagnostic panels instead.`` Under `compare=` the message is prefixed with the member that failed (`compare={'base': ...}: ...`). On the precomputed path residuals are `y_true - y_pred` and studentized residuals are computed without a design matrix. `annotate_metrics=True` (default) adds a top-right corner annotation with `R²`, `RMSE` and `MAE` to the residuals-vs-fitted panel, in the single-panel and the multi-panel layouts. With `compare=`, `panels="single"` overlays one series per model in a `model` color group, and a multi-panel value returns a `ConcatChart` with one panel grid per model.

**`importance_chart`.** `show_values=True` (default) prints each bar's formatted importance at the bar's end.

**`learning_curve_chart`, `validation_curve_chart`.** The training and cross-validation curves are labelled directly at their right ends, `"Training Score"` and `"Cross-Validation Score"`, and the chart has no color legend. `validation_curve_chart` requires `values=` (`validation_curve_chart(values=...) is required — pass an explicit list of values to sweep for the given param.`) and is titled `"Validation Curve — <param>"`.

**`discrimination_threshold_chart`** (binary classifiers). The sweep's metrics are drawn as lines over the threshold, titled `"Discrimination Threshold"`, with a `"Metric"` legend of `Precision`, `Recall`, `F1` and `Queue rate`. An unknown entry in `metrics=` is refused, naming the valid set. `optimum_label` labels the F1 optimum (`max F1 = 0.936 @ t=0.63`) and is on by default; `threshold_line=True` adds a vertical rule at the same threshold. Both need `"f1"` in `metrics=`: an explicit `optimum_label=True` or `threshold_line=True` without it raises at render (`mark_discrimination_threshold(optimum_label=True) has no F1-optimum row to draw from: ...`), while the unset default omits the label silently. `compare=` returns a `ConcatChart` of small multiples, one panel per model with shared x and y scales, each marking its own model's optimum.

**SHAP.** `shap_beeswarm_chart`, `shap_bar_chart` and `shap_waterfall_chart` are the SHAP figures; `shap_waterfall_chart` requires `sample_idx`. `shap_chart` is a deprecated dispatcher: every call emits `DeprecationWarning: shap_chart(kind=...) is deprecated; use shap_beeswarm_chart / shap_bar_chart / shap_waterfall_chart instead.` and forwards to the sibling that `kind` names. `kind` must be `"beeswarm"`, `"bar"` or `"waterfall"`, and `kind="waterfall"` without `sample_idx` raises `shap_chart(kind='waterfall') requires sample_idx=<int>.` The visualizer classes split the same way (§3.15).

**Ranking.** `rank1d_chart` and `rank2d_chart` are the ranking figures. `rank_chart` is a deprecated dispatcher that validates `rank` (`"1d"` or `"2d"`), emits `DeprecationWarning: rank_chart(rank=...) is deprecated; use rank1d_chart / rank2d_chart instead.` on every call, and forwards to the matching sibling.

**First-parameter aliases.** The family's first parameter has one canonical name per kind of input (`model`, `data_or_source`). `cluster_diagnostics` also accepts the keyword `X=` for `model`, and `rank_chart`, `rank1d_chart` and `rank2d_chart` the keyword `source=` for `data_or_source`; positional callers are unaffected. Supplying both raises `TypeError`: `cluster_diagnostics() got both model= and X=; X= is a deprecated alias for model=, supply only one.` Supplying neither raises `TypeError: cluster_diagnostics() missing required argument: 'model'`.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#314-figure-level-functions).

---

### 3.15 Sklearn-Protocol Visualizers

Visualizers implement `fit` / `score` / `show` for drop-in use in sklearn workflows. Unlike Yellowbrick, `.show()` always returns a ferrum chart object, never a matplotlib figure.

**Base class:**

```
class FerrumVisualizer:
    def fit(self, X, y=None) -> self
    def score(self, X, y) -> float        # delegates to model.score; 0.0 if no model
    def show(self) -> Chart
    def __repr__(self) -> str             # summary string with key metrics
    @property
    def has_score(self) -> bool           # True iff score() returns a real metric
```

- **fit.** `fit` refuses a malformed `mark=` first, then builds the source (by default a `ModelSource` wrapping the model), computes the headline metrics into `_metrics`, and builds the chart, all in the class's voice: every warning and refusal raised during `fit` names the visualizer class (`ROCVisualizer: ...`), not the figure function it shares a builder with. It returns `self`.
- **show.** `show` returns the chart built by `fit`, and raises `RuntimeError: <Class> must be fit before .show(); call .fit(X, y) first.` before `fit`.
- **repr.** The repr is `<Class>(unfit)` before `fit` and `<Class>(metric=value, ...)` afterwards, one entry per `_metrics` key, each formatted to four decimals.
- **score.** `score` returns `float(self.model.score(X, y))` whenever the wrapped model has a callable `.score`, and `0.0` for the no-model visualizers (`ClassBalanceVisualizer`, `Rank1DVisualizer`, `Rank2DVisualizer`, `ParallelCoordinatesVisualizer`, `ElbowVisualizer`). `has_score` is a read-only property derived from the same condition, so it is `True` exactly when `score()` returns a real metric rather than the `0.0` fallback. Two subclasses override both together:
  - `ROCVisualizer.score` returns the ROC AUC (`roc_auc_score` on the positive-class column for a binary model, one-vs-rest for multiclass) when the model has `predict_proba`, falling back to `model.score`, then `0.0`. Its `has_score` is `True` when the model has `predict_proba` or a callable `.score`, so a `predict_proba`-only model reports `has_score=True` and scores its AUC.
  - `CalibrationVisualizer` given a symmetric multi-model overlay (two or more positional models, or one dict of models) has no single metric: `has_score` is `False` and `score()` returns `0.0`. A single model, or one base model with `compare=`, scores the base model's `.score`.
- A visualizer defines no `get_params` / `set_params`, so it is not clonable by `sklearn.base.clone` and cannot sit inside a `Pipeline` or `GridSearchCV`; fit the visualizer on the search's best estimator instead.

**Concrete visualizers:**

| Class | Wraps |
|---|---|
| `ROCVisualizer(model, *, micro=True, macro=True, per_class=True, annotate_auc=True, subtitle=None, compare=None, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | ROC chart. `micro` / `macro` translate to the builder's `average=`: `"macro"` when `macro=True`, else `"micro"` when `micro=True`, else `None`. As in `roc_chart`, `average` matters only with `per_class=False`, where it picks the single averaged curve (`None` keeps the per-class curves). `_metrics["auc_mean"]`. |
| `PRVisualizer(model, *, per_class=True, average="macro", annotate_ap=True, iso_lines=False, subtitle=None, compare=None, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | PR chart. As in `pr_chart`, `average` picks the single averaged curve drawn with `per_class=False`. `_metrics["ap_mean"]`. |
| `ConfusionMatrixVisualizer(model, *, normalize="true", annotate=True, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Confusion matrix. `_metrics["accuracy"]`. |
| `ClassificationReportVisualizer(model, *, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Heatmap of precision/recall/F1 per class. `_metrics["f1_macro"]`. |
| `CalibrationVisualizer(*models, n_bins=10, strategy="uniform", annotate_brier=True, subtitle=None, compare=None, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Calibration chart. At least one model is required (`TypeError: CalibrationVisualizer requires at least one model`). |
| `ResidualsVisualizer(model, *, kind="studentized", cook_threshold=None, panels=None, annotate_metrics=True, subtitle=None, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Residuals diagnostic. `panels=None` renders the single residuals-vs-fitted panel; `panels="auto"` renders the same 4-panel grid as `residuals_chart(panels="auto")`. |
| `PredictionErrorVisualizer(model, *, reference_line=True, ci=None, reference_band=False, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Prediction error |
| `FeatureImportancesVisualizer(model, *, method="builtin", top_k=20, orient="horizontal", error_bars=True, show_values=True, subtitle=None, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Importance chart |
| `LearningCurveVisualizer(model, *, cv=5, scoring=None, train_sizes=None, ci_style="band", subtitle=None, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Learning curve |
| `ValidationCurveVisualizer(model, param, values, *, cv=5, scoring=None, log_scale="auto", ci_style="band", subtitle=None, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Validation curve |
| `SilhouetteVisualizer(model, *, k=None, subtitle=None, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Silhouette chart (clusterers) |
| `ElbowVisualizer(model_class, *, ks, metric="distortion", mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Elbow chart. Takes a model **class**, not a fitted model, and fits one instance per `k` inside `fit()`; the class must accept `n_clusters`, `random_state` and `n_init`. |
| `ManifoldVisualizer(model=None, *, method="umap", mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Embedding projection |
| `ClassBalanceVisualizer(*, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Class frequency bar chart. No model required. |
| `CooksDistanceVisualizer(model, *, threshold=None, kind="studentized", panels=("residuals_vs_leverage",), annotate_metrics=True, subtitle=None, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Cook's distance. Defaults to the residuals-vs-leverage panel alone, which needs a linear estimator exposing `coef_`. |
| `SHAPVisualizer(model, *, kind="beeswarm", max_display=20, sample_idx=None, order="abs_mean", background=None, per_class=False, zero_line=True, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | SHAP chart, **deprecated**: constructing it emits `DeprecationWarning: SHAPVisualizer(kind=...) is deprecated; use SHAPBeeswarmVisualizer / SHAPBarVisualizer / SHAPWaterfallVisualizer instead.` Use the three sibling classes below. |
| `SHAPBeeswarmVisualizer(model, *, max_display=20, order="abs_mean", background=None, per_class=False, zero_line=True, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | SHAP beeswarm |
| `SHAPBarVisualizer(model, *, max_display=20, order="abs_mean", background=None, per_class=False, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | SHAP mean-absolute bar |
| `SHAPWaterfallVisualizer(model, *, sample_idx, max_display=20, order="abs_mean", background=None, per_class=False, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | SHAP single-sample waterfall |
| `DiscriminationThresholdVisualizer(model, *, n_thresholds=50, metrics=("precision", "recall", "f1", "queue_rate"), cv=None, threshold_line=False, optimum_label, subtitle=None, compare=None, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Discrimination threshold chart (binary classifiers only). `_metrics["best_threshold"]` and `["best_f1"]`. |
| `ParallelCoordinatesVisualizer(*, features=None, hue=None, rescale="minmax", alpha=0.5, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Parallel coordinates. No model required; `fit(X, y)` accepts a raw feature matrix. `hue` is applied from `y` if provided. |
| `ClassPredictionErrorVisualizer(model, *, normalize=False, show_counts=True, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Class prediction error chart. `_metrics["accuracy"]`. |
| `PCAVarianceVisualizer(model, *, n_components=None, cumulative_line=True, threshold=0.95, subtitle=None, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | PCA scree plot. The model must expose `explained_variance_ratio_` (sklearn `PCA`, `TruncatedSVD`, or compatible). |
| `Rank1DVisualizer(*, algorithm="shapiro", orient="horizontal", top_k=None, color_field=None, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Univariate feature ranking bar chart. No model required. |
| `Rank2DVisualizer(*, algorithm="pearson", annot=True, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Pairwise feature correlation heatmap. No model required. |
| `AlphaSelectionVisualizer(model, alphas, *, cv=5, scoring=None, log_scale=True, highlight_best=True, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Regularization parameter selection curve |
| `InterclusterDistanceVisualizer(model, *, method="mds", k=None, mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Intercluster distance map |
| `CVScoresVisualizer(model, *, cv=5, scoring=None, kind="box", split="both", mark=None, encode=None, properties=None, layers=None, random_state=None, theme=None)` | Cross-validation score distribution |

`DiscriminationThresholdVisualizer`'s `optimum_label` is keyword-only with an effective default of `True`, held as an internal unset sentinel and forwarded unset to `mark_discrimination_threshold`, so the mark can tell "never asked" from "asked". The F1-optimum label and the `threshold_line` rule need `"f1"` in `metrics=`: without it, an explicit `optimum_label=True` or `threshold_line=True` raises `ValueError` when the chart renders (`mark_discrimination_threshold(optimum_label=True) has no F1-optimum row to draw from: metrics=('precision',) excludes 'f1', ...`), while the unset default quietly draws neither (the same split as `discrimination_threshold_chart` and the mark itself, see §3.3 Model Diagnostic Marks). An integer `cv=` resolves to an unshuffled `StratifiedKFold`, so the sweep is deterministic; a splitter object is used as given and receives `y` in `.split(X, y)`.

**Parity with the chart builders.** Each visualizer takes, as explicit keyword-only constructor parameters, every keyword-only parameter of the chart builder its `fit` calls, including the override quartet `mark=` / `encode=` / `properties=` / `layers=` on every visualizer, and forwards each to the same builder the figure function uses. No constructor takes `**kwargs`, so an unknown keyword raises `TypeError` naming it (`ROCVisualizer.__init__() got an unexpected keyword argument 'bogus'`). Defaults match the public figure function's, with these deliberate differences:

- Spelling: `ROCVisualizer` takes `micro=` / `macro=` booleans where `roc_chart` takes `average=` (translated as in the table row), and `CooksDistanceVisualizer` takes `threshold=` where the builder takes `cook_threshold=`.
- `ResidualsVisualizer(panels=None)` renders one panel where `residuals_chart` defaults to `panels="auto"`, and `CooksDistanceVisualizer` defaults to the leverage panel alone; both reach every layout through `panels=`.
- `Rank1DVisualizer` / `Rank2DVisualizer` name `"shapiro"` / `"pearson"` outright where `rank1d_chart` / `rank2d_chart` spell the same default `algorithm=None`.

`tests/diagnostics/test_visualizer_chart_parity.py` enforces this by diffing each visualizer's keyword-only signature and defaults against its builder and figure function, with a rationale-carrying skip-map for exactly the differences above, and fails on any `FerrumVisualizer` subclass missing from its registry.

**`compare=`.** `ROCVisualizer`, `PRVisualizer`, `DiscriminationThresholdVisualizer` and `CalibrationVisualizer` accept `compare={"name": model, ...}`. The positional `model` is the base, labelled `"base"`; the merge goes through `ModelSource._from_base_and_compare`, the same path the figure functions use, so the refusals are shared: a non-dict raises `TypeError: compare= must be dict[str, model] or None; got list.` and a `"base"` key raises `ValueError: compare= must not use the key 'base' -- it is reserved for the primary model= argument. ...`. Every derived metric (`auc_mean`, `ap_mean`, `best_threshold` / `best_f1`, `calibration_error`) and therefore `__repr__` describes the base model only, computed from its rows and equal to what the visualizer reports without `compare=`, never a value pooled across models or a rival's; `.score()` and `has_score` likewise describe the base model. `ROCVisualizer`, `PRVisualizer` and `CalibrationVisualizer` overlay the models in one panel, each curve labelled with its model name; `DiscriminationThresholdVisualizer` renders small multiples, one `ConcatChart` panel per model built from that model's own sweep (its color channel already carries the four metrics), so the `"base"` panel's F1-optimum annotation names the same threshold and F1 that `_metrics` and `__repr__` report. `CalibrationVisualizer` with `compare=` renders the same chart as `calibration_chart` called with the same model, data and `compare=`.

**`CalibrationVisualizer`'s two multi-model shapes** answer different questions:

- **Base plus rivals.** A positional model with `compare={"rival": other}` has one distinguished base, with the `compare=` semantics above: `has_score=True`, `.score()` scores the base, and `_metrics` holds one `calibration_error`, the base model's own.
- **Symmetric overlay.** Two or more positional models (named `model_0`, `model_1`, ...) or one dict of models (the caller's names) is symmetric with no base: `has_score=False`, `score()` returns `0.0`, and `_metrics` holds one `calibration_error[<model name>]` key per model.
- **Not both.** The two are mutually exclusive. Several positional models or a dict of models together with `compare=` raises `ValueError` at construction, naming both idioms: `CalibrationVisualizer takes either a symmetric multi-model overlay (*models) or one base model plus compare= rivals, not both; got 2 positional models together with compare=. ...`.

**`random_state`.** Every visualizer accepts `random_state: int | None = None` and forwards it to its `ModelSource` (or, for `ElbowVisualizer`, to every per-`k` model). It is consumed only where the computation is randomized: `LearningCurveVisualizer` (seeds sklearn's training-data shuffle), `FeatureImportancesVisualizer` with `method="permutation"` (seeds the permutation shuffles), `ManifoldVisualizer` and `InterclusterDistanceVisualizer` (seed the embedding), and `ElbowVisualizer` (the per-`k` models' `random_state`). `None` means seed `0` in each of these, so default output is reproducible. Every other visualizer's computation is deterministic and ignores the value; this includes the integer-`cv=` paths of `ValidationCurveVisualizer`, `AlphaSelectionVisualizer`, `CVScoresVisualizer` and `DiscriminationThresholdVisualizer`, which use sklearn's unshuffled splitters. `random_state` never seeds the wrapped estimator itself.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#315-sklearn-protocol-visualizers).

---

### 3.16 Output and Rendering

#### `RenderConfig`

```
RenderConfig(raster_threshold=500_000, raster_behavior="warn", raster_aggregate="count", raster_scheme=None, raster_cmap=None, raster_field=None)
```

`RenderConfig` is a frozen dataclass that holds one chart's auto-raster policy and nothing else. Attach it with `.properties(render_config=RenderConfig(...))`; on a composition view, `.properties(render_config=...)` forwards to every child chart, and each leaf applies its own policy. For a one-off override, the `Chart` output methods take `raster=`: `False` disables auto-raster for that call (as `raster_threshold=None` would), `True` forces it (as `raster_threshold=0` would, still subject to the eligibility rules below), and `None` (the default) uses the chart's policy. Composition output methods take no `raster=`.

| Field | Default | Meaning |
|---|---|---|
| `raster_threshold` | `500_000` | Row count at or above which auto-raster is considered. `None` disables auto-raster. |
| `raster_behavior` | `"warn"` | What happens when an eligible chart reaches the threshold: `"warn"` substitutes and emits a `UserWarning`, `"silent"` substitutes without a warning, `"error"` raises `ValueError` (`Auto-raster: <n> marks exceed threshold <t>. Pass raster=False to .show()/.save() to disable.`) instead of substituting. |
| `raster_aggregate` | `"count"` | Aggregate of the substituted `mark_raster`: one of `"count"`, `"density"`, `"mean"`, `"sum"`, `"any"`. |
| `raster_field` | `None` | Column to aggregate. Required by `"mean"` and `"sum"`; ignored by `"count"`, `"density"` and `"any"`. |
| `raster_scheme` | `None`, read back as `"viridis"` | Color scheme of the substituted `mark_raster`. |
| `raster_cmap` | `None` | Alias of `raster_scheme`. Pass at most one of the two (or both with the same value); after construction both read back as the resolved scheme. |

**Validation at construction.** `raster_aggregate` is checked against the `Raster` transform's own vocabulary, so `"max"`, `"min"` or `"median"` raise `ValueError: RenderConfig: raster_aggregate must be one of ['any', 'count', 'density', 'mean', 'sum']; got 'max'`. `"mean"` or `"sum"` without `raster_field` raises `RenderConfig: raster_aggregate='mean' requires raster_field=... (the column to aggregate); pass raster_field=... or use raster_aggregate='count'|'density'|'any', which need no field.` A `raster_field` given with `"count"`, `"density"` or `"any"` is dropped before the substitution, so it is never resolved as a column and never raises. Different `raster_scheme` and `raster_cmap` values raise `RenderConfig: pass either scheme= or cmap= (its alias), not both with different values; ...`. The aggregated column may be any integer, unsigned or float dtype (coerced to `Float64`); a non-numeric column raises at render time with `stat_raster: column '<name>' must be numeric, got <Dtype>`.

**When auto-raster substitutes.** At render time a chart is replaced by a `mark_raster` equivalent when all of these hold: the threshold is not `None`; the chart's data has at least `raster_threshold` rows; its mark is `point`, `bar`, `rect`, `tick`, `rule` or `segment`; it is not a composite or statistical mark (histogram, density, hex, boxplot and the like, whose row count is not their element count); both `x` and `y` are explicitly quantitative (`"x:Q"` or `type="quantitative"`); and it has no non-quantitative `color` encoding. A `color` channel's type comes from its annotation, or from the column dtype when it has none, so a bare `color="z"` on a numeric column counts as quantitative.

- A quantitative `color` channel drives the substitution: it uses `aggregate="mean"` over the color field and the channel's own `scheme=` (else `raster_scheme`), so the color information survives. A `RenderConfig` whose `raster_aggregate` or `raster_field` differs from the defaults wins wholesale over this; because the dataclass cannot tell an explicit `raster_aggregate="count"` from the default, that spelling also takes the color-driven path.
- A categorical, ordinal, temporal or untyped `color` channel is excluded from auto-raster permanently, since rasterizing would collapse the categories into one pixel color per bin. An over-threshold chart in this case renders per element and warns, naming the field and its kind and pointing to `.mark_raster(aggregate=..., field=...)` for an explicit raster.
- An over-threshold chart whose `x` or `y` is missing or not explicitly quantitative renders per element and warns `Chart has <n> marks which may produce large output. ...`. Other ineligible charts (another mark type, a statistical mark) render per element without a warning.

`raster_threshold=None` and `raster_behavior="silent"` are not equivalent: `None` disables auto-raster entirely (every mark renders per element, however many there are); `"silent"` keeps auto-raster active and only suppresses the substitution warning.

The policy is read only from the chart's `RenderConfig` and the `raster=` argument. `ferrum.config` accepts a `raster_threshold` key, but rendering does not read it.

**Concerns that are not `RenderConfig` fields.** Pixel density is the `scale=` argument of `to_png()` and `save()` (PNG and PDF only). Chart size is `.properties(width=..., height=...)`, defaulting to `ferrum.config`'s `width` and `height` (640 x 480). The background is a theme key (`Theme(background=...)`, §3.13). The output format is the `save()` path extension or its `format=` argument. Fonts are not configurable: SVG output embeds the bundled Inter Regular font as an `@font-face` data URI and sets `font-family="Inter"`, and PNG rasterization loads that same bundled font only. The renderer is chosen by the output method (§3.17); there is no backend or engine field, and no Vega-Lite output.

#### Chart output methods

The output surface follows one convention: `to_*` returns an in-memory value, `show()` displays, `save(path)` writes to disk.

```
chart.show(*, raster=None)
chart.to_svg(*, raster=None) -> str
chart.to_png(*, raster=None, scale=2.0) -> bytes
chart.to_html(*, embed_wasm=True, toolbar=True, raster=None, csp_nonce=None) -> str
chart.save(path, *, format=None, embed_wasm=True, raster=None, scale=2.0, toolbar=True, csp_nonce=None)
chart.interactive(*, toolbar=True) -> InteractiveChart
chart.to_spec() -> ChartSpec
chart.to_dict() -> dict
chart.to_json(*, indent=None) -> str
chart.pipe(fn, *args, **kwargs) -> Any
```

| Method | Returns or does |
|---|---|
| `show` | Displays the static SVG (see **Environment detection order for `.show()`**). |
| `to_svg` | SVG markup. |
| `to_png` | PNG bytes: the SVG rasterized at `scale` times its intrinsic size (`scale=2.0` turns a 640 x 480 chart into 1280 x 960 pixels). |
| `to_html` | A self-contained interactive HTML document backed by the WASM renderer, byte-identical to what `save()` writes for `.html`. `embed_wasm=True` base64-inlines the WASM binary; `False` references a `ferrum_wasm_bg.wasm` sidecar served beside the page. `toolbar=False` hides the toolbar. `csp_nonce=` puts a `nonce="..."` on the inline `<style>` and `<script type="module">` tags. |
| `save` | Writes one file; see the format table below. `scale` affects PNG and PDF only; `embed_wasm`, `toolbar` and `csp_nonce` affect HTML only. |
| `interactive` | An `InteractiveChart` for the WASM renderer (§3.17). |
| `to_spec` | The Rust `ChartSpec` the chart lowers to. |
| `to_dict`, `to_json` | The declaration dict, and its JSON (compact by default, pretty-printed with `indent=`). For a `Chart`, `to_json` serializes the `ChartSpec` that `to_spec` returns, and `to_dict()` is `json.loads(chart.to_json())`. |
| `pipe` | Calls `fn` with the chart as its first argument and returns the result, so a chart can pass through a fluent pipeline. |

| `save()` format | Written |
|---|---|
| `svg` | `to_svg()` |
| `png` | `to_png()` at the given `scale` |
| `html` | `to_html(...)`; with `embed_wasm=False` the `ferrum_wasm_bg.wasm` sidecar is copied next to the file |
| `json` | The rendered scene graph as JSON (the payload the interactive renderer draws), not the declaration from `to_json()` |
| `pdf` | The PNG render at `scale`, wrapped in a single-page PDF (raster, not vector) |

The format is `format=` when given, else the path extension, lower-cased. A path with no extension and no `format=` raises `ValueError: save('<path>') requires a format= or a path with extension.`; any other format raises `unknown extension '<ext>'; supported: svg, png, html, json, pdf.` `save()` takes no keyword arguments beyond those in its signature, and none selects a renderer.

**Composition views.** `HConcatChart`, `VConcatChart`, `LayerChart`, `ConcatChart`, `JointChart`, `RepeatChart` and `ClusterMapChart` share one output surface, routed through the same save, show and HTML paths as `Chart`:

```
HConcatChart.show()
HConcatChart.to_svg() -> str
HConcatChart.to_png(*, scale=2.0) -> bytes
HConcatChart.to_html(*, embed_wasm=True, toolbar=True, csp_nonce=None) -> str
HConcatChart.save(path, *, format=None, scale=2.0, toolbar=True, embed_wasm=True, csp_nonce=None)
HConcatChart.interactive(*, toolbar=True)
HConcatChart.to_dict() -> dict
HConcatChart.to_json(*, indent=None) -> str
HConcatChart.pipe(fn, *args, **kwargs) -> Any
```

These differ from `Chart` in two ways: they take no `raster=` argument, and they have no `to_spec()`, because a `ChartSpec` describes one leaf chart and a composition has none to return. A composition's `to_json()` is `json.dumps(to_dict(), indent=indent)`.

A composition's `to_dict()` is recursive: `{"kind": <layout>, <layout fields>, <child slots>}`, where `kind` is `"hconcat"`, `"vconcat"`, `"overlay"` (`LayerChart`), `"wrap"` (`ConcatChart`), `"joint"`, `"repeat"` or `"cluster_map"`. The concat views and `LayerChart` list their children under `children`. A leaf child embeds as `json.loads(child.to_spec().to_json())`, and a nested composition embeds as its own `to_dict()`. For `JointChart`, `RepeatChart` and `ClusterMapChart`, `to_dict()` equals the `.spec` layout property, plus figure chrome when set; `.spec` itself never carries chrome.

**Figure chrome.** A composition's figure `title`, `subtitle` and `caption` are part of its declaration and appear as top-level `to_dict()` keys of those names, each omitted when unset. `title` serializes as `{"text": ...}`, the same shape as an embedded child's title; `subtitle` and `caption` are plain strings. The six non-layer views carry all three. `LayerChart` carries `title` only: it has one title slot, and `.properties(subtitle=..., caption=...)` on it forwards to its member charts, where a subtitle lands in each member's title and a caption is not kept anywhere, so it neither renders nor appears in the dict.

`.properties(title=, subtitle=, caption=)` on a composition accepts a `str` or a `ferrum.Title`, normalized when `.properties()` is called. The composite renderer draws plain text in each slot, so:

- `title=fm.Title(text, subtitle=...)` on one of the six non-layer views routes the `Title`'s subtitle into the composition's `subtitle` slot, where it behaves like an explicit `subtitle=`. Giving both in the same `.properties()` call raises `ValueError` (`properties(title=fm.Title(..., subtitle=...)) and properties(subtitle=...) both declare the figure subtitle; a composite has one subtitle slot. ...`). Across separate calls the later value replaces the earlier one without a warning, as with any repeated `subtitle=`.
- Any `Title` field set away from its default that has no slot is refused with a `ValueError` naming the field and the fix, e.g. `properties(title=fm.Title(...)) with anchor set is refused: a composite figure title carries text plus one routed subtitle line only, with no slot for anchor. ...`. That covers `anchor`, `offset`, `font_size`, `font_weight`, `color`, `subtitle_font_size` and `subtitle_color` on all three keys, `subtitle` on `subtitle=` and `caption=`, and `subtitle` on a `LayerChart` title. A declaration that `.properties()` accepts always renders.

**Construction checks.** Each composition view raises `TypeError: <Class> children must be Chart or composition objects, got <TypeName>` at construction when a child is neither a `Chart` nor a composition (e.g. `fm.HConcatChart([a_dataframe])`). `|` and `&` on a `Chart` or a composition return `NotImplemented` for any other operand, so `chart | 3` raises Python's own `TypeError: unsupported operand type(s) for |: 'Chart' and 'int'`.

**The declaration dict is write-only.** The dict from `to_dict()` (a plain `dict`), and the JSON from `to_json()`, are an export for inspection, logging and diffing. Ferrum does not read them back: no public API builds a `Chart` or a composition from either, and `ferrum.__all__` has no `from_dict`. (`ChartSpec.from_json` parses a leaf's JSON into a `ChartSpec` value, but that value does not rebuild the chart: the dict carries no data rows, size, caption, theme, `configure_*` settings or annotations.) The key layout is not a versioned schema. It may change in any release, without a deprecation period. To rebuild a chart, keep the Python code that built it.

**Deprecated aliases.** `show_svg()` and `show_png()` forward to `to_svg()` and `to_png()` with the same arguments and emit a `DeprecationWarning` (`Chart.show_svg() is deprecated; use Chart.to_svg() instead.`). They have been deprecated since `0.16.0` and will be removed in a future release. New code calls `to_svg()`, `to_png()` or `to_html()`.

#### Environment detection order for `.show()`

`.show()` always displays static SVG, on `Chart` and on every composition view:

1. Inside an IPython shell whose class is `ZMQInteractiveShell` (Jupyter Notebook and Lab, VS Code notebooks, any ipykernel front end) or `TerminalInteractiveShell` (terminal IPython), it calls `IPython.display.display(IPython.display.SVG(chart.to_svg()))`. Terminal IPython takes this branch too, so it shows no image there. If that display call raises, `.show()` falls through to step 2.
2. Otherwise it writes the SVG, wrapped in a minimal HTML page whose `<title>` and heading are the figure title (`"Ferrum chart"` when untitled), to a temporary `.html` file that is not deleted, and opens it with `webbrowser.open`.

Sixel or other terminal-graphics output is not supported. `.show()` never displays the interactive renderer: to see it in a notebook, display the `InteractiveChart` that `.interactive()` returns (for example as a cell's last expression), and outside a notebook use `.to_html()` or `.save("chart.html")`. A chart that is a notebook cell's value displays through its rich-display hooks instead of `.show()`: a `Chart` provides `_repr_svg_` and `_repr_html_` (the SVG in a `<div>`), and falls back to its text repr if rendering fails; a composition provides `_repr_svg_` and an `image/svg+xml` `_repr_mimebundle_`.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#316-output-and-rendering).

---

### 3.17 Rendering Backends

Ferrum has two renderers, chosen by the output method alone: the static renderer for SVG, PNG and PDF, and the wgpu/WASM renderer for interactive output. Mark count never switches renderers; a chart with many marks stays on the same renderer, and auto-raster (§3.16) may change the chart itself by substituting `mark_raster`.

#### SVG Backend

- Rust (`ferrum-core`) renders the chart's scene graph to SVG. It backs `to_svg()`, `save(".svg")`, `.show()` and notebook rich display, and every PNG and PDF starts from its output.
- Resolution-independent and text-searchable. Text uses `font-family="Inter"`, with the bundled Inter Regular embedded as an `@font-face` data URI, so the SVG renders the same without the font installed.
- There is no mark-count cap or warning on SVG output itself; size control is the auto-raster policy (§3.16).

#### tiny-skia Backend (CPU Rasterizer)

- `to_png()` and `save(".png")` rasterize the SVG from the SVG backend with `resvg` (`usvg` parsing, `tiny-skia` drawing): pure Rust, no system dependencies (no Cairo, libpng or X11), bundled in the wheel.
- Text is drawn from the bundled Inter Regular only; system fonts are not loaded.
- Output pixel size is the SVG's intrinsic size times `scale` (default `2.0`, retina).
- `save(".pdf")` wraps this PNG in a single-page PDF, so PDF output is raster, not vector.
- Not used for interactive output. Limitations: no vector output; zooming a PNG does not re-render it.

#### wgpu / WASM Backend (GPU Interactive)

- Used for `.interactive()` and for every HTML export (`to_html()`, `save(".html")`), which is always interactive. Rust renders the scene graph plus packed mark data, and the browser draws it with the `ferrum-wasm` module; `save(".json")` writes that scene graph.
- `wgpu` requests WebGPU where the browser provides it and falls back to WebGL2, with device limits held to the WebGL2 downlevel defaults.
- No compute shaders, so the same pipeline runs on WebGPU and WebGL2: simple marks (points, bars) are instanced draws and curves and areas are tessellated with `lyon`. Vello is not used, since its compute-shader approach cannot run on the WebGL2 fallback.
- Text and axis labels are real DOM text in an SVG overlay above the canvas.
- Pan and zoom are view transforms applied in the browser, with the overlay text re-placed; redrawing needs no Python round-trip.
- Jupyter integration is through `anywidget` (the `jupyter` extra, `ferrum-viz[jupyter]`): selection state syncs to Python as `InteractiveChart.selection_state`, and `on_selection_change(callback)` registers a Python callback. Without `anywidget`, an `InteractiveChart` displays as its text repr, and `to_html()` and `save()` still work.
- An `InteractiveChart` has `to_html` (keyword-only `embed_wasm` and `csp_nonce`) and `save` (`path`, then keyword-only `format`, `embed_wasm`, `csp_nonce` and `scale`), with the same defaults as on `Chart`. Both route through the same HTML and save paths as `Chart` and carry the `toolbar` flag given to `.interactive()`; it has no `show`, `to_svg` or `to_png`.
- The WASM binary is several megabytes. HTML export base64-inlines it by default; `embed_wasm=False` references a `ferrum_wasm_bg.wasm` sidecar instead.

#### Backend selection

| Call | Backend |
|---|---|
| `.to_svg()`, `.save("chart.svg")` | SVG |
| `.to_png()`, `.save("chart.png")` | SVG, rasterized by tiny-skia |
| `.save("chart.pdf")` | SVG, rasterized by tiny-skia, wrapped in PDF |
| `.show()`, notebook rich display of a chart | SVG (inline in a notebook, else in a browser page) |
| `.interactive()`, `.to_html()`, `.save("chart.html")` | wgpu / WASM |
| `.save("chart.json")` | Scene graph JSON (the WASM renderer's input) |

The auto-raster threshold (`RenderConfig.raster_threshold`, default `500_000`) never changes this table; it decides only whether eligible marks are substituted by `mark_raster` before rendering.

---

### 3.18 Data Source Compatibility

`Chart(data=...)` and the figure-level functions accept the inputs below. Every input is converted to one Arrow table by `ferrum._coerce.to_arrow_table` at render time, normalized once by `normalize_for_rust`, and crosses to Rust once over the Arrow C Data Interface.

| Type | Behavior |
|---|---|
| `polars.DataFrame` | Exported to Arrow by polars' own `to_arrow`, zero-copy except for the columns cast first: `Date` and non-millisecond `Datetime` become `Datetime("ms")`, and `Categorical` and `Enum` become strings. |
| `polars.LazyFrame` | Collected once, then handled as a `polars.DataFrame`. |
| `polars.Series` | One column, named by the series. |
| `pandas.DataFrame` | Converted through narwhals (`narwhals>=1.0,<2` is a dependency; pandas is not). |
| `pandas.Series` | One column, named by the series, or `"value"` when it is unnamed. |
| Other eager frames narwhals accepts (modin, cuDF) | Converted through narwhals' `from_native` with `eager_only=True`. Lazy frames other than `polars.LazyFrame` (dask, ibis) are not accepted. |
| `pyarrow.Table` | Used as is, after `date32`/`date64` and non-millisecond `timestamp` columns are cast to `timestamp[ms]`. |
| `pyarrow.RecordBatch` | Wrapped as a one-batch table. |
| `dict[str, list]` | `pyarrow.Table.from_pydict`. A GeoJSON `FeatureCollection` dict, or a bare GeoJSON geometry dict, is read as features instead: property columns plus a geometry column. |
| `list[dict]` | One dict per row, `pyarrow.Table.from_pylist`. An empty list raises `ValueError: Cannot construct Chart from empty list`; a list of non-dicts raises `TypeError: Chart(list) expects a list of dicts (one per row), got list of int`. |
| `numpy.ndarray` (2D) | Columns named `col_0`, `col_1`, ... A 1D array raises `TypeError` asking for column names (`Chart({'value': arr})`); an array with `ndim > 2` raises `TypeError: numpy arrays with ndim=3 not supported (use 2D)`. |
| `None` | A chart with no data of its own, for layers that bring theirs: compose independent-data charts with `+` or `ferrum.layer(...)` (§3.12). Rendering a chart whose data is `None` raises `ValueError: Chart(data=None) requires per-layer data. ...`. |

Any other type goes to narwhals, and one narwhals cannot read raises `TypeError: Unsupported data type: <type>. Supported: ...`. File paths are not data: a `str` or `pathlib.Path` is refused like any other unsupported type, so load a file with polars or pyarrow first. `ModelSource` and `ComparedModelSource` (§3.1) are not `Chart` data either; their derivation methods (`predictions()`, `roc_curve()`, ...) return `polars.DataFrame`s that are, a `ComparedModelSource`'s with an added `model` column, and the diagnostic figure functions take a source directly (§3.14).

**Column normalization.** Before the table reaches Rust, `normalize_for_rust` applies one dtype contract on every path (static and interactive): `date32`/`date64` and `timestamp` of any unit become `timestamp[ms]`; `duration` of any unit becomes `timestamp[ms]` with the same temporal scaling; a dictionary-encoded column is decoded to its value type; an all-null `Null` column becomes `float64`. On the Rust side, a string column carried as `Utf8View` or `LargeUtf8` (as polars and newer pyarrow export strings) is cast to `Utf8` with `arrow::compute::cast` and renders identically to the same column carried as `Utf8`. A cast that cannot complete is a typed render error, `could not normalize column '<name>' to Utf8: <reason>` (`RenderError::StringNormalizationFailed`).

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#318-data-source-compatibility).

---

### 3.19 Utilities

**Color helpers (`ferrum.color`).** The functions read the one palette registry (`crates/ferrum-core/src/render/color/palette.rs` and `.../color/continuous.rs`) through the `ferrum._core` accessors `list_palettes`, `palette_kind`, `palette_colors` and `palette_sample`, so they return the colors that render.

```
ferrum.color.palette(name, n=None)
ferrum.color.sequential(name, n=256)
ferrum.color.diverging(name, n=11)
ferrum.color.to_hex(color, *, scale=None)
```

- `palette` returns hex strings from any registered palette (10 categorical, 16 sequential, 4 diverging). A categorical palette returns all its colors when `n` is `None` and cycles when `n` exceeds its length. A continuous palette is sampled `n` times evenly at `t = i / (n - 1)`, endpoints included, with the renderer's own interpolator (`n` defaults to 7; `n=1` returns the midpoint; `n=0` returns `[]`). An unknown name raises `ValueError: Unknown palette: 'nope'. Available: <every name, sorted>`.
- `sequential` and `diverging` sample a continuous palette the same way, and accept only their own kind: `ValueError: Unknown sequential palette: 'rdbu'. Available: ...` (and `Unknown diverging palette: ...`).
- `to_hex` normalizes a color string (the one vocabulary of §3.2, parsed by the Rust color parser) or an RGB(A) tuple to `#rrggbb`, or `#rrggbbaa` when the alpha is not fully opaque; a fully opaque alpha is dropped on both input forms, so `to_hex` is idempotent on its own output. `scale="unit"` reads tuple components in `[0, 1]`, `scale="byte"` in `[0, 255]`; with `scale=None` a tuple is byte-scaled when any of r, g, b exceeds 1 and unit-scaled otherwise, so `(1, 0, 0)` and `(1.0, 0.0, 0.0)` are both `#ff0000`. A byte reading forced by a non-integer component above 1, such as `(0.9, 0.9, 1.1)`, warns (`to_hex: ambiguous color tuple ...`). A paint-clearing sentinel (`"none"`, `"transparent"`) raises `ValueError` saying it has no hex form, and so do a non-numeric component, a tuple shorter than 3, and a `scale` other than `"unit"`, `"byte"` or `None`.

`scheme=` on `Color`, `Fill` and `Stroke` is checked against the registry at construction (`ValueError: Color(scheme=...): Unknown palette: 'bogus'. Available: ...`). `scheme=` is the colormap keyword of `mark_raster`, `mark_contour`, `mark_hex`, `heatmap` and `clustermap` (and `raster_scheme=` of `RenderConfig`), with `cmap=` (`raster_cmap=`) as an alias: passing both with different values raises `ValueError: mark_raster: pass either scheme= or cmap= (its alias), not both with different values; ...`, and a name that is not a registered continuous colormap raises `ValueError: mark_raster: color scheme 'bogus' is not recognized; available: ...`.

**Process defaults (`ferrum.config`).** A per-context store (`contextvars.ContextVar`), so a value set in one thread or task does not leak into another.

```
ferrum.config.set(**kwargs)
ferrum.config.get(key)
ferrum.config.defaults(**kwargs)
ferrum.config.reset()
```

`set` merges keyword values into the store; `get` returns a stored value, else the built-in default, and raises `KeyError: "Unknown config key: 'nope'"` for a key with neither; `defaults` is a context manager that applies its values inside the `with` block and restores the prior store on exit; `reset` clears every stored value. `set` and `defaults` do not check key names, so a misspelled key is stored and has no effect.

| Key | Built-in default | Read by |
|---|---|---|
| `width` | `640` | The render viewport of a chart with no width of its own (a `width=` given to `Chart` or `.properties` wins). |
| `height` | `480` | The same, for height. |
| `renderer` | `"svg"` | Stored only; no render path reads it. |
| `max_rows` | `5000` | Stored only; no row limit is enforced. |
| `raster_threshold` | `500_000` | Stored only; the auto-raster threshold is `RenderConfig`'s `raster_threshold` (§3.16). |

**Titles.** A leaf chart's `title=` (`Chart(title=...)`, `.properties(title=...)`) takes a string or a `Title`; a string is `Title(text=string)`. On a leaf every field renders. Composite figure titles accept only what their chrome can draw (§3.16).

```
ferrum.Title(text, subtitle=None, anchor="start", offset=None, font_size=None, font_weight=None, color=None, subtitle_font_size=None, subtitle_color=None)
```

`anchor` is `"start"`, `"middle"` or `"end"`; `subtitle=None` draws no second line.

**Gridlines.** `Grid` is the value of the theme's `grid` key (`Theme(grid=...)`, `theme.update(grid=ferrum.Grid(...))`); a bare `bool` there sets `major` alone.

```
ferrum.Grid(major=True, minor=False, color=None, width=None, dash=None, opacity=None, major_color=None, minor_color=None, major_width=None, minor_width=None, major_dash=None, minor_dash=None, major_opacity=None, minor_opacity=None)
```

The bare `color`, `width`, `dash` and `opacity` set both levels; a `major_*` or `minor_*` value overrides its own level. Only the resolved per-level keys reach the renderer. Major gridlines sit on the major ticks, which are scale-projected, so a gridline at value v is on the pixel of a mark at v. `minor=True` draws minor gridlines strictly between the majors on continuous axes (linear, log, time, pow, sqrt, symlog); a categorical or discretizing axis (band, ordinal, quantile, threshold, bin-ordinal) has no continuum to subdivide, so it draws none and does not raise.

**Aggregation values.** `Aggregate` is the group-by stat transform passed to `.transform(...)`; each `AggregateOp` names one input column, one function from the aggregation vocabulary of §3.5 and its output column (`AggregateOp: unknown fn 'bogus'; expected mean|sum|count|...` otherwise). The dict-spec builders `transform_aggregate`, `transform_join_aggregate` and `transform_window` (whose ops are `{"op": ..., "as": ...}` dicts; there is no window value class) are in §3.5.

```
ferrum.AggregateOp(field, fn_, as_)
ferrum.Aggregate(ops, *, groupby=None, name=None)
```

ferrum has no `ferrum.data` namespace and ships no sample datasets.

> **Consolidated 2026-10-12.** Earlier dated notes for this section are kept verbatim in [spec history](design-docs/spec-history.md#319-utilities).

---

---

## Part IV: Extension Points

ferrum has no plugin registry. There is no `register_mark`, `register_stat` or `register_renderer`, and no `MarkProtocol`, `StatProtocol` or `RendererProtocol`: a new mark, stat transform or output format is added inside the library, not by user registration. What a user extends without the library's help is below.

### Custom Marks

Not provided as a protocol. A custom visual form is built by composing shipped marks: layering charts with `+` or `ferrum.layer(...)` (§3.12), or giving a mark precomputed columns.

### Custom Stat Transforms

Not provided as a protocol. A custom statistic is computed on the data before it is passed (for example, with polars), and the result is charted with the shipped marks.

### Custom Themes

Themes are `Theme` values, not registered: build one with `Theme(**keys)` or derive one from a built-in with `.update(...)` (which returns a new `Theme`), and pass it to `.theme()` or `ferrum.set_default_theme()` (§3.13). An unknown key raises `ValueError: Unknown Theme key(s): [...]`.

### Renderer Plugins

Not provided. The output formats are the built-in ones of §3.16 (`to_svg`, `to_png`, `to_html`, `to_json`, `save`, `show`, `interactive`).

---

---

## Appendix: Version Target

This document describes the shipped ferrum API as of 0.24.0.

The intent is that 1.0 be complete enough that a practitioner moving from Altair, Seaborn, or Yellowbrick finds every chart type they reach for without falling back to matplotlib.

Not in the shipped API, and candidates for after 1.0:

- `mark_network` / graph layout
- `mark_gantt` / timeline
- Geographic tile layers (Mapbox, OpenStreetMap)
- 3D coordinate system
- Animation / `frame` encoding
- Real-time streaming data sources
- Julia and R bindings
