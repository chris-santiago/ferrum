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
- `.mark_point()`, `.mark_line()`, ... (§3.3): Set the chart's mark; a later primitive mark call replaces the earlier one. A desugaring mark that adds a fitted layer (`.mark_smooth()`, `.mark_function()`) keeps every existing layer, each with its own style.
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

**Bars and dodge.** A bar row whose computed width or height is exactly `0.0` emits no rect (`render/marks/bar.rs::is_zero_extent`). When that guard empties a batch that had rows to draw, the render warns once, e.g. `bar layer 0 drew nothing: mark_bar() computed a zero width or height for all 4 of its rects (...)`; a batch empty for another reason (no rows, null categories, non-finite values) stays quiet. Under `Dodge` on a band axis that carries a paired span channel (`x`/`x2`, `bin_start`/`bin_end`), both ends of each row's span move into the group's own sub-slot of that row's span (`render/position.rs::apply_dodge`), so a dodged histogram, including one with a single bin, draws side-by-side sub-bars rather than offset full-width bars, and a null endpoint stays null.

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
| `mark_boxen(...)` | nested letter-value rectangles + outlier points (`LetterValue` transform) | `k_depth` (`"tukey"`\|`"proportion"`\|`"trustworthy"`\|`"full"`, default `"tukey"`, capped at depth 12), `k_proportion` (default `0.007`), `outlier_threshold` (default `1.5`, a multiple of the outermost band's span), `palette` (`str`\|`Sequence[str]`\|`None`) |
| `mark_errorbar(...)` | rule + tick caps | `method` (`"ci"`, `"stderr"`, `"stdev"`, `"iqr"`; default `"ci"`), `ticks` |
| `mark_errorband(...)` | ribbon (+ optional border lines) | `method` (as `mark_errorbar`), `borders` |
| `mark_ribbon(...)` | area between Y and Y2 | `opacity` (default `0.3`), `interpolate` (default `"linear"`) |

**Deprecated `extent=`.** `mark_boxplot(extent=)` is accepted for one release as an alias of `whisker_mult=`, and `mark_errorbar(extent=)` / `mark_errorband(extent=)` as an alias of `method=`, each with a deprecation warning: `extent` is reserved for data-domain bounds.

**`mark_boxen` depth bands and `palette=`.** `LetterValue` names each depth by the Tukey letter-value sequence, `M` (depth 1, the median), `F`, `E`, `D`, `C`, `B`, `A`, `Z`, then `Y`, `X`, `W`, `V` for depths 9 to 12. Depth is capped at 12 for every `k_depth` strategy (Rust `letter_value::K_MAX_NAMED`), and `desugar_boxen` builds one layer per depth from the transform's declared secondary outputs. The median is a band-confined `tick`, not a full-panel rule. Each layer's positional encoding carries the user's own field names as axis titles. The outlier layer draws only rows the transform flags as outliers, against a fence of `outlier_threshold` times the span beyond the deepest materialized band.

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
| `LetterValue(value, *, group=None, k_depth="proportion", k_proportion=0.007, outlier_threshold=1.5, sort=None)` | `group`, `depth`, `lower`, `upper`, `level`; secondary `outliers` and one `depth_N` table per depth | Letter-value (boxen) statistics. `k_depth` is `"tukey"`, `"proportion"`, `"trustworthy"` or `"full"`. |
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

**Positional axes.** On a layered chart (`+`, `Chart.layer(...)` or `fm.layer(...)`), the first caller-written continuous scale in layer order establishes the channel's axis, whichever layer carries it. A layer that declares no scale on the channel (or carries only a bar's injected zero anchor), or that carries an equal copy of the establishing scale, resolves on that axis; its data, together with any paired `y2`/`x2`, widens the extent only when the establishing scale states no `domain` (a stated `domain` is used as written). A layer with a different scale, or with its own `domain`, stays out of the axis and warns once per layer and channel (`RenderWarning::LayerScaleExcluded`: `layer <i>: its <channel> scale differs from the one layer <j> establishes the <channel> axis with, so it stays out of that axis; ...`). On `fm.layer` such a layer draws on its own unlabelled scale; on `+` there is one scale per axis, so its marks fall outside the shared axis and are clipped. Both routes resolve the same axis for the same layers in any order (`tests/test_layer_union_provenance.py`). A categorical first scale keeps the ordinal band union and never refuses. When layer 0 (on y, the first layer that is not a secondary-y layer) binds a categorical column, the panel axis is layer 0's categorical axis, and a later layer's continuous caller scale, including one a per-layer `CoordFlip` moves onto that channel, stays out with the same warning. A flipped layer that binds layer 0's own categorical column is refused with the `LayerCoordUnplaceable` reason on `+` and `fm.layer`, in static and interactive output. One route difference remains: a bar's injected zero anchor merges into the chart-level scale copy only, so a reversed-scale bar beside a scale-less point spans to zero on `fm.layer` but not on `+`.

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

#### `Axis`

```
Axis(title=None, *, orient=None, ticks=True, tick_count=None, tick_extra=False, tick_min_step=None,
     grid=True, grid_dash=None, grid_width=None, grid_color=None, grid_opacity=None,
     labels=True, label_angle=None, label_flush=False, label_overlap="greedy",
     label_format=None, label_format_type=None, label_font_size=None, label_color=None,
     domain=True, domain_width=None, domain_color=None,
     offset=None, translate=None, min_band=None, max_band=None,
     title_orient=None, title_font_size=None, title_color=None, title_padding=None,
     values=None, encode=None, zindex=None)
```

> **Dated note (2026-09-19, wave 5 — GH #182, closing #94: a gutter is reserved only for chrome that is drawn):** layout reserved margins for chrome it never drew. Hiding an axis kept the axis title's gutter, and a polar chart — which has no Cartesian axes to hide at all — reserved Cartesian margins regardless, so the plot area came out smaller than the drawn chrome justified and the chart sat off-centre in its own frame. Measured before the fix: `.axis(show=False)` reserved a 23.729px phantom title gutter per side plus a 14.046px phantom label band on the y side, and a polar chart reserved 23.214px on the left and 43.039px at the bottom. The gate now asks whether the axis actually **draws** — which is what the sibling label-band check three lines away already did — so `.axis(show=False)` on a 640x480 chart with 16px padding collapses to exactly `(16, 16, 608, 448)`, `fm.Axis(labels=False)` collapses the label band alone, and polar reserves nothing Cartesian. Scoping the gate to the `show` flags would have closed #182 and left #94 open, which is why it reads "is it drawn". **`AxesInput::show_x`/`show_y` accordingly change meaning** to "this Cartesian axis is drawn", with the coord folded in by the producer; a consumer wanting the raw user toggle reads `spec.axis_x`. `plots/matrix.py`'s clustermap dendrogram `title=""` workaround — which existed only to reach the title-width route so a hidden axis's phantom gutter stopped eating the dendrogram's share of `dendrogram_ratio` — is **removed** rather than left beside the rule that replaces it. **This reverses a prior decision:** `tests/test_finding_p8.py` (2026-08-27 batch) pinned the phantom as "the architecturally correct outcome" and asserted a polar panel's edge equals a single-axis chart's; both tests are inverted with rewritten rationales, and the reversal is recorded for the campaign's decision-flip audit. **All output with hidden or non-Cartesian axis chrome changes geometry.**

> **Dated note (2026-09-19, wave 5 — GH #111: a swatch shows the encoding it stands for):** when `stroke_dash` or `shape` shares its field with `color`, the aux legend block is suppressed so the reader sees one legend instead of two — and the surviving color legend then drew three identical **solid circles** over marks that were solid, `6,3` and `2,3`, or over marks that were circle, square and cross. The mapping the legend exists to explain was unreadable. The surviving entries now carry the suppressed channel's identity: a dash-carrying entry draws as a dashed **line** swatch, a shape-carrying entry draws its glyph. The issue's own prescription (thread the dash pattern into the color entries) was measured to be a no-op — the color swatch is a hard-coded circle and the pattern is only read inside the line arm — so the swatch **kind** had to change; `shape` needed only the thread, since the renderer already reads the shape name ahead of the symbol-kind match. Identities are matched to color entries **by label, never by index**, because both domains are built over the same field with independent sort specs, so index pairing silently transposes under `sort="descending"`.
>
> Three adjacent behaviors are settled rather than left undefined. (1) A suppressed channel's explicit `title=` is no longer silently discarded: where the color channel states no explicit title and the suppressed same-field channel does, that title is used; where both state explicit titles and they differ, the color channel's wins and a `RenderWarning` names the discarded one. (2) `color=fm.Color(..., legend=None)` with a same-field `stroke_dash`/`shape` no longer erases **both** blocks — suppression exists to avoid a duplicate, and with no color legend there is nothing to duplicate. That gate now asks whether the color legend **enumerates** the field rather than whether its scale is categorical, so a colorbar plus a categorical shape or dash renders the aux block too. (3) A chart binding `color`, `shape` **and** `dash` to one field cannot show both folded identities, because the glyphs are painted fill-only and a dash array on them paints nothing: **shape wins**, matching what a standalone shape legend already draws, the losing pattern is not written onto the entry at all so the scene carries only what it paints, and the drop warns. `size` is not part of this: `size=` on a `Utf8` column refuses, so the categorical-color size-suppression branch is unreachable for a string field, and the numeric same-field color+size merge already worked.

> **Vocabulary change (2026-06-22, cohesion-campaign T3.3b / D-EXTENT-1):** the
> axis layout-band overrides were renamed `min_extent`/`max_extent` →
> `min_band`/`max_band` so that `extent` is reserved for the data-domain sense
> only (the layout already uses `band` internally). `min_extent`/`max_extent`
> remain accepted as deprecated keyword aliases (emit a `DeprecationWarning`;
> supplying both a canonical and its alias raises `TypeError`). The same rename
> applies to `configure_axis(...)`.
>
> **Dated note (2026-09-23, wave 6B, `#149` — `Axis`/`AxisConfig` use the generated dataclass constructor, and every alias warning points at the caller):** the `min_extent=`/`max_extent=` `DeprecationWarning` from `fm.Axis(...)` and `fm.AxisConfig(...)` is now attributed to the caller's line; it previously pointed at ferrum's own source, so a `-W error::DeprecationWarning` filter keyed on the user's module never matched it. `configure_axis(x=False)`/`configure_axis(y=False)`'s vestigial-flag warning is attributed to the caller the same way, and `configure_axis` no longer forwards those no-op flags to the `AxisConfig` it builds, so that config carries the defaults (`x=True, y=True`) in `repr`/`__eq__` where it carried the caller's literal; wire output is identical because `to_dict()` drops both keys and no renderer reads them. Two further by-design changes: `Axis().min_extent`/`max_extent` (and the same on `AxisConfig`) read back the unset sentinel on an instance where they raised `AttributeError`, and `__match_args__` gains those two entries. `repr`, `==`, `hash`, `dataclasses.fields()`, `replace`, pickle, positional construction (`Axis('T', 'left')`), the unknown-kwarg `TypeError`, `to_dict()`, the wire and every rendered output are unchanged. `inspect.signature` shows both aliases as `InitVar[object]` parameters.
>
> **Amended 2026-09-24 (wave 6B close):** the attribution also holds through `dataclasses.replace(Axis(...), min_extent=...)` / `replace(AxisConfig(...), ...)`, which added frames above the generated `__init__` and pointed the warning at `dataclasses.py`; the stack level is now computed from the frames rather than fixed.

> **Design decision (2026-05-31, flexibility-campaign D3):** Per-channel
> `Axis(label_format=...)` and `tick_count` are honored at layout time (the
> renderer previously hardcoded the per-channel override to `None` at
> `render/prepare.rs`, so only chart-level `configure_axis` reached the
> formatter). Format support is the **full d3 grammar, single source of truth
> in Rust**, feeding SVG, PNG, and the strings baked into interactive exports
> identically: **numbers** use a hand-rolled d3-format implementation
> (`[[fill]align][sign][symbol][0][width][,][.precision][type]`, type chars
> incl. `s % p r g`, plus the `~` trim flag); **time** uses `chrono` strftime
> (`%b %Y`, `%Y-%m-%d`, `%H:%M`), replacing the prior hand-rolled date math.

> **Dated note (2026-09-03, batch B tasks 2/4 — format presets are honored everywhere, chart-level time formatting works, malformed specs refuse):** a named format preset (`ferrum.format_presets.ALL_PRESETS` — e.g. `"currency"`, `"percent"`, `"date_iso"`, `"ordinal"`) now resolves through one shared function (`resolve_format_or_raw`) on **all five** surfaces that accept a format string: chart-level `AxisConfig(label_format=...)`/`configure_axis(label_format=...)`, per-channel `Axis(label_format=...)`, per-channel `Legend(format=...)`, encoding `format=` (`Text`, `TooltipField`, and any channel routed through the shared encoding emitter), and the raw-dict normalize paths for both axis and legend (including the camelCase serde-alias spellings, e.g. `labelFormat`). Previously only `Axis(label_format=...)` resolved presets; the other four surfaces passed a preset name straight to Rust's d3-format parser, which rendered it as literal control characters into the SVG. A name that isn't a recognized preset is treated as an already-valid raw d3-format/strftime spec and passed through unchanged — this function never raises. `AxisConfig.label_format` is the one preset-names-only exception: it validates eagerly at construction (raises `ValueError` for an unrecognized name) since it has no raw-spec sibling field of its own; a raw d3-format/strftime string belongs in the parallel `label_format_raw` field instead (mutually exclusive with `label_format`).
>
> Chart-level **time** presets now actually format tick labels as dates: `configure_axis(label_format="date_iso")` previously rendered `300.000000%` (the chart-level path never threaded `format_type` through to the formatter); it now renders ISO dates as expected. `"ordinal"` produces real ordinal suffixes (`1st`, `2nd`, `3rd`, `4th`, …, with the `11th`–`13th` exceptions), not raw integers. `label_format=`/`format=` compose with `values=` in both directions on the surfaces that accept both. An explicit per-channel format (`Axis(label_format=...)`, `X(..., format=...)`) always beats a conflicting chart-level `configure_axis(...)` value — a cascade inversion that previously let chart-level silently win regardless of which was actually set, fixed at the one write site both former hand-copied call sites shared. **Malformed format specs now raise a typed error** (`RenderError::InvalidFormatSpec`) instead of emitting control characters into the SVG or, for a malformed time-with-trailing-token spec (e.g. `"curency%"`), panicking through the PyO3 boundary; the diagnosis routes real `%`-directive strftime specs before the ambiguous date/time guess, so the error names the actual problem. `configure_axis(x=..., y=...)` remain deprecated, inert flags (`Chart.axis(x=/y=)`, documented below, is the replacement for axis visibility; they carry no formatting meaning).
>
> **Dated note (2026-09-11, wave-1 hardening, #133):** an unambiguously
> time-typed format (`format_type` explicitly `"time"` — a strftime spec,
> or a named time preset like `"date_iso"`) now refuses with a typed error
> naming the axis/channel when the resolved scale is not temporal, on all
> three surfaces that accept a format: per-channel (`Axis(label_format=)`,
> encoding `format=`), chart-level (`configure_axis(label_format=)` /
> `axis_x=`/`axis_y=`/`axis_y2=`, both the plain `label_format` branch and
> the `tick_values` sibling branch), and the facet independent-axis path
> (`TickFormatMode::Immediate`, the surface the original issue's "both
> paths" phrasing missed). Previously a time format on a mixed
> temporal/quantitative chart's shared `axis=`/`configure_axis()` key
> silently applied to both axes, rendering a nonsense `"1970-01-01"` epoch
> artifact on the non-temporal one. The refusal is deliberately narrower
> than "any spec containing `%`": an ambiguous bare-`%` spec with no
> declared `format_type` (e.g. a plain d3 percent format like `".1%"`) is
> tolerated exactly as before and keeps falling through to the
> numeric/d3 grammar — only a format the caller (or a preset) unambiguously
> declared as time-typed refuses on a non-temporal scale.

> **Dated note (2026-09-22, wave 6A task 13, GH #147 + #146 — one answer to "is this a time format", and an independent facet axis stops dropping its temporal tick format):** six render sites (the shared axis, the independent facet axis, the chart-level applier, the colorbar, `mark_text`'s `text=` channel, and the tooltip column) each re-derived from the wire's `(format, format_type)` pair whether a format is d3, strftime or the ordinal preset, and they disagreed at the edges. They now share one resolution. A spec is read as strftime only when it contains a `%` escape; an undeclared `%`-bearing spec is read as time where the value it formats is temporal (an axis by its scale, a legend or the text channel by its field, see (8)), and the remaining sites (the tooltip column and tick thinning) read only the declared `format_type`. The #133 refusal above is unchanged except that it now also covers a `%`-free time-declared spec on the shared axis and no longer covers the ordinal preset, and that it now reaches the colorbar, the size, shape and stroke-dash legends and the text channel, see (3), (4) and (5). A second refusal mirrors it for the ordinal preset on a temporal value, see (7), and a third for a d3 number format on a temporal value, see (8). Eight outputs move, all of them inputs that were previously dropped or rendered as nonsense: **(1)** On a facet whose x scale resolves independently, `fm.Axis(label_format="%Y/%m/%d")` (with or without `label_format_type="time"`) under `.facet(col=…).share_scale(x="independent")` rendered the scale's own default labels (`2020 2020 2020 2020`), while the shared axis rendered the same chart's dates in the requested pattern. The independent axis now applies the pattern, and the two paths agree. This is the second silent drop found on the independent-facet-axis path in one wave, after `cull_threshold` (2026-09-20 note under §3.13). **(2)** `fm.Text("d", format="%Y/%m/%d", format_type="time")` rendered the default day-granularity date (`2020-01-01`) instead of the pattern it was given; it now renders `2020/01/01`. The tooltip column carries the same fix, which is latent today because `TooltipField(format_type=)` does not yet reach the wire. **(3)** A `%`-free spec declared as time (`label_format=".2f"` or `".3e"` with `label_format_type="time"`) meant three things: a colorbar and thinned ticks echoed its literal text (`.2f` on every label), a temporal per-channel axis applied it as a d3 number to the year labels (`2001.00`, `2.001e+3`), and an independent facet axis re-read those year labels as epoch milliseconds (`1970-01-01`). Every site now falls back to its own temporal default (the axis keeps its scale's labels, the colorbar and text mark their day-granularity default). The #133 refusal's message changed for every case it covers, the existing strftime and time-preset refusals included: it now reads `format ".2f" is typed "time", which on the 'x' axis requires a temporal scale, but the resolved scale is not temporal`, naming the declared type as the cause, where it previously read `format ".2f" on the 'x' axis requires ...`; code matching the old prefix needs updating. On a non-temporal axis the same pair is refused as a time format, by the #133 refusal above, on the shared axis as it already was on the independent one. The thinned-tick case is reachable only through a hand-written `chart_config` dict. The ordinal preset keeps exactly one wire token, `"__ordinal__"` (the `"ordinal"` preset still resolves to it, and both render `1st`, `2nd`, …). What is retired is a second, internal encoding of that sentinel, which let a literal one-character `label_format="\x01"` validate and render ordinals; that input is now an ordinary unrecognized d3 type character and renders plain numbers. **(4)** The ordinal preset beside a stray time type (`format="__ordinal__", format_type="time"`) rendered ordinals on a shared axis, `1970-01-01` on a colorbar and on `mark_text`, and was refused as a time format on a non-temporal independent facet axis. The declared preset now wins everywhere: on a non-temporal field every surface renders `1st`, `2nd`, …, and no axis refuses it as a time format. On a temporal field it is refused, see (7). **(5)** A time-typed format on a non-temporal field rendered on the colorbar, the size legend and `mark_text` as a column of `1970-01-01` labels, the epoch read of small numbers: `fm.Color("n", legend=fm.Legend(format=".2f", format_type="time"))` and `fm.Text("n", format=".2f", format_type="time")` both did this, as did a strftime pattern or a bare `format_type="time"` with no spec. These surfaces now raise the same #133 refusal, naming the surface the format was set on and checking the field rather than an axis scale: `format ".2f" is typed "time", which on the 'color' legend requires a temporal field, but the field is not temporal` (the size legend reads `on the 'size' legend`, the text channel `on the 'text' channel`; a bare type reads `format type "time" on the …`). The shape and stroke-dash legends, which enumerate categories and never format a label, dropped the pair silently where the color legend refused it (`fm.Shape("g", legend=fm.Legend(format=".2f", format_type="time"))` rendered, `fm.Color("g", legend=…)` refused); they now refuse it the same way (`on the 'shape' legend`, `on the 'stroke_dash' legend`), with the same disable exemption. The legend check now walks the one list of legend channels the legend builder walks, so a legend channel cannot be added without meeting it (wave 6A task 22). A field is temporal when its declared type is temporal or, undeclared, its column is a date or datetime, the same rule scale resolution uses, so `"n:T"` over epoch-millisecond numbers keeps its date labels. A string field is not temporal, so `fm.Text("s", format="%Y", format_type="time")` over a string column, which rendered the strings verbatim and dropped the format, now refuses too. A bare `format_type="time"` with no spec now gets the same answer on every surface: on a non-temporal axis (`fm.Axis(label_format_type="time")`, `fm.X("n:Q", format_type="time")`, the shared and the independent facet axis alike) it used to be dropped silently and the scale's numbers rendered; it is now refused (`format type "time" on the 'x' axis requires a temporal scale, …`), and on a temporal axis it still renders the scale's own date labels. A bare `fm.Axis(label_format_type=)` also types the shorthand `fm.X(format=)` beside it, where the type used to be dropped, and the pair so merged meets the same checks as its one-place spellings: `fm.X("d:T", format="%Y", axis=fm.Axis(label_format_type="number"))` refuses with `invalid format spec "%Y": unrecognized token "Y" …`, exactly as `fm.Axis(label_format="%Y", label_format_type="number")` and `fm.X(format="%Y", format_type="number")` do. The shorthand pair an Axis type overrides still meets the #133 refusal, so `fm.X("n:Q", format=".1f", format_type="time", axis=fm.Axis(label_format_type="number"))` refuses as `fm.X("n:Q", format=".1f", format_type="time")` does. The legend refusal validates the spec the caller wrote, not what happens to be drawn: a time format on a non-temporal legend field is refused unless the caller disabled that legend, and the two ways to disable it count alike, per channel (`fm.Color(…, legend=None)`, `Legend(orient="none")`) and chart-level (`.configure_legend(orient="none")`). Axes have no such exemption, on purpose: `fm.X("n:Q", axis=None, format=".2f", format_type="time")` refuses. An `X`/`Y` `Legend(format=)` cascades onto the color field, so validating a legend nobody draws would judge a field the caller never formatted, while an axis format is written on the channel it formats; do not "fix" either side toward the other. A colorbar the renderer itself suppresses with a warning (a `mark_line` or `mark_ribbon` layer, which cannot draw continuous color) still refuses, as does a color legend on a categorical scale. The tooltip column is not covered, because no Python spelling puts a `format_type` on its wire yet. (Wave 6A task 21.) **(6)** An untyped `%`-bearing spec on a non-temporal field meant two things. `fm.X("n", axis=fm.Axis(label_format="%Y"))` over a numeric column with no declared type rendered `250.000000% 300.000000% …` on the shared and the independent facet axis (the axis guessed a date pattern, found no dates, and handed the pattern to the lenient d3 reader), while `Legend(format="%Y")` and `Text(format="%Y")` refused with `invalid format spec "%Y": unrecognized token "Y" …`. The axis now gives that same refusal. A real d3 percent spec (`".1%"`) still renders percentages there, and on a temporal axis an untyped `"%Y"` still renders years. The stroke-dash legend's `Legend(format=)` was never grammar-checked and rendered any spec; it is now checked like the shape legend's. **(7)** The ordinal preset (`"ordinal"`, `"__ordinal__"`, with or without `format_type="time"`) on a temporal field gave three answers: `2001st 2002nd …` on the shared axis, the format dropped on an independent facet axis, and `959817600000th …` (the ordinal of the epoch-millisecond count) on the colorbar, the size legend and `mark_text`. The preset formats numbers, so every surface now refuses it: `the "ordinal" format on the 'x' axis formats numbers and requires a non-temporal scale, but the resolved scale is temporal` (a legend or text channel reads `… requires a non-temporal field, but the field is temporal`). Chart-level `.configure_axis(label_format="ordinal")` over a temporal axis refuses the same way. A shape or stroke-dash legend cannot be built over a temporal field at all, so it never reaches this check. (Wave 6A task 22.) **(8)** Two untyped specs on a temporal field, each settled by the user's ruling (2026-09-22, wave 6A task 23). *A d3 number format* (`.2f`, `,.0f`, `d`, or a `"number"`-typed preset such as `"percent"`) on a temporal field gave three answers: the x and y axes formatted the calendar year (`2001.00 2002.00 …`, `2,001 …`), an independent facet axis dropped the format and showed its date labels, and the colorbar, the size legend and `mark_text` formatted the epoch-millisecond count (`959817600000.00 …`). Every surface now refuses it: `the number format ".2f" on the 'x' axis requires a non-temporal scale, but the resolved scale is temporal; a date pattern (e.g. "%Y") formats a temporal value` (a legend or the text channel reads `… requires a non-temporal field, but the field is temporal; …`). Chart-level `.configure_axis(label_format="percent")` refuses the same way over a temporal axis; that format reaches both axes, so a percent-valued y beside a temporal x needs `.configure(axis_y=AxisConfig(label_format=…))` or the per-channel `Axis`. The typed pair `.2f` + `"time"` is unchanged: it names no pattern and falls back to the temporal default. A split spelling whose `Axis` type retypes a number-format shorthand as time (`fm.X("d:T", format=".2f", axis=fm.Axis(label_format_type="time"))`) gives its one-place spellings' answer, the temporal default: only a time-typed shorthand is checked as written, as in (5), since that is the half an `Axis` type could launder (user ruling, wave 6A task 23 cycle 2). *An undeclared date pattern* (`"%Y"`) on a temporal field rendered years on the axes but was refused by the legends and the text channel as a d3 spec (`invalid format spec "%Y": unrecognized token "Y"`). A legend or the text channel now reads an undeclared `%`-bearing spec by its field's temporality, as an axis reads it by its scale's, so `fm.Text("d", format="%Y")` and `fm.Color("d", legend=fm.Legend(format="%Y"))` over a datetime column render years; over a non-temporal field the spec stays d3 and still refuses, as in (6). An undeclared `%` spec on a temporal field that is not a date pattern (`".1%"`) now refuses on every surface with `invalid format spec ".1%": invalid strftime pattern ".1%" …`, where the colorbar, the size legend and the text channel formatted the epoch-millisecond count as a percentage. The tooltip column is not covered, because its type does not reach the wire. An untyped `".1%"` on a numeric axis is now a d3 spec from the start, so it also formats explicit tick values: `fm.X("q", format=".1%", axis=fm.Axis(values=[0.2, 0.6]))` rendered `0.2`, `0.6` (the spec was guessed as a date pattern, un-guessed and applied to the computed ticks only) and now renders `20.0%`, `60.0%`, as `".2f"` and the `"percent"` preset already did (wave 6A task 23 cycle 3). (Close remediation, wave 6A task 17: the format surfaces answer each `(format, format_type)` pair once, and the resolution now names the "time-typed, no pattern" case itself, so no site re-reads `format_type` after resolving.) No golden moved, because no golden sets a caller-specified tick format; the pins live in `tests/test_format_predicate_unification.py`.

> **Dated note (2026-09-03, batch B task 7 — explicit-equals-default now serializes):** the seven fields shown above with a concrete default (`ticks`, `tick_extra`, `grid`, `labels`, `label_flush`, `label_overlap`, `domain`) follow an **omit-vs-explicit** wire contract: omitting the parameter entirely (the common case, and what every signature default above represents) means "not specified" and the renderer's own default applies as before — byte-identical to today. Passing the parameter **explicitly**, even with the exact value shown above (e.g. `Axis(ticks=True)` or `Axis(label_overlap="greedy")`), now always reaches the wire, which is a real behavior change: an explicit per-channel value beats a conflicting chart-level `configure_axis(...)`/theme value for that field, where chart-level previously won silently regardless of whether the per-channel field was actually named. Previously an explicit value equal to the default was indistinguishable from "not specified" and could be silently overridden by chart-level. `Legend`'s `orient`/`direction` fields (below) carry the identical contract.

> **Dated note (2026-09-03, batch B task 1 — chart-level config refuses unknown keys):** every `configure_*()` domain — `axis`/`axis_x`/`axis_y`/`axis_y2`, `legend`, `title`, `grid`, `padding` (below), `color` — shares one wire-level gate: an unknown key inside any section (a typo, a removed field, a spelling from a different surface) raises `chart config: unknown key '<k>' in <section>; accepted: <sorted list>` before the value ever reaches the Rust deserializer, instead of silently vanishing. Every field the wire schema parses is dispositioned `Honored`/`Rejected` (with a reason) in a completeness-tested manifest, so a field that is accepted syntactically but not yet consumed cannot drift back into silent no-op territory unnoticed.
>
> **Dated note (2026-09-11, wave-1 hardening, #148 — the array-section carve-out closes):** the two ARRAY-shaped `chart_config` sections, `annotations` and `structural`, were the one carve-out this gate didn't cover — each item's `type` tag routed to serde's own internally-tagged deserialize with no per-item key check, so an unknown key inside an `annotations[]`/`structural[]` item (e.g. `annotations[0]` of `type: "text"`) either vanished silently or, depending on the variant, slipped through. The gate now walks each array item, reads its `type` tag, and validates it against that variant's own accepted-key set (8 annotation variants, 2 structural variants — `BreakAxis`/`Inset`), refusing with the same shape the object-keyed sections use, now naming the item index and type: `chart config: unknown key '<k>' in annotations[0] (text); accepted: <sorted list>`. Every one of the 87 newly-covered fields is dispositioned in the same completeness-tested manifest the object-keyed sections already use.
>
> **Dated note (2026-09-11, wave-1 hardening, #92 — `BreakAxis.gap` construction AND wire refusal):** a `BreakAxis(gap=...)` with a non-finite (`NaN`/`±inf`) bound, a malformed pair shape, or a reversed `(start, end)` pair now refuses at BOTH the boundaries the spec's validation-placement doctrine says it should: Python construction (`BreakAxis.__post_init__`, mirroring the `Inset`/`_validate_inset_bounds` idiom already four lines away in the same file), and — for a hand-built `chart_config` dict that bypasses Python construction entirely — the same per-item wire gate #148 closed above, which now also value-checks a `break_axis` item's `gaps[]` entries directly off the raw Python floats (not the post-JSON-bridge deserialized value: `json.dumps(float('nan'))` emits a bare `NaN` token the JSON parser already rejects with an opaque message before any check downstream of it could run, so the value check has to sit upstream of that bridge to give the NaN case a real, pinned message and to actually reach the reversed-pair case, which is otherwise valid JSON that sails through untouched). `Facet(ncols=)`/`Facet(nrows=)` — both the `.facet(ncols=)` chart method and `encode(facet=fm.Facet(columns=))` — gain the matching #238 construction-time check (`validate_positive_int`, since renamed `coerce_positive_int`, refusing anything that isn't a genuine `int >= 1`; `bool` is excluded despite being an `int` subclass); a non-positive or non-integer wrap count used to construct silently and either render nonsense or get overridden by inference. `FacetRow`/`FacetCol(columns=)` never reach this check (see the #240 dated note above under §3.2 — those two channels stay warn-and-drop by design, message corrected only).

> **Dated note (2026-09-03, batch B task 8 — axis config plumbing honored end to end):** `domain_min`/`domain_max`/`nice`/`zero` (chart-level `configure_axis(...)` and the per-channel fields above) now feed positional scale resolution at the X axis, the Y axis, and the secondary Y axis alike. An encoding's own `scale=` domain still wins silently over a chart-level `domain_min`/`domain_max` (the documented cascade), an ordinal axis warns rather than applying a numeric domain it cannot use, a user-set domain on a `log` scale refuses with `LogScale`'s own sentence, `nice=True` on every continuous kind (including `log` and `time`) rounds via that scale kind's own `nice()` too (chart-level and per-scale construction read identically on every field, including `nice` — that parity was fixed 2026-09-03 in the batch-B design-review remediation: a `log` axis under `configure_axis(nice=True)` previously refused with `InvalidScaleDomainConfig` instead of rendering, because the pre-fix implementation rounded every scale kind with the same linear step regardless of kind), and a degenerate `domain_min == domain_max` refuses. Per-axis grid gains a real bool-and-style contract via the new public **`ferrum.GridAxisConfig`** (`enabled`, `color`, `width`, `dash`, `opacity`) — `GridConfig.x`/`GridConfig.y` accept a bare `bool` (enable/disable only, the historical spelling), a `GridAxisConfig`, or its dict form; anything else refuses naming the accepted shapes. Chart-level `tick_count`, `labels`, `ticks`, and the tri-state `title` (omit/`None`/string, matching the per-channel three-way contract above) are honored; `label_overlap` is honored on the **y** axis too (previously x-only). `axis_y2` (`configure(axis_y2=AxisConfig(...))`) is a real consumer end to end — style, axis preparation, and domain resolution all read it for the secondary y axis on a `SecondaryY` chart (previously silently dropped in full: neither color nor `tick_count` reached the rendered SVG). `offset` and `label_flush` are typed `AxisConfig`/`configure_axis()` fields (shift the axis off the plot edge; flush-align the first/last tick label to the axis ends).
>
> **Behavior changes:** grid and domain settings compose by **precedence, not AND** — a chart-level and a per-channel setting no longer both have to agree for either to take effect. `axis_x`-scoped configuration no longer leaks onto the y axis (a pre-existing cross-talk bug). `.override(x_axis_tick_size=...)` now correctly beats `configure_axis(tick_size=...)`, matching the documented per-channel-over-chart-level cascade. `.override(axis_label_format=<raw d3 spec>)` now validates the spec at the override boundary (previously an unvalidated raw string reached the wire unchecked); the escape hatch for a raw, unvalidated spec is `axis_label_format_raw=` on the same surface. `.override(axis_x=...)` / `.override(axis_y=...)` — and the four `x_axis_x`/`x_axis_y`/`y_axis_x`/`y_axis_y` spellings — now raise a typed `FerrumOverrideError` naming the replacement (`Chart.axis(x=False)` / `Chart.axis(y=False)`, documented below) instead of silently no-opping; these were the deprecated, do-nothing spellings of the `x`/`y` visibility flags (see `Chart.axis()` below), not a formatting or layout override.

#### `Legend`

```
Legend(title=None, *, orient="right", direction="vertical", type=None,
       tick_count=None, tick_min_step=None, values=None,
       format=None, format_type=None,
       label_font_size=None, label_color=None, label_limit=None,
       symbol_size=None, symbol_stroke_width=None, symbol_type=None,
       gradient_length=None, gradient_thickness=None,
       columns=None, column_padding=None, row_padding=None,
       clip_height=None, title_font_size=None, title_padding=None,
       offset=None, padding=None, zindex=None)
```

Set `legend=None` on a channel that owns a legend surface (`Color`, `Size`, `Shape`, and the other appearance channels) to suppress the legend for that channel. `X`/`Y` are the exception — see the batch B task 7 dated note below.

> **Dated note (2026-09-03, batch B task 7 — legend contract):** `orient` and `direction` are fully independent. `orient` places the legend block on a chart edge (`"right"`/`"left"`/`"top"`/`"bottom"`); `direction` arranges the entries within it (`"vertical"`/`"horizontal"`) and now also **sizes** the reserved block, so all eight combinations render every entry. Absent `direction`, the edge implies it (side legends stack, top/bottom strips run across). A colorbar honors `direction` too: `"horizontal"` draws a left→right gradient bar with its tick labels centered beneath it.
>
> `orient` and `direction` follow the same **omit-vs-explicit** wire contract as the seven `Axis` fields above: omitting either parameter (the signature defaults `orient="right"`/`direction="vertical"` shown above represent this) means "not specified," and the renderer's own orient-implied default applies — byte-identical to today. Passing either explicitly, even as `"right"`/`"vertical"`, now always reaches the wire, so an explicit per-channel value beats a conflicting chart-level `configure_legend(...)` (behavior change 1 below).
>
> Further clarifications land with it:
>
> - `Legend(orient="none")` on a channel suppresses that channel's legend — the per-channel spelling of chart-level `configure_legend(orient="none")`, identical in effect to `legend=None`.
> - `Legend(values=[...])` on a **categorical** legend filters and orders the entries to the listed values (previously honored only on gradient/colorbar legends, where it replaces the tick labels — unchanged). A value naming no category has no swatch to draw: it is skipped and reported as a `RenderWarning`.
> - `X(legend=...)` / `Y(legend=...)` are honored for explicit style fields. The positional channels have no legend block of their own, so an explicit `Legend(...)` (or a raw dict) addresses the chart's legend, filling any field the `color` channel's own `legend=` left unset. Per-channel precedence is `color` > `x` > `y`. Bare `legend=None` / `legend=False` on `x`/`y` is a no-op (see behavior change 3 below) — there is no "x's/y's own legend" for that spelling to suppress.
> - `orient`/`direction` tokens (both here and on `configure_legend(...)`) validate at construction: an unrecognized value raises `ValueError` naming the accepted set, rather than silently falling back to a default or reaching the wire unrecognized.
> - `Legend(format=...)` resolves named presets through the same shared resolver `Axis(label_format=...)` uses (`configure_legend()`/`LegendConfig` has no `format` field of its own — colorbar/gradient formatting is per-channel only) — see the format-preset dated note under `Axis` above (batch B tasks 2/4) for the full contract (all five surfaces, malformed-spec refusals, `format=`/`values=` composition).
> - **Behavior changes (4):**
>   1. Per-channel `Legend(orient=/columns=/title_font_size=)` now beats chart-level `configure_legend(...)` for those three fields, matching the documented cascade (mark literal > per-channel > chart-level > theme). Previously chart-level silently overwrote the per-channel value.
>   2. `configure_legend(label_font_size=)` now sizes legend labels only; it used to write a slot shared with the axes and so resized axis tick labels as a side effect. Use `configure_axis(label_font_size=)` for those.
>   3. `X(legend=...)` / `Y(legend=...)` reach the chart's legend for every **explicit** style field (`orient=`, `title=`, etc.), through the same per-channel cascade `color`'s `legend=` uses. Bare `legend=None` / `legend=False` is a **no-op** — see the 2026-09-03 design-review-remediation dated note directly below for why this is not "suppress the chart's legend" the way `Color(legend=None)` is.
>   4. A `Legend`/`configure_legend` block oriented `"top"`/`"bottom"` with **no explicit `direction`** now defaults to a **horizontal** gradient bar on a continuous (colorbar) legend, matching the orient-implied default the categorical arm already had. It previously always rendered as a tall vertical bar regardless of orient. Pass `direction="vertical"` explicitly to keep the old bar shape.
>
> **Dated note (2026-09-03, batch B design-review remediation — `X`/`Y(legend=None)` no longer suppresses `color`'s legend):** task 7's original behavior change 3 above routed bare `legend=None`/`False` on `x`/`y` through the exact same "suppress" mapping `Color(legend=None)` uses, so `X("a", legend=None)` suppressed the **color channel's** legend — a cross-channel action at a distance under the most idiomatic "not applicable here" spelling, and a contradiction of the None-means-unspecified rule this same batch codifies for `orient=None`/`ticks=None`/etc. elsewhere (`ferrum._title_sentinel.is_unspecified`). `x`/`y` own no legend surface of their own, so bare `None`/`False` is now a no-op on those two channels: nothing is serialized, exactly as if `legend=` had been omitted, and the color legend (or whichever channel actually owns one) is left untouched. An **explicit** `Legend(...)`/dict passed to `x`/`y` — including one that spells `{"disabled": True}` itself — is an unambiguous, typed ask and still reaches the cascade unchanged; only the bare-boolean spelling's meaning changed. Suppress the rendered legend from the channel that actually owns it (`Color(legend=None)`, `Size(legend=None)`, `Shape(legend=None)`, ...).

#### `Chart.axis()` — spec-level axis suppression (added 2026-05-11)

```
Chart.axis(*, x: bool | None = None,
              y: bool | None = None,
              show: bool | None = None) -> Chart
```

Hides (or shows) the chart's x/y axis line, ticks, tick labels, and axis
title at layout time. Returns a new `Chart`. Mutually exclusive with
per-channel `Axis(...)` configuration on individual encodings (the per-axis
`show=False` always wins).

- `x=False` / `y=False` hides the respective axis.
- `show=False` is shorthand for `axis(x=False, y=False)`.
- Plot-area pixel rect is unchanged — gutters reserved for axis decorations
  stay reserved, so compound views can author each child chart at a fixed
  size and compose with a stable grid.

Used internally by `clustermap()` (dendrogram panels) and `JointChart`
(marginal panels). Replaces an earlier post-render SVG-regex stripper which
was theme-color-fragile.

Serializes to two optional booleans on `ChartSpec`: `axis_x: bool | None`,
`axis_y: bool | None`. Both default to `None` (visible).

> **Dated note (2026-09-03, batch B task 1b — deprecated `override(axis_x=)`/`override(axis_y=)` now refuse):** `Chart.override(axis_x=..., ...)` / `Chart.override(axis_y=..., ...)` — and the four `x_axis_x=`/`x_axis_y=`/`y_axis_x=`/`y_axis_y=` spellings on the same low-level `.override()` escape hatch — previously emitted a `DeprecationWarning` and then silently did nothing (they addressed `AxisConfig`'s vestigial, already-deprecated `x`/`y` no-op flags, not this method). They now raise a typed `FerrumOverrideError` naming the offending kwarg and pointing at the real replacement, `Chart.axis(x=False)` / `Chart.axis(y=False)` documented above. This is a **breaking change**: code relying on the old warn-and-ignore behavior now raises instead.

#### Chart-level Padding — `configure_padding()` / `PaddingConfig`

```
Chart.configure_padding(*, top=None, right=None, bottom=None, left=None, auto: bool | None = None) -> Chart
```

Sets the minimum pixel padding on each of the plot's four sides, independent of the theme's flat `padding` default (§3.13). Each side is optional and independent: an unset side falls back to the theme value; a set side always wins there, whether or not `auto` is on.

- `top`, `right`, `bottom`, `left`: pixel values. Each must be a finite, non-negative number (`int` or `float`; `bool` is excluded even though it is an `int` subclass). A negative value, a non-numeric value (including a `NormCoord` or any other non-`Real` type; these are pixel slots, not relative-coordinate slots), or a non-finite value (`NaN`, `±inf`) raises a typed `ValueError` naming the pixel contract. Validation is construction-time only, in `PaddingConfig.__post_init__`; there is exactly one entry point, not several that must be kept in sync. The low-level `.override(padding_top=..., ...)` escape hatch (scoped to the four numeric side leaves; `padding_auto` is a separate bool leaf, checked in the override's own voice as described under `auto` below) reaches this same check because it constructs a `PaddingConfig` from the whole `padding_*` section named in one `.override(...)` call and takes what that construction produces, so it is a caller of `__post_init__`, not a second validator. `PaddingConfig.to_dict()` performs no validation at all; it only serializes whatever a construction-time-valid instance already holds, so a value already refused at construction can never reach it.
- `auto` (`bool | None`, default `None`): per-side auto-expand. `None` means unset: `auto` is left off the wire and the built-in default applies, which is **on** (there is no theme key for it). `auto=False` turns it off and `auto=True` turns it on. A configure layer that sets only sides leaves `auto` as an earlier layer (or a composite parent) set it. A set `auto` must be `True` or `False` (`numpy.bool_` is accepted); anything else is refused in the voice of the door used: `padding.auto must be True or False; received 'yes'` at the call, from `configure_padding(auto=)` and `PaddingConfig(auto=)`, and `FerrumOverrideError: Chart.override(padding_auto=...) must be True or False; received 'yes'` from `.override(padding_auto=)`, at render, when overrides are applied (since 2026-10-08; see the dated note below). `.override(padding_auto=None)` is no override. Until 2026-10-07 the default was `False`; see the dated note below.

  When `auto` is on, a side that is **not** explicitly set (`top=`/`right=`/`bottom=`/`left=` all still `None` on that side) is expanded (beyond the theme's flat default) to keep two specific things from clipping past the rendered viewport edge:
  1. **Edge tick-label overhang** on a **continuous** axis: the first/last tick label commonly extends past the plot's nominal edge; `auto` measures the real overhang and reserves enough margin to contain it, net of whatever a legend, the chart title or a secondary-y band already reserves on that side, respecting any `max_band` cap on that axis's own reserved gutter.
  2. **A clipped axis title** — solved via a closed-form, asymmetric recentering pass that runs *after* the tick-label pass, against the post-tick-expansion insets (the two passes compose sequentially, not independently).

  A side you *do* set explicitly is never touched by `auto`, on that side, regardless of what the measured overhang would otherwise ask for. `auto` does **not** expand for: annotations; ordinal/nominal axis tick labels (which sit inside fixed band slots and never overhang the plot edge by construction); a hidden axis (`Chart.axis(x=False)`/`(y=False)` — the title pass gates on axis visibility); or, on either axis, a title wider than the viewport itself — that axis's title can never be recentered into view no matter how much padding `auto` adds, so `auto` disables itself entirely on that axis, tick-label rescue included, rather than reserving margin that cannot help. A y-axis whose gutter is capped tight via `max_band` can still leave its own tick label or title outside the canvas on that axis — `auto`'s tick-label pass respects the cap rather than overriding it, so `max_band` is a hard ceiling `auto` will not breach even at the cost of a clipped label.

  `auto`'s total contribution is capped to whatever the viewport can actually still afford: in an infeasible regime (the fixed/floor padding alone already leaves no usable panel area), `auto` contributes **nothing** on that axis — the render is byte-identical to `auto=False` rather than partially shifting the clip point — and `auto=True` can never itself convert a chart that would otherwise render into a `PaddingExceedsViewport` refusal. When the desired expansion doesn't fully fit but the floor does, `auto` applies the largest expansion that does fit (scaled down proportionally, preserving which side needed more) rather than either refusing or overshooting.

- **`PaddingExceedsViewport`** — raised when a padding value (explicit or the theme default) leaves no usable plot area on an axis. The error reports the **caller's actual value and the specific side that overflowed** (`"padding {side}={padding} exceeds viewport dimension {viewport_dim}"`, `side` one of `"top"`/`"right"`/`"bottom"`/`"left"`) — previously it always reported the theme's default of `16` regardless of what the caller actually set, which misdirected debugging on any large explicit override (e.g. `configure_padding(top=1e9)`).

> **Dated note (2026-09-03, batch B task 6 — padding validates at every entry point):** see the pixel-contract refusal described above (`top`/`right`/`bottom`/`left`); this is the point where `NF-B5`/`NF-B7`'s Python half closed. Previously an invalid value (negative, non-numeric, or non-finite) reached the JSON serializer unchecked and surfaced as an artifact message naming neither the field nor the contract (`ValueError: chart_config: expected value at line 1 column 27`).

> **Dated note (2026-09-03, batch B task 9 — `padding.auto` is implemented):** `auto` did not exist before this task; the field, its opt-in default, the tick-label/title expansion passes, the `max_band` interaction, and the honest `PaddingExceedsViewport` reporting are all new in this batch (closes `F-L07-08`/`NF-B6`). No golden output changes for any chart that does not pass `auto=True`.

> **Dated note (2026-09-25, wave 6C, `#139`(2)(3)(4) — `padding.auto` hardening):** `auto=False` output stays byte-identical; at `auto=True`, per-axis infeasibility is now decided before the tick-label contributions are assembled (a suppressed axis's tick contribution is zeroed too, not just its title's), the title pass solves against that already-suppressed geometry, and an explicit side's own extras are zeroed before the viewport cap rather than charged against the auto budget.

> **Dated note (2026-10-07, #135: `padding.auto` is on by default, with one owner; supersedes the "Default stays `False`" text of the 2026-09-03 implementation):** The built-in layout default (`ThemePadding::default().padding_auto` in Rust) is now `true`, and it is the only place the default lives; there is no theme key. `PaddingConfig.auto` and `configure_padding(auto=)` changed from `bool = False` to `bool | None = None`, and an unset `auto` never reaches the wire.
>
> - **Charts that never mention `auto` get the rescue.** Before, a chart with no padding configuration ran with `auto` off. Measured on a 180x200 `mark_point()` chart over x in 0..1e6 with `tick_values=[0, 1_000_000]`: with `label_format="integer"`, the last x tick label overhung the canvas by 1 px and now ends 9 px inside it, the plot narrowing from 101.056 px to 90.832 px; with `label_format="currency"`, the label overhung by 4 px and now ends 9 px inside, the plot narrowing from 101.056 px to 87.303 px. Charts whose labels already fit render byte-identical. `configure_padding(auto=False)` restores the previous geometry exactly.
> - **A later configure layer no longer resets `auto`.** Python used to send `auto: false` on every `PaddingConfig`, even one that only set a side, so `configure_padding(auto=True).configure_padding(left=5)` merged to `{"auto": False, "left": 5.0}` and turned `auto` off. It now merges to `{"auto": True, "left": 5.0}`. The same holds for a composite parent's `auto=True` reaching a leaf that sets its own sides. `PaddingConfig().to_dict()` was `{"auto": False}` and is now `{}`. The render path and the chrome path share one configure-layer merge.
> - **`fm.compute_layout` runs with `auto` on.** The binding has no `auto` argument and uses the built-in default, so it went from off to on; turning it off there is a logged follow-up.
> - **`auto` subtracts the chrome already reserved on a side.** The rescue used to ignore a legend, the chart title and secondary-y bands, so `configure_padding(auto=True)` over-padded a chart with a right legend or a secondary y axis whose last x label already fit inside that gutter: by 9 px with a right legend and 14 px with a secondary y axis on a 200 px wide layout. It now pads 0 px in those cases, and pads exactly the remainder when a label overhangs past the gutter. The legend and title reservations depend on the padding (a legend is capped at half the remaining width or height), so the extra padding is solved as a bounded fixed point: each pass measures the real reservations at the candidate padding, accepting once they agree within 1e-6 px; a chart without a capped legend settles in 2 passes. Explicit `configure_padding(auto=True)` charts with a legend, chart title or secondary y axis can therefore move.
> - **A non-bool `auto` is refused at the call.** Before, `configure_padding(auto="yes")` was accepted at the call and failed at render with `ValueError: chart_config: invalid type: string "yes", expected a boolean at line 1 column 26`. It now raises `ValueError: padding.auto must be True or False; received 'yes'`, and `.override(padding_auto="yes")` raises `ValueError: override 'padding_auto' must be True or False; received 'yes'`. Each door refuses in its own voice (user ruling 2026-10-08, part of this change). `1` and `0` are refused the same way, never read by truthiness.

> **Dated note (2026-10-08, #135: `padding.auto` measures the x labels the axis draws; supersedes the 2026-10-07 note's headline measurements and its `override 'padding_auto'` refusal text):** The edge-label rescue used to charge half the width of the raw first and last x labels as if they were flat, before the axis chose an angle, shrank its font or thinned its labels, and as if the edge tick sat exactly on the plot edge. An explicit `label_angle` and top-oriented labels skipped the correction entirely. The extra margin then narrowed the plot, which could itself change the angle the axis chose.
>
> - **The rescue measures what the renderer draws.** Each pass of the bounded fixed point lays the plot out at its candidate padding, runs the real x-axis presentation (angle, font shrink, label thinning), and measures every drawn x label's ink at its drawn angle and anchor, using the tick's real inset from the plot edge. There is no second angle predictor. Explicit angles, cascade angles and top-oriented labels all go through this one measure. Only a drawn, labeled, continuous x axis is measured; other charts lay out as before. The bound is unchanged: at most 64 passes, accepting once the chrome credit agrees within 1e-6 px and the measured overhang has settled, else falling back to the last covering pass.
> - **Each label is charged by its true lever (user ruling 2026-10-08).** A label that overhangs a side by `o` px asks for `o / lever` px on that side, where `lever` is how far the label's tick moves per pixel of that margin, measured by laying the plot out with that margin 1 px wider. With the scale's capped 8 px inset, a tick at domain fraction `t` has lever `1 - t` against the left margin and `t` against the right, so a tick pinned at the far end has lever 0 against the near margin; with an uncapped inset, that far-end tick has lever `padding_frac`. An outer facet panel's tick moves by its panel's share of the plot region. Each side takes the maximum over every drawn label of every panel. A label whose lever is below 1e-9 contributes nothing to that side.
> - **An infeasible charge contributes nothing (user ruling 2026-10-08).** A label whose charge exceeds the room its panel has on that axis (its width less the 1 px minimum panel size, per pixel of the panel's measured shrink) adds nothing to that side. It is not capped, and it does not reduce the other side's extra. This mirrors the rule for a title wider than the viewport.
> - **A collapsed pass never covers (user ruling 2026-10-08).** A pass whose plot has a panel at or below the 1 px minimum on either dimension never counts as covering, and a pass that settles on a collapse ends the search. When no pass covers, the per-side maximum of the extra each pass needed ships only if laying it out does not collapse the plot; otherwise `auto` adds nothing and the chart lays out exactly as with `auto=False`. `padding.auto` never blanks a chart that `auto=False` draws.
> - **Headline, re-measured.** On the 180x200 chart above, the last label now ends 4 px inside the canvas with `label_format="integer"` (plot width 101.056 px to 95.612 px) and 5 px inside with `label_format="currency"` (101.056 px to 91.898 px). The 2026-10-07 note measured 9 px for both (plot widths 90.832 px and 87.303 px); the difference is the tick's 8 px inset, which the old flat formula ignored. A flat edge label's extra is the overhang less that inset.
> - **Rotated and thinned labels that fit are byte-identical to `auto=False`.** On a 180x200 `mark_point()` over `x = [0, 5e5, 1e6]` with `label_format="currency"`, the axis draws `$0`, `$300,000`, `$600,000` and `$900,000` at -90 degrees, and they fit. `auto` used to add 13.753 px on the right for the culled `$1,000,000` (plot width 104.273 px to 90.520 px); it now adds nothing.
> - **Explicit angles and top-oriented labels are now contained.** On a 240x200 chart over x in 1e6..2e6 with currency ticks at both ends and `label_angle=-45`: on the bottom axis with the y axis hidden, the first label ran 15 px past the left edge and now ends 12 px inside it; with `orient="top"`, the last label ran 10 px past the right edge and now ends 19 px inside it. Rotated ascent and descent use the line height as their measure, so a rotated label is contained with a few pixels to spare.
> - **Cost.** A plain chart settles in two passes. Each pass lays the plot out once, and twice more (the lever probes) when its x axis is continuous and labeled; measured on a 200-point 640x480 scatter, the render is within noise of `auto=False`. A chart in a width band where the axis's presentation alternates between passes (flat labels that need an extra at a width where they no longer stay flat) runs all 64 passes: about +0.8 ms on a 15.3 ms render.
> - **The override door refuses as `FerrumOverrideError`, in the `Chart.override` voice (user ruling 2026-10-08).** `FerrumOverrideError`'s contract is that every `.override` refusal is that class and names `Chart.override`, so a figure that wraps `.override` can re-voice it. `.override(padding_auto="yes")` raised `ValueError: override 'padding_auto' must be True or False; received 'yes'` and now raises `FerrumOverrideError: Chart.override(padding_auto=...) must be True or False; received 'yes'`; `.override(padding_top=-1)` raised `ValueError: override 'padding_top' must be non-negative; received -1` and now raises `FerrumOverrideError: Chart.override(padding_top=...) must be non-negative; received -1`, and the same for the other three sides. `FerrumOverrideError` is a `ValueError`. `PaddingConfig(auto=)` and `configure_padding(auto=)` keep `padding.auto must be True or False; received ...`.
> - **An unset `auto` inherits.** `auto=None` keeps whatever an earlier `configure_padding` layer or a composite parent set, and falls back to the built-in default (on) only when none did. `configure_padding(auto=False).configure_padding(left=5)` stays off, and a composite parent's `auto=False` reaches a leaf whose `auto` is unset.

> **Dated note (2026-10-09, #135: `padding.auto` keeps half the plot, never draws fewer labels, and judges each axis on its own; refines, in the 2026-10-08 note above, the 1 px floor in "An infeasible charge contributes nothing" and "A collapsed pass never covers", and supersedes nothing older):** The 2026-10-08 rescue's only floor was the 1 px minimum panel, so it could trade almost the whole data area for an edge label or a title: on a 220 px wide chart with 1e9 y labels, a 26-character x title was recentred on an 8.565 px plot and a 27-character one on a 1.469 px plot.
>
> - **A usable plot keeps half of `auto=False`'s.** `auto` may take at most half of the `auto=False` plot width (both x sides together) and half of its height (both y sides together), per panel. The floors and the per-axis budgets are measured once per chart from the `auto=False` layout, so they are exact on facets and beside a capped legend. A pass is usable only when every panel is above the 1 px minimum and at least its floor in both dimensions. The same floor is read by the collapse test, the per-label room in the infeasible-charge check, title feasibility (a title whose recentring needs more than its axis's budget is infeasible, and its axis contributes neither a title nor a tick extra) and the cap. Those two titles now render exactly as with `auto=False` and clip. With `auto=False`, nothing changes: the 1 px minimum panel stays its only floor.
> - **Over half adds nothing.** A single label or title whose rescue exceeds the budget adds nothing on that side. Rescues on one axis that each fit but together exceed its budget are not scaled down to a partial rescue: that axis is closed (its budget becomes 0) and `auto` settles again, so the axis adds nothing and lays out as with `auto=False`.
> - **Labels an x axis loses.** Replaced on 2026-10-11: the tick-by-tick "never fewer labels" rule this bullet stated is superseded by the own-ticks rule in the 2026-10-11 note below.
> - **Each axis is judged on its own.** When no pass is usable, each open axis is judged by the settle with the other axis closed. An axis whose rescue alone collapses a panel, leaves under half the plot or culls an `auto=False` label is closed, and `auto` settles again, so the other axis's valid rescue still ships. The rule is the same for x and y. When both axes' rescues are unusable, or each is valid alone but together they collapse the plot or cull a label, `auto` adds nothing and the chart lays out exactly as with `auto=False`. Closing is bounded: a closed axis cannot close again, so there are at most three settles, and the loop stops when no still-open axis was closed.
> - **A pass covers by what it ships.** A pass covers when, on every side, the tick extra it ships (after infeasible-axis zeroing, explicit-side zeroing and the cap) is at least the tick extra the probe at that pass needs, solved by the same rules against the pass's chrome credit and measured overhang. An explicit side and an infeasible axis are covered trivially, and a capped side is judged at its capped size. Explicit sides are zeroed before the title pass, so a tick extra an explicit side discards never moves the plot centre a title is recentred on.
> - **The x label band is the drawn band.** The band is reserved from the same label layout the axis draws (tick pixels, per-tick budget, cascade or explicit angle), at the width the axis is drawn at: after the y and secondary-y bands, at the facet cell width, inside a composite's imposed region, and, for independent-axis facets, from each panel's own tick inputs (domain, inset, format), which one owner resolves for both the band reservation and the drawn per-panel axes. Rotated labels reserve the widest drawn label at its drawn angle; flat or wrapped labels reserve the standoff plus one step of 1.2 times the font size per further line, the pitch the renderer draws. Under `CoordFixed`, the band is a bounded fixed point at the panel the aspect leaves, with a starting band that keeps the panel drawable. The x title is placed from the same band, so it clears the last drawn label line by the axis title padding, on top and bottom axes, and a top axis stacks its wrapped lines away from the plot.
> - **The band has a height budget.** When the band the cascade would draw leaves a panel at or below the 1 px minimum, the cascade culls labels by stride and shrinks until the band fits; when no stride fits, the axis draws tick marks only. The budget overrides `cull_labels=False` when the alternative is a blank plot, and `label_overlap="true"` never triggers it. It guarantees a plot above the 1 px minimum, not a roomy one.
> - **A non-finite viewport is refused.** `compute_layout` and every render refuse a non-finite width or height, with `auto` on or off: `properties(width=float("nan"))` raises `ValueError: invalid viewport: width=NaN, height=200 (both must be finite and > 0)`, where it used to hang with `auto` on.

> **Dated note (2026-10-11, #135 close: a rescue is judged by how many of an x axis's own labels it loses; supersedes the 2026-10-09 note's "Never fewer labels" bullet, its "Those two titles now render exactly as with `auto=False` and clip" sentence, and "culls an `auto=False` label" / "cull a label" in its "Each axis is judged on its own" bullet):**
>
> - **The own-ticks rule (user ruling 2026-10-11).** An x axis loses a tick when the tick has a generated label and the axis does not draw it in full: it is culled, or cut short with an ellipsis. A pass is unusable when it lays out a different number of x axes than `auto=False`, or when any x axis loses more of its own ticks than that axis loses at `auto=False`. Which ticks are labelled does not matter, only how many are lost, so a categorical rescue that labels a different subset of the same size ships: on a 240x160 chart with six 17-character categories and a 40-character x title, `auto=False` labels categories 0 and 4 on a 161.47 px plot, and the default labels categories 0 and 5 on a 137.05 px plot with the title inside the canvas. An axis with hidden labels loses nothing.
> - **An even thinning of a continuous axis is not a loss (user ruling 2026-10-11).** On a continuous x axis, a pass whose drawn labels are exactly the ticks at `i % s == 0` for one stride `s` (counted over the ticks with generated labels), with none cut short and at least three drawn, is usable even when it loses more than `auto=False`. Thinning to two labels, an irregular cull, an ellipsis and tick marks only remain losses. On a 220 px wide chart with 1e9 y labels, 26-, 27- and 28-character x titles are now rescued on plots 118.098 px, 110.798 px and 103.498 px wide (the `auto=False` plot is 137.946 px), each labelling `0`, `0.3`, `0.6` and `0.9` of its 11 ticks flat, where `auto=False` draws its 6 ticks shrunk to 9.02 px. On a 160x200 chart with a 19-character x title, `auto=False` draws `0`, `0.3`, `0.6` and `0.9` rotated on an 81.056 px plot and the default draws `0`, `0.4` and `0.8` flat on a 78.725 px plot. On a 176x100 chart with two `,.4f` ticks, a rescue would lose a label `auto=False` draws, so the chart ships as `auto=False`, both labels drawn.
> - **A pass that loses a label is still declined.** On a 124x150 chart with one `,.4f` label at -45 degrees, the rescue would draw none, and the chart ships as `auto=False`. Apart from an even thinning of a continuous axis, `auto` never loses more of an axis's labels than `auto=False` does.
---

### 3.8 Coordinate Systems

Applied via `.coord()` on `Chart`.

| Class | Description | Key Parameters |
|---|---|---|
| `CoordCartesian(xlim=None, ylim=None, expand=True, clip=True)` | Default Cartesian | |
| `CoordFlip()` | Flipped Cartesian (swap X/Y roles) | |
| `CoordPolar(theta="x", radius="y", *, start_angle=0, end_angle=None, direction="clockwise")` | Polar (pie, radar) | |
| `CoordGeo(projection=None, *, center=None, scale=None, translate=None, rotate=None, precision=None, clip_angle=None, clip_extent=None)` | Geographic | Any D3 projection name |
| `CoordFixed(ratio=1.0, *, xlim=None, ylim=None, expand=True, clip=True)` | Fixed aspect ratio. `ratio` = y units per x unit; default 1.0 enforces equal scaling on both axes. Required for geographic projections, correlation matrices, and any chart where axis unit equality is semantically meaningful. | `ratio` |

> **Dated note (2026-09-06, batch E task 6 — F-L02-09 + verified arc-error sibling, `mark_geoshape` and `mark_arc` refuse a wrong coord instead of rendering wrong or empty):** `mark_geoshape` with a `__geometry__` column present under any coord other than `CoordGeo` (the default `CoordCartesian`, or an explicitly set `CoordFlip`/`CoordPolar`/`CoordFixed`), and `mark_arc` under any coord other than `CoordPolar`, both now raise `ValueError` (`MarkRequiresCoord`, mirroring the mold `UnsupportedChannelCombination` already uses) naming the mark and the `.coord()` call that fixes it — e.g. `"mark_arc requires a Polar coord; add .coord(fm.CoordPolar(...))"`. Neither mark previously refused: `mark_arc` under a non-Polar coord failed later with a generic missing-channel error that did not name the real cause, and `mark_geoshape` with geometry under a non-Geo coord rendered a silently empty panel — axis furniture only, no `<path>`/`<circle>`/`<polyline>` geometry (`geoshape.rs`'s non-Geo arm fell through to `MarkBuildResult::empty`). `mark_geoshape` with **no** geometry column present is unaffected under any coord (stays an empty scene, as documented). `mark_image` is explicitly untouched by this guard — it keeps its own documented behavior of an empty scene under a non-Cartesian coord. A chart already using the correct coord for its mark renders byte-identically. Interactive-only, disclosed: `fm.layer(cartesian_chart, polar_arc_chart).interactive()` now refuses with this same `MarkRequiresCoord` error instead of silently rendering the arc layer as if it were Cartesian — the merged-flat interactive lowering drops a secondary layer's own `coord` (already warned at `composition.py:3303`: "secondary layer's theme/facet/coord is ignored"), so by the time the arc mark resolves it genuinely has no Polar coord to satisfy its own guard. **Static** `to_svg()`/`to_png()` on the identical mixed-coord layered chart is unaffected — the composite overlay tree renders each layer against its own coord and was already correct; verified live (`fm.layer(line_chart, arc_chart.coord(fm.CoordPolar(theta="x"))).to_svg()` renders, `.interactive()` on the same chart raises). The coord-drop itself (rather than the refusal) is a filed follow-up candidate, not fixed by this batch.

> **Dated note (2026-09-19, wave 5 T13 — GH #172(1), a layer carries its own coordinate system):** the note above is superseded in two respects, and the coord-drop it filed as a follow-up is closed. (1) **A layer carries a coord.** `Layer` gained an additive `coord` wire slot (skip-if-absent, so a spec declaring none serializes byte-identically), and `Chart + Chart` — hence `LayerChart`, `.interactive()`, and `resolve={"y": "independent"}`, which all merge through it — carries each secondary member's own coordinate system onto the layers it contributes instead of dropping it. `fm.layer(bar, arc_chart.coord(fm.CoordPolar(theta="x")))` therefore renders the arc member's wedges on **both** output kinds rather than refusing on the interactive one, and `fm.layer(bar, point_chart.coord(fm.CoordFlip()))` renders the point member **flipped** rather than silently unflipped — the half of the defect that produced no diagnostic at all. A layer's coord places that layer's marks; the panel's chrome (Cartesian axes, the polar boundary), its aspect ratio and its scale domains stay the chart-level coord's, which matches what the static composite path draws for the same composition. A per-layer coord that declares a PANEL property instead of a placement — `xlim=`/`ylim=`, `expand=False`, `clip=False`, or `CoordFixed(ratio=)` — is refused with a typed `ValueError` naming the layer, the coord and the property, because the layers share one panel and one of them cannot have its own domain or aspect ratio. Placement-only coords (`CoordFlip`, `CoordPolar`, `CoordGeo`, a bare `CoordCartesian()`) are honored. The `Chart + Chart` merge warning accordingly no longer names `coord`: it now reads "secondary layer's theme/facet is ignored", since the coord is no longer ignored. (2) **The refusal messages no longer prescribe a fix.** `"mark_arc requires a Polar coord; add .coord(fm.CoordPolar(...))"` became `"mark_arc requires a Polar coord"` (and the `mark_geoshape`/`CoordGeo` twin with it). The prescription was false on exactly the path that mattered: the merged-flat layer reached it with the prescribed call already written and dropped. What still refuses is a chart with no such coord anywhere, for which the state is the whole diagnosis.
>
> **Dated note (2026-09-20, wave 5 closing review — GH #172(1), corrected scope on a CATEGORICAL axis):** the note above states the contract for continuous positional axes and **overclaimed** for ordinal/band ones, where a member's own coord was still silently dropping its marks. The cause: a band ladder is built by a different mechanism than a numeric extent (`positional_category_union`, not `numeric_domain_union`), and only the numeric one read each layer's OWN effective coord — the ordinal one read the CHART's for every layer. The corrected contract, normative, is **honored where the panel can hold the member, refused loudly where it cannot, on both output kinds, and never a silently different chart**:
>
> 1. **Honored.** A member's post-flip positional channel now widens the axis it actually occupies, whatever that axis's kind. Measured before: `bar(x="cat:N", y="val") + point(x="val2", y="cat2:N").coord(CoordFlip())` across two frames enumerated the member's VALUE column as three phantom bands (`['a','b','c','40','50','60']`) and drew 0 of its 3 marks on all three merged-flat routes; it now enumerates `['a','b','c','x','y','z']` and draws all 3.
> 2. **Refused.** A coord is unplaceable in two ways, and both raise ONE typed `ValueError` in one voice, naming the layer and the coord: it declares a **panel** property (the note above), or its **channel exchange lands a column on an axis that cannot draw it** — `layer 1: fm.CoordFlip leaves this layer's positional channels in the opposite space from the panel's axes, and the panel's y axis is continuous while the column that lands on it ('cat': Utf8) cannot be drawn against a continuous axis`. The second is `fm.layer(bar, point.coord(CoordFlip()))` over a shared `x="cat:N", y="v:Q"`: the exchange asks the panel's quantitative y to hold a categorical column, and one merged panel cannot hold both. Honoring it would mean resolving that layer through SWAPPED scale slots, i.e. a second scale-resolution mechanism for one feature and a change to the scope stated above; refusing keeps one mechanism and one answer. Before this it rendered with the member's marks simply absent, no error and no warning, while the identical caller mistake spelled `fm.CoordFixed` refused loudly.
> 3. **Scoped per reason, to the routes that genuinely cannot honor it.** The panel-property reason refuses on all three merged-flat routes; the channel-exchange reason refuses on the two that put every member on one shared axis pair (static `chart + chart`, `.interactive()`) and RENDERS on `resolve={"y": "independent"}`, whose member resolves its own y slot from its own post-flip column, and on the static shared-y route, whose members are separate composite leaves. Both were verified to draw every mark. The question the gate asks is per column and per axis, not "a flip plus an ordinal axis": the same composition over an INTEGER category column is placeable and renders, because a continuous axis can read it.

> **Dated note (2026-09-19, wave 5 — GH #95, #113, #172(2), #172(3): what travels through `CoordFlip`):** four flip-fidelity defects close together. All four move rendered output on the shapes they reach; every unflipped chart that does not use `mark_density(orientation="horizontal")` is byte-identical.
>
> 1. **A sort shorthand names the channel you wrote it on (#95).** `sort="-y"` ordered a bar chart's categories correctly until `.coord(fm.CoordFlip())` was added, at which point it aggregated the *category* column, warned twice, and fell back to insertion order; the spelling that worked under flip was `sort="-x"`, for a value the caller had authored on `y`. The channel token in a sort shorthand is the caller's own text and is now resolved in **authored space**, so the shorthand and the `sort=fm.SortField(...)` object form — already flip-invariant, since it names a field rather than a channel — agree under flip as they always did without it. The same two-spaces defect sat in the `shape`, `stroke_dash` and `color` categorical domains, which build their sort the same way, and is fixed with it. Separately, an *ignored* sort spec now warns exactly **once** instead of twice (the provisional placeholder pass and the panel-real pass each reported it); a chart whose data is empty after transforms still gets its one warning.
> 2. **A flipped chart's marks and its axis read one scale (#113).** Under `CoordFlip`, scene construction merged the **authored** chart-level encoding into the post-flip layer-0 encoding, so a channel the flip had already moved (`y2` → `x2`) survived in its pre-flip slot and widened the wrong axis. Measured on a flipped `mark_bar` with a chart-level `y2`: the marks were placed on a y scale spanning roughly `0..72.9` while the axis ticks in the same SVG read `0..3`, a 23-fold divergence inside one file. The chart-level encoding is now moved into post-flip space before it is merged.
> 3. **A series closes its fill on its own baseline (#172(2)).** A `CoordFlip`'d `mark_area` traced its vertices correctly and then closed the filled region on the panel **bottom** — byte-for-byte the closure an unflipped chart emits — so the filled region came out very nearly the complement of the correct one. The same hard-code is reached with **no flip at all** by `mark_density(orientation="horizontal")`, whose near-zero tails degenerate the spurious region to a sliver, which is how it sat latent inside a blessed golden. The value axis is now **declared** rather than inferred: `mark_style` carries an additive, skip-if-none `value_axis` slot (the enum GH #77 added for this purpose), `desugar_density` stamps it, and the three signals compose by **precedence** — a declaration answers outright, and `CoordFlip` plus the stack stamp describe a transformation of a series that was vertical by default and speak only in its absence. As a disjunction a producer-horizontal density under `CoordFlip` stayed "horizontal" on the flip's word alone; the two flips must cancel. A first attempt inferred horizontality from traced geometry and misclassified a vertical area whose rows arrive unsorted (ferrum traces in row order and does not sort); that inference (`traced_edge_axis`) is deleted rather than kept beside the declaration.
> 4. **A flipped ribbon draws (#172(3)).** A `CoordFlip`'d `mark_ribbon` rendered **nothing**, silently: the flip had moved the authored `y2` into `x2` and the drawer read only `y2`. `mark_errorband`, `mark_smooth(ci=)`, the learning and validation curves and the QQ CI region all went band-less with it. The band's curve axis is now a function of the channel actually read. `tests/test_phase_9_e2e/goldens/jointplot_kde_marginals.svg` moved for this: its right marginal now closes flat on its own left edge.
>
> **Dated note (2026-09-20, wave 5 closing review — `CoordFlip` composed with an explicit `Stack(value_axis=)`):** the precedence rule in item 3 above governs `mark_style.value_axis`, whose copy `prepare` swaps into post-flip space with the channels — so by the time a mark reads it, the flip has been spent and `Some(Y)` means "vertical as rendered". `PositionAdjust::Stack::value_axis` is the same kind of declaration in the OPPOSITE space: nothing swaps a layer's `position`, so it arrives as AUTHORED, and it composes with the flip as an exchange rather than a precedence. A declared value axis therefore **moves with the flip**: `Stack(value_axis="x")` under `CoordFlip` puts the value back on y, and `Stack(value_axis="y")` under `CoordFlip` puts it on x, exactly as the undeclared default (vertical) does. Copying item 3's precedence onto the authored field spent the flip zero times, which is what made `mark_histogram(bins=3, orientation="horizontal", multiple="stack").coord(fm.CoordFlip())` draw nine rects — four sharing one `y`/`height` with overlapping widths, the bin column cumulated and the count column used as the bin key — where both its unflipped spelling and its vertical twin drew the correct three. It now matches its vertical twin pixel for pixel, and a caller-written `Stack(value_axis="y")` under `CoordFlip` renders where it used to raise `Stack: x must be a numeric dtype ...; got Utf8`. Unflipped charts and every chart with no explicit `value_axis` are byte-identical; the reconciliation lives at the one triad all three consumers (the draw-time cumulation and the two domain-widening previews) already funnel through, so they cannot disagree about which axis holds the value.

---

### 3.9 Faceting

#### `FacetSpec`

Wraps a `Chart` and repeats it across values of a facet field. Created by `.facet()` on `Chart`.

```
FacetSpec(spec, facet, *, ncols=None, nrows=None, spacing=None, bounds="full",
          columns=None, resolve=None)
```

#### `RepeatSpec`

Repeat a chart across a list of fields (e.g., pairwise matrix).

```
RepeatSpec(spec, *, row=None, column=None, layer=None, spacing=None, columns=None)
```

#### Resolution

```
Resolve(scale=None, legend=None)
```

`scale` / `legend` each accept a dict mapping channel name to `"shared"` or
`"independent"`. `scale` spans `"x"`, `"y"`, `"color"`, `"size"`; `legend`
spans `"color"` and `"size"` only (the channels whose scales can be shared).

> **2026-07-12 (#16):** the `legend` resolution axis is implemented: legend
> resolution **defaults to following scale resolution** — a composite whose
> color/size scale resolves shared renders **one figure-level legend**
> outside the panel grid (per-panel legends suppressed), and
> `legend={"color": "independent"}` opts back into per-panel legends.
> `legend={ch: "shared"}` over a non-shared scale raises `ValueError` at
> lowering. `pairplot(hue=)` and `jointplot(hue=)` get the shared legend by
> default. A plain dict passed to `resolve=` keeps meaning scale resolution
> (back-compat). The originally-declared third axis, `axis=`, is **not
> implemented** and has been narrowed out of this signature; shared axis
> rendering across composite panels is tracked as its own follow-up issue.

> **2026-07-12 (#74):** nested-composite resolve is now a defined rule, not
> emergent behavior. For **`color`/`size`**: a composite node whose
> *effective* mode for the channel is `"shared"` unions the domain (and, per
> the #16 legend band, captures the figure legend) across its **entire leaf
> span** — every descendant leaf, through nested composites and spliced
> overlay subtrees — not just leaves at matching grid positions. A node's
> effective mode is its own explicit `resolve=`/`Resolve(scale=...)` setting
> for that channel when given; an unset node **inherits the nearest
> ancestor's effective mode**. An explicit `"independent"` child opts its
> whole subtree out and resets the chain, so a re-shared node beneath it
> starts its own, separate union. The figure legend band attaches at the
> outermost node whose effective mode is `"shared"`; leaves it covers get
> per-panel suppression, so all three nesting shapes (outer-shared subtree,
> nested-and-outer both shared, a layered chart spliced into a shared
> concat) render exactly **one** legend. **`x`/`y`** are unaffected: they
> keep the pre-existing positional tree-path pairing across congruent direct
> children (grids pair by column — pairplot/compare) with no inheritance.
>
> **`configure_legend(orient="none")`** now suppresses legends through the
> same disabled-legend mechanism `Color(legend=None)` uses: per-panel
> legends, gutter reservations, and the figure-level legend band are all
> cleared. This applies both to a single chart (previously a silent no-op —
> the value parsed but nothing suppressed) and to a composite, where it
> disables every legend the node covers; per the existing all-disabled rule,
> if every participating leaf ends up disabled, no figure legend band is
> emitted. `orient="bottom"` (and every other valid orient) is unaffected.

> **Dated note (2026-10-11, #135 close: a facet panel draws the axes the layout lays out for it):** the scene no longer rebuilds an independent panel's axis on its own; every panel draws the x and y axes the layout lays out for that panel, shared or independent. Four visible consequences: (1) on independent facets, as on shared ones, the x title is drawn only under the bottom row (an upper panel with no panel below it in a short wrap draws its x axis but no x title) and the y title only beside the left column, so a 2x2 facet with either axis independent draws 2 of each where it drew 4; (2) `Chart.axis(x=False)` / `axis(y=False)` hide that axis on every panel of an independent facet (before, a 2x2 facet still drew 28 x tick labels and 4 x titles, or 64 y tick labels and 4 y titles); (3) an explicit `BandScale(range=...)` / `PointScale(range=...)` axis is re-anchored by each panel's offset from the first panel, by its range endpoints, so its ticks and labels sit exactly on that panel's marks on shared axes, under `CoordFlip`, and on a secondary y axis (before, panels after the first drew them at the first panel's pixels); (4) under an explicit range, each entry of a raw scale dict's domain gets its own slot, so a repeated category (accepted only at the dict door) is labelled in its own, empty slot rather than on its first occurrence's bar. A collapsed panel draws no axis, as before.

---

### 3.10 Selections (Interactivity)

Selections define interactive state. They are declared in the spec and resolved by the WASM renderer. In SVG/PNG mode, selections are silently ignored.

| Class | Description | Key Parameters |
|---|---|---|
| `selection_point(*, fields=None, encodings=None, nearest=False, toggle="event.shiftKey", on="click", clear="mouseout", resolve="global")` | Single or multi-point selection | |
| `selection_interval(*, fields=None, encodings=None, translate=True, zoom=True, mark=None, resolve="global")` | Brush / rectangular interval selection | |
| `selection_single(...)` | Alias for `selection_point` with toggle disabled | |
| `selection_multi(...)` | Alias for `selection_point` with shift-toggle enabled | |
| `SelectionMark(fill=None, stroke=None, fill_opacity=None, stroke_opacity=None, stroke_width=None, stroke_dash=None)` | Style the brush rectangle | |

**Using selections in encodings:**

There are two working forms, both driven through the same conditional
pipeline. `encode(<channel>=<ConditionalSpec>)` is the more common one; the
wire channel is taken from the encode key itself:

```
chart.encode(color=selection.when(Color("category")).otherwise(value("#aaa")))
```

The second form embeds the conditional directly on the channel object, for
when the base field encoding needs its own kwargs (`type`, `scale`, `title`,
...) alongside the conditional — `encode()` translates it into the chart's
conditionals list the same way, stamping the channel from the encode key and
keeping the base field encoding:

```
chart.encode(color=Color("category", type="N", condition=selection.when(Color("category")).otherwise(value("#aaa"))))
```

A pre-built `ConditionalSpec` can also be applied without going through
`encode()` at all, via `Chart.conditional(spec)` — the chart-level sugar for
a spec already carrying its own selection.

> **2026-06-01, superseded 2026-09-11 (batch H, F-L13-03/F-L15-01):** a
> conditional built by `sel.when(...).otherwise(...)` or
> `fm.when(sel).then(...).otherwise(...)` may be passed directly to
> `encode(<channel>=cond)` — the first working form above. The wire channel
> is taken from the encode key (so `encode(opacity=fm.when(sel).then(1.0)
> .otherwise(0.2))` resolves the numeric branches as opacity, not colour),
> and the conditional's source selection is auto-registered — no separate
> `.add_selection()` is required. `Chart.conditional(spec)` likewise
> auto-registers the spec's source selection when it carries one. A bare
> number with no channel context (e.g. via `fm.value(0.5)` outside an
> `encode` key) defaults to the `opacity` channel.
>
> Batch H closed the gap this note left open: the second working form —
> `ChannelClass(field=..., condition=<ConditionalSpec>)` — previously
> serialized into the per-channel `EncodingSpec.condition` field, a nested
> blob the wire accepted but no renderer consumed. `encode()` now intercepts
> it the same way: the conditional is appended to the chart's conditionals
> list (channel stamped from the encode key, selection auto-registered), and
> the channel itself encodes normally minus the `condition` kwarg. A
> `condition=` value that is not a `ConditionalSpec` raises `TypeError` at
> encode time, naming both working forms. **Which channels a conditional may
> target — and the typed `ValueError` for the ones it may not — is stated
> once under §3.2's appearance table (dated note, 2026-09-11); all FOUR
> entry points here (`encode(<channel>=cond)`,
> `<Channel>(field, condition=cond)`, `Chart.conditional(spec)`, and
> `Chart.layer(Layer(encoding={<channel>: cond}))` — batch H close cycle
> 3/4, H3-1, see §3.2's fourth-entry-point note below) resolve through that
> one vocabulary.** (This supersedes the stale example
> and the `.conditional(sel, color=..., else_color=...)` sentence that used
> to sit above this note — `Chart.conditional()` never took per-channel
> kwargs; it takes a single pre-built `spec`.)
>
> Separately, `sel.when(...)`/`.otherwise(...)` branch values now accept a
> bare `str` as field shorthand (parsed with the same grammar `encode()`
> uses) — a bare string means a **field** everywhere in ferrum; a literal
> still requires `fm.value(...)`. This is what makes
> `brush.when("category:N").otherwise(fm.value("lightgray"))` (the Altair
> migration guide's example) work as printed.

> **2026-09-02 (batch A):** selection styling (`SelectionMark(fill=...)` and
> `fm.value(<string>)` used as a color) now routes color strings through the
> same 148-name/hex/`rgb()` parser as mark construction (see §3.3's color
> vocabulary note) — a parseable string resolves to a color dict, a number
> still resolves to opacity, and an unparseable string raises `ValueError`
> naming the accepted forms; the former silent opacity-`1.0` fallback for a
> bad string is removed. Selection styling deliberately **diverges** from
> the mark boundary on the two clearing spellings (`"none"`/`"transparent"`):
> the selection wire dict has no cleared-paint representation, so both are
> refused with a message naming the reason ("selection styling cannot
> express '`none`' (no paint); provide a color") rather than the generic
> accepted-forms text. A wire representation for a cleared-paint selection
> is a logged follow-up, not silently absent. A non-string, non-number
> literal passed to `fm.value(...)` now raises `TypeError` (matches the
> sibling terminal raise) instead of silently falling through.

> **Dated note (2026-09-11, batch H task 7 — F-L13-01, `Parameter.ref()` is now consumed at all three documented sites):** `fm.param(...).ref()` — a `{"param": <name>}` marker distinct from passing the `Parameter` object itself — reaches real behavior at every site the docstring names, through one shared decoder. At a scale `domain=` (§3.6), a ref-shaped domain is now treated identically to a raw `Parameter` domain (emits `domainParam` by name); the parameter still needs `add_params()` to reach the wire, since a `.ref()`'d domain carries only the name, not the auto-registration a raw `Parameter` domain gets. At `transform_filter(...)` (§3.5), a ref no longer silently mangles into the garbage predicate `datum.param == "<name>"` (a dict with a `"param"` key previously matched the constraint-dict branch before reaching the parameter check) — `transform_filter(q.ref())` now equals `transform_filter(q)` exactly. At a conditional value site (`sel.when(q.ref())`/`.otherwise(q.ref())`, above), resolution needs the chart's registered parameters, not just a name: `Chart.to_spec()` now builds and threads that registry into conditional serialization, so a real chart resolves a ref to the named `fm.param()`'s current static value, raises `ValueError` naming the parameter when it isn't a registered value-bearing `fm.param()` (e.g. a `Selection` ref, or an unregistered name), and raises when the parameter is registered but carries no `value`. Reactive (WASM-side) re-resolution of a conditional ref when the underlying parameter changes at runtime is a logged follow-up — this batch ships static resolution only, matching every other conditional branch's semantics.

> **Dated note (2026-09-23, wave 6B, `#128` — a conditional literal is typed by its channel):** the batch-A note above resolved a `fm.value(<string>)` branch as a color before looking at the channel, so `encode(size=sel.when(fm.value("red")).otherwise(fm.value(30)))` serialized a `kind=color` value under the `size` channel and the interactive runtime silently repainted the mark's fill on selection. The same held on every non-color conditional channel. One channel-keyed literal table now decides, over the conditional channels listed in §3.2 (plus the `fill` alias): `color` parses a color string (or a number, unchanged); `stroke_dash` takes a dash pattern through the same normalizer as `mark_line(stroke_dash=)` (a named linetype, a comma string, a number or a list), so `fm.value("dashed")`, `fm.value("4,2")` and `fm.value([4, 2])` all serialize `{"kind": "stroke_dash", "value": [4.0, 2.0]}` where they were refused with a color message or a `TypeError`; `size`, `opacity`, `stroke_width`, `stroke_opacity`, `fill_opacity` and `angle` take a finite number and nothing else, so a string there (including a numeric string such as `"0.5"`) is refused. The rule is the same for all three spellings (`fm.value`, `fm.when(...).then(...)`, `fm.param(...).ref()`). Every refusal is a `ValueError` of one shape, naming the call and the channel: `fm.value('red') for channel='size' needs a number, got str: 'red'`. This replaces the batch-A note's `TypeError` for a non-string, non-number literal, and `bool` is now refused on every channel, `color` included. `fm.value` accepts a list or tuple. `color` conditionals, and `to_spec_dict()` with no channel, are byte-identical on the wire. A dash pattern outside the interactive GPU palette is accepted and draws solid interactively, like every other non-palette dash (user ruling Q8).
>
> **Amended 2026-09-24 (wave 6B close):** a literal is typed by its channel on `color` too: a number on a `color` conditional is refused (`fm.value(0.2) for channel='color' needs a color string, got float: 0.2`) where it was re-typed as an opacity value. That fallback existed because `Chart.conditional(sel.when(a).otherwise(b))`, which carries no channel, read the channel from `if_selected` only, so `sel.when(fm.value(20)).otherwise(fm.Size("v"))` serialized an opacity conditional. The channel is now read from both branches (an encoding-channel branch wins; otherwise the first literal names it, a number meaning opacity), checked against the conditional vocabulary, and stamped at `Chart.conditional()`, so a conditional on a channel no conditional can target (e.g. `fm.X`) is refused there instead of serializing as a color conditional. `nan`/`inf` are refused on every numeric conditional channel.

> **Dated note (2026-10-06, wave 7 close round 6 — a conditional reaches only the marks that declared it, and a selection reacts only to its own chart's panels; user rulings R2 and R2b, 2026-10-04):** three changes to interactive scope, each byte-identical where it cannot apply.
>
> - **Conditional scope, every channel.** A conditional of any channel restyles only the marks of the chart or `fm.Layer` that declared it. A chart-level conditional (a `.layer()` host's, `fm.SecondaryY`'s) still reaches every layer attached to it, as ruled 2026-10-03. Before, only `size` was scoped: a `color`, `opacity` or other conditional declared on one chart of `a | b`, or one layer of `a + b`, restyled every mark in every panel when its selection fired, the undeclared side's marks taking `if_selected`/`if_not` too. Linked views declare the conditional on each chart over one shared selection, which works under either rule. On the scene wire each `MarkBatch` lists the conditionals it answers to as `conditionals` (indices into the scene's list, skipped when empty), replacing the size-only `size_conditionals`; a scene with no conditionals is byte-identical. This generalizes the `Size` channel's 2026-10-03 (close round 4) note: a conditional of any channel, not only `size`, stays on the `+` operand that declared it.
> - **Selection input scope.** A selection reacts to clicks and brushes only in the panels of the chart that declared it: every facet panel of that chart, and an overlay group as one owner. Before, input on any panel drove every selection of its kind, so a click on `b` in `a | b` changed `a`'s selection. Each scene selection carries `panels`, its owner panel indices, omitted on a one-panel scene, whose selections are byte-identical. A selection declared on several charts under one name with one definition (linked views) is one scene selection owning the union of their panels; before, each chart appended its own copy.
> - **Redefinition refuses.** One name with two different wire definitions refuses, naming the differing fields: across a composition at render, `hconcat: selection 's' is declared twice with different definitions: chart 1 of hconcat declares nearest=False, chart 2 of hconcat declares nearest=True. One name is one selection across the composition: share one selection between the charts, or give each its own name.`; within one chart at the declaring call (`add_selection()`, `add_params()`, `encode()`, `Chart.layer()`, `Chart.conditional()`, `chart + chart`), `add_selection(): selection 's' is declared twice with different definitions: first declares type='point', second declares type='interval'. One name is one selection on this chart: pass the same Selection object to every call that declares 's', or give the second one its own name.` Before, within one chart or `+` the first definition silently won, and across a composition both definitions sat in the scene under one name. Definitions are compared by their wire form, so `fields=['x']` and `fields=('x',)` are one definition.

---

### 3.11 Annotations

Annotations are lightweight overlays that don't participate in scale domain calculation.

| Class | Description | Key Parameters |
|---|---|---|
| `annotate_hline(y, *, label=None, label_position="right", stroke=None, stroke_dash=None)` | Horizontal reference line | |
| `annotate_vline(x, *, label=None, stroke=None, stroke_dash=None)` | Vertical reference line | |
| `annotate_rect(x1, x2, y1, y2, *, fill=None, opacity=0.1, label=None)` | Shaded rectangle region | |
| `annotate_text(x, y, text, *, dx=0, dy=0, anchor=None, align=None, baseline="middle", font_size=None, color=None, angle=None)` | Free text annotation | |
| `annotate_arrow(x1, y1, x2, y2, *, label=None, label_side="start", stroke=None)` | Arrow with optional label | |
| `AUCLabel(*, position="end", format=".3f", prefix="AUC = ")` | Auto-placed AUC annotation on ROC curves | |
| `OutlierLabel(*, threshold=3.0, field=None, label_field=None, max_labels=10)` | Label high-leverage or high-residual points | |

> **2026-06-22 (COMP-06):** `annotate_text` now takes the canonical `anchor=`
> keyword in the SVG vocabulary (`"start"`/`"middle"`/`"end"`), matching
> [`annotation.text`][]. The former `align=` keyword (`"left"`/`"center"`/
> `"right"`) is retained as a non-breaking **deprecated alias**, mapped via
> `{left: start, center: middle, right: end}`. When neither is supplied the
> resolved anchor is `"middle"` (the same render as the historical
> `align="center"` default). Supplying **both** `anchor=` and `align=` raises
> `ValueError`. This reconciles the annotation-pair vocabulary drift: the
> `annotate_text` (mark-Chart) and `annotation.text` (dataclass) surfaces now
> share one anchor vocabulary.

---

### 3.12 Compound Views

> **Dated note (2026-09-19, wave 5 — GH #247B: a band exists only if something is drawn in it):** a layered chart built from two DataFrames grew a `null` band on its ordinal axis that no mark ever occupied — the merge's diagonal concat pads each frame's columns over the other frame's rows, and those manufactured nulls reached categorical domain resolution as if the caller had supplied them. Every real band was narrower for it, and the axis described a category the chart did not draw. The rule is now stated positively: **a null in a positional ordinal column claims the null-category band only when some null-category row carries a non-null value on the opposite positional channel.** Nullness only — a non-finite paired value (`NaN`/`±inf`) counts as occupancy, because the marks still resolve it to a pixel rather than skip the row; only a true null means the row has no value at all. The standing policy that a **genuine** null category carrying a real value gets its own band is intact and pinned in both directions, unlayered and inside a merge, so a null group and a real `0`/`false`/`""` group remain two categories (FA-9). Null-free columns are byte-identical.
>
> The fix is not scoped to the merge, which is where the design doc first placed it. A single **unmerged** chart over `cat=["a","b","c",None], val=[1.0,2.0,3.0,None]` renders the same phantom band and draws three marks, so a merge-scoped fix was provably incomplete — and no merge-side data-shape change could be complete either, since a merged batch must be at least as tall as its taller frame, so with unequal heights the shorter frame's columns necessarily carry entries its data never supplied and all three candidate fill values are wrong. **Disclosed:** any chart whose positional ordinal column carries unoccupied nulls — including every cross-DataFrame layered chart — loses that band, so its remaining bands widen and its axis labels shrink by one.
>
> **Per layer**, the null band is judged against the positional field **that layer itself binds**, and a layer binding none drops the null category rather than inheriting the "no opposite channel bound → keep the band" arm. Carrying that arm across the layer boundary would keep a band for an injected reference-line column, whose single authored value sits in structural nulls and whose only paired coordinate is *inherited*; measured, it draws four rule lines where one is correct. The narrower rule is the same principle applied honestly: a layer with one positional channel has no paired value, so nothing is drawn in that band. It is non-widening — such a layer draws no mark there today either.

> **Dated note (2026-09-19, wave 5 — the shared ordinal axis covers every layer's categories; user ruling):** `Chart({cat: [a, b, c]}) + Chart({cat: [x, y, z]})` drew **three marks for six rows**. A layered chart's ordinal axis read its categories off the chart-level column alone and never unioned its layers', so the second layer's marks were dropped for want of a band and its categories never even reached the axis labels. The numeric domain has had a cross-layer union for exactly this; the ordinal domain had no sibling, so the invariant "the shared axis covers every layer's extent" was half implemented. Every category any layer's positional column carries now gets a band, **ordered by first appearance in layer order** — not a new semantic, but the existing within-layer first-appearance rule carried across the layer boundary rather than replaced by sorting. The disjoint case renders six bands and six marks; overlapping layers contribute only what they add, and a layered chart whose layers share identical categories renders byte-identically (16 of 16 shapes, measured), which is the control proving this is a widening and not a rewrite. Ordering and scope were a user ruling, taken because the choice sets every band position and is visible in every render. **Disclosed: charts may now draw marks they previously dropped silently, and their band widths change accordingly.** Deliberately unchanged: the **explicit-scale** ordinal path still does not union layers, because the explicit numeric path has the identical property and fixing one side alone would break that symmetry.

> **Dated note (2026-09-19, wave 5 — GH #182 follow-on: which cells share an extent is declared, not inferred):** removing #182's phantom gutter (see §3.7) exposed a real defect it had been masking: **a composite's satellite panels were never aligned with the panel they annotate.** A `jointplot` marginal and a `clustermap` dendrogram are each their own leaf, laid out independently and fitted into a slot, so their plot rects and the centre panel's had no reason to agree; they *looked* aligned because the phantom inset them by roughly the centre's axis band. Measured post-#182, the `jointplot_kde_hist` top marginal spanned x `53.377..624.0` against the joint panel's `67.423..547.025`, and the `clustermap_basic` column dendrogram spanned `176.0..744.0` against the heatmap's `213.4..674.8`. The rule that closes it: **a panel that annotates another spans exactly that panel's extent on the axis the two share, and keeps its own on the other**, imposed at the same single point in layout that the #89A overlay pre-pass uses, and scaled by the follower's native size over the leader's because the grid planner ratio-fits each cell.
>
> The trigger is **declared on the wire**: `CompositeNode::Composite` carries `extent_alignments: Vec<{follower, leader, axis}>` (cell positions, additive, `skip_serializing_if` empty, byte-identical round-trip for a spec declaring none), emitted by the `JointChart` and `ClusterMapChart` lowerings. The shape predicate that recognized a jointplot by its grid geometry is **deleted**, not kept beside it: a clustermap is also a 2x2 grid with a hole, so the predicate separated two different figures by which corner was empty. A declaration naming a **nested composite** cell cannot be honored (such a cell has no single plot rect) and now warns (`RenderWarning::ExtentAlignmentUnresolvable`) naming the follower, leader and axis, instead of dropping silently. A follower that keeps a legend of its own is still left unaligned rather than drawn across it. **Named as pre-existing and NOT fixed here:** a dendrogram's own x scale places its leaves at the extremes of its plot rect (≈1.7% and 98.3%) while the heatmap places its columns at band centres (12.5% and 87.5%), so leaf-over-column is closer after this change (worst error 123px → 50px) but not exact. That is the dendrogram chart's scale, not the panel's extent, and it is filed as a separate fix. **All `jointplot` and `clustermap` output changes geometry**; five goldens moved for it (`clustermap_basic`, `clustermap_row_col_dendrograms`, `jointplot_kde_hist`, `jointplot_kde_marginals`, `composite_figure_title/joint_figure_title`), each regenerated and PNG-inspected against its pre-fix rasterization rather than judged alone.
>
> **Dated note (2026-10-04, wave 7 close): `clustermap` leaves sit on the cells they name.** The leaf-scale gap named above is closed, with two further defects. Each dendrogram's leaf axis is `LinearScale(domain=[-0.5, n-0.5])`, the heatmap band scale's continuous image, so leaf `k` lands on the centre of cell `k`. The row dendrogram's leaf and distance axes are both reversed, so under `CoordFlip` its leaf 0 sits beside the heatmap's top row and its leaves face the heatmap, as the column tree's do. The heatmap's columns follow the column clustering: the center chain is `Linkage(rows), Linkage(columns), Reorder(row order), Reorder(column order, axis="columns"), Unpivot(value_vars=None)` over the id and numeric value columns only, so the heatmap's column band order is the column tree's leaf order. **All `clustermap` output changes:** columns reorder, the row tree flips, and leaves move onto cell centres.

> **2026-07-05 (Phase B, composite render unification / #45):** every compound
> view in this section now renders through ONE Rust composite entry per output
> kind (`render_composite_svg` / `render_composite_interactive`): Python lowers
> the composition to a composite spec tree; Rust resolves `resolve=` sharing
> across the tree's leaves (x/y/color/size; congruent position-wise pairing),
> plans the layout (ratio cells via per-panel `LayoutScale`, holes for empty
> corner/trailing/empty-data cells), and emits one scene. The
> `compose_svg_*` helpers referenced by the historical notes below NO LONGER
> EXIST (removed from the public API); static and interactive output share the
> same layout, subsuming the former W4/W5 interactive limitations. Behavior
> change: an all-empty compound view (every child zero-row) raises a
> `ValueError` instead of rendering a blank grid. Historical notes below are
> retained for the pixel-semantics decisions they record (spacing, ratio math),
> which carried over into the composite layout pass.

> **Dated note (2026-09-11, wave-1 hardening, #140 — figure-level config warnings dedupe, per-leaf data warnings still fan out):** a composite whose N leaves are structural clones of the same chart-level config declaration (e.g. `Repeat`/facet applying one identical `ChartConfig` override to every panel) used to emit N identical `RenderWarning::ConfigSurfaceNotPresent` warnings for what is one user mistake made once — the composite's aggregated warning list now dedupes that one variant by `PartialEq`, so it survives exactly once regardless of leaf count. This is a narrow, class-scoped dedup, not a global one: every other warning class a leaf can emit — a leaf's own degrading transform, `GroupCardinalityDegenerate`, `MarkBuildEmpty`, the per-axis annotation-suppression rollups, and (deliberately, since it resolves per leaf's own `encoding.scheme`) `UnknownColorScheme` — still fans out per leaf unconditionally, because two leaves independently hitting the same-looking data condition are two independent facts, not one declaration replicated.
> a 2×2 layout (center + optional top/right marginals) backed by
> `ferrum._core.compose_svg_grid`; it supports `.theme()`, `.properties()`,
> `.save()`, `.show()`, and `_repr_svg_`. `RepeatChart` gains two new
> parameters: `diagonal=` (a separate template chart used for cells where
> `row_field == col_field` in symmetric n×n repeats) and `corner: bool`
> (filters the expanded grid to the lower triangle, including the
> diagonal). Per-cell field substitution is keyed by three typed
> sentinels — `Repeat.column`, `Repeat.row`, `Repeat.layer` — passed to
> the template's `.encode(...)`; string sentinels (e.g. `"repeat:column"`)
> are not supported. A new compound view `ClusterMapChart` is added:
> a 2×2 grid combining a clustered heatmap with row and/or column
> dendrograms; `dendrogram_ratio: float in (0, 1)` controls relative
> dendrogram-to-heatmap size. Most users construct `ClusterMapChart` via
> `ferrum.clustermap(...)`.

| Class / Operator | Description | Key Parameters |
|---|---|---|
| `HConcatChart(*charts)` / `chart1 \| chart2` | Horizontal concatenation | `spacing`, `resolve`, `title` |
| `VConcatChart(*charts)` / `chart1 & chart2` | Vertical concatenation | `spacing`, `resolve`, `title` |
| `LayerChart(*charts)` | Layer overlay (same axes) | `resolve`, `title` |
| `ConcatChart(*charts, columns=None)` | General wrapping concatenation | `spacing`, `resolve`, `columns` |
| `RepeatChart(spec, row=None, column=None, layer=None)` | Repeat across field lists | `spacing`, `columns`, `resolve` |

> **2026-07-11 (secondary y-axis, #52):** the `LayerChart(*charts)` row's
> "same axes" description is the default (`resolve={"y": "shared"}`, the
> byte-stable pre-#52 path) but is no longer the *only* behavior.
> `LayerChart(a, b, resolve={"y": "independent"})` now renders a real
> dual-axis chart: layer 0's y-scale drives the left axis and gridlines,
> and each subsequent independent layer resolves its own y-scale and
> gets its own right-side axis (stacked outward, unbounded layer count).
> One implementation serves both static SVG and interactive output (the
> merged flat single-panel path — see CLAUDE.md "Composite rendering").
> `x` stays shared across every layer regardless of `resolve`; per-layer
> independent x (dual-x) remains a typed `ValueError` naming GH #55. See
> `SecondaryY` below for the sugar built on top of this mechanism.

> **2026-08-27 (P2 chrome dedup; retired 2026-08-28 by residuals batch
> #89 — current behavior below, original gate preserved as amendment
> history):** static shared-`y` `LayerChart` output previously drew every
> axis line, tick label, grid line, and title once **per layer**,
> overprinting at identical coordinates (2 layers = 2x every chrome
> element; layers binding different y fields overprinted both axis
> titles). Dedup is now **total**. For an Overlay composite node whose
> direct children are all leaves, every leaf lays out against one shared
> plot rect — the intersection, per side, of the leaves' natural gutters
> (title/legend/axis bands) — computed in a composite planning pre-pass,
> so layout never needs the old post-layout `plot_area` overwrite.
> Suppression (clearing grid, axes, above-marks axis chrome, and scene
> title on non-primary leaves) is coupled to imposition: it applies to
> every leaf that actually laid out against the shared rect, which after
> the shared pre-pass is every leaf in a well-formed group. The three
> former refusal shapes — a non-primary leaf with its own legend, a
> `zindex >= 1` axis, or a `z="below_marks"` annotation — now render with
> one chrome and full content preserved (legend renders, annotation
> renders below marks, above-marks axis renders once); legends are never
> suppressed, so their gutters still participate in the shared rect on
> every leaf. Three cases still keep per-leaf chrome, never suppression
> without geometric alignment: (1) an Overlay node whose children are NOT
> uniformly direct leaves — e.g. one child is itself a nested composite —
> is structurally excluded from grouping up front, so every child keeps
> its own rect and chrome exactly as before this batch (unreachable from
> Python lowering: `LayerChart`, the sole producer of an Overlay node,
> rejects any non-leaf layer with a typed `ValueError` before a tree is
> built; the guard exists only because a directly-constructed wire spec
> could still express it); (2) a degenerate intersection — the leaves'
> combined gutters leave no common plot area — drops that group's
> suppression and emits `RenderWarning::OverlayGuttersDiverged` naming
> the layer count, so the resulting doubled chrome is diagnosable rather
> than silent; and (3) a member whose own layout fails during the
> imposition pre-pass also drops its group's suppression, with **no**
> warning — that leaf's render re-runs the identical layout moments later
> and raises the failure as a typed error at its canonical position, so a
> chrome-suppression warning here would be noise ahead of an outright
> render failure, not a silently-wrong chart. Interactive LayerChart
> (merged flat path) and independent-y are unchanged. See CLAUDE.md
> "Composite rendering" for the Rust-side mechanism.
>
> **Amendment history:** the original 2026-08-27 note described dedup as
> gated by a per-leaf safety check (`overlay_imposition_safe`): a
> non-primary layer carrying its own color/size legend, an axis drawn
> above marks (`zindex=1`), or a `z="below_marks"` text annotation kept
> its pre-fix per-layer chrome at its own rect rather than sharing the
> primary layer's, with the gate's removal tracked as a follow-up issue.
> Residuals batch #89 (2026-08-28) deleted that gate, along with
> `overlay_imposition_safe` and `chrome_suppressed` themselves — neither
> symbol exists in the codebase anymore — and the totality behavior
> described above replaced it.

> **2026-09-10 (batch F, F-L10-01 rider):** a layered chart (any compound
> view lowering to `spec.layers` -- `LayerChart`, catplot's `"point"`/
> `"bar"` kinds, and every composite-mark desugar) now refuses, rather
> than silently drops, a layer at index >= 1 whose POSITIONAL encoding
> (`x`/`y`/`x2`/`y2`) names a column absent from that layer's own resolved
> batch: `RenderError::LayerUnknownColumn` names the layer index, the
> channel, and the column. Previously only layer 0's encoding was
> validated before scale resolution; every subsequent layer's per-mark
> builder swallowed the missing column via `.ok()` and rendered nothing
> for that layer, with no error and no visible signal beyond an
> inexplicably absent series. Explicitly unaffected, by design: an
> INHERITED color channel absent from a layer's batch still renders
> (structural, owned by #106 -- see the chrome-dedup note above); a
> legitimately empty (zero-row) layer still skips rather than raising; a
> `data_source=` layer is checked against its OWN resolved batch, not
> layer 0's. The gate surfaced two pre-existing, now-fixed bugs it made
> visible for the first time: the discrimination-threshold desugar was
> rendering its optimum-label/threshold-line layers even when the
> f1-gated sentinel columns were never injected, and a secondary-y smooth
> test asserted a fit polyline against a column name the transform never
> actually emitted.

#### `JointChart`

Compound view with a center plot and optional marginal distribution plots sharing the center's x-axis (top) and y-axis (right). The marginals are independent `Chart` objects, typically `mark_histogram` or `mark_density`.

```
JointChart(center, *, top=None, right=None, ratio=5, spacing=10.0)
```

`center`: any `Chart`. `top` / `right`: marginal `Chart` objects; x-axis of `top` is shared with center x; y-axis of `right` is shared with center y. `ratio`: size ratio of center panel to each marginal. `spacing`: gap between panels in pixels.

> **2026-05-11 (Phase 9, P2.6):** `spacing` is **pixels**, not a fraction.
> The original spec described `spacing` as "a fraction of total size" but
> the Rust grid compositor has always treated the value as pixels —
> `spacing=0.02` rendered as zero gap (one-fiftieth of a pixel). The
> default for `JointChart` / `RepeatChart` / `ClusterMapChart` was
> bumped from `0.02` (effectively zero) to `10.0` to match
> `HConcatChart` / `VConcatChart`. Affected goldens were re-blessed.

> **2026-05-12 (Phase 9, F20 audit):** The grid compositor allocates
> slot dimensions as `K * ratio[i]`, where `K = min_i(native_dim[i] /
> ratio[i])`. The dominant row/column renders at its native size; smaller
> cells are stretched into their declared ratio via nested
> `<svg viewBox preserveAspectRatio="none">` wrappers. This is the
> algorithm that gives `JointChart(ratio=5)` its 5:1 center-to-marginal
> proportions and that drives the `row_ratios` / `col_ratios` parameters
> on the internal `compose_svg_grid` binding.
>
> **Non-uniform-stretch caveat:** the `preserveAspectRatio="none"`
> wrapper distorts glyphs when the slot's aspect ratio diverges from
> the cell's native aspect ratio. For `JointChart` this is harmless —
> the shared data axis stays aligned by construction; only the redundant
> count/density axis stretches. Compound views with intrinsic shape
> constraints (dendrograms, geographic projections, fixed-aspect
> coordinate systems) must pre-resize their cells so the slot ratios are
> already satisfied at native size. `ClusterMapChart` is the example
> here: dendrogram heights and widths are pre-computed to match the
> heatmap's row/column slots so `K_w` and `K_h` collapse to unity and
> no scaling fires.

> **2026-05-11 (Phase 9, P2.5):** `RepeatChart` now ships the previously
> dormant `columns`, `layer`, and `resolve` kwargs:
>
> - `columns: int` — wrap width for 1-D repeats (only `row=` or only
>   `column=` is set). Defaults to a single row for column-only repeats
>   and a single column for row-only repeats.
> - `layer: list[str]` — each cell stacks one resolved template copy
>   per layer field, combined via the `Chart +` overlay operator.
>   Diagonal cells (when `diagonal=` is set on a 2-D grid) skip layering.
> - `resolve: dict[str, "shared" | "independent"]` — per-channel scale
>   sharing. `"shared"` computes the union domain across every layer of
>   every cell and injects an explicit `scale=` dict so the participating
>   axes match. `"independent"` keeps per-cell domains (the default for
>   unlisted channels).
>
> Asymmetric `diagonal=` (set with `row != column`) is now a `ValueError`
> at `expand()` time — previously it emitted a `UserWarning` and silently
> dropped the diagonal template.

Most users construct `JointChart` via `ferrum.jointplot(...)`.

#### `SecondaryY`

```
SecondaryY(field, mark="line", axis=None, color=None, opacity=None, scale=None)
```

`chart + SecondaryY(...)` adds a second, independent y-axis to `chart`.
`field`: data field mapped to the secondary y axis (required). `mark`:
mark type for the secondary series (default `"line"`). `axis`: per-axis
`Axis` config applied to the right-side y2 axis. `color` / `opacity`:
literal mark styling for the secondary series. `scale`: `Scale` config for
the secondary y axis. The base chart must carry an `x` encoding for the
secondary layer to inherit; adding `SecondaryY` to a chart with no `x`
raises `ValueError`.

> **2026-07-11 (secondary y-axis, #52):** `chart + SecondaryY(...)`
> desugars, at `Chart.__add__` time, to one appended layer on `chart` —
> mark `mark`, `y` encoding on `field` (carrying `axis`/`scale`), `x`
> inherited from the base chart's own x encoding, color literal `color`,
> opacity `opacity` — flagged as an independent-y layer using the same
> per-layer y-scale-slot mechanism `resolve={"y": "independent"}` uses
> above. The base chart's own layer(s) are unchanged: a layered base
> chart keeps its existing layers sharing the left axis while only the
> appended `SecondaryY` layer gets its own right axis; adding multiple
> `SecondaryY` instances stacks multiple right axes outward, same as
> stacking multiple independent layers directly.
>
> This re-bases `SecondaryY` off its former standalone Rust mechanism
> (`SecondaryYSpec` / `StructuralSpec::SecondaryY`, which rendered in the
> static overlay path only and was inert in interactive output) — that
> Rust mechanism no longer exists in the crate. One secondary-axis
> mechanism remains. Behavior deltas from the pre-#52 renderer: (1) the
> secondary series' plot area now genuinely narrows to reserve a
> right-side margin band, rather than the axis overdrawing the plot; (2)
> the secondary axis is fully interactive (tooltips, zoom/pan,
> hit-testing) like any other layer; (3) `color`, when omitted, no longer
> falls back to a hardcoded `#E45756` — the secondary mark defaults like
> any other mark (theme default).

> **Dated note (2026-10-01, wave 7 — `#141`, a secondary y axis owns its gridlines, off unless asked):** Before this note a secondary axis drew no gridlines whatever was asked: `axis_y2=AxisConfig(grid=True)`, the secondary layer's own `axis=Axis(grid=True)`, and the `axis_y2.grid_color`/`grid_dash`/`grid_width`/`grid_opacity` styling were accepted and inert (the config manifest listed `axis_y2.grid` as unhonored), while the secondary's layout `show_grid` silently followed the theme's grid default. **Default off** (user ruling 2026-10-01): a secondary's show flag defaults to `false`, so every existing dual-axis chart renders byte-identically; only its layout JSON's `show_grid` reads `false` where it read `true`. **Show chain:** the secondary layer's per-channel `Axis(grid=)` > `axis_y2=AxisConfig(grid=)` > off. `Theme(grid=True)`, a shared `configure_axis(grid=)` and `configure_grid(y=)` never light it, since `axis_y2` never falls back to shared config. The chart-level `axis_y2` broadcasts to every stacked secondary, and a per-channel `Axis(grid=False)` overrides it for one. **Style chain:** per-channel `grid_*` on the secondary layer > `axis_y2.grid_*` > theme major (which carries a flat `configure_grid(color=, width=, dash=, opacity=)`); style without show draws nothing. When lit, horizontal gridlines draw across the plot at that secondary's own tick positions, after the primary's, per panel and per lit secondary; band colors stay primary-only; minors draw only when `Theme(grid_minor=True)` and the secondary's own show are both on; a primary-y `break_axis` never remaps them; they are flat top-level lines, fixed under interactive zoom like the primary's. There is no `configure_grid(y2=)` slot (user ruling 2026-10-01): right-axis gridlines are controlled through `axis_y2` and the layer's own `Axis` only.

All compound views accept `.theme()`, `.properties()`, `.save()`, `.show()`.

#### `BreakAxis`

```
BreakAxis(axis, gap, break_size=12, break_style="slash")
```

`chart + BreakAxis(...)` removes one or more ranges from a continuous positional scale, so a distant outlier cannot compress the rest of the data into an unreadable band, and draws a break indicator where each removed range was.

**This block is the current contract, consolidated 2026-09-13.** Everything under it, from the 2026-09-11 batch-H note down, is the historical record of how each piece of this contract was reached: several of those notes retract or supersede claims in earlier ones, and they are kept verbatim for the decisions and measured symptoms they record. Where a note and this block disagree, **this block is current**.

**Parameters.**

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `axis` | `str` | required | Which **physical** axis to break: `"x"` or `"y"`. Physical, not encoding-relative: under `CoordFlip()`, `axis="y"` still breaks the physical y axis, which by then carries the flipped chart's x encoding. |
| `gap` | `tuple` or `list` | required | One `(start, end)` pair, or a sequence of pairs. Each bound is numeric (`int`, `float`, or a numpy scalar: `np.int64`, `np.float32`, ...) or temporal (ISO-8601 date/datetime `str`, `datetime.date`, `datetime.datetime`, `numpy.datetime64`, `pandas.Timestamp`). |
| `break_size` | `float` | `12` | Pixels of axis reserved for each drawn break indicator. |
| `break_style` | `str` | `"slash"` | Indicator style: `"slash"`, `"zigzag"`, `"wave"`, or `"gap"` (whitespace, no indicator drawn). |

**Construction and validation** (`ferrum.structural.BreakAxis.__post_init__`). Every refusal below is a `ValueError` at construction time naming the offending value. A temporal bound is coerced to epoch-milliseconds UTC by `ferrum.annotation.coords.temporal_coord_to_epoch_ms`, literally the same function the encoding layer's TimeScale-domain coercion and annotation coordinates go through, so those three Python surfaces accept the same types; naive bounds are treated as UTC, aware ones converted to UTC first. One surface stays narrower, by omission rather than design: the Rust-backed `TimeScale(domain=[...])` constructor extracts its domain in Rust and refuses `numpy.datetime64`, which the three Python seams accept.

- `axis` must be `"x"` or `"y"`; `break_style` must be one of the four named styles.
- `gap` must be a tuple or list, must hold at least one pair, and every pair must have exactly two bounds.
- Both bounds of one pair must be the same kind: a pair mixing numeric and temporal is refused.
- Each coerced pair must be finite (no NaN, no ±inf) and ordered (`start < end`; equal bounds are a zero-width gap and are refused).
- A string bound that is not ISO-8601 is refused as an unparseable date, not as "not numeric".
- A not-a-time bound (`pandas.NaT`, `numpy.datetime64("NaT")`) is refused by name, as is a `numpy.datetime64` too large to represent as a microsecond-precision epoch offset (proven in arbitrary-precision integers before any fixed-width cast can wrap).
- Not construction errors, because the constructor cannot see the data: a gap outside the resolved domain, and a temporal gap on a non-temporal axis. Both render as if the break had not been declared, and warn (see Warnings).

The stored `gap` keeps the container type it was given (a tuple of pairs stays a tuple, so a tuple-spelled `BreakAxis` stays hashable) with each pair canonicalized to a 2-tuple of coerced floats. The wire form is `{"type": "break_axis", "axis", "gaps": [[start, end], ...], "break_size", "break_style"}`; the Rust `chart_config` gate accepts `f64` bounds only, so a hand-built config dict must carry already-converted epoch-ms numbers, not date strings.

**Render model.** Gaps are normalized once per axis before any geometry is built: non-finite bounds dropped, each gap clipped to the resolved domain, empty results dropped, then sorted and merged, so overlapping, nested, adjacent, and unsorted gaps all reconcile into the same union and an indicator only reserves pixels for a gap it actually draws. Declaration order does not matter. The surviving gaps become a piecewise scale whose segments tile the **broken scale's own padded pixel range**, not the raw plot rect, so a broken axis's ticks, marks, gridlines and indicator sit on the same pixels as the unbroken chart's, and a value past the domain clamps to the domain's own end pixel. Each surviving gap consumes `break_size` pixels.

**Per-mark-kind behavior**, decided by what the geometry *encodes* rather than by node shape (`BreakPlacement::for_kind`, exhaustive over mark batch kinds):

| Mark batch kind | Behavior under a gap |
|---|---|
| `bar`, `area`, `rect`, `ribbon`, `polygon` (incl. `mark_violin`) | Clipped: one sub-node per visible segment it intersects, nothing when it lies wholly inside a gap. A cut edge that the gap itself produced is **squared** on a rounded (`corner_radius`) piece; corners inherited from the source rect keep their radius. |
| `point`, `tick`, `text`, `image` | Translated whole to the anchor's compressed position, keeping drawn size and shape, and hidden whole when the anchor is inside a gap. The anchor is the row's own scale-mapped position, so an asymmetric shape and a `dx`/`dy`-offset label hide on their data position, not on their drawn extent. |
| `label` | The text moves as a glyph; the leader line's two ends are two data positions and are placed independently. |
| `line`, `rule`, `arc` | Split: one sub-node per visible piece, cut at each gap edge with the cross-axis coordinate interpolated at the crossing, and the in-gap portion not drawn. A stroke with no visible piece is dropped. |
| `segment` | The shaft splits as above; an `arrow=True` head is rebuilt from the endpoints of the piece reaching its tip, keeping its drawn size and pointing along that piece. A tip inside a gap drops the head. |

Split pieces carry their row metadata through 1→N bookkeeping, so `data_indices`, tooltips, hrefs, descriptions and polar spans stay one entry per node; a `key=` sidecar suffixes every piece after the first as `{key}#{i}`, choosing the smallest free `i` so a second axis's pass cannot collide with the first's.

**Axis chrome.** Only nodes a tick value positioned are re-placed, and only on the axis they were built along: a tick dash and its label translate **rigidly on that tick's own anchor**, so a label keeps its drawn baseline offset bit for bit and hides if and only if its own tick's anchor is in a gap. The axis title and the domain line are exempt (their positions come from the axis rect, not from tick values), as is all cross-axis chrome, so a y break moves horizontal gridlines and leaves vertical ones spanning the full plot rect.

**Annotations** (§3.11) are routed by what their geometry means, mirroring the mark side. `rect` and `span` are clipped into one piece per visible segment, silently (a span's label re-anchors on the largest surviving piece, ties keeping the first). `line` and `arrow` split through the same vertex splitter marks use. `text`, `callout` and `image` keep a plain membership test on their coordinate. `bracket` is suppressed whole, permanently and by design: its two endpoint tips say where the bracketed range begins and ends, so a cut would either strand them or invent new ones. An annotation is suppressed only when nothing of it survives, or when its kind admits no honest split; suppression rolls up into one warning per axis per mechanism (a coordinate inside a gap, versus an interval reaching across one). Pixel- and norm-space annotation coordinates are never touched by a break.

**Facets and composites.** A break is built per facet panel, against that panel's own plot rect and scale pixel range, so identical data places identically in every panel and every panel draws its own indicator. Diagnosis is per chart, not per panel: a fact every reporting panel agrees on is reported once, and a divergent one names the panel it was measured on and how many panels share it. Composite leaves are separate scenes with their own break build and their own warnings.

**Multiple specs on one axis** merge into a single break result before any geometry is built: their gap lists union through the same normalization, so N same-axis specs render byte-identically to one spec declaring the union. A conflicting `break_style`/`break_size` among merged specs keeps the first declaration's value and warns once per axis per field.

**Coordinate systems.** `CoordPolar` and `CoordGeo` combined with any `BreakAxis` refuse at the render funnel with a typed `ValueError` (`RenderError::BreakAxisIncompatibleCoord`) naming both the coord and the axis, rather than silently corrupting already-projected geometry. Every other coord renders normally.

**Warnings** (all `RenderWarning`, surfaced as Python warnings): `BreakAxisIneffective` (no gap overlaps the axis domain, or the axis is ordinal — the chart then renders exactly as if the break had not been declared, padding included), `BreakAxisOnNonlinearScale` (a `log`/`pow`/`sqrt`/`symlog` axis: placement uses a linear pixel-to-data map, so the removed range can differ from the gap asked for), `BreakAxisMergeConflict` (same-axis specs disagreeing on `break_style`/`break_size`), and `BreakClipSidecarMisaligned` (a split batch whose per-node metadata length disagreed, so tooltips/links/keys on it may name the wrong marks).

**Disclosed residuals.** A curved line (`mark_line(interpolate="basis"/"monotone"/"cardinal")`) keeps the per-vertex fallback and can still draw a stroke through the hidden band, because cutting a placed curve honestly needs subdividing it. A piece that is simultaneously rotated (`angle != 0`), rounded, and cut keeps its original corner radius, because converting it to a path would relocate its rotation anchor. Non-linear-scale placement is linear (warned, above). Interactive (WASM) output builds its scene through the same break-aware pass, so marks are placed identically, but break-aware interaction (hit-testing, keyed transition pairing across a split) remains a documented non-goal.

> **Dated note (2026-09-11, batch H tasks 3/4 — F-L08-06/F-L08-07, `BreakAxis` (`ferrum.structural.BreakAxis`) clipping, glyph placement, tick behavior, and annotation suppression+remap go from broken/undocumented to matching the guide's contract):** the guide's clipping model (`docs/site/guide/concepts/break-axes.md`) is now real. A mark's placement under a `BreakAxis` gap is decided by what its geometry *encodes*, not by node shape: an extent mark (bar, area, rect, ribbon, and a filled/closed shape like `mark_violin`'s ring) whose data extent intersects a gap is split into one clipped sub-node per visible segment; a glyph mark (point, tick, text, label, image) is translated whole to its anchor's compressed position and hidden whole (never resized, split, or dropped) when its anchor falls inside a gap; a vertex-placed mark (line, rule, segment, arc) places each vertex independently, hidden per-vertex — this is the one shape that still spikes toward the hidden sentinel for *connected* stroked geometry crossing a gap, a documented, unfixed residual (per-segment line splitting is a filed follow-up). A `mark_label(leader_line=True)` connector and a `mark_tick` rug dash are visually the same node kind (`Line`) inside a glyph batch with opposite meanings (a two-point connector vs. a fixed-length dash) and are now disambiguated by the mark's batch kind, not the node shape. In-gap axis ticks and gridlines were already hidden correctly before this batch and remain so; what this batch found and fixed in that chain was two things: the axis title — a `Text` node at the axis midpoint that reverse-mapped into a gap straddling it and teleported off-canvas (all three pre-existing break goldens carried this defect and are now corrected, one element each: the restored title) — and, on the OTHER axis, its tick marks. Which chrome belongs to a broken axis used to be inferred from a node's own pixel coordinate, and a downward x-axis tick starts ON the plot floor — inside the y axis's own pixel span — so on a chart with no scale padding a y-axis break reverse-mapped and clamped every x-axis tick dash to a single pixel (rendered as zero-length lines, the tick marks vanishing entirely while their labels survived). Ownership is now decided by the axis's own dimension instead of a coordinate range, which closes this for both directions. A `SecondaryY` layer's marks are exempt from a break declared on the primary axis, and overlapping/nested/domain-edge-touching gaps normalize (merge, sort, clip to domain) before segments are computed so an indicator reserves pixels only for a gap it actually draws. Separately, an annotation (`annotate_*`/`ann.*`, §3.11) whose data-space coordinate on a broken axis falls inside that axis's gap is now suppressed with a named `RenderWarning` (`annotation '<kind>' at <axis>=<value> falls inside BreakAxis gap (<lo>, <hi>); suppressed`) instead of silently rendering at the unbroken (wrong) position; a surviving data-space annotation is remapped through every same-axis break in declaration order, landing on the same compressed pixel a mark at that value would; pixel-/norm-space annotation coordinates are never touched by a break. An extent-shaped annotation (rect, span, line, arrow, bracket) is suppressed when its interval INTERSECTS a gap, not only when one of its endpoints lies inside one — the spec's "any in-gap data coordinate" includes the interval's interior, so a rect large enough to contain the gap is suppressed like every smaller overlapping one instead of painting straight across the break (a distinct warning names the extent and the gap it crosses). Point-shaped coordinates (text, callout anchors, image) keep the plain membership test. Suppression is TOTAL for extents under this contract: a band that brackets every visible segment (e.g. `annotate_rect` spanning the whole axis as a background shade) is dropped whole rather than clipped into one piece per segment, which is what the equivalent MARK geometry does — unifying the two answers is the filed follow-up, not this batch. An out-of-domain annotation value now clamps to the correct plot edge, and that correction applies to EVERY chart, not only broken ones: the prior sign error put an out-of-domain value on the mirrored side on a break-free chart too (`annotate_hline(y=-50)` on a `[0, 100]` axis rendered above the plot); the break's fold only made it conspicuous by clamping the value back onto the visible canvas. `Inset(connect_to=...)` resolves its own out-of-domain anchor through that same corrected rule. A `BreakAxis` that resolves to zero effective gaps (every gap outside the domain, or an ordinal axis) now emits a warning AND renders exactly as if it had never been declared — the render, not only the diagnosis, matches "had no effect": no indicator, no remap, and (for a continuous axis whose gaps simply miss the domain) no collapse of the scale's own pixel padding either, which an earlier cut of this fix still let through as an up-to-8px shift in every mark, tick, and gridline. A break on a reversed (descending-domain) scale is now effective instead of being silently discarded, and a break on a non-linear scale (`log`, `pow`/`sqrt`, `symlog`) warns that its placement is computed with a linear pixel-to-data map and therefore removes a different data range than requested — unfixed, disclosed, and a filed follow-up. A `mark_segment(arrow=True)` arrowhead under a break keeps both its drawn size and its alignment with the shaft: the head is rebuilt from the shaft's own placed endpoints — the same thing `annotate_arrow` already did on the annotation path — rather than having its three vertices translated or placed independently, which previously sheared a 20px head into a ~90px one, and then (a later fix) kept the head's size but left it pointing along its pre-break direction instead of the shaft's new one. The shaft itself still places per vertex. Disclosed residuals, not fixed by this batch: `Line`-shaped extent marks (`mark_rule`, box/violin whiskers, area borders) still spike across a gap rather than splitting (same root as the vertex-placed-mark residual above); a bar's `corner_radius` is replicated onto each split segment, rounding the internal cut edge; a mark sitting within a few px of a gap edge can resolve to "hidden" due to bbox-anchor imprecision for asymmetric glyphs; a faceted chart applies its `BreakAxis` to panel 0 only; interactive (WASM) rendering under a break remains a documented non-goal.

> **Dated note (2026-09-11, wave-1 hardening, #240-3 — the in-gap annotation warning rolls up per axis, not per suppressed element):** the batch-H note above documented per-instance suppression warnings; a chart with N annotation elements falling in the same axis's break gap now emits exactly ONE rolled-up `RenderWarning` naming the axis and how many annotation elements were dropped (both the point-coordinate-in-gap mechanism and the extent-spans-a-gap mechanism roll up independently, since they report different facts) — previously each suppressed PRIMITIVE warned separately, so a chart with 4 `annotate_hline()` calls (each desugaring to a line + its own label, 8 primitives) emitted 8 near-identical warnings for what a user experiences as one fact. The count is honestly of annotation ELEMENTS, not of the user-facing `annotate_*()` calls that produced them (this layer has no way to attribute a desugared primitive back to its originating call), and the rolled-up kinds list is deduplicated (first-appearance order) in the printed text while the stored count is not — a 4-call/8-primitive repro reads `"8 annotation elements (line, text) on the y axis fell inside a BreakAxis gap and were suppressed"`.
>
> **Dated note (2026-09-11, wave-1 hardening, #118 — mark builders stop returning empty in silence):** a mark builder that discovers it has nothing to draw now names why, through a new `RenderWarning::MarkBuildEmpty { layer, mark, reason }`, at five sites that previously rendered a silent blank panel with no diagnostic: `mark_segment()` bound to x/y only (missing x2 or y2), `mark_image()` with no `url=` encoding and no valid Raster-transform data, `mark_geoshape()` whose batch has no `__geometry__` column at all (a present-but-all-null column stays silent — the pre-existing, still-legitimate per-panel/facet-filter case), `mark_rect()`'s IQR box-body layer when neither its x nor y axis resolved to an ordinal scale (the render-time surface of #183, below), and `mark_image()` under `CoordPolar`/`CoordGeo` (both its url-tile and raster paths place images in Cartesian pixel space with no fallback under either coord; the Cartesian case is unaffected and stays silent). Every OTHER empty return in the mark-builder family is exempted by design, not by omission: a zero-row input batch, a documented per-row null/non-finite/out-of-domain skip, and a facet panel that is legitimately empty after a filter all still render silently — warning on those would make a correct, intended filter result look like a bug. `RenderWarning::GroupCardinalityDegenerate`'s singular form (`"1 category was placed..."`) is also corrected alongside this (previously always plural, even at N=1).
>
> **Dated note (2026-09-11, wave-1 hardening, #183 — sub-pixel ordinal cardinality warns):** an ordinal-domain positional scale that resolves to a sub-pixel bandwidth (too many categories for the available pixel range) now emits `RenderWarning::GroupCardinalityDegenerate { categories, bandwidth_px }`, sibling to the existing `ColorPaletteOverflowed` warning, instead of silently drawing bars/points that overlap into an unreadable smear with no diagnostic. This fires per panel on a faceted chart (matching `ColorPaletteOverflowed`'s existing per-panel convention — N independently-degenerate panels are N independent facts, not one to be deduplicated). The frozen report's own repro (`catplot(x="cont:N", ..., kind="box")` on a Float64 column, which resolves the box-body layer's ordinal-vs-continuous mismatch silently) is closed at its actual render-time site by the #118 `mark_rect()` case above; the underlying Python-side root cause (a composite-mark desugar losing the chart-level explicit axis `type_` across the chart↔layer merge) is a distinct, unfixed follow-up outside this note's scope.

> **Dated note (2026-09-11, batch H tasks 5/6 — F-L08-08, `Inset` (`ferrum.structural.Inset`) bounds go from a docstring/stopgap mismatch to the design's original resolution model):** `Inset.bounds`'s four entries — `(left, top, right, bottom)` — resolve two coordinate kinds, each entry independent of the others: a bare `float` or `NormCoord` (`fm.norm(f)`) is a normalized `[0, 1]` fraction of the parent plot area (unchanged, as designed since the 2026-05-24 inset-panels design); `PixelCoord` (`fm.px(n)`) is now a real absolute pixel offset from the plot area's origin, resolved in Rust against the computed plot rect at layout time (`InsetCoord::Px`, `chart_config.rs`) — previously `fm.px(...)` in an `Inset` bounds tuple raised `TypeError` as a stopgap. Mixed kinds within one bounds tuple are allowed and resolve independently (e.g. `bounds=(px(10), norm(0.0), px(220), norm(1.0))`). The docstring's former claim that a bare float meant a *data-space* coordinate was same-commit authorship drift against the original design (which always specified normalized-or-pixel, never data-space, and reserved data-space coordinates for the annotation layer only) — corrected here, not implemented, since implementing it would have overridden a located, real design decision. `Inset(bounds=...)` now refuses at construction time (`ValueError`) for a normalized entry outside `[0, 1]`, a negative pixel entry, or an empty box (`left >= right` / `top >= bottom`, compared only within like kinds — a mixed-kind pair, e.g. `left=px(50), right=norm(0.1)`, is never compared for emptiness); previously all three constructed silently and rendered off-canvas or as a degenerate zero-extent inset.

> **Dated note (2026-09-23, wave 6B, `#239` — a malformed `Inset` bound is refused by name):** each `bounds` entry is now classified once (`norm`, `px`, or a bare number read as `norm`) and validated raw through the package's shared fraction/pixel validators, with the entry's label. Refusal is the only thing that changes; the wire for every accepted input (`norm(0.5)`, `px(10)`, bare `0.5`, numpy integer and floating scalars, `Decimal`) is byte-identical. Before, the classifier ran `float(entry)` before any validator saw the value, so `"a"` raised an unlabeled `ValueError`, `None` raised a `TypeError`, and `True`/`False` and numeric strings such as `"0.5"` were silently laundered into fractions that every sibling validator refuses. All five now raise one labeled `ValueError`, e.g. `Inset.bounds[0] (left) must be a number between 0 and 1, not str; received 'a'` (user ruling Q6). The accepted set for a bare entry is closed: `int`, `float`, any other non-`bool` `numbers.Real`, numpy integer and floating scalars, and `Decimal`. Bytes, numpy string/bytes/complex/datetime/timedelta scalars, arrays, Python `complex` and an arbitrary object defining `__float__` are refused by type name. `nan`/`inf` keep the validator's existing answer.
>
> **Amended 2026-09-24 (wave 6B close):** "is this a real number" has one answer across the package, `ferrum._validate.is_real_number` (any `numbers.Real` or `Decimal`, never `bool`, never a timedelta), shared by the Inset classifier, the pixel/fraction validators, dash lengths, `BreakAxis.gap` and the filter/impute literal serializers. Two consequences: a wrapped bound accepts what a bare one does (`px(np.int64(10))` was refused while bare `np.int64` was accepted), and every accepted bound reaches the wire as a float, so `px(10)` and `norm(1)` now serialize `10.0`/`1.0` where they serialized the int (rendering is unchanged). The same predicate makes `PaddingConfig`, `GridConfig`/`GridAxisConfig` `width`/`opacity` and `override(padding_*=)` accept numpy scalars and `Decimal`, and those configs now store and serialize the float (`PaddingConfig(top=5).to_dict()` gives `{"top": 5.0, ...}`); `transform_filter`/`transform_impute` accept `Decimal`; an `np.timedelta64` `BreakAxis.gap` bound refuses with a typed error instead of a raw `TypeError`.

> **Dated note (2026-09-24, wave 6C, `#120`(6) — every numeric door shares one integer predicate and one finite-real coercion, and refuses the same way):** `ferrum._validate` gains `is_integer_number` (an `int`/numpy-integer sibling of `is_real_number`; never `bool`, `timedelta64`, or `Decimal`) and `coerce_finite_real(label, value, *, required=False)`. A door that reaches it now raises `ValueError` reading `"{label} must be a real number (int, float, numpy scalar, or Decimal), not {type}; received {value!r}"` when the value is not a real number, or `"{label} must be a finite number a float can hold; received {value!r}"` when its `float()` is not finite (`10**400`, `Decimal("sNaN")` included); otherwise it returns a builtin `int` (for an integer) or `float`. Every real-number door that used to run its own conversion now calls it: `heatmap(vmin=`/`vmax=`/`center=...)` ran `math.isfinite` on the raw value, so a numpy integer or `Decimal` bound constructed and then leaked a raw `TypeError: Object of type int64/Decimal is not JSON serializable` at render, and `heatmap(vmax=True)` rendered as `1` — it now refuses, `"heatmap(vmax=...) must be a real number (int, float, numpy scalar, or Decimal), not bool; received True"`. `AxisConfig(domain_min=`/`domain_max=...)` (and, through it, every `override(<axis>_domain_min/max=...)` leaf) had the same serialization leak and, for the same reason, now also accepts `Decimal` (`AxisConfig(domain_min=Decimal("2.5"))` stores and serializes `2.5`, where it previously refused). `ModelSource.embeddings(method="tsne", perplexity=...)` and `BreakAxis.gap`'s bounds (whose bare `float()` let an oversized integer's `OverflowError` and a `Decimal("sNaN")`'s `ValueError` escape unlabeled) route through the same coercion. `coerce_positive_int` (behind `Chart.facet(ncols=`/`nrows=...)`, `Facet(columns=...)`, and the new `catplot(n_boot=...)` door, a door not previously listed in the spec's parameter table) now asks `is_integer_number`, accepts a numpy integer, and above `2**53` — the largest integer a float holds exactly — refuses `"{label} must be at most 2**53, the largest integer a float holds exactly; received {value!r}"`; `Chart.facet(ncols=...)` between `2**32` and `2**53` still passes Python and is refused by Rust's `u32` field at render, naming `facet:` rather than the door — a residual, not fixed here. Refusal text moves with the unification: heatmap's finiteness message reads "must be a finite number a float can hold" (was "must be finite"); `AxisConfig`'s domain-bound refusal names the call-form door (`AxisConfig(domain_min=...)`) instead of the internal field label; `Chart.facet`'s count refusal names `Chart.facet(ncols=...)`/`Facet(columns=...)` instead of the internal `_Facet` attribute; catplot's `n_boot` house text adds "and at most 2**53". Separately, `catplot(ci=`/`n_boot=...)` gains real validation — `ci="95"` previously laundered through a bare `try: float(v)` and now refuses — and, on the five kinds that draw no interval (`strip`, `swarm`, `box`, `violin`, `boxen`) plus `count`, a non-default `ci=`/`n_boot=`, previously accepted and silently ignored, now refuses reading `"catplot({name}={value!r}) is not supported with kind={kind!r} -- that kind draws no confidence interval, so {name}= would be ignored. Drop {name}=, or use one of the estimating kinds ('bar', 'point')."` The defaults (`ci=95`, `n_boot=1000`) stay silent on every kind, since an omitted argument cannot be told apart from a written default. This replaces the earlier rule that a non-default `ci=`/`n_boot=` on these kinds must never raise. Every already-accepted numeric shape's rendered output, and every accepted numpy/`Decimal` bound's serialized value, is otherwise as documented by the amendment directly above; no golden moved. The mark-style doors and the remaining number literals join the same coercion. The numeric `mark_*` style kwargs (`size`, `opacity`/`alpha`, `stroke_width`, `corner_radius`, `font_size`, `dx`, `dy`, `angle`, `limit`, `band_size`, `arrow_size`, on primitive and composite marks alike) and `transform_impute(value=...)` now store a numpy scalar or `Decimal` as a builtin number, where it leaked `TypeError: Object of type int64 is not JSON serializable` at `to_json()`/render, and refuse a non-number or non-finite value in the door's voice, `"mark_point(size=...) must be a real number (int, float, numpy scalar, or Decimal), not str; received 'a'"`, where the renderer refused it later with its own text (`mark_style: invalid type: string "a", expected f64`); a composite mark refuses when the chart is resolved, as its other style checks do. `transform_filter`'s and `transform_impute(keyvals=...)`'s number literals refuse through the same helper, reading `"transform_filter: value for field 'x' must be a finite number a float can hold; received nan"` (was `"transform_filter: non-finite float nan not allowed for field 'x'; accepted: string, number, bool, or None"`) and `"transform_impute: keyvals value for key 'x' must be a finite number a float can hold; received nan"` (was `"transform_impute: non-finite float nan not allowed in keyvals for key 'x'; accepted: string, number, or bool"`), and now also refuse an integer a float cannot hold (`10**400`, previously emitted as a literal) and `Decimal("sNaN")` (previously a bare `ValueError`). A dash length and a conditional numeric literal (`fm.value(...)` on a numeric channel) refuse `10**400` and `Decimal("sNaN")` in their existing refusal shapes instead of leaking a raw `OverflowError` or an unlabeled `ValueError`. The count predicate is renamed `coerce_positive_int` and returns the builtin `int` every door now stores, so `Chart.facet(ncols=np.int64(2))` stores `2`. **Amended 2026-09-26 (wave 6C close):** the mark-style numeric fields split by wire type. `ferrum._core.mark_integer_kwargs()` publishes the unsigned-integer (`usize`) subset of `mark_numeric_kwargs()`, today `limit`, and those fields go through `coerce_non_negative_int` (the `>= 0` sibling of `coerce_positive_int`) instead of `coerce_finite_real`: `mark_text(limit=2.5)` refuses at construction with `"mark_text(limit=...) must be a non-negative int, not float; received 2.5"` and `limit=-1` with `"mark_text(limit=...) must be >= 0; received -1"`, where both constructed and then failed at render with `mark_style: invalid type: floating point '2.5', expected usize`. And a style kwarg the mark does not honor is refused as unknown before its value is checked, so `mark_point(corner_radius="x")` raises the unknown-keyword `TypeError` rather than a "must be a real number" `ValueError` about a value the mark would never use.

> **Dated note (2026-09-12, wave-2 break-axis #226 — a break's segments tile the scale's own padded pixel range, so the axis keeps its padding):** a `BreakAxis`'s piecewise scale is built against the broken scale's own pixel range (`ScaleKind::pixel_range()`) rather than the raw plot rect. The two differ by the positional-scale padding inset (5% of the span, capped at 8px per end), and the remap that re-places every coordinate onto the broken axis always reverse-mapped from the padded range: laying the segments over the rect made that round trip lossy in exactly the padding band, so EVERY effective break silently deleted the axis's padding. Observable consequences, all corrected: a broken axis's ticks, gridlines, marks and break indicator now sit at the same pixels as the unbroken chart's (on the reference chart, y ticks run 24.0…412.961 broken and unbroken alike, where a broken axis previously ran 16.0…420.961); a gap that only just reaches into the domain no longer steps the whole chart by the padding width the moment it crosses the domain boundary (the zero-effective-gaps case already rendered exactly unbroken, so the two sides of that boundary used to disagree by 8px); and a data value past the domain clamps to the domain's own end pixel — the padded edge — instead of the plot rect's edge. The axis DOMAIN LINE is exempt from the remap alongside the axis title, for the same reason: its endpoints are the axis rect's corners rather than tick values, so under a break they clamped to the domain's ends and the line shrank onto the segment span, drawing an axis shorter than the plot it bounds. It now spans the full plot rect exactly as on an unbroken chart, with the break indicators overlaying it. The same rule governs CROSS-AXIS chrome, which is the general form of it: only chrome a tick value on the broken axis positioned is re-placed, so a y break moves the horizontal gridlines and leaves the vertical ones (placed at x tick values, and spanning the plot rect's height from corner to corner) untouched, and vice versa — a broken chart's cross-axis gridlines meet the opposite axis line exactly as an unbroken chart's do. A grid band fill's boundaries ARE tick values on the y axis, so a y break still clips it per visible segment, but its outermost boundary is the plot rect's own edge and stays there (the padding strip at each end of the axis keeps its stripe), and an x break leaves a band's full rect width alone. Behavior-changing for every chart carrying an effective `BreakAxis` (the five break goldens moved and were re-blessed under the PNG-inspection protocol); unbroken charts and ineffective-gap breaks are byte-identical. This is the fix the batch-H note above names as "the scale-pixel-range fix tracked as a follow-up".

> **Dated note (2026-09-12, wave-2 break-axis #226/#231 — a tick label translates rigidly on its own tick's anchor, not on its own drawn baseline coordinate; D2):** `AxisNodes::data_positioned` now carries `AxisTickNode { node, anchor }` — every data-positioned axis-chrome node (a tick dash or a tick label) paired with its own tick's data-mapped pixel position (`TickLayout::position`) on the unbroken axis, tagged at `build_axis` time in `marks/axis.rs`. `BreakRemap::remap_axis_chrome` reads this pairing and calls `remap_axis_tick_node`, which shares the `translate_on_anchor` primitive with `remap_glyph_node`: the whole node translates by the DISPLACEMENT its anchor undergoes under the break, rather than being reverse-mapped from its own drawn coordinate. A tick label's drawn position is `tick.position` plus a baseline offset (`+ font/3`, a `label_flush` edge shift, rotation geometry) — drawn size, not a second data value — so reverse-mapping the label's own coordinate (the pre-fix model) answered "where does this baseline offset belong on the broken scale", compressing each label's own gap from its dash by a different amount than its neighbor's and reading a flush-edge label's offset as exactly zero. Rigid translation on the anchor keeps every label's drawn offset from its own dash bit-for-bit identical to the unbroken chart's, for every visible tick, and a label now hides if and only if its own tick's anchor falls inside a gap — never at a different offset than its siblings, and never independently of whether its own dash/gridline is visible. A tick dash's own two endpoints already equal its anchor, so this is a byte-identical no-op for dashes; only labels change. An axis-chrome node reaching `remap_axis_tick_node` that is neither a `Line` (dash) nor a `Text` (label) — the only two kinds `build_axis` tags with an anchor — fails a `debug_assert!` in a debug build rather than silently falling back to the old per-coordinate remap; in release this check compiles out and the node still translates rigidly via `translate_on_anchor`. Pinned by a new label↔dash offset-pairing assertion (`tests/test_break_axis_geometry.py`) satisfying acceptance criterion 2. Behavior-changing for every chart carrying an effective `BreakAxis` with visible axis labels (part of the same golden re-bless as the #226 pixel-range fix above — the five break goldens' bottom-label baseline moved again, to its final, sibling-matching offset); unbroken charts are byte-identical.

> **Dated note (2026-09-12, wave-2 break-axis #230-4/#230-9 — same-axis `BreakAxis` specs merge into one break result, and `BREAK_HIDDEN` is poison):** two or more `BreakAxis` specs declared on the same axis (`fm.Chart(...) + fm.BreakAxis(axis="y", gap=(150, 400)) + fm.BreakAxis(axis="y", gap=(500, 850))`) previously built one `BreakResult` PER spec, each resolved against the ORIGINAL unbroken domain, and folded marks/surviving annotations through all of them in sequence — numerically wrong (the second break's pixel-to-data reverse map re-interpreted the first break's already-broken pixel as an unbroken-domain data value) and, worse, capable of resurrecting a mark the first break had hidden: the second break's reverse map clamped the first break's `BREAK_HIDDEN` sentinel (`-99999.0`, far outside any real pixel) back into the domain like any other out-of-range pixel, landing it at a domain edge instead of leaving it hidden (measured: a mark at `y=200`, inside a `(150, 400)` gap, rendered at the plot's top pixel under a second `(500, 850)` break declared as a separate spec, overplotting a mark at `y=1000`) — silently, with no warning, even though the annotation-suppression path (which reasons in data space, not pixels) correctly suppressed the same value. Same-axis specs now merge into exactly ONE `BreakResult` before any break is applied: their gaps union (through the same `normalize_gaps` a single spec's own gap list always ran through — overlapping, nested, and unsorted gaps still reconcile the same way), so N same-axis specs render byte-identically to one spec declaring the union of gaps, and the resurrection is structurally unreachable (there is only ever one remap per axis for the mark/annotation fold to run through). A conflicting `break_style` or `break_size` among merged same-axis specs keeps the FIRST declaration's value (matching what "one spec with the union of gaps" would carry) and warns once per axis per conflicting field via the new `RenderWarning::BreakAxisMergeConflict { axis, field }` (`"multiple BreakAxis specs on the {axis} axis disagree on {field}; the first declaration's value was kept"`); merging the gaps themselves is the documented semantics of a same-axis break and never warns on its own. Independently, as belt-and-braces, `BREAK_HIDDEN` is now poison at the mechanism level (`BreakRemap::data_at`): a pixel already parked at the sentinel reverse-maps to `NaN` rather than a clamped domain value, so it can never re-enter view through `data_at`/`coord` regardless of caller — not only the specific double-remap sequence the same-axis merge eliminates. Two annotation-side unit tests (`resolve_x_folds_through_multiple_same_axis_breaks`, `resolve_x_folds_same_axis_breaks_in_declaration_order`) that hand-built a `ScaleContext` with two same-axis `BreakRemap`s to pin the old sequential-composition behavior are retired (see the rationale comment in `annotation.rs`) rather than kept green, since that construction is no longer reachable from production and keeping them passing would pin the replaced defect as if it were still supported. Behavior-changing only for charts declaring 2+ `BreakAxis` specs on the same axis; a chart with at most one `BreakAxis` per axis (the overwhelming majority) is byte-identical, and no golden moves.

> **Dated note (2026-09-12, wave-2 break-axis #230-8 — Polar and Geo refuse a BreakAxis at the render funnel instead of corrupting geometry silently; D3):** `scene_build::build_scene` now checks, before any geometry (break or otherwise) is built, whether `spec.coord` is `Polar` or `Geo` AND `chart_config.structural` carries a `BreakAxis`; if so it returns the new `RenderError::BreakAxisIncompatibleCoord { coord, axis }` (`coord` is `"Polar"`/`"Geo"`, `axis` is the offending `BreakAxis.axis` token), surfacing as a Python `ValueError` naming both. This closes what was previously silent geometry destruction, not suppression, for `Polar`: `mark_arc`'s wedge transform runs during batch build, before any break geometry exists, so the pre-fix break fold reverse-mapped already-projected wedge pixels through the Cartesian scale the projection replaced — measured: every polar wedge destroyed (11 `BREAK_HIDDEN` sentinel occurrences in one repro) — while still painting a break indicator over Cartesian axes that `resolve_panel_axes` suppresses under this coord, with no warning. The false doc claim that Cartesian breaks are "suppressed" under polar coords (a comment on `BreakPlacement::for_kind`'s `Arc` arm) is removed. The gate is extended to `Geo` on a structural argument: `mark_geoshape` places every vertex by bbox-fitting its `projection::forward`-projected lon/lat into the plot rect, independent of the resolved x/y scales, so `BreakRemap`'s reverse-map through the Cartesian scale it replaces has no scale-relative position left to correct — the break contract cannot be honored for `Geo` geometry either. This funnel is the one every standalone render and every composite leaf's render passes through, so a composite leaf combining either coord with a break refuses per-leaf, exactly like any other render error. Behavior-changing only for a chart (or leaf) combining `CoordKind::Polar`/`CoordKind::Geo` with any `BreakAxis`; every other coord/structural pairing, including a `Polar`/`Geo` chart with no `BreakAxis`, is unaffected.

> **Dated note (2026-09-12, wave-2 break-axis #230-6 — `BreakAxis.gap` bounds accept temporal values, coerced through the encoding layer's own UTC contract):** each `BreakAxis.gap` bound now accepts, besides `int`/`float`, `str` (ISO-8601 date/datetime), `datetime.date`, `datetime.datetime`, `numpy.datetime64`, or `pandas.Timestamp` (a `datetime.datetime` subclass). A temporal bound is coerced to epoch-milliseconds UTC at `BreakAxis.__post_init__` (construction) time by reusing `ferrum.annotation.coords.temporal_coord_to_epoch_ms` — the same conversion `ferrum.encoding._scale`'s TimeScale-domain coercion and annotation coordinates already use (SPEC-04) — for `str`/`datetime.date`/`datetime.datetime`/`pd.Timestamp`; a naive (timezone-unaware) bound is treated as UTC, a timezone-aware one is converted to UTC first, matching that function's documented contract. `numpy.datetime64` has no analog in that function and is converted directly via a dedicated arbitrary-precision-integer proof (`_epoch_us_for_datetime64`) that refuses an out-of-range value before any fixed-width numpy cast can silently wrap, uniformly for every datetime64 unit (`Y M W D h m s ms us ns ps fs as`), including a fine unit whose dtype `count` multiplies it back up past microsecond resolution. `BreakAxis` is a frozen dataclass; the coerced float values replace `self.gap` in place via `object.__setattr__` so the Rust wire (which accepts only `[f64; 2]`) and every downstream reader see plain floats, never a raw date string. A gap pairing one numeric and one temporal bound raises `ValueError`; an unparseable string bound raises `ValueError` naming the value; a `pd.NaT`/`np.datetime64('NaT')` bound raises `ValueError` naming it rather than propagating a NaN into the existing finite/ordering checks, which still apply to the coerced values. A temporal gap that lands outside a non-temporal axis's domain is not a construction-time error — it renders as an out-of-domain gap and warns `BreakAxisIneffective` at render time, the same as a numeric out-of-domain gap. This closes the `docs/site/guide/concepts/break-axes.md` temporal example, which previously called this combination unsupported (see the guide's own updated note). Behavior-changing only for a `BreakAxis.gap` bound of a temporal type; existing numeric `gap=` calls are unaffected and construct identically.

> **Dated note (2026-09-12, wave-2 break-axis #230-1 — connected stroked geometry is CUT at a gap instead of spiking through the hidden sentinel):** a vertex-placed mark (`mark_line`, `mark_rule`, `mark_segment`'s shaft, and the box-plot whiskers that lower to a ranged `rule`) crossing a `BreakAxis` gap now emits ONE SUB-NODE PER VISIBLE SEGMENT instead of one node whose in-gap vertices sat at `BREAK_HIDDEN`. This RETRACTS three claims of the batch-H note above, which named the spike a documented unfixed residual: "a vertex-placed mark (line, rule, segment, arc) places each vertex independently, hidden per-vertex — this is the one shape that still spikes toward the hidden sentinel", "The shaft itself still places per vertex", and the disclosed residual "`Line`-shaped extent marks (`mark_rule`, box/violin whiskers, area borders) still spike across a gap rather than splitting". Measured pre-fix symptom: a three-point `mark_line` across a gap emitted `<polyline points="74.714,420.961 345.357,-99999 616,16">`, one stroke detouring far off-canvas and back. The geometry contract: each vertex is its own data position, an edge between two consecutive vertices is straight in data space, and the drawn parts of that edge are the parts of its data interval that fall in the break's visible segments — so one edge may contribute to several pieces when it spans several gaps, and the whole of a segment an edge passes straight through is still drawn. At each crossing the piece ends exactly ON the gap edge, at the point where the edge actually meets it: the cross-axis coordinate is INTERPOLATED there (`x` at the crossing, for a y break), not dropped back to the last visible vertex, which would bend the stroke toward that vertex and draw a slope the data never had. A vertex lying inside a gap therefore contributes no point of its own — the piece before it ends at the near gap edge, the piece after it starts at the far edge. A stroke with no visible piece at all (a rule whose single value is in a gap, a line segment wholly inside one) is DROPPED rather than parked off-canvas, which is the same fate the extent clip has always given a bar wholly inside a gap; correspondingly, a `mark_segment(arrow=True)` arrowhead whose tip is in a gap is dropped too, since no shaft piece reaches it, and an arrowhead whose tip IS visible is rebuilt on the endpoints of the piece that reaches it (keeping the drawn size and now pointing along the piece a reader sees it attached to, rather than along the whole compressed shaft). The rule belongs to the GEOMETRY rather than to the batch it arrives in: an extent-clipped batch carries connected stroked geometry too whenever `mark_area(line=True)` or `mark_area(borders=True)` draws an outline over the fill, and that OPEN path is cut by the same splitter, beside the fill's own clip. A broken area's outline therefore agrees with the band boundaries of the fill beneath it, instead of spiking off-canvas at a border vertex that landed in the gap or drawing one connected stroke straight across the break band when no vertex did. Split pieces go through the same 1→N bookkeeping the extent clip uses, so a batch's `data_indices`, tooltips, hrefs, descriptions, `keys` and polar spans stay one-entry-per-node and a dropped stroke takes its row with it; `keys` are still DUPLICATED across the pieces of one split node (the `{key}#{i}` suffixing is #230-5's own work, tracked separately, and interactive rendering under a break remains a documented non-goal either way). The split boundary is a property of the GEOMETRY, not of the node kind: straight-line commands split, including the `HLineTo`/`VLineTo` shorthand `mark_line(interpolate="step"/"step-before"/"step-after")` emits (whose pieces are re-spelled with explicit `MoveTo`/`LineTo` vertices — a bytes change confined to a step line a gap actually CUTS, which includes one that merely ends or begins inside a gap rather than passing through it, and is exactly where the old bytes carried a sentinel spike; whether a cut happened is reported by the splitter itself rather than inferred from the pieces it returns, since a cut that drops a boundary vertex and inserts a crossing in the same edge leaves the piece and vertex counts unchanged), while a CURVED path (`interpolate="basis"/"monotone"/"cardinal"`, i.e. `QuadTo`/`CubicTo`/`ArcTo`) keeps the per-vertex fallback and its sentinel, because cutting a placed curve honestly needs subdivision of the curve rather than a straight cut through it — unfixed, disclosed, and the one remaining shape that can spike. Axis chrome is untouched by this: a gridline is one value wide along the broken axis, so an in-gap gridline still hides whole at the sentinel as before. Behavior-changing only for charts whose stroked geometry actually crosses an effective gap; unbroken charts, ineffective-gap breaks, and curved lines are byte-identical, and no golden moves (the five break goldens carry point, bar and area geometry only, and a plain `mark_area` emits no border node at all — no golden sets `line=` or `borders=`).

> **Dated note (2026-09-13, wave-2 break-axis #230-2 — a split rounded bar/rect no longer rounds the cut edge a gap produced):** the batch-H note above's disclosed residual — "a bar's `corner_radius` is replicated onto each split segment, rounding the internal cut edge" — is RETRACTED and fixed. A `mark_bar`/`mark_rect`/`annotate_rect` piece produced by `BreakAxis` clipping now rounds only the corners it inherits from the source rect's own boundary; a corner touching an edge the gap itself produced is squared. Measured pre-fix symptom: `mark_bar(corner_radius=8)` split by a `(150, 850)` gap emitted `rx="8"` on both pieces, rounding the flat cut edge exactly like the two real corners. A piece stays a plain `Rect` (byte-identical to every pre-#230-2 render) in three cases: no cut edge touches it at all; every one of its corners is squared (the middle slice of a two-gap break, which has nothing left to round and would otherwise pay for a `Path` outline identical in geometry to `Rect { corner_radius: 0.0 }`); or its style carries a nonzero rotation `angle` — `SceneNode::Rect` rotates about its own centre while `SceneNode::Path` rotates about its first vertex, so converting a rotated piece would silently relocate its rotation anchor relative to its unbroken rounded sibling. **Disclosed residual, not fixed by this note:** a rotated (`angle`-bearing), rounded, cut piece therefore keeps its ORIGINAL uncut corner radius rather than squaring its cut edge — the specific three-way combination of rotation, rounding, and a break gap is the one case this fix does not resolve, to avoid corrupting the rotation anchor instead. Interactive (WASM) hit-testing of a squared split piece remains covered by the general "interactive rendering under a break remains a documented non-goal" line above and is tracked as a separate follow-up, not by this note. Behavior-changing only for a rounded (`corner_radius > 0`) bar/rect/annotation-rect piece a gap actually cuts; unbroken charts, zero-radius marks, and pieces a gap does not touch are byte-identical, and no golden moves (no golden sets `corner_radius` under a break).

> **Dated note (2026-09-13, wave-2 break-axis #230-3 — a glyph's move/hide test now uses its exact data anchor, not a bbox measured off its own drawn geometry):** the batch-H note above's disclosed residual — "a mark sitting within a few px of a gap edge can resolve to 'hidden' due to bbox-anchor imprecision for asymmetric glyphs" — is RETRACTED and fixed. `BreakRemap::remap_glyph_node`'s anchor (the position a `MoveGlyph`/`MoveGlyphAndConnector` batch — point, tick, text, label, image — moves/hide-tests on) is now the row's own scale-mapped `(x, y)` position, threaded from batch build (`MarkBatch::glyph_anchors`, one entry per emitted node, cleared before the scene is packed/serialized since no downstream reader needs it), rather than the midpoint of the node's own drawn extent. The old bbox-midpoint model was exact only for a shape symmetric around its data position (`Circle`/`Square`/`Diamond`/`Cross`/`VLine`/`HLine`, and a single-coordinate node like `Text` before an offset is baked in) — every point-mark break golden uses circles, so none moved. It was wrong for `TriangleUp`/`TriangleDown`, whose bbox midpoint sits a quarter of the triangle's height off the true anchor: measured, a `TriangleUp` hidden at a y position where a `Circle` at the exact same data value still rendered. It was also wrong for `mark_text`'s `dx`/`dy` offsets, which `text.rs` bakes into the node's own `x`/`y` before the break ever sees it — the old model measured and hide-tested the OFFSET position instead of the row's true data position, so a label could hide (or wrongly survive) based on where its offset happened to land rather than where its data point did. Both are fixed the same way: the anchor is recorded once, before any shape-specific drawn geometry (a point's radius, a text label's `dx`/`dy`) is added on top of it, and the glyph translates by that anchor's own displacement under the break — so an asymmetric shape keeps its drawn size and a text label's `dx`/`dy` translates rigidly with its anchor, in both directions (a label can now correctly hide on an in-gap data position even when its offset lands outside the gap, and correctly stay visible when its offset alone would have landed inside it). `remap_glyph_node`'s other callers are unaffected: a `mark_label` leader-line connector (`remap_label_node`'s `Line` branch) already placed its two ends independently and never read this anchor; an arrowhead's rigid fallback (`remap_arrowhead`) is a different mechanism entirely (rebuilt from the shaft's own placed endpoints), untouched by this change. A `MoveGlyph`/`MoveGlyphAndConnector` batch missing this sidecar (unreachable from every real builder) fails loudly under a debug build rather than silently degrading. Behavior-changing only for an asymmetric point shape (`TriangleUp`/`TriangleDown`) or a `dx`/`dy`-offset `mark_text`/`mark_label` node within a few pixels of a gap boundary; every symmetric-shape glyph, every glyph away from a gap boundary, and every existing break golden (all circles) is byte-identical, and no golden moves.

> **Dated note (2026-09-12, wave-2 break-axis #228 — an extent annotation is CLIPPED at a gap like the mark geometry it mirrors; only what cannot be split honestly is dropped):** an annotation with a data-space extent on a broken axis is no longer dropped whole the moment a gap touches it. This SUPERSEDES the batch-H note above's final annotation clause — "Suppression is TOTAL for extents under this contract: a band that brackets every visible segment (e.g. `annotate_rect` spanning the whole axis as a background shade) is dropped whole rather than clipped into one piece per segment, which is what the equivalent MARK geometry does — unifying the two answers is the filed follow-up, not this batch" — and closes that follow-up. The answer is now stratified by what the annotation's geometry MEANS, the same question the mark side has always asked (`BreakPlacement`):
>
> - **`rect` and `span`** (`annotate_rect`, `ann.rect`, `ann.span`) are CLIPPED into one piece per visible segment, through the same machinery a bar's height goes through, and SILENTLY: a clipped annotation rendered, so there is nothing to warn about. `annotate_rect(y1=0, y2=1000)` on a `[0, 1000]` axis broken at `(150, 850)` now paints two pieces, each stopping exactly at its gap edge, where before it vanished with a warning. A `span`'s optional label anchors on the midpoint of the LARGEST surviving piece (ties keep the first, in segment order), because a straddling span's pre-clip midpoint is inside the gap — the one place its label must not be drawn.
> - **`line` and `arrow`** (`annotate_hline`/`annotate_vline`'s rule primitive, `ann.line`, `annotate_arrow`, `ann.arrow`) SPLIT per visible piece through the same vertex splitter `mark_line`/`mark_segment` use (#230-1's mechanism, not a second implementation): each piece ends on the gap edge at the point the stroke actually crosses it, with the cross-axis coordinate interpolated there. An arrow's head rides the piece that reaches its TIP, rebuilt on that piece's own endpoints so it keeps its drawn size and points along the piece a reader sees it attached to; a tip inside a gap leaves no piece to terminate the head on, so the head alone drops — the shaft's own visible pieces still render, truncated at the gap edge they end on, exactly as a `mark_segment(arrow=True)` shaft's pieces do for the same case (#230-1). This is clip-fate geometry, not a silent drop of the annotation: what is missing is the part of the drawing that was never visible, the same way a bar clipped at a gap edge is not "silently" missing the sliver inside the gap.
> - **`bracket`** (`ann.bracket`) stays suppress-whole, and this is the PERMANENT contract rather than a residual: its two endpoint tips are what say where the bracketed range begins and ends, so cutting the baseline at a gap edge would either strand the tips at positions the data never had or invent new tips at the cut. A bracket reaching across a gap asserts a continuity the broken axis cannot carry, so it is dropped with the warning naming the axis — where a `rect` of the same extent clips.
> - **The point-anchored kinds** (`text`, `callout`, `image`) keep the plain membership test, unchanged: the coordinate is either inside a gap or it is not. A callout's background rect is a TEXT BOX — drawn size around an anchor, never data — so it is never clipped even when it happens to straddle the indicator band in pixels.
>
> Suppression survives as the narrower fact it should always have been, with its warnings and their per-axis rollup (#240-3) unchanged in shape: an extent annotation is dropped only when NOTHING OF IT SURVIVES the gaps, or when its kind admits no honest split. The endpoint-first shortcut is retired for the four clip-routed kinds — an extent with an endpoint inside a gap but a non-empty surviving portion is CLIPPED (`annotate_rect(y1=100, y2=400)` under a `(150, 850)` gap renders its visible 100–150 band instead of vanishing) — and the endpoint test now decides only WHICH rollup a dropped annotation lands in, since "a coordinate of yours is in the gap" (`AnnotationSuppressedByBreak`) and "your interval crosses the gap" (`AnnotationSpansBreakGap`) remain different, equally true facts. The knife-edge case where a clip-routed extent still lands in the second bucket is an extent whose bounds are exactly a gap's bounds: both endpoints are visible values, yet no length of it is. One `annotate_*` call that desugars into several primitives (`annotate_hline(label=...)` emits a `line` plus its own label `text` at the same y) has each primitive routed on its own, so a fate can legitimately differ between them — a full-width `line` can clip on an x break while its label, anchored inside that gap, point-suppresses, and the warning then names only the kind actually dropped. Behavior-changing for charts whose extent annotations meet an effective gap (strictly more renders than before, strictly fewer warnings); unbroken charts, ineffective-gap breaks, and annotations clear of every gap are byte-identical, and no golden moves (no golden carries an annotation).

> **Dated note (2026-09-12, wave-2 break-axis #230-5 — a keyed mark split 1→N by a break suffixes its piece keys instead of duplicating the source key; spec §6):** when `BreakAxis` clipping splits one keyed node into N pieces, `ClippedNodes::realign_keys` now suffixes every piece after the first with `{key}#{i}` (`i` starting at 1) instead of copying the pre-clip `keys` sidecar verbatim onto every piece — the prior verbatim-copy behavior made WASM's key-based transition pairing (`ferrum-wasm`'s `transition.rs`, which treats key equality as row identity) treat N split pieces as one logical row, degrading the whole batch's keyed pairing to `IndexZip`. The first piece of a split keeps the original key unsuffixed; `i` is not simply the piece's position within its run: a chart broken on two axes re-suffixes a batch's `keys` sidecar once per axis (`scene_build.rs`'s `break_remaps` loop calls `realign_keys` once per axis pass), so the second pass's input is already-suffixed output from the first, and naively restarting the `#1, #2, ...` counter each pass can mint a suffix an earlier pass already used. `i` is therefore the smallest integer such that `{key}#{i}` is absent from both the sidecar the current pass received and every key already emitted earlier in that same pass — identical to the plain sequential count whenever nothing collides (every currently-golden/pinned single-break chart), and additionally closes a collision shape where a user-supplied key already contains `#` (e.g. `"a#1"`). An unkeyed batch (`keys` is `None`) is untouched. Behavior-changing only for a keyed mark batch a break actually splits 1→N; an unkeyed batch or a batch no gap touches is byte-identical, and no golden moves (no golden sets `key=` under a break). Interactive rendering under a break remains a documented non-goal; this note only fixes what key each piece CARRIES, not WASM grouped-key pairing across a break (a logged follow-up, not built here).

> **Dated note (2026-09-13, wave-2 break-axis #230-7 — a `BreakAxis` applies to EVERY facet panel, and its diagnosis is spoken once for the chart):** the batch-H note above's disclosed residual, "a faceted chart applies its `BreakAxis` to panel 0 only", is RETRACTED and fixed. The whole structural build was gated on the first panel, so a faceted chart broke panel 0 and left panels 1..N on the unbroken scale: measured on a two-panel `.facet(col=…)` bar chart with `gap=(150, 850)`, panel 0's `y=900` bar clipped into two pieces at the compressed positions while panel 1's was drawn full height on the linear scale, under ONE shared y axis whose tick labels described only panel 0's geometry, with no warning that half the chart disagreed with its own axis. Breaks are now built per panel: every panel lays the chart's same data-space gaps over its OWN plot rect and its own scale pixel range (the #226 range fix above, now read per panel), so identical data in two panels places identically, and every panel draws its own break indicator inside its own rect. `Inset`, the other structural spec, stays chart-level and is still built once against the reference panel: an inset's clip-path ids are namespaced per inset within one SVG document, so repeating it per panel would emit colliding ids rather than N insets. One ordering consequence of that split: on a chart declaring BOTH a `BreakAxis` and an `Inset`, break indicators are now emitted before inset bodies rather than interleaved in declaration order, so an inset paints over an indicator it overlaps (axis chrome under embedded content). The DIAGNOSIS cadence is deliberately not per panel. `BreakAxisMergeConflict` is a pure function of the spec, and `BreakAxisIneffective`/`BreakAxisOnNonlinearScale` are functions of the spec and the resolved domain. In the common case — a shared-scale facet, the default — every panel resolves the same domain and reports the same payload, so each distinct fact is reported ONCE per render however many panels observe it: a faceted chart's warning text is identical to the single-panel chart's. The ledger does not derive that agreement from the facet's resolve mode; it compares each reporting panel's payload against the first one it stored, because a per-panel resolved domain can diverge panel to panel for reasons the resolve mode alone does not settle. Two shapes are pinned (Rust and Python) where it does: `Chart.share_scale(x=/y="independent")`, where each panel resolves its own domain by request and a gap can be effective on one panel and miss another's entirely — there the single `BreakAxisIneffective` report names the panel it was measured on and the fraction of panels that share it (`"no gap overlaps the axis domain (10, 20) (facet panel 1); this holds on 1 of 2 panels, and the break applies on the rest"`), rather than going silent on the panels the gap missed or repeating itself once per panel; and a **default-resolve** facet whose panels still resolve different domains for a reason unrelated to `share_scale` — a bar mark's zero-baseline domain resolves per panel even under the default shared resolve, so two panels with different group maxima report different `(0, max)` domains without any `share_scale` call at all (reproduced live: `.facet(col="g")` with no `share_scale`, groups `p` (max `900`) and `q` (max `30`), `gap=(2000, 3000)`, which overlaps neither domain — the report names `facet panel 0` exactly as the independent-scale shape does; the per-panel bar-domain resolution itself is pre-existing and unchanged by this task, verified byte-identical against an unbroken chart on released `ferrum-viz==0.23.0`, so what's new here is the ledger noticing the divergence and the report naming it honestly, not the domain resolution). The panels the gap DID reach still break; a partial report is a qualification, not a suppression. The same qualification also fires when a gap misses EVERY panel but the reporting panels' payloads genuinely differ: the pre-fix gate qualified only on a SHORT reporting count (`panels_reporting < panels`), so a gap missing every one of two panels with different domains flushed the first panel's domain verbatim and unattributed, as if it spoke for the chart — reproduced live with groups `p` (`10, 900`) and `q` (`10, 30`), `share_scale(y="independent")`, and `gap=(2000, 3000)` (which overlaps neither domain): the report now reads `"no gap overlaps the axis domain (0, 900) (measured on facet panel 0); other panels reported this fact with a different value, so this is panel 0's, not every panel's"` rather than the bare, unattributed `"no gap overlaps the axis domain (0, 900)"`. The qualifier states only what the ledger checked — that another panel's report disagreed — and does not name a cause; it does not say the panels are independently scaled, since the divergence is reachable under the default shared resolve too. The fix keys the qualification on payload DIVERGENCE across reporting panels, not merely on a short count, and the two shapes read differently: a short count still says "applies on the rest" (some panels broke), while an all-report divergence never claims a "rest" the break applied on, since every panel missed its own gap. `BreakAxisOnNonlinearScale` deliberately does NOT get this same partial-panel qualification: unlike `BreakAxisIneffective`, it is a claim about the SPEC itself (a log/pow/symlog axis's inverse is not the break's linear reverse map), equally true on every panel the break reaches regardless of how many panels observed it, so naming a subset would tell the reader nothing actionable. One consequence: a free-scale facet whose gap is effective on only some panels can render a flat, unqualified `BreakAxisOnNonlinearScale` warning beside a qualified `BreakAxisIneffective` one in the same warning list — reproduced live with `share_scale(y="independent")`, a log-scaled y, and `gap=(150, 850)`: the nonlinear warning names no panel at all, while the ineffective warning reads `"...no gap overlaps the axis domain (10, 20) (facet panel 1); this holds on 1 of 2 panels..."` right beside it. Annotation suppression is unchanged and cannot diverge per panel by construction: chart-level annotations are built in ONE place, against the reference panel's scales, so an in-gap annotation on a faceted broken chart is suppressed everywhere and reported once. Composite/concat leaves are unaffected (each leaf is its own scene with its own break build and its own warnings, so an hconcat of two leaves whose annotations both drop still warns twice, once per leaf). Behavior-changing only for a FACETED chart carrying an effective `BreakAxis`; single-panel broken charts, unbroken charts, and ineffective-gap breaks are byte-identical, and no golden moves (no golden facets under a break). That byte-identity claim covers SVG output only: break-axis diagnostics now flush after the whole panel loop rather than interleaving at panel 0's structural point, so on a single-panel broken chart they now follow the annotation-suppression and mark-remap warnings that previously preceded them in the emitted warning list — no test or downstream consumer reads warning order (`reason` text is read nowhere but the warning's own `Display`/serde), but the sequence itself is a disclosed behavior change.

> **Dated note (2026-09-20, wave 6A task 10, `#229`(2) — the wire's `break_axis.axis` is a two-token vocabulary, and a third token now refuses instead of silently breaking the x axis):** `BreakAxis.axis` is typed end to end in Rust (one shared axis enum, also carrying `ExtentAlignment`'s axis), so the `structural[]` item's `axis` key accepts exactly `"x"` and `"y"`. A hand-built `chart_config` dict naming any other token is REFUSED with a typed `ValueError` reading, verbatim, `chart config: structural[0] (break_axis).axis: axis must be one of ['x', 'y']; got 'z'` — where it previously deserialized as a plain string, failed every downstream `axis == "y"` test, and rendered as an **x** break, a chart the caller did not ask for, with no diagnostic. The refusal comes from the **array-item value gate** (`binding.rs::validate_break_axis_axis_value`), the same gate the sibling `gaps` refusals come from, so it names the item and its index and mirrors Python's own `validate_choice("BreakAxis.axis", ..., ("x", "y"))` wording. That gate runs on the raw dict BEFORE the JSON bridge, and covers both spellings a unit variant has on the wire — the string (`"axis": "y"`) and the externally-tagged single-key map (`"axis": {"y": null}`), which renders identically to it. Its coverage of the map form is quantified over `json.dumps`, the transform the bridge runs: that encoder coerces exactly five key types into JSON keys (`str`, `float`, `bool`, `None`, `int`) and raises `TypeError` for any other, so the gate mirrors those five spellings — including `true`/`false` for a `bool` key, `null` for `None`, `NaN`/`Infinity`/`-Infinity` for a non-finite `float`, and an `int` at arbitrary precision — and **no out-of-vocabulary token reaches serde through the bridge, in any spelling `json.dumps` can produce**. What serde keeps is the SHAPE errors, which it reports better than the gate could (`invalid type: integer \`7\`, expected string or map`; `invalid value: map, expected map with a single key`), and the unknown-variant backstop for a consumer that reaches `Deserialize` without this gate in front of it. Note the key gate is a third, separate mechanism: it lists `axis` among `break_axis`'s accepted KEYS (`STRUCTURAL_BREAK_AXIS_KEYS`) and refuses only unknown keys, with its own pinned `chart config: unknown key '<k>' in <section>; accepted: ...` text this change does not touch. **Wire-level only, and unreachable from the public Python API:** `ferrum.BreakAxis` is a frozen dataclass whose `__post_init__` has always run `validate_choice("BreakAxis.axis", ..., ("x", "y"))`, and no override-prefix rule targets the `structural` section, so the sole route to the new refusal is a dict passed directly to the private `ferrum._core` render entries. Every rendered byte and every break-family warning and refusal message is unchanged for every well-formed chart, `BreakAxis(axis="y")`'s quoted axis token included.

> **Dated note (2026-09-25, wave 6C, N4 — `BreakAxisIneffective` states the domain low to high, even on a reversed axis):** the warning's domain pair now reads `(min, max)` regardless of the scale's own declared order, so a reversed y domain of `[100, 10]` with unmatched gaps reads `"no gap overlaps the axis domain (10, 100)"` rather than the previous `"... (100, 10)"`. Only the message text moves; gap effectiveness and every other break behavior is unchanged.

---

### 3.13 Themes

A `Theme` is an immutable value object. All properties are optional; unset properties fall back to the Ferrum defaults.

```
Theme(
    # Canvas
    background=None, width=None, height=None, padding=None,

    # Typography
    font_family=None, font_size=None, font_weight=None, font_color=None,
    title_font_family=None, title_font_size=None, title_font_weight=None,
    title_color=None, title_anchor=None, title_offset=None,
    label_font_family=None, label_font_size=None, label_color=None,

    # Grid
    grid=True, grid_color=None, grid_dash=None, grid_width=None, grid_opacity=None,

    # Axes
    axis_line=True, axis_line_width=None, axis_line_color=None,
    tick_size=None, tick_width=None, tick_color=None,
    cull_threshold=None,  # int; minimum pixel gap between adjacent tick
                           # labels — see the 2026-09-03 and 2026-09-20 dated
                           # notes below. Default 8 (Rust).
    cull_labels=None,      # bool; may the layout engine DROP tick labels to
                           # reach that gap? Independent of the magnitude —
                           # see the 2026-09-20 dated note below.

    # Marks
    point_size=None, point_opacity=None,
    line_stroke_width=None, bar_corner_radius=None,
    area_opacity=None, opacity=None,

    # Colors
    color_scheme=None, mark_color=None, background_color=None,

    # Legend
    legend_orient=None, legend_direction=None, legend_title_font_size=None,

    # Spacing
    axis_title_padding=None, column_padding=None, row_padding=None,
)
```

> **2026-05-11 (Themes-T1):** Every key listed in the `Theme(...)` block above
> is now plumbed end-to-end. The Python `Theme` class validates unknown
> kwargs at construction time (raises `ValueError`); the Rust `theme_from_dict`
> binding likewise rejects unknown keys. Spec key aliases (e.g. `background`
> ↔ `background_color`) are accepted by both. Fallback chains (`title_color
> → font_color`, `label_color → font_color`, `title_font_family →
> font_family`, `label_font_family → font_family`) are resolved Python-side
> in `Theme.to_theme_inputs_dict()` so the Rust binding sees a fully-populated
> dict. Render-side consumption of the newly-plumbed keys lands in Themes-T2
> through T4; defaults remain at their pre-T1 values in this sub-phase.

> **2026-05-11 (Themes-T2 → T4):** Render-side consumers now read every
> plumbed key. New `ThemeInputs::default()` ships an Observable Plot-flavored
> visual identity: `mark_color="#4C78A8"` (tableau blue, was Okabe orange
> `#E69F00`), faint visible grid (`grid_color="#DDDDDD"` width `0.5`, was
> `#EEEEEE` width `1.0`), left-aligned semibold title (`title_anchor="start"`,
> `title_font_weight="600"`), three-stop text color ramp (body `#222` / label
> `#555` / axis `#888`), `point_size=36` (was `30`), `padding=16`,
> `column_padding=row_padding=12`. `theme.color_scheme` defaults flip from
> `okabe_ito` (§3.6) to `tableau10` — `okabe_ito` remains accessible via
> `Theme(color_scheme="okabe_ito")`. `theme.grid` now actually emits
> gridlines via `axis::draw_grid` (was a no-op before T3). `theme.axis_line:
> bool` suppresses the axis stroke when `False`. Font family is kept as
> `"Inter"` (bundled in `crates/ferrum-core/src/render/embed_font.rs`) rather
> than the design spec's `"DejaVu Sans"` — Inter is deterministic across CI
> hosts, DejaVu Sans would resolve through the system font path. The 8
> built-in themes are rebuilt to use the newly-plumbed keys; each is visibly
> distinct from the others on the same chart (see
> `tests/goldens/theme_gallery/`).

> **2026-06-22 (D-THEME-1, T2.1):** The Python `Theme._KNOWN_KEYS` and color-key
> sets are now *derived* from the Rust key manifest (`ferrum._core.theme_known_keys()`
> / `theme_color_keys()`) so the Python and Rust accepted-key contracts cannot
> drift. The accepted set now includes the per-level grid keys (`major_grid_*`,
> `minor_grid_*`, `minor`) in `Theme(...)` as well (previously Python rejected
> them though Rust accepted). The resolved-dict method is `Theme.to_spec_dict()`
> (the older spec mention of `to_theme_inputs_dict()` is renamed). The
> title/label → body-text fallback chain is now **complete**: in addition to the
> four documented above it resolves `title_font_weight → font_weight` and
> `title_font_size → font_size` (every `title_*`/`label_*` key that has a
> body-text counterpart). There is no public `label_font_weight`/`label_font_size`
> key — the public `font_size` key *is* the label/body font size (it routes to
> the Rust binding's `label_font_size`, the same way `background` is the public
> alias for `background_color`). CSS shorthand hex (`#rgb`/`#rgba`) is expanded
> by the Rust color parser (`from_hex_str`); the prior redundant Python-side
> expansion was removed.

> **Dated note (2026-09-03, batch B task 3 — `cull_threshold` is a real pixel-gap minimum, and culling is reachable):** `cull_threshold` is the minimum pixel gap the layout engine keeps between adjacent tick labels; when the natural label set would pack tighter than that, labels are dropped until the survivors clear the gap. `cull_threshold=0` disables culling entirely. Before this task, culling was largely **unreachable**: a floating-point artifact (`cos(±90°)` evaluates to about `6.12e-17` in `f64`, not exactly `0`) made a vertically-rotated (`-90°`) label's computed footprint collapse to near-zero, so the rotation stage always reported "fits" and skipped straight past the cull/elide stages regardless of `cull_threshold`. The footprint calculation now includes the label's own line-height contribution at any rotation angle, so a `-90°` label gets a real fit test and dense categorical/tick axes now visibly cull. `cull_threshold` folds into **every** layout cascade stage as a minimum-gap budget (not just the final stage), so wrapping and packing degrade uniformly under it (shared word-balanced wrapping, never a one-word-per-line collapse) and elision (mid-label truncation with an ellipsis) only fires where truncating the label can actually free enough width to help — an infeasible truncation leaves the label intact rather than truncating for no gain. After culling removes some labels, the survivors re-select their own presentation (flat / rotated / wrapped / elided) at the new, sparser density — a survivor that no longer needs rotation renders flat rather than keeping the rotation the denser pre-cull set required. At `cull_threshold=0` the wrap/pack path is byte-identical to before this task (the verbatim pre-existing accumulator loop).

> **Dated note (2026-09-20, wave 6A task 9 — `cull_threshold` splits into gap magnitude and cull permission, and it now reaches an independent-x facet axis):** Two changes, one additive and one a bug fix. **(1) The split.** `cull_threshold` used to answer two questions with one number: how much gap to require, and whether the layout engine may drop labels to reach it. `0` meant both "require no gap" and "never drop a label", so "keep an 8 px gap driving the flat / wrap / font-shrink / rotate stages, but never drop a label" was unspellable. A new boolean Theme key, `cull_labels`, now carries the permission on its own. Every pre-existing spelling is unchanged in meaning: `cull_threshold=0` still means gap 0 AND never cull; any positive `cull_threshold` still means that gap AND culling permitted; `fm.Axis(label_overlap="true")` still abandons the cascade entirely (independent-x faceted *output* does move for some of these spellings — see the fix below). What the split adds is the other two corners: `Theme(cull_threshold=30, cull_labels=False)` requires a 30 px gap of every presentation stage (S0–S3; S5's elision fallback is deliberately exempt, see the code comment at its fit test) and then keeps every label rather than dropping one — every label survives regardless of density, but WHICH stage the survivors land in does not: sparse enough and the gap is satisfiable early (flat or a shallow rotation), dense enough and the cascade falls through to its last resort, rotated -90° and possibly overlapping (elision, the nominal step after that, is gated on freeing width a -90° truncation can't actually reclaim, so it essentially never fires). Both arms are pinned by `tests/test_label_gap_policy.py::test_cull_labels_false_stage_depends_on_density`. And `Theme(cull_threshold=0, cull_labels=True)` drops labels only where they genuinely overlap, with no extra gap. The default is unchanged (gap 8, culling permitted) and its magnitude and mode are now two independent constants in Rust, so the still-open question of what the default gap should be can be settled without another change to this key surface. **(2) The fix.** On a facet whose x scale resolves independently (`.facet(col=…).share_scale(x="independent")`), the per-panel x axis was rebuilt with the compile-time default gap instead of the theme's, so `Theme(cull_threshold=…)` was entirely INERT there: a 60-category axis rendered the same label count at `cull_threshold` 0, 8 and 40, while the shared-x and un-faceted paths tracked the setting. That axis now reads the theme like every other axis, and an independent-x facet and a shared-x facet over the same data cull identically. Independent-x faceted output is the only output this task allows to move, and only where the setting actually changes which presentation the cascade lands in — a sparse independent-x facet whose presentation `cull_threshold` cannot change stays byte-identical; everything else in the corpus is byte-identical outright. The y axis was never affected — culling is an x-label concern.

**Built-in themes** (`ferrum.themes`)

| Name | Description |
|---|---|
| `default` | Ferrum defaults: light background, subtle grid, OKabe-Ito |
| `minimal` | No grid lines, no axis lines, generous whitespace |
| `dark` | Dark background (#1a1a2e), light text, neon-adjacent accents |
| `publication` | Print-ready: no background, high-contrast, Tableau10 |
| `economist` | Red accent, left-axis labels, horizontal grid only |
| `fivethirtyeight` | Grey background, red/blue diverging, no axis lines |
| `solarized_light` | Solarized palette |
| `solarized_dark` | Solarized dark palette |

**Theme composition:**

```python
my_theme = fr.themes.minimal.update(
    font_family="Berkeley Mono",
    grid=fr.Grid(major=True, minor=False, color="#f5f5f5"),
    title_font_weight="bold",
)
```

`Theme.update(**kwargs)` returns a new `Theme`; source is unchanged.

**Global default** (process-scoped, not module-level mutable):

```python
ferrum.set_default_theme(my_theme)  # returns a context manager
# or
with ferrum.theme_context(my_theme):
    ...
```

> **2026-05-10 (Phase 8a):** `set_default_theme(theme)` is implemented as a
> contextvars-backed setter that returns a context manager. Per-chart
> `Chart.theme(t)` always overrides this default. CLAUDE.md §"Hard
> constraints" documents this as the single sanctioned exception to
> "no global mutable state."

---

### 3.14 Figure-Level Functions

> **2026-05-10 (Phase 9):** All 8 Group A figure-level functions —
> `displot`, `catplot`, `lmplot`, `residplot`, `pairplot`, `heatmap`,
> `clustermap`, `jointplot` — land in Phase 9 with every parameter
> advertised in this section honored (no `NotImplementedError`,
> no silent warn-fallbacks). Each function returns either a `Chart` or
> a compound view (`JointChart`, `RepeatChart`, `ClusterMapChart`) whose
> `.spec` / `.charts` / `.expand()` is a fully-formed object that
> round-trips through `ChartSpec.from_json`. Group B (model-diagnostic
> figure-level functions: `roc_chart`, `pr_chart`,
> `confusion_matrix_chart`, `calibration_chart`, `gain_chart`,
> `lift_chart`, `residuals_chart`, `importance_chart`, `shap_chart`,
> `learning_curve_chart`, `validation_curve_chart`,
> `cluster_diagnostics`, `decision_boundary_chart`,
> `discrimination_threshold_chart`, `parallel_coordinates_chart`,
> `class_prediction_error_chart`, `pca_scree_chart`, `rank_chart`,
> `alpha_selection_chart`, `intercluster_distance_chart`,
> `cv_scores_chart`) remains scheduled for Phase 10, alongside
> `ModelSource` and the model-diagnostic marks they depend on.

> **2026-05-11 (Phase 10 — Model Diagnostics):** All Group B figure
> functions ship in Phase 10 with every spec parameter implemented.
> Multi-model overlay is supported via three entry points:
> (a) pass a dict positional argument
> (`roc_chart({"a": model_a, "b": model_b}, X, y)`),
> (b) pass `compare=` kwarg
> (`roc_chart(model_a, X, y, compare={"alt": model_b})`), or
> (c) construct a `ComparedModelSource` explicitly via
> `ModelSource.compare(...)` and pass it positionally. In all three
> cases each per-model DataFrame is concatenated with a `model: Utf8`
> column that downstream chart builders route to `color="model"`.
> `random_state` affects compute on figure functions backed by
> ModelSource methods that consume randomness (see §3.1 note); it is a
> no-op on the rest. `parallel_coordinates_chart` and `rank_chart` do
> not need a model — both accept a raw DataFrame / 2D array directly.

> **2026-06-27 (compare= aggregate rendering, #35):** `compare=` now
> renders **small multiples** — one panel per model, composed as a
> `ConcatChart` — for the previously-gated aggregate diagnostics.
> Affected families: the explanation family (`importance_chart`,
> `shap_beeswarm_chart`, `shap_bar_chart`, `shap_waterfall_chart`,
> `shap_chart`, `pdp_chart`); the model-selection family
> (`learning_curve_chart`, `validation_curve_chart`, `cv_scores_chart`,
> `alpha_selection_chart`); the regression aggregate family
> (`cooks_distance_chart`; `residuals_chart` with a multi-panel layout;
> `prediction_error_chart` with `ci=` or `reference_band=`, each band
> computed per-model from that model's residuals only, never pooled);
> and the source-based clustering charts (`pca_scree_chart`,
> `intercluster_distance_chart`, `silhouette_chart`, `manifold_chart`)
> with independent per-panel scales. Charts whose single-model output
> is itself composite (`pdp_chart`, multi-panel `residuals_chart`) nest:
> the outer `ConcatChart` children are per-model composites, each
> carrying the model name as a figure-level title. The two sweep-based
> clustering charts (`cluster_diagnostics`, `elbow_chart`) remain
> excluded — they sweep one clusterer class over *k* on a feature
> matrix and wrap no per-model `ModelSource`; algorithm/method
> comparison is tracked in #43.

> **2026-07-10 (dodge-by-model compare= layout, #42):** for the
> dodge-eligible diagnostics, `compare=` now renders a **single
> shared-axis panel** with marks grouped (dodged) by model — the
> canonical seaborn/yellowbrick comparison plot — instead of the
> 2026-06-27 small-multiples `ConcatChart`. **Landed:** `importance_chart`
> returns one `Chart` whose bars are dodged by model within each feature
> band, with a model legend and error rules / value labels dodged
> alongside their bars; features are ranked once globally across models
> (mean *absolute* importance, descending — mirrors shap_bar's
> `_shap_order_features` magnitude ranking) and the shared top-`top_k` set
> is used for every model; the value-axis domain is computed over the
> combined frame.
> `orient="horizontal"` (default) uses the vertical desugar form plus
> `CoordFlip`; `orient="vertical"` renders the dodged layout directly.
> `shap_bar_chart` (`per_class=False`) likewise returns one dodged `Chart`:
> per-model SHAP values are stacked before `_shap_order_features` ranks the
> pooled values once, globally; the shared top-`max_display` feature set is
> aggregated per model, and the always-horizontal single-model layout uses
> the same vertical-desugar-plus-`CoordFlip` idiom (`mark_shap_bar` gained
> an internal `orient=`/`color_field=` pair mirroring `mark_importance`).
> `shap_bar_chart(per_class=True)` keeps the 2026-06-27 small-multiples
> `ConcatChart` (class is a competing facet dimension). `cv_scores_chart`
> (`kind="box"`/`"strip"`, the default) likewise returns one dodged `Chart`
> (spec D3): `split` (`train`/`test`, filtered by `split=` exactly as
> today) stays the shared categorical axis, and each split band gets one
> box/strip group per model, dodged and colored by model — the seaborn
> `hue` convention, since `split` already has ≤ 2 levels and reads as
> x-categories (facet-by-split would reintroduce panels to solve a problem
> two bands don't have). `kind="strip"` drops the single-model jitter
> under dodge (position adjustments are not composable — same rule
> catplot already records); `desugar_cv_scores` gained a `color_field=`
> parameter (`mark_cv_scores(color_field=...)`) that threads into
> `desugar_boxplot`'s groupby for `"box"` and replaces the jittered point
> layer's color for `"strip"`. `kind="bar"` keeps the 2026-06-27
> small-multiples `ConcatChart` (fold x split x model is three grouping
> dimensions — no coherent single dodge). The remaining explanation /
> model-selection charts keep the 2026-06-27 small-multiples layout.
> Single-model output is unchanged.

> **2026-09-13 (wave 3 T5, #213 — `discrimination_threshold_chart(compare=)`
> joins the small multiples):** `discrimination_threshold_chart` and
> `DiscriminationThresholdVisualizer(compare=)` now render **small
> multiples** — one `ConcatChart` panel per model, shared x/y scales,
> titled with the model name — instead of the single pooled panel they
> produced before. This is the one classification-curve chart that composes
> rather than overlaying by `color="model"` (the 2026-05 classification
> default), because its colour channel is already spent on the four-metric
> legend; with N models overlaid there is no free channel left to tell the
> models apart. **Breaking, and the geometry moves, not just the
> annotation.** The builder's `unpivot` to long form listed neither `model`
> in `index=` nor among the resolved metric columns, so polars dropped the
> column outright: both models' sweeps stacked into one frame with nothing
> distinguishing them, and the line mark drew each metric as a polyline that
> traversed the threshold axis once per model — an interleaved zig-zag, not
> an overlay. Measured on the pinned `binary_logistic` fixture vs. a
> `GaussianNB` rival at `n_thresholds=20`: the old single panel held 160
> rows over only 80 distinct `(threshold, metric)` pairs (every pair
> duplicated); each panel now holds 80 rows, one per pair. The
> F1-optimum annotation and the `threshold_line` rule were computed by a
> single `arg_max` over that pooled frame, so they reported whichever model
> won globally; each panel now computes them from its own model's sweep,
> because `_compose_compare` builds every panel by re-entering the
> single-model builder. Measured: base 0.918 @ t=0.42, rival 0.886 @ t=0.53
> — two distinct labels where there was one. This also settles the render
> half of the `compare=` metric-semantics note (F-L10-07) below:
> `DiscriminationThresholdVisualizer(GaussianNB, compare={"rival": logistic})`
> reprs `best_threshold=0.5263, best_f1=0.8865` (its base model, restricted
> by `_base_model_rows` since batch G) while rendering `max F1 = 0.918 @
> t=0.42` (the rival's). The `"base"` panel now annotates exactly the pair
> the repr prints. The single-model path (no `compare=`) is unchanged, and
> both single-model discrimination-threshold goldens are byte-identical.

> **2026-07-02 (explicit scales survive composite-mark desugar, #45):**
> an explicit `scale=` on a chart-level positional channel (`x`/`y`) now
> propagates onto the layers a composite mark desugars into — previously
> it was silently dropped for every composite mark (box, violin,
> cv_scores, letter-value, …). Rule: a layer channel with no scale
> inherits the chart-level scale; a layer scale without a `domain` (e.g.
> validation_curve's log x) merges in the domain only, keeping its own
> `type`/`range`; a layer scale that already carries a `domain` (e.g.
> SHAP's mark-computed `x_scale_domain`) always wins. Consequence for
> `compare=`: shared-scale resolution now visibly applies to
> composite-mark panels with flat data — e.g. `cv_scores_chart`
> `compare=` panels render one union y-axis (the #45 report case).
> Grid-composite children (multi-panel `residuals_chart` under
> `compare=`) still do not share; that lands with the composite-render
> unification (Phase B spec, 2026-07-02).

Figure-level functions return `Chart` or compound view objects. They handle data reshaping, faceting, axis labeling, and legend placement automatically. All accept `theme=` and `**encode_kwargs` to override defaults.

> **Dated note (2026-10-11, #135 close: a figure's size arguments are checked at its entry, by their own names):** `displot`, `catplot`, `relplot` and `pairplot` check `height=` (a finite pixel value above 0) and `aspect=` (a finite number above 0), `jointplot` checks `height=`, and `clustermap` checks that `figsize=` is a `(width, height)` pair of finite pixel values above 0, before any size is derived from them. A malformed value (0, negative, NaN, infinity, a `bool`, a non-number) is refused with a `ValueError` in the figure's voice: `displot(height=...) must be positive; received 0`, `displot(aspect=...) must be a finite number a float can hold; received nan`, `clustermap(figsize=...) must be a (width, height) pair; received (300,)`. Before: `height=0` and `aspect=0` rendered silently at the default size, NaN reached the renderer, and `figsize=(300,)` raised `IndexError`. A size that is 0 only because the data is empty (row facets over an empty frame) is left unset, so the empty chart still renders.

#### Distribution

```
ferrum.displot(data, *, x=None, y=None, hue=None, col=None, row=None,
               kind="hist",          # "hist" | "kde" | "ecdf" | "rug"
               fill=True, cumulative=False, log_scale=False, stat="count",
               bins="sturges", bandwidth="scott", bw_adjust=1.0,
               multiple="layer",     # "layer" | "stack" | "fill" | "dodge"
               kde=False, rug=False, height=None, aspect=None, theme=None)
```

> **2026-09-10 (batch F, F-L09-01 + F-L09-10):** `kind="kde"`'s
> `cumulative=` reaches the `Kde` transform end to end (grouped fits
> included) -- previously `_displot_build` silently dropped it for the
> primary KDE mark, so `cumulative=True` rendered byte-identically to
> `cumulative=False`. It now produces a real CDF. The `kde=True` overlay on
> `kind="hist"` inherits the same `cumulative` value, so a cumulative
> histogram never carries a mismatched (non-cumulative) PDF overlay.
>
> The `kind="hist", kde=True` overlay is also count-scaled and per-hue,
> closing two adjacent bugs. Under `kind="hist", stat="count"`,
> `multiple != "fill"` (the only combination where the histogram's y-axis
> is actually in counts -- `multiple="fill"` normalizes bars to `[0, 1]`,
> which a count-scaled curve would not match), the overlay's `Kde`
> transform gains an opt-in `count_scale` (`n_group x bin_width` for the
> density, `n_group` for the cumulative CDF) computed from the same
> bin-width routine `Bin` uses on the histogram's own bins, so the curve is
> visible against the bars instead of a flat ~2%-height sliver (measured
> curve/bar height ratio 0.02 before, matching the bars after). The overlay
> also now threads `groupby=hue` with a matching color channel --
> previously a `hue=` histogram drew ONE pooled KDE curve in the theme's
> flat default color over the per-hue bars, silently ignoring the
> grouping. `stat=` is validated against its documented `{"count",
> "density"}` vocabulary for every `kind=`, not just `"hist"`; an
> unsupported value (e.g. the undocumented `"probability"`) now raises
> instead of silently degrading to a count-mode render. `stat="density"`
> overlays and `kind="ecdf"` are byte-identical (scaling is opt-in and
> applies only to the count-axis case above). The overlay also carries an
> explicit `Identity()` position so it can never inherit a `Stack`/
> normalize adjustment from `multiple=` -- fixing a pre-existing crash
> where `rug=True` combined with `hue=` and `multiple="stack"`/`"fill"`
> raised `"Stack: value (y) encoding required"` (the rug tick layer gets
> the same `Identity()` fix, so rug ticks are never dodged/stacked either).

> **2026-09-18 (wave 4, #193 — `bins=` accepts a vocabulary instead of
> swallowing one):** `displot`'s `bins=` was passed to a door that read only
> an integer count, so every other spelling was silently discarded and the
> chart fell back to Sturges. Measured pre-fix: `bins="fd"`,
> `"freedman_diaconis"`, `"auto"`, `"scott"`, `"stone"`, `bins=0.5` and
> `bins=np.int64(9)` all rendered the identical 11-rect Sturges histogram
> with an empty transform payload; `bins=True` rendered a 1-bin histogram
> (because `bool` subclasses `int`); `bins=0` raised a `ValueError` naming
> `Bin` rather than `displot`; and `bins=-3` leaked a raw `OverflowError`
> out of the renderer. The accepted set is now explicit and closed:
>
> | Value | Meaning |
> |---|---|
> | `"sturges"` | Sturges' rule (the default) |
> | `"fd"` / `"freedman_diaconis"` | Freedman–Diaconis rule |
> | `int` ≥ 1 | an exact bin count |
> | positive finite `float` | a bin **width** |
> | `None` | unset; falls through to `bin_count`/`bin_width` |
>
> Everything else refuses with a typed `ValueError` in the voice of the entry
> the caller invoked. Measured live: `"auto"`, `"scott"`, `"sqrt"`, `"rice"`,
> `"doane"`, `"stone"`, and the near-misses `"FD"` and `"fd "` all give
> `displot: unknown bins value 'auto'; expected 'sturges'|'fd'|int|float`
> (no case or whitespace coercion — the vocabulary is closed); `True`/`False`
> give `displot: bins must not be a bool (True); expected
> 'sturges'|'fd'|int|float`, checked *before* the integer arm; `0`, `-3` and
> `10**30` give `displot: bins as an int must be a bin count >= 1 within
> machine range; got 0` — no `OverflowError`, and no fall-through into the
> width arm; `0.0`, `-0.5`, `nan` and `inf` give `displot: bins as a float
> must be a positive finite bin width; got 0`; and a `list`/`tuple`/`dict`/
> arbitrary object gives `displot: bins must be 'sturges'|'fd'|int|float`.
>
> `bins=` is validated for **every** `kind`, matching the `stat=` precedent —
> so `displot(kind="kde", bins="auto")` and `displot(kind="rug",
> bins="auto")` now raise where `bins=` was previously inert on those kinds.
> A valid `bins=` on those kinds is still accepted and still inert.
>
> Two doors that already spoke this vocabulary inherit the hardened parser
> and therefore gain refusals they did not have: `Bin2D(bins_x=)`/`bins_y=`
> and `Kde(count_scale=)` now refuse `bool`, non-positive and out-of-range
> integers, and non-positive/non-finite floats at construction, each naming
> itself (`Bin2D: …`, `stat_kde: …`). Valid values are unaffected.
>
> Serialization is fixed with it: `BinMode::FreedmanDiaconis` previously
> mapped to the same empty wire payload as Sturges, so `to_dict()`/
> `to_json()` reported Sturges for a chart that had drawn Freedman–Diaconis.
> FD now carries an explicitly-named additive `bin_rule` key; Sturges stays
> the wire default (the absence of all three mode keys), so **every
> pre-existing payload keeps its exact bytes**. A payload naming `bin_rule`
> *and* a numeric key refuses rather than dropping one answer.
> `repr(Bin("x", bins="fd"))` gains a trailing `, bins='freedman_diaconis'`;
> every other mode's repr is byte-identical.
>
> **Correction (2026-09-23, wave 6B, `#195`):** the sentence above was true when written and is false after `#195`: every `Bin` repr now also renders the five parameters it used to omit (`shared_extent`, `groupby`, `agg_field`, `agg`, `name`), whatever the mode. The mode rendering this note describes is kept: a count or width mode renders as before, and FD still ends in `bins='freedman_diaconis'`. See §3.5's `#195` note for the full repr change.

> **2026-09-18 (wave 4, #194 — `multiple="stack"` means stacked):**
> `displot` built its own position adjustment from `multiple=` and never
> forwarded the mode into `mark_histogram`, so the underlying `Bin` ran at
> its default and binned **each group over its own data range**. The two
> groups' bars therefore shared no x position and were not even the same
> width — measured on two 300-point normals at means 0 and 3: bar widths
> `31.34 px` vs `34.19 px`, and **16 distinct x starts where a stack needs
> 8**. The stack adjustment had no pairs to pair, and the render was a
> side-by-side chart wearing a stack's name. What a `multiple=` mode means is
> now answered in one place (`marks/_multiple.py`), which decides the
> position adjustment *and* the binning extent together — because bars can
> only stack if every group binned against the same edges. `displot` forwards
> the mode; `"layer"`, `"dodge"`, the no-hue `"stack"` path and the
> `kind="kde"` branch are byte-identical.
>
> **`mark_histogram(position_by=)` is new public API** (additive,
> keyword-only): the column the `multiple=` position adjustment groups by, so
> forwarding the mode does not cost the caller its grouping. It accepts a
> column name (`str`) or `None` and refuses anything else in its own voice —
> `mark_histogram: position_by must be a column name (str) or None, not
> Color; received Color('g').` — appending `Pass the column name itself:
> position_by='g'.` when a channel object was passed. It is documented as a
> column name, not as the figure layer's "str or encoding" spelling, on
> purpose: `position_by` feeds `Stack(by=)`/`Dodge(by=)`, whose contract is
> `Optional[str]`, and unwrapping would silently widen a contract this change
> is only now creating.
>
> Known limit, unchanged: `displot(kind="kde", multiple="stack")` still does
> not forward the mode to `mark_density`.

> **2026-09-18 (wave 4, #200 — `kind="ecdf"` keeps the column it colors
> by):** the ecdf branch is the one `displot` path that builds its own `Bin`,
> and it built one without the group key, so the transform discarded the
> `hue=` column before the encoding referred to it. Per-class curves could
> not render. The branch now hands the `Bin` the group key, on a support the
> groups share, so `displot(kind="ecdf", hue=...)` draws one curve per class.
> The no-hue ecdf path is byte-identical.

> **2026-09-18 (wave 4, #201-B — the rug gets the axis its curve was drawn
> against):** `displot(x=..., kind="rug", kde=True)` is a composition the
> docstring advertised twice and named rug for explicitly, and it **raised**
> (`encoding 'y' expected EncodingSpec, got None`) because the x-only root
> encoding fell through the generic x/y check. #201-D (above) made it render
> with the correct extent and tick labels, but through the binding *fallback*
> — which resolves a scale, not a chart-level binding — so the density axis
> came out **untitled**. `displot` now binds the density column the overlay
> already names at chart level, and the axis is titled `density`: the KDE
> transform's own output column name, which is exactly what
> `displot(kind="kde")` and `displot(kind="kde", rug=True)` already title the
> same axis, and what `kind="ecdf"` does with `count`. The curve's extent and
> tick labels are unchanged from #201-D; the rug's 200 ticks still sit at the
> axis base; `hue=` still yields two curves and split ticks, now titled.
>
> `displot(y="v", kind="rug", kde=True)` used to silently drop the KDE
> overlay (the overlay block is gated on the x variable) and render a bare
> vertical rug. It now refuses — see the close note below.

> **2026-09-18 (wave 4 close, A2/A3/A5 — `displot` answers in its own voice,
> and stops dropping a mark it was asked for):** three related changes to
> what `displot` accepts.
>
> 1. **One refusal for the x-requiring kinds.** `mark_histogram() requires
>    .encode(x=...)` and its `mark_density()` twin no longer escape from this
>    path: one helper answers for `hist`, `kde` and `ecdf` alike, naming the
>    entry the caller invoked, the binding observed, and the fix in the
>    caller's own column name. Measured:
>    `displot(kind='hist', y='v') is not supported -- kind='hist' draws the
>    distribution of the variable x= names and computes the other axis itself
>    (a count, a density, a cumulative frequency), so a y= variable has
>    nowhere to go. Pass x='v', or kind='rug', which draws its ticks against
>    either axis.` `kind="rug"` is deliberately excluded: a tick mark draws
>    against either axis.
> 2. **`kind="rug", y=..., kde=True` refuses rather than dropping the
>    overlay.** Measured: `displot(kind='rug', y='v', kde=True) is not
>    supported -- the kde= overlay is drawn against the variable x= names,
>    and this call binds y= instead, so the overlay has no variable to read.
>    Pass x='v' (kind='rug' draws the same ticks from x= or y=), or drop
>    kde=.` Refusal rather than a warning because the overlay is not a knob
>    that got ignored, it is a whole requested mark with nowhere to stand —
>    there is no partially-honored reading of it, unlike `bins=` below. A
>    previously-accepted call now raises. The end state this defers is a
>    y-oriented density curve, which is an orientation feature, not a guard.
> 3. **`bins=` warns where it is inert.** `kind="kde"` and `kind="rug"`
>    evaluate no binned statistic, so a stated rule cannot act. Measured:
>    `displot(kind='kde', bins=7): bins= is inert for kind='kde' -- it
>    evaluates no binned statistic, so the rule you named cannot act on
>    anything and the render is identical for every value. Drop bins=, or use
>    kind='hist'/'ecdf', which bin.` Only a rule the caller actually *stated*
>    warns: `bins=None` and `bins="sturges"` both mean the default a
>    non-binning kind would have used anyway, so nothing is discarded and
>    there is nothing to report. This is the warn-degraded half of the #193
>    refusal contract above, which validates the vocabulary for every kind.

> **2026-09-18 (wave 4 close, A1 — `multiple="dodge"` bins on shared
> edges):** `"dodge"` now maps to a shared binning extent, joining `"stack"`
> and `"fill"`; `"layer"` is the only mode that may still bin per group,
> because it is the only one that asks nothing of its neighbours. The band a
> dodged bar takes its slot in *is* the bin, so the bins have to be the same
> bins for every group before a slot means anything. Previously each hue
> group binned over its own range: measured on a two-group frame, 9 bars of
> one width against 10 of another, sharing **zero** x positions — two
> histograms wearing a dodge's name. Measured after, on 200 rows in two
> groups at `bins=10`: both groups draw 10 bars of equal width `22.57 px` on
> one grid of step `50.16 px`, offset from each other by exactly one slot
> (`22.57 px`), with no overlapping pairs and the two slots plus padding
> spanning the band. **This is a user-visible rendering change** — a chart
> built on the old geometry will look different, correctly so. (The
> equal-width half of this depends on the R6 span fix noted under
> `ferrum.Layer`; shared edges alone fixed the grid but not the widths.)

#### Categorical

```
ferrum.catplot(data, *, x=None, y=None, hue=None, col=None, row=None,
               kind="strip",         # "strip" | "swarm" | "box" | "violin" |
                                     # "boxen" | "point" | "bar" | "count"
               order=None, hue_order=None, orient=None,
               dodge=False, jitter=True, native_scale=False,
               ci=95, n_boot=1000, seed=None,
               height=None, aspect=None,   # > **2026-05-31:** added; same semantics as displot
               theme=None)
```

> **2026-09-18 (wave 4, #192 — `kind="count"` accepts a channel object like
> its seven siblings):** `count` was the only one of catplot's eight kinds
> that read the **raw** `x=`/`y=` argument again after the shared splitter
> had already resolved it, so `catplot(..., kind="count", x=fm.X("g"))` died
> with a bare PyO3 `TypeError: argument 'field': 'X' object is not an
> instance of 'str'`, naming neither `catplot`, nor `kind="count"`, nor a
> working route — while `x : str or encoding` was already the documented
> contract for the parameter. It now derives the aggregated field from the
> resolved channel, the same idiom the estimating kinds already use. This
> closes a second, latent defect on the same line: an
> `encode={"x": fm.X("other_col")}` override on `kind="count"` now aggregates
> **the column its axis actually binds**, not the column named by the
> original `x=`/`y=` keyword. The bare-string path is byte-identical
> (SHA-256-pinned).

> **2026-09-10 (batch F, F-L09-04):** `hue_order=` is honored for
> `kind="box"` and `kind="violin"` -- previously silently ignored (byte-
> identical output regardless of the argument). catplot already built
> `Color(hue, sort=hue_order)` correctly; the drop point was
> `_ResolveContext._color_field()` flattening the channel to a bare string
> before the box/violin desugars ever saw the sort. The color sort now
> rides the same threading pipeline `x_sort`/`y_sort` already use: a
> `sort=` on a bare `Color(...)` channel -- catplot's `hue_order=`, or a
> caller's own `mark_boxplot()`/`mark_violin()`/`mark_errorbar()`/
> `mark_errorband()` plus `Color(sort=...)` -- now reaches the desugared
> layers' color encoding and the legend order. `kind="boxen"` and
> `kind="swarm"` keep their pre-existing, documented exclusion from this
> mechanism (their own `hue_order=` already worked through chart-level
> color, unaffected by this fix). `sort=None` (or omitted) is
> byte-identical.
>
> **Dated note (2026-09-11, wave-1 hardening, #190 — `catplot(hue=)` honors both a channel object's `sort=` AND `hue_order=`, kwarg wins):** `catplot(hue=fm.Color(...), hue_order=[...])` used to raise `TypeError` on every one of catplot's eight kinds — the `hue_order=` path always handed the raw `hue` argument straight into `Color(hue, sort=..., type_="nominal")` as `Color`'s positional `field`, and `ChannelBase.__init__` rejects a non-`str` `field`. `hue` is now routed through `_catplot_cat_field_parts`, the same channel-agnostic splitter `x`/`order` already use, so a channel-object `hue=` composes with `hue_order=` instead of crashing. When `hue_order=` is given, it OVERRIDES a channel's own `sort=` (checkpoint ruling: kwarg wins, matching the `order=` precedent — a documented divergence from the issue text's own parenthetical, which proposed "explicit channel sort wins"; splitting the builder's precedence rule between its two parallel channels was rejected as worse). When `hue_order=` is NOT given, a channel object's own `sort=` survives untouched, and — closing a sibling defect of the same root cause — `nominal_color_channel`'s previously-silent `sort=` drop on an already-built channel object now honors it instead of discarding it. An explicit `type_=` on the caller's own channel is honored verbatim alongside `hue_order=`, mirroring how `x_sort`/`type_=` already compose. A caller-supplied numeric `hue=fm.Color(<numeric col>)` with NO `hue_order=` keeps its continuous colorbar exactly as before (this composes with #240-4a below only through the shared `resolve_cmap_alias` refusal, not through the hue-typing decision itself).
>
> **Dated note (2026-09-11, wave-1 hardening, #240-4a — unknown cmap/scheme name refuses at desugar, warns at the Rust resolve site):** `resolve_cmap_alias` (the shared `scheme=`/`cmap=` alias resolver used by `mark_raster`, `mark_hex`, `mark_contour`, `heatmap`, `clustermap`, and `RenderConfig`) now validates the resolved name against the registered continuous colormap list and refuses a bad name at Python desugar/construction time, closing the two channel-less Rust mark sites (`mark_raster`, the quantitative-color arm of `mark_hex`) that previously fell back to Viridis with no diagnostic at all, since the channel-bound warning below only fires when a `Color` scale is actually in the loop. The Rust side of the same fix (`resolve_continuous_scheme`) emits `RenderWarning::UnknownColorScheme` naming the bad name and falling back to the theme default, for the one path that IS channel-bound and therefore reachable only through a hand-built wire spec that bypasses the typed Python API — confirmed unreachable from `fm.Color(scheme="bad")` today, which already refuses at Python construction.
>
> **2026-09-10 (batch F, F-L09-02/03):** `kind="point"` and `kind="bar"` are
> ESTIMATING kinds. Each draws one mean per (category × `hue`) group from a
> `Summary` transform, with a bootstrap CI whisker: the whisker paints under
> the point marker and over the bar. Previously both drew one mark per raw row
> (200 overplotted circles for a 200-row frame) and any non-default `ci=`/
> `n_boot=` raised "not yet supported". `ci` is now a percent in the open
> interval `(0, 100)` and `n_boot` must be `>= 1`; both are validated at the
> catplot boundary for these two kinds only (they remain inert on the other
> six). `seed=` seeds the bootstrap as well as `Jitter`, with `None` mapping
> to the fixed default seed, so output is byte-deterministic. Both kinds are
> layered charts now, so `mark=` takes per-layer keys (`"whisker"` plus
> `"point"`/`"bar"`) rather than a flat mark-kwargs dict, and a call missing
> either the category or the value channel raises instead of drawing a
> value-less strip. `ci`/`n_boot` also refuse a `bool` (it subclasses `int`,
> so `n_boot=True` would silently mean one resample); the numeric `1` stays
> valid for both. An explicit positional `scale=` on the value channel (via
> `y=`/`x=`, `encode=`, or `**encode_kwargs`) is honored on these kinds as it
> is on the others **when it carries a `domain`**; a domain-less one (e.g. a
> bare `{"type": "log"}`) now raises instead — main accepted it when these
> kinds were single-mark charts, but on a layered chart the renderer resolves
> an explicit domain-less scale from one column and would silently clip the CI
> whiskers (the same gap drops marks on `kind="box"`/`"boxen"` today, without
> an error). Adding a `domain` is the documented route; the refusal is deleted,
> not relaxed, once the explicit-scale path gains the cross-layer domain union
> the auto-inferred path has (tracked as SCALE-1 in the code-archaeology
> followups doc). The other six kinds are byte-identical.
>
> **2026-09-10 (batch F close):** every option stated on a channel the CALLER
> built reaches all eight kinds. `x=fm.X("cat", sort=[...])` and
> `hue=fm.Color("g", sort=[...])` were honored by the six per-row kinds and
> dropped by `kind="point"`/`"bar"`, whose layers rebuilt both channels from
> bare field names and re-threaded only `type_`, `order=` and the
> native-scale domain -- so the encoding spelling of an order was inert while
> the `order=`/`hue_order=` kwarg spelling worked. Both spellings resolve at
> one site now and produce byte-identical output for the same order; where
> both are given the kwarg wins (unchanged, and the same on every kind). The
> third door is uniform too: `encode={"x": ...}` is a raw override that
> REPLACES the builder's normalized category channel (discarding its typing
> and its native-scale padding, as `encode=` is documented to). It always did
> that on the six per-row kinds; it now does it on the two estimating kinds
> as well, which previously kept their own channel and left the door inert.
> `encode`'s `x` is the layer-space CATEGORY channel under both orientations,
> the same key space the value-scale refusal and the scale propagation
> already read it in.
> **Refused, not dropped:** a PER-OBSERVATION appearance channel on the two
> estimating kinds -- `size`, `shape`, `opacity`, `fill_opacity`,
> `stroke_opacity`, `stroke_width`, `stroke_dash`, `angle`, `color`, `fill`,
> `stroke`, through a bare keyword or the `encode=` dict -- raises naming the
> route. These kinds draw one mark per (category × `hue`) group, so the
> mapping has no row to read: it reached the chart-level encoding only, where
> it drew the channel's LEGEND over marks that carried none (measured:
> `size=` rendered a six-swatch legend beside uniform-radius estimators;
> `opacity=` was byte-identical to passing nothing). `hue=` is deliberately
> NOT in the set -- it joins the `Summary` groupby, so its legend is true.
> The six per-row kinds accept all of these unchanged.
>
> **2026-09-10 (batch F, T5r rider — renderer-wide):** a LAYERED chart's
> auto-inferred value-axis domain now extends to zero when a bar layer binds
> that channel, mirroring the zero-anchor a single-chart `mark_bar` already
> gets from its injected `scale.zero`; it is suppressed when the bar binds an
> explicit span channel (`y2`/`x2`), whose extent is taken literally, matching
> the same suppression on the single-chart rule. This is what keeps
> `catplot(kind="bar")`'s bars on a zero baseline without an explicit
> per-layer scale, which would have replaced the cross-layer domain union and
> clipped the CI whiskers. The rule is not catplot-specific: any layered spec
> with a bar layer on an auto-inferred value axis is affected, and one
> phase-10 golden (`class_prediction_error_multiclass.svg`) moved its baseline
> accordingly. `mark_bar(zero=False)` opts a layer out of the new rule as it
> already opted a single chart out of the injected `scale.zero`: the opt-out
> now travels with the mark as `mark_style.zero` when a chart is composed into
> a layer, so it survives on the merged single-panel path
> (`resolve={"y": "independent"}`, `SecondaryY`, and every `.interactive()`
> `LayerChart`), where the chart-level injection never ran. On
> `catplot(kind="bar")` itself -- whose bar layer is built directly as a
> `_Layer`, not composed from a standalone `mark_bar()` chart -- the identical
> opt-out is reachable through the ordinary per-layer `mark=` override:
> `catplot(kind="bar", mark={"bar": {"zero": False}})` merges `zero: False`
> into that layer's `mark_kwargs` (`_apply_mark_overrides`), which
> `_build_layers_list` serializes wholesale as the layer's wire `mark_style` --
> no special-case wiring needed, live and honored.

> **2026-09-10 (batch F, F-L09-03):** the categorical channel is now ALWAYS
> explicitly typed, for every one of the 8 `kind=` values, instead of never
> being typed at all. `native_scale=False` (the default) types it
> `"nominal"` -- **disclosed behavior change:** a numeric category column
> (e.g. an integer year) previously inferred `"quantitative"` and rendered
> on a linear axis, silently contradicting this documented default; it now
> renders as equal-spaced categorical bands, matching a string column and
> matching seaborn. `native_scale=True` types it `"quantitative"` and
> places category values at their real numeric positions; it raises,
> naming the column and its dtype, when the column is not numeric. The
> former `native_scale=True` refusal ("the renderer does not support
> quantitative categorical axes") is deleted -- it was factually wrong; the
> renderer's continuous-band `Dodge` and non-ordinal `Jitter` paths already
> existed and needed no Rust change. `kind="strip"`'s `Jitter` width on a
> native scale is rescaled Python-side from the ordinal 0.4 band-fraction
> to `0.4 x` the category column's own median unique-value spacing (data
> units), mirroring the bandwidth proxy `Dodge`'s continuous-band path
> already computes; a single-category column falls back to the literal 0.4.
> One exclusion, orthogonal to this change: `kind="boxen"`'s letter-value
> transform hard-requires a Utf8 groupby column regardless of
> `native_scale` (a pre-existing Rust-transform gap, not introduced or
> closed here).
>
> **2026-09-10 addendum (cycle-2 close-out, same finding F-L09-03):** the
> category channel's typing now covers the documented `x : str or encoding`
> shape uniformly -- an already-built channel object (`x=fm.X("year")`) is
> typed and `order=`-sorted exactly like a bare field name, honoring an
> explicit `type_=` the caller stated on their own channel verbatim when
> present. A quantitative axis's domain is otherwise just the raw data
> extent, with no room for `Jitter`/`Dodge` to move a mark into; when
> `kind="strip"` jitters (default `jitter=True`) or any kind dodges
> (`dodge=True` + `hue=`), the category channel's domain is padded by half
> that position adjustment's own maximum offset (plus a 5% safety margin)
> so no mark renders outside it and is silently dropped by the renderer,
> which resolves no awareness of the pending offset on its own. This
> padding never overrides a domain the caller set explicitly (`x=fm.X(...,
> scale=...)` or `encode={"x": ...}`) -- an explicit caller domain still
> governs, and a mark that clips against IT is the caller's own choice.

#### Relational

```
ferrum.relplot(data, *, x, y, hue=None, size=None, style=None,
               col=None, row=None,
               kind="scatter",        # "scatter" | "line"
               height=None, aspect=None, theme=None)
```

`size` maps to the `Size` channel (point area or line width). `style`
maps to `Shape` for `kind="scatter"` or `StrokeDash` for `kind="line"`.

#### Regression

```
ferrum.lmplot(data, *, x, y, hue=None, col=None, row=None,
              method="lm",           # "lm" | "loess" | "quadratic" | "cubic" | "log" | "sqrt" | "gam" | "logistic" | "glm" | "robust" (smooth-family names derive live from the registry, 2026-09-05 batch D; "glm" maps to the dedicated Glm transform; mark_smooth's alias spellings are not accepted here)
              ci=95, order=None, scatter=True, scatter_kws=None, line_kws=None,   # order=None = the method's own default degree; an explicit order forwards into the method's degree-acceptance matrix (2026-09-05, batch D; previously order=1 and silently absorbed)
              truncate=True, x_bins=None, x_estimator=None, x_jitter=None,
              logx=False, show_metrics=True, theme=None)
# 2026-09-05 (batch D): this block previously mis-documented truncate=False (the
# shipped default has been True since the function landed) and omitted
# show_metrics= — both corrected above to the live signature.

ferrum.regplot(data, *, x, y, hue=None,
               method="lm",          # same as lmplot
               ci=95, order=None, scatter=True, scatter_kws=None, line_kws=None,   # order=None = the method's own default degree; an explicit order forwards into the method's degree-acceptance matrix (2026-09-05, batch D; previously order=1 and silently absorbed)
               truncate=True, x_bins=None, x_estimator=None, x_jitter=None,
               logx=False, show_metrics=True, theme=None)
# 2026-09-10 (batch F, F-L10-09): logx, show_metrics, x_bins, x_estimator are
# now real named parameters (identical defaults to lmplot's), corrected above
# to the live signature -- previously reachable only through **encode_kwargs,
# invisible to introspection. col=/row= raise TypeError naming lmplot as the
# faceting route (previously silently faceted) -- see the prose note below.

ferrum.residplot(data, *, x, y, lowess=False, order=1, robust=False,
                 dropna=True, show_metrics=True, zero_line=True,
                 label=None, color=None, theme=None)
# 2026-09-10 (batch F): the block above adds show_metrics= and zero_line=,
# which the signature has always accepted but this doc omitted.
```

`regplot` is the axes-level equivalent of `lmplot` — identical API minus
`col=` and `row=`: passing either now raises `TypeError` naming `lmplot` as
the faceting route (2026-09-10, batch F, F-L10-09 -- previously they
silently faceted through `**encode_kwargs`). This exclusion is specific to
`col=`/`row=`; it does not extend to the grammar-level `facet=` encoding
channel, which `regplot` accepts the same as any other `Chart`. Every other
parameter is forwarded to `lmplot` unchanged.

> **2026-09-10 (batch F, F-L10-04):** `residplot`'s `color=` works, both
> documented halves. A value that names a column of `data` groups the
> residuals -- `Smooth`/`Robust` refit per group (via the transform's
> `groupby`, the same mechanism `label=` already used) and the points/
> lowess overlay color by that column; previously EVERY `color=` value
> raised `ValueError: unknown column`, whether it named a column or not.
> A value that is not a column is parsed as a literal color (CSS name,
> hex, `rgb()`/`rgba()`, or an RGB tuple) and painted onto the points via
> the mark's own `fill=` -- never through `encode(color=)` -- composing
> with `label=` (a labeled legend swatch in that literal color, via a
> pinned single-range `OrdinalScale`). A value that is neither raises,
> naming both interpretations. Column-membership against the INPUT frame
> is checked first, so a column literally named `"red"` groups rather than
> paints, and a column name that collides with a transform OUTPUT column
> (e.g. `"x"`) still resolves deterministically by input-frame membership.
> `Robust` gained `groupby` in Rust to make this possible (see the
> `Robust` dated note in §3.4) -- as a side effect, `robust=True,
> label=...` now shows the label (previously a silent no-op, since
> `Robust` had no `groupby` to carry the injected `_label` column into its
> output). `lowess=True` with a grouping `color=` draws one curve per
> group instead of one curve through the pooled cloud. `show_metrics=True`'s
> corner annotation is suppressed only when `color=` actually groups
> (mirrors `lmplot`'s `hue`-gated `metrics_applied`); a `label=`-only
> grouping (one level by construction) keeps its corner. A user column
> literally named `_label` plus `label=` now refuses with a typed
> collision error instead of the label silently overwriting it.
> `color=None` is byte-identical to before.

> **2026-09-10 (batch F close, F-L10-02):** `lmplot`/`regplot`'s scatter
> layer keeps the OBSERVATIONS. With `ci=None` and `show_metrics=False` the
> fit chart collapsed to a single line mark bound to a chain-advancing
> (unnamed) `Smooth`, and composing it with the scatter applied that
> transform to BOTH layers: every rendered "observation" landed exactly on a
> fit vertex (measured: 200 of 200, zero spread about the curve) for `lm`,
> `loess`, `quadratic`, `gam`, `log` and `sqrt`, in both functions -- or
> raised `unknown column` when the frame's columns were not themselves named
> `x`/`y`. The fit now binds to a NAMED transform through `data_source=`
> whenever a scatter layer is present, the same shape the `ci=` path and the
> `logistic`/`glm`/`robust` methods already used; with `scatter=False` the
> single-mark shape is kept, since there is no sibling layer to corrupt.
> (That shape's chart-level encoding titles the axes with the caller's own
> column names -- but it is reached only at `ci=None, show_metrics=False`
> on a `mark_smooth`-routed method, i.e. 7 of the 40 `scatter=False` cells,
> so it was never what titled the other 33; see the 2026-09-17 note below.)
> **One layer
> renames:** on that combination the fit layer is
> `"line"` rather than `"fit"` -- which every other `lmplot` shape already
> called it -- so a per-layer `mark={"fit": ...}` override for it becomes
> `mark={"line": ...}`. Separately, `x_jitter=` on a non-Float64 predictor
> now renders: `Jitter` writes a Float64 offset back into the column it
> adjusts, so an Int64 predictor failed Arrow's schema check on three of the
> four `ci`/`show_metrics` combinations and was masked on the fourth by the
> corruption above; the predictor is widened before either chart is built,
> and the cast is deleted once `apply_jitter` coerces the column it writes.

> **2026-09-17 (wave 4, #199):** every fit path titles its axes with the
> caller's own columns, and honors `line_kws` on every method. The fit
> layers (ribbon / line / metrics) are bound to the transform's output
> columns, and the renderer resolves an axis title from layer 0 only
> (spec-level channel `title`, then layer-0 channel `title`, then layer-0
> field name), so whenever no scatter happened to sit at layer 0 the value
> axis was titled `ci_lower` -- the confidence band's LOWER BOUND -- and the
> predictor axis `x`. Measured over the full 40-cell `method × ci ×
> show_metrics` matrix under `scatter=False` (all ten `lmplot` methods ×
> `ci ∈ {95, None}` × `show_metrics ∈ {True, False}`), only 7 cells were
> right — the seven `mark_smooth`-routed methods at `ci=None,
> show_metrics=False`, where the single-line shortcut keeps the chart-level
> encoding; the other 33 were wrong. All 40 are correct now.
> The fit-layer builder now carries the caller's field names as layer
> titles, which fixes `lmplot(scatter=False)`, `regplot(scatter=False)` and
> a standalone `Chart(df).mark_smooth(...)` from one site; an explicit
> channel `title=` still outranks it. `residplot` gets the same treatment on
> its DEFAULT path, where no flag was needed to reach the defect: its axes
> were titled `x` and `residual`, and are now the caller's x column and
> `"<y> residual"` (the one golden this moves is
> `tests/test_phase_9_e2e/goldens/residplot_lowess.svg`, whose y-axis title
> changes and nothing else). Separately, `line_kws` was silently discarded
> for the seven `mark_smooth`-routed methods (`lm`, `loess`, `quadratic`,
> `cubic`, `log`, `sqrt`, `gam`) under `scatter=False` — 28 of the 40 cells;
> the three dedicated methods (`logistic`, `glm`, `robust`) already forwarded
> it at construction and were never affected. It was
> applied by rewriting the finished chart's layers, which on that branch are
> still unmaterialized (the chart carries a pending stat mark until render).
> Every branch now forwards it at construction, to the fit line alone, and
> the same capability is exposed on the grammar as
> `layer_kwargs={"<layer>": {...}}` — a composite mark's own `**mark_kwargs`
> reach every emitted layer, so a `stroke=` passed that way repaints the CI
> band too. **Precedence between the two is by specificity:** when both name
> the same style key, the one aimed at the named layer wins, and the
> catch-all still applies wherever nothing narrower was said — measured,
> `mark_smooth(ci=0.95, stroke="#0000ff", layer_kwargs={"line": {"stroke":
> "#ff0000"}})` draws the fit line `#ff0000` while the CI band keeps its
> `#0000ff`-derived fill, on the layered shape and the single-line shortcut
> alike. Charts with a scatter present and the single-line shortcut render
> byte-identically. (This parameter was first written as a
> `mark_smooth`-only `line_kwargs=` earlier in the same unreleased wave and
> reshaped before release into the general door; see the 2026-09-18 note
> under `ferrum.Layer` for the shipped contract. `line_kwargs=` is not
> accepted.)

> **2026-09-18 (wave 4 close, B3/B6 — the fit paths accept a channel object
> on every method, and refuse a missing fit column by name):** `lmplot`'s
> `x=`/`y=` are documented as "str or encoding", and seven of the ten methods
> already honored the encoding spelling — the `mark_smooth`-routed ones,
> because `.encode()` takes a channel object. The three that construct a
> transform directly did not, and neither did `residplot`, and all four
> leaked a **binding-layer** error out of a figure-level call: measured
> pre-fix, `lmplot(x=fm.X("bill"), method="robust")` raised the raw PyO3
> `TypeError: argument 'x': 'X' object is not an instance of 'str'`,
> `residplot(x=fm.X("bill"))` raised polars' `TypeError: cannot turn
> 'ferrum.encoding.positional.X' into selector`, and any method with
> `logx=True` raised `X: field must be str, ...`.
>
> **Resolved by accepting, not refusing** — refusing would have regressed
> seven working paths and contradicted the family's own documented contract.
> Each consumer wants the column *name*, which is now narrowed once at the
> figure's own entry; the channel itself still reaches `.encode()`, so the
> caller's `title=`/`scale=` survive. Measured:
> `lmplot(x=fm.X("bill", title="Total bill (USD)"), method="robust")` renders
> that title, not the bare column name, and `residplot(x=fm.X("bill"))`
> renders. **`residplot` now accepts an encoding object where it previously
> died** — a widened public surface at a second entry, deliberately, to match
> the family.
>
> Two new typed refusals come with the narrowing, replacing untyped errors:
> `lmplot(x=None): x names the column to fit and has no default. Pass a
> column name, or an encoding object wrapping one.` (and its
> `residplot(y=None)` twin, which names `residplot`), and the contradiction
> `lmplot(logx=True, x=<channel with its own scale=>): two instructions for
> one x scale. Keep your scale= and drop logx=True, or pass x as a column
> name and let logx set the log scale.` The `None` case also closed a latent
> instance of this same note's title defect: `residplot` built its titles
> from the **raw** parameter, so `y=None` would have rendered the axis title
> `"None residual"` and a channel object would have rendered an object repr —
> unreachable only because polars raised first.

#### Matrix / Pairwise

```
ferrum.pairplot(data, *, vars=None, x_vars=None, y_vars=None,
                hue=None, kind="scatter",     # off-diagonal
                diag_kind="auto",             # "hist" | "kde" | None
                markers=None, height=None, aspect=None,
                corner=False, dropna=False, theme=None)

ferrum.heatmap(data, *, annot=True, fmt=".2f", cmap="blues",
               linewidths=None, linecolor=None,   # None: 0.5 px, "white"
               vmin=None, vmax=None, center=None, robust=False,
               square=False, mask=None, theme=None)

ferrum.clustermap(data, *, method="ward", metric="euclidean",
                  cmap="viridis", linewidths=None, linecolor=None,
                  z_score=None, standard_scale=None,
                  figsize=None, dendrogram_ratio=0.2, theme=None)
```

> **2026-09-02 (batch A — heatmap domain/border honesty):** `vmin=`/
> `vmax=`/`center=`/`robust=` are honored end to end via a Diverging (when
> `center=` is set) or Linear color scale's explicit `domain=`; previously
> these fell through to an unscaled render. `vmin`/`vmax`/`center` must be
> finite — `inf`/`-inf`/`NaN` raise `ValueError` immediately, before any
> chart construction. When only one of `vmin=`/`vmax=` is given, the other
> endpoint fills from the pre-mask finite data extent (matching `robust=`'s
> existing convention); if there is no finite data to fill from, the fill is
> dropped with a `UserWarning` naming the reason rather than silently
> discarding the user's explicit bound. A one-sided fill that lands on the
> wrong side of the given bound, or an explicit `vmin > vmax`, still renders
> — deterministically as a single flat color, not a reversed ramp — with a
> `UserWarning` naming both endpoints. A `center=` outside the effective
> `[vmin, vmax]` renders as a one-sided compressed ramp with a
> `UserWarning`, never silently and never as an error. `linewidths=`/
> `linecolor=` are honored (default `linewidths=0.5`, `linecolor="white"`)
> — every heatmap now ships default white 0.5px cell borders;
> `linewidths=0` disables the border.

> **2026-09-10 (batch F, F-L09-08):** `clustermap`'s dendrogram panels no
> longer silently blank at practical ratios. The row/column dendrogram
> leaves reserved axis chrome -- a title gutter and (row side) a label
> band -- for axes their own `axis(show=False)` never draws;
> `_clustermap_build` now blanks the dendrogram leaves' `_X`/`_Y` titles
> the same way the center heatmap already blanks its own
> (`matrix.py:1157-1160`). At `dendrogram_ratio=0.1`, both dendrograms now
> render instead of collapsing to an empty panel; at the default
> `dendrogram_ratio=0.2`, a short center heatmap (e.g.
> `figsize=(480, 200)`) no longer silently blanks the column dendrogram.
> Sub-floor ratios still raise `PaddingExceedsViewport` as before -- the
> fix removes dead chrome, it does not relax the hard floor. **All
> clustermap output changes geometry** as a result (more of each
> dendrogram panel is now real branch area rather than reserved-but-unused
> chrome); the `clustermap_basic` and `clustermap_row_col_dendrograms`
> goldens were regenerated and PNG-inspected. The same fix closed two
> adjacent layered-chart wire bugs found alongside it (general, not
> clustermap-specific): `SpecBuildMixin._build_layers_list` was dropping a
> layer's `title=""` via a truthiness check instead of a presence check
> (so `title=None` on a layered channel now also reaches the wire and
> suppresses the axis title, where `title=""` already did), `format=` on
> any layered channel no longer raises an unknown-field error
> (`format_type` is what Rust actually calls the key), and a layer's
> `condition=` now reaches the wire instead of silently vanishing.

> **Dated note (2026-10-07, #125: `clustermap` draws `heatmap`'s default cell border, and one helper owns that border for both):** `clustermap` gains `linewidths=` and `linecolor=`, with the same meaning, default and validation as `heatmap`'s. One helper owns the default (0.5 px, `"white"`), the `None`-means-default rule and the validation for both figures, so the two can no longer drift. Both signatures now default to `None`, which resolves to 0.5 and `"white"`. No Rust or theme default is involved, and `confusion_matrix_chart`, `classification_report_chart`, `rank2d_chart` and `jointplot(kind="hist")` stay borderless (user ruling 2026-10-07; a border stays reachable through `mark=`).
>
> - **`clustermap(df)`'s center cells** had no border and now draw a white 0.5 px stroke, one per cell. Dendrograms, colorbar and labels are unchanged; the `clustermap_basic` and `clustermap_row_col_dendrograms` goldens moved and were PNG-inspected against their previous renders. `clustermap(df, linewidths=0)` draws no border (before, `linewidths=` was not a parameter and was refused as `unknown encoding channel: 'linewidths'`). `clustermap(mark={"stroke": ..., "stroke_width": ...})` still wins on the cells.
> - **`heatmap(df)` with no border arguments is byte-identical.** `heatmap(df, linewidths=0)` is too.
> - **`None` means the default on both figures.** `heatmap(linewidths=None)` raised a raw `TypeError: '>' not supported between instances of 'NoneType' and 'int'` and now draws the default border. `heatmap(linecolor=None)` drew no border, with a warning about `stroke_width`, and now draws the white default.
> - **Malformed values are refused naming the figure and the argument, never `stroke`.** `linewidths` must be a finite, non-negative real number (`bool` excluded); `linecolor` must be a color `ferrum.color.to_hex` accepts, or one of the paint sentinels below. Before and after, for `heatmap`:
>     - `linewidths=-1`: drew no border, silently. Now `ValueError: heatmap(linewidths=...) must be non-negative; received -1`.
>     - `linewidths=float("nan")`: drew no border, silently. Now `heatmap(linewidths=...) must be a finite numeric pixel value; received nan`.
>     - `linewidths=True`: `heatmap(stroke_width=...) must be a real number (int, float, numpy scalar, or Decimal), not bool; received True`. Now `heatmap(linewidths=...) must be a numeric pixel value (int or float), not bool; received True`.
>     - `linewidths="a"`: raw `TypeError: '>' not supported between instances of 'str' and 'int'`. Now `heatmap(linewidths=...) must be a numeric pixel value (int or float), not str; received 'a'`.
>     - `linecolor="notacolor"`: `heatmap: stroke='notacolor' is not a valid color (...)`. Now `heatmap(linecolor=...)='notacolor' is not a valid color (invalid color 'notacolor': expected a CSS color name, #rrggbb[aa]/#rgb[a] hex, or rgb()/rgba())`.
>     - `linecolor=3`: `heatmap: stroke must be a string; got 3`. Now `heatmap(linecolor=...)=3 must be a color string, not int`.
>
>   `clustermap` refuses the same values with the same text, voiced `clustermap(...)`. A malformed `linecolor` is refused even beside `linewidths=0`.
> - **Paint sentinels.** `linecolor="none"` or `"transparent"` draws no border, the same as `linewidths=0`; before, `heatmap` also warned that `stroke_width` had no effect, and now it is silent. `linecolor="theme:label"` is forwarded unchanged and draws the theme's label color, as before.
> - **An explicit value made inert by the other argument warns, naming both.** `heatmap(linewidths=2, linecolor="none")` warned about `stroke_width` and now warns `heatmap: linewidths=2.0 has no effect when linecolor='none' draws no cell border; it was ignored`. `heatmap(linewidths=0, linecolor="red")` was silent and now warns `heatmap: linecolor='red' has no effect when linewidths=0.0 draws no cell border; it was ignored`. `None` on either side is unset and never warns. The warnings `heatmap` and `clustermap` raise from their own argument handling (these two inert-pair warnings and `heatmap`'s three color-domain warnings for `vmin=`/`vmax=`/`center=`) now point at the caller's own line; warnings raised later at render time are unchanged.

#### Joint Distribution

```
ferrum.jointplot(data, *, x, y, hue=None,
                 kind="scatter",       # "scatter" | "kde" | "hist" | "hex" | "reg"
                 marginal_kind="hist", # "hist" | "kde" | "rug" | "box"
                 ratio=5, space=0.05,
                 xlim=None, ylim=None,
                 joint_kws=None, marginal_kws=None,
                 height=None, theme=None)
```

Returns a `JointChart`. `kind` controls the center panel mark; `marginal_kind` controls both marginal panels. `joint_kws` and `marginal_kws` are passed as keyword arguments to the center and marginal chart constructors respectively.

#### Model Diagnostics (figure-level)

```
ferrum.roc_chart(model_or_source=None, X=None, y=None, *,
                 y_true=None, y_pred=None,   # precomputed path (1-D binary or 2-D multiclass scores)
                 per_class=True,
                 average="macro", annotate_auc=True,
                 compare=None,     # dict[str, estimator] for multi-model overlay
                 theme=None)

ferrum.pr_chart(model_or_source=None, X=None, y=None, *,
                y_true=None, y_pred=None,    # precomputed path (1-D binary or 2-D multiclass scores)
                per_class=True,
                annotate_ap=True, iso_lines=True, compare=None, theme=None)
# **2026-09-11 (wave-1 hardening, #220):** `iso_lines=True` on a multiclass
# precomputed source no longer emits polars' internal "Comparisons with
# None always result in null" `UserWarning` — the two group-enumeration
# loops behind iso-line label placement (`_metric_labels.py`'s
# `_apply_metric_label`, `_direct_label.py`'s `_direct_label_endpoint`)
# now filter out a null class label before sorting, mirroring the same
# file's already-correct `_curve_metrics_by_group` idiom, instead of
# comparing `None` against every class value during the sort.
#
# **2026-09-18 (wave 4, #121):** the `iso_lines=True` reference curves now
# render in the grey they always declared. The iso-line layer bound its
# grouping key `_iso_f` to the `color` channel, but the chart's color scale
# is keyed on `class`, so every `"F=0.2".."F=0.8"` lookup missed and the
# line mark's stroke fallback painted `paint.fill` — theme blue at the
# layer's `opacity=0.6` — instead of the literal `stroke="#AAAAAA"` the
# layer had set all along. The key now travels on `detail` (the channel
# that groups without painting), mirroring `desugar_parallel_coordinates`.
# The stale `_iso_f` entry in the color-binding allowlist, whose rationale
# asserted the curves were already grey, is deleted with it.

ferrum.confusion_matrix_chart(model_or_source=None, X=None, y=None, *,
                               y_true=None, y_pred=None,   # precomputed path (1-D hard labels)
                               normalize="true", annotate=True, theme=None)
# **2026-09-10 (batch G doc pass, adjacent transcription fix, pre-existing
# drift, not a batch G behavior change):** corrected to match the live
# signature — the prior listing showed a `cmap="blues"` parameter that does
# not exist on this function and omitted the real `annotate=True` kwarg
# (per-cell text annotations, already documented separately in the
# Schwabish SB3 note below as "docstring already specified `annotate=True`
# default").

ferrum.classification_report_chart(model=None, X=None, y=None, *,
                                    y_true=None, y_pred=None,   # precomputed path (1-D hard labels)
                                    random_state=None, theme=None)
# **2026-09-10 (batch G, F-L11-05 + F-L11-09):** joins the precomputed
# family — eleven `y_true=`/`y_pred=` figure functions now, counting
# `prediction_error_chart` (see the dated note below: the prior "nine"
# framing already undercounted by one before this batch, since
# `prediction_error_chart` documents and renders on this path too and was
# never enumerated in the list). Renders
# a rect-plus-text heatmap: rows are class labels, columns are
# precision/recall/f1-score, color encodes the value, and each cell is
# annotated with its value to two decimal places — no `support` row (a
# deliberate, disclosed non-parity with sklearn's own
# `classification_report`, which does report support counts; adding one
# here would require inventing a count from confusion data that would not
# reconcile with sklearn's on the model path, defeating the parity the
# precomputed and model paths otherwise guarantee). The chart carries a
# bare title, `"Classification Report"` — the family's title-only shape (no
# `subtitle=` parameter), matching `confusion_matrix_chart`. On the
# precomputed path, `y_true=`/`y_pred=` (hard labels) resolve metrics from
# confusion counts with **no sklearn import**; class labels are the
# distinct values of `y_true`/`y_pred` in sklearn `unique_labels` order
# (their own ascending order, raw values — not stringified first), the
# same order the model path's sklearn-backed report produces. On the model path, `X`/`y`
# are both required — an X-less or y-less `ModelSource` (or a bare
# `classification_report_chart(model)` call) raises a typed `ValueError`
# naming the diagnostic and the fix, replacing what used to be a leaked
# sklearn shape error (`"Expected 2D array, got scalar array instead"`) or
# a foreign `InvalidParameterError` naming `y_true`. Both paths order the
# class axis identically (#224a, 2026-09-13): the precomputed path's
# `sorted(..., key=str)` rule — since retired for `confusion_matrix` too
# (close remediation, 2026-09-17; see the dated note) — retired for this
# figure, so the documented divergence between the two paths no longer
# exists. `y_true` and `y_pred`
# must each be 1-D (a 2-D `y_true` now refuses by name, mirroring the
# `y_pred` shape check #208 already had). The label derivation
# (`_curve_frames.report_labels`) is total by partition over the *values*
# the two (1-D) arrays hold: a null in either array, a mix of ordering
# families, and a homogeneous family that cannot serve as a class label —
# either because it does not order (`complex`, `dict`) or because it
# orders but cannot be hashed (`list`, `set`, `bytearray`) — each refuse
# with a typed `ValueError` naming which array and what it holds. A
# homogeneous, orderable, hashable family is the answer, deduplicated by
# hash equality (not adjacency, so a partially-ordered-but-hashable family
# such as `frozenset` cannot return a duplicate label). Nothing else can
# reach the class axis. On the model path (#206 cycle 5, 2026-09-13): when
# `model.classes_` is present, it is coerced into `y`'s dtype and checked
# against `y`, and the call refuses if the two share no value at all (a
# model whose `classes_` describe different labels than `y`, not merely a
# different dtype) — see the dated #206 cycle-5 note above. A partial
# overlap still renders, and so does a model that simply predicts none of
# the classes `y` holds on the slice being reported.

ferrum.calibration_chart(model_or_source=None, X=None, y=None, *,
                          y_true=None, y_pred=None,  # precomputed path (1-D positive-class proba)
                          n_bins=10, theme=None)

ferrum.gain_chart(model_or_source=None, X=None, y=None, *,
                  y_true=None, y_pred=None,          # precomputed path (soft scores)
                  compare=None, theme=None)

ferrum.lift_chart(model_or_source=None, X=None, y=None, *,
                  y_true=None, y_pred=None,          # precomputed path (soft scores)
                  compare=None, theme=None)
# **2026-09-10 (batch G, F-L11-08):** `gain_chart`/`lift_chart` now show the
# model's real class names for every class, not just a synthetic
# `"Class N"` rename for the first two. **Disclosed change:** binary charts
# label classes `"0"`/`"1"` (the stringified real value) instead of
# `"Class 0"`/`"Class 1"`; a 3rd-and-later class, which previously rendered
# unlabeled relative to its siblings, now renders with its own real name
# too. String-named classes (e.g. `classes_=["setosa", "versicolor"]`) were
# already rendered as-is and are unaffected. The only remaining display
# rename is the synthetic baseline row (`gain_frame`/`lift_frame`'s own
# reference-line data) → `"Baseline"`. See the `ModelSource.class_names`
# dated note under §3.1 for the underlying resolution rule
# (`class_names=` → `model.classes_` → real value, never invented).

ferrum.residuals_chart(model_or_source=None, X=None, y=None, *,
                        y_true=None, y_pred=None,    # precomputed path (fitted values); residuals = y_true − y_pred
                        kind="studentized", panels="auto",
                        # panels: "auto" | "single" | None | list of
                        #         "residuals_vs_fitted" | "qq" |
                        #         "scale_location" | "residuals_vs_leverage"
                        theme=None)
# **2026-05-12 (P3.12, F10):** `panels="auto"` ships the canonical
# 4-panel layout — `residuals_vs_fitted`, `qq`, `scale_location`,
# `residuals_vs_leverage` — composed as a 2x2 grid. `panels=None` or
# `panels="single"` returns just the residuals-vs-fitted panel
# (the pre-P3.12 behavior). For non-linear estimators (no `coef_`),
# `ModelSource.predictions()` emits all-NaN `leverage`; the chart
# builder silently drops the leverage panel from the auto layout so
# `panels="auto"` stays safe for RandomForest / GBM / etc.
#
# **2026-05-15 (precomputed path):** `y_true=`, `y_pred=` bypass the model
# entirely. Residuals are computed as `y_true − y_pred`. Leverage and
# Cook's distance are unavailable (no design matrix), so the leverage
# panel is silently dropped when `panels="auto"`. `compare=` is
# incompatible with the precomputed path (raises `ValueError`).
#
# **2026-07-02 (GH #44):** the leverage drop only degrades gracefully when
# at least one other panel survives it. If dropping `residuals_vs_leverage`
# would empty the resolved panel list — an explicit `panels=` of just
# `["residuals_vs_leverage"]`, or `cooks_distance_chart` (which always
# requests that single panel) — on a non-linear estimator or a precomputed
# source, the call raises `ValueError` naming the hat-matrix/`coef_`
# requirement and the estimator type (when a model is available) instead of
# returning an empty/broken chart. `compare=` inherits this: the error
# identifies which compare= member failed. `panels="auto"` degradation is
# unaffected — it still renders the remaining 3 panels.
#
# **2026-09-11 (wave-1 hardening, #208):** nine `_PrecomputedSource` methods
# (`roc_curve`, `pr_curve`, `cumulative_gain`/`lift_curve`, `calibration_curve`,
# `confusion_matrix`, `classification_report`, `predictions`,
# `discrimination_threshold`) now refuse a malformed `y_pred` shape with a
# typed `ValueError` naming the calling figure function and the observed vs.
# required shape, instead of leaking `IndexError`/`ArrowInvalid`/
# `TypeError: unhashable type: 'list'`/a numpy broadcast `ValueError`, or —
# for a 1-D `y_pred` against a genuinely multiclass `y_true` on roc/pr/
# gain/lift — silently computing a wrong-but-plausible curve. A degenerate
# SINGLE-class `y_true` is NOT treated as multiclass (matching the
# pre-existing, correct `binary_class_values` degradation to a NaN
# AUC/AP with a still-rendering curve) — only `> 2` distinct classes
# refuses. `predictions()` serves two public callers (`residuals_chart` and
# `prediction_error_chart`); its refusal now names whichever one the caller
# actually invoked, via a `chart_name=` parameter, rather than always
# naming `residuals_chart`.

ferrum.importance_chart(model_or_source, X=None, y=None, *,
                         method="builtin", top_k=20, orient="horizontal",
                         error_bars=True, theme=None)

ferrum.shap_chart(model_or_source, X=None, *,
                   kind="beeswarm",   # "beeswarm" | "bar" | "waterfall" | "force" | "heatmap"
                   max_display=20, sample_idx=None, theme=None)

# 2026-05-12 (P3.6): `shap_chart(kind=...)` is now a deprecation
# shim for three sibling functions — `shap_beeswarm_chart`,
# `shap_bar_chart`, and `shap_waterfall_chart` — that each take
# kind-specific keyword arguments without the `kind=` dispatcher.
# The visualizer side splits the same way: `SHAPBeeswarmVisualizer`,
# `SHAPBarVisualizer`, `SHAPWaterfallVisualizer`.  Old call sites
# emit `DeprecationWarning` but keep working.
ferrum.shap_beeswarm_chart(model_or_source, X=None, y=None, *,
                            max_display=20, order="abs_mean",
                            background=None, random_state=None, theme=None)
ferrum.shap_bar_chart(model_or_source, X=None, y=None, *,
                       max_display=20, order="abs_mean",
                       background=None, random_state=None, theme=None)
ferrum.shap_waterfall_chart(model_or_source, X=None, y=None, *,
                             sample_idx, max_display=20, order="abs_mean",
                             background=None, random_state=None, theme=None)

ferrum.learning_curve_chart(model, X, y, *, cv=5, scoring=None,
                              train_sizes=None, ci_style="band",
                              n_jobs=None, theme=None)

ferrum.validation_curve_chart(model, X, y, param, values, *,
                                cv=5, scoring=None, log_scale="auto",
                                ci_style="band", theme=None)

ferrum.cluster_diagnostics(model, *, ks, method="kmeans", scoring="both",
                             # scoring: "elbow" | "silhouette" | "both"
                             n_init=10, random_state=None, theme=None)
# **2026-06-22 (T3.4, D-FIRSTPARAM-1):** the first positional parameter was
# renamed `X` -> `model` so the whole model-diagnostic family shares one
# canonical first-param name. Positional callers are unaffected; the legacy
# keyword `X=` is accepted as a deprecated alias (supplying both `model=` and
# `X=` raises `TypeError`).

ferrum.decision_boundary_chart(model, X, y, *,
                                 features=(0, 1),    # column indices or names
                                 grid_resolution=200,
                                 proba=False,         # plot probability surface if True
                                 scatter=True, theme=None)

ferrum.discrimination_threshold_chart(model_or_source=None, X=None, y=None, *,
                                       y_true=None, y_pred=None,  # precomputed path (1-D positive-class scores); cv= not supported
                                       n_thresholds=50,
                                       metrics=("precision", "recall", "f1", "queue_rate"),
                                       highlight_best=True, compare=None, theme=None)
# 2026-05-12: gain_chart, lift_chart, and discrimination_threshold_chart gain
# compare=None for multi-model overlay, consistent with roc_chart / pr_chart.

ferrum.parallel_coordinates_chart(data_or_source, X=None, y=None, *,
                                   features=None, hue=None,
                                   rescale="minmax",   # "minmax" | "zscore" | None
                                   alpha=0.5, theme=None)

ferrum.class_prediction_error_chart(model_or_source=None, X=None, y=None, *,
                                     y_true=None, y_pred=None,  # precomputed path (1-D hard labels)
                                     normalize=False, theme=None)

ferrum.pca_scree_chart(model_or_source, X=None, *, n_components=None,
                        cumulative_line=True, threshold=0.95, theme=None)

ferrum.rank_chart(data_or_source, X=None, y=None, *,
                   rank="2d",       # "1d" | "2d"
                   algorithm=None,  # "1d" default: "shapiro"; "2d" default: "pearson"
                   top_k=None, theme=None)
# **2026-05-12 (P4.4, F4):** `rank_chart(rank=...)` is deprecated;
# use the two sibling functions `ferrum.rank1d_chart(...)` and
# `ferrum.rank2d_chart(...)` directly. They take the same parameters
# minus the `rank` flag and route to the same chart builders.
# `rank_chart` retains the dispatch behavior as a DeprecationWarning
# shim for backward compatibility.
ferrum.rank1d_chart(data_or_source, X=None, y=None, *,
                     algorithm=None, top_k=None, orient="horizontal",
                     color_field=None, random_state=None, theme=None)
ferrum.rank2d_chart(data_or_source, X=None, y=None, *,
                     algorithm=None, annot=True,
                     random_state=None, theme=None)

ferrum.alpha_selection_chart(model, X, y, alphas, *, cv=5, scoring=None,
                               log_scale=True, ci_style="band", theme=None)

ferrum.intercluster_distance_chart(model_or_source, X=None, *, k=None,
                                    method="mds", theme=None)

ferrum.cv_scores_chart(model, X, y, *, cv=5, scoring=None,
                        kind="box", split="both", theme=None)
```

> **2026-05-15 — Precomputed input path** (corrected count, **2026-09-10,
> batch G, F-L11-05**: eleven members, not ten or nine — the original note
> undercounted by omitting `prediction_error_chart`, which already
> documented and rendered on this path before this batch). The
> prediction-evaluation figure functions that appear above — `roc_chart`,
> `pr_chart`, `calibration_chart`, `gain_chart`, `lift_chart`,
> `discrimination_threshold_chart`, `confusion_matrix_chart`,
> `class_prediction_error_chart`, `residuals_chart`, `prediction_error_chart`,
> and (batch G) `classification_report_chart` — each accept `y_true=` and
> `y_pred=` as keyword-only arguments, bypassing the fitted model entirely.
> Callers supply arrays they have already computed; the function returns a
> chart visually identical to the model-backed path. Exactly one of {model
> path, precomputed path} must be active — supplying both, neither, or only
> one of `y_true`/`y_pred` raises `ValueError`. `compare=` is incompatible
> with the precomputed path. `y_pred` semantics vary by function: soft
> scores/probabilities for curve functions, hard labels for matrix functions
> and `classification_report_chart`, fitted values for `residuals_chart` and
> `prediction_error_chart`. See each function's docstring for the exact
> expected shape.

> **Schwabish SB3 — 2026-05-11 — figure-function defaults.** Eight
> Group-B functions adopt the Schwabish defaults from §3.19's principles doc:
>
> - `roc_chart`: `annotate_auc=True` by default; one `AUCLabel` per curve,
>   staggered along the y axis so labels do not collide at the line endpoint.
>   (This bullet's original text gave single-curve charts the active title
>   `"ROC — AUC X.XXX"` and multi-curve charts a descriptive `"ROC"` title;
>   the metric-carrying title is superseded by the 2026-09-13 #219/D4 note in
>   §3.15 — every ROC chart is titled `"ROC Curve"` and the overlay is the
>   sole metric surface.)
> - `pr_chart`: `annotate_ap=True` by default; overlay parallels `roc_chart`
>   (its active title is superseded by the same 2026-09-13 note). Binary
>   classifiers additionally render a dashed horizontal baseline at the
>   positive-class prevalence so the chance-level floor is always visible.
> - `calibration_chart`: new `annotate_brier=True` kwarg (default on). A
>   corner `BrierLabel` carries the per-sample Brier score (see the
>   2026-09-10 dated note immediately below — this bullet's original text,
>   "multi-model charts get one `BrierLabel` per model from the bin-level
>   reliability data," is superseded; its active title is superseded by the
>   2026-09-13 #219/D4 note).

> **2026-09-10 (batch G, F-L11-04 — supersedes the bin-level compared-path
> text above; itself partly superseded by the batch G close note that
> follows it, which replaces the `precomputed_value=` mechanism described
> here with a `brier` column on the frame — read both):** the `BrierLabel`
> overlay and the title now show the exact
> same per-sample Brier score, invariant to `n_bins=`/`strategy=`. Before
> this fix, the title computed the true per-sample Brier from
> `model.predict_proba`, but the overlay independently integrated the
> chart's own *binned* reliability-diagram data — not the Brier score at
> all, and a number that moved with `n_bins=`/`strategy=` even though the
> real Brier score never does (observed: 0.078 → 0.028 on `strategy=` alone
> while the true value held at 0.096). `_apply_metric_label_explicit` gains
> `precomputed_value: float | dict[str, float] | None` — when set, the
> overlay renders this value directly and `metric_fn`/`metric_field` are
> never consulted; the calibration builder passes the per-sample Brier it
> already computes for the title, so overlay and title cannot disagree.
> **Compared-path (`compare=`) semantics change, disclosed:** each model's
> label now shows **its own true per-sample Brier** (a `{model_name: brier}`
> dict passed as `precomputed_value=`), replacing the bin-level number every
> model previously shared. The compared-path title stays the plain
> `"Calibration Curve"` — no single number describes several models.
> **Precomputed path** (`calibration_chart(y_true=, y_pred=)`): the title
> now carries the Brier too (computed from the held `y_true`/`y_prob`
> arrays via `_PrecomputedSource.brier_score()`, the same formula as every
> other path — one Brier formula, `ferrum._metric_labels._brier_score`, used
> everywhere), where previously it stayed blank.

> **2026-09-10 (batch G close — supersedes the `precomputed_value=`
> mechanism described in the note immediately above, which was written
> earlier the same day; the user-visible contract below is the current
> one):** the per-sample Brier now lives
> on the **frame**, not in a builder-side override. `.calibration_curve()`
> gains a broadcast `brier` column (schema above), computed at frame
> assembly from the per-sample predictions and the binarized target — the
> last point at which the per-sample arrays exist — exactly as `auc`/`ap`
> are broadcast onto ROC/PR frames. The chart title, the figure builder's
> overlay, and the composable `chart + BrierLabel()` path all read that one
> column through the same rule (`_curve_metric`: read the kind's broadcast
> column when the frame carries it, integrate the drawn curve only when it
> does not), so `_apply_metric_label_explicit`'s `precomputed_value=`
> parameter and `_PrecomputedSource.brier_score()` are both removed —
> neither has a caller once the number rides on the frame. **User-visible
> consequence:** `calibration_chart(...) + fm.BrierLabel()` — the example
> in `BrierLabel`'s own docstring — now renders the true per-sample Brier
> matching the title, on both `strategy="uniform"` and
> `strategy="quantile"`. It previously rendered the bin-level integral
> (0.030 uniform / 0.009 quantile against a true 0.028 on a reference
> fixture) *and overwrote* the correct label the figure had already placed,
> leaving a chart strictly less correct than its own default. Composing
> `+ AUCLabel()`/`+ APLabel()` onto a `roc_chart`/`pr_chart` likewise now
> reads the kernel's `auc`/`ap` column rather than re-integrating; the
> values agreed already, so no rendered number moves there. A frame with no
> such column — arbitrary user data, the reason the composable path exists
> — still integrates the metric kind's own value function over the drawn
> curve, unchanged. (The `metric_fn` parameter that phrase originally named
> is gone: a surface now names a metric *kind* and the table supplies its
> value function, column and display name together.)
> - `residuals_chart`: new `annotate_metrics=True` kwarg (default on).
>   Overlays a top-right corner annotation with `R²` / `RMSE` / `MAE` on
>   the residuals-vs-fitted view in both the single-panel and the 4-panel
>   `panels="auto"` layouts. Implementation shares a pair of helpers —
>   `_inject_metrics_corner` (data augmentation) and
>   `_overlay_metrics_corner` (chart overlay) — so the orchestration
>   (which panels get the corner) stays separate from the implementation
>   (how to render it).
> - `importance_chart`: new `show_values=True` kwarg (default on). Emits
>   the formatted importance value at each bar's end via a same-data
>   `mark_text` overlay.
> - `learning_curve_chart` and `validation_curve_chart`: replace the
>   categorical color legend with endpoint-anchored direct labels (`train`
>   / `test`) via the new private `_direct_label_endpoint` helper. Legend
>   suppression flows through the new `Color(field, legend=None)` honored
>   kwarg (serialized to the renderer as `legend = {"disabled": true}`).
> - `confusion_matrix_chart`: docstring already specified `annotate=True`
>   default; SB3 added a regression test.
>
> Every Group-B function also accepts an opt-in `subtitle: str = None`
> kwarg that flows through to the `Title(text, subtitle=…)` primitive.
>
> Implementation lives in `src/ferrum/_diagnostics/charts.py` for the
> single-model builders and `src/ferrum/_direct_label.py` for the
> direct-label helper; primitives in `src/ferrum/annotations.py`. The
> renderer-side legend suppression and `mark_text` align / dx / dy
> support land in `crates/ferrum-core/src/render/prepare.rs` and
> `…/marks/text.rs` respectively.

---

### 3.15 Sklearn-Protocol Visualizers

Visualizers implement `fit` / `score` / `show` for drop-in compatibility with sklearn workflows. Unlike Yellowbrick, `.show()` always returns a `Chart` object — not a matplotlib figure.

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

> **Note (2026-06-22, cohesion campaign T4.6):** The base `score(X, y)`
> delegates to `float(self.model.score(X, y))` whenever the wrapped model
> exposes a `.score` method, and returns `0.0` only for genuinely no-model
> visualizers (`model=None`: rank / parallel-coordinates / class-balance /
> elbow). Subclasses whose score is not the estimator's own (e.g.
> `ROCVisualizer` → `roc_auc_score`) override it. `has_score` is a derived
> read-only property (not a hand-set class flag): it reports `True` exactly
> when `score()` returns a real metric, mirroring the model-has-`.score`
> guard, so the two can never drift.
>
> **Refinement (2026-06-23, cohesion campaign S4FIX2):** the "can never drift"
> guarantee now holds for the two non-standard subclass shapes as well.
> `ROCVisualizer` overrides `has_score` to mirror its own `score()` condition
> (scoreable when the model exposes `predict_proba` or a callable `.score`), so
> a `predict_proba`-only model — no `.score`, a valid sklearn shape — reports
> `has_score=True` and `score()` returns the AUC. A multi-model
> `CalibrationVisualizer` overlay (two or more positional models, or a single
> dict-of-models) has no single well-defined metric: it now reports
> `has_score=False` and `score()` returns the documented `0.0` no-single-score
> fallback, rather than silently scoring only the first model. Single-model
> calibration behavior is unchanged.

**Concrete visualizers:**

| Class | Wraps |
|---|---|
| `ROCVisualizer(model, *, micro=True, macro=True, per_class=True, annotate_auc=True, subtitle=None, compare=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | ROC chart |
| `PRVisualizer(model, *, per_class=True, average="macro", annotate_ap=True, iso_lines=False, subtitle=None, compare=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | PR chart |
| `ConfusionMatrixVisualizer(model, *, normalize="true", annotate=True, mark=None, encode=None, properties=None, layers=None, theme=None)` | Confusion matrix |
| `ClassificationReportVisualizer(model, *, mark=None, encode=None, properties=None, layers=None, theme=None)` | Heatmap of precision/recall/F1 per class |
| `CalibrationVisualizer(*models, n_bins=10, strategy="uniform", annotate_brier=True, subtitle=None, compare=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Calibration chart |
| `ResidualsVisualizer(model, *, kind="studentized", cook_threshold=None, panels=None, annotate_metrics=True, subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Residuals diagnostic |
| `PredictionErrorVisualizer(model, *, reference_line=True, ci=None, reference_band=False, mark=None, encode=None, properties=None, layers=None, theme=None)` | Prediction error |
| `FeatureImportancesVisualizer(model, *, method="builtin", top_k=20, orient="horizontal", error_bars=True, show_values=True, subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Importance chart |
| `LearningCurveVisualizer(model, *, cv=5, scoring=None, train_sizes=None, ci_style="band", subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Learning curve |
| `ValidationCurveVisualizer(model, param, values, *, cv=5, scoring=None, log_scale="auto", ci_style="band", subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Validation curve |
| `SilhouetteVisualizer(model, *, k=None, subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Silhouette chart (clusterers) |
| `ElbowVisualizer(model_class, *, ks, metric="distortion", mark=None, encode=None, properties=None, layers=None, theme=None)` | Elbow chart |
| `ManifoldVisualizer(model=None, *, method="umap", mark=None, encode=None, properties=None, layers=None, theme=None)` | Embedding projection |
| `ClassBalanceVisualizer(*, mark=None, encode=None, properties=None, layers=None, theme=None)` | Class frequency bar chart |
| `CooksDistanceVisualizer(model, *, threshold=None, kind="studentized", panels=("residuals_vs_leverage",), annotate_metrics=True, subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Cook's distance |
| `SHAPVisualizer(model, *, kind="beeswarm", max_display=20, sample_idx=None, order="abs_mean", background=None, per_class=False, zero_line=True, mark=None, encode=None, properties=None, layers=None, theme=None)` | SHAP chart — **deprecated** since 2026-05-12 (P3.6); use the three sibling classes below. |
| `SHAPBeeswarmVisualizer(model, *, max_display=20, order="abs_mean", background=None, per_class=False, zero_line=True, mark=None, encode=None, properties=None, layers=None, theme=None)` | SHAP beeswarm |
| `SHAPBarVisualizer(model, *, max_display=20, order="abs_mean", background=None, per_class=False, mark=None, encode=None, properties=None, layers=None, theme=None)` | SHAP mean-absolute bar |
| `SHAPWaterfallVisualizer(model, *, sample_idx, max_display=20, order="abs_mean", background=None, per_class=False, mark=None, encode=None, properties=None, layers=None, theme=None)` | SHAP single-sample waterfall |
| `DiscriminationThresholdVisualizer(model, *, n_thresholds=50, metrics=("precision", "recall", "f1", "queue_rate"), cv=None, threshold_line=False, optimum_label=True, subtitle=None, compare=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Discrimination threshold chart (binary classifiers only) |
| `ParallelCoordinatesVisualizer(*, features=None, hue=None, rescale="minmax", alpha=0.5, mark=None, encode=None, properties=None, layers=None, theme=None)` | Parallel coordinates. No model required; `fit(X, y)` accepts raw feature matrix. `hue` applied from `y` if provided. |
| `ClassPredictionErrorVisualizer(model, *, normalize=False, show_counts=True, mark=None, encode=None, properties=None, layers=None, theme=None)` | Class prediction error chart |
| `PCAVarianceVisualizer(model, *, n_components=None, cumulative_line=True, threshold=0.95, subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | PCA scree plot. Model must expose `explained_variance_ratio_` attribute (sklearn PCA, TruncatedSVD, or compatible). |
| `Rank1DVisualizer(*, algorithm="shapiro", orient="horizontal", top_k=None, color_field=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Univariate feature ranking bar chart. No model required. |
| `Rank2DVisualizer(*, algorithm="pearson", annot=True, mark=None, encode=None, properties=None, layers=None, theme=None)` | Pairwise feature correlation heatmap. No model required. |
| `AlphaSelectionVisualizer(model, alphas, *, cv=5, scoring=None, log_scale=True, highlight_best=True, mark=None, encode=None, properties=None, layers=None, theme=None)` | Regularization parameter selection curve |
| `InterclusterDistanceVisualizer(model, *, method="mds", k=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Intercluster distance map |
| `CVScoresVisualizer(model, *, cv=5, scoring=None, kind="box", split="both", mark=None, encode=None, properties=None, layers=None, theme=None)` | Cross-validation score distribution |

> **2026-09-10 (batch G, F-L11-06 + F-L12-09 + F-L10-07 — full
> visualizer↔chart signature parity):** every visualizer above gained, as
> explicit keyword-only constructor parameters, every kwarg its underlying
> chart builder documents — including the uniform override quartet
> `mark=`/`encode=`/`properties=`/`layers=` on every single visualizer in
> the family (checkpoint decision: no quartet exemptions). This closes the
> gap where `PredictionErrorVisualizer` was the family's only fully-threaded
> member and every other visualizer silently dropped documented builder
> kwargs, with no `mark=`/`encode=`/`properties=`/`layers=` override
> reachable from any visualizer at all. `FerrumVisualizer.__init__`'s prior
> `**_extra: Any` silent swallow is removed — an unknown constructor kwarg
> on any visualizer now raises `TypeError` naming it, instead of being
> silently accepted and discarded. `ResidualsVisualizer(panels="auto")`
> renders the canonical 4-panel grid identical to
> `residuals_chart(panels="auto")` (a shared helper now owns the
> `"auto"` → 4-panel-list expansion, consumed by both callers — a naive
> per-caller forward previously fell back to single-panel, since the
> builder's own vocabulary is `None`/`"single"`/an explicit list, never
> `"auto"`). `CooksDistanceVisualizer`'s new `panels=` defaults to exactly
> its historical `("residuals_vs_leverage",)` tuple — unchanged output for
> every existing call. Three pre-existing spelling drifts are kept for
> backward compatibility, not unified (spelling unification is a logged
> follow-up, not this batch): `ROCVisualizer.micro=`/`macro=` vs. the
> builder's `average=`; `CooksDistanceVisualizer.threshold=` vs. the
> builder's `cook_threshold=`; `CalibrationVisualizer`'s variadic `*models`
> vs. the public `calibration_chart`'s `compare=`. A mechanical parity test
> (`inspect.signature` diff against each visualizer's actual `_build_chart`
> target, patterned on `test_core_stub_parity.py`) enforces every
> visualizer's keyword-only surface against its builder's, with an
> explicit, rationale-carrying skip-map for exactly those three drifts —
> an empty-rationale or stale skip-map entry fails the test, as does any
> `FerrumVisualizer` subclass missing from the test's own completeness
> registry. No existing kwarg was renamed or removed on any visualizer.
>
> **`compare=` metric semantics (F-L10-07).** `ROCVisualizer`, `PRVisualizer`,
> and `DiscriminationThresholdVisualizer` accept `compare=` per the drift
> map above; their derived metrics (`auc_mean`, `ap_mean`,
> `best_threshold`/`best_f1`, and `__repr__`'s summary) always describe the
> **base model** (the visualizer's own `model=` argument), consistent with
> `.score()` — never a value pooled across every compared model, and never
> silently substituted with a stronger/weaker rival's number. Before this
> fix, e.g. `DiscriminationThresholdVisualizer._metrics` searched the
> concatenated per-model frame for the best F1 regardless of which model it
> came from, so the object could report one model's `.score()` and a
> different model's derived metrics simultaneously.
> `CalibrationVisualizer`'s own genuinely-symmetric multi-model shape
> (`has_score=False`, `score()=0.0`, no distinguished base) is unaffected —
> `compare=` on the other three has one distinguished base model (the
> primary `model=` argument) where `CalibrationVisualizer`'s `*models` does
> not. **Rider (2026-09-13, wave 3 T5, #213):** that fix reached the
> *metrics* only. `DiscriminationThresholdVisualizer._build_chart` kept
> forwarding the unfiltered compared source to the pooled chart builder, so
> the rendered F1-optimum annotation could still name the rival while
> `_metrics`/`__repr__` named the base. `compare=` on that visualizer now
> renders small multiples (see the 2026-09-13 note in the `compare=`
> composition section), so the `"base"` panel's annotation and the repr
> describe the same model by construction. `ROCVisualizer` and
> `PRVisualizer` never had the render half of this problem: their charts
> overlay by `color="model"`, so every curve is already labeled with the
> model it belongs to.

> **2026-09-13 (wave 3, #212 — `CalibrationVisualizer(compare=)`).**
> Supersedes both statements above about `CalibrationVisualizer`: the
> drift-map row naming its variadic `*models` as an adjudicated drift from
> `calibration_chart`'s `compare=`, and the sentence saying its
> "genuinely-symmetric multi-model shape" stands apart from `compare=`'s
> distinguished base. The class now accepts **both**, as two different
> questions rather than one drift:
>
> - `CalibrationVisualizer(model, compare={"rival": other})` has one
>   distinguished base — the positional `model`, labelled `"base"` — with
>   exactly the semantics `ROCVisualizer`/`PRVisualizer`/
>   `DiscriminationThresholdVisualizer` already have: `has_score` is `True`,
>   `.score()` scores the base model, and `_metrics` carries a single
>   `calibration_error`, the base model's own (never pooled across rivals).
>   Before this change the kwarg did not exist and the call raised
>   `TypeError: CalibrationVisualizer.__init__() got an unexpected keyword
>   argument 'compare'`.
> - `CalibrationVisualizer(m_a, m_b)` / `CalibrationVisualizer({"a": m_a,
>   "b": m_b})` is **unchanged**: symmetric, no base, `has_score=False`,
>   `score()` returns the `0.0` fallback, and `_metrics` carries one
>   `calibration_error[<model name>]` key per model. The 2026-06-23
>   refinement above still describes this shape exactly.
> - The two are mutually exclusive. Several positional models (or a
>   dict-of-models) together with `compare=` is a self-contradicting
>   request — the two spell different metric contracts — and raises
>   `ValueError` at construction naming both idioms.
>
> `fit` resolves `compare=` through the same shared helper the three curve
> visualizers use, so `compare={"base": ...}` and a non-dict `compare=`
> refuse with the same messages on all four, and the rendered chart is
> byte-identical to `calibration_chart(model, X, y, compare=...)` — the
> parity is the constructor and the metric semantics, not a new render
> path. No default, no existing call's output, and no golden changed.

> **2026-05-11 (Phase 10 — Model Diagnostics):** Every visualizer constructor
> documented above accepts `random_state: int | None = None` as an additional
> keyword argument. The base `FerrumVisualizer.__init__` stores it on `self`
> and forwards it to the underlying `ModelSource` (or to `_diagnostics/stats`
> compute helpers for the no-model variants). Visualizers whose backing
> compute is deterministic (`ROCVisualizer`, `PRVisualizer`,
> `ConfusionMatrixVisualizer`, `CalibrationVisualizer`,
> `ResidualsVisualizer`, `PredictionErrorVisualizer`,
> `DiscriminationThresholdVisualizer`, `PCAVarianceVisualizer`,
> `Rank2DVisualizer` — except `algorithm="kendall"` which is also
> deterministic, `ClassBalanceVisualizer`, `ClassificationReportVisualizer`,
> `ClassPredictionErrorVisualizer`) ignore the value; the rest propagate
> it. (`DiscriminationThresholdVisualizer`'s membership in this deterministic
> list was briefly false while its `cv=` path used a shuffled splitter — see
> the dated 2026-09-13 §3.1 note for that history; the unshuffled
> `StratifiedKFold` restores it.) The `ElbowVisualizer(model_class, *, ks, ...)` signature is unique —
> it takes a model **class** and fits one instance per k inside `fit()`.

---

### 3.16 Output and Rendering

#### `RenderConfig`

```
RenderConfig(
    format="svg",         # "svg" | "png" | "html" | "json"
    scale=2.0,            # pixel density multiplier for PNG
    embed_fonts=True,     # embed fonts in SVG/HTML
    background=None,      # override chart background for export
    width=None, height=None,
    engine="ferrum",      # "ferrum" | "vega-lite"  (vega-lite emits JSON spec)

    # Auto-raster policy (see §3.3 and §3.17)
    raster_threshold=500_000,   # int | None — mark count above which auto-raster
                                # is triggered. None disables auto-raster entirely.
                                # Overrides ferrum.config.set_raster_threshold()
                                # for this render call.
    raster_behavior="warn",     # "warn" | "silent" | "error" — controls whether
                                # a warning is emitted when auto-raster
                                # substitution occurs.
    raster_aggregate="count",   # "count" | "density" | "mean" | "sum" | "any" —
                                # default aggregate for auto-raster substitution
    raster_field=None,          # str | None — column to aggregate when
                                # raster_aggregate is "mean" or "sum"
    raster_cmap="viridis",      # default colormap for auto-raster substitution

    # Backend selection (see §3.17)
    backend=None,               # "svg" | "tiny-skia" | "wgpu" | None (auto-select)
    tile_parallel=False,        # enable Rayon tile parallelism in the tiny-skia
                                # backend. Increases memory usage; recommended
                                # for charts with > 500k marks.
    font_path=None,             # str | list[str] | None — override font search
                                # paths for the tiny-skia backend. None = use
                                # bundled Ferrum fonts with system font fallback.
)
```

`raster_threshold=None` and `raster_behavior="silent"` are not equivalent: `None` disables auto-raster entirely (the chart renders all marks through the requested backend, even if performance suffers); `"silent"` keeps auto-raster active but suppresses the substitution warning.

> **Dated note (2026-09-03, batch B task 5 — `raster_aggregate` validates at construction):** `raster_aggregate` is checked at `RenderConfig` construction against its true accepted set — `"count"` | `"density"` | `"mean"` | `"sum"` | `"any"` — matching the underlying Rust `Raster` transform exactly. `"max"`/`"min"`/`"median"` were never valid (despite being reachable at render time before this task, where they surfaced a confusing runtime error well after construction) and now raise `ValueError` immediately. The new `raster_field=` names the column to aggregate when `raster_aggregate` is `"mean"` or `"sum"` (both require it and raise at construction if it's missing); it is genuinely ignored — dropped before the substitution, never resolved as a column name — for `"count"`, `"density"`, and `"any"`, which need no field. The aggregated column must be `Float64`-typed; a non-`Float64` column raises `stat_raster: column '...' must be Float64` at render time (a pre-existing constraint of `Raster`, not new here). `mark_raster` and `desugar_raster`'s docstrings are truth-fixed to match this accepted set — `min`/`max` were never valid and are no longer listed as if they were.
>
> **Dated note (2026-09-05, batch D task 2 — correcting the note above):** the aggregated column's dtype constraint stated above no longer holds. The batch-D M1 dtype-policy sweep widened `Raster`'s `x`, `y`, and `field` columns to accept any integer, unsigned, or float dtype (coerced to `Float64`); the quoted `stat_raster: column '...' must be Float64` error no longer exists anywhere in the tree. A non-numeric column now raises `stat_raster: column '<name>' must be numeric, got <Dtype>` instead, naming the transform, column, and actual dtype. This is the general M1 dtype-widening policy applied consistently across the transform engine (design spec `2026-09-05-batch-d-transform-engine-design.md` §4.1); `raster` was one of the transforms the batch-D sweep touched.

> **2026-05-09 (Phase 7 implementation note):** Phase 7 honors the following
> `RenderConfig` fields: `scale`, `embed_fonts`, `background`, `width`,
> `height`. The remaining fields (`format`, `engine`, `raster_threshold`,
> `raster_behavior`, `raster_aggregate`, `raster_cmap`, `backend`,
> `tile_parallel`, `font_path`) are deferred to subsequent phases that ship
> their corresponding features. `embed_fonts` is treated as always-true in
> Phase 7 for visual determinism (rendered text uses the bundled Inter
> Regular regardless of system font availability); future phases may surface
> the `False` case for size-conscious users.

> **2026-05-10 (Phase 8a):** `.show()` env detection in 8a covers Jupyter
> inline (`_repr_svg_` / `_repr_html_` rich display) and a browser fallback
> (writes temp HTML, calls `webbrowser.open`). Sixel terminal output and the
> standalone HTML wrapper output are deferred to Phase 9. `.save()` honors
> `.svg` and `.png` extensions; `.html` and `.json` raise
> `NotImplementedError` pointing to Phase 9.

#### Chart output methods

The output surface follows one convention: `to_*` returns an in-memory value,
`show()` displays, `save(path)` writes to disk.

```
chart.show(*, renderer=None)                  # auto-detect environment, display
chart.to_svg() -> str                         # in-memory SVG markup
chart.to_png() -> bytes                        # in-memory PNG bytes
chart.to_html() -> str                         # in-memory interactive HTML
                                               #   (byte-identical to save('.html'))
chart.to_spec() -> ChartSpec                  # internal dataclass (Chart only)
chart.to_json(*, indent=None) -> str
chart.save(path, *, format=None, **render_kwargs)
chart.pipe(fn, *args, **kwargs) -> Any        # apply a function to self
```

These methods exist on `Chart` and on every composition view (`HConcatChart`,
`VConcatChart`, `LayerChart`, `ConcatChart`, `JointChart`, `RepeatChart`,
`ClusterMapChart`), with one exception: `to_spec` exists on `Chart` only — a
`ChartSpec` is leaf-only by construction; composition views expose `to_dict()`/
`to_json()` for their declaration (see the 2026-09-11 dated note below). On
composition views `to_png` is scale-only (it accepts `scale=` but no `raster`
argument).

**Deprecated aliases.** `show_svg()` and `show_png()` are retained as deprecated
aliases that forward to `to_svg()` / `to_png()` and emit a `DeprecationWarning`.
They are slated for removal after `0.16.0`. New code should call `to_svg()` /
`to_png()` / `to_html()`.

> **Note (2026-06-04):** The in-memory output methods were renamed
> `show_svg()` -> `to_svg()` and `show_png()` -> `to_png()`, and `to_html() -> str`
> was added (returns the interactive HTML string, byte-identical to
> `save('.html')`). This unifies the surface on the `to_*` = in-memory-value
> convention, leaving `show()` for display and `save(path)` for disk. The old
> `show_svg`/`show_png` names remain as deprecated aliases that forward to the
> new methods and emit a `DeprecationWarning`; they are slated for removal after
> `0.16.0`. The rename applies to `Chart` and all composition views; on
> composition views `to_png` is scale-only (no `raster` argument).

> **Dated note (2026-09-11, batch H task 11 — `to_dict`/`to_json`/`pipe` generalize; `to_spec` stays `Chart`-only, F-L14-01/F-L14-02/F-L06-04):** `to_json(*, indent=None)` and `pipe(fn, *args, **kwargs) -> Any` are now real on every composition view (`HConcatChart`, `VConcatChart`, `ConcatChart`, `LayerChart`, `JointChart`, `RepeatChart`, `ClusterMapChart`), not just `Chart` — this table's blanket "these methods exist on ... every composition view" claim was itself stale before this task, since neither existed on any of the seven. Each composite also gains `to_dict() -> dict`: a *recursive* declaration dict (`{"kind": <layout>, ...layout fields..., <child slots>: <embedded dict(s)>}`) generalizing the pre-existing `JointChart.spec` / `RepeatChart.spec` / `ClusterMapChart.spec` introspection properties — a leaf `Chart` child embeds as `json.loads(chart.to_spec().to_json())` (round-trippable through `ChartSpec.from_json`), and a nested composite child now embeds *recursively* via its own `to_dict()` instead of the old `None` placeholder. Those three `.spec` properties are unchanged in shape and now recursive-consistent with `to_dict` through the one shared embedding helper (`composition._embed_chart_spec`) — for `JointChart`/`RepeatChart`/`ClusterMapChart`, `to_dict()` returns `.spec` (plus figure chrome when set — see the 2026-09-11 note below).
>
> **Figure chrome in the declaration dict (2026-09-11, batch H close remediation).** A composition's figure-level `title` / `subtitle` / `caption` — set via `.properties(title=..., subtitle=..., caption=...)` — is part of its declaration, so `to_dict()` carries it under those key names, omitted when unset. The **six** `_CompositeBase` views (`HConcatChart`, `VConcatChart`, `ConcatChart`, `JointChart`, `RepeatChart`, `ClusterMapChart`) carry all three, because all three are figure-level slots on those classes. `LayerChart` carries **`title` only**: it is a single-plot overlay with one `_title` slot, and `.properties(subtitle=..., caption=...)` on it fans those kwargs to its member charts (a layer's own title blob) rather than storing figure chrome — that fan-out is pre-existing behavior, lossy for this purpose (a fanned `caption` reaches no layer slot at all), and is named here rather than papered over. Before this correction only `LayerChart` emitted `title` (and emitted it as `None` when unset) while the other six silently dropped every one of the three, so `(a | b).properties(title="t").to_dict()` had no title anywhere. Because unset chrome is omitted, a chrome-free composition's dict is unchanged, and `JointChart`/`RepeatChart`/`ClusterMapChart`'s `to_dict()` still equals `.spec` exactly when untitled — for a titled one it is `.spec` plus the chrome keys. The `.spec` properties themselves stay pure layout introspection and never gain chrome.
>
> **Chrome values are normalized at ingress, LOSSLESSLY where a lossless
> answer exists, and applied to all three chrome keys (2026-09-11, batch H
> close cycle 3, H3-4; made lossless and widened to `subtitle=`/`caption=`
> in cycle 4).** All three of `title=`/`subtitle=`/`caption=` accept a bare
> `str` or a `ferrum.Title`, normalized at `.properties()` — before it ever
> reaches `_figure_title`/`_figure_subtitle`/`_figure_caption` (or
> `LayerChart`'s single `_title`) — not at `to_dict()`/serialization time.
> **`title=fm.Title(text=..., subtitle=...)` routes the `Title`'s own
> `subtitle` into the composite's separate, already-working `subtitle` slot**
> (the six `_CompositeBase` views only — `LayerChart` has no such slot, see
> below) when no explicit `subtitle=` was ALSO given in the same call;
> giving both **in the same `.properties()` call** is refused typed, naming
> both spellings, the same idiom as the conditional-vocabulary collision
> refusals. This scoping is deliberate, not an oversight (2026-09-11, batch
> H close cycle 5): **across separate calls it is not a collision at all** —
> a later `.properties()` call's routed subtitle silently replaces an
> earlier one, the same last-write-wins rule an explicit `subtitle=` already
> follows across chained calls (`.properties(subtitle="OLD").properties(title=fm.Title("T",
> subtitle="NEW"))` ends with the figure subtitle `"NEW"`, no warning — a
> routed subtitle behaves exactly like an explicit one once it lands in the
> `subtitle` slot). This corrects the cycle-3
> shipped behavior, which flattened `title=fm.Title(..., subtitle=...)` to
> `.text` and **silently dropped** the subtitle with no warning — the same
> disease this batch refused `shape` conditionals for elsewhere in the same
> cycle ("accepted and inert"), reached here via the declaration layer
> instead of the wire. A `Title` field with genuinely nowhere to go on ANY
> chrome slot — `anchor`, `offset`, `font_size`, `font_weight`, `color`,
> `subtitle_font_size`, `subtitle_color` on any of the three keys; `subtitle`
> itself on `subtitle=`/`caption=` (nesting makes no sense) or on
> `LayerChart`'s `title=` (no separate subtitle slot exists there) — is
> refused typed rather than silently dropped, naming the field(s) and the
> fix. `to_dict()` still serializes `title`'s (already-plain) text through
> `Title.to_spec_dict()`, so the top-level `title` stays `{"text": ...}`,
> identical in shape to every embedded child's `title`; the routed subtitle
> surfaces as the top-level `subtitle` KEY (a dict sibling of `title`, not
> nested inside `title`'s own dict), since that is the slot that actually
> renders.
>
> **The composite-tree renderer only ever accepted a bare-string title (and
> subtitle/caption), and now the declaration surface agrees for all three
> (2026-09-11, batch H close cycle 3, H3-4; widened in cycle 4).** Every
> composite output path (all six `_CompositeBase` views' `to_svg`/
> `.interactive()`, and `LayerChart.to_svg`'s default/shared-y path) lowers
> the figure title into the one-call Rust composite entry's wire `title`
> field, which is a bare `Optional<String>` — there is no per-figure-title
> richness slot in the composite tree the way a leaf `Chart`'s
> `ChartSpec.title` has one (a full `TitleSpec` with
> subtitle/anchor/offset/colors). Cycle 3 fixed this for `title=` only;
> cycle 4 found `subtitle=fm.Title(...)`/`caption=fm.Title(...)` still
> raised the identical unmapped PyO3 `TypeError: 'Title' object is not an
> instance of 'str'` from `to_svg()` on every wrapper including `LayerChart`
> (whose `subtitle=`/`caption=` fan a normalized string to member charts
> rather than storing composite-level chrome — pre-existing, unrelated to
> this fix), while `to_dict()`/`to_json()` silently accepted the same call.
> All three keys, on all seven wrappers, now agree: `.properties(title=...,
> subtitle=..., caption=...)` with any `fm.Title(...)` either renders or
> refuses typed at `.properties()` time — it never builds a declaration
> `to_svg()` cannot honor. `ferrum.Title`'s own docstring previously claimed
> it was "accepted everywhere a title string is accepted" including
> `HConcat/VConcat/Layer(title=...)`; that claim is corrected to describe
> the routing/refusal rule above.
>
> **`to_spec() -> ChartSpec` stays `Chart`-only.** `ChartSpec` is a leaf-only Rust dataclass (one mark/encoding/transform set); a composite view has no single `ChartSpec` to return, so it does not gain this method — the original wording above implying `to_spec` was universal across composition views was the stale claim being corrected here, not a feature being removed.
>
> **Composite construction now validates its children (H17).** `HConcatChart` / `VConcatChart` / `ConcatChart` / `LayerChart` / `JointChart` / `RepeatChart` / `ClusterMapChart` each raise `TypeError: {Class} children must be Chart or composition objects, got {TypeName}` at construction when a child is neither a `Chart` nor another composition object (e.g. `fm.HConcatChart([a_dataframe])`), instead of constructing silently and failing much later at render time with `_lower_any`'s generic, node-agnostic `ValueError: unrecognized node kind '...'` — that render-time error remains as a backstop for node shapes the construction-time check cannot describe (e.g. a `JointChart` marginal that is a composite but not a leaf `Chart`, still rejected by `_composite_tree`). `Chart.__or__`/`__and__` and every composition view's `__or__`/`__and__` now return `NotImplemented` for a non-chart, non-composition operand (mirroring `Chart.__add__`'s existing `NotImplemented` fallback), so `chart | 3` raises Python's own operator `TypeError` ("unsupported operand type(s) for |: ...") rather than reaching the constructor at all.

#### Environment detection order for `.show()`

1. Jupyter / IPython → inline SVG (static) or WASM widget (if `.interactive()`)
2. VS Code notebook → same
3. ~~Terminal with sixel support → PNG via sixel~~ — **REMOVED (2026-05-15):** sixel is niche, inconsistent across terminal emulators, and ferrum's audience is Jupyter/browser-first. Not worth implementation effort.
4. Otherwise → write temp HTML and open browser

---

### 3.17 Rendering Backends

Ferrum has three rendering backends. The backend used for a given output is determined by the call (`.save()` vs `.show()` vs `.interactive()`), the file extension or format, the mark count, and any explicit override in `RenderConfig` or `ferrum.config`.

#### SVG Backend

- Default for static output when mark count is below threshold.
- Resolution-independent, text-searchable, scalable.
- Not viable above ~50k marks.
- Output: `.svg` files; inline SVG in notebooks.

#### tiny-skia Backend (CPU Rasterizer)

- Pure Rust, zero system dependencies (no Cairo, no libpng, no X11).
- Bundled in the wheel; works on Linux, macOS, Windows, ARM.
- Text rendering via `fontdue` (rasterization) + `rustybuzz` (shaping); fonts bundled or resolved from system font paths.
- Default for `.save("chart.png")` and `.show()` when mark count exceeds threshold.
- Not used for interactive output.
- Performance: ~200–800 ms for 1M antialiased marks (CPU-bound, SIMD-optimized); tile-based Rayon parallelism available as opt-in via `RenderConfig`.
- Default output scale: `2.0` (retina); configurable in `RenderConfig`.
- Limitations: no vector output; zoom invalidates raster and requires re-render.

#### wgpu / WASM Backend (GPU Interactive)

- Used only when `.interactive()` is called.
- `wgpu` targets WebGPU (Chrome 113+, Edge 113+, Safari 17+) with automatic WebGL2 fallback for broader compatibility.
- Vello is not used; its compute-shader-based approach is incompatible with the WebGL2 fallback path. Instanced draws for simple marks (points, bars) and tessellation for curves and areas require no compute shaders and work across both WebGPU and WebGL2.
- Text and axis labels rendered via CSS overlay (real DOM text; accessible, no font bundling required in browser context).
- Pan and zoom are GPU matrix transforms — no Python round-trip, no re-render.
- Selections that trigger Python-side recomputation require a Python kernel round-trip (~50–200 ms in local notebooks); managed via the `anywidget` protocol.
- Jupyter integration via `anywidget`; browser integration via standalone HTML bundle.
- WASM bundle size: ~2–4 MB compressed (CDN-hosted recommended).
- On macOS including M-series, the native `wgpu` backend targets Metal; unified memory architecture reduces GPU buffer upload latency vs. discrete GPUs.
- Viewport changes (zoom, resize) trigger re-rasterization of `mark_raster` layers at the new resolution.

#### Backend selection

The threshold column refers to `raster_threshold` (default `500_000`), configurable in `RenderConfig` and `ferrum.config`.

| Call                                            | Backend                            |
|-------------------------------------------------|------------------------------------|
| `.save("chart.svg")`                            | SVG (warns if `mark_count > 50k`)  |
| `.save("chart.png")`                            | tiny-skia                          |
| `.show()` — `mark_count < threshold` (def 500k) | SVG inline                         |
| `.show()` — `mark_count >= threshold`           | tiny-skia PNG                      |
| `.show()` in headless / no display              | tiny-skia PNG (always)             |
| `.show()` in terminal with sixel support        | ~~tiny-skia → sixel~~ — removed    |
| `.interactive()`                                | wgpu / WASM                        |

Per-call `RenderConfig` takes precedence over global `ferrum.config` settings for all threshold and backend parameters.

---

### 3.18 Data Source Compatibility

Ferrum accepts the following as `data` in `Chart(data=...)` or figure-level functions:

| Type | Notes |
|---|---|
| `polars.DataFrame` | Zero-copy via Arrow |
| `polars.LazyFrame` | Collected at render time; lazy evaluation where possible |
| `pandas.DataFrame` | Converted to Arrow once on first access |
| `pyarrow.Table` | Native |
| `pyarrow.RecordBatch` | Native |
| `dict[str, list]` | Converted to Arrow inline |
| `list[dict]` | Converted to Arrow inline |
| `numpy.ndarray` (2D) | Columns named `col_0`, `col_1`, ... |
| `ModelSource` | Wraps an estimator; derived data accessed via `.predictions()` etc. |
| `ComparedModelSource` | Multi-model wrapper; adds `model` column to all derived data |
| `str` / `pathlib.Path` | CSV, Parquet, JSON, NDJSON; loaded lazily |
| `None` | Data supplied per-layer |

> **2026-05-10 (Phase 8a):** Data input compatibility provided via narwhals
> (~1.x) for pandas, modin, cuDF, dask, ibis. Polars goes via direct CDI
> (zero-copy). pyarrow `Table` and `RecordBatch` accepted as native. Dict,
> list-of-records, and 2D numpy with auto-named columns supported. File path
> inputs (`Chart("file.csv")`) and `ModelSource`/`ComparedModelSource`
> deferred to Phases 9 and 10 respectively.

> **Dated note (2026-09-25, wave 6C, `#167`(1) — `Utf8View`/`LargeUtf8` string columns normalize through an Arrow cast):** a string column arriving as `Utf8View` or `LargeUtf8` (e.g. from a pyarrow/polars source using Arrow's newer string encodings) now casts to `Utf8` via `arrow::compute::cast` and renders byte-identically to the same column carried as plain `Utf8`. A cast that cannot complete refuses with a typed `RenderError::StringNormalizationFailed` naming the column and the reason, instead of an internal error. `Utf8` columns are unaffected.

---

### 3.19 Utilities

```
# ferrum.data — REMOVED (2026-05-15)
# Users source sample data from sklearn/seaborn optional dependencies.
# A ferrum-native dataset loader duplicates existing ecosystem tooling
# with no user-facing benefit. This namespace will not be implemented.

ferrum.color.palette(scheme, n)            # return n colors from a scheme as hex list
ferrum.color.to_hex(color, *, scale=None)  # normalize color to hex; scale="unit"|"byte"
ferrum.color.sequential(name, n=256)       # n render-truth samples of a sequential scheme
ferrum.color.diverging(name, n=11)         # n render-truth samples of a diverging scheme
ferrum.color.diverging_palette(low, mid, high, n)

# 2026-06-22 (T2.2, ENC-06/XNAME-02/XSIB-07/ENC-11): the palette registry in
# crates/ferrum-core/src/render/palette.rs + .../color/continuous.rs is the
# single source of truth. ferrum.color consumes it through the _core accessors
# (list_palettes / palette_kind / palette_colors / palette_sample); the
# hand-mirrored Python hex tables are gone. Surfaced behavior changes:
#   * color.palette()/sequential()/diverging() for the 7 colorous-backed
#     schemes (viridis, plasma, magma, inferno, cividis, blues, rdbu) now
#     return RENDER-TRUTH colors (what actually renders), replacing the old
#     hand-picked 7-stop approximations. The 19 other palettes are unchanged.
#   * scheme= on Color/Fill/Stroke is validated at declaration time against the
#     registry (a bogus name raises ValueError immediately, not at render).
#   * to_hex(color, scale=) takes an explicit "unit"/"byte" override; the
#     default (scale=None) heuristic is now range-based (any component > 1 means
#     byte), so (1,0,0) and (1.0,0.0,0.0) agree. Pass scale="byte" for the old
#     all-integers-are-bytes interpretation of ambiguous <=1 tuples.
#   * scheme= is the canonical colormap kwarg on mark_raster/mark_contour/
#     mark_hex/clustermap; cmap= remains a documented back-compat alias.

ferrum.config.set_max_rows(n)              # raise/lower data size guard (default: None)
ferrum.config.set_renderer(renderer)       # default renderer for .show()
ferrum.config.set_default_width(n)
ferrum.config.set_default_height(n)

ferrum.config.set_raster_threshold(n)      # mark count threshold for auto-raster
                                           # substitution. Pass None to disable.
                                           # Default: 500_000.
ferrum.config.set_raster_behavior(mode)    # "warn" (default), "silent", "error"
ferrum.config.set_default_backend(backend) # "svg", "tiny-skia", "wgpu", or None (auto)
ferrum.config.set_font_paths(*paths)       # prepend paths to the font search list
                                           # used by the tiny-skia backend

ferrum.Title(text, *, subtitle=None, anchor="start", offset=None,
             font_size=None, font_weight=None, color=None,
             subtitle_font_size=None, subtitle_color=None)

ferrum.Grid(major=True, minor=False, *,
            color=None, width=None, dash=None, opacity=None,
            major_color=None, minor_color=None,
            major_dash=None, minor_dash=None,
            major_width=None, minor_width=None,
            major_opacity=None, minor_opacity=None)
# 2026-05-30 (item 18): the bare color=/width=/dash=/opacity= shorthand
# (shown in the Theme example above) is a FALLBACK that sets BOTH the major
# and minor level; an explicit major_*/minor_* overrides that level. minor
# gridlines render on continuous axes only (linear/log/time/pow/sqrt/symlog);
# categorical/discretizing axes have no continuum to subdivide, so minor=True
# is a documented no-op there. Major ticks/gridlines are scale-projected (a
# tick at value v coincides with a data mark at v); minors subdivide between
# them on the same grid.

ferrum.Aggregate(fn, field, *, as_=None)   # used in transform_aggregate
ferrum.WindowTransform(fn, *, field=None, param=None, as_=None, peer=False)
```

---

---

## Part IV: Extension Points

### Custom Marks

Implement the `MarkProtocol`:

```
class MyMark:
    def to_primitive_layers(self, data: ArrowTable, encodings: Encodings) -> list[PrimitiveLayer]:
        ...
```

Register: `ferrum.register_mark("mark_my_mark", MyMark)`

### Custom Stat Transforms

Implement `StatProtocol`:

```
class MyStat:
    def apply(self, data: ArrowTable) -> ArrowTable:
        ...
```

Register: `ferrum.register_stat(MyStat)`

### Custom Themes

Themes are plain dataclass instances. No registration needed — pass directly to `.theme()`.

### Renderer Plugins

Implement `RendererProtocol` and register via `ferrum.register_renderer(name, RendererClass)`. The built-in renderers (`"svg"`, `"png"`, `"html"`, `"json"`) cannot be overridden, only supplemented.

---

---

## Appendix: Version Target

This document describes **Ferrum 1.0 API surface**. Nothing here exists yet.

The intent is that 1.0 be complete enough that a practitioner moving from Altair, Seaborn, or Yellowbrick finds every chart type they reach for without falling back to matplotlib.

Post-1.0 candidates (explicitly out of scope for 1.0):

- `mark_network` / graph layout
- `mark_gantt` / timeline
- Geographic tile layers (Mapbox, OpenStreetMap)
- 3D coordinate system
- Animation / `frame` encoding
- Real-time streaming data sources
- Julia and R bindings
