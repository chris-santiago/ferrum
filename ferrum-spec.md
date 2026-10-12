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
