# Ferrum
## A Statistical Visualization Library — Concept & API Specification

> *Fast. Composable. Statistically honest.*

---

## Part I: Philosophy

### 1. Core Beliefs

#### Grammar first, convenience second
Every visualization — from a scatter plot to a SHAP beeswarm — is a composition of the same primitives: data, encodings, marks, scales, and coordinate systems. Ferrum commits to this without compromise. High-level convenience functions (`displot`, `lmplot`, `roc_chart`) exist, but they are sugar over the grammar, not parallel universes with different rules.

#### Model artifacts are data
A confusion matrix is a cross-tabulation. A ROC curve is a sorted transformation of predicted probabilities. A SHAP value is a column. There is no reason these should require a separate API or a different mental model. Ferrum treats model outputs as data sources that feed into the standard grammar. This eliminates the Yellowbrick problem: diagnostic plots are composable, themeable, and interactive because they are charts, not special objects.

#### Statistics belong in the rendering pipeline, not in userspace
Computing a KDE, bootstrapping a confidence interval, or fitting a LOESS curve should not require the user to call SciPy before plotting. These are stat transforms: first-class operations declared in the chart spec and executed in the Rust engine before rendering. The user declares intent; Ferrum computes.

> **2026-05-11 (Phase 10 — Model Diagnostics):** Phase 10 places model-diagnostic
> compute in the `ModelSource` adapter layer (Python, lazy-imported sklearn
> delegation) rather than as Rust transforms in the rendering pipeline. From the
> user's perspective the figure function (`ferrum.roc_chart`, etc.) *is* the
> rendering pipeline — they are not computing ROC in userspace, which is what
> this constraint actually proscribes. Whether the internal compute is a Rust
> transform or a Python call to sklearn is invisible at the call site.
> Model-diagnostic compute is also entangled with the model-specific protocol
> (`predict_proba`, `classes_`, etc.), which a generic Rust transform cannot
> access without reimplementing sklearn's estimator protocol. The one exception
> is `ferrum._core.kendall_tau_b` (Knight's O(n log n) tau-b), which runs as a
> Rust function called from `ModelSource.rank2d(algorithm="kendall")` — pure
> numeric work over two f64 arrays, no estimator protocol involved.

#### Interactivity is a renderer, not a rewrite
You should not need to learn a different API to make a chart interactive. `.interactive()` switches the render target from SVG to a WASM canvas. Selections, zoom, pan, and linked views are declared in the chart spec and handled by the renderer. Plotly's fatal flaw is that interactive charts and static charts are different objects. Ferrum has one chart object.

#### Zero unnecessary copies
Python is the declaration layer. Rust is the computation layer. Data moves between them once, over the Arrow C Data Interface (CDI). Stat transforms, layout, binning, and aggregation happen in Rust. The Python process never touches row-level data again after the initial handoff.

> **Amendment 2026-05-09:** Original spec said "Arrow IPC" (byte-serialized stream format). After design review for Phase 2, we chose the Arrow C Data Interface instead. Polars DataFrames implement `__arrow_c_stream__` natively — CDI passes the buffer pointer directly with zero copies, whereas IPC would serialize to bytes and deserialize on the Rust side. The `pyo3-arrow` crate mediates the CDI boundary in PyO3. The spirit of the constraint ("data moves once, no row-level Python access after handoff") is preserved; only the wire format changed.

#### Defaults should be correct, not just pretty
Default color schemes are perceptually uniform and colorblind-safe (OKabe-Ito for categorical, Viridis for sequential). Default font sizes pass WCAG contrast. Default bin counts follow Sturges' rule as a floor. These are not aesthetic opinions — they are epistemically correct starting points.

---

### 2. Design Constraints

- **No matplotlib dependency.** Ever. Not as a fallback, not for "legacy support."
- **No mutable global state.** No `rcParams`, no `set_theme()` that mutates a module-level object. Themes are values passed to charts.
- **No magic inference that silently fails.** If Ferrum infers a scale or encoding type incorrectly, it raises a descriptive error with a suggested fix.
- **Sklearn-compatible, not sklearn-dependent.** The model diagnostics layer works with any object that implements `predict`, `predict_proba`, or `transform`. Sklearn is not imported unless the user passes a fitted sklearn model (or uses an sklearn-backed extra such as SHAP). The raw-array path (`y_true=` / `y_pred=`) computes ROC, precision-recall, calibration, confusion-matrix, and threshold-sweep metrics on native Rust kernels with no sklearn dependency.
- **One output format per render call.** A chart produces SVG, PNG, HTML (WASM bundle), or a Vega-Lite JSON spec. Producing all four from the same spec is supported; producing ambiguous mixed output is not.

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

The library is named **Ferrum** (Latin: iron). Iron is the substrate of Rust. It is also load-bearing, structural, and unglamorous — which is what a visualization engine should be.

The Python package imports as `ferrum`. Internal Rust crate: `ferrum-core`. WASM renderer: `ferrum-wasm`.

---

---

## Part II: Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Python Layer                        │
│  Chart spec builder · Encoding DSL · Convenience API   │
└────────────────────┬────────────────────────────────────┘
                     │  PyO3 + Arrow CDI (pyo3-arrow)
┌────────────────────▼────────────────────────────────────┐
│                    Rust Core                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Stat Engine │  │Layout Engine │  │ Scale Engine │  │
│  │  KDE · CI    │  │Constraint    │  │Domain/Range  │  │
│  │  Regression  │  │solver ·      │  │mapping ·     │  │
│  │  Binning     │  │Facet layout  │  │Tick gen      │  │
│  │  Aggregation │  │Legend place  │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                     │
  ┌──────────────────────────────────────────────────┐
  │               Render Pipeline                    │
  │                                                  │
  │   Geometry pass → SceneGraph                     │
  │                       │                          │
  │          ┌────────────┼────────────┐             │
  │          ▼            ▼            ▼             │
  │      SVG backend  tiny-skia    wgpu/WASM         │
  │      (vector)     (CPU raster) (GPU/interactive) │
  └──────────────────────────────────────────────────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
    SVG/PNG      HTML+WASM     JSON spec
   (static)    (interactive)  (interop)
```

SceneGraph is the renderer-agnostic intermediate representation — a flat list of geometric primitives (filled paths, stroked paths, rectangles, circles, glyphs, images) consumed by all three backends.

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

**Parameters**
- `data` — Any of: `pandas.DataFrame`, `polars.DataFrame`, `polars.LazyFrame`, Arrow `Table`, `ModelSource`, or `None` (data supplied per-layer).
- `width` / `height` — Integer pixels, or `"container"` for responsive sizing.
- `title` — String or `Title` object.
- `description` — Accessible alt-text string; included in SVG and HTML output.

**Methods** (all return `self` unless noted)
- `.encode(**channels)` — Bind encoding channels. Channels inherited by all layers unless overridden.
- `.layer(*layers)` — Add one or more `Layer` objects.
- `.mark_*(...)` — Shorthand: creates a single-layer chart with the given mark. Equivalent to `.layer(Layer(mark=mark_*(...)))`.
- `.transform(*transforms)` — Add data transforms applied before any stat computation.
- `.stat(*stats)` — Add stat transforms (computed per-layer if applied at layer level).
- `.facet(field=None, *, row=None, col=None, ncols=None, nrows=None)` — Wrap chart in a `FacetSpec`.
- `.add_selection(*selections)` — Bind `Selection` objects.
- `.conditional(selection, **true_encodings, **false_encodings)` — Conditional encoding based on selection state.
- `.properties(**kwargs)` — Set chart-level metadata (title, width, height, description).
- `.theme(theme)` — Apply a `Theme` object.
- `.interactive(*, toolbar: bool = True)` — Switch render target to WASM. Returns `InteractiveChart`.
- `.resolve(scale=None, axis=None, legend=None)` — Override shared/independent resolution in compound views.
- `.save(path, *, format=None, scale=2.0, backend=None, **render_kwargs)` — Render and write to disk. `format` inferred from extension if `None`. `backend` overrides auto-selection. `scale` applies to raster backends only.

> **2026-05-22 (feat/rtree-toolbar — Toolbar & Auto-tooltips):**
>
> `.interactive(*, toolbar: bool = True)` — the `toolbar` kwarg (default `True`)
> renders a Bokeh-style toolbar overlaid on the canvas. Five tools are included:
> Pan (keyboard shortcut P), Box Zoom (Z), Box Select (S), Reset (R), and Save PNG.
> The default active tool depends on whether the chart has an `selection_interval`
> declared: if so, Box Select is active on load; otherwise Pan is active. Pass
> `toolbar=False` to suppress the toolbar entirely (useful for embedded or
> space-constrained contexts).
>
> **Auto-tooltips** — when rendering interactively, hover tooltips are
> automatically injected from the chart's encoded channels (x, y, color, size,
> etc.). No explicit `tooltip=` encoding is needed for basic hover. If an explicit
> `tooltip=` encoding is present it takes precedence and auto-injection is skipped.
> Auto-tooltips only affect interactive renders; SVG and PNG output are unchanged.
>
> `.save(path, *, ..., toolbar: bool = True)` — for HTML format, `toolbar`
> controls whether the toolbar appears in the exported file. Defaults to `True`.
> Has no effect on SVG/PNG/JSON output.
- `.show()` — Display in current environment (notebook, terminal sixel, or browser).
- `.to_spec()` — Return the internal `ChartSpec` dataclass (serializable).
- `.to_json()` — Return JSON string of the chart spec.
- `__or__(other)` → `HConcatChart`
- `__and__(other)` → `VConcatChart`

---

#### `ferrum.Layer`

Explicit layer constructor. Used when multiple layers need independent data, marks, or encodings.

```
Layer(data=None, mark=None, *, encoding=None, stat=None, transform=None)
```

Most users use `mark_*()` methods on `Chart` instead of constructing `Layer` directly.

---

#### `ferrum.ModelSource`

Wraps a fitted estimator and a dataset; exposes derived data sources as Arrow tables.

```
ModelSource(model, X, y=None, *, feature_names=None, class_names=None,
            sample_weight=None, random_state=None)
```

> **2026-05-11 (Phase 10 — Model Diagnostics):** `random_state: int | None = None`
> was added as the sixth keyword-only constructor argument. It is propagated to
> every derived-data method whose underlying sklearn / shap / umap call accepts
> an RNG seed (importances permutation, SHAP background sampling, UMAP / t-SNE
> embeddings, MDS / t-SNE cluster-center projection, learning_curve / validation_curve /
> cv_scores / alpha_selection cross-validation, partial_dependence sampling).
> Methods whose compute is deterministic by construction (predictions, residuals,
> roc_curve, pr_curve, confusion_matrix, calibration_curve, cumulative_gain,
> lift_curve, discrimination_threshold, pca_variance, rank1d/rank2d) ignore it.

**Methods** (all return `polars.DataFrame` unless noted)
- `.predictions()` — `y_true`, `y_pred`, `residual`, `studentized_residual`
- `.probabilities()` — `y_true`, one column per class with predicted probability
- `.roc_curve(*, average=None, drop_intermediate=True)` — `fpr`, `tpr`, `threshold`, `class`, `auc`
- `.pr_curve(*, average=None)` — `precision`, `recall`, `threshold`, `class`, `ap`
- `.confusion_matrix(*, normalize=None)` — `actual`, `predicted`, `value`, `value_fmt`
- `.calibration_curve(*, n_bins=10, strategy="uniform")` — `mean_predicted`, `fraction_positive`, `count`
- `.cumulative_gain()` — `percent_population`, `gain`, `class`
- `.lift_curve()` — `percent_population`, `lift`, `class`
- `.importances(*, method="builtin", n_repeats=30, scoring=None, random_state=None)` — `feature`, `importance`, `std`, `rank`
- `.shap_values(*, background=None, max_evals=500)` — `sample_id`, `feature`, `shap_value`, `feature_value`, `feature_value_normalized`, `class_label`

> **2026-05-12 (P3.11, D15):** `shap_values()` always includes a `class_label`
> column. Regression sources emit `class_label="target"` on every row.
> Binary classifiers emit the positive-class name (single value, schema
> unchanged in shape). Multi-class classifiers emit one row per
> `(sample, feature, class)` — total height `n_samples * n_features * n_classes`.
> See `mark_shap_*(per_class=...)` for the chart-side overlay hook.
- `.partial_dependence(features, *, grid_resolution=100, kind="average")` — `feature`, `feature_value`, `pd_value`, `sample_id` (if `kind="individual"`)
- `.silhouette(k)` — `sample_id`, `cluster`, `silhouette_value`
- `.embeddings(*, method="umap", n_components=2, **method_kwargs)` — `dim_0`, `dim_1`, (`dim_2`), `label`
- `.learning_curve(*, cv=5, scoring=None, train_sizes=None)` — `train_size`, `split`, `score`, `mean_score`, `std_score`, `lower`, `upper`
- `.validation_curve(param, values, *, cv=5, scoring=None)` — `param_value`, `split`, `score`, `mean_score`, `lower`, `upper`
- `.discrimination_threshold(*, n_thresholds=50, cv=None)` — `threshold`, `precision`, `recall`, `f1`, `queue_rate`. Binary classifiers only. Sweeps the decision threshold and records precision, recall, F1, and queue rate at each point. `cv`: if provided, averages over cross-validation folds.
- `.pca_variance(*, n_components=None)` — `component`, `explained_variance_ratio`, `cumulative_variance_ratio`. Model must expose `explained_variance_ratio_` (e.g. sklearn PCA, TruncatedSVD). `n_components`: limit output to first n components.
- `.rank1d(*, algorithm="shapiro")` — `feature`, `score`, `rank`. Univariate feature ranking. `algorithm`: `"shapiro"` (Shapiro-Wilk normality score) | `"variance"` | `"covariance"` (covariance with target y).
- `.rank2d(*, algorithm="pearson")` — `feature_x`, `feature_y`, `correlation`. Pairwise feature correlation matrix. `algorithm`: `"pearson"` | `"spearman"` | `"kendall"` | `"covariance"`.
- `.cv_scores(*, cv=5, scoring=None)` — `fold`, `split`, `score`. Cross-validation score per fold. `split`: `"train"` | `"test"`. `scoring`: sklearn scorer string or callable; defaults to model's default scorer.
- `.alpha_selection(alphas, *, cv=5, scoring=None)` — `alpha`, `fold`, `score`, `mean_score`, `std_score`. CV score at each regularization parameter value. `alphas`: array-like of alpha values to evaluate. Intended for Ridge, Lasso, ElasticNet, and other regularized estimators.
- `.intercluster_distance(k, *, method="mds")` — `cluster`, `x`, `y`, `size`. MDS or t-SNE projection of cluster centers, with `size` = membership count. `k`: number of clusters. `method`: `"mds"` | `"tsne"`.

**Class method**
- `ModelSource.compare(models: dict[str, estimator], X, y, **kwargs)` → `ComparedModelSource`
  - All derived methods return the same schemas with an additional `model` column.

---

### 3.2 Encoding Channels

> **2026-05-10 (Phase 9):** Adds **position adjustments** as a sibling concept
> to encoding channels. Four immutable value classes — `Identity`, `Dodge`,
> `Jitter`, `Stack` — are passed via `position=` on eligible marks (Chart- or
> Layer-level). They serialize to `{"type": "<kind>", ...}` and are consumed
> by the Rust ChartSpec. Eligibility is enforced at chart-build time. The
> matrix below mirrors `src/ferrum/position.py`:
>
> | Position | Eligible marks |
> |---|---|
> | `Identity` | every mark (default; no-op) |
> | `Dodge` | `bar`, `point`, `box`, `boxplot`, `boxen`, `swarm`, `violin`, `errorbar`, `errorband`, `ribbon`, `histogram`, `density` |
> | `Jitter` | `point`, `swarm`, `tick` |
> | `Stack` | `bar`, `area`, `ribbon`, `histogram`, `density` |
>
> `Stack.offset` accepts `"zero"` (default) or `"normalize"` (percent-fill
> stacking). `Jitter.axis` accepts `"x"`, `"y"`, or `"both"`. `Dodge.by`
> selects the grouping field; if omitted, the active color/fill encoding
> field is used. Ineligible (mark, position) pairs raise `ValueError` at
> build time, not at render time.

Encoding channels are typed objects passed as keyword arguments to `.encode()`. All channels accept a field name string as a shorthand.

#### Positional

| Class | Shorthand alias | Description |
|---|---|---|
| `X(field, *, type=None, bin=False, aggregate=None, scale=None, axis=None, title=None, stack=None, sort=None, impute=None)` | `"fieldname"` | Horizontal position |
| `Y(...)` | `"fieldname"` | Vertical position |
| `X2(field)` | — | Horizontal span end (for `mark_rect`, `mark_rule`) |
| `Y2(field)` | — | Vertical span end |
| `XError(field)` | — | Symmetric error bar extent on X |
| `YError(field)` | — | Symmetric error bar extent on Y |
| `XError2(field)` | — | Asymmetric error upper bound on X |
| `YError2(field)` | — | Asymmetric error upper bound on Y |
| `Theta(field, *, stack=True)` | — | Angular position (arc/pie marks) |
| `Radius(field)` | — | Radial position |

#### Appearance

| Class | Description |
|---|---|
| `Color(field, *, type=None, scheme=None, scale=None, legend=None, title=None)` | Fill and stroke color |
| `Fill(field, ...)` | Fill only (overrides Color fill) |
| `Stroke(field, ...)` | Stroke only |
| `Opacity(field, *, scale=None, legend=None)` | Overall opacity |
| `FillOpacity(field, ...)` | Fill opacity |
| `StrokeOpacity(field, ...)` | Stroke opacity |
| `StrokeWidth(field, *, scale=None)` | Stroke width |
| `StrokeDash(field, *, scale=None)` | Dash pattern |
| `Size(field, *, scale=None, legend=None)` | Point size or line width |
| `Shape(field, *, scale=None, legend=None)` | Point shape (circle, square, cross, diamond, triangle-*) |
| `Angle(field, *, scale=None)` | Point rotation angle |

#### Text / Detail / Tooltip

| Class | Description |
|---|---|
| `Text(field, *, format=None, formatType=None)` | Text label content |
| `Detail(field)` | Group-by without a visual channel |
| `Tooltip(*fields)` | Hover tooltip fields; accepts strings or `TooltipField` objects |
| `TooltipField(field, *, title=None, format=None, formatType=None)` | Tooltip field with formatting |
| `Href(field)` | URL for click-through on interactive charts |
| `Description(field)` | Accessible description per mark |
| `Key(field)` | Identity key for animated transitions |

#### Facet (used with `.facet()`)

| Class | Description |
|---|---|
| `Facet(field, *, type=None, sort=None, title=None)` | Single facet dimension |
| `FacetRow(field, ...)` | Row facet dimension |
| `FacetCol(field, ...)` | Column facet dimension |

---

**Channel shorthand strings**

All positional and appearance channels accept:
- `"fieldname"` — bare field
- `"aggregate(fieldname)"` — e.g. `"mean(price)"`, `"count()"`, `"q50(latency)"`
- `"fieldname:Q"` / `":N"` / `":O"` / `":T"` — inline type annotation (Quantitative, Nominal, Ordinal, Temporal)

> **2026-05-10 (Phase 8a):** All 31 channel classes are constructible Python
> value objects. Renderer honors `x`, `y`, `color`, `size`, `shape`, `opacity`
> in Phase 8a. Other channels (Stroke, Fill, FillOpacity, StrokeOpacity,
> StrokeWidth, StrokeDash, Angle, Text, Detail, Tooltip, TooltipField, Href,
> Description, Key, X2, Y2, XError, YError, XError2, YError2, Theta, Radius)
> are accepted at the API and stored on `EncodingSpec`, but the renderer
> ignores them with a one-time `UserWarning` per (channel, render call).
> Phase 9 wires the remaining channels.
>
> Channel kwargs honored in 8a: `type`, `bin`, `aggregate`, `scale`, `title`.
> Other kwargs (`axis`, `legend`, `sort`, `stack`, `impute`, `scheme`, `format`,
> `formatType`) are accepted, stored typed on `EncodingSpec`, and warn-once.
>
> **2026-05-11 (Phase 9, P1.5):** "one-time `UserWarning` per (channel,
> render call)" is implemented as **one-time per channel, process-wide**
> via `ferrum._warn.warn_once`. The stricter dedupe is the practical
> interpretation: notebook re-renders of the same chart should not stack
> dozens of identical warnings, and per-render dedupe would degrade to a
> per-process registry whose lifetime users can't reason about anyway.
> Tests reset the registry via `ferrum._warn.reset_warnings()` between
> cases; in user code the warning fires the first time a channel is
> accepted-but-not-rendered after import. Phase 9 still drops several
> of the listed channels (Fill, Stroke, FillOpacity, StrokeOpacity,
> StrokeWidth, StrokeDash, Angle, Text, Detail, Tooltip, TooltipField,
> Href, Description, Key, XError, YError, XError2, YError2, Theta,
> Radius); X2, Y2, and the existing `text` slot now render and were
> moved to the honored set.

---

### 3.3 Marks

Marks are constructors that accept visual property overrides as keyword arguments. All marks inherit from `MarkBase`.

> **2026-05-10 (Phase 9):** `mark_segment` is shipped as a primitive mark in
> Phase 9. It draws a line segment from `(x, y)` to `(x2, y2)` and is
> diagonal-capable — distinct from `mark_rule`, which is axis-aligned.
> Required encodings: `X`, `Y`, `X2`, `Y2`. Use cases: dumbbell charts,
> slope graphs, network edges, free-form annotations. Removed from
> `PHASE_9_PLUS_MARKS` warn-list.

#### Primitive Marks

| Mark | Description | Key Parameters |
|---|---|---|
| `mark_point(...)` | Scatter / dot plot | `size`, `shape`, `filled`, `stroke_width` |
| `mark_line(...)` | Line chart | `interpolate`, `stroke_width`, `stroke_cap`, `stroke_join` |
| `mark_area(...)` | Area chart | `interpolate`, `line`, `opacity` |
| `mark_bar(...)` | Bar chart | `orient`, `corner_radius`, `width`, `bin_spacing` |
| `mark_rect(...)` | Filled rectangle; heatmaps | `corner_radius` |
| `mark_rule(...)` | Horizontal or vertical reference line | `stroke_dash`, `stroke_width` |
| `mark_text(...)` | Text labels | `align`, `baseline`, `dx`, `dy`, `font_size`, `font_weight`, `angle`, `limit` |
| `mark_tick(...)` | Rug / tick marks | `band_size`, `orient` |
| `mark_arc(...)` | Arc / pie / donut | `inner_radius`, `outer_radius`, `pad_angle`, `corner_radius` |
| `mark_image(...)` | Image at point | `width`, `height`, `align`, `baseline` |
| `mark_geoshape(...)` | Choropleth / geographic shape | `projection` |
| `mark_segment(...)` | Line segment between (x,y) and (x2,y2). Requires X, Y, X2, Y2 encoding channels. Used for dumbbell charts, slope graphs, network edges, and free-form annotations. | `stroke`, `stroke_width`, `stroke_dash`, `stroke_cap` (`"butt"`\|`"round"`\|`"square"`), `arrow` (bool), `arrow_size` |
| `mark_label(...)` | Text with a solid background fill box. All `mark_text` parameters, plus: | `background_fill`, `background_stroke`, `background_stroke_width`, `background_padding`, `background_corner_radius` |

#### Composite Marks (expand to multiple primitive layers)

> **2026-05-10 (Phase 9):** Adds `mark_boxen` (letter-value plot) as a
> composite mark. Expands to a stack of nested rectangles per letter-value
> depth plus outlier points; uses the `LetterValue` stat transform. Key
> parameters: `k_depth` (`"tukey"` \| `"proportion"` \| `"trustworthy"` \|
> `"full"` \| `int`), `k_proportion` (float in `(0, 1)`, used when
> `k_depth="proportion"`), `outlier_threshold` (float; rows beyond the
> outermost letter value are flagged outliers), `palette` (sequential
> palette name; nested rectangles fade toward the median).

| Mark | Expands To | Key Parameters |
|---|---|---|
| `mark_boxplot(...)` | box + whisker + outlier points | `extent` (`"min-max"` or float IQR multiplier), `size`, `outliers` |
| `mark_boxen(...)` | nested rectangles (letter values) + outlier points | `k_depth` (`"tukey"`\|`"proportion"`\|`"trustworthy"`\|`"full"`\|`int`), `k_proportion`, `outlier_threshold`, `palette` |
| `mark_errorbar(...)` | rule + tick | `extent` (`"ci"`, `"stderr"`, `"stdev"`, `"iqr"`), `ticks` |
| `mark_errorband(...)` | area + line | `extent`, `borders` |
| `mark_ribbon(...)` | area between Y and Y2 | `opacity`, `interpolate` |

#### Statistical Marks (trigger stat engine)

| Mark | Stat Computed | Key Parameters |
|---|---|---|
| `mark_smooth(...)` | LOESS or regression fit + CI band | `method` (`"loess"`, `"lm"`, `"glm"`, `"gam"`), `ci`, `bandwidth`, `degree`, `n` |
| `mark_density(...)` | KDE | `bandwidth` (`"scott"`\|`"silverman"`\|float), `kernel`, `extent`, `cumulative`, `multiple` (`"layer"`\|`"stack"`\|`"fill"`\|`"dodge"`, default `"layer"`) |
| `mark_histogram(...)` | Frequency / density bins | `bin_count`, `bin_width`, `density`, `cumulative`, `right`, `multiple` (`"layer"`\|`"stack"`\|`"fill"`\|`"dodge"`, default `"layer"`) |
| `mark_contour(...)` | 2D density contours | `bandwidth`, `thresholds`, `smooth` |
| `mark_violin(...)` | KDE + optional boxplot overlay | `bandwidth`, `inner` (`"box"`, `"quartile"`, `"point"`, `None`) |
| `mark_qq(...)` | Quantile–quantile | `distribution` (`"normal"`, `"uniform"`, `"exponential"`, or `scipy.stats` dist), `dequantize` |
| `mark_raster(...)` | Aggregates data to a pixel-resolution grid and renders as a colored image | `aggregate` (`"count"`, `"density"`, `"mean"`, `"sum"`, `"any"`), `field` (required if `aggregate` is `"mean"` or `"sum"`), `cmap`, `resolution` (`"screen"`, `int`, `tuple[int, int]`), `blend` (`"alpha"`, `"additive"`), `min_count`, `log_scale` |
| `mark_swarm(...)` | Beeswarm for categorical data. Points are arranged along the categorical axis to avoid overplotting while preserving the value distribution. General-purpose; distinct from `mark_shap_beeswarm`. | `size`, `orient`, `spacing`, `side` (`"both"`\|`"left"`\|`"right"`), `dodge` |
| `mark_hex(...)` | Hexagonal binning. Aggregates data into a hexagonal grid and renders each hex colored by aggregate value. Preferred over `mark_raster` when topology and shape perception matter. | `bin_size`, `aggregate` (`"count"`\|`"mean"`\|`"sum"`), `field` (required if `aggregate` is mean or sum), `cmap`, `stroke`, `stroke_width` |
| `mark_function(fn, ...)` | Plot a Python callable f(x) → y as a line. `fn` receives a numpy array of x values and must return a numpy array of y values. Domain inferred from X scale if not specified. | `domain` (`tuple[float, float]`), `n` (evaluation points, default 200), `clip` |

**Auto-raster (`mark_raster` as a policy):**

`mark_raster` is the explicit form of density rasterization. Auto-raster is a policy layer that implicitly substitutes `mark_raster` for the original mark when the per-layer mark count exceeds a threshold (`raster_threshold`, see §3.16 and §3.18).

When auto-raster fires:
- Color encodings on the original layer are dropped and replaced with a density colormap.
- Tooltip encodings are dropped.
- A warning is emitted (configurable via `raster_behavior`).
- A density colorbar legend replaces the original legend.

Auto-raster will **not** fire if the chart has an active color encoding — doing so would silently discard user intent. Either remove the color encoding or use `mark_raster` explicitly.

Auto-raster behavior is configurable via `raster_behavior`: `"warn"` (default), `"silent"`, or `"error"`.

> *(2026-05-10) Phase 8b: `blend="additive"` is deferred to Phase 11 (interactive renderer). Auto-raster policy (`raster_threshold`, `raster_behavior`) is deferred to Phase 9+; explicit `mark_raster` is implemented in Phase 8b.*

> *(2026-05-10) Phase 8b: `mark_swarm` `dodge=` parameter deferred (single-group swarm only in 8b).*

> **2026-05-12 (Phase 9, F16 audit):** Type inference for the color
> channel follows the same rule as the axis scales: continuous color is
> selected when the `type=` argument on `Color(...)` is `"Q"` /
> `Quantitative` or `"T"` / `Temporal`, or when `type=` is `None` and
> the column dtype is numeric (any width: Float32/64, Int8/16/32/64,
> UInt8/16/32/64) or temporal (Date32/64, Timestamp). All other cases
> fall through to a categorical color scale. A user-supplied
> `type="quantitative"` on a non-numeric column raises
> `EncodingTypeMismatch` per the line-52 "no magic inference that
> silently fails" rule.

> *(2026-05-10) Phase 8b: `mark_hex` only count/mean/sum aggregates supported. Other Vega-Lite aggregates warn-once and fall back to count.*

**Bivariate density:** When both X and Y channels are encoded as quantitative fields, `mark_density` switches to bivariate KDE mode and renders filled contours, equivalent to `mark_contour` with `fill=True`. The `multiple` parameter applies to univariate mode only.

#### Model Diagnostic Marks

| Mark | Description | Key Parameters |
|---|---|---|
| `mark_residuals(...)` | Residuals vs fitted | `kind` (`"raw"`, `"studentized"`, `"scaled"`), `reference_line`, `cook_threshold` |
| `mark_prediction_error(...)` | Actual vs predicted | `identity_line`, `ci`, `reference_band` |
| `mark_confusion(...)` | Confusion matrix heatmap + text | `normalize` (`None`, `"true"`, `"pred"`, `"all"`), `text_fmt` |
| `mark_roc(...)` | ROC curve line | `average` (`None`, `"micro"`, `"macro"`, `"weighted"`), `reference_line`, `annotate_auc` |
| `mark_pr(...)` | Precision-recall curve | `average`, `annotate_ap`, `iso_lines` |
| `mark_calibration(...)` | Calibration curve | `n_bins`, `strategy`, `reference_line` |
| `mark_gain(...)` | Cumulative gain curve | `reference_lines` |
| `mark_lift(...)` | Lift curve | `reference_line` |
| `mark_importance(...)` | Feature importance bar | `orient`, `error_bars`, `top_k` |
| `mark_shap_beeswarm(...)` | SHAP beeswarm | `max_display`, `color_bar`, `order` (`"abs_mean"`, `"max_abs"`) |
| `mark_shap_bar(...)` | SHAP mean absolute bar | `max_display`, `layered` |
| `mark_shap_waterfall(...)` | SHAP waterfall (single prediction) | `max_display`, `show_data` |
| `mark_pdp(...)` | Partial dependence + ICE | `kind` (`"average"`, `"individual"`, `"both"`), `ice_alpha`, `center` |
| `mark_silhouette(...)` | Silhouette plot per cluster | `line_width`, `zero_line` |
| `mark_learning_curve(...)` | Learning curve with CI band | `ci_style` (`"band"`, `"errorbar"`) |
| `mark_validation_curve(...)` | Validation curve with CI band | `log_scale`, `ci_style` |
| `mark_decision_boundary(...)` | 2D classification boundary | `grid_resolution`, `alpha`, `background`, `contour_levels` |
| `mark_discrimination_threshold(...)` | Precision, recall, F1, and queue rate vs decision threshold for binary classifiers. Useful for threshold selection under class imbalance. | `metrics` (list of metrics to display, default all four), `n_thresholds`, `threshold_line` (bool, marks estimated optimal threshold) |
| `mark_parallel_coordinates(...)` | Parallel coordinates plot. Each sample is a polyline drawn across vertically-arranged feature axes. General-purpose: accepts any tabular data, not only model output. | `rescale` (`"minmax"`\|`"zscore"`\|`None`), `alpha`, `highlight_selection` (bool) |
| `mark_class_prediction_error(...)` | Stacked bar chart of predicted class counts, colored by actual class. Distinct from confusion matrix: shows absolute prediction volume per class and reveals systematic over/under-prediction. | `orient`, `normalize` (bool) |
| `mark_pca_scree(...)` | Bar chart of explained variance ratio per principal component with optional cumulative variance line overlay. | `n_components`, `cumulative_line` (bool, default `True`), `threshold_line` (float, draws a horizontal line at this cumulative variance level, e.g. 0.95) |
| `mark_rank1d(...)` | Horizontal bar chart of feature scores from a univariate ranking algorithm. | `algorithm` (`"shapiro"`\|`"variance"`\|`"covariance"`), `orient`, `top_k` |
| `mark_rank2d(...)` | Heatmap of pairwise feature correlations or covariances. | `algorithm` (`"pearson"`\|`"spearman"`\|`"kendall"`\|`"covariance"`), `annot` (bool), `cmap` |
| `mark_intercluster_distance(...)` | MDS or t-SNE projection of cluster centers, with each center sized proportionally to its membership count. Completes the clustering diagnostic suite alongside `mark_silhouette`. | `method` (`"mds"`\|`"tsne"`), `min_size`, `max_size`, `label_clusters` (bool) |
| `mark_cv_scores(...)` | Box, bar, or strip plot of cross-validation score distributions per fold. | `kind` (`"box"`\|`"bar"`\|`"strip"`), `split` (`"test"`\|`"train"`\|`"both"`) |
| `mark_alpha_selection(...)` | CV score vs regularization parameter curve with CI band. Intended for Ridge, Lasso, ElasticNet. Distinct from `mark_validation_curve` in that it assumes a log-spaced alpha domain by default. | `log_scale` (bool, default `True`), `ci_style` (`"band"`\|`"errorbar"`), `highlight_best` (bool) |

> **2026-05-11 (Phase 10 — Model Diagnostics):** Per-mark clarifications surfaced during
> the 10a–10g implementation:
> - `mark_residuals(kind="studentized")` uses the leverage-aware hat-matrix definition
>   when `X` is supplied to the underlying `studentized_residual` helper (linear
>   estimators only). For non-linear estimators ferrum falls back to internally
>   studentized residuals (divide by `std(r, ddof=1)`).
> - `mark_confusion`: color scale on `value` uses Phase 8b's continuous color scale
>   (`viridis` default). Per-cell `value_fmt` (Utf8) is pre-computed by
>   `ModelSource.confusion_matrix` so the renderer's text layer lays out short labels
>   without invoking number formatting per cell.
> - `mark_decision_boundary`: requires exactly 2 features; the figure-level
>   `decision_boundary_chart` fixes the unused features at their column means.
> - `mark_shap_*`: require the `shap` library, installable via `ferrum[shap]`. The
>   `shap_waterfall` mark requires an explicit `sample_idx: int` kwarg.
> - `mark_rank2d(algorithm="kendall")` is the sole Phase 10 path that crosses into
>   Rust at compute time — it calls `ferrum._core.kendall_tau_b` (Knight's
>   O(n log n) merge-sort variant). All other rank2d algorithms run in NumPy.
> - `mark_pca_scree(cumulative_line=True)` overlays a `mark_line` on the
>   `cumulative_variance_ratio` column. Layer-0 is the cumulative line so its
>   wider y range drives axis-scale resolution; the rect-bar layer follows.
> - `mark_intercluster_distance` uses the `size` channel for cluster cardinality;
>   the chart builder pads the x/y domain by 15% so large bubbles don't clip.
> - `mark_parallel_coordinates`: routes `sample_id` through `mark_style.detail` so
>   each sample renders as its own polyline. Requires the Phase 10g `mark_line`
>   composite (color, detail) grouping path and ordinal-x support.

---

### 3.4 Stat Transforms

> **2026-05-10 (Phase 9):** Adds eight new transforms — `Unpivot` (wide →
> long reshape; homogeneous-or-numeric value dtype), `Linkage` (hierarchical
> clustering with three named secondary outputs: `linkage`, `order`,
> `coords` / segment table for dendrogram rendering), `Reorder`
> (permutation by index column produced by `Linkage`), `Bin2D` (2D
> rectangular binning over `(x, y)`), `Logistic` (binary logistic
> regression via IRLS plus Wald CI), `Glm` (5 families × 7 links — see
> compatibility table below), `Robust` (Huber M-estimator + sandwich CI),
> `LetterValue` (boxen-plot statistics; outliers as named secondary
> output). Extends existing transforms: `Bin` gains a `cumulative: bool`
> parameter for ECDF support; `Smooth` gains `x_bins`, `x_estimator`, and
> `output: "fitted" | "residuals"`; `Robust` accepts the same `output`
> parameter for residual diagnostics.
>
> **GLM Family / Link Compatibility (Phase 9)**
>
> | Family | Canonical link | Other valid |
> |---|---|---|
> | Gaussian | Identity | Log, Inverse |
> | Binomial | Logit | Probit, Log |
> | Poisson | Log | Identity, Sqrt |
> | Gamma | Inverse | Identity, Log |
> | InverseGaussian | InverseSquared | Identity, Log |

Stat transforms are applied before rendering, in the Rust engine. They receive a column-oriented Arrow table and return a transformed table. Applied via `.stat()` at the chart or layer level, or implicitly by statistical marks.

| Class | Description | Key Parameters |
|---|---|---|
| `stat_bin(field, *, bin_count=None, bin_width=None, extent=None, nice=True)` | Bin a field; add `bin_start`, `bin_end`, `count`, `density` | |
| `stat_bin_2d(x, y, *, bin_count_x=None, bin_count_y=None)` | 2D binning | |
| `stat_kde(field, *, bandwidth="scott", kernel="gaussian", n=512, extent=None, cumulative=False)` | Kernel density estimate | |
| `stat_kde_2d(x, y, *, bandwidth="scott", n=128)` | 2D KDE | |
| `stat_smooth(x, y, *, method="loess", ci=0.95, bandwidth=0.75, degree=2, n=200)` | Smoothing line + CI | |
| `stat_summary(field, *, fun="mean", error_fn="ci", ci=0.95, n_boot=1000)` | Aggregate with uncertainty | |
| `stat_aggregate(field, fn, *, groupby=None)` | Arbitrary aggregation | |
| `stat_ecdf(field, *, complementary=False)` | Empirical CDF | |
| `stat_qq(field, *, distribution="normal", line=True)` | Quantile–quantile data | |
| `stat_contour(x, y, *, bandwidth="scott", thresholds=6, smooth=True)` | 2D contour levels | |
| `stat_identity()` | No-op; explicit passthrough | |
| `stat_ellipse(x, y, *, type="norm", level=0.95, segments=51)` | Compute a confidence or data ellipse for bivariate scatter. Returns `x`, `y` coordinates of the ellipse boundary. | `type`: `"norm"` (parametric normal ellipse) \| `"t"` (t-distribution) \| `"euclid"` (Euclidean distance from centroid). `level`: confidence level (ignored for `"euclid"`). |
| `stat_function(fn, *, domain=None, n=200, as_=("x", "y"))` | Evaluate a Python callable over a 1D domain. `fn` receives a numpy array and must return a numpy array. `domain`: `(min, max)` tuple; inferred from X scale if `None`. Returns columns named by `as_`. | |
| `stat_hex(x, y, *, bin_size=None, bins=None, aggregate="count", field=None)` | Hexagonal binning. Returns `hex_x`, `hex_y`, `value`. Triggered implicitly by `mark_hex`. | |

> *(2026-05-10) Phase 8b: `stat_kde_2d` is implemented as the `Kde2D` transform (10th transform of the phase). Output is a single-row Arrow batch with `grid_x` / `grid_y` / `density` list columns.*

**Model stat transforms** (accept `ModelSource` or raw arrays)

| Class | Description |
|---|---|
| `stat_roc(y_true, y_score, *, average=None)` | Compute ROC curve data |
| `stat_pr(y_true, y_score, *, average=None)` | Compute PR curve data |
| `stat_confusion(y_true, y_pred, *, normalize=None, labels=None)` | Compute confusion matrix |
| `stat_calibration(y_true, y_prob, *, n_bins=10, strategy="uniform")` | Compute calibration curve |
| `stat_lift(y_true, y_score)` | Compute lift/gain curve |
| `stat_importance(model, X, y, *, method="builtin", n_repeats=30)` | Compute feature importance |
| `stat_shap(model, X, *, background=None)` | Compute SHAP values |
| `stat_pdp(model, X, features, *, grid_resolution=100, kind="average")` | Compute partial dependence |
| `stat_residuals(model, X, y, *, kind="studentized")` | Compute residual diagnostics |
| `stat_learning_curve(model, X, y, *, cv=5, scoring=None, train_sizes=None)` | Compute learning curve |
| `stat_validation_curve(model, X, y, param, values, *, cv=5, scoring=None)` | Compute validation curve |

---

### 3.5 Data Transforms

Data transforms are applied before stat transforms, in the Rust engine. They reshape or filter the input data.

| Class | Description | Key Parameters |
|---|---|---|
| `transform_filter(predicate)` | Row filter | Predicate string (Vega expression), dict, or `Selection` |
| `transform_calculate(as_, expr)` | Derived column | Expression string or callable |
| `transform_aggregate(*aggregates, groupby=None)` | Group-by aggregation | `Aggregate` objects |
| `transform_bin(field, *, as_=None, **bin_kwargs)` | Bin a field | |
| `transform_fold(fields, *, as_=("key", "value"))` | Wide → long (melt) | |
| `transform_pivot(field, value, *, groupby=None, limit=None, op="sum")` | Long → wide | |
| `transform_join_aggregate(*aggregates, groupby=None)` | Add aggregate columns without collapsing rows | |
| `transform_window(*window_transforms, *, sort=None, groupby=None, frame=None, ignore_peers=False)` | Window functions (rolling mean, rank, cumsum, etc.) | |
| `transform_density(field, *, bandwidth="scott", groupby=None, extent=None, minsteps=None, steps=None, cumulative=False, as_=("value", "density"))` | KDE transform | |
| `transform_regression(x, y, *, method="linear", order=3, groupby=None, extent=None, params=False, as_=("x","y"))` | Regression transform | |
| `transform_loess(x, y, *, bandwidth=0.3, groupby=None, as_=("x","y"))` | LOESS transform | |
| `transform_impute(field, *, method="value", value=None, groupby=None, key=None, keyvals=None)` | Fill missing values | |
| `transform_flatten(fields, *, as_=None)` | Expand list-typed columns | |
| `transform_sample(n)` | Random row sample | |
| `transform_top_k(n, *, field, op="mean", sort="descending")` | Keep top-k by aggregate | |
| `transform_stack(field, *, groupby, sort=None, as_=("start","end"), offset="zero")` | Stacking transform for area/bar | |
| `transform_timeunit(field, unit, *, utc=False, as_=None)` | Temporal unit extraction | |

---

### 3.6 Scales

Scales map data domain values to visual range values. Attached to encoding channels via the `scale=` parameter.

#### Scale Classes

| Class | Description |
|---|---|
| `Scale(type=None, domain=None, range=None, nice=None, zero=None, padding=None, clamp=None, reverse=None, round=None)` | Base / linear scale |
| `ScaleLog(base=10, *, domain=None, ...)` | Logarithmic |
| `ScalePow(exponent=2, *, domain=None, ...)` | Power |
| `ScaleSqrt(...)` | Square root (power with exponent=0.5) |
| `ScaleSymlog(constant=1, ...)` | Symmetric log (handles zero) |
| `ScaleTime(...)` | Temporal |
| `ScaleUtc(...)` | UTC temporal |
| `ScaleOrdinal(domain=None, range=None)` | Categorical |
| `ScalePoint(domain=None, *, padding=0.5, align=0.5, reverse=False)` | Ordinal point scale (for dot plots) |
| `ScaleBand(domain=None, *, padding=0.1, padding_inner=None, padding_outer=None, align=0.5)` | Ordinal band scale (for bar plots) |
| `ScaleSequential(scheme=None, *, domain=None, reverse=False, interpolate=None)` | Sequential color |
| `ScaleDiverging(scheme=None, *, domain=None, domainMid=None)` | Diverging color |
| `ScaleThreshold(domain, range)` | Threshold / bin color |
| `ScaleQuantile(domain, range, *, quantiles=None)` | Quantile |
| `ScaleQuantize(domain, range)` | Quantize |
| `ScaleBinOrdinal(bins, scheme=None)` | Binned ordinal |

#### Color Scheme Constants (`ferrum.schemes`)

**Categorical:** `okabe_ito` *(default)*, `tableau10`, `set1`, `set2`, `paired`, `pastel`, `dark2`

**Sequential:** `viridis` *(default)*, `plasma`, `inferno`, `magma`, `cividis`, `blues`, `greens`, `oranges`, `reds`, `purples`, `greys`, `tealblues`

**Diverging:** `redblue`, `redgrey`, `pinkgreen`, `purplegreen`, `brownbluegreen`, `spectral`

**Cyclical:** `rainbow`, `sinebow`

> **Implementation note (2026-05-11, themes-T3):** The 7 categorical schemes
> ship as a const palette registry in `crates/ferrum-core/src/render/palette.rs`
> (`CATEGORICAL_SCHEMES`, `SEQUENTIAL_SCHEMES`, `is_categorical_scheme`,
> `is_sequential_scheme`). `Theme(color_scheme=...)` validates the name
> eagerly at render entry — unknown values raise `ValueError`. Categorical
> color resolution precedence is now `encoding.scheme` (per-encoding override,
> e.g. `mark_heatmap`'s `cmap=`) → `theme.color_scheme` (Theme default) →
> `OKABE_ITO` fallback. Sequential scheme names (`viridis` etc.) used on a
> nominal encoding substitute `tableau10` rather than collapsing silently to
> the categorical default. Gridlines now render on every quantitative axis
> as `<line>` elements drawn behind axis lines and marks, honoring
> `theme.grid` / `grid_color` / `grid_width` / `grid_dash` / `grid_opacity`.
> Cyclical, `tealblues`, and the brewer-extended sequential names beyond
> `viridis/plasma/magma/inferno/cividis` are reserved spec surface — not yet
> implemented and currently rejected by theme-level validation. T4 will flip
> the default to `tableau10`.

> **Implementation note (2026-05-11, themes-T4):** `Scale.padding` (listed in
> the base Scale class above) is now plumbed end-to-end and defaults to
> `0.05` for quantitative / temporal scales when unset. The visual mapping
> reserves 5% of the plot dimension on each side (capped at 8 px) so marks
> do not touch axis lines or the plot edge. Precedence at the renderer:
> `Scale(padding=p)` honors `p` (including `0.0` to disable); a user-supplied
> `Scale(domain=[...])` with no explicit padding suppresses the default to
> `0.0` (the user-explicit domain is treated as authoritative); otherwise
> `0.05` applies. Categorical / ordinal scales (`ScaleBand`, `ScalePoint`,
> `OrdinalScale`) are unaffected — they keep their own internal half-step
> band padding. Pixel ranges supplied via `Scale(range=[...])` bypass the
> inset entirely and are treated as the final scale range. The 4 quantitative
> PyO3 scale classes (`LinearScale`, `LogScale`, `TimeScale`, `SymlogScale`)
> accept a new keyword `padding` and expose a `.padding` getter; the
> `Ordinal` variant's existing `padding` field is unchanged. The default
> `ThemeInputs::default()` color scheme flips from `okabe_ito` to
> `tableau10` (loud divergence from the §3.6 *"`okabe_ito` (default)"*
> annotation above) — see the §3.13 Themes-T2 → T4 note.

> **Design decision (2026-05-31, flexibility-campaign D1):** Categorical /
> ordinal scale `range` accepts **color strings** (hex `#rrggbb`/`#rgb` or CSS
> named colors), not only numeric pixel positions. Paired with `domain`, this
> expresses explicit category→color mapping — the editorial "gray everything,
> accent two" and financial "green-up/red-down" idioms
> (`ScaleOrdinal(domain=["A","B","C"], range=["#ccc","#ccc","#e4572e"])`),
> matching Altair's `Scale(domain=..., range=[...])`. `ScaleQuantize` string
> ranges are the existing precedent. No new value-class is introduced.

> **2026-05-31 (F2, internal wire-format typing):** The ordinal `scale.range`
> wire representation is now a typed array of `Number | String` entries
> internally (Rust `Vec<OrdinalRangeValue>`, serialized `untagged`), replacing
> the prior untyped `serde_json::Value` that was re-discriminated by JSON
> sniffing at three call sites. The **user-facing contract is unchanged**:
> Python still constructs `Scale(range=[...])` / `OrdinalScale(range=[...])`
> with a plain `list[float | str]`, and the emitted JSON is still a flat array
> (e.g. `[0, 300]` or `["#ccc", "#e4572e"]`). Only the Rust-side type and the
> single typed accessor that replaced the sniffers changed; positional and
> color scale resolution are byte-identical.

> **Design decision (2026-05-31, flexibility-campaign D2):** Layering with `+`
> resolves color scales by **unioning their domains** across layers, so the
> result is order-independent (`base + highlight == highlight + base`). On a
> genuine scheme conflict the first encoding-bearing layer wins. Axis titles
> are taken from the first **data-bearing** layer; annotation-only layers
> (e.g. `annotate_rect`) never supply axis titles or rename axes to
> `_x1`/`_y1`. Mirrors Altair's shared-scale resolve default.

---

### 3.7 Axes and Legends

#### `Axis`

```
Axis(title=None, *, orient=None, ticks=True, tick_count=None, tick_extra=False, tick_min_step=None,
     grid=True, grid_dash=None, grid_width=None, grid_color=None, grid_opacity=None,
     labels=True, label_angle=None, label_flush=True, label_overlap="parity",
     label_format=None, label_format_type=None, label_font_size=None, label_color=None,
     domain=True, domain_width=None, domain_color=None,
     offset=None, translate=None, min_extent=None, max_extent=None,
     title_orient=None, title_font_size=None, title_color=None, title_padding=None,
     values=None, encode=None, zindex=None)
```

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

Set `legend=None` on any channel to suppress the legend for that channel.

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
Resolve(scale=None, axis=None, legend=None)
```

`scale` / `axis` / `legend` each accept a dict mapping channel name (`"x"`, `"y"`, `"color"`, etc.) to `"shared"` or `"independent"`.

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

```
Color(field="category", condition=selection.when(Color("category")).otherwise(value("#aaa")))
```

Or via chart-level `.conditional(sel, color=..., else_color=...)`.

> **Note (2026-06-01):** A conditional built by `sel.when(...).otherwise(...)`
> or `fm.when(sel).then(...).otherwise(...)` may be passed directly to
> `encode(<channel>=cond)`. The wire channel is taken from the encode key
> (so `encode(opacity=fm.when(sel).then(1.0).otherwise(0.2))` resolves the
> numeric branches as opacity, not colour), and the conditional's source
> selection is auto-registered — no separate `.add_selection()` is required.
> `Chart.conditional(spec)` likewise auto-registers the spec's source selection
> when it carries one. A bare number with no channel context (e.g. via
> `fm.value(0.5)` outside an `encode` key) defaults to the `opacity` channel.

---

### 3.11 Annotations

Annotations are lightweight overlays that don't participate in scale domain calculation.

| Class | Description | Key Parameters |
|---|---|---|
| `annotate_hline(y, *, label=None, label_position="right", stroke=None, stroke_dash=None)` | Horizontal reference line | |
| `annotate_vline(x, *, label=None, stroke=None, stroke_dash=None)` | Vertical reference line | |
| `annotate_rect(x1, x2, y1, y2, *, fill=None, opacity=0.1, label=None)` | Shaded rectangle region | |
| `annotate_text(x, y, text, *, dx=0, dy=0, align="center", baseline="middle", font_size=None, color=None, angle=None)` | Free text annotation | |
| `annotate_arrow(x1, y1, x2, y2, *, label=None, label_side="start", stroke=None)` | Arrow with optional label | |
| `AUCLabel(*, position="end", format=".3f", prefix="AUC = ")` | Auto-placed AUC annotation on ROC curves | |
| `OutlierLabel(*, threshold=3.0, field=None, label_field=None, max_labels=10)` | Label high-leverage or high-residual points | |

---

### 3.12 Compound Views

> **2026-05-10 (Phase 9):** `JointChart` (sketched in 8b) is implemented as
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

All compound views accept `.theme()`, `.properties()`, `.save()`, `.show()`.

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

Figure-level functions return `Chart` or compound view objects. They handle data reshaping, faceting, axis labeling, and legend placement automatically. All accept `theme=` and `**encode_kwargs` to override defaults.

#### Distribution

```
ferrum.displot(data, *, x=None, y=None, hue=None, col=None, row=None,
               kind="hist",          # "hist" | "kde" | "ecdf" | "rug"
               fill=True, cumulative=False, log_scale=False, stat="count",
               bins="sturges", bandwidth="scott", bw_adjust=1.0,
               multiple="layer",     # "layer" | "stack" | "fill" | "dodge"
               kde=False, rug=False, height=None, aspect=None, theme=None)
```

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
              method="lm",           # "lm" | "logistic" | "glm" | "loess" | "robust"
              ci=95, order=1, scatter=True, scatter_kws=None, line_kws=None,
              truncate=False, x_bins=None, x_estimator=None, x_jitter=None,
              logx=False, theme=None)

ferrum.regplot(data, *, x, y, hue=None,
               method="lm",          # same as lmplot
               ci=95, order=1, scatter=True, scatter_kws=None, line_kws=None,
               truncate=True, x_jitter=None, theme=None)

ferrum.residplot(data, *, x, y, lowess=False, order=1, robust=False,
                 dropna=True, label=None, color=None, theme=None)
```

`regplot` is the axes-level equivalent of `lmplot` — identical API minus
`col=` and `row=` (no faceting). All parameters are forwarded to `lmplot`.

#### Matrix / Pairwise

```
ferrum.pairplot(data, *, vars=None, x_vars=None, y_vars=None,
                hue=None, kind="scatter",     # off-diagonal
                diag_kind="auto",             # "hist" | "kde" | None
                markers=None, height=None, aspect=None,
                corner=False, dropna=False, theme=None)

ferrum.heatmap(data, *, annot=True, fmt=".2f", cmap="blues",
               linewidths=0.5, linecolor="white",
               vmin=None, vmax=None, center=None, robust=False,
               square=False, mask=None, theme=None)

ferrum.clustermap(data, *, method="ward", metric="euclidean",
                  cmap="viridis", z_score=None, standard_scale=None,
                  figsize=None, dendrogram_ratio=0.2, theme=None)
```

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

ferrum.confusion_matrix_chart(model_or_source=None, X=None, y=None, *,
                               y_true=None, y_pred=None,   # precomputed path (1-D hard labels)
                               normalize="true", cmap="blues", theme=None)

ferrum.calibration_chart(model_or_source=None, X=None, y=None, *,
                          y_true=None, y_pred=None,  # precomputed path (1-D positive-class proba)
                          n_bins=10, theme=None)

ferrum.gain_chart(model_or_source=None, X=None, y=None, *,
                  y_true=None, y_pred=None,          # precomputed path (soft scores)
                  compare=None, theme=None)

ferrum.lift_chart(model_or_source=None, X=None, y=None, *,
                  y_true=None, y_pred=None,          # precomputed path (soft scores)
                  compare=None, theme=None)

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

> **2026-05-15 — Precomputed input path.** The nine prediction-evaluation
> figure functions that appear above — `roc_chart`, `pr_chart`,
> `calibration_chart`, `gain_chart`, `lift_chart`,
> `discrimination_threshold_chart`, `confusion_matrix_chart`,
> `class_prediction_error_chart`, and `residuals_chart` — each accept
> `y_true=` and `y_pred=` as keyword-only arguments, bypassing the fitted
> model entirely. Callers supply arrays they have already computed; the
> function returns a chart visually identical to the model-backed path.
> Exactly one of {model path, precomputed path} must be active — supplying
> both, neither, or only one of `y_true`/`y_pred` raises `ValueError`.
> `compare=` is incompatible with the precomputed path. `y_pred` semantics
> vary by function: soft scores/probabilities for curve functions, hard
> labels for matrix functions, fitted values for `residuals_chart`. See each
> function's docstring for the exact expected shape.

> **Schwabish SB3 — 2026-05-11 — figure-function defaults.** Eight
> Group-B functions adopt the Schwabish defaults from §3.19's principles doc:
>
> - `roc_chart`: `annotate_auc=True` by default; single-curve charts get the
>   active title `"ROC — AUC X.XXX"`; multi-curve charts get a descriptive
>   `"ROC"` title plus one `AUCLabel` per curve, staggered along the y axis
>   so labels do not collide at the line endpoint.
> - `pr_chart`: `annotate_ap=True` by default; active title parallels
>   `roc_chart`. Binary classifiers additionally render a dashed horizontal
>   baseline at the positive-class prevalence so the chance-level floor is
>   always visible.
> - `calibration_chart`: new `annotate_brier=True` kwarg (default on).
>   Single-model charts get an active title carrying the per-sample Brier
>   score computed from `model.predict_proba` plus a corner `BrierLabel`;
>   multi-model charts get one `BrierLabel` per model from the bin-level
>   reliability data.
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
    def score(self, X, y) -> float        # computes metric; populates chart state
    def show(self) -> Chart
    def __repr__(self) -> str             # summary string with key metrics
```

**Concrete visualizers:**

| Class | Wraps |
|---|---|
| `ROCVisualizer(model, *, micro=True, macro=True, per_class=True, theme=None)` | ROC chart |
| `PRVisualizer(model, *, theme=None)` | PR chart |
| `ConfusionMatrixVisualizer(model, *, normalize="true", theme=None)` | Confusion matrix |
| `ClassificationReportVisualizer(model, *, theme=None)` | Heatmap of precision/recall/F1 per class |
| `CalibrationVisualizer(*models, *, n_bins=10, theme=None)` | Calibration chart |
| `ResidualsVisualizer(model, *, kind="studentized", theme=None)` | Residuals diagnostic |
| `PredictionErrorVisualizer(model, *, identity_line=True, theme=None)` | Prediction error |
| `FeatureImportancesVisualizer(model, *, method="builtin", top_k=20, theme=None)` | Importance chart |
| `LearningCurveVisualizer(model, *, cv=5, scoring=None, train_sizes=None, theme=None)` | Learning curve |
| `ValidationCurveVisualizer(model, param, values, *, cv=5, scoring=None, theme=None)` | Validation curve |
| `SilhouetteVisualizer(model, *, theme=None)` | Silhouette chart (clusterers) |
| `ElbowVisualizer(model_class, *, ks, metric="distortion", theme=None)` | Elbow chart |
| `ManifoldVisualizer(model, *, method="umap", theme=None)` | Embedding projection |
| `ClassBalanceVisualizer(*, theme=None)` | Class frequency bar chart |
| `CooksDistanceVisualizer(model, *, threshold=None, theme=None)` | Cook's distance |
| `SHAPVisualizer(model, *, kind="beeswarm", background=None, theme=None)` | SHAP chart — **deprecated** since 2026-05-12 (P3.6); use the three sibling classes below. |
| `SHAPBeeswarmVisualizer(model, *, max_display=20, order="abs_mean", background=None, theme=None)` | SHAP beeswarm |
| `SHAPBarVisualizer(model, *, max_display=20, order="abs_mean", background=None, theme=None)` | SHAP mean-absolute bar |
| `SHAPWaterfallVisualizer(model, *, sample_idx, max_display=20, order="abs_mean", background=None, theme=None)` | SHAP single-sample waterfall |
| `DiscriminationThresholdVisualizer(model, *, n_thresholds=50, scoring=None, cv=None, theme=None)` | Discrimination threshold chart (binary classifiers only) |
| `ParallelCoordinatesVisualizer(*, features=None, hue=None, rescale="minmax", theme=None)` | Parallel coordinates. No model required; `fit(X, y)` accepts raw feature matrix. `hue` applied from `y` if provided. |
| `ClassPredictionErrorVisualizer(model, *, normalize=False, theme=None)` | Class prediction error chart |
| `PCAVarianceVisualizer(model, *, n_components=None, theme=None)` | PCA scree plot. Model must expose `explained_variance_ratio_` attribute (sklearn PCA, TruncatedSVD, or compatible). |
| `Rank1DVisualizer(*, algorithm="shapiro", top_k=None, theme=None)` | Univariate feature ranking bar chart. No model required. |
| `Rank2DVisualizer(*, algorithm="pearson", theme=None)` | Pairwise feature correlation heatmap. No model required. |
| `AlphaSelectionVisualizer(model, alphas, *, cv=5, scoring=None, theme=None)` | Regularization parameter selection curve |
| `InterclusterDistanceVisualizer(model, *, method="mds", theme=None)` | Intercluster distance map |
| `CVScoresVisualizer(model, *, cv=5, scoring=None, kind="box", theme=None)` | Cross-validation score distribution |

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
> it. The `ElbowVisualizer(model_class, *, ks, ...)` signature is unique —
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
    raster_aggregate="count",   # default aggregate for auto-raster substitution
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
chart.to_spec() -> ChartSpec                  # internal dataclass
chart.to_json(*, indent=None) -> str
chart.save(path, *, format=None, **render_kwargs)
chart.pipe(fn, *args, **kwargs) -> Any        # apply a function to self
```

These methods exist on `Chart` and on every composition view (`HConcatChart`,
`VConcatChart`, `LayerChart`, `ConcatChart`, `JointChart`, `RepeatChart`,
`ClusterMapChart`). On composition views `to_png` is scale-only (it accepts
`scale=` but no `raster` argument).

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
