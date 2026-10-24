# Marks & encodings

Two primitives carry every Ferrum chart: **marks** (the geometric shapes that visualize your data) and **encodings** (the typed mappings from data fields to visual variables). Picking the right mark + encoding combination is most of what authoring a chart looks like.

This page is the reference for both. It covers the encoding channels, the mark families, the shorthand syntax that compresses common cases, and when to reach for what.

## How a chart is assembled

A Ferrum chart is built by attaching a mark to a data source and declaring which columns drive which visual variables. Every chart follows the same shape:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
chart = (
    fm.Chart(iris)
    .mark_point()
    .encode(x="sepal_length", y="petal_length", color="species:N")
)
chart
```

![Basic scatter](img/marks-encodings_basic_scatter.png)

The three pieces — data, mark, encoding — compose freely. You can change the mark without touching the encoding ([`mark_line()`][ferrum.Chart.mark_line] instead of [`mark_point()`][ferrum.Chart.mark_point]), change the encoding without touching the mark, or compose multiple marks against the same encoding (see [Composition](composition.md)).

## Encoding channels

An encoding channel declares: *this field drives this visual variable*. Channels are typed by the engine: a quantitative field gets a continuous scale, a nominal field gets a categorical color palette, a temporal field gets a time scale. You can be explicit by passing an encoding object ([`fm.X("col", type="Q")`][ferrum.X]) or use the shorthand syntax (described below).

### Positional channels

These channels place marks in space:

| Channel | Purpose |
|---|---|
| `x`, `y` | Primary horizontal / vertical position. |
| `x2`, `y2` | Secondary position. Used for bands, segments, intervals, error extents. |
| `xerror`, `yerror`, `xerror2`, `yerror2` | Error extents around the primary position. |
| `theta`, `radius` | Polar coordinates. Used with [`CoordPolar`][ferrum.CoordPolar]. |

Most charts only declare `x` and `y`. The rest unlock band marks (`mark_area`, `mark_errorband`), intervals (`mark_rect`, `mark_rule`), and polar plots.

### Appearance channels

These channels modulate how marks look:

| Channel | Purpose |
|---|---|
| `color` | Mark color. Continuous fields get a perceptually uniform palette; categorical fields get a discrete palette. |
| `fill`, `stroke` | Override color separately for the fill and stroke. `color` sets both. |
| `opacity`, `fill_opacity`, `stroke_opacity` | Mark opacity. |
| `stroke_width`, `stroke_dash` | Stroke styling. |
| `size` | Mark size. |
| `shape` | Mark glyph (for `mark_point`). |
| `angle` | Rotation. |

Appearance channels can take either a field name (data-driven) or a literal value (constant for all marks). Setting `color="red"` colors every mark red; setting `color="species:N"` colors marks by the `species` column.

### Text and metadata channels

These channels carry information that does not directly map to position or appearance:

| Channel | Purpose |
|---|---|
| `text` | Text content for `mark_text`. |
| `detail` | Additional grouping that does not affect appearance — useful for keeping series separate without coloring them differently. |
| `tooltip`, `tooltip_field` | Field shown on hover. In interactive mode, renders as a tooltip overlay; in static output, becomes accessibility metadata. |
| `href` | URL the mark links to. |
| `description` | Accessibility description. |
| `key` | Stable identity for interactive selections. |

### Faceting channels

These channels split the chart into small multiples:

| Channel | Purpose |
|---|---|
| `facet` | Single faceting variable, wrapped into a grid. |
| `facet_row`, `facet_col` | Row / column facets for a 2-D small-multiples grid. |

Faceting is structural: it produces multiple panels rather than overlaying marks. To layer marks against the same axes, use [Composition](composition.md).

## The shorthand string syntax

Encodings accept a compact string syntax that handles the most common cases without explicit channel objects:

| Shorthand | Meaning |
|---|---|
| `"field"` | Field with inferred type (engine picks Q / N / O / T based on dtype). |
| `"field:Q"` | Explicitly quantitative. |
| `"field:N"` | Nominal (unordered categorical). |
| `"field:O"` | Ordinal (ordered categorical). |
| `"field:T"` | Temporal. |
| `"agg(field):Q"` | Aggregation. Examples: `"mean(price):Q"`, `"count():Q"`, `"sum(qty):Q"`, `"median(value):Q"`. |

The shorthand is purely syntactic sugar over the explicit form. `fm.X("price", type="Q")` and `"price:Q"` produce identical specs. The shorthand keeps simple cases compact; the explicit form unlocks advanced channel options.

When in doubt, use the explicit form:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
chart = (
    fm.Chart(iris)
    .mark_point()
    .encode(
        x=fm.X("sepal_length", type="Q", title="Sepal length"),
        y=fm.Y("petal_length", type="Q", title="Petal length"),
        color=fm.Color("species", type="N", title="Species"),
    )
)
chart
```

![Explicit encoding](img/marks-encodings_explicit_encoding.png)

## Position adjustments

Position adjustments control how marks that share the same x-position are arranged. Pass a position object to any mark's `position=` parameter.

### [`Dodge`][ferrum.Dodge] — side-by-side (grouped bars)

Spreads marks into non-overlapping groups. The `by` parameter selects the grouping channel (defaults to `color`/`fill`).

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "category": ["A", "A", "B", "B"],
    "group": ["x", "y", "x", "y"],
    "value": [10.0, 15.0, 8.0, 12.0],
})
chart = (
    fm.Chart(df)
    .mark_bar(position=fm.Dodge(padding=0.05))
    .encode(x="category:N", y="value:Q", color="group:N")
)
chart
```

![Dodge grouped bars](img/marks-encodings_dodge.png)

### [`Stack`][ferrum.Stack] — stacked bars/areas

Accumulates marks vertically. The `offset` parameter controls the stacking strategy:

- `"zero"` (default) — standard cumulative stack from y = 0.
- `"normalize"` — 100% stack; each x-bin scales to a total of 1.
- `"center"` — streamgraph; symmetric around y = 0.

```python
chart = (
    fm.Chart(df)
    .mark_bar(position=fm.Stack(offset="normalize"))
    .encode(x="category:N", y="value:Q", color="group:N")
)
chart
```

![Stack stacked bars](img/marks-encodings_stack.png)

### [`Jitter`][ferrum.Jitter] — random displacement for overplotting

Adds controlled noise to one or both axes. Output is deterministic for a given dataset and seed.

```python
chart = (
    fm.Chart(df)
    .mark_point(position=fm.Jitter(axis="x", width=0.3, seed=42))
    .encode(x="category:N", y="value:Q")
)
```

## Axis customization

### Scale types

Positional channels accept an explicit scale via the `scale=` parameter. The primary scale classes are:

| Scale | Usage |
|---|---|
| [`LinearScale`][ferrum.LinearScale] | Default continuous scale. |
| [`LogScale`][ferrum.LogScale] | Logarithmic (base-10 by default; configurable via `base=`). |
| [`SymlogScale`][ferrum.SymlogScale] | Symmetric log — handles zero and negatives. |
| [`TimeScale`][ferrum.TimeScale] | Temporal axis. |
| [`OrdinalScale`][ferrum.OrdinalScale] | Discrete/categorical. |

See [Scale Types](../api/scales.md) for the full scale reference including `PowScale`, `BandScale`, `PointScale`, and color scales.

```python
import ferrum as fm
import polars as pl
import numpy as np

rng = np.random.default_rng(42)
df_log = pl.DataFrame({"income": rng.uniform(100, 100000, 80), "score": rng.uniform(20, 90, 80)})
chart = (
    fm.Chart(df_log)
    .mark_point(size=40)
    .encode(
        x=fm.X("income", scale=fm.LogScale(domain=[100, 100000], base=10)),
        y="score:Q",
    )
)
chart
```

![Log scale axis](img/marks-encodings_log_scale.png)

### Axis limits (domain)

Set explicit axis limits by passing a `domain=` list to the scale constructor:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"])
chart = (
    fm.Chart(iris)
    .mark_point(size=40)
    .encode(
        x=fm.X("sepal_length", scale=fm.LinearScale(domain=[4, 8])),
        y="petal_length:Q",
    )
)
chart
```

![Axis limits](img/marks-encodings_axis_limits.png)

### Reversed axis

Pass `reverse=True` to any continuous positional scale (`LinearScale`, `LogScale`, `PowScale`, `SqrtScale`, `SymlogScale`, `TimeScale`) to reverse its axis — marks and axis tick labels flip together:

```python
# Reversed y-axis (high values at bottom)
chart = (
    fm.Chart(df)
    .mark_point()
    .encode(
        x="x:Q",
        y=fm.Y("depth", scale=fm.LinearScale(domain=[0, 100], reverse=True)),
    )
)
```

Equivalently, with an explicit `domain=`, swapping the endpoints by hand produces the identical rendered chart:

```python
y=fm.Y("depth", scale=fm.LinearScale(domain=[100, 0]))
```

`reverse=True` is the clearer spelling when the domain is auto-inferred from data (no `domain=` to swap by hand) or when a `param`/data-driven domain makes writing `[hi, lo]` inconvenient. `PointScale.reverse` reverses its sorted category order instead of swapping a numeric domain pair — a different mechanism under the same name.

## Legend control

The `legend` parameter on appearance channels controls legend rendering.

### Suppressing the legend

Pass `legend=False` or `legend=None` to hide the legend for a channel:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
chart = (
    fm.Chart(iris)
    .mark_point()
    .encode(
        x="sepal_length:Q", y="petal_length:Q",
        color=fm.Color("species", legend=False),
    )
)
chart
```

![Legend suppressed](img/marks-encodings_legend_suppressed.png)

### Legend title

The legend title defaults to the field name. Override it with the `title=` parameter on the encoding channel:

```python
chart = (
    fm.Chart(iris)
    .mark_point()
    .encode(
        x="sepal_length:Q", y="petal_length:Q",
        color=fm.Color("species", title="Iris species"),
    )
)
```

Legend suppression works on [`Color`][ferrum.Color], [`Size`][ferrum.Size], and [`Shape`][ferrum.Shape]. Pass `legend=False` or `legend=None` on any of these channels to hide its legend.

See [Legend](../api/legend.md) for per-channel legend customization via the `Legend(...)` value class.

## Palette cycling

When the number of distinct categories exceeds the palette length, colors **cycle** — category `i` receives `palette[i % len(palette)]`. The same modular-index strategy applies to the shape palette. This means that with many categories, some groups will share a color or glyph. If your data has more groups than palette entries, consider switching to a larger palette via `scheme=` or reducing cardinality before plotting.

See [Color](../api/color.md) for programmatic palette access.

## Mark families

Ferrum ships 56 mark methods on [`Chart`][ferrum.Chart]. They group into families by what they're for.

### Primitive marks

The geometric building blocks. Use these when you want direct control over what gets drawn.

| Method | Geometry |
|---|---|
| [`mark_point()`][ferrum.Chart.mark_point] | Discrete points. The default scatter mark. |
| [`mark_line()`][ferrum.Chart.mark_line] | Polyline connecting points in order. |
| [`mark_area()`][ferrum.Chart.mark_area] | Filled area, optionally banded with `y2`. |
| [`mark_bar()`][ferrum.Chart.mark_bar] | Vertical or horizontal bars. |
| [`mark_rect()`][ferrum.Chart.mark_rect] | Rectangular cells. Used for heatmaps and intervals. |
| [`mark_rule()`][ferrum.Chart.mark_rule] | Reference lines (often horizontal or vertical). |
| [`mark_text()`][ferrum.Chart.mark_text] | Text labels (paired with the `text` encoding). |
| [`mark_label()`][ferrum.Chart.mark_label] | Positioned text labels with automatic collision avoidance. |
| [`mark_image()`][ferrum.Chart.mark_image] | Image tiles from URL fields. |
| [`mark_tick()`][ferrum.Chart.mark_tick] | Short ticks, often used for rug plots. |
| [`mark_segment()`][ferrum.Chart.mark_segment] | Arbitrary line segments from `(x, y)` to `(x2, y2)`. |

Example — basic scatter:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
chart = (
    fm.Chart(iris)
    .mark_point()
    .encode(
        x="sepal_length",
        y="petal_length",
        color="species:N",
        size="sepal_width",
    )
)
chart
```

![Scatter with size](img/marks-encodings_size.png)

### Statistical marks

These marks compute a transform on your data before rendering — KDE, binning, smoothing, contours, quantile-quantile reference, or arbitrary functions. The transform happens in Rust, declared in the chart spec.

| Method | Transform |
|---|---|
| [`mark_smooth()`][ferrum.Chart.mark_smooth] | LOESS or OLS regression overlay (with optional CI band). |
| [`mark_errorbar()`][ferrum.Chart.mark_errorbar] | Error bars with optional terminal ticks. |
| [`mark_errorband()`][ferrum.Chart.mark_errorband] | Filled band between `y` and `y2`. |
| [`mark_histogram()`][ferrum.Chart.mark_histogram] | Binned counts or densities. |
| [`mark_density()`][ferrum.Chart.mark_density] | 1-D kernel density estimate. |
| [`mark_contour()`][ferrum.Chart.mark_contour] | 2-D density contours. |
| [`mark_hex()`][ferrum.Chart.mark_hex] | Hexagonal binning for large datasets. `stroke`/`stroke_width` draw cell borders. |
| [`mark_raster()`][ferrum.Chart.mark_raster] | Pre-aggregated rectangular grid. |
| [`mark_qq()`][ferrum.Chart.mark_qq] | Quantile-quantile plot against a reference distribution. |
| [`mark_function()`][ferrum.Chart.mark_function] | Plot an arbitrary `f(x)` over a domain. |

Transform kwargs (`bandwidth`, `bin_count`, `ci`, …) control the computation; they are independent of the constant mark-style kwargs above. Statistical marks accept both, and the style applies to **every layer the mark emits** — for `mark_smooth(ci=...)` that means the regression line and the CI ribbon together:

```python
# Overlapping translucent KDEs: opacity is a mark-style kwarg, bandwidth is a transform kwarg
fm.Chart(df).mark_density(bandwidth="scott", opacity=0.4).encode(x="value:Q", color="group:N")
```

A few marks reserve a name for the transform: `mark_density(fill=False)` selects a line instead of a filled area, and `mark_hex` reads `cmap`/`stroke`/`stroke_width` as its own parameters — use `color=` for a constant hex fill.

#### `mark_smooth` methods

The smoothing method is selected with the `method=` kwarg. The computation runs in Rust.

| Method | Description |
|---|---|
| `"loess"` (default) | Locally-weighted polynomial regression. |
| `"lm"` | Ordinary least-squares linear fit. |

Key parameters:

- `ci` — confidence interval level (e.g. `0.95`). When set, emits a ribbon + line layered chart. Default `None` (no band).
- `bandwidth` — LOESS span fraction in `(0, 1]`. Default `0.75`. Ignored when `method="lm"`.
- `degree` — LOESS polynomial degree (`1` or `2`). Default `2`. Ignored when `method="lm"`.
- `n` — number of evaluation grid points. Default `200`.

!!! note
    `"logistic"` regression is available via the separate [`Logistic`][ferrum.Logistic] transform (used by [`lmplot`][ferrum.lmplot]), not through [`mark_smooth`][ferrum.Chart.mark_smooth].

Example — 1-D kernel density estimate:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
chart = (
    fm.Chart(iris)
    .mark_density(bandwidth="scott")
    .encode(x="sepal_length")
)
chart
```

![KDE density](img/marks-encodings_kde.png)

Stat marks are described in detail in [Stats in the rendering pipeline](concepts/stats-pipeline.md).

### Grouped transforms with `groupby`

Statistical marks compute their transform over the entire dataset by default. To compute independently per group — one LOESS line per species, one KDE per category — pass `groupby=`:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
chart = (
    fm.Chart(iris)
    .mark_smooth(method="loess", groupby="species")
    .encode(x="sepal_length", y="petal_length", color="species:N")
)
chart
```

![Grouped LOESS](img/marks-encodings_grouped_loess.png)

The `groupby` parameter is available on `mark_smooth`, `mark_density`, and `mark_histogram`. The group column is preserved in the transform output so downstream `color=` encoding maps each group to a distinct visual.

This is especially important when layering a statistical mark with a scatter via `+` — without `groupby`, the transform runs over all data combined and produces a single aggregate line.

### Distribution-summary marks

For comparing categorical distributions at a glance:

| Method | Geometry |
|---|---|
| [`mark_boxplot()`][ferrum.Chart.mark_boxplot] | Tukey boxplot — quartiles, whiskers, outliers. |
| [`mark_violin()`][ferrum.Chart.mark_violin] | Symmetric KDE per group. |
| [`mark_boxen()`][ferrum.Chart.mark_boxen] | Letter-value boxplot — more quantiles for larger samples. |
| [`mark_swarm()`][ferrum.Chart.mark_swarm] | Beeswarm jitter (categorical scatter without overlap). |

Example — boxplot by species:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
chart = (
    fm.Chart(iris)
    .mark_boxplot()
    .encode(x="species:N", y="sepal_length")
)
chart
```

![Boxplot by species](img/marks-encodings_boxplot.png)

### Composition marks

| Method | Geometry |
|---|---|
| [`mark_ribbon()`][ferrum.Chart.mark_ribbon] | Continuous band, typically paired with a line overlay. |

### Other marks

| Method | Geometry |
|---|---|
| [`mark_geoshape()`][ferrum.Chart.mark_geoshape] | Geographic polygons (GeoJSON-backed). |
| [`mark_arc()`][ferrum.Chart.mark_arc] | Arc/wedge segments for pie and donut charts (polar coordinates). |

### Model-diagnostic marks

These marks work with [`ModelSource`][ferrum.ModelSource] to produce evaluation plots. For most use cases, prefer the figure-level helpers ([`roc_chart`][ferrum.roc_chart], [`calibration_chart`][ferrum.calibration_chart], etc.) covered in [Figure-level helpers](figure-helpers.md). The marks exist for when you want grammar-level composition of custom diagnostic views.

**Classification:**

| Method | Purpose |
|---|---|
| [`mark_roc()`][ferrum.Chart.mark_roc] | ROC curve with AUC annotation. |
| [`mark_pr()`][ferrum.Chart.mark_pr] | Precision-recall curve. |
| [`mark_calibration()`][ferrum.Chart.mark_calibration] | Calibration (reliability) curve. |
| [`mark_confusion()`][ferrum.Chart.mark_confusion] | Confusion matrix heatmap. |
| [`mark_class_prediction_error()`][ferrum.Chart.mark_class_prediction_error] | Stacked prediction-error bars by class. |
| [`mark_discrimination_threshold()`][ferrum.Chart.mark_discrimination_threshold] | Metrics vs. decision threshold. |
| [`mark_gain()`][ferrum.Chart.mark_gain] | Cumulative gains chart. |
| [`mark_lift()`][ferrum.Chart.mark_lift] | Lift curve. |

**Regression:**

| Method | Purpose |
|---|---|
| [`mark_residuals()`][ferrum.Chart.mark_residuals] | Residuals vs. fitted values. |
| [`mark_prediction_error()`][ferrum.Chart.mark_prediction_error] | Predicted vs. actual scatter with identity line. |

**Explanation:**

| Method | Purpose |
|---|---|
| [`mark_importance()`][ferrum.Chart.mark_importance] | Feature importance bar chart. |
| [`mark_shap_beeswarm()`][ferrum.Chart.mark_shap_beeswarm] | SHAP beeswarm summary plot. |
| [`mark_shap_bar()`][ferrum.Chart.mark_shap_bar] | SHAP mean-absolute bar plot. |
| [`mark_shap_waterfall()`][ferrum.Chart.mark_shap_waterfall] | SHAP waterfall for a single prediction. |
| [`mark_pdp()`][ferrum.Chart.mark_pdp] | Partial dependence plot. |

**Model selection:**

| Method | Purpose |
|---|---|
| [`mark_learning_curve()`][ferrum.Chart.mark_learning_curve] | Train/test score vs. sample size. |
| [`mark_validation_curve()`][ferrum.Chart.mark_validation_curve] | Score vs. hyperparameter value. |
| [`mark_cv_scores()`][ferrum.Chart.mark_cv_scores] | Cross-validation score distribution. |
| [`mark_alpha_selection()`][ferrum.Chart.mark_alpha_selection] | Regularization path (alpha vs. metric). |

**Clustering and manifold:**

| Method | Purpose |
|---|---|
| [`mark_silhouette()`][ferrum.Chart.mark_silhouette] | Silhouette coefficient per sample. |
| [`mark_pca_scree()`][ferrum.Chart.mark_pca_scree] | PCA explained variance scree plot. |
| [`mark_intercluster_distance()`][ferrum.Chart.mark_intercluster_distance] | Inter-cluster distance map. |
| [`mark_decision_boundary()`][ferrum.Chart.mark_decision_boundary] | 2-D decision boundary contour. |
| [`mark_rank1d()`][ferrum.Chart.mark_rank1d] | Univariate feature ranking. |
| [`mark_rank2d()`][ferrum.Chart.mark_rank2d] | Pairwise feature ranking matrix. |
| [`mark_parallel_coordinates()`][ferrum.Chart.mark_parallel_coordinates] | Parallel coordinates plot by class. |

## Mark shortcuts

Several common mark configurations have dedicated aliases to keep code concise.

**`mark_circle()` and `mark_square()`** are convenience aliases for `mark_point(shape="circle")` and `mark_point(shape="square")` respectively. They accept the same keyword arguments as `mark_point()`.

```python
chart = fm.Chart(df).mark_circle(size=60).encode(x="x", y="y", color="group:N")
chart = fm.Chart(df).mark_square(size=60).encode(x="x", y="y", color="group:N")
```

**`mark_line(point=True)`** overlays point markers at each data coordinate on top of the line — a common pattern for time-series where individual observations should be visible:

```python
chart = fm.Chart(df).mark_line(point=True).encode(x="date:T", y="value:Q")
```

**`"|"` and `"-"` point shapes** render vertical and horizontal line markers respectively. These are useful as per-point glyphs in strip plots or rug-like overlays:

```python
chart = fm.Chart(df).mark_point(shape="|").encode(x="x", y="group:N")
```

## Mark style kwargs

Every `mark_*()` method accepts `**kwargs`: constant style overrides passed straight through to the renderer, layered on top of (or instead of) any data-driven encoding. Ferrum holds them to Rust's style table, which records, for each primitive mark, exactly the style fields that mark draws. A field the mark never draws (`mark_line(fill=...)`, `mark_text(stroke=...)`, `mark_point(font_size=...)`) raises `ValueError` immediately, naming the mark and the fields it takes, rather than being silently accepted and dropped (#243). So does a name that is no style field at all.

The same rule holds at every door that styles a mark, not only the `mark_*()` methods: `fm.Layer(mark_kwargs=...)`, a figure function's `mark=` dict and `*_kws`, a composite's `layer_kwargs=`, and `Chart.override(mark_<field>=...)`. Each refusal is voiced as the public call you made. A composite, statistical or diagnostic mark (`mark_boxplot`, `mark_density`, `mark_roc`, …) takes every field below through its own `**kwargs` and spreads each one over only the marks it draws that draw that field; a field none of its marks draws raises `ValueError`.

This section is the canonical reference for that vocabulary. Docstrings across the codebase that describe a mark's `**kwargs` point back here by name rather than re-listing every accepted key. The "Drawn by" column names the primitive marks that draw each field, as of this release; the table in Rust is the source of truth.

### Style fields

| Kwarg | Meaning | Drawn by |
|---|---|---|
| `fill` | Fill colour override (CSS colour string or hex). | point, bar, area, text, tick, rect, polygon, ribbon, arc, label, geoshape |
| `stroke` | Stroke/outline colour override. | point, line, bar, area, rule, tick, rect, polygon, ribbon, segment, arc, geoshape |
| `opacity` | Overall opacity in `[0, 1]`. | every mark but image |
| `stroke_width` | Stroke/outline width in pixels. | point, line, bar, area, rule, tick, rect, polygon, ribbon, segment, arc, geoshape |
| `stroke_dash` | Dash pattern as alternating on/off pixel lengths, e.g. `[4, 2]`, or a name such as `"dashed"`. | point, line, bar, rule, tick, rect, segment, arc |
| `stroke_cap` | Line cap style: `"butt"`, `"round"`, `"square"`. | point, line, bar, area, rule, tick, rect, segment, arc, geoshape |
| `stroke_join` | Line join style: `"miter"`, `"round"`, `"bevel"`. | point, line, bar, area, rule, tick, rect, polygon, ribbon, segment, arc, geoshape |
| `size` | Constant mark size: point area in px², or line width. Must be a finite number above zero. | point, line |
| `blend` | How the mark composites over what is already drawn: `"normal"` (the default; `"alpha"` is the same mode under its older name) or `"additive"`. Any other value raises `ValueError`. Interactive output adds each mark's color, scaled by its opacity, mark over mark. Static SVG approximates additive with `mix-blend-mode:screen`: a layer's marks in each panel composite as one group, which is then screened over what is beneath. | every mark |
| `corner_radius` | Corner rounding in pixels (per-corner tangent-circle fillets on an arc wedge, including a polar bar). | bar, rect, arc |
| `font_size` | Font size in points. | text, rect, label |
| `font_weight` | CSS font-weight, e.g. `"bold"`, `400`, `700`. | text, label |
| `align` | Horizontal text alignment: `"left"`, `"center"`, `"right"`. | text |
| `baseline` | Vertical text alignment: `"top"`, `"middle"`, `"bottom"`. | text |
| `dx`, `dy` | Fixed pixel offset from the anchor point. | text, label |
| `angle` | Rotation in degrees. | text |
| `limit` | Maximum rendered text width in pixels; clips overflow. | text |
| `radius` | Under `CoordPolar`, the pixel distance from the center at which a text placed by its `theta` wedge sits; default mid-ring. | text |
| `leader_line` | Draw a thin connecting line from the data point to the placed label. Default `False`. | label |
| `interpolate` | Curve interpolation: `"linear"`, `"monotone"`, `"step"`, `"step-before"`, `"step-after"`, `"basis"`, `"cardinal"`. (`mark_ribbon` also takes its own named `interpolate=` parameter.) | line, area, ribbon |
| `line` | Whether to draw the area's top boundary as a line. Default `False`. | area |
| `borders` | Whether to draw border lines at the band edges. (`mark_errorband` takes its own named `borders=` parameter.) | area |
| `filled` | Whether points are filled (default) or open outlines. | point |
| `shape` | Point glyph name; see the [`mark_point()`][ferrum.Chart.mark_point] docstring for the full glyph list. | point |
| `band_size` | Extent of the tick/band along its cross axis. On an ordinal tick this is a fraction of the category band width (default `0.6`); on a rug tick (a single quantitative axis, no categorical channel) it is a fraction of the panel's cross-extent, and unset keeps the fixed tick length. | tick, rect |
| `arrow` | Draw a filled triangle arrowhead at each segment's `(x2, y2)` end. Default `False`. | segment |
| `arrow_size` | Arrowhead size in pixels when `arrow=True`. Default `8.0`. | segment |
| `cmap` | Colormap name. An image draws it only over raster data, which only `mark_raster` produces; a polygon draws it only under a quantitative colour encoding. | polygon, image |
| `width`, `height` | Rendered image size in pixels. An image draws them only when it has a `url` encoding. | image |
| `zero` | Set `zero=False` to suppress the y-scale zero anchor. Default `True`. Taken at every door that styles a bar; on a flat bar chart a door-supplied `zero` is consumed exactly as `mark_bar(zero=...)` consumes it. | bar |

Some fields draw only when another input is present. A stroke field's enabling input differs per mark: a point's stroke fields (`stroke_width`, `stroke_dash`, `stroke_join`; not `stroke_cap`, which a point always draws) need a `stroke=` colour, `filled=False`, or a numeric `stroke_width` encoding; an area's stroke fields need a `stroke=` colour or `line=True`/`borders=True`; a bar's or rect's stroke fields need a `stroke=` colour or a numeric `stroke_width` encoding; a polygon's, ribbon's, or arc's stroke fields need a `stroke=` colour alone. Beyond stroke: `arrow=True` for a segment's `arrow_size`; a quantitative colour encoding for a polygon's `cmap`; raster data for an image's `cmap`; a `url` encoding for an image's `width`/`height`; a band axis paired with a `y2`/`x2` range for a rect's `band_size`; a `text` encoding for a rect's `font_size`; and both `dx` and `dy` for a label's offsets, since a label places itself unless both are set. Such a field set without its input is accepted and warns that it draws nothing, e.g. `mark_bar: stroke_width has no effect on this bar mark without a stroke color (stroke=) or a numeric stroke_width encoding (which stroke='none' turns off); it was ignored`.

### Method arguments

| Kwarg | Meaning |
|---|---|
| `orient` | `orient="horizontal"` flips the mark to a horizontal orientation; `"vertical"` is the default. Taken only by `mark_bar()` and `mark_tick()`, which consume it themselves; every other door refuses it, naming the method. |

Statistical marks' own tunables (`mark_smooth(method=, ci=, bandwidth=, degree=, k=, n=, ...)`, `mark_density(bandwidth=, kernel=, extent=, cumulative=, multiple=, ...)`, `mark_histogram(bins=, bin_count=, bin_width=, density=, cumulative=, multiple=, position_by=, ...)`, `mark_errorbar(method=, ...)`) are named parameters of those specific `mark_*()` methods, not entries in this `**kwargs` vocabulary; see each method's own docstring on [`Chart`][ferrum.Chart] for its full parameter list.

## Friendly kwarg aliases

Every mark — primitive (`mark_point`, `mark_line`), statistical (`mark_density`, `mark_histogram`, `mark_smooth`), and composite (`mark_boxplot`, `mark_violin`) — accepts the [mark style kwargs](#mark-style-kwargs) above, plus short, familiar aliases for the most common ones. Aliases are resolved before the spec is compiled — they have no runtime cost and produce identical output.

| Alias | Canonical | Notes |
|---|---|---|
| `color="red"` | `fill="red"` | Sets the mark fill color directly. |
| `alpha=0.5` | `opacity=0.5` | Mark opacity in `[0, 1]`. |
| `linetype="dashed"` | `stroke_dash=[4, 2]` | Named line patterns; see table below. |

Named `linetype` values and their dash arrays:

| `linetype=` | Dash pattern |
|---|---|
| `"solid"` | `[]` (solid line) |
| `"dashed"` | `[4, 2]` |
| `"dotted"` | `[1, 3]` |
| `"dashdot"` | `[4, 2, 1, 2]` |
| `"longdash"` | `[8, 4]` |

```python
# These three produce identical output:
fm.Chart(df).mark_line(alpha=0.7, color="steelblue", linetype="dashed").encode(x="x", y="y")
fm.Chart(df).mark_line(opacity=0.7, fill="steelblue", stroke_dash=[4, 2]).encode(x="x", y="y")
```

## Picking a mark

A quick decision guide for the common cases:

- **One variable, looking at distribution shape?** `mark_density()` or `mark_histogram()`.
- **One variable across groups?** `mark_boxplot()` (or `mark_violin()` for symmetric KDEs, `mark_swarm()` for full points).
- **Two variables, looking at relationship?** `mark_point()` (low cardinality), `mark_hex()` (high cardinality), `mark_smooth()` overlaid on points (with a regression line).
- **Two variables over time?** `mark_line()`, optionally with `mark_ribbon()` or `mark_errorband()` for uncertainty.
- **Counts by category?** `mark_bar()`. Add `encode(color="...")` and a stacking position adjustment for stacked bars.
- **A discrete grid of values?** `mark_rect()`. The same primitive serves heatmaps and binned 2-D histograms.
- **Model diagnostic?** Use the figure-level helpers in [Figure-level helpers](figure-helpers.md) and [Model diagnostics](model-diagnostics.md) rather than calling diagnostic marks directly.

## A complete example

Combining multiple marks against one data source, each layer binding its own channels (full composition is covered on the [Composition](composition.md) page):

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
points = (
    fm.Chart(iris)
    .mark_point(opacity=0.6)
    .encode(x="sepal_length", y="petal_length", color="species:N")
)
trend = (
    fm.Chart(iris)
    .mark_smooth(method="loess")
    .encode(x="sepal_length", y="petal_length", color="species:N")
)
combined = points + trend
combined
```

![Points + LOESS trend](img/marks-encodings_points_loess.png)

This puts a per-species LOESS overlay on top of a scatter. Each layer binds `color="species:N"` itself, because a binding on one layer never reaches the other; the `+` operator on `Chart` produces a layered view that renders both marks against the same axes and one shared color scale, and the smooth fits one curve per species.

## Where to go next

- [Composition](composition.md) for how to combine multiple marks and charts into compound views (`Layer`, `HConcat`, `VConcat`, `JointChart`, etc.).
- [Themes](themes.md) for changing how marks look without changing the chart spec.
- [Figure-level helpers](figure-helpers.md) for one-line entry points to common chart patterns.
- [Stats in the rendering pipeline](concepts/stats-pipeline.md) for the design rationale behind statistical marks.
- The [API Reference](../api/ferrum-toc.md) for the full method signatures of every mark and encoding channel.
