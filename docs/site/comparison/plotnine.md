# Migrating from plotnine

plotnine is the most faithful ggplot2 port in Python — a full grammar-of-graphics implementation with layers, facets, scales, coordinates, and themes. Ferrum inherits the same grammar-first philosophy but replaces the matplotlib rendering backend with a Rust engine, adds model diagnostics, and extends composition beyond faceting.

If you think in `ggplot() + geom_*() + facet_*()`, the translation to Ferrum is direct.

## Concept mapping

| plotnine | Ferrum | Notes |
|---|---|---|
| `ggplot(df, aes(x, y))` | `fm.Chart(df).encode(x=, y=)` | Aesthetics become encoding channels. |
| `geom_point()` | `.mark_point()` | — |
| `geom_line()` | `.mark_line()` | — |
| `geom_bar(stat="identity")` | `.mark_bar()` | — |
| `geom_col()` | `.mark_bar()` | — |
| `geom_histogram()` | `.mark_histogram()` | Binning computed in Rust. |
| `geom_density()` | `.mark_density()` | KDE computed in Rust. |
| `geom_smooth()` | `.mark_smooth()` | `method=` accepts `"lm"`, `"loess"`, `"lowess"`, `"poly"`. |
| `geom_boxplot()` | `.mark_boxplot()` | — |
| `geom_violin()` | `.mark_violin()` | — |
| `geom_tile()` | `.mark_rect()` | — |
| `geom_text()` | `.mark_text()` | — |
| `geom_hline()` / `geom_vline()` | `.mark_rule()` | — |
| `geom_area()` | `.mark_area()` | — |
| `geom_errorbar()` | `.mark_errorbar()` | — |
| `geom_ribbon()` | `.mark_ribbon()` | — |
| `geom_qq()` | `.mark_qq()` | — |
| `geom_contour()` | `.mark_contour()` | — |
| `geom_hex()` | `.mark_hex()` | — |
| `stat_smooth(method="lm")` | `.mark_smooth(method="lm")` | Stats are part of the mark declaration. |
| `stat_bin()` | `.mark_histogram()` | — |
| `stat_density()` | `.mark_density()` | — |
| `facet_wrap(~var)` | `.encode(facet="var")` | Faceting is an encoding channel. |
| `facet_grid(row~col)` | `.encode(facet_row="row", facet_col="col")` | — |
| `scale_x_log10()` | `.encode(x=fm.X("x", scale=fm.Scale(type="log")))` | Scales are values, not global functions. |
| `coord_flip()` | `.coord("flip")` | — |
| `labs(title=, x=, y=)` | `.properties(title=)` | Axis labels inferred from field names or set via `fm.X("field", title="Label")`. |
| `theme_bw()` | `fm.themes.minimal` | See [Themes](../guide/themes.md) for all 12. |
| `theme_set()` | `fm.set_default_theme()` | Scope-bounded via context manager; per-chart `.theme()` always wins. |
| `ggsave()` | `.save()` | Supports SVG, PNG, HTML. |

## Key differences

### Composition beyond faceting

plotnine composes layers with `+` and panels with `facet_wrap` / `facet_grid`. That covers faceted views of the same data, but not arbitrary chart arrangements.

Ferrum adds horizontal and vertical concatenation for unrelated charts:

<!--pytest.mark.skip-->
```python
# plotnine — no equivalent for side-by-side unrelated charts
# You'd need patchwork (R) or manual matplotlib subplot management

# Ferrum
scatter = fm.Chart(df).mark_point().encode(x="x", y="y")
histogram = fm.Chart(df).mark_histogram().encode(x="value")
combined = scatter | histogram  # side by side
stacked = scatter & histogram   # top and bottom
combined.save("report.svg")
```

### No matplotlib dependency

plotnine renders through matplotlib. Ferrum renders SVG, PNG, and interactive HTML directly from Rust. There is no `plt.show()`, no figure/axes management, no Cairo or Tk dependency, and no implicit display state.

<!--pytest.mark.skip-->
```python
# plotnine
from plotnine import ggplot, aes, geom_point
p = ggplot(df, aes("x", "y")) + geom_point()
p.save("scatter.svg")  # renders through matplotlib

# Ferrum
import ferrum as fm
chart = fm.Chart(df).mark_point().encode(x="x", y="y")
chart.save("scatter.svg")  # renders through Rust
```

### Scale ceiling

plotnine inherits matplotlib's rendering pipeline, which means individual SVG elements per mark. At ~100k marks, renders slow significantly and files balloon. Ferrum's auto-raster transparently switches to rasterized output at high mark counts — the same chart spec works at 100 rows and 10M rows.

### DataFrame pluralism

plotnine requires pandas. Ferrum accepts polars, pandas, modin, cuDF, dask, ibis, and pyarrow through the same `Chart(data)` constructor.

### Interactive rendering

plotnine has no built-in interactivity beyond matplotlib's GUI backends. Ferrum's `.interactive()` produces a GPU-backed WASM renderer with selections, zoom/pan, linked views, and tooltips — in Jupyter via anywidget or as standalone HTML.

### Model diagnostics

plotnine has no model evaluation surface. Ferrum includes 44 figure-level helpers and 28 visualizer classes for classification, regression, clustering, and explainability diagnostics — all returning the same `Chart` objects that compose with grammar operators.

### Themes are values, not global state

plotnine uses `theme_set()` to mutate global matplotlib rcParams. Ferrum themes are immutable values applied per-chart or scope-bounded via context manager:

<!--pytest.mark.skip-->
```python
# plotnine — global mutation
from plotnine import theme_set, theme_bw
theme_set(theme_bw())

# Ferrum — per-chart or scoped
chart.theme(fm.themes.publication)
# or process-wide (scope-bounded):
with fm.theme_context(fm.themes.dark):
    ...
```

### Statistical transforms in Rust

plotnine delegates statistical computation to scipy and statsmodels via matplotlib. Ferrum declares transforms in the chart spec and computes them in Rust — KDE, LOESS, bootstrap CIs, binning, and aggregations run in the engine, not in Python.

## Coverage comparison

| Category | plotnine | Ferrum |
|---|---|---|
| Primitive marks | point, line, bar, area, tile, text, rule, segment, step | All of plotnine's + tick, rect, image |
| Statistical marks | histogram, density, smooth, boxplot, violin, qq, hex, contour, ribbon, errorbar | All of plotnine's + boxen, swarm, raster, function |
| Faceting | `facet_wrap`, `facet_grid` | `facet`, `facet_row`, `facet_col` encoding channels |
| Composition | `+` (layers only) | `+` layer, `\|` hconcat, `&` vconcat, `RepeatChart`, `JointChart` |
| Coordinates | `coord_flip`, `coord_fixed`, `coord_cartesian`, `coord_polar` | `"flip"`, `"polar"`, `"theta"`, `"radial"` |
| Scales | Full ggplot2 scale system | Typed scale values with `fm.Scale(type=, domain=, range=)` |
| Themes | ggplot2 theme system (bw, classic, minimal, etc.) | 12 built-in themes (Paper Ink, Slate Citrus, Dark, Publication, Economist, etc.) |
| Interactivity | matplotlib backends only | WASM/GPU renderer with selections, zoom/pan, linked views |
| Model diagnostics | — | 44 helpers, 28 visualizer classes (ROC, PR, confusion matrix, SHAP, PDP, etc.) |
| DataFrames | pandas only | polars, pandas, modin, cuDF, dask, ibis, pyarrow |
| System deps | matplotlib (Cairo/Tk/Qt) | None — pure wheel |

## Where to go next

- [Marks & encodings](../guide/marks-encodings.md) for the full mark and channel reference.
- [Composition](../guide/composition.md) for the `+`, `|`, `&` operators.
- [Themes](../guide/themes.md) for Ferrum's twelve built-in themes.
- [Gallery](../gallery/index.md) for rendered examples with code.
