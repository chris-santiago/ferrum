# Composition

Composition is how Ferrum combines multiple charts (or multiple marks against shared axes) into a single output. Where encoding controls what one chart looks like, composition controls how charts relate to each other.

Ferrum ships six composition operators, each producing a different kind of compound view:

| Operator | When to use |
|---|---|
| **`+` (Layer)** | Multiple marks against the same axes — scatter + smooth, line + ribbon, bars + text labels. Always layers; never concatenates. |
| **`\|` (HConcat)** | Independent charts laid out left-to-right. |
| **`&` (VConcat)** | Independent charts stacked top-to-bottom. |
| **`fm.hconcat()` / `fm.vconcat()`** | Convenience functions for building concat layouts from more than two charts. |
| **`JointChart`** | Central plot with marginal distributions on the top and right. |
| **`RepeatChart`** | Template chart repeated across a grid of field combinations. |
| **`ClusterMapChart`** | Clustered heatmap with row and column dendrograms. |

The principles that govern these operators are the same as the rest of Ferrum's grammar: composition is structural, declarative, and produces a value that you can theme, save, render statically or interactively, and embed into larger views. Compound views are the same kind of object as base charts — they accept the same theme, the same renderer, and the same composition operators recursively.

## Layering: same axes, multiple marks

Layering is the most common form of composition: you have one set of axes and you want more than one mark on it. Scatter + regression line. Bars + value labels. Line + ribbon for uncertainty.

The `+` operator on `Chart` **always produces a layered view** — it never concatenates. When both charts share the same data, the result reuses the original DataFrame. When data differs, the two DataFrames are merged via null-padded diagonal concatenation; each layer's encoding references only its own columns, so the padding is invisible at render time.

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
layered = points + trend
assert layered.show_svg().startswith("<svg")
```

![Layered scatter + trend](img/composition_01.png)

Layered charts share axes by construction: both marks are drawn against the same x/y scales, the same color scale, and the same plot region. Each layer keeps its own mark and any layer-specific encoding overrides, but the shared encodings apply uniformly.

Use layering when:

- You want a regression overlay (`mark_smooth`) on a scatter (`mark_point`).
- You want a confidence band (`mark_ribbon` / `mark_errorband`) under a line.
- You want value labels (`mark_text`) on top of bars.
- You want multiple statistical summaries (mean line + min/max ribbon) on one chart.

Use concatenation (next section) when the charts should *not* share axes.

## Horizontal and vertical concatenation

Concatenation places independent charts next to each other, with each retaining its own scales, axes, and legend. There is no shared x/y; only the visual layout connects them.

The `|` operator produces an `HConcatChart`; `&` produces a `VConcatChart`:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
scatter = (
    fm.Chart(iris)
    .mark_point()
    .encode(x="sepal_length", y="petal_length", color="species:N")
)
distribution = (
    fm.Chart(iris)
    .mark_boxplot()
    .encode(x="species:N", y="sepal_length")
)
side_by_side = scatter | distribution
assert side_by_side.show_svg().startswith("<svg")
```

![Horizontal concat](img/composition_02.png)

The `&` operator stacks the same two charts vertically:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
scatter = fm.Chart(iris).mark_point().encode(x="sepal_length", y="petal_length", color="species:N")
distribution = fm.Chart(iris).mark_boxplot().encode(x="species:N", y="sepal_length")
stacked = scatter & distribution
assert stacked.show_svg().startswith("<svg")
```

![Vertical concat](img/composition_03.png)

You can chain operators to compose deeper trees: `(a | b) & (c | d)` produces a 2 × 2 grid where the two rows have different charts and the left and right columns differ within each row. The operators are left-associative and follow normal Python precedence.

`HConcatChart` and `VConcatChart` also accept explicit list construction with a `spacing` keyword for fine control:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
a = fm.Chart(iris).mark_point().encode(x="sepal_length", y="petal_length")
b = fm.Chart(iris).mark_point().encode(x="sepal_width", y="petal_width")
c = fm.Chart(iris).mark_histogram().encode(x="sepal_length")
trio = fm.HConcatChart([a, b, c], spacing=24.0)
assert trio.show_svg().startswith("<svg")
```

![Three-chart HConcat](img/composition_04.png)

The explicit form is useful when you want to control spacing or pass more than two charts in one call.

### Top-level convenience functions

`fm.hconcat()` and `fm.vconcat()` are shorthand for building concat layouts from variadic arguments:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
a = fm.Chart(iris).mark_point().encode(x="sepal_length", y="petal_length")
b = fm.Chart(iris).mark_histogram().encode(x="sepal_length")
c = fm.Chart(iris).mark_boxplot().encode(x="species:N", y="sepal_length")
row = fm.hconcat(a, b, c, spacing=20.0)
assert row.show_svg().startswith("<svg")
```

![fm.hconcat convenience](img/composition_05.png)

These are equivalent to `HConcatChart([a, b, c])` / `VConcatChart([a, b, c])` but read more naturally at the call site. Use them when you have more than two charts or want explicit `spacing` control; use the `|` / `&` operators for quick two-chart layouts.

## Joint distribution with marginals

`JointChart` lays out a central chart with optional marginal plots on the top and right. It's the same shape as seaborn's `jointplot` — a scatter with marginal histograms is the canonical example.

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
center = (
    fm.Chart(iris)
    .mark_point()
    .encode(x="sepal_length", y="petal_length", color="species:N")
)
top = fm.Chart(iris).mark_histogram().encode(x="sepal_length")
joint = fm.JointChart(center, top=top)
assert joint.show_svg().startswith("<svg")
```

![JointChart with marginal](img/composition_06.png)

The center chart shares its x-axis with the top marginal. `JointChart` also accepts a `right=` keyword that places a marginal on the right edge sharing the y-axis; the `right` marginal needs to be a chart authored with the correct orientation for that side. The `ratio` parameter (default 5) controls how much vertical space the center chart takes versus the marginal — `ratio=5` means the center is 5× taller than the top marginal.

If you want a one-line entry point that handles both marginals and the orientation for you, [`jointplot`](figure-helpers.md) in `ferrum.plots` is the convenience helper that builds a `JointChart` automatically.

## Repeating a template across fields

`RepeatChart` takes a template chart and replicates it across a grid of fields. Each cell in the grid is the template with one or both encoding channels replaced by the per-cell field name.

The template uses `Repeat.column`, `Repeat.row`, or `Repeat.layer` sentinels (from `ferrum.Repeat`) to mark which encoding channel receives the substitution:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
template = (
    fm.Chart(iris)
    .mark_point()
    .encode(x=fm.Repeat.column, y=fm.Repeat.row, color="species:N")
)
grid = fm.RepeatChart(
    template,
    row=["sepal_length", "petal_length"],
    column=["sepal_width", "petal_width"],
)
assert grid.show_svg().startswith("<svg")
```

![RepeatChart 2x2 grid](img/composition_07.png)

This produces a 2 × 2 grid of scatter plots, each cell pairing one row field on the y axis with one column field on the x axis. Pass only `column=` (with a fixed `y` in the template) for a single row of plots; pass only `row=` (with a fixed `x`) for a single column.

Use `RepeatChart` when:

- You want to see one variable plotted against many others (a pairs-plot column).
- You want a small-multiples layout where the *encoding channel* changes per cell, not just a filter on the data.

For a small-multiples layout where the data is *partitioned* across cells (one panel per group), use the `facet` encoding channel instead. The difference is structural: faceting splits one chart by a categorical field; `RepeatChart` substitutes a different encoding into a template.

`ferrum.plots.pairplot` is the figure-level helper that uses `RepeatChart` internally to produce a full pairs grid.

## Clustered heatmap with dendrograms

`ClusterMapChart` is a specialized composition: a heatmap with row and column dendrograms attached, computed from a hierarchical clustering of the data. The output is a 2 × 2 grid — heatmap (bottom-right), column dendrogram (top-right), row dendrogram (bottom-left, rotated), empty (top-left).

For most use cases, the figure-level helper [`clustermap`](figure-helpers.md) in `ferrum.plots` is the right entry point. Direct `ClusterMapChart` construction is for when you want fine control over the linkage method, the dendrogram styling, or the heatmap encoding details.

## Picking a composition operator

A decision guide for the common cases:

- **Multiple marks, one set of axes?** Use `+` (layer). Examples: scatter + smooth, line + ribbon, bars + text labels.
- **Multiple charts, no shared axes, side-by-side?** Use `|` (`HConcat`).
- **Multiple charts, no shared axes, stacked vertically?** Use `&` (`VConcat`).
- **Central plot with marginals on top and right?** Use `JointChart` (or the `jointplot` helper).
- **Template chart repeated across a grid of fields?** Use `RepeatChart` (or `pairplot` for the canonical pairs case).
- **Heatmap with hierarchical clustering structure?** Use `ClusterMapChart` (or the `clustermap` helper).
- **Same chart broken into panels by a categorical field?** That is not composition — use the `facet` / `facet_row` / `facet_col` encoding channels (see [Marks & encodings](marks-encodings.md#faceting-channels)).

## Composition is recursive

Compound views are themselves charts (in the structural sense): you can layer a `Layer`, concatenate a `JointChart`, or place a `RepeatChart` inside an `HConcatChart`. The operators compose freely.

That recursive composition is what makes complex dashboards-as-static-images viable. A four-panel model report — one ROC curve, one calibration plot, one confusion matrix, one residuals plot — is `(roc | calibration) & (confusion | residuals)`. Same grammar, same theme, same `.save()`.

## Where to go next

- [Marks & encodings](marks-encodings.md) for what goes into each chart before composition starts.
- [Figure-level helpers](figure-helpers.md) for convenience entry points (`jointplot`, `pairplot`, `clustermap`) that wrap the composition operators.
- [Themes](themes.md) for how to apply consistent styling across composed charts.
- [Model diagnostics](model-diagnostics.md) for the canonical use case: composing multiple diagnostic plots into a model evaluation view.
- The [API Reference](../api/ferrum.md) for the full signatures of `HConcatChart`, `VConcatChart`, `JointChart`, `RepeatChart`, and `ClusterMapChart`.
