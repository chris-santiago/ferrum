# First plot

This page gets you from zero to a rendered chart in under a minute. By the end you'll have a scatter plot, a layered chart, and a saved SVG — and you'll know the three-piece pattern that every Ferrum chart follows.

## The pattern

Every Ferrum chart is **data + mark + encoding**:

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
assert chart.show_svg().startswith("<svg")
```

![Iris scatter plot](img/first-plot_01.png)

That's the whole thing: `Chart(data)` binds your DataFrame, `.mark_point()` picks the geometry, `.encode(...)` maps columns to visual channels. The result is a [`Chart`][ferrum.Chart] object — call [`.show_svg()`][ferrum.Chart.show_svg] to render it, [`.save()`][ferrum.Chart.save] to write it to disk, or just display it in a Jupyter notebook (where it renders automatically).

## Add a trend line

Want a regression overlay? Layer it with `+`:

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
    .mark_smooth(method="loess", groupby="species")
    .encode(x="sepal_length", y="petal_length", color="species:N")
)
chart = points + trend
assert chart.show_svg().startswith("<svg")
```

![Scatter with LOESS trend](img/first-plot_02.png)

The `+` operator always layers — both marks share the same axes. The LOESS smooth is computed in Rust; you declared what you wanted, not how to compute it.

## Try a different mark

Different questions call for different marks. The pattern is always the same — data, mark, encoding:

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
    .encode(x="species:N", y="sepal_length", color="species:N")
)
assert chart.show_svg().startswith("<svg")
```

![Boxplot by species](img/first-plot_03.png)

## Apply a theme

Themes are one method call:

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
    .theme(fm.themes.publication)
)
assert chart.show_svg().startswith("<svg")
```

![Publication theme](img/first-plot_04.png)

Ferrum ships [twelve built-in themes](../guide/themes.md) in the [`themes`][ferrum.themes] module — from Paper Ink (the warm default) to dark, publication, and editorial styles.

## What just happened

In four snippets you used:

1. **Data binding** — `fm.Chart(iris)` accepts polars, pandas, modin, cuDF, dask, ibis, pyarrow, or dict-of-arrays. One constructor.
2. **Marks** — [`mark_point()`][ferrum.Chart.mark_point], [`mark_smooth()`][ferrum.Chart.mark_smooth], [`mark_boxplot()`][ferrum.Chart.mark_boxplot]. Ferrum has 28+ marks covering primitives, statistical transforms, distributions, and model diagnostics.
3. **Encodings** — `x`, `y`, `color`. Shorthand strings like `"species:N"` set the type (Nominal); the full form [`fm.X("field", type="Q", title="...")`][ferrum.X] gives finer control.
4. **Composition** — `+` layers marks on shared axes. `|` and `&` concatenate charts side-by-side or stacked.
5. **Themes** — `.theme(fm.themes.publication)` swaps the entire visual style without touching the data or encoding.

## Where to go next

- [Marks & encodings](../guide/marks-encodings.md) — the full mark and encoding reference.
- [Composition](../guide/composition.md) — layering, concatenation, joint charts, repeat grids.
- [Themes](../guide/themes.md) — the twelve built-in themes, custom themes, and scoped defaults.
- [Figure-level helpers](../guide/figure-helpers.md) — one-line entry points for common chart patterns.
- [Model diagnostics](../guide/model-diagnostics.md) — ROC curves, confusion matrices, SHAP — all as charts.
