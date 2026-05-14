# Migrating from seaborn

Ferrum's figure-level helpers follow seaborn's signatures intentionally. If you know `displot`, `catplot`, `lmplot`, `pairplot`, `jointplot`, and `heatmap`, you already know the Ferrum equivalents — the call sites are nearly identical.

The differences are structural: Ferrum returns chart objects you compose with grammar operators, not matplotlib `Figure`/`Axes` pairs. There's no `plt.show()`, no `fig, ax = plt.subplots()`, and no global state.

## Function mapping

| seaborn | Ferrum | Notes |
|---|---|---|
| `sns.displot(df, x=, kind=)` | [`fm.displot(df, x=, kind=)`][ferrum.displot] | Same signature. `kind` ∈ `"hist"`, `"kde"`, `"ecdf"`. |
| `sns.catplot(df, x=, y=, kind=)` | [`fm.catplot(df, x=, y=, kind=)`][ferrum.catplot] | Same `kind` values: `"strip"`, `"box"`, `"violin"`, `"swarm"`, `"bar"`, `"point"`, `"boxen"`, `"count"`. |
| `sns.lmplot(df, x=, y=)` | [`fm.lmplot(df, x=, y=)`][ferrum.lmplot] | `method=` replaces `order=` for polynomial fits. |
| `sns.residplot(df, x=, y=)` | [`fm.residplot(df, x=, y=)`][ferrum.residplot] | Same interface. |
| `sns.pairplot(df)` | [`fm.pairplot(df)`][ferrum.pairplot] | `vars=`, `hue=`, `corner=`, `diag_kind=` all supported. |
| `sns.jointplot(df, x=, y=)` | [`fm.jointplot(df, x=, y=)`][ferrum.jointplot] | `kind=`, `marginal_kind=` match seaborn. |
| `sns.heatmap(data)` | [`fm.heatmap(data)`][ferrum.heatmap] | `annot=`, `cmap=` supported. |
| `sns.clustermap(data)` | [`fm.clustermap(data)`][ferrum.clustermap] | `method=`, `metric=`, `z_score=` supported. |

## Key differences

### No matplotlib dependency

Ferrum renders SVG directly from Rust. There is no `plt.show()`, no figure/axes management, and no implicit display state. You get a [`Chart`][ferrum.Chart] object back and decide what to do with it.

<!--pytest.mark.skip-->
```python
# seaborn
import seaborn as sns
import matplotlib.pyplot as plt
g = sns.displot(df, x="value", kind="kde")
plt.savefig("kde.svg")
plt.close()

# Ferrum
import ferrum as fm
chart = fm.displot(df, x="value", kind="kde")
chart.save("kde.svg")
```

### Composition with operators

seaborn's `FacetGrid` returns a matplotlib figure. Combining two seaborn plots requires manual subplot management. In Ferrum, charts compose with operators:

<!--pytest.mark.skip-->
```python
scatter = fm.lmplot(df, x="x", y="y")
distribution = fm.displot(df, x="x", kind="hist")
report = scatter | distribution  # side by side
report.save("report.svg")
```

### Themes are values

seaborn uses `sns.set_theme()` to mutate global matplotlib rcParams. Ferrum themes are immutable values:

<!--pytest.mark.skip-->
```python
# seaborn — global mutation
sns.set_theme(style="darkgrid", palette="deep")

# Ferrum — per-chart or scoped
chart.theme(fm.themes.dark)
# or process-wide (but scope-bounded):
with fm.theme_context(fm.themes.dark):
    ...
```

### DataFrame pluralism

seaborn requires pandas. Ferrum accepts polars, pandas, modin, cuDF, dask, ibis, and pyarrow — the same `Chart(data)` constructor handles all of them.

## What seaborn has that Ferrum does not (yet)

- **Interactive backends** — seaborn inherits matplotlib's interactive renderers (Qt, Tk, notebook widgets). Ferrum's interactive renderer is Phase 11.
- **`relplot`** — seaborn's relational plot helper. In Ferrum, the same result is `fm.Chart(df).mark_point().encode(x=, y=, hue=).facet(col=)`.
- **Axes-level API** — seaborn's `sns.scatterplot(ax=ax)` pattern for embedding into existing matplotlib figures. In Ferrum, you compose with `+`, `|`, `&` instead.

## Where to go next

- [Figure-level helpers](../guide/figure-helpers.md) for the full helper reference.
- [Themes](../guide/themes.md) for Ferrum's twelve built-in themes.
- [Composition](../guide/composition.md) for the `+`, `|`, `&` operators that replace matplotlib subplot management.
