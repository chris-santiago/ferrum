# Ferrum

Grammar-of-graphics statistical visualization for Python, with a Rust core.

Ferrum builds every chart — scatter plot, faceted histogram, ROC curve, SHAP beeswarm — from the same grammar of data, encodings, marks, scales, and statistical transforms. The declaration layer is Python; the computation engine is Rust compiled via PyO3.

## Quickstart

```bash
pip install ferrum-viz
```

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 3, 5], "group": ["a", "a", "b", "b"]})

chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="group:N")
chart.save("scatter.svg")
```

## Key features

- **One chart model** — scatter plots, statistical graphics, and ML diagnostics share the same grammar and compose with `+`, `|`, `&`.
- **Stat transforms in the pipeline** — KDE, LOESS, bootstrap CIs, binning, and aggregations are declared in the chart spec and computed in Rust.
- **Model diagnostics as charts** — `fm.roc_chart(model, X, y)`, `fm.confusion_matrix_chart(...)`, `fm.shap_chart(...)` return regular `Chart` objects.
- **Zero system dependencies** — no Cairo, no X11, no display server. Renders anywhere `pip install` works.
- **DataFrame pluralism** — polars, pandas, modin, cuDF, dask, and ibis all work through `Chart(data)`.
- **12 built-in themes** — from Paper Ink (warm cream default) to dark, publication, and editorial styles.

## Examples

```python
# Layer a LOESS trend on a scatter plot
points = fm.Chart(df).mark_point(opacity=0.6).encode(x="x", y="y", color="group:N")
trend = fm.Chart(df).mark_smooth(method="loess").encode(x="x", y="y", color="group:N")
chart = points + trend

# Compose diagnostics into a model report
source = fm.ModelSource(model, X_test, y_test)
report = (fm.roc_chart(source) | fm.confusion_matrix_chart(source)) & fm.importance_chart(source)
report.save("model_report.svg")

# Figure-level helpers
fm.displot(df, x="value", hue="group", kind="kde")
fm.catplot(df, x="species", y="measurement", kind="violin")
fm.pairplot(df, vars=["a", "b", "c"], hue="label")
```

## Architecture

| Layer | Role |
|---|---|
| `src/ferrum/` | Python declaration API — Chart, encodings, marks, themes, plots |
| `crates/ferrum-core/` | Rust computation engine — transforms, scales, rendering |
| Arrow CDI | Zero-copy data transport between Python and Rust via `pyo3-arrow` |

## Development

Requires Python 3.10+, Rust toolchain, and [maturin](https://www.maturin.rs/).

```bash
uv sync
unset CONDA_PREFIX && uv run maturin develop       # build Rust extension
uv run pytest                                       # run tests
```

## License

See [LICENSE](LICENSE) for details.
