# Dataframe pluralism

Ferrum is designed for the Python data ecosystem as it exists, not as it would look if everyone standardized on one dataframe library.

Real teams move between pandas, Polars, Arrow tables, modin, cuDF, dask, ibis, and NumPy arrays — sometimes within a single project, often within a single notebook. A plotting library that forces one blessed table type pushes the cost of conversion onto the user every time the source changes. Ferrum's position is that the chart model should meet your data where it already lives, not the other way around.

## One constructor for every dataframe

Pandas, Polars, modin, cuDF, dask, ibis, Arrow tables, and NumPy arrays all flow through the same [`Chart(data)`][ferrum.Chart] constructor. They are internally normalized to Arrow once, then routed through the Rust core unchanged. There are no per-framework adapters in user code, and no special-case ingestion paths inside ferrum.

<!--pytest.mark.skip-->
```python
import ferrum as fm

chart = fm.Chart(pandas_df).mark_point().encode(x="x", y="y")
chart = fm.Chart(polars_df).mark_point().encode(x="x", y="y")
chart = fm.Chart(arrow_table).mark_point().encode(x="x", y="y")
```

The chart object that comes back is the same in every case. The rest of the grammar — encodings, marks, scales, composition, themes — does not know or care which dataframe library the data came from.

## How it works

Two pieces of infrastructure carry the pluralism contract:

**Narwhals** provides the compatibility layer. It normalizes the differences between dataframe APIs into a single interface that Ferrum can program against. When you pass a pandas DataFrame, a Polars DataFrame, a modin frame, a cuDF frame, a dask DataFrame, or an ibis table, Narwhals exposes them through the same operations. Ferrum's ingestion code is written against that single interface.

**The Arrow C Data Interface** is the boundary where Python ends and Rust begins. Once the input has been normalized, columnar buffers cross into the Rust engine without row-level copying. For Polars specifically that handoff is zero-copy — the engine reads the buffers Polars already owns. For other sources, Narwhals produces an Arrow representation and the engine reads through it.

Every other layer of Ferrum — scale resolution, statistical transforms, layout, rendering — runs against the Arrow batch. The internal pipeline does not have a "pandas path" and a "Polars path" and an "ibis path." It has one path, and the dataframe pluralism is resolved before the path begins.

## What this means in practice

The shape of your code does not change when your data source changes. You do not write `if isinstance(df, pl.DataFrame): ...` in user code, you do not call `.to_pandas()` to satisfy the plotting library, and you do not maintain separate ingestion helpers per framework.

Concretely, this is what stops showing up in your codebase:

- Per-framework adapter functions before each plot call.
- `df.to_pandas()` round-trips inserted just to make the plotting library accept the input.
- Schema coercion that converts datetime types or categorical encodings between frameworks for the sake of the renderer.
- Plotting-only utility modules that mirror your real data utilities, one per dataframe type.

And what starts showing up instead is plotting code that looks the same regardless of where the table lives.

## Why "part of the contract"

Dataframe pluralism is not a checklist item. It is a structural commitment that one chart grammar should remain stable across the messy reality of Python data work — the same way it should remain stable across small and large datasets, static and interactive output, and ordinary plots versus model diagnostics.

The bet is that the cost of one good interoperability layer, paid once at ingestion, is much smaller than the accumulated cost of forcing users to convert data every time they want to plot it. The single normalization step that happens at `Chart(data)` is the price of admission. Everything downstream of that step is uniform.

That commitment shapes the rest of the library:

- **Stat transforms** ([Stats in the rendering pipeline](stats-pipeline.md)) run on Arrow columns, not on a specific dataframe type, so the same KDE / binning / smoothing works regardless of source.
- **Performance** ([Performance & scale](performance-scale.md)) depends on a columnar boundary; if every dataframe type were a special case, the zero-copy story would not hold.
- **Composition** ([One chart model](one-chart-model.md)) works because charts that come from different data sources are still the same kind of object — you can [`hconcat`][ferrum.hconcat] a Polars-backed chart with a pandas-backed chart without thinking about it.

## What this does not promise

Dataframe pluralism is about ingestion, not about preserving every framework-specific feature inside the chart. Once data crosses the Arrow boundary, the engine treats it as Arrow data. A pandas index, a Polars categorical encoding, or an ibis lazy expression does not survive in its original form past the ingestion step — it is normalized into the columnar representation the engine consumes.

That is the right trade-off for a plotting library. The engine needs one stable data model to be honest about scale, statistics, and rendering. The user-facing benefit is that you get to keep using whichever framework you already use, without the library forcing a choice.

## Where to go next

- [One chart model](one-chart-model.md) for how the same chart object sits on top of any dataframe input.
- [Stats in the rendering pipeline](stats-pipeline.md) for what the engine does with the columnar data after it crosses the boundary.
- [Performance & scale](performance-scale.md) for the architectural details of the Arrow boundary and the Python/Rust split.
