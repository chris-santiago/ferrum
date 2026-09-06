# Data transforms

Data transforms are explicit pre-processing steps applied to your data before rendering. They let you filter, reshape, aggregate, and compute derived columns directly in the chart specification — without mutating your source DataFrame.

This is different from **stat transforms** (like `mark_smooth`, `mark_density`, `mark_histogram`), which are implicit: they're triggered by the mark you choose and computed automatically by the engine. Data transforms are declared explicitly via `.transform()` and run before any mark-level statistics.

## When to use data transforms

Use data transforms when you need to:

- Filter rows before visualization (e.g. only show values above a threshold)
- Compute derived columns from expressions (e.g. ratios, differences)
- Reshape data between wide and long format
- Aggregate or bin data for summary views
- Apply window functions (running averages, rankings, lag/lead)
- Sample large datasets down to a manageable size

If the operation is purely about *how* marks are drawn (KDE shape, regression line, bin counts), prefer a stat mark. If it's about *which* data or *what* columns exist before any mark logic runs, use a data transform.

## Basic usage

Attach transforms to a chart with `.transform()`:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age": [25, 17, 34, 16, 42],
    "score": [88.0, 72.0, 95.0, 61.0, 79.0],
})

chart = (
    fm.Chart(df)
    .transform(fm.transform_filter("datum.age >= 18"))
    .mark_bar()
    .encode(x="name:N", y="score:Q")
)
```

The transform runs first: only rows where `age >= 18` survive to the mark stage. The bar chart renders three bars (Alice, Carol, Eve).

## Filtering and calculation

### `transform_filter` — row filtering

Filter rows using an expression string:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "country": ["US", "UK", "DE", "FR", "JP"],
    "gdp": [21.0, 2.8, 3.8, 2.7, 5.1],
    "population": [331, 67, 83, 67, 126],
})

chart = (
    fm.Chart(df)
    .transform(fm.transform_filter("datum.gdp > 3"))
    .mark_bar()
    .encode(x="country:N", y="gdp:Q")
)
```

You can also pass a dict for common filter patterns:

```python
# Keep only specific categories
fm.transform_filter({"country": ["US", "DE", "JP"]})

# Comparison operators
fm.transform_filter({"gdp": {">": 3, "<": 20}})
```

### `transform_calculate` — derived columns

Add a new column computed from an expression:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "city": ["NYC", "LA", "Chicago"],
    "revenue": [500.0, 300.0, 200.0],
    "cost": [350.0, 180.0, 150.0],
})

chart = (
    fm.Chart(df)
    .transform(fm.transform_calculate("profit", "datum.revenue - datum.cost"))
    .mark_bar()
    .encode(x="city:N", y="profit:Q")
)
```

The first argument is the output column name; the second is the expression.

## Reshaping

### `transform_fold` — wide to long

Fold (melt) multiple columns into key-value pairs:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "year": [2020, 2021, 2022],
    "revenue": [100.0, 120.0, 150.0],
    "expenses": [80.0, 95.0, 110.0],
})

chart = (
    fm.Chart(df)
    .transform(fm.transform_fold(["revenue", "expenses"], as_=("metric", "amount")))
    .mark_line()
    .encode(x="year:O", y="amount:Q", color="metric:N")
)
```

Each row in the original becomes two rows in the output — one for `revenue`, one for `expenses`. The `as_` parameter names the key and value columns.

### `transform_pivot` — long to wide

The inverse of fold — spread a categorical column into multiple columns:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "date": ["Mon", "Mon", "Tue", "Tue"],
    "category": ["A", "B", "A", "B"],
    "sales": [10.0, 20.0, 15.0, 25.0],
})

# Pivot so each category becomes its own column
fm.transform_pivot("category", "sales", groupby=["date"])
```

### `transform_flatten` — expand list columns

Unnest array/list columns into individual rows:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "person": ["Alice", "Bob"],
    "tags": [["python", "rust"], ["go", "python", "java"]],
})

chart = (
    fm.Chart(df)
    .transform(fm.transform_flatten(["tags"]))
    .mark_bar()
    .encode(x="tags:N", y="count():Q")
)
```

## Aggregation

### `transform_aggregate` — group-by aggregation

Collapse rows into group summaries. Each aggregate spec is a dict with `field`, `fn`, and `as` keys:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "department": ["eng", "eng", "sales", "sales", "sales"],
    "salary": [120.0, 130.0, 80.0, 90.0, 85.0],
})

chart = (
    fm.Chart(df)
    .transform(fm.transform_aggregate(
        {"field": "salary", "fn": "mean", "as": "avg_salary"},
        {"field": "salary", "fn": "count", "as": "headcount"},
        groupby=["department"],
    ))
    .mark_bar()
    .encode(x="department:N", y="avg_salary:Q")
)
```

Supported aggregation functions: `"sum"`, `"mean"`, `"median"`, `"min"`, `"max"`, `"count"`, `"variance"`, `"stdev"`, `"q1"`, `"q3"`, `"distinct"`.

### `transform_join_aggregate` — aggregate without collapsing

Adds aggregate columns to every row (like a SQL window function without ordering). The original rows are preserved:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "region": ["East", "East", "West", "West"],
    "sales": [100.0, 150.0, 200.0, 80.0],
})

# Add total_sales per region to each row, then compute percentage
chart = (
    fm.Chart(df)
    .transform(fm.transform_join_aggregate(
        {"field": "sales", "fn": "sum", "as": "region_total"},
        groupby=["region"],
    ))
    .transform(fm.transform_calculate("pct", "datum.sales / datum.region_total"))
    .mark_bar()
    .encode(x="region:N", y="pct:Q")
)
```

### `transform_top_k` — keep top groups

Keep only the top-k groups ranked by an aggregate:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "product": ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"],
    "revenue": [50.0, 30.0, 80.0, 10.0, 45.0, 55.0, 25.0, 90.0, 15.0, 40.0],
})

chart = (
    fm.Chart(df)
    .transform(fm.transform_top_k(3, field="revenue", op="sum", groupby=["product"]))
    .mark_bar()
    .encode(x="product:N", y="sum(revenue):Q")
)
```

This keeps only the three products with the highest total revenue: the aggregate (`op="sum"` over `revenue`) is computed per `groupby` key, the keys are ranked, and every row belonging to a top-3 key survives. Without `groupby=`, the transform ranks the distinct values of `field` itself by their own aggregate — useful for trimming a long tail of raw values, but not a per-category ranking. Note that a row whose `groupby` key is null belongs to no group and is always dropped from grouped output, regardless of `n`; the ungrouped form instead collects null field values into their own `"__null__"` group, which can survive the ranking.

### `transform_bin` — binning

Bin a continuous field into discrete intervals:

```python
import ferrum as fm
import polars as pl
import numpy as np

rng = np.random.default_rng(42)
df = pl.DataFrame({"value": rng.normal(0, 1, 200).tolist()})

chart = (
    fm.Chart(df)
    .transform(fm.transform_bin("value", maxbins=15))
    .mark_bar()
    .encode(x="value_bin:Q", y="count():Q")
)
```

Parameters: `maxbins` sets the target bin count, `step` sets an explicit bin width, and `nice=True` (the default) rounds boundaries to clean values.

## Window functions

### `transform_window` — rolling operations, rank, lag/lead

Window transforms compute values over a sliding or partitioned frame:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "day": list(range(1, 11)),
    "sales": [12.0, 15.0, 13.0, 18.0, 20.0, 17.0, 22.0, 19.0, 25.0, 23.0],
})

# 3-day rolling average
chart = (
    fm.Chart(df)
    .transform(fm.transform_window(
        {"op": "mean", "field": "sales", "as": "rolling_avg"},
        sort=["day"],
        frame=(-1, 1),  # 1 preceding, current, 1 following
    ))
    .mark_line()
    .encode(x="day:Q", y="rolling_avg:Q")
)
```

Window operations include:

| Operation | Description |
|---|---|
| `"row_number"` | Row index within the window |
| `"rank"` | Competition rank, tie-aware (`1, 1, 3`) |
| `"dense_rank"` | Dense rank, tie-aware, no gaps (`1, 1, 2`) |
| `"lag"` | Previous row value (use `param` for offset) |
| `"lead"` | Next row value (use `param` for offset) |
| `"sum"`, `"mean"`, `"min"`, `"max"` | Rolling aggregates |
| `"first_value"`, `"last_value"` | Frame boundary values |

### Tie-aware ranks

`rank` and `dense_rank` written without a `field` (the common case) derive their ties from the `sort` columns: rows equal on **every** sort column, post-coercion, share a rank — competition-style for `rank` (`1, 1, 3`) and dense-style for `dense_rank` (`1, 1, 2`), matching SQL and Vega. An explicit `field=` keeps its own field-keyed ranking behavior, unaffected. A spec with no `sort=` at all has no key to tie on, so both ops fall back to plain row order.

```python
df = pl.DataFrame({"score": [90, 90, 85, 80, 80, 80]})

fm.transform_window({"op": "rank", "as": "r"}, sort=["score"])
# ranks (ascending by score): 80→1, 80→1, 80→1, 85→4, 90→5, 90→5

fm.transform_window({"op": "dense_rank", "as": "dr"}, sort=["score"])
# dense ranks: 80→1, 80→1, 80→1, 85→2, 90→3, 90→3
```

### Peer-aware rolling frames and `ignore_peers`

The `frame` parameter specifies `(preceding, following)` bounds. Use `None` for unbounded:

```python
# Cumulative sum (unbounded preceding to current row)
fm.transform_window(
    {"op": "sum", "field": "sales", "as": "cumulative"},
    sort=["day"],
    frame=(None, 0),
)

# Rank within each group
fm.transform_window(
    {"op": "rank", "as": "rank"},
    sort=["score"],
    groupby=["department"],
)
```

By default (`ignore_peers=False`), a rolling frame treats rows tied on the `sort` key as one unit: the frame is anchored on the tied run's own bounds and then *both* ends independently expand outward to whole peer groups, so every tied row shares one frame and therefore one aggregate value, and a frame boundary never splits a peer group. Pass `ignore_peers=True` to get plain row-offset frames instead, where each row's frame is computed from its own position regardless of ties.

```python
df = pl.DataFrame({"t": [1, 1, 2, 3, 3], "x": [10.0, 20.0, 30.0, 40.0, 50.0]})

fm.transform_window(
    {"op": "sum", "field": "x", "as": "s"},
    sort=["t"], frame=(-1, 0),
)
# peer-aware (default): [30, 30, 60, 120, 120] — the two t=1 rows share one
# frame/value, and so do the two t=3 rows.

fm.transform_window(
    {"op": "sum", "field": "x", "as": "s"},
    sort=["t"], frame=(-1, 0), ignore_peers=True,
)
# row-offset frames: [10, 30, 50, 70, 90] — no peer grouping.
```

**This is deliberately not Vega's `ignorePeers: false` semantics.** Vega's default also expands both ends of a frame outward to peer-group boundaries, but anchors the *un-expanded* frame on the current row rather than on the peer run's own bounds — so Vega does not guarantee that every tied row ends up sharing the same value. On the fixture above, Vega's `ignorePeers: false` output is `[30, 30, 60, 120, 90]`: only the last row differs, because Vega's trailing `t=3` row anchors its own frame at position 3, not the peer run's start at position 2. Ferrum's peer-anchored rule guarantees tied rows always share one value; treat this as a documented, intentional divergence from Vega, not a bug to reconcile. `ignore_peers=True` sidesteps the difference entirely (both engines produce the same row-offset output), and any sort key with no actual ties is unaffected either way.

## Statistical transforms

These transforms produce new datasets from statistical computations. They replace the original rows with computed output.

### `transform_density` — kernel density estimation

Compute a KDE curve from a single field:

```python
import ferrum as fm
import polars as pl
import numpy as np

rng = np.random.default_rng(42)
df = pl.DataFrame({
    "group": ["A"] * 100 + ["B"] * 100,
    "value": rng.normal(0, 1, 100).tolist() + rng.normal(2, 0.5, 100).tolist(),
})

chart = (
    fm.Chart(df)
    .transform(fm.transform_density("value", groupby=["group"]))
    .mark_area(opacity=0.5)
    .encode(x="value:Q", y="density:Q", color="group:N")
)
```

The output columns default to `"value"` and `"density"` (configurable via `as_`). Use `groupby` to compute separate densities per category.

The `kernel=` parameter selects the kernel function. Supported values: `"gaussian"` (default), `"epanechnikov"` (alias `"epan"`), `"tophat"`, `"uniform"`, `"cosine"`, `"exponential"`, `"linear"`.

```python
chart = (
    fm.Chart(df)
    .transform(fm.transform_density("value", kernel="epanechnikov", groupby=["group"]))
    .mark_area(opacity=0.5)
    .encode(x="value:Q", y="density:Q", color="group:N")
)
```

### `transform_regression` — regression line

Fit a regression model and output the fitted curve:

```python
import ferrum as fm
import polars as pl
import numpy as np

rng = np.random.default_rng(42)
x = rng.uniform(0, 10, 50)
df = pl.DataFrame({"x": x.tolist(), "y": (2 * x + rng.normal(0, 1, 50)).tolist()})

chart = (
    fm.Chart(df)
    .transform(fm.transform_regression("x", "y", method="linear"))
    .mark_line()
    .encode(x="x:Q", y="y:Q")
)
```

Methods: `"linear"`, `"poly"` (set `order=` for degree), `"exp"`, `"log"`, `"pow"`.

Use `groupby=` to fit an independent curve per group, with the group column appended to the output at its original dtype so a `color=` encoding maps to it:

```python
fm.transform_regression("x", "y", method="linear", groupby=["g"])
```

Each group's output grid spans that group's own x-range (not the whole dataset's), matching `mark_smooth`'s grouped behavior.

!!! note "mark_smooth method aliases"
    `mark_smooth(method=)` accepts several friendly aliases in addition to the primary names. `"linear"`, `"quadratic"`, and `"cubic"` map to OLS polynomial fits of degree 1, 2, and 3 respectively; `"log"` and `"sqrt"` fit log and square-root curves. `"lm"` is the primary name for linear fits (pass `degree=` for higher-order polynomials); `"loess"` is the name for locally weighted regression; `"glm"` fits a generalized linear model (Gaussian/Identity); `"gam"` fits a generalized additive model — a penalized cubic B-spline with a GCV-chosen smoothing parameter, deterministic and unaffected by `seed=`. All of the following are valid: `method="lm"`, `method="linear"`, `method="loess"`, `method="quadratic"`, `method="cubic"`, `method="log"`, `method="sqrt"`, `method="glm"`, `method="gam"`.

### `transform_loess` — LOESS smoothing

Non-parametric local regression:

```python
import ferrum as fm
import polars as pl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 4 * np.pi, 80)
df = pl.DataFrame({"x": x.tolist(), "y": (np.sin(x) + rng.normal(0, 0.3, 80)).tolist()})

chart = (
    fm.Chart(df)
    .transform(fm.transform_loess("x", "y", bandwidth=0.3))
    .mark_line()
    .encode(x="x:Q", y="y:Q")
)
```

The `bandwidth` parameter controls smoothness — smaller values follow the data more closely; larger values produce smoother curves. Like `transform_regression`, `groupby=` fits an independent curve per group, each spanning that group's own x-range, with the group column preserved in the output.

## Utilities

### `transform_sample` — random subsample

Downsample large datasets for faster rendering:

```python
import ferrum as fm
import polars as pl
import numpy as np

rng = np.random.default_rng(42)
df = pl.DataFrame({
    "x": rng.normal(0, 1, 10000).tolist(),
    "y": rng.normal(0, 1, 10000).tolist(),
})

chart = (
    fm.Chart(df)
    .transform(fm.transform_sample(500, seed=42))
    .mark_point(opacity=0.5)
    .encode(x="x:Q", y="y:Q")
)
```

The `seed` parameter ensures deterministic output across renders.

### `transform_impute` — fill missing values

Replace nulls in a column using a specified strategy:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "month": [1, 2, 3, 4, 5, 6],
    "sales": [100.0, None, 120.0, None, 140.0, 150.0],
})

chart = (
    fm.Chart(df)
    .transform(fm.transform_impute("sales", method="median"))
    .mark_line()
    .encode(x="month:O", y="sales:Q")
)
```

Methods: `"value"` (constant, set `value=`), `"mean"`, `"median"`, `"min"`, `"max"`. `groupby=` computes each group's fill value from that group's own rows (not the whole column), and fills only that group's null/NaN rows.

If a group has *no* non-null values to compute a statistic from, it falls back to the whole column's statistic instead of degrading to `0.0` (which would silently distort the domain) or `±inf` (which would drop the group's rows from the render entirely).

#### `key=` / `keyvals=` — filling in missing rows, not just missing values

`transform_impute` can also *add* rows, following Vega's `impute` semantics. `key=` names a column whose values define a "should exist" universe within each group; any `(group, key)` combination missing from the data gets one row appended, with the key's own column filled to the missing value and every other non-group column set to null:

```python
df = pl.DataFrame({
    "region": ["East", "East", "West"],
    "month": [1, 2, 1],
    "sales": [100.0, 120.0, 80.0],
})

fm.transform_impute(
    "sales", method="mean",
    groupby=["region"], key="month", keyvals=[1, 2, 3],
)
# East already has months 1 and 2; month 3 is appended with sales = East's
# own mean. West only has month 1; months 2 and 3 are appended the same way.
```

The key universe is `keyvals=` when given, else the distinct values of `key` observed across the whole batch. `keyvals=` requires `key=` — passing it alone raises, since a key universe is meaningless without a key column. Appended rows use the same per-group fill value (including the whole-column fallback above, for a group with no data of its own) as the ordinary null-filling path.

### `transform_stack` — stacking positions

Compute cumulative start/end positions for stacked bar or area charts:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "quarter": ["Q1", "Q1", "Q2", "Q2"],
    "region": ["East", "West", "East", "West"],
    "revenue": [100.0, 80.0, 120.0, 90.0],
})

fm.transform_stack("revenue", groupby=["quarter", "region"], offset="normalize")
```

The `offset` parameter controls stacking strategy: `"zero"` (standard cumulative), `"normalize"` (100% stack), or `"center"` (streamgraph).

### `transform_timeunit` — temporal extraction

Extract a unit from a datetime field:

```python
import ferrum as fm
import polars as pl
from datetime import date

df = pl.DataFrame({
    "date": [date(2023, 1, 15), date(2023, 3, 20), date(2023, 6, 10), date(2023, 11, 5)],
    "value": [10.0, 25.0, 18.0, 30.0],
})

chart = (
    fm.Chart(df)
    .transform(fm.transform_timeunit("date", "month"))
    .mark_bar()
    .encode(x="month_date:O", y="value:Q")
)
```

Units: `"year"`, `"month"`, `"day"`, `"hour"`, `"minute"`, `"second"`, `"day_of_week"`, `"week"`, `"quarter"`.

## Expression syntax

Transform expressions (used in `transform_filter` and `transform_calculate`) follow a Vega-style syntax:

| Syntax | Example |
|---|---|
| Field access | `datum.field_name` |
| Bracket access | `datum["field with spaces"]` |
| Arithmetic | `datum.x * 2 + datum.y` |
| Modulo | `datum.x % 2 == 0` |
| Comparison | `datum.age >= 18` |
| Logical AND | `datum.x > 0 && datum.y > 0` |
| Logical OR | `datum.x < 0 \|\| datum.x > 100` |
| Logical NOT | `!datum.active` |
| Ternary | `datum.x > 0 ? 'positive' : 'non-positive'` |
| String literals | `datum.status == 'active'` |
| Boolean / null literals | `datum.active == true`, `datum.x != null` |
| Scientific notation | `datum.x < 1e1`, `datum.y > 2.5e-3` |
| Array literals | `[1, 2, 3]`, `['US', 'DE', 'JP']` |
| Function calls | `indexof(array, value)` |
| Membership | `indexof([1, 2, 3], datum.id) >= 0` |

`indexof(array, value)` is the one builtin function today (the grammar's call syntax and array literals are general enough to add more without a new syntax); it returns the index of the first array element equal to `value`, or `-1` when not found. Calling an unregistered function name raises a `ValueError` naming the function and the supported set.

## Chaining transforms

Multiple transforms execute in sequence — each one operates on the output of the previous. You can chain separate `.transform()` calls or pass multiple transforms in a single call (`.transform(a, b, c)` is equivalent to `.transform(a).transform(b).transform(c)`):

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "product": ["A", "B", "C", "D", "E"] * 20,
    "region": (["North", "South"] * 50),
    "revenue": [float(i * 7 % 13 + 5) for i in range(100)],
})

chart = (
    fm.Chart(df)
    # Step 1: keep only top-3 products by total revenue
    .transform(fm.transform_top_k(3, field="revenue", op="sum", groupby=["product"]))
    # Step 2: aggregate by product and region
    .transform(fm.transform_aggregate(
        {"field": "revenue", "fn": "sum", "as": "total"},
        groupby=["product", "region"],
    ))
    # Step 3: compute percentage within each product
    .transform(fm.transform_join_aggregate(
        {"field": "total", "fn": "sum", "as": "product_total"},
        groupby=["product"],
    ))
    .transform(fm.transform_calculate("pct", "datum.total / datum.product_total"))
    .mark_bar()
    .encode(x="product:N", y="pct:Q", color="region:N")
)
```

The order matters: filtering before aggregation reduces the data that gets summarized; aggregating before filtering lets you filter on computed values.

## Where to go next

- [Marks & encodings](marks-encodings.md) for the mark types and encoding channels that consume transform output.
- [Composition](composition.md) for combining multiple transformed views into compound charts.
- [Figure-level helpers](figure-helpers.md) for one-line chart functions that handle common transform patterns internally.
- The [API Reference](../api/ferrum-toc.md) for the full signatures of every transform function.
