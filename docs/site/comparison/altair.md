# Migrating from Altair

Ferrum's API is modeled directly on Altair. If you already know Altair, most of your knowledge transfers unchanged — same `Chart(df).mark_*().encode()` pattern, same encoding shorthand, same `field:type` syntax, same composition operators.

The differences are in the rendering layer (Rust instead of Vega-Lite/JavaScript), the extension to model diagnostics, and a handful of API details documented below.

## Quick translation

<!--pytest.mark.skip-->
```python
# Altair
import altair as alt
chart = alt.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
chart.save("scatter.svg")

# Ferrum
import ferrum as fm
chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
chart.save("scatter.svg")
```

Same pattern. Same encoding shorthand. Same `.save()`. Swap the import.

## Encoding shorthand

Ferrum uses the same `"field:type"` shorthand as Altair.

| Type letter | Meaning |
|---|---|
| `:Q` | Quantitative (continuous numeric) |
| `:N` | Nominal (unordered categorical) |
| `:O` | Ordinal (ordered categorical) |
| `:T` | Temporal (date/time) |

<!--pytest.mark.skip-->
```python
# Altair
alt.Chart(df).mark_point().encode(
    x="date:T",
    y="value:Q",
    color="category:N",
)

# Ferrum — identical
fm.Chart(df).mark_point().encode(
    x="date:T",
    y="value:Q",
    color="category:N",
)
```

You can also use typed channel objects for fine-grained control:

<!--pytest.mark.skip-->
```python
fm.Chart(df).mark_point().encode(
    x=fm.X("date", type="temporal", title="Date"),
    y=fm.Y("value", type="quantitative"),
    color=fm.Color("category", type="nominal"),
)
```

## Marks

Most Altair marks map directly. `mark_circle` and `mark_square` are aliases for `mark_point` with a shape override.

| Altair | Ferrum | Notes |
|---|---|---|
| `mark_point()` | [`.mark_point()`][ferrum.Chart.mark_point] | — |
| `mark_circle()` | [`.mark_circle()`][ferrum.Chart.mark_circle] | Alias for `mark_point(shape="circle")` |
| `mark_square()` | [`.mark_square()`][ferrum.Chart.mark_square] | Alias for `mark_point(shape="square")` |
| `mark_line()` | [`.mark_line()`][ferrum.Chart.mark_line] | — |
| `mark_area()` | [`.mark_area()`][ferrum.Chart.mark_area] | — |
| `mark_bar()` | [`.mark_bar()`][ferrum.Chart.mark_bar] | — |
| `mark_rect()` | [`.mark_rect()`][ferrum.Chart.mark_rect] | — |
| `mark_rule()` | [`.mark_rule()`][ferrum.Chart.mark_rule] | — |
| `mark_text()` | [`.mark_text()`][ferrum.Chart.mark_text] | — |
| `mark_tick()` | [`.mark_tick()`][ferrum.Chart.mark_tick] | — |
| `mark_image()` | [`.mark_image()`][ferrum.Chart.mark_image] | — |
| `mark_geoshape()` | — | Not yet implemented |
| — | [`.mark_smooth()`][ferrum.Chart.mark_smooth] | No Altair equivalent — computes LOESS/LM in Rust |
| — | [`.mark_histogram()`][ferrum.Chart.mark_histogram] | Altair uses `mark_bar()` + `transform_bin()` |
| — | [`.mark_density()`][ferrum.Chart.mark_density] | Altair uses `mark_area()` + `transform_density()` |
| — | [`.mark_boxplot()`][ferrum.Chart.mark_boxplot] | Altair uses `mark_boxplot()` (composite) |
| — | [`.mark_violin()`][ferrum.Chart.mark_violin] | No Altair equivalent |
| — | [`.mark_contour()`][ferrum.Chart.mark_contour] | No Altair equivalent |
| — | [`.mark_hex()`][ferrum.Chart.mark_hex] | No Altair equivalent |

## Transforms

In Altair, transforms are chained on the chart: `chart.transform_filter(...)`. In Ferrum, transforms are passed to `.transform()` as objects:

<!--pytest.mark.skip-->
```python
# Altair
alt.Chart(df).mark_point().encode(x="x:Q", y="y:Q").transform_filter(
    alt.datum.value > 0
)

# Ferrum
fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").transform(
    fm.Filter("value > 0")
)
```

Common transform equivalents:

| Altair | Ferrum |
|---|---|
| `.transform_filter(predicate)` | `.transform(fm.Filter("expr"))` |
| `.transform_calculate(field, expr)` | `.transform(fm.Calculate(field="name", expr="expr"))` |
| `.transform_fold(fields, as_=["key","value"])` | `.transform(fm.Fold(fields=["a","b"]))` |
| `.transform_aggregate(groupby=, ...)` | `.transform(fm.Aggregate(groupby=["col"], ops=[...]))` |
| `.transform_bin("binned_x", field="x")` | Mark-level: `.mark_histogram(bin_count=20)` |
| `.transform_density("value")` | Mark-level: `.mark_density()` |
| `.transform_regression("y", on="x")` | Mark-level: `.mark_smooth(method="lm")` |
| `.transform_loess("y", on="x")` | Mark-level: `.mark_smooth(method="loess")` |
| `.transform_window(...)` | `.transform(fm.Window(...))` |
| `.transform_sample(n)` | `.transform(fm.Sample(n=n))` |
| `.transform_flatten(fields)` | `.transform(fm.Flatten(fields=["col"]))` |

Statistical transforms (bin, density, regression, loess) are first-class marks in Ferrum — they compute in Rust at render time and do not require a separate transform step.

## Composition

Ferrum uses the same operators as Altair:

<!--pytest.mark.skip-->
```python
# Layering — same in both
layered = base + smooth

# Horizontal concat — same in both
side_by_side = chart_a | chart_b

# Vertical concat — same in both
stacked = chart_a & chart_b

# Named functions — same in both
fm.layer(base, smooth)
fm.hconcat(chart_a, chart_b)
fm.vconcat(chart_a, chart_b)
```

Repeat charts work similarly but with a Ferrum-specific object:

<!--pytest.mark.skip-->
```python
# Altair
alt.Chart(df).mark_point().encode(
    x=alt.X(alt.repeat("column"), type="quantitative"),
    y="mpg:Q",
).repeat(column=["hp", "weight", "displacement"])

# Ferrum
fm.Chart(df).mark_point().encode(
    x=fm.X(fm.Repeat("column"), type="quantitative"),
    y="mpg:Q",
).repeat(column=["hp", "weight", "displacement"])
```

## Selections and conditions

Altair's selection system is the area with the most API difference:

<!--pytest.mark.skip-->
```python
# Altair
brush = alt.selection_interval()
chart = alt.Chart(df).mark_point().encode(
    color=alt.condition(brush, "category:N", alt.value("lightgray")),
).add_params(brush)

# Ferrum
brush = fm.selection_interval()
chart = fm.Chart(df).mark_point().encode(
    color=brush.when("category:N").otherwise(fm.value("lightgray")),
).add_selection(brush)
```

| Altair | Ferrum |
|---|---|
| `alt.selection_interval()` | [`fm.selection_interval()`][ferrum.selection_interval] |
| `alt.selection_point()` | [`fm.selection_point()`][ferrum.selection_point] |
| `alt.condition(sel, if_true, if_false)` | `sel.when(if_true).otherwise(if_false)` |
| `.add_params(sel)` | `.add_selection(sel)` |

## Scales

Altair's `alt.Scale(...)` maps to typed scale objects in Ferrum:

<!--pytest.mark.skip-->
```python
# Altair
alt.Chart(df).mark_point().encode(
    x=alt.X("x:Q", scale=alt.Scale(type="log")),
    color=alt.Color("cat:N", scale=alt.Scale(scheme="tableau10")),
)

# Ferrum
fm.Chart(df).mark_point().encode(
    x=fm.X("x:Q", scale=fm.Scale(type="log")),
    color=fm.Color("cat:N", scale=fm.Scale(scheme="tableau10")),
)
```

| Altair `Scale` param | Ferrum equivalent |
|---|---|
| `type="log"` | `fm.Scale(type="log")` |
| `type="sqrt"` | `fm.Scale(type="sqrt")` |
| `domain=[0, 100]` | `fm.Scale(domain=[0, 100])` |
| `range=["red","blue"]` | `fm.Scale(range=["red","blue"])` |
| `zero=False` | `fm.Scale(zero=False)` |
| `scheme="tableau10"` | `fm.Scale(scheme="tableau10")` |
| `clamp=True` | `fm.Scale(clamp=True)` |

## Chart properties

`.properties()` works identically:

<!--pytest.mark.skip-->
```python
# Altair — same API in Ferrum
chart.properties(
    width=400,
    height=300,
    title="My chart",
)
```

Title objects with subtitles also work the same way:

<!--pytest.mark.skip-->
```python
# Altair
chart.properties(title=alt.Title("Main", subtitle="Sub"))

# Ferrum
chart.properties(title=fm.Title("Main", subtitle="Sub"))
```

## Saving and export

`.save()` supports the same formats:

<!--pytest.mark.skip-->
```python
chart.save("plot.svg")   # SVG (default)
chart.save("plot.png")   # PNG (rasterized by Rust)
chart.save("plot.html")  # Interactive HTML
chart.save("plot.pdf")   # PDF
```

Serialization also mirrors Altair:

<!--pytest.mark.skip-->
```python
chart.to_json()   # JSON string
chart.to_dict()   # Python dict
```

## Key differences summary

| Feature | Altair | Ferrum |
|---|---|---|
| Rendering backend | Vega-Lite / JavaScript | Rust (native SVG/PNG/PDF) |
| Interactive output | Vega-Lite in Jupyter | WASM GPU renderer via anywidget |
| Statistical transforms | `.transform_*()` on chart | First-class marks (`.mark_smooth()`, `.mark_density()`, etc.) |
| Selection conditions | `alt.condition(sel, a, b)` | `sel.when(a).otherwise(b)` |
| Adding selections | `.add_params(sel)` | `.add_selection(sel)` |
| Repeat shorthand | `alt.repeat("column")` | `fm.Repeat("column")` |
| Title objects | `alt.Title(...)` | `fm.Title(...)` |
| Geographic marks | `mark_geoshape()` supported | Not yet implemented |
| Model diagnostics | — | 44 figure helpers, 28 visualizer classes |
| DataFrame support | pandas, polars (via `from_dict`) | polars, pandas, modin, cuDF, dask, ibis, pyarrow |
| System dependencies | Node.js for `.save()` on some formats | None — pure wheel |

## Where to go next

- [Marks & encodings](../guide/marks-encodings.md) for the full mark and channel reference.
- [Data transforms](../guide/data-transforms.md) for the complete transform list.
- [Composition](../guide/composition.md) for the `+`, `|`, `&` operators and repeat charts.
- [Interactive rendering](../guide/interactive.md) for the WASM renderer and selections.
- [Themes](../guide/themes.md) for Ferrum's twelve built-in themes.
- [Gallery](../gallery/gallery.md) for rendered examples with code.
