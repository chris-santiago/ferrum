# Interactive rendering

This page covers the interactive rendering API: selections, conditional encodings, zoom/pan, linked views, and saving interactive output. The interactive behavior requires a Jupyter notebook with `anywidget` installed; the code patterns on this page work in any Python environment, but the visual interactivity only activates in a live Jupyter session.

For the design rationale — why interactivity is a renderer, not a rewrite — see the [Interactivity concept page](concepts/interactivity.md).

Interactive rendering is `anywidget`-based and works in JupyterLab, VS Code notebooks, Google Colab, and classic Jupyter Notebook. Embedding in Streamlit, Dash, or Panel is not currently supported.

## Setup

Install the interactive extras:

```bash
pip install ferrum-viz[jupyter]
```

This adds `anywidget` and `ipywidgets` as dependencies. The WASM GPU renderer is bundled inside the `ferrum` wheel — no separate download.

## Switching to interactive mode

Any chart becomes interactive by calling [`.interactive()`][ferrum.Chart.interactive]:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 1, 5, 3]})
chart = fm.Chart(df).mark_point().encode(x="x", y="y")

# In a Jupyter cell, this renders as a GPU-backed canvas with zoom/pan:
chart.interactive()
```

The chart object is unchanged — `.interactive()` switches the render target from SVG to a WASM canvas widget. The same chart still works with `.show_svg()`, `.save("out.svg")`, and every other static render path. Selections and zoom/pan are silently ignored in static output.

## Selections

Selections define interactive state: "which marks did the user click?" or "what region did the user brush?" They are declared in the chart spec and resolved by the renderer.

### Point selections

A point selection activates when the user clicks a mark. Use [`selection_point`][ferrum.selection_point]:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [2, 4, 1, 5, 3],
    "group": ["a", "b", "a", "b", "a"],
})

sel = fm.selection_point(fields=["group"])
chart = (
    fm.Chart(df)
    .mark_point(size=100)
    .encode(x="x", y="y", color="group:N")
    .add_selection(sel)
    .interactive()
)
```

Clicking a mark selects all marks that share the same `group` value. Shift-click toggles additional selections (controlled by `toggle="event.shiftKey"`, the default). Use [`selection_single`][ferrum.selection_single] to disable toggling, or [`selection_multi`][ferrum.selection_multi] for explicit multi-select.

Key parameters:

| Parameter | Default | Effect |
|---|---|---|
| `fields` | `None` | Capture these field values on click; marks with matching values are selected. |
| `encodings` | `None` | Alternatively, trigger on encoding channel values (e.g. `["x", "color"]`). |
| `nearest` | `False` | Snap to the nearest mark instead of requiring an exact click. |
| `on` | `"click"` | Trigger event — `"click"`, `"mouseover"`, `"dblclick"`. |
| `clear` | `"mouseout"` | Event that clears the selection. |
| `resolve` | `"global"` | How multi-panel selections are resolved: `"global"`, `"union"`, `"intersect"`. |

### Interval selections

An interval selection activates when the user drags a rectangular brush. Use [`selection_interval`][ferrum.selection_interval]:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "x": [1, 2, 3, 4, 5, 6, 7, 8],
    "y": [2, 4, 1, 5, 3, 6, 2, 4],
})

brush = fm.selection_interval()
chart = (
    fm.Chart(df)
    .mark_point()
    .encode(x="x", y="y")
    .add_selection(brush)
    .interactive()
)
```

Dragging on the canvas creates a rectangular brush. Marks inside the brush are selected; marks outside are not. The brush can be panned (`translate=True`) and zoomed with the mousewheel (`zoom=True`).

To style the brush rectangle, pass a [`SelectionMark`][ferrum.SelectionMark]:

```python
import ferrum as fm

brush = fm.selection_interval(
    mark=fm.SelectionMark(fill="#3388cc", fill_opacity=0.2, stroke="#3388cc"),
)
```

## Conditional encodings

Selections become useful when they drive visual feedback. A **conditional encoding** changes a channel's value based on whether marks are selected.

The pattern is: `sel.when(if_selected).otherwise(if_not)`.

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [2, 4, 1, 5, 3],
    "species": ["setosa", "versicolor", "setosa", "versicolor", "setosa"],
})

sel = fm.selection_point(fields=["species"])
chart = (
    fm.Chart(df)
    .mark_point(size=100)
    .encode(x="x", y="y", color="species:N")
    .add_selection(sel)
    .conditional(sel.when(fm.Color("species")).otherwise(fm.value("#cccccc")))
    .interactive()
)
```

Clicking a point colors all marks of the same species; unselected marks turn grey. The [`value`][ferrum.value] wrapper marks a literal (a hex color, an opacity float) for use in the conditional.

You can also apply conditionals to `opacity` and `size`:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({"x": [1, 2, 3], "y": [3, 1, 2], "g": ["a", "b", "a"]})
sel = fm.selection_point(fields=["g"])
chart = (
    fm.Chart(df)
    .mark_point(size=100)
    .encode(x="x", y="y")
    .add_selection(sel)
    .conditional(sel.when(fm.Opacity("g")).otherwise(fm.value(0.2)))
    .interactive()
)
```

## Zoom and pan

The interactive renderer supports mousewheel zoom and click-drag pan on the canvas. These are controlled by the chart's coordinate system — no extra declaration is needed beyond calling `.interactive()`.

Zooming recomputes the visible domain and re-renders the scene with updated axis ticks and labels. The chart data is not resampled — the renderer draws the full dataset within the visible window.

## Linked views

Because selections live in the chart spec and composition operators pass the spec through, linked views fall out of composition with no extra API.

Declare a selection with `fields` in one chart, reference the same selection object in another, and compose them:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "x": [1, 2, 3, 4, 5, 6, 7, 8],
    "y": [2, 4, 1, 5, 3, 6, 2, 4],
    "category": ["a", "b", "a", "b", "a", "b", "a", "b"],
})

sel = fm.selection_point(fields=["category"])

scatter = (
    fm.Chart(df)
    .mark_point(size=80)
    .encode(x="x", y="y", color="category:N")
    .add_selection(sel)
    .conditional(sel.when(fm.Color("category")).otherwise(fm.value("#cccccc")))
)

bars = (
    fm.Chart(df)
    .mark_bar()
    .transform(fm.transform_aggregate(
        {"field": "category", "fn": "count", "as": "n"}, groupby=["category"]
    ))
    .encode(x="category:N", y="n:Q", color="category:N")
    .add_selection(sel)
    .conditional(sel.when(fm.Color("category")).otherwise(fm.value("#cccccc")))
)

linked = scatter | bars
```

Clicking a point in the scatter selects all marks that share the same `category` value — in both panels. The `|` operator composes both charts into a side-by-side view — the same operator used for static concatenation, no separate "link API." The `fields=["category"]` parameter tells the selection to match marks by field value rather than by data index, which is what enables cross-panel linking even when the two charts have different data shapes (the bar chart uses an aggregate transform).

## Listening for selections in Python

For programmatic responses to user interaction, register a callback with [`on_selection_change`][ferrum._interactive.InteractiveChart.on_selection_change]:

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({"x": [1, 2, 3], "y": [3, 1, 2], "label": ["a", "b", "c"]})
sel = fm.selection_point(fields=["label"])
interactive = (
    fm.Chart(df)
    .mark_point(size=100)
    .encode(x="x", y="y")
    .add_selection(sel)
    .interactive()
)

def handle(state):
    print(f"Selected: {state}")

interactive.on_selection_change(handle)
interactive  # display in Jupyter
```

The callback receives the current selection state as a dict. In Jupyter, output from the callback appears below the chart widget (via an `ipywidgets.Output` area that clears on each new selection).

## Saving interactive output

Save an interactive chart as a self-contained HTML file:

```python
chart.interactive().save("dashboard.html")
```

The HTML file inlines the WASM renderer and the scene data — no external dependencies, no server required. Open it in any modern browser.

## Performance at scale

The interactive renderer uses two optimizations that keep large charts responsive:

- **Binary instance bridge** — GPU mark data bypasses JSON serialization and is sent as a packed binary buffer. This eliminates the deserialization bottleneck that would otherwise make million-point interactive charts impractical.
- **Packed tooltips** — field-level tooltip content is transferred via a binary buffer rather than per-mark JSON objects. Tooltip lookups use a spatial hit-test (`hitTestAt`) that resolves to the nearest mark's data index.

These are transparent — you don't need to opt in. A 1M-point scatter with tooltips uses the same `.interactive()` call as a 100-point chart.

## Static fallback

A chart with selections renders normally in static output — `.show_svg()`, `.save("plot.png")`, `.save("plot.svg")` all work. Selections, conditional encodings, and zoom/pan are silently ignored. This means you can build one chart that serves both a notebook dashboard (interactive) and a report figure (static) without maintaining two specs.

## Where to go next

- [Interactivity is a renderer](concepts/interactivity.md) for the design rationale behind this approach.
- [Composition](composition.md) for the operators that enable linked views.
- [API Reference — ferrum.selection](../api/selection.md) for the full signatures of selection constructors and `SelectionMark`.
- [Marks & encodings](marks-encodings.md) for the `tooltip`, `href`, and `key` channels that interact with the renderer.
