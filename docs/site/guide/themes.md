# Themes

A theme is a bundle of style decisions — background color, mark color, point size, line stroke, grid behavior, padding — that applies uniformly across the marks in a chart. Themes are values, not state: you construct a `Theme` and pass it where you want it. Nothing mutates global module state, and no chart is "secretly" themed by an import side-effect.

Three layers of theme control let you reach for the right scope for each situation: per-chart, process-default, and scoped via a context manager. Per-chart always wins.

## Built-in themes

Ferrum ships eight built-in themes accessible as attributes of `ferrum.themes`:

| Name | Purpose |
|---|---|
| `default` | Ferrum's standard look — white background, perceptually uniform palettes, sensible grid. |
| `minimal` | Stripped-down: no grid, tighter margins, smaller axis decoration. |
| `dark` | Dark background for slides, dashboards, terminal-style outputs. |
| `publication` | Optimized for print: high contrast, no grid, larger font weights. |
| `economist` | An *Economist*-style editorial look. |
| `fivethirtyeight` | A *FiveThirtyEight* editorial look. |
| `solarized_light` | Solarized light palette. |
| `solarized_dark` | Solarized dark palette. |

Apply one to a chart:

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
    .theme(fm.themes.dark)
)
assert chart.show_svg().startswith("<svg")
```

The chart spec carries the theme alongside its encoding and mark — saving the spec preserves the theme. Switching themes is one method call away; the rest of the chart spec is untouched.

## Theme as a value

The `Theme` class is immutable. Once constructed, it cannot be mutated. Methods like `.update()` return a new `Theme` rather than modifying the original:

```python
import ferrum as fm

base = fm.Theme(background_color="#f9f9f9", grid=True, padding=16)
darker = base.update(background_color="#222222", mark_color="#e74c3c")
assert base != darker
assert base.to_theme_inputs_dict()["background_color"] == "#f9f9f9"
assert darker.to_theme_inputs_dict()["background_color"] == "#222222"
```

This immutability is structural: themes are values that compose, like encodings and marks. You can keep a "base theme" in a module and derive variants without worrying about side-effects. Use `Theme.to_theme_inputs_dict()` to inspect a theme's contents — `Theme` does not expose its properties as attributes directly.

### Wired keys vs. round-trip keys

`Theme` accepts arbitrary keyword arguments. Some are wired to the Rust renderer today; others are stored and round-tripped through `update()` / `to_theme_inputs_dict()` but are not yet consumed at render time. Keys reserved for future phases include `font_color`, `title_color`, `color_scheme`, `font_family`, `axis_line`, `grid_color` — see the `Theme` API reference for the live list.

The keys currently honored by the renderer:

- `background_color` — CSS hex string for the chart background.
- `mark_color` — default fill/stroke for marks with no explicit `color` encoding.
- `point_size` — default radius for point marks.
- `line_stroke_width` — default stroke width for line marks.
- `bar_corner_radius` — corner radius applied to bars.
- `area_opacity` — default opacity for area marks.
- `grid` — whether to draw grid lines.
- `padding` — chart padding in pixels (applied to all four sides).

## Process-default themes

When you want the same theme to apply to every chart in a notebook, a script, or an analysis session, set it once with `set_default_theme()`:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
fm.set_default_theme(fm.themes.dark)
chart = (
    fm.Chart(iris)
    .mark_point()
    .encode(x="sepal_length", y="petal_length", color="species:N")
)
assert chart.show_svg().startswith("<svg")
fm.set_default_theme(fm.themes.default)  # reset for downstream cells
```

`set_default_theme()` does not mutate a module-level config object. It writes to a per-thread `contextvars.ContextVar`, which means:

- The default is scoped to the current Python interpreter context.
- Concurrent tasks (asyncio, multiprocessing, threads using contextvars) do not share defaults.
- The previous default can be restored explicitly by calling `set_default_theme()` again with the old theme, or implicitly via the context manager pattern (next section).

This is the single documented exception to Ferrum's "no global mutable state" rule, and the mechanism is deliberately scope-bounded.

## Scoped themes via `with`

For analysis code where the theme should change for one block and revert afterward, `set_default_theme()` returns a context manager. `theme_context()` is a clearer-named alias for the same behavior.

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
with fm.theme_context(fm.themes.publication):
    publication_chart = (
        fm.Chart(iris)
        .mark_point()
        .encode(x="sepal_length", y="petal_length", color="species:N")
    )
    assert publication_chart.show_svg().startswith("<svg")
# Outside the with-block the previous default is restored.
assert fm.get_default_theme() == fm.themes.default
```

The previous default is restored automatically on `__exit__`. This is the right scope when you want a different theme for a single section of a notebook, a single figure-rendering function, or a test fixture — without the rest of the session inheriting it.

`get_default_theme()` returns the currently active process default, useful for debugging or for code that needs to inspect what would apply to a freshly-constructed chart.

## Precedence

When multiple theme sources are in play, the resolution order is:

1. **Per-chart `.theme(t)` always wins.** If a chart calls `.theme(dark)`, that chart renders with `dark` regardless of any process default or context.
2. **Process default applies otherwise.** A chart without an explicit `.theme()` picks up whatever `set_default_theme()` last set (or whatever `theme_context()` currently scopes).
3. **Ferrum's built-in `default` is the bottom of the stack.** If nothing else is set, every chart renders with `themes.default`.

You should think of `.theme()` as a chart-level override and `set_default_theme()` / `theme_context()` as ambient defaults. The chart-level override is always available as an escape hatch.

## Building your own theme

A custom theme is just a `Theme(...)` call with the keys you want:

```python
import ferrum as fm
import polars as pl
from sklearn.datasets import load_iris

raw = load_iris()
iris = pl.DataFrame(raw.data, schema=["sepal_length", "sepal_width", "petal_length", "petal_width"]).with_columns(
    species=pl.Series([raw.target_names[t] for t in raw.target])
)
brand = fm.Theme(
    background_color="#ffffff",
    mark_color="#0d47a1",
    point_size=4.0,
    grid=False,
    padding=24,
)
chart = (
    fm.Chart(iris)
    .mark_point()
    .encode(x="sepal_length", y="petal_length")
    .theme(brand)
)
assert chart.show_svg().startswith("<svg")
```

Stored unknown keys round-trip through `Theme.update()` so they survive theme inheritance, even when not yet wired to the renderer. That lets you encode forward-looking style decisions today and pick them up when a future phase consumes them.

## Where to go next

- [Marks & encodings](marks-encodings.md) for what gets styled by a theme.
- [Composition](composition.md) for how themes apply across layered, concatenated, and joint compound views.
- [Figure-level helpers](figure-helpers.md) for the convenience entry points (most accept a `theme=` keyword).
- The [API Reference](../api/themes.md) for the full `Theme` constructor signature and the list of round-trip keys reserved for future phases.
