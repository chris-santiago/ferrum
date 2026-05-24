# Override

`.override(**kwargs)` is ferrum's escape hatch for the rare case where the typed
configuration surface hasn't caught up to what you need. It lets you inject
spec-path key/value pairs directly into the chart spec at render time.

```python
chart.override(x_axis_label_angle=-60, y_axis_tick_count=8)
```

---

## When to use override

Override is explicitly a **last resort**. Before reaching for it, check whether the
typed surface covers your need:

| Need | Typed surface |
|---|---|
| Rotate axis labels | `.configure_axis(label_angle=...)` |
| Change tick count | `.configure_axis(tick_count=...)` |
| Move legend | `.configure_legend(orient=...)` |
| Adjust title alignment | `.configure_title(anchor=...)` |
| Show/hide grid | `.configure_grid(x=..., y=...)` |
| Set padding | `.configure_padding(...)` |

Use override only when you need to set something that the typed surface doesn't yet
expose. Override should be a temporary measure; if you find yourself using it for the
same thing repeatedly, that's a signal that the typed surface has a gap worth reporting.

---

## Path conventions

Override paths use snake_case with underscores to separate the spec hierarchy:

```python
# axis settings
chart.override(
    x_axis_label_angle=-45,
    y_axis_tick_count=6,
    y_axis_grid_color="#eee",
)

# legend settings
chart.override(
    legend_orient="bottom",
)

# mark settings
chart.override(
    mark_opacity=0.8,
)
```

---

## Validation at render time

Unknown override paths raise `FerrumOverrideError` when the chart renders, not when
`.override()` is called. The error message includes the invalid path and a list of
closest matches:

```
FerrumOverrideError: Unknown override path 'x_axis_lable_angle'.
Did you mean: 'x_axis_label_angle'?
```

This means override calls are syntactically valid Python even with typos, but you will
see the error when you first render or save the chart.

---

## Deprecation warnings

Some override paths have typed equivalents in the current API. Using these paths emits a
`DeprecationWarning` at render time that points to the correct method:

```
DeprecationWarning: override path 'legend_orient' has a typed equivalent:
  .configure_legend(orient=...)
Use the typed method instead.
```

The override still applies even when the warning fires, so existing code continues to
work. Migrate to the typed method at your convenience.

---

## Multiple override calls

Multiple `.override()` calls merge their kwargs. Later calls win on key conflicts:

```python
chart = (
    fm.Chart(df)
    .mark_point()
    .encode(x="x:Q", y="y:Q")
    .override(x_axis_label_angle=-30)
    .override(x_axis_label_angle=-45)  # wins; applied angle is -45
)
```

This is consistent with how `configure_*()` layers stack — later always wins within the
same key.

---

## Override in the cascade

Override sits at the top of the six-level cascade: it beats per-channel `axis=`,
chart-level configure, themes, and Rust defaults.

```
1. chart.override(...)           ← wins everything
2. Per-channel axis= / legend=
3. chart.configure_*()
4. chart.theme(...)
5. set_default_theme(...)
6. Rust renderer defaults
```

A value set via override cannot be overridden by anything else. Use that power
judiciously.

---

## Example

```python
import ferrum as fm
import polars as pl

df = pl.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [2.1, 3.8, 3.2, 5.1, 4.9],
})

# This hypothetical path isn't yet in the typed surface
chart = (
    fm.Chart(df)
    .mark_point(size=80)
    .encode(x="x:Q", y="y:Q")
    .override(x_axis_label_angle=-30)  # uses typed method in practice; shown for illustration
)
```

---

## Reporting gaps

If you find yourself using `.override()` for something that should be in the typed
surface, please open an issue. The goal is that override is never needed for any common
customization.
