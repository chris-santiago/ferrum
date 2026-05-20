# Plotnine / Altair Migration Gap Audit

**Date:** 2026-05-20
**Scope:** Ferrum API compared against plotnine and Altair — identifying bugs, blockers, friction points, and missing features that users migrating from those libraries would encounter.
**Method:** Four parallel exploration agents covering data input & encoding, marks & transforms, scales & themes & composition, and error messages & ergonomics. Findings deduplicated and ranked by severity.

---

## BUGS (silently wrong behavior — fix immediately)

| # | Issue | Severity | Detail |
|---|-------|----------|--------|
| **B1** | `type_=` kwarg silently dropped on channel constructors | **CRITICAL** | Docstrings teach `fm.X("hp", type_="Q")` but `to_encoding_spec_dict()` reads `self._kwargs.get("type")` — the trailing underscore form is accepted, warned as "not yet honored", then ignored. `encoding/base.py:78`, docstrings in `encoding/positional.py:39,62` |
| **B2** | `count():Q` aggregate shorthand crashes | **CRITICAL** | `encode(y="count():Q")` → parser returns `(None, "Q", "count")` → `AggregateOp(field="")` → Rust rejects empty field. The most common altair shorthand is broken. `encoding/base.py:133`, `transform/aggregate.rs` |
| **B3** | Aggregate shorthand missing auto-groupby | **HIGH** | `encode(x="cat:N", y="mean(val):Q")` → implicit Aggregate has `groupby=[]` → after agg, `cat` column is gone → `ValueError: unknown column 'cat'`. Altair auto-infers groupby from sibling channels. `encoding/base.py:128-134`, `chart.py:3729-3757` |
| **B4** | `nice=True` on Scale constructors silently dropped | **HIGH** | All typed Scale classes accept `nice=True` in the constructor, but `_scale_to_dict()` never serializes it. The key never reaches Rust. `encoding/_scale.py` (all branches) |
| **B5** | `mark_smooth(method="linear")` documented but crashes | **HIGH** | Docstring at `chart.py:966` lists `"linear"`, `"quadratic"`, `"cubic"`, `"log"`, `"sqrt"` but the Rust `Smooth` transform only accepts `"lm"` and `"loess"`. Users following the docstring hit a render-time error. `smooth.rs:858-862` |
| **B6** | Non-Int64 integers fail when encoded as nominal | **HIGH** | PyArrow tables with Int32/Int16/UInt8 columns encoded as `:N` crash: `distinct_values_in_order` only handles Utf8, LargeUtf8, Int64, Boolean. Polars users unaffected (always Int64), but pandas/pyarrow users hit this. `arrow_cast.rs:174-201` |
| **B7** | Config defaults (640×480) don't match render defaults (600×400) | **MEDIUM** | `config.py:22-23` defines 640×480 but `_render.py:190` hard-codes `600.0, 400.0`. The config values are never read at render time. |

---

## BLOCKERS (user cannot accomplish common task)

| # | Issue | Affects | Detail |
|---|-------|---------|--------|
| **K1** | `mark_point(color="red")` raises TypeError | All users | `color` not in `_VALID_MARK_KWARGS`. Both altair and plotnine users expect `color=` to work. Must use `fill=` or `stroke=`. `marks/base.py:12-56` |
| **K2** | `mark_point(alpha=0.5)` raises TypeError | plotnine users | `alpha` not in valid kwargs. Must use `opacity=`. `marks/base.py:12-56` |
| **K3** | `mark_line(point=True)` raises TypeError | altair users | `point` not in valid kwargs. Must manually layer `chart.mark_line() + chart.mark_point()`. `marks/base.py:12-56` |
| **K4** | Bars don't auto-stack with color encoding | plotnine users | Plotnine defaults to `position="stack"` for grouped bars. Ferrum requires explicit `position=fm.Stack()`. |
| **K5** | pandas/polars Series rejected as data | plotnine users | `fm.Chart(pd.Series(...))` → TypeError. Series not detected before narwhals fallback. `_coerce.py:129-146` |
| **K6** | Duration/Timedelta columns crash at render | all users | Passes coercion but Rust can't read Duration values. `arrow_cast.rs:60-83` |
| **K7** | PyArrow Date32 columns crash at render | pyarrow users | `col_as_f64` has no Date32/Date64 arms. Polars users unaffected (pre-casts). `arrow_cast.rs:60-83` |

---

## FRICTION (achievable but non-obvious path)

| # | Issue | Affects | Detail |
|---|-------|---------|--------|
| **F1** | No `labs()` for post-hoc axis labels | plotnine users | Must set at encode time: `encode(x=fm.X("col", title="Label"))`. No fluent `.labs(x=..., y=...)` method. |
| **F2** | No standalone `xlim()`/`ylim()` | plotnine users | Requires `chart.coord(fm.CoordCartesian(xlim=(...), ylim=(...)))`. |
| **F3** | `linetype="dashed"` not supported | plotnine users | Must use `stroke_dash="4,2"`. No plotnine-style string names. |
| **F4** | No `mark_circle()` / `mark_square()` | altair users | Must use `mark_point(shape="circle")`. AttributeError on `mark_circle()`. |
| **F5** | `cornerRadius` (camelCase) rejected | altair users | Must use `corner_radius`. All Vega-Lite camelCase forms raise TypeError. |
| **F6** | Transform chaining syntax differs | altair users | Altair: `chart.transform_filter(...)`. Ferrum: `chart.transform(fm.transform_filter(...))`. Module-level functions, not chart methods. |
| **F7** | `fm.value()` only works in conditionals | altair users | Can't do `encode(color=fm.value("red"))` as constant encoding. Must use `mark_point(fill="red")` instead. |
| **F8** | `.interactive()` must be last in chain | altair users | Returns `InteractiveChart`, not `Chart`. Can't chain `.encode()` after it. |
| **F9** | Unknown-column error at render time lacks available-columns list | all users | Error says `unknown column 'xyz'` but doesn't say which columns exist. Validation deferred to render, not at `encode()`. `arrow_cast.rs:52`, `chart.py:3729` |
| **F10** | Unknown-channel error doesn't list valid channels | all users | `encode(z="field")` → `ValueError: unknown encoding channel: 'z'` but doesn't say what IS valid. `chart.py:3732` |
| **F11** | `facet(column="x")` → TypeError | altair users | Ferrum uses `col=`, not `column=`. |
| **F12** | `Scale(zero=False)` not on typed classes | altair users | Only works via raw dict `scale={"type": "linear", "zero": False}`. No `zero=` param on `LinearScale`. |
| **F13** | No DPI/scale control for PNG | all users | Fixed 2x retina. No `dpi=` or `scale=` param on `save()` or `show_png()`. |
| **F14** | No PDF export | all users | `.save("plot.pdf")` → ValueError. |
| **F15** | No `.to_dict()` on Chart | altair users | Must do `json.loads(chart.to_json())`. |
| **F16** | No `reverse=` on continuous positional scales | all users | Must manually reverse domain. |
| **F17** | No categorical label remapping | plotnine users | No `labels={"a": "Group A"}` on scales/axes. Must rename in DataFrame. |
| **F18** | `nudge_x`/`nudge_y` not on mark_text | plotnine users | Ferrum has `dx`/`dy` (pixel offsets, not data-space). |

---

## MISSING (feature gap, workaround or none)

| # | Feature | Workaround |
|---|---------|------------|
| **M1** | No `transform_lookup` (join secondary data) | Pre-join in polars/pandas |
| **M2** | Only 6 aggregate functions (mean/sum/count/min/max/median) | Altair has 15+ (variance, stdev, q1, q3, etc.) |
| **M3** | `mark_density(kernel=...)` only supports `"gaussian"` | Other kernels documented but raise ValueError |
| **M4** | No `mark_trail` (variable-width line) | No direct equivalent |
| **M5** | No `geom_abline` (slope+intercept line) | Use `mark_function()` |
| **M6** | No `mark_polygon` as public API | Internal only |
| **M7** | URL string not accepted as data source | Fetch manually |
| **M8** | No `alt.datum` keyword for constant-position encoding | Use data workarounds |
| **M9** | No Altair migration doc (`docs/site/comparison/altair.md`) | Most important missing doc — ferrum's API is modeled on Altair |

---

## DOCUMENTATION

| # | Issue |
|---|-------|
| **D1** | **No `altair.md` migration guide** — the highest-priority doc gap since ferrum's API shape mirrors Altair most closely |
| **D2** | `mark_smooth` docstring lists wrong method names (`"linear"` vs `"lm"`) |
| **D3** | `mark_tick` docstring says `band_size` is "Tick length in pixels" — it's actually a fraction of band width |

---

## TOP 5 HIGHEST-IMPACT FIXES

1. **B2 + B3**: `count():Q` and `mean(field):Q` shorthands — these are the bread and butter of altair. Broken aggregate shorthand + missing auto-groupby makes the most common altair bar chart pattern fail.
2. **B1**: `type_="Q"` silently dropped — the docstrings actively teach users the broken form.
3. **K1 + K2**: `color=` and `alpha=` not accepted as mark kwargs — every plotnine user will try these first.
4. **B5**: `mark_smooth(method="linear")` documented but crashes — docstring/Rust mismatch.
5. **D1**: Missing altair migration doc.

---

## ALREADY FIXED (this session, 2026-05-20)

These items were identified and fixed before this audit was run:

| Fix | Branch |
|-----|--------|
| Shorthand parser rejects hyphens/dots/spaces in column names | `fix/shorthand-hyphen-parsing` |
| Polars Categorical/Enum columns rejected by Rust renderer | `fix/shorthand-hyphen-parsing` |
| CSS named colors rejected in theme overrides (hex-only) | `fix/shorthand-hyphen-parsing` |
| `mark_tick` renders nothing with ordinal-y-only encoding | `fix/shorthand-hyphen-parsing` |
| Composite mark kwargs (errorbar, boxplot, etc.) silently dropped | `fix/shorthand-hyphen-parsing` |
| No `"|"` / `"-"` point shapes (plotnine `shape="|"` equivalent) | `fix/shorthand-hyphen-parsing` |
