# Ferrum
## A Statistical Visualization Library — Concept & API Specification

> *Fast. Composable. Statistically honest.*

---

## Part I: Philosophy

### 1. Core Beliefs

#### Grammar first, convenience second
Every visualization — from a scatter plot to a SHAP beeswarm — is a composition of the same primitives: data, encodings, marks, scales, and coordinate systems. Ferrum commits to this without compromise. High-level convenience functions (`displot`, `lmplot`, `roc_chart`) exist, but they are sugar over the grammar, not parallel universes with different rules.

#### Model artifacts are data
A confusion matrix is a cross-tabulation. A ROC curve is a sorted transformation of predicted probabilities. A SHAP value is a column. There is no reason these should require a separate API or a different mental model. Ferrum treats model outputs as data sources that feed into the standard grammar. This eliminates the Yellowbrick problem: diagnostic plots are composable, themeable, and interactive because they are charts, not special objects.

#### Statistics belong in the rendering pipeline, not in userspace
Computing a KDE, bootstrapping a confidence interval, or fitting a LOESS curve should not require the user to call SciPy before plotting. These are stat transforms: first-class operations declared in the chart spec and executed in the Rust engine before rendering. The user declares intent; Ferrum computes.

> **2026-05-11 (Phase 10 — Model Diagnostics):** Phase 10 places model-diagnostic
> compute in the `ModelSource` adapter layer (Python, lazy-imported sklearn
> delegation) rather than as Rust transforms in the rendering pipeline. From the
> user's perspective the figure function (`ferrum.roc_chart`, etc.) *is* the
> rendering pipeline — they are not computing ROC in userspace, which is what
> this constraint actually proscribes. Whether the internal compute is a Rust
> transform or a Python call to sklearn is invisible at the call site.
> Model-diagnostic compute is also entangled with the model-specific protocol
> (`predict_proba`, `classes_`, etc.), which a generic Rust transform cannot
> access without reimplementing sklearn's estimator protocol. The one exception
> is `ferrum._core.kendall_tau_b` (Knight's O(n log n) tau-b), which runs as a
> Rust function called from `ModelSource.rank2d(algorithm="kendall")` — pure
> numeric work over two f64 arrays, no estimator protocol involved.

#### Interactivity is a renderer, not a rewrite
You should not need to learn a different API to make a chart interactive. `.interactive()` switches the render target from SVG to a WASM canvas. Selections, zoom, pan, and linked views are declared in the chart spec and handled by the renderer. Plotly's fatal flaw is that interactive charts and static charts are different objects. Ferrum has one chart object.

#### Zero unnecessary copies
Python is the declaration layer. Rust is the computation layer. Data moves between them once, over the Arrow C Data Interface (CDI). Stat transforms, layout, binning, and aggregation happen in Rust. The Python process never touches row-level data again after the initial handoff.

> **Amendment 2026-05-09:** Original spec said "Arrow IPC" (byte-serialized stream format). After design review for Phase 2, we chose the Arrow C Data Interface instead. Polars DataFrames implement `__arrow_c_stream__` natively — CDI passes the buffer pointer directly with zero copies, whereas IPC would serialize to bytes and deserialize on the Rust side. The `pyo3-arrow` crate mediates the CDI boundary in PyO3. The spirit of the constraint ("data moves once, no row-level Python access after handoff") is preserved; only the wire format changed.

#### Defaults should be correct, not just pretty
Default color schemes are perceptually uniform and colorblind-safe (OKabe-Ito for categorical, Viridis for sequential). Default font sizes pass WCAG contrast. Default bin counts follow Sturges' rule as a floor. These are not aesthetic opinions — they are epistemically correct starting points.

---

### 2. Design Constraints

- **No matplotlib dependency.** Ever. Not as a fallback, not for "legacy support."
- **No mutable global state.** No `rcParams`, no `set_theme()` that mutates a module-level object. Themes are values passed to charts.
- **No magic inference that silently fails.** If Ferrum infers a scale or encoding type incorrectly, it raises a descriptive error with a suggested fix.
- **Sklearn-compatible, not sklearn-dependent.** The model diagnostics layer works with any object that implements `predict`, `predict_proba`, or `transform`. Sklearn is not imported unless the user passes a fitted sklearn model (or uses an sklearn-backed extra such as SHAP). The raw-array path (`y_true=` / `y_pred=`) computes ROC, precision-recall, calibration, confusion-matrix, and threshold-sweep metrics on native Rust kernels with no sklearn dependency.
- **One output format per render call.** A chart produces SVG, PNG, HTML (WASM bundle), or a Vega-Lite JSON spec. Producing all four from the same spec is supported; producing ambiguous mixed output is not.

---

### 3. Relationship to Prior Art

| Library | What Ferrum Inherits | What Ferrum Rejects |
|---|---|---|
| **plotnine / ggplot2** | Grammar of Graphics layering, explicit scales, faceting primitives | matplotlib backend, R-centric defaults |
| **Altair** | Typed encoding channels, selection API, `\|` / `&` composition operators | Vega-Lite JSON as the primary data format (too slow at scale), 5000-row default limit |
| **Seaborn** | Statistical vocabulary, figure-level functions, automatic facet label handling | matplotlib coupling, inconsistent axes-level vs figure-level API surface |
| **Plotly** | Interactivity as a first-class feature, hover tooltips, WASM/HTML output | Separate static and interactive APIs, heavy JavaScript bundle for simple charts |
| **Yellowbrick** | sklearn-protocol visualizers, diagnostic vocabulary | matplotlib coupling, non-composable outputs, parallel API surface |
| **scikit-plot** | Lift/gain curves, calibration curve conventions | Function-level API with no composability |

---

### 4. Naming

The library is named **Ferrum** (Latin: iron). Iron is the substrate of Rust. It is also load-bearing, structural, and unglamorous — which is what a visualization engine should be.

The Python package imports as `ferrum`. Internal Rust crate: `ferrum-core`. WASM renderer: `ferrum-wasm`.

---

---

## Part II: Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Python Layer                        │
│  Chart spec builder · Encoding DSL · Convenience API   │
└────────────────────┬────────────────────────────────────┘
                     │  PyO3 + Arrow CDI (pyo3-arrow)
┌────────────────────▼────────────────────────────────────┐
│                    Rust Core                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Stat Engine │  │Layout Engine │  │ Scale Engine │  │
│  │  KDE · CI    │  │Constraint    │  │Domain/Range  │  │
│  │  Regression  │  │solver ·      │  │mapping ·     │  │
│  │  Binning     │  │Facet layout  │  │Tick gen      │  │
│  │  Aggregation │  │Legend place  │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                     │
  ┌──────────────────────────────────────────────────┐
  │               Render Pipeline                    │
  │                                                  │
  │   Geometry pass → SceneGraph                     │
  │                       │                          │
  │          ┌────────────┼────────────┐             │
  │          ▼            ▼            ▼             │
  │      SVG backend  tiny-skia    wgpu/WASM         │
  │      (vector)     (CPU raster) (GPU/interactive) │
  └──────────────────────────────────────────────────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
    SVG/PNG      HTML+WASM     JSON spec
   (static)    (interactive)  (interop)
```

SceneGraph is the renderer-agnostic intermediate representation — a flat list of geometric primitives (filled paths, stroked paths, rectangles, circles, glyphs, images) consumed by all three backends.

---

---

## Part III: API Specification

---

### 3.1 Top-Level Entry Points

#### `ferrum.Chart`
The primary chart constructor. All single-view charts begin here.

```
Chart(data=None, *, width=None, height=None, title=None, description=None)
```

**Parameters**
- `data` — Any of: `pandas.DataFrame`, `polars.DataFrame`, `polars.LazyFrame`, Arrow `Table`, `ModelSource`, or `None` (data supplied per-layer).
- `width` / `height` — Integer pixels, or `"container"` for responsive sizing.
- `title` — String or `Title` object.
- `description` — Accessible alt-text string; included in SVG and HTML output.

**Methods** (all return `self` unless noted)
- `.encode(**channels)` — Bind encoding channels. Channels inherited by all layers unless overridden.
- `.layer(*layers)` — Add one or more `Layer` objects.
- `.mark_*(...)` — Shorthand: creates a single-layer chart with the given mark. Equivalent to `.layer(Layer(mark=mark_*(...)))`.
- `.transform(*transforms)` — Add data transforms applied before any stat computation.
- `.stat(*stats)` — Add stat transforms (computed per-layer if applied at layer level).
- `.facet(field=None, *, row=None, col=None, ncols=None, nrows=None, title=None)` — Wrap chart in a `FacetSpec`.
- `.add_selection(*selections)` — Bind `Selection` objects.
- `.conditional(selection, **true_encodings, **false_encodings)` — Conditional encoding based on selection state.
- `.properties(**kwargs)` — Set chart-level metadata (title, width, height, description).
- `.theme(theme)` — Apply a `Theme` object.
- `.interactive(*, toolbar: bool = True)` — Switch render target to WASM. Returns `InteractiveChart`.
- `.resolve(scale=None, legend=None)` — Override shared/independent resolution in compound views (see §3.9; `axis=` narrowed out 2026-07-12, tracked as a follow-up).
- `.save(path, *, format=None, scale=2.0, backend=None, **render_kwargs)` — Render and write to disk. `format` inferred from extension if `None`. `backend` overrides auto-selection. `scale` applies to raster backends only.

> **2026-05-22 (feat/rtree-toolbar — Toolbar & Auto-tooltips):**
>
> `.interactive(*, toolbar: bool = True)` — the `toolbar` kwarg (default `True`)
> renders a Bokeh-style toolbar overlaid on the canvas. Five tools are included:
> Pan (keyboard shortcut P), Box Zoom (Z), Box Select (S), Reset (R), and Save PNG.
> The default active tool depends on whether the chart has an `selection_interval`
> declared: if so, Box Select is active on load; otherwise Pan is active. Pass
> `toolbar=False` to suppress the toolbar entirely (useful for embedded or
> space-constrained contexts).
>
> **Auto-tooltips** — when rendering interactively, hover tooltips are
> automatically injected from the chart's encoded channels (x, y, color, size,
> etc.). No explicit `tooltip=` encoding is needed for basic hover. If an explicit
> `tooltip=` encoding is present it takes precedence and auto-injection is skipped.
> Auto-tooltips only affect interactive renders; SVG and PNG output are unchanged.
>
> `.save(path, *, ..., toolbar: bool = True)` — for HTML format, `toolbar`
> controls whether the toolbar appears in the exported file. Defaults to `True`.
> Has no effect on SVG/PNG/JSON output.

> **2026-07-11 (secondary y-axis, #52 close-out):** auto-tooltips on a
> **layered** chart (`LayerChart`, or any chart with `+`-appended layers
> such as `SecondaryY`) now report each hovered layer's *own* encoded
> channels, not the chart's first/primary layer's channels for every
> layer. This was a pre-existing gap in the auto-tooltip injection path
> (`_inject_auto_tooltips` and the Rust `MetadataColumns::from_ctx` builder
> were both chart-level-only, collecting one shared field set) that
> surfaced while building per-layer independent y-scales, and the fix is a
> root cause fix applying to **every** layered chart, not just dual-axis
> ones — hovering a non-primary layer's marks now always shows that
> layer's fields. An explicit chart-level `tooltip=` encoding still wins
> for every layer (unchanged, takes precedence over auto-injection as
> documented above). One documented non-regression: a field-based linked
> selection on a layered chart still short-circuits per-layer
> auto-tooltip injection for the selection-driven tooltip content (the
> chart-level selection-tooltip path returns early before the per-layer
> walk) — not a regression, just a narrower auto-tooltip scope than the
> unconditional per-layer case.
- `.show()` — Display in current environment (notebook, terminal sixel, or browser).
- `.to_spec()` — Return the internal `ChartSpec` dataclass (serializable).
- `.to_json()` — Return JSON string of the chart spec.
- `__or__(other)` → `HConcatChart`
- `__and__(other)` → `VConcatChart`

---

#### `ferrum.Layer`

Explicit layer constructor. Used when multiple layers need independent data, marks, or encodings.

```
Layer(data=None, mark=None, *, encoding=None, stat=None, transform=None)
```

Most users use `mark_*()` methods on `Chart` instead of constructing `Layer` directly.

> **2026-09-02 (batch A):** hoisted paint no longer leaks into a sibling
> layer. Python's `LayerChart` lowering copies layer-0's mark kwargs up to
> chart level (a compatibility fallback for chart-level rendering paths),
> but every layer still keeps its own copy — a layer with no `mark_style` of
> its own now strips that inherited paint (`fill`/`stroke` cleared) rather
> than rendering in it, regardless of mark type. `mark_bar(fill="red") +
> mark_point()` now renders the point layer in its own default color, not
> red; `mark_text(fill="red") + mark_bar()` no longer renders the bars red.
> A flat (no-layers) chart is unaffected, and a layer's own paint — when it
> sets one — always wins.

> **2026-09-23 (wave 6, `#119`, Breaking):** a layer's style is its own — the batch-A paint-only rule above is now the general rule for every non-paint `mark_style` field too. Python's `Chart.to_spec()` no longer copies layer-0's mark kwargs up to a layered chart's chart-level `mark_style` at all (the compatibility fallback the batch-A note describes is retired outright, not just stripped of paint); Rust's per-layer fallback reads `layer.mark_style` alone and `MarkKwargsSpec::without_paint` is deleted, so the fallback it fed is gone, not merely unreachable. A kwarg-less sibling layer now renders identically to its flat self on both the static and interactive paths, and across `+`, `LayerChart`, independent-y, `SecondaryY`, and a desugar that adds a prior layer (`.mark_smooth()`, and `.mark_function()`'s own multi-layer append). Six shapes, measured:
>
> | Chart | Before | After |
> |---|---|---|
> | `mark_point(opacity=0.3, stroke_width=5) + mark_line()` | line at 30% opacity, width 5 | line identical to a flat `mark_line()` |
> | `mark_function(f, clip=False) + mark_point()` | points escape the panel clip | points clipped |
> | `mark_line().encode(detail="g") + mark_line()` | layer 1 split by layer 0's column | one polyline per its own grouping |
> | `mark_density(orientation="horizontal") + mark_area()` | area closes to the x baseline | closes to its own baseline |
> | `mark_point(opacity=.3, stroke_dash=[4,2]).mark_smooth()` | fit line translucent and dashed | fit line default |
> | `LayerChart(a, b)` | static clean, interactive leaks | both clean, identical |
>
> A hand-built layered `ChartSpec` that sets a chart-level `mark_style` stops applying it to a kwarg-less layer — the wire no longer carries a top-level `mark_style` on any layered spec Python produces, and Rust has no reader left for one even on a spec built some other way. Unchanged: a flat (no-layers) chart, whose chart-level `mark_style` still applies in full; a layer that sets its own kwargs, which already replaced the chart-level value wholesale before this change and still does; and a composite-mark or kwarg-less LHS, from which chart-level `mark_style` was never emitted in the first place (`_mark_kwargs` stays empty there), so there was nothing to gate.

> **2026-09-18 (wave 4, #201-D):** a layered chart is judged by the marks it
> actually draws, and a positional axis may bind from the layer that binds
> it. Two rules, each now stated once in `render/scale_resolve/`:
>
> 1. **Eligibility for the single-axis exemption** (the carve-out that lets
>    `mark_tick`/`mark_rule` render against one axis) is decided by *every
>    layer that paints*, not by the composed spec's root `mark` field.
>    Lowering a `(tick + line)` composition leaves `mark="tick"` at the root
>    as a placeholder, so the old root-mark test granted the exemption to a
>    chart whose line layer genuinely needs both axes, and the resolver
>    synthesized a `0…1` dummy y scale. Measured on `(tick_chart +
>    line_chart)` — a rug under a density curve — the curve was squashed into
>    the foot of the panel: polyline y extent `(368.316, 416.911)` of a 480px
>    canvas under y tick labels `1, 0.9, … 0`. It now renders at
>    `(24.0, 412.961)` under the density column's own units
>    (`0.13, 0.12, … 0.01`).
> 2. **Binding**: an axis takes the chart encoding's binding when there is
>    one, otherwise the first layer eligible to supply it (`independent_y`
>    layers excluded for `y`, every layer encoding normalized out of authored
>    space first). A chart-level or layer-0 binding always wins, so the
>    change is bounded to specs whose merged chart+layer-0 encoding leaves a
>    positional channel unbound. Specs that resolve through the generic
>    two-axis path are untouched; **specs that resolved through the
>    single-axis exemption with an axis unbound DO move, by design** — that
>    is the defect being fixed, and the `(tick + line)` geometry above is an
>    instance of it. This fallback yields a **scale**, not a plottable
>    field — see the refusal below, and the axis-title consequence in the
>    next paragraph.
>
> **The value axis of a bare layering is deliberately untitled.** A
> `Chart(df).mark_tick(...) + Chart(df).mark_line(...)` composition now
> renders on the line layer's own scale, but with **no y-axis title**, and
> that is the intended contract rather than an omission. An axis title is a
> claim about what the chart is *about*, and the only thing that makes such
> a claim is a binding the caller wrote — chart-level or on layer 0. The
> fallback above is not that: it resolves a scale so a sibling layer can be
> placed, which is precisely why a drawing layer reaching a channel *only*
> through it is refused rather than drawn. Titling from the same fallback
> would have the renderer assert, on the strength of whichever layer bound
> the channel first, something no caller said — and would move rendered text
> on compositions that already render correctly. **The caller's spelling for
> a title is to make the binding real, and it must go where a binding is
> read**: `.encode(y=fm.Y("density", title="Density"))` on the composition's
> own chart encoding, or on the **first** chart of the pair (layer 0). A
> `title=` on any later layer is ignored, because that is exactly the
> fallback binding rule 2 above describes as not title-bearing. Measured on
> `tick + line`, titling the density column six ways:
>
> | Composition | Rendered y-axis title |
> |---|---|
> | `tick_plain + line_plain` | *(none)* |
> | `tick_titled + line_plain` | `Density` |
> | `line_titled + tick_plain` | `Density` |
> | `tick_plain + line_titled` | *(none)* |
> | `line_plain + tick_titled` | `d` — layer 0's own field name; the later layer's `title=` is dropped |
> | `(tick_plain + line_plain).encode(y=fm.Y("d", title="Density"))` | `Density` |
>
> Note the natural-reading trap in row 4: the density column is the *line's*
> column, so titling the line chart is the obvious move, and in `tick + line`
> the line is the second layer, so it does nothing. Put the title on the
> first chart or on the composition. `displot` reaches the chart encoding
> itself, which is why `displot(kind="rug", kde=True)` *is* titled (see
> #201-B in §3.14) while the hand-composed equivalent is not.
>
> **New typed refusal** — `RenderError::LayerMissingPositionalChannel`,
> surfacing as a `ValueError`. A layer that paints and whose mark needs both
> positional channels must reach each one from its own encoding or from the
> chart encoding it inherits; reaching it only through the fallback above is
> now refused instead of silently drawing nothing:
>
> ```
> layer 0: mark_point needs both positional channels, but no 'y' encoding
> reaches it — a sibling layer binds 'y', which resolves the axis but leaves
> mark_point with nothing to plot. Encode 'y' on this layer, or on the
> chart, to draw it.
> ```
>
> The guard runs *after* generic x/y resolution, so a channel bound nowhere
> at all keeps raising the existing channel-level `EncodingTypeMismatch`.
> This **closes 12 silent drops that predate the wave entirely**. Measured by
> an exhaustive sweep of all 144 ordered two-layer pairs over
> `{point, line, tick, rule} × {x}, {y}, {x,y}`, composed through the public
> `+` API and rendered: the pre-wave tree rendered 84, refused 48 and
> **silently dropped 12** (`tick{x} + point{x}` and `rule{y} + point{y}`
> among them); the shipped tree renders 88, refuses 56 and drops **0**. The
> transition census over the same 144 is `rendered→rendered 84`,
> `refused→refused 48`, `dropped→refused 8`, `dropped→rendered 4`, and
> **zero compositions that the pre-wave tree rendered correctly are now
> refused**. The sweep ships as a test
> (`tests/test_single_axis_eligibility.py`), so the class stays closed by
> construction — but note what that test does and does not check: it asserts
> the **zero-drop property alone** — every one of the 144 pairs either
> refuses or draws its full complement of marks, counted for all four mark
> kinds inside the marks group — not the 88/56 split and not the transition
> census. Counting, rather than testing for presence, is what makes the
> assertion cover `tick` and `rule`: both render as `<line>`, which is also
> what every axis tick and gridline renders as, and every chart carries a
> panel `<rect>` whether or not anything drew — so a presence test on either
> element matches chrome and passes for a dropped layer. Scoping the search
> to the marks group (which carries no chrome) and comparing counts closes
> that, including for a pair whose two marks share an element type.
> The shipped `0` is therefore re-runnable on any
> checkout; the pre-wave `84/48/12` row and the census come from runs
> against reverted builds and cannot be reproduced from this tree. (Scope
> bound: the closure claim covers this 4-mark × 3-channel × 2-layer space;
> 3+ layer stacks, other marks and polar compositions are outside it.)

> **2026-09-18 (wave 4, T10b):** a bar with no extent no longer reaches the
> SVG. All four `SceneNode::Rect` push sites in the bar renderer
> (`build_ordinal`, `build_ordinal_y`, `build_quantitative`,
> `build_quantitative_horizontal`) now skip a row whose computed width or
> height is exactly `0.0`, through one shared `is_zero_extent` check. The
> visible output is unchanged — a zero-extent rect painted nothing — but the
> **element is gone from the document**, so code that counts rects sees one
> fewer per empty group-bin. Concretely, a grouped/stacked histogram with an
> empty cell used to emit e.g.
> `<rect x="65.31" y="392.489" width="76.534" height="0" fill="#dc2626"/>`
> for the bin where one group has no rows; it now emits nothing there. The
> `displot_stacked_hist` golden drops from 14 rects to 13. The check is
> exact-zero rather than epsilon on purpose: these extents are always
> `(a - b).abs()` or `.max(0.0)` over two scale-lookup pixel values, so an
> empty-group zero is exactly `0.0` to the bit, and an epsilon would also
> swallow a genuinely tiny-but-nonzero bar.

> **2026-09-18 (wave 4 close, R1 — an all-zero-extent bar batch says why it
> drew nothing):** the guard above skips the push site, and nothing
> downstream could tell that silence from a chart that legitimately had
> nothing to draw. The four rect builders now count their zero-extent drops
> and set the batch's `empty_reason` through one shared helper, which fires
> **only** when the guard emptied a batch that had rows to draw (every row
> dropped, none emitted). A batch already empty for a documented reason —
> zero input rows, null categories, non-finite values, unresolvable
> pixels — never reaches the guard and keeps the `None` reserved for a
> legitimate empty, so the channel stays quiet. Measured on
> `importance_chart` over an all-zero importance column:
>
> ```
> bar layer 0 drew nothing: mark_bar() computed a zero width or height for
> all 4 of its rects (every value equal to its own baseline, or an empty
> group in every bin); a zero-extent fill paints nothing, so none were
> emitted
> ```
>
> The non-zero control emits no warning (measured: the same chart with
> importances `0.4/0.3/0.2/0.1` renders 4 data rects and zero warnings,
> against 0 data rects and this one warning for the all-zero case).
> **SVG bytes are unchanged everywhere** — this adds a diagnosis, not a
> rendering — but it is a new `UserWarning`, so a consumer running under
> `-W error` will now see an exception where it previously saw a silent
> blank panel.

> **2026-09-18 (wave 4 close, R6 — a dodged span mark subdivides its band):**
> `apply_dodge`'s continuous branch offset only the band column (`x` /
> `bin_start`) and left the span's other end (`x2` / `bin_end`) untouched, so
> each group's rect became `[bin_start + offset, bin_end]` — a width of
> `bin_width ∓ offset` instead of a share of the bin. Both endpoints now move
> together into the group's own sub-slot, using `apply_dodge_ordinal`'s
> existing convention (padding off each end of the band, `n_groups` slots laid
> end to end from the leading edge), and the row's **own** span is subdivided
> rather than a median-spacing bandwidth proxy, so a non-uniform final bin
> keeps its slots inside itself. The point-row path (no paired channel) is
> untouched and byte-identical.
>
> This also closes a **silent drop**, not just a mis-sized bar: a negative
> slot offset could push the leftmost `bin_start` below the resolved
> x-domain minimum, where the unclamped linear scale returns `NaN`, the
> pixel resolves to `None`, and the bar builder skips the row entirely — a
> missing bar. Two further notes: the `uniques.len() < 2` bail (which exists
> because median spacing is undefined for a single band value, a proxy the
> span path does not need) no longer applies to span rows, so a **one-bin
> dodged histogram now dodges instead of overlapping**; and the span branch
> preserves nulls rather than coercing a null band value to `0.0` as the
> point branch still does, because a row with a null endpoint is skipped by
> the drawer today and would otherwise have started drawing a rect back to
> the axis origin.
>
> **Radius: every dodged span mark, not only `displot`.** Any chart whose
> band axis carries a paired channel under `Dodge` now draws sub-slots
> instead of offset full-width spans. No golden moved, but a downstream
> chart built on the old overlapping geometry will look different —
> correctly so.

> **Dated note (2026-09-11, wave-1 hardening, #198 — `Chart.__add__` preserves each layer's own `data_source`):** `a + b`'s two rewrite sites (the column-collision branch, and the named-transform-routing branch that promotes an unnamed RHS transform) used to unconditionally overwrite every RHS layer's `data_source` with the merge's own auto-generated name — harmless for a plain single-mark chart (whose one layer starts with `data_source=None`), but every multi-output composite mark (`mark_qq`, `mark_violin`, `mark_boxplot`, …) names its own top-level transform and points each layer's `data_source` at it, so the clobber stranded a layer pointing at a name nothing publishes. Measured: `point + qq` raised `unknown column 'qq_line_x_start'`; `qq + point` happened to work only because `__add__` takes a different branch when the LHS already carries `_transforms`. Both sites now preserve a layer's own `data_source` when it is already set, substituting the merge's name only when a layer's `data_source` is `None` — so a multi-output composite composes via `+` in **either order**. A second, previously-silent defect surfaced by the same code path is now a typed refusal instead of silent data corruption: two DIFFERENT composite-mark charts that happen to name their top-level transform identically (every `mark_qq()` names its transform `"qq_main"` regardless of field) used to compose without error and silently let one side's data overwrite the other's at render time (measured: 100 rendered circles collapsing to only 50 distinct positions). `Chart.__add__` now raises `ValueError` naming the specific layer and the named transform output both sides would collide on, before the merge happens — a self-add (`chart + chart`) and two structurally-identical transforms (`qq(field="y") + qq(field="y")`) are exempted, since a transform that would produce identical output on both sides isn't a real collision.

---

#### `ferrum.ModelSource`

Wraps a fitted estimator and a dataset; exposes derived data sources as Arrow tables.

```
ModelSource(model, X=None, y=None, *, feature_names=None, class_names=None,
            sample_weight=None, random_state=None)
```

> **2026-05-11 (Phase 10 — Model Diagnostics):** `random_state: int | None = None`
> was added as the sixth keyword-only constructor argument. It is propagated to
> every derived-data method whose underlying sklearn / shap / umap call accepts
> an RNG seed (importances permutation, SHAP background sampling, UMAP / t-SNE
> embeddings, MDS / t-SNE cluster-center projection, learning_curve / validation_curve /
> cv_scores / alpha_selection cross-validation, partial_dependence sampling).
> Methods whose compute is deterministic by construction (predictions, residuals,
> roc_curve, pr_curve, confusion_matrix, calibration_curve, cumulative_gain,
> lift_curve, discrimination_threshold, pca_variance, rank1d/rank2d) ignore it.

> **2026-09-13 (wave 3 T4, #211/N3 — correcting a false claim):** the
> `discrimination_threshold` entry above, and the matching
> `DiscriminationThresholdVisualizer` entry in §3.15's `random_state` note,
> were **false** for `cv=<int>` between 2026-05-11 and this fix: the hand-rolled
> splitter was `KFold(n_splits=cv, shuffle=True, random_state=self._random_state
> or 0)`, which genuinely consumed `random_state` (it was reproducible only by
> the accident of the `or 0` default when `random_state` was left unset).
> `discrimination_threshold(cv=<int>)` now resolves via sklearn's own
> `check_cv(cv, y, classifier=True)` convention — an **unshuffled**
> `StratifiedKFold` — which has no seed to thread, making both claims true
> again. This is a disclosed, number-moving change for every integer-`cv=`
> call (stratification changes which rows land in which fold); no golden
> exercises `cv=`. A user-supplied splitter's `.split()` now also receives
> `y` (previously `.split(X)` with no `y`, which raised for a stratified
> splitter). A **train** fold left with only one class (rare under
> stratification, not impossible at very small class counts) now emits a
> `UserWarning` naming the fold and the missing class before the clone-and-fit
> step fails; a degenerate **test** fold is unaffected — it is a routine
> property of the split, not a bad request, and still silently contributes
> `precision = recall = f1 = 0` with a real `queue_rate` (unchanged since the
> 2026-09-11 close review, cycle 4).

> **2026-09-10 (batch G, F-L12-01):** `X` is now optional
> (`ModelSource(model, X=None, y=None, ...)`; back-compatible — every existing
> positional/keyword call site is unaffected). Diagnostics computed purely
> from model attributes work with no `X` at all — `pca_variance()` when the
> model exposes `explained_variance_ratio_`, and `intercluster_distance()`.
> Every other derived-data method raises `ValueError` naming the diagnostic
> and the fix when called on an X-less source —
> `"ModelSource.<method>() requires the training data: ModelSource(model, X)."`
> — instead of the prior `TypeError: __init__() missing 1 required positional
> argument: 'X'` (constructing directly) or a leaked
> `Chart(data=None) requires per-layer data...` error (routed through a
> figure function's own `X=None` default, e.g. `silhouette_chart(model)`).
>
> A public, read-only `ModelSource.class_names -> list[str] | None` property
> is added alongside `.X`/`.y`/`.feature_names`, proxied on
> `ComparedModelSource`. Resolution is two arms only — the constructor's
> `class_names=` if set, else `model.classes_` (stringified) if the model
> exposes it, else `None`. It deliberately does **not** attempt a third,
> positional `class_{i}` fallback: that arm needs a definite class count (the
> width of a computed prediction matrix), which isn't knowable from
> constructor state alone without running inference. The internal
> `class_{i}` fallback that `probabilities()`/`confusion_matrix()`/etc.
> complete after inference is unaffected and unrelated to this property.

**Methods** (all return `polars.DataFrame` unless noted)
- `.predictions()` — `y_true`, `y_pred`, `residual`, `studentized_residual`
- `.probabilities()` — `y_true`, one column per class with predicted probability
- `.roc_curve(*, average=None, drop_intermediate=True)` — `fpr`, `tpr`, `threshold`, `class`, `auc`
- `.pr_curve(*, average=None)` — `precision`, `recall`, `threshold`, `class`, `ap`
- `.confusion_matrix(*, normalize=None)` — `actual`, `predicted`, `value`, `value_fmt`
- `.calibration_curve(*, n_bins=10, strategy="uniform")` — `mean_predicted`, `fraction_positive`, `count`, `brier`
- `.cumulative_gain()` — `percent_population`, `gain`, `class`
- `.lift_curve()` — `percent_population`, `lift`, `class`
- `.importances(*, method="builtin", n_repeats=30, scoring=None, random_state=None)` — `feature`, `importance`, `std`, `rank`
- `.shap_values(*, background=None, max_evals=500)` — `sample_id`, `feature`, `shap_value`, `feature_value`, `feature_value_normalized`, `class_label`

> **2026-05-12 (P3.11, D15) — the second sentence is SUPERSEDED for multi-output regressors by the 2026-09-13 note below:** `shap_values()` always includes a `class_label`
> column. Regression sources emit `class_label="target"` on every row.
> Binary classifiers emit the positive-class name (single value, schema
> unchanged in shape). Multi-class classifiers emit one row per
> `(sample, feature, class)` — total height `n_samples * n_features * n_classes`.
> See `mark_shap_*(per_class=...)` for the chart-side overlay hook.

> **2026-09-13 (wave 3, #209 / D2 — supersedes the sentence above that reads
> "Regression sources emit `class_label="target"` on every row", which is no
> longer true for a *multi-output* regressor):** the `class_label` vocabulary
> now has one owner (`_output_class_labels`), shared with
> `.partial_dependence()` so the two frames cannot label the same model
> differently, and that owner gained the case neither caller previously
> answered. A single-output regressor is unchanged and still emits `"target"`
> on every row; a model with no `classes_` but several outputs — a native
> multi-output regressor, e.g. `RandomForestRegressor` fit on a 2-column `y` —
> now emits `target_0 ... target_{n-1}` instead of repeating a bare `"target"`.
> **Breaking (number mover) on the shap figure functions:** `class_label` is
> what `shap_beeswarm_chart`/`shap_bar_chart`/`shap_waterfall_chart` (and the
> deprecated `shap_chart` dispatcher) select and facet on, so those outputs
> move for this model shape. With a repeated `"target"` there was nothing to select, and
> `per_class=False` averaged two unrelated outputs' SHAP values together;
> it now selects the first output's rows. Measured on 120 rows, three
> standard-normal features `a`/`b`/`c` and
> `y = [3a + 0.2c, 4b + 0.2c]` through
> `RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)`:
> `shap_values()`'s `class_label` uniques go `["target"]` → `["target_0",
> "target_1"]` at an unchanged height of 720; the `per_class=False` selection
> goes 720 rows → 360; and `shap_bar_chart(source)`'s `abs_mean_shap` goes
> `a` 1.181774 → 2.011961 and `b` 1.626124 → 0.268149 (`c` 0.0 on both), so
> the rendered bar order flips from `b, a, c` to `a, b, c`. `per_class=True`
> on the same source now facets into one panel per output where it previously
> rendered a single unfaceted panel (`_should_facet_by_class` needs more than
> one class). Classifiers of every arity, and single-output regressors, are
> unaffected.
- `.partial_dependence(features, *, grid_resolution=100, kind="average")` — `feature`, `feature_value`, `pd_value`, `class_label`, `sample_id`

> **2026-09-13 (wave 3, #209 / D2 — multiclass partial dependence emits per-class
> rows):** `.partial_dependence()` always includes a `class_label` column naming
> the model output each row belongs to, using exactly `shap_values()`'
> vocabulary above (one shared owner, `_output_class_labels`). Regression sources
> emit `class_label="target"`; binary classifiers emit the positive-class name
> (sklearn's partial dependence reports the positive class only, so the rows are
> unchanged). **Breaking (schema + number mover):** a K-class classifier now
> emits one row group per class instead of silently returning class 0's curve
> alone — for a 3-class model with 2 features at `grid_resolution=10`,
> `kind="average"` measured 20 rows before and 60 after, and `kind="individual"`
> on 300 samples at `grid_resolution=6` measured 1800 rows before and 5400 after.
> Both `kind` fan-outs apply. The per-class values are sklearn's own per-output
> results (`partial_dependence(...)["average"][c]` / `["individual"][c]`),
> previously discarded for every `c > 0`. Single-output frames keep their rows
> and row order byte-identical; only the new column is added. A multi-output
> regressor (no `classes_`, several outputs) numbers its outputs `target_0 ...
> target_{n-1}` rather than repeating a bare `"target"`, since `class_label` is
> the downstream curve-grouping key. Chart side: `pdp_chart` keeps one facet
> panel per feature and binds `class_label` to the color channel when the frame
> carries more than one class, so each panel shows K legended curves; a
> single-class frame keeps the color-free rendering it had (existing PDP goldens
> are byte-identical). Grouping keys that identify one curve — the `center=`
> anchor, the `kind="both"` average join, the ICE `detail` pairing — all carry
> `class_label`, so curves cannot chain or cross-join between classes.
- `.silhouette(k)` — `sample_id`, `cluster`, `silhouette_value`
- `.embeddings(*, method="umap", n_components=2, **method_kwargs)` — `dim_0`, `dim_1`, (`dim_2`), `label`
- `.learning_curve(*, cv=5, scoring=None, train_sizes=None)` — `train_size`, `split`, `score`, `mean_score`, `std_score`, `lower`, `upper`
- `.validation_curve(param, values, *, cv=5, scoring=None)` — `param_value`, `split`, `score`, `mean_score`, `lower`, `upper`
- `.discrimination_threshold(*, n_thresholds=50, cv=None)` — `threshold`, `precision`, `recall`, `f1`, `queue_rate`. Binary classifiers only. Sweeps the decision threshold and records precision, recall, F1, and queue rate at each point. `cv`: if provided, averages over cross-validation folds; an int resolves to sklearn's `check_cv(cv, y, classifier=True)` — an unshuffled `StratifiedKFold` — and a splitter object is called `.split(X, y)` (see the 2026-09-13 dated note above).
- `.pca_variance(*, n_components=None)` — `component`, `explained_variance_ratio`, `cumulative_variance_ratio`. Model must expose `explained_variance_ratio_` (e.g. sklearn PCA, TruncatedSVD). `n_components`: limit output to first n components.
- `.rank1d(*, algorithm="shapiro")` — `feature`, `score`, `rank`. Univariate feature ranking. `algorithm`: `"shapiro"` (Shapiro-Wilk normality score) | `"variance"` | `"covariance"` (covariance with target y).
- `.rank2d(*, algorithm="pearson")` — `feature_x`, `feature_y`, `correlation`. Pairwise feature correlation matrix. `algorithm`: `"pearson"` | `"spearman"` | `"kendall"` | `"covariance"`.
- `.cv_scores(*, cv=5, scoring=None)` — `fold`, `split`, `score`. Cross-validation score per fold. `split`: `"train"` | `"test"`. `scoring`: sklearn scorer string or callable; defaults to model's default scorer.
- `.alpha_selection(alphas, *, cv=5, scoring=None)` — `alpha`, `fold`, `score`, `mean_score`, `std_score`. CV score at each regularization parameter value. `alphas`: array-like of alpha values to evaluate. Intended for Ridge, Lasso, ElasticNet, and other regularized estimators.
- `.intercluster_distance(k, *, method="mds")` — `cluster`, `x`, `y`, `size`. MDS or t-SNE projection of cluster centers, with `size` = membership count. `k`: number of clusters. `method`: `"mds"` | `"tsne"`.

**Class method**
- `ModelSource.compare(models: dict[str, estimator], X, y, **kwargs)` → `ComparedModelSource`
  - All derived methods return the same schemas with an additional `model` column.

> **2026-09-10 (batch G, F-L12-08 — `.partial_dependence`/`.importances` stop
> leaking sklearn feature-name warnings):** when the model exposes
> `feature_names_in_` (i.e. it was fit on a named frame) and pandas is
> importable, `.partial_dependence(...)` and `.importances(method=
> "permutation")` now hand sklearn a pandas DataFrame rebuilt from the
> numpy feature matrix with those exact names, so sklearn receives what it
> was fit on and emits zero `UserWarning`s; numeric output is identical to
> the numpy path to machine precision. A model without `feature_names_in_`
> is unaffected (the numpy path, which never warned). Provenance — whether
> the caller's `X` genuinely carries no real names, vs. merely inherited
> synthetic ones — is captured once, structurally, at `ModelSource`
> construction time (a bare `numpy.ndarray`, or a pandas `DataFrame` whose
> `.columns` is a default `RangeIndex`, are both detected as nameless; a
> pandas frame with a real column index, or any polars `DataFrame`, is
> genuinely named), not guessed later from column-name strings. **Disclosed
> residual:** a bare-array **polars** `DataFrame` the caller builds directly
> with no `schema=` (`pl.DataFrame(array)`, which auto-generates
> `column_0, column_1, ...` labels) is treated as genuinely named — polars
> exposes no structural marker distinguishing this from a deliberately
> user-named frame — so passing one to a source whose model was fit under
> different real names now raises sklearn's own feature-name-mismatch
> `ValueError` where it previously computed a silently positionally-correct
> result with a warning. This is a deliberate trade (a loud, actionable
> error beats a silent guess), and it affects only this one specific input
> shape: a bare `numpy.ndarray` and `pd.DataFrame(array)` — the two most
> common ways to hand over a nameless feature matrix — are both detected
> structurally and compute correctly with zero warnings.

> **2026-09-10 (batch G, F-L12-02 — t-SNE perplexity clamp corrected):**
> `.intercluster_distance(k, method="tsne")`'s perplexity clamp is
> `min(5, k - 2)`, not `min(5, k - 1)` as previously documented — t-SNE's
> kNN size for `k` embedded cluster centers is `k - 1` (each center's
> `k - 1` co-embedded peers), and perplexity must be strictly less than
> that. Under the old (wrong) clamp, `k` in `{4, 5, 6}` raised
> `tsne_embedding failed: perplexity (n) must be strictly less than the kNN
> size (n)` — every documented-supported `k` except 7+. Every `k >= 4` now
> renders; `k < 4` keeps its existing typed refusal, unchanged.

> **2026-09-13 (wave 3, #222 + N4 — `.embeddings(method="tsne")` perplexity
> parity):** the same physical constraint `.intercluster_distance`'s clamp
> encodes for `k` cluster centers — manifolds-rs (Barnes-Hut t-SNE) requires
> `perplexity < kNN size`, which composes in the small-`n` regime to
> `perplexity <= n_samples - 2` — now also governs `.embeddings(method=
> "tsne")`'s user-visible `perplexity`. **Breaking (number mover):**
> the default (no `perplexity=` supplied) now resolves to
> `min(30, n_samples - 2)` instead of forwarding `None` straight to
> manifolds-rs's own unclamped default of 30; measured before/after: every
> call with `n_samples <= 31` previously raised
> `tsne_embedding failed: perplexity (30) must be strictly less than the
> kNN size (n_samples - 1)` and now renders (e.g. `n_samples=20` resolves
> to `perplexity=18` and renders; `n_samples=200` is unaffected, since
> `min(30, 198) == 30`, its pre-existing value). This is the majority
> failure mode — it fired on `manifold_chart` and `ManifoldVisualizer` too,
> since neither exposes a `perplexity` parameter of its own and both always
> hit the default path. An explicit `perplexity=` is validated, never
> silently reduced: non-numeric, non-finite (NaN/±inf), non-positive, or
> greater than `n_samples - 2` all raise a typed `ValueError` naming
> `perplexity`, `n_samples`, and the bound — replacing both the raw
> manifolds-rs `"kNN size"` message (an internal derived quantity no
> ferrum parameter names) and a `PanicException` a negative or NaN value
> previously produced at the PyO3 boundary (`k_float.to_usize().unwrap()`
> has no valid `usize` for those inputs), which no `except ValueError`
> could catch. Unrelated to the perplexity fix but closed in the same
> pass: an unknown `method_kwargs` key now refuses by name, checked
> per-`method` (not the union of all three methods' keys, which would
> accept `perplexity` on `"umap"` and silently drop it) — previously every
> read was `method_kwargs.get(...)`, so a typo (or a key belonging to a
> different method) reached nothing and the embedding was silently
> computed with defaults. No golden added or moved: t-SNE/UMAP output has
> no golden by deliberate design (manifolds-rs's RNG is not the
> byte-deterministic `ChaCha8Rng` this repo's other transforms use, so no
> t-SNE/UMAP output is cross-platform-stable enough to pin — see
> `design-docs/architecture/ARCHITECTURE.md`'s "Randomness" section);
> within one process/platform/toolchain, `random_state=` determinism is
> unaffected by this change.

> **2026-09-10 (batch G, F-L12-03 — `rank1d(algorithm="covariance")` raw
> routing):** `rank1d_chart(df, y=y, algorithm="covariance")` on a raw
> DataFrame/2-D array now renders — previously it unconditionally routed
> through `ModelSource(model=df, X=None, y=y)` (treating the raw frame as a
> fitted estimator to wrap), which always hit the `X`-required typed
> refusal. Raw tabular input (`polars.DataFrame`, a 2-D `numpy.ndarray`, or
> a `pandas.DataFrame`) now routes to the same Rust covariance entry the
> `Rank1DVisualizer` raw-input path already used; a fitted estimator input
> (`ModelSource`, or a bare fitted model) is unaffected and keeps its exact
> prior code path. `algorithm="covariance"` still requires `y`.

> **Dated note (2026-09-11, wave-1 hardening, #204 — `rank1d(algorithm="covariance")` refuses a length mismatch, drops null-`y` rows instead of zero-imputing them):** `y` whose length doesn't match `X`'s row count now raises a typed `ValueError` naming the mismatch (mirroring the existing `hat_matrix_stats` guard's wording) instead of panicking with an out-of-bounds index inside the Rust covariance kernel — the check runs uniformly before the `algorithm` dispatch, even though only `"covariance"` reads `y`. A null `y` slot — previously read straight off the Arrow value buffer's default `0.0` and folded into the covariance as a real zero-valued observation, corrupting every feature's score — now drops out of the computation entirely (a validity-aware read, not a zero-impute); both Python callers (`rank1d_compute_with_y`, `RankingMixin.rank1d`) now warn naming the algorithm and the dropped row count. An all-null `y` degrades to n=0, which the kernel's existing degenerate-input guards already handle without panicking.
>
> **Dated note (2026-09-11, wave-1 hardening, #216 — `rank1d`/`rank2d` refuse an X-less fitted estimator on every algorithm):** `rank1d_chart`/`rank2d_chart` previously only probed "is this raw tabular data or a fitted estimator" inside the `algorithm="covariance"` branch of `rank1d_chart`, and not at all in `rank2d_chart` — so `rank1d_chart(model, y=y, algorithm="shapiro"/"variance")` and EVERY `rank2d_chart(model)` call on an X-less fitted estimator leaked a raw `TypeError` from `np.asarray()` instead of the typed `ModelSource` refusal ("...requires the training data: `ModelSource(model, X)`.") the covariance branch already had. The raw/model probe is now hoisted to the top of both dispatch functions, routing every algorithm through the same typed refusal; `rank2d_chart(model, df)` (with X) is unaffected.
>
> **Breaking (2026-09-13, wave-3 diagnostics, N1 — a multi-chunk `pyarrow.Table` was silently truncated to its first chunk):** `polars_or_array_to_record_batch` (the raw-data coercer behind `rank1d_chart`/`rank2d_chart`/`Rank1DVisualizer`/`Rank2DVisualizer`) took `X.to_batches()[0]` for a `pyarrow.Table` input, discarding every chunk after the first. A real `pa.Table` is routinely multi-chunk — `pa.concat_tables`, a multi-file `parquet.read_table`, or a `RecordBatchReader` drain all produce one — so every raw-data rank statistic on such a Table was silently computed over a fraction of the rows while `ModelSource`'s own path (which already routes through `ferrum._coerce.to_polars`) computed the correct answer on the same input: measured on a 200-row regression fixture split into two 100-row chunks, `rank1d_chart(chunked, algorithm="variance")`'s top score for column `b` read **1.169347** (the first 100-row chunk alone) against the correct **1.018316** (all 200 rows) on the equivalent single-chunk Table and the `ModelSource` path. The covariance path did not compute a silently wrong number; it refused with a misleading `"y length must match X row count"`, because the coercer silently halved X's row count while the caller's `y` kept its full length. The pyarrow arm now routes through `ferrum._coerce.to_polars` (the same canonical Arrow-CDI seam `ModelSource` already uses), so every chunk is consumed and the raw-data path, the `ModelSource` path, and a single-chunk Table of the same data all agree. A secondary, smaller instance of the same defect is also closed: an empty 2-column `pyarrow.Table` used to return zero features (the old empty-Table branch, `pa.RecordBatch.from_pydict({})`, discarded the schema along with the rows), where an empty 2-column `polars.DataFrame` correctly returned 2 features scored 0.0 — both spellings of "no rows, two columns" now agree. `#217` itself (the covariance-refuses-on-`pa.Table` shape) was already closed by `545be19a`; this note documents the residual defect one layer below it, in the coercer `545be19a` did not touch.

> **Cache contract (2026-09-13, wave-3 diagnostics, #224b/N5 — `BaseSource._cache_key` can never return the wrong frame):** every `ModelSource` derived-data method caches its result under a key built from the method name plus its kwargs. `_cache_key` normalizes an `ndarray`/`list`/`tuple` kwarg value (including one nested inside another) into a nested tuple of its own values — content-preserving, no rounding, no truncation — before checking hashability. Every `int` and `float` leaf also carries its own numeric kind through normalization, so equal content produces an equal, cacheable key only when the values' type matches too: an int-valued array-like and an equal-valued float-valued array-like (e.g. `train_sizes=[1]` vs. `train_sizes=[1.0]`) key *differently*, because that distinction is not just container bookkeeping — sklearn's own `learning_curve(train_sizes=)` reads an `int` array as an absolute sample count and a `float` array as a fraction of the training set, two different requests that would otherwise collide onto one cached frame via Python's cross-type numeric equality (`1 == 1.0`, `hash(1) == hash(1.0)`). A value that is still unhashable after normalization (a dict, a custom object with no usable `__hash__`, a cv splitter object) is never cached at all: that call gets a key built from a fresh, never-repeating `_Uncacheable` marker, so the lookup always misses, the method always recomputes, and the result is never stored either — `self._cache` (a `_SourceCache`) recognizes the marker at write time and skips the assignment, so a caller who repeatedly passes the same reused unhashable value (e.g. looping `discrimination_threshold(cv=my_splitter)`) does not grow `self._cache` by one throwaway entry per call; correctness costs a cache miss, never a memory leak. This is a disclosed behavior change from the prior `repr()` fallback, which cached under an identity- or print-derived key that could collide two logically different calls. Four such collisions were verified end-to-end and are now impossible: (a) `discrimination_threshold(cv=<splitter>)` with a splitter class that defines `__eq__` (disabling identity `__hash__`) without overriding `__repr__` — CPython's default `object.__repr__` prints the memory address, and two short-lived temporaries of the same class reliably land at the same freed address (measured 10/10 trials), so `cv=Splitter(3)` followed by `cv=Splitter(7)` built the SAME key and the second call silently returned the first's frame; (b) `learning_curve(train_sizes=<ndarray>)`, where numpy's `repr()` rounds (`precision=8`) and truncates (`threshold=1000` → `...`), so two different `train_sizes` arrays could print identically and share a cache slot; (c) `shap_values(background=...)`, which keyed on a hand-rolled `str(background)[:64]` — two backgrounds differing only past character 64 of their printed repr collided outright; (d) `learning_curve(train_sizes=<int array>)` vs. `learning_curve(train_sizes=<equal-valued float array>)` — the dtype-blind collision described above, on the same method as (b) via a different vector. `validation_curve`'s own local `tuple(float(v) for v in vals)` normalization (previously the one caller that worked around the `repr()` fallback correctly) is now redundant and has collapsed onto the central normalizer.

---

### 3.2 Encoding Channels

> **2026-09-18 (wave 4, #90 — a channel's `title=` is the title its legend
> prints):** `title=` on `Color`/`Size`/`Shape`/`StrokeDash` and the other
> legend-bearing channels was honored on axes but discarded by every legend
> builder, which read the field name directly. A caller who wrote
> `fm.Color("species", type="N", title="Species")` got a legend headed
> `species`. One resolver now serves all of them — the swatch legend, both
> colorbars (continuous and binned), the conditional legend, the facet band
> and every auxiliary block — and legend-title suppression is spelled the
> way axes already spell it. The field name remains the default when no
> `title=` is given, so a chart that never set one is byte-identical. (The
> guide's "Legend title" section, `docs/site/guide/marks-encodings.md`, has
> illustrated this parameter since before it worked; its rendered image was
> regenerated with this note.)

> **2026-09-18 (wave 4, #130 — an untyped `Color` on `hue=` is a
> discriminator, not a gradient):** the figure-level helpers decided whether
> a `hue=` argument was categorical by asking whether the caller passed a
> bare string or a channel *object*, so `hue="species"` and
> `hue=fm.Color("species")` read the same column two different ways — the
> typed object fell through to a continuous color scale. The helper now asks
> the question that actually matters: whether the caller *stated* a type.
> `fm.Color("species")` with no `type=` is treated exactly like the bare
> string; `fm.Color("v", type="Q")` still gets its gradient. Bare-string
> `hue=` and an explicitly typed `Color` are both unaffected.

> **2026-05-10 (Phase 9):** Adds **position adjustments** as a sibling concept
> to encoding channels. Four immutable value classes — `Identity`, `Dodge`,
> `Jitter`, `Stack` — are passed via `position=` on eligible marks (Chart- or
> Layer-level). They serialize to `{"type": "<kind>", ...}` and are consumed
> by the Rust ChartSpec. Eligibility is enforced at chart-build time. The
> matrix below mirrors `src/ferrum/position.py`:
>
> | Position | Eligible marks |
> |---|---|
> | `Identity` | every mark (default; no-op) |
> | `Dodge` | `bar`, `point`, `box`, `boxplot`, `boxen`, `swarm`, `violin`, `errorbar`, `errorband`, `ribbon`, `histogram`, `density` |
> | `Jitter` | `point`, `swarm`, `tick` |
> | `Stack` | `bar`, `area`, `ribbon`, `histogram`, `density` |
>
> `Stack.offset` accepts `"zero"` (default) or `"normalize"` (percent-fill
> stacking). `Jitter.axis` accepts `"x"`, `"y"`, or `"both"`. `Dodge.by`
> selects the grouping field; if omitted, the active color/fill encoding
> field is used. Ineligible (mark, position) pairs raise `ValueError` at
> build time, not at render time.
>
> > **Dated note (2026-09-19, wave 5 — GH #184: one answer to which dtypes a position adjustment can handle):** the three adjustments held three separate opinions about which columns they could read, so the same frame behaved differently depending on which adjustment it met. **The value column:** `Dodge` and `Jitter` gated on the Arrow dtype rather than on the resolved scale kind, so a `Float32`, `UInt8`, `UInt16`, `UInt32` or `Timestamp` measure was refused where a `Float64` was accepted. All three now retype through one helper and the eleven dtypes the crate's numeric reader accepts agree across `Stack`, `Dodge` and `Jitter`. **The category column:** `Stack` carried a second, separate gate taking only `Float64`/`Utf8`, so `mark_bar(position=Stack()).encode(x="<Int64 column>:N", y=..., color=...)` refused while the identical column dodged, jittered and banded fine. That bespoke chain is deleted; the category column now keys through the same ordinal-category reader the band ladder is enumerated with, falling back to the temporal epoch reader — so two rows share a stack bin if and only if they share a **band** wherever a band ladder exists, and bin by exact instant where none does. Neither hole is a regression introduced by #171: the refusal was measured byte-identical with and without it.
> >
> > **`mark_area(position=Stack())` over a date or timestamp `x` now renders.** It raised for every input before, and that temporal arm is what closes it. It is not a second keying model: a stack over a continuous quantitative axis already binned by exact value (an `Int32` `x:Q` stacks to its group sum, measured at ladder max `9.0` against an unstacked control of `6.0`), so exact-value keying was the established behavior and `Timestamp` was the one dtype the reader could not spell.
> >
> > Two `Float64` partitions change, both toward the band ladder: `-0.0` and `0.0` now share a bin instead of stacking twice at one pixel, and a **null** category keys into its own band instead of merging into the real `0.0` bin. `Utf8` stacks stay byte-identical. A temporal column declared **ordinal** still refuses, from a different site (the band ladder cannot enumerate `Timestamp`); that refusal is pinned rather than deleted, and filed separately.

Encoding channels are typed objects passed as keyword arguments to `.encode()`. All channels accept a field name string as a shorthand.

#### Positional

| Class | Shorthand alias | Description |
|---|---|---|
| `X(field, *, type=None, bin=False, aggregate=None, scale=None, axis=None, title=None, stack=None, sort=None, impute=None)` | `"fieldname"` | Horizontal position |
| `Y(...)` | `"fieldname"` | Vertical position |
| `X2(field)` | — | Horizontal span end (for `mark_rect`, `mark_rule`) |
| `Y2(field)` | — | Vertical span end |
| `XError(field)` | — | Symmetric error bar extent on X |
| `YError(field)` | — | Symmetric error bar extent on Y |
| `XError2(field)` | — | Asymmetric error upper bound on X |
| `YError2(field)` | — | Asymmetric error upper bound on Y |
| `Theta(field, *, stack=True)` | — | Angular position (arc/pie marks) |
| `Radius(field)` | — | Radial position |

> **Dated note (2026-09-06, batch E tasks 7/12 — F-L01-01, encode-level `bin=True`, `Bin` gains per-bin aggregation and the encode path finally renders it):** `X(field, bin=True)`/`Y(field, bin=True)` were accepted (in the honored-kwarg set) but silently produced nothing renderable before this batch — the channel kept pointing at the original, unbinned field even after a `Bin` transform replaced the batch schema with `bin_start`/`bin_end`/`count`/`density` columns, so the chart either raised an unknown-column error or rendered garbage. `_lower_bin_encoding` (one implementation, shared by both the encode-time path — `Chart.encode()` building the initial spec — and the layer-resolution path — `_resolve_layer_bins`, so the two cannot drift as they once threatened to) now rewrites the binned channel to `bin_start` and, when no explicit span channel is already bound, adds `x2`/`y2` pointing at `bin_end` — a binned bar now spans its own bin width instead of collapsing to the mark's default width (verified live: bin rects render at their true per-bin pixel width, not a uniform default). The Rust `Bin` transform itself gains an optional `agg_field`/`agg` pair (drawn from the same `Aggregate` vocabulary `stat_aggregate` uses) that computes a **second, non-cumulative** aggregate per bin — even when `cumulative=True` is also set for the primary count/density curve — emitted as a new `agg_value: Float64` output column (registry-declared, drift-guarded). The encode path folds `X(field, bin=True)` + an aggregate on the *opposing* channel into one `Bin`: `Y(None, aggregate="count")` (fieldless) re-points at the `Bin`'s own `count` column — rows per bin, matching the pre-existing histogram idiom exactly — while `Y("v", aggregate="mean")` (or any non-count method, or `"count"` naming a real column) folds into `agg_field="v"`/`agg="mean"` and re-points at `agg_value`; `Y("v", aggregate="count")` specifically counts non-null `v` values per bin, not total rows, the same distinction `Aggregate`'s own `Count` makes elsewhere. This makes the FA-4 test suite's "the standard histogram pattern... is NOT an error" docstring claim actually true, now backed by a rendering test rather than spec-JSON-only coverage. Every rewritten channel's axis title recovers the user's own field name (`X("v", bin=True)` titles its axis `"v"`, `Y("w", aggregate="mean")` titles its axis `"w"`) instead of the `Bin` transform's internal `bin_start`/`agg_value` column names — verified live on both a single-layer and a layered (mixed bin + non-bin layer) chart. Six typed refusals replace what used to be either a silent no-op or a confusing downstream error: more than one `bin=` in an encoding; more than one channel carrying `aggregate=`; the same channel carrying both `bin=` and `aggregate=`; a fieldless non-`count` aggregate (`aggregate="mean"` with no source column); a pre-built `bin=Bin(...)` asked to also absorb an opposing `aggregate=` (a pre-built `Bin` exposes no readable parameters to rebuild with the fold — name `agg_field`/`agg` on the `Bin(...)` call directly instead); and an unknown aggregate method name, via the shared `Aggregate` vocabulary. A chart using neither `bin=` nor the new `agg_field`/`agg` parameters is byte-identical; layered-bin span recovery (the `x2`/`y2` addition) is the one disclosed byte-family change on charts that already passed `bin=True` (batch's disclosed-changes list, plan §4).

#### Appearance

| Class | Description |
|---|---|
| `Color(field, *, type=None, scheme=None, scale=None, legend=None, title=None)` | Fill and stroke color |
| `Fill(field, ...)` | Fill only (overrides Color fill) |
| `Stroke(field, ...)` | Stroke only |
| `Opacity(field, *, scale=None, legend=None)` | Overall opacity |
| `FillOpacity(field, ...)` | Fill opacity |
| `StrokeOpacity(field, ...)` | Stroke opacity |
| `StrokeWidth(field, *, legend=None)` | Stroke width — per-row constant, `scale=` accepted but not honored (see 2026-09-02 note below) |
| `StrokeDash(field, *, scale=None, legend=None, title=None, sort=None)` | Dash pattern |
| `Size(field, *, scale=None, legend=None)` | Point size or line width |
| `Shape(field, *, scale=None, legend=None)` | Point shape (circle, square, cross, diamond, triangle-*) — `scale=` domain and range now honored, see 2026-09-23 note below |
| `Angle(field, *, legend=None)` | Point rotation angle — per-row constant, `scale=` accepted but not honored (see 2026-09-02 note below) |

> **Dated note (2026-09-23, #123, wave 6B — every appearance-channel kwarg works or is refused):** four accepted-but-inert cells are fixed, and two already-inert kwargs flip from silent to refused.
>
> - **`Shape(scale=...)` is wired, domain and range, mirroring `Color`'s explicit-ordinal-domain/range path.** Before this, any `scale=` value — including one carrying a `range=` of point-shape names — was accepted, serialized, and silently ignored: `pairplot(markers=[...])` (a documented seaborn-compat argument) drew the same glyphs for every markers list. `domain` reorders the category-to-glyph assignment; `range` selects the glyphs directly, parsed all-or-nothing through the strict point-shape table (`ferrum.marks.base._VALID_POINT_SHAPES`, without `mark_point(shape=...)`'s no-match-falls-back-to-circle leniency). An unrecognized range entry is refused eagerly at construction naming `Shape(scale=...)`, and by a typed `RenderError` at render for a hand-built spec.
> - **`StrokeDash(scale=...)` wires the domain; an explicit `range=` is refused**, naming the fixed 4-slot GPU dash palette (solid plus three patterns) — that representation cannot carry an arbitrary domain-to-range mapping, so this refuses honestly instead of accepting silently (user ruling Q1b). Arbitrary interactive dash patterns remain a separable north-star (follow-up F6).
> - **`title=` on `Opacity`/`FillOpacity`/`StrokeOpacity` is refused**, not silently accepted: none of the three draws a legend, so there was nothing for a title to label. The three keep `scale=` (genuinely wired) and `legend=None`/`legend=False` (the harmless "no legend" idiom stays silent); only a truthy `legend=` value — a dict or `Legend(...)` — now refuses too, on those three plus `StrokeWidth`/`Angle`.
> - **A `Size`/`Shape`/`StrokeDash` legend value whose `orient=` is anything other than `"none"` is refused**, stating that auxiliary legends always share the chart's legend block — place it with `Color(legend=Legend(orient=...))` or `configure_legend(orient=...)`. `orient="none"` still disables that channel's own legend block.
> - **`FacetRow(columns=...)`/`FacetCol(columns=...)` flip from warn-and-drop to the same typed refusal** every other permanently-inert kwarg above gets, so the table has one policy instead of warn in one place and refuse in another.
>
> **Wire consequence of the `title=`/`legend=` refusals:** `Opacity`/`FillOpacity`/`StrokeOpacity`/`StrokeWidth`/`Angle` no longer emit a `legend` key at all, even for the harmless `legend=None`/`legend=False` idiom. Before this, those five channels shared the "own legend surface" emitter, so `Opacity("f", legend=None).to_encoding_spec_dict()` carried `{"legend": {"disabled": true}}`; now it carries no `legend` key, because there is no legend for a wire consumer to disable. No renderer ever read that key for these five channels (there was no legend to disable), so no SVG or interactive output moves — this is a wire-shape-only change, listed here because a wire diff against a saved pre-#123 spec would show it.
>
> Every other appearance-channel kwarg is unchanged. See `tests/test_appearance_kwarg_matrix.py` for the full channel × kwarg cross-surface matrix this note documents.

> **Dated note (2026-09-11, wave-1 hardening, #166 — `shape` reads Int64/Float64/Boolean columns):** `mark_point`'s `shape` reader switched from `col_as_str` to `col_as_ordinal_category_str` — the same ordinal-category reader `color` already uses — so a `shape` column typed `Int64`/`Float64`/`Boolean` now resolves each row's real shape kind instead of silently default-circling every row while the legend still enumerated every category (a genuine reader-switch, not a refusal: every input that previously read a real shape keeps reading it byte-identically; only the previously-blank case starts rendering). This is one of three sibling readers the same issue found swallowing a typed error into a blank render: `mark_text`'s and `mark_label`'s `text` channel, and `mark_rect`'s heatmap cell-annotation `text` channel, now PROPAGATE the typed error they used to discard via `.ok()` — a `text=` field naming a column absent from the layer's batch (or unreadable by either the string or numeric-fallback reader) now refuses instead of rendering a blank/degraded panel with no diagnostic (on an ordinal y axis this previously skipped every row silently, since the numeric fallback formatter has nothing to read). The pre-existing, load-bearing carve-out at `channels.rs`'s color-channel loader — a color column absent from ONE layer's own batch (but present on a sibling layer sharing the same encoding) renders as "no color for this layer" rather than an error, which is what keeps a multi-output composite chart rendering — is untouched by any of this: none of the three fixed readers touch the color channel, and a dedicated regression test on `mark_text` pins the same carve-out for its own text-channel path.

> **Dated note (2026-09-11, batch H close remediation): which channels accept
> a `condition=`.** A selection conditional — `encode(<channel>=sel.when(...)
> .otherwise(...))` or `<Channel>(field, condition=...)`, the two spellings
> being equivalent — may target exactly these eight channels: `color`,
> `size`, `opacity`, `stroke_width`, `stroke_opacity`,
> `stroke_dash`, `fill_opacity`, `angle`. `fill` is accepted and carries the
> conditional onto `color`, the same target it aliases to for its base
> encoding, so `encode(fill=cond)` and
> `encode(fill=Fill(f, condition=cond))` produce the identical wire
> conditional — and that value genuinely repaints the fill, which is what
> `fill` names. **`stroke` is NOT a conditional target** (corrected
> 2026-09-11 in the same batch's close re-review, before shipping): the only
> conditional colour value the wire carries is `EncodingValue::Color`, and
> the WASM runtime applies it to the mark's **fill**
> (`apply_value_common`; `FillStrokeView` exposes no stroke-colour member,
> though `CircleInstance`/`RectInstance` each carry a `stroke_color` the
> conditional path never touches). Aliasing a stroke conditional onto
> `color` would therefore recolour the interior while the user asked for the
> outline, so it is refused by name with the same typed error every other
> non-conditional channel gets. Filed follow-up: add
> `EncodingValue::StrokeColor` in Rust, after which `stroke` joins the list
> for real. **`shape` is NOT a conditional target either** (corrected the
> same day, same re-review): `ChannelName::Shape` exists, so a shape
> conditional serialized and round-tripped cleanly, but `EncodingValue` has
> no `Shape` variant — the conditional could only carry a field reference,
> which the runtime applies as a no-op. It was accepted and **inert**, which
> is the silent-drop shape this batch exists to remove, so it is refused by
> name like `stroke`. Filed follow-up: `EncodingValue::Shape`. (The batch's
> earlier docstring calling `Shape`'s `condition=` honored was written on the
> assumption that reaching the wire meant working; a mirror test against the
> Rust enum disproved it before the batch shipped.) **Every other channel
> raises `ValueError` at `encode()` time**,
> naming the supported set — previously a conditional on `fill`/`stroke` (or
> on any non-conditional channel such as `text`/`tooltip`/`facet`) died with
> an unmapped serde `unknown variant` error from inside `to_spec()`. The set
> is one Python constant, `ferrum.encoding._channel_policy._CONDITIONAL_CHANNELS`,
> mirrored from Rust's `EncodingValue` variants (the value the WASM runtime
> actually dispatches on) and **test-enforced in both directions** against
> that enum's source, so a future Rust conditional value cannot land without
> Python noticing; `ChannelName::X`/`Y` exist in the Rust enum for a
> *selection's* `encodings=` projection list, not for conditionals, and are
> therefore not conditional targets. Every wire channel name with no value
> variant — `x`, `y`, `shape` — is an explicit, documented refusal carrying
> its own reason, in one registry
> (`_channel_policy._CONDITIONAL_REFUSALS`, which also holds the `stroke`
> alias refusal); the mirror test asserts that registry covers the whole gap,
> so the next such channel is a test failure rather than a discovery. The
> channels' `_honored_kwargs` tables agree with this set: `StrokeWidth`,
> `StrokeOpacity`, `FillOpacity` and `Angle` declare `condition` honored
> (their `condition=` demonstrably renders; the warn guard used to call it
> ignored), while `Stroke` and `Shape` do not.
>
> **One conditional per wire channel (2026-09-11, close re-review; extended
> to `Chart + Chart` in batch-H close cycle 4 — see the dated note below).**
> A wire channel carries at most one conditional. Two declarations that
> resolve to the same channel — `encode(color=condA, fill=condB)`, since
> `fill` aliases onto `color` — raise `ValueError` naming both spellings and
> the shared channel, at the call that creates the conflict. Previously both
> were emitted with the same `channel` and no defined winner, so whichever
> the runtime applied last silently discarded the other declaration. The
> rule is enforced at all FOUR declaration entry points — `encode()`'s two
> spellings, `Chart.conditional()`, and `Chart.layer(Layer(encoding=...))`
> (see the dated note below) — **and, as of cycle 4, across `Chart + Chart`'s
> layer merge too** (see the dedicated dated note below for why the earlier
> exemption there was wrong).
>
> **`Chart.layer(Layer(encoding=...))` is a fourth declaration entry point
> (2026-09-11, batch-H close cycle 3, H3-1; both spellings covered as of
> cycle 4).** A `Layer`'s `encoding=` dict never passes through `encode()`,
> so a `condition=` inside it used to serialize into the dead per-channel
> `EncodingSpec.condition` blob (nothing on the Rust side reads it) instead
> of the chart's real `conditionals` pipeline — accepted, serialized, and
> **inert**, the exact shape this section exists to remove, one code path
> over. It is now lifted into the chart's `conditionals` list at
> `Chart.layer()` time, gated by the same `_CONDITIONAL_REFUSALS` vocabulary
> (a `stroke`/`shape`/`x`/`y` conditional inside `Layer(encoding=...)`
> raises the same typed error), and subject to the same
> one-conditional-per-wire-channel rule as `encode()` — including against a
> conditional already on the chart from an `encode()` call, or from another
> layer attached in the same `Chart.layer(*layers)` call. Both of
> `encode()`'s spellings are mirrored here: the channel-carried form
> (`Layer(encoding={"color": fm.Color(f, condition=cond)})`, cycle 3) and
> the bare `ConditionalSpec` form (`Layer(encoding={"color": cond})`, cycle
> 4 — the cycle-3 code only recognized `ChannelBase` values in this loop, so
> the bare spelling silently vanished with no warning and no conditional,
> the identical silent-drop shape this whole section exists to remove, one
> spelling over). This makes a layer-declared conditional LIVE where it was
> previously silently inert: a disclosed, deliberate behavior change (see
> the changelog).
>
> **`Chart + Chart`'s layer merge is not exempt from the one-conditional-
> per-wire-channel rule (2026-09-11, batch-H close cycle 4 correction).** An
> earlier version of this note (and of `Chart.conditional()`'s docstring)
> exempted `a + b` from the rule on the premise that "each layer's
> conditional belongs to that layer's marks." That premise is false: the
> WASM interactive runtime's `conditionals` list is flat and carries no
> per-layer scope at all — every conditional applies to every mark batch in
> every panel (`resolve_conditionals_with_packed`,
> `crates/ferrum-wasm/src/conditional.rs`). So `a + b`, where each side
> conditions the same wire channel (`color`, say), used to build a spec with
> two conditionals stamped the same `channel` and no defined winner — the
> exact undefined-runtime-behavior shape this rule exists to refuse, reached
> through the one entry point that used to skip the check. `Chart + Chart`
> now raises `ValueError` naming the wire channel when both sides condition
> it, checked cumulatively across a chained `a + b + c` merge; two layers
> conditioning DIFFERENT channels still compose freely, unchanged. **Breaking
> change:** `a + b` where both sides carry a conditional on the same wire
> channel now raises where it previously built (with undefined
> last-writer-wins runtime behavior); see the changelog.
>
> **Alias base-field-vs-own-conditional collision refused (2026-09-11,
> batch-H close cycle 3, H3-2; corrected in cycle 4).** The
> one-conditional-per-channel rule above covers two conditionals colliding
> on one wire channel; it did not cover an alias source's OWN base field
> colliding with its OWN conditional — `encode(color=fm.Color("d"),
> fill=fm.Fill("c", condition=cond))` used to build silently: `fill`'s field
> `"c"` is dropped by the `fill -> color` alias's `keep_target` policy
> (`color` is already encoded), while `fill`'s conditional survives and
> overrides `color` at selection time. The user's field mapping vanished
> while the call still visibly did something. This is now refused with the
> same message shape as the two-conditional collision, at `encode()` and at
> `Chart.layer(Layer(encoding=...))` alike, for **both same-call kwarg
> orderings** and for the forward cross-call ordering (`encode(color=...)`
> then a later `encode(fill=Fill(f, condition=...))`). The refusal requires
> the alias source to actually carry a field: `encode(color=fm.Color("g"),
> fill=fm.Fill(condition=cond))` — a fieldless `fill`, matching the
> equivalent bare-`ConditionalSpec` spelling `encode(color=...,
> fill=cond)` — builds; there is no field for the alias to drop (cycle-4
> correction: the cycle-3 code refused this unconditionally, a false
> refusal on a call that built before cycle 3).
>
> **Disclosed residual, not closed:** the REVERSE cross-call ordering —
> `.encode(fill=fm.Fill(f, condition=cond))` in one call, then a *later,
> separate* `.encode(color=...)` — still builds silently with the same drop.
> Closing it needs the declaration layer to remember, across chart clones,
> which encode key's OWN `condition=` produced a given wire conditional (a
> chart-private `wire_channel -> source_key` map carried alongside
> `_conditionals`, never serialized, never public — not a field on the
> public `ConditionalSpec` dataclass, which earlier cycles correctly
> declined to extend for this). Filed as a follow-up, not yet built.

#### Text / Detail / Tooltip

| Class | Description |
|---|---|
| `Text(field, *, format=None, formatType=None)` | Text label content |
| `Detail(field)` | Group-by without a visual channel |
| `Tooltip(*fields)` | Hover tooltip fields; accepts strings or `TooltipField` objects |
| `TooltipField(field, *, title=None, format=None, formatType=None)` | Tooltip field with formatting |
| `Href(field)` | URL for click-through on interactive charts |
| `Description(field)` | Accessible description per mark |
| `Key(field)` | Identity key for animated transitions |

> **Dated note (2026-09-03, batch B tasks 2/4):** `Text(format=...)`/`TooltipField(format=...)` resolve named format presets (`"currency"`, `"date_iso"`, `"ordinal"`, …) through the same shared resolver as `Axis`/`Legend` — see the format-preset dated note under §3.7 `Axis` for the full five-surface contract.

#### Facet (used with `.facet()`)

| Class | Description |
|---|---|
| `Facet(field, *, type=None, sort=None, title=None)` | Single facet dimension |
| `FacetRow(field, ...)` | Row facet dimension |
| `FacetCol(field, ...)` | Column facet dimension |

> **Dated note (2026-09-06, batch E task 13 — F-L01-04, facet `title=` wired):** `Facet`/`FacetRow`/`FacetCol`'s `title=` kwarg was in the honored-kwarg set (accepted silently) but was completely discarded before this fix — the rendered strip and the serialized `facet` dict carried no title information regardless of what was passed. It is now threaded through `Chart.encode()`'s facet synthesis and `Chart.facet(title=...)` into the Rust `FacetSpec.title`/`row_title` fields, and the strip text renders `"{title}: {value}"` when a title is set (the bare value otherwise — never the field name, correcting the three channels' prior docstring claim). `Chart.facet(title=...)` always sets the primary/wrap-dimension strip's title (the column strip in grid mode; the only strip in wrap mode). Routing note: `FacetRow(title=...)` used ALONE (no `facet_col`) lowers to **wrap mode**, where the column/wrap strip is the one that renders and reads `title` — so a standalone `FacetRow`'s title routes to `title`, not `row_title` (which has no effect outside grid mode). In grid mode (`facet_col` + `facet_row` both set), `FacetCol.title` → `title` (column strip) and `FacetRow.title` → `row_title` (row strip) are independent. No-title facets remain byte-identical (new wire keys are omitted when unset).

> **Dated note (2026-09-11, batch H task 12 — F-L06-02, `Facet(columns=)` honored):** `columns` on `Facet` (the `facet` wrap channel only — `FacetRow`/`FacetCol` are grid-mode, where a wrap count has no meaning and `columns=` there still warns and drops) is no longer a warn-and-noop. It writes the same `ncols` sink `Chart.facet(ncols=N)` already writes and Rust's `build_facet_dict` already consumes, with last-writer-wins precedence: a `columns=` on the channel overrides a prior `.facet(ncols=)` value carried into the same `encode()` call, and a *later* `.facet(ncols=N)` call still overrides an earlier channel `columns=`. No `columns=` on the channel preserves whatever `ncols` a prior `.facet()` call set.

---

**Channel shorthand strings**

All positional and appearance channels accept:
- `"fieldname"` — bare field
- `"aggregate(fieldname)"` — e.g. `"mean(price)"`, `"count()"`, `"q50(latency)"`
- `"fieldname:Q"` / `":N"` / `":O"` / `":T"` — inline type annotation (Quantitative, Nominal, Ordinal, Temporal)

> **2026-05-10 (Phase 8a):** All 31 channel classes are constructible Python
> value objects. Renderer honors `x`, `y`, `color`, `size`, `shape`, `opacity`
> in Phase 8a. Other channels (Stroke, Fill, FillOpacity, StrokeOpacity,
> StrokeWidth, StrokeDash, Angle, Text, Detail, Tooltip, TooltipField, Href,
> Description, Key, X2, Y2, XError, YError, XError2, YError2, Theta, Radius)
> are accepted at the API and stored on `EncodingSpec`, but the renderer
> ignores them with a one-time `UserWarning` per (channel, render call).
> Phase 9 wires the remaining channels.
>
> Channel kwargs honored in 8a: `type`, `bin`, `aggregate`, `scale`, `title`.
> Other kwargs (`axis`, `legend`, `sort`, `stack`, `impute`, `scheme`, `format`,
> `formatType`) are accepted, stored typed on `EncodingSpec`, and warn-once.
>
> **2026-05-11 (Phase 9, P1.5):** "one-time `UserWarning` per (channel,
> render call)" is implemented as **one-time per channel, process-wide**
> via `ferrum._warn.warn_once`. The stricter dedupe is the practical
> interpretation: notebook re-renders of the same chart should not stack
> dozens of identical warnings, and per-render dedupe would degrade to a
> per-process registry whose lifetime users can't reason about anyway.
> Tests reset the registry via `ferrum._warn.reset_warnings()` between
> cases; in user code the warning fires the first time a channel is
> accepted-but-not-rendered after import. Phase 9 still drops several
> of the listed channels (Fill, Stroke, FillOpacity, StrokeOpacity,
> StrokeWidth, StrokeDash, Angle, Text, Detail, Tooltip, TooltipField,
> Href, Description, Key, XError, YError, XError2, YError2, Theta,
> Radius); X2, Y2, and the existing `text` slot now render and were
> moved to the honored set.
>
> **2026-08-27 (P1 remediation, findings-remediation batch):** the
> Phase 9 amendment above is stale — most of the listed channels now
> render, and `encode()` is a total function over
> `ferrum.encoding._channel_class_map()`: every channel falls into
> exactly one of five disjoint, test-enforced buckets
> (`tests/test_finding_p1.py`).
>
> - **Honored** (own `EncodingSpec`, both chart-level and per-layer):
>   `x`, `y`, `x2`, `y2`, `color`, `size`, `shape`, `opacity`, `text`,
>   `tooltip`, `href`, `description`, `url`, `stroke_opacity`,
>   `stroke_width`, `stroke_dash`, `angle`, `fill_opacity`, and — newly
>   promoted — `key` (the Rust wire already existed via
>   `ChartSpec(key=...)` and `scene_build::extract_keys`; Python simply
>   never passed it through before this fix). `key` reaches the scene
>   graph (`MarkBatch.keys`) on both paths; at the time of this note the
>   WASM runtime did not yet read it, so it was bucketed Honored because
>   it reached its own `EncodingSpec` and the scene graph, not because it
>   was visually rendered — **superseded by the 2026-08-28 note below:
>   the WASM runtime now consumes `key` for transition pairing.**
> - **Alias** (redirect to another channel or to mark-style kwargs, no
>   warning unless noted): `fill`/`stroke` alias to `color`
>   (`stroke` warns once if `color` is already bound); `detail` aliases
>   to `mark_style.detail` on every mark, but only `mark_line`,
>   `mark_area`, and `mark_polygon`'s Rust builders read it — on any
>   other mark it now warns once (chart-level and per-layer both; a
>   layer's own `detail` previously reached no alias logic at all and
>   was dropped silently).
> - **Warn** (accepted, `warn_once`, absent from the resulting spec and
>   output — never reaches an `EncodingSpec` or a Rust `Encoding` field):
>   `x_error`, `y_error`, `x_error2`, `y_error2` (no
>   explicit-error-column feature exists for `mark_errorbar`; it
>   computes its own extents), and `tooltip_field` used as a top-level
>   channel (it is documented as valid only inside `Tooltip(*fields)`).
> - **Polar** (`theta`, `radius`, `theta2`, `radius2`): remapped to `x`/`y` and
>   rendered when the encoding's **effective** `CoordPolar` is set, warned once
>   (and absent from the spec) when it is not. For the chart-level encoding the
>   effective coord is the chart's; for a **layer's own** encoding
>   (`Chart + Chart`, `Chart.layer()`) it is that layer's own `coord` if it
>   declares one, else the chart's.
>
>   > **Dated note (2026-09-19, wave 5 T13 — GH #172(1)):** the layer half of
>   > this bullet changed. It previously read "there is no per-layer `CoordPolar`
>   > remap, so a layer's own polar channel warns once and is never rendered
>   > *regardless of whether the chart's coord is `CoordPolar`*". That is no
>   > longer true: layers carry a coordinate system of their own (see §3.16's
>   > per-layer coord note) and the layer pass runs the same remap against the
>   > layer's effective coord, so `fm.layer(bar, arc_chart.coord(fm.CoordPolar(theta="x")))`
>   > renders the arc member's `theta=` channel, and a polar chart's own layers
>   > carrying `theta=` render too. A polar channel on a layer whose effective
>   > coord is NOT polar still warns once and is still absent from the spec —
>   > that is the case where it could never render, and it is the discriminating
>   > control in `tests/test_finding_p1.py`.
> - **Facet** (`facet`, `facet_row`, `facet_col`): unchanged — resolved
>   through `resolved._facet`, never through the encoding-warn path.
>
> All warn messages use `ferrum._warn.warn_once("encoding", <channel>)`;
> dedupe key is the channel name, scoped per-context via
> `reset_warnings()` in tests exactly as before.
>
> **2026-08-28 (residuals batch, #93):** the P1 remediation note above is
> stale on one point — `key` no longer lacks a consumer. The WASM
> interactive runtime now pairs marks by `key` instead of flat index
> during a scene transition: matched keys lerp geometry and color as
> before, a key present only on the new side enters (final geometry,
> opacity ramping 0 → target), and a key present only on the old side
> exits (old geometry, opacity ramping target → 0, dropped at `t=1`).
> Pairing is per paired batch (batches still pair positionally by
> panel/batch index); a batch falls back to the pre-#93 index-zip pairing
> wholesale whenever either side lacks usable keys — absent keys, a
> duplicate key within the batch, a keys/instance-count mismatch, or a
> mixed-kind batch — never a partial keying. A key column
> whose values are not distinct per row degrades to that same index-zip
> fallback, silently: `Boolean` is non-injective above 2 rows by
> construction, null key values from any dtype collapse together to `""`,
> `Float32`/`Float64` saturate (every value `>= 2^63` collapses to one
> key, every `NaN` collapses to one key), and a key column of a dtype
> neither coercer covers (`Duration`, `List`, `Struct`) yields no keys at
> all for that batch — in every case the chart still renders, object
> constancy just never engages, with no warning (see
> `scene_build.rs::extract_keys`'s doc comment for the full dtype
> breakdown). Packed batches (≥1000 marks) carry keys through a
> `HAS_KEYS` sidecar section written by the packer and decoded by the
> WASM loader into `PackedBatchMeta.keys`, decoupled from the JSON
> `nodes` path. The runtime sources the OLD side's keys from an in-memory
> snapshot of the outgoing scene taken at load time, not by re-parsing
> its JSON — a packed old batch's JSON carries neither instances nor
> keys, so the snapshot is the only place a large old batch's identity
> survives; JSON re-parsing remains only as the first-load fallback, when
> there is no predecessor to snapshot. Static SVG output is unaffected
> and remains byte-identical with and without `key=` — this is a
> WASM-runtime-only consumer, not a static-render change — and unkeyed
> interactive scenes are byte-identical in behavior to before #93.
> Key-based selection membership and key-addressed hit-testing remain
> unbuilt (logged follow-ups, not this batch); nor is there a transition
> for mesh-backed marks (line/area/path), which carry no per-mark
> identity.

> **Dated note (2026-09-11, wave-1 hardening, #240):** `ferrum._warn.warn_once`'s
> dedupe key widens from `(channel, kwarg)` to `(channel, kwarg, message)` —
> two `warn_once` calls on the same `(channel, kwarg)` pair that carry
> genuinely different messages now BOTH emit (once each), where the pair
> alone used to swallow the second regardless of what it said. A repeated
> call with the identical message still dedupes exactly as before. Two
> carve-outs that previously said "planned for a future Phase" — a
> permanently-untrue statement for a refusal that will never become
> supported — now say the true, permanent fact instead: `condition=` on a
> channel `_channel_policy._CONDITIONAL_REFUSALS` names as permanently
> unsupported (`shape`, `stroke`, `x`, `y`) states that plainly, and
> `columns=` on `FacetRow`/`FacetCol` (grid-mode channels, where a wrap
> count has no meaning) states that the setting is permanently
> grid-mode-inapplicable rather than merely unimplemented. Two long-standing
> key-hacks that smuggled per-mark/per-column text into the dedupe key
> itself (`chart.py`'s `stack_on_{mark}`, `encoding/_aliases.py`'s
> `stroke_dropped_by_color`) are retired in favor of the real
> `(channel, kwarg, message)` identity — both now key on their real kwarg
> name and rely on the message carrying the mark/column-specific text, so
> two distinct marks/columns each still warn once apiece rather than the
> first blocking the second. `("encoding", "detail")`'s existing
> per-mark-interpolated message needed no key hack at all once the rekey
> landed — it was already the exact defect this fix closes.
>
> The uncapped per-in-gap-annotation warning fan-out and the unknown-cmap
> silent-Viridis-fallback the same issue named are Rust-side and ride the
> #162 transform-warning channel and #190's routed rider respectively — see
> the `BreakAxis` annotation-suppression dated note under §3.12 for the
> per-axis rollup, and the color-vocabulary dated note under §3.2 above for
> the cmap refusal/warning split.

> **2026-09-02 (batch A, appearance-resolution):** color and appearance-channel
> resolution honesty fixes (the api-contract-audit remediation campaign, batch A;
> follow-ups tracked as GH #107–#128).
>
> - **Figure-level `hue=` typing.** A figure function's `hue=`/group-color
>   binding types the color channel nominally (categorical palette, categorical
>   legend), so an integer-coded category column gets the categorical palette
>   rather than a continuous ramp or a fabricated colorbar. `catplot` answers
>   uniformly across all eight kinds (integer group keys reach the box kind too
>   via the shared group-partition entry point). The one deliberate carve-out:
>   `relplot(kind="scatter")` keeps seaborn-parity dtype-driven hue — a numeric
>   hue column there renders a continuous ramp by design (pinned). One tracked
>   deviation: an untyped `fm.Color(...)` object passed as `hue=` currently
>   bypasses the nominal typing (GH #130). The boundary is enforced by an AST
>   completeness test over every color-channel binding in `marks/` and `plots/`.
> - **Color vocabulary.** Every color-string site now accepts one shared
>   vocabulary: 148 CSS Color 4 named colors (case-insensitive, trimmed),
>   `#rgb`/`#rgba`/`#rrggbb`/`#rrggbbaa` hex, and `rgb(r,g,b)`/`rgba(r,g,b,a)`
>   with integer 0-255 channels and a float `0-1` alpha (a percentage-free
>   0-255 alpha integer is not accepted). The former silent bare-hex
>   normalization (accepting near-hex strings without validation) is removed.
>   `mediumpurple` is corrected to the CSS Color 4 value `(147, 112, 219)`.
>   Enforcement is **not** uniform across every site the vocabulary reaches —
>   an unparseable string raises a typed error naming the accepted forms only
>   at five mandated strict boundaries: mark construction (`fill=`/`stroke=`/
>   `color=` kwargs, Python `ValueError` at `MarkBase.__init__`), encoding
>   resolution at render time (Rust `resolve_mark_style`, typed
>   `RenderError`), `ferrum.color.to_hex` (Python `ValueError`), selection
>   styling (`selection.py`, Python `ValueError`), and the `background=`
>   render/theme override (Rust `RenderConfig`/`ThemeOverrides`, typed error).
>   **Closed 2026-09-11 (#126, wave-1 hardening).** `Color`/`Fill`/`Stroke`
>   scale `range=` entries — every one of the six sites that build a
>   position-indexed color range or gradient from explicit strings (three
>   encoding-level: categorical `range=`, `Gradient([...])` stops,
>   discretizing `bucket_colors`; three chart-config-level:
>   `configure_color(range=)`'s categorical, discretizing, and continuous
>   arms) — now refuse with a typed `RenderError` naming the first
>   unparseable entry, instead of the former warn-and-discard: an
>   unparseable entry used to emit a `UserWarning` and silently discard the
>   **entire** explicit range (not just the bad entry) in favor of the theme
>   palette, so one bad string dropped every good one alongside it with no
>   way to recover the user's intended mapping. The `configure_color(range=)`
>   continuous arm was the one site that wasn't even all-or-nothing before
>   this fix — it `filter_map`'d bad stops out with **no warning at all** —
>   folded into the same #126 refusal as a sibling defect of the same root
>   cause. All-or-nothing parsing itself is unchanged: a range/gradient is
>   indexed by position, so a partial parse would re-point every later entry
>   rather than shortening a cycle — a WRONG mapping, not a degraded one;
>   only the disposition (warn → raise) flipped.
>   **Closed 2026-09-11 (#107, wave-1 hardening).** The chart-config/
>   legend-styling/axis-band/title color fields (`AxisConfig`, `LegendConfig`,
>   `TitleConfig`, `GridAxisConfig`, `GridConfig`) now refuse an unparseable
>   color string at construction (Python `ValueError`, wrapping
>   `ferrum.color.to_hex`'s message), the same boundary shape as the five
>   mandated strict boundaries above — this is a sixth family of construction-
>   time gates, layered in front of them rather than added to the list. The
>   deeper Rust scene-build sites these fields feed (`apply_chart_config`,
>   `axis_style_fill_from`, legend swatch/label color, chart title/subtitle
>   color, axis band fills, inset background/border) keep their `.ok()`/
>   `unwrap_or` fallback shape unchanged — construction now refuses first, so
>   those fallbacks are unreachable from the typed Python API rather than
>   removed. `GridConfig.band_colors`' `"transparent"` entries stay legal
>   (an intentional "draw no rect for this band" skip, matching
>   `render/marks/axis.rs`'s own exemption) and are exempted from the color
>   parse, not from list membership.
>   **Closed 2026-09-11 (#122, wave-1 hardening).** `ferrum.color.to_hex`
>   now keeps a tuple's 4th component as alpha on every accepted tuple
>   form — `(1.0, 0.0, 0.0, 0.5)` and `(255, 0, 0, 128, scale="byte")` both
>   emit `#rrggbbaa`, using the same resolved `unit`/`byte` scale as
>   R/G/B — where it previously read only the first three components and
>   silently dropped a 4th, undocumented one. A 3-component tuple is
>   byte-identical to before; a 5th-and-later component is still read past
>   and ignored, unchanged. Every component actually read is now checked
>   `isinstance(..., numbers.Real)` before arithmetic: a non-numeric
>   component (e.g. `("a", "b", "c")`) now raises a typed `ValueError`
>   naming the offending component, closing the escaped bare `TypeError`
>   `to_hex` previously let through (`plots/_helpers.py`'s
>   `_column_or_literal_color` kept its `except (ValueError, TypeError)`
>   clause as a defensive belt only — no call site relies on the escaped
>   `TypeError` anymore). `bool` is not excluded from the numeric check
>   (unlike e.g. `configure.py`'s domain-bounds validator) — a
>   `(True, False, True)` tuple rendered `"#ff00ff"` before this change and
>   still does; only the genuinely-non-numeric escape is closed.
> - **Two clearing spellings.** `"none"` and `"transparent"` both clear a
>   mark's fill/stroke paint at the mark boundary, in both Python and Rust,
>   matched case-insensitively after trimming. (Selection styling diverges —
>   see §3.10.) A paint the user actually cleared (never a paint that merely
>   evaluates to transparent, e.g. an explicit `fill="#00000000"`) serializes
>   as no attribute at all. **(2026-09-21, GH #110.)** That rule is now
>   universal across marks: the former exception list (line/rule/tick/segment
>   stroke, and arc/area/geoshape/polygon, which serialized a cleared paint as
>   a zero-alpha value — visually identical, byte-different) is closed, and
>   `StrokeStyle.color` in the scene graph is optional so a stroke-only node
>   omits the attribute rather than carrying `rgba(0,0,0,0.000)`. A cleared
>   *fill* still emits SVG's own `fill="none"` keyword, because SVG's default
>   fill is black and omitting the attribute would paint it; a cleared
>   *stroke* omits the attribute, whose SVG default is already `none`.
>   Text marks follow the fill rule: `TextStyle.color` is optional too, so
>   `mark_text(fill="none")` and `mark_label(fill="none")` write
>   `fill="none"` on each `<text>` element instead of `rgba(0,0,0,0.000)`,
>   `mark_label`'s leader lines (painted in its fill) emit no stroke, and
>   the interactive runtime draws no glyphs for them.
>   The same release makes an explicit clear outrank a **color channel**:
>   `mark_line(stroke="none").encode(x=…, y=…, color=…)` previously rendered
>   byte-identically to the same chart without the clear, painting the
>   categorical palette over it, and the same was true of a cleared `fill=`
>   under a color channel on `area`/`arc`/`bar`/`rect`/`point`/`polygon`/`text`. A
>   clear is a decision to paint nothing, and it now wins over any color the
>   channel resolves.
> - **`FillOpacity`/`StrokeOpacity`/`Opacity`/`StrokeDash` `scale=` is now
>   honored end to end** (previously silently dropped for the first three;
>   `StrokeDash` gained a real `scale=` implementation the same batch).
>   Default (no explicit `scale=`) for `FillOpacity`/`StrokeOpacity` on a
>   quantitative field resolves domain = data extent, range =
>   `[theme.sizes.opacity_min, opacity_max]` — matching `Opacity`'s existing
>   semantics — except when the column's data extent is degenerate (min ==
>   max) and no explicit `scale=` domain is given, in which case no scale
>   resolves and each row keeps its own literal alpha instead of being
>   repainted to the band midpoint (GH #104; `Opacity` itself is excluded
>   from this carve-out and keeps its pre-batch behavior of repainting a
>   constant column to the midpoint). Resolved opacity range endpoints
>   always clamp into `[0.0, 1.0]` before scale construction, so an explicit
>   `range=[0, 5]` cannot produce an out-of-gamut alpha. A non-numeric column
>   bound to `fill_opacity`/`stroke_opacity` is a typed render error naming
>   the channel and dtype (previously a silent default-alpha fallback). A
>   non-Linear `scale=` (Log/Pow/Sqrt) on any opacity-family channel emits a
>   `RenderWarning` naming the channel and the dropped scale type, then falls
>   back to Linear resolution — the curve, domain, and range are not honored
>   yet (logged follow-up).
> - **`StrokeWidth`/`Angle` `scale=` is still not honored.** Both remain
>   per-row constants by documented design (`spec/encoding.rs:746`); the
>   table above lists only the kwargs each channel actually honors (`type`,
>   `legend`, and — since 2026-09-11 — `condition`, per the conditional-channel
>   note under §3.2's appearance table). This corrects a stale advertisement — earlier phase notes
>   implied `scale=` reached the wire for these two channels; it never did,
>   and this batch made no change to that.
> - **`StrokeDash` full contract.** `scale=` and `title=` are honored
>   (2026-08-28); `sort=` is honored (2026-09-01, mirrors `Shape`, since the
>   Rust `build_stroke_dash_scale` domain builder already read it);
>   `condition=` is honored — `stroke_dash` is one of the nine conditional
>   channels (2026-09-11, batch H close remediation; this line previously read
>   "accepted on the wire but not yet consumed by any builder (reserved)",
>   which the shipped behavior already contradicted — see the
>   conditional-channel dated note under §3.2's appearance table). A
>   categorical `stroke_dash` column groups `mark_line`/
>   `mark_ribbon` series by (color, detail, stroke_dash) instead of merging
>   distinct dash categories into one polyline. `mark_line`, `mark_rule`,
>   `mark_point`, `mark_bar`, and `mark_rect` (rect newly gained the read)
>   all honor a bound `stroke_dash` channel: a **numeric** column keeps the
>   pre-existing `DASH_PALETTE`-index contract byte-identically; a
>   **string/categorical** column resolves through the new
>   `StrokeDashScale`. A dedicated dash-swatch legend renders beside
>   `Shape`'s aux legend. A literal `stroke_dash=[...]` mark kwarg on
>   `mark_point` now applies to the point's stroke (previously silently
>   dropped) — see `mark_point`'s docstring.
> - **`StrokeDash(scale=...)` domain now honored; `range=` refused (2026-09-23, #123, wave 6B).** Before this, `scale=`'s presence reached the wire (the line above) but neither `domain` nor `range` was actually read by `build_stroke_dash_scale` — any explicit domain order or range was silently ignored. The domain now reorders the category-to-pattern assignment the same way `Shape`'s does. `range` is refused outright, at construction, naming the fixed 4-slot GPU dash palette (solid plus three patterns, `render/draw.rs::DASH_PALETTE`): that representation has no slot for an arbitrary domain-to-range mapping, so this states the limit instead of accepting a value that can never do anything (user ruling Q1b). Arbitrary interactive dash patterns are a separable north-star, logged as follow-up F6.
> - **Continuous/discretizing color on `mark_line`/`mark_ribbon` fails
>   loudly.** These stroke-continuous marks cannot render per-segment color;
>   binding a continuous or discretizing `Color` scale to either now
>   suppresses the misleading gradient/colorbar legend and emits a
>   `RenderWarning` naming the mark and the unsupported scale kind, instead
>   of silently rendering one solid, wrong color under a full colorbar. True
>   gradient-colored polylines are a logged feature follow-up.

---

### 3.3 Marks

Marks are constructors that accept visual property overrides as keyword arguments. All marks inherit from `MarkBase`.

> **Dated note (2026-09-11, wave-1 hardening, #243 — each mark refuses a style kwarg it doesn't honor; settles #154):** a mark kwarg is now checked against that SPECIFIC mark's own honored vocabulary — a per-mark frozenset derived from `MarkKwargsSpec`'s own family mapping (`crates/ferrum-core/src/spec/mark_style.rs`'s section-header comments, drift-pinned) — rather than one flat allowlist shared identically by every mark. `mark_bar(arrow=True)` (a segment-only field) now raises `TypeError` naming the mark and its honored set, instead of being silently accepted at construction and dropped at the Rust wire with no feedback. Refusal is immediate and unconditional — no warn-then-refuse staging, so #240's same-pair-different-message double-signal shape never arises here. This also closes a second, related gap: 13 names (`method`, `ci`, `bandwidth`, `degree`, `n`, `kernel`, `extent`, `cumulative`, `bin_count`, `bin_width`, `density`, `right`, `multiple`) had lived in the old flat allowlist despite never being real `MarkKwargsSpec` fields — they are named parameters of specific statistical mark FUNCTIONS (`mark_smooth`, `mark_density`, `mark_histogram`, `mark_errorbar`), always intercepted before reaching a mark's style-kwarg validation; passing one of them directly is now refused instead of silently accepted-then-dropped. **Settles #154**: `mark_bar(band_size=0.9)` now raises (`band_size` is honored only by `mark_tick`/`mark_rect`, and `mark_bar` never read it) — previously silently accepted and dropped. The docs sentence this contradicted ("passing a kwarg that doesn't apply to a given mark is harmless — the renderer ignores it") is retired.
>
> **Dated note (2026-09-11, wave-1 hardening, #136 — `mark_raster()` validates its `aggregate`/`field` combination at call time):** `Chart.mark_raster(aggregate=..., field=...)` now validates eagerly, before `.encode()`, reusing the same accepted-set constants `RenderConfig.__post_init__` already checks — previously an invalid `aggregate` (e.g. `"median"`) fell all the way through to a Rust `ValueError` at render time, and a missing `field=` for `aggregate="mean"`/`"sum"` only raised once `desugar_raster` resolved (i.e. once an encoding was set), so `Chart(df).mark_raster(aggregate="mean")` built silently with no field before this fix. Probed per the decision record before writing a redundant check: the docstring's long-standing claim that `raster_field` "must be Float64-typed" is stale — batch D's dtype-widening already coerces any integer/unsigned/float column, so an `Int64` field renders correctly today; the docstring is corrected to state the real current message (`stat_raster: column '...' must be numeric, got <Dtype>`) instead of adding a redundant refusal for coverage that already exists.

> **Dated note (2026-09-19, wave 5 — GH #171: the type you declared is the type that bands):** a `mark_boxplot`, `mark_violin`, `mark_errorbar` or `mark_boxen` whose category column carried an integer, unsigned, float or boolean dtype resolved to a **quantitative** scale, so its tick layer landed in the quantitative fallthrough and drew 8px stubs where band geometry belonged — measured on a 3-category `Int64` column, three box rects dropped entirely and the axis grew from 3 category ticks to 9 quantitative ones. Writing `x="cyl:N"` did not rescue it, and neither did `catplot`'s documented force-nominal default: the composite desugar reduced both positional channels to bare field-name strings before any desugar ran, and six desugar sites then rebuilt the categorical channel untyped, so the declaration never survived long enough to be read. The declared type now travels through the desugar. Where the caller declares nothing, the five category-axis composite marks default that channel **nominal**; `mark_errorband` stays untyped, because its `x` is a genuine value axis. **This changes rendered output**: such a chart now draws band geometry — boxes, violins, whiskers at band centres — where it previously drew rug-like stubs, and its axis draws one tick per category instead of quantitative ticks. `catplot(native_scale=True)` is unaffected and stays deliberately quantitative.
>
> `mark_tick` is deliberately **not** part of that default. It is a primitive, not a composite desugar: it bands when its channel is declared nominal at any dtype, and stubs as a true rug when the channel is an undeclared numeric, which is what #171's own text asks for. (The design spec was amended on 2026-09-19 to say so, after the behavior was measured rather than assumed.)

> **Dated note (2026-09-19, wave 5 — GH #124: `stroke_dash=` accepts the named linetypes on every surface that accepts `stroke_dash=`):** `mark_line(linetype="dashed")` worked and `mark_line(stroke_dash="dashed")` raised, because the gate read the caller's **spelling** rather than the canonical key, so the canonical form fell through to `float("dashed")`. Measured, the delta was exactly the five named linetypes (`"solid"`, `"dashed"`, `"dotted"`, `"dashdot"`, `"longdash"`) — comma strings with or without spaces, float lists, int lists, tuples, both empty spellings and the junk refusal were already symmetric — so the canonical kwarg was a strict **subset** of its own alias. The canonical form is now the superset. One normalization serves four surfaces: `MarkBase` mark kwargs, `annotate_hline`/`annotate_vline`/`annotate_abline` (which take `stroke_dash=` and have **no** `linetype=` parameter of their own, so the named forms were unreachable there by any spelling), and `fm.SelectionMark(stroke_dash=...)` — which went the other way and accepted anything, so `SelectionMark(stroke_dash="wiggly")` constructed silently and now refuses, naming the entry the caller invoked. No value that was accepted before renders differently. **This reverses a shipped decision** (`f8148bdc`, batch A T9, whose two tests pinned the named forms as belonging to `linetype=` only); #124 was filed by that batch's own final design review, dissenting. Both tests are inverted with rewritten rationales, and the reversal is recorded for the campaign's decision-flip audit.

> **2026-05-10 (Phase 9):** `mark_segment` is shipped as a primitive mark in
> Phase 9. It draws a line segment from `(x, y)` to `(x2, y2)` and is
> diagonal-capable — distinct from `mark_rule`, which is axis-aligned.
> Required encodings: `X`, `Y`, `X2`, `Y2`. Use cases: dumbbell charts,
> slope graphs, network edges, free-form annotations. Removed from
> `PHASE_9_PLUS_MARKS` warn-list.

#### Primitive Marks

| Mark | Description | Key Parameters |
|---|---|---|
| `mark_point(...)` | Scatter / dot plot | `size`, `shape`, `filled`, `stroke_width` |
| `mark_line(...)` | Line chart | `interpolate`, `stroke_width`, `stroke_cap`, `stroke_join` |
| `mark_area(...)` | Area chart | `interpolate`, `line`, `opacity` |
| `mark_bar(...)` | Bar chart | `orient`, `corner_radius`, `width`, `bin_spacing` |
| `mark_rect(...)` | Filled rectangle; heatmaps | `corner_radius` |
| `mark_rule(...)` | Horizontal or vertical reference line | `stroke_dash`, `stroke_width` |
| `mark_text(...)` | Text labels | `align`, `baseline`, `dx`, `dy`, `font_size`, `font_weight`, `angle`, `limit` |
| `mark_tick(...)` | Rug / tick marks | `band_size`, `orient` |
| `mark_arc(...)` | Arc / pie / donut | `inner_radius`, `outer_radius`, `pad_angle`, `corner_radius` |
| `mark_image(...)` | Image at point | `width`, `height`, `align`, `baseline` |
| `mark_geoshape(...)` | Choropleth / geographic shape | `projection` |
| `mark_segment(...)` | Line segment between (x,y) and (x2,y2). Requires X, Y, X2, Y2 encoding channels. Used for dumbbell charts, slope graphs, network edges, and free-form annotations. | `stroke`, `stroke_width`, `stroke_dash`, `stroke_cap` (`"butt"`\|`"round"`\|`"square"`), `arrow` (bool), `arrow_size` |
| `mark_label(...)` | Text with a solid background fill box. All `mark_text` parameters, plus: | `background_fill`, `background_stroke`, `background_stroke_width`, `background_padding`, `background_corner_radius` |

> **Dated note (2026-09-11, batch H task 1/task 2 — F-L08-01, `mark_segment(arrow=)` goes from spec-fiction to real geometry; `annotate_arrow` grows a head on every render path):** `mark_segment`'s `arrow` (bool, default `False`) and `arrow_size` (pixels, default `8.0`) — documented in this table's Key Parameters column since Phase 9 but never accepted by the mark-kwargs validator — are now real. `arrow=True` draws a filled triangle arrowhead at `(x2, y2)`, sharing one triangle-geometry helper (`draw::arrowhead_triangle`) with the pre-existing `annotate_arrow`/`ann.arrow()` primitive path, so a mark-path arrowhead and an annotation-path arrowhead are byte-consistent for the same endpoints and size; the head inherits the segment row's own resolved opacity/stroke-opacity/fill-opacity rather than always painting fully opaque. `mark_segment()` without `arrow=` is byte-identical to every pre-existing render (the branch is never reached). `annotate_arrow(x1, y1, x2, y2, ...)` (§3.11) now renders an arrowhead by default on **all three** of its render paths — standalone `.to_svg()`, unlabelled `+` overlay (both via `mark_segment(arrow=True, arrow_size=8)`), and labelled overlay (via `AnnotationArrow.head_size=8.0`, previously `0.0`) — where it previously rendered headless everywhere; `ann.arrow()` is unaffected (it already drew a head). This closes F-L08-01 and the spec:690 drift together.

> **2026-09-02 (batch A — `mark_rule` totality and layered domain union):**
> `mark_rule` now covers every presence-legal channel combination: an input
> that passes the presence gate either renders as real geometry or raises a
> typed `UnsupportedChannelCombination` error listing the supported shapes —
> it can no longer fall through to a silently empty render. This batch adds
> an all-four-numeric `(x, x2, y, y2)` diagonal/segment shape (what
> `mark_qq(line=True)` needs to render its reference diagonal), a
> numeric-`x` + `y` + `y2` vertical span, and a numeric-`y` + `x` + `x2`
> horizontal span (the latter two were previously a silent blank panel or
> silently wrong geometry). The anchor read is keyed off the resolved scale
> kind, not the raw column dtype, so an ordinal scale on an `Int`/`Float`/
> `Utf8` column still takes the categorical (banded) reading; a null
> ordinal-anchor row now bands at the null category exactly as `mark_point`/
> `mark_bar` do, instead of being silently skipped. Making the diagonal
> shape actually visible required fixing `numeric_domain_union` to include a
> layer's `x2`/`y2` fields when computing shared axis domains — it
> previously unioned only bare `x`/`y`, so a layered chart whose `x2`/`y2`
> extended past its `x`/`y` (band charts, error bars, the new rule diagonal)
> could render geometry outside the visible axis domain. The fix is global,
> not rule-specific: any layered chart with wider `x2`/`y2` extents may now
> resolve a correspondingly wider axis domain than before this batch.

> **Dated note (2026-09-06, batch E task 1 — F-L02-02, `interpolate=` gains a real curve module; the typed unknown-value refusal universalizes to every mark carrying the kwarg):** `mark_line`, `mark_area`, and `mark_ribbon` all evaluate `interpolate="monotone"` (Fritsch–Carlson), `"basis"` (uniform cubic B-spline), and `"cardinal"` (Catmull–Rom, tension 0) through one shared Rust module (`render::marks::curve`) instead of each mark silently falling back to a straight line for any value it didn't special-case — before this batch, only `"linear"` and the three step variants had real geometry on `mark_line`/`mark_area`, and `mark_ribbon` refused `interpolate=` outright (see the T14 note on `mark_ribbon` below). An interpolate value outside the seven-member vocabulary (`"linear"`, `"step"`, `"step-before"`, `"step-after"`, `"monotone"`, `"basis"`, `"cardinal"`) now raises `ValueError` naming the accepted set, on every mark that carries the kwarg — the same typed refusal (`RenderError::UnknownInterpolate`), not a per-mark bespoke check. Two disclosed byte-family changes on charts that already pass `interpolate=` (never on charts that don't): (1) a banded/stacked `mark_area`/`mark_ribbon` with `interpolate="step"`/`"step-before"`/`"step-after"` now steps **both** the top and bottom boundary edges — previously only the top edge stepped while the bottom boundary walked straight between vertices, an incoherent shape no golden covered; (2) a series whose independent axis is not the vertical screen axis — `Stack(value_axis="x")` (a horizontal stack) or a chart under `CoordFlip` — now parameterizes its curve along its true independent axis (the one it is actually a function of), rather than always assuming the vertical axis, so `interpolate="monotone"` on a horizontal series is now genuinely monotone in the axis it varies along. Two points or fewer always render as a straight line under every method (no curve has a defined shape with fewer than 3 points). `interpolate` unset, or `interpolate="linear"`/a step value, on a chart that already rendered correctly is byte-identical.

> **Dated note (2026-09-06, batch E task 2 — F-L02-06, `mark_arc(corner_radius=...)` goes from silently inert to real geometry):** `corner_radius` on `mark_arc` was accepted by the mark-kwargs allowlist but never read by the wedge builder — every arc rendered square-cornered regardless of the value passed. It now clips each corner with an exact tangent circle of the requested radius (clamped per-corner so a radius larger than the wedge's own span or the ring's own thickness degrades to the largest radius that still fits, rather than producing self-intersecting geometry), and the same wedge path is shared by a `mark_bar` rendered under `CoordPolar` (a polar bar chart), so `corner_radius` rounds polar bars too — one call site (`bar.rs::build_polar`) passes `ctx.mark_style.paint.corner_radius` into `wedge_path` unconditionally, with no branch on which encoding channel `theta` names, so this holds for either `theta="x"` or `theta="y"`. Verified live under the canonical `theta="x"` orientation (categorical field on `x`, numeric field on `y`): 3 of 4 emitted paths (the per-category wedges; the 4th is the polar frame's own background circle, which is not a bar and does not respond to `corner_radius`) differ between `corner_radius=0` and `corner_radius=8`. The mirrored `theta="y"` orientation requires swapping which field is angular vs. radial (`y` becomes categorical, `x` becomes numeric) to reach the same real-wedge code path — an untested but structurally identical combination. `corner_radius=0` (the default) is byte-identical to every pre-batch arc/polar-bar render. Interactive-only, disclosed: the polar wedge builders (arc and polar bar) now also emit a `polar_spans` key on each node in the interactive scene JSON — the node's true `(angle_start, angle_end)` — so hit-testing reads the real span instead of inferring one from path geometry; a consequence is that a full-circle wedge (a single category spanning the whole circle, e.g. a one-slice pie or a `pad_angle=0` donut with one group) is now hittable at every angle, where before it was hittable only across a half-ring or a single degenerate angle. Static SVG is untouched by this interactive change.

> **Dated note (2026-09-06, batch E task 3 — F-L02-05, rug `mark_tick(band_size=...)` goes from silently inert to real geometry):** a **rug** tick — one quantitative axis encoded and no categorical channel, e.g. `Chart(df).mark_tick().encode(x="x")` — previously ignored `band_size` entirely and always drew a fixed-length tick (`theme.sizes.tick_size * 2.0`). It is now honored there too: when set, the rendered tick length is `band_size` times the panel's **cross-axis** extent (panel height for an x-rug, panel width for a y-rug — the dimension perpendicular to the axis the rug's ticks measure), matching the full-length-factor convention `mark_rect`'s `band_size` already uses (#85). An **unset** `band_size` on a rug keeps the old fixed length exactly — it does **not** pick up the ordinal default of `0.6`, which would turn an untouched rug into a set of panel-spanning bars; this is a deliberate divergence from the ordinal-tick default, not an oversight. Ordinal-tick `band_size` is unchanged, **including when the other axis is also declared** — verified live: `mark_tick(band_size=0.2).encode(x="cat", y="val")` (ordinal x, own y — the boxplot-median/strip-plot idiom) still shrinks each tick from its 0.6-default length to the requested fraction of its category band, and the ordinal-y/own-x mirror behaves identically. The narrower rule the code actually implements (`tick.rs:473-511`) is: only a tick whose **x resolves to a quantitative scale while y is also separately declared** — not a true rug, since a true rug has no y at all — keeps the old fixed length unconditionally and ignores `band_size`. This exists because the composite tick layers that reach this exact shape (boxplot cap, errorbar cap) set `band_size` on themselves unconditionally; scaling by it here would turn their small fixed-length caps into panel-height bars on a chart where the *caller* passed no `band_size`. Out of this fix's scope either way — F-L02-05 only covers the true-rug shape — but the boundary is the x scale's kind, not "both axes encoded." No golden without an explicit `band_size` on a rug moves.

> **Dated note (2026-09-06, batch E task 11 cycle 3 rider — discovered alongside F-L02-04/F-L03-04, `mark_tick` per-row `color=`/`fill=`/`stroke_opacity=` go from silently flat to real geometry):** `mark_tick` previously rendered every row in the theme's flat default stroke color and full opacity regardless of a `color=`/`fill=`/`stroke_opacity=`/`opacity=` encoding — those channels resolved in the scale layer but the tick builder never read the per-row resolved value. Ticks now consume per-row `color=`/`fill=`/`stroke_opacity=`/`opacity=` encodings, but not uniformly: an explicit constant `stroke=` mark-kwarg still wins over an encoded `color=` (verified live, scoped to the tick's own `<line>` elements — `mark_tick(stroke="#123456").encode(x="x", color="c")`'s ticks all carry `stroke="#123456"`, the same value every tick carries with no `color=` encoding at all; the *whole* rendered SVG is not byte-identical between the two, since adding a `color=` encoding also adds a color legend that shifts the panel — the equivalence is in the tick geometry's own resolved stroke, not the file bytes), but a constant `opacity=` mark-kwarg does **not** win over an encoded `opacity=` — the encoded value replaces it outright (verified live at three distinct row values — `mark_tick(opacity=0.9).encode(x="x", opacity="op")` is byte-identical to the same chart built with no `opacity=` mark-kwarg at all, both encoding the *same* `opacity="op"` channel — an apples-to-apples pair with no legend-presence difference between the two sides). The asymmetry is mechanical, not a documented policy choice: encoded `opacity` feeds `to_scene_stroke` as a direct construction parameter (`tick.rs:20`), leaving nothing for a constant to override, whereas `stroke`/`color` resolution has an explicit precedence step. A third channel is a **documented, unclosed exception**: per-row *encoded* `stroke_dash` still has no effect on `mark_tick` at all (verified live, scoped to the tick's own `<line>` elements — `mark_tick().encode(x="x", stroke_dash="c")`'s ticks carry no `stroke-dasharray` attribute at all, identical to the same chart's ticks with no `stroke_dash=` encoding; again the whole-SVG bytes differ, since the encoded chart also gains a dash legend) — `tick.rs:29-45` adjudicates this explicitly as a real, distinct gap left open on purpose, out of this cycle's scope (`mark_rule`/`mark_line`/`mark_ribbon` all resolve per-row encoded `stroke_dash`; `mark_tick` does not). A tick chart passing none of `color=`/`fill=`/`stroke_opacity=`/`opacity=`/`stroke_dash=` renders byte-identically to before this fix.

#### Composite Marks (expand to multiple primitive layers)

> **2026-05-10 (Phase 9):** Adds `mark_boxen` (letter-value plot) as a
> composite mark. Expands to a stack of nested rectangles per letter-value
> depth plus outlier points; uses the `LetterValue` stat transform. Key
> parameters: `k_depth` (`"tukey"` \| `"proportion"` \| `"trustworthy"` \|
> `"full"` \| `int`), `k_proportion` (float in `(0, 1)`, used when
> `k_depth="proportion"`), `outlier_threshold` (float; rows beyond the
> outermost letter value are flagged outliers), `palette` (`str` \|
> `Sequence[str]` \| `None`) — see the 2026-08-27 note below (residuals
> batch, #91): colors the depth bands directly, replacing the opacity
> ramp.

| Mark | Expands To | Key Parameters |
|---|---|---|
| `mark_boxplot(...)` | box + whisker + outlier points | `extent` (`"min-max"` or float IQR multiplier), `size`, `outliers` |
| `mark_boxen(...)` | nested rectangles (letter values) + outlier points | `k_depth` (`"tukey"`\|`"proportion"`\|`"trustworthy"`\|`"full"`\|`int`), `k_proportion`, `outlier_threshold`, `palette` (`str`\|`Sequence[str]`\|`None`) — colors depth bands directly, see 2026-08-27 note below |
| `mark_errorbar(...)` | rule + tick | `extent` (`"ci"`, `"stderr"`, `"stdev"`, `"iqr"`), `ticks` |
| `mark_errorband(...)` | area + line | `extent`, `borders` |
| `mark_ribbon(...)` | area between Y and Y2 | `opacity`, `interpolate` |

> **Dated note (2026-09-06, batch E task 14 — F-L02-03, `mark_ribbon(interpolate=...)` goes from a hard refusal to real geometry):** `desugar_ribbon` previously raised `ValueError` on any `interpolate=` other than the default — the kwarg existed but every value except silence was rejected. It now forwards `interpolate` straight into the synthesized layer's mark kwargs, which the shared curve module (see the task 1 note above) renders on both the ribbon's top and reversed-bottom edges. The full seven-value vocabulary is accepted; an out-of-vocabulary value refuses the same typed way `mark_line`/`mark_area` do (`RenderError::UnknownInterpolate`), not the old Python-side single-value guard. `interpolate` unset (the `"linear"` default) is byte-identical to before.

#### Statistical Marks (trigger stat engine)

| Mark | Stat Computed | Key Parameters |
|---|---|---|
| `mark_smooth(...)` | LOESS or regression fit + CI band | `method` (`"loess"`, `"lm"`, `"glm"`, `"gam"`, + aliases), `ci`, `bandwidth` (loess-only — explicit values on other methods raise, 2026-09-06), `degree`, `k` (gam-only basis dimension, 2026-09-06), `n`, `layer_kwargs` (per-layer mark kwargs, keyed by sub-layer name — `{"ribbon": ..., "line": ..., "metrics": ...}` here, 2026-09-18: a bare style kwarg reaches every emitted layer including the CI band, and a per-layer value outranks it on the layer it names; an unknown name raises. Shared by every composite mark, keyed on the same sub-layer vocabulary the figure-level `mark={...}` override uses; replaces the unreleased `line_kwargs`) |
| `mark_density(...)` | KDE | `bandwidth` (`"scott"`\|`"silverman"`\|float), `kernel`, `extent`, `cumulative`, `multiple` (`"layer"`\|`"stack"`\|`"fill"`\|`"dodge"`, default `"layer"`) |
| `mark_histogram(...)` | Frequency / density bins | `bins` (`"sturges"`\|`"fd"`\|`"freedman_diaconis"`\|`int` bin count\|`float` bin width, 2026-09-18 — mutually exclusive with `bin_count`/`bin_width`), `bin_count`, `bin_width`, `density`, `cumulative`, `right`, `multiple` (`"layer"`\|`"stack"`\|`"fill"`\|`"dodge"`, default `"layer"`), `position_by` (column name the `multiple=` position adjustment groups by, 2026-09-18) |
| `mark_contour(...)` | 2D density contours | `bandwidth`, `thresholds`, `smooth` |
| `mark_violin(...)` | KDE + optional boxplot overlay | `bandwidth`, `inner` (`"box"`, `"quartile"`, `"point"`, `None`) |
| `mark_qq(...)` | Quantile–quantile | `distribution` (`"normal"`, `"uniform"`, `"exponential"`, `"lognormal"` — the latter newly computed 2026-09-05, batch D F-L03-07: normal theoretical quantiles exponentiated; it was documented but refused before — or `scipy.stats` dist), `dequantize` |
| `mark_raster(...)` | Aggregates data to a pixel-resolution grid and renders as a colored image | `aggregate` (`"count"`, `"density"`, `"mean"`, `"sum"`, `"any"`), `field` (required if `aggregate` is `"mean"` or `"sum"`), `cmap`, `resolution` (`"screen"`, `int`, `tuple[int, int]`), `blend` (`"alpha"`, `"additive"`), `min_count`, `log_scale` |
| `mark_swarm(...)` | Beeswarm for categorical data. Points are arranged along the categorical axis to avoid overplotting while preserving the value distribution. General-purpose; distinct from `mark_shap_beeswarm`. | `size`, `orient`, `spacing`, `side` (`"both"`\|`"left"`\|`"right"`), `dodge` |
| `mark_hex(...)` | Hexagonal binning. Aggregates data into a hexagonal grid and renders each hex colored by aggregate value. Preferred over `mark_raster` when topology and shape perception matter. | `bin_size`, `aggregate` (`"count"`\|`"mean"`\|`"sum"`), `field` (required if `aggregate` is mean or sum), `cmap`, `stroke`, `stroke_width` |
| `mark_function(fn, ...)` | Plot a Python callable f(x) → y as a line. `fn` receives a numpy array of x values and must return a numpy array of y values. Domain inferred from X scale if not specified. | `domain` (`tuple[float, float]`), `n` (evaluation points, default 200), `clip` |

> **Dated note (2026-09-06, batch E task 11 — F-L03-04 + verified sibling (boxen's median rule, see task 10 above), `mark_violin(inner="quartile")` ticks confine to the violin body instead of spanning the panel):** `inner="quartile"`'s three `q1`/`median`/`q3` layers used `mark="rule"`, which (per `rule.rs`'s documented shape table) draws a full-panel span for a bare `{x, y}` pair — every quartile tick ran the width of the entire plot area regardless of how many categories shared it, the same root cause as the boxen median-rule bug (T10). They are now `mark="tick"` with `band_size=0.8` (`_VIOLIN_QUARTILE_BAND_SIZE`), confined to the violin's own body width — verified live: an un-hued two-category violin's quartile ticks each render at 226.6px (80% of that category's own band) against a 566.5px full panel width, not the panel's full width. `q1`/`q3` keep their dashed strokes; the median tick is solid, matching the pre-fix visual distinction. With a hue split (`color_field` distinct from the categorical axis) the ticks are **not** per-hue dodged — nothing under `inner=` dodges — so they span the shared category band rather than each narrower per-hue violin body; this is an existing, unchanged limitation, not new drift from this fix.

> **Dated note (2026-09-06, batch E task 4 — F-L03-09, archive-claim-verified provenance item 2, `mark_contour`'s smoothing/level ordering flips and fill mode stops leaving a permanent hole at the density peak):** two related bugs in the `Contour` transform (`contour.rs`). First, levels were computed from the **raw** density grid, and only afterward did the `smooth=true` Gaussian pass (a 3×3 kernel, center weight 4, edge weight 2, corner weight 1, over 16) replace the array the contour bands actually clip against — so under the default `smooth=true`, the isolines/bands were drawn at thresholds computed for a *different* surface than the one being contoured. Smoothing now runs **before** level computation, so both derive from the same (post-smoothing) surface; a chart with `smooth=false` is unaffected (there is no smoothing step to reorder), but a `smooth=true` chart's isolines and filled bands shift to their correct positions — a disclosed, visible change on every default (`smooth=true`, the default) contour/density-2D chart. Second, fill mode's loop built only `thresholds − 1` interior bands between consecutive levels, leaving the region above the highest level — the density peak, always the mode of the distribution — permanently unfilled, a hole no `thresholds` value could close; `thresholds=1` additionally rendered nothing at all (fewer than two levels to pair). The loop now emits `thresholds` bands: the same `thresholds − 1` interior bands plus a final top band `[levels[thresholds−1], dmax]`, so `thresholds=1` now yields exactly one meaningful band (`[levels[0], dmax]`) and every fill-mode contour's darkest band now actually covers the peak — verified live (`mark_contour(thresholds=1, fill=True)` renders a non-empty path where it previously rendered nothing). The top band's `level_value` is its own lower bound (`levels[thresholds−1]`), consistent with every other band, and — because that value did not previously appear as any band's `level_value` at all — the color legend/colorbar keyed off `level_value` now reports a higher true maximum than before, reflecting the real peak instead of the second-highest threshold. Isoline (segment/`groupby`) mode is unaffected by the fill-loop change; it only ever drew lines at the levels themselves, which the ordering fix still corrects under `smooth=true`. `mark_density`'s bivariate mode (`fill=True` equivalent) shares this code path and moves identically. Filled-contour and joint-plot goldens that move go through the PNG-inspection protocol (`tests/_snapshots.py::regen_and_verify`); the `resvg-py` dense-path truncation caveat applies when inspecting them.

> **Dated note (2026-09-06, batch E tasks 9/11 — F-L02-04, `mark_function(clip=False)` goes from a hard raise to a real per-batch clip escape):** `clip=False` previously raised — the parameter existed on the signature but any value other than the default `True` was rejected outright, contradicting the "toggle" framing its own docstring gave it. It is now a genuine toggle: `desugar_function` threads `clip=False` onto the synthesized line layer as `MarkKwargsSpec.clip = false`, carried by the scene graph's new `MarkBatch.clip: bool` field (serde-default `true`, skipped when `true` — every scene not using this escape serializes byte-identically) into both the SVG walker (the batch renders outside the panel's clip group, the same mechanism text/label batches already use to render outside the clip) and the WASM walker (scissor/clip skipped for that batch). The visible effect: with `clip=False`, `fn`'s curve extends past the panel edge for any evaluation point outside the resolved x-domain rather than being cut off at the panel boundary — verified live (`clip=False` and `clip=True` renders of the same chart produce different SVG bytes). `fn` is evaluated over the same resolved x-domain regardless of `clip` — clipping is purely a rendering property, not a domain-narrowing one. `clip=True` (the default) is byte-identical to every render from before this escape existed.

**Auto-raster (`mark_raster` as a policy):**

`mark_raster` is the explicit form of density rasterization. Auto-raster is a policy layer that implicitly substitutes `mark_raster` for the original mark when the per-layer mark count exceeds a threshold (`raster_threshold`, see §3.16 and §3.18).

When auto-raster fires:
- Color encodings on the original layer are dropped and replaced with a density colormap.
- Tooltip encodings are dropped.
- A warning is emitted (configurable via `raster_behavior`).
- A density colorbar legend replaces the original legend.

Auto-raster will **not** fire if the chart has an active color encoding — doing so would silently discard user intent. Either remove the color encoding or use `mark_raster` explicitly.

Auto-raster behavior is configurable via `raster_behavior`: `"warn"` (default), `"silent"`, or `"error"`.

> **Dated note (2026-09-11, batch H task 10 — F-L14-04, the color exclusion above splits by color type):** a **quantitative** color encoding (`:Q`, or `type="quantitative"`) no longer blocks auto-raster — it now substitutes `mark_raster(aggregate="mean", field=<the color field>, cmap=<the color channel's own `scheme=` if declared, else `raster_scheme`>)`, so a large scatter with a continuous color mapping still rasterizes past `raster_threshold` instead of silently staying vector (and, at very large N, silently ballooning past the size guard). A **categorical/ordinal/temporal** color encoding keeps the exclusion above exactly as stated — permanently, not a temporary gap — since no pixel aggregate preserves a category's identity; the emitted warning now names the field, the reason, and the manual `mark_raster(aggregate=..., field=...)` workaround. An explicit `RenderConfig(raster_aggregate=..., raster_field=...)` wins wholesale over the color-driven default regardless of the color channel's type — the color field is ignored entirely when the user has already named an aggregate/field.

> **Dated note (2026-09-11, batch H close remediation — the color type is *inferred* when unannotated):** the split above is decided by the color channel's resolved type, and that resolution reads the column dtype when the user declared no type: `color="z"` on a numeric column is quantitative and substitutes exactly like `color="z:Q"`, while `color="c"` on a string column stays excluded. An explicit annotation always wins (`color="z:N"` on a numeric column keeps the exclusion), and a field whose type can be neither annotated nor inferred (a missing column, or a non-polars data source) stays excluded too, with the warning describing it as untyped rather than asserting a type the gate never determined. Previously only the explicit `:Q` spelling substituted, so the untyped spelling — the one most user code and most of these docs write — fell into the categorical branch and was reported as categorical even for a plain float column. The positional `x`/`y` eligibility gate is unchanged: it still requires explicit quantitative annotations.

> *(2026-05-10) Phase 8b: `blend="additive"` is deferred to Phase 11 (interactive renderer). Auto-raster policy (`raster_threshold`, `raster_behavior`) is deferred to Phase 9+; explicit `mark_raster` is implemented in Phase 8b.*

> *(2026-05-10) Phase 8b: `mark_swarm` `dodge=` parameter deferred (single-group swarm only in 8b).*

> **Dated note (2026-09-05, batch D task 5 — M2, `mark_swarm(size=/spacing=)` and `mark_raster(resolution="screen")` go from inert to live):** Both parameters were **fully inert in every real render** from Phase 8b through v0.21.0: the one production transform-execution site (`render/prepare/mod.rs::run_transforms_with_facet`) always constructed `TransformContext::default()`, so `panel_pixel_size` was `None` and `swarm` silently fell back to a fixed placement radius (`radius_data = 1.0`, ignoring `size`/`spacing` entirely) while `raster(resolution="screen")` silently fell back to a fixed 256×256 grid regardless of chart size. Both now respond to the chart's actual dimensions: the production pipeline seeds `TransformContext.panel_pixel_size` with a **pre-layout ESTIMATE** — the viewport minus the theme's padding, divided per-axis by the facet grid's column/row counts when the chart facets — computed once per render and threaded to every transform. This estimate is *not* the exact rendered panel size: it deliberately excludes axis titles, tick labels, legends, and facet strip bands, all of which are unknowable before layout runs (true panel size is circular pre-layout — layout itself measures the transformed output this estimate seeds). Consequently `mark_swarm(size=N)` and `mark_raster()`'s default `resolution="screen"` now genuinely respond to the chart dimensions, closing F-L03-05/F-L03-06 — but in distinctly different ways. Raster's grid dimensions scale UP as the panel grows (a bigger panel gets a bigger grid). Swarm responds **only to its value-axis panel dimension — height for the default vertical orientation, width for `orient="horizontal"` — and inversely** (corrected 2026-09-06; the original note measured vertical only and claimed width was always inert): a longer value axis produces a smaller pixel spread at the same `size=`, since less cross-axis displacement is needed to avoid collisions. The cross-axis panel dimension has no effect on swarm's placement. Both are close approximations to, not exact matches for, the plot area a user might measure by hand from the rendered SVG. An explicit `mark_raster(resolution=<int>)` or `resolution=(nx, ny)` is unaffected (it never reads the estimate). `violin.rs`'s pre-existing `band_pixels` estimate (used for the inner-quartile/point overlay's positional scale) also improves under the same change, with no new parameter — it read the same field before and still does. A direct `apply()`/`apply_with_context(TransformContext::default())` call outside the render pipeline (unit tests, notebook/WASM direct usage) keeps the old fallbacks (256×256 for raster, `radius_data = 1.0` for swarm, `band_pixels = 100.0` for violin) unchanged. Any golden whose chart's pipeline includes swarm, `raster(resolution="screen")`, or violin may render with different (not necessarily wrong) pixel geometry than before this fix; every other chart's output is byte-identical. A post-layout transform re-run (the only way to get the *exact* panel size instead of an estimate) is a filed north-star follow-up, not built here — see `.claude/output/decisions/2026-09-05-batch-d-transform-engine-decision.md`'s M2 entry.*

> **2026-05-12 (Phase 9, F16 audit):** Type inference for the color
> channel follows the same rule as the axis scales: continuous color is
> selected when the `type=` argument on `Color(...)` is `"Q"` /
> `Quantitative` or `"T"` / `Temporal`, or when `type=` is `None` and
> the column dtype is numeric (any width: Float32/64, Int8/16/32/64,
> UInt8/16/32/64) or temporal (Date32/64, Timestamp). All other cases
> fall through to a categorical color scale. A user-supplied
> `type="quantitative"` on a non-numeric column raises
> `EncodingTypeMismatch` per the line-52 "no magic inference that
> silently fails" rule.

> *(2026-05-10) Phase 8b: `mark_hex` only count/mean/sum aggregates supported. Other Vega-Lite aggregates warn-once and fall back to count.*

**Bivariate density:** When both X and Y channels are encoded as quantitative fields, `mark_density` switches to bivariate KDE mode and renders filled contours, equivalent to `mark_contour` with `fill=True`. The `multiple` parameter applies to univariate mode only.

#### Model Diagnostic Marks

| Mark | Description | Key Parameters |
|---|---|---|
| `mark_residuals(...)` | Residuals vs fitted | `kind` (`"raw"`, `"studentized"`, `"scaled"`), `reference_line`, `cook_threshold` |
| `mark_prediction_error(...)` | Actual vs predicted | `identity_line`, `ci`, `reference_band` |
| `mark_confusion(...)` | Confusion matrix heatmap + text | `normalize` (`None`, `"true"`, `"pred"`, `"all"`) — see 2026-08-27 note below: informational-only at the mark layer, now warns; `text_fmt` |
| `mark_roc(...)` | ROC curve line | `average` (`None`, `"micro"`, `"macro"`, `"weighted"`), `reference_line`, `annotate_auc` |
| `mark_pr(...)` | Precision-recall curve | `average`, `annotate_ap`, `iso_lines` |
| `mark_calibration(...)` | Calibration curve | `reference_line` (see 2026-08-27 note below: `n_bins`/`strategy` removed) |
| `mark_gain(...)` | Cumulative gain curve | `reference_lines` |
| `mark_lift(...)` | Lift curve | `reference_line` |
| `mark_importance(...)` | Feature importance bar | `orient`, `error_bars`, `top_k` |
| `mark_shap_beeswarm(...)` | SHAP beeswarm | `max_display`, `color_bar`, `order` (`"abs_mean"`, `"mean"`, `"max"`, `"none"`) — see 2026-08-27 close-out note below: unified with the `shap_chart` figure-function family's `order=` vocabulary (union, not a narrowing) |
| `mark_shap_bar(...)` | SHAP mean absolute bar | `max_display`, `layered` |
| `mark_shap_waterfall(...)` | SHAP waterfall (single prediction) | `max_display`, `show_data` |
| `mark_pdp(...)` | Partial dependence + ICE | `kind` (`"average"`, `"individual"`, `"both"`), `ice_alpha`, `center` — see 2026-08-27 note below: informational-only at the mark layer, now warns |
| `mark_silhouette(...)` | Silhouette plot per cluster | `line_width`, `zero_line` |
| `mark_learning_curve(...)` | Learning curve with CI band | `ci_style` (`"band"`, `"errorbar"`) |
| `mark_validation_curve(...)` | Validation curve with CI band | `log_scale`, `ci_style` |
| `mark_decision_boundary(...)` | 2D classification boundary | `grid_resolution`, `alpha`, `background`, `contour_levels` — see 2026-08-27 note below: `proba` is informational-only at the mark layer and now warns |
| `mark_discrimination_threshold(...)` | Precision, recall, F1, and queue rate vs decision threshold for binary classifiers. Useful for threshold selection under class imbalance. | `metrics` (list of metrics to display, default all four), `n_thresholds`, `threshold_line` (bool, marks estimated optimal threshold) — see 2026-08-27 note below: `n_thresholds` is informational-only at the mark layer and now warns |
| `mark_parallel_coordinates(...)` | Parallel coordinates plot. Each sample is a polyline drawn across vertically-arranged feature axes. General-purpose: accepts any tabular data, not only model output. | `rescale` (`"minmax"`\|`"zscore"`\|`None`), `alpha`, `highlight_selection` (bool) |
| `mark_class_prediction_error(...)` | Stacked bar chart of predicted class counts, colored by actual class. Distinct from confusion matrix: shows absolute prediction volume per class and reveals systematic over/under-prediction. | `orient`, `normalize` (bool) |
| `mark_pca_scree(...)` | Bar chart of explained variance ratio per principal component with optional cumulative variance line overlay. | `cumulative_line` (bool, default `True`), `threshold_line` (float, draws a horizontal line at this cumulative variance level, e.g. 0.95) — see 2026-08-27 note below: `n_components` removed |
| `mark_rank1d(...)` | Horizontal bar chart of feature scores from a univariate ranking algorithm. | `algorithm` (`"shapiro"`\|`"variance"`\|`"covariance"`), `orient`, `top_k` |
| `mark_rank2d(...)` | Heatmap of pairwise feature correlations or covariances. | `algorithm` (`"pearson"`\|`"spearman"`\|`"kendall"`\|`"covariance"`), `annot` (bool), `cmap` |
| `mark_intercluster_distance(...)` | MDS or t-SNE projection of cluster centers, with each center sized proportionally to its membership count. Completes the clustering diagnostic suite alongside `mark_silhouette`. | `method` (`"mds"`\|`"tsne"`), `min_size`, `max_size`, `label_clusters` (bool) |
| `mark_cv_scores(...)` | Box, bar, or strip plot of cross-validation score distributions per fold. | `kind` (`"box"`\|`"bar"`\|`"strip"`), `split` (`"test"`\|`"train"`\|`"both"`) |
| `mark_alpha_selection(...)` | Mean CV score vs regularization parameter, single line (no CI band). Intended for Ridge, Lasso, ElasticNet. Distinct from `mark_validation_curve` in that it assumes a log-spaced alpha domain by default. | `log_scale` (bool, default `True`), `highlight_best` (bool) — see 2026-08-27 note below: `ci_style` removed |

> **2026-05-11 (Phase 10 — Model Diagnostics):** Per-mark clarifications surfaced during
> the 10a–10g implementation:
> - `mark_residuals(kind="studentized")` uses the leverage-aware hat-matrix definition
>   when `X` is supplied to the underlying `studentized_residual` helper (linear
>   estimators only). For non-linear estimators ferrum falls back to internally
>   studentized residuals (divide by `std(r, ddof=1)`).
> - `mark_confusion`: color scale on `value` uses Phase 8b's continuous color scale
>   (`viridis` default). Per-cell `value_fmt` (Utf8) is pre-computed by
>   `ModelSource.confusion_matrix` so the renderer's text layer lays out short labels
>   without invoking number formatting per cell.
> - `mark_decision_boundary`: requires exactly 2 features; the figure-level
>   `decision_boundary_chart` fixes the unused features at their column means.
> - `mark_shap_*`: require the `shap` library, installable via `ferrum[shap]`. The
>   `shap_waterfall` mark requires an explicit `sample_idx: int` kwarg.
> - `mark_rank2d(algorithm="kendall")` is the sole Phase 10 path that crosses into
>   Rust at compute time — it calls `ferrum._core.kendall_tau_b` (Knight's
>   O(n log n) merge-sort variant). All other rank2d algorithms run in NumPy.
> - `mark_pca_scree(cumulative_line=True)` overlays a `mark_line` on the
>   `cumulative_variance_ratio` column. Layer-0 is the cumulative line so its
>   wider y range drives axis-scale resolution; the rect-bar layer follows.
> - `mark_intercluster_distance` uses the `size` channel for cluster cardinality;
>   the chart builder pads the x/y domain by 15% so large bubbles don't clip.
> - `mark_parallel_coordinates`: routes `sample_id` through `mark_style.detail` so
>   each sample renders as its own polyline. Requires the Phase 10g `mark_line`
>   composite (color, detail) grouping path and ordinal-x support.

> **2026-06-22 (annotation default divergence — intentional):** `mark_roc`
> defaults to `annotate_auc=False`, `mark_pr` to `annotate_ap=False`, while the
> figure functions `roc_chart` / `pr_chart` (and `calibration_chart`'s
> `annotate_brier`) default to `True`. This divergence is by design: a raw
> primitive mark does not auto-annotate, whereas a figure function is a
> finished plot that should carry its metric. The figure builders own the
> overlay (calling `mark_roc(annotate_auc=False)` then layering the metric label
> themselves), so the AUC/AP/Brier value and its overlay-text formatting come
> from a single source — the metric-kind table in `ferrum._metric_labels`
> (`_METRIC_LABEL_SPECS`), shared by both the direct-mark `AUCLabel`/`APLabel`/
> `BrierLabel` `__radd__` path and the figure-function explicit-field path.

> **2026-09-10 (batch G, F-L11-01):** every AUC/AP a user sees — the
> `roc_chart`/`pr_chart` title, `_inject_metric_into_legend_labels`'s legend
> text, and the mark-level `AUCLabel`/`APLabel` overlay — now reads the same
> number, sourced from one place. `roc_chart`/`pr_chart`'s title and legend
> read the frame's Rust-kernel-computed `auc`/`ap` column (one value per
> curve/group, already broadcast) instead of re-integrating the curve in
> Python; for `per_class=False` (averaged) titles the column value is the
> **weighted mean of per-class AUCs/APs** (sklearn parity), not a trapezoid
> over the interpolated averaged curve — a different, and in general
> different-valued, computation. `_trapezoid_auc`/`_ap_step` (the composable
> `+ AUCLabel()`/`+ APLabel()` path on arbitrary user data, which has no
> metric column to read) now sort with `kind="stable"`; ROC/PR curves
> routinely repeat one x/recall value across several rows, and an unstable
> sort's within-tie permutation previously changed the integral (observed:
> AUC 0.8571 unstable vs. 0.8543 stable on a tie-heavy fixture). The other
> unstable `np.argsort` sites in diagnostics frame assembly — PR macro
> interpolation, gain/lift score ordering — are stable too, so tied-score
> curves no longer depend on sort internals.
>
> Separately: `mark_roc(annotate_auc=True)`/`mark_pr(annotate_ap=True)` now
> render their metric label instead of raising. `_inject_curve_annotation`
> (previously dead code with zero `src/` callers) is wired as the injector
> the mark desugar path calls when the `_auc_label_*`/`_ap_label_*` columns
> its text layer binds are absent: it reads the frame's `auc`/`ap` column
> when present, else computes via the same stable-sorted helper, on raw user
> data. Before this fix, `mark_roc(annotate_auc=True)` on user data raised
> `layer N: unknown column '_auc_label_x'` under the batch-F layer gate
> (silently rendered nothing before that gate existed). Disclosed: AUC/AP
> values move wherever the old unstable re-derivation was wrong — verified
> against the kernel column independently on the phase-10 golden fixtures
> (seven goldens regenerated and PNG-inspected).

> **2026-09-13 (wave 3, #219 / D4 — one metric surface; supersedes the two
> notes above wherever they say the title or the legend carries a metric):**
> `annotate_auc` / `annotate_ap` / `annotate_brier` now drive **exactly one**
> surface, the per-curve endpoint overlay. Two surfaces retire:
>
> - The chart title's metric suffix. `roc_chart` is titled `"ROC Curve"`,
>   `pr_chart` `"Precision–Recall Curve"` and `calibration_chart`
>   `"Calibration Curve"` in every shape — single-curve, per-class, averaged,
>   compared and precomputed alike. The single-vs-multi-curve title split and
>   `calibration_chart`'s compared-source title guard are gone with it: there
>   is no longer a title branch for them to guard.
> - The legend rename. A ROC/PR legend entry is the bare class or model name
>   (`"1"`, `"macro"`, `"a"`), not `"1 (AUC = 0.984)"`.
>
> This is the repo's own Schwabish T2 rule (a direct label replaces the
> legend's copy of the number) applied to the three curve families, and it is
> what makes `annotate_*=False` mean what it says: the title used to print the
> metric regardless of the flag, so opting out never removed the number from
> the figure. It now does — measured metric text runs across all eleven
> phase-10 goldens this moved: `roc_chart_binary` and
> `roc_chart_binary_annotate_auc` 3 → 1; `roc_chart_multiclass` and
> `roc_chart_multiclass_annotate_auc` 6 → 3; `roc_chart_compare_two_models`
> 4 → 2; `pr_chart_binary`, `pr_chart_binary_annotate_ap` and
> `pr_chart_binary_iso_lines` 3 → 1; `pr_chart_multiclass_annotate_ap_iso`
> 6 → 3; `pr_chart_multiclass_macro_average` 3 → 1;
> `calibration_chart_binary` 2 → 1. No metric *value* changes; only how many
> times each is printed.
>
> Two further contract changes ship with it:
>
> - **Re-applying a metric label replaces it.** `roc_chart(...) + AUCLabel()`
>   used to stack a second text layer carrying the same string at the same
>   coordinates (an exact overstrike, since the figure already places one).
>   The new label now supersedes the old one, so a re-application is how a
>   caller changes `position=` / `format=` / `prefix=` on a figure's label
>   rather than a way to double it.
> - **`BrierLabel` on a frame with no `brier` column refuses.** A reliability
>   diagram's binned `(mean_predicted, fraction_positive)` points do not define
>   a Brier score — integrating them gives a bin-level statistic that moves
>   with `n_bins=` / `strategy=`. That silent fallback is deleted; the label
>   raises a `ValueError` naming the missing column. `AUCLabel` / `APLabel`
>   are unaffected: the area under the drawn curve *is* their metric, so the
>   composable path over arbitrary user data keeps integrating.
>
> One owner now produces every user-visible metric string
> (`ferrum._metric_labels._render_metric`), taking the display name and the
> `.3f` format from the kind's row of `_METRIC_LABEL_SPECS`; the six sites
> that each carried their own literal are gone or consume it. The mark-level
> annotation (`mark_roc(annotate_auc=True)` / `mark_pr(annotate_ap=True)`)
> keeps its `"{group}: AUC = 0.987"` placement and text byte-for-byte — it was
> already a single surface for the direct-mark path — but composes that text
> from the same owner.

> **2026-09-10 (batch G close — binary label conventions):** every binary
> classification diagnostic now binarizes the target against the positive
> class before the Rust kernels see it, so a target's *labels* no longer
> change its metrics. The kernels count positives by summing `y_true`, so
> they require a 0/1 indicator; the multiclass arms always supplied one
> (via `one_hot`), but the binary arms of `roc_curve`/`pr_curve` and all of
> `calibration_curve` passed the raw target straight through. **Disclosed
> behavior change, all of it corrections:** `y ∈ {-1, 1}` (the SVM
> convention) rendered `"ROC Curve — AUC -1.747"` with `tpr` ranging to
> -4.375 against a true AUC of 0.9994; `y ∈ {1, 2}` rendered `AUC 0.302`
> against 0.9989 and a Brier of 1.035 (a Brier score is bounded in
> `[0, 1]`); string labels raised `could not convert string to float`
> outright, from `roc_chart`, `pr_chart`, `calibration_chart` and the
> precomputed `y_true=`/`y_pred=` path alike. All four now produce the same
> numbers the `{0, 1}` labelling produces for the same probabilities.
> `{0, 1}` targets — every fixture and golden in the repo — are
> byte-identical: `y == positive_class` is the identity there. The
> precomputed path additionally stops comparing against the *string* `"1"`:
> its binary class values are now read from `y_true` in `y_true`'s own
> dtype (ascending, larger value positive, matching
> `discrimination_threshold`'s long-standing rule), and a degenerate
> single-class `y_true` labels its curve with the class actually present
> instead of a fabricated `"1"`.
>
> Relatedly, a present-but-NaN metric column now **propagates** instead of
> triggering the integration fallback. "The frame carries no such metric"
> and "the metric is undefined for this curve" are different facts;
> collapsing them turned the one honest signal a degenerate (single-class)
> curve emits into a plausible-looking number integrated from the broken
> curve. Such a chart now renders `AUC = nan` rather than a fabricated
> figure. Frames with no metric column at all are unaffected.

> **2026-09-10 (batch G close — `confusion_matrix` predictions):** the same
> display-label-vs-comparison-value rule reaches the third of its call
> sites. `confusion_matrix()` coerced its *encode values* into `y`'s domain
> but left `model.predict`'s output in `classes_`'s, so an int-coded
> `classes_` against a `Utf8`/`Categorical`/`Enum` `y` — the ordinary shape
> for a string-labelled target — encoded every predicted value to the
> kernel's out-of-vocabulary code and produced a matrix of **all zeros**,
> rendered as such by `confusion_matrix_chart` and
> `ConfusionMatrixVisualizer`. Predictions are now translated through the
> same positional `classes_ -> y`-domain correspondence that builds the
> encoding, so the two cannot disagree. Numeric `y` is byte-identical (the
> translation is the identity). Found by the close's own dtype matrix, not
> by a user report.

> **2026-09-13 (wave 3, #203 — the target-hygiene matrix):** supersedes the
> "scope is deliberately narrow" clause of the 2026-09-10 note below. A
> target that disagrees with the model is now answered by one contract
> instead of by the target's dtype and the model's resolution tier.
>
> What the contract covers today: the seven `ModelSource` classification
> methods — `roc_curve`, `pr_curve`, `calibration_curve`, `cumulative_gain`,
> `lift_curve`, `discrimination_threshold`, `confusion_matrix` — and every
> figure function built on them (`class_prediction_error_chart` included,
> since it renders the confusion frame); plus the eight classification
> methods of the precomputed (`y_true=`/`y_pred=`) path: `roc_chart`,
> `pr_chart`, `calibration_chart`, `gain_chart`, `lift_chart`,
> `discrimination_threshold_chart`, `confusion_matrix_chart`, and
> `classification_report_chart`; plus, since 2026-09-13 (wave 3, #206), the
> **model-backed** `classification_report_chart` and
> `ClassificationReportVisualizer`, previously the one classification
> surface outside this contract -- narrowly: only the null cell, below.
> Its model path now runs the same null-target guard before predicting, so
> a null-bearing `y` refuses in this contract's own typed voice instead of
> leaking scikit-learn's `ValueError: Input y_true contains NaN` (numeric)
> / `Classification metrics can't handle a mix of unknown and binary
> targets` (`Utf8`), and both classification surfaces now answer a null
> identically. The model path does **not** run the stray-value check
> (`check_resolved_target`'s absorb-and-warn cell) -- sklearn's own
> `classification_report` resolves its class union directly from whatever
> `y_true`/`y_pred` it is handed, with no comparison-value resolution step
> for that guard to attach to. It **does**, since 2026-09-13 (#206 cycle 5,
> below), run a narrower total-mismatch check of its own
> (`check_translated_classes_match_y`): when the model's `classes_`,
> translated into `y`'s domain, match none of `y`'s observed values, the
> call refuses instead of rendering an all-zero report in silence. One
> consequence, disclosed rather than fixed: a model exposing no `classes_`
> still hands sklearn `model.predict`'s raw, untranslated output (there is
> no positional correspondence to translate through, and therefore nothing
> for the total-mismatch check above to translate either -- see the
> 2026-09-13 #206 note below), so a non-numeric `y` against such a model's
> numeric predictions can still surface sklearn's own untyped
> `ValueError: Mix of label input types (string and number)` -- a narrower
> instance of the exact failure #206 closes for the `classes_`-present
> case, tracked as a follow-up rather than closed here.
>
> One classification surface is **outside** the contract entirely, stated
> here so it is not read wider than it is: `class_balance_chart` counts
> `y`'s own values rather than comparing them against a model, so no class
> resolution happens there to disagree with.
>
> - *Nulls refuse.* A target carrying any null row raises `ValueError`
>   naming the diagnostic invoked, the null count out of the row count, and
>   `y`'s observed values. A null is not a class observation, so counting it
>   as a negative would fabricate data — `REFUSE MALFORMED` governs, and
>   scikit-learn refuses a NaN-bearing `y` too. This replaces three
>   different old behaviors: on a `Utf8`/`Categorical`/`Enum` target it
>   replaces an unowned `TypeError: '<' not supported between instances of
>   'str' and 'NoneType'` raised from whichever of four internal
>   `np.unique`/`sorted` sites the call reached first; on a numeric target
>   it replaces silence, where the nulls arrived as `NaN`, became a third
>   distinct class, and shifted the metric (measured: AUC 0.9395 against a
>   clean 0.9843 on a 200-row fixture with 10 nulls) or pushed a
>   `classes_`-less model from resolution tier 2 to tier 3; and on the
>   precomputed `confusion_matrix_chart` / `classification_report_chart` it
>   replaces rendering `None` as a class of its own (their
>   `sorted(..., key=str)` stringified it). "Null" means null in every dtype
>   family a polars target can arrive in, including the temporal one: a
>   `Date`/`Datetime`/`Duration` column's nulls arrive as `NaT` rather than
>   as `NaN` or `None`, and until this note they were seen by nothing —
>   `confusion_matrix_chart` rendered the `NaT` as a class of its own
>   (measured 2026-09-13), the same silent accept as the string case above.
>   One spelling stays outside: a *pandas* nullable target (`pd.NA`) never
>   reaches this guard at all, because ferrum's polars ingestion refuses it
>   first with its own untyped `TypeError: unexpected value while building
>   Series of type String; found value of type Object: <NA>`. That is an
>   ingestion-layer gap, not a silent accept.
> - *Stray values warn.* Rows holding a value outside an otherwise-matching
>   class set now emit one `UserWarning` per computed call naming the
>   diagnostic, how many such rows there were, and which values they held.
>   What happens to those rows is unchanged, and it is not the same on both
>   families, so the two messages say different things. On a **curve**
>   (`roc_curve`, `pr_curve`, `calibration_curve`, `cumulative_gain`,
>   `lift_curve`, `discrimination_threshold`, and their charts) the row
>   still counts as negative — scikit-learn's own `pos_label` semantics and
>   the right arithmetic — and the warning says *counted as negative*
>   (measured: the 200-row fixture with 10 rows set to a stray `7` returns
>   AUC 0.9395, and says so). On the **confusion matrix** (`confusion_matrix`,
>   `confusion_matrix_chart`, `class_prediction_error_chart`) the kernel
>   maps the row to its out-of-vocabulary code and **drops** it — it is
>   counted in no cell, not absorbed into one — so the warning says
>   *dropped*. That drop is not new; the silence around it was total, and
>   measured (2026-09-13): cells summing to 190 of 200 rows on the binary
>   fixture and 290 of 300 on the multiclass one, with no diagnostic at all,
>   while `roc_curve` on the identical source warned.
> - *A binary pair whose positive class matches no row says nothing extra.*
>   `binary_indicator` refuses it a moment later, so the call returns no
>   number; a stray warning ahead of that refusal would assert an absorb
>   that never happened, and one used to (measured with `classes_ = [0, 2]`
>   against a `{0, 1}` target: "96 row(s) … were counted as negative",
>   immediately followed by the refusal). The refusal is the whole answer.
> - *A resolution nothing matches refuses, on both reaches.* `cumulative_gain`
>   and `lift_curve` join the total-mismatch refusal every other binary
>   builder already had: a binary target holding no positive row used to
>   yield a gain curve reaching 1.0 and a lift chart carrying only its
>   baseline. A multiclass resolution where **zero** resolved class values
>   match any row also refuses, closing the silent all-NaN case the resolver's
>   own docstring called a known gap (measured: three NaN curves where
>   sklearn's one-vs-rest AUCs were 0.872 / 0.942 / 0.930).
> - *Unchanged.* Multiclass one-vs-rest still tolerates *some* classes being
>   absent — "no row is class k" is a real answer, and `lift_curve` still
>   skips a zero-base-rate one-vs-rest class. An empty target is still not an
>   error, an all-positive target still yields the kernels' honest NaN, and
>   the request-vs-slice rule for `discrimination_threshold(cv=...)` below
>   still holds. A clean target is byte-identical everywhere: no golden moved.
> - *One asymmetry, by construction.* A stray value on a `classes_`-less
>   **binary** model still refuses rather than absorbing. The stray inflates
>   `y`'s cardinality past the score-column count, so resolution falls to the
>   parsed `class_{i}` labels, which match no row at all — a total mismatch,
>   not a stray among valid rows. There is no class set to absorb into.

> **2026-09-13 (wave 3, #206 + #224a — classification_report dtype
> translation and class-order unification):** two `classification_report`
> fixes, both scoped to the model path's dtype handling and the class-axis
> render order; the confusion matrix and every other classification
> diagnostic are untouched.
>
> *#206 — `model.predict` dtype translation.* The model path (both
> `classification_report_chart`'s builder and
> `ClassificationReportVisualizer._materialize`'s `f1_macro`) now translates
> `model.predict`'s output into `y`'s domain through the SAME positional
> `classes_` ↔ `encode_values` correspondence the batch-G confusion fix
> built, before calling sklearn — reused, not re-derived. Before this, a
> `Utf8`/`Categorical`/`Enum` `y` fit against an int-coded `classes_` (the
> ordinary shape when a user label-encodes for training and keeps the
> readable target for plotting) made `y_true` and `model.predict`'s raw
> output disagree on type, and sklearn's own `unique_labels` raised
> `ValueError: Mix of label input types (string and number)` before any
> metric was computed — a different failure shape than confusion's silent
> all-zeros, since `classification_report`'s own `unique_labels` guard
> catches the type split where confusion's kernel just encoded everything
> to the out-of-vocabulary code. A numeric `y` is byte-identical (the
> translation is the identity); `class_prediction_error_chart` needed no
> change and was verified already-correct by transitivity — it consumes
> `ModelSource.confusion_matrix()` wholesale and never touches
> `model.predict` directly, so it inherited the batch-G fix already.
>
> **Disclosed, not closed by this fix.** The translation above applies only
> when the model exposes `classes_`; the no-`classes_` branch
> (`report_predictions_in_y_domain`'s documented passthrough, mirroring
> `confusion_matrix`'s own no-`classes_` fallback) hands `model.predict`'s
> raw output straight to sklearn unchanged, since there is no positional
> `classes_`↔`encode_values` correspondence to translate through. A
> non-numeric `y` against such a model's numeric predictions can therefore
> still surface sklearn's own untyped `ValueError: Mix of label input types
> (string and number)` (measured 2026-09-13 with a duck-typed
> `classes_`-less wrapper over a `Utf8` `y`) — the identical failure #206
> closes for the `classes_`-present case, surviving in this narrower one.
> Filed as a follow-up, not fixed here.
>
> *classification_report's model path now sits inside the target-hygiene
> contract (#203) too* — see the correction to the 2026-09-13 hygiene-matrix
> note above: a null-bearing `y` on the model path now refuses through the
> same `require_no_null_target` guard, naming the diagnostic and the null
> count, instead of leaking scikit-learn's own null error. The visualizer's
> own guard (`ClassificationReportVisualizer._materialize`) names
> `ClassificationReportVisualizer`, not the figure function it never calls
> directly, matching `require_sklearn("ClassificationReportVisualizer")`'s
> own naming two lines above it; and `ClassificationReportVisualizer(model)
> .fit(X)` (`y` omitted — `FerrumVisualizer.fit`'s own documented default)
> now refuses the same typed way instead of leaking `TypeError: 'NoneType'
> object is not iterable` out of `require_no_null_target`'s own
> `_null_mask` internals.
>
> *#224a — one class-axis order.* Both `classification_report_chart` paths
> (model and precomputed) now order the class axis by sklearn's
> `unique_labels` convention: the union of `y_true`/`y_pred`'s own distinct
> values, in their natural ascending order. The precomputed path's
> `sorted(..., key=str)` rule — inherited from `confusion_matrix`'s own
> fallback, and unchanged there — retires for `classification_report`
> specifically: it sorted the confusion matrix's already-stringified
> `actual`/`predicted` columns, which is decimal-string order — digit count
> before value, so `"10"` lands between `"1"` and `"2"` — where sorting the
> raw values first (matching sklearn's own `unique_labels`, which never
> stringifies before resolving) sorts ascending. **Breaking, disclosed:**
> this moves the rendered class-axis order for any precomputed
> `classification_report_chart` call whose numeric class labels do not sort
> the same way as their decimal strings. That condition is *unequal digit
> counts*, not a class count: **two** classes are enough. Measured
> 2026-09-13: `{2, 10}` rendered `10, 2` and now renders `2, 10`;
> `{1, 2, 10}` rendered `1, 10, 2` and now renders `1, 2, 10`; a 12-class
> target rendered `0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9` and now renders
> `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11`, matching the model path.
> (An earlier draft of this note scoped the change to "10 or more numeric
> classes" — that boundary was false, and a binary target labelled
> `{2, 10}` moves.) Class labels that all have the same digit count
> (`{0, 1, 2}`, `{10, 11, 12}`) are unaffected, as is any non-numeric
> (string/Categorical/Enum) target — a purely lexicographic domain sorts
> the same way either rule reads it. No golden exercises this path (no
> `classification_report` golden exists at all). The documented divergence
> test (`test_classification_report_chart_class_axis_order_documented_divergence`)
> is flipped to
> `test_classification_report_chart_class_axis_order_now_identical`,
> pinning identity at both the frame and the rendered SVG's y-axis tick
> order, with the sub-10-class move pinned alongside it
> (`test_precomputed_classification_report_order_moves_below_ten_classes`).
>
> *The class-axis label derivation is total by partition.* Sorting raw
> values is not total the way the retired `sorted(..., key=str)` rule was
> (that rule stringified first, which is why it accepted everything and
> ordered it wrongly), so `_curve_frames.report_labels` owns the whole
> input space in four named cells rather than patching shapes as they are
> reported: a **null in either array** refuses in the null guard's
> vocabulary, naming which array, through `_null_mask` so every dtype
> family's null spelling counts; a **mix of ordering families** refuses as
> a mix of label types, matching scikit-learn's own `unique_labels` stance
> (an `int`/`float` or `bool`/`int` union is one family, not a mix); a
> **homogeneous orderable** union is the answer; and a **homogeneous
> unorderable** one (`complex`, `dict`, a type with no `<`) refuses as
> unorderable rather than as a mix, so the message does not send the
> caller after a defect their input does not have.
>
> **Breaking, disclosed — three shapes that rendered now refuse.** Measured
> 2026-09-13 against the pre-fix code at this wave's branch point
> (`8bfb4bbb`), whose label expression — `sorted(set(actual) |
> set(predicted), key=str)` — is character-identical to v0.23.0's. These
> are before/after against what a user has, not against this task's own
> unreleased intermediate states (its first cycle's unowned `TypeError`
> never shipped):
>
> - `y_true=[0,1,0,1]` with `y_pred=["0","1","1","0"]` **rendered** classes
>   `'0'`/`'1'` with every metric 0.5, silently scoring string predictions
>   against integer targets; it now refuses as a mix of label types.
> - A null-bearing `y_pred` **rendered** a phantom class of its own:
>   `["a",None,"a","b"]` produced a `'None'` row at 0/0/0 with class `'b'`'s
>   recall at 0.5, and `[0.0,nan,0.0,1.0]` produced a `'nan'` row the same
>   way. Both now refuse, naming `y_pred` and the null count. (`y_true`'s
>   nulls already refused, via #203's `require_no_null_target`.)
> - `complex` class labels **rendered** (`'(1+1j)'`, `'(2+0j)'`, through the
>   stringified sort) and now refuse as unorderable; `dict` labels leaked
>   `TypeError: unhashable type: 'dict'` and now refuse by name too.
>
> `confusion_matrix_chart` did **not** move with any of these when #224a
> shipped: measured on the identical inputs, it still rendered the mixed
> union (16 cells), rendered a null `y_pred` as a `'None'`/`'nan'` class,
> and ordered its own axis `10, 2`. The asymmetry was deliberate — #224a is
> about the report's class axis, and widening these refusals to the
> confusion family moves a second figure's behavior — but it meant two
> figures answered one malformed input differently, which is what #203 set
> out to end. **Superseded in part on 2026-09-17; see the note directly
> below.**

> **2026-09-17 (wave 3 close remediation, finding 2 — the confusion family
> joins the cells it sat outside):** the wave's whole-change design review
> read the target-hygiene header's claim ("one contract ... applied by
> every binary and multiclass diagnostic on both source paths") against the
> disclosure above and returned it as a concern: the claim was not true, and
> the gap was user-visible inside one figure family. Three cells move, on
> **malformed inputs only** — clean targets are byte-identical and no golden
> moved:
>
> - **A null prediction refuses** on both `confusion_matrix`
>   implementations, model path and precomputed, through the same
>   `require_no_null_predictions` guard `classification_report_chart` uses,
>   in every dtype family a null arrives in. Measured 2026-09-13 on
>   `y_true=["a","b","a","b"]` with `y_pred=["a",None,"a","b"]`: the
>   `predicted` axis held `['None', 'a', 'b']`, a class nothing ever
>   predicted, and the row holding the null was scored against it; the
>   float twin (`[0.0, nan, 0.0, 1.0]`) rendered `'nan'` the same way. The
>   fabricated class is gone.
> - **A fully disjoint translated class vocabulary refuses** on
>   `confusion_matrix_chart` and `class_prediction_error_chart`, through
>   the same `check_translated_classes` guard the report path runs
>   (#206 cycle 5). `classes_ = [0, 1, 2]` against `y = ["c0", "c1", "c2"]`
>   translated every prediction to `"0"`/`"1"`/`"2"`, values `y` never
>   holds, so every row encoded out of vocabulary and the kernel dropped
>   all of them: the matrix rendered every cell as zero behind a warning
>   saying 300 of 300 rows had been dropped. A **partial** overlap still
>   warns and drops, unchanged — it is the ordinary stray-value case.
> - **`ModelSource.confusion_matrix`'s no-`classes_` branch derives its
>   class axis through `report_labels`**, closing the totality gap the note
>   above disclosed: the bare `sorted(set(...))` leaked
>   `TypeError: '<' not supported between instances of 'str' and 'int'` on
>   a mixed target and `TypeError: unhashable type` one statement later on
>   a `dict`-valued prediction; both now refuse by name, in the report
>   family's own voice. The order is unchanged for every input the old rule
>   accepted, so only the malformed cases move. T1's disposition for that
>   branch (its labels are the `y_true`/`y_pred` union, so no row can fall
>   outside them and it needs no stray-value guard) still holds.
>
> One thing was left standing at this point and is **superseded by the note
> directly below**: `_PrecomputedSource.confusion_matrix` kept its
> `sorted(..., key=str)` class order, described here as a presentation
> difference rather than a hygiene one. The close review's second cycle
> measured what that rule rendered and the characterization did not
> survive.

> **2026-09-17 (wave 3 close remediation cycle 2, finding A — one class axis
> for the whole label-domain family):** `_PrecomputedSource.confusion_matrix`
> derives its class axis from `_target_hygiene.report_labels`, the same
> partition `classification_report_chart` uses on both paths and
> `ModelSource.confusion_matrix`'s no-`classes_` branch joined above. The
> rule it replaces, `sorted(set(y_true) | set(y_pred), key=str)`, was carried
> for two cycles as an *ordering* preference. It was not only an ordering
> rule: stringifying before comparing makes `str` totally order every value,
> which is why it accepted every input — including ones the chart cannot
> draw.
>
> Measured 2026-09-17 against the pre-fix code on `y_true = [0, 1, 0, 1]`
> with `y_pred = ["0", "1", "0", "1"]` (object dtype): it derived four labels
> `[0, '0', 1, '1']` that stringify to two, returned a **16-row** frame
> carrying only **4 distinct** `(actual, predicted)` keys — each repeated
> four times, with the true diagonal mass split across visually identical
> cells — and `confusion_matrix_chart` rendered **four superimposed `<text>`
> nodes at each of four identical pixel positions** (four nodes at
> `x=178.166, y=133.537`, and the same at three further positions).
> `class_prediction_error_chart`, which renders the same frame, drew the same
> thing. `classification_report_chart` refused that input by name throughout.
> Overplotted duplicate class labels are not an ordering preference, and
> under the standing REFUSE-MALFORMED policy the correct end state is a
> refusal.
>
> Two user-visible moves, both on `confusion_matrix_chart` and
> `class_prediction_error_chart`:
>
> - **Inputs that rendered now refuse**, by name and in the report's own
>   voice: a mixed-family union, `complex` labels, and
>   `list`/`set`/`bytearray` labels (the last previously leaked an unowned
>   `TypeError: unhashable type` out of the old rule's own `set()`).
> - **Numeric class labels order by value rather than by digit count.**
>   Measured: `{1, 2, 10}` moves from `['1', '10', '2']` to
>   `['1', '2', '10']` — which is what `classification_report_chart` and the
>   model path already rendered, so this retires the last divergence in the
>   family's class axis rather than creating one. Two classes of unequal
>   digit count are enough to see it.
>
> String-labelled matrices sort identically under both rules and do not move;
> that is every confusion golden in the repo, and no golden moved. The
> earlier note's "presentation difference, not a hygiene one" reading is
> retired: there is no label-rule difference left between the four
> label-domain surfaces.

> **2026-09-17 (wave 3 close remediation cycle 2 — three smaller contract
> notes):**
>
> - *`classification_report_chart`'s model path refuses a null-bearing
>   prediction vector.* An estimator whose `predict` returns a null produced
>   scikit-learn's own `ValueError: Classification metrics can't handle a mix
>   of binary and unknown targets` (measured), naming neither the array nor
>   the entry, while `confusion_matrix_chart` and
>   `class_prediction_error_chart` refused typed on the identical input. The
>   guard now runs in `report_predictions_in_y_domain`, which both report
>   consumers share, so the null-prediction cell is answered the same way by
>   all three label-domain figures.
> - *A model-path refusal names a lever the caller holds.* When `y_pred` is
>   `model.predict`'s output rather than an array the caller passed, "drop or
>   impute the null rows" and "use one consistent value type for y_true and
>   y_pred" name something the caller cannot do. Those refusals now close
>   with a clause naming the model instead. The diagnosis, counts and
>   observed values are unchanged, and the precomputed path's wording is
>   byte-identical.
> - *One spelling for the entry name.* The precomputed adapter's shape and
>   arity guards took `chart_name=` while every hygiene guard took
>   `diagnostic=` — two keywords for one concept, passed the same local value
>   three lines apart. They are now all `diagnostic=`. Internal only
>   (`_PrecomputedSource` is private); no message text changed.

> **2026-09-17 (wave 3 close remediation, finding 4 — a refusal names the
> entry you called):** every target-hygiene refusal and warning opens with
> the name of *the public entry the caller actually invoked*. That is what
> the contract always said; the model path did not honor it. Measured:
> `fm.roc_chart(model, X, y_with_nulls)` raised `"roc_curve: y has 10 null
> row(s) ... Drop or impute the null rows before calling roc_curve."`,
> naming a source method the figure-function caller never invoked and
> telling them to fix it there, while the precomputed
> `fm.roc_chart(y_true=, y_pred=)` on the same defect raised
> `"roc_chart: ..."`. A caller who switched input paths saw one defect
> reported under two names.
>
> Every classification method on both source kinds now takes a
> `diagnostic=` keyword that defaults to the method's own name, and each
> public entry passes its own: a message raised through `roc_chart` opens
> `roc_chart:`, through `ROCVisualizer` opens `ROCVisualizer:` (matching
> `ClassificationReportVisualizer`, which named itself from the start), and
> a direct `source.roc_curve()` call still opens `roc_curve:`. The most
> visible instance is the one that could not be fixed by baking a name into
> the source method: `confusion_matrix` serves two figure functions, so its
> stray-drop warning now opens `class_prediction_error_chart:` when that is
> what was called, and `confusion_matrix_chart:` when that was. This is a
> **message-text change only** — no number, no frame, and no rendered
> geometry moves — and the name is deliberately not part of the cache key,
> since it labels messages rather than changing what is computed.
>
> **2026-09-13 (fix cycle 4 — cell 4 extends from unorderable to
> unkeyable, and `y_true`'s shape joins `y_pred`'s).** The prose above
> ("a homogeneous unorderable [family] refuses as unorderable") was true
> of `report_labels` in isolation but not of what shipped: the very next
> statement in `classification_report` (`self.confusion_matrix(...)`)
> keys each class by its label value in a `dict`, so an
> orderable-but-unhashable label (`list`, `set`, `bytearray`) answered
> inside `report_labels` and then leaked `TypeError: unhashable type`
> one statement later — measured on all three. A `frozenset` pair (hashable,
> but only partially ordered by subset) surfaced a second defect: the
> former adjacency-only dedupe does not group equal values next to each
> other under a partial order, so it returned a duplicate label (measured:
> three labels for two classes). Cell 4 is now named *unkeyable* rather
> than *unorderable*: it refuses both a value that does not order and one
> that orders but cannot be hashed, each with its own message naming the
> reason; the answer cell dedupes by hash equality (`dict.fromkeys`) after
> confirming hashability, not by adjacency. **Breaking, disclosed:**
> `list`/`set`/`bytearray` class labels now refuse by name — previously an
> unowned `TypeError: unhashable type` leaked from the confusion matrix,
> not from `report_labels`, so this closes a leak rather than narrowing an
> accepted input. `frozenset` labels still render; the fix is to their
> deduplication, not their acceptance. Also fixed: `y_true`'s shape was
> never checked (`report_labels`' docstring claimed it was "the caller's
> precondition, enforced upstream", which was true for `y_pred` via
> `_require_1d_array` — renamed from `_require_1d_y_pred`, since it is now
> called on both arrays — but not for `y_true`); a 2-D `y_true` was
> misdiagnosed as "a mix of label types" (each row read as a `list`
> value, one ordering family, against `y_pred`'s `number` family) instead
> of the correct 1-D shape refusal `_require_1d_array` already gives
> `y_pred`. `classification_report_chart` now checks both arrays' shape
> before either reaches `report_labels`. `y_pred`'s own pre-existing 1-D
> message also moved as part of this: it read `requires 1-D hard class
> labels; got an array with shape ...` (unsuffixed, when it was the only
> such check at this call site) and now reads `requires 1-D hard class
> labels (y_pred); ...`, disambiguating it from the new `y_true` check
> that now runs beside it. `confusion_matrix_chart`'s own 1-D `y_pred`
> check is unaffected and keeps its unsuffixed wording.

> **2026-09-13 (fix cycle 5 — a total mismatch between the model's
> translated `classes_` and `y` refuses).** One residual carried forward from
> cycle 1 (rated S1 there, revised upward to S2 on cycle-4 review): a
> model whose `classes_` describe *different* labels than `y` --
> `classes_ = [0, 1, 2]` fit against `y = ["c0", "c1", "c2"]` is the shape
> -- translated every prediction to `"0"`/`"1"`/`"2"` through the
> #206 correspondence, values `y` never held, and
> `classification_report_chart` rendered an 18-row, six-class report with
> **every metric 0.0** and no warning, while `confusion_matrix_chart` on
> the identical source warned `"confusion_matrix: 300 row(s) of y out of
> 300 hold values outside the class set ['0', '1', '2'] ... and were
> dropped from it entirely"`. Before #206's translation existed, this exact
> shape raised sklearn's own loud `ValueError: Mix of label input types
> (string and number)` -- so #206, closing the dtype-translation gap for
> the ordinary (same-labels, different-dtype) case, traded that loud
> refusal for a silent, wrong-looking-plausible chart on this adjacent
> (different-labels) case. This is the total-mismatch shape #203's
> standing policy already refuses elsewhere -- a resolution matching no row
> of `y` refuses rather than answering something plausible and meaningless
> (`check_resolved_target`'s zero-match cell, `binary_indicator`'s
> positive-class cell) -- so it now refuses here too, via the new
> `_curve_frames.check_translated_classes_match_y`, called from
> `report_predictions_in_y_domain` on the coerced class vocabulary before
> anything is translated, on both model-path consumers
> (`classification_report_chart`'s builder and
> `ClassificationReportVisualizer._materialize`). **Breaking, disclosed:**
> a total mismatch between the model's translated `classes_` and `y` now
> refuses instead of rendering; a *partial* overlap is unaffected and still
> renders, since it is the ordinary out-of-vocabulary case sklearn's own
> report already tolerates. What is tested is the class *vocabulary*, not
> the predictions the model happened to make: a correctly-labelled model
> evaluated on a slice it gets entirely wrong (`y` holds only one class,
> the model predicts only another) shares no predicted value with `y` while
> its `classes_` and `y` still describe the same labels, so that report
> renders, unchanged from the released behavior. This closes the gap for the `classes_`-present
> branch only -- the `classes_`-less branch's own disclosed leak (above)
> is untouched, since there is no translation for this check to run after
> there.

> **2026-09-10 (batch G close, cycle 3 — positive-class resolution and
> target hygiene):** two consequences of making binarization mandatory,
> both corrections.
>
> *Resolution.* The positive class is now always a value from `y`'s own
> domain. Three tiers: the model's `classes_` coerced into `y`'s dtype;
> else `y`'s own distinct values when their count matches the score
> columns; else, as a last resort, the display label parsed back into
> `y`'s dtype. Tier 2 is new. Tier 3 used to be the only fallback, and it
> is unsound as a general rule: `probabilities()` names its columns from
> `class_names`, which is a user alias or a synthetic `class_{i}`, so a
> model without `classes_` binarized `y` against a string `y` never
> contains and every binary curve collapsed (measured: `AUC nan`, `AP
> 0.0`, a flat reliability curve, and a Brier of `mean(p²)` against a true
> 0.0959). Models exposing `classes_` are unaffected, with or without a
> `class_names=` alias.
>
> *Hygiene.* A binary curve whose target matches **no** row of the positive
> class now raises `ValueError` naming the class value and `y`'s observed
> values, rather than computing against an entirely-negative target. An
> all-null target is the same case. This is the same refusal-over-plausible-
> number principle as the NaN note above, applied at the other end of the
> pipeline. Scope is deliberately narrow: multiclass one-vs-rest still
> tolerates an absent class ("no row is class k" is a real answer, and
> `lift_curve` documents skipping zero-base-rate classes), an empty target
> is not an error, and an all-positive target still yields the kernels'
> honest NaN. A *partially* corrupted target — some null or out-of-domain
> rows among valid ones — is still absorbed as negative and is **not**
> covered here. *(Superseded 2026-09-13 by the hygiene-matrix note above:
> nulls refuse, strays warn-and-absorb, and gain/lift plus zero-match
> multiclass joined the refusal. The all-null case still refuses, now by the
> null rule and with the true diagnosis rather than a `classes_` mismatch
> message.)*
>
> The refusal judges the **request**, never an internal slice of it.
> `discrimination_threshold(cv=...)` resolves and validates the positive
> class once against the whole target, then sweeps each fold from rows of
> that validated indicator. A held-out fold that happens to contain no
> positives is a property of the split, not a bad request — routine on the
> imbalanced data this diagnostic exists for, and more so because the
> default generator is plain `KFold(shuffle=True)` — so it contributes the
> kernel's honest `precision = recall = f1 = 0` with its real `queue_rate`,
> exactly as it always has. A target that genuinely never matches still
> refuses, with or without `cv=`.

> **2026-08-27 (P9 remediation, findings-remediation batch):** three
> mark-level parameters listed above were accepted and silently dropped
> (`del <param>` with no wiring); each is now either implemented or
> removed, never a silent no-op. Removed — passing any of these now raises
> `TypeError` naming the argument:
>
> - `mark_calibration(n_bins=..., strategy=...)` — removed. The mark
>   receives already-binned curve rows (`mean_predicted`,
>   `fraction_positive`, `count`); binning happens in
>   `ModelSource.calibration_curve(n_bins=, strategy=)` (or the
>   `calibration_chart` figure function, which forwards them there), so no
>   raw prediction column ever reaches the mark layer to rebin.
> - `mark_pca_scree(n_components=...)` — removed. Never real: no desugar
>   ever declared it as a wired parameter, and `pca_scree_chart`'s own
>   `n_components` is consumed entirely by
>   `ModelSource.pca_variance(n_components=)` upstream of the mark.
> - `mark_alpha_selection(ci_style=...)` — removed. The data contract
>   (`alpha`, `mean_score`) carries no lower/upper variance columns, so
>   there is no CI band to style; unlike `mark_learning_curve`/
>   `mark_validation_curve` (both keep `ci_style`, real `{"band",
>   "errorbar"}` vocabulary), this mark renders a single curve.
>
> Other P9 sites became functional rather than removed (`average` on
> `mark_roc`/`mark_pr`, `split` on `mark_cv_scores`, `reference_line` on
> `mark_gain`/`mark_lift`, `metrics` on `mark_discrimination_threshold`,
> `order`/`color_bar` on `mark_shap_beeswarm`) — those already matched
> this spec's Key Parameters columns and needed no table edit. **Correction
> (2026-08-27 close-out):** `order` did not in fact match this spec's table
> (which read `"max_abs"`, a value the implementation never accepted), and
> `color_bar=False` was accepted and validated but never wired to suppress
> the beeswarm's color bar/legend — both are now real; see the two
> close-out notes below.

> **2026-08-27 (P9 AST guard, findings-remediation batch — Task 14):**
> closing the P9 desugar-parameter guard's own blind spot (a `del` on a
> *declared* parameter with no wiring and no warning) surfaced two more
> mark-level parameters carrying the same defect the paragraph above
> fixed, neither of which was implementable or removable — their real
> effect lives entirely upstream, before the data ever reaches the mark.
> Both are now registered in
> `ferrum.marks._informational_kwargs.INFORMATIONAL_KWARGS` and warn once
> (`UserWarning`) when passed directly to the mark method with a
> non-default value; neither changes a single byte of rendered output.
>
> - `mark_decision_boundary(proba=...)` — informational at the mark
>   layer. The grid's `z` column (class index vs. predicted probability)
>   is already computed by the time it reaches this mark; `proba`'s real
>   effect is in `decision_boundary_chart`'s upstream grid construction,
>   which selects which `z` gets computed and no longer forwards `proba`
>   into the `mark_decision_boundary()` call at all.
> - `mark_discrimination_threshold(n_thresholds=...)` — informational at
>   the mark layer. The threshold sweep is already fixed and the data
>   already pre-melted by the time it reaches this mark; `n_thresholds`'s
>   real effect is in `ModelSource.discrimination_threshold(n_thresholds=)`
>   / `discrimination_threshold_chart(n_thresholds=)`, upstream of the
>   mark, which likewise no longer forwards it into the
>   `mark_discrimination_threshold()` call.
>
> Both were previously silent no-ops (`del <param>` with neither wiring
> nor a warning); they now match the "works or rejects/warns loudly"
> contract every other P9 site above already meets.

> **2026-09-13 (wave 3 T10, #189 — `mark_discrimination_threshold`
> explicit-vs-default `optimum_label`/`threshold_line`):** both flags
> default to a private `_UNSET` sentinel (`ferrum._desugar`) instead of a
> materialized bool, so `_prep_discrimination_threshold_sentinels` can tell
> "the caller explicitly asked for this" apart from "the parameter's own
> default." A `metrics=` subset that excludes `"f1"` leaves neither the
> `_threshold_best` nor the `_optimum_x`/`_optimum_y`/`_optimum_text`
> sentinel column for the layer to read (see the 2026-05-12 note on this
> mark above): an EXPLICIT `optimum_label=True` or `threshold_line=True` in
> that state now raises `ValueError` naming the flag and `metrics=`, since
> there is no F1-optimum row to label/rule at and the request is self-
> contradicting; the un-passed default keeps the pre-existing silent
> downgrade to `False` (batch F T3's crash-shape fix). `discrimination_
> threshold_chart` and `DiscriminationThresholdVisualizer` forward
> `optimum_label` through unchanged (still `_UNSET` by default) rather than
> materializing their own `True` default before calling the mark, so the
> silent-default behavior survives through the figure-function and
> visualizer entry points too, not just the direct chart-API call this
> issue was filed against. `threshold_line` keeps a concrete `False`
> default at every layer — unlike `optimum_label`, `False` is never the
> value the refusal fires on, so there is no default-vs-explicit ambiguity
> to preserve for it there.

> **2026-08-27 (close-out, findings-remediation batch):** two further gaps
> closed on `mark_shap_beeswarm`/`shap_chart`, found by the closing design
> and intent reviews.
>
> - **`data_transform` silently no-op'd on non-polars charts.**
>   `_set_composite_mark` (`chart.py`) applied every P9 "become functional"
>   `data_transform` closure (`average` on `mark_roc`/`mark_pr`, `split` on
>   `mark_cv_scores`, `reference_line` on `mark_gain`/`mark_lift`, `metrics`
>   on `mark_discrimination_threshold`, `order` on `mark_shap_beeswarm`)
>   only when `isinstance(new._data, pl.DataFrame)` -- a pandas- or
>   pyarrow-backed `Chart` silently skipped the filter/reorder entirely,
>   with no error and no warning. Now routed through the batch's own
>   `ferrum._coerce.to_polars` at that seam, so every supported input type
>   is coerced before the closure runs; already-polars input is an identity
>   passthrough (`to_polars` returns the same object), so this is byte-
>   identical for the previously-working polars path.
> - **`mark_shap_beeswarm(order=)` and `shap_chart`'s feature-ranking
>   `order=` carried two different closed vocabularies.** The mark accepted
>   `{"abs_mean", "mean", "none"}` (row *display* order); the figure-side
>   `_shap_order_features` accepted `{"abs_mean", "max"}` (feature
>   *selection* ranking, pre-existing, documented, implemented behavior --
>   `"max"` ranks by descending `max(|shap_value|)` to surface
>   high-impact-outlier features) -- `"abs_mean"` meant the same thing on
>   both, `"mean"`/`"max"` each raised on the other, and
>   `shap_beeswarm_chart` papered over the mismatch with a hardcoded
>   `order="none"` mark call. Unified to the **union** of both
>   vocabularies (not a narrowing -- an earlier pass of this fix wrongly
>   retired `"max"`, breaking working pre-batch behavior; corrected),
>   canonically defined once in
>   `ferrum.marks._desugar_helpers.SHAP_ORDER_VALUES = {"abs_mean", "mean",
>   "max", "none"}` and imported by both sides: `"abs_mean"` ranks by
>   descending `mean(|shap_value|)`; `"max"` ranks by descending
>   `max(|shap_value|)` (now implemented on the mark side too, via the same
>   `expr.max()` branch `_shap_order_features` always had); `"mean"` by
>   descending signed `mean(shap_value)`; `"none"` performs no ranking.
>   `_shap_bar_chart_from_source`'s bar value follows the same rule it had
>   at merge-base: `order="max"` plots `max(|shap_value|)` (byte-identical
>   to pre-batch behavior); every other `order` plots `mean(|shap_value|)`,
>   matching the `abs_mean_shap` column name.
> - **`mark_shap_beeswarm(color_bar=False)` accepted, validated, and
>   silently discarded the kwarg.** The desugar's own per-layer
>   `Color(..., legend=...)` was correct, but the Rust renderer's
>   colorbar-legend construction for a layered/composite chart reads its
>   `legend=` config from the *chart-level* `encoding.color`, never from a
>   per-layer color channel -- so neither `color_bar=False`'s suppression
>   nor `color_bar=True`'s documented `"Low"`/`"High"` tick labels ever
>   reached the rendered SVG; a plain numeric-tick colorbar rendered
>   unconditionally. `Chart.mark_shap_beeswarm` now also mirrors the same
>   `Color(...)` config onto the chart-level encoding, verified by
>   rendering (not merely by inspecting the emitted spec JSON). This
>   changes the *default* (`color_bar=True`) rendered output too -- the
>   `shap_chart_beeswarm` golden was regenerated and visually confirmed to
>   show the "Low"/"High" tick labels the mark always claimed to render.

> **2026-08-27 (P9 AST guard extension, findings-remediation batch —
> Task 14, quality-review cycle 3):** extending the guard in the paragraph
> above from "every `del`" to "every declared parameter, `del`eted or
> simply never referenced" (closing the guard's own P9-class blind spot —
> a parameter that is declared and never read passes a `del`-only check
> just by not being explicitly deleted) surfaced three more mark-level
> parameters in that exact state. Two match the `proba`/`n_thresholds`
> shape above (effect lives entirely upstream); one does not:
>
> - `mark_confusion(normalize=...)` — informational at the mark layer.
>   The cell values are already normalized (or not) by the time they
>   reach this mark; `normalize`'s real effect is in
>   `ModelSource.confusion_matrix(normalize=)` /
>   `confusion_matrix_chart(normalize=)`, upstream, which no longer
>   forwards it into the `mark_confusion()` call. Registered in
>   `ferrum.marks._informational_kwargs.INFORMATIONAL_KWARGS`; warns once
>   when passed directly with a non-`None` value; no effect on rendered
>   output either way.
> - `mark_pdp(center=...)` — informational at the mark layer. ICE
>   polylines are already re-based to start at 0 by the time they reach
>   this mark; `center`'s real effect is in `pdp_chart(center=)`
>   (`_pdp_center_curves`, applied before the `Chart` is constructed),
>   which no longer forwards it into the `mark_pdp()` call. Same registry
>   + warn-once treatment; no effect on rendered output either way.
> - `mark_boxen(palette=...)` — **not** the same shape as the two above.
>   Unlike `normalize`/`center`, no call site anywhere — mark, mixin, or
>   figure function — ever gave `palette` an effect: it was simply never
>   read. Registered as a stopgap (warned once on a non-`None` value)
>   purely so it stopped being a *silent* drop, tracked as an open item in
>   `design-docs/superpowers/followups/2026-05-15-code-archaeology.md`.
>
> **2026-08-27 (residuals batch, #91; current behavior after three rounds
> of quality-review correction — see the amendment history below):**
> `mark_boxen(palette=...)` implemented. `palette: str | Sequence[str] |
> None = None`. The color mapping anchors on the **base band**: `colors[0]`
> is spent on `k=2`, the innermost real letter-value interval beyond the
> median — guaranteed to render whenever `LetterValue` produces *any*
> non-degenerate depth at all, for *any* dataset — and later colors
> consume outward (`k=3`, `k=4`, ...) as richer data materializes more
> bands; a `k` that never materializes for a given dataset simply leaves
> its color, and every color past it, unused (the same tail-truncation as
> a palette longer than the actual number of categories elsewhere). `k=1`
> — the median tick's own row (see the 2026-09-06 note below: a
> band-confined `tick`, not a full-panel `rule`), `lower == upper == median`
> by construction,
> always zero-pixel on every render — never consumes a color of its own
> (it borrows `k=2`'s). Fills apply at opacity 1.0, replacing the opacity
> ramp. A named palette (`str`) is expanded via `ferrum.color.palette(name,
> n=<colorable band count>)` — exactly the number of colorable band slots
> (11 as of the 2026-09-06 note below; was 5 pre-batch), so a
> full-depth dataset shows every generated color, including a continuous
> palette's endpoint (`viridis` reaches `#fde725`) — raising the same
> `ValueError` shape `scheme=` raises for an unrecognized name; an
> explicit color sequence is applied in inside-out order and cycled if
> shorter than the colorable band count. Bands paint **widest-first**
> (largest present `k` painted first/under, `k=1` last/on top) so the
> letter-value nesting stays visible once the ramp's alpha blending is
> gone; `palette=None` keeps the original ascending-`k` layer order and is
> byte-identical to the pre-batch shading. This is depth-band coloring,
> not a seaborn-style hue mapping — **boxen has no hue channel today**
> (hue-vs-palette support is the tracked violin-hue frontier work, out of
> scope here), and `palette` does not interact with `color_field` (which
> only changes the `LetterValue` groupby, not color — every band still
> renders in one theme color at ramp opacities, or the flat palette fill,
> regardless of `color_field`). A chart-level `.encode(color=...)` channel
> *does* conflict with `palette`: the color encoding always overrides a
> layer's `fill=` (the same precedence a plain `fill=` has under any
> encoded channel), so combining the two raises `ValueError`
> (`ferrum._desugar._prep_boxen_palette_color_conflict`, both call orders)
> — drop the color encoding, or drop `palette=`; there is no hue-channel
> fallback to redirect to. A non-`str`, non-iterable `palette=` value
> (e.g. an `int`) raises the same named `ValueError` shape as the
> empty-sequence case, not a bare `TypeError`. An explicit user
> `fill=`/`opacity=` mark kwarg still wins over either shading (existing
> `apply_user_mark_kwargs` precedence). The `warn_informational_kwarg`
> call and the `INFORMATIONAL_KWARGS["boxen"]` registry entry are removed
> — `palette` is now genuinely used by `desugar_boxen`.
>
> **Amendment history:** the original 2026-08-27 note above said "band k
> (outermost = 1) gets `fill = colors[k-1]`" — backwards (`k=1` is
> innermost) — and a first correction pass (same day) fixed the
> band-indexing description but kept `colors[k-1]` as a literal raw-index
> formula, which put `colors[0]` on the innermost, always-zero-pixel `k=1`
> band. A second quality-review pass rasterized the output and proved the
> requested first palette color rendered in zero pixels, and also caught a
> false remedy in the `.encode(color=...)` conflict error/docstrings ("use
> `color_field=...` for per-group hue instead" — `color_field=` grants no
> hue at all, contradicted by this very note's own "no hue channel"
> sentence two lines earlier); the hue-remedy fix landed clean, but the
> color-mapping fix that pass shipped anchored `colors[0]` at the widest
> *configured* band (`k=<the mark's full configured depth cap>`, see the
> 2026-09-06 note below for how that cap is derived) instead of the widest *rendered*
> one — real per-group depth is a Rust-side quantity this desugar function
> has no access to, so that anchor only rendered `colors[0]` when a
> dataset happened to reach full configured depth, which is not the
> common case: measured (third quality-review pass) at zero pixels for
> every group under 1000-ish rows on the mark's default `k_depth`, with
> only the palette's *last* two colors ever appearing at n=200. The
> now-shipped base-band anchor (`k=2`, above) replaces it, along with
> sizing the palette request to the actual number of colorable slots
> (the full configured depth cap minus one, not the full cap itself — the
> previous request generated
> one color, `colors[-1]`, that no mapping could ever reach, truncating
> every continuous-palette ramp one step short of its endpoint
> unconditionally). The "architecturally-unavoidable-in-Python" framing
> attached to the previous anchor's residual gap is retracted: it was true
> of the *literal* "outermost rendered band" reading (still not decidable
> without a Rust round-trip, since depth is data- and even per-group-
> dependent), but not of removing the gap itself, which the base-band
> anchor does in pure Python by choosing a different, always-present
> anchor point instead.
>
> **2026-09-06 (batch-E T10, F-L03-02/04b/10/11/12 — closes the stale
> `_BOXEN_K_MAX` references above):** `_BOXEN_K_MAX` (the hardcoded literal
> `6` this section's earlier notes name) is **deleted**. `desugar_boxen`'s
> depth-band count now derives from `letter_value`'s registry-declared
> `secondary_outputs` (`depth_1..depth_N`), which the Rust `K_MAX_NAMED`
> constant governs — currently **12**, raised from 6/8 by the same batch's
> T5. Every place the notes above say "the mark's full configured depth
> cap" now means 12, not 6; the colorable-band count (`colors[0..]`'s
> length) is therefore **11**, not 5.
>
> `letter_value`'s `level` column names each depth by the real Tukey letter-value continuation, not a `"depth_N"` fallback spelling — `M` (depth 1, the median), `F`, `E`, `D`, `C`, `B`, `A`, `Z`, and, past the classical seven-letter table, continuing `Y`, `X`, `W`, `V` for depths 9 through 12 (verified live via `cargo test -p ferrum-core --lib letter_value::tests::test_letter_value_tukey_n8192_reaches_depth_9_with_continuation_letter`, which asserts depth 9's `level` is `"Y"`).
>
> **Disclosed rendered-output
> consequences** (byte-stability exceptions for this batch):
> - `palette=` boxen charts sample `ferrum.color.palette(name, n=11)`
>   instead of `n=5`. For a *continuous* palette this shifts every sampled
>   position (resampled evenly across `[0, 1]` at the requested count), so
>   colors on an existing `palette=` chart using a continuous scheme change
>   even when its real letter-value depth is unchanged. A *categorical*
>   palette's early colors (`colors[0]`, `colors[1]`, …) are unaffected as
>   long as the palette has ≥5 entries (`colors[i % len(colors)]` for
>   `i < len(colors)` doesn't depend on the requested count).
> - A chart whose real per-group depth exceeds 6 (n large enough that
>   `k_depth` computes more than 6 real bands — e.g. `k_depth="full"` on a
>   few hundred rows) now renders those previously-silently-dropped deeper
>   bands. Depths 7 and 8 already existed in Rust's own pre-batch output
>   (its cap was 8, not 6 — `desugar_boxen`'s Python-side cap of 6 was the
>   tighter, binding limit) but had no `desugar_boxen` layer to read them;
>   depths 9 through 12 are newly computed by Rust as well, since T5 raised
>   `K_MAX_NAMED` from 8 to 12 in the same batch.
> - The median mark is now `mark="tick"` (band-confined to its category),
>   not `mark="rule"` (which drew a bare `{x, y}` pair with no `x2`/`y2` —
>   `rule.rs`'s documented shape table resolves that pattern to a
>   full-panel `HorizontalSpan`, so the median line used to run across
>   every category, not just its own).
> - Every layer's positional encoding titles both axes with the user's own
>   field names (`X(title=...)`/`Y(title=...)`); previously the axes showed
>   `LetterValue`'s internal `"group"`/`"lower"`/`"upper"`/`"value"` column
>   names verbatim.
> - `letter_value`'s `"outliers"` secondary batch now emits only rows where
>   `is_outlier == true` (T5); `desugar_boxen`'s own outlier-point encoding
>   is unchanged, so the *encoding* side is purely a batch-filtering fix —
>   the outlier layer draws exactly the flagged-true rows instead of every
>   observation. What counts as flagged-true is not itself unchanged,
>   though: `is_outlier`'s own fence (`letter_value.rs:394-396`,
>   `outlier_threshold * span` beyond the deepest materialized band) reads
>   off whichever depth this same batch's raised cap now lets a large
>   dataset reach. For `n ≳ 4096` under Tukey's own formula (the point
>   where its raw depth first exceeds the pre-batch cap of 8), that fence
>   now derives from one or more letter-value depths deeper than before as `n`
>   grows (exactly one depth deeper just past the threshold; more at larger
>   `n`, clamped once the raised cap of 12 is reached) — a
>   narrower `span`, hence a tighter absolute fence — layered on top of the
>   pure filtering change. A dataset too small to reach the old cap-8 depth
>   sees only the filtering fix; its fence position is unchanged.
> - The default (`palette=None`) opacity ramp is renormalized against the
>   depth THIS chart actually materializes (its real Tukey/proportion/
>   trustworthy/full depth, computed from the bound data via a lightweight
>   arithmetic estimate — never the quantile math, which stays Rust-only),
>   guarded to a minimum of 2 bands — not against the full configured cap
>   of 12. Normalizing by the cap unconditionally would have flattened the
>   ramp's visual contrast on every chart whose real depth is small (the
>   overwhelmingly common case), even though that chart's own rendered
>   depth did not change: adjacent-band opacity steps would have shrunk
>   from ≈0.11-0.18 (typical pre-batch contrast) to ≈0.05 (normalized
>   against 12) for no data-driven reason. A chart whose real depth cannot
>   be determined (no bound data reachable, or the data shape can't be
>   read) falls back to normalizing against the full cap, matching the
>   batch's initial (pre-cycle-2) behavior — never a crash, since this is a
>   styling estimate, not a correctness-critical value.
> - `whisker_mult="min-max"` (boxplot, not boxen) now suppresses the
>   outlier point layer entirely, regardless of `outliers=`: whiskers
>   already reach the data extremes, so nothing can lie beyond them.
>   Previously the `Outliers` transform still ran at a fixed `1.5`×IQR
>   fence unrelated to the whiskers' own `"min-max"` extent, so points
>   between that fence and the true min/max rendered as "outliers" —
>   contradicting the min-max whisker plot's own geometry, which already
>   reached those same points.
>
> `k_depth` (every strategy) clamps at depth 12: two `k_depth` rules that
> differ only in the depth they'd compute beyond 12 render identically —
> reachable only at group sizes from roughly 4,000 rows (`"full"`) up into
> the tens of thousands (`"tukey"`/`"trustworthy"`), depending on the
> strategy's own formula.

---

### 3.4 Stat Transforms

> **Dated note (2026-09-19, wave 5 — GH #185: `stat_letter_value` takes the group dtypes its siblings already took):** the issue reported that `stat_letter_value` rejected `Utf8` group columns; it was inverted. `Utf8` and `Categorical` were the only dtypes it accepted, and it refused every integer, unsigned, float and boolean that every sibling transform takes — so `mark_boxen` on an `Int64` category column raised outright where `mark_boxplot`, `mark_violin` and `mark_errorbar` all rendered. `letter_value` was the last transform still carrying its own private group-key representation (an `Option<String>`) and its own hand-rolled dtype gate, years after the shared `group_key/` toolkit was created so that int/uint/bool support "exists once and cannot drift". Both private copies are **retired**, not supplemented: validation routes through `validate_groupby_columns` and partitioning through `group_partition_multi`, the same route `box_stats` takes, and the group column comes back in its **original dtype** rather than stringified (a `.to_string()` shortcut would have satisfied the error message while producing `Utf8` where `box_stats` produces `Int64`). Routing through the shared partitioner surfaced a second defect: groups whose values are entirely null survived into the sort, where a `NaN` aggregate degrades the comparator to `Equal` against everything — measured, a `sort=median, Desc` request came back ascending. Empty groups are now skipped. **Disclosed:** refusals now read "groupby column", the shared validator's wording, where they read "group column" before. A temporal (`Date32`/`Timestamp`) group column still refuses. Archaeology row NS-2 closes with this.
>
> `mark_boxen` on a numeric category column needed **both** this and #171: #171 alone left it raising, #185 alone left it drawing 8px stubs with no boxes, which is why the family's contract is pinned as one matrix (`tests/_category_dtype_matrix.py` and its consumers) rather than in either half's own module.

> **2026-05-10 (Phase 9):** Adds eight new transforms — `Unpivot` (wide →
> long reshape; homogeneous-or-numeric value dtype), `Linkage` (hierarchical
> clustering with three named secondary outputs: `linkage`, `order`,
> `coords` / segment table for dendrogram rendering), `Reorder`
> (permutation by index column produced by `Linkage`), `Bin2D` (2D
> rectangular binning over `(x, y)`), `Logistic` (binary logistic
> regression via IRLS plus Wald CI), `Glm` (5 families × 7 links — see
> compatibility table below), `Robust` (Huber M-estimator + sandwich CI),
> `LetterValue` (boxen-plot statistics; outliers as named secondary
> output). Extends existing transforms: `Bin` gains a `cumulative: bool`
> parameter for ECDF support; `Smooth` gains `x_bins`, `x_estimator`, and
> `output: "fitted" | "residuals"`; `Robust` accepts the same `output`
> parameter for residual diagnostics.
>
> **2026-09-10 (batch F, F-L10-04):** `Robust` gains `groupby` (a column
> name or list, serde-default empty, wire-compatible), mirroring
> `Smooth`'s existing grouped-fit pipeline: a per-group Huber IRLS refit,
> with `x_range`/`inject_zero_ref`/`inject_metrics` all honored per group,
> and the group-key columns re-emitted at their original dtypes. Ungrouped
> output (the default) is byte-identical. This is what lets `residplot`'s
> `robust=True` compose with `color=` (grouped refits) and `label=`
> (previously a silent no-op, since `Robust` had nothing to carry the
> injected `_label` column into its output -- see `residplot`'s dated note
> in §3.14). The grouped-stack-plus-group-key-append block is shared with
> `Smooth` via one helper (`group_key.rs::stack_grouped_batches`); a
> `groupby` column colliding with a transform's own output column name, or
> repeated within `groupby` itself, now refuses with a typed error naming
> the real output columns -- previously a duplicate-named Arrow field was
> silently mis-resolved by `Schema::index_of`, corrupting output instead
> of refusing (applies to both `Smooth` and `Robust`).
>
> **Dated note (2026-09-11, wave-1 hardening, #179/#197 — the eleven
> hand-rolled partitioners migrate onto one shared partitioning idiom;
> BREAKING for two disclosed behaviors):** `Aggregate`, `Summary`,
> `BoxStats`, `ErrorExtent`, `Violin`, `Pivot`, `DataAggregate`,
> `DensityData`, `Bin`, `Kde`, `Kde2D`, `Outliers`, `TopK`,
> `RegressionData`, `LoessData`, `DataStack`, `JoinAggregate`, and
> `DataWindow` now all partition their `groupby` through the one shared
> `group_key::group_partition_multi` (or, for `DataWindow`'s
> sorted-adjacent-run algorithm, the same validator plus its own
> null-key warning) instead of a hand-rolled per-transform loop, closing
> the repeated-groupby-name gap for good and installing one null-key
> policy everywhere at once.
>
> - **BREAKING: group row order is first-appearance, not sorted**, for
>   `stat_aggregate`, `stat_summary`, `stat_box_stats`, `stat_error_extent`,
>   `stat_violin`, `data_pivot`, `data_aggregate`, and `data_density`.
>   Downstream this moves the category-axis order, the legend order, and
>   therefore palette assignment, because a categorical domain is built in
>   first-appearance order of the post-transform batch. A frame whose
>   groupby column already arrives in sorted order is unaffected (the two
>   orders coincide); measured across the full test/golden corpus, this
>   moved exactly two goldens (`cv_scores_ridge_box.svg`,
>   `compare_cv_scores_two_models.svg` — both a pure element-order
>   permutation, byte-length-identical, no geometry change).
> - **BREAKING: a null groupby key now excludes its row from every group**,
>   for the same eight transforms, replacing the previous behavior where a
>   null key was pooled into its own (arguably meaningless) group. A
>   groupby column that is null on every row degrades to an empty group
>   set and emits a named `UserWarning` (`"...so the data could not be
>   split into groups; the grouped output is empty"`) instead of silently
>   rendering one unlabeled group; a *partially*-null groupby column does
>   not warn (groups were genuinely formed; nothing degraded). Three named
>   exemptions keep their pre-existing behavior and gain the same warning
>   rather than dropping rows: `Impute` keeps its whole-column fallback
>   (a row-preserving contract — every input row must still produce an
>   output row); `Smooth`/`Robust` keep their pooled whole-batch fit (the
>   batch-D-documented degraded-but-meaningful render); `DataStack`,
>   `JoinAggregate`, and `DataWindow` — a sibling class the original issue
>   did not name, added because all three are row-preserving with no
>   appended group column, the same shape as `Impute` — pool null-keyed
>   rows into one group rather than dropping the rows they'd otherwise
>   have to leave with a fabricated `0.0`/`NaN` value.
> - `transform_density(field, groupby=[...])` now honors **every** groupby
>   dimension — it previously grouped by only the first and silently
>   discarded the rest, so `groupby=["a", "b"]` behaved as `groupby=["a"]`
>   with `b` invisibly dropped from the output. The output now gains one
>   column per extra dimension.
> - `transform_pivot`, `transform_aggregate`, and `transform_density` — the
>   three data/stat transforms that build their spec as a raw wire dict
>   with no PyO3 constructor of their own to refuse a repeat at — now refuse
>   a repeated `groupby` name at the Python construction boundary (matching
>   `Aggregate`'s existing `#[new]`-time refusal), instead of deferring the
>   catch to Rust's apply-time partitioning.
> - Repeated-groupby-name and output-column-collision refusals now cover
>   every append site across the whole partitioning family, closing the
>   gap the constructor-time refusal alone left open for every one of
>   these transforms reached through a raw-dict/wire spec rather than its
>   typed pyclass.
>
> **Transform degrade-warning channel (2026-09-11, wave-1 hardening,
> #162).** A Rust `TransformWarning` sink (`Rc<RefCell<Vec<_>>>` on
> `TransformContext`, drained into the render pipeline's warnings vec and
> delivered as a Python `UserWarning` through the existing
> `binding::emit_warnings` — no new delivery path) now names every
> transform-level degrade that previously happened in silence: the
> all-NaN-panel family (`Smooth`, `Robust`, `RegressionData`, `LoessData`,
> `Kde` — naming the transform, the group when grouped, and the reason;
> the render/output is unchanged, only a warning is added), `stat_smooth`'s
> `gam` `k` clamp, `LetterValue`'s named-depth cap, `Impute`'s
> empty-group and whole-column fallbacks, and the #179/#197 null-key
> warnings above. Behavior for every one of these was already the
> documented degrade (a NaN range, a clamped parameter, a pooled fit, an
> imputed fallback) — this note is additive: a warning now names what was
> always happening.
>
> **GLM Family / Link Compatibility (Phase 9)**
>
> | Family | Canonical link | Other valid |
> |---|---|---|
> | Gaussian | Identity | Log, Inverse |
> | Binomial | Logit | Probit, Log |
> | Poisson | Log | Identity, Sqrt |
> | Gamma | Inverse | Identity, Log |
> | InverseGaussian | InverseSquared | Identity, Log |

Stat transforms are applied before rendering, in the Rust engine. They receive a column-oriented Arrow table and return a transformed table. Applied via `.stat()` at the chart or layer level, or implicitly by statistical marks.

| Class | Description | Key Parameters |
|---|---|---|
| `stat_bin(field, *, bins=None, bin_count=None, bin_width=None, extent=None, nice=True)` | Bin a field; add `bin_start`, `bin_end`, `count`, `density` | `bins` (2026-09-18): `"sturges"` \| `"fd"` \| `"freedman_diaconis"` \| a positive `int` (bin count) \| a positive finite `float` (bin width) \| `None`. Mutually exclusive with `bin_count`/`bin_width` — naming both refuses. |
| `stat_bin_2d(x, y, *, bin_count_x=None, bin_count_y=None)` | 2D binning | |
| `stat_kde(field, *, bandwidth="scott", kernel="gaussian", n=512, extent=None, cumulative=False)` | Kernel density estimate | |
| `stat_kde_2d(x, y, *, bandwidth="scott", n=128)` | 2D KDE | |
| `stat_smooth(x, y, *, method="loess", ci=0.95, bandwidth=0.75, degree=None, n=200)` | Smoothing line + CI | 2026-09-05: `method="glm"` and `method="gam"` are newly accepted — `glm` fits a real GLM (IRLS, Gaussian/Identity default) and `gam` fits a real generalized additive model (a penalized cubic B-spline: uniform-knot cubic B-spline basis of dimension `clamp(distinct_x, 4, 10)`, second-order difference penalty, smoothing parameter chosen by GCV over a fixed 33-point grid, Bayesian posterior CI band on `n − edf` degrees of freedom). Both previously refused as unknown methods, so no existing chart changes. `gam` is fully deterministic — it uses no randomness, so `seed=` does not affect it — it takes no `degree=` (the spline degree is fixed at 3 and smoothness comes from the penalty), and outside the observed x range it holds its boundary value rather than extrapolating. A `gam` fit the data cannot support (fewer than 4 distinct x values, or zero x variance) degrades to a NaN range like `lm` on a zero-variance x — gam adds no whole-frame refusal, since no caller parameter can make its design absurd. `method="lm"` + `degree≥2` routes to polynomial regression, and the `degree=` default changed `2` → `None` (bare `method="lm"` stays linear — the old default would have silently promoted it to quadratic). `degree=` on `log`/`sqrt` now raises where it was silently ignored (`glm` refuses it too, from birth); `quadratic`/`cubic` accept only their own degree (a conflicting `degree=` raises, naming `method="lm", degree=d` as the escape hatch). A whole-frame polynomial fit whose `degree` ≥ usable point count raises instead of rendering an empty plot — this includes `quadratic`/`cubic` on 2-/3-row frames that previously rendered blank; a degenerate *group* in a `groupby` fit, or a degenerate *facet panel* of a faceted chart, still degrades to a NaN range per the grouped-fit semantics (batch D, F-L03-08). 2026-09-06 (close-review hardening): `gam` gains `k=` — basis dimension, default 10, clamped to [4, distinct-x count], the knob for under-smoothing on high-frequency data; refuses `k < 4` and `k=` on non-gam methods. Explicit `bandwidth=` on any non-loess method now raises typed instead of being silently ignored, matching `degree=`'s policy (an explicit value equal to the 0.75 default is accepted) |
| `stat_summary(field, *, fun="mean", error_fn="ci", ci=0.95, n_boot=1000)` | Aggregate with uncertainty | |
| `stat_aggregate(field, fn, *, groupby=None)` | Arbitrary aggregation | |
| `stat_ecdf(field, *, complementary=False)` | Empirical CDF | |
| `stat_qq(field, *, distribution="normal", line=True)` | Quantile–quantile data | |
| `stat_contour(x, y, *, bandwidth="scott", thresholds=6, smooth=True)` | 2D contour levels | |
| `stat_identity()` | No-op; explicit passthrough | |
| `stat_ellipse(x, y, *, type="norm", level=0.95, segments=51)` | Compute a confidence or data ellipse for bivariate scatter. Returns `x`, `y` coordinates of the ellipse boundary. | `type`: `"norm"` (parametric normal ellipse) \| `"t"` (t-distribution) \| `"euclid"` (Euclidean distance from centroid). `level`: confidence level (ignored for `"euclid"`). |
| `stat_function(fn, *, domain=None, n=200, as_=("x", "y"))` | Evaluate a Python callable over a 1D domain. `fn` receives a numpy array and must return a numpy array. `domain`: `(min, max)` tuple; inferred from X scale if `None`. Returns columns named by `as_`. | |
| `stat_hex(x, y, *, bin_size=None, bins=None, aggregate="count", field=None)` | Hexagonal binning. Returns `hex_x`, `hex_y`, `value`. Triggered implicitly by `mark_hex`. | |

> *(2026-05-10) Phase 8b: `stat_kde_2d` is implemented as the `Kde2D` transform (10th transform of the phase). Output is a single-row Arrow batch with `grid_x` / `grid_y` / `density` list columns.*

**Model stat transforms** (accept `ModelSource` or raw arrays)

| Class | Description |
|---|---|
| `stat_roc(y_true, y_score, *, average=None)` | Compute ROC curve data |
| `stat_pr(y_true, y_score, *, average=None)` | Compute PR curve data |
| `stat_confusion(y_true, y_pred, *, normalize=None, labels=None)` | Compute confusion matrix |
| `stat_calibration(y_true, y_prob, *, n_bins=10, strategy="uniform")` | Compute calibration curve |
| `stat_lift(y_true, y_score)` | Compute lift/gain curve |
| `stat_importance(model, X, y, *, method="builtin", n_repeats=30)` | Compute feature importance |
| `stat_shap(model, X, *, background=None)` | Compute SHAP values |
| `stat_pdp(model, X, features, *, grid_resolution=100, kind="average")` | Compute partial dependence |
| `stat_residuals(model, X, y, *, kind="studentized")` | Compute residual diagnostics |
| `stat_learning_curve(model, X, y, *, cv=5, scoring=None, train_sizes=None)` | Compute learning curve |
| `stat_validation_curve(model, X, y, param, values, *, cv=5, scoring=None)` | Compute validation curve |

---

### 3.5 Data Transforms

Data transforms are applied before stat transforms, in the Rust engine. They reshape or filter the input data.

| Class | Description | Key Parameters |
|---|---|---|
| `transform_filter(predicate)` | Row filter | Predicate string (Vega expression), dict, or `Selection` |
| `transform_calculate(as_, expr)` | Derived column | Expression string or callable |
| `transform_aggregate(*aggregates, groupby=None)` | Group-by aggregation | `Aggregate` objects |
| `transform_bin(field, *, as_=None, **bin_kwargs)` | Bin a field | |
| `transform_fold(fields, *, as_=("key", "value"))` | Wide → long (melt) | |
| `transform_pivot(field, value, *, groupby=None, limit=None, op="sum")` | Long → wide | |
| `transform_join_aggregate(*aggregates, groupby=None)` | Add aggregate columns without collapsing rows | |
| `transform_window(*window_transforms, *, sort=None, groupby=None, frame=None, ignore_peers=False)` | Window functions (rolling mean, rank, cumsum, etc.) | 2026-09-05 (batch D, M4): `rank`/`dense_rank` written without a `field` are now tie-aware from the `sort` columns (competition ranks for `rank`, dense ranks for `dense_rank`; rows equal on every sort column, post-coercion, share a rank) — previously they silently fell back to `row_number`, so tied sort values drew distinct, order-dependent ranks. `ignore_peers` is now implemented: under the default `False`, a rolling-aggregate frame is anchored on the current row's own tied peer run (rows equal on every `sort` column) and then both ends independently expand outward to whole peer groups, so tied rows always share one frame and one aggregate value and a frame boundary never splits a peer group; under `True`, frames are row-offset only (unaffected by ties) — this is what every frame did before this release. This is *not* Vega's `ignorePeers: false` contract: Vega also expands both ends outward to peer boundaries, but anchors the un-expanded frame on the current row rather than the peer run's own bounds, so it does not guarantee tied rows share a value (example: sort key `t=[1,1,2,3,3]`, values `x=[10,20,30,40,50]`, `frame=(-1,0)` — ferrum gives `[30,30,60,120,120]`, Vega gives `[30,30,60,120,90]`, differing only on the trailing tied row). Also: `sort=`/`field=` columns now coerce every integer/unsigned/float/temporal dtype instead of silently comparing `Ordering::Equal` (leaving the window unsorted) or reading `NaN`; an unsupported dtype raises naming the transform, column, and dtype; null sort values order last (previously compared equal to everything, an intransitive non-order); and a `frame` lying entirely before its own row now selects no rows instead of wrapping an unsigned cast and aggregating the whole partition. |
| `transform_density(field, *, bandwidth=None, kernel="gaussian", groupby=None, extent=None, steps=None, cumulative=False, as_=("value", "density"))` | KDE transform | 2026-09-05: gained `kernel=` — routed through the one shared KDE kernel dispatch, which also gained `"exponential"` and `"linear"` (batch D, F-L05-12/F-L03-07); the row also mis-documented `bandwidth="scott"` and a nonexistent `minsteps=` — corrected to the shipped signature |
| `transform_regression(x, y, *, method="linear", order=1, groupby=None, as_=("x","y"))` | Regression transform | 2026-09-05 (batch D, M3a): row corrected — the shipped signature has never had `extent=`/`params=`, and `order`'s default has always been `1`, not `3`. `groupby=` was declared on the wire and accepted by Python but silently ignored by the Rust transform (F-L05-03); it is now real — the fit is computed independently per group, the group column is appended to the output at its original dtype, and each group's output grid spans that group's own x-range (not the whole dataset's), matching `mark_smooth`'s grouped behavior. Ungrouped output is byte-identical. |
| `transform_loess(x, y, *, bandwidth=0.3, groupby=None, as_=("x","y"))` | LOESS transform | 2026-09-05 (batch D, M3a): `groupby=` was declared but inert (same F-L05-03 gap as `transform_regression`); now real, with the same per-group-own-x-range and byte-identical-when-ungrouped contract described above. |
| `transform_impute(field, *, method="value", value=None, groupby=None, key=None, keyvals=None)` | Fill missing values | 2026-09-05 (batch D, M3/M3b, F-L05-04): `groupby=` was declared but computed one global fill value regardless of group; now real — the fill value is computed from each group's own rows, and a group with no non-null values of its own falls back to the whole column's statistic (not `0.0` or `±inf`, which would silently distort the domain or drop the group's rows). `key=`/`keyvals=` are newly implemented with Vega semantics: within each group (or the whole batch, with no `groupby`), the key universe is `keyvals=` when given, else the distinct `key` values observed across the whole batch; a key missing from a group gets one row appended after the group's own rows, with the key and group columns set and every other column null (its `Field` widens to nullable only when a row is actually appended). `keyvals=` without `key=` raises, as does a `keyvals` element outside the key column's own dtype range (previously silent corruption via integer wraparound for out-of-width values). |
| `transform_flatten(fields, *, as_=None)` | Expand list-typed columns | |
| `transform_sample(n, *, seed=42)` | Random row sample | 2026-09-06: row corrected to the shipped signature (seeded — byte-deterministic) |
| `transform_top_k(n, *, field, op="sum", sort="descending", groupby=None)` | Keep top-k by aggregate | 2026-09-05: gained `groupby=` — the aggregate is computed and ranked per groupby key (batch D, F-L05-01); previously the row also mis-documented `op="mean"` (the shipped default has always been `"sum"`) |
| `transform_stack(field, *, groupby, sort=None, as_=("y0","y1"), offset="zero")` | Stacking transform for area/bar | 2026-09-06: row corrected to the shipped `as_` default |
| `transform_timeunit(field, unit, *, utc=False, as_=None)` | Temporal unit extraction | |

> **Dated note (2026-09-05, batch D, M5 — expression grammar gains arrays, calls, and `indexof`; F-L05-06 dict-list literal fix):** the expression grammar used by `transform_filter`/`transform_calculate` gained array literals (`[e1, e2, …]`, any expression as an element), function-call syntax (`ident(args…)`) resolved against a builtin registry, and one builtin, `indexof(array, value) -> Number` (index of the first array element equal to `value`, else `-1`) — the documented membership form `indexof([1, 2, 3], datum.id) >= 0` now actually parses and evaluates; before this release it raised `unexpected character: ','` inside the array literal. `transform_filter`'s dict-list branch (`{"field": [...]}`) now compiles to exactly this `indexof(...)` form, with every element serialized through the grammar's own literal rules — Python's `repr()` previously leaked `True`/`False`/`None` (capitalized, not the grammar's `true`/`false`/`null`) into the emitted predicate for any bool- or `None`-valued list/dict/scalar filter, which raised `unexpected identifier`. All three dict-filter branches (list membership, comparison-dict, scalar equality) now share one literal encoder that also normalizes numpy/pandas scalars (`np.int64`, `np.float64`, `np.bool_`) to their Python-typed equivalents and refuses non-finite floats and unsupported types by name. `transform_calculate`'s expression evaluates to `ValueError` (naming the expression and row) if its *overall* result is an array — arrays are legitimate only as an `indexof(...)` argument, not as a materialized column value. `ferrum._core.transform_registry(kind: str) -> {"methods": [...], "outputs": [(name, dtype), ...], "secondary_outputs": [(key, [(name, dtype), ...]), ...], "params": [...]}` (the fourth key added 2026-09-06, close-review hardening: the named secondary batches — qq reference line, letter-value bands/outliers, linkage dendrogram — are declared schema under default construction, a caller-supplied `name` prefixing the keys) is a new Rust-owned registry (raises `ValueError` listing known kinds for an unrecognized one) that is the single source of truth for method vocabularies, output schemas, and accepted spec fields per transform kind — Python derives `mark_smooth`'s method validation, KDE kernel/qq-distribution vocabularies, and output-column names used by figure/desugar consumers from it, replacing several previously-hand-maintained Python mirrors that could drift from the Rust implementation.

---

### 3.6 Scales

Scales map data domain values to visual range values. Attached to encoding channels via the `scale=` parameter.

#### Scale Classes

| Class | Description |
|---|---|
| `Scale(type=None, domain=None, range=None, nice=None, zero=None, padding=None, clamp=None, reverse=None, round=None)` | Base / linear scale |
| `ScaleLog(base=10, *, domain=None, ...)` | Logarithmic |
| `ScalePow(exponent=2, *, domain=None, ...)` | Power |
| `ScaleSqrt(...)` | Square root (power with exponent=0.5) |
| `ScaleSymlog(constant=1, ...)` | Symmetric log (handles zero) |
| `ScaleTime(...)` | Temporal |
| `ScaleUtc(...)` | UTC temporal |
| `ScaleOrdinal(domain=None, range=None)` | Categorical |
| `ScalePoint(domain=None, *, padding=0.5, align=0.5, reverse=False)` | Ordinal point scale (for dot plots) |
| `ScaleBand(domain=None, *, padding=0.1, padding_inner=None, padding_outer=None, align=0.5)` | Ordinal band scale (for bar plots) |
| `ScaleSequential(scheme=None, *, domain=None, reverse=False, interpolate=None)` | Sequential color |
| `ScaleDiverging(scheme=None, *, domain=None, domainMid=None)` | Diverging color |
| `ScaleThreshold(domain, range)` | Threshold / bin color |
| `ScaleQuantile(domain, range, *, quantiles=None)` | Quantile |
| `ScaleQuantize(domain, range)` | Quantize |
| `ScaleBinOrdinal(bins, scheme=None)` | Binned ordinal |

> **Dated note (2026-09-19, wave 5 — GH #153: `nice=True` on a typed scale class does what the same key does in a raw dict):** `LinearScale(nice=True)`, `TimeScale(nice=True)` and their `LogScale`/`SymlogScale` siblings accepted the kwarg and always serialized `nice: false`, so a chart written with the typed class rendered un-niced ticks while the identical chart written as `{"type": "linear", "nice": true}` rendered niced ones. Measured on the same data: ticks `1..9` against `0..10`, and a first time tick of `2024-01-04` against `2023-12-28`. The gap was exactly the **data-derived** domain, where nicing is necessarily deferred to render and the flag had nowhere to live — the structs had no `nice` field, so the value was discarded at construction and the wire conversion hardcoded `false`. All four classes now retain and emit it. No new wire key was needed; `nice` already exists on `ScaleSpec` and the raw-dict path already set it, so only the class path had to reach it. The **explicit-domain** case was never broken (those classes nice eagerly at construction, so `.domain` and `.ticks()` already reflected it) and stays byte-stable. `TimeScale`'s own docstring has promised this behavior all along and was simply false; it is left untouched because the fix makes it true. **Disclosed:** existing charts passing a typed scale class with `nice=True` and no `domain=` now get the niced domain they asked for, so their ticks move.

> **2026-09-02 (batch A — discretizing color scales honored):**
> `Color(scale=...)` now honors `ScaleQuantize`, `ScaleQuantile`,
> `ScaleThreshold`, and `ScaleBinOrdinal` with real bucketed semantics
> instead of collapsing to a continuous linear scale. **Quantize** buckets a
> `domain=[lo, hi]` (explicit, or the data extent when omitted) into
> `len(range)` uniform buckets. **Quantile** buckets at data quantiles.
> **Threshold** takes `k` explicit domain thresholds mapping to `k + 1`
> colors. **BinOrdinal** takes explicit bin boundaries with scheme-derived
> or explicit colors. Bucket colors: an explicit string `range=` wins; else
> a **categorical** scheme name contributes its entries in declaration
> order (cycling with a `RenderWarning` when the bucket count exceeds the
> palette length); else a sequential/diverging scheme is sampled at evenly
> spaced points. A descending `domain=[hi, lo]` (Quantize) or a
> fully-descending threshold/bin-boundary list normalizes to ascending with
> the swatch order reversed; a non-monotonic boundary list (reachable only
> via raw-dict scales that bypass the `pyclass` constructors' validation) is
> a typed render error. `configure_color(range=...)` whose length mismatches
> the resolved bucket count emits a `RenderWarning` naming both counts
> rather than silently truncating or padding. The colorbar legend renders
> discrete labeled swatches for a discretizing scale instead of an
> interpolated gradient. `fm.continuous_palette(name)` (`ContinuousScheme`)
> is constructible and usable as `Color(scale=...)`, equivalent to
> `ScaleSequential(scheme=name)`; `Color(scale=fm.Gradient([...]))` is also
> implemented — explicit gradient stops thread through to the render path
> and compose with `reverse`.

#### Color Scheme Constants (`ferrum.schemes`)

**Categorical:** `okabe_ito` *(default)*, `tableau10`, `set1`, `set2`, `paired`, `pastel`, `dark2`

**Sequential:** `viridis` *(default)*, `plasma`, `inferno`, `magma`, `cividis`, `blues`, `greens`, `oranges`, `reds`, `purples`, `greys`, `tealblues`

**Diverging:** `redblue`, `redgrey`, `pinkgreen`, `purplegreen`, `brownbluegreen`, `spectral`

**Cyclical:** `rainbow`, `sinebow`

> **Implementation note (2026-05-11, themes-T3):** The 7 categorical schemes
> ship as a const palette registry in `crates/ferrum-core/src/render/palette.rs`
> (`CATEGORICAL_SCHEMES`, `SEQUENTIAL_SCHEMES`, `is_categorical_scheme`,
> `is_sequential_scheme`). `Theme(color_scheme=...)` validates the name
> eagerly at render entry — unknown values raise `ValueError`. Categorical
> color resolution precedence is now `encoding.scheme` (per-encoding override,
> e.g. `mark_heatmap`'s `cmap=`) → `theme.color_scheme` (Theme default) →
> `OKABE_ITO` fallback. Sequential scheme names (`viridis` etc.) used on a
> nominal encoding substitute `tableau10` rather than collapsing silently to
> the categorical default. Gridlines now render on every quantitative axis
> as `<line>` elements drawn behind axis lines and marks, honoring
> `theme.grid` / `grid_color` / `grid_width` / `grid_dash` / `grid_opacity`.
> Cyclical, `tealblues`, and the brewer-extended sequential names beyond
> `viridis/plasma/magma/inferno/cividis` are reserved spec surface — not yet
> implemented and currently rejected by theme-level validation. T4 will flip
> the default to `tableau10`.

> **Implementation note (2026-05-11, themes-T4):** `Scale.padding` (listed in
> the base Scale class above) is now plumbed end-to-end and defaults to
> `0.05` for quantitative / temporal scales when unset. The visual mapping
> reserves 5% of the plot dimension on each side (capped at 8 px) so marks
> do not touch axis lines or the plot edge. Precedence at the renderer:
> `Scale(padding=p)` honors `p` (including `0.0` to disable); a user-supplied
> `Scale(domain=[...])` with no explicit padding suppresses the default to
> `0.0` (the user-explicit domain is treated as authoritative); otherwise
> `0.05` applies. Categorical / ordinal scales (`ScaleBand`, `ScalePoint`,
> `OrdinalScale`) are unaffected — they keep their own internal half-step
> band padding. Pixel ranges supplied via `Scale(range=[...])` bypass the
> inset entirely and are treated as the final scale range. The 4 quantitative
> PyO3 scale classes (`LinearScale`, `LogScale`, `TimeScale`, `SymlogScale`)
> accept a new keyword `padding` and expose a `.padding` getter; the
> `Ordinal` variant's existing `padding` field is unchanged. The default
> `ThemeInputs::default()` color scheme flips from `okabe_ito` to
> `tableau10` (loud divergence from the §3.6 *"`okabe_ito` (default)"*
> annotation above) — see the §3.13 Themes-T2 → T4 note.

> **Design decision (2026-05-31, flexibility-campaign D1):** Categorical /
> ordinal scale `range` accepts **color strings** (hex `#rrggbb`/`#rgb` or CSS
> named colors), not only numeric pixel positions. Paired with `domain`, this
> expresses explicit category→color mapping — the editorial "gray everything,
> accent two" and financial "green-up/red-down" idioms
> (`ScaleOrdinal(domain=["A","B","C"], range=["#ccc","#ccc","#e4572e"])`),
> matching Altair's `Scale(domain=..., range=[...])`. `ScaleQuantize` string
> ranges are the existing precedent. No new value-class is introduced.

> **2026-05-31 (F2, internal wire-format typing):** The ordinal `scale.range`
> wire representation is now a typed array of `Number | String` entries
> internally (Rust `Vec<OrdinalRangeValue>`, serialized `untagged`), replacing
> the prior untyped `serde_json::Value` that was re-discriminated by JSON
> sniffing at three call sites. The **user-facing contract is unchanged**:
> Python still constructs `Scale(range=[...])` / `OrdinalScale(range=[...])`
> with a plain `list[float | str]`, and the emitted JSON is still a flat array
> (e.g. `[0, 300]` or `["#ccc", "#e4572e"]`). Only the Rust-side type and the
> single typed accessor that replaced the sniffers changed; positional and
> color scale resolution are byte-identical.

> **Design decision (2026-05-31, flexibility-campaign D2):** Layering with `+`
> resolves color scales by **unioning their domains** across layers, so the
> result is order-independent (`base + highlight == highlight + base`). On a
> genuine scheme conflict the first encoding-bearing layer wins. Axis titles
> are taken from the first **data-bearing** layer; annotation-only layers
> (e.g. `annotate_rect`) never supply axis titles or rename axes to
> `_x1`/`_y1`. Mirrors Altair's shared-scale resolve default.

> **Dated note (2026-09-04, batch C tasks 1/2 — `reverse=` is real on every continuous positional scale; F-L04-07):** `LinearScale`, `LogScale`, `PowScale`, `SqrtScale`, `SymlogScale`, and `TimeScale` all accept a `reverse: bool = False` keyword (and a `.reverse` getter of the same name); the raw-dict spelling `scale={"type": <continuous>, "reverse": true}` is honored identically. Semantics are **domain-swap sugar**, applied at the resolver after domain resolution: `reverse=True` on an explicit `domain=[a, b]` swaps the resolved pair, exactly equivalent to writing `domain=[b, a]` by hand (verified live: `LinearScale(domain=[0, 100], reverse=True)` renders byte-identical SVG to `LinearScale(domain=[100, 0])`). On an auto-inferred domain (no `domain=` supplied), the swap applies after the usual padding inference, so the equivalence claim is scoped to the explicit-domain case. Composition rules: `reverse=True` + an explicit `range=[hi, lo]` applies both as stated (a reversed explicit range is legal and means what it says); `.xlim(lo, hi)`/`.ylim(lo, hi)` (or the underlying `.coord(fm.CoordCartesian(xlim=…, ylim=…))`) replaces the domain wholesale and therefore wins over the encoding scale's `reverse` unconditionally. Range orientation, the structural y-inversion predicate, and tick label/fraction pairing are untouched — a reversed scale flips marks **and** axis tick labels together, never one without the other. Wire byte-identity is preserved for every pre-existing spec (`reverse` serializes only when explicitly `True`; the 14 frozen `test_scale_spec_parity` baselines are unchanged). **`PointScale.reverse` predates this batch (GH #65) and works differently**: it reverses the sorted domain **vector** before layout, not the resolved domain pair — the two `reverse` fields share a name but not a mechanism, and `BandScale` has no `reverse` field at all. **Known gap:** the scale-key gate (below) is keyed by scale *type*, not by encoding *channel* — `reverse` is a legitimate `Linear`/`Log`/etc. field, so `{"type": "linear", "reverse": true}` on a `color`/`size`/`opacity` channel deserializes without error but is silently inert there (only positional resolution reads `reverse`); closing that needs a channel-aware gate, not built by this batch.

> **`reverse=` in one place — the three behaviors it names (2026-09-04, batch C close remediation).** The note above and the `PointScale` / gate notes each document one of these in isolation; a design review asked for them presented together, because one kwarg name selects three different mechanisms depending on the scale class and the channel it is attached to. (1) **Continuous positional — render-time domain swap.** On `LinearScale`/`LogScale`/`PowScale`/`SqrtScale`/`SymlogScale`/`TimeScale` attached to `x`/`y`, `reverse=True` swaps the resolved domain pair at the render resolver. It is **inert on the compute facade**: `fm.LinearScale(domain=[0, 10], range=[0, 100], reverse=True).scale(0)` returns `0.0`, exactly as `reverse=False` does, because the swap has not happened yet at that call. (2) **`PointScale` — facade-active domain-vector reversal.** `PointScale.reverse` reverses the sorted domain **vector** before layout, so it *is* live on the compute facade: `fm.PointScale(domain=list("abc"), range=[0, 100], reverse=True).scale("a")` returns `83.33` where `reverse=False` returns `16.67`. Same name, different mechanism, different observable moment. (3) **Non-positional channels — accepted and inert.** The scale-key gate is keyed by scale *type*, not by encoding *channel*, so `{"type": "linear", "reverse": true}` on a `color`/`size`/`opacity` channel deserializes cleanly and then does nothing, since only positional resolution reads `reverse`. `BandScale` has no `reverse` field at all and refuses it. Rule of thumb: `reverse` means "flip the axis" on a positional channel, means "flip the category order" on `PointScale` regardless of where you ask, and means nothing anywhere else.

> **Dated note (2026-09-04, batch C task 4 — unknown raw-dict scale keys refuse at the wire; F-L04-07 completeness, F-L04-10 raw-dict path):** every raw-dict `scale={...}` value is now validated at the wire boundary against a schema-derived, per-type accepted-key set before serde's internally-tagged deserialize would otherwise flatten and silently drop an unrecognized field — the last documented `#[serde(flatten)]` silent-drop carve-out on `ScaleSpec` is closed. An unknown key refuses with `scale: unknown key '<key>' for type '<type>'; accepted: <sorted list>` (e.g. `{"type": "linear", "clammp": True}` names `clammp`, `linear`, and `linear`'s accepted keys). The accepted-key table is published as `ferrum._core.scale_accepted_keys(scale_type: str) -> list[str]` (raises `ValueError` for an unrecognized type) — the single source both the gate and `Chart.override(<channel>_scale_type=...)`'s type-switch merge consume, so the two cannot drift apart. **The refusal survives override composition**: `Chart.override(<channel>_scale_type=...)` on a channel that already carries an explicit `scale=` first validates the base scale's own keys against its own claimed type — a base scale that was already invalid (a typo, or a field its own type doesn't have) refuses immediately, exactly as the bare gate would, rather than letting the type switch decide the key's fate by coincidence (silently dropping it if the new type also rejects it, or, worse, promoting it into effect if the new type happens to define a same-named field) — then keeps only the base scale's keys that the new type also accepts. Raw-dict `{"type": "time"/"utc", "domain": [datetime.date(...), datetime.datetime(...), "2024-01-01", ...]}` converts every domain element under the identical rule `TimeScale(domain=...)`'s own constructor uses (below) before the dict is serialized to JSON — a Python `datetime` object cannot survive `json.dumps` at all, so this conversion happens upstream of any wire format existing, not inside it — or refuses naming the accepted forms.

> **Dated note (2026-09-04, batch C close remediation — `Chart.override(<channel>_scale_<leaf>=...)`'s accepted leaves are the gate's own wire keys):** the override leaf registry is now derived from `ferrum._core.scale_accepted_keys` (the same table the gate enforces) instead of from the `*Scale` classes' Python attribute names, so the two can no longer disagree. Additive effects: `nice`, `zero`, `stops`, `domainParam`, `paddingInner`, `paddingOuter`, and `domainMid` gain override paths they never had, and every camelCase wire key also accepts a snake_case spelling (`x_scale_padding_inner` == `x_scale_paddingInner`; both emit `paddingInner`). Two former leaves are removed: `<channel>_scale_utc` and `<channel>_scale_quantiles` named no wire key of any scale type — `utc` is a `TimeScale` constructor flag whose real override spelling is `<channel>_scale_type="utc"`, and `quantiles` is `QuantileScale`'s computed thresholds, a read-only getter. Neither had a working spelling before (both resolved through the registry and were then hard-refused at the wire); they now refuse at the override boundary with `FerrumOverrideError: Unknown override path ...`, which names the mistake instead of an internal key. A leaf whose wire key the effective scale type does not accept still refuses at the gate, naming that type's accepted list — e.g. `.override(x_scale_padding_inner=0.3)` on a default (linear) x refuses with `unknown key 'paddingInner' for type 'linear'`, and works once the effective type accepts it (`.override(x_scale_type="band", x_scale_padding_inner=0.3)`).

> **Dated note (2026-09-04, batch C task 4 — `.override(<channel>_scale_domain=...)` now suppresses the bar zero-anchor like an equivalent explicit `scale=`; behavior change):** the override-scale merge for `mark_bar()` now runs **before** the zero-anchor injection, so `mark_bar().override(y_scale_domain=[50, 200])` renders identically to the explicit `Y("val", scale={"domain": [50, 200], "zero": False})` spelling — matching the pre-existing rule that an encoding's own `scale={"domain": ...}` already suppresses the zero-anchor — and differs from the `zero: True` spelling (verified live on both counts). Before this change the override merged the domain **after** the zero-anchor had already widened an empty scale to include zero, so `.override(y_scale_domain=...)` on a bar chart silently included a zero anchor the equivalent explicit `scale=` spelling would not have. This is a deliberate, disclosed correction, not a bug carried forward.

> **Dated note (2026-09-19, wave 5 — GH #247A: a scale ferrum wrote is not a scale the caller wrote):** a bar chart's value scale is injected by ferrum itself, to anchor the axis at zero. That injected scale was indistinguishable from one the caller wrote, and a caller-written scale rightly short-circuits the cross-mark domain union — so a bar **overlaid with another mark** bypassed the union and drew its bars against a domain its own axis did not describe. The injection now marks its authorship with an additive `systemInjected` bit on `ScaleSpec::Linear` (wire key `systemInjected`, `skip_serializing_if` false, so every pre-existing payload's bytes are unchanged), and the positional resolver's auto-zero-anchor gate reaches layered specs through `spec.layers` rather than reading any present scale as the caller's word. A scale the caller wrote keeps the flag withheld, so it still wins.
>
> **The scope is narrow and was measured**, over a 25-shape SHA-256 corpus rendered both ways. Four shapes move: `bar + point`, that pair under `CoordFlip`, and two faceted layered bars — in each case a bar overlaid with a mark whose values extend **beyond** the bar's, which now shares one value axis with it, so the moved renders are the ones whose marks agree with the ladder drawn beside them. Byte-identical and explicitly **not** moved: `catplot(kind="bar")` and its `hue=`/`col=` variants, `catplot(kind="box")`, every `displot(kind="hist")` and `mark_histogram` shape, every non-layered bar, `point + bar`, plain `fm.layer(...)`, `resolve={"y": "independent"}`, and `bar + line` on one frame with one value column (its union equals the bar's own extent). `catplot` is exempt by mechanism rather than by luck: its chart-level mark is a placeholder point with no injected scale, so the new gate has nothing to act on.
>
> **The `systemInjected` key is hand-settable, and that is disclosed rather than guarded.** The Python producer crosses the same wire boundary every user dict crosses, so the key gate accepts `scale={"type": "linear", "systemInjected": True}` on a linear scale — a way for a caller to assert ferrum's authorship of a scale the caller in fact wrote. Since the field's stated purpose is distinguishing the two authorships, a reader deserves to know that on this one spelling the distinction is advisory. It is pinned by `tests/test_layer_union_provenance.py::test_hand_written_system_injected_scale_is_accepted_and_yields`, which asserts the hand-written flag is accepted and does make that leaf yield to the union. The `Chart.override(...)` registry, whose scale leaves are derived from the gate's own accepted-key table, deliberately **excludes** this key: there is no `.override(y_scale_systemInjected=True)` path, and a test guard asserts the excluded key really is in the gate's table, so the exclusion cannot rot into a coincidence.

> **Dated note (2026-09-04, batch C task 3 — the UTC contract is pinned; F-L04-06):** `TimeScale(utc=True)` and `TimeScale(utc=False)` render byte-identical SVG, by contract: all temporal rendering — tick values, calendar `nice()`, and `format_time`/`format_time_spec` — is UTC, and a naive `datetime`/`date` means UTC because there is no other timezone the renderer could honor (byte-determinism bars any timezone-dependent rendering path, permanently — CLAUDE.md hard constraint). The `utc` wire tag now survives resolution (`ScaleSpec::Utc` previously hardcoded `utc=false` on the resolved scale regardless of the source tag; it now round-trips as `true`), though no rendering branch currently reads it — the tag exists for downstream consumers, not to select behavior today.

> **Dated note (2026-09-04, batch C task 3 — `TimeScale` joins its continuous siblings, optional datetime-accepting domain; F-L04-10):** `TimeScale(domain=None)` — the default — infers its domain from data like `LinearScale`/`LogScale`/etc.; a domain-less `TimeScale` no longer requires an explicit `domain=`. Each domain element accepts `float` (epoch-ms, unchanged), `datetime.date` (midnight UTC), `datetime.datetime` (naive means UTC; aware converts via the object's own `.timestamp()`, honoring any `tzinfo`), or an ISO-8601 date/datetime string (parsed via the real `datetime.date.fromisoformat`/`datetime.datetime.fromisoformat`, not a hand-rolled parser, so the accepted grammar always matches whatever Python version is running the extension). Plain `int` is accepted as a numeric epoch-ms value, matching every other continuous scale's `int`-domain acceptance — a deliberate divergence from `ferrum.annotation.temporal_coord_to_epoch_ms`'s own accepted-type union, which has no numeric type at all; `bool` is explicitly refused (Python's `bool` is an `int` subclass, and silently reading `True`/`False` as `1.0`/`0.0` epoch-ms is a footgun no sibling scale's domain should inherit). Any other value raises naming the accepted forms. A cross-language parity test proves the Rust-side conversion equal to `temporal_coord_to_epoch_ms` across the full input taxonomy (naive/aware/date/string, both directions), so the two rules cannot fork.

> **Dated note (2026-09-04, batch C tasks 5-7 — `BandScale`/`PointScale` padding and align are real render geometry, closing GH #67; F-L04-03):** `BandScale(padding=/padding_inner=/padding_outer=/align=)` and `PointScale(padding=)` now observably change single-series bar/box/tick geometry, following d3's own `scaleBand`/`scalePoint` layout model end to end — both mark widths and axis tick-label centers are read from the same resolved scale (previously mark widths ignored these parameters entirely, and axis tick centers used a second, symmetric, padding-blind model that could disagree with the bars by several pixels; both halves were finding F-L04-03 / GH #67, now collapsed onto one placement path). Default (auto-inferred) categorical axes construct unpadded (`padding_inner = padding_outer = 0, align = 0.5`) and reduce to the pre-existing symmetric model bit-exactly, so every default chart in the golden corpus is byte-identical; naming the scale class explicitly does not inherit that default, though — `fm.BandScale()` carries d3's own `padding = 0.1` on both sides, so a chart that merely writes `scale=fm.BandScale()` (or any explicit `BandScale(...)` call with no `padding=` argument) renders narrower bands than the same chart's auto-inferred axis, with no padding argument anywhere in user code (four categories, `mark_bar()`: `112.327`px auto-inferred vs. `98.629`px for `scale=fm.BandScale()`).
>
> **Breaking change to a public compute API:** `BandScale.scale()` returns d3's band leading edge from the single band model the renderer now shares, so for `padding_inner > 0` the reported edge moves back by half the inner gap — `BandScale(domain=list("abcd"), range=[40, 260]).scale("a")` is now `45.366` where earlier releases returned `48.049` — correcting a placement that let the last band extend past the declared range end. `BandScale.bandwidth()` is unchanged (`|step|·(1 − padding_inner)` never had the inset), and `PointScale.scale()` is unchanged (the point arm was never affected).
>
> **Mark widths derive from the scale's padding-aware `bandwidth()`** times the existing `band_size` factors (0.8 bar, 0.6 box/tick default); a mark's width now follows the resolved scale's own range, whatever that range's provenance, rather than falling back to the panel extent when a scale's own explicit range disagreed with its panel. The category count used is the scale's **domain**, not the data batch's — a layer missing categories (an empty facet cell, a filtered layer) now sizes its marks against the full domain, matching its complete sibling panels (a deliberate correctness change). A **padded heatmap cell is its drawn band on each axis** — `mark_rect`'s cell **is** the band, so a padded `BandScale` opens a visible gutter between cells rather than tiling them edge-to-edge, the same meaning padding has between bars (at the default zero padding, cells still tile exactly; every heatmap golden is byte-identical). A mark drawn against an explicit `PointScale`, which has no inner gap to size against, takes a width of `|step|` (the full slot) times its `band_size` factor, since a point scale's positions are zero-width by construction and a renderer must draw *something*.
>
> `BandScale.align` renders visibly only when the band denominator `n − padding_inner + 2·padding_outer` falls below 1 and is clamped to 1 — in practice a single-category domain with a large `padding_inner` (default-padding `n=1` is not enough on its own: verified live, `align=0.0` and `align=1.0` render byte-identically there); `PointScale.align` is algebraically inert unconditionally, for every domain size, since a point scale's padded positions always fill the range exactly — both documented as d3-faithful in the class docstrings. **Dodge composition**: scale padding defines the drawn band; `Dodge` subdivides that band and applies its own `padding` within it — the two compose rather than one overriding the other, and GH #66's clamp-width metadata now derives from the padding-aware bandwidth. **Violin is the one documented exception**: its transform stage runs before scale resolution and has no resolved scale to consume, so it sizes each polygon as panel pixels ÷ the batch's group count while every sibling ordinal width uses the resolved scale's range ÷ domain count, and the two diverge on three independent axes — under `BandScale(padding_inner=p)` sibling bars/boxes narrow to `(1 − p)` of the slot while violin polygons keep the full, unpadded slot width; under an explicit `domain=` containing a category absent from the data the siblings' band is narrower than `panel/groups` even at `p = 0`; and under an explicit `range=` narrower or wider than the plot area the siblings' band moves off the panel extent entirely, also at `p = 0`. Agreement is scoped to the auto-scale case: with no explicit scale the domain is the batch's own groups, the range is the panel, and padding is zero, so the two formulas agree exactly only there.
>
> Two documented divergences from upstream d3 remain (`crates/ferrum-core/src/scale/discrete.rs`'s module doc is the source of truth): (1) upstream folds outer padding into the `align` term; this model applies `padding_outer·step` directly to `start` and lets `align` distribute only the denominator clamp's leftover — the two agree whenever `align = 0.5` or `padding_outer = 0`, and differ only for a non-default `align` combined with outer padding. (2) upstream normalizes an inverted range before laying out bands, so its `band(x)` is always the low-coordinate edge; this model preserves the range's sign, so a negative-extent scale's `lead(i)` is the band's leading edge **in domain order** (the high-coordinate edge) — positions agree, only which edge is called the "lead" differs.

> **Dated note (2026-09-04, batch C task 7 — `tick_values`/`Axis(values=)` pairing on a categorical axis):** when an explicit tick-value override supplies exactly one value per category, the relabeled ticks land on the scale's own category centers — the same centers the marks sit on — for both an explicit-`range=` scale (the pre-#67 behavior) and a padded no-`range=` scale (this batch's aligned-centers behavior, extended to the relabeled case). When the override's length differs from the category count, the values have no per-category correspondence, so the ticks fall back to evenly spaced uniform slots across the plot area instead — unchanged behavior, and true for every categorical axis regardless of whether `range=` is explicit.

> **Dated note (2026-09-04, batch C — base `Scale` row correction):** the `Scale(...)` row's `round=None` parameter, in the table above, documents reserved spec vocabulary only — no concrete PyO3 scale class (`LinearScale`, `LogScale`, `PowScale`, `SqrtScale`, `SymlogScale`, `TimeScale`) accepts a `round` keyword today, and none is planned as part of batch C (batch C's non-goals explicitly name `round=` on continuous scales as remaining unimplemented). Treat the table's `round=None` as a documented gap, not a live parameter.

> **Dated note (2026-09-11, wave-1 hardening, #152):** an explicit `domain=`
> or `range=` whose length isn't exactly 2, on any of the seven continuous
> scale kinds (`Linear`, `Log`, `Time`, `Utc`, `Symlog`, `Pow`, `Sqrt`), now
> refuses at render time with `validate_continuous_domain`'s own wording
> (`"domain must have length 2; got {n}"` / the range equivalent) instead
> of panicking (`index out of bounds`) partway through resolution — this
> closes the literal reported crash, reachable from a raw-dict scale (e.g.
> `scale={"type": "linear", "domain": [5.0]}`), which bypassed the
> constructor-level length check the pyclass spellings already had. **Scope
> note, a deliberate narrowing from the original ask:** a *degenerate*
> `[c, c]` domain (equal, not merely short) is intentionally NOT refused at
> this funnel, even though the two construction-time surfaces (the pyclass
> constructors, chart-config surfaces) do refuse it — the reactive
> `domainParam` substitution path writes a parameter's *current* value
> straight into this same `domain` field, with no surviving marker
> distinguishing "the user hand-typed a degenerate domain" from "a
> slider mid-drag happens to be at a degenerate value right now"; refusing
> here would break the pre-existing, tested, sanctioned
> `domainParam`-collapses-to-a-midpoint contract (GH #104) for every
> reactive chart whose current value collapses. The length check has no
> such conflict — a too-short param value is already filtered to `None`
> (auto-infer) upstream, before it ever reaches this field. The continuous-
> degrading scale family (`Sequential`/`Diverging`/`Quantize`/`BinOrdinal`/
> `Quantile`/`Threshold`) is unaffected either way — `ScaleSpec::positional_extent()`
> already normalizes its domain to exactly 2 elements or `None` before this
> funnel runs, so the length panic can't reach it, and it carries shapes
> (Diverging's 3-element `[lo, mid, hi]`) the original report never named.

---

### 3.7 Axes and Legends

#### `Axis`

```
Axis(title=None, *, orient=None, ticks=True, tick_count=None, tick_extra=False, tick_min_step=None,
     grid=True, grid_dash=None, grid_width=None, grid_color=None, grid_opacity=None,
     labels=True, label_angle=None, label_flush=False, label_overlap="greedy",
     label_format=None, label_format_type=None, label_font_size=None, label_color=None,
     domain=True, domain_width=None, domain_color=None,
     offset=None, translate=None, min_band=None, max_band=None,
     title_orient=None, title_font_size=None, title_color=None, title_padding=None,
     values=None, encode=None, zindex=None)
```

> **Dated note (2026-09-19, wave 5 — GH #182, closing #94: a gutter is reserved only for chrome that is drawn):** layout reserved margins for chrome it never drew. Hiding an axis kept the axis title's gutter, and a polar chart — which has no Cartesian axes to hide at all — reserved Cartesian margins regardless, so the plot area came out smaller than the drawn chrome justified and the chart sat off-centre in its own frame. Measured before the fix: `.axis(show=False)` reserved a 23.729px phantom title gutter per side plus a 14.046px phantom label band on the y side, and a polar chart reserved 23.214px on the left and 43.039px at the bottom. The gate now asks whether the axis actually **draws** — which is what the sibling label-band check three lines away already did — so `.axis(show=False)` on a 640x480 chart with 16px padding collapses to exactly `(16, 16, 608, 448)`, `fm.Axis(labels=False)` collapses the label band alone, and polar reserves nothing Cartesian. Scoping the gate to the `show` flags would have closed #182 and left #94 open, which is why it reads "is it drawn". **`AxesInput::show_x`/`show_y` accordingly change meaning** to "this Cartesian axis is drawn", with the coord folded in by the producer; a consumer wanting the raw user toggle reads `spec.axis_x`. `plots/matrix.py`'s clustermap dendrogram `title=""` workaround — which existed only to reach the title-width route so a hidden axis's phantom gutter stopped eating the dendrogram's share of `dendrogram_ratio` — is **removed** rather than left beside the rule that replaces it. **This reverses a prior decision:** `tests/test_finding_p8.py` (2026-08-27 batch) pinned the phantom as "the architecturally correct outcome" and asserted a polar panel's edge equals a single-axis chart's; both tests are inverted with rewritten rationales, and the reversal is recorded for the campaign's decision-flip audit. **All output with hidden or non-Cartesian axis chrome changes geometry.**

> **Dated note (2026-09-19, wave 5 — GH #111: a swatch shows the encoding it stands for):** when `stroke_dash` or `shape` shares its field with `color`, the aux legend block is suppressed so the reader sees one legend instead of two — and the surviving color legend then drew three identical **solid circles** over marks that were solid, `6,3` and `2,3`, or over marks that were circle, square and cross. The mapping the legend exists to explain was unreadable. The surviving entries now carry the suppressed channel's identity: a dash-carrying entry draws as a dashed **line** swatch, a shape-carrying entry draws its glyph. The issue's own prescription (thread the dash pattern into the color entries) was measured to be a no-op — the color swatch is a hard-coded circle and the pattern is only read inside the line arm — so the swatch **kind** had to change; `shape` needed only the thread, since the renderer already reads the shape name ahead of the symbol-kind match. Identities are matched to color entries **by label, never by index**, because both domains are built over the same field with independent sort specs, so index pairing silently transposes under `sort="descending"`.
>
> Three adjacent behaviors are settled rather than left undefined. (1) A suppressed channel's explicit `title=` is no longer silently discarded: where the color channel states no explicit title and the suppressed same-field channel does, that title is used; where both state explicit titles and they differ, the color channel's wins and a `RenderWarning` names the discarded one. (2) `color=fm.Color(..., legend=None)` with a same-field `stroke_dash`/`shape` no longer erases **both** blocks — suppression exists to avoid a duplicate, and with no color legend there is nothing to duplicate. That gate now asks whether the color legend **enumerates** the field rather than whether its scale is categorical, so a colorbar plus a categorical shape or dash renders the aux block too. (3) A chart binding `color`, `shape` **and** `dash` to one field cannot show both folded identities, because the glyphs are painted fill-only and a dash array on them paints nothing: **shape wins**, matching what a standalone shape legend already draws, the losing pattern is not written onto the entry at all so the scene carries only what it paints, and the drop warns. `size` is not part of this: `size=` on a `Utf8` column refuses, so the categorical-color size-suppression branch is unreachable for a string field, and the numeric same-field color+size merge already worked.

> **Vocabulary change (2026-06-22, cohesion-campaign T3.3b / D-EXTENT-1):** the
> axis layout-band overrides were renamed `min_extent`/`max_extent` →
> `min_band`/`max_band` so that `extent` is reserved for the data-domain sense
> only (the layout already uses `band` internally). `min_extent`/`max_extent`
> remain accepted as deprecated keyword aliases (emit a `DeprecationWarning`;
> supplying both a canonical and its alias raises `TypeError`). The same rename
> applies to `configure_axis(...)`.

> **Design decision (2026-05-31, flexibility-campaign D3):** Per-channel
> `Axis(label_format=...)` and `tick_count` are honored at layout time (the
> renderer previously hardcoded the per-channel override to `None` at
> `render/prepare.rs`, so only chart-level `configure_axis` reached the
> formatter). Format support is the **full d3 grammar, single source of truth
> in Rust**, feeding SVG, PNG, and the strings baked into interactive exports
> identically: **numbers** use a hand-rolled d3-format implementation
> (`[[fill]align][sign][symbol][0][width][,][.precision][type]`, type chars
> incl. `s % p r g`, plus the `~` trim flag); **time** uses `chrono` strftime
> (`%b %Y`, `%Y-%m-%d`, `%H:%M`), replacing the prior hand-rolled date math.

> **Dated note (2026-09-03, batch B tasks 2/4 — format presets are honored everywhere, chart-level time formatting works, malformed specs refuse):** a named format preset (`ferrum.format_presets.ALL_PRESETS` — e.g. `"currency"`, `"percent"`, `"date_iso"`, `"ordinal"`) now resolves through one shared function (`resolve_format_or_raw`) on **all five** surfaces that accept a format string: chart-level `AxisConfig(label_format=...)`/`configure_axis(label_format=...)`, per-channel `Axis(label_format=...)`, per-channel `Legend(format=...)`, encoding `format=` (`Text`, `TooltipField`, and any channel routed through the shared encoding emitter), and the raw-dict normalize paths for both axis and legend (including the camelCase serde-alias spellings, e.g. `labelFormat`). Previously only `Axis(label_format=...)` resolved presets; the other four surfaces passed a preset name straight to Rust's d3-format parser, which rendered it as literal control characters into the SVG. A name that isn't a recognized preset is treated as an already-valid raw d3-format/strftime spec and passed through unchanged — this function never raises. `AxisConfig.label_format` is the one preset-names-only exception: it validates eagerly at construction (raises `ValueError` for an unrecognized name) since it has no raw-spec sibling field of its own; a raw d3-format/strftime string belongs in the parallel `label_format_raw` field instead (mutually exclusive with `label_format`).
>
> Chart-level **time** presets now actually format tick labels as dates: `configure_axis(label_format="date_iso")` previously rendered `300.000000%` (the chart-level path never threaded `format_type` through to the formatter); it now renders ISO dates as expected. `"ordinal"` produces real ordinal suffixes (`1st`, `2nd`, `3rd`, `4th`, …, with the `11th`–`13th` exceptions), not raw integers. `label_format=`/`format=` compose with `values=` in both directions on the surfaces that accept both. An explicit per-channel format (`Axis(label_format=...)`, `X(..., format=...)`) always beats a conflicting chart-level `configure_axis(...)` value — a cascade inversion that previously let chart-level silently win regardless of which was actually set, fixed at the one write site both former hand-copied call sites shared. **Malformed format specs now raise a typed error** (`RenderError::InvalidFormatSpec`) instead of emitting control characters into the SVG or, for a malformed time-with-trailing-token spec (e.g. `"curency%"`), panicking through the PyO3 boundary; the diagnosis routes real `%`-directive strftime specs before the ambiguous date/time guess, so the error names the actual problem. `configure_axis(x=..., y=...)` remain deprecated, inert flags (`Chart.axis(x=/y=)`, documented below, is the replacement for axis visibility; they carry no formatting meaning).
>
> **Dated note (2026-09-11, wave-1 hardening, #133):** an unambiguously
> time-typed format (`format_type` explicitly `"time"` — a strftime spec,
> or a named time preset like `"date_iso"`) now refuses with a typed error
> naming the axis/channel when the resolved scale is not temporal, on all
> three surfaces that accept a format: per-channel (`Axis(label_format=)`,
> encoding `format=`), chart-level (`configure_axis(label_format=)` /
> `axis_x=`/`axis_y=`/`axis_y2=`, both the plain `label_format` branch and
> the `tick_values` sibling branch), and the facet independent-axis path
> (`TickFormatMode::Immediate`, the surface the original issue's "both
> paths" phrasing missed). Previously a time format on a mixed
> temporal/quantitative chart's shared `axis=`/`configure_axis()` key
> silently applied to both axes, rendering a nonsense `"1970-01-01"` epoch
> artifact on the non-temporal one. The refusal is deliberately narrower
> than "any spec containing `%`": an ambiguous bare-`%` spec with no
> declared `format_type` (e.g. a plain d3 percent format like `".1%"`) is
> tolerated exactly as before and keeps falling through to the
> numeric/d3 grammar — only a format the caller (or a preset) unambiguously
> declared as time-typed refuses on a non-temporal scale.

> **Dated note (2026-09-22, wave 6A task 13, GH #147 + #146 — one answer to "is this a time format", and an independent facet axis stops dropping its temporal tick format):** six render sites (the shared axis, the independent facet axis, the chart-level applier, the colorbar, `mark_text`'s `text=` channel, and the tooltip column) each re-derived from the wire's `(format, format_type)` pair whether a format is d3, strftime or the ordinal preset, and they disagreed at the edges. They now share one resolution. A spec is read as strftime only when it contains a `%` escape; an undeclared `%`-bearing spec is read as time where the value it formats is temporal (an axis by its scale, a legend or the text channel by its field, see (8)), and the remaining sites (the tooltip column and tick thinning) read only the declared `format_type`. The #133 refusal above is unchanged except that it now also covers a `%`-free time-declared spec on the shared axis and no longer covers the ordinal preset, and that it now reaches the colorbar, the size, shape and stroke-dash legends and the text channel, see (3), (4) and (5). A second refusal mirrors it for the ordinal preset on a temporal value, see (7), and a third for a d3 number format on a temporal value, see (8). Eight outputs move, all of them inputs that were previously dropped or rendered as nonsense: **(1)** On a facet whose x scale resolves independently, `fm.Axis(label_format="%Y/%m/%d")` (with or without `label_format_type="time"`) under `.facet(col=…).share_scale(x="independent")` rendered the scale's own default labels (`2020 2020 2020 2020`), while the shared axis rendered the same chart's dates in the requested pattern. The independent axis now applies the pattern, and the two paths agree. This is the second silent drop found on the independent-facet-axis path in one wave, after `cull_threshold` (2026-09-20 note under §3.13). **(2)** `fm.Text("d", format="%Y/%m/%d", format_type="time")` rendered the default day-granularity date (`2020-01-01`) instead of the pattern it was given; it now renders `2020/01/01`. The tooltip column carries the same fix, which is latent today because `TooltipField(format_type=)` does not yet reach the wire. **(3)** A `%`-free spec declared as time (`label_format=".2f"` or `".3e"` with `label_format_type="time"`) meant three things: a colorbar and thinned ticks echoed its literal text (`.2f` on every label), a temporal per-channel axis applied it as a d3 number to the year labels (`2001.00`, `2.001e+3`), and an independent facet axis re-read those year labels as epoch milliseconds (`1970-01-01`). Every site now falls back to its own temporal default (the axis keeps its scale's labels, the colorbar and text mark their day-granularity default). The #133 refusal's message changed for every case it covers, the existing strftime and time-preset refusals included: it now reads `format ".2f" is typed "time", which on the 'x' axis requires a temporal scale, but the resolved scale is not temporal`, naming the declared type as the cause, where it previously read `format ".2f" on the 'x' axis requires ...`; code matching the old prefix needs updating. On a non-temporal axis the same pair is refused as a time format, by the #133 refusal above, on the shared axis as it already was on the independent one. The thinned-tick case is reachable only through a hand-written `chart_config` dict. The ordinal preset keeps exactly one wire token, `"__ordinal__"` (the `"ordinal"` preset still resolves to it, and both render `1st`, `2nd`, …). What is retired is a second, internal encoding of that sentinel, which let a literal one-character `label_format="\x01"` validate and render ordinals; that input is now an ordinary unrecognized d3 type character and renders plain numbers. **(4)** The ordinal preset beside a stray time type (`format="__ordinal__", format_type="time"`) rendered ordinals on a shared axis, `1970-01-01` on a colorbar and on `mark_text`, and was refused as a time format on a non-temporal independent facet axis. The declared preset now wins everywhere: on a non-temporal field every surface renders `1st`, `2nd`, …, and no axis refuses it as a time format. On a temporal field it is refused, see (7). **(5)** A time-typed format on a non-temporal field rendered on the colorbar, the size legend and `mark_text` as a column of `1970-01-01` labels, the epoch read of small numbers: `fm.Color("n", legend=fm.Legend(format=".2f", format_type="time"))` and `fm.Text("n", format=".2f", format_type="time")` both did this, as did a strftime pattern or a bare `format_type="time"` with no spec. These surfaces now raise the same #133 refusal, naming the surface the format was set on and checking the field rather than an axis scale: `format ".2f" is typed "time", which on the 'color' legend requires a temporal field, but the field is not temporal` (the size legend reads `on the 'size' legend`, the text channel `on the 'text' channel`; a bare type reads `format type "time" on the …`). The shape and stroke-dash legends, which enumerate categories and never format a label, dropped the pair silently where the color legend refused it (`fm.Shape("g", legend=fm.Legend(format=".2f", format_type="time"))` rendered, `fm.Color("g", legend=…)` refused); they now refuse it the same way (`on the 'shape' legend`, `on the 'stroke_dash' legend`), with the same disable exemption. The legend check now walks the one list of legend channels the legend builder walks, so a legend channel cannot be added without meeting it (wave 6A task 22). A field is temporal when its declared type is temporal or, undeclared, its column is a date or datetime, the same rule scale resolution uses, so `"n:T"` over epoch-millisecond numbers keeps its date labels. A string field is not temporal, so `fm.Text("s", format="%Y", format_type="time")` over a string column, which rendered the strings verbatim and dropped the format, now refuses too. A bare `format_type="time"` with no spec now gets the same answer on every surface: on a non-temporal axis (`fm.Axis(label_format_type="time")`, `fm.X("n:Q", format_type="time")`, the shared and the independent facet axis alike) it used to be dropped silently and the scale's numbers rendered; it is now refused (`format type "time" on the 'x' axis requires a temporal scale, …`), and on a temporal axis it still renders the scale's own date labels. A bare `fm.Axis(label_format_type=)` also types the shorthand `fm.X(format=)` beside it, where the type used to be dropped, and the pair so merged meets the same checks as its one-place spellings: `fm.X("d:T", format="%Y", axis=fm.Axis(label_format_type="number"))` refuses with `invalid format spec "%Y": unrecognized token "Y" …`, exactly as `fm.Axis(label_format="%Y", label_format_type="number")` and `fm.X(format="%Y", format_type="number")` do. The shorthand pair an Axis type overrides still meets the #133 refusal, so `fm.X("n:Q", format=".1f", format_type="time", axis=fm.Axis(label_format_type="number"))` refuses as `fm.X("n:Q", format=".1f", format_type="time")` does. The legend refusal validates the spec the caller wrote, not what happens to be drawn: a time format on a non-temporal legend field is refused unless the caller disabled that legend, and the two ways to disable it count alike, per channel (`fm.Color(…, legend=None)`, `Legend(orient="none")`) and chart-level (`.configure_legend(orient="none")`). Axes have no such exemption, on purpose: `fm.X("n:Q", axis=None, format=".2f", format_type="time")` refuses. An `X`/`Y` `Legend(format=)` cascades onto the color field, so validating a legend nobody draws would judge a field the caller never formatted, while an axis format is written on the channel it formats; do not "fix" either side toward the other. A colorbar the renderer itself suppresses with a warning (a `mark_line` or `mark_ribbon` layer, which cannot draw continuous color) still refuses, as does a color legend on a categorical scale. The tooltip column is not covered, because no Python spelling puts a `format_type` on its wire yet. (Wave 6A task 21.) **(6)** An untyped `%`-bearing spec on a non-temporal field meant two things. `fm.X("n", axis=fm.Axis(label_format="%Y"))` over a numeric column with no declared type rendered `250.000000% 300.000000% …` on the shared and the independent facet axis (the axis guessed a date pattern, found no dates, and handed the pattern to the lenient d3 reader), while `Legend(format="%Y")` and `Text(format="%Y")` refused with `invalid format spec "%Y": unrecognized token "Y" …`. The axis now gives that same refusal. A real d3 percent spec (`".1%"`) still renders percentages there, and on a temporal axis an untyped `"%Y"` still renders years. The stroke-dash legend's `Legend(format=)` was never grammar-checked and rendered any spec; it is now checked like the shape legend's. **(7)** The ordinal preset (`"ordinal"`, `"__ordinal__"`, with or without `format_type="time"`) on a temporal field gave three answers: `2001st 2002nd …` on the shared axis, the format dropped on an independent facet axis, and `959817600000th …` (the ordinal of the epoch-millisecond count) on the colorbar, the size legend and `mark_text`. The preset formats numbers, so every surface now refuses it: `the "ordinal" format on the 'x' axis formats numbers and requires a non-temporal scale, but the resolved scale is temporal` (a legend or text channel reads `… requires a non-temporal field, but the field is temporal`). Chart-level `.configure_axis(label_format="ordinal")` over a temporal axis refuses the same way. A shape or stroke-dash legend cannot be built over a temporal field at all, so it never reaches this check. (Wave 6A task 22.) **(8)** Two untyped specs on a temporal field, each settled by the user's ruling (2026-09-22, wave 6A task 23). *A d3 number format* (`.2f`, `,.0f`, `d`, or a `"number"`-typed preset such as `"percent"`) on a temporal field gave three answers: the x and y axes formatted the calendar year (`2001.00 2002.00 …`, `2,001 …`), an independent facet axis dropped the format and showed its date labels, and the colorbar, the size legend and `mark_text` formatted the epoch-millisecond count (`959817600000.00 …`). Every surface now refuses it: `the number format ".2f" on the 'x' axis requires a non-temporal scale, but the resolved scale is temporal; a date pattern (e.g. "%Y") formats a temporal value` (a legend or the text channel reads `… requires a non-temporal field, but the field is temporal; …`). Chart-level `.configure_axis(label_format="percent")` refuses the same way over a temporal axis; that format reaches both axes, so a percent-valued y beside a temporal x needs `.configure(axis_y=AxisConfig(label_format=…))` or the per-channel `Axis`. The typed pair `.2f` + `"time"` is unchanged: it names no pattern and falls back to the temporal default. A split spelling whose `Axis` type retypes a number-format shorthand as time (`fm.X("d:T", format=".2f", axis=fm.Axis(label_format_type="time"))`) gives its one-place spellings' answer, the temporal default: only a time-typed shorthand is checked as written, as in (5), since that is the half an `Axis` type could launder (user ruling, wave 6A task 23 cycle 2). *An undeclared date pattern* (`"%Y"`) on a temporal field rendered years on the axes but was refused by the legends and the text channel as a d3 spec (`invalid format spec "%Y": unrecognized token "Y"`). A legend or the text channel now reads an undeclared `%`-bearing spec by its field's temporality, as an axis reads it by its scale's, so `fm.Text("d", format="%Y")` and `fm.Color("d", legend=fm.Legend(format="%Y"))` over a datetime column render years; over a non-temporal field the spec stays d3 and still refuses, as in (6). An undeclared `%` spec on a temporal field that is not a date pattern (`".1%"`) now refuses on every surface with `invalid format spec ".1%": invalid strftime pattern ".1%" …`, where the colorbar, the size legend and the text channel formatted the epoch-millisecond count as a percentage. The tooltip column is not covered, because its type does not reach the wire. An untyped `".1%"` on a numeric axis is now a d3 spec from the start, so it also formats explicit tick values: `fm.X("q", format=".1%", axis=fm.Axis(values=[0.2, 0.6]))` rendered `0.2`, `0.6` (the spec was guessed as a date pattern, un-guessed and applied to the computed ticks only) and now renders `20.0%`, `60.0%`, as `".2f"` and the `"percent"` preset already did (wave 6A task 23 cycle 3). (Close remediation, wave 6A task 17: the format surfaces answer each `(format, format_type)` pair once, and the resolution now names the "time-typed, no pattern" case itself, so no site re-reads `format_type` after resolving.) No golden moved, because no golden sets a caller-specified tick format; the pins live in `tests/test_format_predicate_unification.py`.

> **Dated note (2026-09-03, batch B task 7 — explicit-equals-default now serializes):** the seven fields shown above with a concrete default (`ticks`, `tick_extra`, `grid`, `labels`, `label_flush`, `label_overlap`, `domain`) follow an **omit-vs-explicit** wire contract: omitting the parameter entirely (the common case, and what every signature default above represents) means "not specified" and the renderer's own default applies as before — byte-identical to today. Passing the parameter **explicitly**, even with the exact value shown above (e.g. `Axis(ticks=True)` or `Axis(label_overlap="greedy")`), now always reaches the wire, which is a real behavior change: an explicit per-channel value beats a conflicting chart-level `configure_axis(...)`/theme value for that field, where chart-level previously won silently regardless of whether the per-channel field was actually named. Previously an explicit value equal to the default was indistinguishable from "not specified" and could be silently overridden by chart-level. `Legend`'s `orient`/`direction` fields (below) carry the identical contract.

> **Dated note (2026-09-03, batch B task 1 — chart-level config refuses unknown keys):** every `configure_*()` domain — `axis`/`axis_x`/`axis_y`/`axis_y2`, `legend`, `title`, `grid`, `padding` (below), `color` — shares one wire-level gate: an unknown key inside any section (a typo, a removed field, a spelling from a different surface) raises `chart config: unknown key '<k>' in <section>; accepted: <sorted list>` before the value ever reaches the Rust deserializer, instead of silently vanishing. Every field the wire schema parses is dispositioned `Honored`/`Rejected` (with a reason) in a completeness-tested manifest, so a field that is accepted syntactically but not yet consumed cannot drift back into silent no-op territory unnoticed.
>
> **Dated note (2026-09-11, wave-1 hardening, #148 — the array-section carve-out closes):** the two ARRAY-shaped `chart_config` sections, `annotations` and `structural`, were the one carve-out this gate didn't cover — each item's `type` tag routed to serde's own internally-tagged deserialize with no per-item key check, so an unknown key inside an `annotations[]`/`structural[]` item (e.g. `annotations[0]` of `type: "text"`) either vanished silently or, depending on the variant, slipped through. The gate now walks each array item, reads its `type` tag, and validates it against that variant's own accepted-key set (8 annotation variants, 2 structural variants — `BreakAxis`/`Inset`), refusing with the same shape the object-keyed sections use, now naming the item index and type: `chart config: unknown key '<k>' in annotations[0] (text); accepted: <sorted list>`. Every one of the 87 newly-covered fields is dispositioned in the same completeness-tested manifest the object-keyed sections already use.
>
> **Dated note (2026-09-11, wave-1 hardening, #92 — `BreakAxis.gap` construction AND wire refusal):** a `BreakAxis(gap=...)` with a non-finite (`NaN`/`±inf`) bound, a malformed pair shape, or a reversed `(start, end)` pair now refuses at BOTH the boundaries the spec's validation-placement doctrine says it should: Python construction (`BreakAxis.__post_init__`, mirroring the `Inset`/`_validate_inset_bounds` idiom already four lines away in the same file), and — for a hand-built `chart_config` dict that bypasses Python construction entirely — the same per-item wire gate #148 closed above, which now also value-checks a `break_axis` item's `gaps[]` entries directly off the raw Python floats (not the post-JSON-bridge deserialized value: `json.dumps(float('nan'))` emits a bare `NaN` token the JSON parser already rejects with an opaque message before any check downstream of it could run, so the value check has to sit upstream of that bridge to give the NaN case a real, pinned message and to actually reach the reversed-pair case, which is otherwise valid JSON that sails through untouched). `Facet(ncols=)`/`Facet(nrows=)` — both the `.facet(ncols=)` chart method and `encode(facet=fm.Facet(columns=))` — gain the matching #238 construction-time check (`validate_positive_int`, refusing anything that isn't a genuine `int >= 1`; `bool` is excluded despite being an `int` subclass); a non-positive or non-integer wrap count used to construct silently and either render nonsense or get overridden by inference. `FacetRow`/`FacetCol(columns=)` never reach this check (see the #240 dated note above under §3.2 — those two channels stay warn-and-drop by design, message corrected only).

> **Dated note (2026-09-03, batch B task 8 — axis config plumbing honored end to end):** `domain_min`/`domain_max`/`nice`/`zero` (chart-level `configure_axis(...)` and the per-channel fields above) now feed positional scale resolution at the X axis, the Y axis, and the secondary Y axis alike. An encoding's own `scale=` domain still wins silently over a chart-level `domain_min`/`domain_max` (the documented cascade), an ordinal axis warns rather than applying a numeric domain it cannot use, a user-set domain on a `log` scale refuses with `LogScale`'s own sentence, `nice=True` on every continuous kind (including `log` and `time`) rounds via that scale kind's own `nice()` too (chart-level and per-scale construction read identically on every field, including `nice` — that parity was fixed 2026-09-03 in the batch-B design-review remediation: a `log` axis under `configure_axis(nice=True)` previously refused with `InvalidScaleDomainConfig` instead of rendering, because the pre-fix implementation rounded every scale kind with the same linear step regardless of kind), and a degenerate `domain_min == domain_max` refuses. Per-axis grid gains a real bool-and-style contract via the new public **`ferrum.GridAxisConfig`** (`enabled`, `color`, `width`, `dash`, `opacity`) — `GridConfig.x`/`GridConfig.y` accept a bare `bool` (enable/disable only, the historical spelling), a `GridAxisConfig`, or its dict form; anything else refuses naming the accepted shapes. Chart-level `tick_count`, `labels`, `ticks`, and the tri-state `title` (omit/`None`/string, matching the per-channel three-way contract above) are honored; `label_overlap` is honored on the **y** axis too (previously x-only). `axis_y2` (`configure(axis_y2=AxisConfig(...))`) is a real consumer end to end — style, axis preparation, and domain resolution all read it for the secondary y axis on a `SecondaryY` chart (previously silently dropped in full: neither color nor `tick_count` reached the rendered SVG). `offset` and `label_flush` are typed `AxisConfig`/`configure_axis()` fields (shift the axis off the plot edge; flush-align the first/last tick label to the axis ends).
>
> **Behavior changes:** grid and domain settings compose by **precedence, not AND** — a chart-level and a per-channel setting no longer both have to agree for either to take effect. `axis_x`-scoped configuration no longer leaks onto the y axis (a pre-existing cross-talk bug). `.override(x_axis_tick_size=...)` now correctly beats `configure_axis(tick_size=...)`, matching the documented per-channel-over-chart-level cascade. `.override(axis_label_format=<raw d3 spec>)` now validates the spec at the override boundary (previously an unvalidated raw string reached the wire unchecked); the escape hatch for a raw, unvalidated spec is `axis_label_format_raw=` on the same surface. `.override(axis_x=...)` / `.override(axis_y=...)` — and the four `x_axis_x`/`x_axis_y`/`y_axis_x`/`y_axis_y` spellings — now raise a typed `FerrumOverrideError` naming the replacement (`Chart.axis(x=False)` / `Chart.axis(y=False)`, documented below) instead of silently no-opping; these were the deprecated, do-nothing spellings of the `x`/`y` visibility flags (see `Chart.axis()` below), not a formatting or layout override.

#### `Legend`

```
Legend(title=None, *, orient="right", direction="vertical", type=None,
       tick_count=None, tick_min_step=None, values=None,
       format=None, format_type=None,
       label_font_size=None, label_color=None, label_limit=None,
       symbol_size=None, symbol_stroke_width=None, symbol_type=None,
       gradient_length=None, gradient_thickness=None,
       columns=None, column_padding=None, row_padding=None,
       clip_height=None, title_font_size=None, title_padding=None,
       offset=None, padding=None, zindex=None)
```

Set `legend=None` on a channel that owns a legend surface (`Color`, `Size`, `Shape`, and the other appearance channels) to suppress the legend for that channel. `X`/`Y` are the exception — see the batch B task 7 dated note below.

> **Dated note (2026-09-03, batch B task 7 — legend contract):** `orient` and `direction` are fully independent. `orient` places the legend block on a chart edge (`"right"`/`"left"`/`"top"`/`"bottom"`); `direction` arranges the entries within it (`"vertical"`/`"horizontal"`) and now also **sizes** the reserved block, so all eight combinations render every entry. Absent `direction`, the edge implies it (side legends stack, top/bottom strips run across). A colorbar honors `direction` too: `"horizontal"` draws a left→right gradient bar with its tick labels centered beneath it.
>
> `orient` and `direction` follow the same **omit-vs-explicit** wire contract as the seven `Axis` fields above: omitting either parameter (the signature defaults `orient="right"`/`direction="vertical"` shown above represent this) means "not specified," and the renderer's own orient-implied default applies — byte-identical to today. Passing either explicitly, even as `"right"`/`"vertical"`, now always reaches the wire, so an explicit per-channel value beats a conflicting chart-level `configure_legend(...)` (behavior change 1 below).
>
> Further clarifications land with it:
>
> - `Legend(orient="none")` on a channel suppresses that channel's legend — the per-channel spelling of chart-level `configure_legend(orient="none")`, identical in effect to `legend=None`.
> - `Legend(values=[...])` on a **categorical** legend filters and orders the entries to the listed values (previously honored only on gradient/colorbar legends, where it replaces the tick labels — unchanged). A value naming no category has no swatch to draw: it is skipped and reported as a `RenderWarning`.
> - `X(legend=...)` / `Y(legend=...)` are honored for explicit style fields. The positional channels have no legend block of their own, so an explicit `Legend(...)` (or a raw dict) addresses the chart's legend, filling any field the `color` channel's own `legend=` left unset. Per-channel precedence is `color` > `x` > `y`. Bare `legend=None` / `legend=False` on `x`/`y` is a no-op (see behavior change 3 below) — there is no "x's/y's own legend" for that spelling to suppress.
> - `orient`/`direction` tokens (both here and on `configure_legend(...)`) validate at construction: an unrecognized value raises `ValueError` naming the accepted set, rather than silently falling back to a default or reaching the wire unrecognized.
> - `Legend(format=...)` resolves named presets through the same shared resolver `Axis(label_format=...)` uses (`configure_legend()`/`LegendConfig` has no `format` field of its own — colorbar/gradient formatting is per-channel only) — see the format-preset dated note under `Axis` above (batch B tasks 2/4) for the full contract (all five surfaces, malformed-spec refusals, `format=`/`values=` composition).
> - **Behavior changes (4):**
>   1. Per-channel `Legend(orient=/columns=/title_font_size=)` now beats chart-level `configure_legend(...)` for those three fields, matching the documented cascade (mark literal > per-channel > chart-level > theme). Previously chart-level silently overwrote the per-channel value.
>   2. `configure_legend(label_font_size=)` now sizes legend labels only; it used to write a slot shared with the axes and so resized axis tick labels as a side effect. Use `configure_axis(label_font_size=)` for those.
>   3. `X(legend=...)` / `Y(legend=...)` reach the chart's legend for every **explicit** style field (`orient=`, `title=`, etc.), through the same per-channel cascade `color`'s `legend=` uses. Bare `legend=None` / `legend=False` is a **no-op** — see the 2026-09-03 design-review-remediation dated note directly below for why this is not "suppress the chart's legend" the way `Color(legend=None)` is.
>   4. A `Legend`/`configure_legend` block oriented `"top"`/`"bottom"` with **no explicit `direction`** now defaults to a **horizontal** gradient bar on a continuous (colorbar) legend, matching the orient-implied default the categorical arm already had. It previously always rendered as a tall vertical bar regardless of orient. Pass `direction="vertical"` explicitly to keep the old bar shape.
>
> **Dated note (2026-09-03, batch B design-review remediation — `X`/`Y(legend=None)` no longer suppresses `color`'s legend):** task 7's original behavior change 3 above routed bare `legend=None`/`False` on `x`/`y` through the exact same "suppress" mapping `Color(legend=None)` uses, so `X("a", legend=None)` suppressed the **color channel's** legend — a cross-channel action at a distance under the most idiomatic "not applicable here" spelling, and a contradiction of the None-means-unspecified rule this same batch codifies for `orient=None`/`ticks=None`/etc. elsewhere (`ferrum._title_sentinel.is_unspecified`). `x`/`y` own no legend surface of their own, so bare `None`/`False` is now a no-op on those two channels: nothing is serialized, exactly as if `legend=` had been omitted, and the color legend (or whichever channel actually owns one) is left untouched. An **explicit** `Legend(...)`/dict passed to `x`/`y` — including one that spells `{"disabled": True}` itself — is an unambiguous, typed ask and still reaches the cascade unchanged; only the bare-boolean spelling's meaning changed. Suppress the rendered legend from the channel that actually owns it (`Color(legend=None)`, `Size(legend=None)`, `Shape(legend=None)`, ...).

#### `Chart.axis()` — spec-level axis suppression (added 2026-05-11)

```
Chart.axis(*, x: bool | None = None,
              y: bool | None = None,
              show: bool | None = None) -> Chart
```

Hides (or shows) the chart's x/y axis line, ticks, tick labels, and axis
title at layout time. Returns a new `Chart`. Mutually exclusive with
per-channel `Axis(...)` configuration on individual encodings (the per-axis
`show=False` always wins).

- `x=False` / `y=False` hides the respective axis.
- `show=False` is shorthand for `axis(x=False, y=False)`.
- Plot-area pixel rect is unchanged — gutters reserved for axis decorations
  stay reserved, so compound views can author each child chart at a fixed
  size and compose with a stable grid.

Used internally by `clustermap()` (dendrogram panels) and `JointChart`
(marginal panels). Replaces an earlier post-render SVG-regex stripper which
was theme-color-fragile.

Serializes to two optional booleans on `ChartSpec`: `axis_x: bool | None`,
`axis_y: bool | None`. Both default to `None` (visible).

> **Dated note (2026-09-03, batch B task 1b — deprecated `override(axis_x=)`/`override(axis_y=)` now refuse):** `Chart.override(axis_x=..., ...)` / `Chart.override(axis_y=..., ...)` — and the four `x_axis_x=`/`x_axis_y=`/`y_axis_x=`/`y_axis_y=` spellings on the same low-level `.override()` escape hatch — previously emitted a `DeprecationWarning` and then silently did nothing (they addressed `AxisConfig`'s vestigial, already-deprecated `x`/`y` no-op flags, not this method). They now raise a typed `FerrumOverrideError` naming the offending kwarg and pointing at the real replacement, `Chart.axis(x=False)` / `Chart.axis(y=False)` documented above. This is a **breaking change**: code relying on the old warn-and-ignore behavior now raises instead.

#### Chart-level Padding — `configure_padding()` / `PaddingConfig`

```
Chart.configure_padding(*, top=None, right=None, bottom=None, left=None, auto=False) -> Chart
```

Sets the minimum pixel padding on each of the plot's four sides, independent of the theme's flat `padding` default (§3.13). Each side is optional and independent: an unset side falls back to the theme value; a set side always wins there, whether or not `auto=True` is also given.

- `top`, `right`, `bottom`, `left` — pixel values. Each must be a finite, non-negative number (`int` or `float`; `bool` is excluded even though it is an `int` subclass). A negative value, a non-numeric value (including a `NormCoord` or any other non-`Real` type — these are pixel slots, not relative-coordinate slots), or a non-finite value (`NaN`, `±inf`) raises a typed `ValueError` naming the pixel contract. Validation is construction-time only, in `PaddingConfig.__post_init__`; there is exactly one entry point, not several that must be kept in sync. The low-level `.override(padding_top=..., ...)` escape hatch (scoped to the four numeric side leaves; `padding_auto` is a separate bool leaf and passes through unvalidated here) reaches this same check because it constructs a `PaddingConfig` from the whole `padding_*` section named in one `.override(...)` call and takes what that construction produces — it is a caller of `__post_init__`, not a second validator. `PaddingConfig.to_dict()` performs no validation at all; it only serializes whatever a construction-time-valid instance already holds, so a value already refused at construction can never reach it.
- `auto` (`bool`, default `False`) — opt-in, per-side auto-expand. **Default stays `False`, so every existing chart renders byte-identical**; `auto=True` is a deliberate opt-in, not a retroactive behavior change.

  When `auto=True`, a side that is **not** explicitly set (`top=`/`right=`/`bottom=`/`left=` all still `None` on that side) is expanded — beyond the theme's flat default — to keep two specific things from clipping past the rendered viewport edge:
  1. **Edge tick-label overhang** on a **continuous** axis — the first/last tick label commonly extends past the plot's nominal edge; `auto` measures the real overhang and reserves enough margin to contain it, respecting any `max_band` cap on that axis's own reserved gutter.
  2. **A clipped axis title** — solved via a closed-form, asymmetric recentering pass that runs *after* the tick-label pass, against the post-tick-expansion insets (the two passes compose sequentially, not independently).

  A side you *do* set explicitly is never touched by `auto`, on that side, regardless of what the measured overhang would otherwise ask for. `auto` does **not** expand for: annotations; ordinal/nominal axis tick labels (which sit inside fixed band slots and never overhang the plot edge by construction); a hidden axis (`Chart.axis(x=False)`/`(y=False)` — the title pass gates on axis visibility). A y-axis whose gutter is capped tight via `max_band` can still leave its own tick label or title outside the canvas on that axis — `auto`'s tick-label pass respects the cap rather than overriding it, so `max_band` is a hard ceiling `auto` will not breach even at the cost of a clipped label.

  `auto`'s total contribution is capped to whatever the viewport can actually still afford: in an infeasible regime (the fixed/floor padding alone already leaves no usable panel area), `auto` contributes **nothing** on that axis — the render is byte-identical to `auto=False` rather than partially shifting the clip point — and `auto=True` can never itself convert a chart that would otherwise render into a `PaddingExceedsViewport` refusal. When the desired expansion doesn't fully fit but the floor does, `auto` applies the largest expansion that does fit (scaled down proportionally, preserving which side needed more) rather than either refusing or overshooting.

- **`PaddingExceedsViewport`** — raised when a padding value (explicit or the theme default) leaves no usable plot area on an axis. The error reports the **caller's actual value and the specific side that overflowed** (`"padding {side}={padding} exceeds viewport dimension {viewport_dim}"`, `side` one of `"top"`/`"right"`/`"bottom"`/`"left"`) — previously it always reported the theme's default of `16` regardless of what the caller actually set, which misdirected debugging on any large explicit override (e.g. `configure_padding(top=1e9)`).

> **Dated note (2026-09-03, batch B task 6 — padding validates at every entry point):** see the pixel-contract refusal described above (`top`/`right`/`bottom`/`left`); this is the point where `NF-B5`/`NF-B7`'s Python half closed. Previously an invalid value (negative, non-numeric, or non-finite) reached the JSON serializer unchecked and surfaced as an artifact message naming neither the field nor the contract (`ValueError: chart_config: expected value at line 1 column 27`).

> **Dated note (2026-09-03, batch B task 9 — `padding.auto` is implemented):** `auto` did not exist before this task; the field, its opt-in default, the tick-label/title expansion passes, the `max_band` interaction, and the honest `PaddingExceedsViewport` reporting are all new in this batch (closes `F-L07-08`/`NF-B6`). No golden output changes for any chart that does not pass `auto=True`.

---

### 3.8 Coordinate Systems

Applied via `.coord()` on `Chart`.

| Class | Description | Key Parameters |
|---|---|---|
| `CoordCartesian(xlim=None, ylim=None, expand=True, clip=True)` | Default Cartesian | |
| `CoordFlip()` | Flipped Cartesian (swap X/Y roles) | |
| `CoordPolar(theta="x", radius="y", *, start_angle=0, end_angle=None, direction="clockwise")` | Polar (pie, radar) | |
| `CoordGeo(projection=None, *, center=None, scale=None, translate=None, rotate=None, precision=None, clip_angle=None, clip_extent=None)` | Geographic | Any D3 projection name |
| `CoordFixed(ratio=1.0, *, xlim=None, ylim=None, expand=True, clip=True)` | Fixed aspect ratio. `ratio` = y units per x unit; default 1.0 enforces equal scaling on both axes. Required for geographic projections, correlation matrices, and any chart where axis unit equality is semantically meaningful. | `ratio` |

> **Dated note (2026-09-06, batch E task 6 — F-L02-09 + verified arc-error sibling, `mark_geoshape` and `mark_arc` refuse a wrong coord instead of rendering wrong or empty):** `mark_geoshape` with a `__geometry__` column present under any coord other than `CoordGeo` (the default `CoordCartesian`, or an explicitly set `CoordFlip`/`CoordPolar`/`CoordFixed`), and `mark_arc` under any coord other than `CoordPolar`, both now raise `ValueError` (`MarkRequiresCoord`, mirroring the mold `UnsupportedChannelCombination` already uses) naming the mark and the `.coord()` call that fixes it — e.g. `"mark_arc requires a Polar coord; add .coord(fm.CoordPolar(...))"`. Neither mark previously refused: `mark_arc` under a non-Polar coord failed later with a generic missing-channel error that did not name the real cause, and `mark_geoshape` with geometry under a non-Geo coord rendered a silently empty panel — axis furniture only, no `<path>`/`<circle>`/`<polyline>` geometry (`geoshape.rs`'s non-Geo arm fell through to `MarkBuildResult::empty`). `mark_geoshape` with **no** geometry column present is unaffected under any coord (stays an empty scene, as documented). `mark_image` is explicitly untouched by this guard — it keeps its own documented behavior of an empty scene under a non-Cartesian coord. A chart already using the correct coord for its mark renders byte-identically. Interactive-only, disclosed: `fm.layer(cartesian_chart, polar_arc_chart).interactive()` now refuses with this same `MarkRequiresCoord` error instead of silently rendering the arc layer as if it were Cartesian — the merged-flat interactive lowering drops a secondary layer's own `coord` (already warned at `composition.py:3303`: "secondary layer's theme/facet/coord is ignored"), so by the time the arc mark resolves it genuinely has no Polar coord to satisfy its own guard. **Static** `to_svg()`/`to_png()` on the identical mixed-coord layered chart is unaffected — the composite overlay tree renders each layer against its own coord and was already correct; verified live (`fm.layer(line_chart, arc_chart.coord(fm.CoordPolar(theta="x"))).to_svg()` renders, `.interactive()` on the same chart raises). The coord-drop itself (rather than the refusal) is a filed follow-up candidate, not fixed by this batch.

> **Dated note (2026-09-19, wave 5 T13 — GH #172(1), a layer carries its own coordinate system):** the note above is superseded in two respects, and the coord-drop it filed as a follow-up is closed. (1) **A layer carries a coord.** `Layer` gained an additive `coord` wire slot (skip-if-absent, so a spec declaring none serializes byte-identically), and `Chart + Chart` — hence `LayerChart`, `.interactive()`, and `resolve={"y": "independent"}`, which all merge through it — carries each secondary member's own coordinate system onto the layers it contributes instead of dropping it. `fm.layer(bar, arc_chart.coord(fm.CoordPolar(theta="x")))` therefore renders the arc member's wedges on **both** output kinds rather than refusing on the interactive one, and `fm.layer(bar, point_chart.coord(fm.CoordFlip()))` renders the point member **flipped** rather than silently unflipped — the half of the defect that produced no diagnostic at all. A layer's coord places that layer's marks; the panel's chrome (Cartesian axes, the polar boundary), its aspect ratio and its scale domains stay the chart-level coord's, which matches what the static composite path draws for the same composition. A per-layer coord that declares a PANEL property instead of a placement — `xlim=`/`ylim=`, `expand=False`, `clip=False`, or `CoordFixed(ratio=)` — is refused with a typed `ValueError` naming the layer, the coord and the property, because the layers share one panel and one of them cannot have its own domain or aspect ratio. Placement-only coords (`CoordFlip`, `CoordPolar`, `CoordGeo`, a bare `CoordCartesian()`) are honored. The `Chart + Chart` merge warning accordingly no longer names `coord`: it now reads "secondary layer's theme/facet is ignored", since the coord is no longer ignored. (2) **The refusal messages no longer prescribe a fix.** `"mark_arc requires a Polar coord; add .coord(fm.CoordPolar(...))"` became `"mark_arc requires a Polar coord"` (and the `mark_geoshape`/`CoordGeo` twin with it). The prescription was false on exactly the path that mattered: the merged-flat layer reached it with the prescribed call already written and dropped. What still refuses is a chart with no such coord anywhere, for which the state is the whole diagnosis.
>
> **Dated note (2026-09-20, wave 5 closing review — GH #172(1), corrected scope on a CATEGORICAL axis):** the note above states the contract for continuous positional axes and **overclaimed** for ordinal/band ones, where a member's own coord was still silently dropping its marks. The cause: a band ladder is built by a different mechanism than a numeric extent (`positional_category_union`, not `numeric_domain_union`), and only the numeric one read each layer's OWN effective coord — the ordinal one read the CHART's for every layer. The corrected contract, normative, is **honored where the panel can hold the member, refused loudly where it cannot, on both output kinds, and never a silently different chart**:
>
> 1. **Honored.** A member's post-flip positional channel now widens the axis it actually occupies, whatever that axis's kind. Measured before: `bar(x="cat:N", y="val") + point(x="val2", y="cat2:N").coord(CoordFlip())` across two frames enumerated the member's VALUE column as three phantom bands (`['a','b','c','40','50','60']`) and drew 0 of its 3 marks on all three merged-flat routes; it now enumerates `['a','b','c','x','y','z']` and draws all 3.
> 2. **Refused.** A coord is unplaceable in two ways, and both raise ONE typed `ValueError` in one voice, naming the layer and the coord: it declares a **panel** property (the note above), or its **channel exchange lands a column on an axis that cannot draw it** — `layer 1: fm.CoordFlip leaves this layer's positional channels in the opposite space from the panel's axes, and the panel's y axis is continuous while the column that lands on it ('cat': Utf8) cannot be drawn against a continuous axis`. The second is `fm.layer(bar, point.coord(CoordFlip()))` over a shared `x="cat:N", y="v:Q"`: the exchange asks the panel's quantitative y to hold a categorical column, and one merged panel cannot hold both. Honoring it would mean resolving that layer through SWAPPED scale slots, i.e. a second scale-resolution mechanism for one feature and a change to the scope stated above; refusing keeps one mechanism and one answer. Before this it rendered with the member's marks simply absent, no error and no warning, while the identical caller mistake spelled `fm.CoordFixed` refused loudly.
> 3. **Scoped per reason, to the routes that genuinely cannot honor it.** The panel-property reason refuses on all three merged-flat routes; the channel-exchange reason refuses on the two that put every member on one shared axis pair (static `chart + chart`, `.interactive()`) and RENDERS on `resolve={"y": "independent"}`, whose member resolves its own y slot from its own post-flip column, and on the static shared-y route, whose members are separate composite leaves. Both were verified to draw every mark. The question the gate asks is per column and per axis, not "a flip plus an ordinal axis": the same composition over an INTEGER category column is placeable and renders, because a continuous axis can read it.

> **Dated note (2026-09-19, wave 5 — GH #95, #113, #172(2), #172(3): what travels through `CoordFlip`):** four flip-fidelity defects close together. All four move rendered output on the shapes they reach; every unflipped chart that does not use `mark_density(orientation="horizontal")` is byte-identical.
>
> 1. **A sort shorthand names the channel you wrote it on (#95).** `sort="-y"` ordered a bar chart's categories correctly until `.coord(fm.CoordFlip())` was added, at which point it aggregated the *category* column, warned twice, and fell back to insertion order; the spelling that worked under flip was `sort="-x"`, for a value the caller had authored on `y`. The channel token in a sort shorthand is the caller's own text and is now resolved in **authored space**, so the shorthand and the `sort=fm.SortField(...)` object form — already flip-invariant, since it names a field rather than a channel — agree under flip as they always did without it. The same two-spaces defect sat in the `shape`, `stroke_dash` and `color` categorical domains, which build their sort the same way, and is fixed with it. Separately, an *ignored* sort spec now warns exactly **once** instead of twice (the provisional placeholder pass and the panel-real pass each reported it); a chart whose data is empty after transforms still gets its one warning.
> 2. **A flipped chart's marks and its axis read one scale (#113).** Under `CoordFlip`, scene construction merged the **authored** chart-level encoding into the post-flip layer-0 encoding, so a channel the flip had already moved (`y2` → `x2`) survived in its pre-flip slot and widened the wrong axis. Measured on a flipped `mark_bar` with a chart-level `y2`: the marks were placed on a y scale spanning roughly `0..72.9` while the axis ticks in the same SVG read `0..3`, a 23-fold divergence inside one file. The chart-level encoding is now moved into post-flip space before it is merged.
> 3. **A series closes its fill on its own baseline (#172(2)).** A `CoordFlip`'d `mark_area` traced its vertices correctly and then closed the filled region on the panel **bottom** — byte-for-byte the closure an unflipped chart emits — so the filled region came out very nearly the complement of the correct one. The same hard-code is reached with **no flip at all** by `mark_density(orientation="horizontal")`, whose near-zero tails degenerate the spurious region to a sliver, which is how it sat latent inside a blessed golden. The value axis is now **declared** rather than inferred: `mark_style` carries an additive, skip-if-none `value_axis` slot (the enum GH #77 added for this purpose), `desugar_density` stamps it, and the three signals compose by **precedence** — a declaration answers outright, and `CoordFlip` plus the stack stamp describe a transformation of a series that was vertical by default and speak only in its absence. As a disjunction a producer-horizontal density under `CoordFlip` stayed "horizontal" on the flip's word alone; the two flips must cancel. A first attempt inferred horizontality from traced geometry and misclassified a vertical area whose rows arrive unsorted (ferrum traces in row order and does not sort); that inference (`traced_edge_axis`) is deleted rather than kept beside the declaration.
> 4. **A flipped ribbon draws (#172(3)).** A `CoordFlip`'d `mark_ribbon` rendered **nothing**, silently: the flip had moved the authored `y2` into `x2` and the drawer read only `y2`. `mark_errorband`, `mark_smooth(ci=)`, the learning and validation curves and the QQ CI region all went band-less with it. The band's curve axis is now a function of the channel actually read. `tests/test_phase_9_e2e/goldens/jointplot_kde_marginals.svg` moved for this: its right marginal now closes flat on its own left edge.
>
> **Dated note (2026-09-20, wave 5 closing review — `CoordFlip` composed with an explicit `Stack(value_axis=)`):** the precedence rule in item 3 above governs `mark_style.value_axis`, whose copy `prepare` swaps into post-flip space with the channels — so by the time a mark reads it, the flip has been spent and `Some(Y)` means "vertical as rendered". `PositionAdjust::Stack::value_axis` is the same kind of declaration in the OPPOSITE space: nothing swaps a layer's `position`, so it arrives as AUTHORED, and it composes with the flip as an exchange rather than a precedence. A declared value axis therefore **moves with the flip**: `Stack(value_axis="x")` under `CoordFlip` puts the value back on y, and `Stack(value_axis="y")` under `CoordFlip` puts it on x, exactly as the undeclared default (vertical) does. Copying item 3's precedence onto the authored field spent the flip zero times, which is what made `mark_histogram(bins=3, orientation="horizontal", multiple="stack").coord(fm.CoordFlip())` draw nine rects — four sharing one `y`/`height` with overlapping widths, the bin column cumulated and the count column used as the bin key — where both its unflipped spelling and its vertical twin drew the correct three. It now matches its vertical twin pixel for pixel, and a caller-written `Stack(value_axis="y")` under `CoordFlip` renders where it used to raise `Stack: x must be a numeric dtype ...; got Utf8`. Unflipped charts and every chart with no explicit `value_axis` are byte-identical; the reconciliation lives at the one triad all three consumers (the draw-time cumulation and the two domain-widening previews) already funnel through, so they cannot disagree about which axis holds the value.

---

### 3.9 Faceting

#### `FacetSpec`

Wraps a `Chart` and repeats it across values of a facet field. Created by `.facet()` on `Chart`.

```
FacetSpec(spec, facet, *, ncols=None, nrows=None, spacing=None, bounds="full",
          columns=None, resolve=None)
```

#### `RepeatSpec`

Repeat a chart across a list of fields (e.g., pairwise matrix).

```
RepeatSpec(spec, *, row=None, column=None, layer=None, spacing=None, columns=None)
```

#### Resolution

```
Resolve(scale=None, legend=None)
```

`scale` / `legend` each accept a dict mapping channel name to `"shared"` or
`"independent"`. `scale` spans `"x"`, `"y"`, `"color"`, `"size"`; `legend`
spans `"color"` and `"size"` only (the channels whose scales can be shared).

> **2026-07-12 (#16):** the `legend` resolution axis is implemented: legend
> resolution **defaults to following scale resolution** — a composite whose
> color/size scale resolves shared renders **one figure-level legend**
> outside the panel grid (per-panel legends suppressed), and
> `legend={"color": "independent"}` opts back into per-panel legends.
> `legend={ch: "shared"}` over a non-shared scale raises `ValueError` at
> lowering. `pairplot(hue=)` and `jointplot(hue=)` get the shared legend by
> default. A plain dict passed to `resolve=` keeps meaning scale resolution
> (back-compat). The originally-declared third axis, `axis=`, is **not
> implemented** and has been narrowed out of this signature; shared axis
> rendering across composite panels is tracked as its own follow-up issue.

> **2026-07-12 (#74):** nested-composite resolve is now a defined rule, not
> emergent behavior. For **`color`/`size`**: a composite node whose
> *effective* mode for the channel is `"shared"` unions the domain (and, per
> the #16 legend band, captures the figure legend) across its **entire leaf
> span** — every descendant leaf, through nested composites and spliced
> overlay subtrees — not just leaves at matching grid positions. A node's
> effective mode is its own explicit `resolve=`/`Resolve(scale=...)` setting
> for that channel when given; an unset node **inherits the nearest
> ancestor's effective mode**. An explicit `"independent"` child opts its
> whole subtree out and resets the chain, so a re-shared node beneath it
> starts its own, separate union. The figure legend band attaches at the
> outermost node whose effective mode is `"shared"`; leaves it covers get
> per-panel suppression, so all three nesting shapes (outer-shared subtree,
> nested-and-outer both shared, a layered chart spliced into a shared
> concat) render exactly **one** legend. **`x`/`y`** are unaffected: they
> keep the pre-existing positional tree-path pairing across congruent direct
> children (grids pair by column — pairplot/compare) with no inheritance.
>
> **`configure_legend(orient="none")`** now suppresses legends through the
> same disabled-legend mechanism `Color(legend=None)` uses: per-panel
> legends, gutter reservations, and the figure-level legend band are all
> cleared. This applies both to a single chart (previously a silent no-op —
> the value parsed but nothing suppressed) and to a composite, where it
> disables every legend the node covers; per the existing all-disabled rule,
> if every participating leaf ends up disabled, no figure legend band is
> emitted. `orient="bottom"` (and every other valid orient) is unaffected.

---

### 3.10 Selections (Interactivity)

Selections define interactive state. They are declared in the spec and resolved by the WASM renderer. In SVG/PNG mode, selections are silently ignored.

| Class | Description | Key Parameters |
|---|---|---|
| `selection_point(*, fields=None, encodings=None, nearest=False, toggle="event.shiftKey", on="click", clear="mouseout", resolve="global")` | Single or multi-point selection | |
| `selection_interval(*, fields=None, encodings=None, translate=True, zoom=True, mark=None, resolve="global")` | Brush / rectangular interval selection | |
| `selection_single(...)` | Alias for `selection_point` with toggle disabled | |
| `selection_multi(...)` | Alias for `selection_point` with shift-toggle enabled | |
| `SelectionMark(fill=None, stroke=None, fill_opacity=None, stroke_opacity=None, stroke_width=None, stroke_dash=None)` | Style the brush rectangle | |

**Using selections in encodings:**

There are two working forms, both driven through the same conditional
pipeline. `encode(<channel>=<ConditionalSpec>)` is the more common one; the
wire channel is taken from the encode key itself:

```
chart.encode(color=selection.when(Color("category")).otherwise(value("#aaa")))
```

The second form embeds the conditional directly on the channel object, for
when the base field encoding needs its own kwargs (`type`, `scale`, `title`,
...) alongside the conditional — `encode()` translates it into the chart's
conditionals list the same way, stamping the channel from the encode key and
keeping the base field encoding:

```
chart.encode(color=Color("category", type="N", condition=selection.when(Color("category")).otherwise(value("#aaa"))))
```

A pre-built `ConditionalSpec` can also be applied without going through
`encode()` at all, via `Chart.conditional(spec)` — the chart-level sugar for
a spec already carrying its own selection.

> **2026-06-01, superseded 2026-09-11 (batch H, F-L13-03/F-L15-01):** a
> conditional built by `sel.when(...).otherwise(...)` or
> `fm.when(sel).then(...).otherwise(...)` may be passed directly to
> `encode(<channel>=cond)` — the first working form above. The wire channel
> is taken from the encode key (so `encode(opacity=fm.when(sel).then(1.0)
> .otherwise(0.2))` resolves the numeric branches as opacity, not colour),
> and the conditional's source selection is auto-registered — no separate
> `.add_selection()` is required. `Chart.conditional(spec)` likewise
> auto-registers the spec's source selection when it carries one. A bare
> number with no channel context (e.g. via `fm.value(0.5)` outside an
> `encode` key) defaults to the `opacity` channel.
>
> Batch H closed the gap this note left open: the second working form —
> `ChannelClass(field=..., condition=<ConditionalSpec>)` — previously
> serialized into the per-channel `EncodingSpec.condition` field, a nested
> blob the wire accepted but no renderer consumed. `encode()` now intercepts
> it the same way: the conditional is appended to the chart's conditionals
> list (channel stamped from the encode key, selection auto-registered), and
> the channel itself encodes normally minus the `condition` kwarg. A
> `condition=` value that is not a `ConditionalSpec` raises `TypeError` at
> encode time, naming both working forms. **Which channels a conditional may
> target — and the typed `ValueError` for the ones it may not — is stated
> once under §3.2's appearance table (dated note, 2026-09-11); all FOUR
> entry points here (`encode(<channel>=cond)`,
> `<Channel>(field, condition=cond)`, `Chart.conditional(spec)`, and
> `Chart.layer(Layer(encoding={<channel>: cond}))` — batch H close cycle
> 3/4, H3-1, see §3.2's fourth-entry-point note below) resolve through that
> one vocabulary.** (This supersedes the stale example
> and the `.conditional(sel, color=..., else_color=...)` sentence that used
> to sit above this note — `Chart.conditional()` never took per-channel
> kwargs; it takes a single pre-built `spec`.)
>
> Separately, `sel.when(...)`/`.otherwise(...)` branch values now accept a
> bare `str` as field shorthand (parsed with the same grammar `encode()`
> uses) — a bare string means a **field** everywhere in ferrum; a literal
> still requires `fm.value(...)`. This is what makes
> `brush.when("category:N").otherwise(fm.value("lightgray"))` (the Altair
> migration guide's example) work as printed.

> **2026-09-02 (batch A):** selection styling (`SelectionMark(fill=...)` and
> `fm.value(<string>)` used as a color) now routes color strings through the
> same 148-name/hex/`rgb()` parser as mark construction (see §3.3's color
> vocabulary note) — a parseable string resolves to a color dict, a number
> still resolves to opacity, and an unparseable string raises `ValueError`
> naming the accepted forms; the former silent opacity-`1.0` fallback for a
> bad string is removed. Selection styling deliberately **diverges** from
> the mark boundary on the two clearing spellings (`"none"`/`"transparent"`):
> the selection wire dict has no cleared-paint representation, so both are
> refused with a message naming the reason ("selection styling cannot
> express '`none`' (no paint); provide a color") rather than the generic
> accepted-forms text. A wire representation for a cleared-paint selection
> is a logged follow-up, not silently absent. A non-string, non-number
> literal passed to `fm.value(...)` now raises `TypeError` (matches the
> sibling terminal raise) instead of silently falling through.

> **Dated note (2026-09-11, batch H task 7 — F-L13-01, `Parameter.ref()` is now consumed at all three documented sites):** `fm.param(...).ref()` — a `{"param": <name>}` marker distinct from passing the `Parameter` object itself — reaches real behavior at every site the docstring names, through one shared decoder. At a scale `domain=` (§3.6), a ref-shaped domain is now treated identically to a raw `Parameter` domain (emits `domainParam` by name); the parameter still needs `add_params()` to reach the wire, since a `.ref()`'d domain carries only the name, not the auto-registration a raw `Parameter` domain gets. At `transform_filter(...)` (§3.5), a ref no longer silently mangles into the garbage predicate `datum.param == "<name>"` (a dict with a `"param"` key previously matched the constraint-dict branch before reaching the parameter check) — `transform_filter(q.ref())` now equals `transform_filter(q)` exactly. At a conditional value site (`sel.when(q.ref())`/`.otherwise(q.ref())`, above), resolution needs the chart's registered parameters, not just a name: `Chart.to_spec()` now builds and threads that registry into conditional serialization, so a real chart resolves a ref to the named `fm.param()`'s current static value, raises `ValueError` naming the parameter when it isn't a registered value-bearing `fm.param()` (e.g. a `Selection` ref, or an unregistered name), and raises when the parameter is registered but carries no `value`. Reactive (WASM-side) re-resolution of a conditional ref when the underlying parameter changes at runtime is a logged follow-up — this batch ships static resolution only, matching every other conditional branch's semantics.

---

### 3.11 Annotations

Annotations are lightweight overlays that don't participate in scale domain calculation.

| Class | Description | Key Parameters |
|---|---|---|
| `annotate_hline(y, *, label=None, label_position="right", stroke=None, stroke_dash=None)` | Horizontal reference line | |
| `annotate_vline(x, *, label=None, stroke=None, stroke_dash=None)` | Vertical reference line | |
| `annotate_rect(x1, x2, y1, y2, *, fill=None, opacity=0.1, label=None)` | Shaded rectangle region | |
| `annotate_text(x, y, text, *, dx=0, dy=0, anchor=None, align=None, baseline="middle", font_size=None, color=None, angle=None)` | Free text annotation | |
| `annotate_arrow(x1, y1, x2, y2, *, label=None, label_side="start", stroke=None)` | Arrow with optional label | |
| `AUCLabel(*, position="end", format=".3f", prefix="AUC = ")` | Auto-placed AUC annotation on ROC curves | |
| `OutlierLabel(*, threshold=3.0, field=None, label_field=None, max_labels=10)` | Label high-leverage or high-residual points | |

> **2026-06-22 (COMP-06):** `annotate_text` now takes the canonical `anchor=`
> keyword in the SVG vocabulary (`"start"`/`"middle"`/`"end"`), matching
> [`annotation.text`][]. The former `align=` keyword (`"left"`/`"center"`/
> `"right"`) is retained as a non-breaking **deprecated alias**, mapped via
> `{left: start, center: middle, right: end}`. When neither is supplied the
> resolved anchor is `"middle"` (the same render as the historical
> `align="center"` default). Supplying **both** `anchor=` and `align=` raises
> `ValueError`. This reconciles the annotation-pair vocabulary drift: the
> `annotate_text` (mark-Chart) and `annotation.text` (dataclass) surfaces now
> share one anchor vocabulary.

---

### 3.12 Compound Views

> **Dated note (2026-09-19, wave 5 — GH #247B: a band exists only if something is drawn in it):** a layered chart built from two DataFrames grew a `null` band on its ordinal axis that no mark ever occupied — the merge's diagonal concat pads each frame's columns over the other frame's rows, and those manufactured nulls reached categorical domain resolution as if the caller had supplied them. Every real band was narrower for it, and the axis described a category the chart did not draw. The rule is now stated positively: **a null in a positional ordinal column claims the null-category band only when some null-category row carries a non-null value on the opposite positional channel.** Nullness only — a non-finite paired value (`NaN`/`±inf`) counts as occupancy, because the marks still resolve it to a pixel rather than skip the row; only a true null means the row has no value at all. The standing policy that a **genuine** null category carrying a real value gets its own band is intact and pinned in both directions, unlayered and inside a merge, so a null group and a real `0`/`false`/`""` group remain two categories (FA-9). Null-free columns are byte-identical.
>
> The fix is not scoped to the merge, which is where the design doc first placed it. A single **unmerged** chart over `cat=["a","b","c",None], val=[1.0,2.0,3.0,None]` renders the same phantom band and draws three marks, so a merge-scoped fix was provably incomplete — and no merge-side data-shape change could be complete either, since a merged batch must be at least as tall as its taller frame, so with unequal heights the shorter frame's columns necessarily carry entries its data never supplied and all three candidate fill values are wrong. **Disclosed:** any chart whose positional ordinal column carries unoccupied nulls — including every cross-DataFrame layered chart — loses that band, so its remaining bands widen and its axis labels shrink by one.
>
> **Per layer**, the null band is judged against the positional field **that layer itself binds**, and a layer binding none drops the null category rather than inheriting the "no opposite channel bound → keep the band" arm. Carrying that arm across the layer boundary would keep a band for an injected reference-line column, whose single authored value sits in structural nulls and whose only paired coordinate is *inherited*; measured, it draws four rule lines where one is correct. The narrower rule is the same principle applied honestly: a layer with one positional channel has no paired value, so nothing is drawn in that band. It is non-widening — such a layer draws no mark there today either.

> **Dated note (2026-09-19, wave 5 — the shared ordinal axis covers every layer's categories; user ruling):** `Chart({cat: [a, b, c]}) + Chart({cat: [x, y, z]})` drew **three marks for six rows**. A layered chart's ordinal axis read its categories off the chart-level column alone and never unioned its layers', so the second layer's marks were dropped for want of a band and its categories never even reached the axis labels. The numeric domain has had a cross-layer union for exactly this; the ordinal domain had no sibling, so the invariant "the shared axis covers every layer's extent" was half implemented. Every category any layer's positional column carries now gets a band, **ordered by first appearance in layer order** — not a new semantic, but the existing within-layer first-appearance rule carried across the layer boundary rather than replaced by sorting. The disjoint case renders six bands and six marks; overlapping layers contribute only what they add, and a layered chart whose layers share identical categories renders byte-identically (16 of 16 shapes, measured), which is the control proving this is a widening and not a rewrite. Ordering and scope were a user ruling, taken because the choice sets every band position and is visible in every render. **Disclosed: charts may now draw marks they previously dropped silently, and their band widths change accordingly.** Deliberately unchanged: the **explicit-scale** ordinal path still does not union layers, because the explicit numeric path has the identical property and fixing one side alone would break that symmetry.

> **Dated note (2026-09-19, wave 5 — GH #182 follow-on: which cells share an extent is declared, not inferred):** removing #182's phantom gutter (see §3.7) exposed a real defect it had been masking: **a composite's satellite panels were never aligned with the panel they annotate.** A `jointplot` marginal and a `clustermap` dendrogram are each their own leaf, laid out independently and fitted into a slot, so their plot rects and the centre panel's had no reason to agree; they *looked* aligned because the phantom inset them by roughly the centre's axis band. Measured post-#182, the `jointplot_kde_hist` top marginal spanned x `53.377..624.0` against the joint panel's `67.423..547.025`, and the `clustermap_basic` column dendrogram spanned `176.0..744.0` against the heatmap's `213.4..674.8`. The rule that closes it: **a panel that annotates another spans exactly that panel's extent on the axis the two share, and keeps its own on the other**, imposed at the same single point in layout that the #89A overlay pre-pass uses, and scaled by the follower's native size over the leader's because the grid planner ratio-fits each cell.
>
> The trigger is **declared on the wire**: `CompositeNode::Composite` carries `extent_alignments: Vec<{follower, leader, axis}>` (cell positions, additive, `skip_serializing_if` empty, byte-identical round-trip for a spec declaring none), emitted by the `JointChart` and `ClusterMapChart` lowerings. The shape predicate that recognized a jointplot by its grid geometry is **deleted**, not kept beside it: a clustermap is also a 2x2 grid with a hole, so the predicate separated two different figures by which corner was empty. A declaration naming a **nested composite** cell cannot be honored (such a cell has no single plot rect) and now warns (`RenderWarning::ExtentAlignmentUnresolvable`) naming the follower, leader and axis, instead of dropping silently. A follower that keeps a legend of its own is still left unaligned rather than drawn across it. **Named as pre-existing and NOT fixed here:** a dendrogram's own x scale places its leaves at the extremes of its plot rect (≈1.7% and 98.3%) while the heatmap places its columns at band centres (12.5% and 87.5%), so leaf-over-column is closer after this change (worst error 123px → 50px) but not exact. That is the dendrogram chart's scale, not the panel's extent, and it is filed as a separate fix. **All `jointplot` and `clustermap` output changes geometry**; five goldens moved for it (`clustermap_basic`, `clustermap_row_col_dendrograms`, `jointplot_kde_hist`, `jointplot_kde_marginals`, `composite_figure_title/joint_figure_title`), each regenerated and PNG-inspected against its pre-fix rasterization rather than judged alone.

> **2026-07-05 (Phase B, composite render unification / #45):** every compound
> view in this section now renders through ONE Rust composite entry per output
> kind (`render_composite_svg` / `render_composite_interactive`): Python lowers
> the composition to a composite spec tree; Rust resolves `resolve=` sharing
> across the tree's leaves (x/y/color/size; congruent position-wise pairing),
> plans the layout (ratio cells via per-panel `LayoutScale`, holes for empty
> corner/trailing/empty-data cells), and emits one scene. The
> `compose_svg_*` helpers referenced by the historical notes below NO LONGER
> EXIST (removed from the public API); static and interactive output share the
> same layout, subsuming the former W4/W5 interactive limitations. Behavior
> change: an all-empty compound view (every child zero-row) raises a
> `ValueError` instead of rendering a blank grid. Historical notes below are
> retained for the pixel-semantics decisions they record (spacing, ratio math),
> which carried over into the composite layout pass.

> **Dated note (2026-09-11, wave-1 hardening, #140 — figure-level config warnings dedupe, per-leaf data warnings still fan out):** a composite whose N leaves are structural clones of the same chart-level config declaration (e.g. `Repeat`/facet applying one identical `ChartConfig` override to every panel) used to emit N identical `RenderWarning::ConfigSurfaceNotPresent` warnings for what is one user mistake made once — the composite's aggregated warning list now dedupes that one variant by `PartialEq`, so it survives exactly once regardless of leaf count. This is a narrow, class-scoped dedup, not a global one: every other warning class a leaf can emit — a leaf's own degrading transform, `GroupCardinalityDegenerate`, `MarkBuildEmpty`, the per-axis annotation-suppression rollups, and (deliberately, since it resolves per leaf's own `encoding.scheme`) `UnknownColorScheme` — still fans out per leaf unconditionally, because two leaves independently hitting the same-looking data condition are two independent facts, not one declaration replicated.
> a 2×2 layout (center + optional top/right marginals) backed by
> `ferrum._core.compose_svg_grid`; it supports `.theme()`, `.properties()`,
> `.save()`, `.show()`, and `_repr_svg_`. `RepeatChart` gains two new
> parameters: `diagonal=` (a separate template chart used for cells where
> `row_field == col_field` in symmetric n×n repeats) and `corner: bool`
> (filters the expanded grid to the lower triangle, including the
> diagonal). Per-cell field substitution is keyed by three typed
> sentinels — `Repeat.column`, `Repeat.row`, `Repeat.layer` — passed to
> the template's `.encode(...)`; string sentinels (e.g. `"repeat:column"`)
> are not supported. A new compound view `ClusterMapChart` is added:
> a 2×2 grid combining a clustered heatmap with row and/or column
> dendrograms; `dendrogram_ratio: float in (0, 1)` controls relative
> dendrogram-to-heatmap size. Most users construct `ClusterMapChart` via
> `ferrum.clustermap(...)`.

| Class / Operator | Description | Key Parameters |
|---|---|---|
| `HConcatChart(*charts)` / `chart1 \| chart2` | Horizontal concatenation | `spacing`, `resolve`, `title` |
| `VConcatChart(*charts)` / `chart1 & chart2` | Vertical concatenation | `spacing`, `resolve`, `title` |
| `LayerChart(*charts)` | Layer overlay (same axes) | `resolve`, `title` |
| `ConcatChart(*charts, columns=None)` | General wrapping concatenation | `spacing`, `resolve`, `columns` |
| `RepeatChart(spec, row=None, column=None, layer=None)` | Repeat across field lists | `spacing`, `columns`, `resolve` |

> **2026-07-11 (secondary y-axis, #52):** the `LayerChart(*charts)` row's
> "same axes" description is the default (`resolve={"y": "shared"}`, the
> byte-stable pre-#52 path) but is no longer the *only* behavior.
> `LayerChart(a, b, resolve={"y": "independent"})` now renders a real
> dual-axis chart: layer 0's y-scale drives the left axis and gridlines,
> and each subsequent independent layer resolves its own y-scale and
> gets its own right-side axis (stacked outward, unbounded layer count).
> One implementation serves both static SVG and interactive output (the
> merged flat single-panel path — see CLAUDE.md "Composite rendering").
> `x` stays shared across every layer regardless of `resolve`; per-layer
> independent x (dual-x) remains a typed `ValueError` naming GH #55. See
> `SecondaryY` below for the sugar built on top of this mechanism.

> **2026-08-27 (P2 chrome dedup; retired 2026-08-28 by residuals batch
> #89 — current behavior below, original gate preserved as amendment
> history):** static shared-`y` `LayerChart` output previously drew every
> axis line, tick label, grid line, and title once **per layer**,
> overprinting at identical coordinates (2 layers = 2x every chrome
> element; layers binding different y fields overprinted both axis
> titles). Dedup is now **total**. For an Overlay composite node whose
> direct children are all leaves, every leaf lays out against one shared
> plot rect — the intersection, per side, of the leaves' natural gutters
> (title/legend/axis bands) — computed in a composite planning pre-pass,
> so layout never needs the old post-layout `plot_area` overwrite.
> Suppression (clearing grid, axes, above-marks axis chrome, and scene
> title on non-primary leaves) is coupled to imposition: it applies to
> every leaf that actually laid out against the shared rect, which after
> the shared pre-pass is every leaf in a well-formed group. The three
> former refusal shapes — a non-primary leaf with its own legend, a
> `zindex >= 1` axis, or a `z="below_marks"` annotation — now render with
> one chrome and full content preserved (legend renders, annotation
> renders below marks, above-marks axis renders once); legends are never
> suppressed, so their gutters still participate in the shared rect on
> every leaf. Three cases still keep per-leaf chrome, never suppression
> without geometric alignment: (1) an Overlay node whose children are NOT
> uniformly direct leaves — e.g. one child is itself a nested composite —
> is structurally excluded from grouping up front, so every child keeps
> its own rect and chrome exactly as before this batch (unreachable from
> Python lowering: `LayerChart`, the sole producer of an Overlay node,
> rejects any non-leaf layer with a typed `ValueError` before a tree is
> built; the guard exists only because a directly-constructed wire spec
> could still express it); (2) a degenerate intersection — the leaves'
> combined gutters leave no common plot area — drops that group's
> suppression and emits `RenderWarning::OverlayGuttersDiverged` naming
> the layer count, so the resulting doubled chrome is diagnosable rather
> than silent; and (3) a member whose own layout fails during the
> imposition pre-pass also drops its group's suppression, with **no**
> warning — that leaf's render re-runs the identical layout moments later
> and raises the failure as a typed error at its canonical position, so a
> chrome-suppression warning here would be noise ahead of an outright
> render failure, not a silently-wrong chart. Interactive LayerChart
> (merged flat path) and independent-y are unchanged. See CLAUDE.md
> "Composite rendering" for the Rust-side mechanism.
>
> **Amendment history:** the original 2026-08-27 note described dedup as
> gated by a per-leaf safety check (`overlay_imposition_safe`): a
> non-primary layer carrying its own color/size legend, an axis drawn
> above marks (`zindex=1`), or a `z="below_marks"` text annotation kept
> its pre-fix per-layer chrome at its own rect rather than sharing the
> primary layer's, with the gate's removal tracked as a follow-up issue.
> Residuals batch #89 (2026-08-28) deleted that gate, along with
> `overlay_imposition_safe` and `chrome_suppressed` themselves — neither
> symbol exists in the codebase anymore — and the totality behavior
> described above replaced it.

> **2026-09-10 (batch F, F-L10-01 rider):** a layered chart (any compound
> view lowering to `spec.layers` -- `LayerChart`, catplot's `"point"`/
> `"bar"` kinds, and every composite-mark desugar) now refuses, rather
> than silently drops, a layer at index >= 1 whose POSITIONAL encoding
> (`x`/`y`/`x2`/`y2`) names a column absent from that layer's own resolved
> batch: `RenderError::LayerUnknownColumn` names the layer index, the
> channel, and the column. Previously only layer 0's encoding was
> validated before scale resolution; every subsequent layer's per-mark
> builder swallowed the missing column via `.ok()` and rendered nothing
> for that layer, with no error and no visible signal beyond an
> inexplicably absent series. Explicitly unaffected, by design: an
> INHERITED color channel absent from a layer's batch still renders
> (structural, owned by #106 -- see the chrome-dedup note above); a
> legitimately empty (zero-row) layer still skips rather than raising; a
> `data_source=` layer is checked against its OWN resolved batch, not
> layer 0's. The gate surfaced two pre-existing, now-fixed bugs it made
> visible for the first time: the discrimination-threshold desugar was
> rendering its optimum-label/threshold-line layers even when the
> f1-gated sentinel columns were never injected, and a secondary-y smooth
> test asserted a fit polyline against a column name the transform never
> actually emitted.

#### `JointChart`

Compound view with a center plot and optional marginal distribution plots sharing the center's x-axis (top) and y-axis (right). The marginals are independent `Chart` objects, typically `mark_histogram` or `mark_density`.

```
JointChart(center, *, top=None, right=None, ratio=5, spacing=10.0)
```

`center`: any `Chart`. `top` / `right`: marginal `Chart` objects; x-axis of `top` is shared with center x; y-axis of `right` is shared with center y. `ratio`: size ratio of center panel to each marginal. `spacing`: gap between panels in pixels.

> **2026-05-11 (Phase 9, P2.6):** `spacing` is **pixels**, not a fraction.
> The original spec described `spacing` as "a fraction of total size" but
> the Rust grid compositor has always treated the value as pixels —
> `spacing=0.02` rendered as zero gap (one-fiftieth of a pixel). The
> default for `JointChart` / `RepeatChart` / `ClusterMapChart` was
> bumped from `0.02` (effectively zero) to `10.0` to match
> `HConcatChart` / `VConcatChart`. Affected goldens were re-blessed.

> **2026-05-12 (Phase 9, F20 audit):** The grid compositor allocates
> slot dimensions as `K * ratio[i]`, where `K = min_i(native_dim[i] /
> ratio[i])`. The dominant row/column renders at its native size; smaller
> cells are stretched into their declared ratio via nested
> `<svg viewBox preserveAspectRatio="none">` wrappers. This is the
> algorithm that gives `JointChart(ratio=5)` its 5:1 center-to-marginal
> proportions and that drives the `row_ratios` / `col_ratios` parameters
> on the internal `compose_svg_grid` binding.
>
> **Non-uniform-stretch caveat:** the `preserveAspectRatio="none"`
> wrapper distorts glyphs when the slot's aspect ratio diverges from
> the cell's native aspect ratio. For `JointChart` this is harmless —
> the shared data axis stays aligned by construction; only the redundant
> count/density axis stretches. Compound views with intrinsic shape
> constraints (dendrograms, geographic projections, fixed-aspect
> coordinate systems) must pre-resize their cells so the slot ratios are
> already satisfied at native size. `ClusterMapChart` is the example
> here: dendrogram heights and widths are pre-computed to match the
> heatmap's row/column slots so `K_w` and `K_h` collapse to unity and
> no scaling fires.

> **2026-05-11 (Phase 9, P2.5):** `RepeatChart` now ships the previously
> dormant `columns`, `layer`, and `resolve` kwargs:
>
> - `columns: int` — wrap width for 1-D repeats (only `row=` or only
>   `column=` is set). Defaults to a single row for column-only repeats
>   and a single column for row-only repeats.
> - `layer: list[str]` — each cell stacks one resolved template copy
>   per layer field, combined via the `Chart +` overlay operator.
>   Diagonal cells (when `diagonal=` is set on a 2-D grid) skip layering.
> - `resolve: dict[str, "shared" | "independent"]` — per-channel scale
>   sharing. `"shared"` computes the union domain across every layer of
>   every cell and injects an explicit `scale=` dict so the participating
>   axes match. `"independent"` keeps per-cell domains (the default for
>   unlisted channels).
>
> Asymmetric `diagonal=` (set with `row != column`) is now a `ValueError`
> at `expand()` time — previously it emitted a `UserWarning` and silently
> dropped the diagonal template.

Most users construct `JointChart` via `ferrum.jointplot(...)`.

#### `SecondaryY`

```
SecondaryY(field, mark="line", axis=None, color=None, opacity=None, scale=None)
```

`chart + SecondaryY(...)` adds a second, independent y-axis to `chart`.
`field`: data field mapped to the secondary y axis (required). `mark`:
mark type for the secondary series (default `"line"`). `axis`: per-axis
`Axis` config applied to the right-side y2 axis. `color` / `opacity`:
literal mark styling for the secondary series. `scale`: `Scale` config for
the secondary y axis. The base chart must carry an `x` encoding for the
secondary layer to inherit; adding `SecondaryY` to a chart with no `x`
raises `ValueError`.

> **2026-07-11 (secondary y-axis, #52):** `chart + SecondaryY(...)`
> desugars, at `Chart.__add__` time, to one appended layer on `chart` —
> mark `mark`, `y` encoding on `field` (carrying `axis`/`scale`), `x`
> inherited from the base chart's own x encoding, color literal `color`,
> opacity `opacity` — flagged as an independent-y layer using the same
> per-layer y-scale-slot mechanism `resolve={"y": "independent"}` uses
> above. The base chart's own layer(s) are unchanged: a layered base
> chart keeps its existing layers sharing the left axis while only the
> appended `SecondaryY` layer gets its own right axis; adding multiple
> `SecondaryY` instances stacks multiple right axes outward, same as
> stacking multiple independent layers directly.
>
> This re-bases `SecondaryY` off its former standalone Rust mechanism
> (`SecondaryYSpec` / `StructuralSpec::SecondaryY`, which rendered in the
> static overlay path only and was inert in interactive output) — that
> Rust mechanism no longer exists in the crate. One secondary-axis
> mechanism remains. Behavior deltas from the pre-#52 renderer: (1) the
> secondary series' plot area now genuinely narrows to reserve a
> right-side margin band, rather than the axis overdrawing the plot; (2)
> the secondary axis is fully interactive (tooltips, zoom/pan,
> hit-testing) like any other layer; (3) `color`, when omitted, no longer
> falls back to a hardcoded `#E45756` — the secondary mark defaults like
> any other mark (theme default).

All compound views accept `.theme()`, `.properties()`, `.save()`, `.show()`.

#### `BreakAxis`

```
BreakAxis(axis, gap, break_size=12, break_style="slash")
```

`chart + BreakAxis(...)` removes one or more ranges from a continuous positional scale, so a distant outlier cannot compress the rest of the data into an unreadable band, and draws a break indicator where each removed range was.

**This block is the current contract, consolidated 2026-09-13.** Everything under it, from the 2026-09-11 batch-H note down, is the historical record of how each piece of this contract was reached: several of those notes retract or supersede claims in earlier ones, and they are kept verbatim for the decisions and measured symptoms they record. Where a note and this block disagree, **this block is current**.

**Parameters.**

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `axis` | `str` | required | Which **physical** axis to break: `"x"` or `"y"`. Physical, not encoding-relative: under `CoordFlip()`, `axis="y"` still breaks the physical y axis, which by then carries the flipped chart's x encoding. |
| `gap` | `tuple` or `list` | required | One `(start, end)` pair, or a sequence of pairs. Each bound is numeric (`int`, `float`, or a numpy scalar: `np.int64`, `np.float32`, ...) or temporal (ISO-8601 date/datetime `str`, `datetime.date`, `datetime.datetime`, `numpy.datetime64`, `pandas.Timestamp`). |
| `break_size` | `float` | `12` | Pixels of axis reserved for each drawn break indicator. |
| `break_style` | `str` | `"slash"` | Indicator style: `"slash"`, `"zigzag"`, `"wave"`, or `"gap"` (whitespace, no indicator drawn). |

**Construction and validation** (`ferrum.structural.BreakAxis.__post_init__`). Every refusal below is a `ValueError` at construction time naming the offending value. A temporal bound is coerced to epoch-milliseconds UTC by `ferrum.annotation.coords.temporal_coord_to_epoch_ms`, literally the same function the encoding layer's TimeScale-domain coercion and annotation coordinates go through, so those three Python surfaces accept the same types; naive bounds are treated as UTC, aware ones converted to UTC first. One surface stays narrower, by omission rather than design: the Rust-backed `TimeScale(domain=[...])` constructor extracts its domain in Rust and refuses `numpy.datetime64`, which the three Python seams accept.

- `axis` must be `"x"` or `"y"`; `break_style` must be one of the four named styles.
- `gap` must be a tuple or list, must hold at least one pair, and every pair must have exactly two bounds.
- Both bounds of one pair must be the same kind: a pair mixing numeric and temporal is refused.
- Each coerced pair must be finite (no NaN, no ±inf) and ordered (`start < end`; equal bounds are a zero-width gap and are refused).
- A string bound that is not ISO-8601 is refused as an unparseable date, not as "not numeric".
- A not-a-time bound (`pandas.NaT`, `numpy.datetime64("NaT")`) is refused by name, as is a `numpy.datetime64` too large to represent as a microsecond-precision epoch offset (proven in arbitrary-precision integers before any fixed-width cast can wrap).
- Not construction errors, because the constructor cannot see the data: a gap outside the resolved domain, and a temporal gap on a non-temporal axis. Both render as if the break had not been declared, and warn (see Warnings).

The stored `gap` keeps the container type it was given (a tuple of pairs stays a tuple, so a tuple-spelled `BreakAxis` stays hashable) with each pair canonicalized to a 2-tuple of coerced floats. The wire form is `{"type": "break_axis", "axis", "gaps": [[start, end], ...], "break_size", "break_style"}`; the Rust `chart_config` gate accepts `f64` bounds only, so a hand-built config dict must carry already-converted epoch-ms numbers, not date strings.

**Render model.** Gaps are normalized once per axis before any geometry is built: non-finite bounds dropped, each gap clipped to the resolved domain, empty results dropped, then sorted and merged, so overlapping, nested, adjacent, and unsorted gaps all reconcile into the same union and an indicator only reserves pixels for a gap it actually draws. Declaration order does not matter. The surviving gaps become a piecewise scale whose segments tile the **broken scale's own padded pixel range**, not the raw plot rect, so a broken axis's ticks, marks, gridlines and indicator sit on the same pixels as the unbroken chart's, and a value past the domain clamps to the domain's own end pixel. Each surviving gap consumes `break_size` pixels.

**Per-mark-kind behavior**, decided by what the geometry *encodes* rather than by node shape (`BreakPlacement::for_kind`, exhaustive over mark batch kinds):

| Mark batch kind | Behavior under a gap |
|---|---|
| `bar`, `area`, `rect`, `ribbon`, `polygon` (incl. `mark_violin`) | Clipped: one sub-node per visible segment it intersects, nothing when it lies wholly inside a gap. A cut edge that the gap itself produced is **squared** on a rounded (`corner_radius`) piece; corners inherited from the source rect keep their radius. |
| `point`, `tick`, `text`, `image` | Translated whole to the anchor's compressed position, keeping drawn size and shape, and hidden whole when the anchor is inside a gap. The anchor is the row's own scale-mapped position, so an asymmetric shape and a `dx`/`dy`-offset label hide on their data position, not on their drawn extent. |
| `label` | The text moves as a glyph; the leader line's two ends are two data positions and are placed independently. |
| `line`, `rule`, `arc` | Split: one sub-node per visible piece, cut at each gap edge with the cross-axis coordinate interpolated at the crossing, and the in-gap portion not drawn. A stroke with no visible piece is dropped. |
| `segment` | The shaft splits as above; an `arrow=True` head is rebuilt from the endpoints of the piece reaching its tip, keeping its drawn size and pointing along that piece. A tip inside a gap drops the head. |

Split pieces carry their row metadata through 1→N bookkeeping, so `data_indices`, tooltips, hrefs, descriptions and polar spans stay one entry per node; a `key=` sidecar suffixes every piece after the first as `{key}#{i}`, choosing the smallest free `i` so a second axis's pass cannot collide with the first's.

**Axis chrome.** Only nodes a tick value positioned are re-placed, and only on the axis they were built along: a tick dash and its label translate **rigidly on that tick's own anchor**, so a label keeps its drawn baseline offset bit for bit and hides if and only if its own tick's anchor is in a gap. The axis title and the domain line are exempt (their positions come from the axis rect, not from tick values), as is all cross-axis chrome, so a y break moves horizontal gridlines and leaves vertical ones spanning the full plot rect.

**Annotations** (§3.11) are routed by what their geometry means, mirroring the mark side. `rect` and `span` are clipped into one piece per visible segment, silently (a span's label re-anchors on the largest surviving piece, ties keeping the first). `line` and `arrow` split through the same vertex splitter marks use. `text`, `callout` and `image` keep a plain membership test on their coordinate. `bracket` is suppressed whole, permanently and by design: its two endpoint tips say where the bracketed range begins and ends, so a cut would either strand them or invent new ones. An annotation is suppressed only when nothing of it survives, or when its kind admits no honest split; suppression rolls up into one warning per axis per mechanism (a coordinate inside a gap, versus an interval reaching across one). Pixel- and norm-space annotation coordinates are never touched by a break.

**Facets and composites.** A break is built per facet panel, against that panel's own plot rect and scale pixel range, so identical data places identically in every panel and every panel draws its own indicator. Diagnosis is per chart, not per panel: a fact every reporting panel agrees on is reported once, and a divergent one names the panel it was measured on and how many panels share it. Composite leaves are separate scenes with their own break build and their own warnings.

**Multiple specs on one axis** merge into a single break result before any geometry is built: their gap lists union through the same normalization, so N same-axis specs render byte-identically to one spec declaring the union. A conflicting `break_style`/`break_size` among merged specs keeps the first declaration's value and warns once per axis per field.

**Coordinate systems.** `CoordPolar` and `CoordGeo` combined with any `BreakAxis` refuse at the render funnel with a typed `ValueError` (`RenderError::BreakAxisIncompatibleCoord`) naming both the coord and the axis, rather than silently corrupting already-projected geometry. Every other coord renders normally.

**Warnings** (all `RenderWarning`, surfaced as Python warnings): `BreakAxisIneffective` (no gap overlaps the axis domain, or the axis is ordinal — the chart then renders exactly as if the break had not been declared, padding included), `BreakAxisOnNonlinearScale` (a `log`/`pow`/`sqrt`/`symlog` axis: placement uses a linear pixel-to-data map, so the removed range can differ from the gap asked for), `BreakAxisMergeConflict` (same-axis specs disagreeing on `break_style`/`break_size`), and `BreakClipSidecarMisaligned` (a split batch whose per-node metadata length disagreed, so tooltips/links/keys on it may name the wrong marks).

**Disclosed residuals.** A curved line (`mark_line(interpolate="basis"/"monotone"/"cardinal")`) keeps the per-vertex fallback and can still draw a stroke through the hidden band, because cutting a placed curve honestly needs subdividing it. A piece that is simultaneously rotated (`angle != 0`), rounded, and cut keeps its original corner radius, because converting it to a path would relocate its rotation anchor. Non-linear-scale placement is linear (warned, above). Interactive (WASM) output builds its scene through the same break-aware pass, so marks are placed identically, but break-aware interaction (hit-testing, keyed transition pairing across a split) remains a documented non-goal.

> **Dated note (2026-09-11, batch H tasks 3/4 — F-L08-06/F-L08-07, `BreakAxis` (`ferrum.structural.BreakAxis`) clipping, glyph placement, tick behavior, and annotation suppression+remap go from broken/undocumented to matching the guide's contract):** the guide's clipping model (`docs/site/guide/concepts/break-axes.md`) is now real. A mark's placement under a `BreakAxis` gap is decided by what its geometry *encodes*, not by node shape: an extent mark (bar, area, rect, ribbon, and a filled/closed shape like `mark_violin`'s ring) whose data extent intersects a gap is split into one clipped sub-node per visible segment; a glyph mark (point, tick, text, label, image) is translated whole to its anchor's compressed position and hidden whole (never resized, split, or dropped) when its anchor falls inside a gap; a vertex-placed mark (line, rule, segment, arc) places each vertex independently, hidden per-vertex — this is the one shape that still spikes toward the hidden sentinel for *connected* stroked geometry crossing a gap, a documented, unfixed residual (per-segment line splitting is a filed follow-up). A `mark_label(leader_line=True)` connector and a `mark_tick` rug dash are visually the same node kind (`Line`) inside a glyph batch with opposite meanings (a two-point connector vs. a fixed-length dash) and are now disambiguated by the mark's batch kind, not the node shape. In-gap axis ticks and gridlines were already hidden correctly before this batch and remain so; what this batch found and fixed in that chain was two things: the axis title — a `Text` node at the axis midpoint that reverse-mapped into a gap straddling it and teleported off-canvas (all three pre-existing break goldens carried this defect and are now corrected, one element each: the restored title) — and, on the OTHER axis, its tick marks. Which chrome belongs to a broken axis used to be inferred from a node's own pixel coordinate, and a downward x-axis tick starts ON the plot floor — inside the y axis's own pixel span — so on a chart with no scale padding a y-axis break reverse-mapped and clamped every x-axis tick dash to a single pixel (rendered as zero-length lines, the tick marks vanishing entirely while their labels survived). Ownership is now decided by the axis's own dimension instead of a coordinate range, which closes this for both directions. A `SecondaryY` layer's marks are exempt from a break declared on the primary axis, and overlapping/nested/domain-edge-touching gaps normalize (merge, sort, clip to domain) before segments are computed so an indicator reserves pixels only for a gap it actually draws. Separately, an annotation (`annotate_*`/`ann.*`, §3.11) whose data-space coordinate on a broken axis falls inside that axis's gap is now suppressed with a named `RenderWarning` (`annotation '<kind>' at <axis>=<value> falls inside BreakAxis gap (<lo>, <hi>); suppressed`) instead of silently rendering at the unbroken (wrong) position; a surviving data-space annotation is remapped through every same-axis break in declaration order, landing on the same compressed pixel a mark at that value would; pixel-/norm-space annotation coordinates are never touched by a break. An extent-shaped annotation (rect, span, line, arrow, bracket) is suppressed when its interval INTERSECTS a gap, not only when one of its endpoints lies inside one — the spec's "any in-gap data coordinate" includes the interval's interior, so a rect large enough to contain the gap is suppressed like every smaller overlapping one instead of painting straight across the break (a distinct warning names the extent and the gap it crosses). Point-shaped coordinates (text, callout anchors, image) keep the plain membership test. Suppression is TOTAL for extents under this contract: a band that brackets every visible segment (e.g. `annotate_rect` spanning the whole axis as a background shade) is dropped whole rather than clipped into one piece per segment, which is what the equivalent MARK geometry does — unifying the two answers is the filed follow-up, not this batch. An out-of-domain annotation value now clamps to the correct plot edge, and that correction applies to EVERY chart, not only broken ones: the prior sign error put an out-of-domain value on the mirrored side on a break-free chart too (`annotate_hline(y=-50)` on a `[0, 100]` axis rendered above the plot); the break's fold only made it conspicuous by clamping the value back onto the visible canvas. `Inset(connect_to=...)` resolves its own out-of-domain anchor through that same corrected rule. A `BreakAxis` that resolves to zero effective gaps (every gap outside the domain, or an ordinal axis) now emits a warning AND renders exactly as if it had never been declared — the render, not only the diagnosis, matches "had no effect": no indicator, no remap, and (for a continuous axis whose gaps simply miss the domain) no collapse of the scale's own pixel padding either, which an earlier cut of this fix still let through as an up-to-8px shift in every mark, tick, and gridline. A break on a reversed (descending-domain) scale is now effective instead of being silently discarded, and a break on a non-linear scale (`log`, `pow`/`sqrt`, `symlog`) warns that its placement is computed with a linear pixel-to-data map and therefore removes a different data range than requested — unfixed, disclosed, and a filed follow-up. A `mark_segment(arrow=True)` arrowhead under a break keeps both its drawn size and its alignment with the shaft: the head is rebuilt from the shaft's own placed endpoints — the same thing `annotate_arrow` already did on the annotation path — rather than having its three vertices translated or placed independently, which previously sheared a 20px head into a ~90px one, and then (a later fix) kept the head's size but left it pointing along its pre-break direction instead of the shaft's new one. The shaft itself still places per vertex. Disclosed residuals, not fixed by this batch: `Line`-shaped extent marks (`mark_rule`, box/violin whiskers, area borders) still spike across a gap rather than splitting (same root as the vertex-placed-mark residual above); a bar's `corner_radius` is replicated onto each split segment, rounding the internal cut edge; a mark sitting within a few px of a gap edge can resolve to "hidden" due to bbox-anchor imprecision for asymmetric glyphs; a faceted chart applies its `BreakAxis` to panel 0 only; interactive (WASM) rendering under a break remains a documented non-goal.

> **Dated note (2026-09-11, wave-1 hardening, #240-3 — the in-gap annotation warning rolls up per axis, not per suppressed element):** the batch-H note above documented per-instance suppression warnings; a chart with N annotation elements falling in the same axis's break gap now emits exactly ONE rolled-up `RenderWarning` naming the axis and how many annotation elements were dropped (both the point-coordinate-in-gap mechanism and the extent-spans-a-gap mechanism roll up independently, since they report different facts) — previously each suppressed PRIMITIVE warned separately, so a chart with 4 `annotate_hline()` calls (each desugaring to a line + its own label, 8 primitives) emitted 8 near-identical warnings for what a user experiences as one fact. The count is honestly of annotation ELEMENTS, not of the user-facing `annotate_*()` calls that produced them (this layer has no way to attribute a desugared primitive back to its originating call), and the rolled-up kinds list is deduplicated (first-appearance order) in the printed text while the stored count is not — a 4-call/8-primitive repro reads `"8 annotation elements (line, text) on the y axis fell inside a BreakAxis gap and were suppressed"`.
>
> **Dated note (2026-09-11, wave-1 hardening, #118 — mark builders stop returning empty in silence):** a mark builder that discovers it has nothing to draw now names why, through a new `RenderWarning::MarkBuildEmpty { layer, mark, reason }`, at five sites that previously rendered a silent blank panel with no diagnostic: `mark_segment()` bound to x/y only (missing x2 or y2), `mark_image()` with no `url=` encoding and no valid Raster-transform data, `mark_geoshape()` whose batch has no `__geometry__` column at all (a present-but-all-null column stays silent — the pre-existing, still-legitimate per-panel/facet-filter case), `mark_rect()`'s IQR box-body layer when neither its x nor y axis resolved to an ordinal scale (the render-time surface of #183, below), and `mark_image()` under `CoordPolar`/`CoordGeo` (both its url-tile and raster paths place images in Cartesian pixel space with no fallback under either coord; the Cartesian case is unaffected and stays silent). Every OTHER empty return in the mark-builder family is exempted by design, not by omission: a zero-row input batch, a documented per-row null/non-finite/out-of-domain skip, and a facet panel that is legitimately empty after a filter all still render silently — warning on those would make a correct, intended filter result look like a bug. `RenderWarning::GroupCardinalityDegenerate`'s singular form (`"1 category was placed..."`) is also corrected alongside this (previously always plural, even at N=1).
>
> **Dated note (2026-09-11, wave-1 hardening, #183 — sub-pixel ordinal cardinality warns):** an ordinal-domain positional scale that resolves to a sub-pixel bandwidth (too many categories for the available pixel range) now emits `RenderWarning::GroupCardinalityDegenerate { categories, bandwidth_px }`, sibling to the existing `ColorPaletteOverflowed` warning, instead of silently drawing bars/points that overlap into an unreadable smear with no diagnostic. This fires per panel on a faceted chart (matching `ColorPaletteOverflowed`'s existing per-panel convention — N independently-degenerate panels are N independent facts, not one to be deduplicated). The frozen report's own repro (`catplot(x="cont:N", ..., kind="box")` on a Float64 column, which resolves the box-body layer's ordinal-vs-continuous mismatch silently) is closed at its actual render-time site by the #118 `mark_rect()` case above; the underlying Python-side root cause (a composite-mark desugar losing the chart-level explicit axis `type_` across the chart↔layer merge) is a distinct, unfixed follow-up outside this note's scope.

> **Dated note (2026-09-11, batch H tasks 5/6 — F-L08-08, `Inset` (`ferrum.structural.Inset`) bounds go from a docstring/stopgap mismatch to the design's original resolution model):** `Inset.bounds`'s four entries — `(left, top, right, bottom)` — resolve two coordinate kinds, each entry independent of the others: a bare `float` or `NormCoord` (`fm.norm(f)`) is a normalized `[0, 1]` fraction of the parent plot area (unchanged, as designed since the 2026-05-24 inset-panels design); `PixelCoord` (`fm.px(n)`) is now a real absolute pixel offset from the plot area's origin, resolved in Rust against the computed plot rect at layout time (`InsetCoord::Px`, `chart_config.rs`) — previously `fm.px(...)` in an `Inset` bounds tuple raised `TypeError` as a stopgap. Mixed kinds within one bounds tuple are allowed and resolve independently (e.g. `bounds=(px(10), norm(0.0), px(220), norm(1.0))`). The docstring's former claim that a bare float meant a *data-space* coordinate was same-commit authorship drift against the original design (which always specified normalized-or-pixel, never data-space, and reserved data-space coordinates for the annotation layer only) — corrected here, not implemented, since implementing it would have overridden a located, real design decision. `Inset(bounds=...)` now refuses at construction time (`ValueError`) for a normalized entry outside `[0, 1]`, a negative pixel entry, or an empty box (`left >= right` / `top >= bottom`, compared only within like kinds — a mixed-kind pair, e.g. `left=px(50), right=norm(0.1)`, is never compared for emptiness); previously all three constructed silently and rendered off-canvas or as a degenerate zero-extent inset.

> **Dated note (2026-09-12, wave-2 break-axis #226 — a break's segments tile the scale's own padded pixel range, so the axis keeps its padding):** a `BreakAxis`'s piecewise scale is built against the broken scale's own pixel range (`ScaleKind::pixel_range()`) rather than the raw plot rect. The two differ by the positional-scale padding inset (5% of the span, capped at 8px per end), and the remap that re-places every coordinate onto the broken axis always reverse-mapped from the padded range: laying the segments over the rect made that round trip lossy in exactly the padding band, so EVERY effective break silently deleted the axis's padding. Observable consequences, all corrected: a broken axis's ticks, gridlines, marks and break indicator now sit at the same pixels as the unbroken chart's (on the reference chart, y ticks run 24.0…412.961 broken and unbroken alike, where a broken axis previously ran 16.0…420.961); a gap that only just reaches into the domain no longer steps the whole chart by the padding width the moment it crosses the domain boundary (the zero-effective-gaps case already rendered exactly unbroken, so the two sides of that boundary used to disagree by 8px); and a data value past the domain clamps to the domain's own end pixel — the padded edge — instead of the plot rect's edge. The axis DOMAIN LINE is exempt from the remap alongside the axis title, for the same reason: its endpoints are the axis rect's corners rather than tick values, so under a break they clamped to the domain's ends and the line shrank onto the segment span, drawing an axis shorter than the plot it bounds. It now spans the full plot rect exactly as on an unbroken chart, with the break indicators overlaying it. The same rule governs CROSS-AXIS chrome, which is the general form of it: only chrome a tick value on the broken axis positioned is re-placed, so a y break moves the horizontal gridlines and leaves the vertical ones (placed at x tick values, and spanning the plot rect's height from corner to corner) untouched, and vice versa — a broken chart's cross-axis gridlines meet the opposite axis line exactly as an unbroken chart's do. A grid band fill's boundaries ARE tick values on the y axis, so a y break still clips it per visible segment, but its outermost boundary is the plot rect's own edge and stays there (the padding strip at each end of the axis keeps its stripe), and an x break leaves a band's full rect width alone. Behavior-changing for every chart carrying an effective `BreakAxis` (the five break goldens moved and were re-blessed under the PNG-inspection protocol); unbroken charts and ineffective-gap breaks are byte-identical. This is the fix the batch-H note above names as "the scale-pixel-range fix tracked as a follow-up".

> **Dated note (2026-09-12, wave-2 break-axis #226/#231 — a tick label translates rigidly on its own tick's anchor, not on its own drawn baseline coordinate; D2):** `AxisNodes::data_positioned` now carries `AxisTickNode { node, anchor }` — every data-positioned axis-chrome node (a tick dash or a tick label) paired with its own tick's data-mapped pixel position (`TickLayout::position`) on the unbroken axis, tagged at `build_axis` time in `marks/axis.rs`. `BreakRemap::remap_axis_chrome` reads this pairing and calls `remap_axis_tick_node`, which shares the `translate_on_anchor` primitive with `remap_glyph_node`: the whole node translates by the DISPLACEMENT its anchor undergoes under the break, rather than being reverse-mapped from its own drawn coordinate. A tick label's drawn position is `tick.position` plus a baseline offset (`+ font/3`, a `label_flush` edge shift, rotation geometry) — drawn size, not a second data value — so reverse-mapping the label's own coordinate (the pre-fix model) answered "where does this baseline offset belong on the broken scale", compressing each label's own gap from its dash by a different amount than its neighbor's and reading a flush-edge label's offset as exactly zero. Rigid translation on the anchor keeps every label's drawn offset from its own dash bit-for-bit identical to the unbroken chart's, for every visible tick, and a label now hides if and only if its own tick's anchor falls inside a gap — never at a different offset than its siblings, and never independently of whether its own dash/gridline is visible. A tick dash's own two endpoints already equal its anchor, so this is a byte-identical no-op for dashes; only labels change. An axis-chrome node reaching `remap_axis_tick_node` that is neither a `Line` (dash) nor a `Text` (label) — the only two kinds `build_axis` tags with an anchor — fails a `debug_assert!` in a debug build rather than silently falling back to the old per-coordinate remap; in release this check compiles out and the node still translates rigidly via `translate_on_anchor`. Pinned by a new label↔dash offset-pairing assertion (`tests/test_break_axis_geometry.py`) satisfying acceptance criterion 2. Behavior-changing for every chart carrying an effective `BreakAxis` with visible axis labels (part of the same golden re-bless as the #226 pixel-range fix above — the five break goldens' bottom-label baseline moved again, to its final, sibling-matching offset); unbroken charts are byte-identical.

> **Dated note (2026-09-12, wave-2 break-axis #230-4/#230-9 — same-axis `BreakAxis` specs merge into one break result, and `BREAK_HIDDEN` is poison):** two or more `BreakAxis` specs declared on the same axis (`fm.Chart(...) + fm.BreakAxis(axis="y", gap=(150, 400)) + fm.BreakAxis(axis="y", gap=(500, 850))`) previously built one `BreakResult` PER spec, each resolved against the ORIGINAL unbroken domain, and folded marks/surviving annotations through all of them in sequence — numerically wrong (the second break's pixel-to-data reverse map re-interpreted the first break's already-broken pixel as an unbroken-domain data value) and, worse, capable of resurrecting a mark the first break had hidden: the second break's reverse map clamped the first break's `BREAK_HIDDEN` sentinel (`-99999.0`, far outside any real pixel) back into the domain like any other out-of-range pixel, landing it at a domain edge instead of leaving it hidden (measured: a mark at `y=200`, inside a `(150, 400)` gap, rendered at the plot's top pixel under a second `(500, 850)` break declared as a separate spec, overplotting a mark at `y=1000`) — silently, with no warning, even though the annotation-suppression path (which reasons in data space, not pixels) correctly suppressed the same value. Same-axis specs now merge into exactly ONE `BreakResult` before any break is applied: their gaps union (through the same `normalize_gaps` a single spec's own gap list always ran through — overlapping, nested, and unsorted gaps still reconcile the same way), so N same-axis specs render byte-identically to one spec declaring the union of gaps, and the resurrection is structurally unreachable (there is only ever one remap per axis for the mark/annotation fold to run through). A conflicting `break_style` or `break_size` among merged same-axis specs keeps the FIRST declaration's value (matching what "one spec with the union of gaps" would carry) and warns once per axis per conflicting field via the new `RenderWarning::BreakAxisMergeConflict { axis, field }` (`"multiple BreakAxis specs on the {axis} axis disagree on {field}; the first declaration's value was kept"`); merging the gaps themselves is the documented semantics of a same-axis break and never warns on its own. Independently, as belt-and-braces, `BREAK_HIDDEN` is now poison at the mechanism level (`BreakRemap::data_at`): a pixel already parked at the sentinel reverse-maps to `NaN` rather than a clamped domain value, so it can never re-enter view through `data_at`/`coord` regardless of caller — not only the specific double-remap sequence the same-axis merge eliminates. Two annotation-side unit tests (`resolve_x_folds_through_multiple_same_axis_breaks`, `resolve_x_folds_same_axis_breaks_in_declaration_order`) that hand-built a `ScaleContext` with two same-axis `BreakRemap`s to pin the old sequential-composition behavior are retired (see the rationale comment in `annotation.rs`) rather than kept green, since that construction is no longer reachable from production and keeping them passing would pin the replaced defect as if it were still supported. Behavior-changing only for charts declaring 2+ `BreakAxis` specs on the same axis; a chart with at most one `BreakAxis` per axis (the overwhelming majority) is byte-identical, and no golden moves.

> **Dated note (2026-09-12, wave-2 break-axis #230-8 — Polar and Geo refuse a BreakAxis at the render funnel instead of corrupting geometry silently; D3):** `scene_build::build_scene` now checks, before any geometry (break or otherwise) is built, whether `spec.coord` is `Polar` or `Geo` AND `chart_config.structural` carries a `BreakAxis`; if so it returns the new `RenderError::BreakAxisIncompatibleCoord { coord, axis }` (`coord` is `"Polar"`/`"Geo"`, `axis` is the offending `BreakAxis.axis` token), surfacing as a Python `ValueError` naming both. This closes what was previously silent geometry destruction, not suppression, for `Polar`: `mark_arc`'s wedge transform runs during batch build, before any break geometry exists, so the pre-fix break fold reverse-mapped already-projected wedge pixels through the Cartesian scale the projection replaced — measured: every polar wedge destroyed (11 `BREAK_HIDDEN` sentinel occurrences in one repro) — while still painting a break indicator over Cartesian axes that `resolve_panel_axes` suppresses under this coord, with no warning. The false doc claim that Cartesian breaks are "suppressed" under polar coords (a comment on `BreakPlacement::for_kind`'s `Arc` arm) is removed. The gate is extended to `Geo` on a structural argument: `mark_geoshape` places every vertex by bbox-fitting its `projection::forward`-projected lon/lat into the plot rect, independent of the resolved x/y scales, so `BreakRemap`'s reverse-map through the Cartesian scale it replaces has no scale-relative position left to correct — the break contract cannot be honored for `Geo` geometry either. This funnel is the one every standalone render and every composite leaf's render passes through, so a composite leaf combining either coord with a break refuses per-leaf, exactly like any other render error. Behavior-changing only for a chart (or leaf) combining `CoordKind::Polar`/`CoordKind::Geo` with any `BreakAxis`; every other coord/structural pairing, including a `Polar`/`Geo` chart with no `BreakAxis`, is unaffected.

> **Dated note (2026-09-12, wave-2 break-axis #230-6 — `BreakAxis.gap` bounds accept temporal values, coerced through the encoding layer's own UTC contract):** each `BreakAxis.gap` bound now accepts, besides `int`/`float`, `str` (ISO-8601 date/datetime), `datetime.date`, `datetime.datetime`, `numpy.datetime64`, or `pandas.Timestamp` (a `datetime.datetime` subclass). A temporal bound is coerced to epoch-milliseconds UTC at `BreakAxis.__post_init__` (construction) time by reusing `ferrum.annotation.coords.temporal_coord_to_epoch_ms` — the same conversion `ferrum.encoding._scale`'s TimeScale-domain coercion and annotation coordinates already use (SPEC-04) — for `str`/`datetime.date`/`datetime.datetime`/`pd.Timestamp`; a naive (timezone-unaware) bound is treated as UTC, a timezone-aware one is converted to UTC first, matching that function's documented contract. `numpy.datetime64` has no analog in that function and is converted directly via a dedicated arbitrary-precision-integer proof (`_epoch_us_for_datetime64`) that refuses an out-of-range value before any fixed-width numpy cast can silently wrap, uniformly for every datetime64 unit (`Y M W D h m s ms us ns ps fs as`), including a fine unit whose dtype `count` multiplies it back up past microsecond resolution. `BreakAxis` is a frozen dataclass; the coerced float values replace `self.gap` in place via `object.__setattr__` so the Rust wire (which accepts only `[f64; 2]`) and every downstream reader see plain floats, never a raw date string. A gap pairing one numeric and one temporal bound raises `ValueError`; an unparseable string bound raises `ValueError` naming the value; a `pd.NaT`/`np.datetime64('NaT')` bound raises `ValueError` naming it rather than propagating a NaN into the existing finite/ordering checks, which still apply to the coerced values. A temporal gap that lands outside a non-temporal axis's domain is not a construction-time error — it renders as an out-of-domain gap and warns `BreakAxisIneffective` at render time, the same as a numeric out-of-domain gap. This closes the `docs/site/guide/concepts/break-axes.md` temporal example, which previously called this combination unsupported (see the guide's own updated note). Behavior-changing only for a `BreakAxis.gap` bound of a temporal type; existing numeric `gap=` calls are unaffected and construct identically.

> **Dated note (2026-09-12, wave-2 break-axis #230-1 — connected stroked geometry is CUT at a gap instead of spiking through the hidden sentinel):** a vertex-placed mark (`mark_line`, `mark_rule`, `mark_segment`'s shaft, and the box-plot whiskers that lower to a ranged `rule`) crossing a `BreakAxis` gap now emits ONE SUB-NODE PER VISIBLE SEGMENT instead of one node whose in-gap vertices sat at `BREAK_HIDDEN`. This RETRACTS three claims of the batch-H note above, which named the spike a documented unfixed residual: "a vertex-placed mark (line, rule, segment, arc) places each vertex independently, hidden per-vertex — this is the one shape that still spikes toward the hidden sentinel", "The shaft itself still places per vertex", and the disclosed residual "`Line`-shaped extent marks (`mark_rule`, box/violin whiskers, area borders) still spike across a gap rather than splitting". Measured pre-fix symptom: a three-point `mark_line` across a gap emitted `<polyline points="74.714,420.961 345.357,-99999 616,16">`, one stroke detouring far off-canvas and back. The geometry contract: each vertex is its own data position, an edge between two consecutive vertices is straight in data space, and the drawn parts of that edge are the parts of its data interval that fall in the break's visible segments — so one edge may contribute to several pieces when it spans several gaps, and the whole of a segment an edge passes straight through is still drawn. At each crossing the piece ends exactly ON the gap edge, at the point where the edge actually meets it: the cross-axis coordinate is INTERPOLATED there (`x` at the crossing, for a y break), not dropped back to the last visible vertex, which would bend the stroke toward that vertex and draw a slope the data never had. A vertex lying inside a gap therefore contributes no point of its own — the piece before it ends at the near gap edge, the piece after it starts at the far edge. A stroke with no visible piece at all (a rule whose single value is in a gap, a line segment wholly inside one) is DROPPED rather than parked off-canvas, which is the same fate the extent clip has always given a bar wholly inside a gap; correspondingly, a `mark_segment(arrow=True)` arrowhead whose tip is in a gap is dropped too, since no shaft piece reaches it, and an arrowhead whose tip IS visible is rebuilt on the endpoints of the piece that reaches it (keeping the drawn size and now pointing along the piece a reader sees it attached to, rather than along the whole compressed shaft). The rule belongs to the GEOMETRY rather than to the batch it arrives in: an extent-clipped batch carries connected stroked geometry too whenever `mark_area(line=True)` or `mark_area(borders=True)` draws an outline over the fill, and that OPEN path is cut by the same splitter, beside the fill's own clip. A broken area's outline therefore agrees with the band boundaries of the fill beneath it, instead of spiking off-canvas at a border vertex that landed in the gap or drawing one connected stroke straight across the break band when no vertex did. Split pieces go through the same 1→N bookkeeping the extent clip uses, so a batch's `data_indices`, tooltips, hrefs, descriptions, `keys` and polar spans stay one-entry-per-node and a dropped stroke takes its row with it; `keys` are still DUPLICATED across the pieces of one split node (the `{key}#{i}` suffixing is #230-5's own work, tracked separately, and interactive rendering under a break remains a documented non-goal either way). The split boundary is a property of the GEOMETRY, not of the node kind: straight-line commands split, including the `HLineTo`/`VLineTo` shorthand `mark_line(interpolate="step"/"step-before"/"step-after")` emits (whose pieces are re-spelled with explicit `MoveTo`/`LineTo` vertices — a bytes change confined to a step line a gap actually CUTS, which includes one that merely ends or begins inside a gap rather than passing through it, and is exactly where the old bytes carried a sentinel spike; whether a cut happened is reported by the splitter itself rather than inferred from the pieces it returns, since a cut that drops a boundary vertex and inserts a crossing in the same edge leaves the piece and vertex counts unchanged), while a CURVED path (`interpolate="basis"/"monotone"/"cardinal"`, i.e. `QuadTo`/`CubicTo`/`ArcTo`) keeps the per-vertex fallback and its sentinel, because cutting a placed curve honestly needs subdivision of the curve rather than a straight cut through it — unfixed, disclosed, and the one remaining shape that can spike. Axis chrome is untouched by this: a gridline is one value wide along the broken axis, so an in-gap gridline still hides whole at the sentinel as before. Behavior-changing only for charts whose stroked geometry actually crosses an effective gap; unbroken charts, ineffective-gap breaks, and curved lines are byte-identical, and no golden moves (the five break goldens carry point, bar and area geometry only, and a plain `mark_area` emits no border node at all — no golden sets `line=` or `borders=`).

> **Dated note (2026-09-13, wave-2 break-axis #230-2 — a split rounded bar/rect no longer rounds the cut edge a gap produced):** the batch-H note above's disclosed residual — "a bar's `corner_radius` is replicated onto each split segment, rounding the internal cut edge" — is RETRACTED and fixed. A `mark_bar`/`mark_rect`/`annotate_rect` piece produced by `BreakAxis` clipping now rounds only the corners it inherits from the source rect's own boundary; a corner touching an edge the gap itself produced is squared. Measured pre-fix symptom: `mark_bar(corner_radius=8)` split by a `(150, 850)` gap emitted `rx="8"` on both pieces, rounding the flat cut edge exactly like the two real corners. A piece stays a plain `Rect` (byte-identical to every pre-#230-2 render) in three cases: no cut edge touches it at all; every one of its corners is squared (the middle slice of a two-gap break, which has nothing left to round and would otherwise pay for a `Path` outline identical in geometry to `Rect { corner_radius: 0.0 }`); or its style carries a nonzero rotation `angle` — `SceneNode::Rect` rotates about its own centre while `SceneNode::Path` rotates about its first vertex, so converting a rotated piece would silently relocate its rotation anchor relative to its unbroken rounded sibling. **Disclosed residual, not fixed by this note:** a rotated (`angle`-bearing), rounded, cut piece therefore keeps its ORIGINAL uncut corner radius rather than squaring its cut edge — the specific three-way combination of rotation, rounding, and a break gap is the one case this fix does not resolve, to avoid corrupting the rotation anchor instead. Interactive (WASM) hit-testing of a squared split piece remains covered by the general "interactive rendering under a break remains a documented non-goal" line above and is tracked as a separate follow-up, not by this note. Behavior-changing only for a rounded (`corner_radius > 0`) bar/rect/annotation-rect piece a gap actually cuts; unbroken charts, zero-radius marks, and pieces a gap does not touch are byte-identical, and no golden moves (no golden sets `corner_radius` under a break).

> **Dated note (2026-09-13, wave-2 break-axis #230-3 — a glyph's move/hide test now uses its exact data anchor, not a bbox measured off its own drawn geometry):** the batch-H note above's disclosed residual — "a mark sitting within a few px of a gap edge can resolve to 'hidden' due to bbox-anchor imprecision for asymmetric glyphs" — is RETRACTED and fixed. `BreakRemap::remap_glyph_node`'s anchor (the position a `MoveGlyph`/`MoveGlyphAndConnector` batch — point, tick, text, label, image — moves/hide-tests on) is now the row's own scale-mapped `(x, y)` position, threaded from batch build (`MarkBatch::glyph_anchors`, one entry per emitted node, cleared before the scene is packed/serialized since no downstream reader needs it), rather than the midpoint of the node's own drawn extent. The old bbox-midpoint model was exact only for a shape symmetric around its data position (`Circle`/`Square`/`Diamond`/`Cross`/`VLine`/`HLine`, and a single-coordinate node like `Text` before an offset is baked in) — every point-mark break golden uses circles, so none moved. It was wrong for `TriangleUp`/`TriangleDown`, whose bbox midpoint sits a quarter of the triangle's height off the true anchor: measured, a `TriangleUp` hidden at a y position where a `Circle` at the exact same data value still rendered. It was also wrong for `mark_text`'s `dx`/`dy` offsets, which `text.rs` bakes into the node's own `x`/`y` before the break ever sees it — the old model measured and hide-tested the OFFSET position instead of the row's true data position, so a label could hide (or wrongly survive) based on where its offset happened to land rather than where its data point did. Both are fixed the same way: the anchor is recorded once, before any shape-specific drawn geometry (a point's radius, a text label's `dx`/`dy`) is added on top of it, and the glyph translates by that anchor's own displacement under the break — so an asymmetric shape keeps its drawn size and a text label's `dx`/`dy` translates rigidly with its anchor, in both directions (a label can now correctly hide on an in-gap data position even when its offset lands outside the gap, and correctly stay visible when its offset alone would have landed inside it). `remap_glyph_node`'s other callers are unaffected: a `mark_label` leader-line connector (`remap_label_node`'s `Line` branch) already placed its two ends independently and never read this anchor; an arrowhead's rigid fallback (`remap_arrowhead`) is a different mechanism entirely (rebuilt from the shaft's own placed endpoints), untouched by this change. A `MoveGlyph`/`MoveGlyphAndConnector` batch missing this sidecar (unreachable from every real builder) fails loudly under a debug build rather than silently degrading. Behavior-changing only for an asymmetric point shape (`TriangleUp`/`TriangleDown`) or a `dx`/`dy`-offset `mark_text`/`mark_label` node within a few pixels of a gap boundary; every symmetric-shape glyph, every glyph away from a gap boundary, and every existing break golden (all circles) is byte-identical, and no golden moves.

> **Dated note (2026-09-12, wave-2 break-axis #228 — an extent annotation is CLIPPED at a gap like the mark geometry it mirrors; only what cannot be split honestly is dropped):** an annotation with a data-space extent on a broken axis is no longer dropped whole the moment a gap touches it. This SUPERSEDES the batch-H note above's final annotation clause — "Suppression is TOTAL for extents under this contract: a band that brackets every visible segment (e.g. `annotate_rect` spanning the whole axis as a background shade) is dropped whole rather than clipped into one piece per segment, which is what the equivalent MARK geometry does — unifying the two answers is the filed follow-up, not this batch" — and closes that follow-up. The answer is now stratified by what the annotation's geometry MEANS, the same question the mark side has always asked (`BreakPlacement`):
>
> - **`rect` and `span`** (`annotate_rect`, `ann.rect`, `ann.span`) are CLIPPED into one piece per visible segment, through the same machinery a bar's height goes through, and SILENTLY: a clipped annotation rendered, so there is nothing to warn about. `annotate_rect(y1=0, y2=1000)` on a `[0, 1000]` axis broken at `(150, 850)` now paints two pieces, each stopping exactly at its gap edge, where before it vanished with a warning. A `span`'s optional label anchors on the midpoint of the LARGEST surviving piece (ties keep the first, in segment order), because a straddling span's pre-clip midpoint is inside the gap — the one place its label must not be drawn.
> - **`line` and `arrow`** (`annotate_hline`/`annotate_vline`'s rule primitive, `ann.line`, `annotate_arrow`, `ann.arrow`) SPLIT per visible piece through the same vertex splitter `mark_line`/`mark_segment` use (#230-1's mechanism, not a second implementation): each piece ends on the gap edge at the point the stroke actually crosses it, with the cross-axis coordinate interpolated there. An arrow's head rides the piece that reaches its TIP, rebuilt on that piece's own endpoints so it keeps its drawn size and points along the piece a reader sees it attached to; a tip inside a gap leaves no piece to terminate the head on, so the head alone drops — the shaft's own visible pieces still render, truncated at the gap edge they end on, exactly as a `mark_segment(arrow=True)` shaft's pieces do for the same case (#230-1). This is clip-fate geometry, not a silent drop of the annotation: what is missing is the part of the drawing that was never visible, the same way a bar clipped at a gap edge is not "silently" missing the sliver inside the gap.
> - **`bracket`** (`ann.bracket`) stays suppress-whole, and this is the PERMANENT contract rather than a residual: its two endpoint tips are what say where the bracketed range begins and ends, so cutting the baseline at a gap edge would either strand the tips at positions the data never had or invent new tips at the cut. A bracket reaching across a gap asserts a continuity the broken axis cannot carry, so it is dropped with the warning naming the axis — where a `rect` of the same extent clips.
> - **The point-anchored kinds** (`text`, `callout`, `image`) keep the plain membership test, unchanged: the coordinate is either inside a gap or it is not. A callout's background rect is a TEXT BOX — drawn size around an anchor, never data — so it is never clipped even when it happens to straddle the indicator band in pixels.
>
> Suppression survives as the narrower fact it should always have been, with its warnings and their per-axis rollup (#240-3) unchanged in shape: an extent annotation is dropped only when NOTHING OF IT SURVIVES the gaps, or when its kind admits no honest split. The endpoint-first shortcut is retired for the four clip-routed kinds — an extent with an endpoint inside a gap but a non-empty surviving portion is CLIPPED (`annotate_rect(y1=100, y2=400)` under a `(150, 850)` gap renders its visible 100–150 band instead of vanishing) — and the endpoint test now decides only WHICH rollup a dropped annotation lands in, since "a coordinate of yours is in the gap" (`AnnotationSuppressedByBreak`) and "your interval crosses the gap" (`AnnotationSpansBreakGap`) remain different, equally true facts. The knife-edge case where a clip-routed extent still lands in the second bucket is an extent whose bounds are exactly a gap's bounds: both endpoints are visible values, yet no length of it is. One `annotate_*` call that desugars into several primitives (`annotate_hline(label=...)` emits a `line` plus its own label `text` at the same y) has each primitive routed on its own, so a fate can legitimately differ between them — a full-width `line` can clip on an x break while its label, anchored inside that gap, point-suppresses, and the warning then names only the kind actually dropped. Behavior-changing for charts whose extent annotations meet an effective gap (strictly more renders than before, strictly fewer warnings); unbroken charts, ineffective-gap breaks, and annotations clear of every gap are byte-identical, and no golden moves (no golden carries an annotation).

> **Dated note (2026-09-12, wave-2 break-axis #230-5 — a keyed mark split 1→N by a break suffixes its piece keys instead of duplicating the source key; spec §6):** when `BreakAxis` clipping splits one keyed node into N pieces, `ClippedNodes::realign_keys` now suffixes every piece after the first with `{key}#{i}` (`i` starting at 1) instead of copying the pre-clip `keys` sidecar verbatim onto every piece — the prior verbatim-copy behavior made WASM's key-based transition pairing (`ferrum-wasm`'s `transition.rs`, which treats key equality as row identity) treat N split pieces as one logical row, degrading the whole batch's keyed pairing to `IndexZip`. The first piece of a split keeps the original key unsuffixed; `i` is not simply the piece's position within its run: a chart broken on two axes re-suffixes a batch's `keys` sidecar once per axis (`scene_build.rs`'s `break_remaps` loop calls `realign_keys` once per axis pass), so the second pass's input is already-suffixed output from the first, and naively restarting the `#1, #2, ...` counter each pass can mint a suffix an earlier pass already used. `i` is therefore the smallest integer such that `{key}#{i}` is absent from both the sidecar the current pass received and every key already emitted earlier in that same pass — identical to the plain sequential count whenever nothing collides (every currently-golden/pinned single-break chart), and additionally closes a collision shape where a user-supplied key already contains `#` (e.g. `"a#1"`). An unkeyed batch (`keys` is `None`) is untouched. Behavior-changing only for a keyed mark batch a break actually splits 1→N; an unkeyed batch or a batch no gap touches is byte-identical, and no golden moves (no golden sets `key=` under a break). Interactive rendering under a break remains a documented non-goal; this note only fixes what key each piece CARRIES, not WASM grouped-key pairing across a break (a logged follow-up, not built here).

> **Dated note (2026-09-13, wave-2 break-axis #230-7 — a `BreakAxis` applies to EVERY facet panel, and its diagnosis is spoken once for the chart):** the batch-H note above's disclosed residual, "a faceted chart applies its `BreakAxis` to panel 0 only", is RETRACTED and fixed. The whole structural build was gated on the first panel, so a faceted chart broke panel 0 and left panels 1..N on the unbroken scale: measured on a two-panel `.facet(col=…)` bar chart with `gap=(150, 850)`, panel 0's `y=900` bar clipped into two pieces at the compressed positions while panel 1's was drawn full height on the linear scale, under ONE shared y axis whose tick labels described only panel 0's geometry, with no warning that half the chart disagreed with its own axis. Breaks are now built per panel: every panel lays the chart's same data-space gaps over its OWN plot rect and its own scale pixel range (the #226 range fix above, now read per panel), so identical data in two panels places identically, and every panel draws its own break indicator inside its own rect. `Inset`, the other structural spec, stays chart-level and is still built once against the reference panel: an inset's clip-path ids are namespaced per inset within one SVG document, so repeating it per panel would emit colliding ids rather than N insets. One ordering consequence of that split: on a chart declaring BOTH a `BreakAxis` and an `Inset`, break indicators are now emitted before inset bodies rather than interleaved in declaration order, so an inset paints over an indicator it overlaps (axis chrome under embedded content). The DIAGNOSIS cadence is deliberately not per panel. `BreakAxisMergeConflict` is a pure function of the spec, and `BreakAxisIneffective`/`BreakAxisOnNonlinearScale` are functions of the spec and the resolved domain. In the common case — a shared-scale facet, the default — every panel resolves the same domain and reports the same payload, so each distinct fact is reported ONCE per render however many panels observe it: a faceted chart's warning text is identical to the single-panel chart's. The ledger does not derive that agreement from the facet's resolve mode; it compares each reporting panel's payload against the first one it stored, because a per-panel resolved domain can diverge panel to panel for reasons the resolve mode alone does not settle. Two shapes are pinned (Rust and Python) where it does: `Chart.share_scale(x=/y="independent")`, where each panel resolves its own domain by request and a gap can be effective on one panel and miss another's entirely — there the single `BreakAxisIneffective` report names the panel it was measured on and the fraction of panels that share it (`"no gap overlaps the axis domain (10, 20) (facet panel 1); this holds on 1 of 2 panels, and the break applies on the rest"`), rather than going silent on the panels the gap missed or repeating itself once per panel; and a **default-resolve** facet whose panels still resolve different domains for a reason unrelated to `share_scale` — a bar mark's zero-baseline domain resolves per panel even under the default shared resolve, so two panels with different group maxima report different `(0, max)` domains without any `share_scale` call at all (reproduced live: `.facet(col="g")` with no `share_scale`, groups `p` (max `900`) and `q` (max `30`), `gap=(2000, 3000)`, which overlaps neither domain — the report names `facet panel 0` exactly as the independent-scale shape does; the per-panel bar-domain resolution itself is pre-existing and unchanged by this task, verified byte-identical against an unbroken chart on released `ferrum-viz==0.23.0`, so what's new here is the ledger noticing the divergence and the report naming it honestly, not the domain resolution). The panels the gap DID reach still break; a partial report is a qualification, not a suppression. The same qualification also fires when a gap misses EVERY panel but the reporting panels' payloads genuinely differ: the pre-fix gate qualified only on a SHORT reporting count (`panels_reporting < panels`), so a gap missing every one of two panels with different domains flushed the first panel's domain verbatim and unattributed, as if it spoke for the chart — reproduced live with groups `p` (`10, 900`) and `q` (`10, 30`), `share_scale(y="independent")`, and `gap=(2000, 3000)` (which overlaps neither domain): the report now reads `"no gap overlaps the axis domain (0, 900) (measured on facet panel 0); other panels reported this fact with a different value, so this is panel 0's, not every panel's"` rather than the bare, unattributed `"no gap overlaps the axis domain (0, 900)"`. The qualifier states only what the ledger checked — that another panel's report disagreed — and does not name a cause; it does not say the panels are independently scaled, since the divergence is reachable under the default shared resolve too. The fix keys the qualification on payload DIVERGENCE across reporting panels, not merely on a short count, and the two shapes read differently: a short count still says "applies on the rest" (some panels broke), while an all-report divergence never claims a "rest" the break applied on, since every panel missed its own gap. `BreakAxisOnNonlinearScale` deliberately does NOT get this same partial-panel qualification: unlike `BreakAxisIneffective`, it is a claim about the SPEC itself (a log/pow/symlog axis's inverse is not the break's linear reverse map), equally true on every panel the break reaches regardless of how many panels observed it, so naming a subset would tell the reader nothing actionable. One consequence: a free-scale facet whose gap is effective on only some panels can render a flat, unqualified `BreakAxisOnNonlinearScale` warning beside a qualified `BreakAxisIneffective` one in the same warning list — reproduced live with `share_scale(y="independent")`, a log-scaled y, and `gap=(150, 850)`: the nonlinear warning names no panel at all, while the ineffective warning reads `"...no gap overlaps the axis domain (10, 20) (facet panel 1); this holds on 1 of 2 panels..."` right beside it. Annotation suppression is unchanged and cannot diverge per panel by construction: chart-level annotations are built in ONE place, against the reference panel's scales, so an in-gap annotation on a faceted broken chart is suppressed everywhere and reported once. Composite/concat leaves are unaffected (each leaf is its own scene with its own break build and its own warnings, so an hconcat of two leaves whose annotations both drop still warns twice, once per leaf). Behavior-changing only for a FACETED chart carrying an effective `BreakAxis`; single-panel broken charts, unbroken charts, and ineffective-gap breaks are byte-identical, and no golden moves (no golden facets under a break). That byte-identity claim covers SVG output only: break-axis diagnostics now flush after the whole panel loop rather than interleaving at panel 0's structural point, so on a single-panel broken chart they now follow the annotation-suppression and mark-remap warnings that previously preceded them in the emitted warning list — no test or downstream consumer reads warning order (`reason` text is read nowhere but the warning's own `Display`/serde), but the sequence itself is a disclosed behavior change.

> **Dated note (2026-09-20, wave 6A task 10, `#229`(2) — the wire's `break_axis.axis` is a two-token vocabulary, and a third token now refuses instead of silently breaking the x axis):** `BreakAxis.axis` is typed end to end in Rust (one shared axis enum, also carrying `ExtentAlignment`'s axis), so the `structural[]` item's `axis` key accepts exactly `"x"` and `"y"`. A hand-built `chart_config` dict naming any other token is REFUSED with a typed `ValueError` reading, verbatim, `chart config: structural[0] (break_axis).axis: axis must be one of ['x', 'y']; got 'z'` — where it previously deserialized as a plain string, failed every downstream `axis == "y"` test, and rendered as an **x** break, a chart the caller did not ask for, with no diagnostic. The refusal comes from the **array-item value gate** (`binding.rs::validate_break_axis_axis_value`), the same gate the sibling `gaps` refusals come from, so it names the item and its index and mirrors Python's own `validate_choice("BreakAxis.axis", ..., ("x", "y"))` wording. That gate runs on the raw dict BEFORE the JSON bridge, and covers both spellings a unit variant has on the wire — the string (`"axis": "y"`) and the externally-tagged single-key map (`"axis": {"y": null}`), which renders identically to it. Its coverage of the map form is quantified over `json.dumps`, the transform the bridge runs: that encoder coerces exactly five key types into JSON keys (`str`, `float`, `bool`, `None`, `int`) and raises `TypeError` for any other, so the gate mirrors those five spellings — including `true`/`false` for a `bool` key, `null` for `None`, `NaN`/`Infinity`/`-Infinity` for a non-finite `float`, and an `int` at arbitrary precision — and **no out-of-vocabulary token reaches serde through the bridge, in any spelling `json.dumps` can produce**. What serde keeps is the SHAPE errors, which it reports better than the gate could (`invalid type: integer \`7\`, expected string or map`; `invalid value: map, expected map with a single key`), and the unknown-variant backstop for a consumer that reaches `Deserialize` without this gate in front of it. Note the key gate is a third, separate mechanism: it lists `axis` among `break_axis`'s accepted KEYS (`STRUCTURAL_BREAK_AXIS_KEYS`) and refuses only unknown keys, with its own pinned `chart config: unknown key '<k>' in <section>; accepted: ...` text this change does not touch. **Wire-level only, and unreachable from the public Python API:** `ferrum.BreakAxis` is a frozen dataclass whose `__post_init__` has always run `validate_choice("BreakAxis.axis", ..., ("x", "y"))`, and no override-prefix rule targets the `structural` section, so the sole route to the new refusal is a dict passed directly to the private `ferrum._core` render entries. Every rendered byte and every break-family warning and refusal message is unchanged for every well-formed chart, `BreakAxis(axis="y")`'s quoted axis token included.

---

### 3.13 Themes

A `Theme` is an immutable value object. All properties are optional; unset properties fall back to the Ferrum defaults.

```
Theme(
    # Canvas
    background=None, width=None, height=None, padding=None,

    # Typography
    font_family=None, font_size=None, font_weight=None, font_color=None,
    title_font_family=None, title_font_size=None, title_font_weight=None,
    title_color=None, title_anchor=None, title_offset=None,
    label_font_family=None, label_font_size=None, label_color=None,

    # Grid
    grid=True, grid_color=None, grid_dash=None, grid_width=None, grid_opacity=None,

    # Axes
    axis_line=True, axis_line_width=None, axis_line_color=None,
    tick_size=None, tick_width=None, tick_color=None,
    cull_threshold=None,  # int; minimum pixel gap between adjacent tick
                           # labels — see the 2026-09-03 and 2026-09-20 dated
                           # notes below. Default 8 (Rust).
    cull_labels=None,      # bool; may the layout engine DROP tick labels to
                           # reach that gap? Independent of the magnitude —
                           # see the 2026-09-20 dated note below.

    # Marks
    point_size=None, point_opacity=None,
    line_stroke_width=None, bar_corner_radius=None,
    area_opacity=None, opacity=None,

    # Colors
    color_scheme=None, mark_color=None, background_color=None,

    # Legend
    legend_orient=None, legend_direction=None, legend_title_font_size=None,

    # Spacing
    axis_title_padding=None, column_padding=None, row_padding=None,
)
```

> **2026-05-11 (Themes-T1):** Every key listed in the `Theme(...)` block above
> is now plumbed end-to-end. The Python `Theme` class validates unknown
> kwargs at construction time (raises `ValueError`); the Rust `theme_from_dict`
> binding likewise rejects unknown keys. Spec key aliases (e.g. `background`
> ↔ `background_color`) are accepted by both. Fallback chains (`title_color
> → font_color`, `label_color → font_color`, `title_font_family →
> font_family`, `label_font_family → font_family`) are resolved Python-side
> in `Theme.to_theme_inputs_dict()` so the Rust binding sees a fully-populated
> dict. Render-side consumption of the newly-plumbed keys lands in Themes-T2
> through T4; defaults remain at their pre-T1 values in this sub-phase.

> **2026-05-11 (Themes-T2 → T4):** Render-side consumers now read every
> plumbed key. New `ThemeInputs::default()` ships an Observable Plot-flavored
> visual identity: `mark_color="#4C78A8"` (tableau blue, was Okabe orange
> `#E69F00`), faint visible grid (`grid_color="#DDDDDD"` width `0.5`, was
> `#EEEEEE` width `1.0`), left-aligned semibold title (`title_anchor="start"`,
> `title_font_weight="600"`), three-stop text color ramp (body `#222` / label
> `#555` / axis `#888`), `point_size=36` (was `30`), `padding=16`,
> `column_padding=row_padding=12`. `theme.color_scheme` defaults flip from
> `okabe_ito` (§3.6) to `tableau10` — `okabe_ito` remains accessible via
> `Theme(color_scheme="okabe_ito")`. `theme.grid` now actually emits
> gridlines via `axis::draw_grid` (was a no-op before T3). `theme.axis_line:
> bool` suppresses the axis stroke when `False`. Font family is kept as
> `"Inter"` (bundled in `crates/ferrum-core/src/render/embed_font.rs`) rather
> than the design spec's `"DejaVu Sans"` — Inter is deterministic across CI
> hosts, DejaVu Sans would resolve through the system font path. The 8
> built-in themes are rebuilt to use the newly-plumbed keys; each is visibly
> distinct from the others on the same chart (see
> `tests/goldens/theme_gallery/`).

> **2026-06-22 (D-THEME-1, T2.1):** The Python `Theme._KNOWN_KEYS` and color-key
> sets are now *derived* from the Rust key manifest (`ferrum._core.theme_known_keys()`
> / `theme_color_keys()`) so the Python and Rust accepted-key contracts cannot
> drift. The accepted set now includes the per-level grid keys (`major_grid_*`,
> `minor_grid_*`, `minor`) in `Theme(...)` as well (previously Python rejected
> them though Rust accepted). The resolved-dict method is `Theme.to_spec_dict()`
> (the older spec mention of `to_theme_inputs_dict()` is renamed). The
> title/label → body-text fallback chain is now **complete**: in addition to the
> four documented above it resolves `title_font_weight → font_weight` and
> `title_font_size → font_size` (every `title_*`/`label_*` key that has a
> body-text counterpart). There is no public `label_font_weight`/`label_font_size`
> key — the public `font_size` key *is* the label/body font size (it routes to
> the Rust binding's `label_font_size`, the same way `background` is the public
> alias for `background_color`). CSS shorthand hex (`#rgb`/`#rgba`) is expanded
> by the Rust color parser (`from_hex_str`); the prior redundant Python-side
> expansion was removed.

> **Dated note (2026-09-03, batch B task 3 — `cull_threshold` is a real pixel-gap minimum, and culling is reachable):** `cull_threshold` is the minimum pixel gap the layout engine keeps between adjacent tick labels; when the natural label set would pack tighter than that, labels are dropped until the survivors clear the gap. `cull_threshold=0` disables culling entirely. Before this task, culling was largely **unreachable**: a floating-point artifact (`cos(±90°)` evaluates to about `6.12e-17` in `f64`, not exactly `0`) made a vertically-rotated (`-90°`) label's computed footprint collapse to near-zero, so the rotation stage always reported "fits" and skipped straight past the cull/elide stages regardless of `cull_threshold`. The footprint calculation now includes the label's own line-height contribution at any rotation angle, so a `-90°` label gets a real fit test and dense categorical/tick axes now visibly cull. `cull_threshold` folds into **every** layout cascade stage as a minimum-gap budget (not just the final stage), so wrapping and packing degrade uniformly under it (shared word-balanced wrapping, never a one-word-per-line collapse) and elision (mid-label truncation with an ellipsis) only fires where truncating the label can actually free enough width to help — an infeasible truncation leaves the label intact rather than truncating for no gain. After culling removes some labels, the survivors re-select their own presentation (flat / rotated / wrapped / elided) at the new, sparser density — a survivor that no longer needs rotation renders flat rather than keeping the rotation the denser pre-cull set required. At `cull_threshold=0` the wrap/pack path is byte-identical to before this task (the verbatim pre-existing accumulator loop).

> **Dated note (2026-09-20, wave 6A task 9 — `cull_threshold` splits into gap magnitude and cull permission, and it now reaches an independent-x facet axis):** Two changes, one additive and one a bug fix. **(1) The split.** `cull_threshold` used to answer two questions with one number: how much gap to require, and whether the layout engine may drop labels to reach it. `0` meant both "require no gap" and "never drop a label", so "keep an 8 px gap driving the flat / wrap / font-shrink / rotate stages, but never drop a label" was unspellable. A new boolean Theme key, `cull_labels`, now carries the permission on its own. Every pre-existing spelling is unchanged in meaning: `cull_threshold=0` still means gap 0 AND never cull; any positive `cull_threshold` still means that gap AND culling permitted; `fm.Axis(label_overlap="true")` still abandons the cascade entirely (independent-x faceted *output* does move for some of these spellings — see the fix below). What the split adds is the other two corners: `Theme(cull_threshold=30, cull_labels=False)` requires a 30 px gap of every presentation stage (S0–S3; S5's elision fallback is deliberately exempt, see the code comment at its fit test) and then keeps every label rather than dropping one — every label survives regardless of density, but WHICH stage the survivors land in does not: sparse enough and the gap is satisfiable early (flat or a shallow rotation), dense enough and the cascade falls through to its last resort, rotated -90° and possibly overlapping (elision, the nominal step after that, is gated on freeing width a -90° truncation can't actually reclaim, so it essentially never fires). Both arms are pinned by `tests/test_label_gap_policy.py::test_cull_labels_false_stage_depends_on_density`. And `Theme(cull_threshold=0, cull_labels=True)` drops labels only where they genuinely overlap, with no extra gap. The default is unchanged (gap 8, culling permitted) and its magnitude and mode are now two independent constants in Rust, so the still-open question of what the default gap should be can be settled without another change to this key surface. **(2) The fix.** On a facet whose x scale resolves independently (`.facet(col=…).share_scale(x="independent")`), the per-panel x axis was rebuilt with the compile-time default gap instead of the theme's, so `Theme(cull_threshold=…)` was entirely INERT there: a 60-category axis rendered the same label count at `cull_threshold` 0, 8 and 40, while the shared-x and un-faceted paths tracked the setting. That axis now reads the theme like every other axis, and an independent-x facet and a shared-x facet over the same data cull identically. Independent-x faceted output is the only output this task allows to move, and only where the setting actually changes which presentation the cascade lands in — a sparse independent-x facet whose presentation `cull_threshold` cannot change stays byte-identical; everything else in the corpus is byte-identical outright. The y axis was never affected — culling is an x-label concern.

**Built-in themes** (`ferrum.themes`)

| Name | Description |
|---|---|
| `default` | Ferrum defaults: light background, subtle grid, OKabe-Ito |
| `minimal` | No grid lines, no axis lines, generous whitespace |
| `dark` | Dark background (#1a1a2e), light text, neon-adjacent accents |
| `publication` | Print-ready: no background, high-contrast, Tableau10 |
| `economist` | Red accent, left-axis labels, horizontal grid only |
| `fivethirtyeight` | Grey background, red/blue diverging, no axis lines |
| `solarized_light` | Solarized palette |
| `solarized_dark` | Solarized dark palette |

**Theme composition:**

```python
my_theme = fr.themes.minimal.update(
    font_family="Berkeley Mono",
    grid=fr.Grid(major=True, minor=False, color="#f5f5f5"),
    title_font_weight="bold",
)
```

`Theme.update(**kwargs)` returns a new `Theme`; source is unchanged.

**Global default** (process-scoped, not module-level mutable):

```python
ferrum.set_default_theme(my_theme)  # returns a context manager
# or
with ferrum.theme_context(my_theme):
    ...
```

> **2026-05-10 (Phase 8a):** `set_default_theme(theme)` is implemented as a
> contextvars-backed setter that returns a context manager. Per-chart
> `Chart.theme(t)` always overrides this default. CLAUDE.md §"Hard
> constraints" documents this as the single sanctioned exception to
> "no global mutable state."

---

### 3.14 Figure-Level Functions

> **2026-05-10 (Phase 9):** All 8 Group A figure-level functions —
> `displot`, `catplot`, `lmplot`, `residplot`, `pairplot`, `heatmap`,
> `clustermap`, `jointplot` — land in Phase 9 with every parameter
> advertised in this section honored (no `NotImplementedError`,
> no silent warn-fallbacks). Each function returns either a `Chart` or
> a compound view (`JointChart`, `RepeatChart`, `ClusterMapChart`) whose
> `.spec` / `.charts` / `.expand()` is a fully-formed object that
> round-trips through `ChartSpec.from_json`. Group B (model-diagnostic
> figure-level functions: `roc_chart`, `pr_chart`,
> `confusion_matrix_chart`, `calibration_chart`, `gain_chart`,
> `lift_chart`, `residuals_chart`, `importance_chart`, `shap_chart`,
> `learning_curve_chart`, `validation_curve_chart`,
> `cluster_diagnostics`, `decision_boundary_chart`,
> `discrimination_threshold_chart`, `parallel_coordinates_chart`,
> `class_prediction_error_chart`, `pca_scree_chart`, `rank_chart`,
> `alpha_selection_chart`, `intercluster_distance_chart`,
> `cv_scores_chart`) remains scheduled for Phase 10, alongside
> `ModelSource` and the model-diagnostic marks they depend on.

> **2026-05-11 (Phase 10 — Model Diagnostics):** All Group B figure
> functions ship in Phase 10 with every spec parameter implemented.
> Multi-model overlay is supported via three entry points:
> (a) pass a dict positional argument
> (`roc_chart({"a": model_a, "b": model_b}, X, y)`),
> (b) pass `compare=` kwarg
> (`roc_chart(model_a, X, y, compare={"alt": model_b})`), or
> (c) construct a `ComparedModelSource` explicitly via
> `ModelSource.compare(...)` and pass it positionally. In all three
> cases each per-model DataFrame is concatenated with a `model: Utf8`
> column that downstream chart builders route to `color="model"`.
> `random_state` affects compute on figure functions backed by
> ModelSource methods that consume randomness (see §3.1 note); it is a
> no-op on the rest. `parallel_coordinates_chart` and `rank_chart` do
> not need a model — both accept a raw DataFrame / 2D array directly.

> **2026-06-27 (compare= aggregate rendering, #35):** `compare=` now
> renders **small multiples** — one panel per model, composed as a
> `ConcatChart` — for the previously-gated aggregate diagnostics.
> Affected families: the explanation family (`importance_chart`,
> `shap_beeswarm_chart`, `shap_bar_chart`, `shap_waterfall_chart`,
> `shap_chart`, `pdp_chart`); the model-selection family
> (`learning_curve_chart`, `validation_curve_chart`, `cv_scores_chart`,
> `alpha_selection_chart`); the regression aggregate family
> (`cooks_distance_chart`; `residuals_chart` with a multi-panel layout;
> `prediction_error_chart` with `ci=` or `reference_band=`, each band
> computed per-model from that model's residuals only, never pooled);
> and the source-based clustering charts (`pca_scree_chart`,
> `intercluster_distance_chart`, `silhouette_chart`, `manifold_chart`)
> with independent per-panel scales. Charts whose single-model output
> is itself composite (`pdp_chart`, multi-panel `residuals_chart`) nest:
> the outer `ConcatChart` children are per-model composites, each
> carrying the model name as a figure-level title. The two sweep-based
> clustering charts (`cluster_diagnostics`, `elbow_chart`) remain
> excluded — they sweep one clusterer class over *k* on a feature
> matrix and wrap no per-model `ModelSource`; algorithm/method
> comparison is tracked in #43.

> **2026-07-10 (dodge-by-model compare= layout, #42):** for the
> dodge-eligible diagnostics, `compare=` now renders a **single
> shared-axis panel** with marks grouped (dodged) by model — the
> canonical seaborn/yellowbrick comparison plot — instead of the
> 2026-06-27 small-multiples `ConcatChart`. **Landed:** `importance_chart`
> returns one `Chart` whose bars are dodged by model within each feature
> band, with a model legend and error rules / value labels dodged
> alongside their bars; features are ranked once globally across models
> (mean *absolute* importance, descending — mirrors shap_bar's
> `_shap_order_features` magnitude ranking) and the shared top-`top_k` set
> is used for every model; the value-axis domain is computed over the
> combined frame.
> `orient="horizontal"` (default) uses the vertical desugar form plus
> `CoordFlip`; `orient="vertical"` renders the dodged layout directly.
> `shap_bar_chart` (`per_class=False`) likewise returns one dodged `Chart`:
> per-model SHAP values are stacked before `_shap_order_features` ranks the
> pooled values once, globally; the shared top-`max_display` feature set is
> aggregated per model, and the always-horizontal single-model layout uses
> the same vertical-desugar-plus-`CoordFlip` idiom (`mark_shap_bar` gained
> an internal `orient=`/`color_field=` pair mirroring `mark_importance`).
> `shap_bar_chart(per_class=True)` keeps the 2026-06-27 small-multiples
> `ConcatChart` (class is a competing facet dimension). `cv_scores_chart`
> (`kind="box"`/`"strip"`, the default) likewise returns one dodged `Chart`
> (spec D3): `split` (`train`/`test`, filtered by `split=` exactly as
> today) stays the shared categorical axis, and each split band gets one
> box/strip group per model, dodged and colored by model — the seaborn
> `hue` convention, since `split` already has ≤ 2 levels and reads as
> x-categories (facet-by-split would reintroduce panels to solve a problem
> two bands don't have). `kind="strip"` drops the single-model jitter
> under dodge (position adjustments are not composable — same rule
> catplot already records); `desugar_cv_scores` gained a `color_field=`
> parameter (`mark_cv_scores(color_field=...)`) that threads into
> `desugar_boxplot`'s groupby for `"box"` and replaces the jittered point
> layer's color for `"strip"`. `kind="bar"` keeps the 2026-06-27
> small-multiples `ConcatChart` (fold x split x model is three grouping
> dimensions — no coherent single dodge). The remaining explanation /
> model-selection charts keep the 2026-06-27 small-multiples layout.
> Single-model output is unchanged.

> **2026-09-13 (wave 3 T5, #213 — `discrimination_threshold_chart(compare=)`
> joins the small multiples):** `discrimination_threshold_chart` and
> `DiscriminationThresholdVisualizer(compare=)` now render **small
> multiples** — one `ConcatChart` panel per model, shared x/y scales,
> titled with the model name — instead of the single pooled panel they
> produced before. This is the one classification-curve chart that composes
> rather than overlaying by `color="model"` (the 2026-05 classification
> default), because its colour channel is already spent on the four-metric
> legend; with N models overlaid there is no free channel left to tell the
> models apart. **Breaking, and the geometry moves, not just the
> annotation.** The builder's `unpivot` to long form listed neither `model`
> in `index=` nor among the resolved metric columns, so polars dropped the
> column outright: both models' sweeps stacked into one frame with nothing
> distinguishing them, and the line mark drew each metric as a polyline that
> traversed the threshold axis once per model — an interleaved zig-zag, not
> an overlay. Measured on the pinned `binary_logistic` fixture vs. a
> `GaussianNB` rival at `n_thresholds=20`: the old single panel held 160
> rows over only 80 distinct `(threshold, metric)` pairs (every pair
> duplicated); each panel now holds 80 rows, one per pair. The
> F1-optimum annotation and the `threshold_line` rule were computed by a
> single `arg_max` over that pooled frame, so they reported whichever model
> won globally; each panel now computes them from its own model's sweep,
> because `_compose_compare` builds every panel by re-entering the
> single-model builder. Measured: base 0.918 @ t=0.42, rival 0.886 @ t=0.53
> — two distinct labels where there was one. This also settles the render
> half of the `compare=` metric-semantics note (F-L10-07) below:
> `DiscriminationThresholdVisualizer(GaussianNB, compare={"rival": logistic})`
> reprs `best_threshold=0.5263, best_f1=0.8865` (its base model, restricted
> by `_base_model_rows` since batch G) while rendering `max F1 = 0.918 @
> t=0.42` (the rival's). The `"base"` panel now annotates exactly the pair
> the repr prints. The single-model path (no `compare=`) is unchanged, and
> both single-model discrimination-threshold goldens are byte-identical.

> **2026-07-02 (explicit scales survive composite-mark desugar, #45):**
> an explicit `scale=` on a chart-level positional channel (`x`/`y`) now
> propagates onto the layers a composite mark desugars into — previously
> it was silently dropped for every composite mark (box, violin,
> cv_scores, letter-value, …). Rule: a layer channel with no scale
> inherits the chart-level scale; a layer scale without a `domain` (e.g.
> validation_curve's log x) merges in the domain only, keeping its own
> `type`/`range`; a layer scale that already carries a `domain` (e.g.
> SHAP's mark-computed `x_scale_domain`) always wins. Consequence for
> `compare=`: shared-scale resolution now visibly applies to
> composite-mark panels with flat data — e.g. `cv_scores_chart`
> `compare=` panels render one union y-axis (the #45 report case).
> Grid-composite children (multi-panel `residuals_chart` under
> `compare=`) still do not share; that lands with the composite-render
> unification (Phase B spec, 2026-07-02).

Figure-level functions return `Chart` or compound view objects. They handle data reshaping, faceting, axis labeling, and legend placement automatically. All accept `theme=` and `**encode_kwargs` to override defaults.

#### Distribution

```
ferrum.displot(data, *, x=None, y=None, hue=None, col=None, row=None,
               kind="hist",          # "hist" | "kde" | "ecdf" | "rug"
               fill=True, cumulative=False, log_scale=False, stat="count",
               bins="sturges", bandwidth="scott", bw_adjust=1.0,
               multiple="layer",     # "layer" | "stack" | "fill" | "dodge"
               kde=False, rug=False, height=None, aspect=None, theme=None)
```

> **2026-09-10 (batch F, F-L09-01 + F-L09-10):** `kind="kde"`'s
> `cumulative=` reaches the `Kde` transform end to end (grouped fits
> included) -- previously `_displot_build` silently dropped it for the
> primary KDE mark, so `cumulative=True` rendered byte-identically to
> `cumulative=False`. It now produces a real CDF. The `kde=True` overlay on
> `kind="hist"` inherits the same `cumulative` value, so a cumulative
> histogram never carries a mismatched (non-cumulative) PDF overlay.
>
> The `kind="hist", kde=True` overlay is also count-scaled and per-hue,
> closing two adjacent bugs. Under `kind="hist", stat="count"`,
> `multiple != "fill"` (the only combination where the histogram's y-axis
> is actually in counts -- `multiple="fill"` normalizes bars to `[0, 1]`,
> which a count-scaled curve would not match), the overlay's `Kde`
> transform gains an opt-in `count_scale` (`n_group x bin_width` for the
> density, `n_group` for the cumulative CDF) computed from the same
> bin-width routine `Bin` uses on the histogram's own bins, so the curve is
> visible against the bars instead of a flat ~2%-height sliver (measured
> curve/bar height ratio 0.02 before, matching the bars after). The overlay
> also now threads `groupby=hue` with a matching color channel --
> previously a `hue=` histogram drew ONE pooled KDE curve in the theme's
> flat default color over the per-hue bars, silently ignoring the
> grouping. `stat=` is validated against its documented `{"count",
> "density"}` vocabulary for every `kind=`, not just `"hist"`; an
> unsupported value (e.g. the undocumented `"probability"`) now raises
> instead of silently degrading to a count-mode render. `stat="density"`
> overlays and `kind="ecdf"` are byte-identical (scaling is opt-in and
> applies only to the count-axis case above). The overlay also carries an
> explicit `Identity()` position so it can never inherit a `Stack`/
> normalize adjustment from `multiple=` -- fixing a pre-existing crash
> where `rug=True` combined with `hue=` and `multiple="stack"`/`"fill"`
> raised `"Stack: value (y) encoding required"` (the rug tick layer gets
> the same `Identity()` fix, so rug ticks are never dodged/stacked either).

> **2026-09-18 (wave 4, #193 — `bins=` accepts a vocabulary instead of
> swallowing one):** `displot`'s `bins=` was passed to a door that read only
> an integer count, so every other spelling was silently discarded and the
> chart fell back to Sturges. Measured pre-fix: `bins="fd"`,
> `"freedman_diaconis"`, `"auto"`, `"scott"`, `"stone"`, `bins=0.5` and
> `bins=np.int64(9)` all rendered the identical 11-rect Sturges histogram
> with an empty transform payload; `bins=True` rendered a 1-bin histogram
> (because `bool` subclasses `int`); `bins=0` raised a `ValueError` naming
> `Bin` rather than `displot`; and `bins=-3` leaked a raw `OverflowError`
> out of the renderer. The accepted set is now explicit and closed:
>
> | Value | Meaning |
> |---|---|
> | `"sturges"` | Sturges' rule (the default) |
> | `"fd"` / `"freedman_diaconis"` | Freedman–Diaconis rule |
> | `int` ≥ 1 | an exact bin count |
> | positive finite `float` | a bin **width** |
> | `None` | unset; falls through to `bin_count`/`bin_width` |
>
> Everything else refuses with a typed `ValueError` in the voice of the entry
> the caller invoked. Measured live: `"auto"`, `"scott"`, `"sqrt"`, `"rice"`,
> `"doane"`, `"stone"`, and the near-misses `"FD"` and `"fd "` all give
> `displot: unknown bins value 'auto'; expected 'sturges'|'fd'|int|float`
> (no case or whitespace coercion — the vocabulary is closed); `True`/`False`
> give `displot: bins must not be a bool (True); expected
> 'sturges'|'fd'|int|float`, checked *before* the integer arm; `0`, `-3` and
> `10**30` give `displot: bins as an int must be a bin count >= 1 within
> machine range; got 0` — no `OverflowError`, and no fall-through into the
> width arm; `0.0`, `-0.5`, `nan` and `inf` give `displot: bins as a float
> must be a positive finite bin width; got 0`; and a `list`/`tuple`/`dict`/
> arbitrary object gives `displot: bins must be 'sturges'|'fd'|int|float`.
>
> `bins=` is validated for **every** `kind`, matching the `stat=` precedent —
> so `displot(kind="kde", bins="auto")` and `displot(kind="rug",
> bins="auto")` now raise where `bins=` was previously inert on those kinds.
> A valid `bins=` on those kinds is still accepted and still inert.
>
> Two doors that already spoke this vocabulary inherit the hardened parser
> and therefore gain refusals they did not have: `Bin2D(bins_x=)`/`bins_y=`
> and `Kde(count_scale=)` now refuse `bool`, non-positive and out-of-range
> integers, and non-positive/non-finite floats at construction, each naming
> itself (`Bin2D: …`, `stat_kde: …`). Valid values are unaffected.
>
> Serialization is fixed with it: `BinMode::FreedmanDiaconis` previously
> mapped to the same empty wire payload as Sturges, so `to_dict()`/
> `to_json()` reported Sturges for a chart that had drawn Freedman–Diaconis.
> FD now carries an explicitly-named additive `bin_rule` key; Sturges stays
> the wire default (the absence of all three mode keys), so **every
> pre-existing payload keeps its exact bytes**. A payload naming `bin_rule`
> *and* a numeric key refuses rather than dropping one answer.
> `repr(Bin("x", bins="fd"))` gains a trailing `, bins='freedman_diaconis'`;
> every other mode's repr is byte-identical.

> **2026-09-18 (wave 4, #194 — `multiple="stack"` means stacked):**
> `displot` built its own position adjustment from `multiple=` and never
> forwarded the mode into `mark_histogram`, so the underlying `Bin` ran at
> its default and binned **each group over its own data range**. The two
> groups' bars therefore shared no x position and were not even the same
> width — measured on two 300-point normals at means 0 and 3: bar widths
> `31.34 px` vs `34.19 px`, and **16 distinct x starts where a stack needs
> 8**. The stack adjustment had no pairs to pair, and the render was a
> side-by-side chart wearing a stack's name. What a `multiple=` mode means is
> now answered in one place (`marks/_multiple.py`), which decides the
> position adjustment *and* the binning extent together — because bars can
> only stack if every group binned against the same edges. `displot` forwards
> the mode; `"layer"`, `"dodge"`, the no-hue `"stack"` path and the
> `kind="kde"` branch are byte-identical.
>
> **`mark_histogram(position_by=)` is new public API** (additive,
> keyword-only): the column the `multiple=` position adjustment groups by, so
> forwarding the mode does not cost the caller its grouping. It accepts a
> column name (`str`) or `None` and refuses anything else in its own voice —
> `mark_histogram: position_by must be a column name (str) or None, not
> Color; received Color('g').` — appending `Pass the column name itself:
> position_by='g'.` when a channel object was passed. It is documented as a
> column name, not as the figure layer's "str or encoding" spelling, on
> purpose: `position_by` feeds `Stack(by=)`/`Dodge(by=)`, whose contract is
> `Optional[str]`, and unwrapping would silently widen a contract this change
> is only now creating.
>
> Known limit, unchanged: `displot(kind="kde", multiple="stack")` still does
> not forward the mode to `mark_density`.

> **2026-09-18 (wave 4, #200 — `kind="ecdf"` keeps the column it colors
> by):** the ecdf branch is the one `displot` path that builds its own `Bin`,
> and it built one without the group key, so the transform discarded the
> `hue=` column before the encoding referred to it. Per-class curves could
> not render. The branch now hands the `Bin` the group key, on a support the
> groups share, so `displot(kind="ecdf", hue=...)` draws one curve per class.
> The no-hue ecdf path is byte-identical.

> **2026-09-18 (wave 4, #201-B — the rug gets the axis its curve was drawn
> against):** `displot(x=..., kind="rug", kde=True)` is a composition the
> docstring advertised twice and named rug for explicitly, and it **raised**
> (`encoding 'y' expected EncodingSpec, got None`) because the x-only root
> encoding fell through the generic x/y check. #201-D (above) made it render
> with the correct extent and tick labels, but through the binding *fallback*
> — which resolves a scale, not a chart-level binding — so the density axis
> came out **untitled**. `displot` now binds the density column the overlay
> already names at chart level, and the axis is titled `density`: the KDE
> transform's own output column name, which is exactly what
> `displot(kind="kde")` and `displot(kind="kde", rug=True)` already title the
> same axis, and what `kind="ecdf"` does with `count`. The curve's extent and
> tick labels are unchanged from #201-D; the rug's 200 ticks still sit at the
> axis base; `hue=` still yields two curves and split ticks, now titled.
>
> `displot(y="v", kind="rug", kde=True)` used to silently drop the KDE
> overlay (the overlay block is gated on the x variable) and render a bare
> vertical rug. It now refuses — see the close note below.

> **2026-09-18 (wave 4 close, A2/A3/A5 — `displot` answers in its own voice,
> and stops dropping a mark it was asked for):** three related changes to
> what `displot` accepts.
>
> 1. **One refusal for the x-requiring kinds.** `mark_histogram() requires
>    .encode(x=...)` and its `mark_density()` twin no longer escape from this
>    path: one helper answers for `hist`, `kde` and `ecdf` alike, naming the
>    entry the caller invoked, the binding observed, and the fix in the
>    caller's own column name. Measured:
>    `displot(kind='hist', y='v') is not supported -- kind='hist' draws the
>    distribution of the variable x= names and computes the other axis itself
>    (a count, a density, a cumulative frequency), so a y= variable has
>    nowhere to go. Pass x='v', or kind='rug', which draws its ticks against
>    either axis.` `kind="rug"` is deliberately excluded: a tick mark draws
>    against either axis.
> 2. **`kind="rug", y=..., kde=True` refuses rather than dropping the
>    overlay.** Measured: `displot(kind='rug', y='v', kde=True) is not
>    supported -- the kde= overlay is drawn against the variable x= names,
>    and this call binds y= instead, so the overlay has no variable to read.
>    Pass x='v' (kind='rug' draws the same ticks from x= or y=), or drop
>    kde=.` Refusal rather than a warning because the overlay is not a knob
>    that got ignored, it is a whole requested mark with nowhere to stand —
>    there is no partially-honored reading of it, unlike `bins=` below. A
>    previously-accepted call now raises. The end state this defers is a
>    y-oriented density curve, which is an orientation feature, not a guard.
> 3. **`bins=` warns where it is inert.** `kind="kde"` and `kind="rug"`
>    evaluate no binned statistic, so a stated rule cannot act. Measured:
>    `displot(kind='kde', bins=7): bins= is inert for kind='kde' -- it
>    evaluates no binned statistic, so the rule you named cannot act on
>    anything and the render is identical for every value. Drop bins=, or use
>    kind='hist'/'ecdf', which bin.` Only a rule the caller actually *stated*
>    warns: `bins=None` and `bins="sturges"` both mean the default a
>    non-binning kind would have used anyway, so nothing is discarded and
>    there is nothing to report. This is the warn-degraded half of the #193
>    refusal contract above, which validates the vocabulary for every kind.

> **2026-09-18 (wave 4 close, A1 — `multiple="dodge"` bins on shared
> edges):** `"dodge"` now maps to a shared binning extent, joining `"stack"`
> and `"fill"`; `"layer"` is the only mode that may still bin per group,
> because it is the only one that asks nothing of its neighbours. The band a
> dodged bar takes its slot in *is* the bin, so the bins have to be the same
> bins for every group before a slot means anything. Previously each hue
> group binned over its own range: measured on a two-group frame, 9 bars of
> one width against 10 of another, sharing **zero** x positions — two
> histograms wearing a dodge's name. Measured after, on 200 rows in two
> groups at `bins=10`: both groups draw 10 bars of equal width `22.57 px` on
> one grid of step `50.16 px`, offset from each other by exactly one slot
> (`22.57 px`), with no overlapping pairs and the two slots plus padding
> spanning the band. **This is a user-visible rendering change** — a chart
> built on the old geometry will look different, correctly so. (The
> equal-width half of this depends on the R6 span fix noted under
> `ferrum.Layer`; shared edges alone fixed the grid but not the widths.)

#### Categorical

```
ferrum.catplot(data, *, x=None, y=None, hue=None, col=None, row=None,
               kind="strip",         # "strip" | "swarm" | "box" | "violin" |
                                     # "boxen" | "point" | "bar" | "count"
               order=None, hue_order=None, orient=None,
               dodge=False, jitter=True, native_scale=False,
               ci=95, n_boot=1000, seed=None,
               height=None, aspect=None,   # > **2026-05-31:** added; same semantics as displot
               theme=None)
```

> **2026-09-18 (wave 4, #192 — `kind="count"` accepts a channel object like
> its seven siblings):** `count` was the only one of catplot's eight kinds
> that read the **raw** `x=`/`y=` argument again after the shared splitter
> had already resolved it, so `catplot(..., kind="count", x=fm.X("g"))` died
> with a bare PyO3 `TypeError: argument 'field': 'X' object is not an
> instance of 'str'`, naming neither `catplot`, nor `kind="count"`, nor a
> working route — while `x : str or encoding` was already the documented
> contract for the parameter. It now derives the aggregated field from the
> resolved channel, the same idiom the estimating kinds already use. This
> closes a second, latent defect on the same line: an
> `encode={"x": fm.X("other_col")}` override on `kind="count"` now aggregates
> **the column its axis actually binds**, not the column named by the
> original `x=`/`y=` keyword. The bare-string path is byte-identical
> (SHA-256-pinned).

> **2026-09-10 (batch F, F-L09-04):** `hue_order=` is honored for
> `kind="box"` and `kind="violin"` -- previously silently ignored (byte-
> identical output regardless of the argument). catplot already built
> `Color(hue, sort=hue_order)` correctly; the drop point was
> `_ResolveContext._color_field()` flattening the channel to a bare string
> before the box/violin desugars ever saw the sort. The color sort now
> rides the same threading pipeline `x_sort`/`y_sort` already use: a
> `sort=` on a bare `Color(...)` channel -- catplot's `hue_order=`, or a
> caller's own `mark_boxplot()`/`mark_violin()`/`mark_errorbar()`/
> `mark_errorband()` plus `Color(sort=...)` -- now reaches the desugared
> layers' color encoding and the legend order. `kind="boxen"` and
> `kind="swarm"` keep their pre-existing, documented exclusion from this
> mechanism (their own `hue_order=` already worked through chart-level
> color, unaffected by this fix). `sort=None` (or omitted) is
> byte-identical.
>
> **Dated note (2026-09-11, wave-1 hardening, #190 — `catplot(hue=)` honors both a channel object's `sort=` AND `hue_order=`, kwarg wins):** `catplot(hue=fm.Color(...), hue_order=[...])` used to raise `TypeError` on every one of catplot's eight kinds — the `hue_order=` path always handed the raw `hue` argument straight into `Color(hue, sort=..., type_="nominal")` as `Color`'s positional `field`, and `ChannelBase.__init__` rejects a non-`str` `field`. `hue` is now routed through `_catplot_cat_field_parts`, the same channel-agnostic splitter `x`/`order` already use, so a channel-object `hue=` composes with `hue_order=` instead of crashing. When `hue_order=` is given, it OVERRIDES a channel's own `sort=` (checkpoint ruling: kwarg wins, matching the `order=` precedent — a documented divergence from the issue text's own parenthetical, which proposed "explicit channel sort wins"; splitting the builder's precedence rule between its two parallel channels was rejected as worse). When `hue_order=` is NOT given, a channel object's own `sort=` survives untouched, and — closing a sibling defect of the same root cause — `nominal_color_channel`'s previously-silent `sort=` drop on an already-built channel object now honors it instead of discarding it. An explicit `type_=` on the caller's own channel is honored verbatim alongside `hue_order=`, mirroring how `x_sort`/`type_=` already compose. A caller-supplied numeric `hue=fm.Color(<numeric col>)` with NO `hue_order=` keeps its continuous colorbar exactly as before (this composes with #240-4a below only through the shared `resolve_cmap_alias` refusal, not through the hue-typing decision itself).
>
> **Dated note (2026-09-11, wave-1 hardening, #240-4a — unknown cmap/scheme name refuses at desugar, warns at the Rust resolve site):** `resolve_cmap_alias` (the shared `scheme=`/`cmap=` alias resolver used by `mark_raster`, `mark_hex`, `mark_contour`, `heatmap`, `clustermap`, and `RenderConfig`) now validates the resolved name against the registered continuous colormap list and refuses a bad name at Python desugar/construction time, closing the two channel-less Rust mark sites (`mark_raster`, the quantitative-color arm of `mark_hex`) that previously fell back to Viridis with no diagnostic at all, since the channel-bound warning below only fires when a `Color` scale is actually in the loop. The Rust side of the same fix (`resolve_continuous_scheme`) emits `RenderWarning::UnknownColorScheme` naming the bad name and falling back to the theme default, for the one path that IS channel-bound and therefore reachable only through a hand-built wire spec that bypasses the typed Python API — confirmed unreachable from `fm.Color(scheme="bad")` today, which already refuses at Python construction.
>
> **2026-09-10 (batch F, F-L09-02/03):** `kind="point"` and `kind="bar"` are
> ESTIMATING kinds. Each draws one mean per (category × `hue`) group from a
> `Summary` transform, with a bootstrap CI whisker: the whisker paints under
> the point marker and over the bar. Previously both drew one mark per raw row
> (200 overplotted circles for a 200-row frame) and any non-default `ci=`/
> `n_boot=` raised "not yet supported". `ci` is now a percent in the open
> interval `(0, 100)` and `n_boot` must be `>= 1`; both are validated at the
> catplot boundary for these two kinds only (they remain inert on the other
> six). `seed=` seeds the bootstrap as well as `Jitter`, with `None` mapping
> to the fixed default seed, so output is byte-deterministic. Both kinds are
> layered charts now, so `mark=` takes per-layer keys (`"whisker"` plus
> `"point"`/`"bar"`) rather than a flat mark-kwargs dict, and a call missing
> either the category or the value channel raises instead of drawing a
> value-less strip. `ci`/`n_boot` also refuse a `bool` (it subclasses `int`,
> so `n_boot=True` would silently mean one resample); the numeric `1` stays
> valid for both. An explicit positional `scale=` on the value channel (via
> `y=`/`x=`, `encode=`, or `**encode_kwargs`) is honored on these kinds as it
> is on the others **when it carries a `domain`**; a domain-less one (e.g. a
> bare `{"type": "log"}`) now raises instead — main accepted it when these
> kinds were single-mark charts, but on a layered chart the renderer resolves
> an explicit domain-less scale from one column and would silently clip the CI
> whiskers (the same gap drops marks on `kind="box"`/`"boxen"` today, without
> an error). Adding a `domain` is the documented route; the refusal is deleted,
> not relaxed, once the explicit-scale path gains the cross-layer domain union
> the auto-inferred path has (tracked as SCALE-1 in the code-archaeology
> followups doc). The other six kinds are byte-identical.
>
> **2026-09-10 (batch F close):** every option stated on a channel the CALLER
> built reaches all eight kinds. `x=fm.X("cat", sort=[...])` and
> `hue=fm.Color("g", sort=[...])` were honored by the six per-row kinds and
> dropped by `kind="point"`/`"bar"`, whose layers rebuilt both channels from
> bare field names and re-threaded only `type_`, `order=` and the
> native-scale domain -- so the encoding spelling of an order was inert while
> the `order=`/`hue_order=` kwarg spelling worked. Both spellings resolve at
> one site now and produce byte-identical output for the same order; where
> both are given the kwarg wins (unchanged, and the same on every kind). The
> third door is uniform too: `encode={"x": ...}` is a raw override that
> REPLACES the builder's normalized category channel (discarding its typing
> and its native-scale padding, as `encode=` is documented to). It always did
> that on the six per-row kinds; it now does it on the two estimating kinds
> as well, which previously kept their own channel and left the door inert.
> `encode`'s `x` is the layer-space CATEGORY channel under both orientations,
> the same key space the value-scale refusal and the scale propagation
> already read it in.
> **Refused, not dropped:** a PER-OBSERVATION appearance channel on the two
> estimating kinds -- `size`, `shape`, `opacity`, `fill_opacity`,
> `stroke_opacity`, `stroke_width`, `stroke_dash`, `angle`, `color`, `fill`,
> `stroke`, through a bare keyword or the `encode=` dict -- raises naming the
> route. These kinds draw one mark per (category × `hue`) group, so the
> mapping has no row to read: it reached the chart-level encoding only, where
> it drew the channel's LEGEND over marks that carried none (measured:
> `size=` rendered a six-swatch legend beside uniform-radius estimators;
> `opacity=` was byte-identical to passing nothing). `hue=` is deliberately
> NOT in the set -- it joins the `Summary` groupby, so its legend is true.
> The six per-row kinds accept all of these unchanged.
>
> **2026-09-10 (batch F, T5r rider — renderer-wide):** a LAYERED chart's
> auto-inferred value-axis domain now extends to zero when a bar layer binds
> that channel, mirroring the zero-anchor a single-chart `mark_bar` already
> gets from its injected `scale.zero`; it is suppressed when the bar binds an
> explicit span channel (`y2`/`x2`), whose extent is taken literally, matching
> the same suppression on the single-chart rule. This is what keeps
> `catplot(kind="bar")`'s bars on a zero baseline without an explicit
> per-layer scale, which would have replaced the cross-layer domain union and
> clipped the CI whiskers. The rule is not catplot-specific: any layered spec
> with a bar layer on an auto-inferred value axis is affected, and one
> phase-10 golden (`class_prediction_error_multiclass.svg`) moved its baseline
> accordingly. `mark_bar(zero=False)` opts a layer out of the new rule as it
> already opted a single chart out of the injected `scale.zero`: the opt-out
> now travels with the mark as `mark_style.zero` when a chart is composed into
> a layer, so it survives on the merged single-panel path
> (`resolve={"y": "independent"}`, `SecondaryY`, and every `.interactive()`
> `LayerChart`), where the chart-level injection never ran. On
> `catplot(kind="bar")` itself -- whose bar layer is built directly as a
> `_Layer`, not composed from a standalone `mark_bar()` chart -- the identical
> opt-out is reachable through the ordinary per-layer `mark=` override:
> `catplot(kind="bar", mark={"bar": {"zero": False}})` merges `zero: False`
> into that layer's `mark_kwargs` (`_apply_mark_overrides`), which
> `_build_layers_list` serializes wholesale as the layer's wire `mark_style` --
> no special-case wiring needed, live and honored.

> **2026-09-10 (batch F, F-L09-03):** the categorical channel is now ALWAYS
> explicitly typed, for every one of the 8 `kind=` values, instead of never
> being typed at all. `native_scale=False` (the default) types it
> `"nominal"` -- **disclosed behavior change:** a numeric category column
> (e.g. an integer year) previously inferred `"quantitative"` and rendered
> on a linear axis, silently contradicting this documented default; it now
> renders as equal-spaced categorical bands, matching a string column and
> matching seaborn. `native_scale=True` types it `"quantitative"` and
> places category values at their real numeric positions; it raises,
> naming the column and its dtype, when the column is not numeric. The
> former `native_scale=True` refusal ("the renderer does not support
> quantitative categorical axes") is deleted -- it was factually wrong; the
> renderer's continuous-band `Dodge` and non-ordinal `Jitter` paths already
> existed and needed no Rust change. `kind="strip"`'s `Jitter` width on a
> native scale is rescaled Python-side from the ordinal 0.4 band-fraction
> to `0.4 x` the category column's own median unique-value spacing (data
> units), mirroring the bandwidth proxy `Dodge`'s continuous-band path
> already computes; a single-category column falls back to the literal 0.4.
> One exclusion, orthogonal to this change: `kind="boxen"`'s letter-value
> transform hard-requires a Utf8 groupby column regardless of
> `native_scale` (a pre-existing Rust-transform gap, not introduced or
> closed here).
>
> **2026-09-10 addendum (cycle-2 close-out, same finding F-L09-03):** the
> category channel's typing now covers the documented `x : str or encoding`
> shape uniformly -- an already-built channel object (`x=fm.X("year")`) is
> typed and `order=`-sorted exactly like a bare field name, honoring an
> explicit `type_=` the caller stated on their own channel verbatim when
> present. A quantitative axis's domain is otherwise just the raw data
> extent, with no room for `Jitter`/`Dodge` to move a mark into; when
> `kind="strip"` jitters (default `jitter=True`) or any kind dodges
> (`dodge=True` + `hue=`), the category channel's domain is padded by half
> that position adjustment's own maximum offset (plus a 5% safety margin)
> so no mark renders outside it and is silently dropped by the renderer,
> which resolves no awareness of the pending offset on its own. This
> padding never overrides a domain the caller set explicitly (`x=fm.X(...,
> scale=...)` or `encode={"x": ...}`) -- an explicit caller domain still
> governs, and a mark that clips against IT is the caller's own choice.

#### Relational

```
ferrum.relplot(data, *, x, y, hue=None, size=None, style=None,
               col=None, row=None,
               kind="scatter",        # "scatter" | "line"
               height=None, aspect=None, theme=None)
```

`size` maps to the `Size` channel (point area or line width). `style`
maps to `Shape` for `kind="scatter"` or `StrokeDash` for `kind="line"`.

#### Regression

```
ferrum.lmplot(data, *, x, y, hue=None, col=None, row=None,
              method="lm",           # "lm" | "loess" | "quadratic" | "cubic" | "log" | "sqrt" | "gam" | "logistic" | "glm" | "robust" (smooth-family names derive live from the registry, 2026-09-05 batch D; "glm" maps to the dedicated Glm transform; mark_smooth's alias spellings are not accepted here)
              ci=95, order=None, scatter=True, scatter_kws=None, line_kws=None,   # order=None = the method's own default degree; an explicit order forwards into the method's degree-acceptance matrix (2026-09-05, batch D; previously order=1 and silently absorbed)
              truncate=True, x_bins=None, x_estimator=None, x_jitter=None,
              logx=False, show_metrics=True, theme=None)
# 2026-09-05 (batch D): this block previously mis-documented truncate=False (the
# shipped default has been True since the function landed) and omitted
# show_metrics= — both corrected above to the live signature.

ferrum.regplot(data, *, x, y, hue=None,
               method="lm",          # same as lmplot
               ci=95, order=None, scatter=True, scatter_kws=None, line_kws=None,   # order=None = the method's own default degree; an explicit order forwards into the method's degree-acceptance matrix (2026-09-05, batch D; previously order=1 and silently absorbed)
               truncate=True, x_bins=None, x_estimator=None, x_jitter=None,
               logx=False, show_metrics=True, theme=None)
# 2026-09-10 (batch F, F-L10-09): logx, show_metrics, x_bins, x_estimator are
# now real named parameters (identical defaults to lmplot's), corrected above
# to the live signature -- previously reachable only through **encode_kwargs,
# invisible to introspection. col=/row= raise TypeError naming lmplot as the
# faceting route (previously silently faceted) -- see the prose note below.

ferrum.residplot(data, *, x, y, lowess=False, order=1, robust=False,
                 dropna=True, show_metrics=True, zero_line=True,
                 label=None, color=None, theme=None)
# 2026-09-10 (batch F): the block above adds show_metrics= and zero_line=,
# which the signature has always accepted but this doc omitted.
```

`regplot` is the axes-level equivalent of `lmplot` — identical API minus
`col=` and `row=`: passing either now raises `TypeError` naming `lmplot` as
the faceting route (2026-09-10, batch F, F-L10-09 -- previously they
silently faceted through `**encode_kwargs`). This exclusion is specific to
`col=`/`row=`; it does not extend to the grammar-level `facet=` encoding
channel, which `regplot` accepts the same as any other `Chart`. Every other
parameter is forwarded to `lmplot` unchanged.

> **2026-09-10 (batch F, F-L10-04):** `residplot`'s `color=` works, both
> documented halves. A value that names a column of `data` groups the
> residuals -- `Smooth`/`Robust` refit per group (via the transform's
> `groupby`, the same mechanism `label=` already used) and the points/
> lowess overlay color by that column; previously EVERY `color=` value
> raised `ValueError: unknown column`, whether it named a column or not.
> A value that is not a column is parsed as a literal color (CSS name,
> hex, `rgb()`/`rgba()`, or an RGB tuple) and painted onto the points via
> the mark's own `fill=` -- never through `encode(color=)` -- composing
> with `label=` (a labeled legend swatch in that literal color, via a
> pinned single-range `OrdinalScale`). A value that is neither raises,
> naming both interpretations. Column-membership against the INPUT frame
> is checked first, so a column literally named `"red"` groups rather than
> paints, and a column name that collides with a transform OUTPUT column
> (e.g. `"x"`) still resolves deterministically by input-frame membership.
> `Robust` gained `groupby` in Rust to make this possible (see the
> `Robust` dated note in §3.4) -- as a side effect, `robust=True,
> label=...` now shows the label (previously a silent no-op, since
> `Robust` had no `groupby` to carry the injected `_label` column into its
> output). `lowess=True` with a grouping `color=` draws one curve per
> group instead of one curve through the pooled cloud. `show_metrics=True`'s
> corner annotation is suppressed only when `color=` actually groups
> (mirrors `lmplot`'s `hue`-gated `metrics_applied`); a `label=`-only
> grouping (one level by construction) keeps its corner. A user column
> literally named `_label` plus `label=` now refuses with a typed
> collision error instead of the label silently overwriting it.
> `color=None` is byte-identical to before.

> **2026-09-10 (batch F close, F-L10-02):** `lmplot`/`regplot`'s scatter
> layer keeps the OBSERVATIONS. With `ci=None` and `show_metrics=False` the
> fit chart collapsed to a single line mark bound to a chain-advancing
> (unnamed) `Smooth`, and composing it with the scatter applied that
> transform to BOTH layers: every rendered "observation" landed exactly on a
> fit vertex (measured: 200 of 200, zero spread about the curve) for `lm`,
> `loess`, `quadratic`, `gam`, `log` and `sqrt`, in both functions -- or
> raised `unknown column` when the frame's columns were not themselves named
> `x`/`y`. The fit now binds to a NAMED transform through `data_source=`
> whenever a scatter layer is present, the same shape the `ci=` path and the
> `logistic`/`glm`/`robust` methods already used; with `scatter=False` the
> single-mark shape is kept, since there is no sibling layer to corrupt.
> (That shape's chart-level encoding titles the axes with the caller's own
> column names -- but it is reached only at `ci=None, show_metrics=False`
> on a `mark_smooth`-routed method, i.e. 7 of the 40 `scatter=False` cells,
> so it was never what titled the other 33; see the 2026-09-17 note below.)
> **One layer
> renames:** on that combination the fit layer is
> `"line"` rather than `"fit"` -- which every other `lmplot` shape already
> called it -- so a per-layer `mark={"fit": ...}` override for it becomes
> `mark={"line": ...}`. Separately, `x_jitter=` on a non-Float64 predictor
> now renders: `Jitter` writes a Float64 offset back into the column it
> adjusts, so an Int64 predictor failed Arrow's schema check on three of the
> four `ci`/`show_metrics` combinations and was masked on the fourth by the
> corruption above; the predictor is widened before either chart is built,
> and the cast is deleted once `apply_jitter` coerces the column it writes.

> **2026-09-17 (wave 4, #199):** every fit path titles its axes with the
> caller's own columns, and honors `line_kws` on every method. The fit
> layers (ribbon / line / metrics) are bound to the transform's output
> columns, and the renderer resolves an axis title from layer 0 only
> (spec-level channel `title`, then layer-0 channel `title`, then layer-0
> field name), so whenever no scatter happened to sit at layer 0 the value
> axis was titled `ci_lower` -- the confidence band's LOWER BOUND -- and the
> predictor axis `x`. Measured over the full 40-cell `method × ci ×
> show_metrics` matrix under `scatter=False` (all ten `lmplot` methods ×
> `ci ∈ {95, None}` × `show_metrics ∈ {True, False}`), only 7 cells were
> right — the seven `mark_smooth`-routed methods at `ci=None,
> show_metrics=False`, where the single-line shortcut keeps the chart-level
> encoding; the other 33 were wrong. All 40 are correct now.
> The fit-layer builder now carries the caller's field names as layer
> titles, which fixes `lmplot(scatter=False)`, `regplot(scatter=False)` and
> a standalone `Chart(df).mark_smooth(...)` from one site; an explicit
> channel `title=` still outranks it. `residplot` gets the same treatment on
> its DEFAULT path, where no flag was needed to reach the defect: its axes
> were titled `x` and `residual`, and are now the caller's x column and
> `"<y> residual"` (the one golden this moves is
> `tests/test_phase_9_e2e/goldens/residplot_lowess.svg`, whose y-axis title
> changes and nothing else). Separately, `line_kws` was silently discarded
> for the seven `mark_smooth`-routed methods (`lm`, `loess`, `quadratic`,
> `cubic`, `log`, `sqrt`, `gam`) under `scatter=False` — 28 of the 40 cells;
> the three dedicated methods (`logistic`, `glm`, `robust`) already forwarded
> it at construction and were never affected. It was
> applied by rewriting the finished chart's layers, which on that branch are
> still unmaterialized (the chart carries a pending stat mark until render).
> Every branch now forwards it at construction, to the fit line alone, and
> the same capability is exposed on the grammar as
> `layer_kwargs={"<layer>": {...}}` — a composite mark's own `**mark_kwargs`
> reach every emitted layer, so a `stroke=` passed that way repaints the CI
> band too. **Precedence between the two is by specificity:** when both name
> the same style key, the one aimed at the named layer wins, and the
> catch-all still applies wherever nothing narrower was said — measured,
> `mark_smooth(ci=0.95, stroke="#0000ff", layer_kwargs={"line": {"stroke":
> "#ff0000"}})` draws the fit line `#ff0000` while the CI band keeps its
> `#0000ff`-derived fill, on the layered shape and the single-line shortcut
> alike. Charts with a scatter present and the single-line shortcut render
> byte-identically. (This parameter was first written as a
> `mark_smooth`-only `line_kwargs=` earlier in the same unreleased wave and
> reshaped before release into the general door; see the 2026-09-18 note
> under `ferrum.Layer` for the shipped contract. `line_kwargs=` is not
> accepted.)

> **2026-09-18 (wave 4 close, B3/B6 — the fit paths accept a channel object
> on every method, and refuse a missing fit column by name):** `lmplot`'s
> `x=`/`y=` are documented as "str or encoding", and seven of the ten methods
> already honored the encoding spelling — the `mark_smooth`-routed ones,
> because `.encode()` takes a channel object. The three that construct a
> transform directly did not, and neither did `residplot`, and all four
> leaked a **binding-layer** error out of a figure-level call: measured
> pre-fix, `lmplot(x=fm.X("bill"), method="robust")` raised the raw PyO3
> `TypeError: argument 'x': 'X' object is not an instance of 'str'`,
> `residplot(x=fm.X("bill"))` raised polars' `TypeError: cannot turn
> 'ferrum.encoding.positional.X' into selector`, and any method with
> `logx=True` raised `X: field must be str, ...`.
>
> **Resolved by accepting, not refusing** — refusing would have regressed
> seven working paths and contradicted the family's own documented contract.
> Each consumer wants the column *name*, which is now narrowed once at the
> figure's own entry; the channel itself still reaches `.encode()`, so the
> caller's `title=`/`scale=` survive. Measured:
> `lmplot(x=fm.X("bill", title="Total bill (USD)"), method="robust")` renders
> that title, not the bare column name, and `residplot(x=fm.X("bill"))`
> renders. **`residplot` now accepts an encoding object where it previously
> died** — a widened public surface at a second entry, deliberately, to match
> the family.
>
> Two new typed refusals come with the narrowing, replacing untyped errors:
> `lmplot(x=None): x names the column to fit and has no default. Pass a
> column name, or an encoding object wrapping one.` (and its
> `residplot(y=None)` twin, which names `residplot`), and the contradiction
> `lmplot(logx=True, x=<channel with its own scale=>): two instructions for
> one x scale. Keep your scale= and drop logx=True, or pass x as a column
> name and let logx set the log scale.` The `None` case also closed a latent
> instance of this same note's title defect: `residplot` built its titles
> from the **raw** parameter, so `y=None` would have rendered the axis title
> `"None residual"` and a channel object would have rendered an object repr —
> unreachable only because polars raised first.

#### Matrix / Pairwise

```
ferrum.pairplot(data, *, vars=None, x_vars=None, y_vars=None,
                hue=None, kind="scatter",     # off-diagonal
                diag_kind="auto",             # "hist" | "kde" | None
                markers=None, height=None, aspect=None,
                corner=False, dropna=False, theme=None)

ferrum.heatmap(data, *, annot=True, fmt=".2f", cmap="blues",
               linewidths=0.5, linecolor="white",
               vmin=None, vmax=None, center=None, robust=False,
               square=False, mask=None, theme=None)

ferrum.clustermap(data, *, method="ward", metric="euclidean",
                  cmap="viridis", z_score=None, standard_scale=None,
                  figsize=None, dendrogram_ratio=0.2, theme=None)
```

> **2026-09-02 (batch A — heatmap domain/border honesty):** `vmin=`/
> `vmax=`/`center=`/`robust=` are honored end to end via a Diverging (when
> `center=` is set) or Linear color scale's explicit `domain=`; previously
> these fell through to an unscaled render. `vmin`/`vmax`/`center` must be
> finite — `inf`/`-inf`/`NaN` raise `ValueError` immediately, before any
> chart construction. When only one of `vmin=`/`vmax=` is given, the other
> endpoint fills from the pre-mask finite data extent (matching `robust=`'s
> existing convention); if there is no finite data to fill from, the fill is
> dropped with a `UserWarning` naming the reason rather than silently
> discarding the user's explicit bound. A one-sided fill that lands on the
> wrong side of the given bound, or an explicit `vmin > vmax`, still renders
> — deterministically as a single flat color, not a reversed ramp — with a
> `UserWarning` naming both endpoints. A `center=` outside the effective
> `[vmin, vmax]` renders as a one-sided compressed ramp with a
> `UserWarning`, never silently and never as an error. `linewidths=`/
> `linecolor=` are honored (default `linewidths=0.5`, `linecolor="white"`)
> — every heatmap now ships default white 0.5px cell borders;
> `linewidths=0` disables the border.

> **2026-09-10 (batch F, F-L09-08):** `clustermap`'s dendrogram panels no
> longer silently blank at practical ratios. The row/column dendrogram
> leaves reserved axis chrome -- a title gutter and (row side) a label
> band -- for axes their own `axis(show=False)` never draws;
> `_clustermap_build` now blanks the dendrogram leaves' `_X`/`_Y` titles
> the same way the center heatmap already blanks its own
> (`matrix.py:1157-1160`). At `dendrogram_ratio=0.1`, both dendrograms now
> render instead of collapsing to an empty panel; at the default
> `dendrogram_ratio=0.2`, a short center heatmap (e.g.
> `figsize=(480, 200)`) no longer silently blanks the column dendrogram.
> Sub-floor ratios still raise `PaddingExceedsViewport` as before -- the
> fix removes dead chrome, it does not relax the hard floor. **All
> clustermap output changes geometry** as a result (more of each
> dendrogram panel is now real branch area rather than reserved-but-unused
> chrome); the `clustermap_basic` and `clustermap_row_col_dendrograms`
> goldens were regenerated and PNG-inspected. The same fix closed two
> adjacent layered-chart wire bugs found alongside it (general, not
> clustermap-specific): `SpecBuildMixin._build_layers_list` was dropping a
> layer's `title=""` via a truthiness check instead of a presence check
> (so `title=None` on a layered channel now also reaches the wire and
> suppresses the axis title, where `title=""` already did), `format=` on
> any layered channel no longer raises an unknown-field error
> (`format_type` is what Rust actually calls the key), and a layer's
> `condition=` now reaches the wire instead of silently vanishing.

#### Joint Distribution

```
ferrum.jointplot(data, *, x, y, hue=None,
                 kind="scatter",       # "scatter" | "kde" | "hist" | "hex" | "reg"
                 marginal_kind="hist", # "hist" | "kde" | "rug" | "box"
                 ratio=5, space=0.05,
                 xlim=None, ylim=None,
                 joint_kws=None, marginal_kws=None,
                 height=None, theme=None)
```

Returns a `JointChart`. `kind` controls the center panel mark; `marginal_kind` controls both marginal panels. `joint_kws` and `marginal_kws` are passed as keyword arguments to the center and marginal chart constructors respectively.

#### Model Diagnostics (figure-level)

```
ferrum.roc_chart(model_or_source=None, X=None, y=None, *,
                 y_true=None, y_pred=None,   # precomputed path (1-D binary or 2-D multiclass scores)
                 per_class=True,
                 average="macro", annotate_auc=True,
                 compare=None,     # dict[str, estimator] for multi-model overlay
                 theme=None)

ferrum.pr_chart(model_or_source=None, X=None, y=None, *,
                y_true=None, y_pred=None,    # precomputed path (1-D binary or 2-D multiclass scores)
                per_class=True,
                annotate_ap=True, iso_lines=True, compare=None, theme=None)
# **2026-09-11 (wave-1 hardening, #220):** `iso_lines=True` on a multiclass
# precomputed source no longer emits polars' internal "Comparisons with
# None always result in null" `UserWarning` — the two group-enumeration
# loops behind iso-line label placement (`_metric_labels.py`'s
# `_apply_metric_label`, `_direct_label.py`'s `_direct_label_endpoint`)
# now filter out a null class label before sorting, mirroring the same
# file's already-correct `_curve_metrics_by_group` idiom, instead of
# comparing `None` against every class value during the sort.
#
# **2026-09-18 (wave 4, #121):** the `iso_lines=True` reference curves now
# render in the grey they always declared. The iso-line layer bound its
# grouping key `_iso_f` to the `color` channel, but the chart's color scale
# is keyed on `class`, so every `"F=0.2".."F=0.8"` lookup missed and the
# line mark's stroke fallback painted `paint.fill` — theme blue at the
# layer's `opacity=0.6` — instead of the literal `stroke="#AAAAAA"` the
# layer had set all along. The key now travels on `detail` (the channel
# that groups without painting), mirroring `desugar_parallel_coordinates`.
# The stale `_iso_f` entry in the color-binding allowlist, whose rationale
# asserted the curves were already grey, is deleted with it.

ferrum.confusion_matrix_chart(model_or_source=None, X=None, y=None, *,
                               y_true=None, y_pred=None,   # precomputed path (1-D hard labels)
                               normalize="true", annotate=True, theme=None)
# **2026-09-10 (batch G doc pass, adjacent transcription fix, pre-existing
# drift, not a batch G behavior change):** corrected to match the live
# signature — the prior listing showed a `cmap="blues"` parameter that does
# not exist on this function and omitted the real `annotate=True` kwarg
# (per-cell text annotations, already documented separately in the
# Schwabish SB3 note below as "docstring already specified `annotate=True`
# default").

ferrum.classification_report_chart(model=None, X=None, y=None, *,
                                    y_true=None, y_pred=None,   # precomputed path (1-D hard labels)
                                    random_state=None, theme=None)
# **2026-09-10 (batch G, F-L11-05 + F-L11-09):** joins the precomputed
# family — eleven `y_true=`/`y_pred=` figure functions now, counting
# `prediction_error_chart` (see the dated note below: the prior "nine"
# framing already undercounted by one before this batch, since
# `prediction_error_chart` documents and renders on this path too and was
# never enumerated in the list). Renders
# a rect-plus-text heatmap: rows are class labels, columns are
# precision/recall/f1-score, color encodes the value, and each cell is
# annotated with its value to two decimal places — no `support` row (a
# deliberate, disclosed non-parity with sklearn's own
# `classification_report`, which does report support counts; adding one
# here would require inventing a count from confusion data that would not
# reconcile with sklearn's on the model path, defeating the parity the
# precomputed and model paths otherwise guarantee). The chart carries a
# bare title, `"Classification Report"` — the family's title-only shape (no
# `subtitle=` parameter), matching `confusion_matrix_chart`. On the
# precomputed path, `y_true=`/`y_pred=` (hard labels) resolve metrics from
# confusion counts with **no sklearn import**; class labels are the
# distinct values of `y_true`/`y_pred` in sklearn `unique_labels` order
# (their own ascending order, raw values — not stringified first), the
# same order the model path's sklearn-backed report produces. On the model path, `X`/`y`
# are both required — an X-less or y-less `ModelSource` (or a bare
# `classification_report_chart(model)` call) raises a typed `ValueError`
# naming the diagnostic and the fix, replacing what used to be a leaked
# sklearn shape error (`"Expected 2D array, got scalar array instead"`) or
# a foreign `InvalidParameterError` naming `y_true`. Both paths order the
# class axis identically (#224a, 2026-09-13): the precomputed path's
# `sorted(..., key=str)` rule — since retired for `confusion_matrix` too
# (close remediation, 2026-09-17; see the dated note) — retired for this
# figure, so the documented divergence between the two paths no longer
# exists. `y_true` and `y_pred`
# must each be 1-D (a 2-D `y_true` now refuses by name, mirroring the
# `y_pred` shape check #208 already had). The label derivation
# (`_curve_frames.report_labels`) is total by partition over the *values*
# the two (1-D) arrays hold: a null in either array, a mix of ordering
# families, and a homogeneous family that cannot serve as a class label —
# either because it does not order (`complex`, `dict`) or because it
# orders but cannot be hashed (`list`, `set`, `bytearray`) — each refuse
# with a typed `ValueError` naming which array and what it holds. A
# homogeneous, orderable, hashable family is the answer, deduplicated by
# hash equality (not adjacency, so a partially-ordered-but-hashable family
# such as `frozenset` cannot return a duplicate label). Nothing else can
# reach the class axis. On the model path (#206 cycle 5, 2026-09-13): when
# `model.classes_` is present, it is coerced into `y`'s dtype and checked
# against `y`, and the call refuses if the two share no value at all (a
# model whose `classes_` describe different labels than `y`, not merely a
# different dtype) — see the dated #206 cycle-5 note above. A partial
# overlap still renders, and so does a model that simply predicts none of
# the classes `y` holds on the slice being reported.

ferrum.calibration_chart(model_or_source=None, X=None, y=None, *,
                          y_true=None, y_pred=None,  # precomputed path (1-D positive-class proba)
                          n_bins=10, theme=None)

ferrum.gain_chart(model_or_source=None, X=None, y=None, *,
                  y_true=None, y_pred=None,          # precomputed path (soft scores)
                  compare=None, theme=None)

ferrum.lift_chart(model_or_source=None, X=None, y=None, *,
                  y_true=None, y_pred=None,          # precomputed path (soft scores)
                  compare=None, theme=None)
# **2026-09-10 (batch G, F-L11-08):** `gain_chart`/`lift_chart` now show the
# model's real class names for every class, not just a synthetic
# `"Class N"` rename for the first two. **Disclosed change:** binary charts
# label classes `"0"`/`"1"` (the stringified real value) instead of
# `"Class 0"`/`"Class 1"`; a 3rd-and-later class, which previously rendered
# unlabeled relative to its siblings, now renders with its own real name
# too. String-named classes (e.g. `classes_=["setosa", "versicolor"]`) were
# already rendered as-is and are unaffected. The only remaining display
# rename is the synthetic baseline row (`gain_frame`/`lift_frame`'s own
# reference-line data) → `"Baseline"`. See the `ModelSource.class_names`
# dated note under §3.1 for the underlying resolution rule
# (`class_names=` → `model.classes_` → real value, never invented).

ferrum.residuals_chart(model_or_source=None, X=None, y=None, *,
                        y_true=None, y_pred=None,    # precomputed path (fitted values); residuals = y_true − y_pred
                        kind="studentized", panels="auto",
                        # panels: "auto" | "single" | None | list of
                        #         "residuals_vs_fitted" | "qq" |
                        #         "scale_location" | "residuals_vs_leverage"
                        theme=None)
# **2026-05-12 (P3.12, F10):** `panels="auto"` ships the canonical
# 4-panel layout — `residuals_vs_fitted`, `qq`, `scale_location`,
# `residuals_vs_leverage` — composed as a 2x2 grid. `panels=None` or
# `panels="single"` returns just the residuals-vs-fitted panel
# (the pre-P3.12 behavior). For non-linear estimators (no `coef_`),
# `ModelSource.predictions()` emits all-NaN `leverage`; the chart
# builder silently drops the leverage panel from the auto layout so
# `panels="auto"` stays safe for RandomForest / GBM / etc.
#
# **2026-05-15 (precomputed path):** `y_true=`, `y_pred=` bypass the model
# entirely. Residuals are computed as `y_true − y_pred`. Leverage and
# Cook's distance are unavailable (no design matrix), so the leverage
# panel is silently dropped when `panels="auto"`. `compare=` is
# incompatible with the precomputed path (raises `ValueError`).
#
# **2026-07-02 (GH #44):** the leverage drop only degrades gracefully when
# at least one other panel survives it. If dropping `residuals_vs_leverage`
# would empty the resolved panel list — an explicit `panels=` of just
# `["residuals_vs_leverage"]`, or `cooks_distance_chart` (which always
# requests that single panel) — on a non-linear estimator or a precomputed
# source, the call raises `ValueError` naming the hat-matrix/`coef_`
# requirement and the estimator type (when a model is available) instead of
# returning an empty/broken chart. `compare=` inherits this: the error
# identifies which compare= member failed. `panels="auto"` degradation is
# unaffected — it still renders the remaining 3 panels.
#
# **2026-09-11 (wave-1 hardening, #208):** nine `_PrecomputedSource` methods
# (`roc_curve`, `pr_curve`, `cumulative_gain`/`lift_curve`, `calibration_curve`,
# `confusion_matrix`, `classification_report`, `predictions`,
# `discrimination_threshold`) now refuse a malformed `y_pred` shape with a
# typed `ValueError` naming the calling figure function and the observed vs.
# required shape, instead of leaking `IndexError`/`ArrowInvalid`/
# `TypeError: unhashable type: 'list'`/a numpy broadcast `ValueError`, or —
# for a 1-D `y_pred` against a genuinely multiclass `y_true` on roc/pr/
# gain/lift — silently computing a wrong-but-plausible curve. A degenerate
# SINGLE-class `y_true` is NOT treated as multiclass (matching the
# pre-existing, correct `binary_class_values` degradation to a NaN
# AUC/AP with a still-rendering curve) — only `> 2` distinct classes
# refuses. `predictions()` serves two public callers (`residuals_chart` and
# `prediction_error_chart`); its refusal now names whichever one the caller
# actually invoked, via a `chart_name=` parameter, rather than always
# naming `residuals_chart`.

ferrum.importance_chart(model_or_source, X=None, y=None, *,
                         method="builtin", top_k=20, orient="horizontal",
                         error_bars=True, theme=None)

ferrum.shap_chart(model_or_source, X=None, *,
                   kind="beeswarm",   # "beeswarm" | "bar" | "waterfall" | "force" | "heatmap"
                   max_display=20, sample_idx=None, theme=None)

# 2026-05-12 (P3.6): `shap_chart(kind=...)` is now a deprecation
# shim for three sibling functions — `shap_beeswarm_chart`,
# `shap_bar_chart`, and `shap_waterfall_chart` — that each take
# kind-specific keyword arguments without the `kind=` dispatcher.
# The visualizer side splits the same way: `SHAPBeeswarmVisualizer`,
# `SHAPBarVisualizer`, `SHAPWaterfallVisualizer`.  Old call sites
# emit `DeprecationWarning` but keep working.
ferrum.shap_beeswarm_chart(model_or_source, X=None, y=None, *,
                            max_display=20, order="abs_mean",
                            background=None, random_state=None, theme=None)
ferrum.shap_bar_chart(model_or_source, X=None, y=None, *,
                       max_display=20, order="abs_mean",
                       background=None, random_state=None, theme=None)
ferrum.shap_waterfall_chart(model_or_source, X=None, y=None, *,
                             sample_idx, max_display=20, order="abs_mean",
                             background=None, random_state=None, theme=None)

ferrum.learning_curve_chart(model, X, y, *, cv=5, scoring=None,
                              train_sizes=None, ci_style="band",
                              n_jobs=None, theme=None)

ferrum.validation_curve_chart(model, X, y, param, values, *,
                                cv=5, scoring=None, log_scale="auto",
                                ci_style="band", theme=None)

ferrum.cluster_diagnostics(model, *, ks, method="kmeans", scoring="both",
                             # scoring: "elbow" | "silhouette" | "both"
                             n_init=10, random_state=None, theme=None)
# **2026-06-22 (T3.4, D-FIRSTPARAM-1):** the first positional parameter was
# renamed `X` -> `model` so the whole model-diagnostic family shares one
# canonical first-param name. Positional callers are unaffected; the legacy
# keyword `X=` is accepted as a deprecated alias (supplying both `model=` and
# `X=` raises `TypeError`).

ferrum.decision_boundary_chart(model, X, y, *,
                                 features=(0, 1),    # column indices or names
                                 grid_resolution=200,
                                 proba=False,         # plot probability surface if True
                                 scatter=True, theme=None)

ferrum.discrimination_threshold_chart(model_or_source=None, X=None, y=None, *,
                                       y_true=None, y_pred=None,  # precomputed path (1-D positive-class scores); cv= not supported
                                       n_thresholds=50,
                                       metrics=("precision", "recall", "f1", "queue_rate"),
                                       highlight_best=True, compare=None, theme=None)
# 2026-05-12: gain_chart, lift_chart, and discrimination_threshold_chart gain
# compare=None for multi-model overlay, consistent with roc_chart / pr_chart.

ferrum.parallel_coordinates_chart(data_or_source, X=None, y=None, *,
                                   features=None, hue=None,
                                   rescale="minmax",   # "minmax" | "zscore" | None
                                   alpha=0.5, theme=None)

ferrum.class_prediction_error_chart(model_or_source=None, X=None, y=None, *,
                                     y_true=None, y_pred=None,  # precomputed path (1-D hard labels)
                                     normalize=False, theme=None)

ferrum.pca_scree_chart(model_or_source, X=None, *, n_components=None,
                        cumulative_line=True, threshold=0.95, theme=None)

ferrum.rank_chart(data_or_source, X=None, y=None, *,
                   rank="2d",       # "1d" | "2d"
                   algorithm=None,  # "1d" default: "shapiro"; "2d" default: "pearson"
                   top_k=None, theme=None)
# **2026-05-12 (P4.4, F4):** `rank_chart(rank=...)` is deprecated;
# use the two sibling functions `ferrum.rank1d_chart(...)` and
# `ferrum.rank2d_chart(...)` directly. They take the same parameters
# minus the `rank` flag and route to the same chart builders.
# `rank_chart` retains the dispatch behavior as a DeprecationWarning
# shim for backward compatibility.
ferrum.rank1d_chart(data_or_source, X=None, y=None, *,
                     algorithm=None, top_k=None, orient="horizontal",
                     color_field=None, random_state=None, theme=None)
ferrum.rank2d_chart(data_or_source, X=None, y=None, *,
                     algorithm=None, annot=True,
                     random_state=None, theme=None)

ferrum.alpha_selection_chart(model, X, y, alphas, *, cv=5, scoring=None,
                               log_scale=True, ci_style="band", theme=None)

ferrum.intercluster_distance_chart(model_or_source, X=None, *, k=None,
                                    method="mds", theme=None)

ferrum.cv_scores_chart(model, X, y, *, cv=5, scoring=None,
                        kind="box", split="both", theme=None)
```

> **2026-05-15 — Precomputed input path** (corrected count, **2026-09-10,
> batch G, F-L11-05**: eleven members, not ten or nine — the original note
> undercounted by omitting `prediction_error_chart`, which already
> documented and rendered on this path before this batch). The
> prediction-evaluation figure functions that appear above — `roc_chart`,
> `pr_chart`, `calibration_chart`, `gain_chart`, `lift_chart`,
> `discrimination_threshold_chart`, `confusion_matrix_chart`,
> `class_prediction_error_chart`, `residuals_chart`, `prediction_error_chart`,
> and (batch G) `classification_report_chart` — each accept `y_true=` and
> `y_pred=` as keyword-only arguments, bypassing the fitted model entirely.
> Callers supply arrays they have already computed; the function returns a
> chart visually identical to the model-backed path. Exactly one of {model
> path, precomputed path} must be active — supplying both, neither, or only
> one of `y_true`/`y_pred` raises `ValueError`. `compare=` is incompatible
> with the precomputed path. `y_pred` semantics vary by function: soft
> scores/probabilities for curve functions, hard labels for matrix functions
> and `classification_report_chart`, fitted values for `residuals_chart` and
> `prediction_error_chart`. See each function's docstring for the exact
> expected shape.

> **Schwabish SB3 — 2026-05-11 — figure-function defaults.** Eight
> Group-B functions adopt the Schwabish defaults from §3.19's principles doc:
>
> - `roc_chart`: `annotate_auc=True` by default; one `AUCLabel` per curve,
>   staggered along the y axis so labels do not collide at the line endpoint.
>   (This bullet's original text gave single-curve charts the active title
>   `"ROC — AUC X.XXX"` and multi-curve charts a descriptive `"ROC"` title;
>   the metric-carrying title is superseded by the 2026-09-13 #219/D4 note in
>   §3.15 — every ROC chart is titled `"ROC Curve"` and the overlay is the
>   sole metric surface.)
> - `pr_chart`: `annotate_ap=True` by default; overlay parallels `roc_chart`
>   (its active title is superseded by the same 2026-09-13 note). Binary
>   classifiers additionally render a dashed horizontal baseline at the
>   positive-class prevalence so the chance-level floor is always visible.
> - `calibration_chart`: new `annotate_brier=True` kwarg (default on). A
>   corner `BrierLabel` carries the per-sample Brier score (see the
>   2026-09-10 dated note immediately below — this bullet's original text,
>   "multi-model charts get one `BrierLabel` per model from the bin-level
>   reliability data," is superseded; its active title is superseded by the
>   2026-09-13 #219/D4 note).

> **2026-09-10 (batch G, F-L11-04 — supersedes the bin-level compared-path
> text above; itself partly superseded by the batch G close note that
> follows it, which replaces the `precomputed_value=` mechanism described
> here with a `brier` column on the frame — read both):** the `BrierLabel`
> overlay and the title now show the exact
> same per-sample Brier score, invariant to `n_bins=`/`strategy=`. Before
> this fix, the title computed the true per-sample Brier from
> `model.predict_proba`, but the overlay independently integrated the
> chart's own *binned* reliability-diagram data — not the Brier score at
> all, and a number that moved with `n_bins=`/`strategy=` even though the
> real Brier score never does (observed: 0.078 → 0.028 on `strategy=` alone
> while the true value held at 0.096). `_apply_metric_label_explicit` gains
> `precomputed_value: float | dict[str, float] | None` — when set, the
> overlay renders this value directly and `metric_fn`/`metric_field` are
> never consulted; the calibration builder passes the per-sample Brier it
> already computes for the title, so overlay and title cannot disagree.
> **Compared-path (`compare=`) semantics change, disclosed:** each model's
> label now shows **its own true per-sample Brier** (a `{model_name: brier}`
> dict passed as `precomputed_value=`), replacing the bin-level number every
> model previously shared. The compared-path title stays the plain
> `"Calibration Curve"` — no single number describes several models.
> **Precomputed path** (`calibration_chart(y_true=, y_pred=)`): the title
> now carries the Brier too (computed from the held `y_true`/`y_prob`
> arrays via `_PrecomputedSource.brier_score()`, the same formula as every
> other path — one Brier formula, `ferrum._metric_labels._brier_score`, used
> everywhere), where previously it stayed blank.

> **2026-09-10 (batch G close — supersedes the `precomputed_value=`
> mechanism described in the note immediately above, which was written
> earlier the same day; the user-visible contract below is the current
> one):** the per-sample Brier now lives
> on the **frame**, not in a builder-side override. `.calibration_curve()`
> gains a broadcast `brier` column (schema above), computed at frame
> assembly from the per-sample predictions and the binarized target — the
> last point at which the per-sample arrays exist — exactly as `auc`/`ap`
> are broadcast onto ROC/PR frames. The chart title, the figure builder's
> overlay, and the composable `chart + BrierLabel()` path all read that one
> column through the same rule (`_curve_metric`: read the kind's broadcast
> column when the frame carries it, integrate the drawn curve only when it
> does not), so `_apply_metric_label_explicit`'s `precomputed_value=`
> parameter and `_PrecomputedSource.brier_score()` are both removed —
> neither has a caller once the number rides on the frame. **User-visible
> consequence:** `calibration_chart(...) + fm.BrierLabel()` — the example
> in `BrierLabel`'s own docstring — now renders the true per-sample Brier
> matching the title, on both `strategy="uniform"` and
> `strategy="quantile"`. It previously rendered the bin-level integral
> (0.030 uniform / 0.009 quantile against a true 0.028 on a reference
> fixture) *and overwrote* the correct label the figure had already placed,
> leaving a chart strictly less correct than its own default. Composing
> `+ AUCLabel()`/`+ APLabel()` onto a `roc_chart`/`pr_chart` likewise now
> reads the kernel's `auc`/`ap` column rather than re-integrating; the
> values agreed already, so no rendered number moves there. A frame with no
> such column — arbitrary user data, the reason the composable path exists
> — still integrates the metric kind's own value function over the drawn
> curve, unchanged. (The `metric_fn` parameter that phrase originally named
> is gone: a surface now names a metric *kind* and the table supplies its
> value function, column and display name together.)
> - `residuals_chart`: new `annotate_metrics=True` kwarg (default on).
>   Overlays a top-right corner annotation with `R²` / `RMSE` / `MAE` on
>   the residuals-vs-fitted view in both the single-panel and the 4-panel
>   `panels="auto"` layouts. Implementation shares a pair of helpers —
>   `_inject_metrics_corner` (data augmentation) and
>   `_overlay_metrics_corner` (chart overlay) — so the orchestration
>   (which panels get the corner) stays separate from the implementation
>   (how to render it).
> - `importance_chart`: new `show_values=True` kwarg (default on). Emits
>   the formatted importance value at each bar's end via a same-data
>   `mark_text` overlay.
> - `learning_curve_chart` and `validation_curve_chart`: replace the
>   categorical color legend with endpoint-anchored direct labels (`train`
>   / `test`) via the new private `_direct_label_endpoint` helper. Legend
>   suppression flows through the new `Color(field, legend=None)` honored
>   kwarg (serialized to the renderer as `legend = {"disabled": true}`).
> - `confusion_matrix_chart`: docstring already specified `annotate=True`
>   default; SB3 added a regression test.
>
> Every Group-B function also accepts an opt-in `subtitle: str = None`
> kwarg that flows through to the `Title(text, subtitle=…)` primitive.
>
> Implementation lives in `src/ferrum/_diagnostics/charts.py` for the
> single-model builders and `src/ferrum/_direct_label.py` for the
> direct-label helper; primitives in `src/ferrum/annotations.py`. The
> renderer-side legend suppression and `mark_text` align / dx / dy
> support land in `crates/ferrum-core/src/render/prepare.rs` and
> `…/marks/text.rs` respectively.

---

### 3.15 Sklearn-Protocol Visualizers

Visualizers implement `fit` / `score` / `show` for drop-in compatibility with sklearn workflows. Unlike Yellowbrick, `.show()` always returns a `Chart` object — not a matplotlib figure.

**Base class:**

```
class FerrumVisualizer:
    def fit(self, X, y=None) -> self
    def score(self, X, y) -> float        # delegates to model.score; 0.0 if no model
    def show(self) -> Chart
    def __repr__(self) -> str             # summary string with key metrics
    @property
    def has_score(self) -> bool           # True iff score() returns a real metric
```

> **Note (2026-06-22, cohesion campaign T4.6):** The base `score(X, y)`
> delegates to `float(self.model.score(X, y))` whenever the wrapped model
> exposes a `.score` method, and returns `0.0` only for genuinely no-model
> visualizers (`model=None`: rank / parallel-coordinates / class-balance /
> elbow). Subclasses whose score is not the estimator's own (e.g.
> `ROCVisualizer` → `roc_auc_score`) override it. `has_score` is a derived
> read-only property (not a hand-set class flag): it reports `True` exactly
> when `score()` returns a real metric, mirroring the model-has-`.score`
> guard, so the two can never drift.
>
> **Refinement (2026-06-23, cohesion campaign S4FIX2):** the "can never drift"
> guarantee now holds for the two non-standard subclass shapes as well.
> `ROCVisualizer` overrides `has_score` to mirror its own `score()` condition
> (scoreable when the model exposes `predict_proba` or a callable `.score`), so
> a `predict_proba`-only model — no `.score`, a valid sklearn shape — reports
> `has_score=True` and `score()` returns the AUC. A multi-model
> `CalibrationVisualizer` overlay (two or more positional models, or a single
> dict-of-models) has no single well-defined metric: it now reports
> `has_score=False` and `score()` returns the documented `0.0` no-single-score
> fallback, rather than silently scoring only the first model. Single-model
> calibration behavior is unchanged.

**Concrete visualizers:**

| Class | Wraps |
|---|---|
| `ROCVisualizer(model, *, micro=True, macro=True, per_class=True, annotate_auc=True, subtitle=None, compare=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | ROC chart |
| `PRVisualizer(model, *, per_class=True, average="macro", annotate_ap=True, iso_lines=False, subtitle=None, compare=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | PR chart |
| `ConfusionMatrixVisualizer(model, *, normalize="true", annotate=True, mark=None, encode=None, properties=None, layers=None, theme=None)` | Confusion matrix |
| `ClassificationReportVisualizer(model, *, mark=None, encode=None, properties=None, layers=None, theme=None)` | Heatmap of precision/recall/F1 per class |
| `CalibrationVisualizer(*models, n_bins=10, strategy="uniform", annotate_brier=True, subtitle=None, compare=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Calibration chart |
| `ResidualsVisualizer(model, *, kind="studentized", cook_threshold=None, panels=None, annotate_metrics=True, subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Residuals diagnostic |
| `PredictionErrorVisualizer(model, *, reference_line=True, ci=None, reference_band=False, mark=None, encode=None, properties=None, layers=None, theme=None)` | Prediction error |
| `FeatureImportancesVisualizer(model, *, method="builtin", top_k=20, orient="horizontal", error_bars=True, show_values=True, subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Importance chart |
| `LearningCurveVisualizer(model, *, cv=5, scoring=None, train_sizes=None, ci_style="band", subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Learning curve |
| `ValidationCurveVisualizer(model, param, values, *, cv=5, scoring=None, log_scale="auto", ci_style="band", subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Validation curve |
| `SilhouetteVisualizer(model, *, k=None, subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Silhouette chart (clusterers) |
| `ElbowVisualizer(model_class, *, ks, metric="distortion", mark=None, encode=None, properties=None, layers=None, theme=None)` | Elbow chart |
| `ManifoldVisualizer(model=None, *, method="umap", mark=None, encode=None, properties=None, layers=None, theme=None)` | Embedding projection |
| `ClassBalanceVisualizer(*, mark=None, encode=None, properties=None, layers=None, theme=None)` | Class frequency bar chart |
| `CooksDistanceVisualizer(model, *, threshold=None, kind="studentized", panels=("residuals_vs_leverage",), annotate_metrics=True, subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Cook's distance |
| `SHAPVisualizer(model, *, kind="beeswarm", max_display=20, sample_idx=None, order="abs_mean", background=None, per_class=False, zero_line=True, mark=None, encode=None, properties=None, layers=None, theme=None)` | SHAP chart — **deprecated** since 2026-05-12 (P3.6); use the three sibling classes below. |
| `SHAPBeeswarmVisualizer(model, *, max_display=20, order="abs_mean", background=None, per_class=False, zero_line=True, mark=None, encode=None, properties=None, layers=None, theme=None)` | SHAP beeswarm |
| `SHAPBarVisualizer(model, *, max_display=20, order="abs_mean", background=None, per_class=False, mark=None, encode=None, properties=None, layers=None, theme=None)` | SHAP mean-absolute bar |
| `SHAPWaterfallVisualizer(model, *, sample_idx, max_display=20, order="abs_mean", background=None, per_class=False, mark=None, encode=None, properties=None, layers=None, theme=None)` | SHAP single-sample waterfall |
| `DiscriminationThresholdVisualizer(model, *, n_thresholds=50, metrics=("precision", "recall", "f1", "queue_rate"), cv=None, threshold_line=False, optimum_label=True, subtitle=None, compare=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Discrimination threshold chart (binary classifiers only) |
| `ParallelCoordinatesVisualizer(*, features=None, hue=None, rescale="minmax", alpha=0.5, mark=None, encode=None, properties=None, layers=None, theme=None)` | Parallel coordinates. No model required; `fit(X, y)` accepts raw feature matrix. `hue` applied from `y` if provided. |
| `ClassPredictionErrorVisualizer(model, *, normalize=False, show_counts=True, mark=None, encode=None, properties=None, layers=None, theme=None)` | Class prediction error chart |
| `PCAVarianceVisualizer(model, *, n_components=None, cumulative_line=True, threshold=0.95, subtitle=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | PCA scree plot. Model must expose `explained_variance_ratio_` attribute (sklearn PCA, TruncatedSVD, or compatible). |
| `Rank1DVisualizer(*, algorithm="shapiro", orient="horizontal", top_k=None, color_field=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Univariate feature ranking bar chart. No model required. |
| `Rank2DVisualizer(*, algorithm="pearson", annot=True, mark=None, encode=None, properties=None, layers=None, theme=None)` | Pairwise feature correlation heatmap. No model required. |
| `AlphaSelectionVisualizer(model, alphas, *, cv=5, scoring=None, log_scale=True, highlight_best=True, mark=None, encode=None, properties=None, layers=None, theme=None)` | Regularization parameter selection curve |
| `InterclusterDistanceVisualizer(model, *, method="mds", k=None, mark=None, encode=None, properties=None, layers=None, theme=None)` | Intercluster distance map |
| `CVScoresVisualizer(model, *, cv=5, scoring=None, kind="box", split="both", mark=None, encode=None, properties=None, layers=None, theme=None)` | Cross-validation score distribution |

> **2026-09-10 (batch G, F-L11-06 + F-L12-09 + F-L10-07 — full
> visualizer↔chart signature parity):** every visualizer above gained, as
> explicit keyword-only constructor parameters, every kwarg its underlying
> chart builder documents — including the uniform override quartet
> `mark=`/`encode=`/`properties=`/`layers=` on every single visualizer in
> the family (checkpoint decision: no quartet exemptions). This closes the
> gap where `PredictionErrorVisualizer` was the family's only fully-threaded
> member and every other visualizer silently dropped documented builder
> kwargs, with no `mark=`/`encode=`/`properties=`/`layers=` override
> reachable from any visualizer at all. `FerrumVisualizer.__init__`'s prior
> `**_extra: Any` silent swallow is removed — an unknown constructor kwarg
> on any visualizer now raises `TypeError` naming it, instead of being
> silently accepted and discarded. `ResidualsVisualizer(panels="auto")`
> renders the canonical 4-panel grid identical to
> `residuals_chart(panels="auto")` (a shared helper now owns the
> `"auto"` → 4-panel-list expansion, consumed by both callers — a naive
> per-caller forward previously fell back to single-panel, since the
> builder's own vocabulary is `None`/`"single"`/an explicit list, never
> `"auto"`). `CooksDistanceVisualizer`'s new `panels=` defaults to exactly
> its historical `("residuals_vs_leverage",)` tuple — unchanged output for
> every existing call. Three pre-existing spelling drifts are kept for
> backward compatibility, not unified (spelling unification is a logged
> follow-up, not this batch): `ROCVisualizer.micro=`/`macro=` vs. the
> builder's `average=`; `CooksDistanceVisualizer.threshold=` vs. the
> builder's `cook_threshold=`; `CalibrationVisualizer`'s variadic `*models`
> vs. the public `calibration_chart`'s `compare=`. A mechanical parity test
> (`inspect.signature` diff against each visualizer's actual `_build_chart`
> target, patterned on `test_core_stub_parity.py`) enforces every
> visualizer's keyword-only surface against its builder's, with an
> explicit, rationale-carrying skip-map for exactly those three drifts —
> an empty-rationale or stale skip-map entry fails the test, as does any
> `FerrumVisualizer` subclass missing from the test's own completeness
> registry. No existing kwarg was renamed or removed on any visualizer.
>
> **`compare=` metric semantics (F-L10-07).** `ROCVisualizer`, `PRVisualizer`,
> and `DiscriminationThresholdVisualizer` accept `compare=` per the drift
> map above; their derived metrics (`auc_mean`, `ap_mean`,
> `best_threshold`/`best_f1`, and `__repr__`'s summary) always describe the
> **base model** (the visualizer's own `model=` argument), consistent with
> `.score()` — never a value pooled across every compared model, and never
> silently substituted with a stronger/weaker rival's number. Before this
> fix, e.g. `DiscriminationThresholdVisualizer._metrics` searched the
> concatenated per-model frame for the best F1 regardless of which model it
> came from, so the object could report one model's `.score()` and a
> different model's derived metrics simultaneously.
> `CalibrationVisualizer`'s own genuinely-symmetric multi-model shape
> (`has_score=False`, `score()=0.0`, no distinguished base) is unaffected —
> `compare=` on the other three has one distinguished base model (the
> primary `model=` argument) where `CalibrationVisualizer`'s `*models` does
> not. **Rider (2026-09-13, wave 3 T5, #213):** that fix reached the
> *metrics* only. `DiscriminationThresholdVisualizer._build_chart` kept
> forwarding the unfiltered compared source to the pooled chart builder, so
> the rendered F1-optimum annotation could still name the rival while
> `_metrics`/`__repr__` named the base. `compare=` on that visualizer now
> renders small multiples (see the 2026-09-13 note in the `compare=`
> composition section), so the `"base"` panel's annotation and the repr
> describe the same model by construction. `ROCVisualizer` and
> `PRVisualizer` never had the render half of this problem: their charts
> overlay by `color="model"`, so every curve is already labeled with the
> model it belongs to.

> **2026-09-13 (wave 3, #212 — `CalibrationVisualizer(compare=)`).**
> Supersedes both statements above about `CalibrationVisualizer`: the
> drift-map row naming its variadic `*models` as an adjudicated drift from
> `calibration_chart`'s `compare=`, and the sentence saying its
> "genuinely-symmetric multi-model shape" stands apart from `compare=`'s
> distinguished base. The class now accepts **both**, as two different
> questions rather than one drift:
>
> - `CalibrationVisualizer(model, compare={"rival": other})` has one
>   distinguished base — the positional `model`, labelled `"base"` — with
>   exactly the semantics `ROCVisualizer`/`PRVisualizer`/
>   `DiscriminationThresholdVisualizer` already have: `has_score` is `True`,
>   `.score()` scores the base model, and `_metrics` carries a single
>   `calibration_error`, the base model's own (never pooled across rivals).
>   Before this change the kwarg did not exist and the call raised
>   `TypeError: CalibrationVisualizer.__init__() got an unexpected keyword
>   argument 'compare'`.
> - `CalibrationVisualizer(m_a, m_b)` / `CalibrationVisualizer({"a": m_a,
>   "b": m_b})` is **unchanged**: symmetric, no base, `has_score=False`,
>   `score()` returns the `0.0` fallback, and `_metrics` carries one
>   `calibration_error[<model name>]` key per model. The 2026-06-23
>   refinement above still describes this shape exactly.
> - The two are mutually exclusive. Several positional models (or a
>   dict-of-models) together with `compare=` is a self-contradicting
>   request — the two spell different metric contracts — and raises
>   `ValueError` at construction naming both idioms.
>
> `fit` resolves `compare=` through the same shared helper the three curve
> visualizers use, so `compare={"base": ...}` and a non-dict `compare=`
> refuse with the same messages on all four, and the rendered chart is
> byte-identical to `calibration_chart(model, X, y, compare=...)` — the
> parity is the constructor and the metric semantics, not a new render
> path. No default, no existing call's output, and no golden changed.

> **2026-05-11 (Phase 10 — Model Diagnostics):** Every visualizer constructor
> documented above accepts `random_state: int | None = None` as an additional
> keyword argument. The base `FerrumVisualizer.__init__` stores it on `self`
> and forwards it to the underlying `ModelSource` (or to `_diagnostics/stats`
> compute helpers for the no-model variants). Visualizers whose backing
> compute is deterministic (`ROCVisualizer`, `PRVisualizer`,
> `ConfusionMatrixVisualizer`, `CalibrationVisualizer`,
> `ResidualsVisualizer`, `PredictionErrorVisualizer`,
> `DiscriminationThresholdVisualizer`, `PCAVarianceVisualizer`,
> `Rank2DVisualizer` — except `algorithm="kendall"` which is also
> deterministic, `ClassBalanceVisualizer`, `ClassificationReportVisualizer`,
> `ClassPredictionErrorVisualizer`) ignore the value; the rest propagate
> it. (`DiscriminationThresholdVisualizer`'s membership in this deterministic
> list was briefly false while its `cv=` path used a shuffled splitter — see
> the dated 2026-09-13 §3.1 note for that history; the unshuffled
> `StratifiedKFold` restores it.) The `ElbowVisualizer(model_class, *, ks, ...)` signature is unique —
> it takes a model **class** and fits one instance per k inside `fit()`.

---

### 3.16 Output and Rendering

#### `RenderConfig`

```
RenderConfig(
    format="svg",         # "svg" | "png" | "html" | "json"
    scale=2.0,            # pixel density multiplier for PNG
    embed_fonts=True,     # embed fonts in SVG/HTML
    background=None,      # override chart background for export
    width=None, height=None,
    engine="ferrum",      # "ferrum" | "vega-lite"  (vega-lite emits JSON spec)

    # Auto-raster policy (see §3.3 and §3.17)
    raster_threshold=500_000,   # int | None — mark count above which auto-raster
                                # is triggered. None disables auto-raster entirely.
                                # Overrides ferrum.config.set_raster_threshold()
                                # for this render call.
    raster_behavior="warn",     # "warn" | "silent" | "error" — controls whether
                                # a warning is emitted when auto-raster
                                # substitution occurs.
    raster_aggregate="count",   # "count" | "density" | "mean" | "sum" | "any" —
                                # default aggregate for auto-raster substitution
    raster_field=None,          # str | None — column to aggregate when
                                # raster_aggregate is "mean" or "sum"
    raster_cmap="viridis",      # default colormap for auto-raster substitution

    # Backend selection (see §3.17)
    backend=None,               # "svg" | "tiny-skia" | "wgpu" | None (auto-select)
    tile_parallel=False,        # enable Rayon tile parallelism in the tiny-skia
                                # backend. Increases memory usage; recommended
                                # for charts with > 500k marks.
    font_path=None,             # str | list[str] | None — override font search
                                # paths for the tiny-skia backend. None = use
                                # bundled Ferrum fonts with system font fallback.
)
```

`raster_threshold=None` and `raster_behavior="silent"` are not equivalent: `None` disables auto-raster entirely (the chart renders all marks through the requested backend, even if performance suffers); `"silent"` keeps auto-raster active but suppresses the substitution warning.

> **Dated note (2026-09-03, batch B task 5 — `raster_aggregate` validates at construction):** `raster_aggregate` is checked at `RenderConfig` construction against its true accepted set — `"count"` | `"density"` | `"mean"` | `"sum"` | `"any"` — matching the underlying Rust `Raster` transform exactly. `"max"`/`"min"`/`"median"` were never valid (despite being reachable at render time before this task, where they surfaced a confusing runtime error well after construction) and now raise `ValueError` immediately. The new `raster_field=` names the column to aggregate when `raster_aggregate` is `"mean"` or `"sum"` (both require it and raise at construction if it's missing); it is genuinely ignored — dropped before the substitution, never resolved as a column name — for `"count"`, `"density"`, and `"any"`, which need no field. The aggregated column must be `Float64`-typed; a non-`Float64` column raises `stat_raster: column '...' must be Float64` at render time (a pre-existing constraint of `Raster`, not new here). `mark_raster` and `desugar_raster`'s docstrings are truth-fixed to match this accepted set — `min`/`max` were never valid and are no longer listed as if they were.
>
> **Dated note (2026-09-05, batch D task 2 — correcting the note above):** the aggregated column's dtype constraint stated above no longer holds. The batch-D M1 dtype-policy sweep widened `Raster`'s `x`, `y`, and `field` columns to accept any integer, unsigned, or float dtype (coerced to `Float64`); the quoted `stat_raster: column '...' must be Float64` error no longer exists anywhere in the tree. A non-numeric column now raises `stat_raster: column '<name>' must be numeric, got <Dtype>` instead, naming the transform, column, and actual dtype. This is the general M1 dtype-widening policy applied consistently across the transform engine (design spec `2026-09-05-batch-d-transform-engine-design.md` §4.1); `raster` was one of the transforms the batch-D sweep touched.

> **2026-05-09 (Phase 7 implementation note):** Phase 7 honors the following
> `RenderConfig` fields: `scale`, `embed_fonts`, `background`, `width`,
> `height`. The remaining fields (`format`, `engine`, `raster_threshold`,
> `raster_behavior`, `raster_aggregate`, `raster_cmap`, `backend`,
> `tile_parallel`, `font_path`) are deferred to subsequent phases that ship
> their corresponding features. `embed_fonts` is treated as always-true in
> Phase 7 for visual determinism (rendered text uses the bundled Inter
> Regular regardless of system font availability); future phases may surface
> the `False` case for size-conscious users.

> **2026-05-10 (Phase 8a):** `.show()` env detection in 8a covers Jupyter
> inline (`_repr_svg_` / `_repr_html_` rich display) and a browser fallback
> (writes temp HTML, calls `webbrowser.open`). Sixel terminal output and the
> standalone HTML wrapper output are deferred to Phase 9. `.save()` honors
> `.svg` and `.png` extensions; `.html` and `.json` raise
> `NotImplementedError` pointing to Phase 9.

#### Chart output methods

The output surface follows one convention: `to_*` returns an in-memory value,
`show()` displays, `save(path)` writes to disk.

```
chart.show(*, renderer=None)                  # auto-detect environment, display
chart.to_svg() -> str                         # in-memory SVG markup
chart.to_png() -> bytes                        # in-memory PNG bytes
chart.to_html() -> str                         # in-memory interactive HTML
                                               #   (byte-identical to save('.html'))
chart.to_spec() -> ChartSpec                  # internal dataclass (Chart only)
chart.to_json(*, indent=None) -> str
chart.save(path, *, format=None, **render_kwargs)
chart.pipe(fn, *args, **kwargs) -> Any        # apply a function to self
```

These methods exist on `Chart` and on every composition view (`HConcatChart`,
`VConcatChart`, `LayerChart`, `ConcatChart`, `JointChart`, `RepeatChart`,
`ClusterMapChart`), with one exception: `to_spec` exists on `Chart` only — a
`ChartSpec` is leaf-only by construction; composition views expose `to_dict()`/
`to_json()` for their declaration (see the 2026-09-11 dated note below). On
composition views `to_png` is scale-only (it accepts `scale=` but no `raster`
argument).

**Deprecated aliases.** `show_svg()` and `show_png()` are retained as deprecated
aliases that forward to `to_svg()` / `to_png()` and emit a `DeprecationWarning`.
They are slated for removal after `0.16.0`. New code should call `to_svg()` /
`to_png()` / `to_html()`.

> **Note (2026-06-04):** The in-memory output methods were renamed
> `show_svg()` -> `to_svg()` and `show_png()` -> `to_png()`, and `to_html() -> str`
> was added (returns the interactive HTML string, byte-identical to
> `save('.html')`). This unifies the surface on the `to_*` = in-memory-value
> convention, leaving `show()` for display and `save(path)` for disk. The old
> `show_svg`/`show_png` names remain as deprecated aliases that forward to the
> new methods and emit a `DeprecationWarning`; they are slated for removal after
> `0.16.0`. The rename applies to `Chart` and all composition views; on
> composition views `to_png` is scale-only (no `raster` argument).

> **Dated note (2026-09-11, batch H task 11 — `to_dict`/`to_json`/`pipe` generalize; `to_spec` stays `Chart`-only, F-L14-01/F-L14-02/F-L06-04):** `to_json(*, indent=None)` and `pipe(fn, *args, **kwargs) -> Any` are now real on every composition view (`HConcatChart`, `VConcatChart`, `ConcatChart`, `LayerChart`, `JointChart`, `RepeatChart`, `ClusterMapChart`), not just `Chart` — this table's blanket "these methods exist on ... every composition view" claim was itself stale before this task, since neither existed on any of the seven. Each composite also gains `to_dict() -> dict`: a *recursive* declaration dict (`{"kind": <layout>, ...layout fields..., <child slots>: <embedded dict(s)>}`) generalizing the pre-existing `JointChart.spec` / `RepeatChart.spec` / `ClusterMapChart.spec` introspection properties — a leaf `Chart` child embeds as `json.loads(chart.to_spec().to_json())` (round-trippable through `ChartSpec.from_json`), and a nested composite child now embeds *recursively* via its own `to_dict()` instead of the old `None` placeholder. Those three `.spec` properties are unchanged in shape and now recursive-consistent with `to_dict` through the one shared embedding helper (`composition._embed_chart_spec`) — for `JointChart`/`RepeatChart`/`ClusterMapChart`, `to_dict()` returns `.spec` (plus figure chrome when set — see the 2026-09-11 note below).
>
> **Figure chrome in the declaration dict (2026-09-11, batch H close remediation).** A composition's figure-level `title` / `subtitle` / `caption` — set via `.properties(title=..., subtitle=..., caption=...)` — is part of its declaration, so `to_dict()` carries it under those key names, omitted when unset. The **six** `_CompositeBase` views (`HConcatChart`, `VConcatChart`, `ConcatChart`, `JointChart`, `RepeatChart`, `ClusterMapChart`) carry all three, because all three are figure-level slots on those classes. `LayerChart` carries **`title` only**: it is a single-plot overlay with one `_title` slot, and `.properties(subtitle=..., caption=...)` on it fans those kwargs to its member charts (a layer's own title blob) rather than storing figure chrome — that fan-out is pre-existing behavior, lossy for this purpose (a fanned `caption` reaches no layer slot at all), and is named here rather than papered over. Before this correction only `LayerChart` emitted `title` (and emitted it as `None` when unset) while the other six silently dropped every one of the three, so `(a | b).properties(title="t").to_dict()` had no title anywhere. Because unset chrome is omitted, a chrome-free composition's dict is unchanged, and `JointChart`/`RepeatChart`/`ClusterMapChart`'s `to_dict()` still equals `.spec` exactly when untitled — for a titled one it is `.spec` plus the chrome keys. The `.spec` properties themselves stay pure layout introspection and never gain chrome.
>
> **Chrome values are normalized at ingress, LOSSLESSLY where a lossless
> answer exists, and applied to all three chrome keys (2026-09-11, batch H
> close cycle 3, H3-4; made lossless and widened to `subtitle=`/`caption=`
> in cycle 4).** All three of `title=`/`subtitle=`/`caption=` accept a bare
> `str` or a `ferrum.Title`, normalized at `.properties()` — before it ever
> reaches `_figure_title`/`_figure_subtitle`/`_figure_caption` (or
> `LayerChart`'s single `_title`) — not at `to_dict()`/serialization time.
> **`title=fm.Title(text=..., subtitle=...)` routes the `Title`'s own
> `subtitle` into the composite's separate, already-working `subtitle` slot**
> (the six `_CompositeBase` views only — `LayerChart` has no such slot, see
> below) when no explicit `subtitle=` was ALSO given in the same call;
> giving both **in the same `.properties()` call** is refused typed, naming
> both spellings, the same idiom as the conditional-vocabulary collision
> refusals. This scoping is deliberate, not an oversight (2026-09-11, batch
> H close cycle 5): **across separate calls it is not a collision at all** —
> a later `.properties()` call's routed subtitle silently replaces an
> earlier one, the same last-write-wins rule an explicit `subtitle=` already
> follows across chained calls (`.properties(subtitle="OLD").properties(title=fm.Title("T",
> subtitle="NEW"))` ends with the figure subtitle `"NEW"`, no warning — a
> routed subtitle behaves exactly like an explicit one once it lands in the
> `subtitle` slot). This corrects the cycle-3
> shipped behavior, which flattened `title=fm.Title(..., subtitle=...)` to
> `.text` and **silently dropped** the subtitle with no warning — the same
> disease this batch refused `shape` conditionals for elsewhere in the same
> cycle ("accepted and inert"), reached here via the declaration layer
> instead of the wire. A `Title` field with genuinely nowhere to go on ANY
> chrome slot — `anchor`, `offset`, `font_size`, `font_weight`, `color`,
> `subtitle_font_size`, `subtitle_color` on any of the three keys; `subtitle`
> itself on `subtitle=`/`caption=` (nesting makes no sense) or on
> `LayerChart`'s `title=` (no separate subtitle slot exists there) — is
> refused typed rather than silently dropped, naming the field(s) and the
> fix. `to_dict()` still serializes `title`'s (already-plain) text through
> `Title.to_spec_dict()`, so the top-level `title` stays `{"text": ...}`,
> identical in shape to every embedded child's `title`; the routed subtitle
> surfaces as the top-level `subtitle` KEY (a dict sibling of `title`, not
> nested inside `title`'s own dict), since that is the slot that actually
> renders.
>
> **The composite-tree renderer only ever accepted a bare-string title (and
> subtitle/caption), and now the declaration surface agrees for all three
> (2026-09-11, batch H close cycle 3, H3-4; widened in cycle 4).** Every
> composite output path (all six `_CompositeBase` views' `to_svg`/
> `.interactive()`, and `LayerChart.to_svg`'s default/shared-y path) lowers
> the figure title into the one-call Rust composite entry's wire `title`
> field, which is a bare `Optional<String>` — there is no per-figure-title
> richness slot in the composite tree the way a leaf `Chart`'s
> `ChartSpec.title` has one (a full `TitleSpec` with
> subtitle/anchor/offset/colors). Cycle 3 fixed this for `title=` only;
> cycle 4 found `subtitle=fm.Title(...)`/`caption=fm.Title(...)` still
> raised the identical unmapped PyO3 `TypeError: 'Title' object is not an
> instance of 'str'` from `to_svg()` on every wrapper including `LayerChart`
> (whose `subtitle=`/`caption=` fan a normalized string to member charts
> rather than storing composite-level chrome — pre-existing, unrelated to
> this fix), while `to_dict()`/`to_json()` silently accepted the same call.
> All three keys, on all seven wrappers, now agree: `.properties(title=...,
> subtitle=..., caption=...)` with any `fm.Title(...)` either renders or
> refuses typed at `.properties()` time — it never builds a declaration
> `to_svg()` cannot honor. `ferrum.Title`'s own docstring previously claimed
> it was "accepted everywhere a title string is accepted" including
> `HConcat/VConcat/Layer(title=...)`; that claim is corrected to describe
> the routing/refusal rule above.
>
> **`to_spec() -> ChartSpec` stays `Chart`-only.** `ChartSpec` is a leaf-only Rust dataclass (one mark/encoding/transform set); a composite view has no single `ChartSpec` to return, so it does not gain this method — the original wording above implying `to_spec` was universal across composition views was the stale claim being corrected here, not a feature being removed.
>
> **Composite construction now validates its children (H17).** `HConcatChart` / `VConcatChart` / `ConcatChart` / `LayerChart` / `JointChart` / `RepeatChart` / `ClusterMapChart` each raise `TypeError: {Class} children must be Chart or composition objects, got {TypeName}` at construction when a child is neither a `Chart` nor another composition object (e.g. `fm.HConcatChart([a_dataframe])`), instead of constructing silently and failing much later at render time with `_lower_any`'s generic, node-agnostic `ValueError: unrecognized node kind '...'` — that render-time error remains as a backstop for node shapes the construction-time check cannot describe (e.g. a `JointChart` marginal that is a composite but not a leaf `Chart`, still rejected by `_composite_tree`). `Chart.__or__`/`__and__` and every composition view's `__or__`/`__and__` now return `NotImplemented` for a non-chart, non-composition operand (mirroring `Chart.__add__`'s existing `NotImplemented` fallback), so `chart | 3` raises Python's own operator `TypeError` ("unsupported operand type(s) for |: ...") rather than reaching the constructor at all.

#### Environment detection order for `.show()`

1. Jupyter / IPython → inline SVG (static) or WASM widget (if `.interactive()`)
2. VS Code notebook → same
3. ~~Terminal with sixel support → PNG via sixel~~ — **REMOVED (2026-05-15):** sixel is niche, inconsistent across terminal emulators, and ferrum's audience is Jupyter/browser-first. Not worth implementation effort.
4. Otherwise → write temp HTML and open browser

---

### 3.17 Rendering Backends

Ferrum has three rendering backends. The backend used for a given output is determined by the call (`.save()` vs `.show()` vs `.interactive()`), the file extension or format, the mark count, and any explicit override in `RenderConfig` or `ferrum.config`.

#### SVG Backend

- Default for static output when mark count is below threshold.
- Resolution-independent, text-searchable, scalable.
- Not viable above ~50k marks.
- Output: `.svg` files; inline SVG in notebooks.

#### tiny-skia Backend (CPU Rasterizer)

- Pure Rust, zero system dependencies (no Cairo, no libpng, no X11).
- Bundled in the wheel; works on Linux, macOS, Windows, ARM.
- Text rendering via `fontdue` (rasterization) + `rustybuzz` (shaping); fonts bundled or resolved from system font paths.
- Default for `.save("chart.png")` and `.show()` when mark count exceeds threshold.
- Not used for interactive output.
- Performance: ~200–800 ms for 1M antialiased marks (CPU-bound, SIMD-optimized); tile-based Rayon parallelism available as opt-in via `RenderConfig`.
- Default output scale: `2.0` (retina); configurable in `RenderConfig`.
- Limitations: no vector output; zoom invalidates raster and requires re-render.

#### wgpu / WASM Backend (GPU Interactive)

- Used only when `.interactive()` is called.
- `wgpu` targets WebGPU (Chrome 113+, Edge 113+, Safari 17+) with automatic WebGL2 fallback for broader compatibility.
- Vello is not used; its compute-shader-based approach is incompatible with the WebGL2 fallback path. Instanced draws for simple marks (points, bars) and tessellation for curves and areas require no compute shaders and work across both WebGPU and WebGL2.
- Text and axis labels rendered via CSS overlay (real DOM text; accessible, no font bundling required in browser context).
- Pan and zoom are GPU matrix transforms — no Python round-trip, no re-render.
- Selections that trigger Python-side recomputation require a Python kernel round-trip (~50–200 ms in local notebooks); managed via the `anywidget` protocol.
- Jupyter integration via `anywidget`; browser integration via standalone HTML bundle.
- WASM bundle size: ~2–4 MB compressed (CDN-hosted recommended).
- On macOS including M-series, the native `wgpu` backend targets Metal; unified memory architecture reduces GPU buffer upload latency vs. discrete GPUs.
- Viewport changes (zoom, resize) trigger re-rasterization of `mark_raster` layers at the new resolution.

#### Backend selection

The threshold column refers to `raster_threshold` (default `500_000`), configurable in `RenderConfig` and `ferrum.config`.

| Call                                            | Backend                            |
|-------------------------------------------------|------------------------------------|
| `.save("chart.svg")`                            | SVG (warns if `mark_count > 50k`)  |
| `.save("chart.png")`                            | tiny-skia                          |
| `.show()` — `mark_count < threshold` (def 500k) | SVG inline                         |
| `.show()` — `mark_count >= threshold`           | tiny-skia PNG                      |
| `.show()` in headless / no display              | tiny-skia PNG (always)             |
| `.show()` in terminal with sixel support        | ~~tiny-skia → sixel~~ — removed    |
| `.interactive()`                                | wgpu / WASM                        |

Per-call `RenderConfig` takes precedence over global `ferrum.config` settings for all threshold and backend parameters.

---

### 3.18 Data Source Compatibility

Ferrum accepts the following as `data` in `Chart(data=...)` or figure-level functions:

| Type | Notes |
|---|---|
| `polars.DataFrame` | Zero-copy via Arrow |
| `polars.LazyFrame` | Collected at render time; lazy evaluation where possible |
| `pandas.DataFrame` | Converted to Arrow once on first access |
| `pyarrow.Table` | Native |
| `pyarrow.RecordBatch` | Native |
| `dict[str, list]` | Converted to Arrow inline |
| `list[dict]` | Converted to Arrow inline |
| `numpy.ndarray` (2D) | Columns named `col_0`, `col_1`, ... |
| `ModelSource` | Wraps an estimator; derived data accessed via `.predictions()` etc. |
| `ComparedModelSource` | Multi-model wrapper; adds `model` column to all derived data |
| `str` / `pathlib.Path` | CSV, Parquet, JSON, NDJSON; loaded lazily |
| `None` | Data supplied per-layer |

> **2026-05-10 (Phase 8a):** Data input compatibility provided via narwhals
> (~1.x) for pandas, modin, cuDF, dask, ibis. Polars goes via direct CDI
> (zero-copy). pyarrow `Table` and `RecordBatch` accepted as native. Dict,
> list-of-records, and 2D numpy with auto-named columns supported. File path
> inputs (`Chart("file.csv")`) and `ModelSource`/`ComparedModelSource`
> deferred to Phases 9 and 10 respectively.

---

### 3.19 Utilities

```
# ferrum.data — REMOVED (2026-05-15)
# Users source sample data from sklearn/seaborn optional dependencies.
# A ferrum-native dataset loader duplicates existing ecosystem tooling
# with no user-facing benefit. This namespace will not be implemented.

ferrum.color.palette(scheme, n)            # return n colors from a scheme as hex list
ferrum.color.to_hex(color, *, scale=None)  # normalize color to hex; scale="unit"|"byte"
ferrum.color.sequential(name, n=256)       # n render-truth samples of a sequential scheme
ferrum.color.diverging(name, n=11)         # n render-truth samples of a diverging scheme
ferrum.color.diverging_palette(low, mid, high, n)

# 2026-06-22 (T2.2, ENC-06/XNAME-02/XSIB-07/ENC-11): the palette registry in
# crates/ferrum-core/src/render/palette.rs + .../color/continuous.rs is the
# single source of truth. ferrum.color consumes it through the _core accessors
# (list_palettes / palette_kind / palette_colors / palette_sample); the
# hand-mirrored Python hex tables are gone. Surfaced behavior changes:
#   * color.palette()/sequential()/diverging() for the 7 colorous-backed
#     schemes (viridis, plasma, magma, inferno, cividis, blues, rdbu) now
#     return RENDER-TRUTH colors (what actually renders), replacing the old
#     hand-picked 7-stop approximations. The 19 other palettes are unchanged.
#   * scheme= on Color/Fill/Stroke is validated at declaration time against the
#     registry (a bogus name raises ValueError immediately, not at render).
#   * to_hex(color, scale=) takes an explicit "unit"/"byte" override; the
#     default (scale=None) heuristic is now range-based (any component > 1 means
#     byte), so (1,0,0) and (1.0,0.0,0.0) agree. Pass scale="byte" for the old
#     all-integers-are-bytes interpretation of ambiguous <=1 tuples.
#   * scheme= is the canonical colormap kwarg on mark_raster/mark_contour/
#     mark_hex/clustermap; cmap= remains a documented back-compat alias.

ferrum.config.set_max_rows(n)              # raise/lower data size guard (default: None)
ferrum.config.set_renderer(renderer)       # default renderer for .show()
ferrum.config.set_default_width(n)
ferrum.config.set_default_height(n)

ferrum.config.set_raster_threshold(n)      # mark count threshold for auto-raster
                                           # substitution. Pass None to disable.
                                           # Default: 500_000.
ferrum.config.set_raster_behavior(mode)    # "warn" (default), "silent", "error"
ferrum.config.set_default_backend(backend) # "svg", "tiny-skia", "wgpu", or None (auto)
ferrum.config.set_font_paths(*paths)       # prepend paths to the font search list
                                           # used by the tiny-skia backend

ferrum.Title(text, *, subtitle=None, anchor="start", offset=None,
             font_size=None, font_weight=None, color=None,
             subtitle_font_size=None, subtitle_color=None)

ferrum.Grid(major=True, minor=False, *,
            color=None, width=None, dash=None, opacity=None,
            major_color=None, minor_color=None,
            major_dash=None, minor_dash=None,
            major_width=None, minor_width=None,
            major_opacity=None, minor_opacity=None)
# 2026-05-30 (item 18): the bare color=/width=/dash=/opacity= shorthand
# (shown in the Theme example above) is a FALLBACK that sets BOTH the major
# and minor level; an explicit major_*/minor_* overrides that level. minor
# gridlines render on continuous axes only (linear/log/time/pow/sqrt/symlog);
# categorical/discretizing axes have no continuum to subdivide, so minor=True
# is a documented no-op there. Major ticks/gridlines are scale-projected (a
# tick at value v coincides with a data mark at v); minors subdivide between
# them on the same grid.

ferrum.Aggregate(fn, field, *, as_=None)   # used in transform_aggregate
ferrum.WindowTransform(fn, *, field=None, param=None, as_=None, peer=False)
```

---

---

## Part IV: Extension Points

### Custom Marks

Implement the `MarkProtocol`:

```
class MyMark:
    def to_primitive_layers(self, data: ArrowTable, encodings: Encodings) -> list[PrimitiveLayer]:
        ...
```

Register: `ferrum.register_mark("mark_my_mark", MyMark)`

### Custom Stat Transforms

Implement `StatProtocol`:

```
class MyStat:
    def apply(self, data: ArrowTable) -> ArrowTable:
        ...
```

Register: `ferrum.register_stat(MyStat)`

### Custom Themes

Themes are plain dataclass instances. No registration needed — pass directly to `.theme()`.

### Renderer Plugins

Implement `RendererProtocol` and register via `ferrum.register_renderer(name, RendererClass)`. The built-in renderers (`"svg"`, `"png"`, `"html"`, `"json"`) cannot be overridden, only supplemented.

---

---

## Appendix: Version Target

This document describes **Ferrum 1.0 API surface**. Nothing here exists yet.

The intent is that 1.0 be complete enough that a practitioner moving from Altair, Seaborn, or Yellowbrick finds every chart type they reach for without falling back to matplotlib.

Post-1.0 candidates (explicitly out of scope for 1.0):

- `mark_network` / graph layout
- `mark_gantt` / timeline
- Geographic tile layers (Mapbox, OpenStreetMap)
- 3D coordinate system
- Animation / `frame` encoding
- Real-time streaming data sources
- Julia and R bindings
