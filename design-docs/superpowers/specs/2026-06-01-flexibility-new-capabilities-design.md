# Flexibility New Capabilities & Polish Design Spec

> Phase B of the 0.14.0 flexibility-audit remediation. Source: `/tmp/ferrum-ux-audit/SYNTHESIS.md` (defects D6–D10). Builds on Phase A (`2026-06-01-flexibility-core-fixes-design.md`).

## 1. Scope

Five additions that lift ferrum's expressive ceiling and polish its output: a reactive parameter system (D6), polar second-extent channels (D7), per-series `detail` grouping on `mark_line` (D8), title hygiene for transform-derived columns (D9), and a figure-level title/caption slot (D10). The center of gravity is D6 — a genuinely new subsystem — with four smaller additive surfaces around it. Where Phase A made declared behavior actually happen, Phase B adds capability that does not exist today.

## 2. Goals

- A **reactive parameter** can be referenced anywhere a value, scale domain, or filter predicate is expected. The interactive runtime updates dependents live; the static renderer resolves each parameter to its initial value. This unblocks true crossfilter, overview+detail reactive rescale, legend toggling, and widget-bound values.
- Polar **`theta2`/`radius2`** second-extent channels make radial bars stack outward and make wind rose, Nightingale coxcomb, and hand-built sunburst/icicle expressible.
- `mark_line` with a **`detail`** channel renders one polyline per distinct detail value, with no color legend — enabling hand-built parallel coordinates and multi-series lines.
- Axis and legend **titles** derive from the encoding's source field or an explicit title, never from an internal transform output column.
- Composite charts carry a **figure-level title/subtitle/caption** rendered once around the composed panels.

## 3. Non-goals

- Composite-mark helpers (`fm.sunburst`, `fm.wind_rose`): generic channels only this phase. Hierarchy rectangling is user-supplied or precomputed.
- Variable-width `mark_trail` (the Minard ribbon workaround already covers tapering flows).
- Gridded `contourf`, half-violin, and all Phase A correctness fixes.
- A new figure wrapper type — figure chrome reuses `.properties()`.

## 4. System behavior

**Reactive parameters (D6).** A `Parameter` is a named reactive value of one of two kinds: a *selection* (interval or point — today's `selection_interval`/`selection_point` become parameters) or a *variable* (a free value, optionally bound to an input control). A parameter may be referenced as: an encoding `value=param`, a `scale` domain (`X("t", scale={"domain": param})`), a `transform_filter(param)` predicate, or a conditional test. In an interactive render (`.interactive()` / HTML / WASM), referencing dependents recompute when the parameter changes: dragging an overview brush rescales the detail panel's domain; brushing one panel filters rows from a linked panel (true crossfilter); clicking a legend entry toggles a series. In a static SVG render, every parameter resolves to its declared initial value, so output stays deterministic. `bind="legend"` on a point selection binds it to the legend.

**Polar second extents (D7).** Under `CoordPolar`, `theta`/`theta2` define an angular span and `radius`/`radius2` a radial span. A mark with both extents draws an annular wedge. Stacking under polar offsets `radius`, so stacked radial bars accumulate outward (wind rose, coxcomb) rather than overlapping at r=0. Sunburst/icicle are expressible by binding `theta`/`theta2` and `radius`/`radius2` from pre-laid-out hierarchy rows.

**Line `detail` grouping (D8).** `mark_line().encode(x=, y=, detail="g")` renders one connected polyline per distinct value of `g`, introducing no color legend. With per-axis layout this produces hand-built parallel coordinates; with a single line per group it produces multi-series lines that are not color-encoded.

**Title hygiene (D9).** The default axis/legend title for a channel is its source field name (or explicit `title=`), regardless of any internal column a transform produced. `contour_x`, `hex_x`, `lo`, and similar derived names never surface as titles.

**Figure title/caption (D10).** `composite.properties(title=, subtitle=, caption=)` renders the title and subtitle above, and the caption below, the whole composed figure — once, not per panel. Per-panel titles on child charts remain available and independent.

## 5. Architecture

- **Parameter system** spans three layers. (1) Python: `Parameter`/`Selection` declaration, the ability to pass a parameter into encoding `value`, `scale.domain`, `transform_filter`, and conditionals, and serialization of a `params` section plus reference markers into the spec JSON. (2) Static resolver: at SVG render, parameters resolve to their initial values; no event loop. (3) Interactive runtime (WASM/JS): an evaluator wires input events (brush, click, widget) to parameter updates and re-renders the dependent encodings/scales/filters. Selections unify under `Parameter`; existing selection constructors keep their signatures.
- **Polar channels** add `theta2`/`radius2` channel definitions in the Python encoding layer and second-extent + radial-stack-offset geometry in the Rust polar coordinate path.
- **Line `detail`** is handled in the Rust line batcher by grouping on the detail key, mirroring the existing parallel-coordinates batcher.
- **Title hygiene** lives in render-layer title resolution: read the encoding's declared field/title, not the post-transform column name.
- **Figure chrome** is rendered by the composite (`vconcat`/`hconcat`/`facet`) layout, which reserves a title/caption band around the child panels; children are unchanged.

## 6. Canonical interfaces / data contracts

Parameter / selection (selections become the selection-kind of parameter):

```python
fm.param(name: str, value=None, bind: Bind | str | None = None)   # variable parameter
selection_interval(...) -> Parameter   # selection-kind parameter (existing signature)
selection_point(..., bind: str | None = None) -> Parameter        # bind="legend" toggles series
```

Reference sites (a `Parameter` is accepted wherever these appear):

```python
X("t", scale={"domain": param})     # reactive rescale (interactive) / initial domain (static)
transform_filter(param)             # crossfilter by the parameter's predicate
encode(opacity=fm.when(param).then(1).otherwise(0.2))   # conditional on a parameter
encode(size=fm.value(param))        # value bound to a parameter
```

Polar second-extent channels:

```python
Theta2(field: str)    # second angular extent under CoordPolar
Radius2(field: str)   # second radial extent under CoordPolar
```

Figure chrome:

```python
composite.properties(title: str | Title | None = None,
                     subtitle: str | None = None,
                     caption: str | None = None)
```

Static-render contract: any spec containing parameters renders deterministically by substituting each parameter's initial value; a spec with no parameters serializes and renders byte-identically to today.

## 7. Invariants and constraints

- Goldens affected by polar, line-detail, title, and figure-chrome changes must be re-blessed and visually inspected (project hard constraint); byte-equality is necessary but not sufficient.
- `cargo test` passes before the phase is marked done; full `pytest -n auto` green.
- No global mutable state; no matplotlib.
- **Static determinism:** parameters resolve to initial values in SVG; a chart that uses no parameters produces byte-identical static output to today. The parameter system must not alter non-interactive renders of param-free charts.
- **Backward compatibility:** `selection_interval`/`selection_point` keep their current signatures and existing behavior; they gain referenceability by scales/filters. `CoordPolar` charts that bind only `theta`/`radius` are unchanged. `.properties(title=)` on a single chart keeps its current meaning.
- Polar stacking must not regress Cartesian stacking.

## 8. Key decisions and tradeoffs

- **Broader reactive-parameter layer over a minimal extension (user decision).** Selections unify with variable parameters, and any value/scale/filter/conditional can reference a parameter. Rationale: future-proofs the interactive idioms (crossfilter, rescale, toggle, widgets) under one model rather than three special cases. Cost and risk: a new `params` section in the spec JSON, a runtime evaluator, and the largest single build in either phase — explicitly the riskiest item. Mitigation: static-render determinism keeps the SVG path simple, and existing selections are absorbed rather than replaced.
- **Static render resolves parameters to initial values.** SVG has no event loop; deterministic substitution is the only coherent static semantics and preserves golden stability for param-free charts.
- **Generic polar channels over dedicated marks (user decision).** `theta2`/`radius2` compose; wind rose, coxcomb, and sunburst are assembled through the grammar. Hierarchy rectangling is user-supplied this phase; a layout transform and `fm.sunburst` helper are deferred.
- **Figure chrome via `.properties()` (user decision).** No new public type; the composite layout grows a title/caption band.
- **`detail` fix scoped to grouping; `mark_trail` deferred.** Splitting `mark_line` by detail unblocks parallel coordinates and multi-series lines; variable-width trails are out of scope.
- **Title hygiene: source field or explicit title always wins** over any internal transform column.

## 9. Acceptance criteria

- Overview+detail: brushing the overview rescales the detail panel's domain in the interactive export (verified by the emitted parameter wiring and scene references); the static render shows the initial domain.
- Crossfilter: `transform_filter(selection)` removes rows from the linked panel in the interactive export; `selection_point(bind="legend")` toggles the corresponding series.
- A variable parameter bound to a control reactively drives an encoding/filter in the interactive export; the static render uses its initial value.
- Wind rose / coxcomb: stacked radial bars accumulate outward with no r=0 overlap; a hand-built sunburst renders nested wedges from laid-out hierarchy rows.
- `mark_line(detail="g")` renders N separate polylines with no color legend; hand-built parallel coordinates render across N axes.
- No chart surfaces an internal column name (`contour_x`, `hex_x`, `lo`) as an axis or legend title.
- `vconcat`/`hconcat`/`facet` with `properties(title=, subtitle=, caption=)` renders the chrome once around the figure, not per panel.
- The Phase-B-tagged audit designs (parallel coords + N-axis brushing, overview+detail, crossfilter, legend toggle, wind rose, coxcomb, sunburst) render or verify; param-free charts remain byte-stable.

## 10. Validation strategy

- Per-item regression tests added before each fix (project TDD convention). Interactive items are validated by inspecting the emitted HTML/JS parameter wiring and scene JSON (the method used in the interactive audit, since there is no browser in CI), asserting that the parameter, its references, and its event bindings are present and correct — not merely that an export succeeded.
- Golden regeneration for polar, line-detail, title, and figure-chrome charts, each rasterized to PNG and visually inspected before blessing; confirm param-free goldens are byte-stable.
- Re-run the Phase-B flexibility-audit categories (multivariate, scientific, categorical, interactive, faceting) and confirm the previously-blocked designs now render/verify; diff the regenerated `SYNTHESIS.md`.
- `cargo test` and full `pytest -n auto` green.

## 11. Open questions

- **Widget binding surface.** Which input controls ship this phase? Leaning: legend bind, interval brush, and point selection are in; HTML slider/dropdown widgets are minimal or deferred to a follow-up, since they add UI surface beyond the audit's blocked designs.
- **Sunburst hierarchy layout.** User precomputes the rectangled rows this phase (generic channels only); a Rust partition/rectangling transform and an `fm.sunburst` helper are a later pass. Confirm that hand-built is acceptable for now.
- **Parameter scope.** Parameters are scoped to a single composed figure/spec this phase; cross-spec/multi-view shared parameters are out of scope. Confirm no immediate need for cross-figure linking.
