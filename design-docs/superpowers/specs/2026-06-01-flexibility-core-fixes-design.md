# Flexibility Core Fixes Design Spec

> Phase A of the 0.14.0 flexibility-audit remediation. Source: `/tmp/ferrum-ux-audit/SYNTHESIS.md` (defects D1–D5). Phase B (D6–D10) is a separate spec.

## 1. Scope

Five renderer- and coercion-side defects that silently break charts across the entire grammar: continuous-color scale resolution, the second positional extent (`y2`/`x2`) on bar and area marks, two-way and statistical/composite faceting, integer/nominal dtype handling, and the surfacing of silent failures. Each was independently hit by multiple audit categories; together they account for the majority of the blocked designs. This phase is correctness and coercion work on existing surfaces — no new marks, channels, or coordinate systems.

## 2. Goals

- A continuous color channel bound through `SequentialScale`/`DivergingScale` resolves to the requested palette — every named scheme (viridis, magma, inferno, reds, greens, oranges, purples, rdbu, …) renders its actual colors, and `domain`/`clamp` are honored.
- `mark_bar` and `mark_area` honor a second positional extent (`y2`/`x2`); bar zero-anchoring becomes opt-out so floating, ranged, and diverging bars are expressible.
- `facet(row=, col=)` produces a true 2-D grid with inferred cardinality and row/column headers; faceting a statistical or composite mark re-runs the transform per partition; faceted layers carrying distinct DataFrames are preserved; a faceted `Chart` supports per-channel scale resolution.
- Integer and nominal columns are accepted on positional, color, and stacking inputs wherever quantitative/ordinal columns are, normalized once at the transport boundary — no mark renders blank because of column dtype.
- Previously-silent failures become observable: dropped or unsupported kwargs warn, empty partitions warn, and the windowed-aggregate frame convention matches its documentation.

## 3. Non-goals

- New marks or layouts (sunburst, sankey, half-violin), polar second extents, interactive selection→scale binding, `detail`-split lines, figure super-title/footnote. All are Phase B.
- Any matplotlib dependency (permanent project constraint).
- API redesign. The only additive surface is the bar `zero=` escape and broadened dtype acceptance; everything else is making declared behavior actually happen.

## 4. System behavior

**Continuous color (D1).** Binding `color="z:Q"` with a `SequentialScale(scheme="viridis")` paints marks along the viridis ramp; the colorbar legend shows the same ramp. `DivergingScale(scheme="rdbu", domain=[-1, 0, 1], clamp=True)` centers at the supplied midpoint and clamps out-of-domain values to the endpoints. Non-blue named schemes render their real palette rather than falling back to blue. The mark-level `cmap=` path (`mark_raster`, `mark_hex`) and the color-scale resolver draw from the same palette source, so a scheme name means the same colors in both.

**Second extent on bar/area (D3).** `mark_bar().encode(x=, y=, y2=)` draws a bar spanning `[y, y2]` rather than `[baseline, y]`; the horizontal form honors `x`/`x2`. Bars retain today's zero-anchored default; passing `zero=False` (or binding `y2`) suppresses the zero anchor so the extent is taken literally. `mark_area` fills between `y` and `y2` when `y2` is bound. This makes candlestick bodies, floating/gap bars, and diverging (mixed-sign) stacks expressible without dropping to `mark_rect`.

**Faceting (D2).** `chart.facet(row="a", col="b")` lays out a grid with one panel per `(a, b)` pair, row and column headers, and inferred row/column counts when not given. Faceting a statistical or composite mark (`mark_density`, `mark_histogram`, `mark_boxplot`, …) computes the transform within each panel's partition, so panels show per-partition results rather than rendering blank. A layered chart whose layers carry different DataFrames keeps all layers under faceting. A faceted `Chart` accepts a scale-resolution request per channel (shared vs. independent), and the resolution visibly changes the rendered domains.

**Dtype acceptance (D4).** Integer columns are usable on the same channels as floats; nominal (string) columns are usable on positional channels for every mark that accepts a categorical axis, not only a subset. A categorical heatmap keyed on integer or string columns renders all cells. Where a column's storage dtype previously caused a silent blank or a hard `unsupported dtype` error, it now coerces and renders.

**Silent failures surfaced (D5).** A kwarg that the renderer cannot honor (for example an annotation `label_position`, or `Y(stack="normalize")` on a path that ignores it) emits a warning naming the dropped field rather than silently no-op'ing. An empty render partition emits a warning identifying the partition. The windowed-aggregate `frame` follows the documented Vega convention — `frame=(-k, 0)` means the `k` preceding rows through the current row — so a rolling mean does not silently produce an all-null series.

## 5. Architecture

- **Color resolution** lives in the Rust render-layer color scale resolver. The resolver is the single place a scale spec maps to a concrete palette; `Sequential` and `Diverging` scale specs are resolved there from the same named-palette table the mark-level `cmap` baking already uses. Python continues to serialize scheme/domain/clamp on the scale object unchanged.
- **Bar/area geometry** is owned by the Rust mark batchers for bar and area; the zero-anchor decision currently injected in the Python `Chart` layer becomes conditional on the `zero=`/`y2`-bound state and is passed through as part of the mark spec.
- **Faceting** spans the Python `Chart` facet construction (cardinality inference, grid shape, resolve API) and the Rust render preparation stage (partition keys must include both the row and column fields; per-partition transform execution). Grammar-level `facet` reaches parity with the existing `RepeatChart`/`ConcatChart` paths rather than redirecting to them.
- **Dtype coercion** is centralized at the existing coerce/transport boundary (the narwhals→Arrow ingestion path), normalizing integer and nominal columns once so downstream marks and transforms see a consistent shape. No per-mark dtype special-casing.
- **Diagnostics** are a cross-cutting policy: a small set of warning categories emitted at the points where kwargs are consumed and partitions are built. No new global state; warnings use the standard Python warnings machinery.

## 6. Canonical interfaces / data contracts

Scale objects (existing surface; behavior now honored):

```python
SequentialScale(scheme: str = "blues", domain: tuple[float, float] | None = None, clamp: bool = False)
DivergingScale(scheme: str = "rdbu", domain: tuple[float, float, float] | None = None, clamp: bool = False)
```

Bar zero-anchor escape (additive, default preserves current behavior):

```python
mark_bar(..., zero: bool = True)   # zero=False, or binding y2/x2, takes the extent literally
```

Faceting (existing method; cardinality now inferred, row consumed, resolve added):

```python
Chart.facet(row: str | None = None, col: str | None = None,
            nrows: int | None = None, ncols: int | None = None)
Chart.facet(...).share_scale(x="shared" | "independent", y="shared" | "independent")
```

Coercion rule: a column's *encoding type* (`:Q`/`:O`/`:N`, explicit or inferred) — not its storage dtype — determines channel acceptance. Integer storage defaults to quantitative; an explicit `:O`/`:N` makes it ordinal/nominal. Nominal storage is accepted on any positional channel that accepts a categorical scale.

Window frame convention: `frame=(preceding, following)` with negative `preceding` counting rows before the current row, matching Vega/Altair and the published docstring.

## 7. Invariants and constraints

- **Goldens must be re-blessed and visually inspected.** Any change to `tests/goldens/**` or `tests/test_phase_9_e2e/goldens/*` requires PNG rasterization and human inspection per the project hard constraint; byte-equality alone is insufficient. Color and bar/area changes will move many goldens.
- `cargo test` must pass before the phase is marked done.
- No global mutable state; no matplotlib.
- Backward compatibility: charts that render correctly today must be byte-stable except where they were silently wrong (continuous color now correct, faceted stat panels now populated, etc.). The bar `zero=` default remains `True`. Diagnostics are warnings, not errors, except the window-frame sign, which is a correctness fix.
- The continuous-color fix must not regress the categorical color cycle or the working `mark_raster`/`mark_hex` `cmap` path.

## 8. Key decisions and tradeoffs

- **Fix color in the resolver, not per mark.** The resolver is the single source of truth; adding `Sequential`/`Diverging` arms there fixes every mark at once and keeps `cmap` baking and scale resolution agreeing. Rejected: patching each mark, which would re-fragment palette logic.
- **Zero-anchor becomes opt-out, not removed.** Removing it would regress the common zero-based bar. An additive `zero=False` (implied when `y2` is bound) preserves defaults while unlocking floating/diverging bars.
- **Grammar-level `facet` reaches parity rather than redirecting to `RepeatChart`.** Faceting is core grammar and should be first-class; silently rewriting `.facet()` into a repeat would leak an abstraction and surprise users who layer or resolve on the faceted chart.
- **Coerce once at the transport boundary.** Centralizing dtype normalization matches ferrum's "data crosses the boundary once" architecture and avoids N per-mark dtype branches that drift.
- **Silent → warning, not error.** Converting dropped kwargs and empty partitions to warnings surfaces the problem without breaking existing pipelines. The window-frame sign is the one exception: it is a wrong-result bug, so it is corrected outright (with a changelog note, since any pipeline that compensated with the inverted sign will shift).
- **Diverging midpoint.** A 3-tuple `domain` carries the explicit midpoint; a 2-tuple is allowed with the midpoint inferred as the domain center.
- **Integer dtype defaults to quantitative.** An integer column is treated as continuous unless its encoding type is explicitly `:O`/`:N`. Matches Altair, avoids silently binning continuous data; ordinal/nominal intent is opt-in.

## 9. Acceptance criteria

- A scatter colored by `SequentialScale(scheme="viridis")` renders the viridis ramp (mark fills and colorbar), verified against a known palette, not blue; the same holds for magma, reds, greens, oranges, purples; `DivergingScale(domain=[-1,0,1], clamp=True)` centers and clamps.
- A candlestick built from `mark_bar`/`mark_rect` bodies with `y`/`y2` renders bodies spanning open→close; a diverging Likert stacked bar renders negative segments; a floating gap bar renders off the baseline.
- `facet(row=, col=)` on a 3×3 categorical cross renders 9 populated panels with headers; `mark_density(...).facet(col=)` renders a populated KDE per panel (no blank); a calendar-heatmap-style `mark_rect` on a week×weekday grid renders all cells.
- A faceted `Chart` with `share_scale(y="independent")` shows visibly different per-panel y domains; with `"shared"`, identical domains.
- Integer-keyed and string-keyed categorical heatmaps both render all cells; `mark_bar` accepts a nominal `y`.
- A rolling mean with `frame=(-13, 0)` produces a non-null trailing average; binding an unsupported kwarg emits a warning naming it; an empty partition emits a warning.
- The previously-blocked audit designs in distributions, timeseries, faceting, multivariate, scientific, and categorical that traced to D1–D5 now render correctly on visual inspection.

## 10. Validation strategy

- Per-defect regression tests (Python, and Rust where the fix is crate-side) asserting the observable behavior above, added before the corresponding fix per project TDD convention.
- Golden regeneration for affected charts, each rasterized to PNG and visually inspected before blessing; spot-check that unaffected goldens stay byte-stable.
- Re-run the flexibility-audit categories tied to D1–D5 and confirm the specific designs flagged blocked/buggy now pass; diff the regenerated `SYNTHESIS.md` against this baseline.
- `cargo test` and the full `pytest -n auto` suite green.

## 11. Open questions

None blocking correctness. Resolved at Phase A review: integer dtype → quantitative by default (explicit `:O`/`:N` to opt out); diverging 2-tuple domain → midpoint inferred as center; empty-partition diagnostic → warning only (no in-panel placeholder this phase); window-frame sign → corrected outright with a changelog note (no deprecation shim).
