# Band/Point Explicit Pixel Range — Band-Geometry Unification Design Spec

GH #39, phase 2. Phase 1 (already on `fix/band-point-scale-range`): `ScaleSpec::Band`/`Point`
carry an optional `range`, the pyclass bridges emit it, and the positional resolver honors it
with fallback to the panel extent — so mark *positions* already honor an explicit range.

## 1. Scope

Make the resolved ordinal positional scale the single source of truth for band geometry at
render time. Today, band step/width geometry is independently re-derived from the panel extent
(`panel.w / n_categories`-style arithmetic) by bar/box/heatmap/tick mark builders and by
categorical axis-tick placement in layout. Those re-derivations were exactly equal to the
scale's geometry only under the historical invariant *"band/point pixel range == panel
extent"*, which phase 1 removed. This spec closes the divergence: when a band/point/ordinal
scale carries an explicit pixel range, every consumer of band geometry derives it from the
resolved scale; when no explicit range is set, output is byte-identical to today.

## 2. Goals

- A chart with `BandScale(domain=[...], range=[a, b])` (or `PointScale`, or a positional
  `OrdinalScale` with numeric range) renders **all** category-band geometry within `[a, b]`:
  mark positions (done in phase 1), bar/box widths, heatmap cell sizes, tick-mark extents.
- Categorical axis tick labels align with mark band centers under an explicit range.
- Charts without an explicit range render byte-identical SVG to current `main` + phase 1.
- Dodged sub-band geometry stays consistent with the band geometry it subdivides.

## 3. Non-goals

- Polar/angular band geometry (`tau / n_cats`): angular bands have no pixel range; unaffected.
- `PointScale(reverse=)` / `align=` render-time consumption: these are serialized but never
  consumed by the resolver — a pre-existing, separable silent drop tracked as its own issue.
- Continuous-scale range handling (already correct) and non-positional (color) ordinal ranges.
- Scale-`padding`-aware mark widths: mark width formulas keep their existing shape factors
  (e.g. `* 0.8`); only the extent they scale by changes. Coupling widths to band `padding` /
  `padding_inner` is a separate design question.

## 4. System behavior

- **Explicit range set** (`range=[a, b]` on `BandScale` / `PointScale` / positional
  `OrdinalScale`): categories are laid out by the resolved scale over `[a, b]`. Bars, boxes,
  heatmap cells, and tick marks size themselves from the scale's band extent (`|b − a|`), so
  they nest inside their bands instead of spilling across the panel. Categorical axis ticks
  and grid lines are placed at the scale's band centers, so the axis agrees with the marks.
- **No explicit range**: geometry derives from the panel extent through the *identical
  arithmetic used today* — not a recomputation through the scale that could drift by one ulp
  and flip a printed SVG decimal. Output is byte-identical.
- **Coordinate frame**: an explicit range is in absolute panel pixel coordinates, exactly as
  for continuous scales (which pass the user's `range` through unchanged). In faceted or
  composite charts the same `[a, b]` therefore applies within each panel that resolves the
  scale; ferrum does not rescale it per panel. This mirrors existing continuous behavior and
  is the documented contract, not a bug.
- **Interactive / composite output**: both consume the same resolved scene (single resolver
  path); they inherit this behavior with no separate implementation.
- **Degenerate ranges**: a wire-level range with fewer than 2 entries falls back to the panel
  extent (phase 1 contract). A reversed range `[hi, lo]` is passed through as-is, matching
  continuous scales.

## 5. Architecture

- **Resolver** (`ferrum-core` scale resolution) is the only component that knows whether a
  range was explicit. It records that fact on the resolved ordinal scale at construction.
- **Resolved scale** (`OrdinalScale` behind `ScaleKind`) owns band geometry: range, extent,
  step, bandwidth, and explicitness. It is the single queryable source for all downstream
  band-geometry consumers.
- **Mark builders** (bar, rect/box/heatmap, tick) query the resolved scale for the band
  extent and fall back to panel extent when the scale reports no explicit range. They retain
  their own shape factors (0.8 width ratio, `band_size`, dodge group division).
- **Layout / axis placement** queries the same resolved scale: explicit range → project
  categorical tick centers through the scale's band centers; otherwise → existing
  uniform-center placement over the panel. The y-channel ordinal orientation (no reversal for
  ordinal axes) must produce the same alignment guarantee as x.
- **Dodge** already reads `bandwidth()` from the resolved scale and needs no change — it is
  the in-tree precedent for scale-derived band geometry.

## 6. Canonical interfaces / data contracts

```rust
/// ScaleKind (render-internal). Some(|r1 - r0|-signed extent... see semantics)
/// only when this is an ordinal positional scale whose pixel range was
/// explicitly supplied by the user (via ScaleSpec range); None otherwise.
fn explicit_band_extent(&self) -> Option<f64>;
```

Semantics that bind implementers:

- `explicit_band_extent()` returns `Some(r1 − r0)` (signed, in range order) only when the
  resolver consumed a user-supplied range; the *fallback* range (panel extent) must yield
  `None`, even though it is numerically a valid range. Explicitness is recorded at
  construction by the resolver — consumers must not infer it by comparing floats.
- Consumer contract (marks): `extent = scale.explicit_band_extent().map(f64::abs).unwrap_or(panel_extent)`,
  where `panel_extent` is the exact expression used today (`panel.w` / `panel.h`). No other
  term in the width/size formulas changes.
- Consumer contract (layout): categorical tick label/grid positions under an explicit range
  are the scale's band centers (the same pixels `to_pixel_str(category)` yields for marks);
  without an explicit range, placement is unchanged (`uniform_center`).

## 7. Invariants and constraints

- **Byte-identity without explicit range** — the no-range render path must execute arithmetic
  identical to today's, not merely numerically close. Guarded by the existing golden corpus
  and the frozen wire fixture; no golden may need regeneration.
- **Marks and axis agree** — for any chart where a categorical axis and its marks share a
  resolved scale, tick label centers coincide with mark band centers (within SVG print
  precision), with and without an explicit range.
- **Single resolver path** — no band-geometry decision may be duplicated in Python, WASM, or
  composite code; the resolved scene remains the only transport (Phase B contract).
- **`cargo test` and the full pytest suite pass**; goldens byte-identical (hard constraint,
  CLAUDE.md).
- The `ferrum-spec.md` §scale-range contract ("pixel ranges supplied via `Scale(range=[...])`
  bypass the inset entirely and are treated as the final scale range") now holds for band and
  point scales; no spec text change required.

## 8. Key decisions and tradeoffs

- **Explicitness flag over float comparison** (decided): the resolver records "range was
  user-supplied" on the resolved scale. Rejected: consumers comparing the scale's range
  against the panel extent — the fallback constructs its range *from* the panel values so
  equality would hold today, but it couples correctness to float identity across separately
  computed expressions and breaks silently if either side changes form.
- **Gate on explicitness rather than always deriving from the scale** (decided): always
  deriving (`extent = r1 − r0`) is the cleaner end-state but recomputes today's `panel.w` as
  `(panel.x + panel.w) − panel.x`, which can differ by 1 ulp and flip a rounded SVG decimal
  somewhere in the golden corpus. Byte-identity is a hard constraint; the gate makes it
  structural. North-star follow-up: collapse the gate by making the scale the unconditional
  source once a golden-regeneration window is acceptable.
- **Mark width formulas keep their shape factors** (decided): substituting only the extent
  preserves every existing visual ratio (0.8 bar fill, `band_size` tick fraction, dodge
  subdivision) for both gated and ungated paths. Rejected: switching widths to scale
  `bandwidth()` — it silently couples mark width to band `padding`, changing no-range output
  for explicit `BandScale(padding_inner=...)` users (byte-identity violation).
- **Explicit range is absolute panel pixels, per panel, mirroring continuous scales**
  (decided): no per-facet rescaling. Consistency across scale families beats cleverness.
- **Ordinal `range` on the positional `Ordinal` arm already resolves via
  `ordinal_pixel_range`** — phase 2 extends the *geometry consumers*, so an explicit ordinal
  positional range gains the same mark/axis agreement. Band, point, and ordinal positional
  scales must behave identically here; none is special-cased.

## 9. Acceptance criteria

- `tests/test_regression_band_point_range.py::test_band_scale_range_constrains_bar_positions`
  (currently RED) passes: all bar rects within `[40, 260] ± 0.5`.
- New behavioral checks (Rust or Python, level chosen by the plan), each discriminating
  (fails on today's build):
  - Ordinal-**y** (horizontal bar) with explicit range: bar heights and y-tick label centers
    within/aligned to the range.
  - Heatmap (`mark_rect`, both axes categorical) with an explicit range on one axis: cell
    extent on that axis equals `|range| / n_categories`; other axis unchanged.
  - Tick mark with explicit band range: tick half-extent derives from the range extent.
  - Categorical axis alignment: for an explicit-range band chart, x tick label centers equal
    mark band centers (assert against `to_pixel_str` values or SVG text x positions).
  - Dodged bar chart with explicit range: sub-band bars do not overlap and stay within range.
- Byte-identity: full golden corpus and `tests/test_scale_spec_parity.py::TestByteIdentity`
  pass without regeneration.
- Full `uv run pytest -n auto` and `cargo test` green.

## 10. Validation strategy

- Behavior-level: SVG-geometry assertions (rect x/width, circle cx, text x) comparing marks
  against the requested range and against axis tick positions — the same probe style that
  exposed the divergence.
- Regression-level: the golden corpus as an unmodified-behavior oracle for the ungated path;
  the phase-1 wire fixture for serialization.
- Visual: rasterize at least one explicit-range band chart and one heatmap via
  `scripts/snapshot-goldens.py` helpers and inspect the PNG before close (goldens-not-blessed
  rule applies to any new golden).

## 11. Open questions

None blocking. (Follow-ups tracked outside this spec: PointScale `reverse`/`align` render
drop; north-star ungated scale-derived geometry.)
