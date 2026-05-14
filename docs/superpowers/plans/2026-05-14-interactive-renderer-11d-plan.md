# Phase 11d — Coordinate Systems + Deferred Marks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Deliver all four coordinate systems (CoordCartesian, CoordFixed, CoordPolar, CoordGeo), three deferred marks (mark_arc, mark_label, mark_geoshape), and coord-awareness for mark_image. Zero `NotImplementedError`s for coordinate classes and zero `deferred_mark_error` calls for arc/label/geoshape after this phase.

## 2. Spec references

- `docs/superpowers/specs/2026-05-13-interactive-renderer-design.md` §7 — Coordinate systems (Cartesian, Fixed, Polar, Geo)
- Same spec §7.3 — Theta/Radius channel strategy: `CoordPolar(theta="x")` reinterprets x encoding as angular; Rust never sees theta/radius channels, only x/y reinterpreted by coord
- Same spec §10.6 — Python coord classes (frozen dataclasses, `_to_spec_dict()`)
- Same spec §12.4 — Testing requirements for coords and deferred marks

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Create | `crates/ferrum-core/src/projection.rs` | Pure-Rust map projection math: 6 projections (free functions, not methods — orphan rule) |
| Create | `crates/ferrum-core/src/render/marks/arc.rs` | mark_arc: pie/donut wedge geometry via Path+ArcTo |
| Create | `crates/ferrum-core/src/render/marks/geoshape.rs` | mark_geoshape: GeoJSON → projected Polygon nodes |
| Create | `crates/ferrum-core/src/render/marks/label.rs` | mark_label: positioned text + optional leader lines |
| Create | `tests/test_phase_11d/` | All coord and deferred mark tests + goldens |
| Modify | `crates/ferrum-core/src/spec/coord.rs` | Extend `CoordKind` from 2 bare variants to 5: Cartesian (with fields), Flip (bare, unchanged), Fixed, Polar, Geo |
| Modify | `crates/ferrum-core/src/spec/chart.rs` | `coord` param: `Option<&str>` → `Option<&Bound<'_, PyAny>>`; no `geojson_geometries` field (geometry travels as `__geometry__` column in RecordBatch) |
| Modify | `crates/ferrum-core/src/spec/mark.rs` | Add Arc, Geoshape, Label to Mark enum + `for_each_mark!` |
| Modify | `crates/ferrum-core/src/render/marks/mod.rs` | Register arc, geoshape, label modules |
| Modify | `crates/ferrum-core/src/render/scene_build.rs` | Spec→scene coord conversion; polar/geo code paths; suppress axes for Geo |
| Modify | `crates/ferrum-core/src/render/scale_resolve.rs` | xlim/ylim domain overrides; expand flag; polar scale normalization |
| Modify | `crates/ferrum-core/src/layout/mod.rs` | CoordFixed aspect-ratio constraint |
| Modify | `crates/ferrum-core/Cargo.toml` | Add `geojson = "0.24"` |
| Modify | `src/ferrum/coord.py` | Replace NotImplementedError stubs with frozen dataclasses per spec §10.6 |
| Modify | `src/ferrum/chart.py` | Wire all coord types; remove polar channel gate (lines ~4388-4398); wire mark_arc/geoshape/label; theta/radius→x/y remapping in `to_spec()` |
| Modify | `src/ferrum/marks/deferred.py` | Remove arc, geoshape, label, image from PHASE_9_PLUS_MARKS |
| Modify | `src/ferrum/_coerce.py` | GeoJSON FeatureCollection detection: split `features[*].properties`→DataFrame columns, `features[*].geometry`→`__geometry__` string column in the same RecordBatch |
| Modify | `src/ferrum/__init__.py` | Export CoordCartesian, CoordFixed, CoordPolar, CoordGeo |

## 4. Constraints

- **CoordKind in ferrum-scene is NOT modified.** The scene-side `CoordKind` in `ferrum-scene/src/types.rs` is already fully field-carrying (Cartesian/Fixed/Polar/Geo) — extend the spec-side to match it. `MarkBatchKind` in ferrum-scene is also NOT modified: `Label` reuses `MarkBatchKind::Text`, `Geoshape` reuses `MarkBatchKind::Polygon`.
- **Backward compat:** Old JSON `{"kind":"cartesian"}` must round-trip with all fields defaulted. Old `Some(CoordKind::Cartesian)` match sites → `Some(CoordKind::Cartesian { .. })`.
- **String coord path preserved:** `ChartSpec(..., coord="flip")` must still work (back-compat for existing tests).
- **Projection functions are free functions**, not methods on `GeoProjection` — orphan rule forbids inherent impls on foreign types.
- **Arc marks only make sense in CoordPolar.** Return empty result without error if coord is not Polar.
- **`inner_radius`** (donut) flows through `mark_style` kwargs, NOT through `CoordPolar` — the coord carries only theta/start/direction.
- **Polar theta scale:** normalized to [0,1] in `scale_resolve.rs`; the arc mark builder multiplies by 2π itself.
- **mark_label collision avoidance (dodging) is out of scope for 11d** — basic offset positioning only.
- **mark_image is inherently Cartesian.** Return empty result for Polar/Geo coords.
- **Geoshape rendering is two-pass:** first pass computes projected bounding extent across all geometries, second pass projects and scales to pixel space.
- **Polygon exterior ring only** for 11d; interior rings (holes) skipped.
- **GeoJSON geometry storage:** `__geometry__` column in RecordBatch (not a ChartSpec sidecar field). `_coerce.py` adds it during FeatureCollection detection. `geoshape.rs` reads it as a string column. This preserves the one-batch-per-panel invariant and ensures facet filtering trims geometries correctly.
- **Flip → scene-Cartesian:** `CoordKind::Flip` in the spec must have an explicit arm in `scene_build.rs` that emits `CoordKind::Cartesian { x_domain: None, y_domain: None, expand: true, clip: true }`. Python handles channel swapping before `to_spec()` — Rust never sees the flip.
- **Newton-Raphson convergence (EqualEarth/NaturalEarth inverse):** tolerance `1e-12`, max 20 iterations. AlbersUsa round-trip test tolerance `1e-4` (conic inset boundaries accumulate error); Mercator/Equirectangular/Orthographic round-trip tolerance `1e-10`.
- All existing golden SVGs must pass unchanged.

## 5. Tasks

### Task 11d0: Coord serialization plumbing (critical path — all others depend on this)
- [ ] Extend spec-side `CoordKind` to 5 variants: `Cartesian` (add fields matching scene-side), `Flip` (keep bare), `Fixed`, `Polar`, `Geo` — all with serde defaults so old `{"kind":"cartesian"}` JSON round-trips
- [ ] Change PyO3 `coord` param to accept dicts via `pyo3_serde::from_py`; preserve string back-compat
- [ ] Replace Python coord stubs with frozen dataclasses + `_to_spec_dict()` per spec §10.6
- [ ] Wire `Chart.coord()` for all types; add theta/radius→x/y remapping in `to_spec()`; remove polar channel gate
- [ ] Export new coord classes; clean up deferred marks list
- [ ] Verify: `maturin develop`, coord round-trip smoke test, `cargo test`, `uv run pytest`

### Task 11d1: CoordCartesian (xlim/ylim, expand, clip)
- [ ] Read coord domain overrides in `scale_resolve.rs`; implement `override_scale_domain` + `strip_scale_padding`
- [ ] Convert spec→scene coord in `scene_build.rs`; handle clip=false (expand clip rect)
- [ ] Verify: golden tests for xlim, expand=False, clip=False

### Task 11d2: CoordFixed (aspect ratio constraint)
- [ ] Implement aspect-ratio constraint in `compute_layout()` — shrink binding dimension, center
- [ ] Verify: golden test for ratio=1.0 (square panel); Rust unit test

### Task 11d3: CoordPolar + mark_arc (pie/donut)
- [ ] Add Arc to Mark enum + `for_each_mark!`
- [ ] Create `arc.rs`: wedge Path geometry (inner_radius from mark_style, angles from theta scale)
- [ ] Polar scale handling in `scale_resolve.rs`: normalize theta scale to [0,1]
- [ ] Polar axis rendering in `scene_build.rs` (circular axis, radial tick marks)
- [ ] Wire `mark_arc` in Python
- [ ] Verify: golden tests for pie, donut, polar point

### Task 11d4: mark_label (positioned text + leader lines)
- [ ] Add Label to Mark enum; create `label.rs`: offset text + optional leader line
- [ ] Wire `mark_label` in Python
- [ ] Verify: golden tests for basic labels and leader-line labels

### Task 11d5: CoordGeo + mark_geoshape (projections + GeoJSON)
- [ ] Create `projection.rs`: 6 forward/inverse free functions (Mercator, Equirectangular, EqualEarth, NaturalEarth, Orthographic, AlbersUsa)
- [ ] Add `geojson = "0.24"` dep; add Geoshape to Mark enum; create `geoshape.rs`: read `__geometry__` string column from RecordBatch, deserialize GeoJSON, project, emit `Polygon` nodes (`MarkBatchKind::Polygon`)
- [ ] Implement GeoJSON FeatureCollection detection in `_coerce.py`: split properties→DataFrame columns, geometry→`__geometry__` string column; no ChartSpec sidecar field needed
- [ ] Suppress axes for Geo coord in `scene_build.rs`
- [ ] Verify: projection round-trip Rust tests (1e-10 tolerance, AlbersUsa 1e-4/1e-6); golden tests for mercator + equal_earth

### Task 11d6: mark_image coord-awareness
- [ ] Add validation in `image.rs`: return empty for Polar/Geo coords
- [ ] Remove deferred error for mark_image in Python
- [ ] Verify: existing raster/image tests still pass

## 6. Acceptance checks

- `unset CONDA_PREFIX && uv run --no-sync maturin develop` — builds clean
- `DYLD_LIBRARY_PATH=... cargo test` — all Rust tests pass including projection round-trips
- `uv run pytest tests/ -x --timeout=120` — all tests pass
- All existing golden SVGs byte-identical (no regression)
- New golden SVGs rasterized via `snapshot-goldens.py` and visually inspected: coord_cartesian_xlim, coord_fixed_ratio1, polar_pie, polar_donut, mark_label_basic, geo_mercator, geo_equal_earth
- Zero `NotImplementedError` for coords; zero `deferred_mark_error` for arc/label/geoshape/image
- `ChartSpec(..., coord='flip')` still works (string back-compat)

## 7. Deferred gaps from 11c to address in 11d

| Item | Notes |
|---|---|
| `requestAnimationFrame` transition loop (§6.8) | Rust lerp/ease implemented. JS animation loop + GPU buffer re-upload per frame needed. |
| Enter/exit fade for unmatched keys (§6.8) | Key diffing identifies matched pairs. Unmatched old/new keys need opacity-fade transitions. |
| `interaction_config` traitlet (§11.2) | `InteractiveChart` syncs scene_json and selection_state but not interaction_config. |
| Recomputation flow on zoom (§6.8 anywidget) | Python-side callable re-evaluation over new domain, partial SceneGraph rebuild. |
| `Raw` node rendering (§3.4 divergence #4) | Legend colorbar gradients skip with console.warn. Needs typed gradient or DOM SVG overlay. |
| `CoordFixed` uniform scale constraint on zoom (§6.5) | `zoom_pan.rs` supports arbitrary zoom; CoordFixed panels should constrain sx=sy. |
| `Chart.conditional()` convenience method (§10.4) | Primary `.when().otherwise()` path works. `.conditional()` is sugar. |

## 9. Intentional gaps from 11d (→ 11e)

| Item | Spec reference | Notes |
|---|---|---|
| Polar transform for non-arc marks | §7.3 | `mark_point` / `mark_line` in polar would use Cartesian x/y scales rather than angle/radius mapping. Only `mark_arc` has full polar support. |
| Polar axis rendering | §7.3 | Circular angular axis + radial tick marks are not emitted. |
| Per-slice color from color encoding (mark_arc) | §7.3 | All arc slices use the same mark_style fill; `color="category"` encoding does not produce per-slice colors. Needs color-scale lookup in `arc.rs`. |
| CoordFixed uniform scale constraint on zoom | §7.2, §6.5 | `zoom_pan.rs` supports arbitrary zoom; CoordFixed panels should constrain sx=sy. Carried from 11c. |
| Interactive zoom recomputation with xlim/ylim | §7.1 | CoordCartesian/CoordFixed are passive (Python sets bounds at spec time). Sending new bounds from WASM on zoom requires the 11e anywidget round-trip flow. |
| Interactive polar hit-testing | §7.3 | WASM `hit_test` uses Cartesian geometry; inverse polar transform (`atan2`, sqrt) not implemented. |
| `GeoProjection.forward()` / `.inverse()` as enum methods | §7.4 | Spec says "methods on GeoProjection" but orphan rule forbids implementing on foreign types. Used free functions in `projection.rs` instead — behavior identical. |
| `inverse()` gated to `#[cfg(test)]` | §7.4 | `inverse` is needed for interactive geo hit-testing (Phase 11e). Gated under test for 11d to avoid dead-code clippy. Un-gate when 11e wires it. |
| Golden SVGs with visual inspection | §12.4 | Smoke tests confirm SVG renders without error. Pixel-level goldens (`coord_cartesian_xlim.svg`, `polar_pie.svg`, `geo_mercator.svg`, etc.) were not generated. |
| GeoJSON GeometryCollection / Geometry input (non-FeatureCollection) | §7.4 | `_coerce.py` detects FeatureCollection only. Single Geometry or GeometryCollection root is not coerced. |

## 8. Open questions (all resolved)

- **MarkBatchKind for Label/Geoshape:** RESOLVED — `MarkBatchKind::Arc` already exists in ferrum-scene. `Label` reuses `MarkBatchKind::Text`; `Geoshape` reuses `MarkBatchKind::Polygon`. No ferrum-scene MarkBatchKind changes needed.
- **Newton-Raphson convergence:** RESOLVED — tolerance `1e-12`, max 20 iterations for EqualEarth/NaturalEarth inverse. Test tolerances: AlbersUsa `1e-4` (conic inset error), all others `1e-10`.
- **GeoJSON geometry storage:** RESOLVED — `__geometry__` column in RecordBatch (geometry-as-column). See §4 Constraints for rationale.

### Intentional divergences from spec §3 (required for byte-identical golden SVGs)

The spec's type definitions assumed a clean WASM-first design. The actual
implementation needed adjustments so the SVG walker (`svg_walk.rs`) could
reproduce the *exact* byte output of the old `render_svg` path. All changes
are additive — no spec fields were removed or renamed.

| # | Type | Spec says | Implementation has | Reason |
|---|---|---|---|---|
| 1 | `SceneGraph` | `decorations: Vec<SceneNode>` | `title: Vec<SceneNode>`, `legend: Vec<SceneNode>`, `decorations: Vec<SceneNode>` | Old `render_svg` emits title → panels → legend in that order. A single `decorations` vec loses this ordering, producing different SVG. |
| 2 | `Panel.strip_title` | `Option<SceneNode>` | `Vec<SceneNode>` | Strip title is 2 nodes (background rect + text). `Option<SceneNode>` forces a `Group` wrapper → extra `<g>` in SVG not present in old output. |
| 3 | `MarkBatch` | no cap/join fields | `stroke_cap: Option<StrokeCap>`, `stroke_join: Option<StrokeJoin>` | `mark_line` and `mark_area` wrap output in `<g stroke-linecap="..." stroke-linejoin="...">`. This is a batch-level attribute, not per-node. |
| 4 | `SceneNode` | 7 variants (Rect, Circle, Line, Path, Text, Image, Polygon) | +3 variants: `Polyline`, `Group`, `Raw` | `Polyline`: old `mark_line` emits `<polyline>` for linear interpolation, not `<path>`. `Group`: needed for `<g>` attribute wrappers. `Raw`: legend colorbar gradient `<defs>` can't be expressed as typed nodes (`fill="url(#...)"` is not a `Color`). |
| 5 | `FontWeight` | `Normal`, `Bold` | + `Custom(String)` | Themes use numeric CSS weights like `"600"` for axis titles. |
| 6 | `TextBaseline` | `Top`, `Middle`, `Bottom`, `Alphabetic` | + `Custom(String)` | `mark_text(baseline="top")` passes the user-facing string verbatim to SVG `dominant-baseline`; `"top"` ≠ `"hanging"` (the SVG-canonical name). |
| 7 | `PathCmd` | `MoveTo`, `LineTo`, `QuadTo`, `CubicTo`, `ArcTo`, `Close` | + `HLineTo`, `VLineTo` | Step interpolation in `mark_line` emits `H`/`V` SVG path commands. |
| 8 | `PathCmd` field style | positional tuples: `MoveTo(f64, f64)` | named fields: `MoveTo { x: f64, y: f64 }` | serde `#[serde(tag = "op")]` requires struct variants, not tuple variants. |
| 9 | `StrokeStyle` | `color`, `width`, `opacity`, `dash` | + `stroke_cap: Option<StrokeCap>`, `stroke_join: Option<StrokeJoin>` | Needed on `Polyline` nodes so the SVG walker can detect and emit the `<g>` wrapper. (Plan §"Type gaps" identified this pre-implementation.) |
| 10 | `TextStyle` | no `font_family` | + `font_family: String` | Every SVG `<text>` needs a `font-family` attribute. (Plan §"Type gaps" identified this pre-implementation.) |
