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
| Modify | `crates/ferrum-core/src/spec/coord.rs` | Extend `CoordKind` from 2 bare variants to 6 field-carrying variants |
| Modify | `crates/ferrum-core/src/spec/chart.rs` | `coord` param: `Option<&str>` → `Option<&Bound<'_, PyAny>>`; add `geojson_geometries` field |
| Modify | `crates/ferrum-core/src/spec/mark.rs` | Add Arc, Geoshape, Label to Mark enum + `for_each_mark!` |
| Modify | `crates/ferrum-core/src/render/marks/mod.rs` | Register arc, geoshape, label modules |
| Modify | `crates/ferrum-core/src/render/scene_build.rs` | Spec→scene coord conversion; polar/geo code paths; suppress axes for Geo |
| Modify | `crates/ferrum-core/src/render/scale_resolve.rs` | xlim/ylim domain overrides; expand flag; polar scale normalization |
| Modify | `crates/ferrum-core/src/layout/mod.rs` | CoordFixed aspect-ratio constraint |
| Modify | `crates/ferrum-core/Cargo.toml` | Add `geojson = "0.24"` |
| Modify | `src/ferrum/coord.py` | Replace NotImplementedError stubs with frozen dataclasses per spec §10.6 |
| Modify | `src/ferrum/chart.py` | Wire all coord types; remove polar channel gate (lines ~4388-4398); wire mark_arc/geoshape/label; theta/radius→x/y remapping in `to_spec()` |
| Modify | `src/ferrum/marks/deferred.py` | Remove arc, geoshape, label, image from PHASE_9_PLUS_MARKS |
| Modify | `src/ferrum/_coerce.py` | GeoJSON FeatureCollection detection: split properties→DataFrame, geometry→JSON string |
| Modify | `src/ferrum/__init__.py` | Export CoordCartesian, CoordFixed, CoordPolar, CoordGeo |

## 4. Constraints

- **Two CoordKind enums must stay in sync:** spec-side (`ferrum-core/src/spec/coord.rs`) and scene-side (`ferrum-scene/src/types.rs`). Scene-side is NOT modified — spec-side is extended to match.
- **Backward compat:** Old JSON `{"kind":"cartesian"}` must round-trip with all fields defaulted. Old `Some(CoordKind::Cartesian)` match sites → `Some(CoordKind::Cartesian { .. })`.
- **String coord path preserved:** `ChartSpec(..., coord="flip")` must still work (back-compat for existing tests).
- **Projection functions are free functions**, not methods on `GeoProjection` — orphan rule forbids inherent impls on foreign types.
- **Arc marks only make sense in CoordPolar.** Return empty result without error if coord is not Polar.
- **mark_image is inherently Cartesian.** Return empty result for Polar/Geo coords.
- All existing golden SVGs must pass unchanged.

## 5. Tasks

### Task 11d0: Coord serialization plumbing (critical path — all others depend on this)
- [ ] Extend spec-side `CoordKind` to 6 field-carrying variants with serde defaults per spec §7
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
- [ ] Add `geojson = "0.24"` dep; add Geoshape to Mark enum; create `geoshape.rs`: deserialize GeoJSON, project, emit Polygon nodes
- [ ] Add `geojson_geometries` field to ChartSpec; implement GeoJSON detection in `_coerce.py`
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

## 7. Open questions

- Does `MarkBatchKind` in ferrum-scene need an `Arc` and/or `Label` variant, or should they reuse `Polygon`/`Text`? Check existing variants before deciding.
- EqualEarth and NaturalEarth inverse projections require Newton-Raphson iteration — what convergence tolerance and max iterations?
