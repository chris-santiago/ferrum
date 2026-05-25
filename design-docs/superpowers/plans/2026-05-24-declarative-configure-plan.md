# Declarative Configuration Surface — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Add a composable declarative configuration surface to ferrum: six typed config objects with `.configure_*()` sugar, ~20 format presets, 8 annotation primitives in 3 coordinate systems, 3 structural features (SecondaryY, BreakAxis, Inset), an override escape hatch, and comprehensive documentation.

## 2. Spec references

- `design-docs/superpowers/specs/2026-05-24-declarative-configure-design.md` — full spec (all sections)
- `ferrum-spec.md §3.7` — existing Axis/Legend contract
- `ferrum-spec.md §3.13` — existing Theme contract
- `design-docs/superpowers/specs/2026-05-11-themes-overhaul-design.md` — Theme cascade context

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Create | `src/ferrum/configure.py` | Six config dataclasses (AxisConfig, LegendConfig, TitleConfig, GridConfig, PaddingConfig, ColorConfig) + Configure container |
| Create | `src/ferrum/format_presets.py` | Preset name → d3-format string mapping + resolution logic |
| Create | `src/ferrum/annotation/` | Package: `__init__.py`, `primitives.py` (8 primitives), `coords.py` (px/norm wrappers), `container.py` (Annotate) |
| Create | `src/ferrum/structural.py` | SecondaryY, BreakAxis, Inset frozen dataclasses |
| Modify | `src/ferrum/chart.py` | `.configure_*()` methods, `.override()`, extend `__add__` dispatch |
| Modify | `src/ferrum/annotations.py` | Re-export new `fm.annotation.*` namespace; preserve existing `annotate_hline`/`annotate_vline` |
| Modify | `src/ferrum/__init__.py` | Export new public API: config objects, annotation, structural, px, norm |
| Modify | `src/ferrum/_render.py` | Thread config into render pipeline; resolve cascade |
| Modify | `crates/ferrum-core/src/render/binding.rs` | Accept config dict, annotation specs, structural specs from Python |
| Modify | `crates/ferrum-core/src/layout/axis.rs` | Consume AxisConfig (label angle, format, domain bounds) |
| Modify | `crates/ferrum-core/src/layout/legend.rs` | Consume LegendConfig |
| Modify | `crates/ferrum-core/src/layout/mod.rs` | Padding auto-expansion for annotations; grid band fills |
| Create | `crates/ferrum-core/src/render/annotation.rs` | Annotation scene-graph node construction + auto-placement |
| Create | `crates/ferrum-core/src/render/secondary_axis.rs` | Y2 scale creation + right-side axis rendering |
| Create | `crates/ferrum-core/src/render/break_axis.rs` | Scale splitting, mark clipping, break indicators |
| Create | `crates/ferrum-core/src/render/inset.rs` | Sub-chart embedding at bounds |
| Modify | `crates/ferrum-core/src/render/format.rs` | Ordinal format preset (Rust-side custom logic) |
| Test | `tests/test_configure.py` | Config objects, `.configure_*()` methods, cascade precedence |
| Test | `tests/test_format_presets.py` | All ~20 presets → d3-format resolution |
| Test | `tests/test_annotation_layer.py` | 8 primitives, coordinate systems, z-ordering, composition |
| Test | `tests/test_structural.py` | SecondaryY, BreakAxis, Inset rendering + composition |
| Test | `tests/test_override.py` | Valid paths apply, unknown paths error, deprecation warnings |
| Create | `docs/site/guide/customizing-charts.md` | Conceptual guide: cascade, theme vs configure, composition |
| Create | `docs/site/guide/concepts/configuration.md` | AxisConfig–ColorConfig reference with examples |
| Create | `docs/site/guide/concepts/format-presets.md` | Preset table + usage |
| Create | `docs/site/guide/concepts/annotations.md` | 8 primitives, coords, z-order |
| Create | `docs/site/guide/concepts/secondary-axes.md` | Dual-axis usage |
| Create | `docs/site/guide/concepts/break-axes.md` | Discontinuous axes |
| Create | `docs/site/guide/concepts/inset-panels.md` | Inset embedding |
| Create | `docs/site/guide/concepts/override.md` | Escape hatch docs |
| Create | `docs/site/recipes/customization/` | 12 recipe scripts (see spec §8 doc plan) |

## 4. Constraints

- All config/annotation/structural objects must be frozen dataclasses; `+` and `.configure_*()` return new Chart — never mutate
- No callables cross FFI — format presets resolve to d3-format strings in Python before Rust
- Cascade order is strict: override > per-channel > configure > theme > default theme > Rust defaults
- No matplotlib dependency (hard constraint)
- Existing `annotate_hline`/`annotate_vline` API preserved — new annotation system is additive
- Existing `__add__` for `Chart + Chart` layering must not regress
- `cargo test` must pass after each Rust task
- Goldens visually inspected via `snapshot-goldens.py` before commit
- Ordinal format preset requires Rust implementation; all others are Python→d3 mapping only

## 5. Tasks

### Task 1: Config objects + format presets (Python)
- [ ] Create `src/ferrum/configure.py` with 6 frozen dataclasses + `Configure` container (spec §6)
- [ ] Create `src/ferrum/format_presets.py` with preset→d3 mapping table and `resolve_format()` function
- [ ] Create `src/ferrum/annotation/` package with coordinate wrappers (`px`, `norm`), 8 primitives, `Annotate` container
- [ ] Create `src/ferrum/structural.py` with `SecondaryY`, `BreakAxis`, `Inset` frozen dataclasses
- [ ] Update `src/ferrum/__init__.py` to export all new public names
- [ ] Verify: `uv run pytest tests/test_configure.py tests/test_format_presets.py -v`

### Task 2: Chart integration (Python)
- [ ] Add `.configure_axis()`, `.configure_legend()`, `.configure_title()`, `.configure_grid()`, `.configure_padding()`, `.configure_color()`, `.configure()` to `Chart`
- [ ] Add `.override()` to `Chart` — store overrides, validate at render time
- [ ] Extend `Chart.__add__` to dispatch on `Configure`, `Annotate`, annotation primitives, `SecondaryY`, `BreakAxis`, `Inset`
- [ ] Wire cascade resolution in `_render.py`: merge config layers in precedence order before passing to Rust
- [ ] Re-export `fm.annotation.*` namespace from `src/ferrum/annotations.py`; preserve existing `annotate_hline`/`annotate_vline`
- [ ] Verify: `uv run pytest tests/test_configure.py tests/test_override.py -v`

### Task 3: Rust — config consumption + format (Rust)
- [ ] Extend `binding.rs` to accept config dict from Python; thread into layout
- [ ] Update `axis.rs` to consume AxisConfig fields (label angle, format, domain bounds, tick overrides)
- [ ] Update `legend.rs` to consume LegendConfig fields
- [ ] Update `layout/mod.rs` for PaddingConfig auto-expansion and GridConfig band fills
- [ ] Add ordinal format to `format.rs` (1st, 2nd, 3rd, etc.)
- [ ] Verify: `cargo test` + `uv run pytest tests/test_configure.py -v`

### Task 4: Rust — annotation rendering (Rust)
- [ ] Create `render/annotation.rs`: scene-graph node construction for all 8 primitives
- [ ] Implement coordinate resolution (data→pixel, norm→pixel, px passthrough) using computed plot-area rect
- [ ] Implement callout auto-placement heuristic (quadrant-based; uses existing label-collision system)
- [ ] Implement z-ordering (below_marks / above_marks / above_axis)
- [ ] Implement margin expansion for annotations near edges
- [ ] Verify: `cargo test` + `uv run pytest tests/test_annotation_layer.py -v`

### Task 5: Rust — structural features (Rust)
- [ ] Create `render/secondary_axis.rs`: independent Y2 scale + right-side axis
- [ ] Create `render/break_axis.rs`: scale splitting, mark clipping into segments, break indicator rendering (slash/zigzag/wave/gap)
- [ ] Create `render/inset.rs`: sub-chart embedding with independent scales, border/shadow/connect rendering
- [ ] Wire all three into `binding.rs` and the render pipeline
- [ ] Verify: `cargo test` + `uv run pytest tests/test_structural.py -v`

### Task 6: Golden tests + visual verification
- [ ] Create golden SVGs: one per annotation primitive, one per structural feature, one combined chart
- [ ] Run `python scripts/snapshot-goldens.py` to rasterize all new goldens
- [ ] Visually inspect each PNG and confirm correct rendering
- [ ] Verify: `uv run pytest -n auto` (full suite, no regressions)

### Task 7: Documentation
- [ ] Write `docs/site/guide/customizing-charts.md` — conceptual guide (cascade diagram, theme vs configure, `+` composition, coordinate systems)
- [ ] Write 7 concept pages in `docs/site/guide/concepts/` (spec §8 documentation plan)
- [ ] Write 12 recipe scripts in `docs/site/recipes/customization/` (spec §8 recipe list)
- [ ] Add matplotlib migration table to customizing-charts guide
- [ ] Verify: `nox -s docs` passes

## 6. Acceptance checks

- `uv run pytest -n auto` — all pass (including new + existing)
- `cargo test` — all pass
- All new goldens rasterized and visually confirmed
- Format presets: unit tests verify all ~20 presets produce correct d3-format strings
- Cascade: test with conflicting config at all 6 levels; highest-precedence wins
- Override: unknown path raises `FerrumOverrideError`; valid path applies
- Backward compat: `annotate_hline`, `Chart + Chart` layering, existing `.theme()` unchanged
- `nox -s docs` — no warnings

## 7. Open questions

- **Auto-placement algorithm**: Quadrant-based (simpler, predictable) vs. force-directed (higher quality, more Rust complexity). Recommend quadrant-based for initial implementation; can upgrade later.
- **Ordinal format**: Confirm worth the Rust complexity for "1st, 2nd, 3rd" vs. deferring. English-only initially acceptable?
- **BreakAxis + SecondaryY**: Confirm independent scales (break on primary Y does not affect Y2).
