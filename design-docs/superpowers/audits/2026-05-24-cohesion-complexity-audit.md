# Ferrum Codebase Audit: Cohesion & Complexity

**Date:** 2026-05-24
**Scope:** Full read-only audit of `src/ferrum/` (Python) and `crates/ferrum-core/` (Rust)
**Branch:** `feat/declarative-configure`

---

## Executive Summary

**Python:** The package-level organization is sound — marks, encoding, plots, composition, and rendering are well-separated. The dominant problem is `chart.py` at **5,779 lines**, a God class that merges 30+ mark methods, encoding resolution, spec serialization, composition operators, and configure surface. The second systemic issue is duplicated `configure_*` signatures across `Chart` and `_ChartLike`.

**Rust:** The module structure is clean, with excellent macro-driven dispatch (`for_each_transform!`, `for_each_mark!`). The biggest structural problems are: (1) the render pipeline is **copy-pasted** between `render_svg` and `render_scene_json`, (2) god structs with 23–57 fields (`PreparedInputs`, `ThemeInputs`), and (3) `theme_from_dict` is **400+ lines** of repetitive manual dict extraction.

---

## Python — Top Issues

### 1. `chart.py` God Class (5,779 lines)

`Chart` combines mark declaration (30+ methods), encoding resolution, `to_spec()` (252 lines), `__add__` (200 lines), `_resolve_pending` (170 lines), configure surface, and inline data_transform closures — all in one class.

**Fix:** Extract into focused mixins — diagnostic marks (~1,700 lines), statistical marks (~900 lines), and configure methods (~350 lines) — cutting the file by ~50% without changing the public API.

### 2. Duplicated `configure_*` Signatures

Six `configure_*` methods are copy-pasted between `Chart` (chart.py:4473–4816) and `_ChartLike` (composition.py:290–577). ~500 lines of near-identical code. Adding a parameter requires updating both.

**Fix:** Shared mixin or thin wrappers generated from the config dataclass definitions.

### 3. Inline `data_transform` Closures

Nine closures defined inside `mark_*` methods capture outer kwargs, mixing Polars manipulation with mark wiring. Untestable in isolation.

**Fix:** Move to named functions in `marks/diagnostic/*.py`, pass via `functools.partial`.

### 4. `_clone` is Manual Slot-by-Slot Copy

24 slots manually copied — adding a slot and forgetting `_clone` causes **silent data loss** during fluent chaining.

**Fix:** Auto-generate from `__slots__` with explicit shallow-copy for mutable containers.

### 5. `to_spec()` — 252 Lines, 5 Interleaved Concerns

Channel aliasing, aggregate remapping, polar remapping, transform serialization, and auto-tooltip synthesis in one method.

**Fix:** Extract `_build_encoding_specs()`, `_resolve_aggregates()`, `_apply_polar_remapping()`.

### Python Complexity Hotspots

| Method | Lines | Nesting |
|---|---|---|
| `Chart.to_spec()` | 252 | 4 |
| `Chart.__add__()` | 200 | 4 |
| `Chart._resolve_pending()` | 170 | 4 |
| `Chart.mark_function()` | 122 | 3 |
| `Chart.encode()` | 117 | 3 |

### Python Module-by-Module Assessment

| Module | Lines | Cohesion | Complexity | Notes |
|---|---|---|---|---|
| `chart.py` | 5,779 | **Low** | **High** | God class. 30+ mark methods, spec serialization, composition operators, configure surface, inline data_transform closures. |
| `composition.py` | 2,450 | **Medium** | **Medium** | `_ChartLike` base + 7 composition classes + 6 scene-merge helpers. configure_* methods copy-pasted from Chart. |
| `_render.py` | 598 | **High** | **Low** | Clean mixin. Single responsibility: rendering and display methods. |
| `_coerce.py` | 247 | **High** | **Low** | Clean dispatch chain for data normalization. |
| `_interactive.py` | 327 | **High** | **Low** | Clean widget class. Single responsibility. |
| `_html.py` | 307 | **High** | **Low** | HTML assembly. Focused and well-scoped. |
| `configure.py` | 372 | **High** | **Low** | Clean frozen dataclasses. Good use of `_to_dict_omit_none`. |
| `_metric_labels.py` | 388 | **Medium** | **Medium** | 4 label classes + application logic. Reasonable scope. |
| `_overrides.py` | 151 | **High** | **Low** | Clean utility with layer-name registry. |
| `display.py` | 348 | **High** | **Low** | Output orchestration. Well-structured. |
| `selection.py` | 550 | **High** | **Low** | Immutable selection descriptors. Clean frozen dataclasses. |
| `color.py` | 380 | **High** | **Low** | Color utilities. Cohesive. |
| `transforms.py` | 741 | **High** | **Low** | Pure dict constructors. Clean and repetitive by design. |
| `themes/__init__.py` | 363 | **High** | **Low** | Theme value class. Clean immutable API. |
| `themes/builtins.py` | 242 | **High** | **Low** | Named theme factories. Cohesive. |
| `encoding/base.py` | 197 | **High** | **Low** | `ChannelBase` + `_PendingAggregate`. Clean base class. |
| `encoding/positional.py` | 369 | **High** | **Low** | Positional channel subclasses. Consistent. |
| `encoding/appearance.py` | 365 | **High** | **Low** | Appearance channel subclasses. Consistent. |
| `encoding/text.py` | 263 | **High** | **Low** | Text/tooltip channel subclasses. Consistent. |
| `annotation/primitives.py` | 649 | **High** | **Low** | Annotation primitive dataclasses. Clean. |
| `marks/composite.py` | 612 | **High** | **Medium** | Composite desugar functions. Well-structured. |
| `marks/heavy_stat.py` | 854 | **Medium** | **Medium** | Statistical mark desugars. Getting long. |
| `marks/statistical.py` | 567 | **High** | **Low** | Core statistical desugar. Clean. |
| `marks/base.py` | 223 | **High** | **Low** | `MarkBase` — clean value object. |
| `plots/classification.py` | 1,823 | **Medium** | **Medium** | 10 classification figure functions. Repetitive pattern (intentionally). |
| `plots/regression.py` | 1,336 | **Medium** | **Medium** | Regression figure functions. Similar repetitive pattern. |
| `plots/distribution.py` | 820 | **High** | **Low** | Distribution figure functions. Clean. |
| `plots/explanation.py` | 1,121 | **Medium** | **Medium** | SHAP/PDP figure functions. Long but well-structured. |
| `plots/ranking.py` | 1,019 | **Medium** | **Low** | Feature ranking figure functions. |
| `plots/clustering.py` | 876 | **Medium** | **Low** | Clustering figure functions. Clean. |
| `plots/matrix.py` | 1,030 | **Medium** | **Medium** | Heatmap/clustermap/pairplot/jointplot. Some complexity. |
| `plots/model_selection.py` | 679 | **High** | **Low** | Model selection figure functions. |
| `plots/_helpers.py` | 284 | **High** | **Low** | Shared builder helpers. Well-factored. |
| `_diagnostics/` (all) | ~4,500 | **High** | **Low** | ModelSource + visualizers. Clean OO hierarchy. |

### Python API Consistency Issues

**A. `configure_*` method parameter lists** — `configure_axis` on `Chart` (line 4475) and `_ChartLike` (line 290) have identical parameter lists, but if one is updated and the other is not, users get different behavior depending on context.

**B. `mark_*` docstring pattern inconsistency** — Some methods use `**mark_kwargs` as a catch-all, others explicitly name keyword args before `**kwargs`. Diagnostic marks consistently use `**mark_kwargs`; statistical marks are mixed.

**C. `data_transform` parameter naming** — Closures sometimes guard against non-polars data (try/except ImportError) and sometimes assume polars unconditionally.

**D. Return type consistency in composition** — `Chart.interactive()` returns `InteractiveChart`, but `Chart.__add__` does not handle `InteractiveChart` as an operand.

---

## Rust — Top Issues

### 1. `render_svg` / `render_scene_json` Duplication

The override-layering pipeline (~50 lines) is copy-pasted verbatim between the two functions (render/mod.rs:561–769). The secondary-Y padding block is **already missing** from `render_scene_json` — demonstrating the drift this causes.

**Fix:** Extract `prepare_and_layout_for_render()` returning `(PreparedInputs, LayoutResult, ThemeInputs, Vec<RenderWarning>)`. Both functions call it, then diverge only at the output stage.

### 2. `ThemeInputs` — 57 Fields, Flat Struct

Mixes layout, typography, axis, grid, mark, palette, legend, and reference-line concerns. Every config-application function must manually cascade each field, producing 130+ lines of `if let Some(ref ...) = ...` in `apply_chart_config` alone.

**Fix:** Decompose into sub-structs (`ThemePadding`, `ThemeTypography`, `ThemeAxis`, `ThemeGrid`, `ThemeMarks`, `ThemePalette`, `ThemeLegend`), each with its own `apply_overrides`.

### 3. `PreparedInputs` — 23 Fields, 12 Legend-Specific

12 of 23 fields are legend overrides extracted from the same JSON map. Adding a new legend override requires touching 4 sites.

**Fix:** Introduce `LegendPreparedOverrides` sub-struct. One field replaces twelve.

### 4. `theme_from_dict` — 400+ Lines of Boilerplate

57 fields extracted with the identical `if let Some(v) = dict.get_item("key")?` pattern.

**Fix:** `#[derive(Deserialize)]` on `ThemeInputs` + `pyo3_serde::from_py`, or a macro generating the extraction.

### 5. `scale_resolve.rs` — 2,205 Lines, No Decomposition

Largest file in the crate. Scale resolution for all channel types, tick generation, domain computation, and sort handling.

**Fix:** Split into `resolve_positional.rs`, `resolve_color.rs`, `resolve_auxiliary.rs`, `resolve_domain.rs`.

### 6. `Encoding::inherit_from` / `inherit_non_positional` Duplication

Identical inner closure duplicated across both methods, enumerated for all 20 channels. Adding a channel requires editing both.

**Fix:** Module-level `inherit` function + `POSITIONAL_CHANNELS` list to drive the non-positional variant.

### Rust Complexity Hotspots

| Function | Lines | Notes |
|---|---|---|
| `prepare_render_inputs` | ~487 | Transform pipeline + scale + axis + facet + legend in one function |
| `apply_stack` | ~620 | Stack position adjustment with normalize/center |
| `theme_from_dict` | ~400 | Repetitive dict extraction |
| `build_scene` | ~400+ | Per-panel loop building everything in one pass |
| `apply_chart_config` | ~130 | Pure if-let-Some cascading |
| `point::build` | ~737 | 6 shapes × categorical/quantitative dispatch |

### Rust Module-by-Module Assessment

| Module | Cohesion | Complexity | Notes |
|---|---|---|---|
| `lib.rs` | **High** | **Low** | Clean PyO3 module registration. Good macro use. |
| `pyo3_serde.rs` | **High** | **Low** | Focused bridge utility. |
| `transport.rs` | **Medium** | **Low** | `process_batch` is vestigial Phase 1 demo. |
| `diagnostics.rs` | **High** | **Low** | Kendall tau-b. `assert_eq!` should be `Result`. |
| `projection.rs` | **High** | **Low** | Clean map projections. |
| `spec/chart.rs` | **Medium** | **High** | `ChartSpec::new` has 35 parameters. |
| `spec/encoding.rs` | **Medium** | **Medium** | 20 `Option<EncodingSpec>` fields. Duplicated inheritance. |
| `spec/mark.rs` | **High** | **Low** | Excellent macro-driven pattern. |
| `spec/layer.rs` | **High** | **Low** | Duplicate `name: None` on line 42 (copy-paste artifact). |
| `spec/coord.rs` | **High** | **Low** | Clean `to_scene_coord` mapping. |
| `spec/mark_style.rs` | **Medium** | **Low** | 29 optional fields — bag-of-optionals pattern. |
| `spec/position.rs` | **High** | **Low** | Well-structured tagged enum. |
| `spec/title.rs` | **High** | **Low** | Clean, uses `deny_unknown_fields`. |
| `render/mod.rs` | **Low** | **High** | 1834 lines. Duplicated render pipeline. |
| `render/prepare.rs` | **Medium** | **High** | `prepare_render_inputs` is 487 lines. God struct. |
| `render/scale_resolve.rs` | **Medium** | **High** | 2205 lines (largest file). |
| `render/binding.rs` | **Medium** | **High** | `theme_from_dict` is 400+ lines of boilerplate. |
| `render/draw.rs` | **Medium** | **Medium** | `MarkStyle` 29 fields mirrors `MarkKwargsSpec`. |
| `render/scene_build.rs` | **Medium** | **High** | 1257 lines. Per-panel loop is complex. |
| `render/marks/*` | **High** | **Medium** | Individual mark renderers well-scoped. `point.rs` (920 lines) largest. |
| `render/annotation.rs` | **Medium** | **Medium** | 3 `#[allow(dead_code)]` on reserved fields. |
| `render/break_axis.rs` | **Medium** | **Low** | 3 `#[allow(dead_code)]` annotations. |
| `render/format.rs` | **High** | **Low** | One `#[allow(dead_code)]` on `format_ordinal_number`. |
| `scale/*` | **High** | **Low** | Well-structured. `core.rs` provides shared validation. |
| `transform/core.rs` | **High** | **Low** | Excellent macro. Clean pipeline orchestration. |
| `transform/*` (individual) | **High** | **Medium** | Self-contained. `smooth.rs` (1866 lines) and `stats.rs` (1457 lines) large but necessarily so. |
| `layout/mod.rs` | **Medium** | **High** | `ThemeInputs` 57 fields. `compute_layout` ~991 lines. |
| `layout/axis.rs` | **Medium** | **High** | 1647 lines. `layout_x_axis` ~957 lines. |

### Rust Type Design Issues

**A. Encoding: 20 `Option<EncodingSpec>` fields without grouping** — Makes `inherit_from`, `overlay_from`, and serialization all O(channels) with per-field code.

**B. ScaleSpec enum variants duplicate field sets** — `Linear`, `Log`, `Time`, `Symlog`, `Pow`, `Sqrt`, `Utc` all share `domain`, `range`, `clamp`, `padding`. Could factor into `ContinuousScaleCommon`.

**C. DataRef is a single-variant enum** — `DataRef::Named { name: String }` could be simplified to a newtype.

**D. `MarkStyle` mirrors `MarkKwargsSpec` field-for-field** — Both have 25+ overlapping fields. `resolve_mark_style` manually maps between them.

### Rust Dead Code

| Location | What | Status |
|---|---|---|
| `transport::process_batch` | Phase 1 demo — renames first column | Vestigial, never used in rendering |
| `annotation.rs` (3 sites) | `z`, `curve` fields | Reserved for future features |
| `break_axis.rs` (3 sites) | `ScaleSegment`, `segments` field | Claimed "used by broken_scale_map" |
| `format.rs` | `format_ordinal_number` | Pending wiring |
| `diagnostics.rs:126` | `assert_eq!` in library code | Should return `Result`, not panic |

---

## Cross-Cutting Observations

1. **The configure pipeline is the single most duplicated concern** — it appears in Python (`Chart` vs `_ChartLike`) and Rust (`render_svg` vs `render_scene_json` vs `theme_from_dict`). Unifying it on both sides would eliminate ~1,500 lines of near-duplicate code.

2. **God structs are the Rust equivalent of the Python God class** — `ThemeInputs` (57 fields), `PreparedInputs` (23 fields), `MarkStyle` (29 fields), and `ChartSpec::new` (35 parameters) all follow the same flat-bag-of-optionals pattern. Sub-grouping would improve every function that touches them.

3. **The Python/Rust boundary is clean** — the `_coerce.py` → Arrow CDI → Rust pipeline respects the design. No leaky abstractions detected at the FFI boundary.

---

## Prioritized Recommendations

### High Impact, Low Effort

1. **Extract shared Rust render pipeline** — eliminates copy-paste drift between `render_svg` and `render_scene_json`
2. **Group legend overrides into `LegendPreparedOverrides`** — 12 fields → 1 field
3. **Auto-generate `Chart._clone` from `__slots__`** — prevents silent data loss

### High Impact, Moderate Effort

4. **Split `chart.py` into mark-method mixins** — ~2,600 lines extracted
5. **Decompose `ThemeInputs` into sub-structs** — simplifies every config function
6. **Auto-derive `theme_from_dict` via serde** — eliminates 400 lines of boilerplate
7. **Unify `configure_*` methods via shared mixin** — eliminates ~500 lines of Python duplication

### Medium Impact, Moderate Effort

8. **Split `scale_resolve.rs` into sub-modules**
9. **Extract `to_spec()` internals into focused helpers**
10. **Factor common `ScaleSpec` fields into `ContinuousScaleCommon`**
