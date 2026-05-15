# Code Archaeology Report — Unimplemented / Unconnected Hints

**Date:** 2026-05-15  
**Scope:** Full source sweep — `src/ferrum/`, `crates/ferrum-core/src/`, `crates/ferrum-wasm/src/`, `tests/`, `design-docs/`, `docs/superpowers/`, `ferrum-spec.md`  
**Method:** Three parallel agents searching for TODO/FIXME/STUB markers, `todo!()`/`unimplemented!()` macros, silent-drop patterns, skipped tests, and spec-vs-impl gaps.

---

## Active Bugs (code wired incorrectly right now)

| ID | Location | Issue | Status |
|---|---|---|---|
| B2 | `src/ferrum/chart.py:4400` + `src/ferrum/encoding/base.py:152` | `to_encoding_spec_dict()` emits `"type_"` but `_build_layers_list` reads `d.get("type")` — data type silently dropped for **all** composite-mark layer encodings | ✅ Fixed `dbc9f41` |
| F8 | `crates/ferrum-wasm/src/hit_test.rs:116` | `Tick`, `Text`, `Ribbon`, `Segment`, `Image` all fall to `_ => None` — tooltips/selections silently broken for those marks in WASM | ✅ Fixed `175664a` |
| F17 | `crates/ferrum-core/src/transform/core.rs:177` | Only `Qq` and `Linkage` dispatch to `secondary_outputs`; every other transform returns empty — any transform needing secondary batches silently drops them | ✅ Not a bug — `LetterValue` already wired via explicit arm; no other transforms implement `secondary_outputs` |

> **2026-05-15 update:** All three active bugs resolved. B2 fix normalises the key read and expands shorthand type strings. F8 wires five missing hit-test arms (Tick/Segment→`hit_test_lines`, Ribbon→new `Path` arm in `hit_test_lines`, Text→`hit_test_texts`, Image→`hit_test_images`); 16 new Rust tests added. F17 was already correct — `LetterValue` had been wired via an explicit arm outside the macro at some prior point. Note: `nearest_in_batch` still only handles Circle and Rect — the five newly wired kinds won't participate in nearest-mark hover selection (separate follow-up).

---

## Skipped Tests (known bugs, never fixed)

| File | Status |
|---|---|
| `tests/marks/test_heavy_stats.py:188` — `mark_violin(inner=None)` scale-resolve on small samples | ✅ Fixed `531f10d` |
| `tests/test_phase_8b_e2e.py:60` — `test_contour_renders` | ✅ Skip removed `30c4732` (was already passing) |
| `tests/test_phase_8b_e2e.py:87` — `test_hex_renders` | ✅ Skip removed `30c4732` (was already passing) |

> **Violin fix:** loosened `vals.len() < 2` guard to `vals.is_empty()` in `violin.rs`; single-element groups now emit degenerate vertices (visible to scale resolution, invisible to renderer). Regression tests added `aab2b0d` — 3 tests covering single-element, all-equal, and mixed-group paths; verified to fail on reverted guard.

---

## High-Severity Rust Gaps

| ID | Location | Issue | Status |
|---|---|---|---|
| F3 | `crates/ferrum-core/src/render/marks/label.rs:42` | Leader lines hardcoded `draw_leader = false` | ✅ Fixed `87e8c6b` — `leader_line: Option<bool>` wired through `MarkStyleSpec` → `mark_label()` kwarg |
| F1 | `crates/ferrum-core/src/render/arrow_cast.rs:35` | `is_numeric` helper written but never called; stale `#[allow(dead_code)]` | ✅ Fixed `f87bccd` — stale attribute removed (was already called at `scale_resolve.rs`) |
| F14 | `crates/ferrum-core/src/render/scene_build.rs:464` | Polar transform only handled `Circle`/`Polyline` | ✅ Fixed `a5d1de0` — extended to `Line`, `Text`, `Rect`→`Polygon` arc sampling |
| F13 | `crates/ferrum-core/src/render/marks/geoshape.rs:111` | Interior polygon rings discarded; `Point`/`LineString` produced empty output | ✅ Fixed `9d299f2` — `SceneNode::Polygon` now carries `rings: Vec<Vec<...>>`; holes via `fill-rule evenodd`; `Point`→`Circle`; `LineString`→`Polyline` |
| F9 | `crates/ferrum-wasm/src/conditional.rs:148` | `Size` conditional dropped for Rect/Bar | ✅ Fixed `26a77c0` — `Size` arm added to `apply_value_to_rect` |
| F11 | `crates/ferrum-wasm/src/zoom_pan.rs:5` | `ScaleMode::Independent` stale `#[allow(dead_code)]` | ✅ Fixed `26a77c0` — variant was already live; stale attribute removed |
| F7 | `crates/ferrum-core/src/transform/swarm.rs:253` | `eprintln!` debug output in production code | ✅ Fixed `f87bccd` — replaced with comment |

> **Regression tests added `aab2b0d`:** 16 new `#[cfg(test)]` tests across all four previously untested fix paths. Each test was verified to fail on the reverted code: geoshape `polygon_with_hole_has_two_rings` fails if holes are dropped; polar transform tests assert coordinate changes and Rect→Polygon conversion; leader-line tests assert `SceneNode::Line` presence/absence by flag.

---

## Python Silent Drops (accepted by Python API, never reach Rust)

### 11 mark kwargs with no `MarkKwargsSpec` path

✅ **Resolved `82a1496`** — TDD investigation found 10 of 11 were already fully implemented (exist in `MarkKwargsSpec`, `to_mark_kwargs_dict()`, and Rust renderers with passing tests). Only `width=` on `mark_boxplot()` was genuinely missing; fixed as an alias to `size=`.

### Channels accepted, never rendered (static SVG)

| Channel | Status |
|---|---|
| `stroke_opacity`, `stroke_width`, `stroke_dash`, `angle` | ✅ Promoted to `_RENDERER_HONORED_CHANNELS` — SVG attribute emission wired in `point.rs`, `bar.rs`, `line.rs`, `rule.rs`; WASM GPU instances wired via `FillStroke`. Commits `26f20b3`, `e387017`, `a8d8da8` |
| `Description` / `Key` | Open — TODO(G1) still present; `ChartSpec` has no `description` field |
| `Theta` / `Radius` | Stale docstrings (say `NotImplementedError` but `CoordPolar` shipped Phase 11) |
| `Href`, `Description` (encoding channels) | Open — only `type` honored |
| `fill_opacity` (via StrokeOpacity alias) | Open — `pass` branch, no warning |

### 5 `EncodingSpec` fields deserialized, never read by Rust renderer

| Field | Status |
|---|---|
| `sort` | ✅ Confirmed — SVG tick order verified in `test_silent_drop_verification.py` |
| `stack` | ✅ Confirmed — `apply_position` now called unconditionally (`f641f39`); encoding-level `stack=` honored. SVG bar positions verified. |
| `axis` | ✅ Confirmed — `label_angle`, `ticks`, `labels`, `title` verified in SVG output |
| `format_type` | ✅ Confirmed — tick label format verified in SVG output |
| `impute` | ✅ Confirmed — missing rows filled; polyline point count verified |

### Features that raise `ValueError` at runtime

| Feature | Status |
|---|---|
| `mark_histogram(multiple="stack"/"fill"/"dodge")` | ✅ Confirmed — all three modes verified in SVG output. `be32daf` fixed bin-edge alignment for stack. |
| `mark_density(multiple="dodge")` | ✅ Confirmed — verified in SVG output |
| `mark_ribbon(interpolate=...)` | Open — still no-op |
| `lmplot(truncate=False)` / `regplot(truncate=False)` | ✅ Confirmed — `x_range` now forwarded to `mark_smooth()` (`91dd487`). Fit line extends to axis boundary. Verified in SVG output. |
| `Chart(data=None)` with per-layer data | ✅ Confirmed — both layers render; verified in SVG output |
| `Layer(data=...)` via `Chart.layer()` | ✅ Confirmed — verified in SVG output |
| `mark_hex(stroke=..., stroke_width=...)` | Open — still raises |
| `mark_function(clip=False)` | Open — still no-op |

> **2026-05-15:** `mark_raster(blend="additive")` SVG already implemented; WASM additive pipeline wired `26f20b3`. `mark_swarm(dodge=...)` already wired. Legend kwargs fully confirmed: `orient` ✅ `title` ✅ `format` ✅ `columns` ✅ — all four verified by behavioral tests (`test_silent_drop_verification.py::TestLegendKwargsSVGPosition`). `format` and `columns` wired in `10c1931`.

### `VisualBase.score()` not overridden in 14 visualizer subclasses

✅ **Resolved `8232d62`** — Two-track fix with 20 TDD tests:
- **Group A** (`LearningCurveVisualizer`, `ValidationCurveVisualizer`, `CVScoresVisualizer`, `AlphaSelectionVisualizer`): `score(X, y)` implemented as `estimator.score(X, y)`, matching the `ResidualsVisualizer` reference pattern.
- **Group B** (remaining 10): `FerrumVisualizer.score()` base changed from `raise NotImplementedError` to `return 0.0` — no-op with a docstring explaining why. These visualizers describe data structure or model internals; test-set scoring is not semantically meaningful for them.

---

## Missing Spec Implementations

| Item | Spec Location | Notes |
|---|---|---|
| Phase 12 extension points (`register_mark`, `register_stat`, `register_renderer`, `MarkProtocol`, `StatProtocol`, `RendererProtocol`) | `ferrum-spec.md §Part IV` | No code, no spec doc written — `ferrum-phases.md` status: `pending` |
| `ferrum.data` namespace (`sample_datasets()`, `load(name)`) | `ferrum-spec.md §3.19` | ~~Not implemented~~ **Intentionally dropped** — users get sample data from sklearn/seaborn optional deps; a ferrum-native dataset loader adds maintenance cost for no real value |
| `ferrum.color` namespace (`palette()`, `to_hex()`, `diverging_palette()`) | `ferrum-spec.md §3.19` | Entirely absent |
| `ferrum.config` namespace (`set_max_rows()`, `set_renderer()`, `set_default_width/height()`, `set_raster_threshold()`, `set_raster_behavior()`, `set_default_backend()`, `set_font_paths()`) | `ferrum-spec.md §3.19` | Entirely absent |
| `Axis(...)` value class | `ferrum-spec.md §3.7` | Not publicly constructable; `axis=` kwarg accepted but stored as opaque dict and ignored by Rust renderer |
| `Legend(...)` kwargs beyond `disabled` | `ferrum-spec.md §3.7` | `orient`, `values`, `format`, `tick_count`, `label_font_size`, `columns`, etc. silently dropped |
| Auto-raster policy (`raster_threshold`, `raster_behavior`, `raster_aggregate`, `raster_cmap`) | `ferrum-spec.md §3.16/3.18` | Documented, not implemented |
| `RenderConfig` Python class (public) | `ferrum-spec.md §3.16` | No public `RenderConfig` class exists; `embed_fonts=False` is untestable |
| `ferrum.Grid` utility class | `ferrum-spec.md §3.19` | Absent from source |
| `ferrum.WindowTransform` | `ferrum-spec.md §3.19` | Absent from source |
| Full palette library (cyclical schemes, tealblues, brewer extended sequential) | `ferrum-spec.md §3.13` | Rejected at validation time |
| `mark_text` multiline via `<tspan>` | `docs/superpowers/followups/2026-05-12-mark-text-multiline-tspan.md` | `\n` in text collapses to single space in SVG; `marks/text.rs` does not split on `\n` |
| Sixel terminal rendering | `ferrum-spec.md §3.16` | **Intentionally dropped (2026-05-15)** — niche format, inconsistent across terminal emulators, audience is Jupyter/browser-first |
| `SceneNode::Raw` support in WASM renderer | `crates/ferrum-wasm/src/scene_load.rs:181` | Silently skipped with `console.warn` only |
| `share_x` / `share_y` enforcement in grid compositor | `crates/ferrum-core/src/render/grid_compose.rs:4` | Accepted, silently ignored — alignment left to caller |
| Axis tick-label formatting via `format=` on X/Y | `crates/ferrum-core/src/render/format.rs:1` | `format=` only honored for `mark_text`, not axis ticks |
| `compare=` routing in `gain_chart`, `lift_chart`, `discrimination_threshold_chart` | `docs/superpowers/followups/2026-05-12-schwabish-audit-remaining.md` | Only `roc_chart`, `pr_chart`, `calibration_chart` route the explicit-kwarg `compare=` form |

---

## Rust Dead Code / Suppressed Warnings

| ID | Location | Issue |
|---|---|---|
| F2 | `crates/ferrum-core/src/scale/ticks.rs:3` | `#![allow(dead_code)]` blankets entire module — which helpers are actually unused is invisible to the compiler |
| — | `crates/ferrum-core/src/render/color/scheme.rs` | Entire `CategoricalPalette` / `Scheme` color module unreferenced; 105 pre-existing dead-code warnings suppressed |
| — | `crates/ferrum-core/src/transform/letter_value.rs` | `OutlierRow` type declared but never constructed |
| — | `crates/ferrum-core/src/transform/core.rs` | `apply_transforms*` entry points unused |
| F16 | `crates/ferrum-core/src/render/marks/label.rs:84` | `mark_label` emits `MarkBatchKind::Text` instead of a dedicated `Label` kind — labels indistinguishable from text in the scene graph, preventing kind-specific dispatch in hit-test, conditional encoding, and WASM rendering |

---

## Stale Documentation / Comments

| Location | Issue |
|---|---|
| `src/ferrum/encoding/positional.py:292–293` | `Theta` / `Radius` docstrings say `CoordPolar` raises `NotImplementedError` — Phase 11 shipped `CoordPolar`; stale |
| `src/ferrum/_coerce.py:60` | Error message references "Phase 8a" — project is post-Phase 11 |
| `crates/ferrum-core/src/render/marks/text.rs:1` | Module header says "Phase 10c-pre extends the Phase 7 stub" — stale label, implementation is complete |
| `crates/ferrum-core/src/transform/contour.rs:505` | Doc comment says `smooth` parameter is "accepted but currently reserved; has no effect" — actually implemented as a 3×3 Gaussian kernel pass |
| `crates/ferrum-core/src/render/format.rs:1` | Module comment says per-axis `FormatSpec` "deferred to Phase 8" — Phase 8 is long done but the deferral was never resolved |
| `crates/ferrum-core/src/layout/binding.rs:2–4` | Comment says `ThemeInputs` wiring "Phase 8 will map ferrum.Theme into ThemeInputs" — never implemented; always uses `ThemeInputs::default()` |

---

## Prioritized Action List

### Immediate (active correctness bugs)
1. **B2** — Fix `"type_"` vs `"type"` key mismatch in `_build_layers_list` (`chart.py:4400`)
2. **F8** — Extend WASM hit-test dispatch table to cover `Tick`, `Text`, `Ribbon`, `Segment`, `Image`
3. **F17** — Wire remaining transforms into `secondary_outputs` dispatch in `transform/core.rs`

### High (features with dead or broken code paths)
4. **F3** — Wire leader-line path in `marks/label.rs` to a `mark_style.line` field, or remove the dead branch
5. **F14** — Extend polar coordinate transform in `scene_build.rs` to cover all scene node types
6. Unblock Task 37: wire per-cell quantitative coloring for `mark_contour` and `mark_hex`
7. Fix `mark_violin(inner=None)` scale-resolve integration for small samples

### Medium (spec-documented but silently dropped)
8. Wire the 11 silent-drop mark kwargs through `MarkKwargsSpec`
9. Implement `Description` → `<desc>` SVG element (TODO(G1) in `chart.py:4670`)
10. Implement `mark_text` multiline via `<tspan>` splitting on `\n`
11. Wire `format=` on X/Y encodings to axis tick-label formatters
12. Wire `Axis(...)` and `Legend(...)` full kwarg sets into Rust renderer

### Low (missing namespaces / Phase 12 scope)
13. ~~Scaffold `ferrum.data`~~ — dropped (users use sklearn/seaborn). Scaffold `ferrum.color`, `ferrum.config` namespaces
14. Clean up 105 suppressed Rust dead-code warnings; remove unused `CategoricalPalette`/`Scheme` module
15. Update stale docstrings (`CoordPolar`, Phase 8a error message, contour `smooth`)
16. Write Phase 12 spec doc and begin extension-point implementation
