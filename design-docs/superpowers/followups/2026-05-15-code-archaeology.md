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
| B3 | `polygon.rs` + `rect.rs` | `CoordFlip` drops mark elements for composite stat marks. **Root cause**: `polygon.rs` read y as float-only (`col_as_f64`); `rect.rs` dispatch didn't route flipped `x2` to ordinal-range builder. **Fix**: polygon.rs gets dual-path `ypx` (float + string fallback); rect.rs extends dispatch and adds horizontal orientation path. | ✅ Fixed — violin 0→3 filled paths, boxplot 1→4 data rects. 6 regression tests. |

> **2026-05-15 update:** All three original active bugs resolved. B3 added: structural smoke tests (`test_render_smoke.py::test_smoke_structural`) found that `CoordFlip` breaks composite stat marks — violin KDE paths and boxplot box rects are dropped during the polar/flip coordinate transform in `scene_build.rs`. Non-flipped variants are unaffected. Other smoke-test findings (boxen renders both rects and lines — correct; image/basic produces no `<image>` without URL column — by design) are not bugs. B2 fix normalises the key read and expands shorthand type strings. F8 wires five missing hit-test arms (Tick/Segment→`hit_test_lines`, Ribbon→new `Path` arm in `hit_test_lines`, Text→`hit_test_texts`, Image→`hit_test_images`); 16 new Rust tests added. F17 was already correct — `LetterValue` had been wired via an explicit arm outside the macro at some prior point. Note: `nearest_in_batch` still only handles Circle and Rect — the five newly wired kinds won't participate in nearest-mark hover selection (separate follow-up).

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

> **Open (noted 2026-06-01, flexibility D2):** `fix_kde_extents_for_facet` in `render/prepare.rs` pins the global x-extent for a single-group faceted KDE so per-panel curves share a comparable axis. It is **KDE-only**: faceting-before-transform drops cross-partition extent unification in general, and `Bin`/`Violin` carry the same `extent`/`shared_extent` field pair. A faceted histogram or violin with auto extent has the same per-panel-drift and is NOT fixed by this helper. Generalize the extent-pin across transforms when faceted `Bin`/`Violin` is next touched — do not file it as a new regression.

> **Open (noted 2026-06-01, flexibility D10):** figure-level `.properties(title=/subtitle=/caption=)` chrome (rendered once around a composed figure) is wired for `_CompositeBase` (`VConcatChart`/`HConcatChart`/`ConcatChart`) and faceted `Chart`. `JointChart` (`composition.py:619`) and `ClusterMapChart` (`composition.py:1268`) override `properties` to route all kwargs into their inner panel, so `joint.properties(title=...)` lands the title on the center panel rather than wrapping the figure. Out of D10's stated scope (vconcat/hconcat/facet); wire figure chrome into those two composites when next touched.

> **Open (noted 2026-06-01, flexibility D7):** `build_polar` in `render/marks/bar.rs` (polar/coxcomb bars) emits `tooltips: None`/`hrefs: None` and applies only flat `mark_style.opacity`, whereas the arc annular path (`build_annular`) wires per-row tooltips and per-row opacity. So a polar `mark_bar` with `tooltip=`/per-row `opacity=` silently loses them. Out of D7's geometry scope; wire to match `build_annular` when polar bars are next touched. Also: the polar channel-mapping convention (theta="x"→radius=y etc.) is duplicated between `arc.rs` and `bar.rs` — extract a shared `polar_channels(ctx)` helper if a third polar mark lands.

---

## Python Silent Drops (accepted by Python API, never reach Rust)

### 11 mark kwargs with no `MarkKwargsSpec` path

✅ **Resolved `82a1496`** — TDD investigation found 10 of 11 were already fully implemented (exist in `MarkKwargsSpec`, `to_mark_kwargs_dict()`, and Rust renderers with passing tests). Only `width=` on `mark_boxplot()` was genuinely missing; fixed as an alias to `size=`.

### Channels accepted, never rendered (static SVG)

| Channel | Status |
|---|---|
| `stroke_opacity`, `stroke_width`, `stroke_dash`, `angle` | ✅ Promoted to `_RENDERER_HONORED_CHANNELS` — SVG attribute emission wired in `point.rs`, `bar.rs`, `line.rs`, `rule.rs`; WASM GPU instances wired via `FillStroke`. Commits `26f20b3`, `e387017`, `a8d8da8` |
| `Description` (chart-level) | ✅ Fixed `6e45ddd` — `chart_description: Option<String>` added to `ChartSpec` + `SceneGraph`; `svg_walk.rs` emits `<desc>` as first child of root `<svg>`. Python `Chart.properties(description=)` → `kw["chart_description"]`. TODO(G1) removed. 7 regression tests. |
| `Theta` / `Radius` | ✅ Docstrings updated — `CoordPolar` shipped Phase 11 (fixed in Stale Documentation section) |
| `Href` (encoding channel) | ✅ Already working — `_RENDERER_HONORED_CHANNELS` → `EncodingSpec` → `MetadataColumns` → `svg_walk.rs` wraps each mark in `<a href="...">`. Verified: 3 `<a>` tags for 3-row DataFrame |
| `Description` (encoding channel) | ✅ Already working — same path → `svg_walk.rs` emits `<desc>` per mark. Verified: 3 `<desc>` tags for 3-row DataFrame |
| `Key` | Intentionally silent — stored for future interactive/animated rendering (`_SILENT_CHANNELS`); not a static SVG gap |
| `fill_opacity` | ✅ Fixed `1623cee` — promoted from `_SILENT_CHANNELS` to `_RENDERER_HONORED_CHANNELS`; `fill_opacity: Option<EncodingSpec>` added to Rust `Encoding`; per-row reading in `point.rs`/`bar.rs`; `fill-opacity` SVG attribute emitted (omitted when 1.0). Old alias to `opacity` removed. 12 regression tests. ✅ WASM GPU shader path also fixed (2026-05-22, feat/rtree-toolbar): `pack_instances.rs` + `tessellate.rs` now use `fill_opacity`/`stroke_opacity` per-channel; `opacity` no longer double-applied to stroke color in circle.wgsl/rect.wgsl. |

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
| `mark_ribbon(interpolate=...)` | ✅ Now rejects non-linear with `ValueError` — deliberate limitation, not a silent no-op |
| `lmplot(truncate=False)` / `regplot(truncate=False)` | ✅ Confirmed — `x_range` now forwarded to `mark_smooth()` (`91dd487`). Fit line extends to axis boundary. Verified in SVG output. |
| `Chart(data=None)` with per-layer data | ✅ Confirmed — both layers render; verified in SVG output |
| `Layer(data=...)` via `Chart.layer()` | ✅ Confirmed — verified in SVG output |
| `mark_hex(stroke=..., stroke_width=...)` | ✅ Fixed `b2aa797` (2026-05-30) — `desugar_hex` passes `stroke`/`stroke_width` into the polygon layer's `mark_kwargs`; ValueError guards removed. Literal semantics: stroke color + width 0 = no visible border (no call-time auto-bump), consistent with other polygon marks. |
| `mark_function(clip=False)` | ✅ Now rejects with `ValueError` — clipping always enabled by design |

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
| `ferrum.color` namespace (`palette()`, `to_hex()`, `diverging_palette()`) | `ferrum-spec.md §3.19` | ✅ **Phase 12 done** — `src/ferrum/color.py`: `palette()`, `to_hex()`, `sequential()`, `diverging()` wrapping Rust palette registry |
| `ferrum.config` namespace (`set_max_rows()`, `set_renderer()`, `set_default_width/height()`, `set_raster_threshold()`, `set_raster_behavior()`, `set_default_backend()`, `set_font_paths()`) | `ferrum-spec.md §3.19` | ✅ **Phase 12 done** — `src/ferrum/config.py`: contextvars-backed `set()`, `get()`, `defaults()`, `reset()` |
| `Axis(...)` value class | `ferrum-spec.md §3.7` | ✅ **Phase 12 done** — `src/ferrum/axis.py` frozen dataclass with all §3.7 params + encoding integration |
| `Legend(...)` kwargs | `ferrum-spec.md §3.7` | ✅ All 11 kwargs confirmed: `orient`, `title`, `format`, `columns` (`10c1931`) + `tickCount`, `labelFontSize`, `gradientLength`, `gradientThickness`, `direction`, `values`, `type` (wired through `LegendOverrides` in layout). 14 regression tests. |
| Auto-raster policy (`raster_threshold`, `raster_behavior`, `raster_aggregate`, `raster_cmap`) | `ferrum-spec.md §3.16/3.18` | ✅ Implemented `5effc0d` — `_apply_auto_raster()` in `chart.py` substitutes `mark_raster` when mark count exceeds threshold (default 500k). Eligible marks: point, bar, rect, tick, rule, segment. Skips composite marks, color-encoded charts. `RenderConfig` dataclass controls policy. 9 acceptance tests. |
| `RenderConfig` Python class (public) | `ferrum-spec.md §3.16` | ✅ Implemented `5effc0d` — `fm.RenderConfig(raster_threshold=, raster_behavior=, raster_aggregate=, raster_cmap=)` exposed via `__init__.py`, accepted by `Chart.properties(render_config=)`. |
| `ferrum.Grid` utility class | `ferrum-spec.md §3.19` | ✅ **Fixed (2026-05-30, `feat/render-gaps-17-19-21`)** — full §3.19 `Grid` shipped with a real minor-tick generation subsystem. Scale engine emits classified minor ticks (`Tick{position,is_major}`; subdivision in transformed space, log 2-9 per decade; categorical/discretizing produce none); layout threads them; per-level `major_*`/`minor_*` theme keys + builtin lighter/thinner minor defaults; `build_grid` emits minors under majors gated on `minor`. Python `ferrum.Grid` frozen dataclass (bare shorthand = both-levels fallback, per-level overrides) ingested by `Theme`. **Prerequisite landed first:** continuous-axis ticks/gridlines now scale-projected (commit `dc69206`) so they coincide with data marks — fixed a pre-existing major-vs-mark misalignment uncovered during this work. `GridConfig` (chart-level) unchanged = major level. Commits `ccce9c8` (scale), `e635c24` (layout), `dc69206` (projection prereq), `dff4aaa` (per-level styling), `dfd2c63` (Python Grid). |
| `ferrum.WindowTransform` | `ferrum-spec.md §3.19` | ✅ **Phase 12 done** — `transform_window` in Rust (`data_window.rs`) + Python API (`transforms.py`). Supports rolling sum/mean/count/min/max, rank, dense_rank, row_number, lag, lead, first_value, last_value with frame/groupby. |
| Full palette library (cyclical schemes, tealblues, brewer extended sequential) | `ferrum-spec.md §3.13` | ⚠️ **Partially resolved** — `ferrum.color` wraps existing Rust registry (7 categorical + 5 sequential + 6 diverging); cyclical schemes (`rainbow`, `sinebow`) and brewer-extended sequential remain deferred |
| `mark_text` multiline via `<tspan>` | `docs/superpowers/followups/2026-05-12-mark-text-multiline-tspan.md` | ✅ Fixed — `SvgBuffer::text()` in `svg.rs` splits `\n` into `<tspan>` elements with `dy="1.2em"` line spacing. Single-line text unchanged. 6 regression tests + 4 Rust unit tests. |
| Sixel terminal rendering | `ferrum-spec.md §3.16` | **Intentionally dropped (2026-05-15)** — niche format, inconsistent across terminal emulators, audience is Jupyter/browser-first |
| `SceneNode::Raw` support in WASM renderer | `crates/ferrum-wasm/src/scene_load.rs` | ✅ Fixed `d094701`+`8f3e9f6` (2026-05-30) — Raw nodes carry a `RawAnchor { Chrome, Data }` discriminant (`#[serde(default)]` → Chrome) and are collected into the scene-export channel, then injected verbatim into the existing SVG DOM overlay. Single-pass id-namespacing across fragments (so the legend's split gradient defs + consuming rect resolve to the same id); chrome Raw fixed, data Raw rides the canvas pan/zoom transform; overlay stays `pointer-events:none`. Restores colorbar/inset/annotation-image in `.interactive()`. |
| `share_x` / `share_y` enforcement in grid compositor | `crates/ferrum-core/src/render/grid_compose.rs:4` | Accepted, silently ignored — not in Phase 12 scope |
| Axis tick-label formatting via `format=` on X/Y | `crates/ferrum-core/src/render/format.rs:1` | ✅ Fixed `fee904d` — `apply_tick_format` rewritten with D3-subset parser: `f`, `e`, `g`, `%`, `,`, `d`, `s` (SI prefix) format specs all honored. Rust unit tests + Python behavioral tests. |
| `compare=` routing in `gain_chart`, `lift_chart`, `discrimination_threshold_chart` | `docs/superpowers/followups/2026-05-12-schwabish-audit-remaining.md` | ✅ All three now accept `compare=` kwarg (2026-05-17 audit) |

---

## Rust Dead Code / Suppressed Warnings

| ID | Location | Issue | Status |
|---|---|---|---|
| F2 | `crates/ferrum-core/src/scale/ticks.rs:3` | `#![allow(dead_code)]` blankets entire module | ✅ Blanket allow removed; no dead code remains (2026-05-17 audit) |
| — | `crates/ferrum-core/src/render/color/scheme.rs` | Entire `CategoricalPalette` / `Scheme` color module unreferenced | ✅ Module removed from codebase (2026-05-17 audit) |
| — | `crates/ferrum-core/src/transform/letter_value.rs` | `OutlierRow` type declared but never constructed | ✅ Type removed from codebase (2026-05-17 audit) |
| — | `crates/ferrum-core/src/transform/core.rs` | `apply_transforms*` entry points unused | ✅ Not dead code — `apply_transforms_named` is actively called from `prepare.rs` (2026-05-17 audit) |
| F16 | `crates/ferrum-core/src/render/marks/label.rs` | `mark_label` emits `MarkBatchKind::Text` instead of a dedicated `Label` kind | ✅ Fixed `6aad1da` (2026-05-30) — dedicated `MarkBatchKind::Label` variant (serde `"label"`) added; `label.rs` emits it at all 5 sites. Tag-only: routes identically to Text in `svg_walk` clip handling and WASM hit-testing, so behavior is unchanged but the scene graph / `to_json()` now distinguish labels from text. |

---

## Stale Documentation / Comments

✅ **All resolved** — `5f5948f` (code fixes) + `f87ba9a` (`_coerce.py` Phase 8a message).

| Location | Status |
|---|---|
| `src/ferrum/encoding/positional.py:292–293` | ✅ Fixed — `Theta`/`Radius` docstrings updated to describe `CoordPolar` as shipped |
| `src/ferrum/_coerce.py:60` | ✅ Fixed earlier (`f87ba9a`) — "Phase 8a" removed |
| `crates/ferrum-core/src/render/marks/text.rs:1` | ✅ Fixed — phase stub references removed |
| `crates/ferrum-core/src/transform/contour.rs:505` | ✅ Fixed — doc now describes the 3×3 Gaussian kernel implementation |
| `crates/ferrum-core/src/render/format.rs:1` | ✅ Fixed — updated to describe current format wiring |
| `crates/ferrum-core/src/layout/binding.rs:2–4` | ✅ Fixed — updated to accurately describe `ThemeInputs` usage |

---

## Prioritized Action List

### Immediate (active correctness bugs) — ✅ all resolved
1. ~~**B2** — Fix `"type_"` vs `"type"` key mismatch~~ — ✅ `dbc9f41`
2. ~~**F8** — Extend WASM hit-test dispatch table~~ — ✅ `175664a`
3. ~~**F17** — Wire remaining transforms into `secondary_outputs`~~ — ✅ not a bug (already correct)

### High (features with dead or broken code paths) — ✅ all resolved
4. ~~**F3** — Wire leader-line path in `marks/label.rs`~~ — ✅ `87e8c6b`
5. ~~**F14** — Extend polar coordinate transform~~ — ✅ `a5d1de0`
6. Unblock Task 37: wire per-cell quantitative coloring for `mark_contour` and `mark_hex` — deferred (separate scope)
7. ~~Fix `mark_violin(inner=None)` scale-resolve~~ — ✅ `531f10d`

### Medium (spec-documented but silently dropped)
8. ~~Wire the 11 silent-drop mark kwargs through `MarkKwargsSpec`~~ — ✅ resolved (10/11 already wired; 1 boxplot `width=` alias added `82a1496`)
9. ~~Implement `Description` → `<desc>` SVG element (TODO(G1))~~ — ✅ fixed `6e45ddd` (chart_description on ChartSpec + SceneGraph)
10. ~~Implement `mark_text` multiline via `<tspan>` splitting on `\n`~~ — ✅ fixed in `svg.rs` (tspan with `dy="1.2em"`)
11. ~~Wire `format=` on X/Y encodings to axis tick-label formatters~~ — ✅ fixed `fee904d` (D3-subset format parser)
12. ~~Wire `Axis(...)` and `Legend(...)` full kwarg sets~~ — ✅ Legend fully wired (11 kwargs). ✅ `Axis(...)` value class implemented Phase 12 (`src/ferrum/axis.py`, frozen dataclass, all §3.7 params, encoding integration).

### Low (missing namespaces / Phase 12 scope)
13. ~~Scaffold `ferrum.data`~~ — dropped (users use sklearn/seaborn). ~~Scaffold `ferrum.color`, `ferrum.config` namespaces~~ — ✅ Phase 12 done (`src/ferrum/color.py`, `src/ferrum/config.py`)
14. ~~Clean up 105 suppressed Rust dead-code warnings; remove unused `CategoricalPalette`/`Scheme` module~~ — ✅ All cleaned up (2026-05-17 audit: blanket allow gone, `CategoricalPalette`/`Scheme` removed, `OutlierRow` removed, `apply_transforms_named` confirmed live)
15. ~~Update stale docstrings (`CoordPolar`, Phase 8a error message, contour `smooth`)~~ — ✅ all 6 fixed `5f5948f` + `f87ba9a`
16. ~~Write Phase 12 spec doc~~ — ✅ Spec + plan written 2026-05-17. ✅ Phase 12 implementation complete on `feat/phase-12-spec-completeness` (2713 pytest + 933 cargo tests pass)

### Remaining open (not covered by Phase 12)
17. ~~`mark_hex(stroke=, stroke_width=)` still raises `ValueError`~~ — ✅ Fixed `b2aa797` (2026-05-30, `feat/render-gaps-17-19-21`)
18. ~~`ferrum.Grid` utility class absent~~ — ✅ Fixed (2026-05-30, `feat/render-gaps-17-19-21`): full §3.19 `Grid` + minor-tick subsystem shipped, preceded by a continuous-axis scale-projection prerequisite that also fixed a pre-existing major-vs-mark gridline misalignment (see Missing Spec Implementations row above)
19. ~~`SceneNode::Raw` WASM support (skipped with warning)~~ — ✅ Fixed `d094701`+`8f3e9f6` (2026-05-30, `feat/render-gaps-17-19-21`)
20. `share_x` / `share_y` grid enforcement — confirmed intentional-by-design (binding documents the params were never functional; sharing is enforced upstream via `share_scale()`). Remaining work is dead-API cleanup of the inert `share` dict in the `.spec` properties, not a functional gap.
21. ~~F16: `MarkBatchKind::Text` for labels — labels indistinguishable from text in scene graph~~ — ✅ Fixed `6aad1da` (2026-05-30, `feat/render-gaps-17-19-21`)

---

## Review findings — `feat/render-gaps-17-19-21` (2026-05-30)

Surfaced by 4 parallel auditors (scene-pipeline, theme-wiring, pyo3-binding, interactive) + a heavyweight `rust-review` before merge. **Zero BUGs / S4 / S5** — the branch is correct and mergeable; these are cohesion (S2/S3), one narrow JS wrong-render, and cosmetic (S1) items.

| ID | Severity | Item | Disposition |
|---|---|---|---|
| R1 | S3 | `minor_tick_fractions` inlines its own projection loop instead of routing through `project_values_to_fractions` — drifted parallel-API; major path is all-or-nothing on non-finite, minor path drops per-element (the difference is legitimate but should be a named policy, not a copied loop). `scale_resolve/mod.rs`. | **Fixing on this branch** (S2/S3 cleanup) |
| R2 | S3 | `AxisInput` field sprawl (20 fields; 4 new). The projection trio (`projected_tick_fractions`, `scale_padding_frac`, `include_minor`) is one concept with a prose-only invariant — group into an `Option<TickProjection>`, which also deletes the redundant `include_minor` gate. `layout/axis.rs` + `prepare.rs`. | **Fixing on this branch** |
| R3 | S2 | `build_grid` has four structurally-identical emission loops (x/y × major/minor) — extract one `emit_gridlines` helper. `render/marks/axis.rs`. | **Fixing on this branch** |
| R4 | S1 | Stale `render_svg`/`render_png` `theme` param docstrings in `binding.rs` — abbreviated key list, never listed the full §3.13 set, now also omits the per-level grid keys. Real contract is `ThemeOverridesSpec` (`deny_unknown_fields`). | **Fixing on this branch** (one-line pointer) |
| R5 | S3 | **WASM colorbar-in-inset id collision** — JS `_buildIdMap` namespaces by literal id per loadScene, so an outer colorbar gradient (`ferrum-colorbar-0`, hardcoded in `legend.rs:58`) and an inset chart that *also* has a colorbar collapse to one namespaced id → the inset's (or outer's) colorbar renders with the wrong gradient. Narrow trigger (continuous-color chart + `.inset()` of another continuous-color chart) but a genuine wrong-render. The only confirmed correctness issue across all reviews. | ✅ **Fixed `0b57b61`** — `legend.rs` merges the colorbar `<defs>`+`<rect>` into one self-contained Raw fragment (removing the only cross-fragment id ref; static SVG byte-identical); `ferrum-anywidget.js` switches to per-fragment id namespacing (`ferrum-raw-{loadIdx}-{fragIdx}-{id}`), now collision-free + reference-complete. Regression test pins old-collapses-to-1 vs new-2-distinct. |
| R6 | S2 | Interactive text-vs-raw z-order flips on the first zoom tick (`_placeTextSvg` re-append moves labels above the raw overlay groups). Cosmetic; rarely-overlapping content. `ferrum-anywidget.js`. | Deferred — follow-up |
| R7 | S2 | Discretizing *positional* scales (`Quantize`/`BinOrdinal`/`Sequential`/`Diverging` declared on an x/y axis) resolve to `ScaleKind::Linear` in `positional.rs` before `minor_ticks_internal` is reached, so they would get *linear-subdivided* minors rather than empty. Unreachable in practice (these are color/size specs); semantic corner only. | Deferred — add a clarifying comment when next touching `positional.rs` |

---

## D6 reactive-parameter runtime (2026-06-01, `feat/flexibility-new-capabilities`)

D6 (reactive parameters) shipped complete (sub-tasks 5a–5e-2b). One pre-existing render-layer limitation surfaced and is recorded here; it is **not** a D6 regression.

| ID | Severity | Item | Disposition |
|---|---|---|---|
| D6-1 | S3 | **Multi-panel simultaneous reactive rescale** is bounded by the single-transform-uniform render path: `render::upload_transform_and_render` uploads one transform uniform per frame and renders the whole frame, so when a brush drives reactive rescale, only one bound target panel's affine takes visible effect at a time (the same constraint that limits `setTransform` to panel 0). Single-panel overview→detail rescale works. A strict multi-panel rescale needs per-panel transform uniforms in the GPU render layer. | Deferred — render-layer change; out of D6 scope. Reactive rescale, crossfilter, and legend toggle all work for the single-target case validated by the audit's blocked designs. |

Note: the packed-batch field-value-point-selection gap (legend toggle on ≥1000-mark batches) that a review flagged was **closed** in 5e-2b (`09ba4f7`) via `scene_load::tooltip_field_value` — packed batches carry `tooltip_bytes`, so field-projected point selections now match on packed marks. No follow-up needed there.

---

## Flexibility re-audit fix campaign (2026-06-02, `feat/flexibility-new-capabilities`)

Re-ran `/audit-flexibility` after D1-D10 + D6; it confirmed 6 of 10 baseline defects closed and surfaced fresh ones. Fixed this campaign (each gated + regression-tested):

| Fix | Commit | What |
|---|---|---|
| G-D6 | a00126d | `fm.when(sel).then(num).otherwise(num)` no longer a silent no-op; channel taken from the encode key; `encode(<ch>=ConditionalSpec)` works; numeric value → opacity default |
| G-D7 | 9aacabf | radial bars stack outward (`Radius(stack=)` honored); `_normalize_stack` shared across X/Y/Theta/Radius fixes a latent `stack=True` PyO3 crash |
| T9 | 45e047e | `transform_top_k` aggregates integer columns instead of silently counting (was Float64-only) |
| T10 | cccf36d | `mark_violin` honors color/hue (per-(x,hue) KDE, overlaid) instead of silently collapsing |
| T11 | 834f126 | `mark_area` splits by `detail=` and non-nominal/ordinal color (was Utf8-color-only collapse) |
| T12 | 8b15f74 | per-layer `aggregate=` no longer dropped when layered (named chart-level Aggregate + data_source routing; both disjoint and column-overlap `__add__` paths) |

### New follow-ups surfaced by the audit (open, not yet fixed)

| ID | Severity | Item | Notes |
|---|---|---|---|
| FA-1 | S3 | **`mark_arc(theta:N, radius:Q)` Nightingale coxcomb renders blank** | falls into the pie path which `col_as_f64`'s the nominal theta → empty. The idiomatic coxcomb path is `mark_bar`+`CoordPolar` (fixed in G-D7); `mark_arc` should either render equal-band value wedges or raise, not silently blank. `arc.rs` build gate + the Python polar dummy-y remapping. |
| FA-2 | S2 | **Polar-bar angular layout is not equal full-circle bands** | G-D7 visual check showed 2-category coxcombs render as two narrow "petals" in the upper arc rather than equal wedges filling 360° (the "value-driven-angle" geometry the categorical agent flagged). Radial stacking is correct; the angular band-scale/extent under `CoordPolar` needs a look (`bar.rs build_polar` angular bands / polar band-scale padding/extent). |
| FA-3 | S3 | **Rust `stat_aggregate` rejects Int64 groupby** (Float64/Utf8 only) | affects both single-chart and layered aggregate; an Int64 x/groupby column errors. `crates/ferrum-core/src/transform/` aggregate path. |
| FA-4 | S3 | **Per-layer `bin=` has the same never-run gap T12 fixed for aggregate** | the layered path now resolves per-layer aggregates into named transforms but NOT per-layer `Bin` sentinels (`_layer_pending_aggregates` keeps only `_PendingAggregate`). A layer with `bin=` encoding silently isn't binned. Same named-transform+data_source fix pattern applies. |
| FA-5 | S1 | **Ordinal/quantitative-color `mark_area` legend swatch ≠ fill** | T11 inspection: legend swatches show categorical colors (blue/red/gold) while area fills use a sequential-ish ramp; the areas split correctly (the T11 fix) but the legend/fill color source for ordinal area diverges. |
| FA-6 | S1 | **violin box-inner layers don't color-encode while quartile/point do** | sibling asymmetry from `desugar_boxplot`'s layer contract (its layers never color-encode); cosmetic under the T10 overlay. |

The larger remaining frontier (annotation categorical-axis anchoring, typed-Scale domain auto-inference, shared legend across Repeat/concat, 2-D `mark_density` hue, custom `fm.Gradient` palettes, gridded `contourf`, flow geometry) is catalogued in the audit synthesis (`/tmp/ferrum-ux-audit/SYNTHESIS.md`, section C-D).
