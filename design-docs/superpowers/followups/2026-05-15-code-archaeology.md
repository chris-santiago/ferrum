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
| B4 | `src/ferrum/chart.py:2517` + `src/ferrum/_render.py` | `Chart.override(**kwargs)` stores kwargs in `self._overrides`, but **nothing in the render pipeline ever reads that dict** — every override key (`x_axis_label_angle`, `width`, `legend_orient`, …) is silently dropped. Fully documented feature (`docs/.../concepts/override.md`: validation via `FerrumOverrideError`, deprecation routing, 6-level cascade) — **none implemented**. `tests/test_override.py` only asserts dict storage, never rendered output, so CI stayed green. | ✅ Fixed `cd1ce00`+`0cea2b7` (`fix/override-wiring`) — override now applies at render, wins the cascade, and fails loud (`FerrumOverrideError` + did-you-mean). Presentation-spec scope; Python-side validation registry generated from live schemas. 62 render-level regression tests; full suite green; no golden churn. RCA/design/plan: `2026-06-14-override-*` |
| B5 | `crates/ferrum-core/src/spec/encoding.rs:285-296` + `crates/ferrum-core/src/render/prepare.rs` | Per-channel `encode(x=fm.X("f", axis={...}))` / `fm.Axis(...)` / `fm.Legend(...)` cross as opaque `AxisSpec.extra`/`LegendSpec.extra` (`serde_json::Map`, zero named fields). `prepare.rs` hand-reads only ~9 axis keys (`labels`, `ticks`, `domain`, `grid`, `label_angle`/`labelAngle`, `title`, `tick_count`/`tickCount`, `labelFormat`, `labelFormatType`) and ~14 legend keys (the D13 `color_legend_extra` block: `orient`, `direction`, `columns`, `title_font_size`, `label_font_size`, `gradient_length`, `gradient_thickness`, `tick_count`, `values`, `type`, `title`, `disabled`, `format`, `tickLabels`). `fm.Axis` advertises ~32 params / `fm.Legend` ~26; the rest (e.g. `label_color`, `grid_color`, `domain_width`, `symbol_size`) **silently drop per-channel** even though the typed `configure_axis`/`configure_legend` → `AxisConfigSpec` path already renders them. Same silent-drop archetype as B4. | ✅ Fixed (`fix/per-channel-axis-legend`, 10 commits `c2148e6`..`abbeffe`) — typed shared `AxisStyleSpec`/`LegendStyleSpec` (`deny_unknown_fields`, fail-loud) routed into the chart-level consumer at per-channel-wins; **every** advertised `fm.Axis`/`fm.Legend` field now renders (orphans + residual read-but-unrendered fields + the `fm.Axis` default-mismatch); `.to_dict()`-only tests upgraded to render-level + per-channel/chart-level parity tests; phase-boundary `rust-review` cohesion fixes (orient `Option` sentinel, `AxisStyleOverrides` bundle). No golden churn. RCA `2026-06-14-per-channel-axis-legend-silent-drop-rca.md`; post-mortem `2026-06-14-silent-drop-postmortem.md`. Follow-up: chart-level `AxisConfig` lacks `label_flush` (per-channel only). |
| B6 | `crates/ferrum-core/src/render/chart_config.rs` (`ChartConfig`) + `crates/ferrum-core/src/transform/*.rs` (`*Spec`) | Neither `ChartConfig` (top-level chart-config keys) nor the transform `*Spec` structs carry `#[serde(deny_unknown_fields)]`, so a misspelled key on a path that bypasses the Python-side override registry (e.g. a raw `transforms_json` dict, or a future chart-config dict) silently drops. No active bug today (the override registry validates Python-side; `transforms.py` emits exact keys), but it is the same latent silent-drop class as B4/B5. **Deliberately deferred** (2026-06-15 audit): adding `deny_unknown_fields` risks the intentional `#[serde(flatten)]` leniency the chart-config path depends on; needs care. | 🔍 Found (2026-06-15 PyO3 audit), deferred — defense-in-depth, no active bug |
| B7 | `crates/ferrum-core/src/render/marks/legend.rs:24-28` | Legend ENTRY labels fall back to `theme.colors.font_color` (`#1F2937`); axis tick labels fall back to `theme.colors.label_color` (`#6B7280`). So `Theme(label_color=...)` changes axis labels but NOT legend labels — "label color" doesn't govern the legend surface. Cross-surface coherence inconsistency, currently a documented design choice. **Deliberately deferred** (2026-06-15 audit): changing legend labels to use `label_color` would churn every legend's label color (golden churn) and is a product/coherence call, not a clear bug. | 🔍 Found (2026-06-15 scene/theme audit), deferred — coherence decision + golden churn |

> **2026-06-15 audit follow-ups:** three audits (PyO3 boundary, scene pipeline, theme cascade) over the B5-touched surface found 6 silent-drop/precedence bugs of the same archetype the post-mortem describes — all fixed on `fix/audit-followups`: the `configure(axis_x=)`-clobbered-by-`configure(axis=)` precedence inversion (NEW B5 regression), per-axis `label_font_size` ignored by layout (NEW B5 regression), `configure_axis(x/y)` vestigial (deprecated → `Chart.axis()`), `configure_title(subtitle_*)` drop, `Legend(label_color)` colorbar drop, and `clustermap(z_score=1)` int crash. Plus fixable WARNs (per-legend `label_font_size` render; W4 doc narrowed to raw-only; theme docstrings). B6/B7 above are the deferred WARNs. Reports: `.claude/output/audit-{pyo3,scene-pipeline,theme}/2026-06-15-audit.md`.

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

> **Update (2026-06-15, GitHub issue #1):** the figure-chrome *positioning* gap is fixed for the composites that emit chrome. The Rust emitter (`render/figure_chrome.rs`) now computes title/subtitle/caption `x` + `text-anchor` from `left_inset`/`right_inset`/`ChromeAnchor` (default `16.0`/`start`, was hardcoded `x=0`), and the Python side (`src/ferrum/_chrome.py` → `composition.py` + `_render.py`) resolves those from the merged configure dict so `configure_padding(left/right=)` and `configure_title(anchor=)` reach the band. Wired into the 3 chrome-emitting compose sites (HConcat, VConcat, ConcatChart grid) plus both single-chart caption `compose_svg_vertical` calls. **Forward-facing caution:** `JointChart` (`composition.py:933`), `RepeatChart` (`:1364`), and `ClusterMapChart` (`:1606`) do **not** pass a figure `title`/`subtitle`/`caption` to their `compose_svg_grid` calls at all — they route titles through a different (working) path, so there is **no silent drop today**. But if anyone later adds figure `title=` to those 3 grid sites, the chrome would reappear at `x=0` unless they also thread `**chrome_kwargs(merge_configure_layers(getattr(self, "_configure_layers", None)))` into the call, matching the 3 wired sites.

> **Open (noted 2026-06-01, flexibility D7):** `build_polar` in `render/marks/bar.rs` (polar/coxcomb bars) emits `tooltips: None`/`hrefs: None` and applies only flat `mark_style.opacity`, whereas the arc annular path (`build_annular`) wires per-row tooltips and per-row opacity. So a polar `mark_bar` with `tooltip=`/per-row `opacity=` silently loses them. Out of D7's geometry scope; wire to match `build_annular` when polar bars are next touched. Also: the polar channel-mapping convention (theta="x"→radius=y etc.) is duplicated between `arc.rs` and `bar.rs` — extract a shared `polar_channels(ctx)` helper if a third polar mark lands.

> **✅ Resolved (2026-06-19, GitHub issues #6/#7/#8, branch `fix/archaeology-bugs-6-7-8-class`):** all three fixed as defect **classes**, not the single instances the issues named. A prior attempt (deleted branch `fix/archaeology-bugs-5-8`) fixed one instance of each and passed every cheap gate while leaving the class live — see the session post-mortem `.claude/output/2026-06-19-session-postmortem-agentic-coding.md`. This pass scoped the full surface up front (spec `design-docs/superpowers/specs/2026-06-19-archaeology-bugs-6-7-8-class-fix-design.md`).
> - **D7 / #6 (metadata/node misalignment):** not one builder — a class across **14 mark builders**. Introduced a `MarkNodes` node+index accumulator (`render/mark_nodes.rs`) so a node cannot be emitted without its source row, with a construction-seam `debug_assert` guarding `nodes.len() == metadata.len()` for each of the tooltip/href/description channels. Migrated bar (5), rect (3, + heatmap rect/text), point (Cross 2-nodes/row — misaligned even with **zero** skipped rows), segment/text/tick/rule (all modes), and the group marks area/line/ribbon/polygon (representative-row). `build_metadata(ctx)` (full-row) deleted. Commits `cb10548`, `782edcf`, `22f5fe0`, `0d9b00b`, `4ffdaed`, `b5f56c0`. Extracted shared `polar_radius_scale` (the duplication this note flagged).
> - **N1 (packed-tooltip corruption >1000 nodes):** *folded into #6* — it is the packed/WASM face of the same misalignment. Fixing the builders to node-order metadata makes the existing `get_tooltip(node_idx)` correct; **no WASM change** (the investigated `data_indices[node_idx]` WASM lookup was explicitly rejected). Packed-path regression tests added. Commit `5324ae3`.
> - **D2 / #7 (faceted Bin/Violin extent drift):** generalized `fix_kde_extents_for_facet` → `fix_transform_extents_for_facet` (`render/prepare.rs`) to dispatch over Kde/Bin/Violin via transform-layer `global_extent` helpers, over the **full pre-facet dataset**, covering the **multi-group/hue** case the old `groupby.is_some()` early-return blocked; also fixed a latent int-column blindness. `ViolinSpec` gained `extent`/`shared_extent`. Bin nices; KDE/Violin pin to the raw global min/max (regression-guarded). Commits `4783bcd`, `5c72e64`, `8d4d74a`; e2e goldens commit `32ad11a`.
> - **D10 / #8 (Joint/ClusterMap/Repeat title → inner panel):** consolidated figure-chrome into `_CompositeBase` and reparented JointChart/ClusterMapChart/RepeatChart onto it (`.properties(title=)` now stores figure-level, never reaches inner panels); threaded chrome into the 3 grid `to_svg` sites (**this closes the forward-caution above** — those sites now pass `**chrome_kwargs(...)`); added full interactive on-canvas parity via a Rust `figure_title_nodes` helper (no WASM change) + a single shared `_inject_figure_chrome`; fixed `to_html` to read the figure title; closed the factory `properties={}` dict-path split. Caption absolute-y matches SVG for the concat family; Joint/ClusterMap caption-y is gated on the pre-existing **W5** interactive-layout limitation (title/subtitle parity is exact). Commits `c3d1b56`, `c4b1716`, `e72b25a`, `d1efe2c`.
> - **N2 (selection `stroke_width` whole-scene rejection):** **could not be reproduced** from source — the contract is sound end-to-end (`ChannelName::StrokeWidth`/`EncodingValue::StrokeWidth`, matching snake_case serde, conditional application, existing W7 tests). Dropped as not-real (the post-mortem itself overclaimed it); no code change.
> Full suite after: **pytest 5688 / 0; cargo 2427 / 0.**

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
| `Key` | ⚠️ Partially fixed 2026-08-27 (P1 remediation) — **Python-side plumbing done**: promoted from the (now-renamed) silent bucket to `_RENDERER_HONORED_CHANNELS`; the Rust wire already existed (`ChartSpec(key=...)`, `scene_build::extract_keys`), Python simply never passed it. Reaches `MarkBatch.keys` in the rendered scene (`crates/ferrum-scene/src/types.rs:110`) on both the chart-level and layered paths, static and interactive scene JSON. **Open sub-item: no consumer.** `MarkBatch.keys` is written by `scene_build.rs:1126` and read by nothing — `svg_walk.rs`, `ferrum-wasm/src/lib.rs`, `selection_state.rs`, `hit_test.rs`, and `spatial_index.rs` all only ever construct `keys: None` or ignore the field. Static SVG is byte-identical with and without `key=`; the WASM runtime never reads it. A visual/identity consumer (e.g. transition-matching on data updates, the original motivating use case per `Key`'s docstring) is unbuilt — this is a real, tracked gap, not a doc nit (quality-review finding, cycle 2: the row was previously closed outright, which is how this became an untracked, undiscovered gap for a second time). |
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
| R7 | S2 | Discretizing *positional* scales (`Quantize`/`BinOrdinal`/`Sequential`/`Diverging` declared on an x/y axis) resolve to `ScaleKind::Linear` in `positional.rs` before `minor_ticks_internal` is reached, so they would get *linear-subdivided* minors rather than empty. Unreachable in practice (these are color/size specs); semantic corner only. | ⚠️ **"Unreachable in practice" was wrong for `Diverging`** — a 3-element `DivergingScale(domain=[lo, mid, hi])` on x/y *is* reachable and truncated the axis to `[lo, mid]`, silently dropping marks above mid (issue #40, fixed 2026-07-10). The deferred clarifying comment is being added to `positional.rs` as part of that fix. Regression guard: `tests/test_scale_spec_parity.py::TestPositionalExtent::test_diverging_positional_all_marks_render`. |

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

### Phase C campaign (2026-06-02) — FA-1..FA-6 + cross-cutting consistency gaps RESOLVED

All FA follow-ups and five cross-cutting synthesis items (SYNTHESIS §C-D) fixed on `feat/flexibility-new-capabilities`, each gated (spec/quality + review-lite) and regression-tested; render changes visually inspected. Full suite after: **pytest 5293 passed / 0 failed; cargo all suites 0 failed.**

| ID | Commit | Resolution |
|---|---|---|
| C1 | 2bfe629 | `title=None` AND `Axis(title="")` truly suppress (no reserved margin, no phantom `<text>`); empty title resolves to None at the prepare boundary (Python forwards `""`, Rust skips the field fallback). |
| C2 | 7dfb0c9 | annotations anchor to categorical/ordinal axes (non-ISO strings → ordinal category coords, not force-parsed temporal); `fm.annotate_*` accept `fm.px`/`fm.norm`; unresolved category warns before center-fallback. |
| C3 | d731f07 | typed continuous scales (Linear/Log/Pow/Sqrt/Symlog) accept optional `domain` and auto-infer from data like the dict form. |
| C4 | 3a2ee59 | `resolve=` on `vconcat`/`hconcat`; `pairplot(hue=)` shares one color domain. **Residual resolved (2026-07-12, [#16](https://github.com/chris-santiago/ferrum/issues/16)):** the compositor renders one figure-level legend band when color/size resolve shared (per-panel legends suppressed at layout; band reuses the panel legend primitives); `fm.Resolve(scale=, legend=)` ships the opt-out, `jointplot(hue=)` opts in, and the fix also repaired hole-cell grids (`corner=True`, jointplot) silently skipping domain union. |
| C5 | 5e2dd63 | 2-D density splits by categorical hue (`Kde2D groupby` → per-group surfaces; `Contour` iterates surfaces; `jointplot(kind='kde', hue=)` + `mark_contour(groupby=)`). **Note:** grouped contours render as isolines colored by group; filled per-group isobands are blocked by per-group `level_id` collision (group A and B both start `level_id=0` → polygon renderer merges them). |
| FA-1 | 059a050 | `mark_arc(theta:N, radius:Q)` renders an equal-band Nightingale coxcomb (was blank). |
| FA-2 | 2480b08 | polar bars render equal full-circle angular bands (was narrow upper-arc petals — root cause a double polar transform on `MarkBatchKind::Arc` geometry). |
| FA-3 | 10ecc4a | `stat_aggregate` accepts integer/uint/bool groupby (KeyValue gained Null/Int/UInt/Bool; output preserves the key dtype). |
| FA-4 | 803d753 | per-layer `bin=` resolved via named transform + data_source routing (mirrors T12); `bin=Bin(...)` kwargs preserved; bin+aggregate on one layer raises rather than clobbering. |
| FA-5 | 4459c29 | ordinal/quantitative-color `mark_area` legend swatches match fills (`build_color_scale` forces categorical for `Mark::Area`). |
| FA-6 | 54da54d | violin/boxplot box-inner layers color-encode by hue (also fixed standalone `mark_boxplot` color threading). |

### New follow-ups surfaced DURING the Phase C campaign (open)

| ID | Severity | Item | Notes |
|---|---|---|---|
| FA-7 | S3 | **RESOLVED (480f72f)** — Int64/uint/bool groupby was rejected by 4 sibling transforms | `violin.rs`, `summary.rs`, `error_extent.rs`, `box_stats.rs` each carried a private `KeyValue` enum (Str+Float only). Extracted canonical keying into `transform/group_key.rs` and migrated all 5 transforms (incl. aggregate); int/bool groupby now works in violin/boxplot/errorband/summary; Float64/Utf8 byte-stable (zero golden movement). |
| FA-8 | S2 | **C5 grouped contours are isoline-only** | filled per-group isobands need globally-unique `level_id` across groups (namespace `level_id` by group index in `contour.rs`); until then `desugar_contour(groupby=)` forces `fill=False`. |
| FA-9 | S1 | **Int64 null groupby key materializes as `0`** | `aggregate.rs materialize_groupby_col` emits `0i64` for a null integer key, which collides with a genuine `0` key (null float → NaN is unambiguous; null int → 0 is not). Pinned/documented by a regression test; emit a proper null instead. |
| FA-10 | S2 | **typed-Scale sentinel+flag is two sources of truth** | C3 stores a sentinel domain + `domain_user_set` bool; crate-internal `_internal` accessors read the sentinel directly. Safe today (only called on data-derived scales) but latent; an `Option<[f64;2]>` inner domain would make the unset case unrepresentable. (Found by C3 quality review.) |

**Resolved from the prior frontier:** annotation categorical-axis anchoring (C2), typed-Scale domain auto-inference (C3), shared color domain across concat/pairplot (C4, legend-dedup residual remains), 2-D `mark_density` hue (C5). **Still open in SYNTHESIS §C-D:** `title=None` was C1 (done); custom `fm.Gradient` continuous palettes; gridded `contourf`/`pcolormesh`/`quiver`; flow geometry (Sankey/variable-width trail); recursive treemap/icicle rectangling; public `mark_polygon` for half-violins/raincloud.

### Structural pass (2026-06-02) — break the fix-exposes-next-instance recursion

A 4-agent duplication + silent-failure audit (Rust transforms, Rust render/marks, Python) found the *mechanisms* behind the recurring bug classes: (1) copy-paste dispatch reimplemented per file (fix one, N drift), (2) transform→render layer coupling (new data shape the renderer never handled), (3) silent-failure as default. Findings were **verified before fixing** — several audit claims were false positives (see below). Fixed the confirmed subset; deferred the risky/unconfirmed with honest status. Full suite after: **pytest 5336 / 0 fail; cargo 2206 / 0 fail.**

| ID | Commit | Resolution |
|---|---|---|
| S1 | (errorbar hue) | `mark_errorbar`/`mark_errorband` computed the error extent pooled across hue groups (`error_extent` groupby `[x]` only) while coloring bands per group — same silent-wrong class as T10/FA-6. Threaded color into the groupby + layer encodings; extracted shared `marks/_desugar_helpers.resolve_color_groupby` so boxplot/violin/errorbar/errorband share one color-threading path (also drops `None` keys — a latent bug). |
| S2 | (shape/offset raise) | `mark_point(shape='typo')` silently drew a circle; `transform_stack(offset='bogus')` silently stacked at zero. Both now validated at the Python API boundary (clear `ValueError`); `data_stack.rs` raises on unknown offset; `point.rs` keeps a non-panicking circle fallback for raw-JSON specs (Python is the guard). |
| S3 | (numeric_util) | Unified byte-identical `clean_float64_values` (bin/kde/smooth) and `quantile_sorted` (qq/letter_value) into `transform/numeric_util.rs`. Pure drift-prevention, byte-stable. |

**Audit false-positives caught by verify-before-fix (NOT bugs):** `smooth.rs` "missing NaN filter" — `extract_xy:500` already filters NaN (audit read the wrong lines); `mark_line` integer color — works (6/6 distinct strokes verified); `mark_ribbon` integer color — splits correctly; legend-swatch≠fill beyond FA-5 — `legend.rs` and `area.rs` already use the same `color_scale.lookup`. Chasing these would have *been* the recursion.

**Deferred (recorded, intentionally NOT fixed — drift-prevention or unconfirmed, not active bugs):**
| ID | Severity | Item |
|---|---|---|
| FA-11 | S2 | **Opacity resolution duplicated + drifted across 5 marks** (point/bar/area/line/rect): per-row vs group-first sampling, scale-applied vs not, `general_opacity` fallback only on bar, `fill_opacity` unread on line. Unifying needs design decisions (is opacity scaled? per-row in grouped marks?) and risks behavior change — no confirmed active bug today. A shared `OpacityChannels` resolver would prevent future drift. `render/marks/{point,bar,area,line,rect}.rs`. |
| FA-12 | S3 | **Group-partition+stack duplicated across bin/kde/kde_2d/smooth** (`apply_grouped`, ~4 near-identical bodies; extent logic already subtly differs). FA-7 unified the *keying*; this is the row-partition + per-group output stacking. Big refactor, drift-risk, no active bug. |
| FA-13 | S3 | **Color/detail grouping duplicated across area/line/ribbon** (`col_as_ordinal_category_str` grouping). Verified working at runtime (line/ribbon int-color split correctly), so drift-prevention only — extract `build_color_detail_groups`. |
| FA-14 | S2 | **`prepare.rs:142` StringView columns bypass string normalization** (`_ => {}`). Potential silent skip; needs a probe to confirm StringView even reaches this path (polars/pyarrow usually yield Utf8). Verify-then-fix-or-close. |

## v0.15.0 post-release audit + v0.15.1 remediation (2026-06-02/03, `fix/v0151-audit-remediation`)

After shipping v0.15.0 (the merged flexibility campaign), a 7-agent audit ran on the `v0.14.0..v0.15.0` diff: 2 bug-hunters (stat transforms, params), 3 seam auditors (scene-pipeline, pyo3-binding, interactive), and 2 heavyweight cohesion reviews (rust, python). The dominant theme was **incomplete unification + sibling drift + silent failure** — two of the three High bugs were unification jobs left half-done. All confirmed bugs fixed on `fix/v0151-audit-remediation` via subagent-driven-development (each bug has a failing-then-passing regression test). Final: **pytest 5445 / 0 fail; cargo all binaries 0 fail; ruff clean.**

| ID | Commit theme | Resolution |
|---|---|---|
| FA-7 (extended) | `2bea218` | The group-keying unification was **incomplete** — `data_window`, `data_stack`, `data_aggregate`, `join_aggregate`, `pivot` still used private String/Float64-only `extract_key` (int/bool groupby silently collapsed to one partition; pivot/data_aggregate also crashed `RecordBatch::try_new` on a declared-vs-actual dtype mismatch). All migrated to `group_key::groupby_key_at` + `materialize_groupby_col`; `density_data`/`bin`/`swarm` now accept int/bool too; `impute` deduped into `numeric_util`. |
| **FA-9** | `2bea218` | **RESOLVED.** Null groupby key now materializes as a distinct Arrow null (+ a positional-only "null" axis band when `null_count()>0`), no longer colliding with a real `0`/`false`/`""`. Groupby output fields made nullable across **all 9** callers — `summary`/`box_stats`/`error_extent`/`violin` had been left non-nullable and would crash `RecordBatch::try_new` on a null key (caught by the cohesion review, not the green suite). Color/shape/legend domains still drop nulls (no golden movement). |
| REN (channel drift) | `2bea218` | `mark_bar` polar (`build_polar`), `mark_rect` range paths, and `mark_arc` (`build_nominal_theta`/`build_annular`) silently dropped channels their siblings honor (opacity/stroke_*/fill_opacity/angle/tooltip and href/description). All now load the shared channel helpers; arc metadata aligned to surviving wedges by `data_indices` (skipped-wedge safe — the `meta.build_metadata` refactor had introduced a node↔row misalignment). |
| TITLE (Axis+Legend) | `09f10eb` | `Axis(title=None)` and `Legend(title=None)` did not suppress (value-object `to_dict` dropped `None` keys while the channel serializer maps `None`→`""`). Shared `serialize_title` helper + Rust legend layout treats `""` as suppress (no phantom `<text>` node). |
| PARAMS (namespace) | `d9de07f` | `add_params`/`add_selection` now raise `TypeError` on wrong-typed args (was silent-drop / late `AttributeError`); a `Selection` + same-named `VariableParameter` now raises a `ValueError` collision instead of the selection silently shadowing the variable's domain. |
| INT-1 (cross-panel rescale) | `35a231f` | **Critical.** `apply_reactive_rescale` built the affine in source-panel pixel space and applied it to target-panel marks → overview→detail brush sent detail marks off-screen. New `rescale_affine_cross_panel` reprojects source-px → shared data domain → target-px (via the existing `reproject_extent` `apply_crossfilter` uses) before building the affine. Single-panel self-rescale preserved. Shipped WASM bundle rebuilt. |
| LOWER BUCKET | `85a51cc`, `6078907` | Inf/NaN in an `fm.param` domain → legible `ValueError` (not cryptic serde); `add_params(selection)` also registers the selection so a `bind="legend"` toggle works; `_core.pyi` `ChartSpec` stub expanded 15→35 kwargs; `mark_area x2` / `mark_bar x2+y2` raise a clear error pointing to `mark_rect` (via centralized `validate_mark_encoding`) instead of silent drop. |

**Still open (separately tracked):**
- **FA-15** — a color-CONDITIONAL encoding (`encoding.color=null` + `sel.when(Color(...))`) builds no legend, so a `bind="legend"` toggle has no categories to toggle. Browser-only-validated; not addressed here.
- **FA-16 — RESOLVED (v0.15.2, browser-validated 2026-06-03).** The line/area "ribbon" under the non-uniform reactive-rescale affine. Fix = un-bake stroke width (store centerline+normal+half_width per `MeshVertex`) + apply the offset in screen space + inverse-transpose direction correction + **bevel joins** (the decisive piece — lyon baked *miter*-elongated, scene-space bisector normals that a per-vertex shader can't recompute under `sx≠sy`; bevel emits unit normals so the screen-space offset is correct at joins). Commits `049f0cd`, `352e802`, `ba51c6a`. Interactive line joins are now beveled; the static SVG renderer keeps miter. Spec: `design-docs/superpowers/specs/2026-06-02-wasm-relayout-rescale-design.md` (Option 3 re-layout remains the deferred broader fix for tick/label/resampling fidelity).
- **FA-17 — RESOLVED (`0e82b88`, browser-confirmed 2026-06-03).** A domain-rescale chart now defaults to brush/box-select mode (`defaultMode='select'` via `hasDomainRescale` from the `domain`-role `param_bindings`). The earlier "still pans" report was a stale bundle/export; the focus+context demo now arms the brush on first drag with no manual tool click. (The FA-16 root-cause audit's FA-17 re-trace confirmed the signal path Python→config→JS is intact.)
- **FA-18 — RESOLVED (`f44b72d`, browser-validated 2026-06-03).** The mesh/instance draw loops bound a single shared transform uniform once per frame, so during a reactive rescale every panel drew with the rescaled panel's affine → sibling shear + the overview line vanishing during the brush. Fix = per-panel transform slots indexed by a new `panel_id` on `MarkMeshPanel`/`DrawCommand`; both draw loops bind their own panel's affine, and the render uploads every panel's affine each frame. Also closed the latent concurrent-multi-panel corruption. Identity uniform (static/annotation/non-mark) unchanged; at-rest + single-panel byte-stable. Spec: `design-docs/superpowers/specs/2026-06-03-wasm-interactive-renderer-correctness-design.md`.
- **FA-19 — fix landed (`7ebfff2`, MSAA), awaiting final browser confirmation.** Root cause was NOT per-segment geometry (the axis line is one `SceneNode::Line` → one lyon stroke) but the mesh pipeline rendering with no MSAA (`sample_count=1`): abutting butt-cap quads (axis line vs tick marks, adjacent facet axis lines) left a 1px hairline gap/step with no edge AA. Interactive-only (static SVG antialiases). Predates FA-16 (confirmed byte-identical line handling); FA-16 could only shift which pixels show it. Fix = 4× MSAA across the whole main render pass (multisampled color target resolving to the surface view; all six pipelines share the sample count; rebuilt on resize), with a silent byte-identical 1× fallback when the backend lacks 4×. Same spec as FA-18.
- **FA-20 — RESOLVED (`1d76567`, browser-validated 2026-06-03).** Surfaced while validating FA-18: box-zooming the overview made the detail vanish. Root cause = the D3-zoom path (wheel/drag-pan/box-zoom) was global and hardcoded to panel 0 (`set_transform`→`set_absolute(0,…)`); pre-existing, made visible by FA-18's per-panel transforms. Fix (focus+context semantics) = `set_transform` gains a `panel_id`; the JS resolves the focus panel from the `domain`-role binding and targets it for wheel/pan/box-zoom, and a box-zoom drawn on a context/overview panel routes through the rescale path (rescales the detail). Charts with no domain binding keep `focusPanel=0` — single-panel + generic multi-panel byte-identical. Spec: `design-docs/superpowers/specs/2026-06-03-focus-context-zoom-semantics-design.md`. (Generic-multi-panel independent per-panel zoom remains out of scope / deferred.)
- FA-11..FA-14 remain deferred (drift-prevention, no active bug).

---

## 2026-06-16 — Open-item triage → GitHub issues

Every still-open item in this doc (excluding the Phase 12 frontier: extension points + full
palette library) was re-verified against current code by 7 parallel read-only audits, then
filed as a GitHub issue. **Three were found already resolved and were NOT filed** — corrections
below.

**Found RESOLVED (no issue; this doc was stale):**
- **FA-14** — StringView/`Utf8View` (and `LargeUtf8`) are now explicitly converted to `Utf8` in `render/prepare.rs` `normalize_string_views` before the `_ => {}` catch-all; no silent skip.
- **D6-1** — superseded by **FA-18** (`f44b72d`): per-panel transform slots (`MarkMeshPanel.panel_id`) bind each panel's own affine and every panel's affine is uploaded per frame, so multi-panel simultaneous reactive rescale no longer shears siblings.
- **`fm.Gradient` custom continuous palettes** (SYNTHESIS §C-D) — already public: `fm.Gradient(stops)` is exported in `ferrum.__all__` (`src/ferrum/schemes.py` → Rust `continuous.rs`). Only a usage doc is missing.

**Filed as issues (ID → #):**

| Issue | ID | Kind |
|---|---|---|
| [#2](https://github.com/chris-santiago/ferrum/issues/2) | B6 | enhancement (deny_unknown_fields on transform Specs) |
| [#3](https://github.com/chris-santiago/ferrum/issues/3) | FA-10 | enhancement (scale sentinel+flag → Option) |
| [#4](https://github.com/chris-santiago/ferrum/issues/4) | B7 | enhancement (legend label_color coherence) |
| [#5](https://github.com/chris-santiago/ferrum/issues/5) | FA-11 | **bug** (fill_opacity unread on mark_line) |
| [#6](https://github.com/chris-santiago/ferrum/issues/6) | D7 | **bug** (polar bar tooltip/href misalign) |
| [#7](https://github.com/chris-santiago/ferrum/issues/7) | D2 | **bug** (faceted Bin/Violin extent drift) |
| [#8](https://github.com/chris-santiago/ferrum/issues/8) | D10 | **bug** (Joint/ClusterMap title → inner panel) |
| [#9](https://github.com/chris-santiago/ferrum/issues/9) | FA-15 | **bug** (conditional-only color → no legend) |
| [#10](https://github.com/chris-santiago/ferrum/issues/10) | R6 | bug (interactive label z-order, cosmetic) |
| [#11](https://github.com/chris-santiago/ferrum/issues/11) | FA-13 | enhancement (color/detail grouping dup + ribbon int risk) |
| [#12](https://github.com/chris-santiago/ferrum/issues/12) | FA-12 | enhancement (group-partition+stack dup) |
| [#13](https://github.com/chris-santiago/ferrum/issues/13) | FA-8 | enhancement (grouped-contour level_id namespacing) |
| [#14](https://github.com/chris-santiago/ferrum/issues/14) | B5-followup | enhancement (chart-level label_flush) |
| [#15](https://github.com/chris-santiago/ferrum/issues/15) | Task 37 | enhancement (per-cell quantitative color contour/hex) |
| [#16](https://github.com/chris-santiago/ferrum/issues/16) | C4-residual | enhancement (figure-level deduped legend) — **resolved 2026-07-12** on `feat/composite-shared-legend` |
| [#17](https://github.com/chris-santiago/ferrum/issues/17) | #20 | enhancement (share_x/y dead-API cleanup) |
| [#18](https://github.com/chris-santiago/ferrum/issues/18) | R7 | documentation (positional discretizing-scale comment) |
| [#19](https://github.com/chris-santiago/ferrum/issues/19) | FA-19 | question (browser-verify MSAA fix; code landed) |
| [#20](https://github.com/chris-santiago/ferrum/issues/20) | feat | enhancement (gridded contourf/pcolormesh/quiver) |
| [#21](https://github.com/chris-santiago/ferrum/issues/21) | feat | enhancement (Sankey/alluvial/trail) |
| [#22](https://github.com/chris-santiago/ferrum/issues/22) | feat | enhancement (treemap/icicle) |
| [#23](https://github.com/chris-santiago/ferrum/issues/23) | feat | enhancement (public mark_polygon) |

Note on verification upgrades: **FA-11** was reclassified from drift-prevention to an active **bug** (`fill_opacity` is silently unread on `mark_line`). **B5-followup** is a Python-side gap only (Rust `AxisStyleSpec` already carries `label_flush`).

---

## 2026-06-19 — issues #6/#7/#8 class-fix + round-5 convergence (`fix/archaeology-bugs-6-7-8-class`)

The three GH-issue bugs were fixed **as defect classes, not instances** (per the session postmortem), then an unscoped review/audit sweep surfaced four more findings (A–D) that were also fixed. The branch ran an autonomous review→remediate loop (rounds 1–5).

**Issue bugs resolved (class-level, pending merge):**

- **#6 / D7** (polar bar tooltip/href misalign) → **RESOLVED as a class.** Root cause was builders constructing the node list with skips/fan-out/grouping while metadata stayed indexed by source row. Introduced `MarkNodes` accumulator (`render/mark_nodes.rs`) so a node cannot be added without its source row; migrated all 14 row-skipping / multi-node / group builders; added a construction-seam `debug_assert_nodes_metadata_aligned` guard over all 5 channels (tooltips/hrefs/descriptions/data_indices/keys). Also fixed `label.rs` leader-line multi-node misalignment and geoshape/image metadata-drop found by the round-1 sweep.
- **#7 / D2** (faceted Bin/Violin extent drift) → **RESOLVED as a class.** `fix_transform_extents_for_facet` (`render/prepare.rs`) now pins the pre-facet shared extent for **all** extent-carrying transforms: Kde, Bin, Violin, Kde2D, Bin2D, DensityData (each with a `global_extent` helper). `ViolinSpec.shared_extent` wired end-to-end (R6).
- **#8 / D10** (Joint/ClusterMap title → inner panel) → **RESOLVED.** Single chrome home in `_CompositeBase`; Rust `figure_title_nodes` PyO3 helper sharing `FigureChrome::layout` with the SVG path; `_inject_figure_chrome` injects title nodes + offsets children for every composite; factory-dict chrome split in `_overrides`.

**Round-5 unscoped-sweep findings (all fixed on this branch):**

- **A (CRITICAL)** — `_merge_packed_data` tooltip-table misparse (assumed a u32 length-prefix; real format has none) dropped/blanked panels for any interactive composite with a >1000-mark child. Fixed: scan the table field-by-field mirroring `scene_load.rs`. Corrected 3 tests that enshrined the wrong byte format.
- **B (HIGH)** — packed GPU instances never received the per-panel concat `(dx, dy)`, so packed marks rendered at the top-left child's coords while scissored to their own offset plot_area. Fixed: `_offset_packed_batch_xy` + per-child `child_xy_offsets` threaded through all 5 merge call sites.
- **C** — `_core.pyi` stub drift (wrong `Violin extent=`; ~19 classes + ~16 functions undeclared). Fixed + a bidirectional stub↔module parity test.
- **D** — Rust hardening: clamped the `scene_load.rs` tooltip-scan slice (latent panic), recorded the actually-loaded `instance_count` (no phantom GPU instances), unified `violin`/`density_data` `global_extent` to `coerce_to_float64` (int-field shared-extent gap).

**Known gaps recorded during round-5 P4 triage (pre-existing, not fixed here):**

| ID | Sev | Item |
|---|---|---|
| KG-1 | S2 | **`groupby` type asymmetry across sibling transforms** — `Violin.groupby` is `Vec<String>` (list) while `Kde`/`Kde2D`/`Smooth.groupby` is `Option<String>` (single). The `_core.pyi` stub faithfully mirrors this (`Optional[List[str]]` vs `Optional[str]`); the asymmetry itself is a Rust-API sibling-drift issue (same family as **FA-7**). Defer: unifying requires an API decision + Python-layer changes; no active bug (callers use the form each accepts). |
| KG-2 | S2 | **`global_extent` residual drift** — after F1, the six extent helpers split into a 1-D inline-fold family (kde/violin/density_data) and a 2-D extracted-helper family (bin_2d/kde_2d `raw_axis_extent`, duplicated verbatim) + bin's `raw_float64_extent`. The fold logic is now uniform; hoisting `raw_axis_extent` into `numeric_util` alongside `coerce_to_float64` would remove the last duplication. Defer: pre-existing, cosmetic, drift-prevention only. |
| KG-3 | S1 | **`MarkBatch.keys`** — built and carried through the packed/interactive path but consumed only by interactive linked-selection (interactive-only); confirm a consumer exists or mark dead. Overlaps the round-1 guard work (keys now in the alignment guard). |
| KG-4 | S1 | **per-mark `description`** — populated for SVG `aria`/`<desc>` but has no WASM/interactive consumer (SVG-only). Same family as the deferred geoshape/image/label **hover-tooltip** limitation (round-3 non-goal). |
| KG-5 | S1 | **dead WASM API** — `onWheel`/`onPan`/`resetZoom`/`selectInRect` exported but unused by the current JS loader (adjacent to issue **#17** share_x/y dead-API cleanup). |
| — | — | **`deny_unknown_fields` asymmetry** (mark_style/coord/facet/position silently drop unknown dict keys while title/axis/legend fail loud) is **already tracked as issue [#2](https://github.com/chris-santiago/ferrum/issues/2) (B6)** — not re-filed. |

---

## 2026-06-20 — convergence (rounds 7–12, `fix/archaeology-bugs-6-7-8-class`)

The autonomous review→remediate loop ran to convergence: a round-12 unscoped sweep (5 chris-code agents — rust/python quality + scene-pipeline/interactive/pyo3 audits) surfaced **zero in-class correctness defects**. Final suite: cargo `ferrum-core` 1603 + `ferrum-wasm` 405 + pytest 5774, all green.

**Classes closed (verified across rounds):** #6 metadata/node alignment (MarkNodes + 5-channel seam guard, all builders); #7 faceted transform-extent, **completed to all extent-DERIVING transforms** (Hex/Raster/DataBin added round-7 to the carrying set Kde/Bin/Violin/Kde2D/Bin2D/DensityData); #8 composite figure-title; round-5 findings A (packed tooltip-table misparse), B (per-panel packed dx/dy offset), C (`_core.pyi` stub drift → now guarded by a **programmatic** live-vs-stub signature-parity test), D (scene_load panic/instance-count hardening, violin/density `global_extent` coerce); the **faceted SHARED-SCALE class across EVERY data-driven channel** — positional x/y (round-7 T4, foundational) + categorical data-aware sort (round-9 T1) + continuous color/size/opacity (round-9 T3) + categorical color (round-9 T3 follow-up) + **shape** (round-11, the last channel); plus the pre-existing interactive **conditional/crossfilter packed-ordering** bug (round-7 T3, user-approved on-branch).

**New known gaps (pre-existing, out of the remediated classes — recorded, NOT fixed on this branch).** All four filed as GitHub issues (2026-06-20): KG-6 → [#24](https://github.com/chris-santiago/ferrum/issues/24), KG-7 → [#25](https://github.com/chris-santiago/ferrum/issues/25), KG-8 → [#26](https://github.com/chris-santiago/ferrum/issues/26), KG-9 → [#27](https://github.com/chris-santiago/ferrum/issues/27).

| ID | Sev | Issue | Item |
|---|---|---|---|
| KG-6 | S2 | [#24](https://github.com/chris-santiago/ferrum/issues/24) | **`facet(col=X)` defaults to `ncols=1`** → panels stack vertically instead of side-by-side (Altair/seaborn convention). Commit `226ba24` (2026-05-11), an established tested contract (`test_d2_facet.py:333` passes `ncols=3` explicitly). NOT a col/row swap — `row=`/`col=` both wrap with `ncols=1`. Fix = infer `ncols = n_distinct(col)` for col-only wrap (grid mode already does). Facet LAYOUT subsystem, not scale resolution. (`src/ferrum/chart.py:2822`.) **Default-layout change → user decision + golden blast radius.** |
| KG-7 | S2 | [#25](https://github.com/chris-santiago/ferrum/issues/25) | **`ShapeKind::Square` coordinate render bug** — square glyph marks can land outside the panel clip bounds in faceted charts (discovered during round-11 shape-test development; the test discriminates on `<circle>` cy to sidestep it). Rendering/coordinate issue in the point/shape mark path, not scale resolution. Pre-existing. |
| KG-8 | S1 | [#26](https://github.com/chris-santiago/ferrum/issues/26) | **shape encoding ignores `sort`** — `build_shape_scale` never honored `EncodingSpec.sort` (orders glyphs by first-appearance only). Pre-existing; sibling of the positional/color sort paths. |
| KG-9 | S1 | [#27](https://github.com/chris-santiago/ferrum/issues/27) | **pyo3 stub fidelity nits** — `EncodingSpec.condition` is a ctor kwarg with no readable getter (write-only); 5 stub defaults are concrete where the live signature shows `=...` (all verified to match the true runtime default, so the stub is *more* informative, not wrong). No caller impact. |

KG-1..KG-5 above remain open/inactive. The faceted shared-scale fix gates strictly on `ResolveMode::Shared` (the documented default) for positional channels and on `spec.facet.is_some()` for the non-positional channels (which have no per-channel independent option); `Independent`, explicit `Scale(domain=)`, and non-faceted output are byte-identical throughout.

---

## 2026-06-22 — cohesion-campaign discovered follow-ups (C1–C4)

Surfaced while executing the cohesion campaign (`fix/cohesion-campaign`, plan `2026-06-21-cohesion-campaign-plan.md`). These are NOT among the 193 campaign findings; they are out-of-scope discoveries logged here at the user's direction ("log all C"). The campaign's own behavior-change carries B1–B3 were *fixed* (not deferred); these C-items are genuine follow-ups. **All filed as GitHub issues (2026-06-22).**

| ID | Sev | Issue | Item |
|---|---|---|---|
| C1 | S3 | [#29](https://github.com/chris-santiago/ferrum/issues/29) | **`cargo clippy -p ferrum-core -D warnings` is RED at a ~180-error pre-existing baseline.** Dominated by a pyo3 0.28.3 `#[pyclass]` deprecation firing on every scale/spec `#[pyclass]` struct, plus warnings across `transform/`, `layout/`, `render/`. The toolchain advanced (rustc 1.95.0 + pyo3 0.28.3; date rolled 2026-06-21→06-22 mid-campaign). NOT introduced by the campaign — every Rust commit was verified to add **zero new** warnings on its touched files, but the crate-wide `-D warnings` gate cannot pass until a dedicated pyo3-deprecation-migration + clippy-cleanup pass runs. Highest-value C-item. |
| C2 | S2 | [#30](https://github.com/chris-santiago/ferrum/issues/30) | **Broad pre-existing pyright type-debt** surfaced on files the campaign edited (it re-reports all diagnostics in a touched file). Clusters: `_RepeatPlaceholder` leaking into `str`-typed returns/args in `chart.py`/`encoding/base.py`; the `_SourceState` mixin protocol gap in `_diagnostics/*` (`_cache`/`_y`/`_model`/`_X` "unknown attribute" — the FA-9-class Protocol the audit recommended); string forward-refs unresolved (`Title`/`HConcatChart`/`VConcatChart`); polars `Series`/`DataFrame` overload mismatches in `plots/*`. Runtime-fine (suite green). Candidate for a dedicated typing pass (add `_SourceState(Protocol)`, real forward-ref imports under `TYPE_CHECKING`). Outside the 193 findings. |
| C3 | S2 | [#31](https://github.com/chris-santiago/ferrum/issues/31) | **Scale-level `scheme=` not eagerly validated** (T2.2/D-COLOR-1 left it deliberately). `DivergingScale(scheme="redblue")` and `ColorConfig(scheme="category10")` use names **not in** `list_palettes()` yet currently resolve late without error. Either those are valid Vega aliases the Rust registry should expose (so `list_palettes()` is incomplete) or they are dead names. Resolve before extending declaration-time validation from the channel path to the Scale/Config path (needs a Rust pass). |
| C4 | S1 | [#32](https://github.com/chris-santiago/ferrum/issues/32) | ✅ **RESOLVED 2026-06-24** (`fix/cmap-vocab-holdouts-32`, see the 2026-06-24 section below). **`cmap`-vocabulary holdouts** `heatmap(cmap=)` (`plots/matrix.py`) and `RenderConfig.raster_cmap` (`render_config.py`) — named in XSIB-07's finding text but outside its fix scope (T2.2 unified mark_raster/contour/hex + clustermap). Both now take canonical `scheme=` with `cmap=`/`raster_cmap` as a back-compat alias, routed through the existing `resolve_cmap_alias` helper. |

C1 is the most worth doing (it restores a real CI-able gate). C2 is the largest. C3/C4 are localized. **C4 resolved 2026-06-24** (see below).

## 2026-06-24 — C4 / issue #32 resolved (`fix/cmap-vocab-holdouts-32`)

**C4 / [#32](https://github.com/chris-santiago/ferrum/issues/32) — the two `cmap`-vocabulary holdouts now take canonical `scheme=`. ✅ RESOLVED.**

Both holdouts route through the existing shared helper `resolve_cmap_alias(*, scheme, cmap, where)` (`marks/_desugar_helpers.py`) — the same one `mark_raster`/`contour`/`hex` and `clustermap` already use — rather than re-implementing the at-most-one rule:

- **`heatmap()`** (`plots/matrix.py`) gained a canonical `scheme=` param (with `cmap=` kept as the back-compat alias), resolved at the public seam via `resolve_cmap_alias(..., where="heatmap")` before the unchanged `_heatmap_build`, mirroring `clustermap`'s public contract.
- **`RenderConfig`** (`render_config.py`) now has a canonical `raster_scheme` field with `raster_cmap` as a back-compat alias. `__post_init__` resolves via the helper (default → `"viridis"`) and writes the resolved value to **both** fields, so `cfg.raster_cmap` reads back as the resolved scheme (no silent-`None` footgun — caught by the design-review gate, which first flagged an `InitVar` variant that left a misleading class-attribute `None`). The single reader (`_render.py`) reads `cfg.raster_scheme`. No Rust change (the Rust/dict key stays `"cmap"`). `palette=` was confirmed out of scope (it is a distinct categorical-list param on `mark_boxen`, not a `scheme`/`cmap` alias).

Verified: 6 regression tests in `tests/test_t2_2_palette_source_of_truth.py` (3 heatmap equivalence/conflict/back-compat + 3 RenderConfig), all proven RED with the source stashed; full close via `verification-before-completion`. Note: the RenderConfig tests were first mis-homed in the slow-gated `test_scale_rendering.py` (deselected by default → would have guarded nothing) and relocated to the not-slow palette source-of-truth module.

**New follow-up (open) — repo-wide deprecation-warning policy for `cmap`/`palette` aliases.** The whole codebase currently *silently* accepts the `cmap=` alias (the helper emits no `DeprecationWarning`; the at-most-one `ValueError` is the only feedback). Whether these aliases should eventually emit a `DeprecationWarning` is a single cross-cutting policy decision that should be applied to *every* aligned site at once (encodings + all four marks + clustermap + heatmap + RenderConfig), not smuggled into one. Deliberately not done here, to keep #32 consistent with its siblings.

## 2026-06-22 — cohesion-campaign Tier-4 deferred follow-ups (filed as issues)

Beyond-scope deferrals surfaced while executing Tier 4 of the cohesion campaign. NOT among the 193 findings (W5/COMP-08 is the audit's COMP-08, deferred by user decision; the rest are out-of-campaign-scope discoveries). All filed as GitHub issues (2026-06-22).

| Source | Sev | Issue | Item |
|---|---|---|---|
| COMP-08 / W5 | S2 | [#33](https://github.com/chris-santiago/ferrum/issues/33) | ✅ **RESOLVED 2026-07-05** (`feat/composite-render-unification`, Phase B / #45). **JointChart/ClusterMap interactive layout ≠ SVG ratio-grid (caption-y drift).** Fixed exactly as prescribed: `Panel.layout_scale` (`{sx,sy,tx,ty}`, serde-default identity) through the scene schema, baked at load by the WASM renderer; both output kinds now share one Rust composite layout pass (`render_composite_interactive`), browser-validated via headless captures (`.claude/output/phase-b-captures/`). The Python interactive scene-merge no longer exists. |
| T4.2b | S2 | [#34](https://github.com/chris-santiago/ferrum/issues/34) | ✅ **RESOLVED 2026-06-23** (`fix/composed-raw-node-offset`, see the 2026-06-23 section below). **Composed interactive renders did not offset inset `<svg>` / data-anchored `<image>` raw nodes.** The `_offset_node` raw branch (COMP-07/W4) offset only `<rect x/y>`; `inset.rs` `<svg x/y>` and `annotation.rs` `<image x/y>` producers stayed at child-local coords. Fixed by wrapping each raw fragment in a `<g transform="translate(dx,dy)">` (browser-verified). |
| T3.5 | S2 | [#35](https://github.com/chris-santiago/ferrum/issues/35) | ✅ **RESOLVED** (small-multiples rendering landed `ae2e305` 2026-07-01 via `_compose_compare`; per-panel scale SHARING completed by Phase B / #45 — residuals `compare=` shares axes position-wise through the Rust composite resolve pass). **Multi-model (`compare=`) RENDERING for the 17 aggregate diagnostics.** |
| T1.5 | S2 | [#36](https://github.com/chris-santiago/ferrum/issues/36) | ✅ **RESOLVED** (already fixed in `f20a135`, shipped v0.18.0; issue was filed ~10 h *after* the fix landed and closed 2026-06-23 as stale). **Precomputed 1-D binary gain/lift ranked negative class by `p`, inconsistent with roc/pr (`1-p`).** Now builds `column_stack([1-p, p])` so the whole precomputed-1D-binary family is consistent; regression test + v0.18.0 changelog entry present. |
| WIRE-ASFIELD-1 | S1 | [#37](https://github.com/chris-santiago/ferrum/issues/37) | **`transform_calculate` emits wire key `as_field` while siblings emit `as_`.** Rust `TransformSpec` serde rename decision (pinned by a test). Candidate for a Tier-6 or dedicated transform-wire pass. |

## 2026-06-23 — issue #34 resolved (`fix/composed-raw-node-offset`)

**T4.2b / [#34](https://github.com/chris-santiago/ferrum/issues/34) — composed interactive renders now offset inset `<svg>` + data-anchored `<image>` raw nodes. ✅ RESOLVED.**

Root cause: `_offset_node`'s `raw` branch (`src/ferrum/_scene_merge.py`) offset only `<rect x/y>` via a regex helper (`_offset_raw_svg_rects`), missing the inset `<svg x/y>` (`inset.rs`) and data-anchored `<image x/y>` (`annotation.rs`) producers, so they stayed at child-local coords in composed `.interactive()` output. The regex was also over-reaching — it shifted an inset's *nested* `<rect>`s in the wrong coordinate space.

Fix: replaced per-element coordinate rewriting with a uniform `<g transform="translate(dx,dy)">` wrapper applied to the whole raw fragment. This mirrors the static compositor's whole-unit-translate strategy, is element-agnostic (any future raw producer is offset automatically), and eliminates the nested-`<rect>` over-shift. Deleted `_offset_raw_svg_rects` and the now-unused `import re`; `_uniquify_clip_ids` is unchanged. Interactive-only and byte-irrelevant to static goldens (the static path composes in Rust). The legend `<clipPath>` def stays inert in interactive (WASM drops the consuming `Group.attrs`), so leaving its baked coords un-rewritten is a no-op.

Verified at three levels: 6 regression tests (5 node-level in `test_bug_hunt_scene_composition.py` + 1 public-path integration in `test_interactive_regression.py`), each proven to fail when the fix is stashed; scene-JSON before/after (`0/2 → 2/2` raw nodes wrapped in `translate(650,0)`, inner coords preserved); and a real headless-Chrome before/after where the data-anchored image annotation moves from the wrong LEFT panel to the correct RIGHT panel (live DOM confirms the nested `<g transform>` composes with the pan/zoom `dataG` matrix; no console errors).

**New follow-up (open) — Option D: promote image annotations to first-class `image` scene nodes.** The data-anchored image annotation is currently a `SceneNode::Raw { <image …> }` opaque string (`annotation.rs`). Promoting it to a first-class `image` scene node (the type `_offset_node` already offsets at the leaf, and which the WASM `ImageQuad` path could render on the GPU with hit-testing) would let image annotations participate in GPU rendering + tooltips like other marks, instead of riding the SVG overlay. The inset `<svg>` is a genuine composite viewport and stays a raw fragment. Not required for #34 (the wrapper fixes both producers); a cohesion improvement to consider when the annotation / scene-node taxonomy is next touched.

## 2026-07-02 — Phase A design-review deferrals (#44/#45/#46 remediation)

- **DSG-1 (S2, structural hardening):** `_resolve_pending_impl` (src/ferrum/_desugar.py)
  threads scale-through-desugar propagation at its three return branches (:686, :702,
  :725); a future fourth desugar path could silently skip propagation. Consider a
  single-exit refactor or a shared epilogue when this function is next touched.
  Channel-clone/private-`_kwargs` + scale-shape normalization are scheduled in the
  Phase B plan Task 10 (sole-ownership consolidation after `_scale_share.py` deletion).

- **DSG-2 (S3-adjacent, Task 5d quality review; expanded by Task 8a):**
  `render/binding.rs::collect_leaf_bindings_walk` duplicates the dict→kind→children
  tree-walk that `spec/composite.rs::composite_node_from_py` owns, guarded only by a
  count-equality check (catches count drift, not reorder-with-equal-count). Task 8a
  raised the stakes: the walk now tracks a THIRD node kind ("hole", which contributes
  no leaf) in lockstep by convention only — a future kind added to one walk but not
  the other is a runtime error. Export the shared kind/children extraction helpers
  `pub(crate)` and consolidate to one walk when the binding layer is next touched.
- **DSG-3 (chore):** `binding.rs` new code uses non-deprecated `Bound::cast`/`cast_into`
  while the rest of the crate (incl. `spec/composite.rs`) uses the deprecated
  `downcast` idiom — crate-wide migration chore, clippy-motivated.

---

## 2026-07-05 — Phase B close design reviews (non-blocking follow-ups)

The Phase B (#45 composite-render-unification) verification close ran both
design reviewers over the whole branch. Rust verdict PASS, Python verdict
CONCERNS (S3s remediated in-branch: share_scale unified onto resolve=,
stale `.spec` docstrings fixed, LayerChart explicit-independent typed error;
see the close commits). ~~The S2-and-below findings below are logged for the
next touch of each subsystem.~~ **ALL RESOLVED 2026-07-05** on
`fix/72h-findings-burndown` (user directive: fix everything now) — plus GH #50
(interactive warning emission) fixed outright, the temporal `_column_minmax`
bug fixed (not deferred to #52), the polars `ColumnNotFoundError` catch gap
closed, and shap_bar aligned to the global-feature-set principle. Remaining
open by explicit user scoping: #51 (upstream wgpu), ~~#52 (secondary-axis
subsystem)~~ **RESOLVED 2026-07-11** (see the 2026-07-11 section below),
#53 (Joint/ClusterMap native resolve=). Full reports:
`.claude/output/phase-b-close/` + `.claude/output/72h-review/`.

| Sev | Area | Item |
|---|---|---|
| S2 | rust layering | `scale_resolve` imports `SharedDomain`/`LeafScaleContext` from `render::composite` while `composite` imports union helpers back — inverted layering; move the seam value-types into the resolver. |
| S2 | rust binding | `composite_tree_from_py` and `collect_leaf_bindings_walk` walk the same Python dicts separately, coupled only by a leaf-count guard (catches count drift, not reorder). Unify into one walk when next touched. |
| S1 | rust interaction | `merge_children` AND-folds zoom/pan across children but hardcodes `toolbar: true`. |
| S2 | python scene | `_scene.py::_empty_scene_json` is a hand-maintained scene-schema mirror (flat-path bootstrap). Consider a Rust-emitted empty scene. |
| S2 | python lowering | `_rebuild_with_charts` sibling signature drift: base + 3 forms carry `resolve=_RESOLVE_UNCHANGED`, Joint/Repeat/ClusterMap don't (unreachable today — Repeat has a semantically-identical bespoke `share_scale`; collapse it onto the base sugar when next touched). `_build_grid_tree` has a 12-parameter signature; composite-node dict construction is triplicated across the lowering sites (unvalidated tree contract — a tiny builder/validator would pin it). |
| S2 | python overlay | LayerChart static (Rust overlay tree) vs interactive (merged flat chart) shared color/size math can diverge (raw-column vs transform-aware unions) — single remaining injection seam, documented; real fix rides GH #52 (secondary axis / per-layer scale slots). ✅ **RESOLVED 2026-07-11** for the independent-y path: `resolve={"y": "independent"}` (and its `SecondaryY` sugar) now routes BOTH static and interactive through the identical merged-flat single-panel spec (CLAUDE.md "Composite rendering" amendment), so there is no separate overlay-tree computation to diverge from for that case. The general raw-column-vs-transform-aware `compute_union_domain` divergence for DEFAULT (`resolve={"y": "shared"}`) shared x/color/size scales across the two paths is unrelated to per-layer y slots and remains open (unaffected by this work; see `compute_union_domain`'s own docstring in `composition.py`). |
| — | wasm | wgpu 29.0.3 scissor workaround sunset: tracked in GH #51. |
| S2 | shap family | `shap_bar` per-class feature selection is class-blind (`head(max_display * n_unique)` over a global sort) — beeswarm and (since #46) waterfall both follow the global-feature-set-per-class principle; bar is the lone outlier. Align when the shap family is next touched. (72h review, 2026-07-05.) |
| S1 | shap family | `is_faceted` computed once in beeswarm vs recomputed inline in bar/waterfall. |
| S2 | wasm/svg parity | LayoutScale anisotropy (sx≠sy, joint/clustermap marginals only): SVG scales per-axis, WASM bakes geometric-mean scalars (D4a-accepted approximation) — the divergence is BETWEEN the two render paths at marginal panels; re-inspect both paths together if marginal mark fidelity is ever questioned. |
| — | adjudicated | 72h python-design S3 "explicit child scale silently defeats resolve=shared" does NOT stand: spec §6's locked decision (explicit `enc.scale` wins) with LEAF-SCOPED exclusion — verified empirically (3-child: pinned keeps domain, other two union). Remedied as documentation (share_scale docstring + composition guide). |

---

## 2026-07-10 — dodge-by-model compare= (#42) fragility note

Not a bug — a known-fragile invariant surfaced while shipping GH #42
(`importance_chart`/`shap_bar_chart`/`cv_scores_chart` dodge-by-model
`compare=` layouts). The dodge band-extent narrowing contract reads the
per-batch group count from the `__dodge_n_groups__` Arrow schema-metadata
key (`DODGE_N_GROUPS_KEY`, `crates/ferrum-core/src/render/position.rs`),
stamped by `apply_dodge_ordinal` alongside the `__pos_x_offset__`/
`__pos_y_offset__` offset columns it emits. `n_dodge_groups` reads this key
back; `bar.rs`, `rect.rs`, and `tick.rs` all call `n_dodge_groups` to shrink
each category's per-group extent to `extent / n_dodge_groups` rather than
inferring the group count by counting distinct offset values (jitter-shaped
offsets without the metadata key correctly report `1`, not the distinct
count).

**Fragility:** the group count lives on `RecordBatch::schema().metadata()`,
which is carried by reference (`Arc<Schema>`) through the render pipeline
today. A future `scene_build.rs` (or transform-pipeline) refactor that
rebuilds the batch's schema from scratch — e.g. via `Schema::new(fields)`
instead of cloning/extending the existing `Arc<Schema>` — would silently
drop the metadata map and every dodge-narrowing consumer would fall back to
its `n_dodge_groups() == 1` no-op path: dodge offsets would still shift bars
apart, but the per-group width would revert to the full undivided band,
producing overlapping (not narrowed) marks with no error or warning. A
guard test exists Rust-side
(`n_dodge_groups_end_to_end_via_apply_dodge_ordinal`, `position.rs`) that
drives the real `apply_dodge_ordinal` producer and asserts the group count
round-trips — it would catch a regression in `apply_dodge_ordinal` itself,
but would not catch a schema-rebuild elsewhere in the pipeline that never
calls `apply_dodge_ordinal` at all. Anyone touching batch/schema
reconstruction in the render pipeline should know this invariant depends on
metadata surviving by reference, not on any structural schema shape.

## 2026-07-10 — issue #40 resolved (`fix/40-diverging-positional-extent`) + deferred follow-up

GH #40 (Diverging 3-element `[lo, mid, hi]` domain truncated to `[lo, mid]`
on positional channels) is fixed: `ScaleSpec::positional_extent()`
(`crates/ferrum-core/src/spec/encoding.rs`) now classifies every variant's
`domain` as positional extent vs discrete-binning artifact in a single
exhaustive match with no wildcard arm, and `build_from_scale_spec`
(`render/scale_resolve/positional.rs`) consumes it in one merged
continuous-fallback arm. A new `ScaleSpec` variant is a compile error until
classified. See the updated R7 row above. Regression guards:
`tests/test_scale_spec_parity.py::TestPositionalExtent::test_diverging_positional_all_marks_render`
plus the `positional_extent_classification` Rust module in
`render/scale_resolve/tests.rs`.

**Deferred follow-up (separable):** the outer-bounds extraction rule
(`[d[0], d[d.len()-1]]` for Sequential/Diverging domains) now exists in two
places — `positional_extent()` and the color path's `scale_explicit_domain`
(`render/scale_resolve/color.rs`). Both design reviews judged a shared
helper not worth it at two call sites; if a third outer-bounds consumer
appears, extract `fn outer_bounds(domain: &[f64]) -> Option<(f64, f64)>`
and route both through it. The larger north star from the #40 defense —
splitting extent-vs-binning domain semantics into distinct types on the
`ScaleSpec` enum itself (composes with FA-10's typed-domain idea) — remains
deliberately unbuilt: it would require custom serde to keep the wire
contract byte-stable across all 15+ variants for enforcement the total
method already provides at the single consumption site.

## 2026-07-11 — secondary y-axis (#52) resolved + new follow-ups

GH #52 (secondary y-axis / per-layer independent y-scales) is resolved on
`feat/secondary-y-axis`, per
`design-docs/superpowers/specs/2026-07-11-secondary-y-axis-design.md`.
`LayerChart(..., resolve={"y": "independent"})` now renders a real dual-axis
chart (layer 0 left, subsequent independent layers stacked right, unbounded
n) through one merged flat single-panel pipeline shared by static SVG and
interactive output; `fm.SecondaryY` is re-based onto the same mechanism
(desugars to an appended independent-y layer at `Chart.__add__` time) and
its former standalone silo (`SecondaryYSpec` / `StructuralSpec::SecondaryY`,
`render/secondary_axis.rs`) no longer exists in the crate. The S2
"python overlay" row above is marked resolved for the independent-y case.
`ferrum-spec.md` §3.12 documents the mechanism and `SecondaryY`; `CLAUDE.md`
"Composite rendering" documents the both-kinds merged-flat routing.

New follow-ups discovered during the run (none blocking #52's close):

| ID | Sev | Item |
|---|---|---|
| SY-1 | S2 | **Intra-member slot grouping.** The per-layer `independent_y` bool wire has no way to group a multi-layer member chart's internal layers into ONE secondary axis slot. A non-first `LayerChart` member that itself decomposes into more than one y-bearing layer — including composite-mark shorthands like `mark_line(point=True)`, which expands to a line layer + point layer both inheriting the chart-level `y` — raises a typed `ValueError` under `resolve={"y": "independent"}` rather than silently rendering one right axis per internal layer. The primary (first) member is exempt (always the left axis regardless of layer count). Fix requires a slot-group id in the wire contract (grouping N layers to 1 axis slot), not just the current bool. `src/ferrum/composition.py::LayerChart._build_merged`. GH #56. |
| SY-2 | S1 | **RepeatChart `resolve={"y":"shared"}` over an independent-y template bypasses the composite conflict raise.** The typed `ValueError` for "parent composite `y: shared` over a subtree containing an independent-y layered leaf" is enforced at `_composite_tree`/`_lower_any`'s ordinary composite-node walk; a `RepeatChart` carrying its own `resolve={"y": "shared"}` over an independent-y template chart routes through `_build_grid_tree` instead, which does not run that guard. Exotic (no test currently exercises it); log only, not a live bug. GH #57. |
| SY-3 | S2 | ~~**Field-based linked selection suppresses per-layer auto-tooltips.** A layered chart with a field-based `selection_point`/`selection_interval` linked tooltip takes the chart-level selection-tooltip injection path, which early-returns before the new per-layer auto-tooltip walk runs — so a selection-bound layered chart still shows one shared tooltip field set rather than per-layer fields. Not a regression (this behavior predates #52); just a narrower scope than the unconditional per-layer case documented in `ferrum-spec.md`'s auto-tooltips note.~~ ✅ **RESOLVED (fix/open-bug-sweep, combined with GH #78)**: `SpecBuildMixin._inject_auto_tooltips` (`src/ferrum/_spec_build.py`) no longer short-circuits on the structural proxy "does any layer carry its own tooltip". It now derives provenance directly — a new `Chart._tooltip_promoted` marker (set by `__add__`, reset by `.encode(tooltip=...)`) distinguishes a promoted primary-layer tooltip from a genuine chart-wide override, and `"tooltip" not in self._encoding` while the wire carries `tooltip_fields` identifies selection-injection (the only source that can produce that combination). When selection-injected, the per-layer walk runs and unions the selection's fields into each layer's own auto fields instead of replacing them. `src/ferrum/chart.py`, `src/ferrum/_spec_build.py`. GH #58, GH #78. |
| SY-4 | S3 | **Explicit `Axis(orient="left")` on a secondary layer is silently forced `Right`.** Spec-compliant (§4 "Axes": secondary layers always render right), but repo precedent elsewhere favors surfacing a contradictory explicit request (warning or typed error) rather than silently overriding it. User decision pending on whether to add one; not a bug today. `crates/ferrum-core/src/render/prepare/mod.rs` (`build_secondary_y_axis_inputs`). GH #59. |
| SY-5 | S2 | ~~**`text_json` right-axis relabel needs ≥2 ticks per secondary axis.** The WASM `text_json` right-axis column-recognition heuristic ranks candidate columns by ascending-x layout geometry; a degenerate single-tick right axis (e.g. a secondary layer whose y-domain collapses to one value) doesn't produce enough ticks for the heuristic to recognize the column, so it won't relabel on zoom/pan. Documented at Task 9's quality-review carry-forward; no crash, just a missed relabel in a narrow domain-collapse case.~~ ✅ **RESOLVED 2026-07-12** (post-v0.19 sweep #71–#74, Task 7, commit `717d35e5`): the proper fix landed as planned — `SceneNode::Text` (and `ferrum-scene`'s `TextElementData`) gains an optional `slot` field, serde-defaulted so untagged/legacy scenes are byte-identical; `build_axis` tags every y-slot's tick-label text nodes (titles stay untagged) and `route_y_axis_slotted` tags all slots uniformly including the primary. Task 8 (commit `17abcda7`, #73) then rewired `text_json` to select the rescale affine by this explicit slot tag instead of the ascending-x column-frequency heuristic, and **deleted** the heuristic machinery (`y2_x_freq`/`ranked_cols`/`col_rank`) outright — a net simplification, not just a workaround. A single-tick right axis now relabels identically to a multi-tick one, and the stray-label guarantee is strengthened structurally (an untagged node can never be mistaken for an axis label, regardless of content matching a tick string), pinned by a dedicated regression test. GH #60 closed. `crates/ferrum-core/src/render/marks/axis.rs`, `crates/ferrum-core/src/render/scene_build.rs`, `crates/ferrum-scene/src/types.rs`, `crates/ferrum-wasm/src/text_json.rs`, `crates/ferrum-wasm/src/scene_load.rs`. |
| SY-6 | S1 | **`build_structural_nodes`'s `StructuralOutput` 4-tuple has two permanently-empty slots.** With `StructuralSpec::SecondaryY` deleted (Task 7), only `BreakAxis`/`Inset` remain, and neither populates the `extra_axes` / `extra_mark_batches` locals (only `extra_annotations` and `break_results` are ever mutated) — those two locals were changed from `let mut` to `let` to silence clippy, but the 4-tuple `StructuralOutput` return shape was deliberately left as-is (a future structural variant could repopulate them). Tidy candidate — collapse to the 2 live fields, or re-justify the 4-tuple with a comment — when this function is next touched. `crates/ferrum-core/src/render/scene_build.rs` (`build_structural_nodes`). GH #61. |
| SY-7 | S2 | **Layer-spec synthesis recipe exists twice** (whole-change gate 2026-07-11): `prepare/mod.rs::build_secondary_y_axis_inputs` and `scene_build.rs::resolve_layer_y_scale` both synthesize a per-layer single-y ChartSpec (overlay layer encoding, strip `layers`, mark from layer) — deliberate and cross-documented (mirrors the primary's provisional-vs-panel double resolution), but a future drift seam. Extract a shared `synthesize_layer_y_spec` helper when either site is next touched. GH #62. |
| SY-9 | S1 | **Per-layer `ResolvedScales` clone carries stale `y_slots`** (design review 2026-07-11): `build_panel_mark_batches` clones `ResolvedScales` per independent layer swapping `.y` to the slot scale, but the clone's `y_slots` still describes all slots — internally inconsistent, never read today (marks read `.y`), a latent trap if a mark ever consults `ctx.scales.y_slots`. Cheapest fix: empty `y_slots` on the per-layer clone. `crates/ferrum-core/src/render/scene_build.rs` (~:977-986). GH #64. |
| SY-8 | S2 | **Two per-slot list-indexing conventions on the wire** (whole-change gate 2026-07-11): `CoordKind::Cartesian.y_domains` includes slot 0; `PanelTickLevels.y_slot_levels` and the wasm `secondary_affines` are slot-1-based (index = slot − 1). Both are documented and tested at each seam; standardize on one convention if a third per-slot list is ever added. GH #63. |

## 2026-07-11 — Band/Point explicit `range=` (#39) resolved + new follow-ups

GH #39 (Band/Point scales: explicit `range=` silently dropped at the wire
boundary) is resolved on `fix/band-point-scale-range`, per
`design-docs/superpowers/specs/2026-07-11-band-point-range-geometry-design.md`.
`ScaleSpec::Band`/`Point` now carry an optional `range`; the positional
resolver honors it and records **range-explicitness** on the resolved
`OrdinalScale` (`explicit_pixel_range`, set only at construction — never
float-inferred); all band-geometry consumers derive from the scale when a
range is explicit: mark widths/cells/tick extents via
`render/marks/channels.rs::band_extent_or`, categorical axis ticks + grid via
`AxisInput.categorical_positions` (absolute band centers). The no-range path
is byte-identical by construction (literal panel-extent fallback expressions,
golden corpus + `test_scale_spec_parity.py` green). Regression module:
`tests/test_regression_band_point_range.py` (10 tests, RED-proven).

New follow-ups discovered during the run (none blocking #39's close):

| ID | Sev | Item |
|---|---|---|
| BR-1 | S2 | **`PointScale(reverse=)` / `align=` are serialized but never consumed at render.** The resolver's Band/Point arms swallow both with `..`; `OrdinalScale::new_internal` has no reverse/align parameter and no `ScaleKind::Ordinal` path honors them — the same silent-drop class as #39's `range=`, one field over. Fix = OrdinalScale reverse/align support (or resolver-side domain/range pre-transform) + regression tests. `crates/ferrum-core/src/render/scale_resolve/positional.rs`, `scale/{point,ordinal}.rs`. GH #65 — **RESOLVED 2026-07-11** (main 1b18f79d..2834756d): `reverse` honored via post-sort domain reversal at the Point arm; `align` adjudicated inert at every layer (no pixel rounding → zero alignment leftover; compute facade ignores it identically) — no semantic to restore, documented in the ScaleSpec→ScaleKind mapping rustdoc; base-position model divergence stays with #67. |
| BR-2 | S2 | **Explicit range + `padding_inner` + dodge is an untested overlap seam.** Mark widths are padding-ignoring (`extent / n / n_groups * shape_factor`) while dodge sub-band offsets are padding-aware (`bandwidth()`); large `padding_inner` under dodge can overlap bars. Pre-existing on the fallback path too (spec §3 declares padding-aware widths a non-goal), but the explicit-range path makes it more reachable. Add a discriminating test for explicit-range+padding+dodge before leaning on this seam (design review 2026-07-11). `crates/ferrum-core/src/render/{marks/bar.rs,position.rs:431-454}`. GH #66. |
| BR-3 | S1 | **North-star: collapse the explicitness gate.** `band_extent_or`'s fallback exists solely to keep no-range arithmetic byte-identical (recomputing `panel.w` as `(panel.x+panel.w)-panel.x` can drift 1 ulp). When a golden-regeneration window is acceptable, make the resolved scale the *unconditional* source of band geometry and delete the gate. Related tidies noted by reviews: `range_user_set` vs `explicit_pixel_range` coexistence (doc-warned at the field, 6f245d69+); 1-entry ordinal range passes through (`ordinal_pixel_range`, pre-existing) while Band/Point fall back — both non-explicit, documented divergence; `tick_projection`/`categorical_positions` mutual exclusivity is discipline-enforced (a `debug_assert` would harden). GH #67. |

## 2026-07-12 — post-v0.19 sweep (#71–#74) resolved + new follow-ups

Four coherent remediations from the release-scoped bug hunt of `v0.19.0..main`
(issues #69–#76), per
`design-docs/superpowers/specs/2026-07-12-post-v019-sweep-71-74-design.md`,
are resolved on `fix/post-v0.19-bug-sweep`: (A) #71 unifies independent-y
semantics across both dual-axis spellings and fixes a rename-sentinel/tooltip
leak; (B) #72 hoists per-layer domain params onto the wire and unifies
per-slot y-domain resolution (`YSlotPlan`, computed once at prepare); (C) #73
makes WASM hit-testing and axis relabeling slot-aware under runtime rescales
(subsumes SY-5/GH #60, resolved above); (D) #74 defines and implements nested
composite shared resolve for color/size (leaf-span union, commit `d3d8e12d`)
and makes `configure_legend(orient="none")` join the legend-disabled
mechanism (commit `d1bf18b6`). See `ferrum-spec.md` §3.9's 2026-07-12 (#74)
note and the #16 composite-shared-legend design doc's matching nesting-rule
update for the user-facing contract.

**#52 §4 "Nesting" — spelling-independent conflict, closed (commit
`10244b0f`, #71 Task 1).** The GH #52 spec's nesting conflict ("a parent
composite's explicit `resolve={"y": "shared"}` colliding with a dual-axis
chart in its subtree raises") was only enforced for the
`LayerChart(resolve={"y": "independent"})` spelling —
`_contains_independent_y_layer` recognized `LayerChart._y_independent()` but
had no path to a plain `Chart` whose `_layers` carry `independent_y=True`
flags from the *other* spelling, `chart + SecondaryY(...)`. A composition
built from the `SecondaryY` spelling silently rendered instead of raising.
`Chart` now exposes a `_has_independent_y_layer()` capability predicate
(mirroring the #16 `_supports_user_resolve` idiom) that
`_contains_independent_y_layer` consults for leaf `Chart`s the same way it
consults `LayerChart._y_independent()` for the layered spelling, and
`LayerChart._composite_tree`'s own shared-y overlay route raises the same
typed error when one of its members carries the flag. Both spellings of
dual-axis now raise identically under an explicit (or default) parent
`resolve={"y": "shared"}`. `src/ferrum/chart.py`, `src/ferrum/composition.py`.

**Dodge band-axis fix (commit `4f2839bb`) + `apply_stack` follow-up (#77).**
Discovered while shipping #75 (jointplot/pairplot builder defects):
`apply_dodge` chose its categorical band axis from `coord_flipped` alone,
which only an explicit `CoordFlip` sets. A natively-horizontal composite
mark (e.g. `mark_boxplot(horizontal=True)`) swaps x/y at Python desugar
*without* setting `CoordFlip`, so a dodged horizontal boxplot offset along
the continuous value axis instead of the category axis — and because each
box sub-layer (rect/whisker/median) carries a different value column, the
sub-layers desynced from each other. The band axis is now chosen by which
resolved scale is `ScaleKind::Ordinal` (the same convention `apply_jitter`
already used, healing the sibling drift between the two), falling back to
`coord_flipped` only when both/neither axis is ordinal (byte-identical for
every currently-passing case). `apply_dodge_ordinal`'s `coord_flipped` param
is renamed `band_on_y`. The commit message flags `apply_stack`'s matching
`coord_flipped`-only band-axis selection as **unproven-broken** by the same
class of bug (no repro yet, but the same natively-horizontal-desugar path
could reach it) — tracked as **GH #77**, an audit of `apply_stack` for the
identical pattern, not yet fixed. `crates/ferrum-core/src/render/position.rs`.

*Back-link to the `__dodge_n_groups__` fragility note (2026-07-10,
above):* that note's producer/consumer contract (`apply_dodge_ordinal`
stamps `DODGE_N_GROUPS_KEY`; `bar.rs`/`rect.rs`/`tick.rs` read it via
`n_dodge_groups` at every band-geometry call site) is **unaffected** by this
fix — `4f2839bb` only changes *which axis* `apply_dodge_ordinal` treats as
the band before computing sub-band offsets, not the metadata contract or
its consumers. Anyone auditing that fragility note after this commit should
know the band-axis selection it depends on is now `ScaleKind`-driven, not
`coord_flipped`-driven, at the same call site; the #77 `apply_stack` audit
should check whether its own band-axis selection needs the analogous
widening before touching `n_dodge_groups`-adjacent code there.

## 2026-08-27 — `mark_boxen(palette=)` open gap (P9 AST guard extension, Task 14 quality-review cycle 3)

**New open item, not yet resolved.** While extending the P9 desugar-parameter
AST guard (`tests/test_mark_kwargs_no_silent_drop.py`) from "every `del`" to
"every declared parameter, `del`eted or simply never referenced", three
mark-level parameters surfaced in that state: `mark_confusion(normalize=)`
and `mark_pdp(center=)` matched the existing `proba`/`n_thresholds` shape
(effect lives entirely upstream in a figure function — both now registered
in `ferrum.marks._informational_kwargs.INFORMATIONAL_KWARGS` and warn once,
see `ferrum-spec.md`'s 2026-08-27 P9 AST guard extension note), but
`mark_boxen(palette=)` did not.

| ID | Sev | Item |
|---|---|---|
| PAL-1 | S2 | **`mark_boxen(palette=...)` is a real, undelivered feature — accepted, now warned, not implemented.** `src/ferrum/marks/composite.py::desugar_boxen` never reads `palette` (now an explicit `del palette` with a comment, previously a silent unreferenced parameter). No call site anywhere — mark, mixin, or any figure function — gives it an effect. Depth-band color follows the ordinary mark-color resolution: an explicit `fill=` override, else the chart's `color` encoding through the theme's categorical palette, else the theme's default `mark_color` — with only opacity ramping by depth; `palette` is never consulted at any step (confirmed by quality review by rendering under `solarized_dark`, the one builtin theme where the default `mark_color` and `color_scheme[0]` differ, after an earlier draft of this note wrongly claimed depth-band color "always comes from the categorical palette" — that claim only coincidentally held under `paper_ink`, where `PAPER_INK[0] == mark_color`). The public docstring in `_chart_methods_statistical.py::mark_boxen` used to falsely claim "Colour palette applied to successive depth bands", contradicted by `ferrum-spec.md`'s own (also-false) "sequential palette name; nested rectangles fade toward the median" prose — both are now corrected to say "accepted but not yet honored" with the actual color-resolution order spelled out. `palette` is registered in `INFORMATIONAL_KWARGS` and warns once via `warn_informational_kwarg` when passed with a non-`None` value (`tests/test_finding_p9.py::test_mark_boxen_palette_*`), which stops it from being *silent*, but this is a stopgap, not a fix — the disposition is explicitly a warn-fallback (CLAUDE.md: "No Warn-fallbacks" — flagged by quality review as only defensible as a bridge to a real fix, not a resting state). **Two real dispositions remain, neither done:** (a) implement per-depth-band palette application in `desugar_boxen` (map an explicit `list[str]`/`None` palette across the nested rect layers by depth, analogous to how `mark_boxplot`/other composite marks resolve categorical color), or (b) remove the parameter entirely (breaking `mark_boxen(palette=...)` callers with a `TypeError`, matching the P9 "removed" disposition applied to `mark_calibration(n_bins=)` etc.). `src/ferrum/marks/composite.py` (desugar), `src/ferrum/marks/_chart_methods_statistical.py` (mixin + docstring), `src/ferrum/marks/_informational_kwargs.py` (registry, full writeup of the disposition split between this and the `normalize`/`center` shape). |
| BRK-1 | S3 | **`apply_break_to_scale` silently erases all data on a NaN gap bound.** (2026-08-27, R1 test-port discovery.) Via `f64::max`/`min` NaN semantics, a `[NaN, NaN]` gap resolves to `[d_lo, d_hi]` — a domain-covering gap that clips every datum and renders a blank chart with no error. Pinned by `crates/ferrum-core/src/render/break_axis.rs::nan_gap_bounds_resolve_to_a_domain_covering_gap` until fixed; the honest behavior is a loud rejection of non-finite gap bounds at spec validation. |
