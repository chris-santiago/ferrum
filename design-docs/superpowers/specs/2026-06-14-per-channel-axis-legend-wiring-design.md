# Per-channel `axis=` / `legend=` Wiring — Design Spec

- **Date:** 2026-06-14
- **Status:** Draft for review (scoping only — no implementation)
- **Related:** RCA `design-docs/superpowers/followups/2026-06-14-per-channel-axis-legend-silent-drop-rca.md` (bug B5)

## 1. Scope

Make per-channel axis/legend options actually render. Today `encode(x=fm.X("f", axis=fm.Axis(label_color=...)))` (and `legend=`) cross to Rust as opaque `serde_json::Map` blobs (`AxisSpec`/`LegendSpec`), and `render/prepare.rs` hand-reads only ~9 axis / ~14 legend keys; every other documented `fm.Axis` (~32) / `fm.Legend` (~26) field is silently dropped — even though the chart-level `configure_axis`/`configure_legend` path already renders those same fields through a typed struct. This spec replaces the opaque per-channel specs with the existing typed config structs and routes them into the per-axis/per-legend config the chart-level path already feeds, at per-channel precedence.

## 2. Goals

- Every `fm.Axis` / `fm.Legend` field that the chart-level `configure_*` path renders also renders when set per-channel (`label_color`, `grid_color`/`grid_dash`/`grid_width`, `domain_color`/`domain_width`, `label_font_size`, `label_overlap`, `tick_values`; legend `symbol_size`/`symbol_type`/`orient`/`columns`/`offset`/`padding`, etc.).
- **Every advertised field renders, including orphans** (Q1 resolved: implement all). Fields `fm.Axis`/`fm.Legend` advertise but no renderer currently honors — axis `orient`, `translate`, `min_extent`/`max_extent`, `tick_extra`, `tick_min_step`, `grid_opacity`, `title_orient`, `zindex`; legend `clip_height`, `row_padding`/`column_padding`, `symbol_stroke_width`, `label_limit`, `tick_min_step`, `zindex` — gain new layout/render support so they render at **both** chart level and per-channel. This is the scope-expanding part: it adds new Rust render code, not just routing.
- **Per-channel wins the cascade** over chart-level config on its own axis/legend: per-channel `x.axis` > `configure_axis` (general + `axis_x`) > theme > default.
- An unknown per-channel axis/legend key **fails loud** (`FerrumOverrideError`-style or a serde error surfaced as `ValueError`) instead of silently dropping.
- No behavioral change for the keys that already work; no golden churn for charts that don't set per-channel axis/legend styling.

## 3. Non-goals

- Changing the chart-level `configure_*` surface or `AxisConfig`/`LegendConfig` Python dataclasses (note: `AxisConfigSpec`/`LegendConfigSpec` Rust structs and the Python config dataclasses DO gain the orphan fields, so chart-level `configure_axis(orient=...)` etc. also start working — a deliberate side effect of "implement all").
- Axis/legend capabilities beyond the fields `fm.Axis`/`fm.Legend` already advertise — only the advertised set is wired; no net-new options are invented.
- The override (B4) feature — separate; though both share the "type the spec + fail loud" remedy (override spec Q3).
- Per-channel options for channels that have no axis/legend (only x/y carry axes; color/size/etc. carry legends).

## 4. System behavior

- **Typed specs.** `EncodingSpec.axis` and `EncodingSpec.legend` deserialize into typed structs whose fields mirror the Python `Axis`/`Legend` value classes (reusing the chart-level `AxisConfigSpec`/`LegendConfigSpec` field set; see §8 D1), not an opaque map.
- **Routing & cascade.** During render prep, each channel's typed axis/legend spec is folded into the same per-axis / per-legend config the chart-level `configure_*` path produces, applied **after** the chart-level config so per-channel wins on conflict. `x.axis` → x-axis; `y.axis` → y-axis; `color.legend` (and other legend-bearing channels) → that channel's legend.
- **Fail-loud.** With the specs typed and `deny_unknown_fields` set, a misspelled per-channel key surfaces as a deserialization error (`ValueError` at render) instead of a silent drop.
- **Casing.** Python `Axis.to_dict()`/`Legend.to_dict()` emit snake_case; the typed structs use snake field names (matching `AxisConfigSpec`), so the snake keys map directly. The legacy camelCase keys the old hand-reader accepted (`labelAngle`, `tickCount`, `labelFormat`, …) are preserved via serde `alias` for back-compat with raw-dict callers.
- **Suppression contract preserved.** `axis=False` / `title=None` semantics (suppress axis / suppress title) continue to work as today.

## 5. Architecture

- **Rust spec (`crates/ferrum-core/src/spec/encoding.rs`).** Replace `AxisSpec { extra: Map }` / `LegendSpec { extra: Map }` with typed structs. Prefer reusing the chart-level `AxisConfigSpec` / `LegendConfigSpec` (`render/chart_config.rs`) directly as the per-channel field type, or a shared "axis style" struct both reference, so there is one schema. Add `#[serde(deny_unknown_fields)]` + camelCase `alias`es.
- **Render prep (`crates/ferrum-core/src/render/prepare.rs`).** Delete the ad-hoc `extra.get("…")` ladder (axis block + the D13 `color_legend_extra` block). Instead, after the chart-level `apply_axis_config_to_axis_input` runs, apply each channel's per-channel axis/legend config to the matching axis/legend input at higher precedence (per-channel-wins). Reuse the existing apply path the chart-level config uses.
- **Python (`src/ferrum/axis.py`, `legend.py`, `encoding/base.py`).** `Axis.to_dict()`/`Legend.to_dict()` already emit the full snake field set — keep. Ensure `_normalize_axis`/`_normalize_legend` and `to_encoding_spec_dict` forward it to the now-typed field. No Python validation needed (Rust deny_unknown_fields is the guard), though a Python-side check could give friendlier errors (open question).
- **Layout consumers.** Grid color/dash/width, label color/size, domain styling, legend symbol geometry are already rendered from the typed config (`layout/axis.rs`, the legend pipeline); routing per-channel into that config lights them up with no new render code. **Orphan fields require new layout/render code** (`layout/axis.rs` for axis `orient`/`translate`/extents/`tick_extra`/`tick_min_step`/`title_orient`/`grid_opacity`; the legend pipeline for `clip_height`/`row_padding`/`column_padding`/`symbol_stroke_width`/`label_limit`) — this is the bulk of the new implementation surface and should be tasked per render subsystem.

## 6. Canonical interfaces / data contracts

- `EncodingSpec.axis: Option<AxisStyleSpec>` and `EncodingSpec.legend: Option<LegendStyleSpec>`. Per Q3 (resolved, §11): `AxisStyleSpec`/`LegendStyleSpec` are **new shared styling+positioning structs**, NOT `AxisConfigSpec` directly — investigation showed `AxisConfigSpec` is not a clean superset (it carries chart-level-only scale-domain fields `domain_min`/`domain_max`/`nice`/`zero` and the `x`/`y` show toggles, which are meaningless per-channel). The chart-level `AxisConfigSpec` embeds `AxisStyleSpec` via `#[serde(flatten)]` and adds those chart-only fields; the per-channel `EncodingSpec.axis` uses `AxisStyleSpec` alone. `LegendConfigSpec` ≈ `LegendStyleSpec` (legend has no chart-only-extra fields), so it flattens/extends the same struct. Field names are snake_case mirroring `fm.Axis`/`fm.Legend`; camelCase legacy keys via `#[serde(alias = "...")]`; `#[serde(deny_unknown_fields)]`. **Scale-domain (`domain_min`/`max`/`nice`/`zero`) is NOT in `AxisStyleSpec`** — per-channel scale domain is set via the scale path (`x_scale_domain`), not the axis.
- **Cascade precedence (per axis):** per-channel `<channel>.axis` > chart-level `axis_x`/`axis_y` > chart-level `axis` > theme > Rust default. Equivalent for legend.
- **Suppression:** `axis=False` → suppression spec (unchanged); `title=None` → empty-title suppress (unchanged).

## 7. Invariants and constraints

- **No silent drops.** Every per-channel axis/legend key either renders or fails loud. (Closes B5.)
- **Per-channel beats chart-level** on its own axis/legend; the other axis keeps its chart-level value.
- **No golden churn** for charts that set only currently-honored keys (or none). The honored keys must render byte-identically; only previously-dropped keys change output (those have no goldens today).
- **Back-compat:** raw-dict callers passing camelCase keys (`axis={"labelAngle": -30}`) keep working via serde aliases.
- **`cargo test` must pass**; the typed-spec migration must not break round-trip JSON (`ChartSpec.from_json`).

## 8. Key decisions and tradeoffs

- **D1 — One shared styling struct per concept (single schema).** Factor `AxisStyleSpec`/`LegendStyleSpec` (styling + positioning), embedded by both the chart-level config struct and the per-channel `EncodingSpec.axis`/`.legend` (see R-Q3, §11 — `AxisConfigSpec` is *not* a clean superset, so we share a styling struct rather than reuse the config struct directly). The already-rendered keys need zero new render code and gain guaranteed per-channel/chart-level parity; orphan fields are added to the shared struct AND given new layout/render support (so chart-level `configure_axis(orient=...)` starts working too). *Rejected:* a fresh, separate per-channel struct (duplicate schema, re-introduces drift); and direct `AxisConfigSpec` reuse (leaks scale-domain + `x`/`y` toggles per-channel).
- **D2 — Per-channel wins the cascade.** Per-channel is the more specific intent; it must beat the chart-wide `configure_axis`. Mirrors the override cascade philosophy. *Tradeoff:* a precedence the docs must state.
- **D3 — `deny_unknown_fields` + serde aliases.** Typing the spec is what makes fail-loud possible; aliases preserve the camelCase back-compat the hand-reader had. This is the B5-local instance of override-spec Q3 (defense-in-depth).
- **D4 — Snake_case is canonical.** Python already emits snake; the typed struct is snake; camelCase is alias-only. Removes the casing-mismatch drop the old reader had.
- **D5 — Delete the `extra.get` ladder, don't extend it.** Extending the hand-reader (RCA option 3) would work but preserves the drift risk; typing + routing removes the second reader entirely.

## 9. Acceptance criteria

- Per-channel renders match chart-level for the previously-dropped keys: `fm.X("f", axis=fm.Axis(label_color="#f00"))` → magenta label fill in SVG; `axis=fm.Axis(grid_color="#ccc"))` → grid stroke `#ccc`; `axis=fm.Axis(domain_width=3))` → domain line width; `color` legend `fm.Legend(symbol_size=...)` → symbol geometry.
- Precedence: `fm.X("f", axis=fm.Axis(label_angle=-45))` combined with `.configure_axis(label_angle=0)` renders −45 on x; the y-axis keeps the configured value.
- A misspelled per-channel key (`axis={"label_colr": "#f00"}`) raises at render (no silent drop).
- camelCase back-compat: `axis={"labelAngle": -30}` still rotates.
- Currently-honored keys + no-axis-styling charts render byte-identically (no golden churn); a representative golden spot-checked.
- `cargo test` + full `pytest` green.

## 10. Validation strategy

- **Render-level tests** (the gap that hid B5): set each previously-dropped per-channel field on a single channel and assert the SVG — not `.to_dict()`. Upgrade `tests/test_phase_12_axis_legend.py` from serialization assertions to render assertions for these fields.
- **Precedence tests:** per-channel vs `configure_axis` conflict resolves to per-channel on its axis, both construction orders; the other axis unaffected.
- **Fail-loud tests:** unknown per-channel key raises; camelCase alias still works.
- **Parity test:** the per-channel typed struct's field set equals the chart-level config struct's (guards future drift), and a key set per-channel produces the same SVG attribute as the same key set via `configure_axis`.
- **Rust round-trip:** `ChartSpec.from_json(s.to_json()) == s` for specs carrying typed per-channel axis/legend.
- **Golden stability:** no `tests/goldens/**` churn from honored-key or no-styling charts.

## 11. Resolved decisions

### R-Q3 — Shared styling struct (not direct `AxisConfigSpec` reuse)

Investigation (RCA §D) found `AxisConfigSpec` is **not** a clean per-channel superset: it carries chart-level scale-domain fields (`domain_min`/`domain_max`/`nice`/`zero`) and `x`/`y` show toggles that are meaningless per-channel. **Resolution:** factor a shared `AxisStyleSpec` (styling + positioning fields only); `AxisConfigSpec` = `#[serde(flatten)] AxisStyleSpec` + the chart-only fields; `EncodingSpec.axis = Option<AxisStyleSpec>`. Scale-domain stays chart-only (per-channel scale domain is the `x_scale_domain` path, not the axis). Legend has no chart-only-extra fields, so `LegendStyleSpec` is the full legend styling set and `LegendConfigSpec` flattens/extends it. One field set per concept → no drift; per-channel cannot set nonsensical fields.

### R-Q2 — Error type: Rust `deny_unknown_fields` → `ValueError` for v1

Rely on the typed-spec deserialization failure surfacing as the existing `ValueError` from the binding (the established PyO3 error path). A friendlier Python-side pre-check with did-you-mean (reusing the override registry) is a deliberate later enhancement, not v1. The hard requirement (fail loud, not silent) is met by `deny_unknown_fields`.

### R-Q1 — Per-orphan render semantics (implement all)

Each orphan field gets the concrete semantic below. Grouped by implementation cost; **★ marks a bounded interpretation** where the field accepts its full type but maps to a constrained behavior (flagged because a fuller version would need a new subsystem).

| Field | Resolved semantic | Anchor / cost |
|---|---|---|
| **axis `orient`** | Place the axis on the named side. Validate against the channel dimension: x→{top,bottom}, y→{left,right}; a cross-dimension value (x="left") **fails loud**. Reserve the label/title margin band on the chosen side. | `AxisLayout.orient` already carried + orient-aware `build_axis` (`marks/axis.rs:38-59`); reserve-on-side in layout. Med. |
| **axis `translate`** | Shift the axis group perpendicular to its line by N px (outward = positive), composing **additively** with the already-honored `offset`. | translate wrapper on the axis scene group (`compositor.rs` pattern). Low. |
| **axis `min_extent`/`max_extent`** | Clamp the reserved axis margin band to `[min,max]` px after the dynamic `estimate_x_label_band` / y-band computation. `min` = reserve at least; `max` = cap (labels may clip past it — documented). | `layout/axis.rs:353-400`. Low-med. |
| **axis `tick_extra`** | After tick generation, append a tick at each domain boundary if not already present. | scale `ticks_internal`. Low-med. |
| **axis `tick_min_step`** | Pass `min_step` into scale tick generation; drop ticks closer than `min_step` in data space. | scale tick methods. Med. |
| **axis `grid_opacity`** | Per-axis grid-line opacity, overriding the theme grid opacity for that axis. | `AxisLayout` field → `build_grid` (`marks/axis.rs:193`). Low. |
| **axis `title_orient`** | Side/orientation of the axis title relative to its axis (e.g. horizontal title on a left axis); adjust title rotation + position. | `AxisTitleLayout` gains orient (`marks/axis.rs:147-173`). Med. |
| **axis/legend `zindex`** ★ | **Bounded:** maps to coarse draw order, not arbitrary integer layering — `zindex >= 1` → axis/grid (or legend) drawn **above** marks; `<= 0` (default) → below marks (current behavior). Reuses the existing annotation above/below-marks ordering; arbitrary int z-order is NOT supported. | reuse annotation z-order mechanism. Med. **(see flag)** |
| **legend `row_padding`/`column_padding`** | Per-legend vertical (row) / horizontal (column) entry spacing, replacing the hardcoded `LEGEND_ENTRY_ROW_PAD=4.0`. | `layout/legend.rs:153,200-211,378-415`. Low. |
| **legend `symbol_stroke_width`** | Stroke width of legend symbols. | `marks/legend.rs:158-187`. Low. |
| **legend `label_limit`** ★ | **Bounded:** max legend-label pixel width; truncate with an ellipsis (`…`) and shrink the legend rect. No wrapping/tooltip. | `layout/legend.rs` label-width calc. Med. |
| **legend `clip_height`** ★ | **Bounded:** cap legend height; hard-clip overflow via an SVG `clipPath`/`overflow` on the legend group. No scrolling. | `marks/legend.rs:9-14`. Med. |
| **legend `tick_min_step`** | Min step between colorbar (gradient) ticks; same as axis `tick_min_step` for the colorbar tick generation. | colorbar tick gen. Med. |

**Flag for the user:** the three ★ bounded semantics (`zindex` → coarse below/above-marks; `label_limit` → ellipsis truncation; `clip_height` → hard clip) are pragmatic interpretations chosen to avoid new subsystems (a full integer z-order, text wrapping, scrolling). If any should be fuller-fidelity, that becomes its own scoped task.
