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

- `EncodingSpec.axis: Option<AxisStyleSpec>` and `EncodingSpec.legend: Option<LegendStyleSpec>` where `AxisStyleSpec`/`LegendStyleSpec` are the typed structs (reused/shared with `AxisConfigSpec`/`LegendConfigSpec`). Field names are snake_case mirroring `fm.Axis`/`fm.Legend`; camelCase legacy keys via `#[serde(alias = "...")]`; `#[serde(deny_unknown_fields)]`.
- **Cascade precedence (per axis):** per-channel `<channel>.axis` > chart-level `axis_x`/`axis_y` > chart-level `axis` > theme > Rust default. Equivalent for legend.
- **Suppression:** `axis=False` → suppression spec (unchanged); `title=None` → empty-title suppress (unchanged).

## 7. Invariants and constraints

- **No silent drops.** Every per-channel axis/legend key either renders or fails loud. (Closes B5.)
- **Per-channel beats chart-level** on its own axis/legend; the other axis keeps its chart-level value.
- **No golden churn** for charts that set only currently-honored keys (or none). The honored keys must render byte-identically; only previously-dropped keys change output (those have no goldens today).
- **Back-compat:** raw-dict callers passing camelCase keys (`axis={"labelAngle": -30}`) keep working via serde aliases.
- **`cargo test` must pass**; the typed-spec migration must not break round-trip JSON (`ChartSpec.from_json`).

## 8. Key decisions and tradeoffs

- **D1 — Reuse the chart-level config struct as the per-channel type (single schema).** The dropped keys are already rendered from `AxisConfigSpec`/`LegendConfigSpec`; making the per-channel spec the same type means zero new render code for those and guaranteed parity between per-channel and chart-level. *Rejected:* a fresh per-channel struct (duplicate schema, re-introduces drift). *Per Q1 (resolved: implement all):* `fm.Axis`/`fm.Legend` fields with no current `AxisConfigSpec`/`LegendConfigSpec` counterpart are **added** to the shared struct and given new layout/render support, so chart-level `configure_axis(orient=...)` starts working too. This is the scope-expanding consequence of the single-schema choice.
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

## 11. Open questions

- **Q1 — RESOLVED: implement all advertised fields.** Orphan `fm.Axis`/`fm.Legend` fields (`orient`, `translate`, `min_extent`/`max_extent`, `tick_extra`, `tick_min_step`, `grid_opacity`, `title_orient`, `zindex`; legend `clip_height`, `row_padding`/`column_padding`, `symbol_stroke_width`, `label_limit`) get added to the shared struct AND new layout/render support, so every advertised field renders (chart-level and per-channel). Sub-decision for the plan: render semantics for a few of these (`zindex` ordering, `translate`/`offset` interaction, `min_extent`/`max_extent` vs auto margins) need a concrete definition per field — enumerate and pin in the plan.
- **Q2 — Error type:** surface the deny_unknown_fields failure as the existing `ValueError` from the binding, or add a Python-side pre-check raising `FerrumOverrideError`-style with did-you-mean (reuse the override registry idea)? Default: rely on the Rust `ValueError` for v1; consider a friendlier Python pre-check later.
- **Q3 — Shared struct vs alias type:** make `EncodingSpec.axis` literally `Option<AxisConfigSpec>`, or introduce an `AxisStyleSpec` that both the encoding and chart-config reference? Default: reuse `AxisConfigSpec` directly if its field set is a superset; otherwise factor a shared struct.
