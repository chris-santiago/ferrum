# Per-channel `axis=` / `legend=` Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task. Rust tasks → `rust-coder`; Python tasks → `python-coder` (per CLAUDE.md dispatch rule).

## 1. Objective

Type the per-channel axis/legend specs and route them into the chart-level config consumer at per-channel-wins precedence, then implement render support for every advertised `fm.Axis`/`fm.Legend` field — closing the B5 silent drop.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-14-per-channel-axis-legend-wiring-design.md` — full design (typed specs, cascade, casing, fail-loud, Q1 resolved = implement all)
- `design-docs/superpowers/followups/2026-06-14-per-channel-axis-legend-silent-drop-rca.md` — root cause, emitted-vs-consumed catalog, provenance

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/spec/encoding.rs` | replace `AxisSpec`/`LegendSpec` `{extra: Map}` with typed structs; `EncodingSpec.axis`/`.legend` field types; `deny_unknown_fields` + camelCase serde aliases |
| Modify | `crates/ferrum-core/src/render/chart_config.rs` | `AxisConfigSpec`/`LegendConfigSpec` gain the orphan fields (shared with the per-channel type) |
| Modify | `crates/ferrum-core/src/render/mod.rs` | `apply_axis_config_to_axis_input` (`:453`) — extend for orphan fields; apply per-channel axis after chart-level (per-channel wins) |
| Modify | `crates/ferrum-core/src/render/prepare.rs` | delete the axis `extra.get` ladder + D13 `color_legend_extra` block; route per-channel axis/legend through the typed config apply |
| Modify | `crates/ferrum-core/src/layout/axis.rs` | render orphan axis fields (`orient`, `translate`, `min_extent`/`max_extent`, `tick_extra`, `tick_min_step`, `grid_opacity`, `title_orient`) |
| Modify | `crates/ferrum-core/src/layout/legend.rs`, `crates/ferrum-core/src/render/marks/legend.rs` | render orphan legend fields (`clip_height`, `row_padding`/`column_padding`, `symbol_stroke_width`, `label_limit`, per-channel legend apply) |
| Modify | `src/ferrum/configure.py` | `AxisConfig`/`LegendConfig` dataclasses gain the orphan fields (chart-level parity) |
| Modify | `src/ferrum/axis.py`, `src/ferrum/legend.py`, `src/ferrum/encoding/base.py` | align `to_dict()` / `_normalize_*` / `to_encoding_spec_dict` emission with the typed snake-case struct |
| Test | `tests/test_phase_12_axis_legend.py` | upgrade `.to_dict()` assertions to render-level; add precedence, fail-loud, parity tests |

## 4. Constraints

- **No silent drops** — every per-channel axis/legend key renders or fails loud (`deny_unknown_fields` → `ValueError` at render). Spec §7.
- **Per-channel wins** the cascade over chart-level `configure_*` on its own axis/legend; the other axis keeps its configured value. Spec §6.
- **No golden churn** for charts that set only currently-honored keys or no axis/legend styling — those must render byte-identically. Spec §7. (Regenerate/inspect only goldens that legitimately change because a previously-dropped key now renders; visually verify per the golden rule.)
- **Back-compat** — raw-dict callers passing camelCase keys keep working via serde aliases. Spec §7.
- **Single schema** — the per-channel type and the chart-level config struct share one field set (no duplicate/drift). Spec §8 D1.
- **`cargo test` must pass**; `ChartSpec.from_json(to_json())` round-trips with typed per-channel specs.
- Snake_case is canonical; camelCase is alias-only. Spec §8 D4.
- Invoke `chris-code:regression-test` before declaring done.

## 5. Tasks

### Task 1: Factor shared style structs; type the specs (Rust + Python schema)
- [ ] In `chart_config.rs` (or a shared spec module), define `AxisStyleSpec`/`LegendStyleSpec` (styling+positioning fields, incl. the orphan fields from spec §11 R-Q1). Refactor `AxisConfigSpec` to `#[serde(flatten)] AxisStyleSpec` + chart-only fields (`x`/`y` toggles, `domain_min`/`domain_max`/`nice`/`zero`); `LegendConfigSpec` flattens `LegendStyleSpec`. (R-Q3: do NOT reuse `AxisConfigSpec` directly — scale-domain/`x`/`y` are not per-channel.)
- [ ] In `encoding.rs`, replace `AxisSpec`/`LegendSpec` `extra` maps so `EncodingSpec.axis: Option<AxisStyleSpec>` / `.legend: Option<LegendStyleSpec>`. Add `#[serde(deny_unknown_fields)]` + `#[serde(alias="labelAngle")]`-style aliases for every camelCase key the old reader accepted. Preserve the `axis=False`/`title=None` suppression contract.
- [ ] In `configure.py`, add the orphan fields to `AxisConfig`/`LegendConfig` (chart-level parity).
- [ ] Verify: `cargo build -p ferrum-core` + `ChartSpec.from_json(s.to_json())==s` for a spec with typed per-channel axis/legend.

### Task 2: Route per-channel into the config consumer (Rust)
- [ ] In `prepare.rs`, delete the axis `extra.get` ladder and the D13 `color_legend_extra` block. Apply each channel's typed axis/legend spec via the chart-level config path (`apply_axis_config_to_axis_input` and the legend equivalent), AFTER chart-level config so per-channel wins; `x.axis`→x-axis, `y.axis`→y-axis, legend-bearing channel→its legend.
- [ ] Verify (existing-renderer keys now work per-channel): `cargo test -p ferrum-core` + a Python render check that `fm.X("f", axis=fm.Axis(grid_color="#ccc"))` produces `#ccc` grid stroke.

### Task 3: Orphan axis render support (Rust)
- [ ] Implement each orphan axis field to the **resolved semantic in spec §11 R-Q1** (`orient` with x→top/bottom, y→left/right validation; `translate` additive with `offset`; `min_extent`/`max_extent` clamp the band; `tick_extra`; `tick_min_step`; `grid_opacity`; `title_orient`; `zindex` ★ coarse below/above-marks via the annotation z-order mechanism). Files per the spec §11 anchors (`layout/axis.rs`, `render/marks/axis.rs`, `mod.rs` apply, scale tick methods). Each field must have a render consumer; cross-dimension `orient` fails loud.
- [ ] Verify: `cargo test -p ferrum-core` + a render check per field.

### Task 4: Orphan legend render support (Rust)
- [ ] Implement each orphan legend field to the **resolved semantic in spec §11 R-Q1** (`row_padding`/`column_padding` replacing `LEGEND_ENTRY_ROW_PAD`; `symbol_stroke_width`; `label_limit` ★ ellipsis truncation; `clip_height` ★ hard clip via SVG clipPath; `tick_min_step` for colorbar; `zindex` ★ coarse below/above-marks). Files: `layout/legend.rs`, `render/marks/legend.rs`.
- [ ] Verify: `cargo test -p ferrum-core` + a render check per field.

### Task 5: Python serialization alignment (Python)
- [ ] Ensure `Axis.to_dict()`/`Legend.to_dict()` and `_normalize_axis`/`_normalize_legend`/`to_encoding_spec_dict` emit the snake-case field set the typed struct expects; drop any reliance on the old camelCase-only path. Add the orphan fields to the Python value classes if not already present.
- [ ] Verify: `uv run pytest tests/test_phase_12_axis_legend.py -v`

### Task 6: Render-level tests + golden stability (Python)
- [ ] Rewrite `test_phase_12_axis_legend.py` per-channel cases from `.to_dict()` to `.to_svg()` render assertions for the previously-dropped + orphan fields (label color, grid color/dash/width, domain styling, legend symbol geometry, orient, extents, clip_height, …). Add: precedence (per-channel beats `configure_axis`, both orders; other axis unaffected), fail-loud (unknown key raises), camelCase alias still works, and a parity test (per-channel field set == chart-level config field set; same key → same SVG attribute both ways).
- [ ] Verify: `uv run pytest tests/test_phase_12_axis_legend.py -v` + full `uv run pytest -n auto`; regenerate + visually inspect any legitimately-changed goldens.

## 6. Acceptance checks

- `cargo test` — all pass (after `unset CONDA_PREFIX && uv run --no-sync maturin develop`)
- `uv run pytest -n auto` — full suite green
- Spec §9 acceptance criteria each have a passing render-level test
- No golden churn except goldens that legitimately render a newly-honored key (visually verified)
- `chris-code:regression-test` invoked

## 7. Resolved decisions (was open questions)

Both resolved in spec §11 — implement to those:
- **Architecture (R-Q3):** factor shared `AxisStyleSpec`/`LegendStyleSpec` (styling+positioning); `AxisConfigSpec` flattens it + chart-only fields (`x`/`y` toggles, `domain_min`/`max`/`nice`/`zero`); `EncodingSpec.axis`/`.legend` use the style struct directly. NOT direct `AxisConfigSpec` reuse (leaks scale-domain per-channel). Drives Task 1.
- **Errors (R-Q2):** `deny_unknown_fields` → existing PyO3 `ValueError` at render for v1 (no Python pre-check yet).
- **Per-orphan semantics (R-Q1):** each orphan's concrete render behavior is pinned in the spec §11 table — implement those in Tasks 3-4. **Three are bounded interpretations (★):** `zindex` → coarse below/above-marks (reuse annotation z-order, not integer layering); `label_limit` → ellipsis truncation; `clip_height` → hard clip. If any should be fuller-fidelity, split it into its own task.
