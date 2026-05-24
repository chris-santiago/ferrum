# Bug Hunt R1+R2 Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `chris-code:subagent-driven-development` (recommended) or `chris-code:executing-plans` to implement this plan task-by-task.

## 1. Objective

Fix all 9 remaining open bugs and 12 latent bugs from bug-hunt rounds 1 and 2 (`.claude/output/bug-hunt/ALL_BUGS_2026-05-24.md`).

## 2. Spec references

- `.claude/output/bug-hunt/ALL_BUGS_2026-05-24.md` — full bug listing with root causes, file locations, test names
- `.claude/output/bug-hunt/BUG_REPORT_2026-05-24_R2.md` — R2 detail

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `src/ferrum/_coerce.py` | R2-3a/3b/3c: RecordBatch, dictionary, Null-type coercion |
| Modify | `src/ferrum/composition.py` | R2-1: add configure_*() to _ChartLike or _CompositeBase |
| Modify | `src/ferrum/plots/regression.py` | R2-4: residplot _label passthrough; R2-6: lmplot int→Float64 |
| Modify | `src/ferrum/plots/distribution.py` | R2-5: displot kde=True column mismatch |
| Modify | `src/ferrum/_interactive.py` | R1-O1: wire annotations into interactive scene |
| Modify | `crates/ferrum-core/src/scale/log.rs` | R2-8: nice() negative domain inversion |
| Modify | `crates/ferrum-core/src/projection.rs` | R2-9: Albers rho0 formula; R2-10: albers_usa_inv insets |
| Modify | `crates/ferrum-core/src/render/svg.rs` | D-1: fmt_f "-0" |
| Modify | `crates/ferrum-core/src/render/svg_walk.rs` | D-2: NaN opacity in emit_text |
| Modify | `crates/ferrum-core/src/render/annotation.rs` | D-3: NaN span; D-4/5/6: baseline divergence; D-7: epsilon arrow |
| Modify | `crates/ferrum-core/src/render/` (tooltip) | D-8: hardcoded day spacing |
| Modify | `crates/ferrum-core/src/layout/mod.rs` | L-3: x_title_gutter per-axis override |
| Modify | `crates/ferrum-core/src/layout/axis.rs` | L-2: negative label_padding guard; L-4: use effective_label_padding |
| Modify | `crates/ferrum-core/src/render/chart_config.rs` | Fix stale inline test assertion (Vec<Value>) |

## 4. Constraints

- **No source-level regressions.** `uv run pytest -n auto` and `cargo test` must pass before and after each task.
- **Rebuild Rust** after every Rust change: `unset CONDA_PREFIX && uv run --no-sync maturin develop`
- **Do not modify test files** written by bug-hunt agents — they are the acceptance criteria.
- **configure_*() on composed types** must delegate to each child chart, not duplicate Chart's implementation. Store config on the composition and apply at render time, or forward to each sub-chart.
- **Interactive annotations** must be injected into the scene JSON after `render_interactive()` returns, matching the SVG path's post-hoc annotation injection pattern — don't try to thread annotations through the Rust ChartSpec.
- **Albers rho0 formula** must match Snyder USGS PP 1395 eq. 14-3a exactly: `rho0 = sqrt(C - 2*n*sin(phi0)) / n`.

## 5. Tasks

### Task 1: Python coercion fixes (R2-3a, R2-3b, R2-3c)
- [ ] `_coerce.py:119-120` — change RecordBatch path to `return to_arrow_table(pa.Table.from_batches([data]))` for recursive re-entry through date cast
- [ ] `_coerce.py` pa.Table handler — after date cast loop, add pass to decode dictionary-encoded columns (`col.dictionary_decode()` when `pa.types.is_dictionary(field_type)`)
- [ ] `_coerce.py` pa.Table handler — cast `pa.types.is_null(field_type)` columns to `pa.float64()`
- [ ] Verify: `uv run pytest tests/test_bug_hunt_coerce_transport.py -k "recordbatch or dictionary_encoded or all_null_column" --tb=short -q`

### Task 2: configure_*() on composed chart types (R2-1)
- [ ] Add all 7 configure methods (`configure`, `configure_axis`, `configure_legend`, `configure_title`, `configure_grid`, `configure_padding`, `configure_color`) to `_ChartLike` or `_CompositeBase` in `composition.py`
- [ ] Implementation: store `_configure_layers` list on the base class; each method appends a Configure layer (same pattern as `Chart`). At render time, `_resolve_chart_config` on each sub-chart merges the composition-level config
- [ ] Verify: `uv run pytest tests/test_bug_hunt_scale_stat.py::test_configure_on_layer_chart tests/test_bug_hunt_model_diagnostics.py::test_residuals_chart_multi_panel_configure --tb=short -q`

### Task 3: Figure-function fixes (R2-4, R2-5, R2-6)
- [ ] `regression.py` residplot — inject `_label` column AFTER the transform, not before. Add `_label` as a passthrough column in the encode dict, or use a separate layer for the labeled series that doesn't go through the Smooth transform
- [ ] `distribution.py:270-276` displot kde=True — the KDE overlay must use the original x column name, not the histogram-renamed column. The `kde_layer` at line 271 already constructs a fresh `Chart(data)` with the original `x` encoding, so the bug is likely in how layers share data batches. Investigate whether the `+` operator merges transforms incorrectly
- [ ] `regression.py` lmplot — add integer→Float64 cast before building the Chart, matching catplot's pattern at `chart.py:443-465`. Apply to both x and y columns when dtype is integer
- [ ] Verify: `uv run pytest tests/test_bug_hunt_figure_api.py -k "residplot_with_label or displot_hist_with_kde or lmplot_integer" --tb=short -q`

### Task 4: Interactive annotations (R1-O1)
- [ ] In `_interactive.py:_render_scene()`, after `render_interactive()` returns `(scene_json, packed)`, parse `scene_json`, inject annotation nodes into each panel's `annotations` list using the same annotation-resolution logic that the SVG path uses (`_annotations.py`)
- [ ] The chart's `_annotations` and `_chart_config_dict.get("annotations")` contain the annotation specs; resolve coordinates against each panel's scale domains/ranges and emit scene-compatible annotation nodes
- [ ] Verify: `uv run pytest tests/test_bug_hunt_phase_11_interactive.py -k "annotate" --tb=short -q`

### Task 5: LogScale.nice() negative domain (R2-8)
- [ ] `log.rs:85-101` — for negative domains, the niced endpoints are sign-flipped. After computing `new_lo` and `new_hi`, the domain-order check `self.domain[0] <= self.domain[1]` produces inverted results because `sign * base^lo_exp` and `sign * base^hi_exp` swap relative magnitudes. Fix: when `neg`, swap new_lo and new_hi before applying the domain-order logic
- [ ] Verify: `uv run pytest tests/test_bug_hunt_scale_stat.py::test_log_scale_negative_domain_nice --tb=short -q`

### Task 6: Albers projection fixes (R2-9, R2-10)
- [ ] `projection.rs:192` — change `rho0 = c.sqrt() / n - phi0.sin() / n` to `rho0 = (c - 2.0*n*phi0.sin()).max(0.0).sqrt() / n` (matches rho computation at line 195 and Snyder eq. 14-3a)
- [ ] `projection.rs:219` — apply same fix to `albers_usa_inv`
- [ ] `projection.rs:214-225` — add Alaska/Hawaii inset detection in `albers_usa_inv`: check if (x,y) falls in the Alaska or Hawaii output region and apply the corresponding inverse (reverse the scale+offset, then inverse conic with the inset's standard parallels)
- [ ] Rebuild: `unset CONDA_PREFIX && uv run --no-sync maturin develop`
- [ ] Verify: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-core --tests -- bug_hunt_r2_albers --nocapture`

### Task 7: Draw/SVG latent bugs (D-1 through D-8)
- [ ] `svg.rs` fmt_f — extend the `trimmed == "-"` check to also catch `"-0"`: `if trimmed == "-" || trimmed == "-0" { "0" }`
- [ ] `svg_walk.rs` emit_text — guard NaN opacity same way as the fixed `with_opacity` path (default to 1.0 or clamp before cast)
- [ ] `annotation.rs` emit_span — guard NaN start/end coordinates (skip or clamp)
- [ ] `annotation.rs` parse_baseline — add `"hanging"` → Top, `"text-before-edge"` → Top, `"ideographic"` → Bottom (match draw.rs `to_scene_text_style`)
- [ ] `annotation.rs` arrow rendering — clamp arrowhead size to shaft length (`head_len = head_len.min(shaft_len)`)
- [ ] Tooltip time formatting — compute `spacing_ms` from actual data range instead of hardcoding `86_400_000`
- [ ] Rebuild and verify: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-core --tests -- bug_hunt_r2 --nocapture`

### Task 8: Layout latent bugs (L-2, L-3, L-4)
- [ ] `axis.rs` — guard negative `label_padding` with `.max(0.0)`
- [ ] `mod.rs:480-483` x_title_gutter — use `axes.x.title_font_size.unwrap_or(theme.title_font_size)` matching the y-axis pattern
- [ ] `axis.rs` estimate_x_label_band — use `effective_label_padding` in the band height calculation (add it to the final height)
- [ ] Note: L-1 (negative facet spacing) is a validation gap, not a render bug — add `.max(0.0)` guard on spacing in FacetGrid compute methods
- [ ] Rebuild and verify: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-core --tests -- bug_hunt --nocapture`

### Task 9: Fix stale Rust inline test
- [ ] `chart_config.rs:495` — update assertion from `Some(vec![0.0, 100.0])` to use `serde_json::Value` (e.g., `Some(vec![Value::from(0.0), Value::from(100.0)])`) to match the new `Vec<Value>` domain type
- [ ] Rebuild and verify full cargo test passes

## 6. Acceptance checks

- `uv run pytest tests/test_bug_hunt_*.py --tb=short -q` — 0 failures (currently 18)
- `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test -p ferrum-core` — compiles and 0 failures
- `uv run pytest -n auto` — full suite passes (no regressions)
- All latent bug tests that previously pinned broken behavior now assert correct behavior

## 7. Open questions

- **Task 4 (interactive annotations):** Should annotations be injected Python-side after `render_interactive()` returns, or should the Rust `render_interactive` be extended to accept and render annotations? Python-side is lower risk and matches SVG path pattern, but duplicates coordinate resolution logic. Recommend Python-side for now.
- **Task 3 (displot kde=True):** Need to confirm whether the bug is in layer data merging or in how the KDE layer references the original column after histogram transform renames it. The `kde_layer` constructs a fresh `Chart(data)` with original column names, so the issue may be deeper in layer composition.
