# Override Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task. All code is Python (`src/ferrum/`, `tests/`) — dispatch each task to the `python-coder` agent.

## 1. Objective

Make `Chart.override(**kwargs)` apply at render with top-of-cascade precedence, validated Python-side against a generated registry, replacing the current silent no-op.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-14-override-wiring-design.md` — full design (scope, behavior, registry, cascade, validation, deprecation, acceptance)
- `design-docs/superpowers/followups/2026-06-14-override-silent-drop-rca.md` — root cause + provenance

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Create | `src/ferrum/_override_apply.py` | path registry (generated from schemas) + resolve/validate/apply; name avoids collision with existing `_overrides.py` |
| Create | `src/ferrum/exceptions.py` | `FerrumOverrideError` public exception (no exceptions module exists today) |
| Modify | `src/ferrum/__init__.py` | export `FerrumOverrideError` |
| Modify | `src/ferrum/_render.py` | call the applicator in/after `_resolve_chart_config` (`:478`); fold chart-config-targeted overrides last |
| Modify | `src/ferrum/chart.py` | apply encoding-scale / mark / coord overrides into the `to_spec` kw dict (`:3340`); apply `width`/`height` to render-time dimensions; keep `override()` (`:2495`) storage-only |
| Modify | `tests/test_override.py` | replace storage-only override tests with render-level + validation + deprecation tests (keep configure/purity tests) |

## 4. Constraints

- **Override wins the cascade:** apply override **last**, after all configure layers / per-channel settings. Spec §7.
- **Python-side validation is mandatory:** Rust spec structs silently drop unknown fields, so an unresolved path must raise `FerrumOverrideError` in Python — never reach Rust as a silent drop. Spec §8 D2.
- **Presentation-only scope:** override sets config / encoding-scale / mark-style / coord / `width`/`height` only. It must NOT set mark type, encoding field bindings, transforms, facet, layers, params. Spec §3.
- **Registry sourced from live schemas** (no hand-maintained leaf lists): `AxisConfig`/`LegendConfig`/`TitleConfig`/`GridConfig`/`PaddingConfig`/`ColorConfig` (`configure.py`); scale fields (`encoding/_scale.py`); `marks/base.py:_VALID_MARK_KWARGS`; `CoordKind` variant fields; `properties()` signature (`width`/`height`).
- **No golden churn:** charts that never call `.override()` render byte-identical.
- **`.override()` stays pure/immutable;** validation happens at render, not at call time. Spec §8 D6.
- **Per-channel encoding `axis=`/`legend=` are out of v1 override scope.** Route axis/legend overrides through the typed chart-config target (`x_axis_*`/`y_axis_*`/`legend_*` → `AxisConfig`/`LegendConfig`). The per-channel `AxisSpec`/`LegendSpec` are opaque `serde_json::Map`s (zero named fields); their de-facto valid keys exist only as scattered `.extra.get("…")` calls in `render/prepare.rs` (alias-cased, e.g. `labelAngle`/`label_angle`) and are near-fully redundant with the typed path. `<channel>_axis_*`/`<channel>_legend_*` raise `FerrumOverrideError` in v1 — accepting them un-validated would reopen the silent-drop hole.
- Invoke `chris-code:regression-test` before declaring done (TDD render-level tests are the load-bearing layer that B4 lacked).

## 5. Tasks

### Task 1: Path registry + `FerrumOverrideError`
- [ ] Create `FerrumOverrideError` in `src/ferrum/exceptions.py`; export from `__init__.py`.
- [ ] Create `src/ferrum/_override_apply.py` with a registry mapping each path/prefix to `(target, spec-location, valid-leaf-set, typed-equivalent)`, leaf-sets generated from the schemas in §4. Targets: chart-config (`x_axis_`/`y_axis_`/`axis_`/`legend_`/`title_`/`grid_`/`padding_`/`color_`), encoding-scale (`<channel>_scale_`), mark (`mark_`), coord (`coord_`), properties (`width`/`height`). Spec §5–6.
- [ ] `resolve(path)` → longest-prefix-first lookup returning the registry entry or marking unknown. Spec §5.
- [ ] Verify: `uv run pytest tests/test_override.py -k registry -v`

### Task 2: Validation + did-you-mean
- [ ] In `_override_apply.py`, validate a full override dict: unknown prefix or invalid leaf → `FerrumOverrideError` naming the path + closest match via `difflib.get_close_matches` over the registry's known paths. Spec §4, §9.
- [ ] Verify: `uv run pytest tests/test_override.py -k "error or suggest" -v`

### Task 3: Render-prep consumer (cascade application)
- [ ] Apply chart-config-targeted overrides by deep-merging into the dict from `_resolve_chart_config` (`_render.py:478`), after configure layers (override wins).
- [ ] Apply encoding-scale / mark / coord overrides into the `to_spec` kw dict (`chart.py:3340`) before `ChartSpec(**kw)`; `mark_*` targets the base/primary mark, error if no single primary (spec §11 Q4).
- [ ] Apply `width`/`height` overrides to the render-time dimensions, beating `.properties()` (spec §8 D5).
- [ ] Run validation (Task 2) once per render before applying.
- [ ] Verify: `uv run pytest tests/test_override.py -k "apply or cascade" -v`

### Task 4: Deprecation routing
- [ ] For a resolved path whose registry entry has a typed equivalent, emit `DeprecationWarning` naming the `.configure_*`/`.properties` method, then apply. Paths without a typed equivalent apply with no warning. Spec §4, §8 D3.
- [ ] Verify: `uv run pytest tests/test_override.py -k deprecat -v`

### Task 5: Replace hollow tests with render-level coverage
- [ ] In `tests/test_override.py`, replace the storage-only override assertions with: per-category render assertions on the SVG (axis angle, width viewBox, legend orient, color scheme, scale domain, mark corner radius, coord) per spec §9; cascade-conflict (both call orders); validation errors (unknown prefix / bad leaf / typo); deprecation warnings; retain purity/merge tests. Keep existing `configure_*` tests untouched.
- [ ] Verify: `uv run pytest tests/test_override.py -v`

## 6. Acceptance checks

- `uv run pytest tests/test_override.py -v` — all pass (render + validation + deprecation + purity)
- `uv run pytest -n auto` — full suite green; no golden churn (spot-check one non-override golden is byte-identical)
- Spec §9 acceptance criteria each have a passing test
- `chris-code:regression-test` invoked

## 7. Open questions

- **Per-channel encoding axis/legend** — decided (see §4 constraint): excluded from v1, raise `FerrumOverrideError`; route via the typed chart-config target. Recorded here only so a future v2 can revisit if a per-channel-only key is genuinely needed.
- **Rust `deny_unknown_fields` backstop** (spec §11 Q3) is a deliberate follow-up, not in this plan.

## 8. Out-of-scope follow-ups (file separately)

- **Per-channel axis/legend silent-drop (latent, unrelated to override).** `render/prepare.rs` hand-reads only ~7 of `AxisConfig`'s fields from the per-channel `AxisSpec.extra` map (`labels`, `ticks`, `domain`, `grid`, `labelAngle`/`label_angle`, `title`, `tick_count`/`tickCount`) and ~3 legend keys (`disabled`, `title`, `format`). Other keys passed via `encode(x=fm.X(..., axis={...}))` (e.g. `label_color`) are silently dropped. Candidate for its own RCA/fix; do not bundle into the override work.
