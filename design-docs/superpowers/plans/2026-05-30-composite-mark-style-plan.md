# Composite-mark constant style Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Make composite/statistical marks accept constant mark-style kwargs (`opacity`, `stroke_width`, …) like simple marks, by splitting transform vs style kwargs in `Chart._resolve_pending` and applying style to the resulting mark/layers.

## 2. Spec references

- `design-docs/superpowers/specs/2026-05-30-composite-mark-style-design.md` — full design (§4 behavior, §5 architecture, §6 helper contract, §7 invariants, §8 decisions, §9 acceptance, §10 validation)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `src/ferrum/chart.py` | Add `_split_style_kwargs` helper; in `_resolve_pending` split kwargs, desugar with transform-only, apply validated style to single-mark `_mark_kwargs` / all emitted layers |
| Modify | `src/ferrum/marks/base.py` | Only if needed: expose a reusable style-validation entry; otherwise reuse `MarkBase(...).to_mark_kwargs_dict()` as-is |
| Test | `tests/marks/test_composite_mark_style.py` | New behavioral tests (split, per-family style passthrough, layered, collisions, prior-layer isolation, typo) |
| Test | `tests/test_golden_configure.py` (or nearest density golden test) | New styled-density golden + check-or-update entry |
| Create | `tests/goldens/configure/density_styled.svg` (name at executor discretion) | Pinned styled-density output, visually inspected |

## 4. Constraints

- **Byte-identical goldens** for any composite mark with no style kwargs — empty `style_kwargs` ⇒ `_mark_kwargs` untouched. This is the behavior-preservation proof; zero diffs under `tests/goldens/**` and `crates/ferrum-core/tests/golden/**`.
- **Signature-based split**: keys naming a desugar parameter → transform; all others → style. Per-mark; no manual style/collision list. Collisions (`fill` on density, `cmap` on hex, `multiple`/`method`/`ci`/`density`) resolve to transform because they are declared desugar params.
- **Reuse `MarkBase`** for style validation, alias resolution (`color→fill`, `alpha→opacity`, `linetype→stroke_dash`), and `TypeError` on unknown keys. Match simple-mark error surface. Validate before desugar runs.
- **Auto-injected transform keys** (`groupby`, `y2_field`, `field`, smooth `name`) stay on the transform side.
- **Layered application**: flat style merged into every emitted layer's `mark_kwargs`, user value winning over desugar-set defaults; **prior primitive layer left untouched**.
- **Mark-kwargs only**: do NOT touch the encoding/`value()` path; constant-encoding is out of scope.
- **Python-only**: no Rust/Arrow/spec-schema changes. `python-coder` only; never general-purpose.
- Covers the full family: density, histogram, smooth, hex, contour, ribbon, errorbar, errorband, boxplot, boxen, violin, qq, raster, swarm, function.

## 5. Tasks

### Task 1: Split helper + style validation
- [ ] Add private `_split_style_kwargs(desugar_fn, user_kwargs) -> (transform_kwargs, style_kwargs)` (introspect `inspect.signature`) per spec §6.
- [ ] Validate/normalize `style_kwargs` via `MarkBase` into a canonical style dict; surface its `TypeError` for unknown keys.
- [ ] Verify: `uv run pytest -n auto -q` (no behavior change yet; helper unit-tested in Task 3).

### Task 2: Wire split + apply into `_resolve_pending`
- [ ] Split stored kwargs before the `desugar_fn(...)` call; pass only `transform_kwargs` (keep existing auto-injection of `groupby`/`y2_field`/`field`/`name` on the transform side).
- [ ] Single-mark result: set `new._mark_kwargs` = style dict (the `chart.py:492` branch, currently empty).
- [ ] Layered result (and the smooth-with-prior-layer branch): merge style dict into each emitted layer's `mark_kwargs`, user value precedence; leave the prior primitive layer untouched.
- [ ] Verify: `uv run pytest -n auto` — all pass, **zero golden diffs** (`git status` shows no `*.svg`/`*.sha256` changes).

### Task 3: Tests + styled golden
- [ ] `tests/marks/test_composite_mark_style.py`: `mark_density(opacity=0.4)` → translucent `rgba` fill; one representative style kwarg per composite family reaches output; `mark_smooth(opacity=0.4)` hits ribbon+line; `chart.mark_point().mark_smooth(opacity=0.4)` leaves scatter untouched; `mark_density(fill=False)` still a line; `mark_hex(cmap=...)` still transform; `mark_density(multiple="stack")` unaffected; typo raises `TypeError`.
- [ ] Add a styled-density golden via the existing check-or-update harness; regenerate with the repo's golden env var; rasterize to PNG (`python scripts/snapshot-goldens.py <name>`), `Read` the PNG, confirm translucent overlap before committing (CLAUDE.md golden rule).
- [ ] Verify: `uv run pytest -n auto -q tests/marks/test_composite_mark_style.py`.

### Task 4: Final gate
- [ ] `uv run pytest -n auto`, `cargo test` (conda/pdm-safe env), `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — all green.
- [ ] Confirm zero golden diffs beyond the one intentionally added styled golden.
- [ ] `/regression-test`.

## 6. Acceptance checks

- `uv run pytest -n auto` — all pass (4665 + new tests)
- `cargo test` — all pass; wasm clippy clean
- `mark_density(opacity=0.4)` renders translucent fills; family-wide style passthrough works; collisions route to transform; layered applies to all emitted layers; prior layer isolated; typo raises `TypeError`
- Zero golden diffs except the new inspected styled-density golden

## 7. Open questions

- None.
