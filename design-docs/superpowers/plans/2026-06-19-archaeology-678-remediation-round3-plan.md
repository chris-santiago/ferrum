# Archaeology #6/#7/#8 Remediation — Round 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: chris-code:subagent-driven-development. Rust → rust-coder, Python → python-coder. EVERY task gets all three gates, each a SEPARATE dispatch: (1) spec-compliance review, (2) quality review, (3) review-lite commit gate. Announce model+agent on every dispatch.

## 1. Objective

Close the three real issues the round-2 re-review surfaced (DensityData faceted extent; factory-dict LayerChart chrome leak + chrome-key dedup; packed instance offset under the figure-title band), then re-run the full review+audit loop.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-19-archaeology-678-remediation-round3.md` — round-3 findings, contracts, non-goals
- Parent: `…-remediation-design.md` §3 (deferrals), §6 (contracts)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/transform/density_data.rs` | global_extent helper; remove dead _shared_extent (T1) |
| Modify | `crates/ferrum-core/src/render/prepare.rs` | dispatch DensityData in fix_transform_extents_for_facet (T1) |
| Modify | `src/ferrum/_overrides.py` | factory-dict chrome split for LayerChart; shared _FIGURE_CHROME_KEYS (T2) |
| Modify | `src/ferrum/composition.py` | _CompositeBase.properties uses shared _FIGURE_CHROME_KEYS (T2); packed-instance y-offset in _inject_figure_chrome (T3) |
| Test | inline `#[cfg(test)]` + `tests/` | per-task regression tests |
| Modify | `design-docs/superpowers/followups/2026-05-15-code-archaeology.md` | record round-3 outcomes + the deferred hover-tooltip limitation |

## 4. Constraints

- T1: pin DensityData extent over the full pre-facet batch, only when unset (never clobber user extent), RAW (no nice — mirror kde); extent math in the transform layer; non-faceted unchanged; remove the dead `_shared_extent` param (no inert interface).
- T2: split chrome keys for ANY chrome-intercepting chart-like (LayerChart + _CompositeBase), not only _CompositeBase; route chrome through `target.properties(**chrome)`, fan only non-chrome to children; subtitle/caption keep fanning to the merged chart for LayerChart; one shared `_FIGURE_CHROME_KEYS` used by both `_overrides` and `_CompositeBase.properties`. No inner-layer title leak; HTML `<title>` correct.
- T3: offset packed instance y by `header_h` wherever `_inject_figure_chrome` shifts a panel's scene nodes, so GPU marks align with the shifted plot_area/axes; small-data and non-titled composites byte-identical; no packed-format change (only the y field value shifts).
- Global: no matplotlib; no global mutable state; no WASM source change. Backward compat byte-identical where noted.
- Build `unset CONDA_PREFIX && uv run --no-sync maturin develop`. Rust tests run with `DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test -p ferrum-core` (binary links miniforge 3.13). RUN tests — do not infer from a launcher exit code.

## 5. Tasks

### Task T1: DensityData faceted extent pin
- [ ] `density_data::global_extent` (raw); dispatch arm in `fix_transform_extents_for_facet` (pin when unset); remove dead `_shared_extent`.
- [ ] Tests: faceted DensityData disjoint-range → shared extent (fail-before); user-extent preserved; raw-not-niced.
- [ ] Verify: `… cargo test -p ferrum-core --lib "fix_extents|global_extent|density"`

### Task T2: factory-dict chrome split for LayerChart + chrome-key dedup
- [ ] `_overrides._apply_overrides` splits chrome for LayerChart too; shared `_FIGURE_CHROME_KEYS` used by `_overrides` and `_CompositeBase.properties`.
- [ ] Tests: `properties={title=}` → figure-level for LayerChart (no leak, HTML `<title>` correct); composites unchanged; chrome-key constant referenced by both.
- [ ] Verify: `uv run pytest -n auto -k "overrides or layer or composite"`

### Task T3: packed instance offset under figure-title band
- [ ] In `_inject_figure_chrome`, offset packed instance y by `header_h` alongside the scene-node offset; align with shifted plot_area.
- [ ] Tests: >1000-mark titled composite → packed instance y shifted by header_h (== scene-node offset); small-data + non-titled unchanged.
- [ ] Verify: `uv run pytest -n auto -k "interactive or composite or packed"`

### Task T4: round-4 close-out + re-review loop
- [ ] Full suite (consistent env, RUN + confirm exit codes): `cargo test -p ferrum-core` and `uv run pytest -n auto`.
- [ ] Full heavyweight rust + python review + scene-pipeline + interactive audits (round 4). If real findings → spec+plan+execute round 5; loop until clean.
- [ ] Update archaeology doc; record deferred hover-tooltip limitation; `chris-code:finishing-a-development-branch` when converged.

## 6. Acceptance checks

- `cargo test -p ferrum-core` exit 0 all binaries; `uv run pytest -n auto` 0 failed.
- T1 faceted DensityData shares extent; T2 factory-dict LayerChart title figure-level + HTML correct + chrome-key single-source; T3 packed marks aligned under the title band.
- Per-task fail-before/pass-after; round-4 review+audits surface no confirmed correctness/class issue → converged.

## 7. Open questions

- None blocking (T1 pattern established; T2 split-predicate = "intercepts chrome"; T3 packed-record y-field offset confirmed during impl).
