# Composite Figure-Level Shared Legend + `Resolve(legend=)` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Render one figure-level legend for composites with a shared `color`/`size` scale (suppressing per-panel legends) and ship the `fm.Resolve(scale=, legend=)` axis, per the design spec.

## 2. Spec references

- `design-docs/superpowers/specs/2026-07-12-composite-shared-legend-design.md` — all sections; §4 behavior, §6 contracts (Python API, wire, semantic rule, seam), §8 decisions, §9 acceptance.
- `ferrum-spec.md §3.9` — contract being implemented/narrowed.

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/spec/composite.rs` | `CompositeResolve` legend sub-struct + effective-mode rule (spec §6) |
| Modify | `crates/ferrum-core/src/render/scale_resolve/seam.rs` | seam: suppression channels in, prepared legend bundle out (spec §6) |
| Modify | `crates/ferrum-core/src/render/composite_render.rs` | pass-2 suppression/capture; pass-3 legend band before root chrome |
| Modify | `crates/ferrum-core/src/render/composite.rs` | expose per-leaf participation per channel to the render pass |
| Modify | `crates/ferrum-core/src/layout/mod.rs` | layout-stage legend suppression hook (`reserve_legends` skip) |
| Modify | `crates/ferrum-core/src/render/prepare/legend.rs` | surface prepared bundle for capture (if not already reachable) |
| Modify | `src/ferrum/composition.py` | `Resolve` class, normalization, mode-matrix validation, wire emission |
| Modify | `src/ferrum/__init__.py` | export `fm.Resolve` |
| Modify | `src/ferrum/plots/matrix.py` | jointplot internal shared resolve; delete pairplot residual comment |
| Test | `crates/ferrum-core` unit tests (same modules) | round-trip, mode matrix, suppression sets, band geometry, capture |
| Test | `tests/test_composite_shared_legend.py` | Python behavior + error surfaces + pairplot/jointplot regression |
| Test | `tests/goldens/**` (new shared-legend goldens) | visual contract |
| Modify | `ferrum-spec.md` | §3.9 dated note: `Resolve(scale, legend)`; axis → follow-up issue |
| Modify | `design-docs/superpowers/followups/2026-05-15-code-archaeology.md` | close C4-residual row (line ~259) |

## 4. Constraints

- **Byte stability:** any composite whose effective legend resolution is all-independent (incl. every composite without scale sharing) must render byte-identically to current main — static and interactive. Existing independent-resolve goldens must not change.
- **One mechanism:** figure legend built once into `SceneGraph.legend`; zero changes in `crates/ferrum-wasm`.
- **Reuse, no parallel legend impl:** figure legend must go through the existing prepare/layout/draw legend primitives (`build_color_legend` outputs, `layout_legend`/`layout_colorbar`/`layout_aux_legends`, `marks::legend::build_legend`).
- **Semantic rule (spec §6):** effective legend mode = explicit `legend[channel]` else `scale[channel]`; `shared` legend over non-shared scale raises typed `ValueError` at Python lowering — never a silent fallback, never `NotImplementedError`.
- **Explicit per-chart `scale=` wins:** leaves excluded from the domain union keep their own panel legend (no suppression).
- **User `legend=None` on a leaf** keeps prepare-stage suppression; such leaves are skipped for capture; all-disabled → no figure legend.
- **Band ordering:** legend band applies at the resolving node after children are placed, before root-chrome injection (title band must offset it).
- **Coding-agent dispatch (CLAUDE.md):** `.rs` → `rust-coder`, `.py` → `python-coder`; lite-review gate before every commit.
- **Goldens:** every new/changed golden rasterized via `tests/_snapshots.py` `regen_and_verify` and the PNG visually inspected before commit.
- **RED proof:** the pairplot one-legend regression test must fail against unpatched main (stash-based proof, flags-before-`--`, no bare `pop`).
- Branch `feat/composite-shared-legend`; no `git push`.
- Rust tests: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test`; rebuilds via `unset CONDA_PREFIX && uv run --no-sync maturin develop`.

## 5. Tasks

### Task 1: Wire contract — `CompositeResolve` legend axis (Rust)
- [ ] Add optional legend sub-struct (`color`/`size`) to `CompositeResolve` per spec §6 wire shape; absent = follow scale; update `is_default` predicate so all-default still serializes to nothing
- [ ] Add effective-legend-mode helper implementing spec §6 semantic rule
- [ ] Round-trip + default-serialization tests (existing payloads deserialize identically)
- [ ] Verify: `cargo test -p ferrum-core composite`

### Task 2: Seam — suppression in, bundle out (Rust)
- Consumes: wire contract from Task 1; seam contract from spec §6
- [ ] Extend the leaf render seam so the compositor can (a) name channels whose legends are layout-suppressed, (b) receive the leaf's prepared legend bundle (entries/colorbar/title/aux/style overrides) for those channels
- [ ] `reserve_legends` skips reservation for suppressed channels while prepare still builds inputs; user `disabled` unchanged (empty bundle)
- [ ] Unit tests: suppressed leaf reserves no gutter, bundle non-empty; user-disabled leaf yields empty bundle
- [ ] Verify: `cargo test -p ferrum-core`

### Task 3: Compositor legend band (Rust)
- Consumes: Task 2 seam; participation info from `resolve_composite_scales` contexts; band idiom per spec §5/§8.1
- [ ] At each composite node with effective shared legend resolution: suppress participating leaves' channel legends in pass 2, capture first non-empty bundle (pre-order)
- [ ] Pass 3: measure via existing layout primitives against merged extent, grow scene on the oriented edge (composition-level/theme `legend_orient`, default right), draw via `build_legend`, append to merged `SceneGraph.legend`; ordered before `inject_root_chrome`
- [ ] Nested attachment: band at the declaring node only; non-participating leaves untouched
- [ ] Unit tests: band geometry per orient, one-legend node count, nested case, all-disabled → no band, color+size stacked band
- [ ] Verify: full `cargo test` (shared-contract change: scene output)

### Task 4: Python `Resolve` + wire emission
- Consumes: wire contract from Task 1 / spec §6
- [ ] `Resolve(scale=None, legend=None)` value class in `composition.py`; flat dict ≡ `Resolve(scale=dict)`; accepted everywhere `resolve=` is today (incl. `share_scale` construction path)
- [ ] Extend `_validate_resolve`/`_composite_resolve_field`: legend channels `color`/`size` only; mode-matrix validation with typed `ValueError` naming channel and both modes; emit `"legend"` sub-object on the node resolve field
- [ ] Export `fm.Resolve` in `__all__`
- [ ] Tests: normalization, wire shape, error surfaces, back-compat flat dict
- [ ] Verify: `uv run pytest tests/test_composite_shared_legend.py -v` then full `uv run pytest -n auto` (shared surface: `resolve=` touches all compositions)

### Task 5: pairplot/jointplot defaults + regression (Python)
- Consumes: Tasks 3–4 (rebuild extension first: `unset CONDA_PREFIX && uv run --no-sync maturin develop`)
- [ ] Delete the pairplot residual comment (`matrix.py:331-335`); behavior now ships
- [ ] `jointplot(hue=)` sets internal shared-color resolve on its grid node (spec §8.6)
- [ ] Regression tests (spec §9.1/§9.9): exactly-one-legend SVG assertions for `pairplot(hue=)` and `jointplot(hue=)`; RED-proof against main per Constraints
- [ ] Tests for spec §9.5 (opt-out), §9.7 (explicit leaf scale), §9.8 (disabled leaves), §9.12 (`markers=` shape collapse preserved)
- [ ] Verify: `uv run pytest -n auto`

### Task 6: Goldens + interactive capture
- Consumes: Tasks 3–5
- [ ] New goldens: pairplot(hue=) categorical; concat shared continuous colorbar; shared size; shared-scale + independent-legend opt-out; nested-share case; title+legend co-render — each via `regen_and_verify`, PNG read and inspected
- [ ] Confirm existing independent-resolve goldens byte-identical (`git status` on `tests/goldens/`)
- [ ] Headless WASM capture of pairplot interactive (spec §9.13): one legend, panels intact
- [ ] Verify: `uv run pytest -n auto` and golden PNGs inspected

### Task 7: Spec, docs, archaeology close-out
- [ ] `ferrum-spec.md` §3.9: dated note narrowing `Resolve` to `(scale, legend)` + default legend-follows-scale; §2 line 154 updated to match
- [ ] File follow-up GH issue for `Resolve(axis=)`; reference it in the §3.9 note
- [ ] Close C4-residual row in the archaeology doc (link #16); check off action-list entry
- [ ] `unset CONDA_PREFIX && uv run --no-sync python scripts/gen_api_pages.py` (home `fm.Resolve`; fix UNHOMED if flagged); cache-cleared docs build check per CLAUDE.md
- [ ] Verify: `nox -s docs`

## 6. Acceptance checks

- `uv run pytest -n auto` — all pass
- `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test` — all pass
- Spec §9 criteria 1–14 all observable; goldens visually inspected; RED proof recorded
- Existing independent-resolve goldens unchanged (byte-identical)

## 7. Open questions

None.
