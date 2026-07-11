# Secondary Y-Axis (GH #52) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development (recommended) or chris-code:executing-plans to implement this plan task-by-task.

## 1. Objective

Implement per-layer independent y-scales for LayerChart (dual axis, both output kinds) per the design spec, re-basing `fm.SecondaryY` onto the new subsystem and deleting `render/secondary_axis.rs`.

## 2. Spec references

- `design-docs/superpowers/specs/2026-07-11-secondary-y-axis-design.md` — all sections; §6 wire/slot contracts, §7 invariants, §9 acceptance
- `design-docs/superpowers/followups/2026-05-15-code-archaeology.md` — S2 row (resolved by this change)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/spec/layer.rs` | `independent_y` flag (serde default false) |
| Modify | `crates/ferrum-core/src/spec/chart.rs` | coerce flag in `coerce_layers` |
| Modify | `crates/ferrum-core/src/render/scale_resolve/mod.rs` | per-slot y scales on `ResolvedScales` |
| Modify | `crates/ferrum-core/src/render/scene_build.rs` | per-layer resolution + DrawCtx slot binding; axis emission per slot; CoordKind y-domain list; remove SecondaryY structural wiring (~:1416-1474) |
| Modify | `crates/ferrum-core/src/render/draw.rs` | DrawCtx y-slot access |
| Modify | `crates/ferrum-core/src/layout/mod.rs` | `reserve_axis_bands` left + n right bands |
| Modify | `crates/ferrum-core/src/layout/axis.rs` | stacked right y-axes, per-axis band offsets |
| Delete | `crates/ferrum-core/src/render/secondary_axis.rs` | superseded by desugar |
| Modify | `crates/ferrum-core/src/render/chart_config.rs` | remove `SecondaryYSpec` / `StructuralSpec::SecondaryY` |
| Modify | `crates/ferrum-wasm/src/param_runtime.rs` | per-slot y domains, pixel↔data per slot |
| Modify | `crates/ferrum-wasm/src/scene_load.rs` | slot index on draw commands |
| Modify | `crates/ferrum-wasm/src/{zoom_pan.rs,hit_test.rs,selection_state.rs}` | per-(panel,slot) rescale affine; readout via owning slot |
| Modify | `src/ferrum/composition.py` | relax `_validate_layer_resolve` (y), re-point x msg to #55; independent-y routing via `_build_merged` both kinds; nested-leaf lowering + shared-y conflict error |
| Modify | `src/ferrum/_spec_build.py` | `_build_layers_list` emits `independent_y` |
| Modify | `src/ferrum/chart.py` | `__add__` SecondaryY desugar (~:1428) |
| Modify | `src/ferrum/structural.py` | SecondaryY docstring (desugar semantics) |
| Test | `tests/test_secondary_y_axis.py` | new feature suite (static/interactive/nested/degenerate) |
| Modify | `tests/test_cohesion_share_scale_resolve_unification.py` | y-rejection tests → feature tests; x stays rejection citing #55 |
| Test | `tests/goldens/**` (new dual-axis + re-blessed SecondaryY) | visual proof |
| Modify | `ferrum-spec.md`, `CLAUDE.md`, archaeology doc | dated contract note; composite-exception amendment; S2 close |

## 4. Constraints

- **Slot contract (spec §6):** slot 0 = primary y (layer 0, left axis, gridlines); slot k = k-th `independent_y` layer (right axis, stacked outward). Mark geometry, axis ticks, and interactive domain state key off the same slot index — one resolution site.
- **Wire back-compat:** layer dicts without `independent_y` deserialize and render byte-identically to today.
- **Shared-path byte-stability:** default / `y:"shared"` LayerChart SVG bytes and scene JSON unchanged; existing goldens must not regress.
- **No Python-side scale math** — domains/nice-ing/pixel mapping stay in Rust; tooltips report raw data values.
- **Zoom/pan is panel-level; only `domainParam`/brush rescale is per-slot** (spec §8.5) — FA-16 screen-space stroke machinery must remain valid unchanged.
- **Goldens:** every new/regenerated golden goes through `tests/_snapshots.py::regen_and_verify` and the PNG is Read and visually confirmed before commit.
- Build: `unset CONDA_PREFIX && uv run --no-sync maturin develop` after every Rust task, before Python tests. Rust tests: `DYLD_LIBRARY_PATH=$(uv run python -c "import sys; print(sys.base_prefix + '/lib')") cargo test`. WASM: `source ~/.cargo/env && wasm-pack build crates/ferrum-wasm --target web --out-dir ../../src/ferrum/_wasm/`; clippy `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` (judge clippy by delta; ~166 pre-existing baseline failures on core).
- Dispatch rule: `.rs` → rust-coder, `.py` → python-coder; lite-review gate before every commit.
- TDD: feature tests written RED first (they currently fail with the overlay-contract ValueError).

## 5. Tasks

### Task 1: Layer flag + wire (rust-coder)
- [ ] Add `independent_y: bool` (serde default false) to `Layer`; coerce in `coerce_layers`
- [ ] Rust tests: absent-key default, roundtrip, coercion from Python dict shape
- [ ] Verify: `cargo test` (DYLD prefix per §4)

### Task 2: Per-slot y-scale resolution (rust-coder, model: opus)
- Consumes: slot contract (§4; spec §6); flag from Task 1
- [ ] `ResolvedScales` gains ordered per-slot y scales + layer→slot mapping; shared layers bind slot 0
- [ ] `resolve_panel_scales` resolves one y `ScaleKind` per independent layer from that layer's own encoding + transform outputs (explicit `scale=` wins; bar zero-anchor and y2 extension per-slot); today only layer 0 seeds — preserve that behavior for slot 0
- [ ] Bind each layer's `DrawCtx` to its slot
- [ ] Rust tests: per-slot domains from per-layer data; shared layers unchanged
- [ ] Verify: `cargo test`

### Task 3: Layout + axis emission (rust-coder)
- Consumes: slots from Task 2 → `ResolvedScales`
- [ ] `reserve_axis_bands`: left band + one right band per secondary slot, stacked outward (reuse `compute_y_label_band_width`/`compute_y_title_width` per axis)
- [ ] Emit one y-axis node per slot: slot 0 left, slots 1..n right at stacked offsets; title from the layer's y field/title; per-encoding `Axis(...)` and theme honored; gridlines from slot 0 only
- [ ] Rust tests: band math for n=1,2,3 secondaries; axis node count/orient/offsets; no plot-area overdraw
- [ ] Verify: `cargo test`

### Task 4: Python validation, routing, spec build (python-coder)
- Consumes: `independent_y` layer-dict key from Task 1 (spec §6)
- [ ] `_validate_layer_resolve`: accept `y:"independent"`; x message re-pointed to GH #55; `share_scale` path follows
- [ ] Independent-y LayerChart routes `to_svg` AND `_render_interactive` through `_build_merged`; flag set on every non-first layer; no y union-domain injection under independent
- [ ] `_build_layers_list` emits `independent_y`
- [ ] Flip y-rejection tests to feature tests in `tests/test_cohesion_share_scale_resolve_unification.py`; keep x rejection (assert #55 in message)
- [ ] New `tests/test_secondary_y_axis.py`: static SVG structural assertions (two y-axes, per-layer positioning, 3-layer stacking, temporal+numeric formatting, no-y-encoding layer joins primary, single-layer degenerate)
- [ ] Verify: `unset CONDA_PREFIX && uv run --no-sync maturin develop && uv run pytest -n auto` (full suite — shared contract files touched)

### Task 5: Nested lowering + conflict error (python-coder)
- Consumes: routing from Task 4 → `src/ferrum/composition.py`
- [ ] `_lower_any`: independent-y LayerChart nests as one leaf carrying its layer slots
- [ ] Parent composite explicit `y:"shared"` over such a leaf raises typed ValueError (spec §6 errors)
- [ ] Tests: HConcat nesting renders; conflict raises
- [ ] Verify: `uv run pytest -n auto`

### Task 6: SecondaryY desugar (python-coder)
- Consumes: `independent_y` key (Task 1); routing (Task 4)
- [ ] `Chart.__add__` structural handling: `+ SecondaryY(...)` appends one independent layer per spec §4 (base layers keep existing sharing; x inherited; axis/scale/color/opacity attached); stop emitting the structural spec
- [ ] Update SecondaryY docstring; update all pre-existing SecondaryY tests; regen affected goldens via `regen_and_verify` + PNG inspection
- [ ] Verify: `uv run pytest -n auto` (10 test files consume SecondaryY — suite-wide only)

### Task 7: Delete the secondary-axis silo (rust-coder)
- Consumes: desugar from Task 6 (nothing emits `SecondaryYSpec` anymore)
- [ ] Delete `render/secondary_axis.rs`; remove `SecondaryYSpec`/`StructuralSpec::SecondaryY` and scene_build wiring
- [ ] Verify: `cargo test` && `grep -rn "SecondaryYSpec\|secondary_axis" crates/` returns nothing && `uv run pytest -n auto` after rebuild

### Task 8: Scene contract — per-slot domains (rust-coder)
- Consumes: slot contract (Task 2)
- [ ] Panel coordinate state (CoordKind capture, `scene_build.rs` ~:231-249) carries ordered y-domain list; mesh + y-axis scene nodes carry slot index
- [ ] Back-compat: single-slot charts emit today's shape or a shape WASM treats identically (assert scene JSON unchanged for shared paths)
- [ ] Tests: scene JSON slot assertions; shared-path JSON byte-check
- [ ] Verify: `cargo test`

### Task 9: WASM runtime per-slot (rust-coder, model: opus)
- Consumes: scene contract from Task 8
- [ ] `param_runtime`: y `axis_domain`/`pixel_to_data`/`data_to_pixel` take a slot; `rescale_affine` per (panel, slot); domainParam/brush rebinds owning layer's slot only
- [ ] `scene_load`: draw commands bind (panel affine ∘ slot rescale affine); zoom/pan panel-level unchanged
- [ ] `hit_test`/`selection_state`: value readout inverts through owning slot
- [ ] Verify: `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` (delta) && wasm-pack build && `cargo test`

### Task 10: Goldens + interactive captures (python-coder + orchestrator inspection)
- Consumes: everything above
- [ ] New goldens: 2-layer dual axis, 3-layer stacked, temporal+numeric — `regen_and_verify`, orchestrator Reads each PNG
- [ ] Headless WASM captures (established harness, see memory `reference_headless_wasm_capture`): zoom-lock (layers move together, both axes relabel), per-layer domainParam rescale, tooltip raw-value readout
- [ ] Verify: acceptance §9.1–3, 7–8 evidence saved under `.claude/output/`

### Task 11: Docs + close-out sweep (python-coder or orchestrator)
- [ ] `ferrum-spec.md`: dated note on LayerChart resolve y + SecondaryY desugar (§3.12/§compound-views table "same axes" row)
- [ ] `CLAUDE.md` composite-rendering section: independent-y merged-path exception (both kinds)
- [ ] Archaeology doc: S2 row resolved; #52 follow-up entries closed
- [ ] Verify: `grep -rn "#52" src/ crates/` returns no stale pointers; `nox -s lint`

## 6. Acceptance checks

- `DYLD_LIBRARY_PATH=... cargo test` — green
- `unset CONDA_PREFIX && uv run --no-sync maturin develop && uv run pytest -n auto` — green
- Spec §9 criteria 1–12 each evidenced (goldens inspected, captures saved, byte-stability of shared paths confirmed against existing goldens)
- `grep -rn "SecondaryYSpec\|secondary_axis" crates/` empty; `grep -rn "#52" src/ crates/` no stale pointers

## 7. Open questions

- None blocking (spec §11). CoordKind's exact defining module is discovered at Task 8 (capture site is pinned).
