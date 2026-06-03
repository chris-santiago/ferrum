# WASM Interactive-Renderer Correctness Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: chris-code:subagent-driven-development. All code is `.rs`/`.wgsl` in `crates/ferrum-wasm/` → `rust-coder`. Tasks 1 and 2 both modify `render.rs` and `pipelines.rs` → they share files and MUST run **sequentially** (Task 1 fully reviewed + committed before Task 2). Task 3 is docs-only. Each WASM fix is browser-validated separately after its commit.

## 1. Objective

Make WASM mark transforms per-panel (FA-18) and add MSAA to the main render pass (FA-19), and mark FA-17 closed in the trackers.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-03-wasm-interactive-renderer-correctness-design.md` §4–§10 (FA-18 §5 per-panel transform, FA-19 §5 MSAA, §6 contracts, §7 invariants).

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-wasm/src/scene_load.rs` | add `panel_id` to `MarkMeshPanel` + mark `DrawCommand`; set it where `plot_area` is recorded |
| Modify | `crates/ferrum-wasm/src/render.rs` | per-panel transform slots in `GpuBuffers`; bind per-panel affine in mesh + instance draw loops; upload all panels' affines; MSAA color target + resolve in `render_frame` |
| Modify | `crates/ferrum-wasm/src/pipelines.rs` | shared `sample_count` on all main-pass pipelines (mesh, textured, instanced rect/circle + additive) |
| Modify | `crates/ferrum-wasm/src/gpu.rs` (if MSAA texture/capability lives here) | backend MSAA capability query + multisampled texture (recreate on resize) |
| Modify | `crates/ferrum-wasm/src/lib.rs` | upload paths read full `zoom.transforms` vector, not a single panel |
| Modify | `src/ferrum/_wasm/*` (rebuilt bundle) | wasm-pack output after each WASM task |
| Modify | `CLAUDE.md`, `design-docs/superpowers/followups/2026-05-15-code-archaeology.md` | mark FA-17 closed (commit `0e82b88`) |

## 4. Constraints

- **At-rest / single-panel byte-stability:** all panels at identity ⇒ per-panel binding yields the same pixels as today's single-uniform path.
- **Identity uniform untouched:** axes/grid/legend/title/annotations stay on identity and fixed under zoom/pan/rescale.
- **Static SVG path untouched** (`ferrum-core`); goldens unchanged.
- **One shared sample count** across the pass color attachment and ALL pipelines in it — mismatch is a wgpu validation error; make it impossible by construction.
- **No hard MSAA dependency:** if the backend reports 4× unsupported, fall back to 1× silently (no panic, no error surfaced). No `NotImplementedError`/warn-fallback that drops functionality.
- `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` clean.

## 5. Tasks

### Task 1: FA-18 — per-panel mark transform (sequential; do first)
- [ ] `scene_load.rs`: add `panel_id: usize` to `MarkMeshPanel` and to mark-kind `DrawCommand`; populate it at the sites that already set `plot_area` (mesh-panel recording + mark draw-command emission).
- [ ] `render.rs`: generalize the single mark uniform to one transform slot per panel (mechanism per spec §5 — dynamic-offset uniform buffer with alignment-padded `Uniforms` per panel, or a `Vec` of per-panel buffers+bind groups). Identity uniform unchanged.
- [ ] `render.rs`: in the mark-mesh loop bind `zoom_transforms[panel.panel_id]`; in the per-batch instance loop bind `zoom_transforms[cmd.panel_id]` for `is_mark` commands (identity for non-mark). Upload every panel's affine each frame from the full `&[Affine2]`.
- [ ] `lib.rs`: ensure rescale/zoom/pan upload paths drive `zoom.transforms[panel]` and the render consumes the whole vector (drop any single-current-panel upload assumption).
- [ ] Unit test: a 2-panel scene records distinct `panel_id`s; with `zoom_transforms` holding one rescaled (sx≠sy) entry, the per-panel selection picks the rescaled affine only for that panel (structural, no GPU).
- [ ] Verify: `cargo test -p ferrum-wasm`; `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings`; `cargo build -p ferrum-wasm --target wasm32-unknown-unknown`.
- [ ] Rebuild bundle: `source ~/.cargo/env && wasm-pack build crates/ferrum-wasm --target web --release --out-dir ../../src/ferrum/_wasm/`.
- [ ] **Browser-validate FA-18 before starting Task 2** (focus+context: siblings fixed, overview line persists during brush).

### Task 2: FA-19 — MSAA on the main render pass (sequential; after Task 1)
- [ ] Choose one sample count `N ∈ {1,4}` from backend capability (query at pipeline build / context init); thread it to every main-pass pipeline builder in `pipelines.rs` (`mesh`, `textured`, `instanced_rect`/`_additive`, `instanced_circle`/`_additive`) via `multisample.count = N`.
- [ ] `render.rs`/`gpu.rs`: when `N > 1`, create an `N`-sample color texture sized to the surface and use it as `color_attachments[0].view` with `resolve_target = surface view`; recreate it on surface resize (PNG capture). When `N == 1`, keep the current direct-to-surface attachment.
- [ ] Unit/structural test: pipeline build selects a single `N` shared by all pipelines (no per-pipeline divergence).
- [ ] Verify: `cargo test -p ferrum-wasm`; `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings`; `cargo build -p ferrum-wasm --target wasm32-unknown-unknown`.
- [ ] Rebuild bundle (same wasm-pack command as Task 1).
- [ ] **Browser-validate FA-19** (axis lines continuous, smooth edges; at-rest parity).

### Task 3: Close FA-17 in trackers (docs-only; parallel-safe, no source overlap)
- [ ] `CLAUDE.md`: mark FA-17 resolved (commit `0e82b88`, `ferrum-anywidget.js:499,540` `hasDomainRescale` default-mode).
- [ ] `design-docs/superpowers/followups/2026-05-15-code-archaeology.md`: flip FA-17 status to RESOLVED with the commit ref.

## 6. Acceptance checks

- `cargo test -p ferrum-wasm` — all pass (incl. new per-panel + sample-count tests).
- `cargo build` + `cargo clippy -- -D warnings` for `wasm32-unknown-unknown` — green.
- `cargo test -p ferrum-core` + static SVG goldens — unchanged.
- Browser (human): FA-18 (no sibling shear, overview line persists), FA-19 (no axis seam, smooth edges), at-rest parity.
- FA-17 marked closed in both trackers.

## 7. Open questions

- None blocking. Backend MSAA fallback to 1× is handled by capability query; FA-18 holds regardless of MSAA support.
