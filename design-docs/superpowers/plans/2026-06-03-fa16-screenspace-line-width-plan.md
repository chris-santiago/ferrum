# FA-16 Screen-Space Stroke Width Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: chris-code:subagent-driven-development. All files are `.rs`/`.wgsl` in `crates/ferrum-wasm/` → `rust-coder`. The three files are one tightly-coupled vertex-format change → a single task, not parallel stages.

## 1. Objective

Make WASM line/area stroke width affine-invariant by capturing lyon's stroke centerline + normal + half-width and applying the width offset in screen space in the mesh shader, eliminating the FA-16 ribbon under non-uniform reactive rescale.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-02-wasm-relayout-rescale-design.md` §1–§10 (Option 1; §11 is the deferred Option-3 alternative — out of scope here).

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-wasm/src/tessellate.rs` | extend `MeshVertex`; capture centerline+normal+half_width in stroke closures; zero-fill fills |
| Modify | `crates/ferrum-wasm/src/shaders/mesh.wgsl` | add normal/half_width inputs; screen-space offset after affine |
| Modify | `crates/ferrum-wasm/src/pipelines.rs` | update mesh vertex-buffer layout (stride + attributes) |
| Modify | `src/ferrum/_wasm/ferrum_wasm_bg.wasm` (+ glue) | rebuilt bundle (wasm-pack output) |

## 4. Constraints

- **At-rest byte-stability:** at identity transform (`sx=sy=1`) the screen-space offset must equal the old baked offset — initial interactive render unchanged.
- **Static SVG path untouched** (`ferrum-core`); circle/rect instance pipelines untouched; fill-mesh appearance unchanged.
- **Preserve lyon topology:** use `StrokeVertex::position_on_path()` / `normal()` / `line_width()` — do NOT replace lyon's stroke tessellation; caps/joins/dashes must render identically.
- **One shared mesh pipeline:** fills carry `normal=[0,0]`, `half_width=0`; shader skips offset when `half_width` is ~0 (single `select`, no second pipeline).
- **Constant width under all transforms** is intended (uniform zoom no longer thickens) — do not special-case uniform vs non-uniform.
- `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` must be clean (this crate gates on `-D warnings`).

## 5. Tasks

### Task 1: Affine-invariant stroke width (single coordinated change)
- [ ] Extend `MeshVertex` (tessellate.rs): add `normal: [f32; 2]`, `half_width: f32`; keep `#[repr(C)]` + bytemuck derives. Add a `size_of` assertion (36 bytes).
- [ ] Update every STROKE closure (`tessellate_line`, `tessellate_polyline`, `tessellate_path` stroke branch, `tessellate_polygon` stroke branch, and the shared `stroke_path_dashed` builder) to set `position = v.position_on_path()`, `normal = v.normal()`, `half_width = v.line_width()/2.0`.
- [ ] Update every FILL closure (`FillVertex` sites) to set `normal: [0.0,0.0], half_width: 0.0`.
- [ ] `mesh.wgsl`: add `@location` inputs for normal + half_width; in `vs_main` transform the centerline by the affine, then `+ select(vec2(0.0), normal*half_width, half_width > 1e-4)` before NDC.
- [ ] `pipelines.rs`: update the mesh `VertexBufferLayout` stride (36) and attributes (position @0 Float32x2, normal Float32x2, half_width Float32x1, color Float32x4) with correct offsets/shader_locations matching the shader.
- [ ] Add tessellation unit tests: a stroked horizontal segment → vertices whose `position` is the centerline and `normal*half_width` reproduces the old offset at identity; a fill → `half_width == 0`.
- [ ] Verify: `cargo test -p ferrum-wasm`; `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings`; `cargo build -p ferrum-wasm --target wasm32-unknown-unknown`.

### Task 2: Rebuild shipped bundle
- [ ] `source ~/.cargo/env && wasm-pack build crates/ferrum-wasm --target web --release --out-dir ../../src/ferrum/_wasm/`.
- [ ] Verify: bundle rebuilt; a quick `.interactive().save()` export still loads.

## 6. Acceptance checks

- `cargo test -p ferrum-wasm` — all pass (incl. new tessellation tests).
- `cargo build` + `cargo clippy -- -D warnings` for `wasm32-unknown-unknown` — green.
- `cargo test -p ferrum-core` + static SVG goldens — unchanged (this fix doesn't touch that path).
- Browser (human): focus+context demo — ribbon gone; constant stroke width under both uniform zoom and non-uniform rescale; dashes/joins intact; at-rest render visually identical.

## 7. Open questions

- None blocking. (Tick/label spacing and curve resampling under rescale remain the affine approximation by design — the deferred Option-3 re-layout in spec §11 addresses them if/when needed.)
