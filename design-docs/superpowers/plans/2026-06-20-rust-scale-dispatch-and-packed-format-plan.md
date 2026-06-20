# Rust Scale-Dispatch Dedup + Packed-Format Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development to implement this plan task-by-task.

## 1. Objective

R1: collapse the triplicated auxiliary-scale build block in `scale_resolve/mod.rs` into one `build_auxiliary_scales` helper (byte-identical). R2: give the packed wire-format named stride/offset consts + a producer-stride test (ferrum-core) and a compile-time size assertion (ferrum-wasm). Both behavior-preserving.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-20-rust-scale-dispatch-and-packed-format-design.md` §6 (interfaces), §7 (invariants), §9 (acceptance), §11 (offset_of caveat)

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/scale_resolve/mod.rs` | R1: add `build_auxiliary_scales`; replace the 3 inline blocks with one call each |
| Modify | `crates/ferrum-core/src/render/pack_instances.rs` | R2: module-level stride/offset consts; capacity hints use them; producer-stride test |
| Modify | `crates/ferrum-wasm/src/scene_load.rs` | R2: compile-time `size_of` assertions for CircleInstance/RectInstance |

## 4. Constraints

- **R1 byte-identical:** identical builder calls, identical order, identical warning push order (`warnings.extend(color_warns)` then later `warnings.push(shape_warn)`). The golden suite is the binding oracle; any byte-diff is a regression.
- **R1 no public API change:** `resolve_scales` / `resolve_scales_with_outputs` signatures unchanged; `build_auxiliary_scales` is `fn`-private to the module; it computes `force_cat = matches!(spec.mark, Mark::Area)` and `aux_shared = facet_aux_shared(spec)` internally.
- **R1 three call sites:** Tick/Rule x-only branch, Tick/Rule y-only branch, and the main path each keep their own positional-axis resolution + `ResolvedScales { … }` assembly; only the auxiliary block is replaced by `let (color, size, shape, opacity) = build_auxiliary_scales(spec, primary_batch, transform_outputs, theme, &mut warnings)?;`.
- **R2 no format change:** emitted/consumed bytes identical; consts equal the current literals — `CIRCLE_STRIDE = 64`, `RECT_STRIDE = 72`, `FIELD_X_OFFSET = 0`, `FIELD_Y_OFFSET = 4`, `CIRCLE_FLOATS = 16`, `RECT_FLOATS = 18`.
- **R2 consumer stays `size_of`-derived:** do NOT replace the `std::mem::size_of::<…>()` stride computation in `scene_load.rs`; only ADD `const _: () = assert!(std::mem::size_of::<CircleInstance>() == 64);` and the rect analog.
- **R2 `offset_of!` only if it compiles** on the project toolchain; if not, omit offset asserts (size asserts suffice). Do NOT bump edition/toolchain or add a dependency.
- **R2 guards must be real discriminators:** confirm a deliberately-wrong const/assert fails (the producer test fails if stride ≠ emitted length; the static assert fails the build if size ≠ 64/72), then set them to the correct values.
- No matplotlib; no global mutable state; `cargo test` must pass.
- Build: `unset CONDA_PREFIX && uv run --no-sync maturin develop` (R1 changes ferrum-core → rebuild before pytest). Source `~/.cargo/env` for cargo.
- Rust tests: `DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test -p ferrum-core` (DYLD required); `cargo test -p ferrum-wasm` (no DYLD). Pytest gets **NO** DYLD prefix.

## 5. Tasks

### Task 1 (R1): extract `build_auxiliary_scales` — `scale_resolve/mod.rs`
- [ ] Add the private `build_auxiliary_scales` helper per spec §6 (reproduce the existing block verbatim, in order; compute `force_cat`/`aux_shared` internally; collect `color_warns` via `warnings.extend`, `shape_warn` via `if let Some(w) = … { warnings.push(w) }`).
- [ ] Replace the three inline auxiliary blocks (Tick/Rule x-only ~870-878, Tick/Rule y-only ~889-898, main path ~959-971) with one `build_auxiliary_scales(...)` call each; keep each site's positional handling + `ResolvedScales` assembly.
- [ ] Verify: `source ~/.cargo/env && DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test -p ferrum-core --lib`
- [ ] Verify: `cargo clippy -p ferrum-core -- -D warnings` (no NEW lints vs the ~181 pre-existing)

### Task 2 (R2): packed-format enforcement — `pack_instances.rs` + `scene_load.rs`
- [ ] `pack_instances.rs`: add module-level consts (`CIRCLE_FLOATS`, `RECT_FLOATS`, `CIRCLE_STRIDE`, `RECT_STRIDE`, `FIELD_X_OFFSET`, `FIELD_Y_OFFSET`) per spec §6; replace the function-local `FLOATS_PER_*` with `CIRCLE_FLOATS`/`RECT_FLOATS` in the capacity hints; add a unit test asserting `pack_circle_batch(&[one_circle]).len() == CIRCLE_STRIDE` and `pack_rect_batch(&[one_rect]).len() == RECT_STRIDE`.
- [ ] `scene_load.rs`: add `const _: () = assert!(std::mem::size_of::<CircleInstance>() == 64);` and `… RectInstance … == 72`; (optionally `offset_of!` X/Y offset asserts only if they compile).
- [ ] Verify: `source ~/.cargo/env && DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/lib cargo test -p ferrum-core` AND `cargo test -p ferrum-wasm`
- [ ] Verify: `cargo clippy -p ferrum-core -- -D warnings` and `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` (no NEW lints)

## 6. Acceptance checks

- `cargo test -p ferrum-core` + `cargo test -p ferrum-wasm` green (orchestrator runs both).
- Full pytest suite green (orchestrator) — golden byte-diffs prove R1 byte-identity end to end.
- `grep -n "build_auxiliary_scales" scale_resolve/mod.rs` shows one definition + three calls; no inline `build_color_scale`/`build_size_scale`/`build_shape_scale`/`build_opacity_scale` sequence remains at the three sites.
- `grep -n "CIRCLE_STRIDE\|RECT_STRIDE" pack_instances.rs` shows the named consts in use; `grep -n "size_of::<CircleInstance>() == 64" scene_load.rs` shows the static assert.

## 7. Open questions

- `offset_of!` toolchain availability (spec §11) — implementer tries it; omit offset asserts if it doesn't compile. Not a blocker.
