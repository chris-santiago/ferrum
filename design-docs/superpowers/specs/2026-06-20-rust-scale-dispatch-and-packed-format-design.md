# Rust Scale-Dispatch Dedup + Packed-Format Enforcement Design Spec

> Source: `/rust-review` of the archaeology #6/#7/#8 effort's Rust surface (2026-06-20). Findings R1 (S3) and R2 (S3/S4). Behavior-preserving. R3 (extent-pin match arms) deliberately left as-is.

## 1. Scope

Two independent, behavior-preserving cohesion fixes in the Rust layer:
- **R1** — collapse the auxiliary-scale (color/size/shape/opacity) build block, currently triplicated across three dispatch branches in `crates/ferrum-core/src/render/scale_resolve/mod.rs`, into one private helper.
- **R2** — convert the packed GPU-instance wire-format from discipline-enforced correctness to compiler/test-enforced correctness: named stride/offset consts in the ferrum-core producer with a test pinning the emitted stride, and compile-time size assertions in the ferrum-wasm consumer.

Both preserve behavior; R1 is byte-identical scene output, R2 adds no runtime behavior (named consts + static asserts + a test).

## 2. Goals

- One definition of "build the four auxiliary scales" (`force_cat` + `aux_shared` + the four builders + warning collection); the three dispatch sites call it.
- The packed circle/rect byte stride (64 / 72) and X/Y field offsets (0 / 4) have one named home per crate, and any future drift in the layout fails a compile-time assertion (consumer) or a unit test (producer) rather than silently corrupting the interactive render.
- No public scale-resolution API change; no behavior change; full cargo + golden suites stay green.

## 3. Non-goals

- **R3 not addressed:** `fix_transform_extents_for_facet`'s one-arm-per-transform match stays as-is. Each arm rebuilds a distinct concrete `TransformSpec` variant; a unifying `ExtentCarrying` trait would be speculative abstraction for a clear, well-tested match.
- No change to the packed binary *format itself* (strides, offsets, flag bits, tooltip-table layout) — only its enforcement.
- No change to the Python `_PACKED_INSTANCE_SIZES` mirror (already a documented named const; reviewed under the 2026-06-20 python-review).
- No change to the ferrum-wasm stride *computation* (already `size_of`-derived and correct) — only an added static assertion that the size equals the canonical value.

## 4. System behavior

Unchanged. R1: every chart's `ResolvedScales` is identical (same builder calls, same order, same warning order) → byte-identical SVG/packed output. R2: no runtime path changes; the producer emits the same bytes, the consumer reads them identically; the new const/assert/test only make the existing contract explicit and self-checking.

## 5. Architecture

**R1** separates *dispatch* from *auxiliary-scale construction*. The three branches of `resolve_scales_with_outputs` (Tick/Rule x-only, Tick/Rule y-only, main path) differ only in their positional-axis handling and final `ResolvedScales` assembly; the auxiliary-scale construction is identical and moves to one helper.

**R2** makes the wire-format contract enforced at its two Rust ends: the producer (`pack_instances.rs`) owns named stride/offset consts and a test asserting its emitted bytes-per-instance equals the stride; the consumer (`scene_load.rs`) keeps its `size_of`-derived stride but gains a compile-time assertion that the struct size equals the canonical 64/72 (and, if `core::mem::offset_of!` is available on the toolchain, that `center`/`position` sit at byte 0). The canonical values are documented in each crate citing the other.

## 6. Canonical interfaces / data contracts

**R1 helper** (private to `scale_resolve`):
```rust
fn build_auxiliary_scales(
    spec: &ChartSpec,
    primary_batch: &RecordBatch,
    transform_outputs: &HashMap<String, RecordBatch>,
    theme: &ThemeInputs,
    warnings: &mut Vec<crate::render::RenderWarning>,
) -> Result<
    (Option<ColorScale>, Option<SizeScale>, Option<ShapeScale>, Option<OpacityScale>),
    RenderError,
>
```
Body reproduces the existing block VERBATIM in order: `force_cat = matches!(spec.mark, Mark::Area)`; `aux_shared = facet_aux_shared(spec)`; `build_color_scale(&spec.encoding, primary_batch, transform_outputs, theme, force_cat, aux_shared)?` then `warnings.extend(color_warns)`; `build_size_scale(&spec.encoding, primary_batch, transform_outputs, aux_shared, theme)?`; `build_shape_scale(&spec.encoding, primary_batch, transform_outputs, aux_shared)?` then `if let Some(w) = shape_warn { warnings.push(w) }`; `build_opacity_scale(&spec.encoding, primary_batch, transform_outputs, aux_shared, theme)?`; return `(color, size, shape, opacity)`.

Each of the three sites replaces its inline block with:
```rust
let (color, size, shape, opacity) =
    build_auxiliary_scales(spec, primary_batch, transform_outputs, theme, &mut warnings)?;
```
and keeps its own positional-axis resolution + `ResolvedScales { … }` assembly.

**R2 producer consts** (`pack_instances.rs`, module-level, replacing the two function-local `FLOATS_PER_*`):
```rust
pub const CIRCLE_FLOATS: usize = 16;
pub const RECT_FLOATS: usize = 18;
pub const CIRCLE_STRIDE: usize = CIRCLE_FLOATS * 4; // 64 bytes
pub const RECT_STRIDE: usize = RECT_FLOATS * 4;     // 72 bytes
pub const FIELD_X_OFFSET: usize = 0; // X is f32[0]
pub const FIELD_Y_OFFSET: usize = 4; // Y is f32[1]
```
`pack_circle_batch` / `pack_rect_batch` use `*_FLOATS` for the capacity hint; a new unit test asserts `pack_circle_batch(&[one_circle]).len() == CIRCLE_STRIDE` and the rect analog.

**R2 consumer assertion** (`scene_load.rs`, compile-time):
```rust
const _: () = assert!(std::mem::size_of::<CircleInstance>() == 64);
const _: () = assert!(std::mem::size_of::<RectInstance>() == 72);
```
(Optionally, if `core::mem::offset_of!` compiles on the project toolchain: assert `center`/`position` at offset 0. If it does not compile, omit the offset asserts — do not add a dependency or bump the toolchain.)

## 7. Invariants and constraints

- **R1 byte-identical:** identical builder calls, identical order, identical warning push order. The golden suite is the oracle.
- **R1 no public API change:** `resolve_scales` / `resolve_scales_with_outputs` signatures unchanged; the helper is `fn`-private to the module.
- **R2 no format change:** the emitted/consumed bytes are identical; consts equal the current literals (64/72, 0/4). The producer test and consumer static-assert must pass against the *current* layout (they encode, not change, it).
- **R2 must not bump the Rust edition/toolchain** or add a dependency. `offset_of!` only if it already compiles.
- No matplotlib; no global mutable state; `cargo test` must pass.

## 8. Key decisions and tradeoffs

- **R1 helper computes `force_cat`/`aux_shared` internally** (from `spec`) rather than taking them as params — one source of truth for how auxiliary scales are derived; the three sites already compute them identically.
- **R1 returns a 4-tuple, not a struct** — the values flow straight into three distinct `ResolvedScales` literals; a wrapper struct would be ceremony. (`ResolvedScales` itself remains the typed home.)
- **R2 keeps the consumer `size_of`-derived** (already correct) and only *asserts* the size — the assertion is the enforcement, not a re-derivation. The producer gets named consts because its stride is currently implicit in the push sequence.
- **R2 does not introduce a shared crate for the format** — the format crosses ferrum-core, ferrum-wasm, and Python; a shared Rust const can't reach Python, and the two Rust crates have legitimately different representations (byte-pusher vs `#[repr(C)]` struct). Per-end enforcement + cross-citing docs is the proportionate fix.

## 9. Acceptance criteria

- R1: `build_auxiliary_scales` defined once; the three former blocks each replaced by one call; `cargo test -p ferrum-core --lib` green; full golden suite byte-identical.
- R2: `CIRCLE_STRIDE`/`RECT_STRIDE`/`FIELD_X_OFFSET`/`FIELD_Y_OFFSET` (and `*_FLOATS`) named in `pack_instances.rs`; producer-stride test passes; `const _: () = assert!(size_of::<…>() == 64/72)` compiles in `scene_load.rs`; `cargo test -p ferrum-core` and `cargo test -p ferrum-wasm` green; `cargo clippy` adds no NEW lints.
- Full pytest suite green (golden byte-diffs) — proves R1 byte-identity end to end.

## 10. Validation strategy

R1 equivalence is proven by byte-identical goldens + the `scale_resolve` unit suite (`tests.rs`); explicit edge cases: Tick/Rule single-axis x-only and y-only branches, Area `force_cat`, faceted `aux_shared`. R2 is proven by the producer-stride unit test (fails if the push sequence drifts from the stride const) and the consumer compile-time assert (fails the build if the struct layout drifts). Both R2 guards must be confirmed to encode the *current* values (a deliberately wrong const must fail) so they are real discriminators, not tautologies.

## 11. Open questions

- `core::mem::offset_of!` availability on the project toolchain — if it does not compile, omit the offset assertions (size asserts alone still catch the most likely drift). The implementer resolves this by trying it; not a blocker.
