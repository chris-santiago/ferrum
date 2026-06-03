# FA-16 Stroke-Width Ribbon Fix — Design Spec

> Status: Option 1 (screen-space stroke width) **chosen 2026-06-03** after a scoping
> spike; supersedes the Option-3 re-layout approach for the FA-16 ribbon. The Option-3
> re-layout design is preserved in §11 as the deferred broader fix (for when correct
> ticks/labels/curve-resampling at the rescaled domain are also required).
>
> Original status: Option-3 re-layout approved 2026-06-02 (now deferred).

## 1. Scope

Line/area mark stroke width is baked into mesh vertices in *scene space* by lyon (the
vertex position already includes the `±normal·half_width` offset), and the WASM vertex
shader then applies the per-panel affine to those positions. Under a **non-uniform**
affine (`sx ≠ sy`, which is exactly what reactive x-domain rescale produces), the baked
width is stretched anisotropically — the "ribbon." This spec makes stroke width
**affine-invariant**: the shader transforms only the stroke *centerline* by the affine,
then applies the width offset in *screen space*, so a stroke renders at its nominal
pixel width regardless of `sx`/`sy`. This fixes the ribbon under reactive rescale and
makes stroke width constant under all interactive transforms (uniform zoom and pan
included), matching standard d3/Vega behavior. The static SVG renderer is untouched.

## 2. Goals

- Strokes render at constant nominal pixel width under any GPU affine (uniform zoom,
  pan, and non-uniform reactive rescale) — no ribbon.
- lyon's full stroke topology (caps, joins, dashes) is preserved unchanged; only the
  *location* of the width offset moves from tessellation-time to the shader.
- Fill meshes, circle/rect instance marks, and the static SVG path are unaffected.
- At rest (identity transform) the rendered output is byte-identical to today.

## 3. Non-goals

- No change to tick/label positions or curve resampling under rescale — those remain
  the affine's approximation (the broader Option-3 re-layout in §11 addresses them when
  needed). This spec fixes stroke width only.
- No change to the static SVG renderer (`ferrum-core` `render/svg.rs`), to circle/rect
  instance pipelines, or to fill-mesh appearance at rest.
- No new user-facing API; internal fidelity fix to existing `.interactive()` behavior.
- No new crate, no Arrow in WASM, no mark-value re-layout payload (those belong to §11).

## 4. System behavior

A stroke (line, polyline, path-stroke, polygon-outline, area-outline) renders at its
nominal pixel width regardless of the active per-panel affine:

- **At rest (identity transform):** byte-identical to today — at `sx=sy=1` the
  screen-space offset equals the previously-baked offset.
- **Uniform zoom / pan (`sx == sy`):** strokes stay constant pixel width (they no longer
  thicken with zoom). This is a deliberate, confirmed behavior change — standard zoom
  behavior — and is what generalizes the ribbon fix to all transforms.
- **Reactive rescale (`sx ≠ sy`):** strokes stay constant pixel width — the ribbon is
  gone. Tick/label spacing under rescale is unchanged from current behavior (still the
  affine approximation; see §11).

Fills (polygon/path fills) are unchanged — they carry no width and are not offset.

## 5. Architecture

The change is contained to the WASM mesh pipeline (`crates/ferrum-wasm/`), three files:

**Tessellation (`src/tessellate.rs`).** The `MeshVertex` struct gains two fields:
`normal: [f32; 2]` and `half_width: f32` (24 → 36 bytes). Every stroke tessellation
closure (currently `BuffersBuilder::new(buffers, |v: StrokeVertex| MeshVertex {
position: v.position(), color })`) changes to capture lyon's un-offset centerline and
offset components instead of the pre-offset position:
- `position` ← `v.position_on_path()` (centerline, **not** `v.position()`)
- `normal` ← `v.normal()`
- `half_width` ← `v.line_width() / 2.0`

Fill closures (`FillVertex`) set `normal: [0.0, 0.0], half_width: 0.0`. Affected stroke
sites: `tessellate_line`, `tessellate_polyline`, `tessellate_path` (stroke branch),
`tessellate_polygon` (stroke branch), and the shared `stroke_path_dashed` builder.

**Shader (`src/shaders/mesh.wgsl`).** `VertexInput` gains `normal` and `half_width`
attributes. `vs_main` transforms the centerline by the affine, then adds the screen-space
width offset only when `half_width > ε`:
`final_px = affine(position) + select(0, normal * half_width, half_width > ε)`.
NDC conversion is unchanged.

**Pipeline (`src/pipelines.rs`).** The mesh vertex-buffer layout updates to the new
stride (36 bytes) and attribute set (position @0, normal, half_width, color).

lyon's `StrokeVertex` (`position_on_path()`, `normal()`, `line_width()`) is stable public
API in the pinned lyon 1.x and exposes exactly these components, so lyon's caps/joins/
dashes geometry is retained — only the width offset is deferred to the shader.

## 6. Canonical interfaces / data contracts

**`MeshVertex` (the one mesh vertex format, shared by strokes and fills):**

```rust
#[repr(C)]
struct MeshVertex {
    position:   [f32; 2], // stroke centerline (scene space); fill position
    normal:     [f32; 2], // unit-ish offset direction from lyon; [0,0] for fills
    half_width: f32,      // half stroke width in px; 0.0 for fills
    color:      [f32; 4],
}
```

**Shader offset contract:** `half_width == 0.0` ⇒ no offset (fills + any zero-width
geometry render exactly as the transformed position). `half_width > 0.0` ⇒ offset by
`normal * half_width` in screen space, *after* the affine, so width is invariant to
`sx`/`sy`. Strokes and fills therefore share one pipeline and one shader, branchless
except for the `select` on `half_width`.

## 7. Invariants and constraints

- **At-rest byte-stability:** with the identity transform, WASM output is unchanged
  (the screen-space offset at `sx=sy=1` equals the old baked offset).
- **Static SVG untouched:** `ferrum-core` render path produces no `MeshVertex`; goldens
  and `cargo test -p ferrum-core` are unaffected.
- **Instance marks untouched:** circle/rect pipelines and their per-instance
  `stroke_width` are not part of the mesh format change.
- **Topology preserved:** caps, joins, and dash patterns render identically (dash
  flattening happens before stroke tessellation; the offset move is post-topology).
- **WASM gates:** `cargo build -p ferrum-wasm --target wasm32-unknown-unknown` succeeds;
  `cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` clean.

## 8. Key decisions and tradeoffs

- **Option 1 (screen-space width) over Option 3 (re-layout).** The browser-confirmed
  evidence is that the defect is *only* the non-uniform stretch of baked width; ticks/
  labels under the affine were not reported as a problem. Option 1 fixes exactly that,
  in 3 WASM files, with no new crate/payload/round-trip. Option 3 additionally fixes
  tick/label spacing and curve resampling but is a much larger build (kept in §11 for
  when that fidelity is required).
- **Capture lyon's centerline + normal, don't replace lyon.** lyon's `StrokeVertex`
  exposes `position_on_path`/`normal`/`line_width`, so caps/joins/dashes stay lyon's;
  only the offset application moves. This is the difference between MODERATE and a LARGE
  rewrite.
- **Constant width under uniform zoom too (confirmed).** Screen-space width applies to
  all transforms, so uniform zoom no longer thickens strokes. This is the standard
  behavior and is what makes the fix general; the prior thickening was an artifact of
  baking width in data space.
- **One shared mesh pipeline, branch on `half_width`.** Fills carry `half_width = 0` and
  skip the offset, avoiding a second pipeline.

## 9. Acceptance criteria

- In the focus+context interactive demo, after a brush rescale the detail lines render
  at constant width (no ribbon) with dashes/joins intact.
- Uniform zoom and pan render strokes at constant width (no thickening), no ribbon.
- At-rest interactive render and all static SVG goldens are byte-identical to before.
- `cargo test -p ferrum-wasm` passes (incl. new tessellation tests); `cargo build`/
  `clippy` for `wasm32-unknown-unknown` are green; the shipped WASM bundle is rebuilt.

## 10. Validation strategy

- **Tessellation unit tests** (`src/tessellate.rs`): a stroked segment yields vertices
  whose `position` is the centerline and whose `normal * half_width` reproduces the old
  baked offset at identity; fills yield `half_width == 0`.
- **WASM build + clippy gates** for `wasm32-unknown-unknown`.
- **Static regression:** `cargo test -p ferrum-core` + golden SVGs unchanged (this fix
  does not touch that path).
- **Browser validation:** export the focus+context demo, confirm ribbon gone, constant
  width under both uniform zoom and non-uniform rescale, dashes/joins correct. (CI-
  unverifiable; human browser inspection, consistent with prior interactive validation.)

## 11. Deferred alternative — Option 3: re-layout in WASM (broader fix)

Retained for when correct **tick/label spacing and curve resampling at the rescaled
domain** are required (Option 1 leaves those as the affine approximation). A scoping
spike (2026-06-02) found this feasible but a larger build:

- Extract a pure-Rust **`ferrum-layout`** crate (the `compute_layout` engine + the 14
  scale kernels) alongside `ferrum-scene`, depended on by both `ferrum-core` (PyO3
  bindings/validators stay there) and `ferrum-wasm`. No `pyo3`/`pyo3-arrow`/`arrow`/
  `rand` in the shared crate (the only `wasm32` blocker is `getrandom/js` via `rand`,
  which is transform-only and stays in core).
- Ship a **packed-binary mark-value payload** (layout-relevant columns only, dtype-
  tagged, string side-pool) in the existing packed-bytes channel, keyed by
  `(panel, batch)`, decoded-once-and-cached in WASM.
- On a `domain`-role rescale (only that trigger; zoom/pan keep the affine), recompute
  the panel layout at the new domain and rebuild instances / re-tessellate meshes
  (identity transform) / re-render ticks+labels; coalesce to one run per animation frame.
- **Open question for Option 3:** whether the WASM runtime can render *arbitrary new*
  tick-label strings (vs. a pre-baked glyph atlas) — may need an HTML/SVG overlay or a
  dynamic glyph path. Resolve before designing the axis-text portion.

Option 3 supersedes the affine entirely for domain rescale (and would moot Option 1 for
that case), at the cost of a new crate, an export-payload format, and a WASM↔layout
round-trip. Prefer Option 1 until tick/label/resampling fidelity at the rescaled domain
is actually needed.
