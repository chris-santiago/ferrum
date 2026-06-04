# WASM Interactive-Renderer Correctness Pass — Design Spec

> Status: **APPROVED 2026-06-03.** Covers FA-18 (per-panel mark transform) and
> FA-19 (mesh MSAA). FA-17 (brush-mode default) was already fixed in commit
> `0e82b88` and needs only a tracker update, not code. Root causes confirmed by a
> 3-agent interactive audit on 2026-06-03.

## 1. Scope

Two independent correctness defects in the WASM interactive renderer
(`crates/ferrum-wasm/`), both rooted in the shared main render pass:

- **FA-18 — multi-panel transform corruption.** The renderer holds a single
  mark-transform uniform that is uploaded once per frame for one panel and bound
  once before the per-panel draw loops. During a reactive domain-rescale (and any
  multi-panel state where two panels are simultaneously non-identity), every
  panel's marks are drawn with the wrong panel's affine — sibling line/area meshes
  shear and the overview's own line is translated out of its scissor and vanishes.
- **FA-19 — interactive axis-line seam.** The mesh pipeline renders with no MSAA
  (`sample_count = 1`); abutting butt-cap quads (axis line vs. tick marks, or
  adjacent facet axis lines) leave a 1px hairline gap/step at their shared edge.
  Interactive-only; the static SVG path antialiases each line as one primitive.

Both fixes are WASM-only. The static SVG renderer (`ferrum-core`) is untouched.

## 2. Goals

- Each mark draw (mesh and instance) is transformed by **its own panel's** affine,
  regardless of how many panels are simultaneously non-identity.
- Reactive rescale of one panel leaves sibling panels fixed and the overview line
  visible throughout the brush drag.
- Stroked mesh edges (axis lines, gridlines, data lines/areas) are antialiased; no
  hairline seam where collinear/abutting quads meet.
- At-rest interactive render and all static SVG goldens are visually unchanged.
- `cargo test -p ferrum-wasm`, `clippy --target wasm32 -- -D warnings`, and the
  wasm32 build stay green; the shipped bundle is rebuilt.

## 3. Non-goals

- No change to the static SVG renderer, to scene-graph construction, or to
  tessellation geometry (the axis line is already one node / one stroke).
- No change to tick/label positions or curve resampling under rescale (still the
  affine approximation per the deferred Option-3 re-layout).
- No new user-facing API.
- FA-17 requires no code; only marking it closed in the trackers.

## 4. System behavior

**FA-18.** When panel P is rescaled (or zoomed/panned), only P's marks move; every
other panel Q renders with Q's own affine (identity unless Q was independently
transformed). In a focus+context chart, brushing the overview rescales the detail
panel in-bounds while the overview's own line stays drawn at its own coordinates.
Single-panel charts behave exactly as today. Static elements (axes, grid, legend,
title, annotations) remain on the identity transform and stay fixed.

**FA-19.** Axis lines, gridlines, and data line/area strokes render with smooth
antialiased edges; the axis domain line reads as one continuous line with no break
or step where tick-mark quads cross it or where adjacent facet panels meet. At rest
the only visible difference from today is the absence of the seam and smoother
stroke edges.

## 5. Architecture

All draws share one render pass in `render_frame` (`render.rs`), with two existing
transform uniforms: a mark uniform (zoom/pan/rescale affine) and an identity
uniform (static + annotation). Both defects live at this pass level.

**FA-18 — per-panel mark transform.** The single mark uniform is generalized to
**one transform slot per panel**, indexed by panel id. Each mark draw unit binds
the slot for the panel it belongs to:

- The mesh draw loop (`render.rs` mark-mesh section) already iterates
  `mark_mesh_panels`; each panel draw binds that panel's transform slot instead of
  the one shared mark bind group.
- The per-batch instance loop binds the transform slot of the command's panel for
  `is_mark` commands; non-mark commands keep the identity uniform.

This requires each mark draw unit to carry a **panel index**. `MarkMeshPanel` and
mark-kind `DrawCommand` gain a `panel_id` set at scene-load time alongside the
existing `plot_area`. The render reads the full `zoom_transforms: &[Affine2]`
(already threaded through `upload_transform_and_render`) and uploads/binds every
panel's affine each frame, so no "current panel" assumption remains. The
mechanism for N transform slots (a dynamic-offset uniform buffer with one
alignment-padded `Uniforms` per panel, vs. a `Vec` of per-panel buffers+bind
groups) is an implementation choice; the contract is only that each mark draw is
bound to its panel's affine. The identity uniform is unchanged.

Upload paths (`apply_reactive_rescale`, `on_wheel`, `on_pan`, `set_absolute`)
continue to mutate `zoom.transforms[panel]`; the render consumes the whole vector
rather than a single uploaded buffer.

**FA-19 — MSAA on the main pass.** The main render pass renders into a
multisampled color texture (sample_count = 4) that resolves to the surface view.
Because all pipelines drawn in the pass must share one sample count, **every**
pipeline built for this pass (mesh, textured, instanced rect/circle and their
additive variants) is created with the same `multisample.count`. The multisampled
texture is sized to the surface and recreated when the surface is resized (the PNG
capture path). If the active backend reports 4× unsupported, the renderer falls
back to sample_count = 1 (current behavior) rather than failing — no panic, no
NotImplementedError.

## 6. Canonical interfaces / data contracts

**Panel association on mark draw units** (`scene_load.rs`):

```rust
struct MarkMeshPanel { panel_id: usize, index_start: u32, index_count: u32, plot_area: [f32; 4] }
// DrawCommand gains: panel_id: usize  (meaningful when is_mark == true)
```

**Per-panel transform contract:** for every mark draw, the bound transform is
`zoom_transforms[panel_id]` (falling back to identity if out of range). Non-mark
and annotation/static draws bind the identity uniform. The `Uniforms` byte layout
(canvas vec4 + transform vec4, 32 bytes) is unchanged per slot.

**MSAA contract:** one sample count `N ∈ {1, 4}` shared by the pass color
attachment and all pipelines in it. `N` is chosen once at pipeline-build time from
backend capability; the pass color attachment uses an `N`-sample texture with
`resolve_target = surface view` when `N > 1`, else the surface view directly with
`resolve_target = None`.

## 7. Invariants and constraints

- **At-rest / single-panel byte-stability (transform):** with all panels at
  identity, per-panel binding produces the same pixels as the single-uniform path.
- **Static SVG untouched:** `ferrum-core` render path and goldens unaffected.
- **Identity uniform unchanged:** axes/grid/legend/title/annotations stay fixed
  under zoom/pan/rescale exactly as today.
- **Uniform sample count:** the pass and all its pipelines share one sample count;
  a mismatch is a wgpu validation error and must not be possible by construction.
- **No hard dependency on 4× MSAA:** unsupported-backend fallback to 1× is silent
  and safe (not a crash, not an error surfaced to the user).
- **WASM gates:** `cargo build`/`clippy` for `wasm32-unknown-unknown` clean
  (`-D warnings`); `cargo test -p ferrum-wasm` passes.

## 8. Key decisions and tradeoffs

- **Robust per-panel uniforms over a minimal in-loop rebind.** Generalizing to one
  transform slot per panel removes the "one transform per frame" assumption
  entirely, fixing FA-18 *and* the latent corruption where two independently
  transformed panels render with each other's affine. The minimal alternative
  (rebind the single buffer inside the loop) fixes the reported symptom but leaves
  the fragile architecture and per-draw upload churn. Chosen: robust.
- **MSAA over per-vertex pixel snapping.** Enabling MSAA is the root-cause fix at
  the rasterization layer and improves every stroked mesh element; vertex snapping
  is a narrow patch that only helps axis/tick quads and not data marks. Chosen:
  MSAA, accepting a render-pass-wide change to all pipelines.
- **FA-19 is not FA-16's fault.** The seam predates the FA-16 bevel/screen-space
  change (confirmed byte-identical line handling pre-FA-16); FA-16 can only shift
  which pixels show it. No FA-16 rollback.
- **FA-17 needs no code.** The `hasDomainRescale` default-mode logic
  (`ferrum-anywidget.js:499,540`) already ships on main; the trackers are stale.

## 9. Acceptance criteria

- Focus+context interactive demo: brushing the overview rescales the detail panel
  while sibling panels stay fixed and the overview line stays visible throughout
  the drag (FA-18 resolved).
- Axis lines render continuous with no seam/step where ticks cross or facets meet;
  stroke edges are antialiased (FA-19 resolved).
- At-rest interactive render visually identical to before (modulo AA smoothing);
  all static SVG goldens byte-identical.
- `cargo test -p ferrum-wasm` passes; `cargo build`/`clippy` for
  `wasm32-unknown-unknown` green; shipped WASM bundle rebuilt.
- FA-17 marked closed in `CLAUDE.md` known-gaps and the archaeology doc.

## 10. Validation strategy

- **Unit (`ferrum-wasm`):** a multi-panel scene records distinct `panel_id`s on its
  `MarkMeshPanel`/`DrawCommand` entries; given a `zoom_transforms` vector with one
  rescaled entry, the render path selects per-panel affines (structural assertion,
  no GPU). Pipeline-build picks one shared sample count for all pipelines.
- **WASM build + clippy gates** for `wasm32-unknown-unknown`.
- **Static regression:** `cargo test -p ferrum-core` + golden SVGs unchanged.
- **Browser validation (human, CI-unverifiable):** export the focus+context demo;
  confirm FA-18 (no sibling shear, overview line persists) and FA-19 (no axis seam,
  smooth edges); confirm at-rest parity. Each fix validated as its own change.

## 11. Open questions

- None blocking. Backend MSAA support is handled by the capability-query fallback;
  the only browser-observable risk is that some GL contexts cap at 1×, in which
  case FA-19 degrades to current behavior (no regression) while FA-18 still holds.
