# Re-layout-in-WASM for Reactive Domain-Rescale Design Spec

> Status: approved 2026-06-02. Fixes FA-16 (line/area stroke-width "ribbon" and
> tick/label distortion under reactive x-domain rescale) at its root.

## 1. Scope

Reactive focus+context rescale (a brush selection that re-domains another panel's
axis) is currently faked in the WASM runtime by writing a non-uniform affine
(`sx` large, `sy = 1`) into the GPU uniforms. Because mark stroke width is baked
into mesh vertices in scene-space, the non-uniform affine stretches that width into
"ribbons," and it only approximates tick positions and label spacing. This spec
defines replacing that affine approximation, **for the reactive-rescale case only**,
with a true re-computation of the affected panel's layout and scales in the browser:
the same Rust layout/scale kernels that produce the static scene run again at the new
domain, and the panel's marks, axes, and labels are rebuilt from the recomputed
geometry. Wheel-zoom and pan are unchanged.

## 2. Goals

- A reactive domain-rescale renders the target panel identically to a statically
  rendered chart at that domain: constant-width strokes, correct tick positions,
  correct label spacing, correctly resampled curves, correct clip.
- The layout/scale kernels have a single source of truth shared by the native
  (PyO3) and browser (WASM) builds — no duplicated or re-implemented layout math.
- Re-layout runs in the browser with no Python round-trip and no network call.
- The native render path is byte-stable across the kernel extraction.
- Wheel-zoom and pan retain their current cheap GPU-affine behavior and performance.

## 3. Non-goals

- No change to wheel-zoom or pan (they keep the uniform GPU affine).
- No Arrow / `pyo3-arrow` dependency in the WASM build.
- Transforms, statistical computation, and RNG stay in `ferrum-core`; they are not
  moved into the shared crate and are not run in the browser.
- The static SVG render path is untouched.
- No new user-facing API. This is an internal fidelity fix to existing
  `.interactive()` behavior.

## 4. System behavior

**Wheel-zoom / pan (unchanged):** the runtime applies a uniform affine to the GPU
uniforms. `sx == sy`, so strokes keep constant apparent width and the result is
visually correct. No re-layout occurs.

**Reactive domain-rescale (new):** when a bound selection changes a panel's domain
(focus+context brushing, or a `domain`-role param binding), the runtime recomputes
that panel's layout and scales at the new domain and rebuilds the panel's rendered
content from the recomputed geometry rather than transforming the existing geometry:

- Mark positions are recomputed at the new domain. Instanced marks (circles, rects)
  get rebuilt instance buffers; mesh marks (lines, areas, paths) are re-tessellated
  at the new positions and drawn with an **identity** transform, so stroke width is
  correct by construction (no affine stretch → no ribbon).
- Axis ticks and labels are recomputed at the new domain (new tick values, new
  positions, correct spacing) and re-rendered.
- The panel clip uses the recomputed plot area.

Re-layout is **coalesced to at most one run per animation frame**: rapid brush
updates within a frame collapse to a single re-layout, and the most recent domain
wins. The first rescale decodes the mark-value payload once; subsequent rescales
reuse the cached decoded columns.

## 5. Architecture

**New crate `ferrum-layout` (pure Rust).** A workspace member alongside
`ferrum-scene`, holding the layout engine (`compute_layout` and its sub-modules)
and the 14 scale kernels. It depends on `ferrum-scene` and on no native-only crate
(no `pyo3`, no `pyo3-arrow`, no `arrow`, no `rand`). It is the single home for
layout and scale math.

**`ferrum-core`** depends on `ferrum-layout` and retains the PyO3 binding layer and
the scale validators (the only parts that touch `pyo3`). Core calls the kernels in
the new crate; its observable behavior is unchanged.

**`ferrum-wasm`** depends on `ferrum-layout` and gains a re-layout path invoked on
reactive domain-rescale. It already compiles to `wasm32-unknown-unknown` and already
links `lyon` for tessellation, so re-tessellation reuses existing machinery.

Layering: `ferrum-scene` (data model) → `ferrum-layout` (geometry + scales) →
`{ferrum-core (native/PyO3), ferrum-wasm (browser/wgpu)}`. This mirrors the existing
`ferrum-scene` split rather than introducing conditional compilation inside
`ferrum-core`.

**Data flow on rescale:** brush selection → new domain for the bound panel →
(decode mark-value columns on first use, else use cache) → `ferrum-layout` recomputes
panel layout + scaled mark positions + ticks/labels → rebuild instance buffers /
re-tessellate meshes (identity transform) / re-render axis text → draw.

## 6. Canonical interfaces / data contracts

**Mark-value payload.** Layout-relevant columns only (the positional and grouping
fields each mark batch consumes — not the full dataframe), carried in the existing
packed-bytes channel, keyed by `(panel_index, batch_index)`. Each column is
dtype-tagged; strings are encoded as `u32` offsets into a per-payload string pool.
The contract is: given a `(panel, batch)` key, the runtime can reconstruct the typed
column vectors needed to re-run that batch's scales and layout. Decoded columns are
cached in WASM memory after first decode; the payload is decoded lazily (only if a
rescale actually occurs).

**Re-layout entry point.** `ferrum-layout` exposes a panel-level layout computation
that takes a scale domain (or domains), panel dimensions / plot area, theme inputs,
a text-metrics provider, and the mark data, and returns laid-out scene geometry
(scaled mark positions, axis ticks, labels, plot area). The native binding and the
WASM runtime call the same entry point; neither re-implements layout.

**Trigger contract.** Only param bindings of role `domain` (reactive rescale)
trigger re-layout. Bindings of role `filter` and `legend`, and the zoom/pan affine,
do not. Wheel/pan continue to write the GPU affine directly.

## 7. Invariants and constraints

- **Native byte-stability:** moving the kernels into `ferrum-layout` must not change
  any native render output. Existing goldens and `cargo test` pass unchanged.
- **No PyO3/Arrow/rand in `ferrum-layout`:** the crate must compile to
  `wasm32-unknown-unknown` on its own. RNG-dependent transforms are excluded by
  construction (they live in `ferrum-core`).
- **WASM build stays green:** `cargo build -p ferrum-wasm --target wasm32-unknown-unknown`
  must succeed; the `getrandom/js` blocker must not appear (it can only enter via
  `rand`, which is not a dependency of the shared crate).
- **Single source of truth:** layout/scale math exists in exactly one place; the WASM
  re-layout must not fork or approximate it.
- **Performance:** wheel/pan performance is unchanged. Reactive re-layout is
  rAF-coalesced and must not block the frame loop; at typical interactive sizes a
  re-layout completes within a frame budget, and at large N degrades to fewer updates
  per second rather than stalling.
- **Payload minimality:** only layout-relevant columns are exported; the payload must
  not balloon the interactive HTML for charts that never rescale (it is inert unless a
  `domain`-role binding exists).

## 8. Key decisions and tradeoffs

- **Extract a shared crate, not feature-gate `ferrum-core`.** `ferrum-scene` already
  establishes the "pure-Rust shared kernel + native crate + browser crate" pattern;
  extraction mirrors it and keeps `ferrum-core` PyO3-only. Feature-gating would
  introduce `#[cfg(feature)]` conditional compilation that exists nowhere in the crate
  today and make `ferrum-core` a dual-target crate — less cohesive. Cost: upfront
  move-and-rewire of the layout/scale modules.
- **One crate (`ferrum-layout`) for both layout and scales**, not two. Layout already
  depends on the scale kernels and they move together; a two-crate split buys nothing.
- **Packed-binary payload, not columnar JSON or Arrow.** Binary is strictly ≥ JSON on
  load and first-decode cost and never slower; it reuses ferrum's existing bytemuck
  packed-bytes path (the channel GPU instances already travel), so it is cohesive with
  how ferrum already moves bulk numeric data for speed and scales to large N without a
  wire-size/parse cliff. Steady-state interaction latency is encoding-independent
  (decode-once-and-cache), so the binary choice is about load/first-decode and large-N
  headroom, accepting the cost of dtype tags + a string side-pool. Arrow is rejected:
  no `arrow` dependency exists in WASM today and `pyo3-arrow` is native-only.
- **Affine for zoom/pan; re-layout only on reactive rescale.** Uniform zoom/pan are
  already correct and cheap under the affine (`sx == sy`, no ribbon); only non-uniform
  domain-rescale (`sx ≠ sy`) distorts. Re-laying out everything would put layout cost on
  the hot path for interactions the affine already handles correctly — a regression.
- **Re-tessellate mesh marks with identity transform on rescale.** Correct stroke
  width by construction; reuses the `lyon` dependency already present in WASM.

## 9. Acceptance criteria

- In the interactive 3-plot demo, after a focus+context brush rescale: lines render at
  constant width (no ribbon), axis ticks sit at correct positions for the new domain,
  and tick labels show the correct values with correct spacing.
- Wheel-zoom and pan behave and perform exactly as before the change.
- `ferrum-layout` compiles to `wasm32-unknown-unknown` standalone; `ferrum-wasm`
  builds for `wasm32-unknown-unknown`.
- Full native `cargo test` and the Python golden suite pass unchanged (kernel
  extraction is byte-stable).
- An interactive chart with no `domain`-role binding ships no meaningfully larger HTML
  than before (payload is inert/absent).

## 10. Validation strategy

- **Kernel-parity unit tests in `ferrum-layout`:** for representative specs, the same
  domain + dimensions produce identical layout/scale output, asserting the extraction
  preserved behavior. These run on the native target and the crate also builds for
  wasm32.
- **Native regression:** existing `cargo test` and golden SVG byte/visual checks
  confirm the static path is unchanged.
- **WASM build gate:** `cargo build -p ferrum-wasm --target wasm32-unknown-unknown`
  succeeds in CI.
- **Browser validation:** the 3-plot interactive demo is exported and visually
  inspected — ribbon gone, ticks/labels correct at the rescaled domain, zoom/pan
  unaffected. (This is the CI-unverifiable path; it requires human inspection in a
  browser, consistent with prior interactive validation.)

## 11. Open questions

- **WASM text rendering of new tick labels.** Re-layout produces tick labels with new
  values at the new domain. Whether the WASM runtime can render *arbitrary new* text
  strings (vs. only a pre-baked glyph atlas of the original labels) is unverified and
  could shift the rendering approach for axes (e.g., an HTML/SVG overlay or a dynamic
  glyph path instead of GPU-drawn text). This must be resolved before the axis-text
  portion of re-layout is designed in the plan; it does not affect the crate
  extraction, payload format, or trigger wiring, which proceed regardless.
- **Reconsider Option 1 (screen-space line width) as the lighter fix — added 2026-06-03.**
  v0.15.1 browser validation of the INT-1 cross-panel rescale fix showed that the
  *uniform* zoom/magnify path (sx==sy) renders lines cleanly (they only thicken
  evenly), and the "ribbon" appears *exclusively* under the *non-uniform* rescale
  affine (sx≠sy). This localizes the defect to the non-uniform case and is strong
  evidence that an affine-invariant **screen-space line width** in the mesh shader —
  i.e. apply the per-vertex width offset *after* the affine so it is not stretched by
  the non-uniform scale — would resolve FA-16 on its own, without the full panel
  re-layout. Option 1 is more surgical than the re-layout this spec currently prefers
  (it touches only line/area mesh width, not tick/label spacing or curve resampling).
  Trade-off: Option 1 fixes only stroke width (ticks/labels under the affine remain
  approximate), whereas re-layout fixes everything; but if the brushed-detail axis
  fidelity is acceptable, Option 1 is the cheaper path. Evaluate Option 1 first when
  FA-16 is picked up; reserve the re-layout for when correct ticks/labels/resampling at
  the rescaled domain are also required.
