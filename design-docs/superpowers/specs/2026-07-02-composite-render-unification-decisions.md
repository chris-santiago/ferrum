# Phase B — §11 Sub-Decision Records (coherent-change decision-only)

Companion to `2026-07-02-composite-render-unification-design.md` §8 D4 / §11. Each
decision was researched against the actual code (file:line evidence in the research
transcripts) and is presented in defended-choice form. Approved by user sign-off
before any implementing task consumes them.

---

## D4a — Per-panel layout-scale representation

**Reframe.** The WASM renderer already has a complete per-panel affine path (the
FA-18 fix): `PanelTransformSlot` per panel indexed by `panel_id`, an
`Affine2 {sx, sy, tx, ty}` uploaded per frame, bind-group plumbing done
(`ferrum-wasm/src/render.rs:20-31, 291-324`). The static SVG walker has NO per-panel
transform construct today; its only ratio precedent (`compositor.rs::write_scaled_cell`)
is **non-uniform** — joint marginals genuinely need sx≠sy (`grid_compose.rs` computes
K_w/K_h independently per axis). Scene-schema back-compat convention is
`#[serde(default)]` (`ferrum-scene/src/types.rs:20, 56, 156`).

**Chosen.** A `layout_scale` field on `Panel`: `{sx, sy, tx, ty}` (f64), serde-default
identity (`sx=sy=1, tx=ty=0`), skip-serialized when identity. WASM composes it as each
panel's *base* affine, pre-multiplied by zoom/pan into the existing slot — zero new GPU
machinery. The static walker wraps the panel's emitted content in a
`<g transform="translate(tx,ty) scale(sx,sy)">` (new construct, semantics identical to
today's `preserveAspectRatio="none"` viewport mapping, strokes/text scale the same way).

**Rejected.**
- *Scalar uniform scale + origin* — cannot represent the sx≠sy joint-marginal case,
  the exact cells this field exists for.
- *Full 2×3 affine* — no consumer: neither renderer uses rotation/shear, the WASM
  `Uniforms` carries only sx/sy/tx/ty (a 2×3 would truncate to the chosen shape at
  upload), and no serde-friendly 2×3 type exists in the scene crates to reuse.
- *Target rect, walker derives scale* — requires the source extent alongside the
  target and forces both walkers to re-derive sx/sy that the layout pass already
  knows; an extra derivation step on both sides for no expressive gain.

---

## D4b — Leaf render seam (resolved domains into per-leaf rendering)

**Reframe.** Three facts. (1) `prepare_and_layout` is already the natural single-leaf
unit (private, one spec + one batch, per-call theme/config, no global state) — but
final per-panel scales are resolved *inside* `build_scene::resolve_panel_scales`, not
in the prepare step. (2) The explicit-domain bypass (`positional.rs:83`) is a
**different rendering path** from facet-shared resolution: padding forced to 0
(`resolve_padding_fraction`, positional.rs:54), nice/clamp flag-driven instead of the
auto path's hardcoded behavior, and no `FINAL_OUTPUT_KEY` union. (3) The spec's
resolve semantics require a user's explicit `enc.scale` to WIN over sharing — so the
sharing mechanism must remain *distinguishable* from user scales.

**Chosen.** Thread an **optional resolved-domain context** through the per-leaf render:
`prepare_and_layout` → provisional pass → `resolve_panel_scales` →
`resolve_scales_with_outputs` → `build_axis_scale`, consulted where `include_final`
is consulted today. Composite-shared leaves therefore render on the **auto path** —
same `DEFAULT_SCALE_PADDING_FRAC`, same nice/clamp behavior as facet-shared panels
(spec D1/D3 intent) — and a leaf whose channel carries a genuine user `enc.scale`
still short-circuits at the existing bypass (user wins, unchanged). Additive
parameter threading, no restructure; `prepare_and_layout`'s signature gains one
optional context.

**Rejected.**
- *Domain injection via `EncodingSpec.set_domain` (the obvious candidate — zero code
  change)* — routes shared leaves through the user-explicit-scale path: padding 0 vs
  the facet path's padding fraction (visibly different axes), flag-driven nice vs the
  auto path's behavior, and — decisive — composite-injected domains become
  indistinguishable from genuine user overrides, so "user scale wins over sharing"
  cannot be enforced. Contradicts spec D3 (shared domains through the facet
  mechanism). Cheapness is not worth a semantic fork between composite-shared and
  facet-shared rendering.
- *Pseudo-facet synthesis* — the facet machinery is structurally bound to one
  `ChartSpec`, one facet field, one merged batch whose panels are row-filters of the
  same data (`filter_batch_by_facet`, `FINAL_OUTPUT_KEY` concat). Heterogeneous
  children (joint = scatter + two marginals) cannot be expressed. The spec reuses the
  facet *mechanism*, not the facet *spec type*.

**Orthogonal obligations recorded for the layout/scene tasks (not seam-dependent):**
`build_scene` owns `SceneGraph` construction and numbers panels locally
(`id = enumerate index`), so the composite scene pass must globally renumber
panel/clip ids and emit N leaves into one graph/viewport; figure chrome uses the
scene-native `build_figure_chrome_nodes` (exists at `figure_chrome.rs:343`), never
the string `wrap_with_chrome`.

---

## D4c — Packed-buffer panel indexing

**Reframe.** The packed format's 20-byte header carries `panel_idx`, and the WASM
loader keys batches by `(enumerate position in scene.panels, batch_idx)` — a hard
contiguity/zero-base constraint (`scene_load.rs:557, 600, 780`). Everything downstream
keys on that same flat position: transform slots (`0..panel_count`), `param_bindings`,
`tick_levels`, `linked_panels`, hit-testing. The facet builder already produces this
invariant natively (`Panel { id: panel_idx }`, header written from the same
enumerate), and the legacy Python merge proves renumbering feasibility — it exists
only to patch headers after the fact.

**Chosen.** **Single flat panel namespace, written correct-first.** The composite
scene pass assigns each leaf's panels their final 0-based positions in the one
`SceneGraph.panels` vec and emits packed batches with those indices directly — no
post-hoc header rewrite exists anywhere. `panel.id`, header `panel_idx`, transform
slots, and all interaction references share the one namespace by construction (the
facet model, generalized).

**Rejected.**
- *Per-leaf namespaces + offset table* — requires a new indirection inside
  `unpack_binary_instances` plus a parallel table threaded to every consumer,
  reintroducing exactly the two-phase patch-after-merge design this phase deletes.
- *Concatenated per-leaf buffers + manifest* — describes per-leaf boundaries the
  unified scene no longer has; consumers would still need per-leaf→flat resolution
  (same indirection, more format).
