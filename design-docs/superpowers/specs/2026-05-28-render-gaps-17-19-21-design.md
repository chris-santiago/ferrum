# Render gaps — hex stroke, Raw-in-WASM, Label scene-kind — Design Spec

**Date:** 2026-05-28
**Tracks:** `design-docs/superpowers/followups/2026-05-15-code-archaeology.md` items 17, 19, 21
**Constraint:** Implemented fully per the project no-defer rule — no warn-fallbacks, no `NotImplementedError`.

## 1. Scope

Close three independent code-archaeology gaps in ferrum's rendering pipeline: (17) `mark_hex(stroke=, stroke_width=)` currently raises `ValueError` instead of rendering hex-cell borders; (19) `SceneNode::Raw` is silently skipped in the WASM/interactive renderer, so insets, legend colorbar gradients, and annotation images vanish in interactive export; (21) `mark_label` emits `MarkBatchKind::Text`, leaving labels indistinguishable from text marks in the scene graph (F16). The three items are otherwise unrelated and share only a feature branch.

## 2. Goals

- `mark_hex` accepts `stroke` and `stroke_width` and renders hex-cell borders, with semantics consistent with every other polygon-family mark.
- Interactive (WASM) charts render the same Raw-backed content as static SVG: continuous-color colorbars, insets, and annotation images.
- The scene graph distinguishes label batches from text batches, so `to_json()` output and downstream consumers (e.g. the Phase 13 renderer-plugin surface) can tell them apart.
- No regression to the static SVG renderer, to non-Raw WASM rendering, or to existing text/label visual output.

## 3. Non-goals

- `ferrum.Grid` (item 18) — separately designed in `ferrum-spec.md §3.19`; gets its own plan.
- `share_x` / `share_y` dead-surface cleanup (item 20).
- Any new label-specific hover/hit-test behavior. The `Label` scene-kind is a tag only; interaction is unchanged.
- Exact data-space re-projection of Raw content under pan/zoom (impossible from an opaque pre-rendered SVG string; see §8).

## 4. System behavior

**Item 17 — hex stroke.** `mark_hex(stroke="#fff", stroke_width=1)` renders each hex cell with a 1px white border. `stroke` and `stroke_width` are independent: a stroke color with `stroke_width` left at its `0` default produces **no visible border** (literal semantics, matching all polygon-family marks). The previous `ValueError` for either argument no longer occurs.

**Item 19 — Raw in WASM.** A chart that produces Raw content (continuous-color legend gradient, inset, annotation image) renders that content in interactive export instead of dropping it. Raw content paints **above** GPU marks. On pan/zoom: chrome Raw (colorbar/legend) stays fixed; data-anchored Raw (annotation image, inset positioned in panel/data space) tracks the same transform applied to the canvas. The Raw overlay does not intercept pointer events, so hover/zoom interaction with underlying marks is unaffected. The static SVG renderer is unchanged.

**Item 21 — Label kind.** No user-visible change. Labels render exactly as before (outside the clip region, hit-tested as text). The scene-graph JSON now carries `"label"` as the batch kind for label marks instead of `"text"`.

## 5. Architecture

**Item 17** lives entirely in the Python desugar layer. `mark_hex` desugars to a `polygon` layer; the existing Rust polygon renderer already reads `stroke`/`stroke_width` from mark kwargs (`resolve_mark_style`) and emits them into `SceneNode::Polygon`'s `FillStroke`. The fix passes the kwargs through; no Rust change.

**Item 19** spans three layers:
- *Scene type* (`ferrum-scene`): `SceneNode::Raw` gains an `anchor` discriminant identifying whether the fragment is chrome (fixed) or data-anchored (transform-tracking).
- *Core producers* (`ferrum-core`): each Raw emitter sets `anchor`. The static walker ignores it.
- *WASM + JS*: `scene_load` forwards Raw fragments (instead of dropping them) into the scene-export channel already used for text nodes; the JS widget injects them into the existing text-overlay `<svg>`, with per-fragment ID namespacing and anchor-based grouping.

The mechanism reuses two existing patterns: the static renderer's verbatim `svg.raw(...)` passthrough, and the WASM renderer's text-only DOM overlay. No new render surface or SVG-parsing/tessellation subsystem is introduced.

**Item 21** is a scene-graph tag plus the handful of consumer match sites that branch on batch kind (clip decision, hit-test dispatch). Label batches are routed identically to text everywhere.

## 6. Canonical interfaces / data contracts

**`SceneNode::Raw` (extended):**

```rust
enum RawAnchor { Chrome, Data }   // serde snake_case: "chrome" | "data"

SceneNode::Raw {
    svg: String,
    #[serde(default)]   // absent => Chrome (fixed); keeps old scenes deserializable
    anchor: RawAnchor,
}
```

`RawAnchor::Chrome` is the default. Producers assign:
- legend colorbar gradient → `Chrome`
- inset → `Chrome`
- annotation image → `Data` when positioned in panel/data space, else `Chrome`

**`MarkBatchKind` (extended):** add `Label` variant; serde snake_case `"label"`.

**Raw ID-namespacing contract (JS):** within each injected Raw fragment, every `id="X"` is rewritten to `id="ferrum-raw-{n}-X"` and every matching `url(#X)` reference is rewritten to `url(#ferrum-raw-{n}-X)`, where `{n}` is a per-fragment counter. This guarantees no `<defs>` ID collisions across multiple Raw fragments or with other overlay content.

## 7. Invariants and constraints

- **Static renderer unchanged.** `svg_walk` continues to emit `SceneNode::Raw.svg` verbatim and ignores `anchor`. The new `anchor` field and `Label` kind produce byte-identical static SVG for existing charts.
- **Serde back-compat.** `anchor` defaults to `Chrome` so scenes serialized before this change deserialize without error. `MarkBatchKind::Label` is safe across the core↔WASM JSON boundary because both crates are version-pinned and built together.
- **No new dependencies / no new render surface** for item 19; the colorbar/inset/annotation fix must not pull in an SVG parser or tessellator.
- **Hex stroke matches sibling marks.** No call-time inference of stroke width from a stroke color — consistent with the codebase, where only `geoshape` has a visible default stroke and it derives from a theme default.
- **Overlay non-interactivity.** The Raw overlay remains `pointer-events: none`.
- **Label routing parity.** Anywhere batch kind is consumed, `Label` behaves exactly as `Text`.

## 8. Key decisions and tradeoffs

- **Hex stroke = literal semantics** (`stroke` with width 0 ⇒ invisible). Chosen over a hairline auto-default for paradigm consistency with all other polygon marks; the ergonomic cost (a stroke color alone does nothing) is documented in the docstring.
- **Raw in WASM = DOM overlay passthrough.** Reuses the static passthrough and the existing text overlay. GPU tessellation (parse + tessellate SVG, synthesize gradient textures) and a hybrid 2D-canvas composite were rejected as new subsystems with no codebase precedent and poor fit for gradients/images/nested SVG.
- **Raw paint order and transform model.** Raw paints above GPU marks; chrome Raw is fixed, data-anchored Raw rides the canvas transform as a unit. Exact per-feature re-projection is impossible from an opaque SVG string, so the transform is applied to the fragment group. This is visually correct for all three real producers and strictly better than the current silent drop. The `anchor` discriminant exists solely to drive this fixed-vs-tracking choice.
- **Label = tag only.** Satisfies F16 (scene-graph distinguishability) with zero behavior change and minimal blast radius. A label-specific hover policy was considered and explicitly deferred as out of scope.
- **Build order 17 → 21 → 19.** Independent items ordered by risk; the cross-language item (19) is last. Single feature branch.

## 9. Acceptance criteria

- `mark_hex(stroke="#fff", stroke_width=1)` emits `stroke`/`stroke-width` attributes on hex polygons in the SVG output; `mark_hex(stroke="#fff")` alone emits no visible border; passing either argument no longer raises.
- A continuous-color chart exported via `.interactive()` includes its colorbar gradient SVG in the exported bundle (previously dropped).
- `SceneNode::Raw` is collected (not skipped) by the WASM scene loader with the correct `anchor`.
- `mark_label` produces a batch whose kind serializes to `"label"`; plain text still serializes to `"text"`; label hit-testing still resolves via the text hit-test path; the `Label` variant round-trips through serde.
- Static SVG output for existing charts is unchanged.
- `cargo test`, `uv run pytest -n auto`, and `cargo clippy` (including the `wasm32-unknown-unknown` target) all pass; any changed/added goldens are visually inspected per CLAUDE.md.

## 10. Validation strategy

- **Item 17:** behavioral SVG-attribute assertions for the stroke / no-stroke / no-longer-raises cases.
- **Item 19:** Rust unit test that the scene loader collects Raw with the correct anchor instead of dropping it; Python integration test asserting the interactive bundle contains the colorbar gradient (direct regression against the silent drop).
- **Item 21:** Rust tests for batch-kind assignment (label vs text), hit-test routing parity, and serde round-trip of the new variant.
- On completion, update items 17/19/21 status columns and the corresponding action-list entries in the code-archaeology followup doc.

## 11. Open questions

None blocking.
