# Open-bug sweep — batch remediation plan (2026-07-23)

Branch: `fix/open-bug-sweep`. Source: all 15 open `bug`-labeled GitHub issues, triaged via `/remediating-issues` (batch path).

## Triage outcome

**Closed as already-fixed (9)** — #60, #69, #70, #71, #72, #73, #74, #75, #76. All were resolved by the post-v0.19 sweep and shipped in v0.20.0 (`ff01dcae` … `2c3e26e4`, release `61f91074`); the GitHub issues were never closed. Verified on current `main`: all 219 named Python evidence tests and 527 `ferrum-wasm` tests green. Each closed with a per-issue evidence comment (fix commit + green-test proof).

**Kept open, not a code fix (1)** — #51 (wgpu scissor-rect): upstream-tracking issue; the in-repo `ensure_scissor!` workaround already shipped and is correct. Clarifying comment posted.

**Remediated in this sweep (5 issues → 4 work items)** — #84, #78+#58 (one fix), #66, #77.

## Work items and defended choices

### W1 — #84 `annotate_arrow(label=)` breaks `+` overlay (Python)

Labeled arrow `&`-composes its label → returns `VConcatChart`; `chart + arrow` raises TypeError and the docstring's "Returns Chart" contract is violated. **Fix:** mirror the #82 `_attach_line_with_label` machinery — keep the arrow a plain `Chart` (standalone renders via `mark_segment`), add the label as a real `_annotations` entry, set `_annotation_primitive = [arrow_primitive, text_primitive]` so `Chart.__add__` expands both on overlay. Rejected: teaching `+` to accept `VConcatChart` (wrong semantics, over-reaching); layer-merge composition (no sibling precedent — every labeled helper uses the primitive path).

### W2 — #78 + #58 tooltip provenance (Python, one fix)

Both issues fire through the same short-circuit at `_spec_build.py:767` (`chart_level_explicit and not any_layer_explicit`), which infers *provenance* from *structure*. **Fix:** a `_tooltip_promoted` slot on `Chart` (the `_mark_zero` slot idiom; auto-cloned by the generic `_clone` loop), set in `Chart.__add__` where promotion actually happens; the short-circuit reads the marker instead of re-deriving history. Selection-injected `tooltip_fields` (#58) likewise do not count as a genuine chart-wide override: the per-layer walk runs and unions the selection's field set into each layer, preserving linked-selection field matching. Rejected: keeping the structural proxy with a selection special-case (perpetuates the inference #78 exists to remove); a #58-only patch (leaves the #78 mixed-layer latent edge live).

### W3 — #66 dodge sub-band overlap (Rust) — issue premise refuted, real sibling defect fixed

Investigation disproved the reported mechanism: `BandScale(padding_inner=)` is geometrically **inert** on the dodge path (byte-identical SVG at 0.0/0.5/0.9 — `OrdinalScale::bandwidth()` is `E/n`, not padding-aware, so widths and offsets already share one notion of sub-band size). The **real defect on the same seam**: `Dodge(padding > 0.1)` — bar width bakes `0.8·bw/g` (dodge-padding-blind, `bar.rs:315`/`bar.rs:437`) while sub-band spacing is `bw·(1−2p)/g` (`position.rs:466-474`), so bars visibly overlap (verified 20 px at p=0.2, 60 px at p=0.4). **Fix:** clamp mark width to its sub-band — enforces the true invariant (a sub-bar fits its sub-band), byte-identical at default padding 0.05 (no golden churn), correct across the whole p range. Plus two discriminating tests: `padding_inner`-inertness, and no-overlap at p=0.2 (RED pre-fix). Rejected: the issue's "consult `bandwidth()`" (a no-op — also padding-blind); full d3 padding-aware band geometry (the #67 north-star; logged there, not re-litigated here).

### W4 — #77 `apply_stack` axis selection (Python + Rust) — audit confirmed a real bug

Two reachable failures, both with `coord_flipped == false` while the value axis lives on x: `mark_histogram(orient="horizontal", multiple="stack")` → hard `ValueError` ("Stack: x column must be Float64 or Utf8") for every input; `mark_density(orient="horizontal", multiple="stack")` → silently corrupted geometry (cumulates the position column binned by density). Root cause: composite desugars (`desugar_histogram`/`desugar_density`) swap x/y via remap without setting CoordFlip, and `apply_stack` (`position.rs:658-662`) guesses the value axis from `coord_flipped` alone. **Fix:** carry the fact on the wire instead of reconstructing it — `Stack` gains an explicit orientation field (serde-defaulted, additive); the desugars set it; `apply_stack` reads it and falls back to `coord_flipped` for primitive marks (byte-identical for all existing output). Precedent: the #60 slot-tag remediation ("carried, not reconstructed"). Rejected: the issue's suggested ordinal-scale mirror of `apply_dodge` (**insufficient** — both axes are continuous in the reachable cases); Rust-side x2/y2 structural inference (fixes histogram, cannot fix density — a banned partial fix); desugars setting CoordFlip (over-reaching — remap is load-bearing for transform column binding).

## Gates per work item

Coder agent (python-coder / rust-coder per file type) → regression test proven RED on pre-fix code (stash-proof) → `*-review-lite` gate → commit → origin issue closed with evidence. Sequential execution (W1→W2→W3→W4); full-suite + cargo test verification at the end before `finishing-a-development-branch`.
