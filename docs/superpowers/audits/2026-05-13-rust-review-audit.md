# Rust Review — `crates/ferrum-core` coherence audit

**Date:** 2026-05-13
**Status:** draft — awaiting approval before implementation
**Scope:** Full `crates/ferrum-core/src/` (82 files, 38,573 lines). Heavyweight `/rust-review` pass.
**Trigger:** User invoked `/rust-review crates/ferrum-core` after completing Phase 10.

---

## TL;DR

The crate is in good shape. The architecture is well-organized: transforms dispatch via a single-source-of-truth macro (`for_each_transform!`), marks dispatch via an analogous `for_each_mark!`, the render pipeline reads top-to-bottom in `render_svg`, and the `ChartSpec` → `prepare` → `scale_resolve` → `draw` flow is clean. The `ThemeOverridesSpec` serde struct replaced a fragile hand-rolled parser. The position-adjustment pass (`position.rs`) is well-isolated.

No S3+ findings. The drift is concentrated in two places: an inline per-channel encoding merge that bypasses `Encoding`'s existing methods, and `ScaleKind` match-arm repetition that could collapse with a small helper. The rest is scanability polish on a 450-line orchestrator function.

---

## Findings

### F1 — Inline encoding merge in `render_svg` bypasses `Encoding` methods [S2, high confidence]

**Location:** `render/mod.rs:364–374`

**Problem:** The per-panel scale resolution needs a merged encoding (chart-level as base, layer-0's channels overlaid). This is done with 9 manual `if is_some() { field = clone() }` lines:

```rust
let mut merged_encoding = spec.encoding.clone();
let layer0_enc = &prep.layers[0].encoding;
if layer0_enc.x.is_some()       { merged_encoding.x       = layer0_enc.x.clone(); }
if layer0_enc.y.is_some()       { merged_encoding.y       = layer0_enc.y.clone(); }
if layer0_enc.color.is_some()   { merged_encoding.color   = layer0_enc.color.clone(); }
// ... 6 more channels
```

`Encoding` already has `inherit_from(&mut self, parent: &Encoding)` in `spec/encoding.rs:488` which does the inverse (child fills gaps from parent). But there's no `overlay_from` method for the direction needed here (overlay wins when `Some`).

**Why it matters:** When a new encoding channel is added (tooltip, href, description — three exist today and aren't in the merge), the manual block silently misses it. This is the kind of bug that's invisible until a user sets `tooltip=` on a layer and it doesn't resolve correctly in scale_resolve.

**Proposed fix:** Add `Encoding::overlay_from(&mut self, overlay: &Encoding)` — the semantic inverse of `inherit_from`. For each channel: if `overlay.{channel}.is_some()`, replace `self.{channel}`. Use it at the call site:

```rust
let mut merged_encoding = spec.encoding.clone();
merged_encoding.overlay_from(&prep.layers[0].encoding);
```

**Impact:** ~15 lines replaced by 1. The method lives alongside `inherit_from` with matching channel coverage — adding a new channel to `Encoding` means adding it to both methods in the same spot. No public API change (both are `pub` on `Encoding`, which is `pub(crate)`).

**Validation:** Existing render tests + SVG goldens. The merge is used for per-panel scale resolution — any regression would show as axis or scale differences.

---

### F2 — `ScaleKind::pixel_range()` five identical arms [S2, medium confidence]

**Location:** `render/scale_resolve.rs:101–125`

**Problem:** `pixel_range()` matches all five variants and calls `.range_pair()` identically on each:

```rust
pub fn pixel_range(&self) -> (f64, f64) {
    match self {
        Self::Linear(s) => { let r = s.range_pair(); (r[0], r[1]) }
        Self::Ordinal(s) => { let r = s.range_pair(); (r[0], r[1]) }
        Self::Time(s) => { let r = s.range_pair(); (r[0], r[1]) }
        Self::Log(s) => { let r = s.range_pair(); (r[0], r[1]) }
        Self::Symlog(s) => { let r = s.range_pair(); (r[0], r[1]) }
    }
}
```

Similarly, `to_pixel_f64` has four near-identical arms calling `.scale_internal(x)` on the continuous variants.

**Why it matters:** Adding a sixth scale type (e.g. `Pow`) requires updating 6 match blocks, most of which are boilerplate.

**Proposed fix:** Add a `macro_rules! dispatch_scale!` local to `ScaleKind` that expands a per-variant call on a method common to all (or all continuous) scale types. Alternatively, a small `ScaleInternal` trait with `range_pair` + `scale_internal` implemented on the four continuous types would let `pixel_range` and `to_pixel_f64` collapse. `tick_labels` genuinely differs per variant (different formatters) and stays as a manual match.

**Trade-off:** The trait adds indirection for ~40 lines saved. The macro is zero-cost but slightly less readable. Either is better than the current 5-way copy-paste.

**Recommendation:** Macro approach — it's local, doesn't change the type hierarchy, and collapses cleanly.

**Impact:** ~40 lines collapsed. No public API change.

**Validation:** `cargo test`. Scale behavior unchanged.

---

### F3 — `render_svg` body is ~450 lines with inlined title and legend rendering [S2, medium confidence]

**Location:** `render/mod.rs:171–500+`

**Problem:** `render_svg` is the top-level pipeline orchestrator. It's well-commented and reads top-to-bottom, but two blocks are self-contained and could be extracted:

1. **Title rendering** (lines 242–288): Resolves per-chart TitleSpec overrides, builds TextStyle, emits title + optional subtitle. ~45 lines with no state dependency on anything after it — pure output to `SvgBuffer`.

2. **Legend rendering** (lines 442–end): Builds a separate rendering_spec, re-resolves scales for the legend color palette, dispatches to `marks::legend::draw`. ~60 lines.

**Why it matters:** Moderate. The function is scannable as-is because the blocks are well-delimited by comments. But extracting them would make the per-panel loop (the core of the function) easier to find when skimming.

**Proposed fix:** Extract `render_title(layout, spec, theme, out)` and `render_legend(layout, spec, prep, theme, out)` as module-private functions. No signature changes, no behavior changes.

**Impact:** `render_svg` drops from ~450 to ~350 lines. Two new ~50-line functions. No public API change.

**Validation:** SVG golden byte-comparison. The extracted functions are pure output — no state escapes.

---

### F4 — `inherit_from` and the inline merge don't cover `tooltip`/`href`/`description` consistently [S2, high confidence]

**Location:** `spec/encoding.rs:524–536` (`inherit_from`) vs. `render/mod.rs:364–374` (inline merge)

**Problem:** `inherit_from` covers 12 channels (x, y, color, size, shape, opacity, x2, y2, text, tooltip, href, description). The inline merge in `render_svg` covers only 9 (missing tooltip, href, description). This means layer-0's tooltip/href/description channels don't overlay onto the merged spec used for scale resolution.

Today this is harmless — tooltip/href/description are `_SILENT_CHANNELS` in the Python layer and don't participate in scale resolution. But if tooltip ever drives a scale (e.g. tooltip formatting), the merge would silently drop it.

**Proposed fix:** Subsumed by F1. The `overlay_from` method would cover all channels in lockstep with `inherit_from`. Both methods enumerate the same channel list — adding a channel means updating both in the same struct impl block.

---

### F5 — `MarkStyle` has 22 fields, many mark-specific [S1, high confidence]

**Location:** `render/draw.rs:28–66`

**Problem:** `MarkStyle` is a flat struct with 22 fields. Text-specific fields (`font_size`, `font_weight`, `align`, `baseline`, `dx`, `dy`, `angle`), polygon-specific fields (`detail`, `cmap`), line-specific fields (`interpolate`, `stroke_cap`, `stroke_join`), and point-specific fields (`filled`, `shape`) are all `Option<_>` and ignored by non-matching marks.

**Assessment:** This is pragmatic and intentional. A per-mark substyle enum or trait hierarchy would add type-level correctness but also indirection and boilerplate for no runtime benefit — the fields are just read-or-skipped in each mark's `draw` function. The flat struct is the simplest representation that works.

**Proposed fix:** None. Leave as-is. The field count is a consequence of supporting 12 mark types with different style surfaces; the flat struct is the right trade-off.

---

### F6 — `scale_resolve.rs` at 1,445 lines [S1, low confidence]

**Location:** `render/scale_resolve.rs`

**Problem:** The file contains axis scale resolution, channel scale builders (color, size, shape, opacity), domain-union logic, tick generation, and sort handling. It's internally well-organized with clear section headers.

**Assessment:** The file could be split into `scale_resolve.rs` (axis + resolution) and `channel_scales.rs` (color/size/shape/opacity builders), but the current organization is logical — all scale-building code lives together. Only worth splitting if the file keeps growing past ~1,800 lines.

**Proposed fix:** None for now. Monitor.

---

## Summary table

| ID | Finding | Severity | Confidence | Public API change | Lines changed (est.) |
|---|---|---|---|---|---|
| F1 | Inline encoding merge bypasses `Encoding` methods | S2 | high | none | -15, +20 |
| F2 | `ScaleKind` five-way match-arm duplication | S2 | medium | none | -40, +15 |
| F3 | `render_svg` inlined title/legend blocks | S2 | medium | none | +0 (restructure) |
| F4 | Encoding merge misses 3 channels | S2 | high | none | subsumed by F1 |
| F5 | `MarkStyle` 22 fields | S1 | high | none | 0 (leave as-is) |
| F6 | `scale_resolve.rs` length | S1 | low | none | 0 (monitor) |

---

## Proposed implementation order

1. **F1 + F4** — Add `Encoding::overlay_from`, use it in `render_svg`. Highest leverage — fixes the missing-channel bug (F4) and removes the inline merge. One change, two findings closed.
2. **F2** — `ScaleKind` dispatch macro. Self-contained, moderate value.
3. **F3** — Extract `render_title` / `render_legend`. Low risk, improves scanability.
4. **F5, F6** — No action.

---

## Decisions

1. **Scope** — Implement F1+F4, F2, F3. Skip F5/F6 (no action needed).
2. **Branch** — Single `refactor/rust-review` branch or direct to main (all changes are internal, no public API).
