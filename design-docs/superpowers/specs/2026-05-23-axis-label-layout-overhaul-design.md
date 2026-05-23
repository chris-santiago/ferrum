# Axis Label Layout Overhaul — Design Spec

**Date:** 2026-05-23
**Depends on:** Phase 6 (Layout Engine), Phase 7 (Static Renderer)
**Modifies:** `crates/ferrum-core/src/layout/axis.rs`, `crates/ferrum-core/src/layout/mod.rs`

---

## 1. Scope

Overhaul the x-axis label collision policy and facet axis-title suppression in the layout engine to fix three bugs (aggressive elision, y-axis title duplication, rotated-label clipping) and add five improvements (multi-line wrapping, graduated angle cascade, dynamic bottom margin, tick culling, font-size reduction). The current policy — flat → -45° → elide — produces degenerate output (2-3 character labels, overlapping titles, clipped rotated text) in real-world faceted charts with categorical axes.

---

## 2. Goals

- Labels on categorical x-axes remain legible without user intervention for up to ~20 categories at default viewport width (600px).
- Faceted charts render exactly one y-axis title (on the leftmost column) and one x-axis title (on the bottom row) — no duplication, no overlap.
- Bottom margin dynamically accommodates rotated label extent so labels are never clipped by the viewport boundary.
- The collision cascade is graduated (wrapping → font reduction → angle escalation → tick culling → elision), exhausting less-destructive strategies before resorting to character truncation.
- All behavior is deterministic and renderer-agnostic (works identically for SVG and WASM).

---

## 3. Non-goals

- Font-file-based text measurement (`FontdueMetrics`). The heuristic `TextMetrics` trait remains the measurement backend; accuracy improvements are a separate concern.
- Y-axis label collision. Y-axis labels are stacked vertically and do not collide horizontally; no changes to `layout_y_axis`.
- User-facing API additions (new `Axis(...)` parameters). This spec changes internal layout behavior; any new knobs (e.g., `Axis(wrap=True)`) are follow-up scope.
- Log/symlog/time scale tick formatting — those scales produce uniform-width numeric labels that rarely trigger collision.

---

## 4. System behavior

### 4.1 Collision cascade (replaces current flat → -45° → elide)

When x-axis tick labels exceed their per-slot horizontal budget (`slot_w × (1 - LABEL_OVERLAP_TOLERANCE)`), the layout engine applies recovery strategies in order, stopping at the first that eliminates all collisions:

| Stage | Strategy | Condition to advance |
|-------|----------|---------------------|
| S0 | **Flat (no action)** | No collision detected → done |
| S1 | **Multi-line wrapping** | Split labels at word boundaries; re-measure using max line width. If all wrapped labels fit within `slot_w` → done |
| S2 | **Font-size reduction** | Reduce label font size by one step (to `label_font_size × FONT_SHRINK_FACTOR`). Re-measure. If all labels fit flat → done. If all wrapped labels fit → done (apply wrapping + reduced font) |
| S3 | **Graduated rotation** | Try angles from `ANGLE_CASCADE` in order. At each angle, check `label_w × cos(|angle|) ≤ slot_w`. First angle where all labels fit → done |
| S4 | **Tick culling** | If `n_labels > CULL_THRESHOLD`, show every Nth label (smallest N where remaining labels fit at the best angle from S3). Culled ticks retain their position mark but lose their label |
| S5 | **Elision** | Last resort. Truncate with `…` to fit budget, as today |

Each stage's output is a `(labels: Vec<String>, angle: f64, font_size: f64, visible: Vec<bool>)` tuple. The cascade is a linear scan — no backtracking, no fixed-point iteration.

### 4.2 Multi-line label wrapping (S1)

Labels are split into lines at natural break points:

1. **Underscore:** `trivial_baseline` → `["trivial", "baseline"]`
2. **Space:** `long category name` → `["long category", "name"]` (greedy line-fill: pack words onto the first line until adding the next word would exceed `slot_w`, then wrap)
3. **camelCase boundary:** `featureImportance` → `["feature", "Importance"]`

Only the first applicable rule fires (underscore > space > camelCase). If no break point exists, the label is not wrappable and passes through to S2. Wrapping is unlimited — a label like `very_long_snake_case_name` splits into 4 lines. The vertical extent (`n_lines × line_height`) is fully accounted for in the bottom margin reservation.

The measured width of a wrapped label is `max(line_widths)`. The vertical extent is `n_lines × line_height`. Multi-line labels are always rendered at 0° (no rotation of wrapped text).

A `TickLayout` gains a `lines: Vec<String>` field (or the existing `label` field contains `\n`-joined lines). The renderer splits on `\n` and emits one `<tspan>` per line (SVG) or stacked text draws (WASM).

### 4.3 Font-size reduction (S2)

A single reduction step: `reduced_font = label_font_size × FONT_SHRINK_FACTOR`. No iterative shrinking — one step only. If the reduced size resolves collision (flat or wrapped), use it. Otherwise, proceed to S3 at the original font size (rotation at a smaller font is hard to read).

`TickLayout` gains an optional `label_font_size: Option<f64>` override. When `None`, the renderer uses the theme default.

### 4.4 Graduated rotation (S3)

Replace the single `-45°` default with an ordered cascade:

```
ANGLE_CASCADE = [0, -30, -45, -60, -90]
```

For each candidate angle θ, check whether `max(label_w × |cos(θ)|) ≤ slot_w`. The first angle that passes wins. At -90°, `cos(90°) ≈ 0`, so nearly any label fits — this makes -90° a near-guaranteed resolution before elision.

The existing `label_angle_override` from `Axis(label_angle=...)` bypasses the cascade entirely (current behavior preserved).

### 4.5 Tick culling (S4)

When rotation alone doesn't resolve collision (very dense axes — 30+ categories in a narrow panel), show every Nth label:

- Compute the minimum stride N such that `max(label_w × |cos(best_angle)|) ≤ slot_w × N`.
- Labels at positions `i % N != 0` have their text cleared but retain their tick mark.
- `TickLayout.culled: bool` flag indicates suppressed labels.
- The stride N is uniform; it does not selectively hide "less important" labels (no heuristic ranking).

`CULL_THRESHOLD` controls the minimum label count before culling is considered (to avoid culling a 5-label axis down to 2). Culling only fires when `n_labels > CULL_THRESHOLD`.

### 4.6 Elision (S5)

Unchanged from current `elide_to_fit()` behavior — character-prefix truncation with `…` suffix. This stage should fire rarely with the preceding cascade in place.

### 4.7 Y-axis title suppression in faceted charts

Mirror the existing x-axis title suppression logic: suppress y-axis titles on all panels except the leftmost column.

```
min_col = panel_rects.iter().map(|(_, c, _, _)| *c).min().unwrap_or(0);

// In the per-panel loop:
if col > min_col && spec.facet.is_some() {
    y_input.title = None;
}
```

This ensures exactly one y-axis title appears per faceted chart, on the left edge.

### 4.8 Dynamic bottom margin

Replace the fixed `x_label_band = line_height(label_font_size)` with a rotation-aware estimate:

1. **Before layout:** compute a preliminary collision check using the worst-case label width (longest label in `axes.x.tick_labels`) against an estimated slot width (`estimated_plot_width / n_labels`).
2. **Estimate the angle** the cascade will likely choose (run stages S1-S3 of the cascade against the worst-case label to determine the probable angle).
3. **Reserve bottom margin:**
   - If wrapping: `n_lines × line_height`
   - If rotated: `max_label_width × |sin(angle)| + line_height × |cos(angle)|`
   - If flat: `line_height` (current behavior)

This is a **bounded two-pass** approach: the first pass estimates the angle from worst-case inputs (O(1) — no per-tick iteration), the second pass is the existing per-panel layout. The plot rect from step 6 remains final — the only change is that the bottom gutter in step 5 is angle-aware rather than fixed.

The single-pass commitment from Phase 6 §6 is preserved in spirit: the worst-case estimate means the plot rect is still determined before per-panel tick layout runs. The estimate may over-reserve slightly (it uses the longest label, not the actual collision outcome), but over-reservation is strictly better than under-reservation (clipping).

---

## 5. Architecture

All changes are confined to the Rust layout engine (`crates/ferrum-core/src/layout/`). No Python-side changes except consuming new `TickLayout` fields in the renderers.

**Computation ownership:**
- `axis.rs` — collision cascade logic, `elide_to_fit()`, new `wrap_label()`, `cascade_collision_recovery()`
- `mod.rs` — bottom margin estimation, y-axis title suppression, `ANGLE_CASCADE` and `CULL_THRESHOLD` constants
- `text_metrics.rs` — `measure_multiline_width()` helper (max of per-line widths)

**Renderer changes (consumers):**
- SVG renderer (`svg_walk.rs` / `marks/axis.rs`) — render multi-line tick labels as `<tspan>` elements; respect per-tick `label_font_size` override
- WASM renderer — same semantics, different output format

---

## 6. Canonical interfaces / data contracts

### 6.1 Extended TickLayout

```rust
pub struct TickLayout {
    pub position: f64,
    pub label: String,         // may contain '\n' for multi-line labels
    pub label_angle: f64,
    pub elided: bool,
    pub culled: bool,          // NEW: tick mark shown, label hidden
    pub label_font_size: Option<f64>,  // NEW: per-tick override (None = theme default)
}
```

### 6.2 Collision cascade output

```rust
struct CascadeResult {
    labels: Vec<String>,       // final labels (wrapped/elided/original)
    angle: f64,                // chosen rotation angle
    font_size: Option<f64>,    // None = theme default; Some = reduced
    visible: Vec<bool>,        // false = culled (tick mark only, no label)
    strategy: CascadeStrategy, // diagnostic: which stage resolved
}

enum CascadeStrategy {
    Flat,
    Wrapped,
    FontReduced,
    Rotated { angle: f64 },
    Culled { stride: u32 },
    Elided { count: u32 },
}
```

`CascadeStrategy` is diagnostic-only (for warnings and debugging). It is not serialized into `LayoutResult`.

### 6.3 Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `ANGLE_CASCADE` | `[0.0, -30.0, -45.0, -60.0, -90.0]` | Ordered rotation candidates |
| `FONT_SHRINK_FACTOR` | `0.82` | Single-step font reduction (11pt → ~9pt) |
| `CULL_THRESHOLD` | `8` | Minimum label count before culling is considered (theme-configurable via `ThemeInputs.cull_threshold`) |
| `LABEL_OVERLAP_TOLERANCE` | `0.10` | Unchanged from Phase 6 |

---

## 7. Invariants and constraints

- **Single-pass plot rect:** The plot rect determined in step 5-6 of `compute_layout` is final. The collision cascade in step 7 mutates label text/angle/visibility, never the panel geometry. The new bottom-margin estimation uses worst-case inputs, not actual cascade outcomes.
- **Deterministic output:** Same inputs produce identical `LayoutResult`. No randomness, no font I/O, no platform-dependent behavior.
- **Heuristic metrics only:** All width measurements use `TextMetrics` trait. The cascade's correctness does not depend on measurement accuracy — it depends on measurement *consistency* (same metric used for checking and rendering).
- **Backward compatibility:** Charts that currently render with flat labels (no collision) produce byte-identical output. The cascade only fires when collision is detected.
- **Renderer agnostic:** `TickLayout` is a pure data struct. SVG and WASM renderers independently interpret `\n` in labels and `culled`/`label_font_size` fields.
- **`label_angle_override` always wins:** When the user sets `Axis(label_angle=...)`, the cascade is bypassed entirely. The override angle is applied unconditionally, then elision fires if labels still collide at that angle. Current behavior preserved.

---

## 8. Key decisions and tradeoffs

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Cascade order | wrap → shrink → rotate → cull → elide | Wrapping preserves full label text and horizontal reading direction. Rotation is more disruptive than wrapping but less destructive than truncation. Culling preserves readability of shown labels at the cost of hiding some. Elision is last resort. |
| 2 | Single font-reduction step | 1 step (×0.82), not iterative | Iterative shrinking produces unreadable 6pt labels. One step is enough to resolve marginal collisions; severe collisions need rotation, not tiny fonts. |
| 3 | Wrap split strategy | underscore > space > camelCase | Matches the most common real-world label formats: `snake_case` variables, natural language, `camelCase` identifiers. Only one strategy fires per label to avoid over-splitting. |
| 4 | Wrapping only at 0° | Multi-line labels are never rotated | Rotated multi-line text is unreadable. If wrapping doesn't resolve collision at 0°, proceed to rotation of the original (unwrapped) single-line label. |
| 5 | Bounded two-pass margin | Estimate angle from worst-case, then layout | Preserves single-pass plot rect guarantee. Over-reservation is acceptable (slightly more bottom padding than needed); under-reservation causes clipping. |
| 6 | Cull threshold default = 8, theme-configurable | `Theme(cull_threshold=N)` | Culling a 5-label axis to 3 labels loses too much information. Default 8 is sane for most charts; power users with dense heatmaps can lower it. |
| 7 | Y-axis suppression mirrors x-axis | Leftmost column only | Exactly parallels the existing bottom-row-only x-axis title behavior. Consistent visual result. |
| 8 | `\n` in label string vs. `Vec<String>` | `\n`-joined string | Minimizes struct changes. Renderers already need to handle text content — splitting on `\n` is trivial. Serde round-trip is cleaner (single string vs. vec). |

**Rejected alternatives:**

- **Fixed-point layout (re-pass after discovering angle):** Violates the single-pass commitment from Phase 6 spec. The bounded two-pass (estimate-then-finalize) achieves the same result without iteration.
- **Smart tick selection (show "important" labels, hide others):** Requires domain knowledge the layout engine doesn't have (which categories matter). Uniform stride culling is predictable.
- **Automatic viewport resizing:** Out of scope — viewport is a caller-provided constraint, not a layout output.

---

## 9. Acceptance criteria

1. A 600×400 chart with 9 snake_case categories (`trivial_baseline`, `negative_prompt`, `persona_constrained`, `minimal_context`, `none`, `generic_coder`, `real_agent_config`, `python_coder`, `long_directive`) renders all labels fully legible (no elision) at default settings.
2. A faceted chart with 2+ columns renders exactly one y-axis title on the leftmost column — no duplication or overlap.
3. A chart with rotated labels (-45° or -90°) does not clip labels at the bottom viewport boundary.
4. `cargo test -p ferrum-core` passes with new tests covering each cascade stage (wrap, shrink, rotate, cull, elide) and the y-axis suppression fix.
5. Existing golden SVGs that don't trigger collision remain byte-identical.
6. The `LabelsElided` warning fires only when elision (S5) is actually used, not for wrapping or rotation.

---

## 10. Validation strategy

- **Unit tests (Rust):** One `MockMetrics`-based test per cascade stage, verifying that the correct stage fires for each density regime. Parameterized test sweeping label count from 4 to 40 in a 600px panel to verify graceful degradation through the cascade.
- **Integration tests (Python):** Render a faceted bar chart with 9 snake_case categories at 600×400 and verify no `LabelsElided` warning. Render a faceted chart and verify only one y-axis title node in the scene graph.
- **Visual regression:** Regenerate affected golden SVGs, rasterize with `snapshot-goldens.py`, and visually confirm labels are readable, titles are not duplicated, and rotated labels are not clipped.

---

## 11. Open questions

None — all resolved.

### Resolved

1. **Wrap line count:** Unlimited wrapping (4+ lines allowed). Labels like `very_long_snake_case_name` split at every `_` boundary. The vertical extent (`n_lines × line_height`) is accounted for in the bottom margin reservation.
2. **Cull threshold:** Theme-configurable via `Theme(cull_threshold=N)`. Exposed as `ThemeInputs.cull_threshold: u32` on the Rust side, default 8. Power users with dense heatmaps can lower it to trigger culling earlier.
