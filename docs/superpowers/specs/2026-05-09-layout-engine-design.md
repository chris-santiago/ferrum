# Phase 6 — Layout Engine: Design Spec

**Date:** 2026-05-09
**Phase:** 6 (Layout Engine)
**Phase slug:** `layout-engine`
**Depends on:** Phase 3 (ChartSpec IR), Phase 4 (Scale engine — tick generation, ordinal padding), Phase 5 (Stat engine — provides post-transform data the caller derives `facet_groups` from)
**Unblocks:** Phase 7 (Static renderer)

---

## §1 Goal

Compute pixel rectangles for every visible region of a chart — panels, axes, legend — from a `ChartSpec`, a `Theme`, and a `Viewport`. The output is a pure-data `LayoutResult` consumed by Phase 7's renderer. No I/O, no rendering, no data values touched.

---

## §2 Scope

### In scope (the binding done-criteria contract)
- Single-chart layout with linear and ordinal axes (Phase 4 surface).
- **Axis titles** — the `Axis.title` field in `ferrum-spec.md §3.7` ("x label" / "y label"). Reserves layout space.
- Legend placement on all four orients: `right`, `left`, `top`, `bottom`.
- Facet **wrap mode** (`ncols` set, rows derived) and **grid mode** (`ncols × nrows` explicit).
- Axis label collision avoidance: rotate to a configurable angle (default `-45°`), then elide with ellipsis if still colliding.
- Facet scale resolution: **shared by default** for `x`, `y`, `color` (matches Vega-Lite convention).

### Out of scope (deferred, named here so future sessions don't re-litigate)
- **Chart titles, panel titles, facet-strip titles.** *Note: facet-strip titles (the per-panel header showing the facet value) are needed for readable faceted output in Phase 7. This is a known Phase 7 follow-up, not an oversight.*
- HConcat / VConcat / Repeat / multi-layer Resolve.
- Polar / Geo coordinates (`§3.8`).
- Container-relative sizing (`width="container"`).
- Multi-layer scale resolution (one mark per chart for now).
- Independent-by-channel `Resolve` overrides.

---

## §3 Architecture

### 3.1 Public surface

```rust
pub fn compute_layout(
    spec: &ChartSpec,
    theme: &Theme,
    viewport: Viewport,
    facet_groups: &[FacetGroup],
    legend_entries: &[LegendEntry],
    metrics: &dyn TextMetrics,
) -> Result<LayoutResult, LayoutError>;
```

A pure function. Same inputs ⇒ same outputs. No global state, no I/O.

### 3.2 Module layout

```
crates/ferrum-core/src/layout/
  mod.rs           // pub fn compute_layout, LayoutResult, LayoutError, LayoutWarning
  panel.rs         // PanelLayout
  axis.rs          // AxisLayout, TickLayout, AxisOrient, label collision policy
  legend.rs        // LegendLayout, LegendEntryLayout, LegendOrient, LegendDirection
  facet.rs         // FacetGrid, FacetMode (Wrap | Grid), cell_rect arithmetic
  text_metrics.rs  // TextMetrics trait, HeuristicMetrics, MockMetrics (test-only)
  geometry.rs      // Rect, Inset, Viewport
```

### 3.3 ChartSpec extension (back-compat)

Phase 6 adds **one** optional field to `ChartSpec`, mirroring how Phase 5 added `transforms`:

```rust
pub struct ChartSpec {
    // ... existing fields ...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facet: Option<FacetSpec>,
}

pub struct FacetSpec {
    pub field: String,
    pub mode: FacetMode,        // Wrap { ncols } | Grid { nrows, ncols }
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spacing: Option<f64>,   // overrides theme.column_padding/row_padding if set
}
```

Existing JSON outputs stay byte-identical (omitted field round-trips as `None`).

`Theme` and `Viewport` are **not** added to `ChartSpec` — themes are values per CLAUDE.md, and viewport is a render-time concern.

---

## §4 Per-component contracts

### 4.1 `geometry.rs`

```rust
pub struct Rect { pub x: f64, pub y: f64, pub w: f64, pub h: f64 }

impl Rect {
    pub const ZERO: Rect = Rect { x: 0.0, y: 0.0, w: 0.0, h: 0.0 };
    pub fn shrink(&self, inset: Inset) -> Rect;       // → Rect::ZERO if collapsed
    pub fn split_top(&self, h: f64) -> (Rect, Rect);
    pub fn split_bottom(&self, h: f64) -> (Rect, Rect);
    pub fn split_left(&self, w: f64) -> (Rect, Rect);
    pub fn split_right(&self, w: f64) -> (Rect, Rect);
}

pub struct Inset { pub top: f64, pub right: f64, pub bottom: f64, pub left: f64 }
pub struct Viewport { pub width: f64, pub height: f64 }
```

### 4.2 `text_metrics.rs`

```rust
pub trait TextMetrics {
    fn measure_width(&self, text: &str, font_size: f64) -> f64;
    fn line_height(&self, font_size: f64) -> f64 {
        font_size * 1.2  // default impl
    }
}

pub struct HeuristicMetrics { pub k: f64 }              // K = 0.6 default
impl Default for HeuristicMetrics { fn default() -> Self { Self { k: 0.6 } } }

impl TextMetrics for HeuristicMetrics {
    fn measure_width(&self, text: &str, font_size: f64) -> f64 {
        text.chars().count() as f64 * font_size * self.k
    }
}

#[cfg(test)]
pub struct MockMetrics { ... }   // user-supplied closure, used in cargo tests
```

Phase 7 will add `FontdueMetrics` in the renderer crate. Phase 6 has no `fontdue` dependency and ships no font assets.

### 4.3 `panel.rs`

```rust
pub struct PanelLayout {
    pub plot_area: Rect,
    pub facet_key: Option<FacetKey>,   // None for non-faceted single chart
    pub row: u32,                      // 0-indexed; (0,0) for single chart
    pub col: u32,
}
pub struct FacetKey { pub field: String, pub value: String }
```

### 4.4 `axis.rs`

```rust
pub enum AxisOrient { Top, Bottom, Left, Right }

pub struct AxisLayout {
    pub orient: AxisOrient,
    pub panel_index: usize,
    pub axis_line: Rect,
    pub ticks: Vec<TickLayout>,
    pub title: Option<AxisTitleLayout>,
}

pub struct TickLayout {
    pub position: f64,        // pixel along the axis (absolute, in viewport coords)
    pub label: String,        // already-elided if elision fired
    pub label_angle: f64,     // degrees; 0 = horizontal
    pub elided: bool,         // diagnostic flag
}

pub struct AxisTitleLayout {
    pub text: String,
    pub anchor: (f64, f64),   // pixel coords of text anchor
    pub angle: f64,           // 0 for x-axis title, -90 for y-axis title
}
```

### 4.5 `legend.rs`

```rust
pub enum LegendOrient { Right, Left, Top, Bottom }
pub enum LegendDirection { Vertical, Horizontal }

pub struct LegendLayout {
    pub rect: Rect,
    pub orient: LegendOrient,
    pub direction: LegendDirection,
    pub entries: Vec<LegendEntryLayout>,
}

pub struct LegendEntryLayout {
    pub label: String,
    pub label_anchor: (f64, f64),
    pub symbol_anchor: (f64, f64),
    pub symbol_kind: SymbolKind,    // Square | Circle | Line — passed in by caller
}
```

### 4.6 `facet.rs`

```rust
pub enum FacetMode {
    Wrap { ncols: u32 },                  // rows derived
    Grid { nrows: u32, ncols: u32 },      // both explicit
}

pub struct FacetGrid {
    pub mode: FacetMode,
    pub n_panels: u32,
    pub cell_w: f64,
    pub cell_h: f64,
    pub gutter_x: f64,                    // theme.column_padding (default 8.0)
    pub gutter_y: f64,                    // theme.row_padding    (default 8.0)
}

impl FacetGrid {
    pub fn compute(spec: &FacetSpec, n_panels: u32, plot_region: Rect, theme: &Theme) -> Self;
    pub fn cell_rect(&self, row: u32, col: u32, origin: Rect) -> Rect;
}
```

For wrap mode: `nrows = ceil(n_panels / ncols)`. For grid mode: panels beyond `nrows * ncols` are dropped with a `LayoutWarning::PanelsDropped` (this is *not* in the Phase 6 done criteria but is the obvious sane behavior).

### 4.7 `mod.rs` — top-level result

```rust
pub struct LayoutResult {
    pub viewport: Rect,
    pub panels: Vec<PanelLayout>,
    pub axes: Vec<AxisLayout>,         // bottom + left per panel; top + right only if spec opts in
    pub legend: Option<LegendLayout>,
    pub warnings: Vec<LayoutWarning>,
}

pub enum LayoutError {
    InvalidViewport { width: f64, height: f64 },
    InvalidFacetSpec(String),
    PaddingExceedsViewport { padding: f64, viewport_dim: f64 },
    EmptyFacetGroups,
}

pub enum LayoutWarning {
    PanelCollapsed { panel_index: usize },
    LabelsElided { axis: usize, count: u32 },
    LegendOverflowed { entries_dropped: u32 },
    PanelsDropped { count: u32 },        // grid mode, n_panels > nrows*ncols
}
```

`LayoutResult`, `LayoutWarning`, all sub-types: `#[derive(Serialize, Deserialize)]` for JSON round-trip and Python dict conversion.

---

## §5 Input contract

```rust
pub fn compute_layout(
    spec: &ChartSpec,                       // post-transform; carries facet field if faceted
    theme: &Theme,                          // value, not global
    viewport: Viewport,                     // pixel canvas
    facet_groups: &[FacetGroup],            // caller-supplied; empty for non-faceted
    legend_entries: &[LegendEntry],         // caller-supplied; empty if no legend
    metrics: &dyn TextMetrics,              // HeuristicMetrics in Phase 6
) -> Result<LayoutResult, LayoutError>;

pub struct FacetGroup { pub key: FacetKey, pub n_rows: u64 }
pub struct LegendEntry { pub label: String, pub symbol: SymbolKind }
```

**Why each input is structured this way:**
- `facet_groups` is caller-pre-computed from the post-transform dataset. Phase 6 stays data-blind.
- `legend_entries` is caller-pre-computed. Without this, "legend doesn't overlap" is theatrical (we'd reserve a fixed default size).
- Tick generation is delegated to Phase 4 scales, called *inside* `compute_layout` against a provisional pixel range. Single-pass — no fixed-point loop.
- `OrdinalScale::range_band()` is the API Phase 6 calls for ordinal padding; Phase 6 does **not** re-implement padding semantics.

### 5.1 Theme fields actually read

Phase 6 reads only layout-affecting theme fields:

| Theme field | Default | Use |
|---|---|---|
| `padding` | `8.0` (all sides) | Outer chart padding |
| `column_padding` | `8.0` | Facet horizontal gutter |
| `row_padding` | `8.0` | Facet vertical gutter |
| `axis_title_padding` | `4.0` | Gap between axis line and axis title |
| `label_font_size` | `11.0` | Tick label measurement |
| `title_font_size` | `13.0` | Axis title measurement |
| `legend_orient` | `Right` | Default if `Legend.orient` unset |

All other theme fields are ignored by Phase 6 (Phase 7 reads them).

---

## §6 Algorithm — single-pass arithmetic

```
1. Validate inputs:
     viewport.w > 0  ∧  viewport.h > 0
     for facet:  ncols > 0  (and nrows > 0 in grid mode)  ∧  n_panels >= 1
   → on failure: return LayoutError

2. inner = viewport.shrink(theme.padding)
   if inner collapsed: return LayoutError::PaddingExceedsViewport

3. legend_rect_size = estimate_legend_size(legend_entries, orient, metrics, theme)
   (inner_after_legend, legend_rect) = inner.split_<orient>(legend_rect_size)

4. left_gutter   = title_height_for(spec.y_axis, metrics) + theme.axis_title_padding
   bottom_gutter = title_height_for(spec.x_axis, metrics) + theme.axis_title_padding
   plot_region   = inner_after_legend.shrink(Inset {
                       left: left_gutter,
                       bottom: bottom_gutter,
                       ..Default::default()
                   })

5. // Reserve worst-case axis-label band — must NOT depend on the (still-unknown) final
   //   plot rect, so use a domain-endpoint upper bound instead of actually generated ticks:
   //   - Linear/log/time scale: format(domain.min) and format(domain.max), take wider.
   //   - Ordinal scale: iterate spec-known categories, take longest.
   //   This is a true upper bound: every intermediate tick label has ≤ chars of the wider endpoint.
   max_y_label_width = estimate_max_y_label_width(spec.y_axis, theme, metrics)
   plot_region = plot_region.shrink(Inset {
                     left: max_y_label_width,
                     bottom: metrics.line_height(theme.label_font_size),
                     ..Default::default()
                 })

6. grid    = FacetGrid::compute(spec.facet, n_panels, plot_region, theme)
   panels  = grid.cells()                       // Vec<PanelLayout>

7. for each panel:
     a. ticks = scale.generate_ticks(panel.plot_area_pixel_range)
     b. for each tick:  label_w = metrics.measure_width(tick.label, theme.label_font_size)
     c. slot_width = panel.w / max(1, ticks.len())
        if any label_w > slot_width * (1 - LABEL_OVERLAP_TOLERANCE):
            for each tick: tick.label_angle = spec.x_axis.label_angle.unwrap_or(-45.0)
            // Re-check after rotation. For a label of length L rotated by θ from horizontal,
            //   anchored at its top-right (or top-left for positive θ) at the tick mark, the
            //   horizontal projection = L * cos(|θ|). Adjacent labels collide if this exceeds
            //   slot_width; we ignore the small label-height contribution since label_h << label_w.
            rotated_w = label_w * cos(|tick.label_angle|)
            if any rotated_w > slot_width:
                elide each label by binary search on prefix length until measured prefix
                  fits slot_width / cos(|angle|); append "…" to result
                tick.elided = true
                emit LayoutWarning::LabelsElided
     d. emit AxisLayout for bottom + left (default); top/right only if spec opts in

8. if !legend_entries.is_empty():
     lay out entries in `direction` (Vertical for Right/Left, Horizontal for Top/Bottom)
     if total entry size > legend_rect:
         drop overflow entries
         emit LayoutWarning::LegendOverflowed

9. for each panel:
     if panel.plot_area.w <= MIN_PANEL_DIM ∨ panel.plot_area.h <= MIN_PANEL_DIM:
         panel.plot_area = Rect::ZERO
         emit LayoutWarning::PanelCollapsed

10. return LayoutResult { viewport: viewport.into_rect(), panels, axes, legend, warnings }
```

**Single-pass commitment:** worst-case label-band reservation in step 5 means the plot rect from step 6 is final. Step 7c only mutates label angle/text, not the plot rect. No fixed-point loop.

### 6.1 Constants

| Constant | Value | Purpose |
|---|---|---|
| `LABEL_OVERLAP_TOLERANCE` | `0.10` | 10% slack before rotation kicks in |
| `DEFAULT_LABEL_ANGLE` | `-45.0` | Rotation when collision fires |
| `DEFAULT_HEURISTIC_K` | `0.6` | `text_chars × font_size × K` |
| `MIN_PANEL_DIM` | `1.0` | Below this, panel clamps to `Rect::ZERO` |

These live as `pub const` in `mod.rs`. They are not configurable from the public API in Phase 6; Phase 8 (grammar API) may surface them via theme.

---

## §7 Error policy (hybrid, mirrors Phase 5 §6)

| Class | Trigger | Response |
|---|---|---|
| **Structural** | Negative viewport, malformed `FacetSpec`, padding > viewport | `LayoutError` → `PyValueError` |
| **Geometric edge** | Panel collapses to ≤ 1px, label too wide even after rotation, legend overflow | Clamp to sensible value (`Rect::ZERO`, ellipsis-elided text, dropped entries), emit a `LayoutWarning` in `LayoutResult.warnings` |
| **Silent** | Empty legend (no entries) | No legend in output, no warning — this is normal |

Phase 7 / Phase 8 will route warnings through Python's `warnings.warn`; Phase 6 only puts them in the result.

---

## §8 New external dependencies

**None.**

Each crate considered and rejected, with reason:

| Crate | Decision | Reason |
|---|---|---|
| `cassowary` | Rejected | Constraint solver for *bidirectional* constraints. Chart layout is a one-way pipeline (legend → axes → panels). ggplot/Vega-Lite/matplotlib don't use one. |
| `taffy` | Rejected | Flexbox engine. Chart layout isn't flex (which is "fit + grow remaining"); reserved-strips model is different. |
| `nalgebra` | Rejected (probably forever) | Phase 11 will want matrix math, but `glam` is the standard graphics-rust choice. Phase 7 needs only a 12-line `Affine2`. |
| `fontdue` | Deferred to Phase 7 | Phase 7's renderer rasterizes glyphs and needs it; Phase 6 only measures widths. The `TextMetrics` trait is the seam. Pulling it forward would add font I/O to a pure-function module and ~100x measurement cost per layout pass. |
| `rustybuzz` | Deferred to Phase 7 | Text shaping is a renderer concern, not a layout one. |

The `TextMetrics` trait + `MockMetrics` in tests means the heuristic's accuracy is **not on the test path** — assertions use exact mock widths, so swapping in `FontdueMetrics` in Phase 7 doesn't break tests.

---

## §9 Test plan

### 9.1 Cargo tests (target ≥ 24 new tests; cargo total ≥ 145)

**`geometry.rs`**
- `Rect::shrink` with normal inset returns expected smaller rect.
- `Rect::shrink` with `inset > rect dim` returns `Rect::ZERO`.
- `split_*` partition is exact (no pixel gap, no overlap).

**`facet.rs` — pinned arithmetic**

| Test | Inputs | Expected |
|---|---|---|
| Wrap, exact fit | viewport 600×400, ncols=3, n_panels=3, gutter=0 | 3 cells × 200×400, rows=1 |
| Wrap, ragged | viewport 600×400, ncols=3, n_panels=5, gutter=0 | rows=2; cells 0–4 at `(0,0),(200,0),(400,0),(0,200),(200,200)`; slot `(400,200)` absent from `panels` |
| Wrap with gutters | viewport 620×420, ncols=3, n_panels=5, gutter_x=10, gutter_y=10 | `cell_w = (620 − 2·10)/3 = 200`; `cell_h = (420 − 1·10)/2 = 205` |
| Grid mode | viewport 600×400, nrows=2, ncols=3, n_panels=6 | identical to 3×2 wrap, all six slots filled |
| Grid drop | nrows=2, ncols=3, n_panels=8 | 6 panels emitted, `LayoutWarning::PanelsDropped { count: 2 }` |
| Single chart | no facet, viewport 600×400, padding=8 | one panel, `plot_area` = inner shrunk by axis gutters |
| Degenerate viewport | viewport 10×10, padding=8 → inner 0×0 | `LayoutError::PaddingExceedsViewport` |
| Empty groups | facet specified, n_panels=0 | `LayoutError::EmptyFacetGroups` |

**`axis.rs` — collision policy (with `MockMetrics`)**

| Test | Setup | Expected |
|---|---|---|
| No collision | OrdinalScale 4 cats, panel_w=400, mock=50px each | all `label_angle == 0`, no elision |
| Rotates | 8 cats, panel_w=400, mock=80px each (slot=50) | all `label_angle == -45.0`, no elision |
| Rotates + elides | 20 cats, panel_w=200, mock=80px each | `label_angle == -45.0`, all `elided == true`, `LabelsElided` warning |
| Custom angle | `spec.x_axis.label_angle = -90.0` | rotation uses `-90.0`, not `-45.0` |
| Y-axis label width | linear 0–100, mock measures `"100"` at 30px | left axis-title gutter ≥ 30 + `axis_title_padding` |

**`legend.rs`**

| Test | Setup | Expected |
|---|---|---|
| Right fits | 3 entries, viewport 600×400, longest 60px | `legend.rect` on right, plot rect width reduced, no overlap |
| Bottom horizontal | 5 entries, longest 40px, viewport 600×400 | `legend.rect` spans bottom, `direction = Horizontal` |
| Overflow | 100 entries in 600×400 with right legend | first N laid out, remaining dropped, `LegendOverflowed` warning |
| All four orients (parameterized) | right, left, top, bottom | plot rect shrinks on the correct side; `legend.rect` on correct side; no overlap |

**`text_metrics.rs`**
- `HeuristicMetrics { k: 0.6 }.measure_width("hello", 12.0) == 36.0` (5 × 12 × 0.6).
- `line_height(12.0) == 14.4`.

**`mod.rs` — end-to-end**
- `compute_layout` on minimal scatter spec → 1 panel, 2 axes (bottom + left), 0 warnings.
- `compute_layout` on faceted (3-panel wrap, 1 color legend, 5 entries) → 3 panels, 6 axes, 1 legend, 0 warnings.
- `LayoutResult` JSON serde round-trip is byte-identical.

### 9.2 Python pytest (target ≥ 6 new tests; pytest total ≥ 78)

`tests/test_layout_engine.py` — validates the binding surface, not arithmetic:
- `from ferrum._core import compute_layout` imports successfully.
- Returns a dict with keys `viewport`, `panels`, `axes`, `legend`, `warnings`.
- Faceted spec: `len(result["panels"]) == n_panels`.
- Invalid viewport raises `ValueError`.
- `_core.pyi` declares `compute_layout(spec, theme, viewport, facet_groups, legend_entries) -> dict`.

### 9.3 Test count baseline at end of phase
- `cargo test -p ferrum-core`: ≥ 145 (currently 121; +~24 layout)
- `uv run pytest`: ≥ 78 (currently 72; +~6 binding)

---

## §10 Done-criteria gate

From `ferrum-phases.md` Phase 6 done criteria:

- [ ] **Facet grid sizes computed correctly for `wrap` and `grid` facet modes** → covered by §9.1 `facet.rs` arithmetic table.
- [ ] **Legend placement does not overlap chart area for a 1-layer scatter plot** → covered by §9.1 `legend.rs` four-orient parameterized test.
- [ ] **Axis label collision avoidance fires at a configurable threshold** → covered by §9.1 `axis.rs` rotation/elision tests with `LABEL_OVERLAP_TOLERANCE` and `spec.x_axis.label_angle` knobs.
- [ ] **`cargo test` covers basic facet layout arithmetic** → 24 new tests, all numerical reference values pinned in §9.1.

A phase-6-done PR must show all four boxes ticked, `cargo test -p ferrum-core` ≥ 145 passing, `uv run pytest` ≥ 78 passing.

---

## §11 Locked decisions table

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Scope | Done-criteria-only | HConcat/VConcat/Repeat/polar/geo/multi-layer-resolve/titles all deferred. |
| 2 | LayoutResult IR shape | Flat pixel rects per region | Renderer consumes raw rects; no transform math, no recursive tree. |
| 3 | Text-width strategy | Heuristic + `TextMetrics` trait | `K=0.6` × char count × font size; trait is the seam Phase 7 plugs `FontdueMetrics` into. |
| 4 | ChartSpec extension | `+ facet: Option<FacetSpec>` only | Theme + viewport stay separate. Mirrors Phase 5 transforms back-compat. |
| 5 | Internal organization | Per-element structs + free fn | No sealed enum (regions compose, not alternatives). Trait dispatch only for `TextMetrics`. |
| 6 | Error policy | Hybrid: structural → `LayoutError`, geometric → clamp + warning | Mirrors Phase 5 §6. |
| 7 | Python API surface | Expose now via `ferrum._core.compute_layout` returning dict | Matches Phase 4/5 cadence; pytest validates binding. |
| 8 | Legend orientation support | All four (right/left/top/bottom) | Same arithmetic, just rotated. `economist` theme needs left support. |
| 9 | Label collision policy | Rotate first, then elide | Configurable angle + threshold. Convention in every charting lib. |
| 10 | Facet scale resolution default | Shared (x, y, color) | Comparison is the point of faceting. Override knob deferred to Phase 8. |
| 11 | New crates | None — heuristic + trait | cassowary/taffy permanently rejected; fontdue deferred to Phase 7. |
| 12 | Tick generation | Phase 6 calls Phase 4 scales internally | Single-pass: provisional pixel range → ticks → label widths → no re-pass. |
| 13 | Iteration | Single pass, no fixed-point | Worst-case label-band reservation in step 5 makes plot rect final. |

---

## §12 Cross-phase notes

**Phase 4 (Scale engine) APIs Phase 6 calls:**
- `LinearScale::generate_ticks(pixel_range, tick_count_hint)` and equivalent for ordinal.
- `OrdinalScale::range_band()` for ordinal padding (Phase 6 does **not** re-implement padding semantics).

**Phase 5 (Stat engine) — what flows through:**
- Stat transforms can change row count (`stat_aggregate` produces fewer rows). Phase 6 sees the **post-transform schema** via `facet_groups` (caller-derived), not the input schema. Phase 6 doesn't see data values.

**Phase 7 (Static renderer) consumes:**
- `LayoutResult` directly as Rust types (zero-copy from in-process Phase 6).
- Phase 7 will replace `HeuristicMetrics` with `FontdueMetrics` in renderer-side callers; Phase 6 internal API is unchanged.
- Phase 7 will need to add **facet-strip titles** (per-panel header showing facet value) for readable faceted output. This is a known follow-up; Phase 6 deliberately does not block on it.

**Phase 8 (Grammar API) — future surface:**
- Will likely surface `LABEL_OVERLAP_TOLERANCE` / `DEFAULT_LABEL_ANGLE` as theme knobs.
- Will add per-channel `Resolve` overrides on top of the shared-by-default Phase 6 behavior.
- May add chart-level `title` / `subtitle` / facet-strip title support, which will need to extend `LayoutResult` (additive, back-compat via `serde(default)`).

**Phase 11 (Interactive renderer):**
- Reuses `LayoutResult`. Will likely add an `Affine2` per panel for zoom/pan, but as a *renderer-side* concern (matrix transforms applied during draw), not a layout concern.

---

## §13 Test count baseline at HEAD (before Phase 6 work)

- `cargo test -p ferrum-core`: 121 passing
- `uv run pytest`: 72 passing
