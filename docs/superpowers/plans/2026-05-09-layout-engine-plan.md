# Phase 6 — Layout Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Phase 6 Rust layout engine — a pure function `compute_layout(spec, theme, viewport, axes, facet_groups, legend_entries, &dyn TextMetrics) -> Result<LayoutResult, LayoutError>` that produces pixel rectangles for panels, axes, and legend. No I/O, no rendering, no data values, no font crate. Exposed to Python via `ferrum._core.compute_layout` returning a dict.

**Architecture:** Per-element structs in their own modules (`geometry`, `text_metrics`, `panel`, `axis`, `legend`, `facet`) orchestrated by a free function in `layout/mod.rs`. ChartSpec gains exactly one optional field (`facet`). Tick labels are caller-pre-computed (Phase 6 stays data-blind). Hybrid error policy: structural failures → `LayoutError` (mapped to `PyValueError`); geometric edges → clamp + `LayoutWarning` in result. No new external crates.

**Tech Stack:** Rust 2021 (PyO3 0.28, abi3-py310, serde, serde_json — all already in workspace deps). No new crates.

**Layout adaptation from spec:** Per §14 spec refinements (commit `dfc0ba1`), tick labels are caller-provided rather than generated inside `compute_layout`. Collision policy fires on the x-axis only. Tests live inline per the project convention (`crate-type = ["cdylib"]`, no integration tests directory).

**Spec reference:** `docs/superpowers/specs/2026-05-09-layout-engine-design.md` (committed `635d6c5`, refined `dfc0ba1`).

**Branch:** `feat/phase-6-layout-engine` (already checked out from `main`).

---

## File map

### New files

| Path | Responsibility |
|---|---|
| `crates/ferrum-core/src/layout/mod.rs` | Module decls, `compute_layout`, `LayoutResult`, `LayoutError`, `LayoutWarning`, top-level constants |
| `crates/ferrum-core/src/layout/geometry.rs` | `Rect`, `Inset`, `Viewport`, `shrink`/`split_*` arithmetic |
| `crates/ferrum-core/src/layout/text_metrics.rs` | `TextMetrics` trait, `HeuristicMetrics`, `MockMetrics` |
| `crates/ferrum-core/src/layout/panel.rs` | `PanelLayout`, `FacetKey` |
| `crates/ferrum-core/src/layout/facet.rs` | `FacetSpec`, `FacetMode`, `FacetGroup`, `FacetGrid`, cell-rect arithmetic |
| `crates/ferrum-core/src/layout/axis.rs` | `AxisOrient`, `AxisInput`, `AxesInput`, `AxisLayout`, `TickLayout`, `AxisTitleLayout`, x/y layout fns, collision policy |
| `crates/ferrum-core/src/layout/legend.rs` | `LegendOrient`, `LegendDirection`, `LegendEntry`, `LegendLayout`, `LegendEntryLayout`, `SymbolKind`, layout fn |
| `crates/ferrum-core/src/layout/binding.rs` | PyO3 binding: `compute_layout` Python function returning `dict` |
| `tests/test_layout_engine.py` | Python smoke tests for the binding surface |

### Modified files

| Path | Change |
|---|---|
| `crates/ferrum-core/src/lib.rs` | `mod layout;` + register `compute_layout` pyfunction |
| `crates/ferrum-core/src/spec/chart.rs` | Add `facet: Option<FacetSpec>` field with `serde(default, skip_serializing_if = "Option::is_none")` |
| `src/ferrum/_core.pyi` | Add `compute_layout` signature stub |
| `src/ferrum/__init__.py` | Re-export `compute_layout` |
| `docs/superpowers/ferrum-phases.md` | Phase 6 status `pending` → `done`; link spec doc |

### Constants table (from spec §6.1, lives in `mod.rs`)

| Constant | Value | Purpose |
|---|---|---|
| `LABEL_OVERLAP_TOLERANCE` | `0.10` | 10% slack before rotation kicks in |
| `DEFAULT_LABEL_ANGLE` | `-45.0` | Rotation when collision fires |
| `DEFAULT_HEURISTIC_K` | `0.6` | Char-width × font-size multiplier |
| `MIN_PANEL_DIM` | `1.0` | Below this, panel clamps to `Rect::ZERO` |
| `DEFAULT_PADDING` | `8.0` | Outer chart padding when theme.padding is None |
| `DEFAULT_LABEL_FONT_SIZE` | `11.0` | Tick label font size |
| `DEFAULT_TITLE_FONT_SIZE` | `13.0` | Axis title font size |
| `DEFAULT_AXIS_TITLE_PADDING` | `4.0` | Gap between axis line and axis title |

---

## Task list

### Task 1: Empty `layout/` module skeleton

**Files:**
- Create: `crates/ferrum-core/src/layout/mod.rs`
- Create: `crates/ferrum-core/src/layout/geometry.rs`
- Create: `crates/ferrum-core/src/layout/text_metrics.rs`
- Create: `crates/ferrum-core/src/layout/panel.rs`
- Create: `crates/ferrum-core/src/layout/facet.rs`
- Create: `crates/ferrum-core/src/layout/axis.rs`
- Create: `crates/ferrum-core/src/layout/legend.rs`
- Create: `crates/ferrum-core/src/layout/binding.rs`
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Create `layout/mod.rs`**

```rust
//! Phase 6 — layout engine. Pure function: ChartSpec + Theme + Viewport ->
//! pixel rectangles for panels, axes, legend. No I/O, no rendering, no data
//! values touched. See docs/superpowers/specs/2026-05-09-layout-engine-design.md.

pub(crate) mod geometry;
pub(crate) mod text_metrics;
pub(crate) mod panel;
pub(crate) mod facet;
pub(crate) mod axis;
pub(crate) mod legend;
pub(crate) mod binding;
```

- [ ] **Step 2: Create empty stub files**

For each of `geometry.rs`, `text_metrics.rs`, `panel.rs`, `facet.rs`, `axis.rs`, `legend.rs`, `binding.rs`, write:

```rust
//! Placeholder — implementation lands in subsequent tasks.
```

- [ ] **Step 3: Register module in `lib.rs`**

Edit `crates/ferrum-core/src/lib.rs`. After the `pub(crate) mod transform;` line, add:

```rust
pub(crate) mod layout;
```

- [ ] **Step 4: Verify build**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

Expected: build succeeds. No new functionality, no new pyclass registration yet.

- [ ] **Step 5: Commit**

```bash
git add crates/ferrum-core/src/layout crates/ferrum-core/src/lib.rs
git commit -m "feat(layout): scaffold layout/ module skeleton

Empty stubs for geometry, text_metrics, panel, facet, axis, legend, binding.
Registered in lib.rs. No public surface yet."
```

---

### Task 2: `geometry.rs` — `Rect`, `Inset`, `Viewport`

**Files:**
- Modify: `crates/ferrum-core/src/layout/geometry.rs`

- [ ] **Step 1: Write failing tests**

Replace `geometry.rs` placeholder content with:

```rust
//! Pixel-space geometry primitives. Coordinates are f64; positive-y is downward
//! (consistent with SVG/screen conventions). All `shrink`/`split_*` operations
//! return new values; `Rect` and `Inset` are `Copy`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    pub const ZERO: Rect = Rect { x: 0.0, y: 0.0, w: 0.0, h: 0.0 };

    /// Shrink by an inset on each side. Returns `Rect::ZERO` if the inset
    /// would collapse either dimension to ≤ 0.
    pub fn shrink(&self, inset: Inset) -> Rect {
        let w = self.w - inset.left - inset.right;
        let h = self.h - inset.top - inset.bottom;
        if w <= 0.0 || h <= 0.0 {
            return Rect::ZERO;
        }
        Rect { x: self.x + inset.left, y: self.y + inset.top, w, h }
    }

    /// Split off a strip of height `h` from the top. Returns `(strip, remainder)`.
    /// If `h >= self.h`, strip == self and remainder == ZERO.
    pub fn split_top(&self, h: f64) -> (Rect, Rect) {
        let h = h.min(self.h).max(0.0);
        let strip = Rect { x: self.x, y: self.y, w: self.w, h };
        let remainder = Rect {
            x: self.x,
            y: self.y + h,
            w: self.w,
            h: self.h - h,
        };
        (strip, remainder)
    }

    pub fn split_bottom(&self, h: f64) -> (Rect, Rect) {
        let h = h.min(self.h).max(0.0);
        let remainder = Rect { x: self.x, y: self.y, w: self.w, h: self.h - h };
        let strip = Rect {
            x: self.x,
            y: self.y + self.h - h,
            w: self.w,
            h,
        };
        (strip, remainder)
    }

    pub fn split_left(&self, w: f64) -> (Rect, Rect) {
        let w = w.min(self.w).max(0.0);
        let strip = Rect { x: self.x, y: self.y, w, h: self.h };
        let remainder = Rect {
            x: self.x + w,
            y: self.y,
            w: self.w - w,
            h: self.h,
        };
        (strip, remainder)
    }

    pub fn split_right(&self, w: f64) -> (Rect, Rect) {
        let w = w.min(self.w).max(0.0);
        let remainder = Rect { x: self.x, y: self.y, w: self.w - w, h: self.h };
        let strip = Rect {
            x: self.x + self.w - w,
            y: self.y,
            w,
            h: self.h,
        };
        (strip, remainder)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Inset {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl Inset {
    pub const fn uniform(v: f64) -> Inset {
        Inset { top: v, right: v, bottom: v, left: v }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub width: f64,
    pub height: f64,
}

impl Viewport {
    pub fn into_rect(self) -> Rect {
        Rect { x: 0.0, y: 0.0, w: self.width, h: self.height }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(x: f64, y: f64, w: f64, h: f64) -> Rect {
        Rect { x, y, w, h }
    }

    #[test]
    fn rect_shrink_normal() {
        let r0 = r(0.0, 0.0, 100.0, 50.0);
        let r1 = r0.shrink(Inset::uniform(5.0));
        assert_eq!(r1, r(5.0, 5.0, 90.0, 40.0));
    }

    #[test]
    fn rect_shrink_collapses_to_zero() {
        let r0 = r(0.0, 0.0, 10.0, 10.0);
        let r1 = r0.shrink(Inset::uniform(10.0));
        assert_eq!(r1, Rect::ZERO);
    }

    #[test]
    fn rect_shrink_collapses_one_dim_to_zero() {
        // Left+right > w but top+bottom fits.
        let r0 = r(0.0, 0.0, 10.0, 100.0);
        let r1 = r0.shrink(Inset { top: 5.0, right: 6.0, bottom: 5.0, left: 6.0 });
        assert_eq!(r1, Rect::ZERO);
    }

    #[test]
    fn split_top_partition_is_exact() {
        let r0 = r(10.0, 20.0, 100.0, 80.0);
        let (top, rest) = r0.split_top(30.0);
        assert_eq!(top, r(10.0, 20.0, 100.0, 30.0));
        assert_eq!(rest, r(10.0, 50.0, 100.0, 50.0));
        // No gap, no overlap.
        assert_eq!(top.y + top.h, rest.y);
    }

    #[test]
    fn split_bottom_partition_is_exact() {
        let r0 = r(10.0, 20.0, 100.0, 80.0);
        let (bottom, rest) = r0.split_bottom(30.0);
        assert_eq!(rest, r(10.0, 20.0, 100.0, 50.0));
        assert_eq!(bottom, r(10.0, 70.0, 100.0, 30.0));
    }

    #[test]
    fn split_left_partition_is_exact() {
        let r0 = r(10.0, 20.0, 100.0, 80.0);
        let (left, rest) = r0.split_left(40.0);
        assert_eq!(left, r(10.0, 20.0, 40.0, 80.0));
        assert_eq!(rest, r(50.0, 20.0, 60.0, 80.0));
    }

    #[test]
    fn split_right_partition_is_exact() {
        let r0 = r(10.0, 20.0, 100.0, 80.0);
        let (right, rest) = r0.split_right(40.0);
        assert_eq!(rest, r(10.0, 20.0, 60.0, 80.0));
        assert_eq!(right, r(70.0, 20.0, 40.0, 80.0));
    }

    #[test]
    fn viewport_into_rect() {
        let v = Viewport { width: 600.0, height: 400.0 };
        assert_eq!(v.into_rect(), r(0.0, 0.0, 600.0, 400.0));
    }

    #[test]
    fn rect_serde_round_trip() {
        let r0 = r(1.0, 2.0, 3.0, 4.0);
        let json = serde_json::to_string(&r0).unwrap();
        let r1: Rect = serde_json::from_str(&json).unwrap();
        assert_eq!(r0, r1);
    }
}
```

- [ ] **Step 2: Run tests, verify they pass**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::geometry
```

Expected: 9 tests pass (`rect_shrink_normal`, `rect_shrink_collapses_to_zero`, `rect_shrink_collapses_one_dim_to_zero`, `split_top_partition_is_exact`, `split_bottom_partition_is_exact`, `split_left_partition_is_exact`, `split_right_partition_is_exact`, `viewport_into_rect`, `rect_serde_round_trip`).

- [ ] **Step 3: Commit**

```bash
git add crates/ferrum-core/src/layout/geometry.rs
git commit -m "feat(layout): geometry primitives — Rect, Inset, Viewport

Rect.shrink returns ZERO if the inset would collapse either dim;
split_{top,bottom,left,right} produce exact partitions (no overlap,
no gap). 9 tests inline."
```

---

### Task 3: `text_metrics.rs` — `TextMetrics` trait + `HeuristicMetrics` + `MockMetrics`

**Files:**
- Modify: `crates/ferrum-core/src/layout/text_metrics.rs`

- [ ] **Step 1: Write failing test + implementation**

Replace `text_metrics.rs` with:

```rust
//! Text width measurement. Phase 6 ships only `HeuristicMetrics` (char_count *
//! font_size * K). Phase 7's renderer will provide a fontdue-backed implementation
//! by implementing this trait. `MockMetrics` is test-only and supports a
//! user-supplied closure for pixel-exact reference values.

pub trait TextMetrics {
    fn measure_width(&self, text: &str, font_size: f64) -> f64;
    fn line_height(&self, font_size: f64) -> f64 {
        font_size * 1.2
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HeuristicMetrics {
    pub k: f64,
}

impl Default for HeuristicMetrics {
    fn default() -> Self {
        Self { k: 0.6 }
    }
}

impl TextMetrics for HeuristicMetrics {
    fn measure_width(&self, text: &str, font_size: f64) -> f64 {
        text.chars().count() as f64 * font_size * self.k
    }
}

#[cfg(test)]
pub(crate) struct MockMetrics<F: Fn(&str, f64) -> f64> {
    pub measure: F,
    pub line_h_factor: f64,
}

#[cfg(test)]
impl<F: Fn(&str, f64) -> f64> TextMetrics for MockMetrics<F> {
    fn measure_width(&self, text: &str, font_size: f64) -> f64 {
        (self.measure)(text, font_size)
    }
    fn line_height(&self, font_size: f64) -> f64 {
        font_size * self.line_h_factor
    }
}

#[cfg(test)]
pub(crate) fn fixed_width(per_char_px: f64) -> impl Fn(&str, f64) -> f64 {
    move |text: &str, _font_size: f64| text.chars().count() as f64 * per_char_px
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_default_k_is_0_6() {
        let m = HeuristicMetrics::default();
        assert!((m.k - 0.6).abs() < 1e-12);
    }

    #[test]
    fn heuristic_measure_hello_at_12pt_equals_36() {
        let m = HeuristicMetrics::default();
        // 5 chars * 12 * 0.6 = 36.0
        assert!((m.measure_width("hello", 12.0) - 36.0).abs() < 1e-12);
    }

    #[test]
    fn heuristic_line_height_default_is_1_2x() {
        let m = HeuristicMetrics::default();
        assert!((m.line_height(12.0) - 14.4).abs() < 1e-12);
    }

    #[test]
    fn heuristic_handles_unicode_chars() {
        let m = HeuristicMetrics::default();
        // 3 chars (each emoji is one char in this count), font_size 10.
        // The actual char count depends on grapheme/scalar count; we use chars()
        // which counts unicode scalars: "abc" = 3, "héllo" = 5.
        assert!((m.measure_width("abc", 10.0) - 18.0).abs() < 1e-12);
        assert!((m.measure_width("héllo", 10.0) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn mock_metrics_uses_closure() {
        let m = MockMetrics {
            measure: fixed_width(10.0),
            line_h_factor: 1.5,
        };
        assert_eq!(m.measure_width("abc", 12.0), 30.0);
        assert!((m.line_height(12.0) - 18.0).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::text_metrics
```

Expected: 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/ferrum-core/src/layout/text_metrics.rs
git commit -m "feat(layout): TextMetrics trait + HeuristicMetrics + MockMetrics

Heuristic K=0.6 default; line_height = 1.2 * font_size.
MockMetrics is test-only (cfg(test)) and accepts an arbitrary
closure so collision-policy tests use exact reference widths
without depending on the heuristic being accurate."
```

---

### Task 4: `panel.rs` — `PanelLayout`, `FacetKey` types

**Files:**
- Modify: `crates/ferrum-core/src/layout/panel.rs`

- [ ] **Step 1: Write the types + a serde round-trip test**

```rust
//! Per-panel layout output. A non-faceted chart yields one PanelLayout with
//! `facet_key = None` and `(row, col) = (0, 0)`.

use serde::{Deserialize, Serialize};

use super::geometry::Rect;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PanelLayout {
    pub plot_area: Rect,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub facet_key: Option<FacetKey>,
    pub row: u32,
    pub col: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FacetKey {
    pub field: String,
    pub value: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_layout_round_trip_no_facet() {
        let p = PanelLayout {
            plot_area: Rect { x: 10.0, y: 20.0, w: 300.0, h: 200.0 },
            facet_key: None,
            row: 0,
            col: 0,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(!json.contains("facet_key"), "facet_key None must be skipped: {json}");
        let parsed: PanelLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn panel_layout_round_trip_with_facet() {
        let p = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 200.0, h: 100.0 },
            facet_key: Some(FacetKey {
                field: "species".into(),
                value: "setosa".into(),
            }),
            row: 1,
            col: 2,
        };
        let json = serde_json::to_string(&p).unwrap();
        let parsed: PanelLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::panel
```

Expected: 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/ferrum-core/src/layout/panel.rs
git commit -m "feat(layout): PanelLayout + FacetKey types

Plain serde-derive structs. facet_key is skipped when None so
non-faceted charts produce minimal JSON output."
```

---

### Task 5: `facet.rs` types + ChartSpec gains `facet: Option<FacetSpec>`

**Files:**
- Modify: `crates/ferrum-core/src/layout/facet.rs`
- Modify: `crates/ferrum-core/src/spec/chart.rs`

- [ ] **Step 1: Add `FacetSpec`, `FacetMode`, `FacetGroup` to `facet.rs`**

```rust
//! Facet input/output types and grid arithmetic. Phase 6 supports two modes:
//! Wrap (ncols set, nrows derived from n_panels) and Grid (both explicit;
//! panels beyond nrows*ncols are dropped with a warning).

use serde::{Deserialize, Serialize};

use super::geometry::Rect;
use super::panel::FacetKey;

/// Spec-side facet declaration. Carried by `ChartSpec.facet`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FacetSpec {
    pub field: String,
    pub mode: FacetMode,
    /// If set, overrides `theme.column_padding` / `theme.row_padding` symmetrically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spacing: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum FacetMode {
    Wrap { ncols: u32 },
    Grid { nrows: u32, ncols: u32 },
}

/// Caller-supplied per-panel input. `n_rows` is informational only — Phase 6
/// does not use it for layout decisions but Phase 7+ may.
#[derive(Debug, Clone, PartialEq)]
pub struct FacetGroup {
    pub key: FacetKey,
    pub n_rows: u64,
}

/// Computed grid sizing. `cell_rect(row, col, origin)` returns the panel rect.
#[derive(Debug, Clone, PartialEq)]
pub struct FacetGrid {
    pub mode: FacetMode,
    pub n_panels: u32,
    pub cell_w: f64,
    pub cell_h: f64,
    pub gutter_x: f64,
    pub gutter_y: f64,
    pub origin: Rect,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facet_spec_round_trip_wrap() {
        let s = FacetSpec {
            field: "species".into(),
            mode: FacetMode::Wrap { ncols: 3 },
            spacing: None,
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: FacetSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
        assert!(json.contains(r#""kind":"wrap""#));
        assert!(json.contains(r#""ncols":3"#));
    }

    #[test]
    fn facet_spec_round_trip_grid() {
        let s = FacetSpec {
            field: "year".into(),
            mode: FacetMode::Grid { nrows: 2, ncols: 3 },
            spacing: Some(12.0),
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: FacetSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
        assert!(json.contains(r#""kind":"grid""#));
    }

    #[test]
    fn facet_spec_omits_spacing_when_none() {
        let s = FacetSpec {
            field: "f".into(),
            mode: FacetMode::Wrap { ncols: 2 },
            spacing: None,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("spacing"));
    }
}
```

- [ ] **Step 2: Add `facet` field to `ChartSpec`**

Edit `crates/ferrum-core/src/spec/chart.rs`. In the `ChartSpec` struct definition (lines ~13-23), append after the `transforms` field:

```rust
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facet: Option<crate::layout::facet::FacetSpec>,
```

So the full struct becomes:

```rust
#[pyclass(eq, module = "ferrum._core")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChartSpec {
    #[serde(default)]
    pub data: DataRef,
    pub mark: Mark,
    #[serde(default)]
    pub encoding: Encoding,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<crate::transform::core::TransformSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facet: Option<crate::layout::facet::FacetSpec>,
}
```

In the `#[new]` constructor (line ~28), update the `pyo3(signature)` and parameters and the struct literal at the bottom. The minimum viable change: do **not** expose `facet` to the Python `__new__` constructor in Phase 6 — leave it `None` by default. (Phase 8 grammar will add the Python builder.)

So just update the struct literal at line ~55 from:

```rust
        Ok(ChartSpec {
            data,
            mark,
            encoding: Encoding { x, y },
            transforms,
        })
```

to:

```rust
        Ok(ChartSpec {
            data,
            mark,
            encoding: Encoding { x, y },
            transforms,
            facet: None,
        })
```

And update **both** existing `#[cfg(test)]` `minimal_scatter` literal AND the literals in `test_canonical_json_shape` and `test_chart_spec_transforms_omitted_in_canonical_json_when_empty` and `test_chart_spec_transforms_round_trip_with_one_bin` — they all initialize `ChartSpec` and now must include `facet: None`.

For each test struct literal in `crates/ferrum-core/src/spec/chart.rs` (search for `ChartSpec {` in the test module), append `facet: None,` as the last field.

- [ ] **Step 3: Add a back-compat test in `chart.rs` test module**

Append inside the `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn test_chart_spec_facet_default_when_omitted() {
        // Pre-Phase-6 JSON shape (no `facet` field) must still deserialize.
        let json = r#"{"data":{"kind":"named","name":"default"},"mark":"point","encoding":{}}"#;
        let parsed: ChartSpec = serde_json::from_str(json).unwrap();
        assert!(parsed.facet.is_none());
    }

    #[test]
    fn test_chart_spec_facet_omitted_in_canonical_json_when_none() {
        let spec = minimal_scatter();
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("facet"), "facet=None should be skipped: {json}");
    }

    #[test]
    fn test_chart_spec_facet_round_trip() {
        use crate::layout::facet::{FacetMode, FacetSpec};
        let mut spec = minimal_scatter();
        spec.facet = Some(FacetSpec {
            field: "species".into(),
            mode: FacetMode::Wrap { ncols: 3 },
            spacing: None,
        });
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""facet":{"#));
        let parsed: ChartSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
    }
```

- [ ] **Step 4: Run tests, then build for Python**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::facet
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core spec::chart
```

Expected: 3 tests pass in `layout::facet`; all existing `spec::chart` tests still pass plus 3 new ones.

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
unset CONDA_PREFIX && uv run --no-sync python -c "from ferrum._core import ChartSpec; s=ChartSpec(mark='point', x='a', y='b'); assert s == ChartSpec.from_json(s.to_json()); print('OK')"
```

Expected: `OK`. Existing `tests/test_chart_spec.py` round-trip still passes.

```bash
uv run pytest tests/test_chart_spec.py -v
```

Expected: all existing chart-spec pytest tests still pass.

- [ ] **Step 5: Commit**

```bash
git add crates/ferrum-core/src/layout/facet.rs crates/ferrum-core/src/spec/chart.rs
git commit -m "feat(spec): ChartSpec gains optional facet field; FacetSpec type

facet: Option<FacetSpec> with serde(default, skip_serializing_if).
Pre-Phase-6 JSON deserializes byte-identical (facet=None omitted on
serialize). FacetMode is a tagged union (kind=wrap|grid).
FacetGroup carries caller-pre-computed group keys + row counts."
```

---

### Task 6: `FacetGrid` — wrap mode arithmetic

**Files:**
- Modify: `crates/ferrum-core/src/layout/facet.rs`

- [ ] **Step 1: Write failing tests**

Append inside the existing `#[cfg(test)] mod tests` in `facet.rs`:

```rust
    fn rect(x: f64, y: f64, w: f64, h: f64) -> Rect {
        Rect { x, y, w, h }
    }

    #[test]
    fn facet_grid_wrap_exact_fit() {
        // viewport 600x400, ncols=3, n_panels=3, gutter=0
        // → 1 row of 3 cells, each 200x400.
        let origin = rect(0.0, 0.0, 600.0, 400.0);
        let grid = FacetGrid::compute_wrap(3, 3, origin, 0.0, 0.0);
        assert_eq!(grid.cell_w, 200.0);
        assert_eq!(grid.cell_h, 400.0);
        assert_eq!(grid.cell_rect(0, 0), rect(0.0, 0.0, 200.0, 400.0));
        assert_eq!(grid.cell_rect(0, 1), rect(200.0, 0.0, 200.0, 400.0));
        assert_eq!(grid.cell_rect(0, 2), rect(400.0, 0.0, 200.0, 400.0));
    }

    #[test]
    fn facet_grid_wrap_ragged() {
        // ncols=3, n_panels=5, gutter=0 → 2 rows; last row has 2 cells.
        let origin = rect(0.0, 0.0, 600.0, 400.0);
        let grid = FacetGrid::compute_wrap(3, 5, origin, 0.0, 0.0);
        assert_eq!(grid.cell_w, 200.0);
        assert_eq!(grid.cell_h, 200.0);
        // Panels 0..5 map to: (0,0),(0,1),(0,2),(1,0),(1,1)
        assert_eq!(grid.cell_rect(0, 0), rect(0.0, 0.0, 200.0, 200.0));
        assert_eq!(grid.cell_rect(0, 1), rect(200.0, 0.0, 200.0, 200.0));
        assert_eq!(grid.cell_rect(0, 2), rect(400.0, 0.0, 200.0, 200.0));
        assert_eq!(grid.cell_rect(1, 0), rect(0.0, 200.0, 200.0, 200.0));
        assert_eq!(grid.cell_rect(1, 1), rect(200.0, 200.0, 200.0, 200.0));
    }

    #[test]
    fn facet_grid_wrap_with_gutters() {
        // viewport 620x420, ncols=3, n_panels=5, gutter_x=10, gutter_y=10
        // cell_w = (620 - 2*10) / 3 = 200
        // cell_h = (420 - 1*10) / 2 = 205
        let origin = rect(0.0, 0.0, 620.0, 420.0);
        let grid = FacetGrid::compute_wrap(3, 5, origin, 10.0, 10.0);
        assert_eq!(grid.cell_w, 200.0);
        assert_eq!(grid.cell_h, 205.0);
        // (0,0) at origin; (0,1) at x = 200 + 10 = 210
        assert_eq!(grid.cell_rect(0, 0), rect(0.0, 0.0, 200.0, 205.0));
        assert_eq!(grid.cell_rect(0, 1), rect(210.0, 0.0, 200.0, 205.0));
        assert_eq!(grid.cell_rect(0, 2), rect(420.0, 0.0, 200.0, 205.0));
        // Row 1 at y = 205 + 10 = 215
        assert_eq!(grid.cell_rect(1, 0), rect(0.0, 215.0, 200.0, 205.0));
    }

    #[test]
    fn facet_grid_wrap_panel_index_to_row_col() {
        // Sanity: panels(...) returns the right (row,col) sequence for ragged.
        let origin = rect(0.0, 0.0, 600.0, 400.0);
        let grid = FacetGrid::compute_wrap(3, 5, origin, 0.0, 0.0);
        let panels = grid.panel_positions();
        assert_eq!(panels, vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]);
    }
```

- [ ] **Step 2: Implement `compute_wrap`, `cell_rect`, `panel_positions`**

In `facet.rs` add an `impl FacetGrid` block after the struct definitions (before the `#[cfg(test)]` module):

```rust
impl FacetGrid {
    /// Wrap mode: ncols is fixed; nrows = ceil(n_panels / ncols).
    pub fn compute_wrap(
        ncols: u32,
        n_panels: u32,
        origin: Rect,
        gutter_x: f64,
        gutter_y: f64,
    ) -> FacetGrid {
        let ncols = ncols.max(1);
        let nrows = (n_panels + ncols - 1) / ncols;
        let nrows = nrows.max(1);
        let total_x_gutter = gutter_x * (ncols.saturating_sub(1) as f64);
        let total_y_gutter = gutter_y * (nrows.saturating_sub(1) as f64);
        let cell_w = ((origin.w - total_x_gutter) / ncols as f64).max(0.0);
        let cell_h = ((origin.h - total_y_gutter) / nrows as f64).max(0.0);
        FacetGrid {
            mode: FacetMode::Wrap { ncols },
            n_panels,
            cell_w,
            cell_h,
            gutter_x,
            gutter_y,
            origin,
        }
    }

    pub fn cell_rect(&self, row: u32, col: u32) -> Rect {
        Rect {
            x: self.origin.x + col as f64 * (self.cell_w + self.gutter_x),
            y: self.origin.y + row as f64 * (self.cell_h + self.gutter_y),
            w: self.cell_w,
            h: self.cell_h,
        }
    }

    /// Returns the (row, col) for each of the `n_panels` panels, in panel-index
    /// order. For wrap mode: row-major. For grid mode (Task 7): same row-major
    /// order, capped at nrows*ncols.
    pub fn panel_positions(&self) -> Vec<(u32, u32)> {
        let ncols = match self.mode {
            FacetMode::Wrap { ncols } => ncols,
            FacetMode::Grid { ncols, .. } => ncols,
        };
        let cap = match self.mode {
            FacetMode::Wrap { .. } => self.n_panels,
            FacetMode::Grid { nrows, ncols } => self.n_panels.min(nrows * ncols),
        };
        (0..cap)
            .map(|i| (i / ncols, i % ncols))
            .collect()
    }
}
```

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::facet
```

Expected: 4 new tests pass (plus the 3 from Task 5 = 7 total).

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/layout/facet.rs
git commit -m "feat(layout): FacetGrid wrap-mode arithmetic

compute_wrap(ncols, n_panels, origin, gutter_x, gutter_y) computes
cell sizes from the available rect; cell_rect(row, col) returns
the panel rect; panel_positions() yields (row, col) per panel index
in row-major order. Pinned numeric tests for exact-fit, ragged,
gutter cases per spec §9.1."
```

---

### Task 7: `FacetGrid` — grid mode + dropped-panels warning

**Files:**
- Modify: `crates/ferrum-core/src/layout/facet.rs`

- [ ] **Step 1: Write failing tests**

Append to the `#[cfg(test)] mod tests` block in `facet.rs`:

```rust
    #[test]
    fn facet_grid_mode_exact_fit() {
        // nrows=2, ncols=3, n_panels=6 → identical to wrap 3x2, all 6 cells filled.
        let origin = rect(0.0, 0.0, 600.0, 400.0);
        let grid = FacetGrid::compute_grid(2, 3, 6, origin, 0.0, 0.0);
        assert_eq!(grid.cell_w, 200.0);
        assert_eq!(grid.cell_h, 200.0);
        assert_eq!(grid.panel_positions().len(), 6);
        assert_eq!(grid.cell_rect(1, 2), rect(400.0, 200.0, 200.0, 200.0));
        assert_eq!(grid.dropped_count(), 0);
    }

    #[test]
    fn facet_grid_mode_overflow_drops_panels() {
        // nrows=2, ncols=3, n_panels=8 → 6 emitted, 2 dropped.
        let origin = rect(0.0, 0.0, 600.0, 400.0);
        let grid = FacetGrid::compute_grid(2, 3, 8, origin, 0.0, 0.0);
        assert_eq!(grid.panel_positions().len(), 6);
        assert_eq!(grid.dropped_count(), 2);
    }

    #[test]
    fn facet_grid_mode_underfill_does_not_panic() {
        // nrows=2, ncols=3, n_panels=4 → 4 emitted, 0 dropped (cells are sized
        // for the 2x3 grid even though 2 are empty).
        let origin = rect(0.0, 0.0, 600.0, 400.0);
        let grid = FacetGrid::compute_grid(2, 3, 4, origin, 0.0, 0.0);
        assert_eq!(grid.cell_w, 200.0);
        assert_eq!(grid.cell_h, 200.0);
        assert_eq!(grid.panel_positions().len(), 4);
        assert_eq!(grid.dropped_count(), 0);
    }
```

- [ ] **Step 2: Implement `compute_grid` and `dropped_count`**

In the `impl FacetGrid` block in `facet.rs`, append:

```rust
    /// Grid mode: nrows and ncols both fixed. Panels beyond nrows*ncols are
    /// dropped (caller should emit `LayoutWarning::PanelsDropped`).
    pub fn compute_grid(
        nrows: u32,
        ncols: u32,
        n_panels: u32,
        origin: Rect,
        gutter_x: f64,
        gutter_y: f64,
    ) -> FacetGrid {
        let nrows = nrows.max(1);
        let ncols = ncols.max(1);
        let total_x_gutter = gutter_x * (ncols.saturating_sub(1) as f64);
        let total_y_gutter = gutter_y * (nrows.saturating_sub(1) as f64);
        let cell_w = ((origin.w - total_x_gutter) / ncols as f64).max(0.0);
        let cell_h = ((origin.h - total_y_gutter) / nrows as f64).max(0.0);
        FacetGrid {
            mode: FacetMode::Grid { nrows, ncols },
            n_panels,
            cell_w,
            cell_h,
            gutter_x,
            gutter_y,
            origin,
        }
    }

    /// In grid mode, returns max(0, n_panels - nrows*ncols). Always 0 in wrap mode.
    pub fn dropped_count(&self) -> u32 {
        match self.mode {
            FacetMode::Grid { nrows, ncols } => {
                let cap = nrows * ncols;
                if self.n_panels > cap { self.n_panels - cap } else { 0 }
            }
            FacetMode::Wrap { .. } => 0,
        }
    }
```

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::facet
```

Expected: 3 new tests pass (10 total in the module).

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/layout/facet.rs
git commit -m "feat(layout): FacetGrid grid mode + dropped-panels accounting

compute_grid(nrows, ncols, n_panels, origin, gutter_x, gutter_y).
dropped_count() returns the overflow count for grid mode (0 in wrap
mode). The orchestrator emits LayoutWarning::PanelsDropped when > 0."
```

---

### Task 8: `axis.rs` — input/output types

**Files:**
- Modify: `crates/ferrum-core/src/layout/axis.rs`

- [ ] **Step 1: Write the types + serde tests**

```rust
//! Axis input (caller-supplied) and axis layout output (engine-computed).
//! Per spec §14.1: tick labels are caller-pre-computed via Phase 4 scales;
//! Phase 6 never touches scale internals.

use serde::{Deserialize, Serialize};

use super::geometry::Rect;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AxisOrient {
    Top,
    Bottom,
    Left,
    Right,
}

/// Caller-supplied per-axis input. Phase 6 takes both x and y always.
#[derive(Debug, Clone, PartialEq)]
pub struct AxisInput {
    pub orient: AxisOrient,
    pub title: Option<String>,
    pub tick_labels: Vec<String>,
    pub label_angle_override: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AxesInput {
    pub x: AxisInput,
    pub y: AxisInput,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisLayout {
    pub orient: AxisOrient,
    pub panel_index: usize,
    pub axis_line: Rect,
    pub ticks: Vec<TickLayout>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub title: Option<AxisTitleLayout>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickLayout {
    pub position: f64,
    pub label: String,
    pub label_angle: f64,
    pub elided: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisTitleLayout {
    pub text: String,
    pub anchor_x: f64,
    pub anchor_y: f64,
    pub angle: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_layout_round_trip() {
        let a = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 50.0, y: 350.0, w: 500.0, h: 1.0 },
            ticks: vec![TickLayout {
                position: 100.0,
                label: "0".into(),
                label_angle: 0.0,
                elided: false,
            }],
            title: Some(AxisTitleLayout {
                text: "Price".into(),
                anchor_x: 300.0,
                anchor_y: 380.0,
                angle: 0.0,
            }),
        };
        let json = serde_json::to_string(&a).unwrap();
        let parsed: AxisLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, a);
    }

    #[test]
    fn axis_layout_serde_lowercases_orient() {
        let a = AxisLayout {
            orient: AxisOrient::Left,
            panel_index: 0,
            axis_line: Rect::ZERO,
            ticks: vec![],
            title: None,
        };
        let json = serde_json::to_string(&a).unwrap();
        assert!(json.contains(r#""orient":"left""#));
        assert!(!json.contains("title"));
    }
}
```

- [ ] **Step 2: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::axis
```

Expected: 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/ferrum-core/src/layout/axis.rs
git commit -m "feat(layout): axis input + output types

AxisInput / AxesInput are caller-supplied (Phase 6 stays data-blind
per spec §14.1). AxisLayout / TickLayout / AxisTitleLayout are the
output structs the renderer consumes. AxisOrient serializes as
lowercase strings (top|bottom|left|right)."
```

---

### Task 9: y-axis layout (no collision policy)

**Files:**
- Modify: `crates/ferrum-core/src/layout/axis.rs`

- [ ] **Step 1: Write failing tests**

Append to the `#[cfg(test)] mod tests` block in `axis.rs`:

```rust
    use crate::layout::text_metrics::{fixed_width, MockMetrics};

    fn mock(per_char_px: f64) -> MockMetrics<impl Fn(&str, f64) -> f64> {
        MockMetrics { measure: fixed_width(per_char_px), line_h_factor: 1.2 }
    }

    #[test]
    fn y_axis_label_band_uses_longest_label() {
        // Labels "0", "100", "10000" — longest is "10000" (5 chars * 10 = 50).
        let input = AxisInput {
            orient: AxisOrient::Left,
            title: None,
            tick_labels: vec!["0".into(), "100".into(), "10000".into()],
            label_angle_override: None,
        };
        let m = mock(10.0);
        let band = compute_y_label_band_width(&input, 11.0, &m);
        assert_eq!(band, 50.0);
    }

    #[test]
    fn y_axis_label_band_empty_labels_returns_zero() {
        let input = AxisInput {
            orient: AxisOrient::Left,
            title: None,
            tick_labels: vec![],
            label_angle_override: None,
        };
        let m = mock(10.0);
        assert_eq!(compute_y_label_band_width(&input, 11.0, &m), 0.0);
    }

    #[test]
    fn y_axis_layout_uniform_tick_positions() {
        // panel_h=200, 4 ticks → slot=50; centers at 25, 75, 125, 175.
        let input = AxisInput {
            orient: AxisOrient::Left,
            title: Some("Price".into()),
            tick_labels: vec!["0".into(), "1".into(), "2".into(), "3".into()],
            label_angle_override: None,
        };
        let panel_area = Rect { x: 100.0, y: 50.0, w: 300.0, h: 200.0 };
        let m = mock(10.0);
        let axis = layout_y_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        assert_eq!(axis.orient, AxisOrient::Left);
        assert_eq!(axis.panel_index, 0);
        assert_eq!(axis.ticks.len(), 4);
        // y-axis: position is the y pixel coord. Slot height = 200/4 = 50.
        // Center positions: panel.y + (i + 0.5) * slot
        assert!((axis.ticks[0].position - (50.0 + 25.0)).abs() < 1e-9);
        assert!((axis.ticks[3].position - (50.0 + 175.0)).abs() < 1e-9);
        // No rotation, no elision on y-axis ever.
        for t in &axis.ticks {
            assert_eq!(t.label_angle, 0.0);
            assert!(!t.elided);
        }
        // Title present and rotated -90° (vertical).
        let title = axis.title.unwrap();
        assert_eq!(title.text, "Price");
        assert!((title.angle - (-90.0)).abs() < 1e-9);
    }
```

- [ ] **Step 2: Implement `compute_y_label_band_width` and `layout_y_axis`**

Append to `axis.rs` (above the `#[cfg(test)] mod tests` block):

```rust
use super::text_metrics::TextMetrics;

/// Returns the pixel width of the widest tick label on the y-axis. Used by the
/// orchestrator to reserve a left gutter before computing the plot rect.
pub fn compute_y_label_band_width(
    input: &AxisInput,
    label_font_size: f64,
    metrics: &dyn TextMetrics,
) -> f64 {
    input
        .tick_labels
        .iter()
        .map(|s| metrics.measure_width(s, label_font_size))
        .fold(0.0_f64, f64::max)
}

/// Returns the title-row width contribution: title text height (rotated 90°,
/// so its "width" along the x-axis is its line height) plus axis_title_padding.
/// Returns 0 if there is no title.
pub fn compute_y_title_width(
    input: &AxisInput,
    title_font_size: f64,
    axis_title_padding: f64,
    metrics: &dyn TextMetrics,
) -> f64 {
    if input.title.is_some() {
        metrics.line_height(title_font_size) + axis_title_padding
    } else {
        0.0
    }
}

/// Build the AxisLayout for the y-axis (Left orient) of a single panel.
/// Tick positions are uniformly spaced across `panel_area.h`; no collision
/// policy applies to y-axis (spec §14.4).
pub fn layout_y_axis(
    input: &AxisInput,
    panel_area: Rect,
    panel_index: usize,
    label_font_size: f64,
    title_font_size: f64,
    axis_title_padding: f64,
    metrics: &dyn TextMetrics,
) -> AxisLayout {
    let n = input.tick_labels.len();
    let slot_h = if n > 0 { panel_area.h / n as f64 } else { 0.0 };
    let ticks: Vec<TickLayout> = input
        .tick_labels
        .iter()
        .enumerate()
        .map(|(i, label)| TickLayout {
            position: panel_area.y + (i as f64 + 0.5) * slot_h,
            label: label.clone(),
            label_angle: 0.0,
            elided: false,
        })
        .collect();

    let axis_line = Rect {
        x: panel_area.x,
        y: panel_area.y,
        w: 1.0,
        h: panel_area.h,
    };

    let title = input.title.as_ref().map(|text| {
        let label_band = compute_y_label_band_width(input, label_font_size, metrics);
        let title_h = metrics.line_height(title_font_size);
        AxisTitleLayout {
            text: text.clone(),
            anchor_x: panel_area.x - label_band - axis_title_padding - title_h / 2.0,
            anchor_y: panel_area.y + panel_area.h / 2.0,
            angle: -90.0,
        }
    });

    AxisLayout {
        orient: AxisOrient::Left,
        panel_index,
        axis_line,
        ticks,
        title,
    }
}
```

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::axis
```

Expected: 3 new tests pass (5 total).

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/layout/axis.rs
git commit -m "feat(layout): y-axis layout — band width, title, uniform ticks

compute_y_label_band_width / compute_y_title_width feed the
orchestrator's left-gutter reservation. layout_y_axis builds the
AxisLayout: uniform tick positions, vertical title rotated -90°.
No collision policy on the y-axis (spec §14.4)."
```

---

### Task 10: x-axis layout — flat default (no collision)

**Files:**
- Modify: `crates/ferrum-core/src/layout/axis.rs`

- [ ] **Step 1: Write failing tests**

Append to the `axis.rs` test module:

```rust
    #[test]
    fn x_axis_no_collision_keeps_labels_flat() {
        // 4 labels, mock width=50px each, panel_w=400, slot=100. Slack = 100*0.9=90.
        // 50 < 90, no rotation.
        let input = AxisInput {
            orient: AxisOrient::Bottom,
            title: None,
            tick_labels: vec!["A".into(), "B".into(), "C".into(), "D".into()],
            label_angle_override: None,
        };
        let panel_area = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 50.0, line_h_factor: 1.2 };
        let (axis, warning) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        assert_eq!(axis.ticks.len(), 4);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, 0.0);
            assert!(!t.elided);
        }
        assert!(warning.is_none());
    }

    #[test]
    fn x_axis_uniform_tick_positions_along_axis() {
        // panel_w=400, 4 ticks → slot=100; centers at 50, 150, 250, 350.
        let input = AxisInput {
            orient: AxisOrient::Bottom,
            title: None,
            tick_labels: vec!["A".into(), "B".into(), "C".into(), "D".into()],
            label_angle_override: None,
        };
        let panel_area = Rect { x: 100.0, y: 50.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 10.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        assert!((axis.ticks[0].position - (100.0 + 50.0)).abs() < 1e-9);
        assert!((axis.ticks[1].position - (100.0 + 150.0)).abs() < 1e-9);
        assert!((axis.ticks[2].position - (100.0 + 250.0)).abs() < 1e-9);
        assert!((axis.ticks[3].position - (100.0 + 350.0)).abs() < 1e-9);
    }
```

- [ ] **Step 2: Implement `layout_x_axis` (collision branches stubbed for next two tasks)**

Append to `axis.rs` (above the `#[cfg(test)]` block):

```rust
use crate::layout::{LABEL_OVERLAP_TOLERANCE, DEFAULT_LABEL_ANGLE};

/// Per-x-axis warning the orchestrator may emit. Internal — consumers translate
/// to `LayoutWarning`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum XAxisWarning {
    LabelsElided { count: u32 },
}

/// Build the AxisLayout for the x-axis (Bottom orient) of a single panel.
/// Tick positions are uniformly spaced across `panel_area.w` (spec §14.3 step 7a).
/// Collision policy: rotate labels then elide if still colliding (spec §14.4).
pub fn layout_x_axis(
    input: &AxisInput,
    panel_area: Rect,
    panel_index: usize,
    label_font_size: f64,
    title_font_size: f64,
    axis_title_padding: f64,
    metrics: &dyn TextMetrics,
) -> (AxisLayout, Option<XAxisWarning>) {
    let n = input.tick_labels.len();
    let slot_w = if n > 0 { panel_area.w / n as f64 } else { 0.0 };

    // Step 1: measure all labels flat.
    let widths: Vec<f64> = input
        .tick_labels
        .iter()
        .map(|s| metrics.measure_width(s, label_font_size))
        .collect();

    // Step 2: decide whether any label exceeds slot * (1 - tolerance).
    let threshold = slot_w * (1.0 - LABEL_OVERLAP_TOLERANCE);
    let any_collision = widths.iter().any(|w| *w > threshold);
    let angle = if any_collision {
        input.label_angle_override.unwrap_or(DEFAULT_LABEL_ANGLE)
    } else {
        0.0
    };

    // Step 3: collision recovery — rotation, then elision (Tasks 11 + 12).
    // Phase 1 of this task: produce flat ticks if no collision.
    let (ticks, warning) = if !any_collision {
        let ticks: Vec<TickLayout> = input
            .tick_labels
            .iter()
            .enumerate()
            .map(|(i, label)| TickLayout {
                position: panel_area.x + (i as f64 + 0.5) * slot_w,
                label: label.clone(),
                label_angle: 0.0,
                elided: false,
            })
            .collect();
        (ticks, None)
    } else {
        // Filled in by Tasks 11 (rotation) and 12 (elision).
        let ticks: Vec<TickLayout> = input
            .tick_labels
            .iter()
            .enumerate()
            .map(|(i, label)| TickLayout {
                position: panel_area.x + (i as f64 + 0.5) * slot_w,
                label: label.clone(),
                label_angle: angle,
                elided: false,
            })
            .collect();
        (ticks, None)
    };

    let axis_line = Rect {
        x: panel_area.x,
        y: panel_area.y + panel_area.h,
        w: panel_area.w,
        h: 1.0,
    };

    let title = input.title.as_ref().map(|text| {
        let title_h = metrics.line_height(title_font_size);
        let label_h = metrics.line_height(label_font_size);
        AxisTitleLayout {
            text: text.clone(),
            anchor_x: panel_area.x + panel_area.w / 2.0,
            anchor_y: panel_area.y + panel_area.h + label_h + axis_title_padding + title_h / 2.0,
            angle: 0.0,
        }
    });

    (AxisLayout { orient: AxisOrient::Bottom, panel_index, axis_line, ticks, title }, warning)
}
```

- [ ] **Step 3: Add the layout-level constants in `mod.rs`**

Edit `crates/ferrum-core/src/layout/mod.rs`. Replace the existing content with:

```rust
//! Phase 6 — layout engine. Pure function: ChartSpec + Theme + Viewport ->
//! pixel rectangles for panels, axes, legend. No I/O, no rendering, no data
//! values touched. See docs/superpowers/specs/2026-05-09-layout-engine-design.md.

pub(crate) mod geometry;
pub(crate) mod text_metrics;
pub(crate) mod panel;
pub(crate) mod facet;
pub(crate) mod axis;
pub(crate) mod legend;
pub(crate) mod binding;

// Spec §6.1 constants.
pub const LABEL_OVERLAP_TOLERANCE: f64 = 0.10;
pub const DEFAULT_LABEL_ANGLE: f64 = -45.0;
pub const DEFAULT_HEURISTIC_K: f64 = 0.6;
pub const MIN_PANEL_DIM: f64 = 1.0;
pub const DEFAULT_PADDING: f64 = 8.0;
pub const DEFAULT_LABEL_FONT_SIZE: f64 = 11.0;
pub const DEFAULT_TITLE_FONT_SIZE: f64 = 13.0;
pub const DEFAULT_AXIS_TITLE_PADDING: f64 = 4.0;
```

- [ ] **Step 4: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::axis
```

Expected: 2 new tests pass (7 total). The `MockMetrics` field type signature differs slightly between the two new tests (one uses `fixed_width(50.0)`, the other uses an inline closure `|_, _| 50.0`); both are valid `Fn(&str, f64) -> f64`.

- [ ] **Step 5: Commit**

```bash
git add crates/ferrum-core/src/layout/axis.rs crates/ferrum-core/src/layout/mod.rs
git commit -m "feat(layout): x-axis layout default (no collision)

layout_x_axis returns uniform tick positions + flat labels when no
label exceeds slot * (1 - LABEL_OVERLAP_TOLERANCE). Collision branch
populates label_angle but does not yet elide (Tasks 11 + 12).
Constants moved to layout/mod.rs."
```

---

### Task 11: x-axis collision policy — rotation when threshold exceeded

**Files:**
- Modify: `crates/ferrum-core/src/layout/axis.rs`

- [ ] **Step 1: Write failing tests**

Append to the `axis.rs` test module:

```rust
    #[test]
    fn x_axis_collision_triggers_default_45_rotation() {
        // 8 labels, mock width=80, panel_w=400 → slot=50, threshold=50*0.9=45.
        // 80 > 45, rotation fires. Rotated projection = 80 * cos(45°) ≈ 56.57.
        // 56.57 > 50 → elision needed too, but Task 11 only checks rotation
        // is set; elision flag handling is Task 12.
        let input = AxisInput {
            orient: AxisOrient::Bottom,
            title: None,
            tick_labels: (0..8).map(|i| format!("L{}", i)).collect(),
            label_angle_override: None,
        };
        let panel_area = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -45.0);
        }
    }

    #[test]
    fn x_axis_rotates_at_custom_angle_override() {
        let input = AxisInput {
            orient: AxisOrient::Bottom,
            title: None,
            tick_labels: (0..8).map(|i| format!("L{}", i)).collect(),
            label_angle_override: Some(-90.0),
        };
        let panel_area = Rect { x: 0.0, y: 0.0, w: 400.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 80.0, line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -90.0);
        }
    }

    #[test]
    fn x_axis_rotation_only_no_elision_when_rotated_fits() {
        // panel_w=600, 6 labels → slot=100, threshold=100*0.9=90.
        // width=95 each → 95 > 90 → rotate. cos(45°)*95 ≈ 67.18 < 100 → no elide.
        let input = AxisInput {
            orient: AxisOrient::Bottom,
            title: None,
            tick_labels: (0..6).map(|i| format!("L{}", i)).collect(),
            label_angle_override: None,
        };
        let panel_area = Rect { x: 0.0, y: 0.0, w: 600.0, h: 200.0 };
        let m = MockMetrics { measure: |_, _| 95.0, line_h_factor: 1.2 };
        let (axis, warning) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -45.0);
            assert!(!t.elided, "rotated projection should fit; no elision");
        }
        assert!(warning.is_none());
    }
```

- [ ] **Step 2: Refine `layout_x_axis` to compute the rotated projection and decide on elision**

Replace the body of the `if !any_collision { ... } else { ... }` branch in `axis.rs::layout_x_axis` with this fuller form (still leaves *actual* string elision to Task 12, but sets the right `elided` flag and emits the warning when the rotated projection still doesn't fit):

```rust
    let (ticks, warning) = if !any_collision {
        let ticks: Vec<TickLayout> = input
            .tick_labels
            .iter()
            .enumerate()
            .map(|(i, label)| TickLayout {
                position: panel_area.x + (i as f64 + 0.5) * slot_w,
                label: label.clone(),
                label_angle: 0.0,
                elided: false,
            })
            .collect();
        (ticks, None)
    } else {
        // Rotated projection: |L * cos(angle)|. Spec §6 step 7c.
        let cos_factor = (angle.to_radians()).cos().abs();
        let any_still_colliding = widths.iter().any(|w| *w * cos_factor > slot_w);
        let mut elided_count: u32 = 0;
        let ticks: Vec<TickLayout> = input
            .tick_labels
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let w = widths[i];
                let needs_elide = any_still_colliding && (w * cos_factor > slot_w);
                if needs_elide { elided_count += 1; }
                TickLayout {
                    position: panel_area.x + (i as f64 + 0.5) * slot_w,
                    // Actual string elision happens in Task 12; here we just
                    // pass through the flat label and set the flag.
                    label: label.clone(),
                    label_angle: angle,
                    elided: needs_elide,
                }
            })
            .collect();
        let warning = if elided_count > 0 {
            Some(XAxisWarning::LabelsElided { count: elided_count })
        } else {
            None
        };
        (ticks, warning)
    };
```

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::axis
```

Expected: 3 new tests pass (10 total). The "LabelsElided count" test from Task 12 is not yet present; the rotation-only-no-elision test should pass because `cos(45°) * 95 ≈ 67.18 < 100`.

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/layout/axis.rs
git commit -m "feat(layout): x-axis rotation policy

Collision = any label_w > slot * (1 - LABEL_OVERLAP_TOLERANCE).
Rotates to label_angle_override or DEFAULT_LABEL_ANGLE (-45°).
After rotation, projects width by |cos(angle)| and flags any
label still exceeding slot_w as elided. String-level ellipsis
truncation lands in Task 12."
```

---

### Task 12: x-axis collision policy — string elision

**Files:**
- Modify: `crates/ferrum-core/src/layout/axis.rs`

- [ ] **Step 1: Write failing tests**

Append to the `axis.rs` test module:

```rust
    #[test]
    fn x_axis_elides_with_ellipsis_when_rotated_still_collides() {
        // panel_w=200, 20 labels → slot=10. width=80 each. After -45° rotation:
        // 80 * cos(45°) ≈ 56.57, still > 10 → elide. Final label fits in 10/cos(45°) ≈ 14.14.
        // For mock width = 10 per char, max chars = floor(14.14 / 10) = 1.
        // Plus ellipsis "…" at 10px = +1 char width. So elided label = "…" alone.
        let input = AxisInput {
            orient: AxisOrient::Bottom,
            title: None,
            tick_labels: (0..20).map(|i| format!("Label_{}", i)).collect(),
            label_angle_override: None,
        };
        let panel_area = Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 };
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let (axis, warning) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        for t in &axis.ticks {
            assert_eq!(t.label_angle, -45.0);
            assert!(t.elided, "expected all 20 labels to be elided");
            assert!(t.label.ends_with('…'), "expected ellipsis suffix; got {:?}", t.label);
        }
        match warning {
            Some(XAxisWarning::LabelsElided { count }) => assert_eq!(count, 20),
            other => panic!("expected LabelsElided{{count: 20}}, got {:?}", other),
        }
    }

    #[test]
    fn x_axis_elision_unicode_safe() {
        // Use multi-byte chars to ensure prefix slicing handles unicode.
        let input = AxisInput {
            orient: AxisOrient::Bottom,
            title: None,
            tick_labels: vec!["héllo wörld".into(); 20],
            label_angle_override: None,
        };
        let panel_area = Rect { x: 0.0, y: 0.0, w: 200.0, h: 200.0 };
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let (axis, _) = layout_x_axis(&input, panel_area, 0, 11.0, 13.0, 4.0, &m);
        for t in &axis.ticks {
            assert!(t.elided);
            // Final label must remain valid UTF-8 (no panic on slice boundary).
            assert!(t.label.is_char_boundary(t.label.len()));
        }
    }
```

- [ ] **Step 2: Add the elision helper and wire it into `layout_x_axis`**

In `axis.rs`, add this function above `layout_x_axis`:

```rust
/// Truncate `label` by char prefix until the measured width plus the ellipsis
/// width fits in `max_width`. Returns the truncated label with "…" appended.
/// If even "…" alone exceeds max_width, returns "…" anyway (caller is already
/// in a degenerate state).
fn elide_to_fit(
    label: &str,
    max_width: f64,
    font_size: f64,
    metrics: &dyn TextMetrics,
) -> String {
    let ellipsis = '…';
    let ellipsis_w = metrics.measure_width(&ellipsis.to_string(), font_size);
    if ellipsis_w >= max_width {
        return ellipsis.to_string();
    }
    let budget = max_width - ellipsis_w;
    let mut out = String::new();
    for ch in label.chars() {
        let mut tentative = out.clone();
        tentative.push(ch);
        if metrics.measure_width(&tentative, font_size) > budget {
            break;
        }
        out = tentative;
    }
    out.push(ellipsis);
    out
}
```

In `layout_x_axis`, replace the elision-flag-only block with elision that mutates the label string. Replace this block:

```rust
                let needs_elide = any_still_colliding && (w * cos_factor > slot_w);
                if needs_elide { elided_count += 1; }
                TickLayout {
                    position: panel_area.x + (i as f64 + 0.5) * slot_w,
                    // Actual string elision happens in Task 12; here we just
                    // pass through the flat label and set the flag.
                    label: label.clone(),
                    label_angle: angle,
                    elided: needs_elide,
                }
```

with:

```rust
                let needs_elide = any_still_colliding && (w * cos_factor > slot_w);
                let final_label = if needs_elide {
                    elided_count += 1;
                    // Available pixel budget for the rotated label projection is slot_w;
                    // the actual measured width budget is slot_w / cos(|angle|).
                    let budget = slot_w / cos_factor.max(1e-6);
                    elide_to_fit(label, budget, label_font_size, metrics)
                } else {
                    label.clone()
                };
                TickLayout {
                    position: panel_area.x + (i as f64 + 0.5) * slot_w,
                    label: final_label,
                    label_angle: angle,
                    elided: needs_elide,
                }
```

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::axis
```

Expected: 2 new tests pass (12 total).

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/layout/axis.rs
git commit -m "feat(layout): x-axis label elision with ellipsis

elide_to_fit truncates by char prefix until measured width + 1
ellipsis fits the rotated-budget = slot_w / cos(|angle|). Walks
chars (not bytes) so multi-byte UTF-8 stays valid. Sets elided=true
and emits XAxisWarning::LabelsElided{count} when fired."
```

---

### Task 13: `legend.rs` types

**Files:**
- Modify: `crates/ferrum-core/src/layout/legend.rs`

- [ ] **Step 1: Write the types + serde tests**

```rust
//! Legend input (caller-supplied entries) and output (engine-computed rects).

use serde::{Deserialize, Serialize};

use super::geometry::Rect;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LegendOrient {
    Right,
    Left,
    Top,
    Bottom,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LegendDirection {
    Vertical,
    Horizontal,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SymbolKind {
    Square,
    Circle,
    Line,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LegendEntry {
    pub label: String,
    pub symbol: SymbolKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegendLayout {
    pub rect: Rect,
    pub orient: LegendOrient,
    pub direction: LegendDirection,
    pub entries: Vec<LegendEntryLayout>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegendEntryLayout {
    pub label: String,
    pub label_anchor_x: f64,
    pub label_anchor_y: f64,
    pub symbol_anchor_x: f64,
    pub symbol_anchor_y: f64,
    pub symbol_kind: SymbolKind,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legend_layout_round_trip() {
        let l = LegendLayout {
            rect: Rect { x: 500.0, y: 50.0, w: 100.0, h: 300.0 },
            orient: LegendOrient::Right,
            direction: LegendDirection::Vertical,
            entries: vec![LegendEntryLayout {
                label: "A".into(),
                label_anchor_x: 520.0,
                label_anchor_y: 70.0,
                symbol_anchor_x: 510.0,
                symbol_anchor_y: 70.0,
                symbol_kind: SymbolKind::Circle,
            }],
        };
        let json = serde_json::to_string(&l).unwrap();
        let parsed: LegendLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, l);
        assert!(json.contains(r#""orient":"right""#));
        assert!(json.contains(r#""symbol_kind":"circle""#));
    }
}
```

- [ ] **Step 2: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::legend
```

Expected: 1 test passes.

- [ ] **Step 3: Commit**

```bash
git add crates/ferrum-core/src/layout/legend.rs
git commit -m "feat(layout): legend input + output types

LegendOrient (right|left|top|bottom), LegendDirection (vertical|
horizontal), SymbolKind (square|circle|line). LegendEntry is the
caller-supplied input; LegendLayout / LegendEntryLayout are the
engine-computed output."
```

---

### Task 14: `layout_legend` — placement on all four orients + overflow

**Files:**
- Modify: `crates/ferrum-core/src/layout/legend.rs`

- [ ] **Step 1: Write failing tests**

Append to the `legend.rs` test module:

```rust
    use crate::layout::text_metrics::{fixed_width, MockMetrics};

    fn mock(per_char_px: f64) -> MockMetrics<impl Fn(&str, f64) -> f64> {
        MockMetrics { measure: fixed_width(per_char_px), line_h_factor: 1.2 }
    }

    fn entries(n: usize, label_chars: usize) -> Vec<LegendEntry> {
        (0..n)
            .map(|i| LegendEntry {
                label: "X".repeat(label_chars).chars().chain(format!("{i}").chars()).collect(),
                symbol: SymbolKind::Circle,
            })
            .collect()
    }

    #[test]
    fn legend_size_right_orient_vertical() {
        // Vertical right legend: width = symbol(12) + symbol_padding(4) + max_label_w + outer_pad(8 each side)
        // height  = n_entries * (line_height) + outer_pad * 2
        let es = vec![
            LegendEntry { label: "abcd".into(), symbol: SymbolKind::Circle }, // 4ch
            LegendEntry { label: "abcdef".into(), symbol: SymbolKind::Circle }, // 6ch
        ];
        let m = mock(10.0);
        let size = estimate_legend_size(&es, LegendOrient::Right, 11.0, &m);
        // max_label_w = 6 * 10 = 60; symbol_w + sep = 12 + 4 = 16; outer pad 8+8=16.
        // total width = 16 + 60 + 16 = 92. height = 2 * 13.2 + 16 = ... line_h = 11*1.2=13.2.
        assert!((size.width - 92.0).abs() < 1e-6);
        assert!((size.height - (2.0 * 13.2 + 16.0)).abs() < 1e-6);
    }

    #[test]
    fn legend_layout_right_does_not_overlap_inner() {
        // viewport_inner = 600x400, legend on right with 3 entries of 4 chars at 10px/char.
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Right, inner, 11.0, &m);
        let legend = legend.expect("legend should be Some");
        assert_eq!(legend.orient, LegendOrient::Right);
        assert_eq!(legend.direction, LegendDirection::Vertical);
        // legend.rect.x must be inner.x + plot_inner.w (no overlap).
        assert!((legend.rect.x - (plot_inner.x + plot_inner.w)).abs() < 1e-6);
        // plot_inner.w must be < inner.w (some space reserved).
        assert!(plot_inner.w < inner.w);
        assert_eq!(plot_inner.h, inner.h);
    }

    #[test]
    fn legend_layout_left_orient() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Left, inner, 11.0, &m);
        let legend = legend.unwrap();
        assert_eq!(legend.rect.x, inner.x);
        // plot_inner starts at inner.x + legend.rect.w.
        assert!((plot_inner.x - (inner.x + legend.rect.w)).abs() < 1e-6);
    }

    #[test]
    fn legend_layout_top_orient_horizontal() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Top, inner, 11.0, &m);
        let legend = legend.unwrap();
        assert_eq!(legend.direction, LegendDirection::Horizontal);
        assert_eq!(legend.rect.y, inner.y);
        // plot_inner starts below the legend.
        assert!((plot_inner.y - (inner.y + legend.rect.h)).abs() < 1e-6);
    }

    #[test]
    fn legend_layout_bottom_orient_horizontal() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let es = entries(3, 4);
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&es, LegendOrient::Bottom, inner, 11.0, &m);
        let legend = legend.unwrap();
        assert_eq!(legend.direction, LegendDirection::Horizontal);
        assert!((legend.rect.y - (inner.y + plot_inner.h)).abs() < 1e-6);
    }

    #[test]
    fn legend_layout_empty_entries_returns_none_and_inner_unchanged() {
        let inner = Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 };
        let m = mock(10.0);
        let (legend, plot_inner) = layout_legend(&[], LegendOrient::Right, inner, 11.0, &m);
        assert!(legend.is_none());
        assert_eq!(plot_inner, inner);
    }

    #[test]
    fn legend_layout_overflow_drops_entries() {
        // viewport 200x100 — small enough to overflow with 50 entries.
        let inner = Rect { x: 0.0, y: 0.0, w: 200.0, h: 100.0 };
        let es = entries(50, 4);
        let m = mock(10.0);
        let (legend, _) = layout_legend(&es, LegendOrient::Right, inner, 11.0, &m);
        let legend = legend.unwrap();
        // At line_height 13.2 + 4 row_pad = 17.2 px per row, 100/17.2 = ~5 rows.
        // Minus 16 outer pad → (100-16)/17.2 ≈ 4.88 → 4 entries fit.
        assert!(legend.entries.len() < 50, "expected overflow drop; got {} entries", legend.entries.len());
    }
```

- [ ] **Step 2: Implement `estimate_legend_size`, `layout_legend`, internal helpers**

Add to `legend.rs` (above the `#[cfg(test)]` module):

```rust
use super::text_metrics::TextMetrics;

const SYMBOL_WIDTH: f64 = 12.0;
const SYMBOL_LABEL_GAP: f64 = 4.0;
const LEGEND_OUTER_PAD: f64 = 8.0;
const LEGEND_ENTRY_ROW_PAD: f64 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegendSize {
    pub width: f64,
    pub height: f64,
}

pub fn estimate_legend_size(
    entries: &[LegendEntry],
    orient: LegendOrient,
    label_font_size: f64,
    metrics: &dyn TextMetrics,
) -> LegendSize {
    let line_h = metrics.line_height(label_font_size);
    let max_label_w = entries
        .iter()
        .map(|e| metrics.measure_width(&e.label, label_font_size))
        .fold(0.0_f64, f64::max);
    let n = entries.len() as f64;

    match orient {
        LegendOrient::Right | LegendOrient::Left => {
            let entry_w = SYMBOL_WIDTH + SYMBOL_LABEL_GAP + max_label_w;
            let width = entry_w + 2.0 * LEGEND_OUTER_PAD;
            let height = if entries.is_empty() {
                0.0
            } else {
                n * line_h + (n - 1.0).max(0.0) * LEGEND_ENTRY_ROW_PAD + 2.0 * LEGEND_OUTER_PAD
            };
            LegendSize { width, height }
        }
        LegendOrient::Top | LegendOrient::Bottom => {
            // Horizontal direction — one row of entries.
            let entry_w = SYMBOL_WIDTH + SYMBOL_LABEL_GAP + max_label_w;
            let width = if entries.is_empty() {
                0.0
            } else {
                n * entry_w + (n - 1.0).max(0.0) * LEGEND_ENTRY_ROW_PAD + 2.0 * LEGEND_OUTER_PAD
            };
            let height = line_h + 2.0 * LEGEND_OUTER_PAD;
            LegendSize { width, height }
        }
    }
}

/// Compute the legend rect on the given orient and the remaining inner rect for
/// the rest of the chart. Returns `(None, inner)` for empty entries.
/// Returns `(Some(legend_with_dropped_entries), reduced_inner)` if the legend
/// overflows the available strip.
pub fn layout_legend(
    entries: &[LegendEntry],
    orient: LegendOrient,
    inner: Rect,
    label_font_size: f64,
    metrics: &dyn TextMetrics,
) -> (Option<LegendLayout>, Rect) {
    if entries.is_empty() {
        return (None, inner);
    }
    let size = estimate_legend_size(entries, orient, label_font_size, metrics);

    // Reserve the strip on the requested side. Cap at half the inner dim so a
    // pathologically wide legend cannot consume the entire chart.
    let (legend_rect, plot_inner) = match orient {
        LegendOrient::Right => {
            let w = size.width.min(inner.w * 0.5);
            let (strip, rest) = inner.split_right(w);
            (strip, rest)
        }
        LegendOrient::Left => {
            let w = size.width.min(inner.w * 0.5);
            let (strip, rest) = inner.split_left(w);
            (strip, rest)
        }
        LegendOrient::Top => {
            let h = size.height.min(inner.h * 0.5);
            let (strip, rest) = inner.split_top(h);
            (strip, rest)
        }
        LegendOrient::Bottom => {
            let h = size.height.min(inner.h * 0.5);
            let (strip, rest) = inner.split_bottom(h);
            (strip, rest)
        }
    };

    let direction = match orient {
        LegendOrient::Right | LegendOrient::Left => LegendDirection::Vertical,
        LegendOrient::Top | LegendOrient::Bottom => LegendDirection::Horizontal,
    };

    // Lay entries out in the strip. If they overflow, drop the tail.
    let line_h = metrics.line_height(label_font_size);
    let entries_laid_out: Vec<LegendEntryLayout> = match direction {
        LegendDirection::Vertical => {
            // Stack top-to-bottom inside legend_rect with outer pad.
            let avail_h = (legend_rect.h - 2.0 * LEGEND_OUTER_PAD).max(0.0);
            let row_pitch = line_h + LEGEND_ENTRY_ROW_PAD;
            let max_rows = if row_pitch > 0.0 {
                ((avail_h + LEGEND_ENTRY_ROW_PAD) / row_pitch).floor() as usize
            } else {
                0
            };
            let n_fit = entries.len().min(max_rows);
            entries
                .iter()
                .take(n_fit)
                .enumerate()
                .map(|(i, e)| {
                    let y = legend_rect.y + LEGEND_OUTER_PAD + (i as f64) * row_pitch + line_h / 2.0;
                    let symbol_x = legend_rect.x + LEGEND_OUTER_PAD + SYMBOL_WIDTH / 2.0;
                    let label_x = legend_rect.x + LEGEND_OUTER_PAD + SYMBOL_WIDTH + SYMBOL_LABEL_GAP;
                    LegendEntryLayout {
                        label: e.label.clone(),
                        label_anchor_x: label_x,
                        label_anchor_y: y,
                        symbol_anchor_x: symbol_x,
                        symbol_anchor_y: y,
                        symbol_kind: e.symbol,
                    }
                })
                .collect()
        }
        LegendDirection::Horizontal => {
            // Single row left-to-right inside legend_rect.
            let avail_w = (legend_rect.w - 2.0 * LEGEND_OUTER_PAD).max(0.0);
            let max_label_w = entries
                .iter()
                .map(|e| metrics.measure_width(&e.label, label_font_size))
                .fold(0.0_f64, f64::max);
            let entry_w = SYMBOL_WIDTH + SYMBOL_LABEL_GAP + max_label_w;
            let pitch = entry_w + LEGEND_ENTRY_ROW_PAD;
            let max_n = if pitch > 0.0 {
                ((avail_w + LEGEND_ENTRY_ROW_PAD) / pitch).floor() as usize
            } else {
                0
            };
            let n_fit = entries.len().min(max_n);
            entries
                .iter()
                .take(n_fit)
                .enumerate()
                .map(|(i, e)| {
                    let entry_x = legend_rect.x + LEGEND_OUTER_PAD + (i as f64) * pitch;
                    let cy = legend_rect.y + legend_rect.h / 2.0;
                    let symbol_x = entry_x + SYMBOL_WIDTH / 2.0;
                    let label_x = entry_x + SYMBOL_WIDTH + SYMBOL_LABEL_GAP;
                    LegendEntryLayout {
                        label: e.label.clone(),
                        label_anchor_x: label_x,
                        label_anchor_y: cy,
                        symbol_anchor_x: symbol_x,
                        symbol_anchor_y: cy,
                        symbol_kind: e.symbol,
                    }
                })
                .collect()
        }
    };

    let legend = LegendLayout {
        rect: legend_rect,
        orient,
        direction,
        entries: entries_laid_out,
    };
    (Some(legend), plot_inner)
}
```

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::legend
```

Expected: 7 new tests pass (8 total).

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/layout/legend.rs
git commit -m "feat(layout): legend size estimation + 4-orient placement + overflow

estimate_legend_size returns (width, height) given orient; vertical
for right/left, horizontal for top/bottom. layout_legend reserves
the strip via Rect::split_*, returns the rect plus reduced inner.
Drops tail entries when the strip cannot fit them; caller emits
LayoutWarning::LegendOverflowed."
```

---

### Task 15: `LayoutResult`, `LayoutError`, `LayoutWarning` types

**Files:**
- Modify: `crates/ferrum-core/src/layout/mod.rs`

- [ ] **Step 1: Append types to `mod.rs`**

Edit `crates/ferrum-core/src/layout/mod.rs` and append after the constants:

```rust
use serde::{Deserialize, Serialize};

pub use self::axis::{
    AxesInput, AxisInput, AxisLayout, AxisOrient, AxisTitleLayout, TickLayout,
};
pub use self::facet::{FacetGroup, FacetMode, FacetSpec};
pub use self::geometry::{Inset, Rect, Viewport};
pub use self::legend::{
    LegendDirection, LegendEntry, LegendEntryLayout, LegendLayout, LegendOrient, SymbolKind,
};
pub use self::panel::{FacetKey, PanelLayout};
pub use self::text_metrics::{HeuristicMetrics, TextMetrics};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayoutResult {
    pub viewport: Rect,
    pub panels: Vec<PanelLayout>,
    pub axes: Vec<AxisLayout>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub legend: Option<LegendLayout>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<LayoutWarning>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LayoutWarning {
    PanelCollapsed { panel_index: usize },
    LabelsElided { axis: usize, count: u32 },
    LegendOverflowed { entries_dropped: u32 },
    PanelsDropped { count: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum LayoutError {
    InvalidViewport { width: f64, height: f64 },
    InvalidFacetSpec(String),
    PaddingExceedsViewport { padding: f64, viewport_dim: f64 },
    EmptyFacetGroups,
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutError::InvalidViewport { width, height } =>
                write!(f, "invalid viewport: width={width}, height={height} (both must be > 0)"),
            LayoutError::InvalidFacetSpec(s) =>
                write!(f, "invalid facet spec: {s}"),
            LayoutError::PaddingExceedsViewport { padding, viewport_dim } =>
                write!(f, "padding {padding} exceeds viewport dimension {viewport_dim}"),
            LayoutError::EmptyFacetGroups =>
                write!(f, "facet specified but facet_groups input is empty"),
        }
    }
}

impl std::error::Error for LayoutError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_result_round_trip_empty() {
        let r = LayoutResult {
            viewport: Rect { x: 0.0, y: 0.0, w: 600.0, h: 400.0 },
            panels: vec![],
            axes: vec![],
            legend: None,
            warnings: vec![],
        };
        let json = serde_json::to_string(&r).unwrap();
        let parsed: LayoutResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, r);
        assert!(!json.contains("legend"));
        assert!(!json.contains("warnings"));
    }

    #[test]
    fn layout_warning_round_trip_each_variant() {
        for w in [
            LayoutWarning::PanelCollapsed { panel_index: 2 },
            LayoutWarning::LabelsElided { axis: 0, count: 5 },
            LayoutWarning::LegendOverflowed { entries_dropped: 3 },
            LayoutWarning::PanelsDropped { count: 1 },
        ] {
            let json = serde_json::to_string(&w).unwrap();
            let parsed: LayoutWarning = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, w);
        }
    }
}
```

- [ ] **Step 2: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout
```

Expected: 2 new tests pass plus all existing layout tests (~22 cumulative so far).

- [ ] **Step 3: Commit**

```bash
git add crates/ferrum-core/src/layout/mod.rs
git commit -m "feat(layout): LayoutResult, LayoutError, LayoutWarning types

Top-level result type with serde round-trip and snake_case tagged
warnings. LayoutError implements Display + Error manually
(no thiserror dep). Re-exports submodule types for downstream use."
```

---

### Task 16: `compute_layout` — orchestration for single-chart (no facet)

**Files:**
- Modify: `crates/ferrum-core/src/layout/mod.rs`

- [ ] **Step 1: Write failing tests**

Append to the `mod.rs` test module:

```rust
    use crate::layout::axis::{AxesInput, AxisInput, AxisOrient};
    use crate::layout::facet::FacetGroup;
    use crate::layout::text_metrics::{fixed_width, MockMetrics};
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    fn minimal_chart_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "a".into(), type_: None }),
                y: Some(EncodingSpec { field: "b".into(), type_: None }),
            },
            transforms: Vec::new(),
            facet: None,
        }
    }

    fn dummy_axes() -> AxesInput {
        AxesInput {
            x: AxisInput {
                orient: AxisOrient::Bottom,
                title: None,
                tick_labels: vec!["0".into(), "1".into(), "2".into(), "3".into()],
                label_angle_override: None,
            },
            y: AxisInput {
                orient: AxisOrient::Left,
                title: None,
                tick_labels: vec!["0".into(), "5".into(), "10".into()],
                label_angle_override: None,
            },
        }
    }

    fn default_theme_inputs() -> ThemeInputs {
        ThemeInputs::default()
    }

    #[test]
    fn compute_layout_single_chart_no_facet_no_legend() {
        let spec = minimal_chart_spec();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            viewport,
            &axes,
            &[],
            &[],
            &m,
        )
        .expect("layout should succeed on minimal spec");

        assert_eq!(result.viewport, viewport.into_rect());
        assert_eq!(result.panels.len(), 1);
        assert_eq!(result.axes.len(), 2); // bottom + left
        assert!(result.legend.is_none());
        assert!(result.warnings.is_empty());

        let panel = &result.panels[0];
        assert!(panel.plot_area.w > 0.0 && panel.plot_area.h > 0.0);
        assert_eq!(panel.row, 0);
        assert_eq!(panel.col, 0);
        assert!(panel.facet_key.is_none());
    }

    #[test]
    fn compute_layout_invalid_viewport_errors() {
        let spec = minimal_chart_spec();
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let err = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 0.0, height: 400.0 },
            &axes,
            &[],
            &[],
            &m,
        )
        .unwrap_err();
        match err {
            LayoutError::InvalidViewport { width, .. } => assert_eq!(width, 0.0),
            other => panic!("expected InvalidViewport, got {:?}", other),
        }
    }

    #[test]
    fn compute_layout_padding_exceeds_viewport_errors() {
        let spec = minimal_chart_spec();
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let theme = ThemeInputs { padding: 100.0, ..ThemeInputs::default() };
        let err = compute_layout(
            &spec,
            &theme,
            Viewport { width: 50.0, height: 50.0 },
            &axes,
            &[],
            &[],
            &m,
        )
        .unwrap_err();
        match err {
            LayoutError::PaddingExceedsViewport { .. } => {}
            other => panic!("expected PaddingExceedsViewport, got {:?}", other),
        }
    }

    #[test]
    fn compute_layout_serde_round_trip() {
        let spec = minimal_chart_spec();
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(10.0), line_h_factor: 1.2 };
        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 600.0, height: 400.0 },
            &axes,
            &[],
            &[],
            &m,
        )
        .unwrap();
        let json = serde_json::to_string(&result).unwrap();
        let parsed: LayoutResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, result);
    }
```

- [ ] **Step 2: Implement `ThemeInputs` + `compute_layout`**

Add to `mod.rs` (above the test module):

```rust
/// Theme fields actually read by Phase 6. Keeps the layout engine decoupled
/// from a full Theme type (which lives in Phase 8 grammar). Phase 7+ will
/// translate `ferrum.Theme` into this shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThemeInputs {
    pub padding: f64,
    pub column_padding: f64,
    pub row_padding: f64,
    pub axis_title_padding: f64,
    pub label_font_size: f64,
    pub title_font_size: f64,
    pub legend_orient: LegendOrient,
}

impl Default for ThemeInputs {
    fn default() -> Self {
        Self {
            padding: DEFAULT_PADDING,
            column_padding: DEFAULT_PADDING,
            row_padding: DEFAULT_PADDING,
            axis_title_padding: DEFAULT_AXIS_TITLE_PADDING,
            label_font_size: DEFAULT_LABEL_FONT_SIZE,
            title_font_size: DEFAULT_TITLE_FONT_SIZE,
            legend_orient: LegendOrient::Right,
        }
    }
}

pub fn compute_layout(
    spec: &crate::spec::chart::ChartSpec,
    theme: &ThemeInputs,
    viewport: Viewport,
    axes: &AxesInput,
    facet_groups: &[FacetGroup],
    legend_entries: &[LegendEntry],
    metrics: &dyn TextMetrics,
) -> Result<LayoutResult, LayoutError> {
    // 1. Validate inputs.
    if viewport.width <= 0.0 || viewport.height <= 0.0 {
        return Err(LayoutError::InvalidViewport {
            width: viewport.width,
            height: viewport.height,
        });
    }
    if let Some(facet) = &spec.facet {
        match &facet.mode {
            FacetMode::Wrap { ncols } if *ncols == 0 => {
                return Err(LayoutError::InvalidFacetSpec("ncols must be > 0".into()));
            }
            FacetMode::Grid { nrows, ncols } if *nrows == 0 || *ncols == 0 => {
                return Err(LayoutError::InvalidFacetSpec("nrows and ncols must be > 0".into()));
            }
            _ => {}
        }
        if facet_groups.is_empty() {
            return Err(LayoutError::EmptyFacetGroups);
        }
    }

    // 2. Apply outer padding.
    let viewport_rect = viewport.into_rect();
    let inner = viewport_rect.shrink(Inset::uniform(theme.padding));
    if inner.w <= 0.0 || inner.h <= 0.0 {
        // Use the smaller viewport dim for the error message.
        let dim = viewport.width.min(viewport.height);
        return Err(LayoutError::PaddingExceedsViewport {
            padding: theme.padding,
            viewport_dim: dim,
        });
    }

    // 3. Reserve legend strip.
    let (legend_layout, inner_after_legend) = legend::layout_legend(
        legend_entries,
        theme.legend_orient,
        inner,
        theme.label_font_size,
        metrics,
    );
    let legend_dropped = legend_entries
        .len()
        .saturating_sub(legend_layout.as_ref().map_or(0, |l| l.entries.len()))
        as u32;

    // 4 + 5. Reserve y-axis title gutter + label band; reserve x-axis label band.
    let y_title_gutter = axis::compute_y_title_width(
        &axes.y,
        theme.title_font_size,
        theme.axis_title_padding,
        metrics,
    );
    let y_label_band = axis::compute_y_label_band_width(&axes.y, theme.label_font_size, metrics);
    let x_label_band = metrics.line_height(theme.label_font_size);
    let x_title_gutter = if axes.x.title.is_some() {
        metrics.line_height(theme.title_font_size) + theme.axis_title_padding
    } else {
        0.0
    };

    let plot_region = inner_after_legend.shrink(Inset {
        top: 0.0,
        right: 0.0,
        bottom: x_label_band + x_title_gutter,
        left: y_label_band + y_title_gutter,
    });

    // 6. Split into facet cells (or a single panel).
    let mut panels: Vec<PanelLayout> = Vec::new();
    let mut warnings: Vec<LayoutWarning> = Vec::new();
    if legend_dropped > 0 {
        warnings.push(LayoutWarning::LegendOverflowed { entries_dropped: legend_dropped });
    }

    let panel_rects: Vec<(u32, u32, Rect, Option<FacetKey>)> = if let Some(facet) = &spec.facet {
        let n_panels = facet_groups.len() as u32;
        let (gx, gy) = facet
            .spacing
            .map(|s| (s, s))
            .unwrap_or((theme.column_padding, theme.row_padding));
        let grid = match facet.mode {
            FacetMode::Wrap { ncols } => {
                facet::FacetGrid::compute_wrap(ncols, n_panels, plot_region, gx, gy)
            }
            FacetMode::Grid { nrows, ncols } => {
                facet::FacetGrid::compute_grid(nrows, ncols, n_panels, plot_region, gx, gy)
            }
        };
        if grid.dropped_count() > 0 {
            warnings.push(LayoutWarning::PanelsDropped { count: grid.dropped_count() });
        }
        grid.panel_positions()
            .into_iter()
            .enumerate()
            .map(|(i, (row, col))| {
                let rect = grid.cell_rect(row, col);
                let key = facet_groups.get(i).map(|g| g.key.clone());
                (row, col, rect, key)
            })
            .collect()
    } else {
        vec![(0, 0, plot_region, None)]
    };

    // 7. Per-panel: clamp degenerate rects, collect axes.
    let mut axis_layouts: Vec<AxisLayout> = Vec::new();
    for (panel_index, (row, col, mut rect, facet_key)) in panel_rects.into_iter().enumerate() {
        if rect.w <= MIN_PANEL_DIM || rect.h <= MIN_PANEL_DIM {
            warnings.push(LayoutWarning::PanelCollapsed { panel_index });
            rect = Rect::ZERO;
        }
        panels.push(PanelLayout { plot_area: rect, facet_key, row, col });

        if rect != Rect::ZERO {
            // y-axis (left)
            let y_axis = axis::layout_y_axis(
                &axes.y,
                rect,
                panel_index,
                theme.label_font_size,
                theme.title_font_size,
                theme.axis_title_padding,
                metrics,
            );
            axis_layouts.push(y_axis);

            // x-axis (bottom)
            let (x_axis, xwarn) = axis::layout_x_axis(
                &axes.x,
                rect,
                panel_index,
                theme.label_font_size,
                theme.title_font_size,
                theme.axis_title_padding,
                metrics,
            );
            if let Some(axis::XAxisWarning::LabelsElided { count }) = xwarn {
                warnings.push(LayoutWarning::LabelsElided {
                    axis: axis_layouts.len(), // index this x-axis is about to occupy
                    count,
                });
            }
            axis_layouts.push(x_axis);
        }
    }

    Ok(LayoutResult {
        viewport: viewport_rect,
        panels,
        axes: axis_layouts,
        legend: legend_layout,
        warnings,
    })
}
```

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout
```

Expected: 4 new tests pass; all earlier layout tests still pass.

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/layout/mod.rs
git commit -m "feat(layout): compute_layout orchestration — single-chart path

Validates inputs (LayoutError on structural failures), reserves
outer padding + legend strip + axis gutters, splits into facet
cells (single rect when no facet), runs per-panel y-axis + x-axis
layout, collects warnings for overflow/elision/collapse. Theme
fields read are bundled into ThemeInputs (Phase 8 will map from
the user-facing Theme)."
```

---

### Task 17: `compute_layout` — faceted with legend, end-to-end

**Files:**
- Modify: `crates/ferrum-core/src/layout/mod.rs`

- [ ] **Step 1: Write failing tests**

Append to the `mod.rs` test module:

```rust
    use crate::layout::facet::{FacetMode, FacetSpec};
    use crate::layout::legend::{LegendEntry, SymbolKind};
    use crate::layout::panel::FacetKey;

    fn faceted_spec(ncols: u32) -> ChartSpec {
        let mut s = minimal_chart_spec();
        s.facet = Some(FacetSpec {
            field: "species".into(),
            mode: FacetMode::Wrap { ncols },
            spacing: None,
        });
        s
    }

    fn three_groups() -> Vec<FacetGroup> {
        vec![
            FacetGroup { key: FacetKey { field: "species".into(), value: "setosa".into() }, n_rows: 50 },
            FacetGroup { key: FacetKey { field: "species".into(), value: "versicolor".into() }, n_rows: 50 },
            FacetGroup { key: FacetKey { field: "species".into(), value: "virginica".into() }, n_rows: 50 },
        ]
    }

    #[test]
    fn compute_layout_faceted_three_panels_one_legend() {
        let spec = faceted_spec(3);
        let groups = three_groups();
        let legend = vec![
            LegendEntry { label: "setosa".into(), symbol: SymbolKind::Circle },
            LegendEntry { label: "versicolor".into(), symbol: SymbolKind::Circle },
            LegendEntry { label: "virginica".into(), symbol: SymbolKind::Circle },
        ];
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(8.0), line_h_factor: 1.2 };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 800.0, height: 400.0 },
            &axes,
            &groups,
            &legend,
            &m,
        )
        .unwrap();

        assert_eq!(result.panels.len(), 3);
        assert_eq!(result.axes.len(), 6);   // 2 axes per panel × 3 panels
        assert!(result.legend.is_some());
        assert!(result.warnings.is_empty(), "unexpected warnings: {:?}", result.warnings);

        // Each panel has the matching facet key.
        assert_eq!(
            result.panels[0].facet_key.as_ref().unwrap().value,
            "setosa"
        );
        assert_eq!(
            result.panels[2].facet_key.as_ref().unwrap().value,
            "virginica"
        );
    }

    #[test]
    fn compute_layout_facet_grid_overflow_warns() {
        let mut spec = minimal_chart_spec();
        spec.facet = Some(FacetSpec {
            field: "species".into(),
            mode: FacetMode::Grid { nrows: 1, ncols: 2 },
            spacing: None,
        });
        let groups = three_groups(); // 3 groups, but grid fits only 2 → 1 dropped
        let axes = dummy_axes();
        let m = MockMetrics { measure: fixed_width(8.0), line_h_factor: 1.2 };

        let result = compute_layout(
            &spec,
            &default_theme_inputs(),
            Viewport { width: 800.0, height: 400.0 },
            &axes,
            &groups,
            &[],
            &m,
        )
        .unwrap();

        assert_eq!(result.panels.len(), 2);
        let dropped = result.warnings.iter().any(|w| matches!(
            w,
            LayoutWarning::PanelsDropped { count: 1 }
        ));
        assert!(dropped, "expected PanelsDropped(1); got {:?}", result.warnings);
    }
```

- [ ] **Step 2: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout
```

Expected: 2 new tests pass; the earlier `compute_layout_faceted` test in the spec test plan is satisfied.

- [ ] **Step 3: Confirm cumulative cargo test count**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -5
```

Expected: total passing test count ≥ 145 (baseline 121 + 24 new).

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/layout/mod.rs
git commit -m "test(layout): end-to-end faceted compute_layout

Three-panel wrap with one color legend + zero warnings.
Grid overflow path emits PanelsDropped warning. Confirms
panels carry the right facet keys from the input groups."
```

---

### Task 18: PyO3 binding — `ferrum._core.compute_layout`

**Files:**
- Modify: `crates/ferrum-core/src/layout/binding.rs`
- Modify: `crates/ferrum-core/src/lib.rs`
- Modify: `src/ferrum/_core.pyi`
- Modify: `src/ferrum/__init__.py`

- [ ] **Step 1: Write the binding**

Replace `binding.rs` placeholder with:

```rust
//! PyO3 binding: `compute_layout(spec, viewport, axes, facet_groups, legend_entries)`
//! returns a Python dict. ThemeInputs and TextMetrics are not yet exposed —
//! Phase 6 always uses HeuristicMetrics + ThemeInputs::default(); Phase 8 will
//! map ferrum.Theme into ThemeInputs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::axis::{AxesInput, AxisInput, AxisOrient};
use super::facet::FacetGroup;
use super::legend::{LegendEntry, LegendOrient, SymbolKind};
use super::panel::FacetKey;
use super::text_metrics::HeuristicMetrics;
use super::{compute_layout as compute_layout_internal, ThemeInputs, Viewport};

#[pyfunction]
#[pyo3(signature = (
    spec,
    *,
    viewport,
    x_tick_labels,
    y_tick_labels,
    x_title = None,
    y_title = None,
    facet_groups = None,
    legend_entries = None,
    legend_orient = "right",
    label_angle = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn compute_layout(
    py: Python<'_>,
    spec: &crate::spec::chart::ChartSpec,
    viewport: (f64, f64),
    x_tick_labels: Vec<String>,
    y_tick_labels: Vec<String>,
    x_title: Option<String>,
    y_title: Option<String>,
    facet_groups: Option<Vec<(String, String, u64)>>,
    legend_entries: Option<Vec<(String, String)>>,
    legend_orient: &str,
    label_angle: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let viewport = Viewport { width: viewport.0, height: viewport.1 };
    let mut theme = ThemeInputs::default();
    theme.legend_orient = parse_legend_orient(legend_orient)?;

    let axes = AxesInput {
        x: AxisInput {
            orient: AxisOrient::Bottom,
            title: x_title,
            tick_labels: x_tick_labels,
            label_angle_override: label_angle,
        },
        y: AxisInput {
            orient: AxisOrient::Left,
            title: y_title,
            tick_labels: y_tick_labels,
            label_angle_override: None,
        },
    };

    let groups: Vec<FacetGroup> = facet_groups
        .unwrap_or_default()
        .into_iter()
        .map(|(field, value, n_rows)| FacetGroup {
            key: FacetKey { field, value },
            n_rows,
        })
        .collect();

    let entries: Vec<LegendEntry> = legend_entries
        .unwrap_or_default()
        .into_iter()
        .map(|(label, kind)| {
            let symbol = match kind.as_str() {
                "circle" => SymbolKind::Circle,
                "square" => SymbolKind::Square,
                "line" => SymbolKind::Line,
                _ => SymbolKind::Circle,
            };
            LegendEntry { label, symbol }
        })
        .collect();

    let metrics = HeuristicMetrics::default();
    let result = compute_layout_internal(
        spec, &theme, viewport, &axes, &groups, &entries, &metrics,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let json = serde_json::to_string(&result)
        .map_err(|e| PyValueError::new_err(format!("internal serde error: {e}")))?;
    let json_module = py.import_bound("json")?;
    let parsed = json_module.call_method1("loads", (json,))?;
    let dict: Py<PyDict> = parsed.extract()?;
    Ok(dict)
}

fn parse_legend_orient(s: &str) -> PyResult<LegendOrient> {
    match s {
        "right" => Ok(LegendOrient::Right),
        "left" => Ok(LegendOrient::Left),
        "top" => Ok(LegendOrient::Top),
        "bottom" => Ok(LegendOrient::Bottom),
        other => Err(PyValueError::new_err(format!(
            "legend_orient must be one of right|left|top|bottom; got '{other}'"
        ))),
    }
}

```

- [ ] **Step 2: Register the pyfunction in `lib.rs`**

Edit `crates/ferrum-core/src/lib.rs`. Inside the `_core` `#[pymodule]` body, append before the closing `Ok(())`:

```rust
    m.add_function(wrap_pyfunction!(layout::binding::compute_layout, m)?)?;
```

- [ ] **Step 3: Add a `.pyi` stub**

Edit `src/ferrum/_core.pyi`. Append to the bottom (or after the `ChartSpec` stub):

```python
def compute_layout(
    spec,
    *,
    viewport: tuple[float, float],
    x_tick_labels: list[str],
    y_tick_labels: list[str],
    x_title: str | None = None,
    y_title: str | None = None,
    facet_groups: list[tuple[str, str, int]] | None = None,
    legend_entries: list[tuple[str, str]] | None = None,
    legend_orient: str = "right",
    label_angle: float | None = None,
) -> dict: ...
```

- [ ] **Step 4: Re-export from `__init__.py`**

Edit `src/ferrum/__init__.py`. Find the import block from `_core`. Append `compute_layout` to the imports and `__all__`. For example:

```python
from ferrum._core import (
    # ...existing imports...
    compute_layout,
)

__all__ = [
    # ...existing names...
    "compute_layout",
]
```

(Match the existing style — if there's no `__all__`, just adding the import is sufficient.)

- [ ] **Step 5: Build and smoke-test**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
unset CONDA_PREFIX && uv run --no-sync python -c "from ferrum._core import compute_layout, ChartSpec; s = ChartSpec(mark='point', x='a', y='b'); r = compute_layout(s, viewport=(600.0,400.0), x_tick_labels=['0','1','2'], y_tick_labels=['0','5','10']); assert 'panels' in r; assert len(r['panels']) == 1; print('OK')"
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add crates/ferrum-core/src/layout/binding.rs crates/ferrum-core/src/lib.rs src/ferrum/_core.pyi src/ferrum/__init__.py
git commit -m "feat(py): expose ferrum._core.compute_layout returning a dict

PyO3 binding takes ChartSpec + tuples for axes/groups/entries to
keep the Python signature flat. Always uses HeuristicMetrics +
ThemeInputs::default(); Phase 8 grammar will provide a richer
Theme→ThemeInputs mapping. .pyi stub + __init__ re-export added."
```

---

### Task 19: Python pytest tests

**Files:**
- Create: `tests/test_layout_engine.py`

- [ ] **Step 1: Write the tests**

```python
"""Phase 6 layout engine — Python binding surface tests.

These validate the dict shape, error mapping, and end-to-end behavior
through the binding. Numeric arithmetic is exhaustively covered on the
Rust side (`cargo test -p ferrum-core layout`); these tests only confirm
the binding wires inputs and outputs correctly.
"""
import pytest

from ferrum._core import ChartSpec, compute_layout


def _spec():
    return ChartSpec(mark="point", x="a", y="b")


def test_compute_layout_returns_dict_with_expected_keys():
    r = compute_layout(
        _spec(),
        viewport=(600.0, 400.0),
        x_tick_labels=["0", "1", "2", "3"],
        y_tick_labels=["0", "5", "10"],
    )
    assert isinstance(r, dict)
    assert "viewport" in r
    assert "panels" in r
    assert "axes" in r
    # legend skipped when None
    assert "legend" not in r or r["legend"] is None
    # warnings skipped when empty
    assert "warnings" not in r or r["warnings"] == []


def test_single_chart_one_panel_two_axes():
    r = compute_layout(
        _spec(),
        viewport=(600.0, 400.0),
        x_tick_labels=["0", "1", "2"],
        y_tick_labels=["0", "5"],
    )
    assert len(r["panels"]) == 1
    assert len(r["axes"]) == 2
    p = r["panels"][0]
    assert p["plot_area"]["w"] > 0
    assert p["plot_area"]["h"] > 0
    assert p["facet_key"] is None
    assert p["row"] == 0 and p["col"] == 0


def test_legend_present_when_entries_supplied():
    r = compute_layout(
        _spec(),
        viewport=(600.0, 400.0),
        x_tick_labels=["a", "b"],
        y_tick_labels=["0", "10"],
        legend_entries=[("setosa", "circle"), ("versicolor", "circle")],
    )
    assert r["legend"] is not None
    assert r["legend"]["orient"] == "right"
    assert len(r["legend"]["entries"]) == 2


def test_invalid_viewport_raises_value_error():
    with pytest.raises(ValueError, match="invalid viewport"):
        compute_layout(
            _spec(),
            viewport=(0.0, 400.0),
            x_tick_labels=["a"],
            y_tick_labels=["a"],
        )


def test_invalid_legend_orient_raises_value_error():
    with pytest.raises(ValueError, match="legend_orient"):
        compute_layout(
            _spec(),
            viewport=(600.0, 400.0),
            x_tick_labels=["a"],
            y_tick_labels=["a"],
            legend_orient="diagonal",
        )


def test_collision_triggers_rotation_in_axis_layout():
    # Many wide labels → x-axis collision → rotation.
    long_labels = [f"category_{i}" for i in range(20)]
    r = compute_layout(
        _spec(),
        viewport=(300.0, 400.0),  # narrow
        x_tick_labels=long_labels,
        y_tick_labels=["0", "10"],
    )
    x_axis = next(a for a in r["axes"] if a["orient"] == "bottom")
    angles = {tick["label_angle"] for tick in x_axis["ticks"]}
    assert -45.0 in angles or any(a != 0.0 for a in angles), (
        f"expected rotation; got angles {angles}"
    )
```

- [ ] **Step 2: Run pytest**

```bash
uv run pytest tests/test_layout_engine.py -v
```

Expected: 6 tests pass.

- [ ] **Step 3: Confirm cumulative pytest count**

```bash
uv run pytest 2>&1 | tail -5
```

Expected: total ≥ 78 (baseline 72 + 6 new).

- [ ] **Step 4: Commit**

```bash
git add tests/test_layout_engine.py
git commit -m "test(layout): Python binding smoke tests

6 tests covering dict shape, single-chart vs legend present,
ValueError on invalid viewport / unknown orient, rotation
trigger on narrow viewport. Numerical assertions stay on the
Rust side; these tests validate the binding wiring only."
```

---

### Task 20: Mark Phase 6 done + final verification

**Files:**
- Modify: `docs/superpowers/ferrum-phases.md`

- [ ] **Step 1: Update phases doc**

Edit `docs/superpowers/ferrum-phases.md`. In the phase table (around line 65), change the Phase 6 row from:

```markdown
| **6** | Layout engine | Constraint solver for facet sizes, legend placement, axis label collision avoidance | 3 | *(not yet written)* | pending |
```

to:

```markdown
| **6** | Layout engine | Constraint solver for facet sizes, legend placement, axis label collision avoidance | 3 | [`2026-05-09-layout-engine-design.md`](specs/2026-05-09-layout-engine-design.md) | **done** |
```

In the per-phase done-criteria section (around line 109), change every Phase 6 unchecked box from `- [ ]` to `- [x]`:

```markdown
### Phase 6 — Layout engine
- [x] Facet grid sizes computed correctly for `wrap` and `grid` facet modes
- [x] Legend placement does not overlap chart area for a 1-layer scatter plot
- [x] Axis label collision avoidance (rotation or elision) fires at a configurable threshold
- [x] `cargo test` covers basic facet layout arithmetic
```

- [ ] **Step 2: Final test sweep**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop --release
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -10
uv run pytest 2>&1 | tail -10
```

Expected:
- `cargo test`: ≥ 145 passing, 0 failing.
- `uv run pytest`: ≥ 78 passing, 0 failing.

If either falls short, **stop** and investigate before proceeding — likely a regression from the ChartSpec change in Task 5.

- [ ] **Step 3: Smoke-test the integration pattern from CLAUDE.md**

```bash
unset CONDA_PREFIX && uv run --no-sync python -c "from ferrum._core import ChartSpec; s=ChartSpec(mark='point', x='a', y='b'); assert s == ChartSpec.from_json(s.to_json()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/ferrum-phases.md
git commit -m "docs(phases): mark Phase 6 layout-engine done

All four done-criteria boxes ticked. Final test counts:
cargo test -p ferrum-core ≥ 145 (baseline 121 + 24 layout tests).
uv run pytest ≥ 78 (baseline 72 + 6 binding tests)."
```

- [ ] **Step 5: Open the merge PR (or merge directly per user pref)**

The user will decide whether to merge `feat/phase-6-layout-engine` into `main` directly or via a PR. Confirm with the user before any push or merge — Phase 5's pattern was a merge commit on main, but the user may want a PR for review.

```bash
git log --oneline main..HEAD
```

Expected: ~20 feature commits visible. Pause for user instruction.

---

## Spec coverage check

Mapping each spec section to the task that satisfies it:

| Spec section | Task |
|---|---|
| §3.2 Module layout | Task 1 |
| §3.3 ChartSpec extension | Task 5 |
| §4.1 geometry.rs | Task 2 |
| §4.2 text_metrics.rs | Task 3 |
| §4.3 panel.rs | Task 4 |
| §4.4 axis.rs | Tasks 8, 9, 10, 11, 12 |
| §4.5 legend.rs | Tasks 13, 14 |
| §4.6 facet.rs | Tasks 5, 6, 7 |
| §4.7 LayoutResult / LayoutError / LayoutWarning | Task 15 |
| §5 Input contract / ThemeInputs | Task 16 |
| §6 Algorithm | Tasks 16, 17 |
| §6.1 Constants | Task 10 |
| §7 Error policy | Tasks 16 (LayoutError → PyValueError), 14 (LegendOverflowed warning), 17 (PanelsDropped warning), 11 (LabelsElided warning), 16 (PanelCollapsed warning) |
| §8 No new deps | (no Cargo.toml changes; verified by Task 1's `maturin develop` step) |
| §9.1 Cargo tests | Tasks 2 (geometry), 3 (text_metrics), 4 (panel), 5/6/7 (facet), 8–12 (axis), 13/14 (legend), 15/16/17 (mod) |
| §9.2 Python pytest | Task 19 |
| §10 Done-criteria gate | Task 20 |
| §14 Refinements | Tasks 8 (AxesInput), 9 (y-axis no collision), 10 (uniform tick positions), 16 (single-pass orchestration) |
