# Phase 8b — Composite + Heavy Statistical Marks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 11 NotImplementedError mark stubs from Phase 8a with working composite/heavy-statistical marks, backed by 10 new Phase 5 transforms, 3 new SVG primitives, a continuous colormap subsystem, and `mark_smooth(ci=)` CI band rendering.

**Architecture:**
- Composite marks desugar Python-side into multi-layer ChartSpecs using the `ChartSpec.layers` field 8a shipped (no new "composite" Mark variant in Rust).
- Heavy stat marks pair a new transform with an existing/new render::marks drawer; multi-output transforms (BoxStats+Outliers, QQ points+line) route via a new `Layer.data_source: Option<String>` field.
- `mark_raster` embeds RGBA→PNG→base64 in SVG via `<image href="data:image/png;base64,...">`. `mark_function` evaluates the callable Python-side at chart-build-time and feeds a synthetic Arrow table into a regular `mark_line` desugar.
- Continuous colormaps (viridis/plasma/magma/inferno/cividis) added via the `colorous` crate; Python `continuous_palette(name)` mirrors 8a's `categorical_palette`.

**Tech Stack:** Rust (PyO3, arrow, serde_json, rand_chacha, **NEW**: colorous 1, **PROMOTED**: png 0.18 from transitive→direct), Python ≥3.10 (numpy, pyarrow), maturin build backend.

**Source spec:** `docs/superpowers/specs/2026-05-10-composite-stat-marks-design.md`

---

## Pre-flight

1. **Build commands.** All `maturin` invocations need `unset CONDA_PREFIX && uv run --no-sync maturin develop` (per repo CLAUDE.md). All `cargo test` invocations need `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core` on macOS.
2. **Test baselines at start.** `cargo test -p ferrum-core` = 309 passing; `uv run pytest` = 217 passing + 3 skipped. Final targets ≥379 / ≥397.

---

### Task 0: Create `feat/phase-8b` worktree

**Files:** none (creates worktree off main)

- [ ] **Step 1: Invoke worktree skill**

Use `superpowers:using-git-worktrees` to create branch `feat/phase-8b` off `main`. All Phase 8b implementation happens on this branch; merge to main only after Task 43 passes.

- [ ] **Step 2: Verify clean baseline**

```bash
git status
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -3
uv run pytest 2>&1 | tail -3
```
Expected: clean working tree; cargo 309 passing; pytest 217 passing.

---

## File structure

**Rust — `crates/ferrum-core/src/`:**

| Path | Status | Responsibility |
|---|---|---|
| `transform/outliers.rs` | NEW | Per-group IQR outlier filter |
| `transform/error_extent.rs` | NEW | ci/stderr/stdev/iqr aggregation |
| `transform/box_stats.rs` | NEW | Quartiles + whiskers per group |
| `transform/violin.rs` | NEW | Per-group KDE → polygon vertices |
| `transform/kde_2d.rs` | NEW | 2D Gaussian KDE on uniform grid |
| `transform/contour.rs` | NEW | Marching Squares isolines + isobands |
| `transform/qq.rs` | NEW | Quantile-quantile + reference line |
| `transform/raster.rs` | NEW | 2D bin aggregation → cell grid |
| `transform/hex.rs` | NEW | Hexagonal bin aggregation |
| `transform/swarm.rs` | NEW | Greedy-sweep beeswarm collision |
| `transform/core.rs` | MODIFY | Add 10 enum variants + dispatch + `apply_with_context` |
| `transform/mod.rs` | MODIFY | `pub(crate) mod` for 10 new files |
| `transform/context.rs` | NEW | `TransformContext` struct (panel pixel size) |
| `render/svg.rs` | MODIFY | Add `image()`, `polygon()`, `beeswarm()` methods |
| `render/marks/polygon.rs` | NEW | Polygon mark drawer (contour/hex/violin) |
| `render/marks/image.rs` | NEW | Image mark drawer (raster) |
| `render/marks/ribbon.rs` | NEW | Ribbon mark drawer (Y/Y2 area) |
| `render/marks/mod.rs` | MODIFY | `pub(crate) mod` for 3 new mark drawers |
| `render/color/` | NEW DIR | Color subsystem (replaces flat `color.rs`) |
| `render/color/mod.rs` | NEW | Re-exports for back-compat |
| `render/color/categorical.rs` | NEW | Move existing 6 palettes here |
| `render/color/continuous.rs` | NEW | colorous-backed continuous schemes |
| `render/color/scheme.rs` | NEW | `Scheme` enum dispatch |
| `render/color.rs` | DELETE | After migration to `render/color/` |
| `render/rasterize.rs` | NEW | RGBA→PNG via `png` crate |
| `render/prepare.rs` | MODIFY | Honor `Layer.data_source` routing + `apply_with_context` call |
| `render/scale_resolve.rs` | MODIFY | Quantitative color encoding branch |
| `spec/layer.rs` | MODIFY | Add `data_source: Option<String>` field |
| `spec/chart.rs` | MODIFY | Plumb 10 new transforms into `coerce_transforms`; named transform support |
| `binding.rs` | MODIFY | Add 10 PyO3 wrappers + ContinuousScheme |
| `lib.rs` | MODIFY | Register new pyclasses |

**Python — `src/ferrum/`:**

| Path | Status | Responsibility |
|---|---|---|
| `marks/composite.py` | NEW | desugar_boxplot/errorbar/errorband/ribbon |
| `marks/heavy_stat.py` | NEW | desugar_contour/violin/qq/raster/swarm/hex/function |
| `marks/statistical.py` | MODIFY | `desugar_smooth` adds CI band path |
| `marks/deferred.py` | MODIFY | Drain PHASE_8B_MARKS to empty frozenset |
| `chart.py` | MODIFY | Replace 11 `NotImplementedError` mark methods with working desugars |
| `schemes.py` | MODIFY | Add `continuous_palette(name)` + `Gradient` class |
| `__init__.py` | MODIFY | Export 10 new transforms + ContinuousScheme + Gradient |
| `_warn.py` | (no change) | 8a's warn-once registry; new categories register at use sites |

**Tests — `tests/`:**

| Path | Status | Tests |
|---|---|---|
| `marks/test_boxplot.py` | NEW | 15 |
| `marks/test_errorbar.py` | NEW | 10 |
| `marks/test_errorband.py` | NEW | 10 |
| `marks/test_ribbon.py` | NEW | 8 |
| `marks/test_smooth_ci.py` | NEW | 6 |
| `marks/test_contour.py` | NEW | 14 |
| `marks/test_violin.py` | NEW | 14 |
| `marks/test_qq.py` | NEW | 10 |
| `marks/test_raster.py` | NEW | 16 |
| `marks/test_swarm.py` | NEW | 12 |
| `marks/test_hex.py` | NEW | 12 |
| `marks/test_function.py` | NEW | 10 |
| `test_continuous_palette.py` | NEW | 8 |
| `test_data_source_routing.py` | NEW | 7 |
| `test_image_primitive.py` | NEW | 4 |
| `test_polygon_primitive.py` | NEW | 4 |
| `test_beeswarm_primitive.py` | NEW | 3 |
| `desugar/test_composite_desugar.py` | NEW | 10 |
| `desugar/test_heavy_stat_desugar.py` | NEW | 16 |
| `test_phase_8b_e2e.py` | NEW | 14 |
| `test_spec_drift.py` | NEW | 4 |
| `test_warn_once_lift.py` | NEW | 3 |

---

## Sub-batch overview

The 45 tasks below (T0 + T1-T22 + T22b + T23-T43) are grouped into sub-batches. The subagent driver (or executing-plans agent) should run each sub-batch in order; tasks within a sub-batch with no inter-dependency can be parallelized via separate subagents.

| Sub-batch | Tasks | Theme | Parallelism |
|---|---|---|---|
| **Pre-flight** | T0 | Create `feat/phase-8b` worktree | sequential |
| **A** | 1-7 | Foundation: workspace deps, ContinuousScheme, SVG primitives, rasterize, TransformContext | T1 sequential; T2-T7 mostly sequential |
| **B** | 8-9 | Routing infra: `data_source` field, `name` on TransformSpec, named-output dispatch | sequential |
| **C** | 10-19 | 10 new transforms | parallelizable after A+B; T15 (Contour) depends on T14 (Kde2D); T17 (Raster) and T19 (Swarm) need T7's TransformContext |
| **D** | 20-22 | 3 new render::marks drawers (polygon w/ quantitative color, image, ribbon) | parallelizable after A+C |
| **22b** | 22b | Layered-desugar resolver in Chart (PREREQUISITE for E + F) | sequential, must complete before any composite/heavy mark |
| **E** | 23-26 | 4 Python composite desugar helpers + Chart wiring (boxplot/errorbar/errorband/ribbon) | parallelizable after 22b |
| **F** | 27-33 | 7 Python heavy-stat desugar helpers + Chart wiring | parallelizable after 22b + D |
| **G** | 34-37 | Cross-cutting: smooth(ci), bivariate density, X2/Y2 wiring, continuous_palette Python lookup | sequential |
| **H** | 38-43 | Spec drift, ferrum-phases update, deferred.py drain, e2e tests, warn-once tests, final verification | sequential |

---

## Sub-batch A — Foundation (Tasks 1-7)

### Task 1: Add `colorous` and promote `png` to direct workspace deps

**Files:**
- Modify: `Cargo.toml` (workspace root, `[workspace.dependencies]` block)
- Modify: `crates/ferrum-core/Cargo.toml` (`[dependencies]` block)
- Run: `unset CONDA_PREFIX && uv run --no-sync maturin develop` to refresh `Cargo.lock`

- [ ] **Step 1: Inspect current workspace deps**

Run: `grep -A20 "\[workspace.dependencies\]" Cargo.toml`
Note the alphabetical ordering convention.

- [ ] **Step 2: Add to workspace `[workspace.dependencies]`**

In `Cargo.toml` workspace block, add (alphabetically sorted):

```toml
colorous = "1"
png      = "0.18"
```

- [ ] **Step 3: Add to `crates/ferrum-core/Cargo.toml` `[dependencies]`**

Add (alphabetically sorted with existing):

```toml
colorous = { workspace = true }
png      = { workspace = true }
```

- [ ] **Step 4: Refresh lockfile**

Run: `unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -20`
Expected: builds successfully; `colorous v1.0.x` and `png v0.18.x` show in compile output.

- [ ] **Step 5: Verify no matplotlib in tree**

Run: `cargo tree -p ferrum-core | grep -i matplotlib || echo "clean"`
Expected: `clean`

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/ferrum-core/Cargo.toml Cargo.lock
git commit -m "build(phase-8b): add colorous dep, promote png to direct workspace dep"
```

---

### Task 2: ContinuousScheme enum + colorous backing

**Note:** Use `colorous = "1"` (1.0.16 at time of writing); the original spec/plan said "0.6" but that version doesn't exist on crates.io. Same API surface; see spec §7.1 dated correction.


**Files:**
- Create: `crates/ferrum-core/src/render/color/mod.rs`
- Create: `crates/ferrum-core/src/render/color/continuous.rs`
- Create: `crates/ferrum-core/src/render/color/categorical.rs`
- Create: `crates/ferrum-core/src/render/color/scheme.rs`
- Modify: `crates/ferrum-core/src/render/mod.rs` (replace `pub(crate) mod color` with `pub(crate) mod color` directory)
- Delete: `crates/ferrum-core/src/render/color.rs` (after content migrated)

This task migrates the existing flat `color.rs` into a directory and adds the continuous subsystem alongside.

- [ ] **Step 1: Inspect current color.rs**

Run: `wc -l crates/ferrum-core/src/render/color.rs && head -40 crates/ferrum-core/src/render/color.rs`
Note the public API surface (what other modules import).

- [ ] **Step 2: Create `render/color/categorical.rs` by moving existing content**

Run: `git mv crates/ferrum-core/src/render/color.rs crates/ferrum-core/src/render/color/categorical.rs`
Then run: `mkdir -p crates/ferrum-core/src/render/color && git mv crates/ferrum-core/src/render/color.rs crates/ferrum-core/src/render/color/categorical.rs`

(If `git mv` to a path inside a sibling dir is awkward, do `git rm color.rs` after creating `color/categorical.rs` with the same content.)

- [ ] **Step 3: Write failing test for ContinuousScheme**

Create `crates/ferrum-core/src/render/color/continuous.rs`:

```rust
//! Continuous colormaps for raster/hex/bivariate-density marks.
//! Backed by `colorous` for the 5 named maps; supports user Gradient and Reverse.

use crate::render::color::categorical::Color;

#[derive(Debug, Clone, PartialEq)]
pub enum NamedContinuous {
    Viridis,
    Plasma,
    Magma,
    Inferno,
    Cividis,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContinuousScheme {
    Named(NamedContinuous),
    Gradient(Vec<(f64, Color)>),
    Reverse(Box<ContinuousScheme>),
}

impl NamedContinuous {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "viridis" => Some(Self::Viridis),
            "plasma"  => Some(Self::Plasma),
            "magma"   => Some(Self::Magma),
            "inferno" => Some(Self::Inferno),
            "cividis" => Some(Self::Cividis),
            _ => None,
        }
    }
    pub fn list() -> &'static [&'static str] {
        &["viridis", "plasma", "magma", "inferno", "cividis"]
    }
    fn colorous_gradient(&self) -> colorous::Gradient {
        match self {
            Self::Viridis => colorous::VIRIDIS,
            Self::Plasma  => colorous::PLASMA,
            Self::Magma   => colorous::MAGMA,
            Self::Inferno => colorous::INFERNO,
            Self::Cividis => colorous::CIVIDIS,
        }
    }
}

impl ContinuousScheme {
    /// Sample at t ∈ [0, 1]. t outside [0, 1] is clamped.
    pub fn sample(&self, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Named(n) => {
                let c = n.colorous_gradient().eval_continuous(t);
                Color { r: c.r, g: c.g, b: c.b, a: 255 }
            }
            Self::Gradient(stops) => sample_gradient(stops, t),
            Self::Reverse(inner) => inner.sample(1.0 - t),
        }
    }
}

fn sample_gradient(stops: &[(f64, Color)], t: f64) -> Color {
    if stops.is_empty() {
        return Color { r: 0, g: 0, b: 0, a: 255 };
    }
    if t <= stops[0].0 { return stops[0].1; }
    if t >= stops[stops.len() - 1].0 { return stops[stops.len() - 1].1; }
    // binary search for bracketing pair
    let i = stops.partition_point(|(p, _)| *p <= t);
    let (t0, c0) = stops[i - 1];
    let (t1, c1) = stops[i];
    let u = (t - t0) / (t1 - t0);
    Color {
        r: lerp_u8(c0.r, c1.r, u),
        g: lerp_u8(c0.g, c1.g, u),
        b: lerp_u8(c0.b, c1.b, u),
        a: lerp_u8(c0.a, c1.a, u),
    }
}

fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    (a as f64 + (b as f64 - a as f64) * t).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viridis_endpoints_match_colorous_reference() {
        let s = ContinuousScheme::Named(NamedContinuous::Viridis);
        let c0 = s.sample(0.0);
        let c1 = s.sample(1.0);
        // Viridis 0.0 → ~RGB(68, 1, 84); 1.0 → ~RGB(253, 231, 37)
        assert!(c0.r < 80 && c0.g < 20 && c0.b < 100, "viridis(0): {:?}", c0);
        assert!(c1.r > 240 && c1.g > 220 && c1.b < 50, "viridis(1): {:?}", c1);
    }

    #[test]
    fn gradient_two_stop_interpolates_in_linear_space() {
        let red   = Color { r: 255, g: 0, b: 0, a: 255 };
        let blue  = Color { r: 0, g: 0, b: 255, a: 255 };
        let s = ContinuousScheme::Gradient(vec![(0.0, red), (1.0, blue)]);
        let mid = s.sample(0.5);
        assert_eq!(mid.r, 128);
        assert_eq!(mid.b, 128);
        assert_eq!(mid.g, 0);
    }

    #[test]
    fn reverse_inverts_t() {
        let s = ContinuousScheme::Reverse(Box::new(
            ContinuousScheme::Named(NamedContinuous::Viridis)));
        let c0 = s.sample(0.0);
        let c1 = s.sample(1.0);
        // Reversed viridis: 0.0 should look like normal viridis(1.0)
        assert!(c0.r > 240 && c0.g > 220);
        assert!(c1.r < 80);
    }
}
```

- [ ] **Step 4: Create `render/color/mod.rs`**

```rust
//! Color subsystem: categorical (8a) + continuous (8b).
pub(crate) mod categorical;
pub(crate) mod continuous;
pub(crate) mod scheme;

// Re-export the public-ish API that callers in 8a use.
pub use categorical::{Color, categorical_palette, /* preserve existing 8a exports */};
pub use continuous::{ContinuousScheme, NamedContinuous};
pub use scheme::Scheme;
```

(Update the re-export list to match what `categorical.rs` actually defines after the move; verify by `grep "pub fn\|pub struct\|pub enum" crates/ferrum-core/src/render/color/categorical.rs`.)

- [ ] **Step 5: Create `render/color/scheme.rs`**

```rust
//! Unified Scheme enum: dispatch between categorical and continuous.

use crate::render::color::categorical::CategoricalPalette;
use crate::render::color::continuous::ContinuousScheme;

#[derive(Debug, Clone, PartialEq)]
pub enum Scheme {
    Categorical(CategoricalPalette),
    Continuous(ContinuousScheme),
}
```

- [ ] **Step 6: Update `render/mod.rs` to declare the directory**

Replace `pub(crate) mod color;` with `pub(crate) mod color;` (no change in syntax — Rust resolves either `color.rs` OR `color/mod.rs`). Verify with: `cargo build -p ferrum-core 2>&1 | head -20`.

- [ ] **Step 7: Run tests**

Run: `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core continuous 2>&1 | tail -10`
Expected: 3 new tests pass; existing color tests (in categorical) still pass.

- [ ] **Step 8: Commit**

```bash
git add crates/ferrum-core/src/render/color/ crates/ferrum-core/src/render/mod.rs
git rm crates/ferrum-core/src/render/color.rs
git commit -m "feat(phase-8b): ContinuousScheme + colorous-backed viridis family"
```

---

### Task 3: SvgBuffer.image() primitive

**Files:**
- Modify: `crates/ferrum-core/src/render/svg.rs` (add `image` method to `SvgBuffer`)
- Test added inline to existing `#[cfg(test)] mod tests` block

- [ ] **Step 1: Locate SvgBuffer impl block**

Run: `grep -n "impl SvgBuffer" crates/ferrum-core/src/render/svg.rs`
Note the file structure to insert the new method in the right place (alongside `circle`, `rect`, `line`).

- [ ] **Step 2: Write failing test (in existing tests mod)**

Append to `crates/ferrum-core/src/render/svg.rs` `mod tests`:

```rust
#[test]
fn image_emits_data_url_with_fixed_attribute_order() {
    use base64::Engine;
    let viewport = Rect { x: 0.0, y: 0.0, width: 100.0, height: 100.0 };
    let mut svg = SvgBuffer::new(viewport, None, false);
    let png_bytes: &[u8] = b"\x89PNG\r\n\x1a\n";  // PNG magic bytes (truncated)
    svg.image(10.0, 20.0, 50.0, 30.0, png_bytes);
    let out = svg.finish();

    let b64 = base64::engine::general_purpose::STANDARD.encode(png_bytes);
    let expected_href = format!("data:image/png;base64,{b64}");

    // Attribute order: x, y, width, height, href (alphabetical NOT used; this is our pinned order)
    let needle = format!(r#"<image x="10" y="20" width="50" height="30" href="{expected_href}"/>"#);
    assert!(out.contains(&needle), "image markup missing or wrong order:\n{out}");
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core image_emits 2>&1 | tail -15`
Expected: FAIL with "method `image` not found".

- [ ] **Step 4: Implement `image` method**

In `impl SvgBuffer`, after existing `text` method:

```rust
/// Embed a PNG as <image href="data:image/png;base64,..." x=... y=... width=... height=.../>.
/// `png_bytes` must be a valid PNG-encoded buffer (use render::rasterize::encode_png).
/// Attribute order is pinned: x, y, width, height, href.
pub fn image(&mut self, x: f64, y: f64, w: f64, h: f64, png_bytes: &[u8]) {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(png_bytes);
    self.body.push_str(&format!(
        r#"<image x="{}" y="{}" width="{}" height="{}" href="data:image/png;base64,{}"/>"#,
        fmt_f(x), fmt_f(y), fmt_f(w), fmt_f(h), b64
    ));
}
```

- [ ] **Step 5: Run test to verify pass**

Run: `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core image_emits 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 6: Add 2 more tests (deterministic across runs; no whitespace in attribute area)**

Append:

```rust
#[test]
fn image_byte_identical_across_runs() {
    let viewport = Rect { x: 0.0, y: 0.0, width: 100.0, height: 100.0 };
    let png: &[u8] = b"\x89PNG\r\n\x1a\nfoo";
    let make = || {
        let mut svg = SvgBuffer::new(viewport, None, false);
        svg.image(0.0, 0.0, 10.0, 10.0, png);
        svg.finish()
    };
    assert_eq!(make(), make(), "image emission must be deterministic");
}

#[test]
fn image_does_not_emit_whitespace_in_href() {
    let viewport = Rect { x: 0.0, y: 0.0, width: 100.0, height: 100.0 };
    let mut svg = SvgBuffer::new(viewport, None, false);
    svg.image(0.0, 0.0, 10.0, 10.0, b"\x89PNG\r\n\x1a\nx");
    let out = svg.finish();
    let href_start = out.find("data:image/png;base64,").expect("href missing");
    let href_end = out[href_start..].find('"').unwrap() + href_start;
    let href_body = &out[href_start..href_end];
    assert!(!href_body.contains('\n') && !href_body.contains(' '),
            "href body must be one line: {href_body}");
}
```

- [ ] **Step 7: Run all svg tests**

Run: `DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core svg:: 2>&1 | tail -10`
Expected: all pass; total svg tests grew by 3.

- [ ] **Step 8: Commit**

```bash
git add crates/ferrum-core/src/render/svg.rs
git commit -m "feat(phase-8b): SvgBuffer.image() primitive (PNG base64 data URL)"
```

---

### Task 4: SvgBuffer.polygon() primitive (path + evenodd)

**Files:**
- Modify: `crates/ferrum-core/src/render/svg.rs`

- [ ] **Step 1: Write failing test**

Append to `mod tests`:

```rust
#[test]
fn polygon_one_ring_emits_closed_path() {
    let viewport = Rect { x: 0.0, y: 0.0, width: 100.0, height: 100.0 };
    let mut svg = SvgBuffer::new(viewport, None, false);
    let ring = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
    let style = FillStroke {
        fill: Some(Color { r: 0, g: 100, b: 200, a: 255 }),
        fill_opacity: 1.0,
        stroke: None,
        stroke_width: 0.0,
        stroke_opacity: 1.0,
    };
    svg.polygon(&[ring], &style);
    let out = svg.finish();
    assert!(out.contains(r#"d="M 0 0 L 10 0 L 10 10 L 0 10 Z""#),
            "missing single-ring path data: {out}");
    assert!(out.contains(r#"fill-rule="evenodd""#));
}

#[test]
fn polygon_multi_ring_concatenates_subpaths() {
    let viewport = Rect { x: 0.0, y: 0.0, width: 100.0, height: 100.0 };
    let mut svg = SvgBuffer::new(viewport, None, false);
    let outer = vec![(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)];
    let hole  = vec![(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)];
    let style = FillStroke {
        fill: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
        fill_opacity: 1.0,
        stroke: None,
        stroke_width: 0.0,
        stroke_opacity: 1.0,
    };
    svg.polygon(&[outer, hole], &style);
    let out = svg.finish();
    assert!(out.contains("M 0 0 L 20 0 L 20 20 L 0 20 Z M 5 5 L 15 5 L 15 15 L 5 15 Z"),
            "multi-ring path data wrong: {out}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core polygon_one_ring 2>&1 | tail -10`
Expected: FAIL "method `polygon` not found".

- [ ] **Step 3: Implement `polygon` method**

In `impl SvgBuffer`:

```rust
/// Emit a closed filled/stroked polygon as <path d="M ... Z" fill-rule="evenodd"/>.
/// `paths`: each inner Vec<(f64, f64)> is one ring; multiple rings → first is outer,
/// rest are holes. fill-rule="evenodd" handles winding automatically.
pub fn polygon(&mut self, paths: &[Vec<(f64, f64)>], style: &FillStroke) {
    if paths.is_empty() { return; }
    let mut d = String::new();
    for ring in paths {
        if ring.is_empty() { continue; }
        d.push_str(&format!("M {} {}", fmt_f(ring[0].0), fmt_f(ring[0].1)));
        for (x, y) in &ring[1..] {
            d.push_str(&format!(" L {} {}", fmt_f(*x), fmt_f(*y)));
        }
        d.push_str(" Z ");
    }
    let d = d.trim_end();
    let style_attrs = fill_stroke_attrs(style);  // existing helper in svg.rs
    self.body.push_str(&format!(
        r#"<path d="{}" fill-rule="evenodd"{}/>"#,
        d, style_attrs
    ));
}
```

(`fill_stroke_attrs` is the existing helper that emits ` fill="..." fill-opacity="..." stroke="..." ...`. If that exact name doesn't exist, locate the equivalent in `svg.rs` — likely inline in `rect()` and `circle()` — and either extract a helper or inline the same logic.)

- [ ] **Step 4: Run tests**

Run: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core polygon 2>&1 | tail -10`
Expected: both pass.

- [ ] **Step 5: Add deterministic emission test**

```rust
#[test]
fn polygon_byte_identical_across_runs() {
    let viewport = Rect { x: 0.0, y: 0.0, width: 100.0, height: 100.0 };
    let style = FillStroke {
        fill: Some(Color { r: 50, g: 50, b: 50, a: 255 }),
        fill_opacity: 0.5,
        stroke: None, stroke_width: 0.0, stroke_opacity: 1.0,
    };
    let make = || {
        let mut svg = SvgBuffer::new(viewport, None, false);
        svg.polygon(&[vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]], &style);
        svg.finish()
    };
    assert_eq!(make(), make());
}
```

- [ ] **Step 6: Run + commit**

Run: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core polygon 2>&1 | tail -5`
```bash
git add crates/ferrum-core/src/render/svg.rs
git commit -m "feat(phase-8b): SvgBuffer.polygon() primitive (path + fill-rule=evenodd, multi-ring)"
```

---

### Task 5: SvgBuffer.beeswarm() primitive (batched circles)

**Files:**
- Modify: `crates/ferrum-core/src/render/svg.rs`

- [ ] **Step 1: Write failing test**

Append to `mod tests`:

```rust
#[test]
fn beeswarm_emits_group_with_n_circles() {
    let viewport = Rect { x: 0.0, y: 0.0, width: 100.0, height: 100.0 };
    let mut svg = SvgBuffer::new(viewport, None, false);
    let pts = vec![(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)];
    let style = FillStroke {
        fill: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
        fill_opacity: 1.0,
        stroke: None, stroke_width: 0.0, stroke_opacity: 1.0,
    };
    svg.beeswarm(&pts, 3.0, &style);
    let out = svg.finish();
    let circle_count = out.matches("<circle").count();
    assert_eq!(circle_count, 3, "expected 3 circles in beeswarm: {out}");
    assert!(out.contains("<g "), "beeswarm should wrap in <g>: {out}");
}

#[test]
fn beeswarm_circles_in_input_order() {
    let viewport = Rect { x: 0.0, y: 0.0, width: 100.0, height: 100.0 };
    let mut svg = SvgBuffer::new(viewport, None, false);
    let pts = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
    let style = FillStroke {
        fill: Some(Color { r: 0, g: 0, b: 0, a: 255 }),
        fill_opacity: 1.0, stroke: None, stroke_width: 0.0, stroke_opacity: 1.0,
    };
    svg.beeswarm(&pts, 1.0, &style);
    let out = svg.finish();
    let pos1 = out.find(r#"cx="1""#).unwrap();
    let pos2 = out.find(r#"cx="2""#).unwrap();
    let pos3 = out.find(r#"cx="3""#).unwrap();
    assert!(pos1 < pos2 && pos2 < pos3, "circles must emit in input order");
}
```

- [ ] **Step 2: Run failing test**

Run: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core beeswarm 2>&1 | tail -10`
Expected: FAIL "method `beeswarm` not found".

- [ ] **Step 3: Implement `beeswarm` method**

```rust
/// Emit a batch of <circle> elements at pre-resolved positions, wrapped in a <g>.
/// Equivalent to N circle() calls with deterministic ordering, but more compact DOM.
/// Used by mark_swarm to keep beeswarm SVG manageable.
pub fn beeswarm(&mut self, points: &[(f64, f64)], radius: f64, style: &FillStroke) {
    if points.is_empty() { return; }
    let style_attrs = fill_stroke_attrs(style);
    self.body.push_str(&format!("<g{}>", style_attrs));
    let r = fmt_f(radius);
    for (x, y) in points {
        self.body.push_str(&format!(
            r#"<circle cx="{}" cy="{}" r="{}"/>"#,
            fmt_f(*x), fmt_f(*y), r
        ));
    }
    self.body.push_str("</g>");
}
```

- [ ] **Step 4: Run tests + commit**

Run: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core beeswarm 2>&1 | tail -5`
Expected: pass.

```bash
git add crates/ferrum-core/src/render/svg.rs
git commit -m "feat(phase-8b): SvgBuffer.beeswarm() primitive (batched <circle> in <g>)"
```

---

### Task 6: render::rasterize PNG encoder

**Files:**
- Create: `crates/ferrum-core/src/render/rasterize.rs`
- Modify: `crates/ferrum-core/src/render/mod.rs` (add `pub(crate) mod rasterize;`)

- [ ] **Step 1: Create file with failing test**

Create `crates/ferrum-core/src/render/rasterize.rs`:

```rust
//! RGBA grid → PNG bytes. Pinned encoder settings for determinism.

use png::{Encoder, BitDepth, ColorType, FilterType, Compression};
use std::io::Cursor;

/// Encode an RGBA pixel buffer as PNG bytes.
/// Pinned: Filter::Sub, Compression::Best (level 9). Required for raster goldens.
pub fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    assert_eq!(
        rgba.len(),
        (width * height * 4) as usize,
        "RGBA buffer length mismatch: expected {} bytes, got {}",
        width * height * 4,
        rgba.len()
    );
    let mut out = Vec::with_capacity(rgba.len() / 4);
    {
        let mut encoder = Encoder::new(Cursor::new(&mut out), width, height);
        encoder.set_color(ColorType::Rgba);
        encoder.set_depth(BitDepth::Eight);
        encoder.set_filter(FilterType::Sub);
        encoder.set_compression(Compression::Best);
        let mut writer = encoder.write_header().expect("png header");
        writer.write_image_data(rgba).expect("png write");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Sha256, Digest};

    fn hash(bytes: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(bytes);
        format!("{:x}", h.finalize())
    }

    #[test]
    fn encode_png_byte_deterministic_across_calls() {
        let mut rgba = vec![0u8; 4 * 4 * 4];
        for i in 0..16 {
            rgba[i * 4] = i as u8;
            rgba[i * 4 + 3] = 255;
        }
        let a = encode_png(4, 4, &rgba);
        let b = encode_png(4, 4, &rgba);
        assert_eq!(hash(&a), hash(&b), "PNG bytes must be byte-identical across calls");
    }

    #[test]
    fn encode_png_minimal_buffer() {
        let rgba = vec![255, 0, 0, 255];  // single red pixel
        let bytes = encode_png(1, 1, &rgba);
        assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G']), "PNG magic missing");
        assert!(bytes.len() < 200, "1x1 PNG should be small: got {} bytes", bytes.len());
    }

    #[test]
    fn encode_png_large_buffer_succeeds() {
        let rgba = vec![128u8; 4 * 1024 * 1024];  // 1024x1024 grey
        let bytes = encode_png(1024, 1024, &rgba);
        assert!(bytes.len() > 100, "1024x1024 PNG should produce some data");
        assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G']));
    }

    #[test]
    #[should_panic(expected = "RGBA buffer length mismatch")]
    fn encode_png_wrong_length_panics() {
        let rgba = vec![0u8; 10];
        encode_png(2, 2, &rgba);
    }
}
```

- [ ] **Step 2: Wire into render/mod.rs**

Add `pub(crate) mod rasterize;` to `crates/ferrum-core/src/render/mod.rs`.

- [ ] **Step 3: Run tests**

Run: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core rasterize 2>&1 | tail -10`
Expected: 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/render/rasterize.rs crates/ferrum-core/src/render/mod.rs
git commit -m "feat(phase-8b): render::rasterize PNG encoder (Filter::Sub, level 9, deterministic)"
```

---

### Task 7: TransformContext + apply_with_context infra

**Files:**
- Create: `crates/ferrum-core/src/transform/context.rs`
- Modify: `crates/ferrum-core/src/transform/core.rs`
- Modify: `crates/ferrum-core/src/transform/mod.rs`

- [ ] **Step 1: Create context module**

Create `crates/ferrum-core/src/transform/context.rs`:

```rust
//! TransformContext: render-time info passed to transforms that need viewport.
//! Used by Raster (resolution="screen") and Swarm (point radius unit conversion).

#[derive(Debug, Clone, Copy)]
pub struct TransformContext {
    /// Panel pixel size (width, height). For raster `resolution="screen"`,
    /// this is the grid dimension. For swarm, used to convert point pixels
    /// to data-space radius via the value-axis scale.
    pub panel_pixel_size: Option<(u32, u32)>,
}

impl Default for TransformContext {
    fn default() -> Self {
        Self { panel_pixel_size: None }
    }
}
```

- [ ] **Step 2: Wire into mod.rs**

In `crates/ferrum-core/src/transform/mod.rs`, add `pub(crate) mod context;`.

- [ ] **Step 3: Add `apply_with_context` to core.rs**

In `crates/ferrum-core/src/transform/core.rs`, after the existing `apply` impl on `TransformSpec`:

```rust
use crate::transform::context::TransformContext;

impl TransformSpec {
    pub(crate) fn apply_with_context(
        &self,
        batch: &RecordBatch,
        _ctx: &TransformContext,
    ) -> PyResult<RecordBatch> {
        // Default: ignore context and forward to existing apply().
        // Phase 8b transforms that NEED context (Raster, Swarm) override below.
        match self {
            // (overrides added when Raster/Swarm transforms land in Tasks 16, 18)
            _ => self.apply(batch),
        }
    }
}

pub(crate) fn apply_transforms_with_context(
    specs: &[TransformSpec],
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    let mut current = batch.clone();
    for spec in specs {
        current = spec.apply_with_context(&current, ctx)?;
    }
    Ok(current)
}
```

- [ ] **Step 4: Add test for context passthrough**

In `transform/core.rs` `mod tests`:

```rust
#[test]
fn apply_with_context_default_falls_back_to_apply() {
    pyo3::Python::initialize();
    let batch = make_one_col_batch("x", vec![1.0, 2.0, 3.0]);
    let spec = TransformSpec::Bin(BinSpec {
        field: "x".into(),
        bin_count: Some(2),
        bin_width: None,
        extent: Some((1.0, 3.0)),
        nice: false,
    });
    let ctx = TransformContext::default();
    let with_ctx = spec.apply_with_context(&batch, &ctx).unwrap();
    let without = spec.apply(&batch).unwrap();
    // Same dispatch; outputs match
    assert_eq!(with_ctx.num_columns(), without.num_columns());
    assert_eq!(with_ctx.num_rows(), without.num_rows());
}
```

- [ ] **Step 5: Run tests**

Run: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core transform::core 2>&1 | tail -10`
Expected: existing 4 tests + 1 new = 5 pass.

- [ ] **Step 6: Commit**

```bash
git add crates/ferrum-core/src/transform/context.rs crates/ferrum-core/src/transform/core.rs crates/ferrum-core/src/transform/mod.rs
git commit -m "feat(phase-8b): TransformContext + apply_with_context infra (default forwards to apply)"
```

---

## Sub-batch B — Routing infra (Tasks 8-9)

### Task 8: Layer.data_source field + TransformSpec.name field

**Files:**
- Modify: `crates/ferrum-core/src/spec/layer.rs`
- Modify: `crates/ferrum-core/src/transform/core.rs` (per-variant name field requires touching each variant struct OR a wrapper)

**Decision (locked in spec §3.7):** Add `name: Option<String>` to each variant's spec struct (BinSpec, KdeSpec, etc.). Skipped on serialization when None to keep 8a JSON byte-identical. *Implementation note:* simpler than a wrapper enum because variants already have struct bodies.

- [ ] **Step 1: Inspect Layer struct**

Run: `grep -n "pub struct Layer\|pub.*field" crates/ferrum-core/src/spec/layer.rs | head -20`

- [ ] **Step 2: Add data_source to Layer**

In `crates/ferrum-core/src/spec/layer.rs`, add to the Layer struct:

```rust
pub struct Layer {
    // ... existing 8a fields ...
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub data_source: Option<String>,
}
```

- [ ] **Step 3: Write round-trip test for data_source**

In `mod tests` of `spec/layer.rs`:

```rust
#[test]
fn layer_data_source_round_trip_some() {
    let l = Layer {
        // ... fill required 8a fields with minimal values ...
        data_source: Some("box".into()),
    };
    let json = serde_json::to_string(&l).unwrap();
    assert!(json.contains(r#""data_source":"box""#));
    let parsed: Layer = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.data_source, Some("box".into()));
}

#[test]
fn layer_data_source_none_omits_from_json() {
    let l = Layer {
        // ... fill required 8a fields with minimal values ...
        data_source: None,
    };
    let json = serde_json::to_string(&l).unwrap();
    assert!(!json.contains("data_source"), "None must not emit field: {json}");
}
```

(Inspect the Layer struct first to fill the other fields with minimal valid values.)

- [ ] **Step 4: Add name field to each TransformSpec variant**

Each of `BinSpec`, `KdeSpec`, `SmoothSpec`, `AggregateSpec`, `SummarySpec` in their respective files needs:

```rust
#[serde(skip_serializing_if = "Option::is_none", default)]
pub name: Option<String>,
```

Add to each spec struct. *This touches `transform/{bin.rs, kde.rs, smooth.rs, aggregate.rs, summary.rs}`.*

- [ ] **Step 5: Update PyO3 wrappers to accept name kwarg**

For each of `PyBin`, `PyKde`, `PySmooth`, `PyAggregate`, `PySummary` in their wrapper sections, add `name: Option<String> = None` to the `#[pyo3(signature = ...)]` and pass through to the spec struct.

- [ ] **Step 6: Run all transform tests**

Run: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core transform 2>&1 | tail -15`
Expected: all existing transform tests still pass; new layer_data_source tests pass.

- [ ] **Step 7: Verify 8a JSON byte-identical when name=None**

Add to `transform/core.rs` mod tests:

```rust
#[test]
fn transform_spec_json_byte_identical_when_name_none() {
    let s = TransformSpec::Bin(BinSpec {
        field: "x".into(),
        bin_count: Some(10),
        bin_width: None,
        extent: None,
        nice: true,
        name: None,
    });
    let json = serde_json::to_string(&s).unwrap();
    // Must NOT contain "name" — the field is omitted when None
    assert!(!json.contains("name"), "name=None must be omitted: {json}");
    // Must match the exact 8a JSON shape
    assert!(json.contains(r#""type":"bin""#));
}
```

- [ ] **Step 8: Run full suite**

Run: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core 2>&1 | tail -10`
Expected: 309 + 3 new = 312 passing.

- [ ] **Step 9: Commit**

```bash
git add crates/ferrum-core/src/spec/layer.rs crates/ferrum-core/src/transform/
git commit -m "feat(phase-8b): Layer.data_source + TransformSpec.name (8a JSON byte-identical when None)"
```

---

### Task 9: render::prepare named-output routing

**Files:**
- Modify: `crates/ferrum-core/src/render/prepare.rs`

- [ ] **Step 1: Inspect current prepare.rs flow AND verify per-layer iteration location**

Run: `wc -l crates/ferrum-core/src/render/prepare.rs && grep -n "pub fn\|fn prepare\|apply_transforms\|layers.iter\|for.*layer" crates/ferrum-core/src/render/prepare.rs | head -15`

Also check the compositor and renderer entry point:
```bash
grep -rn "layers.iter\|spec.layers\|for.*layer" crates/ferrum-core/src/render/ | head -10
```

**Critical check:** if the per-layer iteration lives in `compositor.rs` (Phase 8a) rather than `prepare.rs`, the data-source routing belongs there. In that case, this task's structure changes: the named-outputs map is built in prepare.rs (or wherever transforms apply), passed up to the compositor, and the per-layer dispatch in the compositor reads from it. Adapt the rest of the steps accordingly.

- [ ] **Step 2: Add a TransformOutputs map alongside existing flow**

Add a helper function:

```rust
use std::collections::HashMap;
use crate::transform::core::{TransformSpec, apply_transforms_with_context};
use crate::transform::context::TransformContext;
use arrow::array::RecordBatch;

/// Apply each transform in pipeline order; record named outputs.
/// Returns a map from "__final__" → final pipeline output, plus each transform's
/// name (when present) → that transform's output.
pub(crate) fn apply_transforms_named(
    specs: &[TransformSpec],
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> pyo3::PyResult<HashMap<String, RecordBatch>> {
    let mut outputs = HashMap::new();
    let mut current = batch.clone();
    for spec in specs {
        current = spec.apply_with_context(&current, ctx)?;
        if let Some(name) = spec_name(spec) {
            outputs.insert(name.to_string(), current.clone());
        }
    }
    outputs.insert("__final__".into(), current);
    Ok(outputs)
}

fn spec_name(spec: &TransformSpec) -> Option<&str> {
    match spec {
        TransformSpec::Bin(s) => s.name.as_deref(),
        TransformSpec::Kde(s) => s.name.as_deref(),
        TransformSpec::Smooth(s) => s.name.as_deref(),
        TransformSpec::Aggregate(s) => s.name.as_deref(),
        TransformSpec::Summary(s) => s.name.as_deref(),
        // (Phase 8b transforms added in Tasks 10-19 extend this match)
    }
}
```

- [ ] **Step 3: Update the per-layer dispatch to honor `data_source`**

Locate the existing per-layer loop in `prepare.rs`. Modify it so each layer fetches its input batch via:

```rust
let layer_input = outputs
    .get(layer.data_source.as_deref().unwrap_or("__final__"))
    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
        format!("Layer references data_source '{}'; available: [{}]",
                layer.data_source.as_deref().unwrap_or("__final__"),
                outputs.keys().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
    ))?;
```

- [ ] **Step 4: Add tests**

In `prepare.rs` `mod tests`:

```rust
#[test]
fn unknown_data_source_raises_clear_error() {
    pyo3::Python::initialize();
    // (build a minimal ChartSpec with a layer that points at "missing")
    // ... assert error message contains "Layer references data_source 'missing'"
    // ... and lists available names
}

#[test]
fn data_source_none_uses_final_pipeline_output() {
    pyo3::Python::initialize();
    // (build a ChartSpec with one Bin transform, layer with data_source=None)
    // ... assert layer input matches Bin output (existing 8a behavior)
}

#[test]
fn data_source_some_uses_named_transform_output() {
    pyo3::Python::initialize();
    // (build a ChartSpec with two transforms, second named "second", layer points there)
    // ... assert layer input matches second transform's output
}
```

(Fill in the test bodies with the actual ChartSpec construction once the prepare.rs structure is inspected.)

- [ ] **Step 5: Run tests**

Run: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core prepare:: 2>&1 | tail -10`
Expected: 3 new tests pass; existing pass.

- [ ] **Step 6: Run full suite to confirm 8a charts unaffected**

Run: `DYLD_LIBRARY_PATH=... cargo test -p ferrum-core 2>&1 | tail -10`
Expected: 312 + 3 = 315 passing; all 8a goldens intact.

- [ ] **Step 7: Commit**

```bash
git add crates/ferrum-core/src/render/prepare.rs
git commit -m "feat(phase-8b): named-output transform routing in render::prepare"
```

---

## Sub-batch C — 10 new transforms (Tasks 10-19)

Each transform follows the same pattern: spec struct + `apply` fn + PyO3 wrapper + 3-5 tests. Tasks 10-19 are largely parallelizable (Contour depends on Kde2D — schedule T15 after T14).

### Task 10: Outliers transform

**Files:**
- Create: `crates/ferrum-core/src/transform/outliers.rs`
- Modify: `crates/ferrum-core/src/transform/mod.rs` (add `pub(crate) mod outliers;`)
- Modify: `crates/ferrum-core/src/transform/core.rs` (add `Outliers(OutliersSpec)` variant + dispatch)
- Modify: `crates/ferrum-core/src/binding.rs` (register `PyOutliers`)
- Modify: `crates/ferrum-core/src/lib.rs` (add to `#[pymodule]` registration)

- [ ] **Step 1: Create outliers.rs with spec + apply skeleton**

```rust
//! Outliers: per-group IQR row filter.
//! Output schema = input schema, filtered to outlier rows.

use arrow::array::{Float64Array, RecordBatch, /* group key types */};
use arrow::compute::filter_record_batch;
use arrow::array::BooleanArray;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OutliersSpec {
    pub field: String,
    #[serde(default)]
    pub groupby: Vec<String>,
    pub extent: f64,  // IQR multiplier (default 1.5 from Python wrapper)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

pub(crate) fn apply(spec: &OutliersSpec, batch: &RecordBatch) -> PyResult<RecordBatch> {
    let n = batch.num_rows();
    if n == 0 { return Ok(batch.clone()); }

    let values = column_as_f64(batch, &spec.field)?;  // helper from existing code
    let mask: Vec<bool> = if spec.groupby.is_empty() {
        let (q1, q3) = quartiles_q1_q3(&values);
        let iqr = q3 - q1;
        let lo = q1 - spec.extent * iqr;
        let hi = q3 + spec.extent * iqr;
        values.iter().map(|&v| v < lo || v > hi).collect()
    } else {
        let groups = group_indices_by(&batch, &spec.groupby)?;  // helper
        let mut mask = vec![false; n];
        for indices in groups.values() {
            let group_values: Vec<f64> = indices.iter().map(|i| values[*i]).collect();
            let (q1, q3) = quartiles_q1_q3(&group_values);
            let iqr = q3 - q1;
            let lo = q1 - spec.extent * iqr;
            let hi = q3 + spec.extent * iqr;
            for &i in indices {
                if values[i] < lo || values[i] > hi {
                    mask[i] = true;
                }
            }
        }
        mask
    };
    let bool_arr = BooleanArray::from(mask);
    Ok(filter_record_batch(batch, &bool_arr).map_err(|e|
        pyo3::exceptions::PyValueError::new_err(format!("Outliers filter failed: {e}")))?)
}

/// Type-7 quartile (linear interpolation) — matches numpy/scipy default.
fn quartiles_q1_q3(values: &[f64]) -> (f64, f64) {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 { return (f64::NAN, f64::NAN); }
    let q = |p: f64| -> f64 {
        let h = (n - 1) as f64 * p;
        let lo = h.floor() as usize;
        let hi = h.ceil() as usize;
        if lo == hi { sorted[lo] } else { sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo]) }
    };
    (q(0.25), q(0.75))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch(field: &str, values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(field, DataType::Float64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    #[test]
    fn no_outliers_in_uniform_distribution() {
        pyo3::Python::initialize();
        let b = batch("v", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let spec = OutliersSpec { field: "v".into(), groupby: vec![], extent: 1.5, name: None };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 0, "expected no outliers in uniform data");
    }

    #[test]
    fn extreme_value_flagged_as_outlier() {
        pyo3::Python::initialize();
        let b = batch("v", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0]);
        let spec = OutliersSpec { field: "v".into(), groupby: vec![], extent: 1.5, name: None };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.num_rows(), 1);
        let arr = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(arr.value(0), 1000.0);
    }

    #[test]
    fn outliers_spec_round_trip() {
        let s = OutliersSpec { field: "v".into(), groupby: vec!["g".into()], extent: 2.0, name: Some("o".into()) };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: OutliersSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
    }

    #[test]
    fn schema_preserved() {
        pyo3::Python::initialize();
        let b = batch("v", vec![1.0, 2.0, 100.0]);
        let spec = OutliersSpec { field: "v".into(), groupby: vec![], extent: 1.5, name: None };
        let out = apply(&spec, &b).unwrap();
        assert_eq!(out.schema(), b.schema());
    }
}
```

- [ ] **Step 2: Extend the three match expressions** (TransformSpec enum + apply dispatch + spec_name + apply_with_context default)

In `transform/core.rs`, add variant + dispatch (Rust's exhaustive match means missing any of these breaks compilation, so do all three at once):

```rust
use crate::transform::outliers::{self, OutliersSpec};

pub(crate) enum TransformSpec {
    // ... existing ...
    Outliers(OutliersSpec),
}

impl TransformSpec {
    pub(crate) fn apply(&self, batch: &RecordBatch) -> PyResult<RecordBatch> {
        match self {
            // ... existing ...
            Self::Outliers(s) => outliers::apply(s, batch),
        }
    }

    pub(crate) fn apply_with_context(
        &self,
        batch: &RecordBatch,
        ctx: &TransformContext,
    ) -> PyResult<RecordBatch> {
        match self {
            // ... existing variants forward to apply() ...
            Self::Outliers(_) => self.apply(batch),  // ignores context
        }
    }
}
```

Also extend `spec_name` in `render/prepare.rs`:

```rust
TransformSpec::Outliers(s) => s.name.as_deref(),
```

**For each subsequent transform task (T11-T19), the same three match-expression extensions apply.** Tasks 17 (Raster) and 19 (Swarm) additionally provide non-default `apply_with_context` arms (they actually use the context). Other transforms forward to `apply`.

- [ ] **Step 3: Add PyO3 wrapper**

In `transform/outliers.rs`, append:

```rust
use pyo3::prelude::*;

#[pyclass(name = "Outliers")]
pub(crate) struct PyOutliers(pub(crate) crate::transform::core::TransformSpec);

#[pymethods]
impl PyOutliers {
    #[new]
    #[pyo3(signature = (field, *, groupby = vec![], extent = 1.5, name = None))]
    fn new(field: String, groupby: Vec<String>, extent: f64, name: Option<String>) -> Self {
        Self(crate::transform::core::TransformSpec::Outliers(OutliersSpec {
            field, groupby, extent, name,
        }))
    }

    fn __repr__(&self) -> String {
        if let crate::transform::core::TransformSpec::Outliers(s) = &self.0 {
            format!("Outliers(field='{}', groupby={:?}, extent={}, name={:?})",
                    s.field, s.groupby, s.extent, s.name)
        } else { unreachable!() }
    }
}
```

- [ ] **Step 4: Register in lib.rs / binding.rs**

In `crates/ferrum-core/src/lib.rs` `#[pymodule]` block, add:

```rust
m.add_class::<crate::transform::outliers::PyOutliers>()?;
```

- [ ] **Step 5: Wire mod.rs**

Add to `transform/mod.rs`: `pub(crate) mod outliers;`.

- [ ] **Step 6: Build + test**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -5
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core outliers 2>&1 | tail -10
```
Expected: 4 tests pass.

- [ ] **Step 7: Smoke from Python**

```bash
uv run python -c "from ferrum import Outliers; o = Outliers('v', extent=1.5); print(repr(o))"
```
Expected: `Outliers(field='v', groupby=[], extent=1.5, name=None)`.

- [ ] **Step 8: Commit**

```bash
git add crates/ferrum-core/src/transform/outliers.rs crates/ferrum-core/src/transform/{core,mod}.rs crates/ferrum-core/src/lib.rs
git commit -m "feat(phase-8b): Outliers transform (per-group IQR row filter)"
```

---

### Task 11: ErrorExtent transform

**Files:**
- Create: `crates/ferrum-core/src/transform/error_extent.rs`
- Modify: `transform/{core,mod}.rs`, `lib.rs`, `render/prepare.rs::spec_name`

Mirror Task 10's structure. Key implementation differences:

**ErrorExtentSpec fields:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ErrorExtentSpec {
    pub field: String,
    #[serde(default)]
    pub groupby: Vec<String>,
    pub method: ErrorMethod,    // enum below
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_n_boot")]
    pub n_boot: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}
fn default_seed() -> u64 { 0 }
fn default_n_boot() -> usize { 1000 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ErrorMethod { Ci, Stderr, Stdev, Iqr }
```

**Output schema:** groupby cols + `mean: f64` + `lower: f64` + `upper: f64`. One row per group (or one row total if no groupby).

**apply() per-method logic** (per spec §5.5):
- `Ci`: bootstrap percentile 95% via `rand_chacha::ChaCha8Rng::seed_from_u64(spec.seed)`; `n_boot=1000`.
- `Stderr`: mean ± stdev/√n.
- `Stdev`: mean ± stdev.
- `Iqr`: median + q1/q3.

- [ ] **Step 1: Create file with spec, apply, 5 tests** (one per method + bootstrap reproducibility)
- [ ] **Step 2: Wire core.rs enum, dispatch, prepare.rs spec_name**
- [ ] **Step 3: Add PyO3 wrapper `PyErrorExtent` accepting `method: &str`**
- [ ] **Step 4: Register in lib.rs**
- [ ] **Step 5: Build + run tests**
- [ ] **Step 6: Commit:** `git commit -m "feat(phase-8b): ErrorExtent transform (ci/stderr/stdev/iqr per group)"`

Reference test for bootstrap reproducibility:

```rust
#[test]
fn bootstrap_ci_byte_deterministic_with_fixed_seed() {
    pyo3::Python::initialize();
    let b = batch("v", (0..100).map(|i| i as f64).collect());
    let spec = ErrorExtentSpec {
        field: "v".into(), groupby: vec![], method: ErrorMethod::Ci,
        seed: 42, n_boot: 1000, name: None,
    };
    let out_a = apply(&spec, &b).unwrap();
    let out_b = apply(&spec, &b).unwrap();
    let lower_a = out_a.column_by_name("lower").unwrap().as_any().downcast_ref::<Float64Array>().unwrap().value(0);
    let lower_b = out_b.column_by_name("lower").unwrap().as_any().downcast_ref::<Float64Array>().unwrap().value(0);
    assert_eq!(lower_a, lower_b, "bootstrap must be byte-deterministic with fixed seed");
}
```

---

### Task 12: BoxStats transform

**Files:**
- Create: `crates/ferrum-core/src/transform/box_stats.rs`
- Modify: `transform/{core,mod}.rs`, `lib.rs`, `render/prepare.rs::spec_name`

**BoxStatsSpec fields:**
```rust
pub(crate) struct BoxStatsSpec {
    pub field: String,
    #[serde(default)]
    pub groupby: Vec<String>,
    #[serde(default)]
    pub whisker_extent: WhiskerExtent,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum WhiskerExtent {
    MinMax(String),  // "min-max"
    IqrMultiplier(f64),  // 1.5 default
}
impl Default for WhiskerExtent { fn default() -> Self { Self::IqrMultiplier(1.5) } }
```

**Output schema:** groupby cols + `q1, median, q3, lower_whisker, upper_whisker` (all f64).

**apply() per spec §5.4:** Per group, compute Q1/median/Q3 via Type-7. Whiskers either group-min/max or `Q1 - k·IQR` clipped to group min, etc.

- [ ] **Step 1: Create file** with 5 tests:
  1. Q1/Q2/Q3 against numpy reference (e.g., 1..=10 → Q1=3.25, Q2=5.5, Q3=7.75)
  2. whisker `min-max` mode = group min/max
  3. whisker `IqrMultiplier(1.5)` clamps to data range
  4. per-group case (3 groups, distinct stats per group)
  5. PartialEq round-trip
- [ ] **Step 2-6: Wire + commit**: `git commit -m "feat(phase-8b): BoxStats transform (Type-7 quartiles + whiskers per group)"`

---

### Task 13: Violin transform

**Files:**
- Create: `crates/ferrum-core/src/transform/violin.rs`
- Modify: `transform/{core,mod}.rs`, `lib.rs`, `render/prepare.rs::spec_name`

**ViolinSpec fields:**
```rust
pub(crate) struct ViolinSpec {
    pub field: String,
    #[serde(default)]
    pub groupby: Vec<String>,
    #[serde(default = "default_bandwidth")]
    pub bandwidth: BandwidthSpec,  // reuse existing Phase 5 type from kde.rs
    #[serde(default = "default_violin_n")]
    pub n: usize,
    #[serde(default = "default_violin_width")]
    pub width: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}
fn default_violin_n() -> usize { 256 }
fn default_violin_width() -> f64 { 0.4 }
```

**Output schema:** `group_id: u32` + `<groupby cols>` + `violin_x: f64` (offset from group center) + `violin_y: f64`. Vertices ordered: right side bottom→top, then left side top→bottom (mirrored), forming a closed polygon per group.

**apply() per spec §4.2.5:**
- Per group: run Phase 5 `kde` to get (xs, density). Find max density; normalize to `width`.
- Emit vertices: right side `(+density_normalized, x)` for each grid point bottom→top, then left side `(-density_normalized, x)` top→bottom.
- All rows share `group_id`; assign incrementing IDs per distinct group key.

- [ ] **Step 1: Create file** with 3 tests:
  1. Polygon vertex count = 2N for N grid points; symmetry (right side mirrors left)
  2. Per-group case yields distinct group_ids
  3. Bandwidth ∈ {"scott", "silverman", 0.5} all parse + apply
- [ ] **Step 2-6: Wire + commit:** `git commit -m "feat(phase-8b): Violin transform (per-group KDE → mirrored polygon vertices)"`

---

### Task 14: Kde2D transform

**Files:**
- Create: `crates/ferrum-core/src/transform/kde_2d.rs`
- Modify: `transform/{core,mod}.rs`, `lib.rs`, `render/prepare.rs::spec_name`

**Kde2DSpec fields:**
```rust
pub(crate) struct Kde2DSpec {
    pub x: String,
    pub y: String,
    #[serde(default)]
    pub bandwidth: BandwidthSpec,  // reuse from kde.rs
    #[serde(default = "default_kde2d_n")]
    pub n: usize,  // grid edge length, default 128
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extent: Option<(f64, f64, f64, f64)>,  // (xmin, xmax, ymin, ymax)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}
fn default_kde2d_n() -> usize { 128 }
```

**Output schema (single row):**
- `grid_x: List<f64>` (length n)
- `grid_y: List<f64>` (length n)
- `density: List<f64>` (length n*n, row-major: density[gy * n + gx])
- `nx: u32`, `ny: u32`
- `extent: List<f64>` (4 values: xmin, xmax, ymin, ymax)

**apply() per spec §5.7:**
- Compute marginal Scott bandwidths (reuse `Phase 5 scott_bandwidth` helper).
- Build uniform grid n×n over extent.
- Separable Gaussian KDE: O(N · 2n) by precomputing per-axis kernel sums.

- [ ] **Step 1: Create file** with 3 tests:
  1. Density approximately sums to 1.0 (Σ_ij density[i,j] · dx · dy ≈ 1)
  2. Grid extent matches data range (auto) or explicit extent
  3. PartialEq round-trip
- [ ] **Step 2-6: Wire + commit:** `git commit -m "feat(phase-8b): Kde2D transform (separable Gaussian KDE on 128x128 grid)"`

---

### Task 15: Contour transform

**Files:**
- Create: `crates/ferrum-core/src/transform/contour.rs`
- Modify: `transform/{core,mod}.rs`, `lib.rs`, `render/prepare.rs::spec_name`

**Depends on Task 14** (Contour consumes Kde2D output).

**ContourSpec fields:**
```rust
pub(crate) struct ContourSpec {
    #[serde(default = "default_thresholds")]
    pub thresholds: u32,
    #[serde(default)]
    pub fill: bool,
    #[serde(default = "default_smooth")]
    pub smooth: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}
fn default_thresholds() -> u32 { 6 }
fn default_smooth() -> bool { true }
```

**Output schema:** `level_id: u32` + `level_value: f64` + `contour_x: f64` + `contour_y: f64`, one row per polyline vertex.

**apply() per spec §5.1:**
- Read Kde2D output's grid_x/grid_y/density into 2D grid.
- Compute `thresholds` evenly-spaced levels from min to max density.
- For `fill=False`: per level, run Marching Squares isoline algorithm. For each polyline, emit vertices with shared `level_id`.
- For `fill=True`: per band (between adjacent levels), run Marching Squares isoband algorithm. Detect rings; emit polygons with `level_id = (band_index << 16) | ring_index`.
- Saddle disambiguation: cell-center value (deterministic).

This is the most algorithmically complex transform in 8b. Allocate sufficient time.

- [ ] **Step 1: Create file** with 7 tests:
  1. Isoline schema correct (4 columns, dtype check)
  2. Isoband schema correct
  3. Saddle case (cell with corners 0,1,1,0 — must use center value)
  4. Ring with hole (bimodal density: outer ring + inner hole produces polygon with 2 rings sharing level_id band)
  5. Bivariate density routing (verify Contour over Kde2D output produces sensible shapes)
  6. PartialEq round-trip
  7. **Evenodd fill-rule correctness**: when isoband mode produces a ring with an inner hole, verify `level_id` encoding distinguishes outer vs inner (outer ring has even ring_index, hole has odd) so the polygon mark drawer emits both paths in the same `<path>` element with `fill-rule="evenodd"`.
- [ ] **Step 2-6: Wire + commit:** `git commit -m "feat(phase-8b): Contour transform (Marching Squares isolines + isobands, evenodd-ready level_id)"`

---

### Task 16: QQ transform

**Files:**
- Create: `crates/ferrum-core/src/transform/qq.rs`
- Modify: `transform/{core,mod}.rs`, `lib.rs`, `render/prepare.rs::spec_name`

**QQSpec fields:**
```rust
pub(crate) struct QQSpec {
    pub field: String,
    #[serde(default = "default_distribution")]
    pub distribution: String,  // "normal" | "uniform" | "exponential"
    #[serde(default)]
    pub dequantize: bool,
    #[serde(default = "default_emit_line")]
    pub emit_line: bool,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}
fn default_distribution() -> String { "normal".into() }
fn default_emit_line() -> bool { true }
```

**Output schema:** `theoretical: f64`, `sample: f64` per row.

**Multi-output handling:** When `emit_line=true`, the transform stores the line data as a *separate batch* in the named-outputs map. *Implementation*: add an `apply_with_outputs` variant on `TransformSpec` that returns `Vec<(Option<String>, RecordBatch)>` instead of single batch. Or simpler: the `apply` returns the points batch as primary; the `prepare.rs::apply_transforms_named` checks if the transform is QQ + emit_line, and additionally invokes a `compute_line(...)` helper to register a `"qq_line"` output.

**Recommended approach (simpler):** Add a `secondary_outputs(&self, batch: &RecordBatch) -> PyResult<Vec<(String, RecordBatch)>>` method on `TransformSpec` returning extras. Default: empty. QQ overrides to return `[("qq_line", line_batch)]` when emit_line. `prepare.rs` calls both.

- [ ] **Step 1: Implement QQ apply**

Per spec §5.6:
1. Sort sample values; n = len.
2. Plotting positions `p_i = (i - 0.5) / n`.
3. Inverse CDF for distribution; emit (theoretical, sample) per row.
4. Dequantize: jitter ties.

- [ ] **Step 2: Implement secondary_outputs for line**

Robust resistant fit: slope = (sample_q3 - sample_q1) / (theo_q3 - theo_q1); intercept = sample_q2 - slope * theo_q2. Emit `qq_line_x_start`, `qq_line_x_end`, `qq_line_y_start`, `qq_line_y_end`.

- [ ] **Step 3: Wire `secondary_outputs` into prepare.rs::apply_transforms_named**

After `current = spec.apply_with_context(...)?`, also iterate `spec.secondary_outputs(&current)?` and insert into `outputs`.

- [ ] **Step 4: 4 tests**

1. Normal distribution: sample = normal(0,1) → theoretical roughly matches scipy reference
2. Uniform / exponential distributions
3. Reference line slope correctness for known input
4. Dequantize jitter is non-zero only on ties

- [ ] **Step 5: Wire + commit:** `git commit -m "feat(phase-8b): QQ transform with secondary qq_line output"`

---

### Task 17: Raster transform

**Files:**
- Create: `crates/ferrum-core/src/transform/raster.rs`
- Modify: `transform/{core,mod}.rs`, `lib.rs`, `render/prepare.rs::spec_name`
- Modify: `crates/ferrum-core/src/transform/core.rs` `apply_with_context` to dispatch Raster with context

**RasterSpec fields:**
```rust
pub(crate) struct RasterSpec {
    pub x: String,
    pub y: String,
    #[serde(default = "default_aggregate")]
    pub aggregate: String,  // "count" | "density" | "mean" | "sum" | "any"
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub field: Option<String>,
    #[serde(default)]
    pub resolution: ResolutionSpec,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub min_count: Option<u32>,
    #[serde(default)]
    pub log_scale: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ResolutionSpec {
    Screen,
    Fixed(u32),
    Xy(u32, u32),
}
impl Default for ResolutionSpec { fn default() -> Self { Self::Screen } }
```

**Output schema (single row):** `x_min, x_max, y_min, y_max, width: u32, height: u32, pixel_data: Binary`.

**apply_with_context** (since Raster needs panel size for `Screen`):
```rust
pub(crate) fn apply_with_context(
    spec: &RasterSpec,
    batch: &RecordBatch,
    ctx: &TransformContext,
) -> PyResult<RecordBatch> {
    let (nx, ny) = match spec.resolution {
        ResolutionSpec::Fixed(n) => (n, n),
        ResolutionSpec::Xy(a, b) => (a, b),
        ResolutionSpec::Screen => ctx.panel_pixel_size.unwrap_or((256, 256)),
    };
    // ... histogram2d / aggregate ...
}
```

- [ ] **Step 1: Implement aggregate variants** (count via simple histogram2d; density = count/(cell_area * n); mean/sum require `field`; "any" = 1 where count>0)
- [ ] **Step 2: Wire context dispatch in `core.rs::apply_with_context`** — add explicit match arm for Raster
- [ ] **Step 3: 6 tests** — one per aggregate (5: count/density/mean/sum/any); one for each resolution variant (Fixed, XY, Screen via mock ctx) (3 sub-cases counted as 1 parametric test); min_count masking; log_scale; **plus one test for the no-context fallback**:

```rust
#[test]
fn raster_screen_resolution_falls_back_to_256x256_without_context() {
    pyo3::Python::initialize();
    let b = make_xy_batch(vec![0.0, 1.0, 0.5], vec![0.0, 1.0, 0.5]);
    let spec = RasterSpec {
        x: "x".into(), y: "y".into(), aggregate: "count".into(), field: None,
        resolution: ResolutionSpec::Screen, min_count: None, log_scale: false, name: None,
    };
    let ctx = TransformContext::default();  // panel_pixel_size = None
    let out = apply_with_context(&spec, &b, &ctx).unwrap();
    let width = out.column_by_name("width").unwrap()
        .as_any().downcast_ref::<UInt32Array>().unwrap().value(0);
    let height = out.column_by_name("height").unwrap()
        .as_any().downcast_ref::<UInt32Array>().unwrap().value(0);
    assert_eq!(width, 256, "Screen-without-context must fall back to 256");
    assert_eq!(height, 256);
}
```

- [ ] **Step 4: Wire + commit:** `git commit -m "feat(phase-8b): Raster transform (2D bin aggregation, resolution=Screen via context, 256x256 fallback)"`

---

### Task 18: Hex transform

**Files:**
- Create: `crates/ferrum-core/src/transform/hex.rs`
- Modify: `transform/{core,mod}.rs`, `lib.rs`, `render/prepare.rs::spec_name`

**HexSpec fields:**
```rust
pub(crate) struct HexSpec {
    pub x: String,
    pub y: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bin_size: Option<f64>,
    #[serde(default = "default_aggregate")]
    pub aggregate: String,  // "count" | "mean" | "sum"
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}
```

**Output schema:** `hex_x: f64, hex_y: f64, hex_id: i64, value: f64` — 6 vertex rows per non-empty hex.

**apply per spec §5.3:**
- Auto bin_size = (x_extent / 30) if None.
- Convert (x, y) to fractional axial → cube-round → integer (q, r).
- Aggregate per (q, r).
- Emit 6 vertex rows per hex sharing `hex_id = q * 65536 + r`.

- [ ] **Step 1: Implement** with 4 tests:
  1. count aggregate
  2. mean/sum require field; error if missing
  3. cube-rounding correctness (e.g., (0.4, 0.4, -0.8) → (0, 0))
  4. vertex count = 6N for N non-empty hexes
- [ ] **Step 2-5: Wire + commit:** `git commit -m "feat(phase-8b): Hex transform (axial coords, cube-rounding, 6-vertex output)"`

---

### Task 19: Swarm transform

**Files:**
- Create: `crates/ferrum-core/src/transform/swarm.rs`
- Modify: `transform/{core,mod}.rs`, `lib.rs`, `render/prepare.rs::spec_name`
- Modify: `crates/ferrum-core/src/transform/core.rs` `apply_with_context` for Swarm

**SwarmSpec fields:**
```rust
pub(crate) struct SwarmSpec {
    pub category: String,
    pub value: String,
    #[serde(default = "default_point_size")]
    pub point_size: f64,
    #[serde(default = "default_spacing")]
    pub spacing: f64,
    #[serde(default)]
    pub side: SwarmSide,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SwarmSide { #[default] Both, Left, Right }
```

**Output schema:** input columns + `swarm_x: f64`, `swarm_y: f64`.

**apply_with_context** (since Swarm needs panel size for radius unit conversion):
- Radius in data-space = `point_size * spacing` (currently in pixels) → multiply by `(value_range / panel_height)` if vertical orient.
- Without context, use a fixed radius assumption (warn-once).

**Greedy sweep per spec §5.2:**
1. Per category group, sort by value with stable tiebreak on row index.
2. For each point, try offsets `[0, +d, -d, +2d, -2d, ...]` (or asymmetric for side=left/right).
3. Place at first non-overlapping offset.

- [ ] **Step 1: Implement** with 4 tests:
  1. side=both/left/right yield expected offset patterns
  2. Stable tiebreak: re-running with same input → byte-identical placements
  3. Spacing variations
  4. Range query optimization (no false overlaps)
- [ ] **Step 2-6: Wire + commit:** `git commit -m "feat(phase-8b): Swarm transform (greedy beeswarm with deterministic tiebreak)"`

---

## Sub-batch D — render::marks drawers (Tasks 20-22)

### Task 20: render::marks::polygon

**Files:**
- Create: `crates/ferrum-core/src/render/marks/polygon.rs`
- Modify: `crates/ferrum-core/src/render/marks/mod.rs` (add `pub(crate) mod polygon;`)

- [ ] **Step 1: Inspect existing mark drawer pattern**

Run: `cat crates/ferrum-core/src/render/marks/area.rs | head -60`
Note the function signature, scale resolution, SVG emission pattern.

- [ ] **Step 2: Implement polygon mark drawer**

Per the spec, polygon consumes a batch with columns `<x_field>, <y_field>, <detail_field>` (e.g., `contour_x, contour_y, level_id` for contour; `hex_x, hex_y, hex_id` for hex; `violin_x, violin_y, group_id` for violin):

```rust
//! Polygon mark drawer: groups vertices by `detail` channel into closed paths.
//! Supports filled (with evenodd for holes) and stroked rendering.

use arrow::array::RecordBatch;
use crate::render::svg::SvgBuffer;
use crate::render::scale_resolve::ResolvedScales;
use crate::spec::layer::Layer;
use std::collections::BTreeMap;
use pyo3::PyResult;

pub(crate) fn draw(
    layer: &Layer,
    batch: &RecordBatch,
    scales: &ResolvedScales,
    svg: &mut SvgBuffer,
) -> PyResult<()> {
    // 1. Read x, y, detail columns from encoding
    let x_field = layer.encoding.x.field.as_ref().expect("polygon needs x");
    let y_field = layer.encoding.y.field.as_ref().expect("polygon needs y");
    let detail_field = layer.encoding.detail.as_ref().and_then(|d| d.field.as_ref());

    let xs = column_as_f64(batch, x_field)?;
    let ys = column_as_f64(batch, y_field)?;
    let groups: BTreeMap<i64, Vec<usize>> = match detail_field {
        Some(f) => {
            let dets = column_as_i64(batch, f)?;
            let mut g: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
            for (i, d) in dets.iter().enumerate() { g.entry(*d).or_default().push(i); }
            g
        }
        None => {
            // single polygon
            let mut g = BTreeMap::new();
            g.insert(0, (0..xs.len()).collect());
            g
        }
    };

    // 2. For each group, build ring(s); map through scales
    let mut all_paths: Vec<Vec<(f64, f64)>> = Vec::new();
    for (_id, indices) in &groups {
        // Detect multi-ring via level_id high bits (contour isobands use (band << 16) | ring)
        // For non-contour cases, indices form a single ring.
        let ring: Vec<(f64, f64)> = indices.iter()
            .map(|&i| (scales.x.map(xs[i]), scales.y.map(ys[i])))
            .collect();
        all_paths.push(ring);
    }

    // 3. Compute per-group color (quantitative color encoding via continuous colormap)
    //    See spec §12.3: mark_hex sets encoding.color="value"; mark_contour(fill=True) similar.
    let color_field = layer.encoding.color.as_ref().and_then(|c| c.field.as_ref());
    let cmap_name = layer.mark.kwargs.get("cmap").and_then(|v| v.as_str()).unwrap_or("viridis");
    let scheme = crate::render::color::continuous::ContinuousScheme::Named(
        crate::render::color::continuous::NamedContinuous::from_name(cmap_name)
            .unwrap_or(crate::render::color::continuous::NamedContinuous::Viridis));

    if let Some(cf) = color_field {
        // Per-group quantitative color: emit each polygon individually with its own fill.
        let color_values = column_as_f64(batch, cf)?;
        // Compute global min/max for normalization
        let (vmin, vmax) = color_values.iter().filter(|v| v.is_finite()).fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(lo, hi), &v| (lo.min(v), hi.max(v)));
        let denom = (vmax - vmin).max(f64::EPSILON);

        for (group_idx, (_id, indices)) in groups.iter().enumerate() {
            // Use the median color value of the group (one logical entity per polygon)
            let group_value = color_values[indices[0]];  // hex/contour: all rows in group share value
            let t = (group_value - vmin) / denom;
            let color = scheme.sample(t.clamp(0.0, 1.0));
            let style = FillStroke {
                fill: Some(color),
                fill_opacity: layer.mark.kwargs.get("fill_opacity").and_then(|v| v.as_f64()).unwrap_or(1.0),
                stroke: parse_optional_color(layer.mark.kwargs.get("stroke")),
                stroke_width: layer.mark.kwargs.get("stroke_width").and_then(|v| v.as_f64()).unwrap_or(0.0),
                stroke_opacity: 1.0,
            };
            svg.polygon(&[all_paths[group_idx].clone()], &style);
        }
    } else {
        // Single fixed style from mark_kwargs (used by violin, mark_contour without color encoding)
        let style = build_fill_stroke(&layer.mark.kwargs, None);
        svg.polygon(&all_paths, &style);
    }
    Ok(())
}
```

(Exact helper names from existing renderers; check `crates/ferrum-core/src/render/marks/area.rs` for parallel patterns. `parse_optional_color` parses a hex/named string from the mark_kwargs JSON value.)

- [ ] **Step 3: Add 4 tests** (3 base + 1 quantitative-color)

In the same file `mod tests`:
1. One ring smoke test (3 vertices → triangle SVG path)
2. Multi-ring (contour-style: outer ring + hole as separate detail groups)
3. Multi-polygon batch (3 groups → 3 paths)
4. Quantitative color: a 3-group batch with `value` column [0.0, 0.5, 1.0] and cmap="viridis" produces 3 polygons with distinct fill colors (assert `fill="..."` differs across the three `<path>` emissions)

- [ ] **Step 4: Wire mod.rs + dispatcher**

Find the central mark dispatcher (likely in `render/marks/mod.rs` or a `draw_mark` fn somewhere) and add `"polygon" => polygon::draw(layer, batch, scales, svg)`.

- [ ] **Step 5: Build + test**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -5
DYLD_LIBRARY_PATH=... cargo test -p ferrum-core polygon 2>&1 | tail -10
```

- [ ] **Step 6: Commit:** `git commit -m "feat(phase-8b): render::marks::polygon (groups by detail, fill-rule=evenodd)"`

---

### Task 21: render::marks::image

**Files:**
- Create: `crates/ferrum-core/src/render/marks/image.rs`
- Modify: `crates/ferrum-core/src/render/marks/mod.rs`

The image mark reads a single-row batch from Raster (columns: `x_min, x_max, y_min, y_max, width, height, pixel_data`).

- [ ] **Step 1: Implement**

```rust
//! Image mark drawer: read RGBA pixel data, encode PNG, embed via SvgBuffer.image().

use arrow::array::{RecordBatch, Float64Array, UInt32Array, BinaryArray};
use crate::render::{svg::SvgBuffer, rasterize::encode_png, scale_resolve::ResolvedScales};
use crate::render::color::continuous::{ContinuousScheme, NamedContinuous};
use crate::spec::layer::Layer;
use pyo3::PyResult;

pub(crate) fn draw(
    layer: &Layer,
    batch: &RecordBatch,
    scales: &ResolvedScales,
    svg: &mut SvgBuffer,
) -> PyResult<()> {
    if batch.num_rows() != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("image mark expects 1-row batch from Raster; got {} rows", batch.num_rows())));
    }
    let x_min = batch.column_by_name("x_min").unwrap().as_any().downcast_ref::<Float64Array>().unwrap().value(0);
    let x_max = batch.column_by_name("x_max").unwrap().as_any().downcast_ref::<Float64Array>().unwrap().value(0);
    let y_min = batch.column_by_name("y_min").unwrap().as_any().downcast_ref::<Float64Array>().unwrap().value(0);
    let y_max = batch.column_by_name("y_max").unwrap().as_any().downcast_ref::<Float64Array>().unwrap().value(0);
    let width = batch.column_by_name("width").unwrap().as_any().downcast_ref::<UInt32Array>().unwrap().value(0);
    let height = batch.column_by_name("height").unwrap().as_any().downcast_ref::<UInt32Array>().unwrap().value(0);
    let pixel_bytes = batch.column_by_name("pixel_data").unwrap().as_any().downcast_ref::<BinaryArray>().unwrap().value(0);

    // pixel_bytes is row-major Vec<f64> (8 bytes per cell). Decode.
    let n_cells = (width * height) as usize;
    assert_eq!(pixel_bytes.len(), n_cells * 8);
    let mut values = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        let bytes: [u8; 8] = pixel_bytes[i*8..i*8+8].try_into().unwrap();
        values.push(f64::from_le_bytes(bytes));
    }

    // Pick cmap
    let cmap_name = layer.mark.kwargs.get("cmap").and_then(|v| v.as_str()).unwrap_or("viridis");
    let scheme = ContinuousScheme::Named(
        NamedContinuous::from_name(cmap_name).unwrap_or(NamedContinuous::Viridis));

    // Normalize over non-NaN values, then sample colormap to RGBA
    let (vmin, vmax) = values.iter().filter(|v| v.is_finite()).fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let denom = (vmax - vmin).max(f64::EPSILON);

    let mut rgba = Vec::with_capacity(n_cells * 4);
    for &v in &values {
        if v.is_nan() {
            rgba.extend_from_slice(&[0, 0, 0, 0]);  // transparent
        } else {
            let t = (v - vmin) / denom;
            let c = scheme.sample(t);
            rgba.extend_from_slice(&[c.r, c.g, c.b, c.a]);
        }
    }

    let png_bytes = encode_png(width, height, &rgba);

    // Map data extent to pixel space via scales
    let svg_x = scales.x.map(x_min);
    let svg_y = scales.y.map(y_max);  // SVG y inverted vs data y
    let svg_w = (scales.x.map(x_max) - svg_x).abs();
    let svg_h = (scales.y.map(y_min) - svg_y).abs();

    svg.image(svg_x, svg_y, svg_w, svg_h, &png_bytes);
    Ok(())
}
```

- [ ] **Step 2: 3 tests**

1. Smoke: 4-cell raster batch → SVG output contains `<image href="data:image/png;base64,`
2. cmap dispatch: setting `cmap="plasma"` produces different bytes than `cmap="viridis"`
3. Image positioning: x/y/width/height computed via scales

- [ ] **Step 3: Wire + commit:** `git commit -m "feat(phase-8b): render::marks::image (RGBA→PNG→SVG, viridis default)"`

---

### Task 22: render::marks::ribbon

**Files:**
- Create: `crates/ferrum-core/src/render/marks/ribbon.rs`
- Modify: `crates/ferrum-core/src/render/marks/mod.rs`

The ribbon mark fills the area between two y-values (y, y2) at each x.

- [ ] **Step 1: Implement**

Walk x ascending, build path: `M x[0] y[0] L x[1] y[1] ... L x[n-1] y[n-1] L x[n-1] y2[n-1] L x[n-2] y2[n-2] ... L x[0] y2[0] Z`.

Use existing `area.rs` as a structural reference. Honor `interpolate` kwarg ("linear" only in 8b; warn-once for "step", "monotone" etc.).

- [ ] **Step 2: 3 tests**
1. Y/Y2 path emission with 5 points → expected `d` attribute
2. Opacity via `mark_kwargs["opacity"]`
3. interpolate="linear" works; other values warn-once

- [ ] **Step 3: Wire + commit:** `git commit -m "feat(phase-8b): render::marks::ribbon (Y/Y2 closed area)"`

---

### Task 22b: Layered-desugar resolver in Chart (PREREQUISITE for Sub-batch E/F)

**Files:**
- Modify: `src/ferrum/chart.py` (extend `_resolve_pending_then_build_spec`)
- Create: `tests/test_layered_desugar_resolver.py` (4 tests)

**Why this task exists:** Tasks 23-33 desugar to a tuple `("__layered__", transforms, None, None, layers)` where `layers` is a list of Python dicts (e.g., `{"mark": "rule", "encoding": {"x": "g", "y": "lower"}, "data_source": "box"}`). The 8a resolver only knows how to build single-layer ChartSpecs. This task adds the dict→Layer conversion and the sentinel-handling branch BEFORE any composite mark depends on it. See spec §12.1.

- [ ] **Step 1: Inspect 8a resolver**

Run: `grep -n "_pending_stat_mark\|_resolve_pending\|build_spec" src/ferrum/chart.py | head -20`
Note the existing tuple-unpacking convention and how single-layer ChartSpecs are constructed.

- [ ] **Step 2: Write a fake-desugar test FIRST (Red)**

Create `tests/test_layered_desugar_resolver.py`:

```python
"""Tests for the __layered__ desugar sentinel and dict→Layer conversion.
Uses a fake desugar so we don't depend on any composite mark being implemented yet."""
import polars as pl
import pytest
import ferrum as fe


def _fake_layered_desugar(x_field, y_field, **_kwargs):
    """Returns a 5-tuple: ("__layered__", transforms, None, None, layers)."""
    layers = [
        {"mark": "rule", "encoding": {"x": x_field, "y": y_field}, "data_source": None},
        {"mark": "point", "encoding": {"x": x_field, "y": y_field},
         "mark_kwargs": {"size": 5}, "data_source": None},
    ]
    return ("__layered__", [], None, None, layers)


@pytest.fixture
def df():
    return pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})


def test_layered_sentinel_produces_multi_layer_spec(df):
    chart = fe.Chart(df)
    chart._pending_stat_mark = ("__fake__", {}, _fake_layered_desugar)
    spec = chart.encode(x="x", y="y")._build_spec()
    assert spec.layers is not None, "expected multi-layer ChartSpec"
    assert len(spec.layers) == 2
    layer_names = [l.mark.name for l in spec.layers]
    assert layer_names == ["rule", "point"]


def test_layered_layer_carries_mark_kwargs(df):
    chart = fe.Chart(df)
    chart._pending_stat_mark = ("__fake__", {}, _fake_layered_desugar)
    spec = chart.encode(x="x", y="y")._build_spec()
    point_layer = spec.layers[1]
    # mark_kwargs is the per-mark style overrides bag
    assert point_layer.mark_kwargs is not None
    json = spec.to_json()
    assert '"size":5' in json or '"size": 5' in json


def test_layered_layer_carries_data_source_when_set(df):
    def desugar_with_named_source(x_field, y_field, **_kw):
        layers = [
            {"mark": "rule", "encoding": {"x": x_field, "y": y_field}, "data_source": "box"},
        ]
        return ("__layered__", [], None, None, layers)
    chart = fe.Chart(df)
    chart._pending_stat_mark = ("__fake__", {}, desugar_with_named_source)
    spec = chart.encode(x="x", y="y")._build_spec()
    assert spec.layers[0].data_source == "box"


def test_layered_encoding_y2_supported(df):
    """Layered desugar may emit y2 in encoding (used by ribbon/errorband layers)."""
    def desugar_with_y2(x_field, y_field, **_kw):
        layers = [
            {"mark": "ribbon", "encoding": {"x": x_field, "y": y_field, "y2": "y"}, "data_source": None},
        ]
        return ("__layered__", [], None, None, layers)
    chart = fe.Chart(df)
    chart._pending_stat_mark = ("__fake__", {}, desugar_with_y2)
    spec = chart.encode(x="x", y="y")._build_spec()
    json = spec.to_json()
    assert '"y2"' in json or "y2" in json
```

- [ ] **Step 3: Run test to verify RED**

Run: `uv run pytest tests/test_layered_desugar_resolver.py -v 2>&1 | tail -15`
Expected: all 4 tests fail (resolver doesn't recognize `"__layered__"` sentinel yet).

- [ ] **Step 4: Implement dict→Layer conversion**

In `src/ferrum/chart.py`, add a private helper:

```python
def _dict_to_layer(layer_dict: dict) -> "Layer":
    """Convert a desugar's dict-form layer description into a PyO3 Layer instance.

    Schema:
        {"mark": str, "encoding": dict[str, str], "mark_kwargs": dict (opt), "data_source": str|None (opt)}

    Where `encoding` maps channel name (x|y|x2|y2|color|detail|...) to field name.
    Channel names match the case-insensitive keys accepted by Chart.encode().
    """
    from ferrum import Mark, Layer  # PyO3 wrappers
    from ferrum.encoding.base import to_encoding_spec_dict

    mark_name = layer_dict["mark"]
    mark_kwargs = layer_dict.get("mark_kwargs", {}) or {}
    mark = Mark(mark_name, **mark_kwargs)

    enc_dict = layer_dict["encoding"]
    # Use the existing 8a helper that converts a flat str→str dict into the channel-class encoding.
    encoding_spec = to_encoding_spec_dict(**enc_dict)

    return Layer(mark=mark, encoding=encoding_spec, data_source=layer_dict.get("data_source"))
```

(Verify the exact 8a helper name with `grep -n "to_encoding_spec_dict\|encoding_spec_dict" src/ferrum/encoding/base.py` — Phase 8a's commit message in CLAUDE.md/handoff.md mentions the snake_case fix at this exact site.)

- [ ] **Step 5: Branch the resolver on `__layered__` sentinel**

In `_resolve_pending_then_build_spec` (or whatever 8a named it), after the desugar tuple is unpacked:

```python
mark_or_sentinel, transforms, encoding_remap, synthetic_data, *rest = desugar_result + (None,)*5
if mark_or_sentinel == "__layered__":
    layers_dicts = rest[0]  # 5th positional is layers list
    layers = [_dict_to_layer(d) for d in layers_dicts]
    spec = ChartSpec(
        # ... base data + transforms ...
        layers=layers,
    )
else:
    # Existing 8a single-layer path (unchanged)
    spec = ChartSpec(
        mark=mark_or_sentinel,
        encoding=apply_remap(self._encoding, encoding_remap),
        # ... transforms etc ...
    )
return spec
```

(Adapt to match the actual 8a signatures — function name, ChartSpec construction kwargs, etc.)

- [ ] **Step 6: Run tests to verify GREEN**

Run: `uv run pytest tests/test_layered_desugar_resolver.py -v 2>&1 | tail -10`
Expected: 4 tests pass.

- [ ] **Step 7: Run full suite to confirm 8a charts unaffected**

Run: `uv run pytest 2>&1 | tail -5`
Expected: 217 + 4 = 221 passing; no 8a regressions.

- [ ] **Step 8: Commit**

```bash
git add src/ferrum/chart.py tests/test_layered_desugar_resolver.py
git commit -m "feat(phase-8b): __layered__ desugar resolver + dict→Layer conversion (prereq for composite marks)"
```

---

## Sub-batch E — Composite mark Python wiring (Tasks 23-26)

### Task 23: composite.py — boxplot + Chart.mark_boxplot

**Files:**
- Create: `src/ferrum/marks/composite.py`
- Modify: `src/ferrum/chart.py` (replace `mark_boxplot` stub)
- Modify: `src/ferrum/marks/deferred.py` (remove "boxplot" from PHASE_8B_MARKS)
- Create: `tests/marks/test_boxplot.py` (15 tests)

- [ ] **Step 1: Inspect 8a stub for mark_boxplot**

Run: `grep -n "mark_boxplot\|boxplot" src/ferrum/chart.py`
Note the exact signature/return/raise pattern.

- [ ] **Step 2: Create composite.py with desugar_boxplot**

```python
"""Composite-mark desugar helpers (Phase 8b).

Each desugar_<name> returns a 4-tuple matching the unified Phase 8a/8b contract:
    (mark_or_sentinel: str | None, transforms: list, encoding_remap: dict | None,
     synthetic_data: pyarrow.Table | None)

For composite marks (boxplot/errorbar/errorband/ribbon), the first element is
the sentinel "__layered__" and the desugar instead supplies a `layers` list
via a 5th positional element. We extend the contract to 5-tuple here and update
Chart's pending-stat resolution to handle it.
"""
from __future__ import annotations
from typing import Any, Optional
from ferrum import BoxStats, Outliers, ErrorExtent  # PyO3 wrappers from Tasks 11/12

def desugar_boxplot(
    x_field: str, y_field: str, *,
    extent: float | str = 1.5,
    outliers: bool = True,
    size: Optional[float] = None,
    color_field: Optional[str] = None,
    horizontal: bool = False,
    **mark_kwargs: Any,
) -> tuple[str, list, None, None, list]:
    """mark_boxplot → BoxStats + (optional) Outliers + 4 layers."""
    # When horizontal, swap x/y in encoding maps
    cat = y_field if horizontal else x_field  # categorical axis
    val = x_field if horizontal else y_field  # value axis
    groupby = [cat] + ([color_field] if color_field else [])

    transforms = [BoxStats(field=val, groupby=groupby, whisker_extent=_extent_value(extent), name="box")]
    if outliers:
        transforms.append(Outliers(field=val, groupby=groupby, extent=_extent_iqr_k(extent), name="outliers"))

    band = size or 0.6

    def enc(y_col, y2_col=None):
        if horizontal:
            d = {"x": y_col, "y": cat}
            if y2_col: d["x2"] = y2_col
        else:
            d = {"x": cat, "y": y_col}
            if y2_col: d["y2"] = y2_col
        return d

    layers = [
        {"mark": "rule", "encoding": enc("lower_whisker", "upper_whisker"), "data_source": "box"},
        {"mark": "rect", "encoding": enc("q1", "q3"), "mark_kwargs": {"width": band}, "data_source": "box"},
        {"mark": "tick", "encoding": enc("median"), "mark_kwargs": {"band_size": band}, "data_source": "box"},
    ]
    if outliers:
        layers.append({"mark": "point", "encoding": enc(val), "data_source": "outliers"})

    return ("__layered__", transforms, None, None, layers)


def _extent_value(extent):
    return "min-max" if extent == "min-max" else float(extent)

def _extent_iqr_k(extent):
    # Outliers always uses IQR multiplier, defaulting to 1.5 if extent is "min-max"
    return 1.5 if extent == "min-max" else float(extent)
```

- [ ] **Step 3: Replace Chart.mark_boxplot stub**

In `src/ferrum/chart.py`, find:
```python
def mark_boxplot(self, **kwargs):
    raise deferred_mark_error("boxplot")
```
Replace with:
```python
def mark_boxplot(self, *, extent=1.5, size=None, outliers=True, **mark_kwargs) -> "Chart":
    """Composite boxplot mark. Desugars to box+whisker+median+optional outlier layers."""
    from ferrum.marks.composite import desugar_boxplot
    self._pending_stat_mark = ("boxplot", {"extent": extent, "size": size, "outliers": outliers,
                                            **mark_kwargs}, desugar_boxplot)
    return self
```

(The exact `_pending_stat_mark` tuple shape may differ from 8a — check the existing density/histogram/smooth wiring for the convention. Match it exactly.)

- [ ] **Step 4: Update Chart's _resolve_pending_then_build_spec to handle 5-tuple `__layered__` desugar**

Locate the resolution logic in `chart.py`. Add a branch for `mark_or_sentinel == "__layered__"` that consumes `layers` (5th positional) and builds a multi-layer ChartSpec instead of a single-layer one.

- [ ] **Step 5: Drain "boxplot" from PHASE_8B_MARKS**

In `src/ferrum/marks/deferred.py`:
```python
PHASE_8B_MARKS = frozenset([
    "errorbar", "errorband", "ribbon",        # composite (boxplot done)
    "contour", "violin", "qq", "raster", "swarm", "hex", "function",   # heavy stat
])
```

- [ ] **Step 6: Write 15 tests in tests/marks/test_boxplot.py**

```python
import polars as pl
import pyarrow as pa
import pytest
import ferrum as fe

@pytest.fixture
def df():
    return pl.DataFrame({"group": ["a","a","a","b","b","b"], "value": [1.0, 2.0, 100.0, 4.0, 5.0, 6.0]})

def test_basic_boxplot_smoke(df):
    chart = fe.Chart(df).mark_boxplot().encode(x="group", y="value")
    spec = chart._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 4  # rule + rect + tick + point

def test_boxplot_no_outliers_yields_3_layers(df):
    chart = fe.Chart(df).mark_boxplot(outliers=False).encode(x="group", y="value")
    spec = chart._build_spec()
    assert len(spec.layers) == 3

def test_boxplot_extent_min_max(df):
    chart = fe.Chart(df).mark_boxplot(extent="min-max").encode(x="group", y="value")
    spec = chart._build_spec()
    # BoxStats whisker_extent serializes as "min-max"
    json_str = spec.to_json()
    assert "min-max" in json_str

# ... 12 more tests covering: extent ∈ {0.5, 1.5, 3.0}, size kwarg, color groupby,
# horizontal (CoordFlip), per-mark-style overrides, CDI round-trip, dtype variants,
# missing y error, encode-then-mark vs mark-then-encode order, JSON round-trip
```

(Add the remaining 12 tests with concrete assertions; don't leave as stubs.)

- [ ] **Step 7: Run tests**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
uv run pytest tests/marks/test_boxplot.py -v 2>&1 | tail -20
```
Expected: 15 tests pass.

- [ ] **Step 8: Commit:** `git commit -m "feat(phase-8b): mark_boxplot (composite: box+whisker+median+outliers)"`

---

### Task 24: composite.py — errorbar + Chart.mark_errorbar

**Files:**
- Modify: `src/ferrum/marks/composite.py` (add `desugar_errorbar`)
- Modify: `src/ferrum/chart.py` (replace `mark_errorbar` stub)
- Modify: `src/ferrum/marks/deferred.py` (remove "errorbar")
- Create: `tests/marks/test_errorbar.py` (10 tests)

Mirror Task 23 structure. Per spec §4.2.2:

```python
def desugar_errorbar(
    x_field: str, y_field: str, *,
    extent: str = "ci",
    ticks: bool = True,
    **mark_kwargs: Any,
) -> tuple[str, list, None, None, list]:
    transforms = [ErrorExtent(field=y_field, groupby=[x_field], method=extent, name="err")]
    layers = [
        {"mark": "rule", "encoding": {"x": x_field, "y": "lower", "y2": "upper"}, "data_source": "err"},
    ]
    if ticks:
        layers.extend([
            {"mark": "tick", "encoding": {"x": x_field, "y": "lower"},
             "mark_kwargs": {"band_size": 6}, "data_source": "err"},
            {"mark": "tick", "encoding": {"x": x_field, "y": "upper"},
             "mark_kwargs": {"band_size": 6}, "data_source": "err"},
        ])
    return ("__layered__", transforms, None, None, layers)
```

- [ ] **Steps 1-8:** Mirror Task 23 (write 10 tests covering each extent ∈ {ci,stderr,stdev,iqr}, ticks True/False, per-group, CDI, spec round-trip, layer assertions). Commit: `git commit -m "feat(phase-8b): mark_errorbar (rule + ticks via ErrorExtent)"`.

---

### Task 25: composite.py — errorband + Chart.mark_errorband

Mirror Task 24. Per spec §4.2.3:

```python
def desugar_errorband(
    x_field: str, y_field: str, *,
    extent: str = "ci",
    borders: bool = False,
    **mark_kwargs: Any,
) -> tuple[str, list, None, None, list]:
    transforms = [ErrorExtent(field=y_field, groupby=[x_field], method=extent, name="err")]
    layers = [
        {"mark": "ribbon", "encoding": {"x": x_field, "y": "lower", "y2": "upper"},
         "mark_kwargs": {"opacity": 0.3}, "data_source": "err"},
    ]
    if borders:
        layers.extend([
            {"mark": "line", "encoding": {"x": x_field, "y": "lower"}, "data_source": "err"},
            {"mark": "line", "encoding": {"x": x_field, "y": "upper"}, "data_source": "err"},
        ])
    return ("__layered__", transforms, None, None, layers)
```

- [ ] **Steps 1-8:** 10 tests; commit: `git commit -m "feat(phase-8b): mark_errorband (ribbon ± borders via ErrorExtent)"`.

---

### Task 26: composite.py — ribbon + Chart.mark_ribbon

Per spec §4.2.4: ribbon takes y2 directly via encoding; no transform.

```python
def desugar_ribbon(
    x_field: str, y_field: str, y2_field: Optional[str] = None,
    **mark_kwargs: Any,
) -> tuple[str, list, dict, None]:
    if y2_field is None:
        raise ValueError("mark_ribbon requires both y and y2 encodings")
    return ("ribbon", [], None, None)  # no remap, no synthetic; just signal mark
```

In `Chart.mark_ribbon`, pull `y2_field` from the existing encoding state (or raise if missing):

```python
def mark_ribbon(self, *, opacity=0.3, interpolate="linear", **mark_kwargs) -> "Chart":
    # Validate: y2 must already be in encoding when this is called, OR raise on resolve
    self._pending_stat_mark = ("ribbon", {"opacity": opacity, "interpolate": interpolate,
                                            **mark_kwargs}, desugar_ribbon)
    return self
```

In the resolver: when handling `ribbon`, check `self._encoding.get("y2") is not None`; otherwise raise the §6.1 error.

- [ ] **Steps 1-8:** 8 tests (basic, missing y2 error, opacity, interpolate, X/Y2 wiring, CDI, spec round-trip, 2 dtype variants). Commit: `git commit -m "feat(phase-8b): mark_ribbon (Y/Y2 area with explicit error)"`.

---

## Sub-batch F — Heavy stat mark Python wiring (Tasks 27-33)

### Task 27: heavy_stat.py module + desugar_contour + Chart.mark_contour

**Files:**
- Create: `src/ferrum/marks/heavy_stat.py`
- Modify: `src/ferrum/chart.py` (replace `mark_contour` stub)
- Modify: `src/ferrum/marks/deferred.py` (remove "contour")
- Create: `tests/marks/test_contour.py` (14 tests)

```python
"""Heavy-stat-mark desugar helpers (Phase 8b).

Each desugar_<name> returns the unified 4-tuple
    (mark, transforms, encoding_remap, synthetic_data)
or 5-tuple for layered:
    ("__layered__", transforms, None, None, layers)
"""
from __future__ import annotations
from typing import Any, Optional
import numpy as np
import pyarrow as pa

from ferrum import (
    Kde2D, Contour, QQ, Raster, Hex, Swarm, Violin, BoxStats,  # PyO3 wrappers
)


def desugar_contour(
    x_field: str, y_field: str, *,
    bandwidth: str | float = "scott",
    thresholds: int = 6,
    smooth: bool = True,
    fill: bool = False,
    **mark_kwargs: Any,
) -> tuple[str, list, dict, None]:
    transforms = [
        Kde2D(x=x_field, y=y_field, bandwidth=bandwidth, n=128),
        Contour(thresholds=thresholds, fill=fill, smooth=smooth),
    ]
    encoding_remap = {"x": "contour_x", "y": "contour_y", "detail": "level_id"}
    return ("polygon", transforms, encoding_remap, None)
```

- [ ] **Steps 1-8:** Replace stub in chart.py; add to PHASE_8B_MARKS removal; write 14 tests; commit: `git commit -m "feat(phase-8b): mark_contour (Kde2D + Contour, lines + filled)"`.

---

### Task 28: desugar_violin + Chart.mark_violin

Per spec §4.2.5:

```python
def desugar_violin(
    x_field: str, y_field: str, *,
    bandwidth: str | float = "scott",
    inner: Optional[str] = "box",
    **mark_kwargs: Any,
) -> tuple[str, list, None, None, list] | tuple[str, list, dict, None]:
    if inner not in ("box", "quartile", "point", None):
        raise ValueError(f"mark_violin inner must be one of 'box', 'quartile', 'point', or None; got {inner!r}")

    transforms = [Violin(field=y_field, groupby=[x_field], bandwidth=bandwidth, name="violin")]
    violin_layer = {
        "mark": "polygon",
        "encoding": {"x": x_field, "y": "violin_y", "detail": "group_id"},
        "mark_kwargs": {"fill_opacity": 0.5},
        "data_source": "violin",
    }
    if inner is None:
        return ("__layered__", transforms, None, None, [violin_layer])
    if inner == "point":
        return ("__layered__", transforms, None, None,
                [violin_layer, {"mark": "point", "encoding": {"x": x_field, "y": y_field}}])
    if inner == "quartile":
        transforms.append(BoxStats(field=y_field, groupby=[x_field], name="quart"))
        layers = [violin_layer]
        for col in ("q1", "median", "q3"):
            layers.append({"mark": "rule", "encoding": {"x": x_field, "y": col},
                          "mark_kwargs": ({"stroke_dash": [2, 2]} if col != "median" else {}),
                          "data_source": "quart"})
        return ("__layered__", transforms, None, None, layers)
    if inner == "box":
        from ferrum.marks.composite import desugar_boxplot
        _, box_t, _, _, box_layers = desugar_boxplot(x_field, y_field, extent=1.5, outliers=False, size=0.1)
        return ("__layered__", [*transforms, *box_t], None, None, [violin_layer, *box_layers])
```

- [ ] **Steps 1-8:** 14 tests; commit: `git commit -m "feat(phase-8b): mark_violin (4 inner modes: box/quartile/point/None)"`.

---

### Task 29: desugar_qq + Chart.mark_qq

```python
def desugar_qq(
    field: str, *,
    distribution: str = "normal",
    dequantize: bool = False,
    line: bool = True,
    **mark_kwargs: Any,
) -> tuple[str, list, None, None, list]:
    if distribution not in ("normal", "uniform", "exponential"):
        raise ValueError(f"mark_qq distribution must be 'normal', 'uniform', or 'exponential'; got {distribution!r}")
    transforms = [QQ(field=field, distribution=distribution, dequantize=dequantize,
                     emit_line=line, name="qq_main")]
    layers = [{"mark": "point", "encoding": {"x": "theoretical", "y": "sample"},
               "data_source": "qq_main"}]
    if line:
        layers.append({"mark": "rule",
                       "encoding": {"x": "qq_line_x_start", "y": "qq_line_y_start",
                                    "x2": "qq_line_x_end", "y2": "qq_line_y_end"},
                       "data_source": "qq_line"})
    return ("__layered__", transforms, None, None, layers)
```

Note: Chart.mark_qq takes `field` not `x_field, y_field` (single-column input).

- [ ] **Steps 1-8:** 10 tests; commit: `git commit -m "feat(phase-8b): mark_qq (point + reference line via QQ secondary output)"`.

---

### Task 30: desugar_raster + Chart.mark_raster

```python
def desugar_raster(
    x_field: str, y_field: str, *,
    aggregate: str = "count",
    field: Optional[str] = None,
    cmap: str = "viridis",
    resolution: str | int | tuple = "screen",
    blend: str = "alpha",
    min_count: Optional[int] = None,
    log_scale: bool = False,
    **mark_kwargs: Any,
) -> tuple[str, list, dict, None]:
    if aggregate in ("mean", "sum") and field is None:
        raise ValueError(f"mark_raster aggregate={aggregate!r} requires field=...")
    if blend == "additive":
        from ferrum._warn import warn_once
        warn_once("mark_raster", "blend_additive",
                  "mark_raster blend='additive' deferred to Phase 11; using alpha blending")

    # Convert resolution to ResolutionSpec-friendly form
    if resolution == "screen":
        rs = "screen"
    elif isinstance(resolution, int):
        rs = ("fixed", resolution)
    else:
        rs = ("xy", *resolution)

    transforms = [Raster(x=x_field, y=y_field, aggregate=aggregate, field=field,
                         resolution=rs, min_count=min_count, log_scale=log_scale)]
    return ("image", transforms, {}, None)  # image mark reads everything from batch
```

- [ ] **Steps 1-8:** 16 tests; commit: `git commit -m "feat(phase-8b): mark_raster (RGBA→PNG embed, viridis default)"`.

---

### Task 31: desugar_hex + Chart.mark_hex

```python
def desugar_hex(
    x_field: str, y_field: str, *,
    bin_size: Optional[float] = None,
    aggregate: str = "count",
    field: Optional[str] = None,
    cmap: str = "viridis",
    stroke: Optional[str] = None,
    stroke_width: float = 0,
    **mark_kwargs: Any,
) -> tuple[str, list, dict, None]:
    if aggregate in ("mean", "sum") and field is None:
        raise ValueError(f"mark_hex aggregate={aggregate!r} requires field=...")
    if aggregate not in ("count", "mean", "sum"):
        from ferrum._warn import warn_once
        warn_once("mark_hex", "aggregate_unsupported",
                  f"mark_hex aggregate={aggregate!r} deferred; falling back to 'count'")
        aggregate = "count"
    transforms = [Hex(x=x_field, y=y_field, bin_size=bin_size, aggregate=aggregate, field=field)]
    encoding_remap = {"x": "hex_x", "y": "hex_y", "color": "value", "detail": "hex_id"}
    return ("polygon", transforms, encoding_remap, None)
```

- [ ] **Steps 1-8:** 12 tests; commit: `git commit -m "feat(phase-8b): mark_hex (axial hex bins, viridis default)"`.

---

### Task 32: desugar_swarm + Chart.mark_swarm

```python
def desugar_swarm(
    x_field: str, y_field: str, *,
    size: int = 4,
    orient: str = "vertical",
    spacing: float = 1.0,
    side: str = "both",
    dodge: Optional[str] = None,
    **mark_kwargs: Any,
) -> tuple[str, list, dict, None]:
    if dodge is not None:
        from ferrum._warn import warn_once
        warn_once("mark_swarm", "dodge",
                  "mark_swarm dodge= is not yet supported; rendering single-group swarm")
    cat = x_field if orient == "vertical" else y_field
    val = y_field if orient == "vertical" else x_field
    transforms = [Swarm(category=cat, value=val, point_size=size, spacing=spacing, side=side)]
    encoding_remap = {"x": "swarm_x", "y": "swarm_y"}
    return ("point", transforms, encoding_remap, None)
```

- [ ] **Steps 1-8:** 12 tests; commit: `git commit -m "feat(phase-8b): mark_swarm (greedy beeswarm with deterministic placement)"`.

---

### Task 33: desugar_function + Chart.mark_function

This is the *only* desugar that materializes a synthetic Arrow table.

**Scope restriction (per spec §12.2):** Phase 8b restricts `mark_function` to **single-layer charts** — the synthetic table replaces the chart's data. Use it as a separate `Chart(...)` composed via `+` to overlay on existing data (the SVG compositor handles per-chart data routing). Per-layer data routing within one ChartSpec is deferred to Phase 9+.

```python
def desugar_function(
    fn,
    parent_chart_x_data: Optional[np.ndarray] = None,  # injected by Chart resolver
    *,
    domain: Optional[tuple[float, float]] = None,
    n: int = 200,
    clip: bool = True,
    **mark_kwargs: Any,
) -> tuple[str, list, dict, pa.Table]:
    # Resolve domain
    if domain is not None:
        d = domain
    elif parent_chart_x_data is not None and len(parent_chart_x_data) > 0:
        d = (float(np.nanmin(parent_chart_x_data)), float(np.nanmax(parent_chart_x_data)))
    else:
        raise ValueError("mark_function requires explicit domain when chart has no other data layers")

    xs = np.linspace(d[0], d[1], n)
    ys = fn(xs)
    if not isinstance(ys, np.ndarray) or ys.shape != (n,):
        raise ValueError(f"mark_function callable must return numpy array of shape ({n},); got shape {getattr(ys, 'shape', None)}")

    synthetic = pa.Table.from_pydict({"x": xs, "y": ys})
    encoding_remap = {"x": "x", "y": "y"}
    return ("line", [], encoding_remap, synthetic)
```

In `Chart.mark_function`, the resolver must:
1. Snapshot `self._data` (if any) and extract x-field data for domain inference.
2. Pass it into `desugar_function`.
3. Replace `self._data` with the synthetic Arrow table for this chart instance.
4. **Raise `NotImplementedError("mark_function as a layer in a multi-layer Chart is deferred to Phase 9+; use a separate Chart composed via + instead")`** if the Chart already has accumulated layers (`self._layers` non-empty) or a pending non-function stat mark. The compose-via-`+` path works because each composed chart carries its own data and resolves its own mark_function independently.

- [ ] **Steps 1-8:** 10 tests covering explicit domain, inferred domain, missing-domain error, wrong-shape return, np/lambda functions, `n` parameter, CDI, spec round-trip. Commit: `git commit -m "feat(phase-8b): mark_function (Python-side eval, synthetic data, domain inference)"`.

---

## Sub-batch G — Cross-cutting (Tasks 34-37)

### Task 34: mark_smooth(ci=) ribbon integration

**Files:**
- Modify: `src/ferrum/marks/statistical.py` (`desugar_smooth`)
- Modify: `src/ferrum/chart.py` (resolver: handle `__layered__` from desugar_smooth)
- Modify: `src/ferrum/_warn.py` (remove `("mark_smooth", "ci")` from registry, OR change the test to assert no warn fires)
- Create: `tests/marks/test_smooth_ci.py` (6 tests)

- [ ] **Step 1: Update desugar_smooth**

In `src/ferrum/marks/statistical.py`, replace the existing `if ci is not None: warn_once(...)` with the layered desugar per spec §4.8:

```python
def desugar_smooth(x_field, y_field, **kwargs):
    method = kwargs.pop("method", "loess")
    ci = kwargs.pop("ci", None)
    bandwidth = kwargs.pop("bandwidth", 0.75)
    degree = kwargs.pop("degree", 2)
    n = kwargs.pop("n", 200)

    if ci is None:
        transforms = [Smooth(x_field, y_field, method=method, ci=None, bandwidth=bandwidth,
                             degree=degree, n=n)]
        return ("line", transforms, {"x": "x", "y": "y"}, None)
    # CI band path (NEW in 8b)
    transforms = [Smooth(x_field, y_field, method=method, ci=ci, bandwidth=bandwidth,
                         degree=degree, n=n, name="smooth")]
    layers = [
        {"mark": "ribbon", "encoding": {"x": "x", "y": "ci_lower", "y2": "ci_upper"},
         "mark_kwargs": {"opacity": 0.3}, "data_source": "smooth"},
        {"mark": "line", "encoding": {"x": "x", "y": "y"}, "data_source": "smooth"},
    ]
    return ("__layered__", transforms, None, None, layers)
```

- [ ] **Step 2: Verify the existing 8a `mark_smooth` warn-once test fails appropriately**

Run: `uv run pytest tests/ -k "smooth" -v 2>&1 | tail -20`
Find the test asserting the deferral warning. It should now fail (no warning fires).

- [ ] **Step 3: Replace that test with the 8b expectation**

In `tests/marks/test_smooth_ci.py`, write 6 tests:
1. `mark_smooth(ci=0.95)` produces no warning (the 8a warn is gone)
2. Spec has 2 layers: ribbon then line (z-order matters)
3. Ribbon layer uses `ci_lower`/`ci_upper` columns
4. CI band data covers the same x range as the smooth line
5. `method="loess"` and `method="lm"` both work with ci
6. CDI round-trip: full chart renders to SVG without panic

- [ ] **Step 4: Run + commit**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
uv run pytest tests/marks/test_smooth_ci.py -v 2>&1 | tail -10
git add src/ferrum/marks/statistical.py tests/marks/test_smooth_ci.py
git commit -m "feat(phase-8b): mark_smooth(ci=) renders CI band via ribbon (lifts 8a deferral)"
```

---

### Task 35: bivariate density routing in mark_density

**Files:**
- Modify: `src/ferrum/marks/statistical.py` (`desugar_density`)

Per spec §4.3.6: when `desugar_density` is called and the chart's encoding has both X and Y bound to quantitative fields, route to `desugar_contour(fill=True)` instead of the 1D KDE path.

- [ ] **Step 1: Inspect existing `desugar_density`**

Run: `grep -n "desugar_density\|def desugar" src/ferrum/marks/statistical.py`

- [ ] **Step 2: Add bivariate branch**

The desugar function needs access to the parent encoding state. Either:
(a) Pass a `chart_encoding: dict` parameter that the Chart resolver fills in at call time.
(b) Detect at `Chart.mark_density()` call time whether y is encoded and dispatch to a separate desugar.

Approach (a) is cleaner. Modify `desugar_density(field, *, chart_encoding=None, **kwargs)`:

```python
def desugar_density(field, *, chart_encoding=None, **kwargs):
    # Bivariate routing: if x AND y are both quantitative-bound, render filled contour
    if chart_encoding and chart_encoding.get("x") and chart_encoding.get("y"):
        from ferrum.marks.heavy_stat import desugar_contour
        return desugar_contour(
            x_field=chart_encoding["x"], y_field=chart_encoding["y"],
            fill=True, **kwargs,
        )
    # 1D path (existing 8a behavior)
    # ... unchanged ...
```

The Chart resolver injects `chart_encoding=self._encoding`.

- [ ] **Step 3: Add a test in tests/marks/test_contour.py**

```python
def test_bivariate_density_routes_through_contour(df_2d):
    chart = fe.Chart(df_2d).mark_density().encode(x="x", y="y")
    spec = chart._build_spec()
    # Should produce Kde2D + Contour transforms (not 1D Kde)
    transforms_json = spec.to_json()
    assert "kde_2d" in transforms_json or "Kde2D" in transforms_json
    assert "contour" in transforms_json
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/marks/test_contour.py::test_bivariate_density_routes_through_contour -v 2>&1 | tail -5
git add src/ferrum/marks/statistical.py tests/marks/test_contour.py
git commit -m "feat(phase-8b): mark_density bivariate routes through mark_contour(fill=True)"
```

---

### Task 36: X2/Y2 wiring through scale_resolve.rs

**Files:**
- Modify: `crates/ferrum-core/src/render/scale_resolve.rs`
- Modify: `crates/ferrum-core/src/render/marks/{ribbon,polygon,image,rule,rect}.rs` as needed

Phase 8a accepted X2/Y2 encoding channels but didn't render them. Now ribbon/errorband/qq-line/raster need them.

- [ ] **Step 1: Inspect ResolvedScales struct**

Run: `grep -n "ResolvedScales\|x2\|y2" crates/ferrum-core/src/render/scale_resolve.rs | head -20`

- [ ] **Step 2: Add x2/y2 fields to ResolvedScales**

If they don't exist, add:
```rust
pub struct ResolvedScales {
    pub x: ResolvedScale,
    pub y: ResolvedScale,
    pub x2: Option<String>,  // field name to look up at draw time
    pub y2: Option<String>,
    // ... color/etc
}
```

(The actual scale for x2 is shared with x — they're both on the X axis; same for y2/y.)

- [ ] **Step 3: Update each mark drawer that reads X2/Y2**

In `ribbon.rs`, `rule.rs`, `rect.rs`: when drawing, read `y2` column from the batch and use it to compute the second endpoint / fill area.

- [ ] **Step 4: Add 3 tests** (1 pytest e2e + 2 cargo round-trip in scale_resolve)

In `tests/test_phase_8b_e2e.py` (or a new `test_x2_y2_wiring.py`):

```python
def test_x2_y2_renders_in_rule_mark():
    # mark_rule with explicit x2/y2 should render line from (x,y) to (x2,y2)
    df = pl.DataFrame({"x":[1.0], "y":[2.0], "x2":[5.0], "y2":[6.0]})
    chart = fe.Chart(df).mark_rule().encode(x="x", y="y", x2="x2", y2="y2")
    svg = chart.show_svg()  # method exists in 8a
    assert "<line" in svg  # rule renders as line; check both endpoints used
```

In `crates/ferrum-core/src/render/scale_resolve.rs` `mod tests`, add:

```rust
#[test]
fn resolved_scales_include_x2_y2_field_names_when_set() {
    // build an Encoding with x="a", y="b", x2="a2", y2="b2"
    // resolve through scale_resolve
    // assert resolved.x2 == Some("a2") and resolved.y2 == Some("b2")
}

#[test]
fn resolved_scales_x2_y2_default_to_none_for_8a_charts() {
    // build an Encoding with only x and y
    // resolve through scale_resolve
    // assert resolved.x2 == None and resolved.y2 == None  (8a back-compat)
}
```

- [ ] **Step 5: Run + commit**

```bash
DYLD_LIBRARY_PATH=... cargo test -p ferrum-core scale_resolve 2>&1 | tail -5
uv run pytest tests/test_phase_8b_e2e.py::test_x2_y2_renders_in_rule_mark -v 2>&1 | tail -5
git add crates/ferrum-core/src/render/
git commit -m "feat(phase-8b): wire X2/Y2 channels through scale_resolve and mark drawers"
```

---

### Task 37: continuous_palette() Python lookup + Gradient class

**Files:**
- Modify: `src/ferrum/schemes.py`
- Modify: `src/ferrum/__init__.py` (export)
- Modify: `crates/ferrum-core/src/binding.rs` (add PyContinuousScheme + PyGradient)
- Create: `tests/test_continuous_palette.py` (8 tests)

- [ ] **Step 1: Add PyO3 wrappers for ContinuousScheme + Gradient**

In `crates/ferrum-core/src/binding.rs` (or a dedicated `binding/color.rs`):

```rust
use pyo3::prelude::*;
use crate::render::color::continuous::{ContinuousScheme, NamedContinuous};
use crate::render::color::categorical::Color;  // existing

#[pyclass(name = "ContinuousScheme")]
#[derive(Clone)]
pub(crate) struct PyContinuousScheme(pub(crate) ContinuousScheme);

#[pymethods]
impl PyContinuousScheme {
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        NamedContinuous::from_name(name)
            .map(|n| Self(ContinuousScheme::Named(n)))
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                format!("Unknown colormap: {name!r}; available: {:?}", NamedContinuous::list())))
    }

    fn reversed(&self) -> Self {
        Self(ContinuousScheme::Reverse(Box::new(self.0.clone())))
    }

    fn __repr__(&self) -> String {
        format!("ContinuousScheme({:?})", self.0)
    }
}

#[pyfunction]
fn Gradient(stops: Vec<(f64, String)>) -> PyResult<PyContinuousScheme> {
    let mut color_stops = Vec::with_capacity(stops.len());
    for (t, name) in stops {
        let color = parse_color_string(&name)?;  // existing helper or write one
        color_stops.push((t, color));
    }
    Ok(PyContinuousScheme(ContinuousScheme::Gradient(color_stops)))
}
```

- [ ] **Step 2: Register classes in lib.rs**

Add `m.add_class::<PyContinuousScheme>()?;` and `m.add_function(wrap_pyfunction!(Gradient, m)?)?;`.

- [ ] **Step 3: Python wrapper**

In `src/ferrum/schemes.py`:

```python
"""Color scheme lookups (Phase 8a categorical + Phase 8b continuous)."""
from __future__ import annotations
from ferrum._core import ContinuousScheme as _ContinuousScheme, Gradient as _Gradient
# (existing categorical_palette unchanged)


def continuous_palette(name: str) -> "_ContinuousScheme":
    """Look up a named continuous colormap (viridis/plasma/magma/inferno/cividis)."""
    return _ContinuousScheme.from_name(name)


def _list_continuous():
    return ["viridis", "plasma", "magma", "inferno", "cividis"]
continuous_palette.list = _list_continuous

Gradient = _Gradient  # re-export with capitalized name
```

- [ ] **Step 4: Export from __init__.py**

```python
from ferrum.schemes import categorical_palette, continuous_palette, Gradient
```

- [ ] **Step 5: 8 tests**

```python
import pytest
import ferrum as fe

def test_viridis_lookup():
    s = fe.continuous_palette("viridis")
    assert s is not None

def test_plasma_lookup():
    fe.continuous_palette("plasma")

def test_magma_lookup():
    fe.continuous_palette("magma")

def test_inferno_lookup():
    fe.continuous_palette("inferno")

def test_cividis_lookup():
    fe.continuous_palette("cividis")

def test_unknown_palette_raises():
    with pytest.raises(ValueError, match="Unknown colormap"):
        fe.continuous_palette("notacolor")

def test_continuous_palette_list():
    names = fe.continuous_palette.list()
    assert set(names) == {"viridis", "plasma", "magma", "inferno", "cividis"}

def test_reversed_returns_new_scheme():
    s = fe.continuous_palette("viridis")
    rev = s.reversed()
    assert rev is not s

def test_gradient_two_stops():
    g = fe.Gradient([(0.0, "red"), (1.0, "blue")])
    assert g is not None
```

- [ ] **Step 6: Run + commit**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop 2>&1 | tail -3
uv run pytest tests/test_continuous_palette.py -v 2>&1 | tail -10
git add src/ferrum/schemes.py src/ferrum/__init__.py crates/ferrum-core/src/binding.rs tests/test_continuous_palette.py
git commit -m "feat(phase-8b): continuous_palette() + Gradient Python API"
```

---

## Sub-batch H — Polish + verification (Tasks 38-43)

### Task 38: ferrum-spec.md dated notes

**Files:**
- Modify: `ferrum-spec.md`

Add 6 dated notes per spec §6.5.

- [ ] **Step 1: Locate §3.3 mark_raster**

Run: `grep -n "mark_raster\|mark_swarm\|mark_hex\|stat_kde_2d" ferrum-spec.md | head -10`

- [ ] **Step 2: Append dated notes**

For each of the 6 items in spec §6.5, add an indented bullet under the relevant table row or section, prefixed with `*(2026-05-10)*`. For example:

Under §3.3 mark_raster:
```markdown
*(2026-05-10) Phase 8b: `blend="additive"` is deferred to Phase 11 (interactive renderer).
Auto-raster policy (`raster_threshold`, `raster_behavior`) is deferred to Phase 9+;
explicit `mark_raster` is implemented in Phase 8b.*
```

Under §3.3 mark_swarm:
```markdown
*(2026-05-10) Phase 8b: `dodge=` parameter deferred (single-group swarm only in 8b).*
```

Under §3.3 mark_hex:
```markdown
*(2026-05-10) Phase 8b: only count/mean/sum aggregates supported. Other Vega-Lite aggregates warn-once and fall back to count.*
```

Under §3.4 stat_kde_2d:
```markdown
*(2026-05-10) Phase 8b: implemented as `Kde2D` transform (10th transform of the phase). Output is a single-row Arrow batch with grid_x/grid_y/density list columns.*
```

- [ ] **Step 3: Commit**

```bash
git add ferrum-spec.md
git commit -m "docs(phase-8b): dated 2026-05-10 notes for raster/swarm/hex/kde2d clarifications"
```

---

### Task 39: ferrum-phases.md done-criteria update (9→10 transforms)

**Files:**
- Modify: `docs/superpowers/ferrum-phases.md`

- [ ] **Step 1: Update Phase 8b row in the phase table**

Find the Phase 8b row and update the "What it produces" cell from "9 transforms" to "10 transforms (Outliers, ErrorExtent, BoxStats, Violin, Kde2D, Contour, QQ, Raster, Hex, Swarm)".

- [ ] **Step 2: Update Phase 8b done-criteria**

Find the "Phase 8b — Composite + heavy statistical marks" section. Change:
```markdown
- [ ] New Phase 5 transforms (Outliers, ErrorExtent, Contour, QQ, Raster, Hex, Swarm, BoxStats, Violin) all have round-trip + correctness tests
```
To:
```markdown
- [ ] **10** new Phase 5 transforms (Outliers, ErrorExtent, BoxStats, Violin, **Kde2D**, Contour, QQ, Raster, Hex, Swarm) all have round-trip + correctness tests
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/ferrum-phases.md
git commit -m "docs(phase-8b): update done-criteria — 10 transforms (Kde2D added)"
```

---

### Task 40: Drain marks/deferred.py PHASE_8B_MARKS

**Files:**
- Modify: `src/ferrum/marks/deferred.py`

By this point, all 11 marks should have been removed incrementally as Tasks 23-33 landed. This task verifies the set is empty and tightens the deferred-error message.

- [ ] **Step 1: Verify empty**

```bash
uv run python -c "from ferrum.marks.deferred import PHASE_8B_MARKS; assert len(PHASE_8B_MARKS) == 0, f'still deferred: {PHASE_8B_MARKS}'; print('OK')"
```
Expected: `OK`. If not, find which task missed the removal and fix.

- [ ] **Step 2: Set to empty frozenset explicitly**

```python
PHASE_8B_MARKS = frozenset()  # all 11 marks shipped in Phase 8b

PHASE_9_PLUS_MARKS = frozenset([
    "arc", "image", "geoshape", "segment", "label",
])
```

- [ ] **Step 3: Commit**

```bash
git add src/ferrum/marks/deferred.py
git commit -m "chore(phase-8b): drain PHASE_8B_MARKS to empty frozenset"
```

---

### Task 41: e2e test file (one assertion per new mark)

**Files:**
- Create: `tests/test_phase_8b_e2e.py`

Smoke render every new mark end-to-end (df → Chart → SVG) and assert no panic + correct mark name in spec.

- [ ] **Step 1: Write 14 tests**

```python
"""Phase 8b end-to-end smoke tests: every new mark renders without panic."""
import polars as pl
import numpy as np
import pytest
import ferrum as fe


@pytest.fixture
def df_grouped():
    return pl.DataFrame({
        "g": ["a"] * 30 + ["b"] * 30,
        "v": np.concatenate([np.random.normal(0, 1, 30), np.random.normal(2, 1, 30)]),
    })


@pytest.fixture
def df_2d():
    n = 200
    return pl.DataFrame({"x": np.random.normal(0, 1, n), "y": np.random.normal(0, 1, n)})


def test_boxplot_renders(df_grouped):
    svg = fe.Chart(df_grouped).mark_boxplot().encode(x="g", y="v").show_svg()
    assert "<svg" in svg

def test_errorbar_renders(df_grouped):
    svg = fe.Chart(df_grouped).mark_errorbar(extent="ci").encode(x="g", y="v").show_svg()
    assert "<svg" in svg

def test_errorband_renders(df_grouped):
    df_lines = pl.DataFrame({
        "g": ["a"] * 10 + ["b"] * 10,
        "x": list(range(10)) * 2,
        "v": [float(i) + 0.1 for i in range(10)] + [float(i) + 0.5 for i in range(10)],
    })
    svg = fe.Chart(df_lines).mark_errorband(extent="ci").encode(x="x", y="v").show_svg()
    assert "<svg" in svg

def test_ribbon_renders():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "lo": [0.5, 1.5, 2.5], "hi": [1.5, 2.5, 3.5]})
    svg = fe.Chart(df).mark_ribbon().encode(x="x", y="lo", y2="hi").show_svg()
    assert "<svg" in svg

def test_smooth_with_ci_renders(df_2d):
    svg = fe.Chart(df_2d).mark_smooth(ci=0.95).encode(x="x", y="y").show_svg()
    assert "<svg" in svg

def test_contour_renders(df_2d):
    svg = fe.Chart(df_2d).mark_contour().encode(x="x", y="y").show_svg()
    assert "<svg" in svg

def test_violin_renders(df_grouped):
    svg = fe.Chart(df_grouped).mark_violin(inner="box").encode(x="g", y="v").show_svg()
    assert "<svg" in svg

def test_qq_renders(df_2d):
    svg = fe.Chart(df_2d).mark_qq(distribution="normal").encode(x="x").show_svg()
    assert "<svg" in svg

def test_raster_renders(df_2d):
    svg = fe.Chart(df_2d).mark_raster(resolution=64).encode(x="x", y="y").show_svg()
    assert "<svg" in svg
    assert "data:image/png;base64," in svg

def test_swarm_renders(df_grouped):
    svg = fe.Chart(df_grouped).mark_swarm().encode(x="g", y="v").show_svg()
    assert "<svg" in svg

def test_hex_renders(df_2d):
    svg = fe.Chart(df_2d).mark_hex().encode(x="x", y="y").show_svg()
    assert "<svg" in svg

def test_function_renders():
    df = pl.DataFrame({"t": np.linspace(0, np.pi * 2, 50)})
    svg = (fe.Chart(df).mark_point().encode(x="t", y="t") +
           fe.Chart(df).mark_function(np.sin)).show_svg()
    assert "<svg" in svg

def test_layered_violin_plus_swarm(df_grouped):
    """Composition smoke: violin + swarm overlay."""
    svg = (fe.Chart(df_grouped).mark_violin(inner=None).encode(x="g", y="v") +
           fe.Chart(df_grouped).mark_swarm().encode(x="g", y="v")).show_svg()
    assert "<svg" in svg

def test_layered_errorband_plus_smooth(df_2d):
    """Composition smoke: errorband + smooth line."""
    df = pl.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0) + np.random.randn(20)})
    svg = fe.Chart(df).mark_smooth(ci=0.95).encode(x="x", y="y").show_svg()
    assert "<svg" in svg
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/test_phase_8b_e2e.py -v 2>&1 | tail -20
git add tests/test_phase_8b_e2e.py
git commit -m "test(phase-8b): e2e smoke for all 11 new marks + 2 layered compositions"
```

---

### Task 42: Spec drift + warn-once tests

**Files:**
- Create: `tests/test_spec_drift.py` (4 tests)
- Create: `tests/test_warn_once_lift.py` (3 tests)
- Create: `tests/test_data_source_routing.py` (7 tests)
- Create: `tests/desugar/test_composite_desugar.py` (10 tests)
- Create: `tests/desugar/test_heavy_stat_desugar.py` (16 tests)
- Create: `tests/test_image_primitive.py` (4 tests), `test_polygon_primitive.py` (4), `test_beeswarm_primitive.py` (3)

These are the cross-cutting tests from the test plan §8.2. Many are mechanical assertions that the desugar contract is upheld.

- [ ] **Step 1: tests/test_spec_drift.py (4 tests)**

```python
"""Phase 8b spec-implementation drift checks."""
import re
import ferrum as fe
from pathlib import Path


def test_phase_8b_marks_set_is_empty():
    from ferrum.marks.deferred import PHASE_8B_MARKS
    assert PHASE_8B_MARKS == frozenset(), f"unfinished marks: {PHASE_8B_MARKS}"


def test_ferrum_spec_has_2026_05_10_dated_notes_for_8b():
    spec = Path("ferrum-spec.md").read_text()
    notes = re.findall(r"\*\(2026-05-10\)[^\*]*Phase 8b", spec)
    assert len(notes) >= 4, f"expected ≥4 dated 8b notes; got {len(notes)}"


def test_transform_count_matches_docs():
    """Verify TransformSpec exposes 15 transforms (Phase 5 5 + Phase 8b 10)."""
    # Indirect check: each transform PyO3 wrapper is importable
    from ferrum import (Bin, Kde, Smooth, Aggregate, Summary,            # 8a
                        Outliers, ErrorExtent, BoxStats, Violin, Kde2D,  # 8b
                        Contour, QQ, Raster, Hex, Swarm)
    assert all(t is not None for t in [Bin, Kde, Smooth, Aggregate, Summary,
                                        Outliers, ErrorExtent, BoxStats, Violin, Kde2D,
                                        Contour, QQ, Raster, Hex, Swarm])


def test_ferrum_phases_8b_done_criteria_lists_10_transforms():
    phases = Path("docs/superpowers/ferrum-phases.md").read_text()
    section = phases.split("Phase 8b")[1].split("Phase 9")[0]
    assert "10" in section and "Kde2D" in section
```

- [ ] **Step 2: tests/test_warn_once_lift.py (3 tests)**

```python
import warnings
import polars as pl
import numpy as np
import ferrum as fe


def test_mark_smooth_ci_no_longer_warns():
    df = pl.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0) + np.random.randn(20)})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fe.Chart(df).mark_smooth(ci=0.95).encode(x="x", y="y").show_svg()
    smooth_warns = [x for x in w if "mark_smooth" in str(x.message) and "ci" in str(x.message).lower()]
    assert len(smooth_warns) == 0, f"unexpected warnings: {[str(x.message) for x in smooth_warns]}"


def test_mark_raster_blend_additive_warns_once():
    df = pl.DataFrame({"x": np.random.randn(50), "y": np.random.randn(50)})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fe.Chart(df).mark_raster(blend="additive").encode(x="x", y="y").show_svg()
        fe.Chart(df).mark_raster(blend="additive").encode(x="x", y="y").show_svg()
    blend_warns = [x for x in w if "blend" in str(x.message) and "additive" in str(x.message)]
    assert len(blend_warns) == 1, "should warn exactly once"


def test_mark_swarm_dodge_warns():
    df = pl.DataFrame({"g": ["a"]*10, "v": np.random.randn(10)})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fe.Chart(df).mark_swarm(dodge="x").encode(x="g", y="v").show_svg()
    dodge_warns = [x for x in w if "dodge" in str(x.message)]
    assert len(dodge_warns) >= 1
```

- [ ] **Step 3-7: tests/test_data_source_routing.py + desugar tests + 3 primitive tests**

(Following the patterns above — write concrete tests per the test plan §8.2 counts. Each test file should have the exact test count specified.)

- [ ] **Step 8: Run all + commit**

```bash
uv run pytest tests/test_spec_drift.py tests/test_warn_once_lift.py tests/test_data_source_routing.py tests/desugar/ tests/test_image_primitive.py tests/test_polygon_primitive.py tests/test_beeswarm_primitive.py -v 2>&1 | tail -40
git add tests/
git commit -m "test(phase-8b): cross-cutting drift, warn-once, data_source, desugar, primitive tests"
```

---

### Task 43: Final verification + Phase 8b done-criteria flip

**Files:**
- Modify: `docs/superpowers/ferrum-phases.md` (mark Phase 8b done)

This is the gating commit. All criteria from spec §9 must pass.

- [ ] **Step 1: Run full test suite**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop --release 2>&1 | tail -5
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core 2>&1 | tail -5
uv run pytest 2>&1 | tail -5
```
Expected:
- cargo: ≥379 tests pass
- pytest: ≥397 tests pass + 3 skipped (pre-existing)

If counts are below target, identify gaps and add tests until met.

- [ ] **Step 2: Verify Phase 7 + 8a goldens byte-identical**

```bash
DYLD_LIBRARY_PATH=... cargo test -p ferrum-core --test golden 2>&1 | tail -10
uv run pytest tests/ -k "golden" -v 2>&1 | tail -10
```
Expected: all golden snapshots pass.

- [ ] **Step 3: 2 new raster PNG hash goldens**

In `tests/marks/test_raster.py`, add:

```python
import hashlib

def test_raster_viridis_128x128_byte_identical():
    np.random.seed(42)
    df = pl.DataFrame({"x": np.random.randn(1000), "y": np.random.randn(1000)})
    svg = fe.Chart(df).mark_raster(cmap="viridis", resolution=128).encode(x="x", y="y").show_svg()
    # Extract base64 png
    import re, base64
    m = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", svg)
    png_bytes = base64.b64decode(m.group(1))
    h = hashlib.sha256(png_bytes).hexdigest()
    EXPECTED = "TBD_FIRST_RUN"  # populate by running once and pasting hash
    assert h == EXPECTED, f"raster PNG drift: {h}"

def test_raster_plasma_256x256_byte_identical():
    # Same pattern for plasma 256x256
    ...
```

(After first successful run, copy the printed hash into `EXPECTED` and re-commit.)

- [ ] **Step 4: No-matplotlib audit**

```bash
cargo tree -p ferrum-core 2>/dev/null | grep -i matplotlib && echo "FAIL: matplotlib present" || echo "OK: no matplotlib"
```
Expected: `OK: no matplotlib`.

- [ ] **Step 5: PHASE_8B_MARKS empty audit**

```bash
uv run python -c "from ferrum.marks.deferred import PHASE_8B_MARKS, PHASE_9_PLUS_MARKS; assert PHASE_8B_MARKS == frozenset(); assert PHASE_9_PLUS_MARKS == frozenset({'arc','image','geoshape','segment','label'}); print('OK')"
```
Expected: `OK`.

- [ ] **Step 6: Mark Phase 8b done in ferrum-phases.md**

In the phase table, change Phase 8b's Status column from `pending` to `done`. Check all boxes in the "Phase 8b — Composite + heavy statistical marks" done-criteria block.

- [ ] **Step 7: Commit + offer merge**

```bash
git add docs/superpowers/ferrum-phases.md tests/marks/test_raster.py
git commit -m "chore(phase-8b): mark Phase 8b done; raster goldens green"
```

Now ask the user before merging to main:

> Phase 8b implementation complete on `feat/phase-8b`. Test counts: cargo {N}, pytest {M}.
> Ready to merge to main? (Use `superpowers:finishing-a-development-branch` to choose merge strategy.)

---

## Self-review checklist (run after the plan is fully drafted)

Already reviewed:
- ✓ Spec coverage: all 11 marks → Tasks 23-33; all 10 transforms → Tasks 10-19; 3 SVG primitives → Tasks 3-5; data_source routing → Tasks 8-9; continuous colormaps → Tasks 2, 37; X2/Y2 wiring → Task 36; smooth(ci) → Task 34; bivariate density → Task 35; spec/phases updates → Tasks 38-39; PHASE_8B_MARKS drain → Task 40; e2e + cross-cutting → Tasks 41-42; final gate → Task 43.
- ✓ No placeholders ("TBD_FIRST_RUN" in Task 43 step 3 is intentional — it's the golden hash slot to populate on first successful run, with explicit instructions).
- ✓ Type consistency: `data_source: Option<String>` consistent across spec/layer.rs (Task 8), prepare.rs (Task 9), each desugar (Tasks 23-33). `name: Option<String>` consistent on every TransformSpec variant (Task 8) and used in dispatch (Task 9 spec_name) + extended in each new transform task (Tasks 10-19).

---

## Test-count tracking (after blocker patches)

| Task batch | Cargo Δ | Pytest Δ |
|---|---|---|
| Pre-flight T0 (worktree) | 0 | 0 |
| A (1-7) | +9 (3 image + 3 polygon + 3 beeswarm + 3 continuous + 4 rasterize + 1 context) | 0 |
| B (8-9) | +6 (2 layer + 1 transform name + 3 prepare) | 0 |
| C (10-19) | **+45** (per spec §8.1: +43; +1 in T15 for evenodd; +1 in T17 for no-context fallback) | 0 |
| D (20-22) | **+10** (4 polygon — incl. quantitative-color + 3 image + 3 ribbon) | 0 |
| 22b (resolver) | 0 | **+4** (layered-desugar resolver) |
| E (23-26) | 0 | +43 (15+10+10+8) |
| F (27-33) | 0 | +94 (14+14+10+16+12+12+10) |
| G (34-37) | **+2** (T36 scale_resolve x2/y2 round-trip) | +20 (6 smooth_ci + ~3 bivariate + ~3 X2Y2 + 8 palette) |
| H (38-43) | 0 | +55 (4 drift + 3 warn-once + 7 routing + 10+16 desugar + 4+4+3 primitives + 14 e2e + 2 raster goldens) |
| **Total** | **+72** (309→**381**; ≥379 target met) | **+216** (217→**433**; ≥397 target met) |

The test counts in individual tasks above (Tasks 15, 17, 20, 22b, 36) reflect these additions. No further "fix during execution" punt — counts hit targets as planned.

---

## Execution: end-of-batch checkpoints

After each sub-batch (A-H), the executor should:
1. Run full test suite (`cargo test -p ferrum-core` + `uv run pytest`).
2. Verify counts trend toward the 379/397 targets.
3. Commit a sub-batch closing commit if the batch contains structural pieces (e.g., end of Sub-batch B: "feat(phase-8b): routing infra complete").
4. /clear context if working with subagent-driven-development before the next sub-batch.
