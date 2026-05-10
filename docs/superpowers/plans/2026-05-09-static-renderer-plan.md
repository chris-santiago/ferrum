# Phase 7 — Static Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 7 — a Rust static renderer that takes a `ChartSpec` + `RecordBatch` + `ThemeInputs` + `Viewport` and produces deterministic SVG strings (and PNG bytes via resvg). Pipeline: transforms → scale resolve → layout (Phase 6 with `FontdueMetrics`) → per-mark draw → SVG/PNG. Exposed to Python via `ferrum._core.render_svg` and `ferrum._core.render_png`.

**Architecture:** Per-element modules (`color`, `palette`, `font`, `format`, `svg`, `scale_resolve`, `prepare`, `draw`) plus a `marks/` directory with one module per primitive (8 marks + 3 internal: axis, legend, strip_title). Sealed-enum dispatch (`dispatch_mark`) mirrors Phase 5's transform pattern. Hand-rolled `SvgBuffer` for deterministic output; `palette::Srgba<u8>` as the `Color` type alias; `fontdue` + bundled Inter Regular for layout measurement and font embedding. Hybrid error policy: structural → `RenderError`/`PyValueError`, geometric edge → skip + `RenderWarning`.

**Tech Stack:** Rust 2021; PyO3 0.28 (existing); arrow 58 (existing); serde + serde_json (existing). New deps: `fontdue = "=0.9.3"` (exact), `palette = "~0.7"` (range), `usvg = "~0.47"` (range), `resvg = "~0.47"` (range, must match usvg minor), `base64 = "~0.22"` (range). Bundled asset: `Inter-Regular.ttf` (~310 KB).

**Spec reference:** `docs/superpowers/specs/2026-05-09-static-renderer-design.md` (commit `1a844c4`).

**Branch:** `feat/phase-7-static-renderer` (already checked out from `main`).

**Test target:** `cargo test -p ferrum-core` ≥ 218 (currently 178); `uv run pytest` ≥ 88 (currently 78).

---

## File map

### New files

| Path | Responsibility |
|---|---|
| `crates/ferrum-core/assets/fonts/Inter-Regular.ttf` | Bundled font asset (downloaded in Task 1) |
| `crates/ferrum-core/assets/fonts/Inter-OFL.txt` | SIL Open Font License 1.1 text |
| `NOTICE` | Repo-level third-party attribution (created if absent) |
| `crates/ferrum-core/src/render/mod.rs` | `render_svg`, `render_png`, `RenderOutput`, `RenderError`, `RenderWarning`, constants |
| `crates/ferrum-core/src/render/config.rs` | `RenderConfig` + `Default` |
| `crates/ferrum-core/src/render/color.rs` | `Color` type alias, `from_hex_str`, `with_opacity`, `fmt_svg`, `ColorParseError` |
| `crates/ferrum-core/src/render/palette.rs` | `OKABE_ITO`, `categorical_color` |
| `crates/ferrum-core/src/render/font.rs` | `INTER_REGULAR`, `FontdueMetrics` |
| `crates/ferrum-core/src/render/format.rs` | Tick label formatters (numeric, time, ordinal) |
| `crates/ferrum-core/src/render/svg.rs` | `SvgBuffer`, `FillStroke`, `Stroke`, `TextStyle`, `TextAnchor` |
| `crates/ferrum-core/src/render/embed_font.rs` | `inter_data_url()` — base64 `@font-face` block |
| `crates/ferrum-core/src/render/scale_resolve.rs` | `ScaleKind`, `ColorScale`, `ResolvedScales`, `resolve_scales` |
| `crates/ferrum-core/src/render/prepare.rs` | `PreparedInputs`, `prepare_render_inputs` |
| `crates/ferrum-core/src/render/draw.rs` | `DrawCtx`, `MarkStyle`, `resolve_mark_style`, `dispatch_mark` |
| `crates/ferrum-core/src/render/png.rs` | `svg_string_to_png_bytes` |
| `crates/ferrum-core/src/render/binding.rs` | PyO3 bindings: `render_svg`, `render_png` |
| `crates/ferrum-core/src/render/marks/mod.rs` | Module decls + re-exports |
| `crates/ferrum-core/src/render/marks/point.rs` | `mark_point` draw |
| `crates/ferrum-core/src/render/marks/line.rs` | `mark_line` draw |
| `crates/ferrum-core/src/render/marks/area.rs` | `mark_area` draw |
| `crates/ferrum-core/src/render/marks/bar.rs` | `mark_bar` draw |
| `crates/ferrum-core/src/render/marks/rect.rs` | `mark_rect` draw |
| `crates/ferrum-core/src/render/marks/rule.rs` | `mark_rule` draw |
| `crates/ferrum-core/src/render/marks/text.rs` | `mark_text` draw |
| `crates/ferrum-core/src/render/marks/tick.rs` | `mark_tick` draw |
| `crates/ferrum-core/src/render/marks/axis.rs` | Internal: axis line/ticks/labels/title from `AxisLayout` |
| `crates/ferrum-core/src/render/marks/legend.rs` | Internal: legend swatches/labels from `LegendLayout` |
| `crates/ferrum-core/src/render/marks/strip_title.rs` | Internal: per-panel strip title |
| `crates/ferrum-core/tests/golden/scatter_minimal.svg` | Golden — point, no color |
| `crates/ferrum-core/tests/golden/scatter_color.svg` | Golden — point + color encoding + legend |
| `crates/ferrum-core/tests/golden/bar_grouped.svg` | Golden — bar |
| `crates/ferrum-core/tests/golden/line_simple.svg` | Golden — line |
| `crates/ferrum-core/tests/golden/area_filled.svg` | Golden — area |
| `crates/ferrum-core/tests/golden/faceted_scatter.svg` | Golden — facet + strip titles + legend |
| `crates/ferrum-core/tests/golden/scatter_minimal.png.sha256` | Golden — PNG hash |
| `tests/test_render.py` | Pytest binding tests |

### Modified files

| Path | Change |
|---|---|
| `Cargo.toml` (workspace) | Add `fontdue`, `palette`, `usvg`, `resvg`, `base64` to `[workspace.dependencies]` |
| `crates/ferrum-core/Cargo.toml` | Add same deps to `[dependencies]` |
| `crates/ferrum-core/src/lib.rs` | `mod render;` + register `render_svg`/`render_png` pyfunctions |
| `crates/ferrum-core/src/spec/encoding.rs` | Add `color: Option<EncodingSpec>` to `Encoding` |
| `crates/ferrum-core/src/spec/chart.rs` | Add `color` kwarg to Python `ChartSpec.__init__` |
| `crates/ferrum-core/src/layout/panel.rs` | Add `strip_title: Option<StripTitleLayout>`; new `StripTitleLayout` + `TextAnchor` types |
| `crates/ferrum-core/src/layout/mod.rs` | Add 18 new fields to `ThemeInputs` (additive); populate `strip_title` when faceted; re-export `StripTitleLayout`, `TextAnchor` |
| `src/ferrum/__init__.py` | Re-export `render_svg`, `render_png` |
| `src/ferrum/_core.pyi` | Add `render_svg`/`render_png` signatures + `color` kwarg on `ChartSpec` |
| `tests/test_chart_spec.py` (or equivalent) | Add color round-trip tests |
| `docs/superpowers/ferrum-phases.md` | Phase 7 status `pending` → `done`; link spec + plan |
| `ferrum-spec.md` | §3.16 dated note: Phase 7 honored fields subset |

### Constants table (from spec §6.1, lives in `render/mod.rs`)

| Constant | Value | Purpose |
|---|---|---|
| `FLOAT_PRECISION` | `3` | `{:.N}` precision for SVG attribute floats |
| `DEFAULT_GRID_ENABLED` | `true` | Grid lines drawn behind marks unless theme disables |
| `CLIP_ID_PREFIX` | `"ferrum-clip-"` | `<clipPath>` id namespace |
| `INTER_FONT_FAMILY` | `"Inter"` | `font-family` attribute value |

---

## Task list

### Task 1: Workspace deps + render/ skeleton + Inter font asset

**Files:**
- Modify: `Cargo.toml` (workspace deps)
- Modify: `crates/ferrum-core/Cargo.toml` (crate deps)
- Create: `crates/ferrum-core/assets/fonts/Inter-Regular.ttf` (binary)
- Create: `crates/ferrum-core/assets/fonts/Inter-OFL.txt`
- Create: `NOTICE`
- Create: `crates/ferrum-core/src/render/mod.rs`
- Create: `crates/ferrum-core/src/render/{config,color,palette,font,format,svg,embed_font,scale_resolve,prepare,draw,png,binding}.rs` (stub each)
- Create: `crates/ferrum-core/src/render/marks/mod.rs` (stub)
- Create: `crates/ferrum-core/src/render/marks/{point,line,area,bar,rect,rule,text,tick,axis,legend,strip_title}.rs` (stub each)
- Modify: `crates/ferrum-core/src/lib.rs`

- [ ] **Step 1: Verify current crate versions on crates.io**

```bash
source ~/.cargo/env
cargo search fontdue --limit 1
cargo search palette --limit 1
cargo search usvg --limit 1
cargo search resvg --limit 1
cargo search base64 --limit 1
```

Expected (as of plan-write time): `fontdue 0.9.3`, `palette 0.7.6`, `usvg 0.47.0`, `resvg 0.47.0`, `base64 0.22.1`. If any have advanced, update the version pins below to the new latest stable, and note the bump in the commit message.

- [ ] **Step 2: Add deps to workspace `Cargo.toml`**

Edit `Cargo.toml`. Inside `[workspace.dependencies]`, after the `rand_chacha` line, append:

```toml
# Phase 7 (static-renderer) — exact pin per spec §8.
# fontdue version bumps change glyph advances; refresh goldens via FERRUM_UPDATE_GOLDENS=1.
fontdue = "=0.9.3"
# Color type alias + future colorspace conversions. Range pin per spec §11 row 6.
palette = "~0.7"
# SVG → typed tree → tiny_skia rasterization. usvg/resvg minor versions must match.
usvg    = "~0.47"
resvg   = "~0.47"
# base64 @font-face data URL emission for Phase 7 SVG embed.
base64  = "~0.22"
```

Then **edit the existing `arrow` line** to add the `"compute"` feature (used by `arrow::compute::filter_record_batch` in Task 20 and `arrow::compute::concat_batches` in Task 22):

```toml
arrow      = { version = "58", default-features = false, features = ["ipc", "compute"] }
```

- [ ] **Step 3: Add deps to `crates/ferrum-core/Cargo.toml`**

Edit `crates/ferrum-core/Cargo.toml`. Inside `[dependencies]`, after the `rand_chacha` line, append:

```toml
fontdue     = { workspace = true }
palette     = { workspace = true }
usvg        = { workspace = true }
resvg       = { workspace = true }
base64      = { workspace = true }
```

- [ ] **Step 4: Download Inter Regular font**

```bash
mkdir -p crates/ferrum-core/assets/fonts
curl -L -o crates/ferrum-core/assets/fonts/Inter-Regular.ttf \
  https://github.com/rsms/inter/raw/v4.0/docs/font-files/Inter-Regular.otf
# Note: rsms/inter ships .otf; fontdue accepts .otf despite the .ttf extension.
# If you prefer the .ttf format, use the inter-ui distribution at:
#   https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf
# (variable font; fontdue picks the regular axis automatically).
ls -la crates/ferrum-core/assets/fonts/Inter-Regular.ttf
```

Expected: file present, ~300–400 KB. If the URL 404s, search for "Inter Regular OFL download" — any official OFL build is acceptable.

- [ ] **Step 5: Create OFL license text**

Create `crates/ferrum-core/assets/fonts/Inter-OFL.txt` with the standard SIL OFL 1.1 text (downloadable from `https://openfontlicense.org/` or copied from the Inter repo). Heading line must contain "Copyright 2016 The Inter Project Authors" and the file must include the full OFL 1.1 body verbatim.

- [ ] **Step 6: Create repo-level NOTICE**

Create `NOTICE` at repo root (only if absent — check first with `ls NOTICE`):

```
Ferrum — third-party attributions

Inter font (Regular)
  Copyright 2016 The Inter Project Authors (https://github.com/rsms/inter)
  Licensed under SIL Open Font License 1.1.
  License text: crates/ferrum-core/assets/fonts/Inter-OFL.txt
```

If `NOTICE` exists, append the Inter block instead.

- [ ] **Step 7: Create render/ module skeleton**

```bash
mkdir -p crates/ferrum-core/src/render/marks
```

Create `crates/ferrum-core/src/render/mod.rs`:

```rust
//! Phase 7 — static renderer. Pure functions: ChartSpec + RecordBatch + ThemeInputs +
//! Viewport -> deterministic SVG/PNG. See docs/superpowers/specs/2026-05-09-static-renderer-design.md.

pub(crate) mod config;
pub(crate) mod color;
pub(crate) mod palette;
pub(crate) mod font;
pub(crate) mod format;
pub(crate) mod svg;
pub(crate) mod embed_font;
pub(crate) mod scale_resolve;
pub(crate) mod prepare;
pub(crate) mod draw;
pub(crate) mod png;
pub(crate) mod binding;
pub(crate) mod marks;

// Constants (spec §6.1).
pub const FLOAT_PRECISION: usize = 3;
pub const DEFAULT_GRID_ENABLED: bool = true;
pub const CLIP_ID_PREFIX: &str = "ferrum-clip-";
pub const INTER_FONT_FAMILY: &str = "Inter";
```

For each of `config.rs`, `color.rs`, `palette.rs`, `font.rs`, `format.rs`, `svg.rs`, `embed_font.rs`, `scale_resolve.rs`, `prepare.rs`, `draw.rs`, `png.rs`, `binding.rs`, write:

```rust
//! Placeholder — implementation lands in subsequent tasks.
```

Create `crates/ferrum-core/src/render/marks/mod.rs`:

```rust
//! Per-mark draw functions. Each module exports a free `draw(ctx, out)` fn dispatched
//! from `render::draw::dispatch_mark`. Internal-only helpers (`axis`, `legend`,
//! `strip_title`) are not surfaced as primitive marks.

pub(crate) mod point;
pub(crate) mod line;
pub(crate) mod area;
pub(crate) mod bar;
pub(crate) mod rect;
pub(crate) mod rule;
pub(crate) mod text;
pub(crate) mod tick;
pub(crate) mod axis;
pub(crate) mod legend;
pub(crate) mod strip_title;
```

For each of the 11 mark files, write:

```rust
//! Placeholder — implementation lands in subsequent tasks.
```

- [ ] **Step 8: Register `render` module in `lib.rs`**

Edit `crates/ferrum-core/src/lib.rs`. After the `pub(crate) mod layout;` line, add:

```rust
pub(crate) mod render;
```

(No pyfunction registration yet — those land in Task 22.)

- [ ] **Step 9: Verify build**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

Expected: build succeeds with new deps fetched. No new functionality, no new pyclass.

- [ ] **Step 10: Verify all existing tests still pass**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
uv run pytest
```

Expected: 178 cargo tests pass, 78 pytest tests pass. No regressions.

- [ ] **Step 11: Commit**

```bash
git add Cargo.toml crates/ferrum-core/Cargo.toml \
        crates/ferrum-core/assets crates/ferrum-core/src/render \
        crates/ferrum-core/src/lib.rs NOTICE
git commit -m "feat(render): scaffold render/ module skeleton + bundle Inter font

Adds fontdue, palette, usvg, resvg, base64 to workspace deps.
Bundles Inter-Regular under SIL OFL 1.1 with NOTICE attribution.
Empty stubs for color, palette, font, format, svg, embed_font,
scale_resolve, prepare, draw, png, binding, and marks/{point,line,
area,bar,rect,rule,text,tick,axis,legend,strip_title}. No public
surface yet."
```

---

### Task 2: Encoding gains `color: Option<EncodingSpec>`

**Files:**
- Modify: `crates/ferrum-core/src/spec/encoding.rs`
- Modify: `crates/ferrum-core/src/spec/chart.rs`
- Modify: `src/ferrum/_core.pyi`

- [ ] **Step 1: Add round-trip tests**

In `crates/ferrum-core/src/spec/encoding.rs`, locate the `#[cfg(test)] mod tests { ... }` block. Add these tests at the end:

```rust
#[test]
fn test_encoding_round_trip_with_color() {
    let e = Encoding {
        x: Some(EncodingSpec { field: "price".into(), type_: None }),
        y: Some(EncodingSpec { field: "weight".into(), type_: None }),
        color: Some(EncodingSpec { field: "species".into(), type_: Some(DataType::Nominal) }),
    };
    let json = serde_json::to_string(&e).unwrap();
    assert_eq!(
        json,
        r#"{"x":{"field":"price"},"y":{"field":"weight"},"color":{"field":"species","type":"nominal"}}"#,
    );
    let parsed: Encoding = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, e);
}

#[test]
fn test_encoding_omits_color_when_none() {
    let e = Encoding {
        x: Some(EncodingSpec { field: "a".into(), type_: None }),
        y: Some(EncodingSpec { field: "b".into(), type_: None }),
        color: None,
    };
    let json = serde_json::to_string(&e).unwrap();
    assert_eq!(json, r#"{"x":{"field":"a"},"y":{"field":"b"}}"#);
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core encoding
```

Expected: compile error — `color` is not a field of `Encoding`.

- [ ] **Step 3: Add `color` field to `Encoding`**

Edit the `Encoding` struct in `crates/ferrum-core/src/spec/encoding.rs`:

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Encoding {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x: Option<EncodingSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub y: Option<EncodingSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub color: Option<EncodingSpec>,
}
```

- [ ] **Step 4: Update `ChartSpec` Python `__init__` to accept `color` kwarg**

In `crates/ferrum-core/src/spec/chart.rs`, locate the `#[pymethods] impl ChartSpec` block, find the `#[new]` constructor signature, and:

1. Add `color = None` to the `#[pyo3(signature = ...)]` argument list (alongside `x`, `y`).
2. Add a `color: Option<&Bound<'_, PyAny>>` parameter to the function (mirroring the existing `x` / `y` types).
3. Convert the kwarg the same way `x` and `y` are converted (string → `EncodingSpec` with `field=name`; `EncodingSpec` passed through).
4. Set `encoding.color = Some(...)` when present.

Read the existing `x`/`y` conversion code first to stay consistent — copy that pattern verbatim for `color`.

Also locate any `__repr__` / `to_json` / `from_json` test fixtures and update encoding constructions to set `color: None` where they were previously omitted (default-constructed). Many tests use `Encoding::default()` which is already fine.

Search for fixtures that need updating:

```bash
grep -rn "Encoding {" crates/ferrum-core/src/ tests/
```

Update each constructor that uses field-by-field syntax to add `color: None`.

- [ ] **Step 5: Update `_core.pyi`**

Edit `src/ferrum/_core.pyi`. Replace the `ChartSpec` class block:

```python
class ChartSpec:
    mark: str
    x: Optional[EncodingSpec]
    y: Optional[EncodingSpec]
    color: Optional[EncodingSpec]
    data: str
    transforms: List[object]

    def __init__(
        self,
        *,
        mark: MarkStr,
        x: Union[str, EncodingSpec, None] = None,
        y: Union[str, EncodingSpec, None] = None,
        color: Union[str, EncodingSpec, None] = None,
        data: Optional[str] = None,
        transforms: Optional[List[object]] = None,
    ) -> None: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, s: str) -> "ChartSpec": ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
```

- [ ] **Step 6: Add a Python round-trip test**

In `tests/test_chart_spec.py` (or create if absent), add:

```python
def test_chart_spec_color_round_trip():
    from ferrum import ChartSpec, EncodingSpec
    s = ChartSpec(mark="point", x="a", y="b", color="species")
    j = s.to_json()
    assert '"color"' in j
    parsed = ChartSpec.from_json(j)
    assert parsed == s
```

- [ ] **Step 7: Build and run all tests**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
uv run pytest
```

Expected: cargo +2 (≥ 180); pytest +1 (≥ 79). All previously passing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add crates/ferrum-core/src/spec/ src/ferrum/_core.pyi tests/test_chart_spec.py
git commit -m "feat(spec): Encoding gains color: Option<EncodingSpec>

Additive ChartSpec extension per Phase 7 spec §3.3 / locked decision §11 row 14.
Existing JSON outputs unchanged (omitted field round-trips as None).
Python ChartSpec(__init__) accepts color kwarg with str|EncodingSpec sugar."
```

---

### Task 3: `PanelLayout.strip_title` + `TextAnchor` + `StripTitleLayout`

**Files:**
- Modify: `crates/ferrum-core/src/layout/panel.rs`
- Modify: `crates/ferrum-core/src/layout/mod.rs` (re-exports)

- [ ] **Step 1: Write failing tests**

In `crates/ferrum-core/src/layout/panel.rs`, locate the existing test module (or add `#[cfg(test)] mod tests {}` if absent) and add:

```rust
#[cfg(test)]
mod strip_title_tests {
    use super::*;

    #[test]
    fn strip_title_layout_round_trip() {
        let s = StripTitleLayout {
            text: "setosa".into(),
            anchor: (10.0, 20.0),
            align: TextAnchor::Middle,
            font_size: 13.0,
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: StripTitleLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
    }

    #[test]
    fn panel_layout_strip_title_omitted_when_none() {
        let p = PanelLayout {
            plot_area: crate::layout::Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(!json.contains("strip_title"), "expected omission, got: {json}");
    }

    #[test]
    fn panel_layout_with_strip_title_round_trip() {
        let p = PanelLayout {
            plot_area: crate::layout::Rect { x: 0.0, y: 0.0, w: 100.0, h: 80.0 },
            facet_key: Some(FacetKey { field: "species".into(), value: "setosa".into() }),
            row: 0,
            col: 1,
            strip_title: Some(StripTitleLayout {
                text: "setosa".into(),
                anchor: (50.0, 5.0),
                align: TextAnchor::Middle,
                font_size: 13.0,
            }),
        };
        let json = serde_json::to_string(&p).unwrap();
        let parsed: PanelLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core panel::strip_title_tests
```

Expected: compile errors — `StripTitleLayout`, `TextAnchor`, and `strip_title` field don't exist.

- [ ] **Step 3: Implement `TextAnchor` and `StripTitleLayout`; extend `PanelLayout`**

Read the current `crates/ferrum-core/src/layout/panel.rs` to see the existing `PanelLayout` and `FacetKey` definitions. Then rewrite the file to add the new types and field. The full file should look approximately:

```rust
//! Per-panel layout (single chart or one cell of a facet grid).

use serde::{Deserialize, Serialize};

use super::geometry::Rect;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PanelLayout {
    pub plot_area: Rect,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facet_key: Option<FacetKey>,
    pub row: u32,
    pub col: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strip_title: Option<StripTitleLayout>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FacetKey {
    pub field: String,
    pub value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextAnchor {
    Start,
    Middle,
    End,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StripTitleLayout {
    pub text: String,
    pub anchor: (f64, f64),
    pub align: TextAnchor,
    pub font_size: f64,
}

#[cfg(test)]
mod strip_title_tests {
    // (tests from Step 1)
}
```

Preserve any existing test module verbatim — only add the new fields and types and the new test block.

- [ ] **Step 4: Re-export `StripTitleLayout` and `TextAnchor` from `layout/mod.rs`**

Edit `crates/ferrum-core/src/layout/mod.rs`. Locate the existing `pub use self::panel::{...}` line and replace with:

```rust
pub use self::panel::{FacetKey, PanelLayout, StripTitleLayout, TextAnchor};
```

- [ ] **Step 5: Update existing PanelLayout constructions to set `strip_title: None`**

```bash
grep -rn "PanelLayout {" crates/ferrum-core/src/
```

For each match, add `strip_title: None,` to the struct literal. Watch out for the existing `compute_layout` orchestration in `layout/mod.rs` step 7 — it constructs panels in a loop. The fix is one line per construction site.

- [ ] **Step 6: Build and run all tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
unset CONDA_PREFIX && uv run --no-sync maturin develop
uv run pytest
```

Expected: cargo +3 (≥ 183); all Phase 6 tests still pass; pytest unchanged. The Phase 6 `compute_layout` Python binding test that returns a dict will now have `strip_title: null` in the JSON for all panels — but since the Python test uses `dict.get("panels")` and doesn't assert on the key set, it should still pass.

- [ ] **Step 7: Commit**

```bash
git add crates/ferrum-core/src/layout/panel.rs crates/ferrum-core/src/layout/mod.rs
git commit -m "feat(layout): PanelLayout gains optional strip_title + TextAnchor

Additive extension per Phase 7 spec §3.3 / locked decision §11 row 9.
StripTitleLayout carries text, anchor, align, font_size; serialized only
when present (skip_serializing_if). TextAnchor is the shared anchor enum
used by both strip titles and SvgBuffer text emission (Task 9).
compute_layout still emits None for all panels — Task 12 wires the
faceted-panel population."
```

---

### Task 4: `ThemeInputs` grows 18 new render-only fields

**Files:**
- Modify: `crates/ferrum-core/src/layout/mod.rs`

- [ ] **Step 1: Write failing test**

Add to the `#[cfg(test)] mod tests { ... }` block at the bottom of `crates/ferrum-core/src/layout/mod.rs`:

```rust
#[test]
fn theme_inputs_default_includes_render_fields() {
    let t = ThemeInputs::default();
    // Phase 6 fields preserved.
    assert_eq!(t.padding, DEFAULT_PADDING);
    assert_eq!(t.label_font_size, DEFAULT_LABEL_FONT_SIZE);
    // Phase 7 additions.
    assert_eq!(t.point_size, 30.0);
    assert_eq!(t.line_stroke_width, 1.5);
    assert_eq!(t.bar_corner_radius, 0.0);
    assert_eq!(t.area_opacity, 0.4);
    assert_eq!(t.default_opacity, 1.0);
    assert_eq!(t.axis_line_width, 1.0);
    assert_eq!(t.tick_size, 4.0);
    assert_eq!(t.grid_width, 1.0);
    assert_eq!(t.grid, true);
    assert_eq!(t.strip_text_size, 13.0);
    assert_eq!(t.strip_padding, 4.0);
    // Color fields are exercised in Task 6 once palette::Srgba<u8> exists.
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::tests::theme_inputs_default_includes_render_fields
```

Expected: compile error — fields don't exist.

- [ ] **Step 3: Add fields to `ThemeInputs`**

In `crates/ferrum-core/src/layout/mod.rs`, locate the existing `ThemeInputs` struct (and its `Default` impl). Replace with:

```rust
/// Theme fields actually read by Phase 6 + Phase 7. Kept decoupled from a full
/// Theme type — Phase 8 grammar will translate ferrum.Theme into this shape.
///
/// Color fields use `crate::render::color::Color` (palette::Srgba<u8>); see Task 6.
/// Until Task 6 lands, the color fields below are placeholders typed as
/// `palette::Srgba<u8>`. The crate already has `palette` as a workspace dep (Task 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThemeInputs {
    // Phase 6 layout fields.
    pub padding: f64,
    pub column_padding: f64,
    pub row_padding: f64,
    pub axis_title_padding: f64,
    pub label_font_size: f64,
    pub title_font_size: f64,
    pub legend_orient: LegendOrient,

    // Phase 7 render fields — sizes/widths/opacities (no color yet — Task 6 widens).
    pub point_size: f64,
    pub line_stroke_width: f64,
    pub bar_corner_radius: f64,
    pub area_opacity: f64,
    pub default_opacity: f64,
    pub axis_line_width: f64,
    pub tick_size: f64,
    pub grid_width: f64,
    pub grid: bool,
    pub strip_text_size: f64,
    pub strip_padding: f64,

    // Phase 7 render fields — colors. Populated via from_hex_str in Task 6
    // once `crate::render::color::Color` is wired. For now, stored as
    // palette::Srgba<u8> directly with literal byte values.
    pub mark_color: palette::Srgba<u8>,
    pub axis_line_color: palette::Srgba<u8>,
    pub tick_color: palette::Srgba<u8>,
    pub grid_color: palette::Srgba<u8>,
    pub font_color: palette::Srgba<u8>,
    pub background_color: palette::Srgba<u8>,
    pub strip_background_color: palette::Srgba<u8>,
}

impl Default for ThemeInputs {
    fn default() -> Self {
        // OKABE_ITO[0] = #E69F00 = (230, 159, 0).
        let okabe_orange = palette::Srgba::new(0xE6, 0x9F, 0x00, 0xFF);
        let neutral_888  = palette::Srgba::new(0x88, 0x88, 0x88, 0xFF);
        let neutral_eee  = palette::Srgba::new(0xEE, 0xEE, 0xEE, 0xFF);
        let text_222     = palette::Srgba::new(0x22, 0x22, 0x22, 0xFF);
        let bg_white     = palette::Srgba::new(0xFF, 0xFF, 0xFF, 0xFF);
        let strip_bg     = palette::Srgba::new(0xF0, 0xF0, 0xF0, 0xFF);

        Self {
            // Phase 6.
            padding: DEFAULT_PADDING,
            column_padding: DEFAULT_PADDING,
            row_padding: DEFAULT_PADDING,
            axis_title_padding: DEFAULT_AXIS_TITLE_PADDING,
            label_font_size: DEFAULT_LABEL_FONT_SIZE,
            title_font_size: DEFAULT_TITLE_FONT_SIZE,
            legend_orient: LegendOrient::Right,

            // Phase 7 sizes / widths / opacities.
            point_size: 30.0,
            line_stroke_width: 1.5,
            bar_corner_radius: 0.0,
            area_opacity: 0.4,
            default_opacity: 1.0,
            axis_line_width: 1.0,
            tick_size: 4.0,
            grid_width: 1.0,
            grid: true,
            strip_text_size: 13.0,
            strip_padding: 4.0,

            // Phase 7 colors.
            mark_color: okabe_orange,
            axis_line_color: neutral_888,
            tick_color: neutral_888,
            grid_color: neutral_eee,
            font_color: text_222,
            background_color: bg_white,
            strip_background_color: strip_bg,
        }
    }
}
```

- [ ] **Step 4: Run all tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
```

Expected: cargo +1 (≥ 184). All Phase 6 tests still pass; the previous `Default::default()` ThemeInputs construction still works because all new fields have defaults.

- [ ] **Step 5: Commit**

```bash
git add crates/ferrum-core/src/layout/mod.rs
git commit -m "feat(layout): ThemeInputs grows 18 render-only fields

Additive per Phase 7 spec §5.2 / locked decision §11 row 12.
Sizes (point_size, line_stroke_width, ...), widths, opacities, and
colors (palette::Srgba<u8>). Phase 6 layout arithmetic untouched
because new fields don't affect reservation logic — strip_text_size
and strip_padding are read only when spec.facet.is_some() (Task 12)."
```

---

### Task 5: `RenderConfig` + `RenderError` + `RenderWarning`

**Files:**
- Modify: `crates/ferrum-core/src/render/config.rs`
- Modify: `crates/ferrum-core/src/render/mod.rs`

- [ ] **Step 1: Write failing tests**

Append to `crates/ferrum-core/src/render/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_config_default_values() {
        let c = config::RenderConfig::default();
        assert_eq!(c.scale, 2.0);
        assert!(c.embed_fonts);
        assert!(c.background.is_none());
        assert!(c.width.is_none());
        assert!(c.height.is_none());
    }

    #[test]
    fn render_warning_round_trip_each_variant() {
        use crate::layout::LayoutWarning;
        for w in [
            RenderWarning::Layout(LayoutWarning::PanelCollapsed { panel_index: 0 }),
            RenderWarning::OutOfDomainRows { mark: "point".into(), count: 3 },
            RenderWarning::ColorPaletteOverflowed { categories: 12 },
            RenderWarning::EmptyPanel { panel_index: 1 },
        ] {
            let json = serde_json::to_string(&w).unwrap();
            let parsed: RenderWarning = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, w);
        }
    }

    #[test]
    fn render_error_display_messages_are_meaningful() {
        let err = RenderError::InvalidViewport { width: 0.0, height: 100.0 };
        let msg = format!("{err}");
        assert!(msg.contains("invalid viewport"), "msg: {msg}");
        assert!(msg.contains("0"), "msg: {msg}");

        let err = RenderError::UnknownColumn { name: "missing".into() };
        let msg = format!("{err}");
        assert!(msg.contains("unknown column"), "msg: {msg}");
        assert!(msg.contains("missing"), "msg: {msg}");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::tests
```

Expected: compile error — types don't exist.

- [ ] **Step 3: Implement `RenderConfig`**

Replace `crates/ferrum-core/src/render/config.rs` contents:

```rust
//! RenderConfig — Phase 7 honored fields only. See spec §5.3.

use palette::Srgba;

/// Render-time configuration. Phase 7 honors a small subset of `ferrum-spec.md §3.16`.
/// Future phases will widen this struct as their corresponding features ship
/// (engine, backend, raster_*, tile_parallel, font_path).
#[derive(Debug, Clone, PartialEq)]
pub struct RenderConfig {
    /// PNG density multiplier. Default 2.0 (retina).
    pub scale: f64,
    /// Whether to inline @font-face base64 in SVG output.
    /// Phase 7 always treats this as true (locked decision §11 row 15);
    /// the field is preserved for future phases.
    pub embed_fonts: bool,
    /// Override chart background color for export.
    pub background: Option<Srgba<u8>>,
    /// Override viewport.width if Some.
    pub width: Option<f64>,
    /// Override viewport.height if Some.
    pub height: Option<f64>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self { scale: 2.0, embed_fonts: true, background: None, width: None, height: None }
    }
}
```

- [ ] **Step 4: Implement `RenderError` and `RenderWarning`**

In `crates/ferrum-core/src/render/mod.rs`, after the constants block, add:

```rust
use serde::{Deserialize, Serialize};

use crate::layout::LayoutWarning;

#[derive(Debug, Clone, PartialEq)]
pub enum RenderError {
    InvalidViewport { width: f64, height: f64 },
    EmptyBatch,
    UnknownColumn { name: String },
    InvalidColor(String),
    EncodingTypeMismatch { channel: &'static str, expected: &'static str, got: String },
    TransformFailed(String),
    ScaleResolutionFailed(String),
    LayoutFailed(String),
    ResvgFailed(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidViewport { width, height } =>
                write!(f, "invalid viewport: width={width}, height={height} (both must be > 0)"),
            Self::EmptyBatch =>
                write!(f, "input batch is empty (num_rows == 0)"),
            Self::UnknownColumn { name } =>
                write!(f, "unknown column '{name}' referenced by an encoding"),
            Self::InvalidColor(s) =>
                write!(f, "invalid color string: '{s}' (expected #rrggbb or #rrggbbaa)"),
            Self::EncodingTypeMismatch { channel, expected, got } =>
                write!(f, "encoding '{channel}' expected {expected}, got {got}"),
            Self::TransformFailed(s) =>
                write!(f, "transform failed: {s}"),
            Self::ScaleResolutionFailed(s) =>
                write!(f, "scale resolution failed: {s}"),
            Self::LayoutFailed(s) =>
                write!(f, "layout failed: {s}"),
            Self::ResvgFailed(s) =>
                write!(f, "PNG rasterization failed: {s}"),
        }
    }
}

impl std::error::Error for RenderError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RenderWarning {
    Layout(LayoutWarning),
    OutOfDomainRows { mark: String, count: u64 },
    ColorPaletteOverflowed { categories: u32 },
    EmptyPanel { panel_index: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderOutput<T> {
    pub bytes: T,
    pub layout: crate::layout::LayoutResult,
    pub warnings: Vec<RenderWarning>,
}
```

- [ ] **Step 5: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::tests
```

Expected: 3 new tests pass. Cargo total ≥ 187.

- [ ] **Step 6: Commit**

```bash
git add crates/ferrum-core/src/render/config.rs crates/ferrum-core/src/render/mod.rs
git commit -m "feat(render): RenderConfig + RenderError + RenderWarning + RenderOutput

Honors spec §5.3 field subset (scale, embed_fonts, background, width, height).
Errors mirror Phase 5/6 hybrid: structural variants for PyValueError mapping;
RenderWarning collected into RenderOutput.warnings (wraps LayoutWarning)."
```

---

### Task 6: `color.rs` + `palette.rs`

**Files:**
- Modify: `crates/ferrum-core/src/render/color.rs`
- Modify: `crates/ferrum-core/src/render/palette.rs`

- [ ] **Step 1: Write failing tests**

Replace `crates/ferrum-core/src/render/color.rs` with:

```rust
//! Color = palette::Srgba<u8>. SVG-formatted output, hex parsing, opacity.

use palette::Srgba;

pub type Color = Srgba<u8>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorParseError(pub String);

impl std::fmt::Display for ColorParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid color string: '{}' (expected #rrggbb or #rrggbbaa)", self.0)
    }
}

impl std::error::Error for ColorParseError {}

pub fn from_rgb(r: u8, g: u8, b: u8) -> Color {
    Srgba::new(r, g, b, 0xFF)
}

pub fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Color {
    Srgba::new(r, g, b, a)
}

pub fn from_hex_str(s: &str) -> Result<Color, ColorParseError> {
    let s = s.trim();
    if !s.starts_with('#') {
        return Err(ColorParseError(s.to_string()));
    }
    let hex = &s[1..];
    let parse = |i: usize| -> Result<u8, ColorParseError> {
        u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| ColorParseError(s.to_string()))
    };
    match hex.len() {
        6 => Ok(Srgba::new(parse(0)?, parse(2)?, parse(4)?, 0xFF)),
        8 => Ok(Srgba::new(parse(0)?, parse(2)?, parse(4)?, parse(6)?)),
        _ => Err(ColorParseError(s.to_string())),
    }
}

pub fn with_opacity(c: Color, opacity_0_1: f64) -> Color {
    let a = (c.alpha as f64 * opacity_0_1.clamp(0.0, 1.0)).round() as u8;
    Srgba::new(c.red, c.green, c.blue, a)
}

pub fn fmt_svg(c: Color) -> String {
    if c.alpha == 0xFF {
        format!("#{:02x}{:02x}{:02x}", c.red, c.green, c.blue)
    } else {
        let a = (c.alpha as f64) / 255.0;
        format!("rgba({},{},{},{:.3})", c.red, c.green, c.blue, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_six_digit_hex() {
        let c = from_hex_str("#1f77b4").unwrap();
        assert_eq!(c.red, 0x1f);
        assert_eq!(c.green, 0x77);
        assert_eq!(c.blue, 0xb4);
        assert_eq!(c.alpha, 0xFF);
    }

    #[test]
    fn parse_eight_digit_hex() {
        let c = from_hex_str("#1f77b4cc").unwrap();
        assert_eq!(c.alpha, 0xCC);
    }

    #[test]
    fn parse_named_color_fails() {
        assert!(from_hex_str("red").is_err());
    }

    #[test]
    fn opacity_multiplies() {
        let c = with_opacity(from_rgb(0xFF, 0x00, 0x00), 0.5);
        assert_eq!(c.alpha, 128);
    }

    #[test]
    fn fmt_svg_opaque_uses_hex() {
        assert_eq!(fmt_svg(from_rgb(0x1f, 0x77, 0xb4)), "#1f77b4");
    }

    #[test]
    fn fmt_svg_translucent_uses_rgba() {
        let c = from_rgba(0x1f, 0x77, 0xb4, 0x80);
        assert_eq!(fmt_svg(c), "rgba(31,119,180,0.502)");
    }
}
```

Replace `crates/ferrum-core/src/render/palette.rs` with:

```rust
//! Hardcoded categorical palette (Okabe-Ito). One palette for Phase 7;
//! Phase 8+ may add a scheme registry.

use std::sync::LazyLock;

use super::color::{from_rgb, Color};

/// Okabe-Ito 8-color categorical palette. Lazy-initialized because palette's
/// `Srgba::new` is not const-fn and the internal struct layout (`Alpha<Rgb<...>, u8>`)
/// is not stable enough to literal-construct in a `const`. `LazyLock` (Rust 1.80+)
/// initializes on first access; cost is one-time.
pub static OKABE_ITO: LazyLock<[Color; 8]> = LazyLock::new(|| [
    from_rgb(0xE6, 0x9F, 0x00), // orange
    from_rgb(0x56, 0xB4, 0xE9), // sky blue
    from_rgb(0x00, 0x9E, 0x73), // bluish green
    from_rgb(0xF0, 0xE4, 0x42), // yellow
    from_rgb(0x00, 0x72, 0xB2), // blue
    from_rgb(0xD5, 0x5E, 0x00), // vermillion
    from_rgb(0xCC, 0x79, 0xA7), // reddish purple
    from_rgb(0x00, 0x00, 0x00), // black
]);

pub fn categorical_color(category_index: usize) -> Color {
    OKABE_ITO[category_index % OKABE_ITO.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_category_is_okabe_orange() {
        let c = categorical_color(0);
        assert_eq!(c.red, 0xE6);
        assert_eq!(c.green, 0x9F);
        assert_eq!(c.blue, 0x00);
    }

    #[test]
    fn overflow_wraps() {
        let c = categorical_color(8);
        assert_eq!(c, OKABE_ITO[0]);
    }
}
```

> **Implementer note on MSRV:** `std::sync::LazyLock` requires Rust 1.80+ (released July 2024). If the project's MSRV is older, swap for `once_cell::sync::Lazy` (add `once_cell = "1"` to workspace deps). Check `rust-toolchain.toml` (if present) or current `rustc --version` to confirm. If MSRV ≥ 1.80, no change needed.
>
> **Knock-on for `scale_resolve.rs` (Task 10):** the static now derefs via auto-deref — `palette: OKABE_ITO` won't compile if `OKABE_ITO` is `LazyLock<[Color; 8]>`. Use `palette: &*OKABE_ITO` or `palette: OKABE_ITO.as_slice()` and update `ColorScale::Categorical { palette: &'static [Color] }` accordingly (already a slice, just need a deref at construction). Tests in Task 10 already pass `OKABE_ITO` to assertions — change those to `*OKABE_ITO[0]` or use `categorical_color(0)`.

- [ ] **Step 2: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::color render::palette
```

Expected: 8 tests pass (6 color + 2 palette). Cargo total ≥ 195.

- [ ] **Step 3: Commit**

```bash
git add crates/ferrum-core/src/render/color.rs crates/ferrum-core/src/render/palette.rs
git commit -m "feat(render): Color type alias + Okabe-Ito palette

Color = palette::Srgba<u8> per locked decision §11 row 6.
from_hex_str, with_opacity, fmt_svg helpers; fmt_svg emits #rrggbb when
opaque, rgba(...) when translucent. OKABE_ITO is the Phase 7 default
categorical palette; categorical_color wraps on overflow (warning emitted
by caller in prepare.rs, Task 11)."
```

---

### Task 7: `font.rs` — `FontdueMetrics`

**Files:**
- Modify: `crates/ferrum-core/src/render/font.rs`

- [ ] **Step 1: Write failing tests**

Replace `crates/ferrum-core/src/render/font.rs` with:

```rust
//! Bundled Inter Regular + FontdueMetrics impl of TextMetrics.

use crate::layout::TextMetrics;

pub const INTER_REGULAR: &[u8] = include_bytes!("../../assets/fonts/Inter-Regular.ttf");

pub struct FontdueMetrics {
    font: fontdue::Font,
}

impl FontdueMetrics {
    pub fn new() -> Self {
        let font = fontdue::Font::from_bytes(INTER_REGULAR, fontdue::FontSettings::default())
            .expect("bundled Inter-Regular.ttf must parse");
        Self { font }
    }

    pub fn font(&self) -> &fontdue::Font {
        &self.font
    }
}

impl Default for FontdueMetrics {
    fn default() -> Self { Self::new() }
}

impl TextMetrics for FontdueMetrics {
    fn measure_width(&self, text: &str, font_size: f64) -> f64 {
        text.chars()
            .map(|c| self.font.metrics(c, font_size as f32).advance_width as f64)
            .sum()
    }

    fn line_height(&self, font_size: f64) -> f64 {
        let lm = self
            .font
            .horizontal_line_metrics(font_size as f32)
            .expect("Inter has horizontal line metrics");
        (lm.ascent - lm.descent + lm.line_gap) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_parses() {
        let _ = FontdueMetrics::new();
    }

    #[test]
    fn measure_width_is_positive_for_ascii() {
        let m = FontdueMetrics::new();
        let w = m.measure_width("100", 11.0);
        assert!(w > 0.0, "expected positive width, got {w}");
        // Sanity: "100" at 11pt should be roughly 18-25px in Inter.
        assert!(w < 50.0, "unexpectedly wide: {w}");
    }

    #[test]
    fn line_height_is_positive_and_proportional() {
        let m = FontdueMetrics::new();
        let lh_11 = m.line_height(11.0);
        let lh_22 = m.line_height(22.0);
        assert!(lh_11 > 0.0);
        // Doubling font size roughly doubles line height (within Inter's metrics scaling).
        assert!((lh_22 / lh_11 - 2.0).abs() < 0.1, "lh_11={lh_11}, lh_22={lh_22}");
    }
}
```

- [ ] **Step 2: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::font
```

Expected: 3 tests pass. If `font_parses` panics, the bundled font file is corrupt — re-download in Task 1, Step 4.

- [ ] **Step 3: Commit**

```bash
git add crates/ferrum-core/src/render/font.rs
git commit -m "feat(render): FontdueMetrics + bundled Inter Regular

Replaces Phase 6 HeuristicMetrics in production rendering. Parses Inter
once via include_bytes!; measure_width sums per-char advances;
line_height = ascent - descent + line_gap. Phase 6 binding still uses
HeuristicMetrics (locked decision §11 row 16)."
```

---

### Task 8: `format.rs` — tick label formatters

**Files:**
- Modify: `crates/ferrum-core/src/render/format.rs`

- [ ] **Step 1: Write failing tests**

Replace `crates/ferrum-core/src/render/format.rs` with:

```rust
//! Tick label formatters: numeric, time, ordinal. Hardcoded defaults — no
//! per-axis FormatSpec yet (deferred to Phase 8 per locked decision §11 row 8).

/// Format a numeric tick value:
/// - Integer-valued in normal range: drop decimal ("0", "5", "100").
/// - Decimal with ≤ 4 sig figs: drop trailing zeros ("1.5", "0.25").
/// - |x| >= 1e6 or (0 < |x| < 1e-3): scientific notation ("1.5e6", "1e-4").
pub fn format_numeric(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let abs = x.abs();
    if abs >= 1e6 || abs < 1e-3 {
        // Scientific: "1.5e6", "1e-4"
        let formatted = format!("{x:.3e}");
        // Trim trailing zeros in mantissa: "1.500e6" → "1.5e6"; "1.000e-4" → "1e-4"
        trim_scientific(&formatted)
    } else if x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        let s = format!("{x:.4}");
        trim_trailing_zeros(&s)
    }
}

fn trim_trailing_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_string()
}

fn trim_scientific(s: &str) -> String {
    // s is like "1.500e6" or "1.000e-4"
    let (mantissa, exp) = s.split_once('e').unwrap_or((s, ""));
    let mantissa = trim_trailing_zeros(mantissa);
    if exp.is_empty() { mantissa } else { format!("{mantissa}e{exp}") }
}

/// Time-tick formatter. Picks granularity from inter-tick spacing in milliseconds.
/// - >= 1 year: "2026"
/// - >= 1 month: "Mar 2026"
/// - >= 1 day: "2026-03-15"
/// - >= 1 hour: "15:00"
/// - else: "15:30:45"
pub fn format_time(epoch_ms: i64, spacing_ms: i64) -> String {
    use chrono::{DateTime, NaiveDateTime, Datelike, Timelike, Utc};
    // Phase 7 has no chrono dep — implement minimally without it.
    // Convert epoch_ms to (Y, M, D, h, m, s) using a small helper below.
    let (y, mo, d, h, mi, s) = epoch_ms_to_ymdhms(epoch_ms);
    const DAY: i64 = 86_400_000;
    const HOUR: i64 = 3_600_000;
    if spacing_ms >= 365 * DAY {
        format!("{y}")
    } else if spacing_ms >= 28 * DAY {
        format!("{} {y}", month_short(mo))
    } else if spacing_ms >= DAY {
        format!("{y:04}-{mo:02}-{d:02}")
    } else if spacing_ms >= HOUR {
        format!("{h:02}:{mi:02}")
    } else {
        format!("{h:02}:{mi:02}:{s:02}")
    }
}

fn month_short(m: u32) -> &'static str {
    const NAMES: [&str; 12] = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    NAMES[(m - 1) as usize % 12]
}

/// Convert epoch milliseconds to Gregorian (Y, M, D, h, m, s) in UTC.
/// Self-contained — no chrono dep. Civil-from-days algorithm by Howard Hinnant.
fn epoch_ms_to_ymdhms(epoch_ms: i64) -> (i64, u32, u32, u32, u32, u32) {
    let secs = epoch_ms.div_euclid(1000);
    let ms_part = epoch_ms.rem_euclid(1000);
    let _ = ms_part; // not used in any current format
    let days = secs.div_euclid(86_400);
    let time = secs.rem_euclid(86_400);
    let h = (time / 3600) as u32;
    let mi = ((time % 3600) / 60) as u32;
    let s = (time % 60) as u32;
    // Hinnant: civil_from_days
    let z = days + 719_468;
    let era = if z >= 0 { z / 146_097 } else { (z - 146_096) / 146_097 };
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d, h, mi, s)
}

/// Ordinal/threshold passthrough — caller already has a string.
pub fn format_ordinal(value: &str) -> String {
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_zero() { assert_eq!(format_numeric(0.0), "0"); }
    #[test]
    fn numeric_integer() { assert_eq!(format_numeric(5.0), "5"); }
    #[test]
    fn numeric_decimal_trims() { assert_eq!(format_numeric(1.5), "1.5"); }
    #[test]
    fn numeric_large_uses_scientific() { assert_eq!(format_numeric(1_500_000.0), "1.5e6"); }
    #[test]
    fn numeric_tiny_uses_scientific() {
        // 0.0001 == 1e-4 → "1e-4"
        let s = format_numeric(0.0001);
        assert!(s.starts_with("1") && s.contains("e-4"), "got: {s}");
    }
    #[test]
    fn numeric_near_one_trims_to_one() { assert_eq!(format_numeric(1.000001), "1"); }

    #[test]
    fn time_year_spacing() {
        // 2026-01-01T00:00:00Z = 1767225600000ms
        let s = format_time(1767225600000, 365 * 86_400_000);
        assert_eq!(s, "2026");
    }
    #[test]
    fn time_day_spacing() {
        let s = format_time(1767225600000, 86_400_000);
        assert_eq!(s, "2026-01-01");
    }
    #[test]
    fn time_hour_spacing() {
        // 2026-01-01T15:00:00Z = 1767279600000
        let s = format_time(1767279600000, 3_600_000);
        assert_eq!(s, "15:00");
    }

    #[test]
    fn ordinal_passthrough() {
        assert_eq!(format_ordinal("setosa"), "setosa");
    }
}
```

- [ ] **Step 2: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::format
```

Expected: 10 tests pass. Cargo total ≥ 208.

> **Implementer note:** the time tests above use frozen epoch_ms values for known UTC dates. If `epoch_ms_to_ymdhms` produces off-by-one results (timezone or epoch-anchor bug), debug `civil_from_days` against a third-party reference (e.g. `python -c "import datetime; print(datetime.datetime.fromtimestamp(1767225600, datetime.UTC))"`).

- [ ] **Step 3: Commit**

```bash
git add crates/ferrum-core/src/render/format.rs
git commit -m "feat(render): tick label formatters (numeric, time, ordinal)

Hardcoded defaults per spec §3 / locked decision §11 row 8. Numeric
switches to scientific outside [1e-3, 1e6). Time picks granularity from
spacing_ms (year / month / day / hour / second). Ordinal is passthrough.
Self-contained civil-from-days converter — no chrono dep."
```

---

### Task 9: `svg.rs` + `embed_font.rs` — deterministic SVG buffer

**Files:**
- Modify: `crates/ferrum-core/src/render/svg.rs`
- Modify: `crates/ferrum-core/src/render/embed_font.rs`

- [ ] **Step 1: Implement `svg.rs`**

Replace `crates/ferrum-core/src/render/svg.rs` with:

```rust
//! Hand-rolled SVG buffer with deterministic float formatting and fixed attribute order.
//! Spec §4.4. The element vocabulary covers only what Phase 7 marks need.

use crate::layout::Rect;
use crate::layout::TextAnchor;

use super::color::{fmt_svg, Color};
use super::FLOAT_PRECISION;

pub struct SvgBuffer {
    buf: String,
}

#[derive(Debug, Clone, Copy)]
pub struct FillStroke {
    pub fill: Option<Color>,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
}

#[derive(Debug, Clone)]
pub struct Stroke {
    pub stroke: Color,
    pub stroke_width: f64,
    pub stroke_dash: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Copy)]
pub struct TextStyle {
    pub fill: Color,
    pub font_size: f64,
    pub anchor: TextAnchor,
    pub angle: f64,
}

impl SvgBuffer {
    pub fn new(viewport: Rect, background: Option<Color>, embed_font: bool) -> Self {
        let mut buf = String::with_capacity(8192);
        buf.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
            fmt_f(viewport.w), fmt_f(viewport.h), fmt_f(viewport.w), fmt_f(viewport.h),
        ));
        if embed_font {
            buf.push_str(&super::embed_font::inter_data_url_block());
        }
        if let Some(bg) = background {
            buf.push_str(&format!(
                "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"{}\"/>",
                fmt_f(viewport.w), fmt_f(viewport.h), fmt_svg(bg),
            ));
        }
        Self { buf }
    }

    pub fn finish(mut self) -> String {
        self.buf.push_str("</svg>");
        self.buf
    }

    pub fn g_open(&mut self, transform: Option<&str>) {
        match transform {
            Some(t) => self.buf.push_str(&format!("<g transform=\"{}\">", escape_attr(t))),
            None => self.buf.push_str("<g>"),
        }
    }

    pub fn g_close(&mut self) {
        self.buf.push_str("</g>");
    }

    pub fn rect(&mut self, r: Rect, style: &FillStroke, corner_radius: Option<f64>) {
        self.buf.push_str("<rect");
        push_attr(&mut self.buf, "x", &fmt_f(r.x));
        push_attr(&mut self.buf, "y", &fmt_f(r.y));
        push_attr(&mut self.buf, "width", &fmt_f(r.w));
        push_attr(&mut self.buf, "height", &fmt_f(r.h));
        if let Some(rad) = corner_radius {
            if rad > 0.0 {
                push_attr(&mut self.buf, "rx", &fmt_f(rad));
                push_attr(&mut self.buf, "ry", &fmt_f(rad));
            }
        }
        push_fill_stroke(&mut self.buf, style);
        self.buf.push_str("/>");
    }

    pub fn circle(&mut self, cx: f64, cy: f64, radius: f64, style: &FillStroke) {
        self.buf.push_str("<circle");
        push_attr(&mut self.buf, "cx", &fmt_f(cx));
        push_attr(&mut self.buf, "cy", &fmt_f(cy));
        push_attr(&mut self.buf, "r", &fmt_f(radius));
        push_fill_stroke(&mut self.buf, style);
        self.buf.push_str("/>");
    }

    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, style: &Stroke) {
        self.buf.push_str("<line");
        push_attr(&mut self.buf, "x1", &fmt_f(x1));
        push_attr(&mut self.buf, "y1", &fmt_f(y1));
        push_attr(&mut self.buf, "x2", &fmt_f(x2));
        push_attr(&mut self.buf, "y2", &fmt_f(y2));
        push_stroke(&mut self.buf, style);
        self.buf.push_str("/>");
    }

    pub fn path(&mut self, d: &str, style: &FillStroke) {
        self.buf.push_str("<path");
        push_attr(&mut self.buf, "d", &escape_attr(d));
        push_fill_stroke(&mut self.buf, style);
        self.buf.push_str("/>");
    }

    pub fn polyline(&mut self, points: &[(f64, f64)], style: &Stroke) {
        let pts: Vec<String> = points.iter().map(|(x, y)| format!("{},{}", fmt_f(*x), fmt_f(*y))).collect();
        self.buf.push_str("<polyline");
        push_attr(&mut self.buf, "points", &pts.join(" "));
        push_attr(&mut self.buf, "fill", "none");
        push_stroke(&mut self.buf, style);
        self.buf.push_str("/>");
    }

    pub fn text(&mut self, x: f64, y: f64, content: &str, style: &TextStyle) {
        self.buf.push_str("<text");
        push_attr(&mut self.buf, "x", &fmt_f(x));
        push_attr(&mut self.buf, "y", &fmt_f(y));
        push_attr(&mut self.buf, "fill", &fmt_svg(style.fill));
        push_attr(&mut self.buf, "font-family", super::INTER_FONT_FAMILY);
        push_attr(&mut self.buf, "font-size", &fmt_f(style.font_size));
        push_attr(&mut self.buf, "text-anchor", anchor_str(style.anchor));
        if style.angle != 0.0 {
            let t = format!("rotate({} {} {})", fmt_f(style.angle), fmt_f(x), fmt_f(y));
            push_attr(&mut self.buf, "transform", &t);
        }
        self.buf.push('>');
        self.buf.push_str(&escape_text(content));
        self.buf.push_str("</text>");
    }

    pub fn clip_open(&mut self, id: &str, rect: Rect) {
        self.buf.push_str(&format!(
            "<defs><clipPath id=\"{}\"><rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/></clipPath></defs>",
            escape_attr(id), fmt_f(rect.x), fmt_f(rect.y), fmt_f(rect.w), fmt_f(rect.h),
        ));
    }

    pub fn use_clip_open(&mut self, clip_id: &str) {
        self.buf.push_str(&format!("<g clip-path=\"url(#{})\">", escape_attr(clip_id)));
    }

    pub fn use_clip_close(&mut self) {
        self.buf.push_str("</g>");
    }

    #[cfg(test)]
    pub(crate) fn as_str(&self) -> &str {
        &self.buf
    }
}

fn anchor_str(a: TextAnchor) -> &'static str {
    match a { TextAnchor::Start => "start", TextAnchor::Middle => "middle", TextAnchor::End => "end" }
}

fn push_attr(buf: &mut String, name: &str, value: &str) {
    buf.push(' ');
    buf.push_str(name);
    buf.push_str("=\"");
    buf.push_str(value);
    buf.push('"');
}

fn push_fill_stroke(buf: &mut String, s: &FillStroke) {
    match s.fill {
        Some(c) => push_attr(buf, "fill", &fmt_svg(c)),
        None => push_attr(buf, "fill", "none"),
    }
    if let Some(stroke) = s.stroke {
        push_attr(buf, "stroke", &fmt_svg(stroke));
        if s.stroke_width > 0.0 {
            push_attr(buf, "stroke-width", &fmt_f(s.stroke_width));
        }
    }
}

fn push_stroke(buf: &mut String, s: &Stroke) {
    push_attr(buf, "stroke", &fmt_svg(s.stroke));
    push_attr(buf, "stroke-width", &fmt_f(s.stroke_width));
    if let Some(dash) = &s.stroke_dash {
        let v: Vec<String> = dash.iter().map(|x| fmt_f(*x)).collect();
        push_attr(buf, "stroke-dasharray", &v.join(","));
    }
}

/// Format a float for SVG attribute output.
/// `{:.3}` then trim trailing zeros + dangling decimal point. Negative zero
/// folded to "0". NaN/Inf → "0" with a debug_assert! flag (callers must clamp).
pub fn fmt_f(x: f64) -> String {
    if !x.is_finite() {
        debug_assert!(false, "non-finite float in SVG output: {x}");
        return "0".to_string();
    }
    let x = if x == 0.0 { 0.0 } else { x }; // fold -0.0 → 0.0
    let s = format!("{x:.*}", FLOAT_PRECISION);
    let trimmed = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if trimmed.is_empty() || trimmed == "-" { "0".to_string() } else { trimmed }
}

pub fn escape_text(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

pub fn escape_attr(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::color::{from_rgb, from_rgba};

    fn vp() -> Rect { Rect { x: 0.0, y: 0.0, w: 100.0, h: 80.0 } }

    #[test]
    fn fmt_f_drops_trailing_zeros() {
        assert_eq!(fmt_f(1.5), "1.5");
        assert_eq!(fmt_f(1.0), "1");
        assert_eq!(fmt_f(1.500), "1.5");
        assert_eq!(fmt_f(0.0), "0");
        assert_eq!(fmt_f(-0.0), "0");
    }

    #[test]
    fn empty_buffer_minimal_svg() {
        let buf = SvgBuffer::new(vp(), None, false);
        let out = buf.finish();
        assert!(out.starts_with("<svg "));
        assert!(out.ends_with("</svg>"));
        assert!(out.contains("width=\"100\""));
    }

    #[test]
    fn embed_font_includes_font_face_block() {
        let buf = SvgBuffer::new(vp(), None, true);
        let out = buf.finish();
        assert!(out.contains("@font-face"));
        assert!(out.contains("font-family:\"Inter\""));
    }

    #[test]
    fn background_emitted_when_some() {
        let buf = SvgBuffer::new(vp(), Some(from_rgb(0xFF, 0, 0)), false);
        let out = buf.finish();
        assert!(out.contains("<rect x=\"0\" y=\"0\""));
        assert!(out.contains("fill=\"#ff0000\""));
    }

    #[test]
    fn circle_attribute_order_is_fixed() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        buf.circle(10.5, 20.5, 3.0, &FillStroke { fill: Some(from_rgb(0, 0, 0)), stroke: None, stroke_width: 0.0 });
        let out = buf.finish();
        let needle = "<circle cx=\"10.5\" cy=\"20.5\" r=\"3\" fill=\"#000000\"/>";
        assert!(out.contains(needle), "missing exact element; got: {out}");
    }

    #[test]
    fn text_escapes_lt_gt_amp() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        let style = TextStyle { fill: from_rgb(0,0,0), font_size: 11.0, anchor: TextAnchor::Start, angle: 0.0 };
        buf.text(0.0, 0.0, "Price > 0 & < 1", &style);
        let out = buf.finish();
        assert!(out.contains("Price &gt; 0 &amp; &lt; 1"), "got: {out}");
    }

    #[test]
    fn rect_emits_corner_radius_when_positive() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        let r = Rect { x: 0.0, y: 0.0, w: 10.0, h: 5.0 };
        buf.rect(r, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0 }, Some(2.0));
        assert!(buf.as_str().contains("rx=\"2\""));
        assert!(buf.as_str().contains("ry=\"2\""));
    }

    #[test]
    fn rect_omits_corner_radius_when_zero() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        let r = Rect { x: 0.0, y: 0.0, w: 10.0, h: 5.0 };
        buf.rect(r, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0 }, Some(0.0));
        assert!(!buf.as_str().contains("rx="));
    }

    #[test]
    fn translucent_color_uses_rgba() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        buf.circle(0.0, 0.0, 1.0, &FillStroke { fill: Some(from_rgba(255, 0, 0, 128)), stroke: None, stroke_width: 0.0 });
        assert!(buf.as_str().contains("rgba(255,0,0,0.502)"));
    }

    #[test]
    fn determinism_two_calls_byte_identical() {
        let mut a = SvgBuffer::new(vp(), Some(from_rgb(255,255,255)), false);
        let mut b = SvgBuffer::new(vp(), Some(from_rgb(255,255,255)), false);
        a.circle(1.0, 2.0, 3.0, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0 });
        b.circle(1.0, 2.0, 3.0, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0 });
        assert_eq!(a.finish(), b.finish());
    }

    #[test]
    fn clip_open_close_is_balanced() {
        let mut buf = SvgBuffer::new(vp(), None, false);
        buf.clip_open("c1", Rect { x: 0.0, y: 0.0, w: 50.0, h: 30.0 });
        buf.use_clip_open("c1");
        buf.circle(0.0, 0.0, 1.0, &FillStroke { fill: Some(from_rgb(0,0,0)), stroke: None, stroke_width: 0.0 });
        buf.use_clip_close();
        let out = buf.finish();
        assert!(out.contains("<clipPath id=\"c1\">"));
        assert!(out.contains("<g clip-path=\"url(#c1)\">"));
        assert_eq!(out.matches("</g>").count(), 1);
    }
}
```

- [ ] **Step 2: Implement `embed_font.rs`**

Replace `crates/ferrum-core/src/render/embed_font.rs` with:

```rust
//! Inter Regular as a base64-embedded @font-face block.
//! Emitted unconditionally for SVG output in Phase 7 (locked decision §11 row 15).

use base64::Engine;

use super::font::INTER_REGULAR;

/// Returns the full `<defs><style>@font-face{...}</style></defs>` block,
/// ready to be appended to the SVG header.
pub fn inter_data_url_block() -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(INTER_REGULAR);
    format!(
        "<defs><style>@font-face{{font-family:\"Inter\";src:url(\"data:font/ttf;base64,{}\") format(\"truetype\");}}</style></defs>",
        b64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_is_well_formed() {
        let s = inter_data_url_block();
        assert!(s.starts_with("<defs><style>@font-face"));
        assert!(s.ends_with("</style></defs>"));
        assert!(s.contains("font-family:\"Inter\""));
        assert!(s.contains("data:font/ttf;base64,"));
    }
}
```

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::svg render::embed_font
```

Expected: 11 + 1 = 12 tests pass. Cargo total ≥ 220.

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/render/svg.rs crates/ferrum-core/src/render/embed_font.rs
git commit -m "feat(render): SvgBuffer with deterministic floats + Inter @font-face embed

Hand-rolled per spec §4.4 / locked decision §11 row 4. Float formatting
via {:.3} + trim; fixed attribute order per element; XML escapes for
text content and attribute values. Clip-path / use-clip helpers for
panel-bound mark drawing. embed_font emits inline base64 Inter for
visual determinism (locked decision §11 row 15)."
```

---

### Task 10: `scale_resolve.rs` — `ScaleKind`, `ColorScale`, `ResolvedScales`

**Files:**
- Modify: `crates/ferrum-core/src/render/scale_resolve.rs`

- [ ] **Step 1: Read existing scale module to understand the surface**

```bash
ls crates/ferrum-core/src/scale/
grep -n "pub " crates/ferrum-core/src/scale/linear.rs | head -20
```

Note the public methods on each scale type — `LinearScale::scale(value)`, `LinearScale::ticks(count)`, etc. Phase 4's scales are already PyClasses; we need to wrap them in a Rust-side enum that implements a uniform `value_to_pixel` interface.

- [ ] **Step 2: Implement `scale_resolve.rs`**

Replace `crates/ferrum-core/src/render/scale_resolve.rs` with:

```rust
//! Build ResolvedScales from a ChartSpec + a post-transform RecordBatch.
//! Phase 7 supports: LinearScale, OrdinalScale, TimeScale on x/y; CategoricalColorScale on color.

use arrow::array::Array;
use arrow::datatypes::DataType as ArrowDataType;
use arrow::record_batch::RecordBatch;

use crate::scale::linear::LinearScale;
use crate::scale::ordinal::OrdinalScale;
use crate::scale::time::TimeScale;
use crate::spec::chart::ChartSpec;
use crate::spec::encoding::DataType as SpecDataType;

use super::color::Color;
use super::palette::OKABE_ITO;
use super::RenderError;

/// Sealed-enum wrapper over Phase 4 scales, used during render.
/// Phase 7 only constructs Linear/Ordinal/Time variants.
pub enum ScaleKind {
    Linear(LinearScale),
    Ordinal(OrdinalScale),
    Time(TimeScale),
}

impl ScaleKind {
    /// Map a quantitative or temporal value to a pixel coordinate.
    /// Returns None for ordinal scales (use `to_pixel_str` instead).
    pub fn to_pixel_f64(&self, x: f64) -> Option<f64> {
        match self {
            Self::Linear(s) => Some(s.scale_internal(x)),
            Self::Time(s) => Some(s.scale_internal(x)),
            Self::Ordinal(_) => None,
        }
    }

    /// Map an ordinal/string value to a pixel band center.
    /// Returns None for non-ordinal scales.
    pub fn to_pixel_str(&self, value: &str) -> Option<f64> {
        match self {
            Self::Ordinal(s) => s.scale_internal(value),
            _ => None,
        }
    }

    /// Generate tick values as displayable strings.
    pub fn tick_labels(&self, count_hint: usize) -> Vec<String> {
        match self {
            Self::Linear(s) => s.ticks_internal(count_hint).into_iter().map(super::format::format_numeric).collect(),
            Self::Ordinal(s) => s.ticks_internal().into_iter().map(super::format::format_ordinal).collect(),
            Self::Time(s) => {
                let ticks = s.ticks_internal(count_hint);
                let spacing = if ticks.len() >= 2 { (ticks[1] - ticks[0]) as i64 } else { 86_400_000 };
                ticks.into_iter().map(|t| super::format::format_time(t as i64, spacing)).collect()
            }
        }
    }

    /// Pixel-range used when constructing this scale (min, max).
    pub fn pixel_range(&self) -> (f64, f64) {
        match self {
            Self::Linear(s) => (s.range[0], s.range[1]),
            Self::Ordinal(s) => (s.range[0], s.range[1]),
            Self::Time(s) => (s.range[0], s.range[1]),
        }
    }
}

pub enum ColorScale {
    Categorical {
        domain: Vec<String>,
        palette: &'static [Color],
    },
}

impl ColorScale {
    pub fn lookup(&self, value: &str) -> Option<Color> {
        match self {
            Self::Categorical { domain, palette } => {
                domain.iter().position(|v| v == value)
                    .map(|i| palette[i % palette.len()])
            }
        }
    }
}

pub struct ResolvedScales {
    pub x: ScaleKind,
    pub y: ScaleKind,
    pub color: Option<ColorScale>,
}

/// Build scales from spec + post-transform batch + pixel ranges (x_range, y_range).
/// Pixel ranges are panel-relative; caller passes panel.plot_area bounds.
pub fn resolve_scales(
    spec: &ChartSpec,
    batch: &RecordBatch,
    x_pixel_range: (f64, f64),
    y_pixel_range: (f64, f64),
) -> Result<(ResolvedScales, Vec<crate::render::RenderWarning>), RenderError> {
    let mut warnings = Vec::new();

    let x_enc = spec.encoding.x.as_ref()
        .ok_or(RenderError::EncodingTypeMismatch {
            channel: "x", expected: "EncodingSpec", got: "None".into(),
        })?;
    let y_enc = spec.encoding.y.as_ref()
        .ok_or(RenderError::EncodingTypeMismatch {
            channel: "y", expected: "EncodingSpec", got: "None".into(),
        })?;

    let x = build_axis_scale("x", x_enc, batch, x_pixel_range)?;
    let y = build_axis_scale("y", y_enc, batch, y_pixel_range)?;
    let color = if let Some(c_enc) = &spec.encoding.color {
        let domain = distinct_values_in_order(batch, &c_enc.field)?;
        if domain.len() > OKABE_ITO.len() {
            warnings.push(crate::render::RenderWarning::ColorPaletteOverflowed {
                categories: domain.len() as u32,
            });
        }
        Some(ColorScale::Categorical { domain, palette: OKABE_ITO })
    } else {
        None
    };

    Ok((ResolvedScales { x, y, color }, warnings))
}

fn build_axis_scale(
    channel: &'static str,
    enc: &crate::spec::encoding::EncodingSpec,
    batch: &RecordBatch,
    pixel_range: (f64, f64),
) -> Result<ScaleKind, RenderError> {
    let col = batch.column_by_name(&enc.field)
        .ok_or_else(|| RenderError::UnknownColumn { name: enc.field.clone() })?;
    let dtype = infer_spec_type(enc, col.data_type());
    match dtype {
        SpecDataType::Quantitative => {
            let (min, max) = column_min_max_f64(col)
                .map_err(|e| RenderError::ScaleResolutionFailed(format!("{channel}: {e}")))?;
            // Y-axis pixel range is inverted (top is min y, bottom is max y).
            let pr = if channel == "y" { (pixel_range.1, pixel_range.0) } else { pixel_range };
            Ok(ScaleKind::Linear(LinearScale::new_internal(vec![min, max], vec![pr.0, pr.1], false, false)))
        }
        SpecDataType::Ordinal | SpecDataType::Nominal => {
            let domain = distinct_values_in_order(batch, &enc.field)?;
            let pr = if channel == "y" { (pixel_range.1, pixel_range.0) } else { pixel_range };
            Ok(ScaleKind::Ordinal(OrdinalScale::new_internal(domain, vec![pr.0, pr.1], 0.0)))
        }
        SpecDataType::Temporal => {
            let (min, max) = column_min_max_f64(col)
                .map_err(|e| RenderError::ScaleResolutionFailed(format!("{channel}: {e}")))?;
            let pr = if channel == "y" { (pixel_range.1, pixel_range.0) } else { pixel_range };
            Ok(ScaleKind::Time(TimeScale::new_internal(vec![min, max], vec![pr.0, pr.1], false, false)))
        }
    }
}

fn infer_spec_type(enc: &crate::spec::encoding::EncodingSpec, dtype: &ArrowDataType) -> SpecDataType {
    if let Some(t) = enc.type_ {
        return t;
    }
    match dtype {
        ArrowDataType::Float32 | ArrowDataType::Float64
        | ArrowDataType::Int8 | ArrowDataType::Int16 | ArrowDataType::Int32 | ArrowDataType::Int64
        | ArrowDataType::UInt8 | ArrowDataType::UInt16 | ArrowDataType::UInt32 | ArrowDataType::UInt64
            => SpecDataType::Quantitative,
        ArrowDataType::Date32 | ArrowDataType::Date64
        | ArrowDataType::Timestamp(_, _)
            => SpecDataType::Temporal,
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Boolean
            => SpecDataType::Nominal,
        _ => SpecDataType::Nominal,
    }
}

fn column_min_max_f64(col: &dyn Array) -> Result<(f64, f64), String> {
    use arrow::array::{Float64Array, Int64Array, TimestampMillisecondArray};
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        let min = a.iter().flatten().fold(f64::INFINITY, f64::min);
        let max = a.iter().flatten().fold(f64::NEG_INFINITY, f64::max);
        Ok((min, max))
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        let min = a.iter().flatten().fold(i64::MAX, i64::min) as f64;
        let max = a.iter().flatten().fold(i64::MIN, i64::max) as f64;
        Ok((min, max))
    } else if let Some(a) = col.as_any().downcast_ref::<TimestampMillisecondArray>() {
        let min = a.iter().flatten().fold(i64::MAX, i64::min) as f64;
        let max = a.iter().flatten().fold(i64::MIN, i64::max) as f64;
        Ok((min, max))
    } else {
        Err(format!("unsupported column dtype: {:?}", col.data_type()))
    }
}

fn distinct_values_in_order(batch: &RecordBatch, field: &str) -> Result<Vec<String>, RenderError> {
    use arrow::array::{StringArray, Int64Array, BooleanArray};
    let col = batch.column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    let mut seen = std::collections::HashSet::<String>::new();
    let mut out = Vec::<String>::new();
    let push = |s: String, seen: &mut std::collections::HashSet<String>, out: &mut Vec<String>| {
        if seen.insert(s.clone()) { out.push(s); }
    };
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        for v in a.iter().flatten() { push(v.to_string(), &mut seen, &mut out); }
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        for v in a.iter().flatten() { push(v.to_string(), &mut seen, &mut out); }
    } else if let Some(a) = col.as_any().downcast_ref::<BooleanArray>() {
        for v in a.iter().flatten() { push(v.to_string(), &mut seen, &mut out); }
    } else {
        return Err(RenderError::ScaleResolutionFailed(
            format!("can't enumerate distinct values from column dtype {:?}", col.data_type())
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    fn make_batch_q_q_n() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", ArrowDataType::Float64, false),
            Field::new("y", ArrowDataType::Float64, false),
            Field::new("species", ArrowDataType::Utf8, false),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
            Arc::new(StringArray::from(vec!["a", "b", "a", "c", "b", "c"])),
        ]).unwrap()
    }

    fn make_spec_with_color() -> ChartSpec {
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: Some(EncodingSpec { field: "species".into(), type_: None }),
            },
            transforms: Vec::new(),
            facet: None,
        }
    }

    #[test]
    fn quantitative_x_resolves_to_linear() {
        let s = make_spec_with_color();
        let b = make_batch_q_q_n();
        let (scales, warnings) = resolve_scales(&s, &b, (0.0, 100.0), (0.0, 80.0)).unwrap();
        assert!(matches!(scales.x, ScaleKind::Linear(_)));
        assert!(matches!(scales.y, ScaleKind::Linear(_)));
        assert!(warnings.is_empty());
    }

    #[test]
    fn color_encoding_builds_categorical_in_encounter_order() {
        let s = make_spec_with_color();
        let b = make_batch_q_q_n();
        let (scales, _) = resolve_scales(&s, &b, (0.0, 100.0), (0.0, 80.0)).unwrap();
        let cs = scales.color.unwrap();
        match cs {
            ColorScale::Categorical { domain, .. } => {
                assert_eq!(domain, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
            }
        }
    }

    #[test]
    fn unknown_x_column_errors() {
        let mut s = make_spec_with_color();
        s.encoding.x = Some(crate::spec::encoding::EncodingSpec { field: "missing".into(), type_: None });
        let b = make_batch_q_q_n();
        let err = resolve_scales(&s, &b, (0.0, 100.0), (0.0, 80.0)).unwrap_err();
        assert!(matches!(err, RenderError::UnknownColumn { .. }));
    }

    #[test]
    fn color_overflow_emits_warning() {
        // Build a 10-category color encoding.
        use crate::spec::data_ref::DataRef;
        use crate::spec::encoding::{Encoding, EncodingSpec};
        use crate::spec::mark::Mark;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", ArrowDataType::Float64, false),
            Field::new("y", ArrowDataType::Float64, false),
            Field::new("g", ArrowDataType::Utf8, false),
        ]));
        let groups: Vec<String> = (0..10).map(|i| format!("g{i}")).collect();
        let groups_str: Vec<&str> = groups.iter().map(String::as_str).collect();
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys = xs.clone();
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(xs)),
            Arc::new(Float64Array::from(ys)),
            Arc::new(StringArray::from(groups_str)),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: Some(EncodingSpec { field: "g".into(), type_: None }),
            },
            transforms: Vec::new(),
            facet: None,
        };
        let (_, warnings) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 80.0)).unwrap();
        assert!(matches!(warnings[0], crate::render::RenderWarning::ColorPaletteOverflowed { categories: 10 }));
    }
}
```

> **Implementer note on scale internal accessors:** `LinearScale::scale_internal`, `LinearScale::new_internal`, `LinearScale::ticks_internal`, and the `range` field accessor pattern come from Phase 4's existing crate-private API. Verify these exist on each scale (`LinearScale`, `OrdinalScale`, `TimeScale`) by reading `crates/ferrum-core/src/scale/{linear,ordinal,time}.rs`. If the internal API is named differently (e.g., `scale_internal` doesn't exist but only the `#[pymethods] scale(&self, x: f64)` does), you have two options:
>
> (a) **Add small `pub(crate) fn scale_rust(&self, x: f64) -> f64` shims** to each Phase 4 scale module that the pymethods then call. This is the cleaner fix.
> (b) Call the pymethod variants directly with `Python::with_gil(|py| ...)`. Less clean but no Phase 4 changes.
>
> Pick (a) and add the shims as part of this task. Adjust the test count accordingly (each shim is a one-liner; tests count stays the same).

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::scale_resolve
```

Expected: 4 tests pass. Cargo total ≥ 224.

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/render/scale_resolve.rs crates/ferrum-core/src/scale/
git commit -m "feat(render): scale_resolve — ScaleKind, ColorScale, ResolvedScales

Sealed-enum wrapper over Phase 4 LinearScale/OrdinalScale/TimeScale with
uniform value→pixel + tick-label generation. CategoricalColorScale over
OKABE_ITO; overflow emits ColorPaletteOverflowed. Unknown column raises
RenderError::UnknownColumn. Quantitative inferred from arrow dtype when
spec.type_ is omitted. Adds pub(crate) fn shims on Phase 4 scales for
non-PyO3 internal access."
```

---

### Task 11: `prepare.rs` — orchestrate transforms + scales + axes + facets + legend

**Files:**
- Modify: `crates/ferrum-core/src/render/prepare.rs`

- [ ] **Step 1: Implement `prepare.rs`**

Replace `crates/ferrum-core/src/render/prepare.rs` with:

```rust
//! prepare_render_inputs(spec, batch) →
//!   1. Apply Phase 5 transforms.
//!   2. Build provisional ResolvedScales for tick-label generation.
//!   3. Derive AxesInput (titles, tick_labels).
//!   4. Group rows by facet field (if facet).
//!   5. Build LegendEntry list (if color encoding).
//!
//! The actual final scales are rebuilt per-panel inside render_svg with the
//! correct pixel ranges; this function returns *only* the data plumbing.

use arrow::record_batch::RecordBatch;

use crate::layout::{
    AxesInput, AxisInput, AxisOrient, FacetGroup, FacetKey, LegendEntry, SymbolKind,
};
use crate::spec::chart::ChartSpec;
use crate::transform::apply::apply_transforms;

use super::scale_resolve::{resolve_scales, ResolvedScales};
use super::{RenderError, RenderWarning};

pub struct PreparedInputs {
    pub transformed: RecordBatch,
    pub provisional_scales: ResolvedScales,
    pub axes: AxesInput,
    pub facet_groups: Vec<FacetGroup>,
    pub legend_entries: Vec<LegendEntry>,
    pub warnings: Vec<RenderWarning>,
}

pub fn prepare_render_inputs(
    spec: &ChartSpec,
    batch: &RecordBatch,
) -> Result<PreparedInputs, RenderError> {
    if batch.num_rows() == 0 {
        return Err(RenderError::EmptyBatch);
    }

    // 1. Apply Phase 5 transforms.
    let transformed = if spec.transforms.is_empty() {
        batch.clone()
    } else {
        apply_transforms(&spec.transforms, batch)
            .map_err(|e| RenderError::TransformFailed(e.to_string()))?
    };

    // 2. Build provisional scales over a placeholder pixel range (0..1).
    //    Final pixel ranges are panel-specific; we just need tick labels here.
    let (provisional_scales, scale_warnings) =
        resolve_scales(spec, &transformed, (0.0, 1.0), (0.0, 1.0))?;

    // 3. AxesInput.
    let x_field = spec.encoding.x.as_ref().map(|e| e.field.clone());
    let y_field = spec.encoding.y.as_ref().map(|e| e.field.clone());
    let x_tick_labels = provisional_scales.x.tick_labels(10);
    let y_tick_labels = provisional_scales.y.tick_labels(10);
    let axes = AxesInput {
        x: AxisInput {
            orient: AxisOrient::Bottom,
            title: x_field,
            tick_labels: x_tick_labels,
            label_angle_override: None,
        },
        y: AxisInput {
            orient: AxisOrient::Left,
            title: y_field,
            tick_labels: y_tick_labels,
            label_angle_override: None,
        },
    };

    // 4. Facet groups (in encounter order).
    let facet_groups = if let Some(fspec) = &spec.facet {
        group_rows_by_field(&transformed, &fspec.field)?
    } else {
        Vec::new()
    };

    // 5. Legend entries (only when color encoding present).
    let legend_entries = match &provisional_scales.color {
        Some(super::scale_resolve::ColorScale::Categorical { domain, .. }) => domain
            .iter()
            .map(|v| LegendEntry { label: v.clone(), symbol: SymbolKind::Circle })
            .collect(),
        None => Vec::new(),
    };

    Ok(PreparedInputs {
        transformed,
        provisional_scales,
        axes,
        facet_groups,
        legend_entries,
        warnings: scale_warnings,
    })
}

fn group_rows_by_field(batch: &RecordBatch, field: &str) -> Result<Vec<FacetGroup>, RenderError> {
    use arrow::array::StringArray;
    let col = batch.column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    let arr = col.as_any().downcast_ref::<StringArray>()
        .ok_or_else(|| RenderError::ScaleResolutionFailed(
            format!("facet field '{field}' must be Utf8 (Phase 7 limitation)"),
        ))?;
    let mut order: Vec<String> = Vec::new();
    let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for v in arr.iter().flatten() {
        let s = v.to_string();
        if !counts.contains_key(&s) {
            order.push(s.clone());
        }
        *counts.entry(s).or_insert(0) += 1;
    }
    Ok(order.into_iter()
        .map(|v| FacetGroup {
            key: FacetKey { field: field.to_string(), value: v.clone() },
            n_rows: counts[&v],
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    fn batch3() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("species", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            Arc::new(StringArray::from(vec!["a", "b", "a"])),
        ]).unwrap()
    }

    fn spec_color_facet() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: Some(EncodingSpec { field: "species".into(), type_: None }),
            },
            transforms: Vec::new(),
            facet: Some(crate::layout::FacetSpec {
                field: "species".into(),
                mode: crate::layout::FacetMode::Wrap { ncols: 2 },
                spacing: None,
            }),
        }
    }

    #[test]
    fn prepare_returns_axes_and_groups_and_legend() {
        let spec = spec_color_facet();
        let batch = batch3();
        let prep = prepare_render_inputs(&spec, &batch).unwrap();
        assert_eq!(prep.axes.x.title.as_deref(), Some("x"));
        assert_eq!(prep.axes.y.title.as_deref(), Some("y"));
        assert!(!prep.axes.x.tick_labels.is_empty());
        assert_eq!(prep.facet_groups.len(), 2);   // "a", "b"
        assert_eq!(prep.facet_groups[0].n_rows, 2);
        assert_eq!(prep.facet_groups[1].n_rows, 1);
        assert_eq!(prep.legend_entries.len(), 2);
        assert_eq!(prep.legend_entries[0].label, "a");
    }

    #[test]
    fn empty_batch_errors() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
        ]).unwrap();
        let mut spec = spec_color_facet();
        spec.encoding.color = None;
        spec.facet = None;
        let err = prepare_render_inputs(&spec, &batch).unwrap_err();
        assert!(matches!(err, RenderError::EmptyBatch));
    }
}
```

- [ ] **Step 2: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::prepare
```

Expected: 2 tests pass. Cargo total ≥ 226.

> **Implementer note:** `apply_transforms` is the Phase 5 entry. Read `crates/ferrum-core/src/transform/apply.rs` (or wherever it lives) to confirm the exact name and signature. If the function is named differently (e.g., `run_transforms` or lives in `transform/mod.rs`), update the import.

- [ ] **Step 3: Commit**

```bash
git add crates/ferrum-core/src/render/prepare.rs
git commit -m "feat(render): prepare_render_inputs orchestrates the pre-layout pipeline

Apply Phase 5 transforms, build provisional ResolvedScales over a 0-1
placeholder range, derive AxesInput tick labels, group rows by facet
field (encounter order), build LegendEntry list from color encoding.
EmptyBatch error matches spec §6 step 1."
```

---

### Task 12: `compute_layout` populates `strip_title` when faceted

**Files:**
- Modify: `crates/ferrum-core/src/layout/mod.rs`

- [ ] **Step 1: Write failing test**

Add to the `#[cfg(test)] mod tests {}` block at the bottom of `crates/ferrum-core/src/layout/mod.rs`:

```rust
#[test]
fn compute_layout_faceted_emits_strip_titles() {
    let spec = faceted_spec(3);
    let groups = three_groups();
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
    ).unwrap();

    assert_eq!(result.panels.len(), 3);
    for (i, panel) in result.panels.iter().enumerate() {
        let strip = panel.strip_title.as_ref()
            .unwrap_or_else(|| panic!("panel {i} missing strip_title"));
        assert!(!strip.text.is_empty());
        assert_eq!(strip.font_size, 13.0);
        // Anchor must be inside the panel x-extent.
        assert!(strip.anchor.0 >= panel.plot_area.x);
        assert!(strip.anchor.0 <= panel.plot_area.x + panel.plot_area.w);
    }
}

#[test]
fn compute_layout_unfaceted_omits_strip_titles() {
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
    ).unwrap();
    assert!(result.panels[0].strip_title.is_none());
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout::tests::compute_layout_faceted_emits_strip_titles layout::tests::compute_layout_unfaceted_omits_strip_titles
```

Expected: faceted test fails (`strip_title` is None for all panels); unfaceted test passes.

- [ ] **Step 3: Reserve strip-title band + populate `strip_title`**

In `crates/ferrum-core/src/layout/mod.rs`, modify `compute_layout` in two places:

1. **After step 5 (label-band reservation), before step 6 (split into facet cells)**, add a strip-band reservation when the chart is faceted:

```rust
// 5b. Reserve strip-title band for faceted charts (top of plot_region).
let strip_band_height = if spec.facet.is_some() {
    metrics.line_height(theme.strip_text_size) + 2.0 * theme.strip_padding
} else {
    0.0
};
let plot_region = if strip_band_height > 0.0 {
    plot_region.shrink(Inset { top: strip_band_height, ..Default::default() })
} else {
    plot_region
};
```

(But each panel's plot_area must reserve its OWN strip band, not the whole plot_region. Re-think: the cleaner design is to NOT subtract strip_band_height from `plot_region` here, and instead subtract it from each panel's rect inside the per-panel loop in step 7. Use the latter approach — see step 4.)

Replace the snippet above with:

```rust
let strip_band_height = if spec.facet.is_some() {
    metrics.line_height(theme.strip_text_size) + 2.0 * theme.strip_padding
} else {
    0.0
};
```

(no plot_region change here)

2. **Inside step 7 (per-panel loop)**, before pushing the `PanelLayout`, carve the strip band off the top of each panel rect and build a `StripTitleLayout`:

```rust
let strip_title = if let Some(key) = &facet_key {
    if rect != Rect::ZERO {
        let strip_rect = Rect {
            x: rect.x,
            y: rect.y,
            w: rect.w,
            h: strip_band_height,
        };
        let new_panel_rect = Rect {
            x: rect.x,
            y: rect.y + strip_band_height,
            w: rect.w,
            h: (rect.h - strip_band_height).max(0.0),
        };
        rect = new_panel_rect;
        Some(StripTitleLayout {
            text: key.value.clone(),
            anchor: (
                strip_rect.x + strip_rect.w / 2.0,
                strip_rect.y + theme.strip_padding + theme.strip_text_size,
            ),
            align: TextAnchor::Middle,
            font_size: theme.strip_text_size,
        })
    } else {
        None
    }
} else {
    None
};

panels.push(PanelLayout {
    plot_area: rect,
    facet_key,
    row,
    col,
    strip_title,
});
```

(Replacing the existing `panels.push(PanelLayout { plot_area: rect, facet_key, row, col });` block.)

- [ ] **Step 4: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core layout
```

Expected: 2 new tests pass; all other Phase 6 layout tests still pass. Cargo total ≥ 228.

- [ ] **Step 5: Commit**

```bash
git add crates/ferrum-core/src/layout/mod.rs
git commit -m "feat(layout): compute_layout populates strip_title for faceted panels

Reserves strip_text_size + 2*strip_padding off the top of each faceted
panel rect. Strip text anchors at panel x-center, baseline below the
top of the band. Non-faceted layouts unchanged (strip_title remains
None and is omitted from JSON via skip_serializing_if)."
```

---

### Task 13: `draw.rs` — `DrawCtx`, `MarkStyle`, `dispatch_mark`

**Files:**
- Modify: `crates/ferrum-core/src/render/draw.rs`

- [ ] **Step 1: Implement `draw.rs`**

Replace `crates/ferrum-core/src/render/draw.rs` with:

```rust
//! Per-panel draw context + mark dispatch. Spec §4.5 / §4.6.

use arrow::record_batch::RecordBatch;

use crate::layout::{PanelLayout, ThemeInputs};
use crate::spec::mark::Mark;

use super::color::{with_opacity, Color};
use super::scale_resolve::ResolvedScales;
use super::svg::SvgBuffer;

pub struct DrawCtx<'a> {
    pub panel: &'a PanelLayout,
    pub theme: &'a ThemeInputs,
    pub scales: &'a ResolvedScales,
    pub batch: &'a RecordBatch,
    pub mark_style: &'a MarkStyle,
}

#[derive(Debug, Clone)]
pub struct MarkStyle {
    pub fill: Color,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
    pub opacity: f64,
    pub point_size: f64,
    pub corner_radius: f64,
    pub stroke_dash: Option<Vec<f64>>,
}

/// Resolve effective MarkStyle from theme defaults, mark variant, and any
/// per-mark spec overrides. Phase 7's spec carries no mark-level visual
/// overrides yet (those are Phase 8 grammar surface), so this is theme-only.
pub fn resolve_mark_style(theme: &ThemeInputs, mark: &Mark) -> MarkStyle {
    let base_fill = with_opacity(theme.mark_color, theme.default_opacity);
    match mark {
        Mark::Area => MarkStyle {
            fill: with_opacity(theme.mark_color, theme.area_opacity),
            stroke: Some(theme.mark_color),
            stroke_width: theme.line_stroke_width,
            opacity: 1.0,
            point_size: theme.point_size,
            corner_radius: 0.0,
            stroke_dash: None,
        },
        Mark::Line => MarkStyle {
            fill: theme.mark_color,
            stroke: Some(theme.mark_color),
            stroke_width: theme.line_stroke_width,
            opacity: theme.default_opacity,
            point_size: theme.point_size,
            corner_radius: 0.0,
            stroke_dash: None,
        },
        Mark::Bar | Mark::Rect => MarkStyle {
            fill: base_fill,
            stroke: None,
            stroke_width: 0.0,
            opacity: theme.default_opacity,
            point_size: theme.point_size,
            corner_radius: theme.bar_corner_radius,
            stroke_dash: None,
        },
        Mark::Rule => MarkStyle {
            fill: theme.mark_color,
            stroke: Some(theme.mark_color),
            stroke_width: theme.line_stroke_width,
            opacity: theme.default_opacity,
            point_size: theme.point_size,
            corner_radius: 0.0,
            stroke_dash: None,
        },
        Mark::Tick | Mark::Point | Mark::Text => MarkStyle {
            fill: base_fill,
            stroke: None,
            stroke_width: 0.0,
            opacity: theme.default_opacity,
            point_size: theme.point_size,
            corner_radius: 0.0,
            stroke_dash: None,
        },
    }
}

/// Sealed-enum dispatch (spec §4.6).
pub fn dispatch_mark(mark: &Mark, ctx: &DrawCtx, out: &mut SvgBuffer) {
    match mark {
        Mark::Point => super::marks::point::draw(ctx, out),
        Mark::Line  => super::marks::line::draw(ctx, out),
        Mark::Area  => super::marks::area::draw(ctx, out),
        Mark::Bar   => super::marks::bar::draw(ctx, out),
        Mark::Rect  => super::marks::rect::draw(ctx, out),
        Mark::Rule  => super::marks::rule::draw(ctx, out),
        Mark::Text  => super::marks::text::draw(ctx, out),
        Mark::Tick  => super::marks::tick::draw(ctx, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_style_for_area_uses_area_opacity() {
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(&theme, &Mark::Area);
        // Area opacity = 0.4; alpha = 255 * 0.4 ≈ 102.
        assert!((style.fill.alpha as i32 - 102).abs() <= 1);
    }

    #[test]
    fn resolve_style_for_bar_has_corner_radius_from_theme() {
        let mut theme = ThemeInputs::default();
        theme.bar_corner_radius = 4.0;
        let style = resolve_mark_style(&theme, &Mark::Bar);
        assert_eq!(style.corner_radius, 4.0);
    }

    #[test]
    fn resolve_style_for_point_is_opaque_by_default() {
        let theme = ThemeInputs::default();
        let style = resolve_mark_style(&theme, &Mark::Point);
        assert_eq!(style.fill.alpha, 0xFF);
    }
}
```

> **Implementer note:** the per-mark `super::marks::*::draw` references are stubs for now (Task 1 created `marks/*.rs` placeholders). To make this compile before Tasks 14-17 implement them, each mark stub should expose:
>
> ```rust
> //! Placeholder — implementation lands in subsequent tasks.
>
> #[allow(unused_variables, dead_code)]
> pub fn draw(_ctx: &super::super::draw::DrawCtx, _out: &mut super::super::svg::SvgBuffer) {
>     // no-op stub; real impl in Task 14+.
> }
> ```
>
> Update each of the 8 primitive mark files (and `axis`, `legend`, `strip_title`) with this stub before running `cargo test`. Replace the `//! Placeholder` body in each.

- [ ] **Step 2: Update mark stubs to expose `draw`**

For each of `point.rs`, `line.rs`, `area.rs`, `bar.rs`, `rect.rs`, `rule.rs`, `text.rs`, `tick.rs`, `axis.rs`, `legend.rs`, `strip_title.rs` in `crates/ferrum-core/src/render/marks/`, replace the file contents with:

```rust
//! Stub — implementation lands in subsequent tasks.

#[allow(unused_variables, dead_code)]
pub fn draw(_ctx: &crate::render::draw::DrawCtx, _out: &mut crate::render::svg::SvgBuffer) {}
```

Note: `axis`, `legend`, `strip_title` aren't dispatched via `Mark::*`; their `draw` signatures will diverge in Tasks 18-19. The stub is fine for now since `dispatch_mark` only calls the eight primitive marks.

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::draw
```

Expected: 3 tests pass. Cargo total ≥ 231.

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/render/draw.rs crates/ferrum-core/src/render/marks/
git commit -m "feat(render): DrawCtx + MarkStyle + dispatch_mark

Spec §4.5 / §4.6. resolve_mark_style maps theme + mark variant to a
concrete MarkStyle (fill / stroke / opacity / point_size /
corner_radius / dash). dispatch_mark is the sealed-enum entry point;
mark stubs are no-ops until Task 14+ wires real draw fns."
```

---

### Task 14: `marks/point.rs` — first end-to-end mark

**Files:**
- Modify: `crates/ferrum-core/src/render/marks/point.rs`

**Helper utilities** (used by every mark in Tasks 14-17). Add these to `crates/ferrum-core/src/render/draw.rs` (under the existing `resolve_mark_style` fn), before implementing this task:

```rust
use arrow::array::{Array, Float64Array, Int64Array, StringArray, TimestampMillisecondArray};

/// Try to read a column as `f64`, regardless of whether the underlying type
/// is Float64 / Int64 / Timestamp(ms). Returns None for null rows.
pub fn col_as_f64<'a>(batch: &'a RecordBatch, field: &str) -> Result<Vec<Option<f64>>, super::RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| super::RenderError::UnknownColumn { name: field.to_string() })?;
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        Ok(a.iter().collect())
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        Ok(a.iter().map(|v| v.map(|x| x as f64)).collect())
    } else if let Some(a) = col.as_any().downcast_ref::<TimestampMillisecondArray>() {
        Ok(a.iter().map(|v| v.map(|x| x as f64)).collect())
    } else {
        Err(super::RenderError::ScaleResolutionFailed(
            format!("column '{field}' has unsupported dtype for f64 read: {:?}", col.data_type())
        ))
    }
}

/// Read a column as Vec<Option<String>>.
pub fn col_as_str(batch: &RecordBatch, field: &str) -> Result<Vec<Option<String>>, super::RenderError> {
    let col = batch.column_by_name(field)
        .ok_or_else(|| super::RenderError::UnknownColumn { name: field.to_string() })?;
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        Ok(a.iter().map(|o| o.map(|s| s.to_string())).collect())
    } else {
        Err(super::RenderError::ScaleResolutionFailed(
            format!("column '{field}' must be Utf8 to read as strings: {:?}", col.data_type())
        ))
    }
}

/// Spec carries the encoding fields; this resolves them from a DrawCtx.
pub fn x_field<'a>(ctx: &'a DrawCtx, spec: &'a crate::spec::chart::ChartSpec) -> Option<&'a str> {
    spec.encoding.x.as_ref().map(|e| e.field.as_str())
}
pub fn y_field<'a>(ctx: &'a DrawCtx, spec: &'a crate::spec::chart::ChartSpec) -> Option<&'a str> {
    spec.encoding.y.as_ref().map(|e| e.field.as_str())
}
pub fn color_field<'a>(ctx: &'a DrawCtx, spec: &'a crate::spec::chart::ChartSpec) -> Option<&'a str> {
    spec.encoding.color.as_ref().map(|e| e.field.as_str())
}
```

But — `DrawCtx` doesn't carry the `ChartSpec`. To keep mark draw fns self-contained, extend `DrawCtx`:

```rust
pub struct DrawCtx<'a> {
    pub spec: &'a crate::spec::chart::ChartSpec,   // ← NEW
    pub panel: &'a PanelLayout,
    pub theme: &'a ThemeInputs,
    pub scales: &'a ResolvedScales,
    pub batch: &'a RecordBatch,
    pub mark_style: &'a MarkStyle,
}
```

Update the existing `draw.rs` tests to pass a `spec` reference. Use `crate::spec::chart::ChartSpec { ... }` constructed inline in each test. (One small change to each of the three existing tests.)

- [ ] **Step 1: Apply DrawCtx + helper extensions**

Edit `crates/ferrum-core/src/render/draw.rs`:
1. Add the four helper fns above (after `resolve_mark_style`).
2. Add the `spec:` field to `DrawCtx`.
3. Update the existing three tests to construct a minimal `ChartSpec` and pass it via the new field.

- [ ] **Step 2: Implement `marks/point.rs`**

Replace `crates/ferrum-core/src/render/marks/point.rs` with:

```rust
//! mark_point: render each row as a circle at (scale_x(row.x), scale_y(row.y)).

use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::{ColorScale, ScaleKind};
use crate::render::svg::{FillStroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return };

    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    if xs.len() != ys.len() { return; }

    let color_values: Option<Vec<Option<String>>> = color_field(ctx, spec)
        .and_then(|f| col_as_str(ctx.batch, f).ok());

    // Radius from area: radius = sqrt(point_size / pi).
    let radius = (ctx.mark_style.point_size / std::f64::consts::PI).sqrt();

    for i in 0..xs.len() {
        let (xv, yv) = match (xs[i], ys[i]) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => (a, b),
            _ => continue,
        };
        let cx = match scale_value(&ctx.scales.x, xv, None) { Some(p) => p, None => continue };
        let cy = match scale_value(&ctx.scales.y, yv, None) { Some(p) => p, None => continue };
        let fill = if let (Some(scale), Some(values)) = (&ctx.scales.color, &color_values) {
            match values[i].as_deref() {
                Some(v) => match scale {
                    ColorScale::Categorical { .. } => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                },
                None => ctx.mark_style.fill,
            }
        } else {
            ctx.mark_style.fill
        };
        let fill = with_opacity(fill, ctx.mark_style.opacity);
        out.circle(cx, cy, radius, &FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        });
    }
}

fn scale_value(s: &ScaleKind, v: f64, label: Option<&str>) -> Option<f64> {
    match s {
        ScaleKind::Linear(_) | ScaleKind::Time(_) => s.to_pixel_f64(v),
        ScaleKind::Ordinal(_) => label.and_then(|l| s.to_pixel_str(l)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::color::from_rgb;
    use crate::render::draw::{resolve_mark_style, MarkStyle};
    use crate::render::scale_resolve::{resolve_scales, ResolvedScales};
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;

    fn three_row_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(),
            facet: None,
        }
    }

    fn three_row_batch() -> arrow::record_batch::RecordBatch {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
        ]).unwrap()
    }

    #[test]
    fn three_rows_emit_three_circles() {
        let spec = three_row_spec();
        let batch = three_row_batch();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Point);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<circle ").count(), 3);
    }

    #[test]
    fn out_of_domain_rows_are_skipped() {
        // x=NaN should be skipped silently.
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, f64::NAN, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
        ]).unwrap();
        let spec = three_row_spec();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None, row: 0, col: 0, strip_title: None,
        };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Point);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<circle ").count(), 2);
    }
}
```

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::marks::point render::draw
```

Expected: 2 + 3 = 5 tests pass (point's 2 plus the existing draw 3, now updated for the new `spec` field). Cargo total ≥ 233.

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/render/marks/point.rs crates/ferrum-core/src/render/draw.rs
git commit -m "feat(render): mark_point + DrawCtx.spec + col_as_f64/str helpers

First end-to-end mark draw fn — proves the pipeline. Reads x/y/color
columns, scales each row to pixel coords, looks up categorical color
when present, emits <circle> per row. NaN rows skipped silently.
DrawCtx gains spec field for encoding-field access; col_as_f64 and
col_as_str helpers cover the three Phase 7 column dtypes."
```

---

### Task 15: `marks/line.rs` + `marks/area.rs`

**Files:**
- Modify: `crates/ferrum-core/src/render/marks/line.rs`
- Modify: `crates/ferrum-core/src/render/marks/area.rs`

- [ ] **Step 1: Implement `marks/line.rs`**

```rust
//! mark_line: render rows as a single polyline. If a color encoding is present,
//! one polyline per category (rows of the same category linked, in their batch
//! order). Otherwise one polyline over all rows in batch order.

use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::ColorScale;
use crate::render::svg::{Stroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b),
        _ => return,
    };
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    let cf = color_field(ctx, spec);
    let color_values = cf.and_then(|f| col_as_str(ctx.batch, f).ok());

    // Group row indices by category (or one group if no color encoding).
    let groups: Vec<(Option<String>, Vec<usize>)> = match (color_values.as_ref(), &ctx.scales.color) {
        (Some(values), Some(_)) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in values.iter().enumerate() {
                let key = v.clone();
                let pos = groups.iter().position(|(k, _)| k == &key);
                match pos {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((key, vec![i])),
                }
            }
            groups
        }
        _ => vec![(None, (0..xs.len()).collect())],
    };

    for (key, rows) in groups {
        let mut points: Vec<(f64, f64)> = Vec::new();
        for i in rows {
            let (xv, yv) = match (xs[i], ys[i]) {
                (Some(a), Some(b)) if a.is_finite() && b.is_finite() => (a, b),
                _ => continue,
            };
            let cx = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
            let cy = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
            points.push((cx, cy));
        }
        if points.len() < 2 { continue; }

        let stroke_color = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale @ ColorScale::Categorical { .. })) =>
                scale.lookup(v).unwrap_or(ctx.mark_style.fill),
            _ => ctx.mark_style.fill,
        };
        out.polyline(&points, &Stroke {
            stroke: stroke_color,
            stroke_width: ctx.mark_style.stroke_width,
            stroke_dash: ctx.mark_style.stroke_dash.clone(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::{resolve_mark_style};
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn line_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(), mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        }
    }

    #[test]
    fn line_emits_one_polyline_for_5_rows() {
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0,1.0,2.0,3.0,4.0])),
            Arc::new(Float64Array::from(vec![0.0,1.0,2.0,3.0,4.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<polyline ").count(), 1);
    }

    #[test]
    fn line_skips_when_fewer_than_two_points() {
        let spec = line_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Line);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert!(!out.finish().contains("<polyline"));
    }
}
```

- [ ] **Step 2: Implement `marks/area.rs`**

```rust
//! mark_area: filled region between y(x) and the x-axis baseline. Single area
//! over all rows when no color encoding; one area per category otherwise.

use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::{ColorScale, ScaleKind};
use crate::render::svg::{FillStroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return,
    };
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };

    // Baseline = bottom of panel (y=panel.plot_area.y + panel.plot_area.h).
    let baseline_y = ctx.panel.plot_area.y + ctx.panel.plot_area.h;

    let cf = color_field(ctx, spec);
    let color_values = cf.and_then(|f| col_as_str(ctx.batch, f).ok());

    let groups: Vec<(Option<String>, Vec<usize>)> = match (color_values.as_ref(), &ctx.scales.color) {
        (Some(values), Some(_)) => {
            let mut g: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in values.iter().enumerate() {
                let key = v.clone();
                match g.iter().position(|(k, _)| k == &key) {
                    Some(p) => g[p].1.push(i),
                    None => g.push((key, vec![i])),
                }
            }
            g
        }
        _ => vec![(None, (0..xs.len()).collect())],
    };

    for (key, rows) in groups {
        let mut top: Vec<(f64, f64)> = Vec::new();
        for i in rows {
            let (xv, yv) = match (xs[i], ys[i]) {
                (Some(a), Some(b)) if a.is_finite() && b.is_finite() => (a, b),
                _ => continue,
            };
            let cx = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
            let cy = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
            top.push((cx, cy));
        }
        if top.len() < 2 { continue; }
        let path = build_area_path(&top, baseline_y);
        let fill = match (key.as_deref(), &ctx.scales.color) {
            (Some(v), Some(scale @ ColorScale::Categorical { .. })) => {
                let base = scale.lookup(v).unwrap_or(ctx.mark_style.fill);
                crate::render::color::with_opacity(base, ctx.theme.area_opacity)
            }
            _ => ctx.mark_style.fill,
        };
        out.path(&path, &FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        });
    }
}

fn build_area_path(top: &[(f64, f64)], baseline: f64) -> String {
    use crate::render::svg::fmt_f;
    let mut d = String::new();
    let (x0, y0) = top[0];
    d.push_str(&format!("M{} {}", fmt_f(x0), fmt_f(y0)));
    for &(x, y) in &top[1..] {
        d.push_str(&format!(" L{} {}", fmt_f(x), fmt_f(y)));
    }
    let last_x = top[top.len() - 1].0;
    d.push_str(&format!(" L{} {}", fmt_f(last_x), fmt_f(baseline)));
    d.push_str(&format!(" L{} {}", fmt_f(x0), fmt_f(baseline)));
    d.push_str(" Z");
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn area_spec() -> ChartSpec {
        ChartSpec {
            data: DataRef::default(), mark: Mark::Area,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        }
    }

    #[test]
    fn area_emits_one_path_with_z_close() {
        let spec = area_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0,1.0,2.0,3.0,4.0])),
            Arc::new(Float64Array::from(vec![0.0,1.0,2.0,3.0,4.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Area);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<path ").count(), 1);
        assert!(s.contains(" Z\""), "area path must close with Z");
    }

    #[test]
    fn area_uses_translucent_fill() {
        // Theme area_opacity defaults to 0.4 → alpha ≈ 102.
        // fmt_svg should emit "rgba(...,0.4..)" not opaque #rrggbb.
        let spec = area_spec();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Area);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert!(out.finish().contains("rgba("));
    }
}
```

> **Implementer note:** `fmt_f` in `area.rs` is referenced from `svg::fmt_f`. Make sure `pub fn fmt_f(...)` was exported as `pub` (not `pub(crate)`) in Task 9 step 1. If not, add a `pub use self::fmt_f;` re-export in `svg.rs` or make the fn `pub(super)` and have `area.rs` call via the right path.

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::marks::line render::marks::area
```

Expected: 4 tests pass. Cargo total ≥ 237.

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/render/marks/line.rs crates/ferrum-core/src/render/marks/area.rs
git commit -m "feat(render): mark_line + mark_area

Both group rows by color category (one group if no color encoding).
mark_line emits one <polyline> per group; skips when < 2 points.
mark_area builds an SVG path closed at the panel baseline; fill uses
theme.area_opacity (alpha ≈ 102 by default)."
```

---

### Task 16: `marks/bar.rs` + `marks/rect.rs`

**Files:**
- Modify: `crates/ferrum-core/src/render/marks/bar.rs`
- Modify: `crates/ferrum-core/src/render/marks/rect.rs`

- [ ] **Step 1: Implement `marks/bar.rs`**

```rust
//! mark_bar: ordinal x → quantitative y. One <rect> per row, anchored at
//! the ordinal x-band, extending from baseline (y=0 mapped) to scale_y(row.y).

use crate::layout::Rect;
use crate::render::color::with_opacity;
use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::{ColorScale, ScaleKind};
use crate::render::svg::{FillStroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let xf = match x_field(ctx, spec) { Some(f) => f, None => return };
    let yf = match y_field(ctx, spec) { Some(f) => f, None => return };
    let x_strs = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    if x_strs.len() != ys.len() { return; }

    let panel = ctx.panel.plot_area;
    let baseline_y = panel.y + panel.h;

    let n_categories = match &ctx.scales.x {
        ScaleKind::Ordinal(_) => x_strs.iter().flatten().collect::<std::collections::HashSet<_>>().len().max(1),
        _ => return,
    };
    let bar_width = (panel.w / n_categories as f64) * 0.8;

    let color_values = color_field(ctx, spec).and_then(|f| col_as_str(ctx.batch, f).ok());

    for i in 0..x_strs.len() {
        let xs = match &x_strs[i] { Some(s) => s.as_str(), None => continue };
        let yv = match ys[i] { Some(v) if v.is_finite() => v, _ => continue };
        let cx = match ctx.scales.x.to_pixel_str(xs) { Some(p) => p, None => continue };
        let top_y = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
        let height = (baseline_y - top_y).max(0.0);
        let r = Rect { x: cx - bar_width / 2.0, y: top_y, w: bar_width, h: height };

        let fill = if let (Some(scale), Some(values)) = (&ctx.scales.color, &color_values) {
            match values[i].as_deref() {
                Some(v) => match scale {
                    ColorScale::Categorical { .. } => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                },
                None => ctx.mark_style.fill,
            }
        } else {
            ctx.mark_style.fill
        };
        let fill = with_opacity(fill, ctx.mark_style.opacity);

        out.rect(r, &FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        }, Some(ctx.mark_style.corner_radius));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, ThemeInputs};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{DataType as SDT, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn bar_emits_four_rects_for_four_categories() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Ordinal) }),
                y: Some(EncodingSpec { field: "v".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a","b","c","d"])),
            Arc::new(Float64Array::from(vec![1.0,2.0,3.0,4.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<rect ").count(), 4);
    }

    #[test]
    fn bar_corner_radius_emitted_when_theme_sets_it() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Ordinal) }),
                y: Some(EncodingSpec { field: "v".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(Float64Array::from(vec![1.0])),
        ]).unwrap();
        let mut theme = ThemeInputs::default();
        theme.bar_corner_radius = 3.0;
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Bar);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert!(out.finish().contains("rx=\"3\""));
    }
}
```

- [ ] **Step 2: Implement `marks/rect.rs`**

```rust
//! mark_rect: heatmap-style. Requires both x and y to be ordinal/temporal-binned
//! with a known band width. Phase 7 supports the simplest case: ordinal x,
//! ordinal y → one rect per (x, y) row spanning that band-cell.

use crate::layout::Rect;
use crate::render::color::with_opacity;
use crate::render::draw::{col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::scale_resolve::{ColorScale, ScaleKind};
use crate::render::svg::{FillStroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return,
    };
    let xs = match col_as_str(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_str(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    if xs.len() != ys.len() { return; }

    let panel = ctx.panel.plot_area;
    let n_x = match &ctx.scales.x { ScaleKind::Ordinal(_) => count_distinct(&xs).max(1), _ => return };
    let n_y = match &ctx.scales.y { ScaleKind::Ordinal(_) => count_distinct(&ys).max(1), _ => return };
    let cell_w = panel.w / n_x as f64;
    let cell_h = panel.h / n_y as f64;

    let color_values = color_field(ctx, spec).and_then(|f| col_as_str(ctx.batch, f).ok());

    for i in 0..xs.len() {
        let xs_v = match &xs[i] { Some(s) => s.as_str(), None => continue };
        let ys_v = match &ys[i] { Some(s) => s.as_str(), None => continue };
        let cx = match ctx.scales.x.to_pixel_str(xs_v) { Some(p) => p, None => continue };
        let cy = match ctx.scales.y.to_pixel_str(ys_v) { Some(p) => p, None => continue };

        let r = Rect { x: cx - cell_w / 2.0, y: cy - cell_h / 2.0, w: cell_w, h: cell_h };
        let fill = if let (Some(scale), Some(values)) = (&ctx.scales.color, &color_values) {
            match values[i].as_deref() {
                Some(v) => match scale {
                    ColorScale::Categorical { .. } => scale.lookup(v).unwrap_or(ctx.mark_style.fill),
                },
                None => ctx.mark_style.fill,
            }
        } else {
            ctx.mark_style.fill
        };
        let fill = with_opacity(fill, ctx.mark_style.opacity);
        out.rect(r, &FillStroke {
            fill: Some(fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        }, Some(ctx.mark_style.corner_radius));
    }
}

fn count_distinct(values: &[Option<String>]) -> usize {
    let mut seen = std::collections::HashSet::<&str>::new();
    for v in values.iter().flatten() { seen.insert(v); }
    seen.len()
}

#[cfg(test)]
mod tests {
    // Minimal smoke: rect mark over 2x2 ordinal grid emits 4 rects.
    use super::*;
    use crate::layout::{PanelLayout, ThemeInputs};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{DataType as SDT, Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn rect_emits_four_cells_for_2x2_ordinal_grid() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "row".into(), type_: Some(SDT::Ordinal) }),
                y: Some(EncodingSpec { field: "col".into(), type_: Some(SDT::Ordinal) }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("row", DataType::Utf8, false),
            Field::new("col", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a","a","b","b"])),
            Arc::new(StringArray::from(vec!["x","y","x","y"])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<rect ").count(), 4);
    }

    #[test]
    fn rect_skips_non_ordinal_axes() {
        // Quantitative x → not supported → no rects emitted.
        use arrow::array::Float64Array;
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rect,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Rect);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        assert!(!out.finish().contains("<rect "));
    }
}
```

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::marks::bar render::marks::rect
```

Expected: 4 tests pass. Cargo total ≥ 241.

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/render/marks/bar.rs crates/ferrum-core/src/render/marks/rect.rs
git commit -m "feat(render): mark_bar + mark_rect

Bar: ordinal x + quantitative y → per-row rect from baseline up to
scale_y. Width = panel.w / n_cats * 0.8. corner_radius from theme.
Rect (heatmap): ordinal x + ordinal y → cell at (band_x, band_y) sized
panel.w/n_x by panel.h/n_y. Skips non-ordinal axes silently."
```

---

### Task 17: `marks/rule.rs` + `marks/tick.rs` + `marks/text.rs`

**Files:**
- Modify: `crates/ferrum-core/src/render/marks/rule.rs`
- Modify: `crates/ferrum-core/src/render/marks/tick.rs`
- Modify: `crates/ferrum-core/src/render/marks/text.rs`

- [ ] **Step 1: Implement `marks/rule.rs`**

```rust
//! mark_rule: horizontal or vertical reference line per row. If only y is
//! encoded → horizontal across panel; if only x encoded → vertical; if both
//! → segment from (x_value, panel.y_min) to (x_value, panel.y_max).
//! Phase 7: supports the y-only and x-only cases (full segment is rare).

use crate::render::draw::{col_as_f64, x_field, y_field, DrawCtx};
use crate::render::svg::{Stroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let panel = ctx.panel.plot_area;
    let style = Stroke {
        stroke: ctx.mark_style.fill,
        stroke_width: ctx.mark_style.stroke_width,
        stroke_dash: ctx.mark_style.stroke_dash.clone(),
    };

    if let (Some(yf), None) = (y_field(ctx, spec), x_field(ctx, spec)) {
        // y-only: horizontal rule per row.
        let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
        for yv in ys.into_iter().flatten() {
            if !yv.is_finite() { continue; }
            let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
            out.line(panel.x, py, panel.x + panel.w, py, &style);
        }
        return;
    }
    if let (Some(xf), None) = (x_field(ctx, spec), y_field(ctx, spec)) {
        let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
        for xv in xs.into_iter().flatten() {
            if !xv.is_finite() { continue; }
            let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
            out.line(px, panel.y, px, panel.y + panel.h, &style);
        }
    }
    // Both x and y encoded: out of Phase 7 mark_rule scope (would need x2/y2).
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn y_only_rule_emits_horizontal_lines() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Rule,
            encoding: Encoding {
                x: None,
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        // Two-row batch but resolve_scales requires both x and y; for this
        // unit test, use two y-values and bypass with a synthetic scales build.
        // Easier: include x for resolve_scales but mark spec only encodes y.
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        // Build scales with x present (so we can construct), then drop x from spec.encoding.
        let mut spec_for_scales = spec.clone();
        spec_for_scales.encoding.x = Some(EncodingSpec { field: "x".into(), type_: None });
        let (scales, _) = resolve_scales(&spec_for_scales, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Rule);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<line ").count(), 2);
    }
}
```

- [ ] **Step 2: Implement `marks/tick.rs`**

```rust
//! mark_tick: rug/tick marks. Short vertical (or horizontal) segments centered
//! at each row's x (or y) coordinate. Phase 7 default: vertical at panel bottom.

use crate::render::draw::{col_as_f64, x_field, y_field, DrawCtx};
use crate::render::svg::{Stroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let panel = ctx.panel.plot_area;
    let tick_len = ctx.theme.tick_size * 2.0;
    let style = Stroke {
        stroke: ctx.mark_style.fill,
        stroke_width: ctx.mark_style.stroke_width.max(1.0),
        stroke_dash: None,
    };

    let xf = match x_field(ctx, spec) { Some(f) => f, None => return };
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    // Phase 7 mark_tick is rug-on-x along the bottom edge of the panel.
    let baseline_y = panel.y + panel.h;
    for xv in xs.into_iter().flatten() {
        if !xv.is_finite() { continue; }
        let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
        out.line(px, baseline_y, px, baseline_y - tick_len, &style);
    }
    // y-rug variant deferred.
    let _ = y_field(ctx, spec);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn tick_emits_one_line_per_row() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Tick,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Tick);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<line ").count(), 3);
    }
}
```

- [ ] **Step 3: Implement `marks/text.rs`**

```rust
//! mark_text: render the value of a designated text column at each (x, y).
//! Phase 7 minimal implementation: assumes y is the field whose value is
//! also the displayed text (label = format(y)). Real `text` channel lands
//! with Phase 8's encoding extension; this stub keeps the mark functional
//! for the done-criteria "renders without panic" requirement.

use crate::layout::TextAnchor;
use crate::render::draw::{col_as_f64, x_field, y_field, DrawCtx};
use crate::render::format::format_numeric;
use crate::render::svg::{SvgBuffer, TextStyle};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b), _ => return,
    };
    let xs = match col_as_f64(ctx.batch, xf) { Ok(v) => v, Err(_) => return };
    let ys = match col_as_f64(ctx.batch, yf) { Ok(v) => v, Err(_) => return };
    if xs.len() != ys.len() { return; }

    let style = TextStyle {
        fill: ctx.theme.font_color,
        font_size: ctx.theme.label_font_size,
        anchor: TextAnchor::Middle,
        angle: 0.0,
    };

    for i in 0..xs.len() {
        let (xv, yv) = match (xs[i], ys[i]) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => (a, b), _ => continue,
        };
        let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) => p, None => continue };
        let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) => p, None => continue };
        let label = format_numeric(yv);
        out.text(px, py, &label, &style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::resolve_mark_style;
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn text_emits_one_text_element_per_row() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Text,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let panel = PanelLayout { plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, facet_key: None, row: 0, col: 0, strip_title: None };
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0)).unwrap();
        let mark_style = resolve_mark_style(&theme, &Mark::Text);
        let ctx = DrawCtx { spec: &spec, panel: &panel, theme: &theme, scales: &scales, batch: &batch, mark_style: &mark_style };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<text ").count(), 2);
    }
}
```

- [ ] **Step 4: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::marks::rule render::marks::tick render::marks::text
```

Expected: 3 tests pass. Cargo total ≥ 244.

- [ ] **Step 5: Commit**

```bash
git add crates/ferrum-core/src/render/marks/
git commit -m "feat(render): mark_rule + mark_tick + mark_text

Rule: horizontal segment per y-only row, vertical per x-only row.
Tick: rug-style short vertical segment at each x along panel bottom.
Text: minimal stub — renders format_numeric(y) at (scale_x(x), scale_y(y)).
A real 'text' channel is Phase 8 grammar surface."
```

---

### Task 18: `marks/axis.rs` — internal axis drawing

**Files:**
- Modify: `crates/ferrum-core/src/render/marks/axis.rs`

- [ ] **Step 1: Implement `marks/axis.rs`**

```rust
//! Internal: draw axis line, ticks, tick labels, and axis title from an AxisLayout.

use crate::layout::{AxisLayout, AxisOrient, TextAnchor, ThemeInputs};
use crate::render::svg::{Stroke, SvgBuffer, TextStyle};

pub fn draw(axis: &AxisLayout, theme: &ThemeInputs, out: &mut SvgBuffer) {
    let line_style = Stroke {
        stroke: theme.axis_line_color,
        stroke_width: theme.axis_line_width,
        stroke_dash: None,
    };
    // Axis line.
    let r = axis.axis_line;
    out.line(r.x, r.y, r.x + r.w, r.y + r.h, &line_style);

    // Ticks.
    let tick_style = Stroke {
        stroke: theme.tick_color,
        stroke_width: theme.axis_line_width,
        stroke_dash: None,
    };
    let label_style_base = TextStyle {
        fill: theme.font_color,
        font_size: theme.label_font_size,
        anchor: TextAnchor::Middle,
        angle: 0.0,
    };
    for tick in &axis.ticks {
        let (tx1, ty1, tx2, ty2, label_x, label_y, anchor, angle) = match axis.orient {
            AxisOrient::Bottom => (
                tick.position, r.y, tick.position, r.y + theme.tick_size,
                tick.position, r.y + theme.tick_size + theme.label_font_size + 2.0,
                TextAnchor::Middle, tick.label_angle,
            ),
            AxisOrient::Top => (
                tick.position, r.y, tick.position, r.y - theme.tick_size,
                tick.position, r.y - theme.tick_size - 4.0,
                TextAnchor::Middle, tick.label_angle,
            ),
            AxisOrient::Left => (
                r.x, tick.position, r.x - theme.tick_size, tick.position,
                r.x - theme.tick_size - 2.0, tick.position + theme.label_font_size / 3.0,
                TextAnchor::End, 0.0,
            ),
            AxisOrient::Right => (
                r.x, tick.position, r.x + theme.tick_size, tick.position,
                r.x + theme.tick_size + 2.0, tick.position + theme.label_font_size / 3.0,
                TextAnchor::Start, 0.0,
            ),
        };
        out.line(tx1, ty1, tx2, ty2, &tick_style);
        let mut style = label_style_base;
        style.anchor = anchor;
        style.angle = angle;
        out.text(label_x, label_y, &tick.label, &style);
    }

    // Title.
    if let Some(t) = &axis.title {
        let title_style = TextStyle {
            fill: theme.font_color,
            font_size: theme.title_font_size,
            anchor: TextAnchor::Middle,
            angle: t.angle,
        };
        out.text(t.anchor.0, t.anchor.1, &t.text, &title_style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{AxisLayout, AxisOrient, AxisTitleLayout, Rect, TickLayout};

    #[test]
    fn axis_draws_line_ticks_and_title() {
        let axis = AxisLayout {
            orient: AxisOrient::Bottom,
            panel_index: 0,
            axis_line: Rect { x: 0.0, y: 80.0, w: 100.0, h: 0.0 },
            ticks: vec![
                TickLayout { position: 25.0, label: "0".into(), label_angle: 0.0, elided: false },
                TickLayout { position: 75.0, label: "1".into(), label_angle: 0.0, elided: false },
            ],
            title: Some(AxisTitleLayout {
                text: "x".into(),
                anchor: (50.0, 95.0),
                angle: 0.0,
            }),
        };
        let theme = ThemeInputs::default();
        let mut out = SvgBuffer::new(Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, None, false);
        super::draw(&axis, &theme, &mut out);
        let s = out.finish();
        assert!(s.contains("<line "));
        assert!(s.matches("<line ").count() >= 3); // 1 axis + 2 ticks
        assert!(s.contains(">x</text>") || s.contains(">x<"));
    }
}
```

- [ ] **Step 2: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::marks::axis
```

Expected: 1 test passes. Cargo total ≥ 245.

- [ ] **Step 3: Commit**

```bash
git add crates/ferrum-core/src/render/marks/axis.rs
git commit -m "feat(render): internal axis drawing from AxisLayout

Bottom/Top/Left/Right orient supported. Tick line + label per
TickLayout; axis title from AxisTitleLayout. Theme reads
axis_line_color, axis_line_width, tick_color, tick_size,
label_font_size, title_font_size, font_color."
```

---

### Task 19: `marks/legend.rs` + `marks/strip_title.rs` — internal

**Files:**
- Modify: `crates/ferrum-core/src/render/marks/legend.rs`
- Modify: `crates/ferrum-core/src/render/marks/strip_title.rs`

- [ ] **Step 1: Implement `marks/legend.rs`**

```rust
//! Internal: draw legend swatches + labels from a LegendLayout.
//! Symbol kind drives the swatch shape (circle / square / line).

use crate::layout::{LegendLayout, SymbolKind, TextAnchor, ThemeInputs};
use crate::render::scale_resolve::ColorScale;
use crate::render::svg::{FillStroke, Stroke, SvgBuffer, TextStyle};

pub fn draw(
    legend: &LegendLayout,
    color_scale: Option<&ColorScale>,
    theme: &ThemeInputs,
    out: &mut SvgBuffer,
) {
    let label_style = TextStyle {
        fill: theme.font_color,
        font_size: theme.label_font_size,
        anchor: TextAnchor::Start,
        angle: 0.0,
    };

    for entry in &legend.entries {
        let color = color_scale
            .and_then(|s| match s {
                ColorScale::Categorical { .. } => s.lookup(&entry.label),
            })
            .unwrap_or(theme.mark_color);
        let (sx, sy) = entry.symbol_anchor;
        match entry.symbol_kind {
            SymbolKind::Circle => out.circle(sx, sy, 4.0, &FillStroke {
                fill: Some(color), stroke: None, stroke_width: 0.0,
            }),
            SymbolKind::Square => out.rect(
                crate::layout::Rect { x: sx - 4.0, y: sy - 4.0, w: 8.0, h: 8.0 },
                &FillStroke { fill: Some(color), stroke: None, stroke_width: 0.0 },
                None,
            ),
            SymbolKind::Line => out.line(sx - 6.0, sy, sx + 6.0, sy, &Stroke {
                stroke: color, stroke_width: theme.line_stroke_width, stroke_dash: None,
            }),
        }
        let (lx, ly) = entry.label_anchor;
        out.text(lx, ly, &entry.label, &label_style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{LegendDirection, LegendEntryLayout, LegendOrient, Rect};

    #[test]
    fn legend_emits_circle_swatch_per_entry() {
        let legend = LegendLayout {
            rect: Rect { x: 80.0, y: 0.0, w: 20.0, h: 100.0 },
            orient: LegendOrient::Right,
            direction: LegendDirection::Vertical,
            entries: vec![
                LegendEntryLayout { label: "a".into(), label_anchor: (88.0, 10.0), symbol_anchor: (84.0, 10.0), symbol_kind: SymbolKind::Circle },
                LegendEntryLayout { label: "b".into(), label_anchor: (88.0, 24.0), symbol_anchor: (84.0, 24.0), symbol_kind: SymbolKind::Circle },
            ],
        };
        let theme = ThemeInputs::default();
        let mut out = SvgBuffer::new(Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, None, false);
        super::draw(&legend, None, &theme, &mut out);
        let s = out.finish();
        assert_eq!(s.matches("<circle ").count(), 2);
        assert_eq!(s.matches("<text ").count(), 2);
    }
}
```

- [ ] **Step 2: Implement `marks/strip_title.rs`**

```rust
//! Internal: draw a per-panel strip-title band (background rect + centered text).

use crate::layout::{Rect, StripTitleLayout, ThemeInputs};
use crate::render::svg::{FillStroke, SvgBuffer, TextStyle};

pub fn draw(
    strip: &StripTitleLayout,
    panel_rect: &Rect,
    theme: &ThemeInputs,
    out: &mut SvgBuffer,
) {
    // Reserved band sits directly above panel_rect; height inferred from
    // anchor.y vs panel_rect.y.
    let band_h = (panel_rect.y - (strip.anchor.1 - strip.font_size - theme.strip_padding))
        .abs()
        .max(strip.font_size + 2.0 * theme.strip_padding);
    let band = Rect {
        x: panel_rect.x,
        y: panel_rect.y - band_h,
        w: panel_rect.w,
        h: band_h,
    };
    out.rect(band, &FillStroke {
        fill: Some(theme.strip_background_color),
        stroke: None,
        stroke_width: 0.0,
    }, None);
    out.text(strip.anchor.0, strip.anchor.1, &strip.text, &TextStyle {
        fill: theme.font_color,
        font_size: strip.font_size,
        anchor: strip.align,
        angle: 0.0,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::TextAnchor;

    #[test]
    fn strip_title_emits_background_and_text() {
        let strip = StripTitleLayout {
            text: "setosa".into(),
            anchor: (50.0, 18.0),
            align: TextAnchor::Middle,
            font_size: 13.0,
        };
        let panel = Rect { x: 0.0, y: 22.0, w: 100.0, h: 78.0 };
        let theme = ThemeInputs::default();
        let mut out = SvgBuffer::new(Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 }, None, false);
        super::draw(&strip, &panel, &theme, &mut out);
        let s = out.finish();
        assert!(s.contains("<rect "), "expected strip background rect");
        assert!(s.contains(">setosa</text>") || s.contains(">setosa<"));
    }
}
```

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::marks::legend render::marks::strip_title
```

Expected: 2 tests pass. Cargo total ≥ 247.

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/render/marks/legend.rs crates/ferrum-core/src/render/marks/strip_title.rs
git commit -m "feat(render): internal legend + strip_title draw fns

Legend: per-entry swatch (circle / square / line) + label.
Looks up color from CategoricalColorScale by entry label, falling back
to theme.mark_color. Strip title: band rect + centered text per panel."
```

---

### Task 20: `render_svg` orchestration in `mod.rs`

**Files:**
- Modify: `crates/ferrum-core/src/render/mod.rs`

- [ ] **Step 1: Implement the `render_svg` orchestration function**

Append to `crates/ferrum-core/src/render/mod.rs`:

```rust
use crate::layout::{compute_layout, ThemeInputs, Viewport};
use crate::spec::chart::ChartSpec;
use arrow::record_batch::RecordBatch;

pub fn render_svg(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &config::RenderConfig,
) -> Result<RenderOutput<String>, RenderError> {
    // 1. Validate.
    if viewport.width <= 0.0 || viewport.height <= 0.0 {
        return Err(RenderError::InvalidViewport { width: viewport.width, height: viewport.height });
    }

    // 2. Apply RenderConfig overrides.
    let viewport = Viewport {
        width: config.width.unwrap_or(viewport.width),
        height: config.height.unwrap_or(viewport.height),
    };
    let background = config.background.or(Some(theme.background_color));

    // 3. prepare_render_inputs.
    let prep = prepare::prepare_render_inputs(spec, batch)?;
    let mut warnings = prep.warnings.clone();

    // 4. compute_layout with FontdueMetrics.
    let metrics = font::FontdueMetrics::new();
    let layout = compute_layout(
        spec,
        theme,
        viewport,
        &prep.axes,
        &prep.facet_groups,
        &prep.legend_entries,
        &metrics,
    ).map_err(|e| RenderError::LayoutFailed(e.to_string()))?;
    for w in &layout.warnings {
        warnings.push(RenderWarning::Layout(w.clone()));
    }

    // 5. Initialize SvgBuffer.
    let mut out = svg::SvgBuffer::new(layout.viewport, background, true);

    // 6. Per-panel draw.
    for (panel_idx, panel) in layout.panels.iter().enumerate() {
        if panel.plot_area.w <= 0.0 || panel.plot_area.h <= 0.0 {
            warnings.push(RenderWarning::EmptyPanel { panel_index: panel_idx });
            continue;
        }

        // Find this panel's two axes (bottom + left, by panel_index).
        for axis in layout.axes.iter().filter(|a| a.panel_index == panel_idx) {
            marks::axis::draw(axis, theme, &mut out);
        }

        // Strip title (faceted only).
        if let Some(strip) = &panel.strip_title {
            marks::strip_title::draw(strip, &panel.plot_area, theme, &mut out);
        }

        // Per-panel data slice.
        let panel_batch = if let Some(key) = &panel.facet_key {
            filter_batch_by_facet(&prep.transformed, &key.field, &key.value)?
        } else {
            prep.transformed.clone()
        };
        if panel_batch.num_rows() == 0 { continue; }

        // Per-panel scales (with the panel's actual pixel range).
        let (scales, scale_warnings) = scale_resolve::resolve_scales(
            spec,
            &panel_batch,
            (panel.plot_area.x, panel.plot_area.x + panel.plot_area.w),
            (panel.plot_area.y, panel.plot_area.y + panel.plot_area.h),
        )?;
        warnings.extend(scale_warnings);

        // Clip to panel.plot_area while drawing marks.
        let clip_id = format!("{}{}", CLIP_ID_PREFIX, panel_idx);
        out.clip_open(&clip_id, panel.plot_area);
        out.use_clip_open(&clip_id);

        let mark_style = draw::resolve_mark_style(theme, &spec.mark);
        let ctx = draw::DrawCtx {
            spec, panel, theme, scales: &scales, batch: &panel_batch, mark_style: &mark_style,
        };
        draw::dispatch_mark(&spec.mark, &ctx, &mut out);

        out.use_clip_close();
    }

    // 7. Legend (single, panel-independent — drawn last).
    if let Some(legend) = &layout.legend {
        let color_scale = if let Some(_) = spec.encoding.color {
            // Rebuild a global color scale from the full transformed batch.
            let (gs, _) = scale_resolve::resolve_scales(spec, &prep.transformed, (0.0, 1.0), (0.0, 1.0))?;
            gs.color
        } else {
            None
        };
        marks::legend::draw(legend, color_scale.as_ref(), theme, &mut out);
    }

    let svg_string = out.finish();
    Ok(RenderOutput { bytes: svg_string, layout, warnings })
}

fn filter_batch_by_facet(
    batch: &RecordBatch,
    field: &str,
    value: &str,
) -> Result<RecordBatch, RenderError> {
    use arrow::array::{Array, BooleanArray, StringArray};
    use arrow::compute::filter_record_batch;
    let col = batch.column_by_name(field)
        .ok_or_else(|| RenderError::UnknownColumn { name: field.to_string() })?;
    let arr = col.as_any().downcast_ref::<StringArray>()
        .ok_or_else(|| RenderError::ScaleResolutionFailed(
            format!("facet field '{field}' must be Utf8")
        ))?;
    let mask: BooleanArray = arr.iter()
        .map(|v| Some(v.map(|s| s == value).unwrap_or(false)))
        .collect();
    filter_record_batch(batch, &mask)
        .map_err(|e| RenderError::ScaleResolutionFailed(format!("filter: {e}")))
}

#[cfg(test)]
mod orchestration_tests {
    use super::*;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn scatter_3() -> (ChartSpec, RecordBatch) {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ]).unwrap();
        (spec, batch)
    }

    #[test]
    fn render_svg_minimal_scatter() {
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let result = render_svg(&spec, &batch, &theme, viewport, &config).unwrap();
        let svg = result.bytes;
        assert!(svg.starts_with("<svg "));
        assert!(svg.ends_with("</svg>"));
        assert_eq!(svg.matches("<circle ").count(), 3);
        assert!(svg.contains("@font-face")); // embed_fonts on
    }

    #[test]
    fn render_svg_invalid_viewport_errors() {
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let result = render_svg(
            &spec, &batch, &theme,
            Viewport { width: 0.0, height: 100.0 },
            &config::RenderConfig::default(),
        );
        assert!(matches!(result.unwrap_err(), RenderError::InvalidViewport { .. }));
    }

    #[test]
    fn render_svg_unknown_column_errors() {
        let (mut spec, batch) = scatter_3();
        spec.encoding.x = Some(EncodingSpec { field: "missing".into(), type_: None });
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
        );
        assert!(matches!(result.unwrap_err(), RenderError::UnknownColumn { .. }));
    }

    #[test]
    fn render_svg_faceted_emits_strip_titles() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("species", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
            Arc::new(StringArray::from(vec!["a", "b", "a", "c", "b", "c"])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: Some(EncodingSpec { field: "species".into(), type_: None }),
            },
            transforms: Vec::new(),
            facet: Some(crate::layout::FacetSpec {
                field: "species".into(),
                mode: crate::layout::FacetMode::Wrap { ncols: 3 },
                spacing: None,
            }),
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 800.0, height: 400.0 },
            &config::RenderConfig::default(),
        ).unwrap();
        let svg = result.bytes;
        // 3 facets → 3 strip-title backgrounds (one rect each) + 3 strip-title <text>s.
        assert!(svg.contains(">a<") || svg.contains(">a</text>"));
        assert!(svg.contains(">b<") || svg.contains(">b</text>"));
        assert!(svg.contains(">c<") || svg.contains(">c</text>"));
    }

    #[test]
    fn render_svg_determinism_two_calls_byte_identical() {
        let (spec, batch) = scatter_3();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 600.0, height: 400.0 };
        let config = config::RenderConfig::default();
        let a = render_svg(&spec, &batch, &theme, viewport, &config).unwrap();
        let b = render_svg(&spec, &batch, &theme, viewport, &config).unwrap();
        assert_eq!(a.bytes, b.bytes);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::orchestration_tests
```

Expected: 5 tests pass. Cargo total ≥ 252.

- [ ] **Step 3: Commit**

```bash
git add crates/ferrum-core/src/render/mod.rs
git commit -m "feat(render): render_svg full pipeline orchestration

Spec §6 algorithm. Validates viewport → applies config overrides →
prepare_render_inputs → compute_layout (with FontdueMetrics) →
per-panel: axis draw + strip_title + facet-filter + scale rebuild +
clipped mark dispatch → legend draw → finish. Determinism verified
via two-call byte-identical assertion."
```

---

### Task 21: `png.rs` + `render_png`

**Files:**
- Modify: `crates/ferrum-core/src/render/png.rs`
- Modify: `crates/ferrum-core/src/render/mod.rs`

- [ ] **Step 1: Implement `png.rs`**

Replace `crates/ferrum-core/src/render/png.rs` with:

```rust
//! SVG → PNG via usvg + resvg + tiny_skia.

use super::font::INTER_REGULAR;
use super::RenderError;

pub fn svg_string_to_png_bytes(
    svg: &str,
    width_px: u32,
    height_px: u32,
    scale: f64,
) -> Result<Vec<u8>, RenderError> {
    let mut opts = usvg::Options::default();
    opts.fontdb_mut().load_font_data(INTER_REGULAR.to_vec());

    let tree = usvg::Tree::from_str(svg, &opts)
        .map_err(|e| RenderError::ResvgFailed(format!("usvg parse: {e}")))?;

    let mut pixmap = tiny_skia::Pixmap::new(width_px, height_px)
        .ok_or_else(|| RenderError::ResvgFailed("pixmap allocation".into()))?;
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale as f32, scale as f32),
        &mut pixmap.as_mut(),
    );
    pixmap.encode_png()
        .map_err(|e| RenderError::ResvgFailed(format!("encode_png: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_svg_rasterizes_to_png() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20"><rect x="0" y="0" width="20" height="20" fill="#ff0000"/></svg>"#;
        let png = svg_string_to_png_bytes(svg, 40, 40, 2.0).unwrap();
        // PNG magic: 89 50 4e 47 0d 0a 1a 0a
        assert_eq!(&png[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    }
}
```

- [ ] **Step 2: Implement `render_png` in `mod.rs`**

Append to `crates/ferrum-core/src/render/mod.rs`:

```rust
pub fn render_png(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &config::RenderConfig,
) -> Result<RenderOutput<Vec<u8>>, RenderError> {
    let svg_out = render_svg(spec, batch, theme, viewport, config)?;
    let w = (svg_out.layout.viewport.w * config.scale).round() as u32;
    let h = (svg_out.layout.viewport.h * config.scale).round() as u32;
    let bytes = png::svg_string_to_png_bytes(&svg_out.bytes, w, h, config.scale)?;
    Ok(RenderOutput { bytes, layout: svg_out.layout, warnings: svg_out.warnings })
}

#[cfg(test)]
mod png_tests {
    use super::*;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn render_png_produces_png_magic_bytes() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ]).unwrap();
        let result = render_png(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 100.0, height: 80.0 },
            &config::RenderConfig::default(),
        ).unwrap();
        assert_eq!(&result.bytes[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    }

    #[test]
    fn render_png_determinism_two_calls_byte_identical() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0])),
        ]).unwrap();
        let theme = ThemeInputs::default();
        let viewport = Viewport { width: 100.0, height: 80.0 };
        let config = config::RenderConfig::default();
        let a = render_png(&spec, &batch, &theme, viewport, &config).unwrap();
        let b = render_png(&spec, &batch, &theme, viewport, &config).unwrap();
        assert_eq!(a.bytes, b.bytes);
    }
}
```

- [ ] **Step 3: Run tests**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::png render::png_tests
```

Expected: 3 tests pass. Cargo total ≥ 255.

- [ ] **Step 4: Commit**

```bash
git add crates/ferrum-core/src/render/png.rs crates/ferrum-core/src/render/mod.rs
git commit -m "feat(render): render_png via usvg + resvg + tiny_skia

usvg loads bundled Inter into its fontdb so resvg always finds the
Phase 7 family. Pixmap dimensions = layout.viewport * config.scale.
Determinism verified via two-call byte-identical PNG."
```

---

### Task 22: PyO3 binding — `render_svg`, `render_png`

**Files:**
- Modify: `crates/ferrum-core/src/render/binding.rs`
- Modify: `crates/ferrum-core/src/lib.rs`
- Modify: `src/ferrum/__init__.py`
- Modify: `src/ferrum/_core.pyi`

- [ ] **Step 1: Implement `binding.rs`**

Replace `crates/ferrum-core/src/render/binding.rs` with:

```rust
//! PyO3 bindings: render_svg, render_png. Theme/RenderConfig pass via Python dicts.

use arrow::record_batch::RecordBatchReader;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pyo3_arrow::PyRecordBatchReader;

use crate::layout::{ThemeInputs, Viewport};
use crate::spec::chart::ChartSpec;

use super::{render_png as render_png_internal, render_svg as render_svg_internal};
use super::config::RenderConfig;
use super::RenderError;

#[pyfunction]
#[pyo3(signature = (spec, data, *, viewport, theme = None, config = None))]
pub fn render_svg(
    py: Python<'_>,
    spec: &ChartSpec,
    data: PyRecordBatchReader,
    viewport: (f64, f64),
    theme: Option<&Bound<'_, PyDict>>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let batch = collect_single_batch(data)?;
    let t = theme_from_dict(theme)?;
    let c = config_from_dict(config)?;
    let vp = Viewport { width: viewport.0, height: viewport.1 };
    let result = render_svg_internal(spec, &batch, &t, vp, &c).map_err(render_err_to_py)?;
    emit_warnings(py, &result.warnings)?;
    Ok(result.bytes)
}

#[pyfunction]
#[pyo3(signature = (spec, data, *, viewport, theme = None, config = None))]
pub fn render_png<'py>(
    py: Python<'py>,
    spec: &ChartSpec,
    data: PyRecordBatchReader,
    viewport: (f64, f64),
    theme: Option<&Bound<'_, PyDict>>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let batch = collect_single_batch(data)?;
    let t = theme_from_dict(theme)?;
    let c = config_from_dict(config)?;
    let vp = Viewport { width: viewport.0, height: viewport.1 };
    let result = render_png_internal(spec, &batch, &t, vp, &c).map_err(render_err_to_py)?;
    emit_warnings(py, &result.warnings)?;
    Ok(PyBytes::new(py, &result.bytes))
}

fn collect_single_batch(reader: PyRecordBatchReader) -> PyResult<arrow::record_batch::RecordBatch> {
    let mut iter = reader.into_reader()
        .map_err(|e| PyValueError::new_err(format!("arrow reader: {e}")))?;
    let first = iter.next()
        .ok_or_else(|| PyValueError::new_err("empty record batch stream"))?
        .map_err(|e| PyValueError::new_err(format!("arrow read: {e}")))?;
    // Phase 7 supports single-batch streams. Subsequent batches concatenate.
    let mut all = vec![first];
    for next in iter {
        all.push(next.map_err(|e| PyValueError::new_err(format!("arrow read: {e}")))?);
    }
    if all.len() == 1 {
        Ok(all.into_iter().next().unwrap())
    } else {
        let schema = all[0].schema();
        arrow::compute::concat_batches(&schema, &all)
            .map_err(|e| PyValueError::new_err(format!("concat batches: {e}")))
    }
}

fn theme_from_dict(d: Option<&Bound<'_, PyDict>>) -> PyResult<ThemeInputs> {
    let mut t = ThemeInputs::default();
    let d = match d { Some(x) => x, None => return Ok(t) };
    if let Some(v) = d.get_item("mark_color")? {
        let s: String = v.extract()?;
        t.mark_color = super::color::from_hex_str(&s)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("background_color")? {
        let s: String = v.extract()?;
        t.background_color = super::color::from_hex_str(&s)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if let Some(v) = d.get_item("point_size")? { t.point_size = v.extract()?; }
    if let Some(v) = d.get_item("line_stroke_width")? { t.line_stroke_width = v.extract()?; }
    if let Some(v) = d.get_item("bar_corner_radius")? { t.bar_corner_radius = v.extract()?; }
    if let Some(v) = d.get_item("area_opacity")? { t.area_opacity = v.extract()?; }
    if let Some(v) = d.get_item("grid")? { t.grid = v.extract()?; }
    if let Some(v) = d.get_item("padding")? { t.padding = v.extract()?; }
    // Additional fields can be added as Phase 8 grammar requires.
    Ok(t)
}

fn config_from_dict(d: Option<&Bound<'_, PyDict>>) -> PyResult<RenderConfig> {
    let mut c = RenderConfig::default();
    let d = match d { Some(x) => x, None => return Ok(c) };
    if let Some(v) = d.get_item("scale")? { c.scale = v.extract()?; }
    if let Some(v) = d.get_item("embed_fonts")? { c.embed_fonts = v.extract()?; }
    if let Some(v) = d.get_item("background")? {
        let s: String = v.extract()?;
        c.background = Some(super::color::from_hex_str(&s)
            .map_err(|e| PyValueError::new_err(e.to_string()))?);
    }
    if let Some(v) = d.get_item("width")? { c.width = Some(v.extract()?); }
    if let Some(v) = d.get_item("height")? { c.height = Some(v.extract()?); }
    Ok(c)
}

fn render_err_to_py(e: RenderError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn emit_warnings(py: Python<'_>, warnings: &[super::RenderWarning]) -> PyResult<()> {
    if warnings.is_empty() { return Ok(()); }
    let warnings_mod = py.import("warnings")?;
    for w in warnings {
        let msg = format!("{w:?}");
        warnings_mod.call_method1("warn", (msg,))?;
    }
    Ok(())
}
```

- [ ] **Step 2: Register pyfunctions in `lib.rs`**

Edit `crates/ferrum-core/src/lib.rs`. After the existing `m.add_function(wrap_pyfunction!(layout::binding::compute_layout, m)?)?;` line, add:

```rust
m.add_function(wrap_pyfunction!(render::binding::render_svg, m)?)?;
m.add_function(wrap_pyfunction!(render::binding::render_png, m)?)?;
```

- [ ] **Step 3: Update `src/ferrum/__init__.py`**

Add `render_svg` and `render_png` to the imports and `__all__`:

```python
from ferrum._core import (
    Aggregate,
    AggregateOp,
    Bin,
    ChartSpec,
    EncodingSpec,
    Kde,
    LinearScale,
    LogScale,
    TimeScale,
    SymlogScale,
    OrdinalScale,
    QuantileScale,
    Smooth,
    Summary,
    ThresholdScale,
    compute_layout,
    process_batch,
    render_svg,
    render_png,
)

__version__ = "0.1.0"

__all__ = [
    "Aggregate", "AggregateOp", "Bin", "ChartSpec", "EncodingSpec",
    "Kde", "LinearScale", "LogScale", "TimeScale", "SymlogScale",
    "OrdinalScale", "QuantileScale", "Smooth", "Summary",
    "ThresholdScale", "compute_layout", "process_batch",
    "render_svg", "render_png",
]
```

- [ ] **Step 4: Update `_core.pyi`**

Append to `src/ferrum/_core.pyi`:

```python
def render_svg(
    spec: ChartSpec,
    data: Any,
    *,
    viewport: tuple[float, float],
    theme: Optional[dict] = None,
    config: Optional[dict] = None,
) -> str: ...

def render_png(
    spec: ChartSpec,
    data: Any,
    *,
    viewport: tuple[float, float],
    theme: Optional[dict] = None,
    config: Optional[dict] = None,
) -> bytes: ...
```

- [ ] **Step 5: Build**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
```

Expected: build succeeds.

- [ ] **Step 6: Smoke-test the binding**

```bash
uv run python -c "
import polars as pl
import ferrum as fr
df = pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [10.0, 20.0, 30.0]})
spec = fr.ChartSpec(mark='point', x='x', y='y')
svg = fr.render_svg(spec, df, viewport=(600.0, 400.0))
assert svg.startswith('<svg ')
assert svg.count('<circle ') == 3
print('OK svg', len(svg), 'bytes')
png = fr.render_png(spec, df, viewport=(600.0, 400.0))
assert png[:8] == b'\\x89PNG\\r\\n\\x1a\\n'
print('OK png', len(png), 'bytes')
"
```

Expected: prints `OK svg <n> bytes` and `OK png <n> bytes`.

- [ ] **Step 7: Commit**

```bash
git add crates/ferrum-core/src/render/binding.rs crates/ferrum-core/src/lib.rs \
        src/ferrum/__init__.py src/ferrum/_core.pyi
git commit -m "feat(render): PyO3 binding — render_svg + render_png Python entry points

Spec §13 / locked decision §11 row 13. Two functions, clear typed
returns (str vs bytes). Theme via Python dict (subset of fields wired
in Phase 7; Phase 8 grammar will surface a proper Theme value class).
RenderConfig via dict. Warnings re-emitted through Python warnings.warn
at the binding boundary."
```

---

### Task 23: Pytest tests for the render binding

**Files:**
- Create: `tests/test_render.py`

- [ ] **Step 1: Write the test file**

Create `tests/test_render.py`:

```python
import polars as pl
import pyarrow as pa
import pytest

import ferrum as fr


def _df_3():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})


def _df_color_3cat():
    return pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "species": ["a", "b", "c", "a", "b", "c"],
    })


def test_render_svg_minimal():
    spec = fr.ChartSpec(mark="point", x="x", y="y")
    svg = fr.render_svg(spec, _df_3(), viewport=(600.0, 400.0))
    assert isinstance(svg, str)
    assert svg.startswith("<svg ")
    assert svg.count("<circle ") == 3


def test_render_png_minimal():
    spec = fr.ChartSpec(mark="point", x="x", y="y")
    png = fr.render_png(spec, _df_3(), viewport=(600.0, 400.0))
    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_svg_color_legend():
    spec = fr.ChartSpec(mark="point", x="x", y="y", color="species")
    svg = fr.render_svg(spec, _df_color_3cat(), viewport=(600.0, 400.0))
    assert "<circle " in svg
    # Legend label texts a/b/c should appear.
    for label in ("a", "b", "c"):
        assert f">{label}" in svg


def test_render_svg_theme_dict_applied():
    spec = fr.ChartSpec(mark="point", x="x", y="y")
    svg = fr.render_svg(spec, _df_3(), viewport=(600.0, 400.0), theme={"mark_color": "#ff0000"})
    assert "#ff0000" in svg


def test_render_svg_invalid_viewport_raises():
    spec = fr.ChartSpec(mark="point", x="x", y="y")
    with pytest.raises(ValueError):
        fr.render_svg(spec, _df_3(), viewport=(0.0, 400.0))


def test_render_svg_invalid_color_raises():
    spec = fr.ChartSpec(mark="point", x="x", y="y")
    with pytest.raises(ValueError):
        fr.render_svg(spec, _df_3(), viewport=(600.0, 400.0), theme={"mark_color": "not-a-color"})


def test_render_svg_unknown_column_raises():
    spec = fr.ChartSpec(mark="point", x="missing", y="y")
    with pytest.raises(ValueError):
        fr.render_svg(spec, _df_3(), viewport=(600.0, 400.0))


def test_render_svg_empty_dataframe_raises():
    spec = fr.ChartSpec(mark="point", x="x", y="y")
    empty = pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Float64, "y": pl.Float64})
    with pytest.raises(ValueError):
        fr.render_svg(spec, empty, viewport=(600.0, 400.0))


def test_render_svg_pyarrow_table_works():
    spec = fr.ChartSpec(mark="point", x="x", y="y")
    table = pa.table({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    svg = fr.render_svg(spec, table, viewport=(600.0, 400.0))
    assert svg.count("<circle ") == 3


def test_render_svg_render_config_kwargs_accepted():
    spec = fr.ChartSpec(mark="point", x="x", y="y")
    svg = fr.render_svg(
        spec, _df_3(), viewport=(600.0, 400.0),
        config={"scale": 2.0, "embed_fonts": True, "background": "#000000"},
    )
    assert "<svg " in svg
    # Background = #000000 should appear as the first <rect fill=...>.
    assert "#000000" in svg


def test_render_svg_faceted_emits_three_strip_titles():
    spec_dict_test = fr.ChartSpec.from_json(
        '{"data":{},"mark":"point","encoding":{"x":{"field":"x"},"y":{"field":"y"},"color":{"field":"species"}},"transforms":[],"facet":{"field":"species","mode":{"Wrap":{"ncols":3}}}}'
    )
    svg = fr.render_svg(spec_dict_test, _df_color_3cat(), viewport=(800.0, 400.0))
    # Each facet strip-title is a <text> with the category value.
    for cat in ("a", "b", "c"):
        assert f">{cat}" in svg
```

> **Implementer note on `test_render_svg_faceted_emits_three_strip_titles`:** the `from_json` call assumes `FacetSpec` serializes as `{"Wrap":{"ncols":3}}`. If Phase 6's actual JSON shape differs (it does — sealed-tagged enums in serde use `{"kind":"...","..."}` by default), pull the spec from a Python constructor instead. Phase 8 will add a Python `facet=...` kwarg; for now, you can either:
>
> (a) skip this one test and replace with a structural assertion that calls `render_svg` on a non-faceted spec and just confirms the round-trip parses, or
> (b) construct a faceted ChartSpec by JSON crafted to match the Phase 6 serde format. Read `crates/ferrum-core/src/layout/facet.rs` to see the exact JSON shape.
>
> Pick whichever works. The pytest count target (≥ 88) accommodates either path.

- [ ] **Step 2: Run pytest**

```bash
uv run pytest tests/test_render.py -v
```

Expected: 11 tests pass (or 10 if the faceted strip-title test is skipped per the implementer note). Pytest total ≥ 88.

- [ ] **Step 3: Commit**

```bash
git add tests/test_render.py
git commit -m "test(render): pytest binding tests for render_svg + render_png

Smoke tests, theme dict, RenderConfig dict, error mapping
(invalid viewport / color / column / empty df), pyarrow Table input,
faceted strip-title emission. ≥ 88 pytest total."
```

---

### Task 24: Golden file generation + comparison harness

**Files:**
- Create: `crates/ferrum-core/tests/render_goldens.rs` (Cargo integration test, but Phase 7 uses unit tests; alternatively put inside `crates/ferrum-core/src/render/mod.rs` as `#[cfg(test)] mod golden_tests {}`)
- Create: `crates/ferrum-core/tests/golden/scatter_minimal.svg` (generated)
- Create: `crates/ferrum-core/tests/golden/scatter_color.svg` (generated)
- Create: `crates/ferrum-core/tests/golden/bar_grouped.svg` (generated)
- Create: `crates/ferrum-core/tests/golden/line_simple.svg` (generated)
- Create: `crates/ferrum-core/tests/golden/area_filled.svg` (generated)
- Create: `crates/ferrum-core/tests/golden/faceted_scatter.svg` (generated)
- Create: `crates/ferrum-core/tests/golden/scatter_minimal.png.sha256` (generated)

> **Cargo test harness note:** The repo currently has no `tests/` integration directory under `crates/ferrum-core/` (the crate is `crate-type = ["cdylib"]` and tests live inline). Add the golden tests as `#[cfg(test)] mod golden_tests {}` inside `crates/ferrum-core/src/render/mod.rs`, with the golden files at `crates/ferrum-core/tests/golden/`. Reading those files in tests via `include_str!("../tests/golden/scatter_minimal.svg")` is the simplest path since the path is relative to the source file.

- [ ] **Step 1: Add the golden test harness**

Append to `crates/ferrum-core/src/render/mod.rs`:

```rust
#[cfg(test)]
mod golden_tests {
    //! End-to-end goldens. Refresh via `FERRUM_UPDATE_GOLDENS=1 cargo test`.
    //! See spec §9.4 for refresh discipline.

    use super::*;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn check_golden(name: &str, svg: &str) {
        let path = format!("tests/golden/{name}.svg");
        let abs_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(&path);
        if std::env::var("FERRUM_UPDATE_GOLDENS").is_ok() {
            std::fs::create_dir_all(abs_path.parent().unwrap()).unwrap();
            std::fs::write(&abs_path, svg).expect("write golden");
            return;
        }
        let expected = std::fs::read_to_string(&abs_path)
            .unwrap_or_else(|e| panic!("read golden {path}: {e} — run FERRUM_UPDATE_GOLDENS=1 to create"));
        assert_eq!(svg, expected, "golden mismatch for {name} — run FERRUM_UPDATE_GOLDENS=1 to refresh");
    }

    fn check_png_hash(name: &str, png: &[u8]) {
        use std::io::Write;
        let path = format!("tests/golden/{name}.sha256");
        let abs_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(&path);
        let mut hasher = sha2::Sha256::new();
        sha2::Digest::update(&mut hasher, png);
        let hash = format!("{:x}", sha2::Digest::finalize(hasher));
        if std::env::var("FERRUM_UPDATE_GOLDENS").is_ok() {
            std::fs::create_dir_all(abs_path.parent().unwrap()).unwrap();
            let mut f = std::fs::File::create(&abs_path).unwrap();
            f.write_all(hash.as_bytes()).unwrap();
            return;
        }
        let expected = std::fs::read_to_string(&abs_path)
            .unwrap_or_else(|e| panic!("read png hash {path}: {e}"));
        assert_eq!(hash.trim(), expected.trim(), "PNG hash mismatch for {name}");
    }

    #[test]
    fn scatter_minimal_golden() {
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ]).unwrap();
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
        ).unwrap();
        check_golden("scatter_minimal", &result.bytes);

        let png_result = render_png(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
        ).unwrap();
        check_png_hash("scatter_minimal.png", &png_result.bytes);
    }

    #[test]
    fn scatter_color_golden() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
            Arc::new(StringArray::from(vec!["a","b","c","a","b","c"])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: Some(EncodingSpec { field: "g".into(), type_: None }),
            },
            transforms: Vec::new(), facet: None,
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
        ).unwrap();
        check_golden("scatter_color", &result.bytes);
    }

    #[test]
    fn bar_grouped_golden() {
        use crate::spec::encoding::DataType as SDT;
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a","b","c","d"])),
            Arc::new(Float64Array::from(vec![3.0, 1.0, 4.0, 1.5])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Bar,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "g".into(), type_: Some(SDT::Ordinal) }),
                y: Some(EncodingSpec { field: "v".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
        ).unwrap();
        check_golden("bar_grouped", &result.bytes);
    }

    #[test]
    fn line_simple_golden() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0, 30.0, 80.0, 60.0])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Line,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
        ).unwrap();
        check_golden("line_simple", &result.bytes);
    }

    #[test]
    fn area_filled_golden() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            Arc::new(Float64Array::from(vec![10.0, 50.0, 30.0, 80.0, 60.0])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Area,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: None,
            },
            transforms: Vec::new(), facet: None,
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 600.0, height: 400.0 },
            &config::RenderConfig::default(),
        ).unwrap();
        check_golden("area_filled", &result.bytes);
    }

    #[test]
    fn faceted_scatter_golden() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("species", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 12.0, 22.0, 32.0])),
            Arc::new(StringArray::from(vec!["setosa","setosa","setosa","versicolor","versicolor","versicolor","virginica","virginica","virginica"])),
        ]).unwrap();
        let spec = ChartSpec {
            data: DataRef::default(), mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None }),
                y: Some(EncodingSpec { field: "y".into(), type_: None }),
                color: Some(EncodingSpec { field: "species".into(), type_: None }),
            },
            transforms: Vec::new(),
            facet: Some(crate::layout::FacetSpec {
                field: "species".into(),
                mode: crate::layout::FacetMode::Wrap { ncols: 3 },
                spacing: None,
            }),
        };
        let result = render_svg(
            &spec, &batch, &ThemeInputs::default(),
            Viewport { width: 800.0, height: 400.0 },
            &config::RenderConfig::default(),
        ).unwrap();
        check_golden("faceted_scatter", &result.bytes);
    }
}
```

- [ ] **Step 2: Add `sha2` to dev-deps**

Edit `crates/ferrum-core/Cargo.toml`. Add `[dev-dependencies]` if absent:

```toml
[dev-dependencies]
sha2 = "0.10"
```

- [ ] **Step 3: Generate goldens (first run)**

```bash
mkdir -p crates/ferrum-core/tests/golden
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") FERRUM_UPDATE_GOLDENS=1 cargo test -p ferrum-core render::golden_tests
ls -la crates/ferrum-core/tests/golden/
```

Expected: 6 `.svg` files + 1 `.sha256` file present.

- [ ] **Step 4: Run goldens in compare mode (no env var)**

```bash
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core render::golden_tests
```

Expected: 6 tests pass (each compares the freshly-rendered SVG/PNG against the just-generated golden).

- [ ] **Step 5: Commit**

```bash
git add crates/ferrum-core/Cargo.toml \
        crates/ferrum-core/src/render/mod.rs \
        crates/ferrum-core/tests/golden/
git commit -m "test(render): end-to-end golden tests + 6 SVG goldens + 1 PNG hash

scatter_minimal, scatter_color, bar_grouped, line_simple, area_filled,
faceted_scatter as committed bit-stable references. PNG hash for the
scatter_minimal case verifies the resvg path. Refresh discipline:
FERRUM_UPDATE_GOLDENS=1 cargo test, with a justifying note in the
commit message per spec §9.4."
```

---

### Task 25: Final verification + Phase 7 done

**Files:**
- Modify: `docs/superpowers/ferrum-phases.md`
- Modify: `ferrum-spec.md` (§3.16 dated note)

- [ ] **Step 1: Run all tests, full suite**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
DYLD_LIBRARY_PATH=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") cargo test -p ferrum-core
uv run pytest
```

Expected:
- `cargo test -p ferrum-core`: ≥ 218 passing (per spec §9.3)
- `uv run pytest`: ≥ 88 passing

If either count is below target, identify which task added too few tests; do not weaken the targets without a written justification on the PR.

- [ ] **Step 2: Update phases doc**

Edit `docs/superpowers/ferrum-phases.md`. Find the Phase 7 row and:
1. Change `Status` from `pending` to `**done**`.
2. Replace `*(not yet written)*` in the `Spec doc` column with `[\`2026-05-09-static-renderer-design.md\`](specs/2026-05-09-static-renderer-design.md)`.
3. In the "Phase 7 — Static renderer" done-criteria block, change each `- [ ]` to `- [x]`.

- [ ] **Step 3: Add ferrum-spec.md §3.16 dated note**

Edit `ferrum-spec.md`, find the `### 3.16 Output and Rendering` section, and append (right after the `RenderConfig` block but before "Chart output methods"):

```markdown
> **2026-05-09 (Phase 7 implementation note):** Phase 7 honors the following
> `RenderConfig` fields: `scale`, `embed_fonts`, `background`, `width`,
> `height`. The remaining fields (`format`, `engine`, `raster_threshold`,
> `raster_behavior`, `raster_aggregate`, `raster_cmap`, `backend`,
> `tile_parallel`, `font_path`) are deferred to subsequent phases that ship
> their corresponding features. `embed_fonts` is treated as always-true in
> Phase 7 for visual determinism (rendered text uses the bundled Inter
> Regular regardless of system font availability); future phases may surface
> the `False` case for size-conscious users.
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/ferrum-phases.md ferrum-spec.md
git commit -m "docs(phases): mark Phase 7 static-renderer done

All four done criteria met:
  ✓ Scatter plot from spec → SVG (scatter_minimal/scatter_color goldens)
  ✓ All 8 primitive marks render (per-mark cargo tests + 4 in goldens)
  ✓ PNG output works (resvg path; scatter_minimal.png.sha256 golden)
  ✓ Output includes ticks, axis labels, legend (scatter_color golden)

cargo test -p ferrum-core: ≥ 218 passing.
uv run pytest: ≥ 88 passing.

ferrum-spec.md §3.16 gains a dated note documenting Phase 7's
honored RenderConfig field subset."
```

- [ ] **Step 5: Suggest opening a PR**

```bash
git log --oneline main..HEAD
```

Expected: ~25 Phase 7 commits. Open a PR with:

```bash
gh pr create --title "feat: Phase 7 — static renderer (SVG/PNG)" --body "$(cat <<'EOF'
## Summary

- Implements Phase 7 per `docs/superpowers/specs/2026-05-09-static-renderer-design.md`.
- New: `ferrum.render_svg`, `ferrum.render_png`. Full pipeline (transforms → scales → layout with FontdueMetrics → per-mark draw → SVG/PNG).
- 8 primitive marks (`point`, `line`, `bar`, `area`, `rule`, `text`, `tick`, `rect`) + internal axis/legend/strip-title draw fns.
- `Encoding` gains optional `color`; `PanelLayout` gains optional `strip_title` (both back-compat additive).
- `ThemeInputs` grows 18 render-only fields (sizes, opacities, colors).
- New deps: `fontdue`, `palette`, `usvg`, `resvg`, `base64` + bundled Inter Regular.

## Test plan

- [x] `cargo test -p ferrum-core` ≥ 218 passing
- [x] `uv run pytest` ≥ 88 passing
- [x] 6 SVG goldens + 1 PNG hash committed
- [x] `unset CONDA_PREFIX && uv run --no-sync maturin develop` succeeds clean
- [x] Smoke: `import ferrum; spec = ferrum.ChartSpec(mark='point', x='x', y='y'); ferrum.render_svg(spec, polars_df, viewport=(600, 400))` produces valid SVG

EOF
)"
```

(Do not push or open the PR until the user explicitly asks.)

---

## Self-Review

After all 25 tasks land:

**Spec coverage check:**
- §1 Goal ✓ — Tasks 20+21 implement the full pipeline.
- §2 In-scope items: 8 marks ✓ (Tasks 14-17), color encoding + legend ✓ (Tasks 6+10+19+22), facet strip titles ✓ (Tasks 3+12+19), SVG via SvgBuffer ✓ (Task 9), PNG via resvg ✓ (Task 21), bundled Inter ✓ (Tasks 1+7), tick formatting ✓ (Task 8), RenderConfig fields ✓ (Tasks 5+22).
- §3.3 ChartSpec extensions ✓ — `Encoding.color` Task 2, `PanelLayout.strip_title` Task 3.
- §4 per-component contracts ✓ — Tasks 6 (color), 7 (font), 8 (format), 9 (svg), 13 (draw).
- §5 input contract + ThemeInputs ✓ — Task 4 + Task 5.
- §6 algorithm ✓ — Task 20 step 1 mirrors §6 step-by-step.
- §7 error policy ✓ — Task 5 + every binding/orchestration step.
- §8 dependencies ✓ — Task 1.
- §9 test plan ✓ — Tasks 6-19 cover unit tests; Task 23 covers pytest; Task 24 covers goldens.
- §10 done-criteria gate ✓ — Task 25 ticks all four boxes.
- §11 locked decisions ✓ — every locked decision is implemented in a specific task.

**Type consistency check:**
- `DrawCtx` defined in Task 13, extended in Task 14 (gains `spec` field), used by all marks.
- `Color` alias defined in Task 6, used by `ThemeInputs` (Task 4 — note: Task 4 imports `palette::Srgba<u8>` directly because Task 6 hadn't run yet; this is OK because they're the same type).
- `MarkStyle` defined in Task 13, consumed by all marks.
- `RenderError` / `RenderWarning` defined in Task 5, used by orchestration (Task 20), binding (Task 22).
- `ResolvedScales` / `ScaleKind` / `ColorScale` defined in Task 10, used by marks (14-17) and orchestration (20).
- `FontdueMetrics` defined in Task 7, used by orchestration (Task 20) — implements the `TextMetrics` trait from Phase 6.
- `SvgBuffer` defined in Task 9, used by all marks and orchestration.

**Placeholder scan:** every code block is complete. The only "TBD"-shaped notes are explicit implementer guidance for ambiguous interfaces (Phase 4 internal accessors in Task 10, area `fmt_f` re-export in Task 15, JSON shape of `FacetSpec` in Task 23) — each calls out a specific check to perform and lists alternatives.

**Done.**

