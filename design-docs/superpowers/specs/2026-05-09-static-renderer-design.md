# Phase 7 — Static Renderer (SVG/PNG): Design Spec

**Date:** 2026-05-09
**Phase:** 7 (Static Renderer)
**Phase slug:** `static-renderer`
**Depends on:** Phase 4 (Scale engine — tick generation, scale-value mapping), Phase 5 (Stat engine — pre-render transforms), Phase 6 (Layout engine — `compute_layout`, `LayoutResult`, `TextMetrics` trait)
**Unblocks:** Phase 8 (Grammar API surface)

---

## §1 Goal

Take a `ChartSpec`, a Polars/PyArrow `RecordBatch`, a `ThemeInputs`, and a `Viewport`; emit a deterministic SVG string (and PNG bytes via resvg) representing the chart. Phase 7 owns the full pipeline:

```
spec + batch
  → apply Phase 5 transforms
  → resolve Phase 4 scales (x, y, color)
  → derive AxesInput, FacetGroups, LegendEntries
  → call Phase 6 compute_layout with FontdueMetrics
  → per-mark draw via SvgBuffer
  → emit SVG string  (or rasterize to PNG via usvg + resvg)
```

Output is bit-stable across runs and machines: `render_svg(...)` called twice with the same inputs returns byte-identical strings; `render_png(...)` returns byte-identical bytes (resvg is deterministic with bundled font).

---

## §2 Scope

### In scope (the binding done-criteria contract)
- All 8 primitive marks (`point`, `line`, `bar`, `area`, `rule`, `text`, `tick`, `rect`) render without panics on a minimal spec.
- Single-layer charts: `Mark::*` + `Encoding { x, y, color }`.
- **Color encoding** driving a legend (Okabe-Ito categorical palette, hardcoded).
- Faceted output with **per-panel strip titles** (back-compat additive `LayoutResult.PanelLayout` extension).
- SVG output via hand-rolled `SvgBuffer` with deterministic float formatting (`{:.3}` then trim).
- PNG output via `usvg` + `resvg` + `tiny_skia::Pixmap::encode_png`.
- Bundled Inter Regular for both layout measurement (`FontdueMetrics`) and rendering (font embedded by default for SVG, registered with usvg for PNG).
- Tick label formatting: hardcoded defaults for numeric / time / ordinal / threshold (no public knob yet).
- `RenderConfig` honored fields: `scale` (PNG density), `embed_fonts` (always-on for Phase 7), `background`, `width`, `height`.

### Out of scope (deferred, named here so future sessions don't re-litigate)
- `size` and `shape` encodings — Phase 8.
- `RenderConfig.engine="vega-lite"` — separate emitter module, post-Phase-9.
- `RenderConfig.backend`, `raster_threshold`, `raster_behavior`, `raster_aggregate`, `raster_cmap`, `tile_parallel`, `font_path` — Phase 8+ once tiny-skia direct ships.
- Composite marks (boxplot, errorbar, errorband, ribbon) — Phase 8 desugars to primitives.
- Statistical marks (`mark_density`, `mark_smooth`, etc. as named entry points) — Phase 8/9 sugar over primitive marks + Phase 5 transforms.
- Multi-layer composition, `HConcat`/`VConcat`/`Repeat` — Phase 8.
- Polar/geo coordinates — out of MVP entirely.
- Custom user-defined marks (Phase 12 extension API).
- Per-axis `format=` strings — Phase 8.
- Chart titles, subtitles — Phase 8.
- `rustybuzz` text shaping (kerning, ligatures) — deferred until a real text-heavy mark surfaces a complaint.
- tiny-skia direct-draw path — deferred to backend-policy phase.

---

## §3 Architecture

### §3.1 Public Rust surface

```rust
// crates/ferrum-core/src/render/mod.rs

pub fn render_svg(
    spec: &ChartSpec,
    batch: &arrow::record_batch::RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &RenderConfig,
) -> Result<RenderOutput<String>, RenderError>;

pub fn render_png(
    spec: &ChartSpec,
    batch: &arrow::record_batch::RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &RenderConfig,
) -> Result<RenderOutput<Vec<u8>>, RenderError>;

pub struct RenderOutput<T> {
    pub bytes: T,                      // SVG string OR PNG bytes
    pub layout: LayoutResult,          // for downstream inspection / debugging
    pub warnings: Vec<RenderWarning>,
}
```

Both functions are pure (apart from font-asset reads, which happen once via `include_bytes!`). Same inputs ⇒ same outputs. No global state, no I/O outside font asset loading.

### §3.2 Module layout

```
crates/ferrum-core/src/render/
  mod.rs                  // pub fn render_svg, render_png, RenderOutput, RenderError, RenderWarning
  config.rs               // RenderConfig (Phase 7 honored fields only; Default impl)
  prepare.rs              // prepare_render_inputs(spec, batch) → PreparedInputs
  scale_resolve.rs        // build Scale instances from spec + post-transform batch
  format.rs               // tick label formatting (numeric / time / ordinal / threshold)
  color.rs                // pub type Color = palette::Srgba<u8>; constructors, SVG formatter, opacity multiply
  palette.rs              // OKABE_ITO: &'static [Color; 8]; categorical_color(index)
  font.rs                 // INTER_REGULAR: &'static [u8] (include_bytes!); FontdueMetrics impl of TextMetrics
  svg.rs                  // SvgBuffer: header, g_open/close, rect, circle, line, path, polyline, text; deterministic floats
  png.rs                  // svg_string_to_png_bytes(svg, scale) using usvg + resvg + tiny_skia
  embed_font.rs           // base64 @font-face block injected into SVG when embed_fonts=true (default)
  draw.rs                 // dispatch_mark(mark, ctx, out); DrawCtx struct; orchestrates per-panel draw loop
  binding.rs              // PyO3 binding: render_svg / render_png Python entry points
  marks/
    mod.rs                // re-exports each mark's draw fn
    point.rs              // pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer)
    line.rs
    bar.rs
    area.rs
    rule.rs
    text.rs
    tick.rs
    rect.rs
    strip_title.rs        // internal-only; draws facet-strip headers
    axis.rs               // internal-only; draws axis line + ticks + labels + title from AxisLayout
    legend.rs             // internal-only; draws legend swatches + labels from LegendLayout
```

### §3.3 ChartSpec extensions (additive, back-compat)

#### `Encoding` gains a `color` field

Phase 7 adds **one** field to `Encoding`, mirroring how Phase 5 added `transforms` and Phase 6 added `facet`:

```rust
pub struct Encoding {
    // existing
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub x: Option<EncodingSpec>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub y: Option<EncodingSpec>,
    // new in Phase 7
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub color: Option<EncodingSpec>,
}
```

Existing JSON outputs stay byte-identical (omitted field round-trips as `None`). `EncodingSpec` itself is unchanged.

The Python `ChartSpec.__init__` keyword is `color: Union[str, EncodingSpec, None] = None` for parity with `x`/`y`.

#### `LayoutResult.PanelLayout` gains an optional `strip_title`

```rust
pub struct PanelLayout {
    // existing fields unchanged
    pub plot_area: Rect,
    pub facet_key: Option<FacetKey>,
    pub row: u32,
    pub col: u32,
    // new in Phase 7
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strip_title: Option<StripTitleLayout>,
}

pub struct StripTitleLayout {
    pub text: String,
    pub anchor: (f64, f64),
    pub align: TextAnchor,             // shared with SvgBuffer text emission (§4.4)
    pub font_size: f64,
}
```

`compute_layout` is updated to populate `strip_title` whenever `spec.facet.is_some()`, reserving a strip band above each panel. `ThemeInputs` grows `strip_text_size`, `strip_padding`, `strip_background_color` (see §5.2).

This is the only Phase 6 behavioral change. Existing Phase 6 tests pass because:
- Non-faceted layouts: `strip_title` is always `None` (omitted from JSON via `skip_serializing_if`).
- Faceted layouts: existing tests don't assert on the new field.
- Theme defaults: new theme fields default to values that don't affect non-faceted arithmetic.

#### `ThemeInputs` grows additively

See §5.2 for the full new-field list. No standalone `Theme` Rust type yet — that ships in Phase 8.

---

## §4 Per-component contracts

### §4.1 `color.rs`

```rust
pub type Color = palette::Srgba<u8>;

pub fn from_hex_str(s: &str) -> Result<Color, ColorParseError>;  // "#1f77b4" or "#1f77b4cc"
pub fn from_rgb(r: u8, g: u8, b: u8) -> Color;                    // alpha = 255
pub fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Color;
pub fn with_opacity(c: Color, opacity_0_1: f64) -> Color;         // multiplies alpha; clamped [0, 1]
pub fn fmt_svg(c: Color) -> String;                               // "#1f77b4" if alpha=255, else "rgba(31,119,180,0.502)"

#[derive(Debug)]
pub struct ColorParseError(pub String);
```

Opacity composition pattern: `with_opacity(theme.mark_color, theme.default_opacity * encoding_opacity)`. Conversions to `LinSrgb` / `Oklch` are not part of Phase 7's public API but are available via `palette::*` for future scale-engine work.

### §4.2 `palette.rs`

```rust
pub const OKABE_ITO: &[Color; 8] = &[
    from_rgb(0xE6, 0x9F, 0x00),  // orange
    from_rgb(0x56, 0xB4, 0xE9),  // sky blue
    from_rgb(0x00, 0x9E, 0x73),  // bluish green
    from_rgb(0xF0, 0xE4, 0x42),  // yellow
    from_rgb(0x00, 0x72, 0xB2),  // blue
    from_rgb(0xD5, 0x5E, 0x00),  // vermillion
    from_rgb(0xCC, 0x79, 0xA7),  // reddish purple
    from_rgb(0x00, 0x00, 0x00),  // black
];

pub fn categorical_color(category_index: usize) -> Color {
    OKABE_ITO[category_index % OKABE_ITO.len()]
}
```

Caller (in `prepare.rs`) emits `RenderWarning::ColorPaletteOverflowed { categories: n }` when the distinct-value count exceeds `OKABE_ITO.len()`.

### §4.3 `font.rs`

```rust
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
}

impl crate::layout::TextMetrics for FontdueMetrics {
    fn measure_width(&self, text: &str, font_size: f64) -> f64 {
        text.chars()
            .map(|c| self.font.metrics(c, font_size as f32).advance_width as f64)
            .sum()
    }
    fn line_height(&self, font_size: f64) -> f64 {
        let line_metrics = self.font.horizontal_line_metrics(font_size as f32)
            .expect("Inter font has horizontal line metrics");
        (line_metrics.ascent - line_metrics.descent + line_metrics.line_gap) as f64
    }
}
```

`FontdueMetrics::new()` is constructed once per `render_*` call; cost is parse-time-only and amortized across every label measurement. No global cache (themes-as-values discipline; no module-level mutable state).

### §4.4 `svg.rs` — `SvgBuffer`

```rust
pub struct SvgBuffer { buf: String }

impl SvgBuffer {
    pub fn new(viewport: Rect, background: Option<Color>, embed_font: bool) -> Self;
    pub fn finish(self) -> String;  // closes <svg>, returns string

    pub fn g_open(&mut self, transform: Option<&str>);
    pub fn g_close(&mut self);
    pub fn rect(&mut self, r: Rect, style: &FillStroke, corner_radius: Option<f64>);
    pub fn circle(&mut self, cx: f64, cy: f64, radius: f64, style: &FillStroke);
    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, style: &Stroke);
    pub fn path(&mut self, d: &str, style: &FillStroke);            // for area/line interpolation
    pub fn polyline(&mut self, points: &[(f64, f64)], style: &Stroke);
    pub fn text(&mut self, x: f64, y: f64, content: &str, style: &TextStyle);
    pub fn clip_open(&mut self, id: &str, rect: Rect);              // <clipPath>
    pub fn clip_close(&mut self);
    pub fn use_clip(&mut self, clip_id: &str);                      // wrap subsequent <g> with clip-path attr
}

pub struct FillStroke {
    pub fill: Option<Color>,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
}
pub struct Stroke {
    pub stroke: Color,
    pub stroke_width: f64,
    pub stroke_dash: Option<Vec<f64>>,
}
pub struct TextStyle {
    pub fill: Color,
    pub font_size: f64,
    pub anchor: TextAnchor,
    pub angle: f64,
}
pub enum TextAnchor { Start, Middle, End }
```

**Determinism contract:**
- Floats: `format_f64(x)` formats with `{:.3}`, then trims trailing zeros and a dangling decimal point. `1.500` → `"1.5"`, `1.000` → `"1"`. NaN/Inf forbidden — caller responsibility (Phase 4 scales clamp; renderer asserts in debug builds).
- Attributes within an element: emitted in fixed order (declared per-element in `svg.rs`), never alphabetical-by-iteration. E.g., `<circle>` always emits `cx`, `cy`, `r`, `fill`, `stroke`, `stroke-width` in that order.
- XML escaping: centralized in `escape_text_content` (escapes `&`, `<`, `>`) and `escape_attr_value` (also escapes `"`), applied to every user-supplied string.
- SVG header: `<svg xmlns="http://www.w3.org/2000/svg" width="W" height="H" viewBox="0 0 W H">` with W/H formatted via `format_f64`.

### §4.5 `embed_font.rs`

When `config.embed_fonts == true` (always-on for Phase 7 per §11 row 15), `SvgBuffer::new` emits, immediately after the `<svg>` open tag:

```xml
<defs><style>@font-face{font-family:"Inter";src:url("data:font/ttf;base64,<BASE64>") format("truetype");}</style></defs>
```

`<BASE64>` is the standard base64 encoding of `INTER_REGULAR`. Encoded once per render via `base64::engine::general_purpose::STANDARD.encode(...)`; result is ~410 KB of inline data, acceptable for the determinism guarantee.

### §4.6 `draw.rs` — `DrawCtx` and dispatch

```rust
pub struct DrawCtx<'a> {
    pub panel: &'a PanelLayout,        // pixel rect for this panel
    pub theme: &'a ThemeInputs,
    pub scales: &'a ResolvedScales,    // x, y, color built from spec + batch
    pub batch: &'a RecordBatch,        // post-transform data (rows for THIS panel only)
    pub mark_style: &'a MarkStyle,     // theme + spec overrides resolved into one struct
}

pub struct ResolvedScales {
    pub x: ScaleKind,                  // sealed enum: Linear|Log|Time|Ordinal|Symlog|Quantile|Threshold
    pub y: ScaleKind,
    pub color: Option<ColorScale>,     // Categorical(palette) for Phase 7; Sequential later
}

pub enum ColorScale {
    Categorical {
        domain: Vec<String>,            // distinct values in encounter order
        palette: &'static [Color],      // OKABE_ITO for Phase 7
    },
}

pub struct MarkStyle {
    pub fill: Color,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
    pub opacity: f64,
    pub point_size: f64,               // mark_point/mark_tick
    pub corner_radius: f64,            // mark_bar/mark_rect
    pub stroke_dash: Option<Vec<f64>>, // mark_rule
    // populated from theme defaults; mark spec overrides applied in prepare.rs
}

pub fn dispatch_mark(mark: &Mark, ctx: &DrawCtx, out: &mut SvgBuffer) {
    match mark {
        Mark::Point => marks::point::draw(ctx, out),
        Mark::Line  => marks::line::draw(ctx, out),
        Mark::Bar   => marks::bar::draw(ctx, out),
        Mark::Area  => marks::area::draw(ctx, out),
        Mark::Rule  => marks::rule::draw(ctx, out),
        Mark::Text  => marks::text::draw(ctx, out),
        Mark::Tick  => marks::tick::draw(ctx, out),
        Mark::Rect  => marks::rect::draw(ctx, out),
    }
}
```

Each `marks/<name>.rs::draw(ctx, out)` reads the panel rect + scaled values from `ctx.batch` via `ctx.scales`, applies `ctx.mark_style`, and emits SVG primitives via `out`. Per-mark draw fns are testable in isolation with synthetic `DrawCtx` fixtures.

---

## §5 Input contract & `ThemeInputs` extension

### §5.1 `render_svg` / `render_png` signature

```rust
pub fn render_svg(
    spec: &ChartSpec,
    batch: &RecordBatch,
    theme: &ThemeInputs,
    viewport: Viewport,
    config: &RenderConfig,
) -> Result<RenderOutput<String>, RenderError>;
```

- `spec` carries mark, encoding (x/y/color), transforms, facet.
- `batch` is **pre-transform** data — `prepare_render_inputs` applies `spec.transforms` internally via Phase 5 before deriving scale domains. The public API contract is "raw data in, image out."
- `theme` is a value (CLAUDE.md hard constraint).
- `viewport` is pixel canvas.
- `config` is the small `RenderConfig` slice Phase 7 honors.

### §5.2 `ThemeInputs` additive extension

Phase 6's `ThemeInputs` grows with these fields (all `pub`, all `Default`-resolved):

| Field | Default | Use |
|---|---|---|
| `mark_color` | `OKABE_ITO[0]` | Default fill when no color encoding |
| `point_size` | `30.0` | Mark area in px² (matches `ferrum-spec.md §3.13` default) |
| `line_stroke_width` | `1.5` | mark_line, mark_rule |
| `bar_corner_radius` | `0.0` | mark_bar |
| `area_opacity` | `0.4` | mark_area fill alpha |
| `default_opacity` | `1.0` | All other marks |
| `axis_line_color` | `from_hex_str("#888888").unwrap()` | Axis stroke |
| `axis_line_width` | `1.0` | Axis stroke width |
| `tick_size` | `4.0` | Tick mark length |
| `tick_color` | `from_hex_str("#888888").unwrap()` | Tick mark stroke |
| `grid_color` | `from_hex_str("#eeeeee").unwrap()` | Grid lines (drawn behind marks) |
| `grid_width` | `1.0` | Grid stroke width |
| `font_color` | `from_hex_str("#222222").unwrap()` | Default text fill |
| `background_color` | `from_hex_str("#ffffff").unwrap()` | Canvas fill |
| `grid` | `true` | Whether grid lines are drawn behind marks |
| `strip_text_size` | `13.0` | Facet-strip title font size |
| `strip_padding` | `4.0` | Facet-strip vertical padding |
| `strip_background_color` | `from_hex_str("#f0f0f0").unwrap()` | Facet-strip fill |

Phase 6's existing fields (`padding`, `column_padding`, `row_padding`, `axis_title_padding`, `label_font_size`, `title_font_size`, `legend_orient`) are unchanged. Phase 6 layout tests continue to pass because the new fields default to values that don't affect layout arithmetic except for the strip-title band, which is reserved only when `spec.facet.is_some()`.

### §5.3 `RenderConfig` field-honoring table

```rust
pub struct RenderConfig {
    pub scale: f64,             // PNG density multiplier; default 2.0
    pub embed_fonts: bool,      // SVG @font-face inlining; default true (always-on for Phase 7)
    pub background: Option<Color>,
    pub width: Option<f64>,     // overrides viewport.width if Some
    pub height: Option<f64>,    // overrides viewport.height if Some
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            scale: 2.0,
            embed_fonts: true,
            background: None,
            width: None,
            height: None,
        }
    }
}
```

Fields from `ferrum-spec.md §3.16` that are **not** present in Phase 7's `RenderConfig`: `format`, `engine`, `raster_threshold`, `raster_behavior`, `raster_aggregate`, `raster_cmap`, `backend`, `tile_parallel`, `font_path`. These are deferred to phases that ship the corresponding feature.

`ferrum-spec.md §3.16` will get a dated note flagging Phase 7's subset.

---

## §6 Algorithm — pipeline (single pass)

```
render_svg(spec, batch, theme, viewport, config) →

1. Validate inputs:
     viewport.width > 0  ∧  viewport.height > 0
     batch.num_rows() > 0
     spec.encoding.x and spec.encoding.y must reference valid columns in batch.schema()
     spec.encoding.color (if Some) must reference a valid column
   → on failure: RenderError::{InvalidViewport | EmptyBatch | UnknownColumn}

2. Apply RenderConfig overrides:
     viewport.width  = config.width.unwrap_or(viewport.width)
     viewport.height = config.height.unwrap_or(viewport.height)
     background      = config.background.or(Some(theme.background_color))

3. prepare_render_inputs(spec, batch) → PreparedInputs:
     a. transformed = apply Phase 5 transforms (spec.transforms) to batch
     b. scales      = scale_resolve.rs builds ResolvedScales from spec + transformed:
        - x: pick LinearScale|LogScale|TimeScale|OrdinalScale per encoding type + column dtype
        - y: same
        - color: if spec.encoding.color is Some, build Categorical color scale over distinct values
                  mapped to OKABE_ITO; emit ColorPaletteOverflowed warning if count > 8; else None
     c. axes_input  = format ticks via format.rs; AxisInput { tick_labels, title, ... }:
        - x_title   = spec.encoding.x.field
        - y_title   = spec.encoding.y.field
        - tick_labels generated by scale.ticks() then format_<kind>(value)
     d. facet_groups = if spec.facet, group transformed by spec.facet.field → Vec<FacetGroup>
                       (in stable encounter order; n_rows = group size)
     e. legend_entries = if scales.color is Some, emit LegendEntry { label, symbol: Circle }
                         per category in domain order

4. metrics = FontdueMetrics::new()  // parses Inter once
   layout  = compute_layout(spec, theme, viewport, &axes_input, &facet_groups, &legend_entries, &metrics)
       → may produce LayoutWarnings; collected into RenderOutput.warnings as RenderWarning::Layout(_)
       → populates strip_title on each PanelLayout when facet is set

5. Initialize SvgBuffer:
     out = SvgBuffer::new(layout.viewport, background, config.embed_fonts)
     // if config.embed_fonts (always true for Phase 7): emit <defs><style>@font-face{...}</style></defs>

6. Draw layer order (z-order, back-to-front):
     a. background rect (if background.is_some())
     b. for each panel in layout.panels:
          - draw grid lines from layout.axes ticks (if theme grid enabled — Phase 7 default: enabled)
          - draw axis (axes.rs uses axes layout for this panel)
          - clip to panel.plot_area (via <clipPath> registered with unique id)
          - select rows from transformed RecordBatch belonging to this panel:
                if facet: filter where col(facet.field) == panel.facet_key.value
                else:     all rows
          - dispatch_mark(spec.mark, &ctx, &mut out)  -- single Mark per Phase 7 chart
          - close clip group
          - draw strip_title (if panel.strip_title.is_some())
     c. draw legend (legend.rs uses layout.legend if present)

7. svg_string = out.finish()  -- closes <svg>, returns String
   return RenderOutput { bytes: svg_string, layout, warnings }
```

`render_png` follows the same pipeline through step 7, then:

```
8. tree     = usvg::Tree::from_str(&svg_string, &usvg_options_with_inter_font)
   pixmap_w = (layout.viewport.w * config.scale).round() as u32
   pixmap_h = (layout.viewport.h * config.scale).round() as u32
   pixmap   = tiny_skia::Pixmap::new(pixmap_w, pixmap_h)
       .ok_or(RenderError::ResvgFailed("pixmap allocation"))?
   resvg::render(
       &tree,
       tiny_skia::Transform::from_scale(config.scale as f32, config.scale as f32),
       &mut pixmap.as_mut(),
   )
   bytes    = pixmap.encode_png()
       .map_err(|e| RenderError::ResvgFailed(e.to_string()))?
   return RenderOutput { bytes, layout, warnings }
```

`usvg_options_with_inter_font` registers `INTER_REGULAR` bytes against the family name `"Inter"` so resvg always finds the bundled font regardless of system font availability.

### §6.1 Constants

| Constant | Value | Purpose |
|---|---|---|
| `FLOAT_PRECISION` | `3` | `{:.N}` precision for SVG attribute floats |
| `DEFAULT_GRID_ENABLED` | `true` | Grid lines drawn behind marks unless theme disables |
| `CLIP_ID_PREFIX` | `"ferrum-clip-"` | `<clipPath>` id namespace |
| `INTER_FONT_FAMILY` | `"Inter"` | `font-family` attribute value in SVG and usvg registration |

These live as `pub const` in `render/mod.rs`. Not configurable from the public API in Phase 7.

---

## §7 Error policy (hybrid, mirrors Phase 5/6)

| Class | Trigger | Response |
|---|---|---|
| **Structural** | Invalid viewport, empty batch, encoding references unknown column, unparseable color string, mark/scale/encoding type mismatch, transform failure, scale resolution failure, layout failure, resvg failure | `RenderError` → `PyValueError` |
| **Geometric edge** | Data row outside x/y domain (out-of-domain scale output → `f64::NAN`), single-row line/area, all-NaN column, color category overflow (> 8 cats wraps to OKABE_ITO[0]) | Skip row + emit `RenderWarning`; render proceeds |
| **Layout warnings** | Surfaced from Phase 6 (`LayoutWarning`) | Wrapped as `RenderWarning::Layout(_)` and collected |
| **Silent** | Empty legend (no color encoding), no strip title (no facet) | Normal, no warning |

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum RenderError {
    InvalidViewport { width: f64, height: f64 },
    EmptyBatch,
    UnknownColumn { name: String },
    InvalidColor(String),
    EncodingTypeMismatch { channel: &'static str, expected: &'static str, got: String },
    TransformFailed(String),       // wraps Phase 5 errors
    ScaleResolutionFailed(String), // wraps Phase 4 errors (catch-all for facet-filter / transform-output)
    LayoutFailed(String),          // wraps Phase 6 errors
    ResvgFailed(String),           // PNG path only
    // Phase 9 coherence-pass additions (F5, 2026-05-11): typed variants
    // that replaced `Other(String)` and several `ScaleResolutionFailed`
    // sites. `Other` was retired.
    PositionAdjustFailed { adjustment: &'static str, reason: String },
    // F5 residual cleanup (2026-05-12): renamed `channel` → `field`
    // (the column name is what callers always pass), and added
    // `context: Option<&'static str>` for sites that pre-F5 prefixed
    // the prose with a scale tag ("size", "opacity", "scale"). Display:
    // `"<context>: column '<field>' has unsupported dtype: <dtype>"`
    // when context is set; otherwise `"column '<field>' has …"`.
    UnsupportedDtype { field: String, dtype: String, context: Option<&'static str> },
    EmptyDomain { channel: String, field: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RenderWarning {
    Layout(LayoutWarning),
    OutOfDomainRows { mark: String, count: u64 },
    ColorPaletteOverflowed { categories: u32 },
    EmptyPanel { panel_index: usize },
}
```

> **Note 2026-05-09 (Task 5):** outer tag renamed from `"kind"` to `"type"`.
> `LayoutWarning` (Phase 6) already uses `tag = "kind"` and has Phase 6
> round-trip tests pinning that JSON shape; with both enums tagged `"kind"`
> and internal tagging on a newtype variant, serde flattens the inner
> struct's fields into the outer object and emits two `kind` keys, which
> fails round-trip. `LayoutWarning`'s tag is held stable; `RenderWarning`'s
> outer tag becomes `"type"` to disambiguate.

Python binding (`render/binding.rs`) raises `PyValueError` for `RenderError`; emits `warnings.warn(message)` per warning at the binding boundary, preserving the warning's variant info as part of the message.

---

## §8 New external dependencies

| Crate | Pin discipline | Purpose | Decision |
|---|---|---|---|
| `fontdue` | **Exact pin** in `[workspace.dependencies]` (e.g. `fontdue = "=0.9.3"`) | Glyph metrics + glyph rasterization (used by `FontdueMetrics`; usvg can also consume it transitively) | Adopted. Pinned exactly per CLAUDE.md PyO3 discipline; version bumps require golden-refresh. |
| `palette` | Range pin (e.g. `palette = "~0.7"`) | `Color` type alias `Srgba<u8>`; future `LinSrgb`/`Oklch` conversions | Adopted now (decision §11 row 6). Pinned with `~` to allow patch updates without minor-version churn. |
| `usvg` | Range pin (e.g. `usvg = "~0.45"`) | SVG → typed tree | Adopted. Required for resvg. |
| `resvg` | Range pin (matched to usvg minor) | Tree → tiny_skia rasterization | Adopted. PNG path requires it. |
| `tiny-skia` | Inherited from resvg (re-exported) | Pixmap + PNG encode | Adopted transitively via resvg; we use `Pixmap::encode_png` directly. |
| `base64` | Range pin (e.g. `base64 = "~0.22"`) | `@font-face` data URL emission | Adopted. Tiny dep, no realistic alternative. |
| `rustybuzz` | — | Text shaping (kerning, ligatures) | **Rejected for Phase 7.** Tick labels are short numeric/ordinal strings; visual delta is invisible. Defer until a text-heavy mark surfaces a complaint. |

**Bundled font asset:**
- `crates/ferrum-core/assets/fonts/Inter-Regular.ttf` (~310 KB)
- `crates/ferrum-core/assets/fonts/Inter-OFL.txt` (license text — Inter is OFL-licensed)
- Embedded via `include_bytes!`; no runtime path resolution.
- License notice added to repo `NOTICE` file (or created if absent) attributing Inter Regular under SIL Open Font License 1.1.

**Workspace `[workspace.dependencies]` additions (set in root `Cargo.toml`):**

```toml
fontdue   = "=<exact>"   # pin determined when crate is added; locks golden stability
palette   = "~<minor>"   # range pin
usvg      = "~<minor>"
resvg     = "~<minor>"   # must match usvg minor
base64    = "~<minor>"
```

Each crate's exact version is selected at the time of the implementation plan, recorded in the locked-decisions table addendum, and verified against crates.io for breaking-change history.

---

## §9 Test plan

### §9.1 Cargo tests (target ≥ 40 new; cargo total ≥ 218)

#### `color.rs` (~3 tests)
- `from_hex_str("#1f77b4")` → known `Srgba<u8>`; `"#1f77b4cc"` parses alpha; `"red"` returns `ColorParseError`.
- `with_opacity(red, 0.5)` produces alpha=128 (rounded).
- `fmt_svg(opaque)` → `"#1f77b4"`; `fmt_svg(translucent)` → `"rgba(31,119,180,0.502)"`.

#### `palette.rs` (~2 tests)
- `categorical_color(0)` returns `OKABE_ITO[0]`.
- `categorical_color(8)` wraps to `OKABE_ITO[0]`. (Caller in `prepare.rs` is responsible for the warning.)

#### `font.rs` (~3 tests)
- `FontdueMetrics::new()` succeeds (font asset parses without panic).
- `measure_width("100", 11.0)` is within ±2 px of a hand-measured reference for Inter-Regular (precise reference computed once via fontdue and pinned as the test expectation).
- `line_height(11.0)` returns expected value (Inter ascent + descent + line gap at 11 pt; pinned reference).

#### `format.rs` (~6 tests)
- Numeric: `0.0` → `"0"`, `1.5` → `"1.5"`, `1500000.0` → `"1.5e6"`, `0.0001` → `"1e-4"`, `1.000001` → `"1"`.
- Time: epoch_ms for year-tick spacing → `"2026"`; for day-tick → `"2026-03-15"`; for hour-tick → `"15:00"`.
- Ordinal: passthrough.

#### `svg.rs` (~10 tests)
- `circle(10.5, 20.5, 3.0, &style)` emits `<circle cx="10.5" cy="20.5" r="3" fill="..."/>` in fixed attribute order.
- `text("Price > 0")` escapes `>` → `&gt;`.
- Float formatter: `1.5` → `"1.5"`, `1.0` → `"1"`, `1.500` → `"1.5"`, `0.0001` → `"0"` (precision floor — caveat: very small values lose precision; documented).
- `g_open(Some("translate(10,20)"))` then `g_close()` produces matched `<g transform="translate(10,20)"></g>`.
- Background rect omitted when `None`; emitted as first element after `<defs>` when `Some`.
- Empty buffer + finish produces minimal valid `<svg>...</svg>` (with `<defs>` font-face block).
- Two consecutive `render_svg` calls on the same inputs produce byte-identical strings.
- `<defs>` font-face block contains expected `font-family:"Inter"` substring and base64 prefix.
- `clip_open` / `clip_close` produce matched `<clipPath id="..."><rect.../></clipPath>` with the supplied id.
- `escape_text_content` handles `&`, `<`, `>`; `escape_attr_value` additionally handles `"`.

#### `marks/*.rs` (~16 tests, ~2 per primitive × 8 primitives)
For each of `point`, `line`, `bar`, `area`, `rule`, `text`, `tick`, `rect`:
- **Basic shape test:** synthetic `DrawCtx` with 3 rows → assert correct element kind + count + theme-derived fill in the buffer.
- **Edge case** (varies per mark): empty data → no elements emitted; out-of-domain row → skipped + appropriate warning emitted via the test harness's warning collector.

#### `prepare.rs` (~6 tests)
- Scale resolution: quantitative x → `LinearScale`; ordinal x → `OrdinalScale`; temporal x → `TimeScale`.
- Color encoding builds Categorical from distinct values, in encounter order.
- Color overflow (10 distinct categories) emits `ColorPaletteOverflowed { categories: 10 }`.
- Facet groups: 3-category facet → 3 `FacetGroup` entries with correct row counts.
- Legend entries: 3 distinct color values → 3 `LegendEntry { label, symbol: Circle }` in domain order.
- Empty batch → `RenderError::EmptyBatch`.

#### End-to-end goldens (`tests/golden/`, 6 SVG files + 1 PNG hash, 7 tests)

| Golden | Spec | Asserts |
|---|---|---|
| `scatter_minimal.svg` | 3 rows, mark=point, x/y quantitative, no color encoding | Renders 3 circles in expected pixel positions; default `mark_color`; layout reservations correct |
| `scatter_color.svg` | 6 rows, mark=point, x/y quantitative, color=ordinal-3 | Renders 6 circles in 3 OKABE_ITO colors; legend on right with 3 entries; axis ticks + titles present |
| `bar_grouped.svg` | 4 rows, mark=bar, x ordinal, y quantitative | Renders 4 rects with `bar_corner_radius` from theme; ordinal x-axis tick labels |
| `line_simple.svg` | 5 rows, mark=line, x/y quantitative | Renders single `<polyline>` (or `<path>`) with `line_stroke_width` |
| `area_filled.svg` | 5 rows, mark=area, x/y quantitative | Renders filled `<path>` with `area_opacity=0.4` translated to `fill-opacity` |
| `faceted_scatter.svg` | 9 rows, facet=species(3), mark=point, color=species | 3 panels with strip-title bands above each, 1 legend, 9 circles total distributed across panels |
| `scatter_minimal.png.sha256` | Same spec as `scatter_minimal.svg`, rendered to PNG with `scale=2.0` | sha256 of PNG bytes matches frozen value |

#### Golden-refresh workflow (`§9.4`)

Goldens are refreshed via `FERRUM_UPDATE_GOLDENS=1 cargo test`. The test harness, when this env var is set, writes the produced output to disk in place of comparison. Triggers requiring refresh:
- Font asset change (Inter version, weight, file replacement).
- `fontdue` version change (sub-pixel advance changes propagate).
- `palette` version change affecting `Srgba<u8>` formatting (unlikely but possible).
- `resvg` / `usvg` / `tiny-skia` version change for the PNG hash.
- Intentional `SvgBuffer` formatting change (attribute order, float precision).
- Intentional theme default change.
- Intentional algorithm change in `prepare.rs` / `compute_layout` / mark draw fns.

Each golden refresh requires a justifying note in the commit message naming the trigger.

### §9.2 Pytest tests (target ≥ 10 new; pytest total ≥ 88)

`tests/test_render.py`:

- `from ferrum._core import render_svg, render_png` imports.
- `render_svg(spec, polars_df, viewport=(600, 400))` returns `str` starting with `<?xml` or `<svg`.
- `render_png(spec, polars_df, viewport=(600, 400))` returns `bytes` starting with PNG magic `\x89PNG\r\n\x1a\n`.
- Theme dict round-trip: `render_svg(..., theme={"mark_color": "#ff0000"})` produces SVG containing `"#ff0000"`.
- Invalid viewport raises `ValueError`.
- Invalid color string raises `ValueError`.
- Unknown column raises `ValueError`.
- Faceted spec produces 3 strip-title `<text>` elements in SVG output.
- Empty DataFrame raises `ValueError` (matches `EmptyBatch`).
- `RenderConfig`-equivalent kwargs (`scale=2.0`, `embed_fonts=False`, `background="#000"`) accepted and applied.
- pyarrow Table input works identically to polars DataFrame (Phase 2 transport parity).

### §9.3 Test count baseline at end of Phase 7

- `cargo test -p ferrum-core`: ≥ 218 (currently 178; +40 render)
- `uv run pytest`: ≥ 88 (currently 78; +10 binding)

---

## §10 Done-criteria gate

From `ferrum-phases.md` Phase 7 done criteria:

- [ ] **A scatter plot from a spec file renders to a valid SVG file** → covered by `scatter_minimal.svg` and `scatter_color.svg` end-to-end goldens (§9.1) + pytest `render_svg` smoke test (§9.2).
- [ ] **All eight primitive marks render without panics on a minimal spec** → covered by 8 × per-mark cargo tests in `marks/*.rs` (§9.1) + 4 of 8 marks exercised in goldens (`point`, `bar`, `line`, `area`); remaining 4 (`rule`, `text`, `tick`, `rect`) covered by per-mark cargo tests asserting non-panic + correct element emission.
- [ ] **PNG output works (resvg or equivalent)** → covered by pytest PNG smoke test + `scatter_minimal.png.sha256` golden hash (§9.1, §9.2).
- [ ] **Output includes correct scale ticks, axis labels, and a legend** → covered by `scatter_color.svg` golden (asserts axis ticks, axis titles, color legend rendered) + `format.rs` tick-formatting tests (§9.1).

A Phase-7-done PR must show all four boxes ticked, `cargo test -p ferrum-core` ≥ 218 passing, `uv run pytest` ≥ 88 passing.

---

## §11 Locked decisions table

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Public entry-point pipeline boundary | Full pipeline: `render_svg(spec, batch, theme, viewport, config)` | Done-criterion #1 ("scatter plot from spec → SVG") falls out of one entry point. Internal factoring (`prepare`/`scale_resolve`/`draw`) keeps it modular without doubling public surface. Phase 11 has its own crate; cross-reuse is fictional. |
| 2 | Backend scope | SVG draw path + PNG via resvg | Single set of mark draw fns, single test surface. tiny-skia-direct + auto-raster deferred to where they belong (§3.17 backend-policy phase). |
| 3 | Mark dispatch | Sealed enum + per-module `draw` fn | Mirrors Phase 5 transform pattern (`transform/bin.rs`, `transform/kde.rs`, ...). No trait objects. Phase 12 extension story can be added over the top without breaking Phase 7. |
| 4 | SVG emission | Hand-rolled `SvgBuffer` (~150 LOC) | Determinism is load-bearing for golden tests; controlling attribute order, float formatting, escaping is non-negotiable. SVG element vocabulary needed is small (~10 kinds). |
| 5 | Text measurement & rendering | `fontdue` + bundled Inter Regular; **no** rustybuzz | Real glyph advances drive layout reservations correctly. Bundled font + always-on `@font-face` embed makes SVG visually deterministic across machines. rustybuzz deferred — tick labels don't visibly benefit. |
| 6 | Color type | `pub type Color = palette::Srgba<u8>` | Adopted *now* to prevent mark-code churn when Phase 8+ ships color scales requiring OkLab/LinSrgb interpolation. Mark code is monomorphic on `Srgba<u8>`; conversions stay inside `color.rs`. |
| 7 | Categorical palette | OKABE_ITO hardcoded as `&'static [Color; 8]` in `palette.rs` | Matches `ferrum-spec.md §3.13` default. No scheme-name registry yet; one hardcoded palette is enough to satisfy done-criterion #4. Overflow wraps + warns. |
| 8 | Tick label formatting | Hardcoded defaults per scale kind in `format.rs`; no public knob | "Correct ticks" (done-criterion #4) means readable, not configurable. Per-axis `FormatSpec` deferred to Phase 8 along with the encoding-API redesign. |
| 9 | Facet-strip titles | In scope; `LayoutResult.PanelLayout` gains `strip_title: Option<StripTitleLayout>` (additive, `serde(default)`) | Faceted output without per-panel headers is a known regression flagged in Phase 6 §2. Cheapest to add now (one layout extension + one internal mark module). Phase 8 inherits a working renderer. |
| 10 | Test strategy | Structural per-mark tests + bit-exact end-to-end SVG goldens + PNG hash check | Per-mark gives diagnostics; goldens catch composition regressions; deterministic SVG (Q4) makes goldens stable across machines. PNG hash confirms resvg path runs without binary snapshot bloat. |
| 11 | Error policy | Hybrid: structural → `RenderError`/`PyValueError`, geometric edge → skip + `RenderWarning` | Mirrors Phase 5 §6 / Phase 6 §7. Layout warnings wrap up via `RenderWarning::Layout(_)`. |
| 12 | `ThemeInputs` extension | Additive (~17 new fields, all `Default`-resolved) | No standalone `Theme` Rust type yet — that ships in Phase 8. Phase 6 layout tests untouched because new fields don't affect arithmetic. |
| 13 | Python binding | Two functions: `render_svg`, `render_png` | Clearer typed returns (`str` vs `bytes`), no string-literal dispatch. Phase 8 wraps both behind `chart.show_svg()` / `chart.show_png()`. |
| 14 | `Encoding` extension | `+ color: Option<EncodingSpec>` only | Smallest viable surface for done-criterion #4. `size`/`shape` deferred to Phase 8 (need scale-engine extensions + shape libraries). |
| 15 | Font embedding | Always-on for SVG output (overrides `embed_fonts=true` default to **unconditional** for Phase 7) | Determinism contract: rendered text uses bundled Inter; layout reservations use fontdue advances against the same file. Off-machine fallback would visually break alignment. Future phases may surface `embed_fonts=False` for size-conscious users; Phase 7 keeps it locked. |
| 16 | Phase 6 binding resolution | Leave Phase 6 `compute_layout` Python binding as-is (`HeuristicMetrics`-only). Production rendering goes through Phase 7 `render_svg`/`render_png` which call `compute_layout` internally with `FontdueMetrics`. | Clean separation; no breaking change to Phase 6's surface; layout-only inspection use cases (debugging, tests) keep working. |

---

## §12 Cross-phase notes

### Phase 4 (Scale engine) — what Phase 7 calls
- `LinearScale::ticks(count)`, `OrdinalScale::ticks()`, `TimeScale::ticks(count)`, etc. for tick-value generation.
- `Scale::scale(value)` for per-row coordinate mapping inside mark draw fns.
- `scale_resolve.rs` builds the appropriate `Scale` instance based on encoding type + column dtype.

### Phase 5 (Stat engine) — what Phase 7 calls
- `apply_transforms(spec.transforms, batch)` runs in `prepare.rs` step 3a, returning a transformed `RecordBatch` whose schema may have new columns (e.g., `density`, `bin_start`, `bin_end`). Encoding fields then reference those new columns.

### Phase 6 (Layout engine) — what Phase 7 calls
- `compute_layout(spec, theme, viewport, axes_input, facet_groups, legend_entries, &FontdueMetrics)` from inside `render_svg`/`render_png`.
- `LayoutResult.PanelLayout` is extended additively (new `strip_title` field). Phase 6 tests continue passing.
- The Phase 6 Python `compute_layout` binding is untouched (Decision §11 row 16).

### Phase 8 (Grammar API) — what it inherits from Phase 7
- `render_svg`/`render_png` are the production-path entry points; `chart.show_svg()` / `chart.show_png()` / `chart.save(path)` wrap them.
- `Theme` Python value class will translate to `ThemeInputs` at the binding boundary (currently a dict).
- `RenderConfig` Python value class will translate to the Rust `RenderConfig`; new fields (`engine`, `backend`, `raster_*`) will be added when their corresponding features ship.
- `EncodingSpec` extension pattern is established — Phase 8 adds `size`, `shape`, etc. additively.
- Composite marks (boxplot, errorbar) will desugar into multiple primitive marks at the grammar layer; Phase 7 stays single-mark-per-chart.
- May surface `OKABE_ITO` palette as one entry in a future scheme registry.

### Phase 11 (Interactive renderer) — what it does NOT inherit
- Different crate (`ferrum-wasm`), different draw path (wgpu instanced draws + tessellation). Phase 7's `render_svg`/`render_png` is not designed for reuse.
- May reuse `prepare_render_inputs` / `scale_resolve` / `format` / `color` / `palette` modules — those are pure-data and crate-portable.
- Will reuse `LayoutResult` (already pure-data per Phase 6 §12).

### Phase 12 (Extension points) — what it might want
- A public `MarkDraw` trait could be added over the sealed-enum dispatch without breaking existing marks. Not designed for now; revisited when an external use case lands.

### `ferrum-spec.md` §3.16 update
- Add a dated note: "**2026-05-09 (Phase 7):** Phase 7 honors `RenderConfig` fields `scale`, `embed_fonts`, `background`, `width`, `height`. `engine`, `backend`, `raster_threshold`, `raster_behavior`, `raster_aggregate`, `raster_cmap`, `tile_parallel`, `font_path` are deferred to subsequent phases. `embed_fonts` is treated as always-true in Phase 7 for visual determinism; future phases may surface the false case."

---

## §13 Test count baseline at HEAD (before Phase 7 work)

- `cargo test -p ferrum-core`: 178 passing
- `uv run pytest`: 78 passing

---

## §14 Spec refinements (post-approval, plan-stage)

This section is reserved for refinements that surface during plan drafting (per Phase 6's §14 precedent). Items added here resolve under-specified inputs without changing scope or any locked decision in §11.

*(empty at spec-write time — populate during implementation planning.)*
