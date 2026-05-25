//! Image mark renderer — two independent rendering paths:
//!
//! ## 1. URL-tile path (`mark_image` with a `url` encoding channel)
//!
//! Triggered when `ctx.spec.encoding.url.is_some()`.  Each row in the batch
//! becomes one image tile:
//!
//! - `x`, `y` columns (Float64) — required; rows with nulls are skipped.
//! - The field named by `encoding.url` (Utf8) — holds a `data:<mime>;base64,<payload>`
//!   URL.  Rows with nulls or malformed URLs are skipped silently.
//! - Optional `width`, `height` columns (Float64) for per-row pixel sizing.
//!   When absent, `mark_style.width` / `mark_style.height` are used, defaulting
//!   to 32.0 × 32.0 pixels.
//!
//! Each tile is center-anchored at `(px, py)` — consistent with `mark_point`.
//!
//! ## 2. Raster path (`mark_raster` composite desugaring)
//!
//! Triggered when `ctx.spec.encoding.url.is_none()`.  Reads the single-row
//! Raster transform output batch (columns: `x_min`, `x_max`, `y_min`, `y_max`,
//! `width` (UInt32), `height` (UInt32), `pixel_data` (Binary)).  Applies a
//! colormap and emits one PNG-encoded `SceneNode::Image` spanning the plot
//! extent.
//!
//! Colormap resolution priority (three-step):
//!   1. `mark_style.cmap` name (explicit kwarg on the mark)
//!   2. `theme.sequential_scheme` (theme default)
//!   3. Viridis fallback

use arrow::array::{BinaryArray, Float64Array, UInt32Array};

use crate::render::color::{ContinuousScheme, NamedContinuous};
use crate::render::draw::{col_as_f64, col_as_str, DrawCtx};
use crate::render::rasterize::encode_png;

pub fn build(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::MarkBuildResult;
    use ferrum_scene::MarkBatchKind;

    let empty = || MarkBuildResult::empty(MarkBatchKind::Image);

    // Both paths are Cartesian-only — return empty for Polar/Geo coords.
    match &ctx.spec.coord {
        Some(crate::spec::coord::CoordKind::Polar { .. }) |
        Some(crate::spec::coord::CoordKind::Geo { .. }) => return empty(),
        _ => {}
    }

    // ── URL-tile path ──────────────────────────────────────────────────────
    if let Some(url_enc) = ctx.spec.encoding.url.as_ref() {
        return build_url_tiles(ctx, url_enc.field.as_str());
    }

    // ── Raster path ────────────────────────────────────────────────────────
    build_raster(ctx)
}

/// URL-tile path: one `SceneNode::Image` per row, positioned via x/y scales.
fn build_url_tiles(ctx: &DrawCtx, url_field: &str) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::MarkBuildResult;
    use ferrum_scene::{ImageData, MarkBatchKind, SceneNode};

    let empty = || MarkBuildResult::empty(MarkBatchKind::Image);

    let batch = ctx.batch;
    let n = batch.num_rows();
    if n == 0 {
        return empty();
    }

    // Require x and y fields from the encoding.
    let x_field = match ctx.spec.encoding.x.as_ref() {
        Some(e) => e.field.as_str(),
        None => return empty(),
    };
    let y_field = match ctx.spec.encoding.y.as_ref() {
        Some(e) => e.field.as_str(),
        None => return empty(),
    };

    let xs = match col_as_f64(batch, x_field) {
        Ok(v) => v,
        Err(_) => return empty(),
    };
    let ys = match col_as_f64(batch, y_field) {
        Ok(v) => v,
        Err(_) => return empty(),
    };
    let urls = match col_as_str(batch, url_field) {
        Ok(v) => v,
        Err(_) => return empty(),
    };

    // Optional per-row width/height columns (Float64). If absent, fall back
    // to mark_style.width / mark_style.height, then to 32.0 pixels.
    let widths: Option<Vec<Option<f64>>> = col_as_f64(batch, "width").ok();
    let heights: Option<Vec<Option<f64>>> = col_as_f64(batch, "height").ok();
    let default_w = ctx.mark_style.width.unwrap_or(32.0);
    let default_h = ctx.mark_style.height.unwrap_or(32.0);

    let mut nodes: Vec<SceneNode> = Vec::with_capacity(n);
    let mut data_indices: Vec<usize> = Vec::with_capacity(n);

    for i in 0..n {
        // Skip nulls in required fields.
        let x_val = match xs.get(i).and_then(|v| *v) {
            Some(v) => v,
            None => continue,
        };
        let y_val = match ys.get(i).and_then(|v| *v) {
            Some(v) => v,
            None => continue,
        };
        let url_val = match urls.get(i).and_then(|v| v.as_deref()) {
            Some(s) => s,
            None => continue,
        };

        // Must be a data URL: "data:<mime>;base64,<payload>"
        if !url_val.starts_with("data:") {
            continue;
        }

        // Map data coordinates to pixel space.
        let px = match ctx.scales.x.to_pixel_f64(x_val) {
            Some(p) => p,
            None => continue,
        };
        let py = match ctx.scales.y.to_pixel_f64(y_val) {
            Some(p) => p,
            None => continue,
        };

        // Per-row tile dimensions, falling back to mark_style / default.
        let tile_w = widths.as_ref()
            .and_then(|v| v.get(i).and_then(|v| *v))
            .unwrap_or(default_w);
        let tile_h = heights.as_ref()
            .and_then(|v| v.get(i).and_then(|v| *v))
            .unwrap_or(default_h);

        // Center-anchor: top-left corner of the <image> element.
        let img_x = px - tile_w / 2.0;
        let img_y = py - tile_h / 2.0;

        nodes.push(SceneNode::Image {
            x: img_x,
            y: img_y,
            w: tile_w,
            h: tile_h,
            data: ImageData::Url { url: url_val.to_string() },
        });
        data_indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Image,
        nodes,
        data_indices: Some(data_indices),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    }
}

/// Raster path: single-row Raster transform output → one colormap-rendered PNG.
fn build_raster(ctx: &DrawCtx) -> crate::render::draw::MarkBuildResult {
    use crate::render::draw::MarkBuildResult;
    use ferrum_scene::{ImageData, ImageMime, MarkBatchKind, SceneNode};

    let empty = || MarkBuildResult::empty(MarkBatchKind::Image);

    let batch = ctx.batch;
    if batch.num_rows() != 1 {
        return empty();
    }

    let x_min = match batch
        .column_by_name("x_min")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
    {
        Some(a) => a.value(0),
        None => return empty(),
    };
    let x_max = match batch
        .column_by_name("x_max")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
    {
        Some(a) => a.value(0),
        None => return empty(),
    };
    let y_min = match batch
        .column_by_name("y_min")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
    {
        Some(a) => a.value(0),
        None => return empty(),
    };
    let y_max = match batch
        .column_by_name("y_max")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
    {
        Some(a) => a.value(0),
        None => return empty(),
    };
    let width = match batch
        .column_by_name("width")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
    {
        Some(a) => a.value(0),
        None => return empty(),
    };
    let height = match batch
        .column_by_name("height")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
    {
        Some(a) => a.value(0),
        None => return empty(),
    };
    let pixel_bytes = match batch
        .column_by_name("pixel_data")
        .and_then(|c| c.as_any().downcast_ref::<BinaryArray>())
    {
        Some(a) => a.value(0),
        None => return empty(),
    };

    let n_cells = (width as usize) * (height as usize);
    if pixel_bytes.len() != n_cells * 8 {
        return empty();
    }

    // Resolve colormap: mark_style.cmap → theme.sequential_scheme → Viridis.
    let named = ctx
        .mark_style
        .cmap
        .as_deref()
        .and_then(NamedContinuous::from_name)
        .unwrap_or_else(|| {
            NamedContinuous::from_name(&ctx.theme.sequential_scheme)
                .unwrap_or(NamedContinuous::Viridis)
        });
    let scheme = ContinuousScheme::Named(named);

    // Decode f64 values and map through colormap to produce RGBA8 pixels.
    let mut rgba: Vec<u8> = Vec::with_capacity(n_cells * 4);
    for chunk in pixel_bytes.chunks_exact(8) {
        let v = f64::from_le_bytes(chunk.try_into().expect("invariant: chunks_exact(8) guarantees 8-byte slices"));
        if v.is_nan() {
            rgba.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            let c = scheme.sample(v);
            rgba.push(c.red);
            rgba.push(c.green);
            rgba.push(c.blue);
            rgba.push(c.alpha);
        }
    }

    let png_bytes = encode_png(width, height, &rgba);

    // Map data extent to pixel space via scales.
    let svg_x = match ctx.scales.x.to_pixel_f64(x_min) {
        Some(p) => p,
        None => return empty(),
    };
    let svg_x_far = match ctx.scales.x.to_pixel_f64(x_max) {
        Some(p) => p,
        None => return empty(),
    };
    let svg_y_top = match ctx.scales.y.to_pixel_f64(y_max) {
        Some(p) => p,
        None => return empty(),
    };
    let svg_y_bot = match ctx.scales.y.to_pixel_f64(y_min) {
        Some(p) => p,
        None => return empty(),
    };
    let svg_w = (svg_x_far - svg_x).abs();
    let svg_h = (svg_y_bot - svg_y_top).abs();

    MarkBuildResult {
        kind: MarkBatchKind::Image,
        nodes: vec![SceneNode::Image {
            x: svg_x,
            y: svg_y_top,
            w: svg_w,
            h: svg_h,
            data: ImageData::Inline {
                bytes: png_bytes,
                mime: ImageMime::Png,
            },
        }],
        data_indices: Some(vec![0]),
        tooltips: None,
        hrefs: None,
        descriptions: None,
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
    use crate::spec::encoding::{Encoding, EncodingSpec, ScaleSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{BinaryArray, Float64Array, StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    /// Build an image-mark ChartSpec whose x/y encoding fields point at the
    /// Raster output's `x_min`/`y_min` columns. Explicit linear scale domains
    /// are supplied because the single-row Raster batch is degenerate for
    /// auto-domain inference.
    fn image_spec(x_domain: (f64, f64), y_domain: (f64, f64)) -> ChartSpec {
        let x_scale = ScaleSpec::Linear {
            common: crate::spec::encoding::ContinuousScaleCommon {
                domain: Some(vec![x_domain.0, x_domain.1]),
                range: None, clamp: false, padding: None,
            },
            nice: false, zero: false,
        };
        let y_scale = ScaleSpec::Linear {
            common: crate::spec::encoding::ContinuousScaleCommon {
                domain: Some(vec![y_domain.0, y_domain.1]),
                range: None, clamp: false, padding: None,
            },
            nice: false, zero: false,
        };
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Image,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "x_min".into(), type_: None,
                    scale: Some(x_scale), ..Default::default()
                }),
                y: Some(EncodingSpec {
                    field: "y_min".into(), type_: None,
                    scale: Some(y_scale), ..Default::default()
                }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        title: None,
        axis_x: None, axis_y: None,
        selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        }
    }

    /// Build an image-mark ChartSpec with x/y/url encodings for URL-tile tests.
    fn url_tile_spec(x_domain: (f64, f64), y_domain: (f64, f64)) -> ChartSpec {
        let x_scale = ScaleSpec::Linear {
            common: crate::spec::encoding::ContinuousScaleCommon {
                domain: Some(vec![x_domain.0, x_domain.1]),
                range: None, clamp: false, padding: None,
            },
            nice: false, zero: false,
        };
        let y_scale = ScaleSpec::Linear {
            common: crate::spec::encoding::ContinuousScaleCommon {
                domain: Some(vec![y_domain.0, y_domain.1]),
                range: None, clamp: false, padding: None,
            },
            nice: false, zero: false,
        };
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Image,
            encoding: Encoding {
                x: Some(EncodingSpec {
                    field: "x".into(), type_: None,
                    scale: Some(x_scale), ..Default::default()
                }),
                y: Some(EncodingSpec {
                    field: "y".into(), type_: None,
                    scale: Some(y_scale), ..Default::default()
                }),
                url: Some(EncodingSpec {
                    field: "url".into(), type_: None,
                    ..Default::default()
                }),
                color: None,
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None, axis_y: None,
            selections: Vec::new(), conditionals: Vec::new(),
        chart_description: None,
        }
    }

    /// Build a single-row Raster output batch.
    fn raster_batch(
        x_min: f64, x_max: f64, y_min: f64, y_max: f64,
        width: u32, height: u32, pixel_bytes: Vec<u8>,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x_min", DataType::Float64, false),
            Field::new("x_max", DataType::Float64, false),
            Field::new("y_min", DataType::Float64, false),
            Field::new("y_max", DataType::Float64, false),
            Field::new("width", DataType::UInt32, false),
            Field::new("height", DataType::UInt32, false),
            Field::new("pixel_data", DataType::Binary, false),
        ]));
        let pixel_array = BinaryArray::from(vec![pixel_bytes.as_slice()]);
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![x_min])),
                Arc::new(Float64Array::from(vec![x_max])),
                Arc::new(Float64Array::from(vec![y_min])),
                Arc::new(Float64Array::from(vec![y_max])),
                Arc::new(UInt32Array::from(vec![width])),
                Arc::new(UInt32Array::from(vec![height])),
                Arc::new(pixel_array),
            ],
        )
        .unwrap()
    }

    /// Build a multi-row URL-tile batch.
    fn url_tile_batch(xs: Vec<f64>, ys: Vec<f64>, urls: Vec<&str>) -> RecordBatch {
        let n = xs.len();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("url", DataType::Utf8, false),
        ]));
        let url_strs: Vec<&str> = urls;
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(StringArray::from(url_strs)),
            ],
        )
        .unwrap()
    }

    fn unit_panel() -> PanelLayout {
        PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None,
        }
    }

    #[test]
    fn image_smoke_emits_data_url() {
        // 2x2 grid: 4 f64 values packed as 8-byte LE blobs (32 bytes total).
        let values: Vec<f64> = vec![0.0, 0.33, 0.67, 1.0];
        let mut pixel_data: Vec<u8> = Vec::with_capacity(values.len() * 8);
        for v in &values {
            pixel_data.extend_from_slice(&v.to_le_bytes());
        }
        let spec = image_spec((0.0, 1.0), (0.0, 1.0));
        let batch = raster_batch(0.0, 1.0, 0.0, 1.0, 2, 2, pixel_data);
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image);
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Image { .. })).count(), 1,
            "expected an Image node");
        // Verify PNG data is present.
        let has_png = result.nodes.iter().any(|n| {
            if let ferrum_scene::SceneNode::Image { data, .. } = n {
                matches!(data, ferrum_scene::ImageData::Inline { mime: ferrum_scene::ImageMime::Png, .. })
            } else { false }
        });
        assert!(has_png, "expected inline PNG image data");
    }

    #[test]
    fn image_byte_size_correctness() {
        // 8x8 grid: 64 f64 values packed as 8-byte LE blobs (512 bytes total).
        let n_cells = 8 * 8;
        let mut pixel_data: Vec<u8> = Vec::with_capacity(n_cells * 8);
        for i in 0..n_cells {
            let v = (i as f64) / ((n_cells - 1) as f64);
            pixel_data.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(pixel_data.len(), 512);
        let spec = image_spec((0.0, 8.0), (0.0, 8.0));
        let batch = raster_batch(0.0, 8.0, 0.0, 8.0, 8, 8, pixel_data);
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image);
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx);
        assert_eq!(result.nodes.iter().filter(|n| matches!(n, ferrum_scene::SceneNode::Image { .. })).count(), 1,
            "expected exactly one Image node");
    }

    #[test]
    fn image_position_via_scales() {
        // Data extent x:[0,10], y:[0,10] → with linear scales mapping to [0,100]
        // pixel range, the image should cover the full plot area.
        // 1x1 grid: one f64 value (1.0 = fully saturated) as 8-byte LE blob.
        let pixel_data: Vec<u8> = 1.0_f64.to_le_bytes().to_vec();
        let spec = image_spec((0.0, 10.0), (0.0, 10.0));
        let batch = raster_batch(0.0, 10.0, 0.0, 10.0, 1, 1, pixel_data);
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image);
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = super::build(&ctx);
        // With scales mapping data [0,10] -> pixel [0,100], the image should span
        // x=0, w=100, h=100.
        let img = result.nodes.iter().find(|n| matches!(n, ferrum_scene::SceneNode::Image { .. }));
        assert!(img.is_some(), "expected Image node");
        if let Some(ferrum_scene::SceneNode::Image { x, y, w, h, .. }) = img {
            assert!((*x - 0.0).abs() < 1.0, "expected x near 0, got {x}");
            assert!((*y - 0.0).abs() < 1.0, "expected y near 0, got {y}");
            assert!((*w - 100.0).abs() < 1.0, "expected w near 100, got {w}");
            assert!((*h - 100.0).abs() < 1.0, "expected h near 100, got {h}");
        }
    }

    // ── URL-tile path tests ────────────────────────────────────────────────

    /// Minimal 1×1 red PNG as a base64 data URL.
    fn red_1x1_png_url() -> String {
        // Minimal valid 1x1 RGB PNG (no alpha).
        let png_bytes: Vec<u8> = vec![
            137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,1,0,0,0,1,8,2,0,0,0,
            144,119,83,222,0,0,0,12,73,68,65,84,8,215,99,248,207,192,0,0,0,2,0,1,
            227,33,188,51,0,0,0,0,73,69,78,68,174,66,96,130,
        ];
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
        format!("data:image/png;base64,{b64}")
    }

    #[test]
    fn url_tile_emits_one_image_per_row() {
        let spec = url_tile_spec((0.0, 6.0), (0.0, 6.0));
        let url = red_1x1_png_url();
        let urls = vec![url.as_str(), url.as_str(), url.as_str()];
        let batch = url_tile_batch(
            vec![1.0, 3.0, 5.0],
            vec![2.0, 4.0, 2.0],
            urls,
        );
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image);
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = build(&ctx);
        assert_eq!(
            result.nodes.len(), 3,
            "expected 3 Image nodes for 3 input rows, got {}", result.nodes.len()
        );
    }

    #[test]
    fn url_tile_uses_url_variant() {
        let spec = url_tile_spec((0.0, 6.0), (0.0, 6.0));
        let url = red_1x1_png_url();
        let batch = url_tile_batch(
            vec![1.0],
            vec![2.0],
            vec![url.as_str()],
        );
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image);
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = build(&ctx);
        assert_eq!(result.nodes.len(), 1, "expected 1 Image node");
        let has_url_variant = result.nodes.iter().any(|n| {
            matches!(n, ferrum_scene::SceneNode::Image {
                data: ferrum_scene::ImageData::Url { .. }, ..
            })
        });
        assert!(has_url_variant, "expected ImageData::Url variant for URL-tile path");
    }

    #[test]
    fn url_tile_center_anchored() {
        // With x=3.0, domain [0,6] → pixel 50.0; default tile size 32×32
        // → top-left should be at pixel (50 - 16, 50 - 16) = (34, 34).
        let spec = url_tile_spec((0.0, 6.0), (0.0, 6.0));
        let url = red_1x1_png_url();
        let batch = url_tile_batch(vec![3.0], vec![3.0], vec![url.as_str()]);
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image);
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = build(&ctx);
        assert_eq!(result.nodes.len(), 1);
        if let ferrum_scene::SceneNode::Image { x, y, w, h, .. } = &result.nodes[0] {
            // Center point is at pixel 50; tile half-size is 16.
            assert!((*x - 34.0).abs() < 1.0, "expected x≈34 (center-anchored), got {x}");
            assert!((*y - 34.0).abs() < 1.0, "expected y≈34 (center-anchored), got {y}");
            assert!((*w - 32.0).abs() < 1.0, "expected w=32 (default), got {w}");
            assert!((*h - 32.0).abs() < 1.0, "expected h=32 (default), got {h}");
        } else {
            panic!("expected Image node");
        }
    }

    #[test]
    fn url_tile_skips_non_data_url_rows() {
        let spec = url_tile_spec((0.0, 6.0), (0.0, 6.0));
        let url = red_1x1_png_url();
        let batch = url_tile_batch(
            vec![1.0, 3.0, 5.0],
            vec![2.0, 4.0, 2.0],
            // Second row has a plain URL (not a data URL) → should be skipped.
            vec![url.as_str(), "https://example.com/img.png", url.as_str()],
        );
        let theme = ThemeInputs::default();
        let panel = unit_panel();
        let (scales, _) = resolve_scales(&spec, &batch, (0.0, 100.0), (0.0, 100.0), &theme).unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Image);
        let ctx = DrawCtx {
            spec: &spec, panel: &panel, theme: &theme,
            scales: &scales, batch: &batch, mark_style: &mark_style,
        };
        let result = build(&ctx);
        assert_eq!(result.nodes.len(), 2, "plain URL row should be silently skipped");
    }
}
