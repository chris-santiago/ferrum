//! Image mark drawer: decode normalized f64 cell values from a single-row
//! Raster batch, apply a colormap, encode PNG, and embed via SvgBuffer.image().
//!
//! The Raster transform emits a single-row batch with columns:
//!   `x_min, x_max, y_min, y_max` (Float64), `width, height` (UInt32),
//!   `pixel_data` (Binary, normalized f64 values packed as 8-byte LE blobs,
//!                  length = width*height*8).
//!
//! Colormap resolution priority (three-step, same as polygon.rs):
//!   1. `mark_style.cmap` name (explicit kwarg on the mark)
//!   2. `theme.sequential_scheme` (theme default)
//!   3. Viridis fallback
//!
//! Each f64 in [0.0, 1.0] is sampled through `ContinuousScheme::sample(v)`
//! to produce an RGBA8 pixel, which is then PNG-encoded and embedded.

use arrow::array::{BinaryArray, Float64Array, UInt32Array};

use crate::render::color::{ContinuousScheme, NamedContinuous};
use crate::render::draw::DrawCtx;
use crate::render::rasterize::encode_png;
use crate::render::svg::SvgBuffer;

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let batch = ctx.batch;
    if batch.num_rows() != 1 {
        return; // image expects single-row Raster output
    }

    let x_min = match batch
        .column_by_name("x_min")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
    {
        Some(a) => a.value(0),
        None => return,
    };
    let x_max = match batch
        .column_by_name("x_max")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
    {
        Some(a) => a.value(0),
        None => return,
    };
    let y_min = match batch
        .column_by_name("y_min")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
    {
        Some(a) => a.value(0),
        None => return,
    };
    let y_max = match batch
        .column_by_name("y_max")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
    {
        Some(a) => a.value(0),
        None => return,
    };
    let width = match batch
        .column_by_name("width")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
    {
        Some(a) => a.value(0),
        None => return,
    };
    let height = match batch
        .column_by_name("height")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
    {
        Some(a) => a.value(0),
        None => return,
    };
    let pixel_bytes = match batch
        .column_by_name("pixel_data")
        .and_then(|c| c.as_any().downcast_ref::<BinaryArray>())
    {
        Some(a) => a.value(0),
        None => return,
    };

    let n_cells = (width as usize) * (height as usize);
    if pixel_bytes.len() != n_cells * 8 {
        // Expect normalized f64 (8 bytes/cell). Skip silently on malformed input.
        return;
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
        let v = f64::from_le_bytes(chunk.try_into().unwrap());
        let c = scheme.sample(v);
        rgba.push(c.red);
        rgba.push(c.green);
        rgba.push(c.blue);
        rgba.push(c.alpha);
    }

    let png_bytes = encode_png(width, height, &rgba);

    // Map data extent to pixel space via scales. SVG y axis is inverted vs data y,
    // so the top edge of the image corresponds to data y_max.
    let svg_x = match ctx.scales.x.to_pixel_f64(x_min) {
        Some(p) => p,
        None => return,
    };
    let svg_x_far = match ctx.scales.x.to_pixel_f64(x_max) {
        Some(p) => p,
        None => return,
    };
    let svg_y_top = match ctx.scales.y.to_pixel_f64(y_max) {
        Some(p) => p,
        None => return,
    };
    let svg_y_bot = match ctx.scales.y.to_pixel_f64(y_min) {
        Some(p) => p,
        None => return,
    };
    let svg_w = (svg_x_far - svg_x).abs();
    let svg_h = (svg_y_bot - svg_y_top).abs();

    out.image(svg_x, svg_y_top, svg_w, svg_h, &png_bytes);
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
    use arrow::array::{BinaryArray, Float64Array, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    /// Build an image-mark ChartSpec whose x/y encoding fields point at the
    /// Raster output's `x_min`/`y_min` columns. Explicit linear scale domains
    /// are supplied because the single-row Raster batch is degenerate for
    /// auto-domain inference.
    fn image_spec(x_domain: (f64, f64), y_domain: (f64, f64)) -> ChartSpec {
        let x_scale = ScaleSpec::Linear {
            domain: Some(vec![x_domain.0, x_domain.1]),
            range: None, nice: false, zero: false, clamp: false, padding: None,
        };
        let y_scale = ScaleSpec::Linear {
            domain: Some(vec![y_domain.0, y_domain.1]),
            range: None, nice: false, zero: false, clamp: false, padding: None,
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
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert!(
            s.contains(r#"<image "#),
            "expected an <image> element, got: {s}"
        );
        assert!(
            s.contains("href=\"data:image/png;base64,"),
            "expected a base64 PNG data URL href, got: {s}"
        );
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
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        assert!(s.contains("<image "), "expected <image> element: {s}");
        assert_eq!(s.matches("<image ").count(), 1, "expected exactly one <image> element");
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
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        let s = out.finish();
        // SvgBuffer.image emits attributes in order: x, y, width, height, href.
        // With scales mapping data [0,10] -> pixel [0,100], the image should span
        // x=0, width=100, height=100. The SVG y-axis is inverted (data y_max → svg
        // top), but for a domain spanning the full pixel range either edge yields
        // y=0 in pixel space.
        assert!(
            s.contains(r#"<image x="0" y="0" width="100" height="100" "#),
            "expected image positioned at (0,0) with 100x100 size, got: {s}"
        );
    }
}
