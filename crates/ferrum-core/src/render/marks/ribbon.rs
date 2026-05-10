//! Ribbon mark: closed area between y(x) and y2(x) along a shared x. Used by
//! `mark_errorband`, `mark_smooth(ci=...)`, and the QQ confidence-interval region.
//!
//! Path emission (spec §4.2.4):
//!     M x[0] y[0]
//!     L x[1] y[1] ... L x[n-1] y[n-1]
//!     L x[n-1] y2[n-1] L x[n-2] y2[n-2] ... L x[0] y2[0]
//!     Z
//!
//! Walks x ascending on the top edge (y), then x descending on the bottom edge
//! (y2) to close the polygon. Rows with non-finite x/y/y2 are dropped.
//!
//! When `encoding.y2` is unset the drawer silently skips — ribbon requires y2.

use crate::render::draw::{col_as_f64, x_field, y_field, DrawCtx};
use crate::render::svg::{fmt_f, FillStroke, SvgBuffer};

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b),
        _ => return,
    };
    // Ribbon REQUIRES y2; missing y2 -> silent skip (Python layer warns).
    let y2f = match spec.encoding.y2.as_ref() {
        Some(e) => e.field.as_str(),
        None => return,
    };

    let xs = match col_as_f64(ctx.batch, xf) {
        Ok(v) => v,
        Err(_) => return,
    };
    let ys = match col_as_f64(ctx.batch, yf) {
        Ok(v) => v,
        Err(_) => return,
    };
    let y2s = match col_as_f64(ctx.batch, y2f) {
        Ok(v) => v,
        Err(_) => return,
    };
    let n = xs.len().min(ys.len()).min(y2s.len());

    // Sort row indices by x ascending; drop rows with any null/NaN in x/y/y2.
    let mut indices: Vec<usize> = (0..n)
        .filter(|&i| {
            matches!((xs[i], ys[i], y2s[i]),
                (Some(a), Some(b), Some(c)) if a.is_finite() && b.is_finite() && c.is_finite())
        })
        .collect();
    indices.sort_by(|&a, &b| {
        let xa = xs[a].unwrap_or(f64::NAN);
        let xb = xs[b].unwrap_or(f64::NAN);
        xa.partial_cmp(&xb).unwrap_or(std::cmp::Ordering::Equal)
    });
    if indices.len() < 2 {
        return;
    }

    // Map data values to pixel coordinates; drop any row that fails scale projection.
    let pixels: Vec<(f64, f64, f64)> = indices
        .iter()
        .filter_map(|&i| {
            let xv = xs[i]?;
            let yv = ys[i]?;
            let y2v = y2s[i]?;
            let cx = ctx.scales.x.to_pixel_f64(xv)?;
            let cy = ctx.scales.y.to_pixel_f64(yv)?;
            let cy2 = ctx.scales.y.to_pixel_f64(y2v)?;
            Some((cx, cy, cy2))
        })
        .collect();
    if pixels.len() < 2 {
        return;
    }

    // Build closed path: top edge x ascending, bottom edge x descending, Z.
    let mut d = String::new();
    let (x0, y0, _) = pixels[0];
    d.push_str(&format!("M{} {}", fmt_f(x0), fmt_f(y0)));
    for &(x, y, _) in &pixels[1..] {
        d.push_str(&format!(" L{} {}", fmt_f(x), fmt_f(y)));
    }
    for &(x, _, y2) in pixels.iter().rev() {
        d.push_str(&format!(" L{} {}", fmt_f(x), fmt_f(y2)));
    }
    d.push_str(" Z");

    out.path(
        &d,
        &FillStroke {
            fill: Some(ctx.mark_style.fill),
            stroke: ctx.mark_style.stroke,
            stroke_width: ctx.mark_style.stroke_width,
        },
    );
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
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn ribbon_spec(with_y2: bool) -> ChartSpec {
        ChartSpec {
            data: DataRef::default(),
            mark: Mark::Ribbon,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                y2: if with_y2 {
                    Some(EncodingSpec { field: "y2".into(), type_: None, ..Default::default() })
                } else {
                    None
                },
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
        position: None,
        }
    }

    fn three_col_batch(xs: Vec<f64>, ys: Vec<f64>, y2s: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
                Arc::new(Float64Array::from(y2s)),
            ],
        )
        .unwrap()
    }

    fn two_col_batch(xs: Vec<f64>, ys: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(xs)),
                Arc::new(Float64Array::from(ys)),
            ],
        )
        .unwrap()
    }

    fn render(spec: &ChartSpec, batch: &RecordBatch) -> String {
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None,
        };
        let (scales, _) = resolve_scales(
            spec,
            batch,
            (0.0, 100.0),
            (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Ribbon);
        let ctx = DrawCtx {
            spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch,
            mark_style: &mark_style,
        };
        let mut out = SvgBuffer::new(panel.plot_area, None, false);
        super::draw(&ctx, &mut out);
        out.finish()
    }

    #[test]
    fn ribbon_emits_closed_path() {
        let spec = ribbon_spec(true);
        // Auto-domain is computed from `y` only; keep y2 values within [min(y), max(y)]
        // to avoid out-of-range scale projections in this isolated unit test.
        let batch = three_col_batch(
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![0.0, 2.0, 4.0, 6.0, 8.0],
            vec![1.0, 3.0, 5.0, 7.0, 8.0],
        );
        let svg = render(&spec, &batch);
        assert_eq!(svg.matches("<path ").count(), 1, "svg: {svg}");
        // Path d-attribute starts with M and ends with Z.
        let path_idx = svg.find("d=\"").expect("path d= attr");
        let after = &svg[path_idx + 3..];
        assert!(after.starts_with('M'), "d-attr should start with M: {after}");
        assert!(svg.contains(" Z\""), "ribbon path must close with Z: {svg}");
    }

    #[test]
    fn ribbon_walks_x_ascending_then_descending() {
        let spec = ribbon_spec(true);
        // 3 rows: top edge has 3 vertices (M + 2 L), bottom edge reversed adds 3 more L.
        // Expect 5 ` L` substrings total in the path d-attribute.
        // y values must vary so the auto-domain is non-degenerate; y2 stays inside [min(y), max(y)].
        let batch = three_col_batch(
            vec![0.0, 5.0, 10.0],
            vec![0.0, 4.0, 10.0],
            vec![2.0, 6.0, 8.0],
        );
        let svg = render(&spec, &batch);
        let d_start = svg.find("d=\"").expect("d= attr") + 3;
        let d_end = svg[d_start..].find('"').expect("closing quote") + d_start;
        let d = &svg[d_start..d_end];
        // M + 2 top L + 3 bottom-reversed L + Z = 5 L commands.
        let l_count = d.matches(" L").count();
        assert_eq!(l_count, 5, "expected 5 L commands, got {l_count} in d={d:?}");
        // Ensure the bottom edge starts where the top ended (x descending after rightmost x).
        // The last top vertex has x=10; the next L should also have x=10 (same point, y=y2).
        // Splitting by ' L' yields ["M0 ...", "5 ...", "10 ...", "10 ...", "5 ...", "0 ..."].
        let parts: Vec<&str> = d.split(" L").collect();
        assert_eq!(parts.len(), 6, "expected 6 path segments, got {parts:?}");
        // parts[2] is the rightmost top vertex (x=10), parts[3] is rightmost bottom vertex (x=10).
        let top_right_x = parts[2].split_whitespace().next().unwrap();
        let bot_right_x = parts[3].split_whitespace().next().unwrap();
        assert_eq!(top_right_x, bot_right_x, "x-descend should start at rightmost top x");
    }

    #[test]
    fn ribbon_no_y2_silently_skips() {
        let spec = ribbon_spec(false);
        let batch = two_col_batch(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]);
        let svg = render(&spec, &batch);
        assert!(!svg.contains("<path "), "no path should be emitted without y2: {svg}");
    }
}
