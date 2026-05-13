//! Ribbon mark: closed area between y(x) and y2(x) along a shared x. Used by
//! `mark_errorband`, `mark_smooth(ci=...)`, `mark_learning_curve` /
//! `mark_validation_curve` (with pre-aggregated CI columns), and the QQ
//! confidence-interval region.
//!
//! Path emission (spec §4.2.4), per color group when a color encoding is bound:
//!     M x[0] y[0]
//!     L x[1] y[1] ... L x[n-1] y[n-1]
//!     L x[n-1] y2[n-1] L x[n-2] y2[n-2] ... L x[0] y2[0]
//!     Z
//!
//! Walks x ascending on the top edge (y), then x descending on the bottom edge
//! (y2) to close the polygon. When a color encoding is bound, rows are first
//! partitioned by color category and each group emits its own closed polygon
//! (mirroring `line.rs`); otherwise all rows form one polygon. Without
//! per-color grouping, multi-series CI bands (e.g. train vs test learning
//! curves) collapse into a single zigzag polygon stitching across categories.
//!
//! When `encoding.y2` is unset the drawer silently skips — ribbon requires y2.

use crate::render::draw::{col_as_f64, col_as_str, color_field, x_field, y_field, DrawCtx};
use crate::render::svg::{fmt_f, FillStroke, SvgBuffer};

fn resolve_x_pixels(ctx: &DrawCtx, xf: &str, n: usize) -> Option<(Vec<Option<f64>>, bool)> {
    if let Ok(xs_f) = col_as_f64(ctx.batch, xf) {
        let pixels: Vec<Option<f64>> = xs_f
            .into_iter()
            .take(n)
            .map(|v| v.and_then(|x| ctx.scales.x.to_pixel_f64(x)))
            .collect();
        return Some((pixels, false));
    }
    if let Ok(xs_s) = col_as_str(ctx.batch, xf) {
        let pixels: Vec<Option<f64>> = xs_s
            .into_iter()
            .take(n)
            .map(|v| v.as_deref().and_then(|s| ctx.scales.x.to_pixel_str(s)))
            .collect();
        return Some((pixels, true));
    }
    None
}

pub fn draw(ctx: &DrawCtx, out: &mut SvgBuffer) {
    let spec = ctx.spec;
    let (xf, yf) = match (x_field(ctx, spec), y_field(ctx, spec)) {
        (Some(a), Some(b)) => (a, b),
        _ => return,
    };
    let y2f = match spec.encoding.y2.as_ref() {
        Some(e) => e.field.as_str(),
        None => return,
    };

    let ys = match col_as_f64(ctx.batch, yf) {
        Ok(v) => v,
        Err(_) => return,
    };
    let y2s = match col_as_f64(ctx.batch, y2f) {
        Ok(v) => v,
        Err(_) => return,
    };
    let n = ys.len().min(y2s.len());
    let (x_pixels, x_is_ordinal) = match resolve_x_pixels(ctx, xf, n) {
        Some(v) => v,
        None => return,
    };
    let n = n.min(x_pixels.len());

    // Partition rows by color category when a color encoding is bound; mirrors
    // line.rs so a multi-series ribbon (e.g. train + test CI bands sharing an x
    // axis) emits one closed polygon per series instead of a single zigzag
    // polygon stitching across category boundaries.
    let cf = color_field(ctx, spec);
    let color_values = cf.and_then(|f| col_as_str(ctx.batch, f).ok());
    let groups: Vec<(Option<String>, Vec<usize>)> = match (color_values.as_ref(), &ctx.scales.color) {
        (Some(values), Some(_)) => {
            let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, v) in values.iter().take(n).enumerate() {
                let key = v.clone();
                let pos = groups.iter().position(|(k, _)| k == &key);
                match pos {
                    Some(p) => groups[p].1.push(i),
                    None => groups.push((key, vec![i])),
                }
            }
            groups
        }
        _ => vec![(None, (0..n).collect())],
    };

    // Phase 9c — per-row pixel offsets (Stack/Dodge ordinal).
    let (x_offsets, y_offsets) = crate::render::position::read_position_offsets(ctx.batch);

    // Preserve the original single-series styling when there is no color
    // encoding (fill carries opacity, stroke is a solid color from the
    // resolved mark style) so existing one-series ribbon goldens remain
    // byte-identical. For multi-series ribbons, derive each group's fill
    // from the categorical scale lookup and multiply in the mark-style
    // fill alpha so opacity is preserved.
    let has_color_groups = color_values.is_some() && ctx.scales.color.is_some();
    let alpha_factor = (ctx.mark_style.fill.alpha as f64) / 255.0;

    for (key, rows) in groups {
        let mut indices: Vec<usize> = rows
            .into_iter()
            .filter(|&i| {
                let xp_ok = x_pixels.get(i).and_then(|v| *v).map(|p| p.is_finite()).unwrap_or(false);
                let yv_ok = ys.get(i).and_then(|v| *v).map(|v| v.is_finite()).unwrap_or(false);
                let y2v_ok = y2s.get(i).and_then(|v| *v).map(|v| v.is_finite()).unwrap_or(false);
                xp_ok && yv_ok && y2v_ok
            })
            .collect();
        if !x_is_ordinal {
            indices.sort_by(|&a, &b| {
                let xa = x_pixels[a].unwrap_or(f64::NAN);
                let xb = x_pixels[b].unwrap_or(f64::NAN);
                xa.partial_cmp(&xb).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        if indices.len() < 2 {
            continue;
        }
        let pixels: Vec<(f64, f64, f64)> = indices
            .iter()
            .filter_map(|&i| {
                let cx = x_pixels.get(i).and_then(|v| *v)?;
                let yv = ys[i]?;
                let y2v = y2s[i]?;
                let cy = ctx.scales.y.to_pixel_f64(yv)?;
                let cy2 = ctx.scales.y.to_pixel_f64(y2v)?;
                let xo = x_offsets.get(i).copied().unwrap_or(0.0);
                let yo = y_offsets.get(i).copied().unwrap_or(0.0);
                Some((cx + xo, cy + yo, cy2 + yo))
            })
            .collect();
        if pixels.len() < 2 {
            continue;
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

        // Color resolution.
        // - No color encoding: pass through the resolved mark style verbatim
        //   so single-series ribbons stay byte-identical to pre-Phase-10e
        //   goldens (mark_style.fill already carries the ribbon opacity;
        //   mark_style.stroke is the solid edge color).
        // - With color encoding: look up the group's solid color from the
        //   categorical scale, then multiply in the mark-style fill alpha
        //   so each per-group fill carries the same opacity as the
        //   pre-resolved single-series fill.
        let (fill, stroke) = if has_color_groups {
            let solid = key
                .as_deref()
                .and_then(|v| ctx.scales.color.as_ref().and_then(|s| s.lookup(v)))
                .unwrap_or(ctx.mark_style.fill);
            let fill = crate::render::color::categorical::with_opacity(solid, alpha_factor);
            (Some(fill), ctx.mark_style.stroke)
        } else {
            (Some(ctx.mark_style.fill), ctx.mark_style.stroke)
        };
        out.path(
            &d,
            &FillStroke {
                fill,
                stroke,
                stroke_width: ctx.mark_style.stroke_width,
            },
        );
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
        title: None,
        axis_x: None, axis_y: None,
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
    fn ribbon_emits_one_polygon_per_color_group() {
        // Two interleaved series (A,B,A,B,A,B) sharing an x axis. Without
        // color grouping the renderer would stitch a single zigzag polygon
        // across both categories — regression fixed by the per-color
        // partitioning in `draw`.
        let mut spec = ribbon_spec(true);
        spec.encoding.color = Some(EncodingSpec {
            field: "g".into(),
            type_: None,
            ..Default::default()
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("y2", DataType::Float64, false),
            Field::new("g", DataType::Utf8, false),
        ]));
        use arrow::array::StringArray;
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])),
                Arc::new(Float64Array::from(vec![0.0, 5.0, 2.0, 6.0, 4.0, 7.0])),
                Arc::new(Float64Array::from(vec![1.0, 6.0, 3.0, 7.0, 5.0, 8.0])),
                Arc::new(StringArray::from(vec!["A", "B", "A", "B", "A", "B"])),
            ],
        )
        .unwrap();
        let svg = render(&spec, &batch);
        assert_eq!(
            svg.matches("<path ").count(),
            2,
            "expected one polygon per color group, got: {svg}"
        );
    }

    #[test]
    fn ribbon_no_y2_silently_skips() {
        let spec = ribbon_spec(false);
        let batch = two_col_batch(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]);
        let svg = render(&spec, &batch);
        assert!(!svg.contains("<path "), "no path should be emitted without y2: {svg}");
    }
}
