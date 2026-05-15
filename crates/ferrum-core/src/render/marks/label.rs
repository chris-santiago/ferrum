use ferrum_scene::{MarkBatchKind, SceneNode, StrokeStyle};

use crate::layout::TextAnchor;
use crate::render::arrow_cast::{col_as_f64, col_as_str};
use crate::render::color::with_opacity;
use crate::render::draw::{to_scene_color, to_scene_text_style, DrawCtx, MarkBuildResult};

/// Axis-aligned bounding box for overlap testing.
#[derive(Clone, Copy)]
struct Bbox {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

impl Bbox {
    /// Returns true when `self` and `other` overlap (touching edges do not count).
    fn overlaps(&self, other: &Bbox) -> bool {
        self.x < other.x + other.w
            && self.x + self.w > other.x
            && self.y < other.y + other.h
            && self.y + self.h > other.y
    }

    /// Number of already-placed boxes that overlap `self`.
    fn count_overlaps(&self, placed: &[Bbox]) -> usize {
        placed.iter().filter(|b| self.overlaps(b)).count()
    }
}

/// Candidate (dx, dy) offsets tested in preference order for each label.
///
/// Order: above (default), below, right, left, diagonal NE, diagonal NW,
/// diagonal SE, diagonal SW.  The first candidate with zero overlaps wins;
/// if all overlap something, the candidate with the fewest overlaps is chosen.
fn candidate_offsets(w: f64, h: f64) -> [(f64, f64); 8] {
    let hgap = w / 2.0 + 4.0; // horizontal gap when placing left/right
    let vgap_above = h + 4.0;  // full label height + small padding
    let vgap_below = 4.0;      // just below the point
    [
        (0.0, -(vgap_above)),            // above  (preferred, matches legacy default)
        (0.0, vgap_below + h),           // below
        (hgap, -h / 2.0),               // right
        (-hgap, -h / 2.0),              // left
        (hgap * 0.7, -(vgap_above * 0.7)),  // diagonal NE
        (-hgap * 0.7, -(vgap_above * 0.7)), // diagonal NW
        (hgap * 0.7, vgap_below + h * 0.7), // diagonal SE
        (-hgap * 0.7, vgap_below + h * 0.7),// diagonal SW
    ]
}

/// Build positioned text label nodes with greedy collision avoidance.
///
/// Each row emits one text label at (x, y) + a per-label (dx, dy) offset.
/// When `mark_style.dx` **and** `mark_style.dy` are both explicitly set by the
/// caller the collision avoidance is bypassed and those fixed offsets are used
/// for every label (manual override path).  When either or both are unset the
/// algorithm tries a ranked list of candidate offsets for each label in row
/// order, picking the first placement whose estimated bounding box overlaps no
/// previously-placed box; if every candidate overlaps something the one with
/// the fewest overlaps is chosen.
///
/// Bounding-box estimation uses `len(text) * font_size * 0.6` for width and
/// `font_size * 1.2` for height — fast, approximate, and good enough for the
/// greedy heuristic.
///
/// When `mark_style.leader_line = Some(true)` a thin `SceneNode::Line` is
/// drawn from the data point `(px, py)` to the placed label position.
pub fn build(ctx: &DrawCtx<'_>) -> MarkBuildResult {
    let xf = match ctx.spec.encoding.x.as_ref().map(|e| e.field.as_str()) {
        Some(f) => f,
        None => return MarkBuildResult::empty(MarkBatchKind::Text),
    };
    let yf = match ctx.spec.encoding.y.as_ref().map(|e| e.field.as_str()) {
        Some(f) => f,
        None => return MarkBuildResult::empty(MarkBatchKind::Text),
    };

    let Ok(xs) = col_as_f64(ctx.batch, xf) else {
        return MarkBuildResult::empty(MarkBatchKind::Text);
    };
    let Ok(ys) = col_as_f64(ctx.batch, yf) else {
        return MarkBuildResult::empty(MarkBatchKind::Text);
    };

    // Optional text content: `text` encoding field, else format x value.
    let texts: Vec<Option<String>> = if let Some(tf) = ctx.spec.encoding.text.as_ref().map(|e| e.field.as_str()) {
        col_as_str(ctx.batch, tf).unwrap_or_default()
    } else {
        xs.iter().map(|v| v.map(|f| format!("{:.2}", f))).collect()
    };

    // When the caller explicitly provides both dx AND dy, honour them as fixed
    // offsets and skip collision avoidance entirely (manual positioning path).
    let fixed_dx = ctx.mark_style.dx;
    let fixed_dy = ctx.mark_style.dy;
    let manual_override = fixed_dx.is_some() && fixed_dy.is_some();
    let fallback_dx = fixed_dx.unwrap_or(0.0);
    let fallback_dy = fixed_dy.unwrap_or(-8.0);

    let font_size = ctx.mark_style.font_size.unwrap_or(11.0);
    let color = with_opacity(ctx.mark_style.fill, ctx.mark_style.opacity);
    let draw_leader = ctx.mark_style.leader_line.unwrap_or(false);

    let n = ctx.batch.num_rows();
    let mut nodes: Vec<SceneNode> = Vec::with_capacity(n);
    let mut data_indices: Vec<usize> = Vec::with_capacity(n);

    // Placed bounding boxes accumulate as we emit each label; used only when
    // collision avoidance is active.
    let mut placed: Vec<Bbox> = if manual_override { Vec::new() } else { Vec::with_capacity(n) };

    for i in 0..n {
        let xv = match xs.get(i).and_then(|v| *v) { Some(v) => v, None => continue };
        let yv = match ys.get(i).and_then(|v| *v) { Some(v) => v, None => continue };
        let px = match ctx.scales.x.to_pixel_f64(xv) { Some(p) if p.is_finite() => p, _ => continue };
        let py = match ctx.scales.y.to_pixel_f64(yv) { Some(p) if p.is_finite() => p, _ => continue };
        let content = texts.get(i).and_then(|v| v.clone()).unwrap_or_default();

        // Estimated bounding-box dimensions in pixel space.
        let label_w = (content.len() as f64) * font_size * 0.6;
        let label_h = font_size * 1.2;

        let (dx, dy) = if manual_override {
            (fallback_dx, fallback_dy)
        } else {
            // Greedy placement: try each candidate offset in preference order.
            let candidates = candidate_offsets(label_w, label_h);
            let mut best_dx = fallback_dx;
            let mut best_dy = fallback_dy;
            let mut best_overlaps = usize::MAX;

            for &(cdx, cdy) in &candidates {
                // Bounding box origin at top-left corner of the label text.
                let bbox = Bbox {
                    x: px + cdx - label_w / 2.0,
                    y: py + cdy - label_h,
                    w: label_w,
                    h: label_h,
                };
                let overlaps = bbox.count_overlaps(&placed);
                if overlaps < best_overlaps {
                    best_overlaps = overlaps;
                    best_dx = cdx;
                    best_dy = cdy;
                    if overlaps == 0 {
                        break; // Can't do better.
                    }
                }
            }

            // Record the chosen placement so subsequent labels can avoid it.
            placed.push(Bbox {
                x: px + best_dx - label_w / 2.0,
                y: py + best_dy - label_h,
                w: label_w,
                h: label_h,
            });

            (best_dx, best_dy)
        };

        nodes.push(SceneNode::Text {
            x: px + dx,
            y: py + dy,
            content,
            style: to_scene_text_style(
                color, font_size, TextAnchor::Middle, 0.0,
                ctx.theme.font_family.as_str(),
                ctx.mark_style.font_weight.as_deref(), None, ctx.mark_style.opacity,
            ),
        });

        if draw_leader {
            let lc = with_opacity(ctx.mark_style.fill, ctx.mark_style.opacity * 0.5);
            nodes.push(SceneNode::Line {
                x1: px,
                y1: py,
                x2: px + dx * 0.8,
                y2: py + dy * 0.8,
                style: StrokeStyle {
                    color: to_scene_color(lc),
                    width: 0.75,
                    opacity: ctx.mark_style.opacity * 0.5,
                    dash: None,
                    stroke_cap: None,
                    stroke_join: None,
                },
            });
        }

        data_indices.push(i);
    }

    MarkBuildResult {
        kind: MarkBatchKind::Text,
        nodes,
        data_indices: Some(data_indices),
        tooltips: None,
        hrefs: None,
        descriptions: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{PanelLayout, Rect, ThemeInputs};
    use crate::render::draw::{resolve_mark_style, MarkStyle};
    use crate::render::scale_resolve::resolve_scales;
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Build a minimal two-point label spec + batch wired to x/y columns.
    fn two_point_spec_and_batch() -> (ChartSpec, arrow::record_batch::RecordBatch) {
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Label,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![10.0, 80.0])),
                Arc::new(Float64Array::from(vec![20.0, 70.0])),
            ],
        )
        .unwrap();
        (spec, batch)
    }

    // ── F3 regression tests ──────────────────────────────────────────

    /// When leader_line = Some(true), the output must contain at least one
    /// SceneNode::Line per label (the leader line from data point to label).
    #[test]
    fn leader_line_emitted_when_flag_is_true() {
        let (spec, batch) = two_point_spec_and_batch();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();

        // Build a MarkStyle with leader_line = Some(true).
        let mut mark_style = resolve_mark_style(None, &theme, &Mark::Label);
        mark_style.leader_line = Some(true);

        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };

        let result = build(&ctx);

        let line_count = result.nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count();
        assert!(
            line_count >= 1,
            "expected at least one SceneNode::Line leader line when leader_line=true, got {line_count} lines in {:?}",
            result.nodes.iter().map(std::mem::discriminant).collect::<Vec<_>>()
        );
    }

    /// When leader_line = None (default), the output must contain no
    /// SceneNode::Line nodes — only Text nodes are emitted.
    #[test]
    fn no_leader_line_when_flag_is_none() {
        let (spec, batch) = two_point_spec_and_batch();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();

        // Default resolve — leader_line stays None.
        let mark_style = resolve_mark_style(None, &theme, &Mark::Label);
        assert!(
            mark_style.leader_line.is_none(),
            "resolve_mark_style should leave leader_line as None by default"
        );

        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };

        let result = build(&ctx);

        let line_count = result.nodes.iter().filter(|n| matches!(n, SceneNode::Line { .. })).count();
        assert_eq!(
            line_count,
            0,
            "expected zero SceneNode::Line nodes when leader_line=None (default), got {line_count}"
        );

        // Must still emit Text nodes for the two data points.
        let text_count = result.nodes.iter().filter(|n| matches!(n, SceneNode::Text { .. })).count();
        assert_eq!(text_count, 2, "expected 2 Text nodes for 2-row batch, got {text_count}");
    }
}
