use ferrum_scene::{MarkBatchKind, SceneNode, StrokeStyle};

use crate::layout::TextAnchor;
use crate::render::arrow_cast::{col_as_f64, col_as_str};
use crate::render::color::with_opacity;
use crate::render::draw::{to_scene_color, to_scene_text_style, DrawCtx, MarkBuildResult, MetadataColumns};
use crate::render::mark_nodes::MarkNodes;

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
/// When `mark_style.text.dx` **and** `mark_style.text.dy` are both explicitly set by the
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
/// When `mark_style.text.leader_line = Some(true)` a thin `SceneNode::Line` is
/// drawn from the data point `(px, py)` to the placed label position.
pub fn build(ctx: &DrawCtx<'_>) -> MarkBuildResult {
    let xf = match ctx.spec.encoding.x.as_ref().map(|e| e.field.as_str()) {
        Some(f) => f,
        None => return MarkBuildResult::empty(MarkBatchKind::Label),
    };
    let yf = match ctx.spec.encoding.y.as_ref().map(|e| e.field.as_str()) {
        Some(f) => f,
        None => return MarkBuildResult::empty(MarkBatchKind::Label),
    };

    let Ok(xs) = col_as_f64(ctx.batch, xf) else {
        return MarkBuildResult::empty(MarkBatchKind::Label);
    };
    let Ok(ys) = col_as_f64(ctx.batch, yf) else {
        return MarkBuildResult::empty(MarkBatchKind::Label);
    };

    // Optional text content: `text` encoding field, else format x value.
    let texts: Vec<Option<String>> = if let Some(tf) = ctx.spec.encoding.text.as_ref().map(|e| e.field.as_str()) {
        col_as_str(ctx.batch, tf).unwrap_or_default()
    } else {
        xs.iter().map(|v| v.map(|f| format!("{:.2}", f))).collect()
    };

    // When the caller explicitly provides both dx AND dy, honour them as fixed
    // offsets and skip collision avoidance entirely (manual positioning path).
    let fixed_dx = ctx.mark_style.text.dx;
    let fixed_dy = ctx.mark_style.text.dy;
    let manual_override = fixed_dx.is_some() && fixed_dy.is_some();
    let fallback_dx = fixed_dx.unwrap_or(0.0);
    let fallback_dy = fixed_dy.unwrap_or(-8.0);

    let font_size = ctx.mark_style.text.font_size.unwrap_or(11.0);
    let color = with_opacity(ctx.mark_style.paint.fill, ctx.mark_style.paint.opacity);
    let draw_leader = ctx.mark_style.text.leader_line.unwrap_or(false);

    let n = ctx.batch.num_rows();

    // Pre-read metadata columns so tooltip/href/description reach every node
    // (both the Text and, when leader_line is true, the companion Line node).
    // build_metadata_for_indices is called after the loop using the finalized
    // data_indices, so metadata is in node order regardless of skipped rows or
    // multi-node-per-row shapes (archaeology bug #6 fix).
    let meta = MetadataColumns::from_ctx(ctx);

    // Accumulate nodes and source-row indices in lockstep. For rows without a
    // leader line, push one Text node. For rows with a leader line, push_many
    // emits [Text, Line] both tagged with the same source row i, keeping
    // data_indices[k] == source-row-of-nodes[k] for all k.
    // Capacity is 2× when leader lines are drawn (one text + one line per row).
    let capacity = if draw_leader { n * 2 } else { n };
    let mut acc = MarkNodes::with_capacity(capacity);

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

        let text_node = SceneNode::Text {
            x: px + dx,
            y: py + dy,
            content,
            slot: None,
            style: to_scene_text_style(
                color, font_size, TextAnchor::Middle, 0.0,
                ctx.theme.typography.font_family.as_str(),
                ctx.mark_style.text.font_weight.as_deref(), None, ctx.mark_style.paint.opacity,
            ),
        };

        if draw_leader {
            // Emit text + leader-line as a two-node group both mapped to row i.
            // push_many gives data_indices one entry per emitted node, so
            // nodes.len() == data_indices.len() is maintained (R1 guard enforces).
            let lc = with_opacity(ctx.mark_style.paint.fill, ctx.mark_style.paint.opacity * 0.5);
            let line_node = SceneNode::Line {
                x1: px,
                y1: py,
                x2: px + dx * 0.8,
                y2: py + dy * 0.8,
                style: StrokeStyle {
                    color: to_scene_color(lc),
                    width: 0.75,
                    opacity: ctx.mark_style.paint.opacity * 0.5,
                    stroke_opacity: 1.0,
                    dash: None,
                    stroke_cap: None,
                    stroke_join: None,
                },
            };
            acc.push_many([text_node, line_node], i);
        } else {
            acc.push(text_node, i);
        }
    }

    let (nodes, data_indices) = acc.finalize();
    let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);

    MarkBuildResult {
        kind: MarkBatchKind::Label,
        nodes,
        data_indices: Some(data_indices),
        tooltips,
        hrefs,
        descriptions,
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
            chart_description: None,
            params: Vec::new(),
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

    // ── Task 2: MarkBatchKind::Label tests ──────────────────────────────

    /// mark_label build produces MarkBuildResult with kind == MarkBatchKind::Label.
    #[test]
    fn label_build_produces_label_kind() {
        let (spec, batch) = two_point_spec_and_batch();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Label);
        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };
        let result = build(&ctx);
        assert_eq!(
            result.kind,
            MarkBatchKind::Label,
            "mark_label build must produce MarkBatchKind::Label, got {:?}",
            result.kind
        );
    }

    /// MarkBatchKind::Label serializes to "label" and round-trips through serde.
    #[test]
    fn label_kind_serde_round_trip() {
        let kind = MarkBatchKind::Label;
        let serialized = serde_json::to_string(&kind).expect("serialize MarkBatchKind::Label");
        assert_eq!(serialized, "\"label\"", "Label must serialize to \"label\"");
        let deserialized: MarkBatchKind =
            serde_json::from_str(&serialized).expect("deserialize MarkBatchKind::Label");
        assert_eq!(deserialized, MarkBatchKind::Label, "round-trip must restore Label");
    }

    // ── R2 regression tests ─────────────────────────────────────────

    /// Build a spec + batch that includes a tooltip encoding column alongside x/y.
    fn spec_and_batch_with_tooltip() -> (ChartSpec, arrow::record_batch::RecordBatch) {
        use crate::spec::encoding::EncodingSpec;
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Label,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                tooltip: Some(EncodingSpec { field: "tip".into(), type_: None, ..Default::default() }),
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
            chart_description: None,
            params: Vec::new(),
        };
        use arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("tip", arrow::datatypes::DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![10.0, 80.0])),
                Arc::new(Float64Array::from(vec![20.0, 70.0])),
                Arc::new(StringArray::from(vec!["tip-row-0", "tip-row-1"])),
            ],
        )
        .unwrap();
        (spec, batch)
    }

    /// R2 — Leader-line alignment: nodes.len() == data_indices.len() for a
    /// multi-row batch with leader_line=true.  Pre-fix: nodes.len()==2N,
    /// data_indices.len()==N (2 vs 1 for a single row, 4 vs 2 for two rows).
    #[test]
    fn r2_leader_line_nodes_and_data_indices_are_aligned() {
        let (spec, batch) = two_point_spec_and_batch();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();

        let mut mark_style = resolve_mark_style(None, &theme, &Mark::Label);
        mark_style.text.leader_line = Some(true);

        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };

        let result = build(&ctx);

        let n_nodes = result.nodes.len();
        let n_rows = batch.num_rows(); // 2

        // With leader_line=true: text + line per row → 2*n_rows total nodes.
        assert_eq!(
            n_nodes,
            n_rows * 2,
            "leader_line=true must produce 2 nodes per row; expected {}, got {}",
            n_rows * 2,
            n_nodes,
        );

        // The critical alignment invariant: data_indices must equal nodes.len().
        // Pre-fix: data_indices.len()==N, nodes.len()==2N → invariant violated.
        let data_indices = result.data_indices.as_ref().expect("data_indices must be Some");
        assert_eq!(
            data_indices.len(),
            n_nodes,
            "data_indices.len() ({}) must equal nodes.len() ({}) (R1 guard enforces this)",
            data_indices.len(),
            n_nodes,
        );

        // Each source row appears twice: once for the Text node, once for the Line.
        // For 2 rows the sequence should be [0, 0, 1, 1].
        assert_eq!(
            data_indices,
            &vec![0usize, 0, 1, 1],
            "data_indices must repeat each row index for its text+line pair",
        );
    }

    /// R2 — data_indices alignment is correct when no leader line is drawn: each
    /// row contributes exactly one Text node, so data_indices == [0, 1, ...].
    #[test]
    fn r2_no_leader_line_data_indices_one_per_row() {
        let (spec, batch) = two_point_spec_and_batch();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Label);

        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };

        let result = build(&ctx);
        let n_nodes = result.nodes.len();
        let data_indices = result.data_indices.as_ref().expect("data_indices must be Some");

        assert_eq!(
            data_indices.len(),
            n_nodes,
            "data_indices.len() must equal nodes.len() for no-leader-line label",
        );
        assert_eq!(data_indices, &vec![0usize, 1], "each row maps to exactly one node");
    }

    /// R2 — Metadata reaches Text node: a tooltip encoding column's value appears
    /// in the Text node's slot in the tooltip vector.  Pre-fix: tooltips was
    /// hardcoded to None so no tooltip ever reached any label node.
    #[test]
    fn r2_tooltip_reaches_text_node_without_leader_line() {
        let (spec, batch) = spec_and_batch_with_tooltip();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Label);

        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };

        let result = build(&ctx);

        // Tooltip must NOT be None after the fix.
        let tooltips = result
            .tooltips
            .as_ref()
            .expect("tooltips must be Some when tooltip encoding is present (pre-fix: None)");

        // One tooltip entry per emitted node (2 Text nodes for 2 rows).
        assert_eq!(tooltips.len(), 2, "one tooltip entry per emitted text node");

        // Node 0 (row 0) must carry "tip-row-0".
        assert_eq!(
            tooltips[0].fields[0].value,
            "tip-row-0",
            "node 0 must carry row 0 tooltip",
        );
        // Node 1 (row 1) must carry "tip-row-1".
        assert_eq!(
            tooltips[1].fields[0].value,
            "tip-row-1",
            "node 1 must carry row 1 tooltip",
        );
    }

    /// R2 — Metadata reaches both Text and Line nodes with leader_line=true: the
    /// tooltip vector has 2*N entries, with each row's tooltip duplicated for its
    /// text + line pair.
    #[test]
    fn r2_tooltip_reaches_both_nodes_with_leader_line() {
        let (spec, batch) = spec_and_batch_with_tooltip();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();
        let mut mark_style = resolve_mark_style(None, &theme, &Mark::Label);
        mark_style.text.leader_line = Some(true);

        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };

        let result = build(&ctx);
        let n_nodes = result.nodes.len();

        // 2 rows × 2 nodes per row = 4 nodes total.
        assert_eq!(n_nodes, 4);

        let tooltips = result
            .tooltips
            .as_ref()
            .expect("tooltips must be Some with tooltip encoding + leader_line");

        assert_eq!(tooltips.len(), n_nodes, "tooltip entry per emitted node");

        // Nodes [0, 1] correspond to row 0; nodes [2, 3] to row 1.
        assert_eq!(tooltips[0].fields[0].value, "tip-row-0", "node 0 (text, row 0)");
        assert_eq!(tooltips[1].fields[0].value, "tip-row-0", "node 1 (line, row 0)");
        assert_eq!(tooltips[2].fields[0].value, "tip-row-1", "node 2 (text, row 1)");
        assert_eq!(tooltips[3].fields[0].value, "tip-row-1", "node 3 (line, row 1)");
    }

    /// R2 backward compat: a label without leader_line and without tooltip/href
    /// still produces the same nodes as before (no metadata changes, same
    /// geometry).  This test fails if the migration accidentally changes geometry.
    #[test]
    fn r2_backward_compat_no_leader_no_metadata() {
        let (spec, batch) = two_point_spec_and_batch();
        let theme = ThemeInputs::default();
        let panel = PanelLayout {
            plot_area: Rect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            facet_key: None,
            row: 0,
            col: 0,
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();
        let mark_style = resolve_mark_style(None, &theme, &Mark::Label);

        let ctx = DrawCtx {
            spec: &spec,
            panel: &panel,
            theme: &theme,
            scales: &scales,
            batch: &batch,
            mark_style: &mark_style,
        };

        let result = build(&ctx);

        // 2 rows → 2 Text nodes, 0 Line nodes.
        assert_eq!(result.nodes.len(), 2);
        assert!(result.nodes.iter().all(|n| matches!(n, SceneNode::Text { .. })));
        // No metadata encoding → these remain None.
        assert!(result.tooltips.is_none(), "tooltips must be None when no tooltip encoding");
        assert!(result.hrefs.is_none(), "hrefs must be None when no href encoding");
        assert!(result.descriptions.is_none(), "descriptions must be None when no description encoding");
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
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();

        // Build a MarkStyle with leader_line = Some(true).
        let mut mark_style = resolve_mark_style(None, &theme, &Mark::Label);
        mark_style.text.leader_line = Some(true);

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
            strip_title: None, row_strip_title: None, row_facet_key: None,
        };
        let (scales, _) = resolve_scales(
            &spec, &batch, (0.0, 100.0), (0.0, 100.0),
            &ThemeInputs::default(),
        )
        .unwrap();

        // Default resolve — leader_line stays None.
        let mark_style = resolve_mark_style(None, &theme, &Mark::Label);
        assert!(
            mark_style.text.leader_line.is_none(),
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
