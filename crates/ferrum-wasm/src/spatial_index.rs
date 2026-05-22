//! R*-tree spatial index for accelerated hit-testing.
//!
//! This module builds a per-panel [`RTree`] over all indexed mark nodes
//! (circles and rects from `Point`, `Bar`, and `Rect` batches) so that
//! nearest-neighbor and envelope queries run in O(log n) rather than O(n).
//!
//! The index is built once per scene load via [`SpatialIndex::build`] using
//! `RTree::bulk_load`, which is O(n log n).  Query methods take O(log n).
//!
//! # Distance semantics
//! - **Circle** nodes: Euclidean distance from the query point to the circle
//!   center.  A query point inside the circle has distance 0.0.
//! - **Rect** nodes: Euclidean distance from the query point to the nearest
//!   point on the rect boundary (0.0 when the point is inside the rect).
//!
//! # SNAP_DISTANCE
//! [`nearest`] returns `None` when the closest entry is farther than
//! `SNAP_DISTANCE` (50.0 scene-space pixels).

use ferrum_scene::{MarkBatch, MarkBatchKind, Panel, SceneNode};
use rstar::{PointDistance, RTree, RTreeObject, AABB};

use crate::scene_load::SceneData;

/// Maximum distance (scene-space pixels) for [`SpatialIndex::nearest`] to
/// return a result.  Queries farther than this snap threshold return `None`.
pub const SNAP_DISTANCE: f64 = 50.0;

// ── MarkEntry ────────────────────────────────────────────────────────────────

/// A single indexed mark node inside the R*-tree.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkEntry {
    /// Center point: circle center, or rect center.
    pub point: [f64; 2],
    /// Index of the [`MarkBatch`] within its panel.
    pub batch_idx: usize,
    /// Index of the [`SceneNode`] within the batch's `nodes` vec.
    pub node_idx: usize,
    /// Optional data-row index from `batch.data_indices[node_idx]`.
    pub data_idx: Option<usize>,
    /// For circles: the circle radius.  For rects: `0.0` (use the AABB).
    pub radius: f64,
    /// Axis-aligned bounding box of the mark.
    pub aabb: AABB<[f64; 2]>,
}

impl RTreeObject for MarkEntry {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.aabb
    }
}

impl rstar::PointDistance for MarkEntry {
    /// Squared distance from this entry to `point`.
    ///
    /// For circles the distance is measured from the query point to the
    /// circle center; a point inside the circle contributes distance 0.
    /// For rects the distance is measured to the nearest point on the rect
    /// (0 when the query point is inside).
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let px = point[0];
        let py = point[1];

        if self.radius > 0.0 {
            // Circle: Euclidean distance to center, clamped to 0 when inside.
            let dx = px - self.point[0];
            let dy = py - self.point[1];
            let dist = (dx * dx + dy * dy).sqrt() - self.radius;
            if dist <= 0.0 { 0.0 } else { dist * dist }
        } else {
            // Rect: distance to nearest point on the AABB.
            let lower = self.aabb.lower();
            let upper = self.aabb.upper();
            let cx = px.clamp(lower[0], upper[0]);
            let cy = py.clamp(lower[1], upper[1]);
            let dx = px - cx;
            let dy = py - cy;
            dx * dx + dy * dy
        }
    }
}

// ── PanelIndex ───────────────────────────────────────────────────────────────

/// R*-tree over the indexed marks of a single panel.
pub struct PanelIndex {
    tree: RTree<MarkEntry>,
}

impl PanelIndex {
    fn new(entries: Vec<MarkEntry>) -> Self {
        Self {
            tree: RTree::bulk_load(entries),
        }
    }

    /// Nearest indexed mark to `(x, y)` within this panel's tree.
    ///
    /// Returns `(entry, distance)` where `distance` is the Euclidean distance
    /// from `(x, y)` to the nearest surface of the mark (0 when inside).
    /// Returns `None` when the tree is empty.
    fn nearest(&self, x: f64, y: f64) -> Option<(&MarkEntry, f64)> {
        let query = [x, y];
        self.tree.nearest_neighbor(&query).map(|entry| {
            let dist2 = entry.distance_2(&query);
            (entry, dist2.sqrt())
        })
    }

    /// All entries whose AABB overlaps `aabb`.
    fn in_envelope(&self, aabb: AABB<[f64; 2]>) -> impl Iterator<Item = &MarkEntry> {
        self.tree.locate_in_envelope_intersecting(&aabb)
    }
}

// ── SpatialIndex ─────────────────────────────────────────────────────────────

/// Per-scene spatial index — one [`PanelIndex`] (R*-tree) per panel.
pub struct SpatialIndex {
    /// One entry per panel; `trees[i]` corresponds to `panels[i]`.
    trees: Vec<PanelIndex>,
}

impl SpatialIndex {
    /// Build the index from a slice of [`Panel`]s.
    ///
    /// Only `Circle` nodes (from `Point` batches) and `Rect` nodes (from `Bar`
    /// and `Rect` batches) are indexed.  All other node types are skipped.
    ///
    /// Uses `RTree::bulk_load` — O(n log n) construction.
    pub fn build(panels: &[Panel]) -> Self {
        Self::build_with_packed(panels, None)
    }

    /// Build the index from panels, also indexing packed instances from the
    /// binary sidecar when `scene_data` is provided.
    ///
    /// Packed batches (>= 1000 marks) have empty `batch.nodes` in the scene
    /// graph. Without this method, those marks would be invisible to the
    /// R-tree, making spatial queries O(n) on the fallback path.
    pub fn build_with_packed(panels: &[Panel], scene_data: Option<&SceneData>) -> Self {
        let trees = panels
            .iter()
            .enumerate()
            .map(|(panel_idx, panel)| {
                let mut entries = entries_for_panel(panel);
                // Also index packed instances from the binary sidecar.
                if let Some(data) = scene_data {
                    collect_packed_entries(panel_idx, panel, data, &mut entries);
                }
                PanelIndex::new(entries)
            })
            .collect();

        Self { trees }
    }

    /// Nearest indexed mark to `(x, y)` in the given panel.
    ///
    /// Returns `Some((entry, distance))` when a mark exists within
    /// [`SNAP_DISTANCE`] scene-space pixels; returns `None` when the panel
    /// index does not exist, the tree is empty, or all marks are farther
    /// than the snap threshold.
    pub fn nearest(&self, panel_id: usize, x: f64, y: f64) -> Option<(MarkEntry, f64)> {
        let panel_idx = self.trees.get(panel_id)?;
        let (entry, dist) = panel_idx.nearest(x, y)?;
        if dist > SNAP_DISTANCE {
            return None;
        }
        Some((entry.clone(), dist))
    }

    /// All entries whose bounding box overlaps the given AABB in a panel.
    ///
    /// Returns an empty vec when `panel_id` is out of range.
    pub fn in_envelope(&self, panel_id: usize, aabb: AABB<[f64; 2]>) -> Vec<&MarkEntry> {
        let Some(panel_idx) = self.trees.get(panel_id) else {
            return vec![];
        };
        panel_idx.in_envelope(aabb).collect()
    }

    /// Exact hit-test at `(x, y)` using a small envelope query followed by
    /// per-geometry containment checks.
    ///
    /// `tolerance` expands the query envelope on each side.  A typical value
    /// for interactive use is `5.0` (scene-space pixels).
    ///
    /// Returns the first entry that geometrically contains `(x, y)` (with
    /// tolerance), or `None`.
    pub fn hit_test(
        &self,
        panel_id: usize,
        x: f64,
        y: f64,
        tolerance: f64,
    ) -> Option<MarkEntry> {
        let aabb = AABB::from_corners([x - tolerance, y - tolerance], [x + tolerance, y + tolerance]);
        let candidates = self.in_envelope(panel_id, aabb);
        for entry in candidates {
            if geometry_contains(entry, x, y, tolerance) {
                return Some(entry.clone());
            }
        }
        None
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build the list of [`MarkEntry`]s for one panel.
fn entries_for_panel(panel: &Panel) -> Vec<MarkEntry> {
    let mut entries = Vec::new();
    for (batch_idx, batch) in panel.marks.iter().enumerate() {
        if !is_indexed_kind(batch) {
            continue;
        }
        collect_batch_entries(batch, batch_idx, &mut entries);
    }
    entries
}

/// Return `true` for batch kinds whose nodes are indexed by the spatial index.
fn is_indexed_kind(batch: &MarkBatch) -> bool {
    matches!(
        batch.kind,
        MarkBatchKind::Point | MarkBatchKind::Bar | MarkBatchKind::Rect
    )
}

/// Append [`MarkEntry`] values for each indexable node in `batch`.
fn collect_batch_entries(batch: &MarkBatch, batch_idx: usize, out: &mut Vec<MarkEntry>) {
    for (node_idx, node) in batch.nodes.iter().enumerate() {
        let data_idx = batch
            .data_indices
            .as_ref()
            .and_then(|ids| ids.get(node_idx).copied());

        match node {
            SceneNode::Circle { cx, cy, r, .. } => {
                let r = *r;
                let cx = *cx;
                let cy = *cy;
                let aabb = AABB::from_corners([cx - r, cy - r], [cx + r, cy + r]);
                out.push(MarkEntry {
                    point: [cx, cy],
                    batch_idx,
                    node_idx,
                    data_idx,
                    radius: r,
                    aabb,
                });
            }
            SceneNode::Rect { x, y, w, h, .. } => {
                let x = *x;
                let y = *y;
                let w = *w;
                let h = *h;
                let cx = x + w / 2.0;
                let cy = y + h / 2.0;
                let aabb = AABB::from_corners([x, y], [x + w, y + h]);
                out.push(MarkEntry {
                    point: [cx, cy],
                    batch_idx,
                    node_idx,
                    data_idx,
                    radius: 0.0,
                    aabb,
                });
            }
            _ => {}
        }
    }
}

/// Collect [`MarkEntry`] values for packed instances in a panel.
///
/// For each batch in the panel that has packed metadata in `data.packed_batch_meta`,
/// creates entries from the circle/rect instance buffers at the recorded offsets.
fn collect_packed_entries(
    panel_idx: usize,
    panel: &Panel,
    data: &SceneData,
    out: &mut Vec<MarkEntry>,
) {
    for (batch_idx, _batch) in panel.marks.iter().enumerate() {
        let key = (panel_idx as u32, batch_idx as u32);
        let Some(meta) = data.packed_batch_meta.get(&key) else {
            continue;
        };

        match meta.kind {
            0 => {
                // Packed circles.
                for i in 0..meta.instance_count {
                    let idx = meta.instance_start + i;
                    let Some(ci) = data.circle_instances.get(idx) else { continue };
                    let cx = ci.center[0] as f64;
                    let cy = ci.center[1] as f64;
                    let r = ci.radius as f64;
                    let aabb = AABB::from_corners([cx - r, cy - r], [cx + r, cy + r]);
                    let data_idx = meta.data_indices
                        .as_ref()
                        .and_then(|dis| dis.get(i))
                        .map(|&di| di as usize);
                    out.push(MarkEntry {
                        point: [cx, cy],
                        batch_idx,
                        node_idx: i,
                        data_idx,
                        radius: r,
                        aabb,
                    });
                }
            }
            1 => {
                // Packed rects.
                for i in 0..meta.instance_count {
                    let idx = meta.instance_start + i;
                    let Some(ri) = data.rect_instances.get(idx) else { continue };
                    let x = ri.position[0] as f64;
                    let y = ri.position[1] as f64;
                    let w = ri.size[0] as f64;
                    let h = ri.size[1] as f64;
                    let cx = x + w / 2.0;
                    let cy = y + h / 2.0;
                    let aabb = AABB::from_corners([x, y], [x + w, y + h]);
                    let data_idx = meta.data_indices
                        .as_ref()
                        .and_then(|dis| dis.get(i))
                        .map(|&di| di as usize);
                    out.push(MarkEntry {
                        point: [cx, cy],
                        batch_idx,
                        node_idx: i,
                        data_idx,
                        radius: 0.0,
                        aabb,
                    });
                }
            }
            _ => {}
        }
    }
}

/// Check whether `(x, y)` lies within the geometry of `entry` (with
/// `tolerance` added on all sides).
fn geometry_contains(entry: &MarkEntry, x: f64, y: f64, tolerance: f64) -> bool {
    if entry.radius > 0.0 {
        // Circle: check distance to center vs. (radius + tolerance).
        let dx = x - entry.point[0];
        let dy = y - entry.point[1];
        let r = entry.radius + tolerance;
        dx * dx + dy * dy <= r * r
    } else {
        // Rect: check containment within AABB expanded by tolerance.
        let lower = entry.aabb.lower();
        let upper = entry.aabb.upper();
        x >= lower[0] - tolerance
            && x <= upper[0] + tolerance
            && y >= lower[1] - tolerance
            && y <= upper[1] + tolerance
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use super::*;
    use ferrum_scene::{
        BlendMode, CoordKind, FillStroke, MarkBatch, MarkBatchKind, Panel, Rect, SceneNode,
    };
    use rstar::PointDistance;

    // ── test helpers ──────────────────────────────────────────────────────────

    fn default_style() -> FillStroke {
        FillStroke {
            fill: Some(ferrum_scene::Color { r: 0, g: 0, b: 0, a: 255 }),
            stroke: None,
            stroke_width: 0.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        }
    }

    fn circle(cx: f64, cy: f64, r: f64) -> SceneNode {
        SceneNode::Circle { cx, cy, r, style: default_style() }
    }

    fn rect_node(x: f64, y: f64, w: f64, h: f64) -> SceneNode {
        SceneNode::Rect { x, y, w, h, style: default_style(), corner_radius: 0.0 }
    }

    fn panel_with_batches(batches: Vec<MarkBatch>) -> Panel {
        Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 1000.0, h: 1000.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 1000.0, h: 1000.0 },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
            },
            grid: vec![],
            marks: batches,
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        }
    }

    fn point_batch(nodes: Vec<SceneNode>, data_indices: Option<Vec<usize>>) -> MarkBatch {
        MarkBatch {
            kind: MarkBatchKind::Point,
            nodes,
            data_indices,
            tooltips: None,
            hrefs: None,
            keys: None,
            blend: BlendMode::Normal,
            descriptions: None,
            stroke_cap: None,
            stroke_join: None,
            packed_instances: None,
        }
    }

    fn bar_batch(nodes: Vec<SceneNode>, data_indices: Option<Vec<usize>>) -> MarkBatch {
        MarkBatch {
            kind: MarkBatchKind::Bar,
            nodes,
            data_indices,
            tooltips: None,
            hrefs: None,
            keys: None,
            blend: BlendMode::Normal,
            descriptions: None,
            stroke_cap: None,
            stroke_join: None,
            packed_instances: None,
        }
    }

    fn rect_batch(nodes: Vec<SceneNode>, data_indices: Option<Vec<usize>>) -> MarkBatch {
        MarkBatch {
            kind: MarkBatchKind::Rect,
            nodes,
            data_indices,
            tooltips: None,
            hrefs: None,
            keys: None,
            blend: BlendMode::Normal,
            descriptions: None,
            stroke_cap: None,
            stroke_join: None,
            packed_instances: None,
        }
    }

    fn make_panels(batches: Vec<MarkBatch>) -> Vec<Panel> {
        vec![panel_with_batches(batches)]
    }

    // ── empty-panel tests ─────────────────────────────────────────────────────

    #[test]
    fn empty_panels_slice_builds_empty_index() {
        let idx = SpatialIndex::build(&[]);
        // No panel 0 — nearest must return None.
        assert!(idx.nearest(0, 500.0, 500.0).is_none());
    }

    #[test]
    fn panel_with_no_marks_returns_none_for_nearest() {
        let panels = make_panels(vec![]);
        let idx = SpatialIndex::build(&panels);
        assert!(idx.nearest(0, 500.0, 500.0).is_none());
    }

    #[test]
    fn panel_with_no_marks_returns_empty_for_envelope() {
        let panels = make_panels(vec![]);
        let idx = SpatialIndex::build(&panels);
        let aabb = AABB::from_corners([0.0, 0.0], [1000.0, 1000.0]);
        assert!(idx.in_envelope(0, aabb).is_empty());
    }

    #[test]
    fn panel_with_no_marks_returns_none_for_hit_test() {
        let panels = make_panels(vec![]);
        let idx = SpatialIndex::build(&panels);
        assert!(idx.hit_test(0, 500.0, 500.0, 5.0).is_none());
    }

    // ── single-mark tests ─────────────────────────────────────────────────────

    #[test]
    fn single_circle_nearest_returns_it() {
        let panels = make_panels(vec![
            point_batch(vec![circle(100.0, 100.0, 5.0)], Some(vec![7])),
        ]);
        let idx = SpatialIndex::build(&panels);
        let (entry, dist) = idx.nearest(0, 100.0, 100.0).expect("must find mark");
        assert_eq!(entry.data_idx, Some(7));
        assert!(dist < 1e-9, "distance at center must be ~0, got {dist}");
    }

    #[test]
    fn single_circle_nearest_outside_radius_returns_zero_dist() {
        // Query at the circle center — dist should be 0.
        let panels = make_panels(vec![
            point_batch(vec![circle(200.0, 200.0, 10.0)], None),
        ]);
        let idx = SpatialIndex::build(&panels);
        let (_, dist) = idx.nearest(0, 200.0, 200.0).unwrap();
        assert!(dist < 1e-9);
    }

    #[test]
    fn single_circle_nearest_at_edge_returns_zero_dist() {
        // Query exactly at the circle surface — dist should be 0.
        let panels = make_panels(vec![
            point_batch(vec![circle(0.0, 0.0, 10.0)], None),
        ]);
        let idx = SpatialIndex::build(&panels);
        let (_, dist) = idx.nearest(0, 10.0, 0.0).unwrap();
        assert!(dist < 1e-9, "surface point must have distance 0, got {dist}");
    }

    #[test]
    fn single_circle_nearest_outside_snap_returns_none() {
        let panels = make_panels(vec![
            point_batch(vec![circle(0.0, 0.0, 5.0)], None),
        ]);
        let idx = SpatialIndex::build(&panels);
        // Distance to surface = 100 - 5 = 95 > SNAP_DISTANCE = 50
        assert!(idx.nearest(0, 100.0, 0.0).is_none());
    }

    #[test]
    fn single_circle_hit_test_inside() {
        let panels = make_panels(vec![
            point_batch(vec![circle(50.0, 50.0, 10.0)], Some(vec![3])),
        ]);
        let idx = SpatialIndex::build(&panels);
        let hit = idx.hit_test(0, 55.0, 50.0, 1.0).expect("must hit");
        assert_eq!(hit.data_idx, Some(3));
    }

    #[test]
    fn single_circle_hit_test_miss() {
        let panels = make_panels(vec![
            point_batch(vec![circle(50.0, 50.0, 10.0)], None),
        ]);
        let idx = SpatialIndex::build(&panels);
        // 30 px away from center, radius=10, tolerance=1 → should miss
        assert!(idx.hit_test(0, 80.0, 50.0, 1.0).is_none());
    }

    #[test]
    fn single_rect_nearest_returns_it() {
        // Rect at (90, 90, 20, 20) — center at (100, 100).
        let panels = make_panels(vec![
            bar_batch(vec![rect_node(90.0, 90.0, 20.0, 20.0)], Some(vec![42])),
        ]);
        let idx = SpatialIndex::build(&panels);
        let (entry, dist) = idx.nearest(0, 100.0, 100.0).expect("must find rect");
        assert_eq!(entry.data_idx, Some(42));
        // Query point is inside the rect — distance should be 0.
        assert!(dist < 1e-9, "point inside rect must have distance 0, got {dist}");
    }

    #[test]
    fn single_rect_nearest_outside() {
        // Rect at (0, 0, 10, 10); query at (15, 5) — nearest rect point is (10, 5), dist=5.
        let panels = make_panels(vec![
            bar_batch(vec![rect_node(0.0, 0.0, 10.0, 10.0)], None),
        ]);
        let idx = SpatialIndex::build(&panels);
        let (_, dist) = idx.nearest(0, 15.0, 5.0).unwrap();
        assert!((dist - 5.0).abs() < 1e-6, "expected dist=5, got {dist}");
    }

    #[test]
    fn single_rect_hit_test_inside() {
        let panels = make_panels(vec![
            rect_batch(vec![rect_node(10.0, 10.0, 80.0, 60.0)], Some(vec![99])),
        ]);
        let idx = SpatialIndex::build(&panels);
        let hit = idx.hit_test(0, 50.0, 40.0, 0.0).expect("must hit inside rect");
        assert_eq!(hit.data_idx, Some(99));
    }

    #[test]
    fn single_rect_hit_test_miss() {
        let panels = make_panels(vec![
            rect_batch(vec![rect_node(10.0, 10.0, 80.0, 60.0)], None),
        ]);
        let idx = SpatialIndex::build(&panels);
        assert!(idx.hit_test(0, 5.0, 5.0, 0.0).is_none());
    }

    // ── 100-mark tests ────────────────────────────────────────────────────────

    #[test]
    fn hundred_marks_nearest_correct_vs_linear_scan() {
        // Place 100 circles in a 10x10 grid at positions (i*50, j*50).
        // For a query point, compare R-tree result to brute-force linear scan.
        let mut nodes = Vec::new();
        let mut data_indices = Vec::new();
        for i in 0..10usize {
            for j in 0..10usize {
                let cx = (i as f64) * 50.0 + 25.0;
                let cy = (j as f64) * 50.0 + 25.0;
                nodes.push(circle(cx, cy, 5.0));
                data_indices.push(i * 10 + j);
            }
        }
        let panels = make_panels(vec![
            point_batch(nodes.clone(), Some(data_indices.clone())),
        ]);
        let idx = SpatialIndex::build(&panels);

        // Query at various points and compare to brute-force.
        for (qx, qy) in [(37.0, 37.0), (200.0, 275.0), (450.0, 450.0)] {
            // Brute-force: find circle with minimum center-distance.
            let best_linear = nodes
                .iter()
                .zip(data_indices.iter())
                .map(|(n, di)| {
                    if let SceneNode::Circle { cx, cy, r, .. } = n {
                        let dist = ((qx - cx).powi(2) + (qy - cy).powi(2)).sqrt() - r;
                        (di, dist.max(0.0))
                    } else {
                        unreachable!()
                    }
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let (bf_di, bf_dist) = best_linear.unwrap();

            let rtree_result = idx.nearest(0, qx, qy);

            if bf_dist <= SNAP_DISTANCE {
                let (entry, rt_dist) = rtree_result.expect("R-tree must find a mark within snap");
                // R-tree and linear scan may pick different nodes when distances are tied.
                // Assert distance equality — that is the meaningful correctness property.
                assert!(
                    (rt_dist - bf_dist).abs() < 1e-6,
                    "R-tree distance {rt_dist} vs linear {bf_dist} for query ({qx},{qy}), \
                     rtree data_idx={:?}, linear data_idx={bf_di}",
                    entry.data_idx
                );
            } else {
                assert!(
                    rtree_result.is_none(),
                    "R-tree must return None for query ({qx},{qy}) with bf_dist={bf_dist}"
                );
            }
        }
    }

    #[test]
    fn hundred_marks_envelope_query_correct() {
        // 100 circles at (i*50+25, j*50+25) — query [100, 200] x [100, 200].
        // Expect circles within that box (those whose AABB overlaps).
        let mut nodes = Vec::new();
        let mut data_indices = Vec::new();
        for i in 0..10usize {
            for j in 0..10usize {
                let cx = (i as f64) * 50.0 + 25.0;
                let cy = (j as f64) * 50.0 + 25.0;
                nodes.push(circle(cx, cy, 5.0));
                data_indices.push(i * 10 + j);
            }
        }
        let panels = make_panels(vec![
            point_batch(nodes.clone(), Some(data_indices)),
        ]);
        let idx = SpatialIndex::build(&panels);

        let qbox = AABB::from_corners([100.0, 100.0], [200.0, 200.0]);
        let results = idx.in_envelope(0, qbox);

        // Brute-force: which circles' AABBs overlap [100,200]x[100,200]?
        let expected: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                if let SceneNode::Circle { cx, cy, r, .. } = n {
                    let lo_x = cx - r;
                    let hi_x = cx + r;
                    let lo_y = cy - r;
                    let hi_y = cy + r;
                    hi_x >= 100.0 && lo_x <= 200.0 && hi_y >= 100.0 && lo_y <= 200.0
                } else {
                    false
                }
            })
            .map(|(i, _)| i)
            .collect();

        let mut result_idxs: Vec<usize> = results.iter().map(|e| e.node_idx).collect();
        let mut expected_sorted = expected.clone();
        result_idxs.sort();
        expected_sorted.sort();

        assert_eq!(
            result_idxs, expected_sorted,
            "envelope query result mismatch"
        );
    }

    // ── 100k-mark scale test ──────────────────────────────────────────────────

    #[test]
    fn hundred_thousand_marks_builds_and_queries() {
        // Build index with 100k circles; verify nearest returns correct result.
        let n = 100_000usize;
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            let cx = (i % 1000) as f64 * 1.0 + 0.5;
            let cy = (i / 1000) as f64 * 1.0 + 0.5;
            nodes.push(circle(cx, cy, 0.4));
        }
        let panels = make_panels(vec![
            point_batch(nodes, None),
        ]);
        let idx = SpatialIndex::build(&panels);

        // Query at a known position — the circle at (250.5, 37.5) should be nearest.
        let (entry, dist) = idx.nearest(0, 250.5, 37.5).expect("must find mark in 100k");
        assert_eq!(entry.node_idx, 37 * 1000 + 250, "unexpected nearest mark idx");
        assert!(dist < 1e-6, "expected dist ~0, got {dist}");
    }

    // ── nearest-neighbor correctness vs linear scan ───────────────────────────

    #[test]
    fn nearest_neighbor_correctness_multiple_queries() {
        // 20 circles at random-ish positions; compare R-tree to linear scan for 10 queries.
        let positions: &[(f64, f64)] = &[
            (10.0, 20.0), (50.0, 80.0), (120.0, 35.0), (200.0, 200.0), (300.0, 10.0),
            (15.0, 150.0), (400.0, 300.0), (250.0, 450.0), (180.0, 90.0), (320.0, 220.0),
            (60.0, 410.0), (145.0, 270.0), (380.0, 160.0), (90.0, 340.0), (440.0, 80.0),
            (210.0, 350.0), (330.0, 400.0), (70.0, 230.0), (410.0, 420.0), (155.0, 130.0),
        ];
        let nodes: Vec<SceneNode> = positions
            .iter()
            .map(|(cx, cy)| circle(*cx, *cy, 8.0))
            .collect();
        let panels = make_panels(vec![
            point_batch(nodes.clone(), None),
        ]);
        let idx = SpatialIndex::build(&panels);

        let queries: &[(f64, f64)] = &[
            (10.0, 20.0), (100.0, 100.0), (250.0, 250.0),
            (0.0, 0.0), (450.0, 450.0),
            (200.0, 0.0), (50.0, 50.0), (300.0, 300.0),
            (180.0, 180.0), (420.0, 85.0),
        ];

        for (qx, qy) in queries {
            // Brute-force.
            let (bf_idx, bf_dist) = nodes
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    if let SceneNode::Circle { cx, cy, r, .. } = n {
                        let d = ((qx - cx).powi(2) + (qy - cy).powi(2)).sqrt() - r;
                        (i, d.max(0.0))
                    } else {
                        unreachable!()
                    }
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            let rtree_result = idx.nearest(0, *qx, *qy);

            if bf_dist <= SNAP_DISTANCE {
                let (entry, rt_dist) = rtree_result
                    .expect("R-tree must find a mark within SNAP_DISTANCE");
                assert_eq!(
                    entry.node_idx, bf_idx,
                    "nearest mismatch for query ({qx},{qy}): rtree={}, linear={bf_idx}",
                    entry.node_idx
                );
                assert!(
                    (rt_dist - bf_dist).abs() < 1e-6,
                    "distance mismatch for ({qx},{qy}): rtree={rt_dist}, linear={bf_dist}"
                );
            } else {
                assert!(
                    rtree_result.is_none(),
                    "R-tree must return None for ({qx},{qy}) beyond snap threshold"
                );
            }
        }
    }

    // ── envelope correctness ──────────────────────────────────────────────────

    #[test]
    fn envelope_query_excludes_marks_outside_box() {
        // 4 circles; only 2 are within the query box.
        let nodes = vec![
            circle(50.0, 50.0, 5.0),   // inside  [40,60]x[40,60]
            circle(150.0, 50.0, 5.0),  // outside
            circle(55.0, 55.0, 5.0),   // inside
            circle(50.0, 200.0, 5.0),  // outside
        ];
        let panels = make_panels(vec![
            point_batch(nodes, Some(vec![10, 20, 30, 40])),
        ]);
        let idx = SpatialIndex::build(&panels);

        let qbox = AABB::from_corners([40.0, 40.0], [65.0, 65.0]);
        let mut results: Vec<usize> = idx
            .in_envelope(0, qbox)
            .iter()
            .map(|e| e.data_idx.unwrap())
            .collect();
        results.sort();

        assert_eq!(results, vec![10, 30], "only marks 0 and 2 should be in envelope");
    }

    #[test]
    fn envelope_query_empty_box_returns_empty() {
        let panels = make_panels(vec![
            point_batch(vec![circle(500.0, 500.0, 5.0)], None),
        ]);
        let idx = SpatialIndex::build(&panels);
        // Tiny box far away from all marks.
        let qbox = AABB::from_corners([0.0, 0.0], [1.0, 1.0]);
        assert!(idx.in_envelope(0, qbox).is_empty());
    }

    // ── snap threshold ────────────────────────────────────────────────────────

    #[test]
    fn nearest_just_within_snap_distance_returns_some() {
        // Circle at (0, 0) with radius 0. Query at (SNAP_DISTANCE - 1, 0).
        let panels = make_panels(vec![
            point_batch(vec![circle(0.0, 0.0, 0.0)], None),
        ]);
        let idx = SpatialIndex::build(&panels);
        let dist = SNAP_DISTANCE - 1.0;
        assert!(
            idx.nearest(0, dist, 0.0).is_some(),
            "query within snap threshold must return Some"
        );
    }

    #[test]
    fn nearest_exactly_at_snap_distance_returns_some() {
        // Surface of circle at (0,0) radius=0 is at (0,0). Query at (SNAP_DISTANCE, 0).
        let panels = make_panels(vec![
            point_batch(vec![circle(0.0, 0.0, 0.0)], None),
        ]);
        let idx = SpatialIndex::build(&panels);
        // Distance = SNAP_DISTANCE exactly — should return Some (≤ threshold).
        assert!(
            idx.nearest(0, SNAP_DISTANCE, 0.0).is_some(),
            "query exactly at snap distance must return Some"
        );
    }

    #[test]
    fn nearest_just_beyond_snap_distance_returns_none() {
        let panels = make_panels(vec![
            point_batch(vec![circle(0.0, 0.0, 0.0)], None),
        ]);
        let idx = SpatialIndex::build(&panels);
        let dist = SNAP_DISTANCE + 0.01;
        assert!(
            idx.nearest(0, dist, 0.0).is_none(),
            "query beyond snap threshold must return None"
        );
    }

    // ── out-of-range panel_id ─────────────────────────────────────────────────

    #[test]
    fn out_of_range_panel_id_returns_none_for_nearest() {
        let panels = make_panels(vec![point_batch(vec![circle(100.0, 100.0, 5.0)], None)]);
        let idx = SpatialIndex::build(&panels);
        assert!(idx.nearest(99, 100.0, 100.0).is_none());
    }

    #[test]
    fn out_of_range_panel_id_returns_empty_for_envelope() {
        let panels = make_panels(vec![point_batch(vec![circle(100.0, 100.0, 5.0)], None)]);
        let idx = SpatialIndex::build(&panels);
        let aabb = AABB::from_corners([0.0, 0.0], [1000.0, 1000.0]);
        assert!(idx.in_envelope(99, aabb).is_empty());
    }

    // ── non-indexed batch kinds are not indexed ───────────────────────────────

    #[test]
    fn line_batch_nodes_not_indexed() {
        use ferrum_scene::StrokeStyle;
        let line_style = StrokeStyle {
            color: ferrum_scene::Color { r: 0, g: 0, b: 0, a: 255 },
            width: 1.0,
            opacity: 1.0,
            stroke_opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
        };
        let line_batch = MarkBatch {
            kind: MarkBatchKind::Line,
            nodes: vec![SceneNode::Line {
                x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0,
                style: line_style,
            }],
            data_indices: Some(vec![0]),
            tooltips: None,
            hrefs: None,
            keys: None,
            blend: BlendMode::Normal,
            descriptions: None,
            stroke_cap: None,
            stroke_join: None,
            packed_instances: None,
        };
        let panels = make_panels(vec![line_batch]);
        let idx = SpatialIndex::build(&panels);
        assert!(idx.nearest(0, 50.0, 50.0).is_none(), "Line batch must not be indexed");
    }

    // ── data_idx threading ────────────────────────────────────────────────────

    #[test]
    fn data_idx_none_when_no_data_indices() {
        let panels = make_panels(vec![point_batch(vec![circle(100.0, 100.0, 5.0)], None)]);
        let idx = SpatialIndex::build(&panels);
        let (entry, _) = idx.nearest(0, 100.0, 100.0).expect("must find mark");
        assert_eq!(entry.data_idx, None);
    }

    #[test]
    fn data_idx_correct_for_multiple_nodes() {
        let nodes = vec![
            circle(100.0, 100.0, 5.0),
            circle(200.0, 100.0, 5.0),
            circle(300.0, 100.0, 5.0),
        ];
        let panels = make_panels(vec![
            point_batch(nodes, Some(vec![10, 20, 30])),
        ]);
        let idx = SpatialIndex::build(&panels);

        let (e0, _) = idx.nearest(0, 100.0, 100.0).unwrap();
        assert_eq!(e0.data_idx, Some(10));

        let (e1, _) = idx.nearest(0, 200.0, 100.0).unwrap();
        assert_eq!(e1.data_idx, Some(20));

        let (e2, _) = idx.nearest(0, 300.0, 100.0).unwrap();
        assert_eq!(e2.data_idx, Some(30));
    }

    // ── multiple panels ───────────────────────────────────────────────────────

    #[test]
    fn multiple_panels_indexed_independently() {
        let panels = vec![
            panel_with_batches(vec![
                point_batch(vec![circle(100.0, 100.0, 5.0)], Some(vec![1])),
            ]),
            {
                let mut p = panel_with_batches(vec![
                    point_batch(vec![circle(200.0, 200.0, 5.0)], Some(vec![2])),
                ]);
                p.id = 1;
                p
            },
        ];
        let idx = SpatialIndex::build(&panels);

        let (e0, _) = idx.nearest(0, 100.0, 100.0).unwrap();
        assert_eq!(e0.data_idx, Some(1), "panel 0 must return its own mark");

        let (e1, _) = idx.nearest(1, 200.0, 200.0).unwrap();
        assert_eq!(e1.data_idx, Some(2), "panel 1 must return its own mark");
    }

    // ── rect distance correctness ─────────────────────────────────────────────

    #[test]
    fn rect_distance_inside_is_zero() {
        let entry = MarkEntry {
            point: [50.0, 50.0],
            batch_idx: 0,
            node_idx: 0,
            data_idx: None,
            radius: 0.0,
            aabb: AABB::from_corners([40.0, 40.0], [60.0, 60.0]),
        };
        assert!(entry.distance_2(&[50.0, 50.0]) < 1e-12);
        assert!(entry.distance_2(&[40.0, 40.0]) < 1e-12); // corner
    }

    #[test]
    fn rect_distance_outside_correct() {
        let entry = MarkEntry {
            point: [50.0, 50.0],
            batch_idx: 0,
            node_idx: 0,
            data_idx: None,
            radius: 0.0,
            aabb: AABB::from_corners([40.0, 40.0], [60.0, 60.0]),
        };
        // Right of rect at x=70, y=50 → nearest point (60, 50) → dist=10
        let d2 = entry.distance_2(&[70.0, 50.0]);
        assert!((d2.sqrt() - 10.0).abs() < 1e-6, "expected dist=10, got {}", d2.sqrt());
    }

    #[test]
    fn circle_distance_inside_is_zero() {
        let entry = MarkEntry {
            point: [100.0, 100.0],
            batch_idx: 0,
            node_idx: 0,
            data_idx: None,
            radius: 20.0,
            aabb: AABB::from_corners([80.0, 80.0], [120.0, 120.0]),
        };
        assert!(entry.distance_2(&[100.0, 100.0]) < 1e-12); // center
        assert!(entry.distance_2(&[110.0, 100.0]) < 1e-12); // inside
    }

    #[test]
    fn circle_distance_outside_correct() {
        let entry = MarkEntry {
            point: [0.0, 0.0],
            batch_idx: 0,
            node_idx: 0,
            data_idx: None,
            radius: 10.0,
            aabb: AABB::from_corners([-10.0, -10.0], [10.0, 10.0]),
        };
        // Distance from edge = 30 - 10 = 20 → distance_2 should be 400
        let d2 = entry.distance_2(&[30.0, 0.0]);
        assert!((d2 - 400.0).abs() < 1e-6, "expected d2=400, got {d2}");
    }

    // ── Bug 5: SpatialIndex::build_with_packed indexes packed instances ──

    #[test]
    fn build_with_packed_indexes_circle_instances() {
        use crate::scene_load::{CircleInstance, PackedBatchMeta, SceneData};
        use lyon::tessellation::VertexBuffers;
        use std::collections::HashMap;

        // Panel with one batch that has empty nodes (packed batch).
        let panels = make_panels(vec![
            point_batch(vec![], None), // empty nodes — packed
        ]);

        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 0u32),
            PackedBatchMeta {
                data_indices: Some(vec![10, 20, 30]),
                tooltip_bytes: None,
                kind: 0, // circles
                instance_start: 0,
                instance_count: 3,
            },
        );

        let data = SceneData {
            circle_instances: vec![
                CircleInstance {
                    center: [100.0, 100.0], radius: 5.0,
                    fill_color: [0.0; 4], stroke_color: [0.0; 4],
                    stroke_width: 0.0, opacity: 1.0, stroke_opacity: 0.0,
                    stroke_dash: 0.0, angle: 0.0,
                },
                CircleInstance {
                    center: [200.0, 200.0], radius: 5.0,
                    fill_color: [0.0; 4], stroke_color: [0.0; 4],
                    stroke_width: 0.0, opacity: 1.0, stroke_opacity: 0.0,
                    stroke_dash: 0.0, angle: 0.0,
                },
                CircleInstance {
                    center: [300.0, 300.0], radius: 5.0,
                    fill_color: [0.0; 4], stroke_color: [0.0; 4],
                    stroke_width: 0.0, opacity: 1.0, stroke_opacity: 0.0,
                    stroke_dash: 0.0, angle: 0.0,
                },
            ],
            rect_instances: vec![],
            mesh_buffers: VertexBuffers::new(),
            text_elements: vec![],
            image_quads: vec![],
            background: None,
            width: 500.0,
            height: 500.0,
            packed_batch_meta: packed_meta,
            draw_commands: vec![],
        };

        let idx = SpatialIndex::build_with_packed(&panels, Some(&data));

        // All three packed circles should be indexed and findable.
        let (e0, d0) = idx.nearest(0, 100.0, 100.0).expect("must find packed circle 0");
        assert_eq!(e0.data_idx, Some(10));
        assert!(d0 < 1e-6, "dist must be ~0 at center");

        let (e1, _) = idx.nearest(0, 200.0, 200.0).expect("must find packed circle 1");
        assert_eq!(e1.data_idx, Some(20));

        let (e2, _) = idx.nearest(0, 300.0, 300.0).expect("must find packed circle 2");
        assert_eq!(e2.data_idx, Some(30));
    }

    #[test]
    fn build_with_packed_indexes_rect_instances() {
        use crate::scene_load::{RectInstance, PackedBatchMeta, SceneData};
        use lyon::tessellation::VertexBuffers;
        use std::collections::HashMap;

        // Panel with one batch that has empty nodes (packed batch).
        let panels = make_panels(vec![
            bar_batch(vec![], None), // empty nodes — packed
        ]);

        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 0u32),
            PackedBatchMeta {
                data_indices: Some(vec![42, 99]),
                tooltip_bytes: None,
                kind: 1, // rects
                instance_start: 0,
                instance_count: 2,
            },
        );

        let data = SceneData {
            circle_instances: vec![],
            rect_instances: vec![
                RectInstance {
                    position: [90.0, 90.0], size: [20.0, 20.0], corner_radius: 0.0,
                    fill_color: [0.0; 4], stroke_color: [0.0; 4],
                    stroke_width: 0.0, opacity: 1.0, stroke_opacity: 0.0,
                    stroke_dash: 0.0, angle: 0.0,
                },
                RectInstance {
                    position: [200.0, 200.0], size: [50.0, 30.0], corner_radius: 0.0,
                    fill_color: [0.0; 4], stroke_color: [0.0; 4],
                    stroke_width: 0.0, opacity: 1.0, stroke_opacity: 0.0,
                    stroke_dash: 0.0, angle: 0.0,
                },
            ],
            mesh_buffers: VertexBuffers::new(),
            text_elements: vec![],
            image_quads: vec![],
            background: None,
            width: 500.0,
            height: 500.0,
            packed_batch_meta: packed_meta,
            draw_commands: vec![],
        };

        let idx = SpatialIndex::build_with_packed(&panels, Some(&data));

        // Rect 0 center at (100, 100) — should be findable.
        let (e0, d0) = idx.nearest(0, 100.0, 100.0).expect("must find packed rect 0");
        assert_eq!(e0.data_idx, Some(42));
        assert!(d0 < 1e-6, "point inside rect must have dist 0");

        // Rect 1 center at (225, 215) — should be findable.
        let (e1, _) = idx.nearest(0, 225.0, 215.0).expect("must find packed rect 1");
        assert_eq!(e1.data_idx, Some(99));
    }

    #[test]
    fn build_with_packed_mixes_packed_and_nonpacked() {
        use crate::scene_load::{CircleInstance, PackedBatchMeta, SceneData};
        use lyon::tessellation::VertexBuffers;
        use std::collections::HashMap;

        // Panel with two batches: one non-packed, one packed.
        let panels = make_panels(vec![
            point_batch(vec![circle(50.0, 50.0, 5.0)], Some(vec![1])), // non-packed
            point_batch(vec![], None), // packed
        ]);

        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 1u32), // panel 0, batch 1
            PackedBatchMeta {
                data_indices: Some(vec![2]),
                tooltip_bytes: None,
                kind: 0,
                instance_start: 0,
                instance_count: 1,
            },
        );

        let data = SceneData {
            circle_instances: vec![
                CircleInstance {
                    center: [400.0, 400.0], radius: 5.0,
                    fill_color: [0.0; 4], stroke_color: [0.0; 4],
                    stroke_width: 0.0, opacity: 1.0, stroke_opacity: 0.0,
                    stroke_dash: 0.0, angle: 0.0,
                },
            ],
            rect_instances: vec![],
            mesh_buffers: VertexBuffers::new(),
            text_elements: vec![],
            image_quads: vec![],
            background: None,
            width: 500.0,
            height: 500.0,
            packed_batch_meta: packed_meta,
            draw_commands: vec![],
        };

        let idx = SpatialIndex::build_with_packed(&panels, Some(&data));

        // Non-packed circle at (50, 50).
        let (e_np, _) = idx.nearest(0, 50.0, 50.0).expect("must find non-packed circle");
        assert_eq!(e_np.data_idx, Some(1));

        // Packed circle at (400, 400).
        let (e_pk, _) = idx.nearest(0, 400.0, 400.0).expect("must find packed circle");
        assert_eq!(e_pk.data_idx, Some(2));
    }
}
