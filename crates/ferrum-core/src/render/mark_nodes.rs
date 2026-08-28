//! Node + source-row accumulator for mark builders (archaeology bug #6 fix).
//!
//! # Why this exists
//!
//! A mark builder turns a `RecordBatch` into a list of scene nodes plus a
//! parallel list of *source-row indices* (`data_indices`). The metadata
//! attached to a batch (tooltips, hrefs, descriptions) is indexed by **node
//! position** in both render paths — the SVG walker enumerates nodes and the
//! packed/WASM path's `get_tooltip(node_idx)` indexes the same vectors. So the
//! per-node `data_indices` and the per-node metadata must be built in lockstep
//! with the nodes, or node `j` receives the wrong row's tooltip.
//!
//! The pre-fix bug: builders that *skip* rows (null/degenerate values), emit
//! *several* nodes per row (a point `Cross` shape → 2 line nodes), or *group*
//! rows (line/area/ribbon) built the node list with `continue`/grouping while
//! still calling `build_metadata(ctx)` — which returns full per-*row* vectors
//! indexed `0..n_rows`. Once any row is skipped or fanned out, node index `j`
//! no longer equals source row `j`, and the metadata is silently misaligned.
//!
//! `arc.rs` already does this correctly by hand: it tracks a `data_indices`
//! `Vec` alongside its `nodes` `Vec` and finalizes with
//! [`MetadataColumns::build_metadata_for_indices`](crate::render::draw::MetadataColumns::build_metadata_for_indices).
//! `MarkNodes` generalizes exactly that pattern so a node *cannot* be added
//! without supplying its source row, making the misalignment unrepresentable.
//!
//! # Usage
//!
//! ```ignore
//! let meta = MetadataColumns::from_ctx(ctx);
//! let mut acc = MarkNodes::with_capacity(values.len());
//! for (row, v) in values.iter().enumerate() {
//!     let Some(v) = keep(v) else { continue }; // skipped rows never reach push
//!     acc.push(make_node(v), row);
//! }
//! let (nodes, data_indices) = acc.finalize();
//! let (tooltips, hrefs, descriptions) = meta.build_metadata_for_indices(&data_indices);
//! MarkBuildResult { kind, nodes, data_indices: Some(data_indices), tooltips, hrefs, descriptions }
//! ```

use ferrum_scene::SceneNode;

/// Accumulator that owns scene nodes and their source-row indices together.
///
/// Invariant: `nodes.len() == indices.len()` at all times. `indices[k]` is the
/// source row of `nodes[k]`. Multi-node shapes (e.g. point `Cross` → 2 nodes)
/// map every emitted node to the *same* source row via [`push_many`]; group
/// marks map one node to the group's representative row via [`push`].
///
/// [`push`]: MarkNodes::push
/// [`push_many`]: MarkNodes::push_many
#[derive(Debug, Default)]
pub(crate) struct MarkNodes {
    nodes: Vec<SceneNode>,
    indices: Vec<usize>,
}

impl MarkNodes {
    /// Empty accumulator. Used by group-mark builders (e.g. `geoshape`) whose
    /// output node count is not known upfront, so reserving capacity would not
    /// help.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Empty accumulator with capacity reserved for `n` nodes. Use the row
    /// count when each row maps to ~1 node so the common case avoids reallocs.
    pub(crate) fn with_capacity(n: usize) -> Self {
        Self { nodes: Vec::with_capacity(n), indices: Vec::with_capacity(n) }
    }

    /// Append one node produced by source `row`.
    pub(crate) fn push(&mut self, node: SceneNode, row: usize) {
        self.nodes.push(node);
        self.indices.push(row);
    }

    /// Append several nodes all produced by the same source `row` (e.g. a point
    /// `Cross` shape emits 2 line nodes, both mapped to their row). Every node
    /// in the iterator gets the same `row` so `data_indices` keeps exactly one
    /// entry per emitted node, per the #6 contract.
    pub(crate) fn push_many(
        &mut self,
        nodes: impl IntoIterator<Item = SceneNode>,
        row: usize,
    ) {
        for node in nodes {
            self.nodes.push(node);
            self.indices.push(row);
        }
    }

    /// Number of accumulated nodes (equals the number of source-row indices).
    /// Test-only: production builders read the count off the finalized vectors.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether no nodes have been accumulated.
    /// Test-only: production builders read emptiness off the finalized vectors.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Consume the accumulator, yielding the node list and the parallel
    /// source-row index list in node order. Pass the indices straight to
    /// [`MetadataColumns::build_metadata_for_indices`] and set
    /// `MarkBatch::data_indices` from the same vector so node, index, and
    /// metadata stay aligned.
    ///
    /// [`MetadataColumns::build_metadata_for_indices`]: crate::render::draw::MetadataColumns::build_metadata_for_indices
    pub(crate) fn finalize(self) -> (Vec<SceneNode>, Vec<usize>) {
        // Internal invariant: the two vectors grow in lockstep, so this never
        // trips in practice. It documents the contract for any future edit that
        // touches the push path.
        debug_assert_eq!(
            self.nodes.len(),
            self.indices.len(),
            "MarkNodes node/index lockstep violated",
        );
        (self.nodes, self.indices)
    }
}

/// The canonical batch-construction alignment guard (spec §7).
///
/// At the seam where a [`MarkBatch`](ferrum_scene::MarkBatch)'s nodes are paired
/// with their per-node metadata, every present per-node vector must have exactly
/// one entry per node. A builder whose node count diverges from any vector's
/// length (the #6 defect class) trips this immediately under a debug build.
///
/// The five channels — tooltips, hrefs, descriptions, data_indices, and keys —
/// are populated **independently**. A chart with href but no tooltip produces
/// `tooltips == None` while `hrefs == Some(full_row_vec)`. Checking only
/// `tooltips` would leave that case silent. This function therefore takes the
/// length of each present vector as a separate `Option<usize>` and asserts each
/// one independently, so any misaligned channel trips the guard.
///
/// `data_indices` and `keys` are the alignment vector and the key channel
/// respectively; they carry one entry per emitted node (not per input row) and
/// must equal `nodes.len()` exactly like the metadata channels.
///
/// Each parameter is `Some(n)` when the corresponding vector is present (its
/// length), or `None` when that channel is absent for this batch.
///
/// [`MetadataColumns`]: crate::render::draw::MetadataColumns
#[inline]
pub(crate) fn debug_assert_nodes_metadata_aligned(
    nodes_len: usize,
    tooltips_len: Option<usize>,
    hrefs_len: Option<usize>,
    descriptions_len: Option<usize>,
    data_indices_len: Option<usize>,
    keys_len: Option<usize>,
) {
    if let Some(meta_len) = tooltips_len {
        debug_assert_eq!(
            nodes_len, meta_len,
            "mark batch node count ({nodes_len}) must equal tooltip metadata length \
             ({meta_len}); a builder emitted metadata out of lockstep with its nodes \
             (archaeology bug #6 defect class)",
        );
    }
    if let Some(meta_len) = hrefs_len {
        debug_assert_eq!(
            nodes_len, meta_len,
            "mark batch node count ({nodes_len}) must equal href metadata length \
             ({meta_len}); a builder emitted href metadata out of lockstep with its nodes \
             (archaeology bug #6 defect class)",
        );
    }
    if let Some(meta_len) = descriptions_len {
        debug_assert_eq!(
            nodes_len, meta_len,
            "mark batch node count ({nodes_len}) must equal description metadata length \
             ({meta_len}); a builder emitted description metadata out of lockstep with its nodes \
             (archaeology bug #6 defect class)",
        );
    }
    if let Some(di_len) = data_indices_len {
        debug_assert_eq!(
            nodes_len, di_len,
            "mark batch node count ({nodes_len}) must equal data_indices length \
             ({di_len}); a builder emitted data_indices out of lockstep with its nodes \
             (archaeology bug #6 defect class — alignment vector itself misaligned)",
        );
    }
    if let Some(k_len) = keys_len {
        debug_assert_eq!(
            nodes_len, k_len,
            "mark batch node count ({nodes_len}) must equal keys length \
             ({k_len}); a builder emitted keys out of lockstep with its nodes \
             (archaeology bug #6 defect class)",
        );
    }
}

/// The packed-path sibling of [`debug_assert_nodes_metadata_aligned`] (spec
/// §4.3 / GH #93).
///
/// The JSON-path guard above runs while a batch's `keys` vector is still
/// paired with its `nodes` vector, one-to-one. Packing (`pack_instances.rs`)
/// then clears `nodes` and moves `keys` into the `HAS_KEYS` binary sidecar —
/// at that seam the only length left to check `keys` against is the packed
/// instance count already written into the batch's 20-byte header. This
/// mirrors the JSON guard's `data_indices`/`keys` check exactly, just against
/// `instance_count` instead of `nodes.len()`, so a `keys` vector that
/// desynced from its batch (the #6 defect class) trips at pack time instead
/// of silently shipping a misaligned sidecar to the WASM reader.
#[inline]
pub(crate) fn debug_assert_packed_keys_aligned(instance_count: usize, keys_len: usize) {
    debug_assert_eq!(
        instance_count, keys_len,
        "packed batch instance count ({instance_count}) must equal keys length \
         ({keys_len}); a HAS_KEYS sidecar would desync mark instances from their \
         object-constancy keys (archaeology bug #6 defect class, packed-path sibling)",
    );
}

/// Checked form of the alignment guard for unit testing.
///
/// Returns `Ok(())` when the node count agrees with every present per-node
/// vector (or when a channel is absent), and `Err((channel, nodes_len,
/// vec_len))` on the first misaligned channel. `channel` is one of
/// `"tooltip"`, `"href"`, `"description"`, `"data_indices"`, or `"keys"`.
///
/// This lets tests exercise the invariant on a release build without relying
/// on `debug_assert` being compiled in. The production seam uses
/// [`debug_assert_nodes_metadata_aligned`] (see `scene_build.rs`); this mirrors
/// its predicate exactly, checking all five independent channels.
#[cfg(test)]
pub(crate) fn check_nodes_metadata_aligned(
    nodes_len: usize,
    tooltips_len: Option<usize>,
    hrefs_len: Option<usize>,
    descriptions_len: Option<usize>,
    data_indices_len: Option<usize>,
    keys_len: Option<usize>,
) -> Result<(), (&'static str, usize, usize)> {
    if let Some(meta_len) = tooltips_len {
        if meta_len != nodes_len {
            return Err(("tooltip", nodes_len, meta_len));
        }
    }
    if let Some(meta_len) = hrefs_len {
        if meta_len != nodes_len {
            return Err(("href", nodes_len, meta_len));
        }
    }
    if let Some(meta_len) = descriptions_len {
        if meta_len != nodes_len {
            return Err(("description", nodes_len, meta_len));
        }
    }
    if let Some(di_len) = data_indices_len {
        if di_len != nodes_len {
            return Err(("data_indices", nodes_len, di_len));
        }
    }
    if let Some(k_len) = keys_len {
        if k_len != nodes_len {
            return Err(("keys", nodes_len, k_len));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_scene::{FillStroke, SceneNode};

    /// A trivial distinguishable node. The accumulator is geometry-agnostic, so
    /// a circle at `(x, 0)` with radius 1 suffices to tell nodes apart by `cx`.
    fn node(x: f64) -> SceneNode {
        SceneNode::Circle {
            cx: x,
            cy: 0.0,
            r: 1.0,
            style: FillStroke {
                fill: None,
                stroke: None,
                stroke_width: 0.0,
                opacity: 1.0,
                stroke_dash: None,
                stroke_opacity: 1.0,
                fill_opacity: 1.0,
                angle: 0.0,
            },
        }
    }

    fn cx(n: &SceneNode) -> f64 {
        match n {
            SceneNode::Circle { cx, .. } => *cx,
            _ => panic!("expected Circle"),
        }
    }

    /// `push` records one node against one source row; nodes and indices come
    /// out in lockstep and in insertion order.
    #[test]
    fn push_keeps_node_and_index_in_lockstep() {
        let mut acc = MarkNodes::new();
        acc.push(node(0.0), 5);
        acc.push(node(1.0), 9);
        acc.push(node(2.0), 2);
        assert_eq!(acc.len(), 3);
        assert!(!acc.is_empty());

        let (nodes, indices) = acc.finalize();
        assert_eq!(nodes.len(), indices.len(), "nodes and indices must be 1:1");
        assert_eq!(indices, vec![5, 9, 2], "indices preserve the source rows in order");
        let xs: Vec<f64> = nodes.iter().map(cx).collect();
        assert_eq!(xs, vec![0.0, 1.0, 2.0], "nodes preserve insertion order");
    }

    /// A row that is skipped (never pushed) leaves no gap: indices reflect only
    /// the rows that produced a node, in node order — exactly what
    /// `build_metadata_for_indices` consumes.
    #[test]
    fn skipped_rows_produce_no_gap_in_indices() {
        // Rows 0..4; row 1 and row 3 are "skipped" (degenerate / null).
        let mut acc = MarkNodes::with_capacity(4);
        for row in 0..4 {
            if row == 1 || row == 3 {
                continue;
            }
            acc.push(node(row as f64), row);
        }
        let (nodes, indices) = acc.finalize();
        assert_eq!(nodes.len(), 2);
        assert_eq!(indices, vec![0, 2], "only kept rows appear, in node order");
    }

    /// `push_many` maps every node it appends to the SAME source row, so a
    /// multi-node shape (e.g. point `Cross` → 2 nodes) keeps one `data_indices`
    /// entry per emitted node, all pointing at the producing row.
    #[test]
    fn push_many_maps_all_nodes_to_one_row() {
        let mut acc = MarkNodes::new();
        // Row 0 is a single node; row 1 is a 2-node Cross; row 2 single.
        acc.push(node(0.0), 0);
        acc.push_many([node(10.0), node(11.0)], 1);
        acc.push(node(20.0), 2);

        let (nodes, indices) = acc.finalize();
        assert_eq!(nodes.len(), 4, "1 + 2 + 1 nodes");
        // Both Cross nodes (indices[1], indices[2]) map to row 1.
        assert_eq!(indices, vec![0, 1, 1, 2]);
        let xs: Vec<f64> = nodes.iter().map(cx).collect();
        assert_eq!(xs, vec![0.0, 10.0, 11.0, 20.0]);
    }

    /// `push_many` with an empty iterator is a no-op (zero nodes, zero indices).
    #[test]
    fn push_many_empty_iterator_adds_nothing() {
        let mut acc = MarkNodes::new();
        acc.push_many(std::iter::empty(), 7);
        assert!(acc.is_empty());
        let (nodes, indices) = acc.finalize();
        assert!(nodes.is_empty());
        assert!(indices.is_empty());
    }

    // ── Alignment guard ──────────────────────────────────────────────────────

    /// Helper to call `check_nodes_metadata_aligned` with all five channels,
    /// passing `None` for the two new channels when not needed by a test case.
    fn check(
        nodes: usize,
        tooltips: Option<usize>,
        hrefs: Option<usize>,
        descs: Option<usize>,
        data_indices: Option<usize>,
        keys: Option<usize>,
    ) -> Result<(), (&'static str, usize, usize)> {
        check_nodes_metadata_aligned(nodes, tooltips, hrefs, descs, data_indices, keys)
    }

    /// The checked guard accepts aligned node/metadata lengths and the
    /// no-metadata case (`None`), and rejects a deliberate mismatch on any
    /// of the five independent channels.
    #[test]
    fn check_guard_accepts_aligned_rejects_mismatch() {
        // Aligned: 3 nodes, 3 tooltip entries, no other channels.
        assert!(check(3, Some(3), None, None, None, None).is_ok());
        // All five channels aligned.
        assert!(check(3, Some(3), Some(3), Some(3), Some(3), Some(3)).is_ok());
        // No channels present at all → nothing to check.
        assert!(check(3, None, None, None, None, None).is_ok());
        assert!(check(0, None, None, None, None, None).is_ok());
        // Misaligned tooltip: 3 nodes but 5 tooltip entries.
        assert_eq!(check(3, Some(5), None, None, None, None), Err(("tooltip", 3, 5)));
        // Misaligned tooltip the other way.
        assert_eq!(check(5, Some(3), None, None, None, None), Err(("tooltip", 5, 3)));
        // Misaligned href: tooltips absent, but hrefs are wrong.
        // href-without-tooltip: `tooltips == None` while `hrefs == Some(full_row_vec)`.
        assert_eq!(check(2, None, Some(5), None, None, None), Err(("href", 2, 5)));
        // Misaligned description with tooltips + hrefs absent.
        assert_eq!(check(2, None, None, Some(4), None, None), Err(("description", 2, 4)));
        // Tooltips aligned but href misaligned — all channels checked independently.
        assert_eq!(check(3, Some(3), Some(5), None, None, None), Err(("href", 3, 5)));
        // Tooltips aligned, href aligned, description misaligned.
        assert_eq!(check(3, Some(3), Some(3), Some(7), None, None), Err(("description", 3, 7)));
        // Misaligned data_indices: 3 nodes but 5 data_indices entries.
        // This is the label.rs leader-line shape: 2 nodes per row but 1 data_indices entry
        // — the pre-R1 guard could not detect it because metadata channels were all None.
        assert_eq!(check(3, None, None, None, Some(5), None), Err(("data_indices", 3, 5)));
        // Misaligned data_indices the other way (2 nodes, 3 data_indices).
        assert_eq!(check(2, None, None, None, Some(3), None), Err(("data_indices", 2, 3)));
        // All metadata channels aligned, but data_indices misaligned.
        assert_eq!(
            check(3, Some(3), Some(3), Some(3), Some(5), None),
            Err(("data_indices", 3, 5))
        );
        // Misaligned keys: 3 nodes but 2 keys entries.
        assert_eq!(check(3, None, None, None, None, Some(2)), Err(("keys", 3, 2)));
        // All five aligned — guard is silent.
        assert!(check(4, Some(4), Some(4), Some(4), Some(4), Some(4)).is_ok());
        // data_indices aligned, keys aligned, metadata absent.
        assert!(check(4, None, None, None, Some(4), Some(4)).is_ok());
    }

    /// The debug assertion trips on a deliberately misaligned construction
    /// (tooltip channel). Compiled only in debug builds, where `debug_assert!`
    /// is active — which is how `cargo test` runs by default.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "node count")]
    fn debug_assert_guard_trips_on_misaligned_tooltip() {
        // 2 nodes but 3 tooltip metadata entries — the #6 defect class shape.
        debug_assert_nodes_metadata_aligned(2, Some(3), None, None, None, None);
    }

    /// The debug assertion trips when the href channel is misaligned and
    /// tooltips are absent — the pre-fix soundness hole that motivated the
    /// fix round: `tooltips == None` short-circuited the old single-channel
    /// guard, leaving href misalignment silent.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "href metadata length")]
    fn debug_assert_guard_trips_on_href_only_misalignment() {
        // href-without-tooltip: 2 nodes, no tooltips, 5 href entries.
        // The old guard would short-circuit here because tooltips is None.
        debug_assert_nodes_metadata_aligned(2, None, Some(5), None, None, None);
    }

    /// The debug assertion trips when the description channel is misaligned
    /// and both tooltips and hrefs are absent.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "description metadata length")]
    fn debug_assert_guard_trips_on_description_only_misalignment() {
        debug_assert_nodes_metadata_aligned(2, None, None, Some(4), None, None);
    }

    /// The debug assertion trips when data_indices is misaligned and all
    /// metadata channels are absent. This is the pre-R1 guard gap: the label.rs
    /// leader-line builder emitted 2 nodes per row (text + line) but only 1
    /// data_indices entry per row. With metadata all None, the three-channel
    /// guard passed silently — only the extended five-channel guard catches it.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "data_indices length")]
    fn debug_assert_guard_trips_on_data_indices_misalignment() {
        // 2 nodes per row, 3 rows = 6 nodes; but only 3 data_indices entries (one per row).
        // Represents the pre-fix label.rs leader-line shape: 2N nodes, N data_indices.
        debug_assert_nodes_metadata_aligned(6, None, None, None, Some(3), None);
    }

    /// The debug assertion trips when keys is misaligned while all other
    /// channels (including data_indices) are absent.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "keys length")]
    fn debug_assert_guard_trips_on_keys_misalignment() {
        // 4 nodes but only 2 keys entries.
        debug_assert_nodes_metadata_aligned(4, None, None, None, None, Some(2));
    }

    // ── Packed-path keys guard (spec §4.3 / GH #93) ─────────────────────────

    /// The debug assertion trips when a packed batch's keys vector desyncs
    /// from the instance count already written into the packed header —
    /// exactly the failure mode `pack_instances.rs`'s `HAS_KEYS` writer must
    /// catch before it ships a misaligned sidecar to the WASM reader.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "instance count")]
    fn debug_assert_packed_keys_guard_trips_on_mismatch() {
        // 1200 packed instances but only 1199 keys — one short.
        debug_assert_packed_keys_aligned(1200, 1199);
    }

    /// The debug assertion passes silently when the packed instance count and
    /// keys length agree.
    #[test]
    fn debug_assert_packed_keys_guard_silent_when_aligned() {
        debug_assert_packed_keys_aligned(1200, 1200);
        debug_assert_packed_keys_aligned(0, 0);
    }

    /// The debug assertion passes silently when all present channels are
    /// aligned or when no channels are present (the no-metadata batch case).
    #[test]
    fn debug_assert_guard_silent_when_aligned_or_no_metadata() {
        // All five aligned.
        debug_assert_nodes_metadata_aligned(4, Some(4), Some(4), Some(4), Some(4), Some(4));
        // Tooltips aligned, no other channels.
        debug_assert_nodes_metadata_aligned(4, Some(4), None, None, None, None);
        // href only, aligned.
        debug_assert_nodes_metadata_aligned(4, None, Some(4), None, None, None);
        // desc only, aligned.
        debug_assert_nodes_metadata_aligned(4, None, None, Some(4), None, None);
        // data_indices only, aligned.
        debug_assert_nodes_metadata_aligned(4, None, None, None, Some(4), None);
        // keys only, aligned.
        debug_assert_nodes_metadata_aligned(4, None, None, None, None, Some(4));
        // data_indices + keys, aligned, metadata absent.
        debug_assert_nodes_metadata_aligned(4, None, None, None, Some(4), Some(4));
        // No channels at all.
        debug_assert_nodes_metadata_aligned(4, None, None, None, None, None);
    }
}
