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
//! MarkBuildResult { kind, nodes, data_indices: Some(data_indices), tooltips, hrefs, descriptions, polar_spans: None }
//! ```

use ferrum_scene::SceneNode;

use crate::render::draw::MarkBuildResult;

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

/// The node count of a built batch alongside the length of each per-node
/// channel it carries — the input to the alignment guard (spec §7).
///
/// The six channels — tooltips, hrefs, descriptions, data_indices, keys, and
/// polar_spans — are populated **independently**. A chart with href but no
/// tooltip produces `tooltips == None` while `hrefs == Some(full_row_vec)`, so
/// each channel carries its own `Option<usize>` and is checked on its own;
/// checking only `tooltips` would leave that case silent.
///
/// Named fields, not positional arguments: the guard previously took six
/// adjacent same-typed `Option<usize>` parameters, and every call site was a
/// chain like `(4, None, None, None, Some(4), None, Some(4))`. A transposed
/// argument compiled silently and made the guard check the wrong channel —
/// which is exactly the defect class the guard exists to catch. Build one of
/// these with [`Self::of`] at a production seam; only tests name the fields
/// directly, to construct deliberate misalignments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct NodeChannelLengths {
    /// The batch's emitted node count — what every present channel must match.
    pub(crate) nodes: usize,
    pub(crate) tooltips: Option<usize>,
    pub(crate) hrefs: Option<usize>,
    pub(crate) descriptions: Option<usize>,
    /// The alignment vector itself. One entry per emitted node (not per input
    /// row), so it must equal `nodes` exactly like the metadata channels.
    pub(crate) data_indices: Option<usize>,
    /// The object-constancy key channel, built beside the batch rather than by
    /// the mark builder — the one length [`Self::of`] cannot read off the
    /// build result.
    pub(crate) keys: Option<usize>,
    /// One `(angle_start, angle_end)` per emitted wedge node (batch E Task 2),
    /// populated only by `arc.rs`'s three builders and `bar.rs::build_polar`.
    /// A misaligned `polar_spans` is worse than a misaligned tooltip: it makes
    /// hit-testing itself resolve node `j`'s hover to node `k`'s span, so the
    /// tooltip, selection, and emitted `data_index` all report a different
    /// row's data, silently.
    pub(crate) polar_spans: Option<usize>,
}

impl NodeChannelLengths {
    /// Read every length straight off a finished [`MarkBuildResult`], plus the
    /// separately-built `keys` vector.
    ///
    /// This is the only shape a production seam should use: the lengths are
    /// derived from the very vectors that are about to be moved into the
    /// `MarkBatch`, so there is no argument order for a caller to get wrong.
    pub(crate) fn of(result: &MarkBuildResult, keys: Option<&[String]>) -> Self {
        Self {
            nodes: result.nodes.len(),
            tooltips: result.tooltips.as_ref().map(Vec::len),
            hrefs: result.hrefs.as_ref().map(Vec::len),
            descriptions: result.descriptions.as_ref().map(Vec::len),
            data_indices: result.data_indices.as_ref().map(Vec::len),
            keys: keys.map(<[String]>::len),
            polar_spans: result.polar_spans.as_ref().map(Vec::len),
        }
    }

    /// The six channels paired with the name the guard reports, in check order.
    fn channels(&self) -> [(&'static str, Option<usize>); 6] {
        [
            ("tooltip", self.tooltips),
            ("href", self.hrefs),
            ("description", self.descriptions),
            ("data_indices", self.data_indices),
            ("keys", self.keys),
            ("polar_spans", self.polar_spans),
        ]
    }
}

/// The first channel whose length disagreed with the batch's node count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChannelMisalignment {
    /// One of `"tooltip"`, `"href"`, `"description"`, `"data_indices"`,
    /// `"keys"`, `"polar_spans"`.
    pub(crate) channel: &'static str,
    pub(crate) nodes: usize,
    pub(crate) channel_len: usize,
}

impl std::fmt::Display for ChannelMisalignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { channel, nodes, channel_len } = self;
        write!(
            f,
            "mark batch node count ({nodes}) must equal {channel} length ({channel_len}); \
             a builder emitted a per-node channel out of lockstep with its nodes \
             (archaeology bug #6 defect class)"
        )
    }
}

/// The alignment predicate (spec §7): every present per-node vector must have
/// exactly one entry per node.
///
/// Returns the first misaligned channel, or `Ok(())` when every present channel
/// agrees with the node count (and when a channel is absent, which is not a
/// disagreement). Tests call this directly so the invariant is exercised even on
/// a release build; the production seam wraps it in
/// [`debug_assert_nodes_metadata_aligned`], so there is one predicate rather
/// than two mirrored ones.
///
/// [`MetadataColumns`]: crate::render::draw::MetadataColumns
pub(crate) fn check_nodes_metadata_aligned(
    lengths: NodeChannelLengths,
) -> Result<(), ChannelMisalignment> {
    for (channel, len) in lengths.channels() {
        if let Some(channel_len) = len {
            if channel_len != lengths.nodes {
                return Err(ChannelMisalignment {
                    channel,
                    nodes: lengths.nodes,
                    channel_len,
                });
            }
        }
    }
    Ok(())
}

/// The canonical batch-construction alignment guard (spec §7).
///
/// At the seam where a [`MarkBatch`](ferrum_scene::MarkBatch)'s nodes are paired
/// with their per-node channels, a builder whose node count diverges from any
/// channel's length (the #6 defect class) trips this immediately under a debug
/// build. Release builds compile it out, exactly as before.
#[inline]
pub(crate) fn debug_assert_nodes_metadata_aligned(lengths: NodeChannelLengths) {
    // `cfg!` (not `#[cfg]`) so the block still type-checks in release, where it
    // folds to `if false` and the predicate call goes with it — the guard costs
    // a release render nothing. This is the gate `debug_assert!` would expand
    // to; hoisting it keeps the predicate inside it too, so a plain `panic!` is
    // the whole of the failure path.
    if cfg!(debug_assertions) {
        if let Err(misalignment) = check_nodes_metadata_aligned(lengths) {
            panic!("{misalignment}");
        }
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

    /// `n` nodes with no per-node channel present — the base every case below
    /// varies exactly one field of, which is what naming the fields buys.
    fn lengths(nodes: usize) -> NodeChannelLengths {
        NodeChannelLengths { nodes, ..Default::default() }
    }

    fn misaligned(channel: &'static str, nodes: usize, channel_len: usize) -> ChannelMisalignment {
        ChannelMisalignment { channel, nodes, channel_len }
    }

    /// The predicate accepts aligned node/channel lengths and the no-channel
    /// case, and rejects a deliberate mismatch on each of the six independent
    /// channels — including the shapes where the channels BEFORE the broken one
    /// are absent (`tooltips == None` used to short-circuit the whole guard).
    #[test]
    fn check_guard_accepts_aligned_rejects_mismatch() {
        // Aligned: 3 nodes, 3 tooltip entries, no other channels.
        assert!(check_nodes_metadata_aligned(NodeChannelLengths {
            tooltips: Some(3),
            ..lengths(3)
        })
        .is_ok());
        // All six channels aligned.
        assert!(check_nodes_metadata_aligned(NodeChannelLengths {
            nodes: 3,
            tooltips: Some(3),
            hrefs: Some(3),
            descriptions: Some(3),
            data_indices: Some(3),
            keys: Some(3),
            polar_spans: Some(3),
        })
        .is_ok());
        // No channels present at all → nothing to check.
        assert!(check_nodes_metadata_aligned(lengths(3)).is_ok());
        assert!(check_nodes_metadata_aligned(lengths(0)).is_ok());

        // One misaligned channel at a time, every other channel absent — the
        // href/description/data_indices/keys/polar_spans rows are the ones the
        // old single-channel guard could not see.
        let solo: [(&str, fn(usize) -> NodeChannelLengths); 6] = [
            ("tooltip", |n| NodeChannelLengths { tooltips: Some(n), ..lengths(3) }),
            ("href", |n| NodeChannelLengths { hrefs: Some(n), ..lengths(3) }),
            ("description", |n| NodeChannelLengths { descriptions: Some(n), ..lengths(3) }),
            ("data_indices", |n| NodeChannelLengths { data_indices: Some(n), ..lengths(3) }),
            ("keys", |n| NodeChannelLengths { keys: Some(n), ..lengths(3) }),
            ("polar_spans", |n| NodeChannelLengths { polar_spans: Some(n), ..lengths(3) }),
        ];
        for (channel, build) in solo {
            assert_eq!(
                check_nodes_metadata_aligned(build(5)),
                Err(misaligned(channel, 3, 5)),
                "{channel}: 3 nodes against 5 entries must be reported"
            );
            assert_eq!(
                check_nodes_metadata_aligned(build(3)),
                Ok(()),
                "{channel}: 3 nodes against 3 entries is aligned"
            );
        }

        // Earlier channels aligned, a later one broken: every channel is
        // checked, not just the first present one.
        assert_eq!(
            check_nodes_metadata_aligned(NodeChannelLengths {
                tooltips: Some(3),
                hrefs: Some(5),
                ..lengths(3)
            }),
            Err(misaligned("href", 3, 5))
        );
        assert_eq!(
            check_nodes_metadata_aligned(NodeChannelLengths {
                nodes: 3,
                tooltips: Some(3),
                hrefs: Some(3),
                descriptions: Some(3),
                data_indices: Some(5),
                keys: None,
                polar_spans: None,
            }),
            Err(misaligned("data_indices", 3, 5))
        );
        // The label.rs leader-line shape: 2 nodes per row over 3 rows, but only
        // one data_indices entry per row, with every metadata channel absent.
        assert_eq!(
            check_nodes_metadata_aligned(NodeChannelLengths {
                data_indices: Some(3),
                ..lengths(6)
            }),
            Err(misaligned("data_indices", 6, 3))
        );
    }

    /// `NodeChannelLengths::of` reads the lengths off the build result itself,
    /// so the production seam has no argument order to transpose. A channel the
    /// builder left `None` must stay `None` — reporting it as `Some(0)` would
    /// make every non-empty batch trip the guard.
    #[test]
    fn of_reads_each_channel_length_off_the_build_result() {
        use ferrum_scene::MarkBatchKind;

        let mut result = MarkBuildResult::empty(MarkBatchKind::Arc);
        result.nodes = vec![node(0.0), node(1.0), node(2.0)];
        result.data_indices = Some(vec![0, 1, 2]);
        result.polar_spans = Some(vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]);
        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        assert_eq!(
            NodeChannelLengths::of(&result, Some(&keys)),
            NodeChannelLengths {
                nodes: 3,
                tooltips: None,
                hrefs: None,
                descriptions: None,
                data_indices: Some(3),
                keys: Some(3),
                polar_spans: Some(3),
            },
            "an Arc batch carries data_indices + polar_spans; tooltip/href/description are absent"
        );
        assert_eq!(
            NodeChannelLengths::of(&result, None).keys,
            None,
            "an unkeyed batch reports `keys: None`, not `Some(0)`"
        );

        // The guard is silent on that real batch shape, and trips the moment
        // one channel desyncs.
        debug_assert_nodes_metadata_aligned(NodeChannelLengths::of(&result, Some(&keys)));
        result.polar_spans = Some(vec![(0.0, 1.0)]);
        assert_eq!(
            check_nodes_metadata_aligned(NodeChannelLengths::of(&result, Some(&keys))),
            Err(misaligned("polar_spans", 3, 1))
        );
    }

    /// The debug assertion trips on a deliberately misaligned construction,
    /// naming the channel. Compiled only in debug builds, where `debug_assert!`
    /// is active — which is how `cargo test` runs by default.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "must equal tooltip length")]
    fn debug_assert_guard_trips_on_misaligned_tooltip() {
        // 2 nodes but 3 tooltip metadata entries — the #6 defect class shape.
        debug_assert_nodes_metadata_aligned(NodeChannelLengths {
            tooltips: Some(3),
            ..lengths(2)
        });
    }

    /// The debug assertion trips when `polar_spans` is misaligned while every
    /// other channel is absent — the shape that makes hit-testing resolve one
    /// wedge's hover to a different wedge's angular span, silently, and the one
    /// a guard that stopped at the first absent channel would miss.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "must equal polar_spans length")]
    fn debug_assert_guard_trips_on_polar_spans_misalignment() {
        // 4 nodes (wedges) but only 2 polar_spans entries.
        debug_assert_nodes_metadata_aligned(NodeChannelLengths {
            polar_spans: Some(2),
            ..lengths(4)
        });
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
        // All six aligned.
        debug_assert_nodes_metadata_aligned(NodeChannelLengths {
            nodes: 4,
            tooltips: Some(4),
            hrefs: Some(4),
            descriptions: Some(4),
            data_indices: Some(4),
            keys: Some(4),
            polar_spans: Some(4),
        });
        // Each channel present alone, aligned.
        let solo: [fn(usize) -> NodeChannelLengths; 6] = [
            |n| NodeChannelLengths { tooltips: Some(n), ..lengths(n) },
            |n| NodeChannelLengths { hrefs: Some(n), ..lengths(n) },
            |n| NodeChannelLengths { descriptions: Some(n), ..lengths(n) },
            |n| NodeChannelLengths { data_indices: Some(n), ..lengths(n) },
            |n| NodeChannelLengths { keys: Some(n), ..lengths(n) },
            |n| NodeChannelLengths { polar_spans: Some(n), ..lengths(n) },
        ];
        for build in solo {
            debug_assert_nodes_metadata_aligned(build(4));
        }
        // data_indices + polar_spans, aligned, metadata and keys absent — the
        // real shape of an Arc-kind batch (`arc.rs`/`bar.rs::build_polar`
        // populate both, no `keys` unless an explicit `key=` encoding).
        debug_assert_nodes_metadata_aligned(NodeChannelLengths {
            data_indices: Some(4),
            polar_spans: Some(4),
            ..lengths(4)
        });
        // No channels at all.
        debug_assert_nodes_metadata_aligned(lengths(4));
    }
}
