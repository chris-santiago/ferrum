//! Placement policy — which model a mark batch's geometry meets a broken axis
//! under, plus the off-screen coordinate the hide paths park geometry at.
//!
//! Split out of `break_axis/mod.rs` (#229). The issue's filed four-way seam
//! (`scale`/`indicators`/`remap`/`clip`) left this job homeless even though
//! its own prose names placement as one of the five: the `MarkBatchKind` →
//! [`BreakPlacement`] dispatch is policy rather than geometry, and it is the
//! one item `render/annotation`'s `CutRouting` has to be kept in
//! hand-maintained correspondence with. Giving it a named file is what makes
//! that correspondence findable. Precedent for the split:
//! `render/scale_resolve/`, `projection/`, `transform/stats/`,
//! `transform/smooth/`.

use ferrum_scene::MarkBatchKind;

/// Coordinate used to park geometry that falls inside a break gap off-screen.
///
/// A large off-screen value rather than NaN, which would panic the SVG
/// serializer. Which geometry is hidden this way, and which is clipped
/// instead, is decided per mark batch by [`BreakPlacement`].
pub const BREAK_HIDDEN: f64 = -99999.0;

/// Whether `c` is the hidden-sentinel pixel [`BreakRemap::remap_node`]'s
/// `Line` arm writes for a coordinate that fell inside a gap.
///
/// Exact equality, not a tolerance band: `remap_node`'s `Line` arm writes
/// `BREAK_HIDDEN` verbatim (`*a = self.coord(*a).unwrap_or(BREAK_HIDDEN)`,
/// no offset), which is the producer [`BreakRemap::data_at`]'s poison guard
/// (#230-9) reads from when a coordinate a prior step already hid is handed
/// back through `data_at`/`coord` — so `== BREAK_HIDDEN` is the precise
/// question "did this coordinate hide", not an approximation of it, for all
/// three of its callers: `data_at`, [`BreakRemap::clip_extent`] (whose own
/// poison guard on its raw input pixels was added cycle-2), and
/// [`BreakRemap::remap_arrowhead`]'s shaft-direction filter (defensive since
/// the split hands it placed pieces, never sentinels).
/// [`BreakRemap::remap_glyph_node`]'s hide path is a different producer: it
/// offsets a multi-coordinate glyph's other coordinates by `c - anchor`
/// around the sentinel, so those coordinates do NOT satisfy this exact
/// test — a caller reading glyph-hidden coordinates needs its own,
/// necessarily inexact, threshold. A real pixel is never within a fraction
/// of `BREAK_HIDDEN` either way, so the distinction is academic in
/// practice; it is named here so the next caller does not have to
/// re-derive which producer it is reading from.
///
/// [`BreakRemap::clip_extent`]: super::BreakRemap::clip_extent
/// [`BreakRemap::data_at`]: super::BreakRemap::data_at
/// [`BreakRemap::remap_arrowhead`]: super::BreakRemap::remap_arrowhead
/// [`BreakRemap::remap_glyph_node`]: super::BreakRemap::remap_glyph_node
/// [`BreakRemap::remap_node`]: super::BreakRemap::remap_node
pub(super) fn is_hidden(c: f64) -> bool {
    c == BREAK_HIDDEN
}

/// How a mark batch's geometry meets a broken axis.
///
/// The three models exist because "the mark is inside the gap" means three
/// different things depending on what the mark's pixels encode, and applying
/// one model to all of them is what produced the defects this enum replaces:
/// extent-clipping a *glyph* dropped in-gap diamonds and squares out of the
/// node list (losing their `data_indices`/tooltip rows) while in-gap circles
/// were merely hidden, and cut the surviving ones flat at the gap edge.
///
/// The mapping is per [`MarkBatchKind`] and exhaustive, so a new mark kind has
/// to answer this question rather than inherit an accident.
///
/// `annotation::CutRouting` (private to that module, so not a resolvable
/// doc link from here) asks this same question for annotations and answers
/// it with the same three placements, under the same names (`ClipExtent`,
/// `MoveVertices`, `MoveVerticesAndGlyph`) minus the two glyph-only variants
/// annotations never need. The two enums are not linked by the type system —
/// keep them in correspondence by hand if either grows a new variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakPlacement {
    /// The mark's pixel *extent* along the axis is data (a bar's height, an
    /// area's band, a violin's outline), so it is clipped to the visible
    /// segments: one sub-mark per segment it intersects, none when it lies
    /// wholly inside a gap. A `Polygon` node clips its rings in place, so it
    /// stays one node with one row. See [`BreakRemap::clip_nodes`].
    ///
    /// [`BreakRemap::clip_nodes`]: super::BreakRemap::clip_nodes
    ClipExtent,
    /// The mark is a fixed-size glyph anchored at one data position (a point's
    /// shape, a tick or rug dash, a text label). It translates to its anchor's
    /// compressed position — keeping its drawn size and shape — and hides whole
    /// when the anchor is inside a gap. Never resized, never split, never
    /// dropped, so glyph batches stay 1:1 with their rows whatever `shape=`
    /// they carry. See [`BreakRemap::remap_glyph_node`].
    ///
    /// [`BreakRemap::remap_glyph_node`]: super::BreakRemap::remap_glyph_node
    MoveGlyph,
    /// The mark emits a glyph plus a *connector* to it: `mark_label`'s text
    /// with its optional leader line. The text moves as a glyph; the leader's
    /// two ends are two different data positions, so each is placed on its own
    /// and the data end keeps touching its data point.
    ///
    /// This exists as its own model — rather than as a node-shape test inside
    /// [`BreakPlacement::MoveGlyph`] — because `SceneNode::Line` has two
    /// producers with opposite meanings: a label's leader (a connector) and a
    /// tick's dash (drawn size, not data). Only the batch kind can tell them
    /// apart. See [`BreakRemap::remap_label_node`].
    ///
    /// [`BreakRemap::remap_label_node`]: super::BreakRemap::remap_label_node
    MoveGlyphAndConnector,
    /// The mark is connected *stroked* geometry whose every vertex is its own
    /// data position (a line, a rule, a segment shaft). It is SPLIT into one
    /// sub-node per visible piece: the stroke is cut at the gap edges it
    /// crosses, each piece carries the vertices lying in one segment plus the
    /// interpolated crossing points where it enters and leaves that segment,
    /// and the portion inside a gap is not drawn at all. See
    /// [`BreakRemap::split_nodes`].
    ///
    /// The sentinel model this replaces placed each vertex independently and
    /// parked the in-gap ones at [`BREAK_HIDDEN`], which left the stroke
    /// *connected* to them: a line crossing a gap drew a spike off-canvas and
    /// back (#230-1, measured as
    /// `<polyline points="74.714,420.961 345.357,-99999 616,16">`). It
    /// survives as the 1→1 fallback for stroked geometry no faithful split
    /// exists for — a curved `Path`, whose crossing point is not a vertex of
    /// anything this module can re-spell ([`BreakRemap::remap_node`]).
    ///
    /// [`BreakRemap::remap_node`]: super::BreakRemap::remap_node
    /// [`BreakRemap::split_nodes`]: super::BreakRemap::split_nodes
    MoveVertices,
    /// The mark is vertex-placed stroked geometry that also carries a
    /// fixed-size glyph derived from it: `mark_segment`'s shaft plus its
    /// optional `arrow=True` arrowhead. The shaft splits per visible piece
    /// (exactly [`BreakPlacement::MoveVertices`]); the head is then REBUILT at
    /// its tip's placed position from the endpoints of the piece that REACHES
    /// that tip, so it keeps both its drawn size and its alignment with the
    /// geometry it terminates, whose slope the break changes.
    ///
    /// The mirror of [`BreakPlacement::MoveGlyphAndConnector`], and it exists
    /// for the same reason: only the batch kind knows that a closed `Path` in
    /// a `Segment` batch is an arrowhead. Placing its three vertices one by
    /// one scaled the head by the segment's own compression ratio — a 20px
    /// head became ~90px tall and sheared under a `(150, 850)` gap; rigidly
    /// translating it instead fixed the size but froze the pre-break
    /// direction, leaving the head pointing 16° off the shaft it terminates.
    /// A head whose tip falls inside a gap is dropped: no piece reaches it, so
    /// there is nothing for it to terminate.
    /// See [`BreakRemap::split_segment_nodes`].
    ///
    /// [`BreakRemap::split_segment_nodes`]: super::BreakRemap::split_segment_nodes
    MoveVerticesAndGlyph,
}

impl BreakPlacement {
    /// The placement model for a mark batch of `kind`.
    ///
    /// Exhaustive by construction: a new [`MarkBatchKind`] is a compile error
    /// here rather than an inherited default.
    pub fn for_kind(kind: MarkBatchKind) -> Self {
        match kind {
            // Extent geometry: the span between two data values is the mark.
            // `Polygon` (mark_polygon, mark_geoshape, and mark_violin, which
            // lowers to the polygon mark) is here rather than with the lines
            // because its rings are FILLED: placing their vertices one by one
            // would park the in-gap ones off-canvas and paint the fill as a
            // column across the break, which is worse than the unplaced render
            // it replaced. Clipping the rings is the same answer the area fill
            // already gets.
            MarkBatchKind::Bar
            | MarkBatchKind::Area
            | MarkBatchKind::Rect
            | MarkBatchKind::Ribbon
            | MarkBatchKind::Polygon => Self::ClipExtent,
            // Glyphs: one data position, a drawn size that is not data. `Tick`
            // is here for its dash (a rug's 8px mark is drawn size, so it must
            // survive compression intact); `Image` for the same reason even
            // though no image node is moved yet — see
            // [`BreakRemap::for_each_along`], which names it a deliberate no-op.
            MarkBatchKind::Point
            | MarkBatchKind::Tick
            | MarkBatchKind::Text
            | MarkBatchKind::Image => Self::MoveGlyph,
            // `mark_label` alone mixes a glyph with a connector to it.
            MarkBatchKind::Label => Self::MoveGlyphAndConnector,
            // Connected stroked geometry: every vertex is a data position.
            // `Arc` is polar-only geometry (its batches come from `mark_arc`
            // and polar bars). A chart combining `Polar` coordinates with any
            // `BreakAxis` now refuses at the render funnel
            // (`RenderError::BreakAxisIncompatibleCoord`, #230-8) before this
            // placement model — or any other break geometry — is ever built,
            // so this arm is unreachable for a real polar+break chart. (The
            // prior claim here, that Cartesian breaks are "suppressed" under
            // polar coords, was false: polar projection runs during batch
            // build, then the break fold reverse-mapped the already-projected
            // pixels through the Cartesian scale it replaced, destroying the
            // geometry silently.) It keeps the vertex model — not a separate
            // polar one — as the defensive default should `Arc` ever reach
            // this match under a non-polar coord.
            MarkBatchKind::Line | MarkBatchKind::Rule | MarkBatchKind::Arc => Self::MoveVertices,
            // `mark_segment` alone mixes vertex-placed shafts with an attached
            // fixed-size glyph (`arrow=True`'s head).
            MarkBatchKind::Segment => Self::MoveVerticesAndGlyph,
        }
    }
}
