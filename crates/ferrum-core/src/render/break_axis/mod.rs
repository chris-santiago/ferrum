//! Axis break — piecewise scale that omits a data range plus visual indicators.
//!
//! An axis break removes one or more contiguous ranges from the data domain and
//! inserts a break indicator (slash, zigzag, wave, or blank gap) at the
//! corresponding pixel position. The remaining segments are stretched uniformly
//! to fill the available pixel space minus the indicator widths.
//!
//! # Coordinate conventions
//! The pixel range follows SVG conventions: the origin is at the top-left, so
//! for the Y axis `plot_top` (small pixel value) corresponds to a large data
//! value and `plot_bottom` (large pixel value) corresponds to a small data value.
//!
//! # Module layout
//! The four jobs that do not need `BreakRemap`'s private state live in their
//! own files: [`scale`] (the piecewise domain/pixel math), [`indicators`] (the
//! slash/zigzag/wave glyphs), [`placement`] (which model a mark batch's
//! geometry meets a break under) and [`geometry`] (the `self`-free polygon and
//! path clipping). What remains here is the scene rewrite: [`BreakRemap`] and
//! its single `impl`, kept whole deliberately. `#229` proposed cutting that
//! `impl` along a remap/clip line, but it is one block whose two halves share
//! a dozen private accessors over four private fields, so the cut would have
//! had to publish `axis`, `domain`, `pixels`, `result` and `ClippedNodes`'
//! `sources`/`source_len` to a sibling module — fields whose invariants the
//! types depend on. The four files above are the seam that needs no such
//! widening.

mod geometry;
mod indicators;
mod placement;
mod scale;

use std::collections::HashSet;

use ferrum_scene::{FillStroke, PathCmd, SceneNode};

use crate::render::marks;
use crate::spec::axis::AlignAxis;

pub use self::indicators::build_break_indicators;
pub use self::placement::{BreakPlacement, BREAK_HIDDEN};
pub use self::scale::{
    apply_break_to_scale, broken_scale_map, gap_containing, gap_spanned_by, BreakResult,
    ScaleSegment,
};

use self::geometry::{
    clip_to_band, path_commands, polygon_points, push_line_pieces, rounded_rect_piece,
    set_triangle_vertices, straight_path_points, triangle_vertices,
};
use self::placement::is_hidden;

// ── Scene remapping ───────────────────────────────────────────────────────────

/// A break resolved against the unbroken scale it replaces.
///
/// Scene nodes are built against the unbroken scale, so re-placing them on the
/// broken axis is a round trip: reverse-map a pixel coordinate to data through
/// the unbroken scale, then forward-map that value through the piecewise
/// [`BreakResult`]. This type carries the four numbers that round trip needs
/// alongside the axis they apply to, so every caller (marks, gridlines, axis
/// chrome) performs it identically.
///
/// `Copy` because it is four borrowed/scalar fields and callers hold it in
/// slices they need to re-partition: the annotation router builds a
/// subset of the panel's remaps (the axes whose gaps it is NOT routing through
/// the clip) to hand back to `ScaleContext`, which is a filter over a slice,
/// not a new resolution.
#[derive(Clone, Copy)]
pub struct BreakRemap<'a> {
    /// Which coordinate of a node this break moves.
    axis: AlignAxis,
    /// `(lo, hi)` data domain of the unbroken scale.
    domain: (f64, f64),
    /// `(lo, hi)` pixel range of the unbroken scale, in that scale's own
    /// orientation (for Y this is `(bottom, top)`).
    pixels: (f64, f64),
    result: &'a BreakResult,
}

/// The outcome of rebuilding a node list through a break.
///
/// Both rewrites a break performs are 1→N, and both produce this: the extent
/// clip ([`BreakRemap::clip_nodes`] — a bar crossing one gap becomes two rects,
/// a bar whose whole extent sits inside a gap becomes none) and the vertex
/// split ([`BreakRemap::split_nodes`] / [`BreakRemap::split_segment_nodes`] — a
/// line crossing one gap becomes two sub-lines, a rule inside a gap becomes
/// none). The type is named for the first but belongs to both; `sources`
/// records, for each emitted node, the index of the node it came from, so the
/// batch's per-node sidecars (`data_indices`, tooltips, hrefs, descriptions,
/// keys, polar spans) can be rebuilt in lockstep instead of silently drifting
/// out of alignment.
pub struct ClippedNodes {
    /// The rebuilt node list, in paint order.
    pub nodes: Vec<SceneNode>,
    sources: Vec<usize>,
    source_len: usize,
}

impl ClippedNodes {
    /// True when the rebuild changed the node list's shape, i.e. when
    /// `nodes[i]` no longer comes from source node `i`.
    pub fn is_reindexed(&self) -> bool {
        self.sources.len() != self.source_len
            || self.sources.iter().enumerate().any(|(i, &src)| i != src)
    }

    /// Rebuild one per-node sidecar vector so it stays aligned with `nodes`.
    ///
    /// A sidecar whose length does not match the pre-clip node count is
    /// already misaligned (a builder bug the debug-only alignment guard in
    /// `mark_nodes` exists to catch); it is passed through untouched rather
    /// than indexed into, so a release build cannot panic here.
    ///
    /// This is the DIRECT entry point, and its `debug_assert` guards a caller
    /// that hands over a wrong-length sidecar without having asked. The
    /// production path does not come through here blind: it uses
    /// [`ClippedNodes::realigner`], which asks
    /// [`ClippedNodes::accepts_sidecar_len`] first, records the answer, and
    /// returns a misaligned sidecar untouched without reaching this assert —
    /// so a misattributed batch is REPORTED (as
    /// `RenderWarning::BreakClipSidecarMisaligned`) in both build profiles,
    /// and that reporting has test coverage in the debug gate the project
    /// actually runs.
    pub fn realign<T: Clone>(&self, sidecar: Option<Vec<T>>) -> Option<Vec<T>> {
        let sidecar = match self.checked_sidecar(sidecar?) {
            Ok(sidecar) => sidecar,
            Err(sidecar) => return Some(sidecar),
        };
        Some(self.sources.iter().map(|&i| sidecar[i].clone()).collect())
    }

    /// The wrong-length guard [`ClippedNodes::realign`] and
    /// [`ClippedNodes::realign_keys`] both need before they may index into a
    /// sidecar — extracted so the "one rule" [`ClippedNodes::accepts_sidecar_len`]
    /// documents cannot drift into two implementations that quietly disagree.
    /// `Err` hands the sidecar straight back, untouched: a release build must
    /// never index a wrong-length sidecar, only report it (via
    /// [`SidecarRealign`]).
    fn checked_sidecar<T>(&self, sidecar: Vec<T>) -> Result<Vec<T>, Vec<T>> {
        debug_assert_eq!(
            sidecar.len(),
            self.source_len,
            "per-node sidecar must have one entry per pre-clip node"
        );
        if sidecar.len() == self.source_len {
            Ok(sidecar)
        } else {
            Err(sidecar)
        }
    }

    /// Start realigning a batch's per-node sidecars, accumulating whether any
    /// of them was already misaligned. See [`SidecarRealign`].
    pub fn realigner(&self) -> SidecarRealign<'_> {
        SidecarRealign { clipped: self, misaligned: false }
    }

    /// True when a sidecar of `len` entries can be realigned — i.e. `len` is
    /// the PRE-clip node count, one entry per source node. The single rule
    /// behind both [`ClippedNodes::realign`]'s pass-through and
    /// [`SidecarRealign`]'s misalignment flag.
    pub fn accepts_sidecar_len(&self, len: usize) -> bool {
        len == self.source_len
    }

    /// Realign a batch's `keys` sidecar, disambiguating a 1→N split.
    ///
    /// Every other sidecar this type realigns is a legitimate verbatim copy —
    /// two pieces of one split bar share one `data_index`, one tooltip, one
    /// href, because they ARE one row. `keys` is different: WASM's transition
    /// pairing (`ferrum-wasm`'s `transition.rs`) treats key equality as row
    /// identity, so two nodes sharing a key when they came from a 1→N split
    /// (not from two distinct rows) makes the whole batch's keyed pairing
    /// degrade to `IndexZip` (#230-5). The fix is suffixing, not a new
    /// sidecar: within each run of pieces sharing a source index — `sources`
    /// groups them contiguously, since [`rebuild_nodes`] finishes emitting
    /// one source's pieces before starting the next — the first piece keeps
    /// the original key verbatim and each subsequent piece gets `{key}#{i}`
    /// (spec §6; see [`Self::suffix_keys`] for why `i` is not simply the
    /// run position). An unkeyed batch (`sidecar` is `None`) is untouched,
    /// same as [`ClippedNodes::realign`].
    pub fn realign_keys(&self, sidecar: Option<Vec<String>>) -> Option<Vec<String>> {
        let sidecar = match self.checked_sidecar(sidecar?) {
            Ok(sidecar) => sidecar,
            Err(sidecar) => return Some(sidecar),
        };
        Some(self.suffix_keys(&sidecar))
    }

    /// Suffix `sidecar` (one entry per PRE-clip source node, already length
    /// checked by [`Self::realign_keys`]) into one key per emitted piece.
    ///
    /// A chart broken on two axes runs `install_rebuilt_nodes` — and this
    /// method — ONCE PER AXIS (`scene_build.rs`'s `break_remaps` loop), so
    /// the second pass's `sidecar` is the FIRST pass's already-suffixed
    /// output, not the original user keys. Naively resuming the `#1, #2, ...`
    /// counter from zero in each pass can mint a suffix the first pass
    /// already used (`"ka#1"` reappearing once from pass one's second piece
    /// and again from pass two suffixing pass one's `"ka#1"` piece), which is
    /// exactly the duplicate this whole mechanism exists to prevent — so
    /// checking only "is this the same run" is not enough; the check has to
    /// be "is this candidate string already spoken for, by ANYONE in this
    /// batch". `incoming` is every entry `sidecar` already holds (covers a
    /// user key that happens to already contain `#`, and every not-yet-
    /// visited run's own bare key); `emitted` is every key this call has
    /// produced so far. Within one run, the smallest available `i` is tried
    /// in increasing order starting from 1, so the common case (nothing
    /// collides) still produces the plain `#1, #2, ...` spec §6 sequence;
    /// the first piece of a run is never suffixed, per spec.
    fn suffix_keys(&self, sidecar: &[String]) -> Vec<String> {
        let incoming: HashSet<&str> = sidecar.iter().map(String::as_str).collect();
        let mut emitted: HashSet<String> = HashSet::with_capacity(self.sources.len());
        let mut out = Vec::with_capacity(self.sources.len());
        let mut run_source: Option<usize> = None;
        let mut next_i = 1usize;
        for &src in &self.sources {
            if let Some(prev) = run_source {
                debug_assert!(
                    src >= prev,
                    "sources must be non-decreasing: `rebuild_nodes` finishes \
                     emitting one source's pieces before starting the next, \
                     so a run's source index never appears again after a \
                     later one — the suffix counter above depends on this to \
                     tell pieces of one split apart from unrelated runs; got \
                     source {src} after {prev}"
                );
            }
            let key = if run_source == Some(src) {
                loop {
                    let candidate = format!("{}#{next_i}", sidecar[src]);
                    next_i += 1;
                    if !incoming.contains(candidate.as_str()) && !emitted.contains(&candidate) {
                        break candidate;
                    }
                }
            } else {
                run_source = Some(src);
                next_i = 1;
                sidecar[src].clone()
            };
            emitted.insert(key.clone());
            out.push(key);
        }
        out
    }
}

/// Realigns a clipped batch's per-node sidecars one at a time and remembers
/// whether any of them could not be realigned.
///
/// [`ClippedNodes::realign`]'s pass-through for a wrong-length sidecar is the
/// right release-mode choice — indexing into it would panic — but it leaves
/// that batch's `data_indices`/tooltips/hrefs/keys pointing at the pre-clip
/// node list, i.e. attributed to the wrong marks. The `debug_assert` catches
/// it in a debug build; this type is how a release build still gets to say
/// so, via one `RenderWarning` per affected batch rather than six.
pub struct SidecarRealign<'a> {
    clipped: &'a ClippedNodes,
    misaligned: bool,
}

impl SidecarRealign<'_> {
    /// Realign one sidecar, recording a length mismatch if there is one.
    ///
    /// A misaligned sidecar is passed through from HERE rather than from
    /// [`ClippedNodes::realign`], so this path never trips that method's
    /// `debug_assert`. The degradation is identical (the sidecar is returned
    /// untouched, never indexed into); what changes is that it is now a
    /// reported, testable outcome in a debug build instead of a panic in one
    /// profile and silence in the other.
    pub fn realign<T: Clone>(&mut self, sidecar: Option<Vec<T>>) -> Option<Vec<T>> {
        if self.note_if_misaligned(&sidecar) {
            return sidecar;
        }
        self.clipped.realign(sidecar)
    }

    /// Realign the batch's `keys` sidecar via [`ClippedNodes::realign_keys`],
    /// recording a length mismatch exactly like [`SidecarRealign::realign`].
    /// The only caller is `scene_build::install_rebuilt_nodes`'s `batch.keys`
    /// line — see [`ClippedNodes::realign_keys`] for why `keys` cannot share
    /// the plain clone-per-piece path every other sidecar uses.
    pub fn realign_keys(&mut self, sidecar: Option<Vec<String>>) -> Option<Vec<String>> {
        if self.note_if_misaligned(&sidecar) {
            return sidecar;
        }
        self.clipped.realign_keys(sidecar)
    }

    /// The misalignment check [`SidecarRealign::realign`] and
    /// [`SidecarRealign::realign_keys`] both perform before touching
    /// [`ClippedNodes`] — one implementation of "does this sidecar's length
    /// match the pre-clip node count", so the flag it sets cannot drift
    /// between the verbatim-copy arm and the key-suffixing arm.
    fn note_if_misaligned<T>(&mut self, sidecar: &Option<Vec<T>>) -> bool {
        let misaligned = sidecar
            .as_ref()
            .is_some_and(|s| !self.clipped.accepts_sidecar_len(s.len()));
        if misaligned {
            self.misaligned = true;
        }
        misaligned
    }

    /// True when any sidecar realigned through this accumulator had the wrong
    /// length and was passed through unchanged.
    pub fn misaligned(&self) -> bool {
        self.misaligned
    }
}

/// Rebuild a node list through a 1→N rewrite, recording each emitted node's
/// source index.
///
/// The bookkeeping half of every rewrite a break performs on a mark batch —
/// the extent clip ([`BreakRemap::clip_nodes`]) and the vertex split
/// ([`BreakRemap::split_nodes`], [`BreakRemap::split_segment_nodes`]) — so
/// `sources`/`source_len` are maintained in exactly one place and a new rewrite
/// cannot ship a [`ClippedNodes`] whose provenance is subtly wrong. `place`
/// pushes zero or more nodes for one source node; it takes `&mut` state (the
/// segment walk's shaft pairing) rather than `&self` alone.
fn rebuild_nodes(
    nodes: Vec<SceneNode>,
    mut place: impl FnMut(SceneNode, &mut Vec<SceneNode>),
) -> ClippedNodes {
    let source_len = nodes.len();
    let mut out = Vec::with_capacity(source_len);
    let mut sources = Vec::with_capacity(source_len);
    for (idx, node) in nodes.into_iter().enumerate() {
        let before = out.len();
        place(node, &mut out);
        // Checked BEFORE the resize it guards: `resize` to a shorter length
        // would truncate `sources` and silently corrupt the provenance of every
        // earlier node, so an assert placed after it fires only once the damage
        // is done. `place` is append-only by contract; this is the enforcement.
        debug_assert!(out.len() >= before, "`place` must not shrink the output");
        sources.resize(out.len(), idx);
    }
    ClippedNodes { nodes: out, sources, source_len }
}

/// What a segment batch's most recent shaft left behind for the arrowhead that
/// follows it — see [`BreakRemap::split_segment_nodes`].
enum ShaftTip {
    /// The placed piece that REACHES the shaft's tip, as `(x1, y1, x2, y2)`.
    /// Its `(x2, y2)` is the placed tip itself.
    Placed(f64, f64, f64, f64),
    /// The shaft's tip fell inside a gap, so no piece reaches it.
    Hidden,
}

/// One [`BreakRemap::clip_extent_bands`] result: a pixel-space `(lo, hi)`
/// band plus which of its two edges (`lo`, the smaller pixel value, and
/// `hi`, the larger) a gap cut rather than the extent's own endpoint.
#[derive(Debug, Clone, Copy)]
struct ClippedBand {
    lo: f64,
    hi: f64,
    cut_lo: bool,
    cut_hi: bool,
}

/// What [`BreakRemap::split_polyline`] produced: the surviving pieces, and
/// whether producing them required CUTTING the source.
///
/// `cut` is reported by the producer, set at the moment it departs from the
/// source vertices — it drops one (a vertex inside a gap, an edge wholly
/// inside one, a vertex with no data position at all) or substitutes an
/// interpolated gap-edge crossing for one. It is deliberately NOT derivable
/// from `pieces`: a cut can drop a boundary vertex and insert a crossing in
/// the same edge, so one piece with the source's own vertex count is a shape
/// BOTH an uncut walk and a genuine cut can produce (spec-review cycle-2
/// finding: a `pieces.len() == 1 && piece.len() == points.len()` test read
/// that shape as uncut and handed the in-gap vertex back to `remap_node`,
/// which parks it at `BREAK_HIDDEN` — the #230-1 spike this whole mechanism
/// exists to remove). Geometric comparison of the output against the input
/// vertices is the same class of inference and fails the same way, since a
/// crossing can land exactly on a dropped vertex's position.
struct PolylineSplit {
    pieces: Vec<Vec<(f64, f64)>>,
    cut: bool,
}

/// Which of a clipped rect piece's four edges a break's gap produced — the
/// crate's typed-`Copy`-struct idiom for "which edges" ([`GridRectEnds`],
/// `marks/axis.rs`), in place of four positional `bool`s a call site could
/// silently transpose.
///
/// Exactly two of the four are ever set for one split: a Y split can only
/// cut `top`/`bottom` (the `left`/`right` edges are the rect's own `x` /
/// `x + w`, untouched by a Y clip); an X split can only cut `left`/`right`.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct CutEdges {
    top: bool,
    right: bool,
    bottom: bool,
    left: bool,
}

impl CutEdges {
    /// Whether ANY edge was cut — the common early-out for "this piece is
    /// exactly the source rect, unclipped".
    fn any(self) -> bool {
        self.top || self.right || self.bottom || self.left
    }

    /// Whether each of the four corners — top-left, top-right, bottom-right,
    /// bottom-left, in that order — touches a cut edge. A corner squares
    /// whenever EITHER of its two adjacent edges is cut.
    fn corners(self) -> [bool; 4] {
        [
            self.top || self.left,
            self.top || self.right,
            self.bottom || self.right,
            self.bottom || self.left,
        ]
    }
}

impl<'a> BreakRemap<'a> {
    /// Resolve `result` against the unbroken scale's `domain` and `pixels`.
    pub fn new(
        axis: AlignAxis,
        domain: (f64, f64),
        pixels: (f64, f64),
        result: &'a BreakResult,
    ) -> Self {
        Self { axis, domain, pixels, result }
    }

    /// The axis this break moves.
    ///
    /// Two `scene_build` call sites compare this against a CHANNEL token
    /// (`marks::axis::AxisNodes::placed_along`) and read
    /// [`AlignAxis::wire_name`] to do it, crossing the two vocabularies
    /// explicitly rather than unifying them — see that enum's module doc.
    pub fn axis(&self) -> AlignAxis {
        self.axis
    }

    /// The break's underlying [`BreakResult`] — its segments and indicator
    /// positions, independent of this remap's axis/domain/pixel context.
    ///
    /// Exposed so a caller holding `&[BreakRemap]` (already filtered to axes
    /// with a continuous domain — the canonical list marks fold through) can
    /// run a DATA-SPACE query — e.g. the annotation-suppression pass's
    /// `gap_containing` check — against the same list, rather than keeping a
    /// second, independently-filtered `(axis, BreakResult)` copy that could
    /// drift from it.
    pub fn result(&self) -> &BreakResult {
        self.result
    }

    fn is_y(&self) -> bool {
        self.axis.is_y()
    }

    /// Reverse-map a pixel coordinate to data on the unbroken scale, clamped
    /// to its domain.
    ///
    /// The reverse-map is linear — `data` is `d_lo` plus the pixel's fraction
    /// of `px_hi - px_lo` times `d_hi - d_lo`. That is the true inverse only
    /// where the scale is LINEAR IN DATA (`Linear`, `Time`, and a reversed
    /// linear scale, whose descending `data_domain()` this arithmetic carries
    /// for free). On `Log`/`Pow`/`Symlog` it is NOT: `data_domain()` reports
    /// the untransformed domain while `to_pixel_f64` maps through the data
    /// transform, so this arithmetic names the wrong data value for a pixel
    /// and the break removes the wrong region (a tick outside the gap can
    /// hide while a point inside it draws). That defect is pre-existing and
    /// unfixed here — the honest fix is a scale-level inverse
    /// (`ScaleKind::invert_f64`) or building marks against the piecewise
    /// scale directly, a filed follow-up — but it is no longer silent:
    /// `scene_build::build_panel_breaks` emits
    /// [`RenderWarning::BreakAxisOnNonlinearScale`](crate::render::RenderWarning::BreakAxisOnNonlinearScale)
    /// when a `BreakAxis` targets one of those scale kinds.
    ///
    /// Clamping keeps geometry that overshoots the domain (a bar drawn down
    /// to the plot floor, which sits below the scale's zero) attached to the
    /// nearest segment instead of vanishing.
    ///
    /// `BREAK_HIDDEN` is poison here (#230-9): a pixel already parked at the
    /// sentinel is not "geometry that overshoots the domain" — clamping it
    /// like any other out-of-range pixel silently un-hides whatever it
    /// belonged to. That was the exact resurrection mechanism the same-axis
    /// merge above (`merge_break_specs`) makes unreachable for the
    /// production sequential-remap loop, but poisoning it here too covers any
    /// other caller of `data_at`: `f64::NAN` propagates through
    /// [`broken_scale_map`] as "no segment contains this" (`ScaleSegment::
    /// contains`'s comparisons are false for NaN either side), so `coord`
    /// reports the coordinate as still hidden rather than clamping it to a
    /// domain edge. A caller that instead runs its own `f64::min`/`max` over
    /// two `data_at` results — as [`BreakRemap::clip_extent`] does to order
    /// an extent's endpoints — does NOT get this for free: `min`/`max` drop a
    /// NaN operand rather than propagate it, so that caller needs its own
    /// `is_hidden` guard on the raw input pixels, ahead of calling `data_at`
    /// at all. `clip_extent` carries that guard; a future caller with the
    /// same min/max shape needs its own too.
    fn data_at(&self, px: f64) -> f64 {
        if is_hidden(px) {
            return f64::NAN;
        }
        let (d_lo, d_hi) = self.domain;
        let (px_lo, px_hi) = self.pixels;
        let span = px_hi - px_lo;
        if span.abs() < f64::EPSILON {
            return d_lo;
        }
        let value = d_lo + (px - px_lo) / span * (d_hi - d_lo);
        value.clamp(d_lo.min(d_hi), d_lo.max(d_hi))
    }

    /// Forward-map a data value to its pixel on the *unbroken* scale.
    fn pixel_at(&self, value: f64) -> f64 {
        let (d_lo, d_hi) = self.domain;
        let (px_lo, px_hi) = self.pixels;
        let span = d_hi - d_lo;
        if span.abs() < f64::EPSILON {
            return px_lo;
        }
        px_lo + (value - d_lo) / span * (px_hi - px_lo)
    }

    /// Re-place a single pixel coordinate on the broken axis.
    ///
    /// `None` when the coordinate reverse-maps into a gap.
    pub fn coord(&self, px: f64) -> Option<f64> {
        let (px_lo, px_hi) = self.pixels;
        if (px_hi - px_lo).abs() < f64::EPSILON {
            return Some(px);
        }
        if px.is_nan() {
            return None;
        }
        broken_scale_map(self.data_at(px), self.result)
    }

    /// Clip a pixel extent to the broken scale's visible segments.
    ///
    /// `a`/`b` are the extent's endpoints in the unbroken scale's pixel space,
    /// in either order. Returns one ordered `(lo, hi)` pixel pair per segment
    /// the extent intersects, already mapped onto the broken scale — so a bar
    /// from 0 to 900 under a `(150, 850)` gap yields the two pairs the guide's
    /// clipping model describes, and a bar to 400 under the same gap yields
    /// just its visible 0–150 portion. An extent lying wholly inside a gap
    /// yields nothing, and the caller drops the node rather than emitting the
    /// zero-height artifact the old endpoint-stretching remap left behind. A
    /// zero-length extent (a bar sitting on its own baseline) keeps its single
    /// zero-length pair when visible, so a zero-valued bar still lands in the
    /// right place.
    pub fn clip_extent(&self, a: f64, b: f64) -> Vec<(f64, f64)> {
        self.clip_extent_bands(a, b)
            .into_iter()
            .map(|band| (band.lo, band.hi))
            .collect()
    }

    /// Same intersection as [`BreakRemap::clip_extent`], plus which pixel-space
    /// edge of each returned band a gap produced rather than `a`/`b`
    /// themselves — [`BreakRemap::clip_node_into`]'s `Rect` arm needs this to
    /// know which corners of a split piece stay rounded (#230-2: the prior
    /// code copied `corner_radius` onto every piece, rounding the CUT edge a
    /// gap introduced along with the two edges that were always there).
    ///
    /// "Cut" is exact, not a float-tolerance guess: `data_spans`' own
    /// `lo.max(s_lo)` / `hi.min(s_hi)` clamp returns one of its two operands
    /// UNCHANGED (`f64::max`/`min` never interpolate), so a band's data-space
    /// bound is cut exactly when the clamp picked the segment's edge over the
    /// requested extent's own endpoint — checkable with a plain `>`/`<`
    /// against the same `lo`/`hi` this method already computes. Mapping that
    /// into pixel space needs one more step: `data_spans` walks segments in
    /// ascending DATA order, but a segment's `pixel_for` can run either
    /// direction (a Y axis maps larger data to SMALLER pixels), so which of a
    /// band's two data-cut flags lands on the smaller-pixel edge depends on
    /// that segment's own direction — read off `p0` vs `p1` per band rather
    /// than assumed from array position.
    fn clip_extent_bands(&self, a: f64, b: f64) -> Vec<ClippedBand> {
        // `BREAK_HIDDEN` is poison here too (#230-9 belt-and-braces): tested
        // on the INPUT pixels, before `data_at` runs, because `data_at`'s own
        // poison guard returns `f64::NAN` for a hidden endpoint, and Rust's
        // `f64::min`/`max` (used just below to order `d0`/`d1` into `lo`/
        // `hi`) silently DROP a NaN operand rather than propagate it — so a
        // NaN-only guard here would let one hidden endpoint through as a
        // zero-length pair at the surviving endpoint, and both endpoints
        // hidden through as the segment's FULL extent (verified: `f64::NAN
        // .max(s_lo) == s_lo`, `f64::NAN.min(s_hi) == s_hi`). Early-returning
        // on the raw pixels closes that hole without widening `is_hidden`
        // itself, which stays exact-equality for its other caller
        // (`data_at`).
        if a.is_nan() || b.is_nan() || is_hidden(a) || is_hidden(b) {
            return Vec::new();
        }
        let (d0, d1) = (self.data_at(a), self.data_at(b));
        // Ascending, so `data_spans` needs no travel-order reversal and the
        // pairs come out in segment order, as every caller reads them.
        let (lo, hi) = (d0.min(d1), d0.max(d1));
        self.data_spans(lo, hi)
            .into_iter()
            .map(|(k, c_lo, c_hi)| {
                let seg = &self.result.segments[k];
                let (p0, p1) = (seg.pixel_for(c_lo), seg.pixel_for(c_hi));
                let cut_at_c_lo = c_lo > lo;
                let cut_at_c_hi = c_hi < hi;
                if p0 <= p1 {
                    ClippedBand { lo: p0, hi: p1, cut_lo: cut_at_c_lo, cut_hi: cut_at_c_hi }
                } else {
                    ClippedBand { lo: p1, hi: p0, cut_lo: cut_at_c_hi, cut_hi: cut_at_c_lo }
                }
            })
            .collect()
    }

    /// Whether any part of the closed DATA interval bounded by `a` and `b`
    /// (either order) lands on a visible segment — "would clipping this extent
    /// leave anything to draw".
    ///
    /// The data-space door to the same intersection rule
    /// [`BreakRemap::clip_extent`] reaches through pixels, for the one caller
    /// whose extent IS data: an annotation's bounds are spec values, so asking
    /// in data avoids a round trip through a pixel the spec never had (and
    /// through [`BreakRemap::data_at`]'s domain clamp, which would make an
    /// out-of-domain bound look like a domain-edge one). Used by the
    /// annotation break router to tell "clip it" from "nothing survives, drop
    /// it" (#228).
    ///
    /// Zero-length and merely-touching intervals follow
    /// [`BreakRemap::data_spans`]'s rules: a point ON a visible segment is
    /// visible, a point in a gap is not, and an interval arriving from inside a
    /// gap exactly onto a segment's edge contributes no length and so is not.
    pub fn data_extent_visible(&self, a: f64, b: f64) -> bool {
        !self.data_spans(a.min(b), a.max(b)).is_empty()
    }

    /// The visible slices of the closed data interval from `d0` to `d1`, in
    /// TRAVEL order: `(segment index, entry value, exit value)` per segment the
    /// interval meets, each pair oriented the way the interval runs.
    ///
    /// The one intersection rule behind both shapes a break has to cut along
    /// its axis: an extent's two endpoints ([`BreakRemap::clip_extent`], which
    /// hands in an ascending pair and reads the result as pixel bands) and one
    /// edge of a vertex-placed stroke ([`BreakRemap::split_polyline`], which
    /// needs the direction because it walks a polyline). Keeping them on one
    /// body is what stops "which part of this range is visible" from having two
    /// answers.
    ///
    /// A zero-length interval yields at most one span — the segment containing
    /// it — so a zero-height bar still lands in the right place and a rule
    /// perpendicular to the break still draws. An interval that merely TOUCHES
    /// a segment at one point (arriving from inside a gap exactly onto the
    /// segment's edge) yields nothing for that segment: there is no length
    /// there to draw.
    fn data_spans(&self, d0: f64, d1: f64) -> Vec<(usize, f64, f64)> {
        let (lo, hi) = (d0.min(d1), d0.max(d1));
        let mut spans: Vec<(usize, f64, f64)> = Vec::with_capacity(self.result.segments.len());
        for (k, seg) in self.result.segments.iter().enumerate() {
            if hi <= lo {
                if seg.contains(lo) {
                    spans.push((k, lo, lo));
                    break;
                }
                continue;
            }
            let (s_lo, s_hi) = seg.ordered_data();
            let (c_lo, c_hi) = (lo.max(s_lo), hi.min(s_hi));
            if c_hi > c_lo {
                spans.push((k, c_lo, c_hi));
            }
        }
        if d1 < d0 {
            spans.reverse();
            for span in spans.iter_mut() {
                std::mem::swap(&mut span.1, &mut span.2);
            }
        }
        spans
    }

    /// The index of the segment containing `value`, or `None` when it falls in
    /// a gap or outside the domain.
    fn segment_index(&self, value: f64) -> Option<usize> {
        self.result.segments.iter().position(|seg| seg.contains(value))
    }

    /// Rebuild `nodes` onto the broken axis, clipping extent geometry to the
    /// visible segments.
    ///
    /// For [`BreakPlacement::ClipExtent`] batches only. `Rect`s and closed
    /// polygonal `Path`s (the area fill) are clipped per segment, which can
    /// split one node into several or drop it entirely; an OPEN `Path` (the area
    /// border stroke) is CUT per segment by [`BreakRemap::split_node_into`], so
    /// an outline agrees with the fill it sits on; a `Group` is rebuilt
    /// child-wise but stays one node. Any other node kind found inside such a
    /// batch falls back to [`BreakRemap::remap_node`] and stays 1→1.
    ///
    /// Glyph batches must NOT come through here — extent-clipping a glyph
    /// resizes it and can delete it; they go to
    /// [`BreakRemap::remap_glyph_node`] instead.
    pub fn clip_nodes(&self, nodes: Vec<SceneNode>) -> ClippedNodes {
        rebuild_nodes(nodes, |node, out| self.clip_node_into(node, out))
    }

    fn clip_node_into(&self, node: SceneNode, out: &mut Vec<SceneNode>) {
        match node {
            SceneNode::Rect { x, y, w, h, style, corner_radius } => {
                let is_y = self.is_y();
                let (start, len) = if is_y { (y, h) } else { (x, w) };
                for band in self.clip_extent_bands(start, start + len) {
                    let (nx, nw, ny, nh) = if is_y {
                        (x, w, band.lo, band.hi - band.lo)
                    } else {
                        (band.lo, band.hi - band.lo, y, h)
                    };
                    // A Y split cuts the top/bottom edges (the left/right
                    // edges are `x`/`x+w`, untouched by a Y clip); an X split
                    // cuts left/right. `band.cut_lo` names the smaller-pixel
                    // edge (top for Y, left for X), `band.cut_hi` the larger.
                    let cut = if is_y {
                        CutEdges { top: band.cut_lo, right: false, bottom: band.cut_hi, left: false }
                    } else {
                        CutEdges { top: false, right: band.cut_hi, bottom: false, left: band.cut_lo }
                    };
                    out.push(rounded_rect_piece(nx, ny, nw, nh, style.clone(), corner_radius, cut));
                }
            }
            // A CLOSED polygonal path IS the extent geometry — the area fill —
            // and is clipped per segment. An OPEN path in an extent batch is not
            // an extent at all: it is a stroked BORDER drawn over that fill
            // (`mark_area(line=True)` / `borders=True`), which is connected
            // vertex-placed geometry and therefore splits through the one body
            // that knows how to cut one. Handing it to `remap_node` instead left
            // a broken chart rendering a correctly clipped fill underneath an
            // unclipped outline — in-gap border vertices parked at
            // `BREAK_HIDDEN`, or the stroke running straight across the break
            // band — the two halves of one mark disagreeing (#230-1).
            stroke @ SceneNode::Path { closed: false, .. } => self.split_node_into(stroke, out),
            SceneNode::Path { commands, style, closed: true, widths } => {
                match polygon_points(&commands) {
                    Some(points) => self.clip_polygon_into(&points, &style, out),
                    // A closed ring the polygon reader will not take is CURVED,
                    // and neither the clip nor the split can cut a placed curve
                    // faithfully, so it keeps the per-vertex fallback.
                    None => {
                        let mut node = SceneNode::Path { commands, style, closed: true, widths };
                        self.remap_node(&mut node);
                        out.push(node);
                    }
                }
            }
            SceneNode::Group { attrs, children } => {
                let clipped = self.clip_nodes(children);
                out.push(SceneNode::Group { attrs, children: clipped.nodes });
            }
            // Everything else keeps its node identity and is placed by
            // `remap_node`. That includes `Polygon`, deliberately: a polygon
            // clips WITHIN itself — its rings grow or vanish, the node does
            // not — so a violin cut in two by the break is still one node with
            // one row.
            mut other => {
                self.remap_node(&mut other);
                out.push(other);
            }
        }
    }

    /// Emit one clipped polygon per visible segment.
    ///
    /// The clip runs in the unbroken scale's pixel space (where the polygon's
    /// edges are straight), then each surviving vertex is mapped through its
    /// own segment's affine. Mapping per segment rather than per vertex is
    /// what removes the `BREAK_HIDDEN` spikes: no vertex can reverse-map into
    /// a gap, because the clip already cut the polygon at the gap's edges.
    fn clip_polygon_into(&self, points: &[(f64, f64)], style: &FillStroke, out: &mut Vec<SceneNode>) {
        for seg in &self.result.segments {
            let Some(clipped) = self.clip_ring_to_segment(points, seg) else {
                continue;
            };
            out.push(SceneNode::Path {
                commands: path_commands(&clipped, true),
                style: style.clone(),
                closed: true,
                widths: None,
            });
        }
    }

    /// Clip one closed ring of `(x, y)` vertices to `seg`'s band and map the
    /// survivors onto the broken scale. `None` when nothing survives.
    ///
    /// The single clip body behind both filled shapes a break has to cut: the
    /// area fill's closed `Path` and a `Polygon` node's rings.
    fn clip_ring_to_segment(
        &self,
        points: &[(f64, f64)],
        seg: &ScaleSegment,
    ) -> Option<Vec<(f64, f64)>> {
        let (s_lo, s_hi) = seg.ordered_data();
        let (b0, b1) = (self.pixel_at(s_lo), self.pixel_at(s_hi));
        let along: Vec<(f64, f64)> = points.iter().map(|&p| self.split(p)).collect();
        let clipped = clip_to_band(&along, b0.min(b1), b0.max(b1));
        if clipped.len() < 3 {
            return None;
        }
        Some(
            clipped
                .iter()
                .map(|&(a, across)| self.join(seg.pixel_for(self.data_at(a)), across))
                .collect(),
        )
    }

    /// Clip a `Polygon` node's rings to the visible segments, in place.
    ///
    /// Ring count can grow (a ring spanning a gap becomes one ring per segment
    /// it crosses) or shrink to nothing, but the NODE count does not — a
    /// polygon mark stays one node with one row, so no sidecar can drift.
    /// Ring order is preserved within each source ring, which keeps the
    /// exterior-then-holes `evenodd` reading intact: a hole's clipped pieces
    /// still lie inside their exterior's pieces.
    fn clip_rings(&self, rings: &mut Vec<Vec<[f64; 2]>>) {
        let mut out: Vec<Vec<[f64; 2]>> = Vec::with_capacity(rings.len());
        for ring in rings.iter() {
            let points: Vec<(f64, f64)> = ring.iter().map(|p| (p[0], p[1])).collect();
            for seg in &self.result.segments {
                if let Some(clipped) = self.clip_ring_to_segment(&points, seg) {
                    out.push(clipped.into_iter().map(|(x, y)| [x, y]).collect());
                }
            }
        }
        *rings = out;
    }

    // ── Vertex splitting (#230-1) ────────────────────────────────────────────

    /// Rebuild `nodes` onto the broken axis by SPLITTING vertex-placed stroked
    /// geometry into one sub-node per visible piece.
    ///
    /// The [`BreakPlacement::MoveVertices`] counterpart of
    /// [`BreakRemap::clip_nodes`], and the same 1→N contract: a `Line`,
    /// `Polyline`, or straight-line open `Path` crossing a gap becomes one node
    /// per piece of it that survives, a node lying wholly inside a gap becomes
    /// none, and [`ClippedNodes`] carries the provenance the batch's per-node
    /// sidecars realign through. No coordinate of a split piece is ever
    /// `BREAK_HIDDEN`: the geometry is CUT at the gap edges instead of being
    /// stretched across them through the sentinel (#230-1). The curve fallback
    /// below is the one shape here that still emits one, deliberately — and
    /// the reason cut-ness is reported by [`BreakRemap::split_polyline`] rather
    /// than inferred from its pieces: every route into the fallback has to be
    /// one a caller CHOSE, since the fallback's own placement is the sentinel.
    ///
    /// A node kind with no faithful split falls back to
    /// [`BreakRemap::remap_node`] and stays 1→1. That boundary is a property of
    /// the GEOMETRY, not of the node kind: straight-line commands split, curve
    /// commands (`QuadTo`/`CubicTo`/`ArcTo`) do not, because a curve's crossing
    /// point is not a vertex of anything this module can re-spell — cutting one
    /// would need de Casteljau subdivision of the placed curve, and the
    /// sentinel's honest degradation is preferred to a silently wrong shape.
    /// Closed paths never come here either (a filled ring clips —
    /// [`BreakRemap::clip_polygon_into`]), except in a segment batch, where a
    /// closed path is an arrowhead ([`BreakRemap::split_segment_nodes`]).
    pub fn split_nodes(&self, nodes: Vec<SceneNode>) -> ClippedNodes {
        rebuild_nodes(nodes, |node, out| self.split_node_into(node, out))
    }

    /// Rebuild a whole segment batch's nodes — the
    /// [`BreakPlacement::MoveVerticesAndGlyph`] model.
    ///
    /// `mark_segment(arrow=True)` emits two node species per row: the shaft
    /// `Line`, whose two ends are two data positions, and a CLOSED `Path` — the
    /// arrowhead triangle, whose size and direction are drawn geometry derived
    /// from the shaft. The shaft splits like any other stroke; the head is then
    /// REBUILT from the endpoints of the piece that reaches the shaft's tip, so
    /// it keeps its drawn size and points along the piece it terminates.
    /// Rebuilding rather than moving is what makes the head keep pointing along
    /// its shaft: translating the pre-break triangle preserved its size but
    /// froze its pre-break direction, leaving a visible head-vs-shaft angle
    /// mismatch wherever a break changed the shaft's slope.
    ///
    /// This takes the batch rather than one node because a head is only
    /// meaningful with its shaft, and the pairing is positional:
    /// `marks::segment::build` pushes `[shaft Line, head Path]` per row, in that
    /// order, so the most recently split `Line` is the head's own shaft. Three
    /// fates follow from what that shaft left behind ([`ShaftTip`]):
    ///
    /// - a piece reaching the tip → the head is rebuilt on it;
    /// - the tip inside a gap → the head is DROPPED, because no piece reaches
    ///   it and there is nothing left for it to terminate (the shaft's own
    ///   pieces stop at the gap edge);
    /// - no shaft at all (a malformed batch) → rigid translation onto its own
    ///   placed tip, the previous behavior and the honest degradation.
    pub fn split_segment_nodes(&self, nodes: Vec<SceneNode>) -> ClippedNodes {
        let mut shaft: Option<ShaftTip> = None;
        rebuild_nodes(nodes, |node, out| match node {
            SceneNode::Line { x1, y1, x2, y2, style } => {
                let pieces = self.split_polyline(&[(x1, y1), (x2, y2)]).pieces;
                // The tip is the shaft's own `(x2, y2)`, so the piece that
                // reaches it can only be the LAST one — `split_polyline` walks
                // the vertices in order. Whether one does is asked of the
                // geometry rather than assumed: the last piece's placed end
                // must BE the placed tip, which is an exact comparison (both
                // are `ScaleSegment::pixel_for` of the same data value). It is
                // not, in one knife-edge case: a tip lying exactly on a
                // segment's edge, approached from inside the adjacent gap,
                // places as visible while contributing no length to that
                // segment — so the last piece belongs to a different part of
                // the shaft, and its direction is not the tip's.
                let along_tip = if self.is_y() { y2 } else { x2 };
                let placed_tip = self.coord(along_tip);
                shaft = Some(match (placed_tip, pieces.last()) {
                    (Some(placed), Some(run))
                        if self.split(run[run.len() - 1]).0 == placed =>
                    {
                        let (a, b) = (run[0], run[run.len() - 1]);
                        ShaftTip::Placed(a.0, a.1, b.0, b.1)
                    }
                    _ => ShaftTip::Hidden,
                });
                push_line_pieces(&pieces, &style, out);
            }
            SceneNode::Path { commands, style, closed: true, widths } => {
                let mut head = SceneNode::Path { commands, style, closed: true, widths };
                match shaft {
                    Some(ShaftTip::Placed(x1, y1, x2, y2)) => {
                        self.remap_arrowhead(&mut head, Some((x1, y1, x2, y2)));
                        out.push(head);
                    }
                    Some(ShaftTip::Hidden) => {}
                    None => {
                        self.remap_arrowhead(&mut head, None);
                        out.push(head);
                    }
                }
            }
            other => self.split_node_into(other, out),
        })
    }

    fn split_node_into(&self, node: SceneNode, out: &mut Vec<SceneNode>) {
        match node {
            SceneNode::Line { x1, y1, x2, y2, style } => {
                let pieces = self.split_polyline(&[(x1, y1), (x2, y2)]).pieces;
                push_line_pieces(&pieces, &style, out);
            }
            SceneNode::Polyline { points, style, widths } => {
                let split = self.split_polyline(&points);
                // An uncut split is the source vertices placed one for one, so
                // their widths still align; a cut piece carries none.
                let widths = if split.cut { None } else { widths };
                for piece in split.pieces {
                    out.push(SceneNode::Polyline { points: piece, style: style.clone(), widths: widths.clone() });
                }
            }
            // An OPEN path of straight-line commands is a polyline in path
            // spelling — `mark_line`'s three `step*` interpolations emit exactly
            // that. When a gap CUTS it, each piece is re-spelled as explicit
            // `MoveTo`/`LineTo` (the shorthand `H`/`V` a step path is written
            // with describes a riser between two vertices, and a piece's cut end
            // is not one), so a cut step line's bytes differ from its uncut ones
            // — only under a break that actually reaches it, where the old bytes
            // carried a sentinel spike. A path the split reports as UNCUT keeps
            // its original command stream verbatim, `H`/`V` shorthand included,
            // through the ordinary per-vertex remap: that remap's `coord()`
            // computes the exact same placed pixel `split_polyline` would have
            // for an uncut vertex (both are `data_at` followed by the same
            // segment's `pixel_for`), so nothing but the spelling differs
            // between the two paths.
            //
            // Cut-ness is asked of the producer ([`PolylineSplit::cut`]) and
            // never inferred from the pieces here: `remap_node` is the sentinel
            // path, so a false "uncut" reading hides an in-gap vertex at
            // `BREAK_HIDDEN` — exactly the #230-1 spike this mechanism removes.
            SceneNode::Path { commands, style, closed: false, widths } => {
                match straight_path_points(&commands) {
                    Some(points) => {
                        let split = self.split_polyline(&points);
                        if split.cut {
                            for piece in split.pieces {
                                out.push(SceneNode::Path {
                                    commands: path_commands(&piece, false),
                                    style: style.clone(),
                                    closed: false,
                                    widths: None,
                                });
                            }
                        } else {
                            let mut node = SceneNode::Path { commands, style, closed: false, widths };
                            self.remap_node(&mut node);
                            out.push(node);
                        }
                    }
                    None => {
                        let mut node = SceneNode::Path { commands, style, closed: false, widths };
                        self.remap_node(&mut node);
                        out.push(node);
                    }
                }
            }
            SceneNode::Group { attrs, children } => {
                let split = self.split_nodes(children);
                out.push(SceneNode::Group { attrs, children: split.nodes });
            }
            mut other => {
                self.remap_node(&mut other);
                out.push(other);
            }
        }
    }

    /// Cut a vertex-placed polyline at the gap edges it crosses and place each
    /// surviving piece, in walk order.
    ///
    /// # Geometry contract
    ///
    /// Every vertex is its own data position, recovered by reverse-mapping its
    /// along-axis pixel ([`BreakRemap::data_at`], so out-of-domain vertices
    /// clamp to the domain exactly as the sentinel model placed them). An edge
    /// between two consecutive vertices is a straight line in data space, so
    /// the parts of it that are visible are the parts of its data interval that
    /// fall in the break's segments ([`BreakRemap::data_spans`]), in travel
    /// order — an edge may cross several gaps and therefore contribute to
    /// several pieces.
    ///
    /// At each crossing the piece ends at the GAP EDGE, at the point where the
    /// edge actually meets it: the crossing's cross-axis coordinate is
    /// interpolated linearly along the edge (`data_at` is affine in the pixel,
    /// so interpolating in data is interpolating along the drawn edge). That is
    /// the honest render of "the line goes off here" — dropping back to the last
    /// visible vertex instead would bend the line towards it, inventing a slope
    /// the data never had.
    ///
    /// A vertex that lies inside a gap therefore contributes no point of its
    /// own: the piece before it ends at the gap's near edge and the piece after
    /// it starts at the far edge. A vertex whose pixel is already `BREAK_HIDDEN`
    /// (poisoned, #230-9) has no data position at all — `data_at` reports NaN —
    /// so the pieces on either side of it simply end there rather than being
    /// clamped back into view.
    ///
    /// Pieces are returned in walk order (a piece containing the first vertex
    /// comes first), which is what lets [`BreakRemap::split_segment_nodes`]
    /// find an arrowhead's shaft as the LAST piece.
    ///
    /// The return also REPORTS whether cutting happened at all
    /// ([`PolylineSplit::cut`]), because only this method can know.
    fn split_polyline(&self, points: &[(f64, f64)]) -> PolylineSplit {
        // `(data, across)` per vertex: the space the cut is computed in.
        let vertices: Vec<(f64, f64)> = points
            .iter()
            .map(|&p| {
                let (along, across) = self.split(p);
                (self.data_at(along), across)
            })
            .collect();
        // Each piece is one segment index plus its vertices in `(data, across)`.
        let mut pieces: Vec<(usize, Vec<(f64, f64)>)> = Vec::new();
        // Whether the last piece ends exactly at the previous vertex, so the
        // next edge can continue it instead of starting a new one.
        let mut open = false;
        // Set at every point below where the output departs from the source
        // vertices — see `PolylineSplit::cut` for why it is recorded here
        // rather than inferred from the pieces afterwards.
        let mut cut = false;

        if let [(d, across)] = vertices[..] {
            // A one-vertex polyline draws nothing, but it is not this method's
            // place to decide that: keep it when it is visible, exactly as the
            // per-vertex model did.
            match self.segment_index(d) {
                Some(k) => pieces.push((k, vec![(d, across)])),
                // Dropped: it has no visible position.
                None => cut = true,
            }
        }

        for edge in vertices.windows(2) {
            let ((d0, a0), (d1, a1)) = (edge[0], edge[1]);
            if !d0.is_finite() || !d1.is_finite() {
                // A vertex with no data position at all (a pixel already parked
                // at the sentinel, #230-9): the pieces on either side end here,
                // so this edge is dropped.
                open = false;
                cut = true;
                continue;
            }
            let across_at = |d: f64| {
                if d1 == d0 {
                    a0
                } else {
                    a0 + (d - d0) / (d1 - d0) * (a1 - a0)
                }
            };
            let (start_seg, end_seg) = (self.segment_index(d0), self.segment_index(d1));
            let spans = self.data_spans(d0, d1);
            if spans.is_empty() {
                // The whole edge lies inside one gap: whatever was running up
                // to the previous vertex ends there, and this edge is dropped.
                open = false;
                cut = true;
            }
            for (i, &(k, d_in, d_out)) in spans.iter().enumerate() {
                // A span's end is a VERTEX of the source polyline when it is the
                // edge's own endpoint, and a crossing otherwise. Asked as a
                // segment-membership question rather than by comparing floats,
                // so the vertex keeps its exact coordinates.
                let from_vertex = i == 0 && start_seg == Some(k);
                let to_vertex = i + 1 == spans.len() && end_seg == Some(k);
                // Substituting an interpolated gap-edge crossing for a source
                // endpoint IS the cut, whichever end it happens at.
                cut |= !from_vertex || !to_vertex;
                let entry = if from_vertex { (d0, a0) } else { (d_in, across_at(d_in)) };
                let exit = if to_vertex { (d1, a1) } else { (d_out, across_at(d_out)) };
                match pieces.last_mut() {
                    Some((piece_seg, pts)) if open && *piece_seg == k && from_vertex => {
                        pts.push(exit);
                    }
                    _ => pieces.push((k, vec![entry, exit])),
                }
                open = to_vertex;
            }
        }

        PolylineSplit {
            pieces: pieces
                .into_iter()
                .map(|(k, pts)| {
                    let seg = &self.result.segments[k];
                    pts.into_iter()
                        .map(|(d, across)| self.join(seg.pixel_for(d), across))
                        .collect()
                })
                .collect(),
            cut,
        }
    }

    /// Re-place one fixed-size glyph — the [`BreakPlacement::MoveGlyph`] model.
    ///
    /// `anchor` is the EXACT data anchor the batch build recorded for this
    /// node (`MarkBatch::glyph_anchors`, #230-3) — the scale-mapped position
    /// this row placed, before any shape-specific drawn geometry (a point's
    /// radius, a text label's `dx`/`dy`) was added on top of it. The glyph
    /// translates by the distance that anchor moves under the break, so its
    /// drawn size and shape survive the compression: a diamond stays a
    /// diamond, a square stays square, a `shape=` glyph behaves exactly like
    /// the default circle, and any drawn offset from the anchor (an
    /// asymmetric `TriangleUp`/`TriangleDown`, a text `dx`/`dy`) translates
    /// rigidly with it. An anchor inside a gap hides the whole node at
    /// `BREAK_HIDDEN` rather than any part of it, so a glyph straddling a gap
    /// edge cannot emit a spike, and no glyph is ever dropped — glyph
    /// batches stay 1:1 with their rows, and their `data_indices`/tooltip/key
    /// entries stay valid.
    ///
    /// This replaces the pre-#230-3 model, which measured the anchor off the
    /// node's own drawn bounding box (`(lo + hi) / 2.0` over
    /// [`BreakRemap::along_extent`]): correct only for a shape whose geometry
    /// is symmetric around its data position (true of `Circle`/`Square`/
    /// `Diamond`/`Cross`/`VLine`/`HLine`, and of a single-coordinate node like
    /// `Text` before `dx`/`dy` is baked in), but not of `TriangleUp`/
    /// `TriangleDown` (whose bbox midpoint sits a quarter of the triangle's
    /// height off the data anchor — measured: a `TriangleUp` hidden where a
    /// circle at the same data position renders) or of a `mark_text` node
    /// whose `x`/`y` already has `dx`/`dy` added (the offset position was
    /// measured and reverse-mapped/hide-tested instead of the true data
    /// position).
    ///
    /// A `Line` node reaching here is a glyph too — a tick's dash, including a
    /// rug's, whose 8px length is drawn size and not data — so it moves
    /// rigidly and keeps that length. The one batch whose `Line`s are NOT
    /// glyphs is `mark_label`'s, and that batch is routed to
    /// [`BreakRemap::remap_label_node`] by its kind, never by a node-shape
    /// test here.
    pub fn remap_glyph_node(&self, node: &mut SceneNode, anchor: (f64, f64)) {
        let (along, _across) = self.split(anchor);
        self.translate_on_anchor(node, along);
    }

    /// The pre-#230-3 anchor model: the midpoint of a node's own drawn extent
    /// along this break's axis — kept as the release-mode fallback for a
    /// `MoveGlyph`/`MoveGlyphAndConnector` node whose batch did not carry an
    /// exact data anchor (a producer bug in debug — see the `debug_assert!`
    /// at `remap_mark_batch_through_break`'s call site; every real mark
    /// builder populates `MarkBatch::glyph_anchors`, so this path is
    /// unreached in production). Returns a full `(x, y)` point (via
    /// [`BreakRemap::join`]) so the caller can hand it to
    /// [`BreakRemap::remap_glyph_node`]/[`BreakRemap::remap_label_node`]
    /// exactly like a real anchor; the coordinate this break does not move is
    /// never read back out of it.
    pub(crate) fn bbox_anchor_fallback(&self, node: &mut SceneNode) -> (f64, f64) {
        let along = self.along_extent(node).map(|(lo, hi)| (lo + hi) / 2.0).unwrap_or(0.0);
        self.join(along, 0.0)
    }

    /// Re-place one node of a label batch — the
    /// [`BreakPlacement::MoveGlyphAndConnector`] model.
    ///
    /// `mark_label` emits two node species per row with different semantics,
    /// and only the batch knows that: the text is a glyph, and the optional
    /// leader `Line` is a *connector* running FROM the row's data point TO
    /// that text. Its two ends are two different data positions, so each is
    /// placed on its own — moving the connector rigidly by its midpoint's
    /// displacement pulled the data end ~25px off the point it exists to touch.
    ///
    /// The species test lives behind this batch-kind dispatch rather than
    /// inside [`BreakRemap::remap_glyph_node`] for the reason the rug proved:
    /// `Line` is shared by two producers with opposite meanings, so keying the
    /// rule on the node kind collapsed every tick dash lying along the broken
    /// axis to zero length (200 8px rug dashes → 200 zero-length ones on a
    /// `displot(rug=True)` under a break). What a node *means* is a property of
    /// the mark that emitted it, not of its shape.
    ///
    /// `anchor` is the batch's recorded exact data anchor for this node
    /// (#230-3), forwarded to [`BreakRemap::remap_glyph_node`] for the Text
    /// species; the connector `Line` ignores it entirely — its own two ends
    /// already ARE two data positions; `remap_node` reverse-maps them
    /// directly.
    pub fn remap_label_node(&self, node: &mut SceneNode, anchor: (f64, f64)) {
        match node {
            SceneNode::Line { .. } => self.remap_node(node),
            _ => self.remap_glyph_node(node, anchor),
        }
    }

    /// Re-place one arrowhead triangle, rebuilding it from its shaft's placed
    /// endpoints when they describe a direction.
    ///
    /// The tip is the triangle's first vertex by construction
    /// (`marks::arrowhead::arrowhead_path_with_opacity` writes it as the
    /// leading `MoveTo`) and is the shaft's own `(x2, y2)`; `head_size` is the
    /// distance from that tip to the midpoint of the other two vertices,
    /// which is how `marks::arrowhead::arrowhead_triangle` built them.
    ///
    /// `shaft` is the placed sub-shaft the head terminates, when
    /// [`BreakRemap::split_segment_nodes`] found one; `None` is the malformed
    /// batch (a head with no shaft in front of it), which falls back to rigid
    /// translation onto the head's own placed tip. A tip inside a gap hides the
    /// whole head at [`BREAK_HIDDEN`] — reachable only on that malformed path,
    /// since a well-formed head whose tip is in a gap is dropped by its caller
    /// rather than parked at the sentinel.
    fn remap_arrowhead(&self, node: &mut SceneNode, shaft: Option<(f64, f64, f64, f64)>) {
        let Some(vertices) = triangle_vertices(node) else {
            // Not the three-vertex shape this batch's heads have; place it
            // per vertex rather than guessing at its anatomy.
            self.remap_node(node);
            return;
        };
        let [tip, left, right] = vertices;
        let along_tip = if self.is_y() { tip.1 } else { tip.0 };
        let Some(placed_along) = self.coord(along_tip) else {
            // In a gap: hide the head whole, keeping its shape off-screen.
            self.for_each_along(node, &mut |c| *c = BREAK_HIDDEN + (*c - along_tip));
            return;
        };
        let placed_tip = self.join_point(tip, placed_along);
        // `arrowhead_triangle` places the other two vertices `head_size` back
        // from the tip, half a head to either side, so the tip-to-base-midpoint
        // distance recovers the `arrow_size` this head was drawn with.
        let base_mid = ((left.0 + right.0) / 2.0, (left.1 + right.1) / 2.0);
        let head_size = (tip.0 - base_mid.0).hypot(tip.1 - base_mid.1);

        // The shaft's placed direction, when it has one. A `BREAK_HIDDEN` end
        // is a sentinel rather than a position, so it yields no direction.
        let rebuilt = shaft
            .filter(|(x1, y1, x2, y2)| {
                [*x1, *y1, *x2, *y2].iter().all(|c| !is_hidden(*c))
            })
            .and_then(|(x1, y1, x2, y2)| {
                let (dx, dy) = (x2 - x1, y2 - y1);
                let len = dx.hypot(dy);
                (len > 0.0).then(|| {
                    // Start the triangle exactly `head_size` back along the
                    // placed shaft, so `arrowhead_triangle`'s own
                    // clamp-to-length is a no-op and the head keeps its size.
                    let (ux, uy) = (dx / len, dy / len);
                    (placed_tip.0 - ux * head_size, placed_tip.1 - uy * head_size)
                })
            })
            .and_then(|(sx, sy)| {
                crate::render::marks::arrowhead::arrowhead_triangle(
                    sx,
                    sy,
                    placed_tip.0,
                    placed_tip.1,
                    head_size,
                )
            });

        match rebuilt {
            Some(points) => set_triangle_vertices(node, points),
            // No usable shaft direction: translate rigidly onto the placed
            // tip, which at least keeps the head attached and sized.
            None => self.for_each_along(node, &mut |c| *c = placed_along + (*c - along_tip)),
        }
    }

    /// Put `along` back into `point`'s broken-axis slot, keeping the other
    /// coordinate.
    fn join_point(&self, point: (f64, f64), along: f64) -> (f64, f64) {
        if self.is_y() { (point.0, along) } else { (along, point.1) }
    }

    /// The `(min, max)` of the node's coordinates along this break's axis.
    ///
    /// `None` for node kinds this walker does not visit (`Image`, `Raw`),
    /// which leaves them untouched.
    fn along_extent(&self, node: &mut SceneNode) -> Option<(f64, f64)> {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        self.for_each_along(node, &mut |c| {
            lo = lo.min(*c);
            hi = hi.max(*c);
        });
        (lo <= hi).then_some((lo, hi))
    }

    /// Visit every coordinate of `node` that lies along this break's axis.
    ///
    /// The one walker both glyph passes share (measure, then move or hide), so
    /// a node kind cannot be measured by one rule and rewritten by another.
    /// `Image` and `Raw` yield nothing — the same deliberate no-ops
    /// [`BreakRemap::remap_node`] names.
    ///
    /// A measuring pass leaves the node untouched: the only arm that has to
    /// *write* (a rect, whose second coordinate is derived) writes only when
    /// the closure actually moved something, so measuring cannot drift a
    /// rect's width through a `(start + len) - start` round trip.
    fn for_each_along(&self, node: &mut SceneNode, f: &mut impl FnMut(&mut f64)) {
        let is_y = self.is_y();
        match node {
            SceneNode::Circle { cx, cy, .. } => f(if is_y { cy } else { cx }),
            SceneNode::Rect { x, y, w, h, .. } => {
                // Both edges, so the walker measures the glyph's true extent
                // and the move keeps its size.
                let (start, len) = if is_y { (y, h) } else { (x, w) };
                let (old_start, end) = (*start, *start + *len);
                let mut moved_end = end;
                f(start);
                f(&mut moved_end);
                if *start != old_start || moved_end != end {
                    *len = moved_end - *start;
                }
            }
            SceneNode::Line { x1, y1, x2, y2, .. } => {
                let (a, b) = if is_y { (y1, y2) } else { (x1, x2) };
                f(a);
                f(b);
            }
            SceneNode::Text { x, y, .. } => f(if is_y { y } else { x }),
            SceneNode::Path { commands, .. } => {
                for cmd in commands.iter_mut() {
                    for_each_path_cmd_coord(cmd, is_y, f);
                }
            }
            SceneNode::Polyline { points, .. } => {
                for p in points.iter_mut() {
                    f(if is_y { &mut p.1 } else { &mut p.0 });
                }
            }
            // Visiting a ring's vertices is safe HERE, unlike in the vertex
            // model, because the glyph passes move every coordinate by the
            // same displacement: a rigidly translated ring keeps its shape and
            // cannot stretch across the break.
            SceneNode::Polygon { rings, .. } => {
                for ring in rings.iter_mut() {
                    for p in ring.iter_mut() {
                        f(if is_y { &mut p[1] } else { &mut p[0] });
                    }
                }
            }
            SceneNode::Group { children, .. } => {
                for child in children.iter_mut() {
                    self.for_each_along(child, f);
                }
            }
            SceneNode::Image { .. } | SceneNode::Raw { .. } => {}
        }
    }

    /// Split a point into `(along, across)`, where `along` is the coordinate
    /// this break moves.
    fn split(&self, p: (f64, f64)) -> (f64, f64) {
        if self.is_y() { (p.1, p.0) } else { (p.0, p.1) }
    }

    /// Inverse of [`BreakRemap::split`].
    fn join(&self, along: f64, across: f64) -> (f64, f64) {
        if self.is_y() { (across, along) } else { (along, across) }
    }

    /// Re-place data-positioned axis chrome (tick marks, tick labels) onto the
    /// broken axis.
    ///
    /// Every node given here IS this break's chrome: ownership is decided by
    /// the caller from the axis's own identity
    /// (`marks::axis::AxisNodes::placed_along`), not inferred here from a
    /// coordinate. The pixel-range guard this replaces asked "does this node
    /// sit inside my axis's pixel span", which is true of the wrong axis's
    /// chrome too — an x-axis tick dash starts ON the plot floor, inside the
    /// y range, so a y-break admitted all 11 of them and then clamped both
    /// ends of each onto one pixel, collapsing every x tick to zero length.
    ///
    /// Two of the axis's nodes are never in this list, both for the same
    /// reason: nothing data-shaped positioned them. The *title* is placed
    /// from the axis's own midpoint, so reverse-mapping it is meaningless —
    /// and when that midpoint happened to land in a gap, it teleported the
    /// title to `BREAK_HIDDEN`. The *domain line*'s endpoints are the axis
    /// rect's corners, so under a break they clamped to the domain's ends and
    /// the line shrank to the padded span (#226); it stays full-rect, exactly
    /// as on an unbroken chart. `marks::axis::AxisNodes` keeps all three
    /// apart at the source.
    ///
    /// D2: each node here is translated RIGIDLY on its own tick's `anchor`
    /// (see [`marks::axis::AxisTickNode`]), not reverse-mapped from its own
    /// drawn coordinate — [`BreakRemap::remap_axis_tick_node`].
    pub fn remap_axis_chrome(&self, nodes: &mut [marks::axis::AxisTickNode]) {
        for tick_node in nodes.iter_mut() {
            self.remap_axis_tick_node(&mut tick_node.node, tick_node.anchor);
        }
    }

    /// Re-place one anchored axis-chrome node — a tick dash or a tick label —
    /// by translating it rigidly on `anchor`, its own tick's data-mapped
    /// pixel position (D2, spec §4 "Tick labels").
    ///
    /// Shaped exactly like [`BreakRemap::remap_glyph_node`], except the
    /// anchor is supplied by the caller instead of computed from the node's
    /// own extent: a label's drawn coordinate (`tick.position + font/3`, a
    /// `label_flush` edge shift, rotation geometry...) is NOT the position a
    /// break should reverse-map, because doing so answers "where does this
    /// baseline offset itself belong on the broken scale" rather than "where
    /// does this tick's own value land" — the question whose wrong answer let
    /// siblings' baseline gaps compress by different amounts per segment and
    /// read a flush edge label's offset as exactly zero. Translating by the
    /// anchor's own displacement instead keeps the drawn offset bit for bit,
    /// and hides the whole node — dash and label alike — iff the ANCHOR
    /// (never the label's own baseline) falls inside a gap, so a label can
    /// never hide while its own tick/gridline stays visible.
    ///
    /// A tick dash's own two endpoints already equal its anchor
    /// (`marks::axis::build_axis` draws both from `tick.position`), so for a
    /// dash this reduces to translating both ends by the same displacement
    /// `coord(anchor)` produces for either — byte-identical to the exact
    /// vertex-by-vertex remap dashes had before this method existed.
    ///
    /// Only `Line` (a dash) and `Text` (a label) ever reach here — the two
    /// kinds [`marks::axis::build_axis`] tags with an anchor. Any other kind
    /// arriving here would be a node this module has no anchor contract for;
    /// rather than silently falling back to the vertex-remap's own coordinate
    /// (the bug this method exists to close), a debug build panics so the
    /// producer is fixed instead of the symptom being masked.
    ///
    /// Release-mode note: the `debug_assert!` above is compiled out in
    /// release, so an unanchored kind reaching here is not rejected there —
    /// [`BreakRemap::translate_on_anchor`] still runs and translates
    /// `Circle`/`Rect`/`Path`/`Polyline`/`Polygon`/`Group` rigidly like any
    /// other glyph. The one asymmetry is `Image`/`Raw`: `for_each_along`
    /// visits neither, so such a node is left in place at its unbroken
    /// position — untranslated and NOT hidden — rather than either moving or
    /// disappearing.
    fn remap_axis_tick_node(&self, node: &mut SceneNode, anchor: f64) {
        debug_assert!(
            matches!(node, SceneNode::Line { .. } | SceneNode::Text { .. }),
            "axis-chrome anchor node must be a tick dash or a tick label, got {node:?}"
        );
        self.translate_on_anchor(node, anchor);
    }

    /// Translate every along-axis coordinate of `node` rigidly by the
    /// displacement its own `anchor` undergoes under this break, hiding the
    /// whole node at `BREAK_HIDDEN` (offset by that same displacement) when
    /// `anchor` falls inside a gap.
    ///
    /// The one definition [`BreakRemap::remap_glyph_node`] and
    /// [`BreakRemap::remap_axis_tick_node`] both call — the translation rule
    /// and its hide destination cannot drift between "anchor computed from
    /// the node's own extent" (a plain glyph) and "anchor supplied by the
    /// caller" (a tick dash or label), because both compute the anchor
    /// however they need to and then hand it to the same primitive.
    fn translate_on_anchor(&self, node: &mut SceneNode, anchor: f64) {
        // Hiding is the same rule with `BREAK_HIDDEN` as the destination, so a
        // hidden glyph keeps its shape off-screen instead of collapsing to a
        // zero-size artifact — and a one-coordinate node reduces to
        // `coord = BREAK_HIDDEN`, the assignment it has always had.
        let moved = self.coord(anchor).unwrap_or(BREAK_HIDDEN);
        self.for_each_along(node, &mut |c| *c = moved + (*c - anchor));
    }

    /// Re-place a single node's coordinates along the broken axis, hiding any
    /// that fall inside a gap — the 1→1 model gridlines use, and the FALLBACK
    /// for geometry the 1→N rewrites cannot cut faithfully. Recurses into
    /// `Group` children.
    ///
    /// Every coordinate moves independently, so a node whose coordinates sit on
    /// both sides of a gap keeps the ones it can place and parks the rest at
    /// `BREAK_HIDDEN`. For CONNECTED geometry that leaves the stroke attached to
    /// the sentinel — the spike #230-1 describes — which is why vertex-placed
    /// mark batches no longer come here: they split instead
    /// ([`BreakRemap::split_nodes`]), and what still reaches this arm through a
    /// split is only geometry with no faithful cut (a curved `Path`), plus
    /// chrome whose along-axis extent is not data (a gridline, which is one
    /// value wide along the broken axis and so hides whole).
    pub fn remap_node(&self, node: &mut SceneNode) {
        let is_y = self.is_y();
        match node {
            SceneNode::Circle { cx, cy, .. } => {
                let coord = if is_y { cy } else { cx };
                *coord = self.coord(*coord).unwrap_or(BREAK_HIDDEN);
            }
            // A rect remapped in place cannot split, so it keeps only its
            // first visible segment — the 1→1 degradation of the clip. Callers
            // that can grow their node list (mark batches, gridlines) go
            // through `clip_nodes` and get every visible segment.
            //
            // This arm is unreachable for a `corner_radius`-bearing rect in
            // production: every producer of an extent rect that can carry a
            // radius (bar, rect) is `BreakPlacement::ClipExtent` and routes
            // through `clip_nodes` -> `clip_node_into`'s dedicated `Rect` arm,
            // which owns the cut-edge-aware radius handling `rounded_rect_piece`
            // builds. A rect that DOES reach this fallback (a point mark's
            // `Square` shape, always `corner_radius: 0.0`) never carries a
            // radius, so the assertion below turns that claim into something
            // the suite enforces rather than a prose trace that can drift.
            SceneNode::Rect { x, y, w, h, corner_radius, .. } => {
                debug_assert_eq!(
                    *corner_radius,
                    0.0,
                    "remap_node's Rect arm is the 1->1 fallback and does not apply \
                     rounded_rect_piece's cut-edge-aware radius handling; a \
                     corner_radius-bearing rect must route through clip_nodes -> \
                     clip_node_into instead"
                );
                let (start, len) = if is_y { (y, h) } else { (x, w) };
                match self.clip_extent(*start, *start + *len).first() {
                    Some(&(lo, hi)) => {
                        *start = lo;
                        *len = hi - lo;
                    }
                    None => *len = 0.0,
                }
            }
            SceneNode::Line { x1, y1, x2, y2, .. } => {
                let (a, b) = if is_y { (y1, y2) } else { (x1, x2) };
                *a = self.coord(*a).unwrap_or(BREAK_HIDDEN);
                *b = self.coord(*b).unwrap_or(BREAK_HIDDEN);
            }
            SceneNode::Path { commands, .. } => {
                for cmd in commands.iter_mut() {
                    self.remap_path_cmd(cmd);
                }
            }
            // `mark_line` emits its whole series as one `Polyline` (see
            // `marks/line.rs`), so leaving this arm out — as it was before —
            // meant a line chart under a break drew its line against the
            // unbroken scale while the ticks beside it compressed.
            SceneNode::Polyline { points, .. } => {
                for p in points.iter_mut() {
                    let coord = if is_y { &mut p.1 } else { &mut p.0 };
                    *coord = self.coord(*coord).unwrap_or(BREAK_HIDDEN);
                }
            }
            SceneNode::Text { x, y, .. } => {
                let coord = if is_y { y } else { x };
                *coord = self.coord(*coord).unwrap_or(BREAK_HIDDEN);
            }
            // A filled ring cannot take the vertex model: parking its in-gap
            // vertices at `BREAK_HIDDEN` would paint the fill as a column
            // across the break. It clips instead — the same answer the area
            // fill gets, and the same answer this node kind gets through
            // `clip_nodes`, so a `Polygon` behaves identically whichever model
            // its batch carries.
            SceneNode::Polygon { rings, .. } => self.clip_rings(rings),
            SceneNode::Group { children, .. } => {
                for child in children.iter_mut() {
                    self.remap_node(child);
                }
            }
            // Deliberate no-ops, named rather than wildcarded so a new
            // `SceneNode` kind is a compile error here (the hole cycle-1's
            // `Polyline` and cycle-2's `Polygon` both escaped through).
            // `Image`: an image mark's placement under a break has no
            // established answer and no producer asking for one. `Raw`: an
            // opaque SVG fragment whose coordinates this module cannot parse.
            SceneNode::Image { .. } | SceneNode::Raw { .. } => {}
        }
    }

    /// Re-place a single `PathCmd`'s coordinates along the broken axis.
    fn remap_path_cmd(&self, cmd: &mut PathCmd) {
        for_each_path_cmd_coord(cmd, self.is_y(), &mut |c| {
            *c = self.coord(*c).unwrap_or(BREAK_HIDDEN);
        });
    }
}

/// Visit the coordinates of `cmd` that lie along the broken axis.
///
/// The single answer to "which numbers in this path command does a break on
/// this axis move" — shared by the per-vertex remap and the glyph walker, so
/// the two cannot disagree about a command kind. Control points travel with
/// their endpoint; arc radii are not positions and are left alone.
fn for_each_path_cmd_coord(cmd: &mut PathCmd, is_y: bool, f: &mut impl FnMut(&mut f64)) {
    match cmd {
        PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => f(if is_y { y } else { x }),
        PathCmd::QuadTo { cx, cy, x, y } => {
            if is_y {
                f(cy);
                f(y);
            } else {
                f(cx);
                f(x);
            }
        }
        PathCmd::CubicTo { c1x, c1y, c2x, c2y, x, y } => {
            if is_y {
                f(c1y);
                f(c2y);
                f(y);
            } else {
                f(c1x);
                f(c2x);
                f(x);
            }
        }
        PathCmd::ArcTo { x, y, .. } => f(if is_y { y } else { x }),
        PathCmd::HLineTo { x } => {
            if !is_y {
                f(x);
            }
        }
        PathCmd::VLineTo { y } => {
            if is_y {
                f(y);
            }
        }
        PathCmd::Close => {}
    }
}

// Test coverage lives in `tests.rs` (precedent: `render/scale_resolve/tests.rs`,
// `projection/mod.rs`, `transform/stats/mod.rs`, and this wave's own
// `annotation/mod.rs`) — the module grew too large for an inline
// `mod tests { ... }` block: at 2618 lines the inline test body (`mod tests`
// starting at line 2619) reached 2622 lines, exceeding the source module
// above it and crossing the promotion threshold in this crate's Rust
// test-module convention. The source has since split four ways (see the
// module header), but the tests stay in one file: they exercise the whole
// break through its public surface rather than one job at a time, and
// `use super::*` reaches every re-export, so partitioning them by the new
// file names would cut across what each test actually covers.
#[cfg(test)]
mod tests;
