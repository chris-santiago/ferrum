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

use ferrum_scene::{FillStroke, MarkBatchKind, PathCmd, SceneNode};

use crate::layout::{Rect, ThemeInputs};
use crate::render::draw::to_scene_stroke;

// ── Public types ─────────────────────────────────────────────────────────────

/// Segments of a broken scale. Each segment covers a contiguous range of the
/// original data domain and maps to a contiguous range of pixels.
///
/// Constructed by `apply_break_to_scale` and consumed by `broken_scale_map`,
/// which [`BreakRemap`] calls to re-position scene coordinates onto the broken
/// axis.
#[derive(Debug, Clone)]
pub struct ScaleSegment {
    /// Lower bound of the data-space range covered by this segment.
    pub data_lo: f64,
    /// Upper bound of the data-space range covered by this segment.
    pub data_hi: f64,
    /// Lower bound of the pixel-space range for this segment (top for Y axis).
    pub pixel_lo: f64,
    /// Upper bound of the pixel-space range for this segment (bottom for Y axis).
    pub pixel_hi: f64,
}

impl ScaleSegment {
    /// This segment's data range as an ordered `(lo, hi)` pair.
    ///
    /// `data_lo`/`data_hi` are stored in domain order, which is ascending for
    /// every scale `apply_break_to_scale` builds; the ordering here makes the
    /// range comparisons below independent of that.
    pub fn ordered_data(&self) -> (f64, f64) {
        (self.data_lo.min(self.data_hi), self.data_lo.max(self.data_hi))
    }

    /// True when `value` lies within this segment's data range (inclusive of
    /// both bounds — a value exactly on a gap edge belongs to the segment).
    pub fn contains(&self, value: f64) -> bool {
        let (lo, hi) = self.ordered_data();
        value >= lo && value <= hi
    }

    /// Map a data value onto this segment's pixel range.
    ///
    /// The value is clamped into the segment first, so a caller that has
    /// already intersected an extent against [`ScaleSegment::ordered_data`]
    /// cannot produce an out-of-segment pixel from float drift at the
    /// boundary — the failure mode that made the old per-vertex remap emit
    /// `BREAK_HIDDEN` spikes in the middle of a clipped area.
    pub fn pixel_for(&self, value: f64) -> f64 {
        let t = if self.data_hi != self.data_lo {
            ((value - self.data_lo) / (self.data_hi - self.data_lo)).clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.pixel_lo + t * (self.pixel_hi - self.pixel_lo)
    }
}

/// Output of `apply_break_to_scale` — a piecewise mapping and the pixel
/// positions of the break indicators.
///
/// `segments` is used by `broken_scale_map` for mark position adjustment;
/// `indicator_pixels` is used by `build_break_indicators`.
#[derive(Debug, Clone)]
pub struct BreakResult {
    /// Ordered list of contiguous data segments (excluding the gaps).
    /// Consumed by `broken_scale_map` to map data values to pixel positions,
    /// which [`BreakRemap`] drives for break-axis scene re-positioning.
    pub segments: Vec<ScaleSegment>,
    /// Pixel midpoints of each break indicator, along the broken axis.
    pub indicator_pixels: Vec<f64>,
}

// ── Scale computation ─────────────────────────────────────────────────────────

/// Clip `gaps` to `[d_lo, d_hi]`, then sort and merge them into disjoint,
/// ascending ranges.
///
/// Merging is what makes overlapping and nested gaps well-defined: `[100, 900]`
/// plus `[200, 300]` is one removed range `[100, 900]`, not two. Without it the
/// segment walk below took the *last* gap's upper bound as its cursor, so a
/// nested gap moved the cursor backwards and re-admitted data the enclosing gap
/// had removed; and the reserved indicator space counted a break the scale never
/// draws. Adjacent gaps (`hi == lo`) merge too — they remove one contiguous
/// range and get one indicator, so they must reserve one indicator's pixels.
fn normalize_gaps(gaps: &[[f64; 2]], d_lo: f64, d_hi: f64) -> Vec<(f64, f64)> {
    let mut clipped: Vec<(f64, f64)> = gaps
        .iter()
        .filter_map(|g| {
            let g_lo = g[0].max(d_lo);
            let g_hi = g[1].min(d_hi);
            if g_hi > g_lo { Some((g_lo, g_hi)) } else { None }
        })
        .collect();
    // f64 gap bounds are finite data values; Equal is the safe NaN fallback.
    clipped.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(clipped.len());
    for (g_lo, g_hi) in clipped {
        match merged.last_mut() {
            Some(prev) if g_lo <= prev.1 => prev.1 = prev.1.max(g_hi),
            _ => merged.push((g_lo, g_hi)),
        }
    }
    merged
}

/// Build a piecewise scale that skips `gaps` in the data domain.
///
/// `original_domain` is the full `(lo, hi)` of the axis before breaking.
/// `pixel_range` is `(axis_start_px, axis_end_px)` — for Y this is typically
/// `(plot_top, plot_bottom)`, but the function is axis-agnostic.
/// `break_size` is the pixel height/width reserved for each indicator.
///
/// Segments are allocated proportionally to their data-domain fraction of the
/// total non-gap span. Gaps that fall entirely outside `original_domain` are
/// silently ignored, and gaps that overlap (or merely touch) each other are
/// merged into one before segments are computed — see [`normalize_gaps`].
pub fn apply_break_to_scale(
    original_domain: (f64, f64),
    gaps: &[[f64; 2]],
    pixel_range: (f64, f64),
    break_size: f64,
) -> BreakResult {
    let (d_lo, d_hi) = original_domain;
    let (px_lo, px_hi) = pixel_range;
    let total_px = (px_hi - px_lo).abs();

    let valid_gaps = normalize_gaps(gaps, d_lo, d_hi);

    if valid_gaps.is_empty() {
        // No effective gaps — single segment spanning the full domain.
        return BreakResult {
            segments: vec![ScaleSegment {
                data_lo: d_lo,
                data_hi: d_hi,
                pixel_lo: px_lo,
                pixel_hi: px_hi,
            }],
            indicator_pixels: Vec::new(),
        };
    }

    // Build ordered list of retained data ranges (domain minus gaps).
    // `normalize_gaps` already sorted and merged, so the cursor only ever
    // moves forward: a nested gap can no longer walk it backwards and
    // re-admit a range an enclosing gap had removed.
    let mut segment_ranges: Vec<(f64, f64)> = Vec::new();
    let mut cursor = d_lo;
    for (g_lo, g_hi) in &valid_gaps {
        if cursor < *g_lo {
            segment_ranges.push((cursor, *g_lo));
        }
        cursor = *g_hi;
    }
    if cursor < d_hi {
        segment_ranges.push((cursor, d_hi));
    }

    // Total data span retained (excluding gaps).
    let total_data_span: f64 = segment_ranges.iter().map(|(lo, hi)| hi - lo).sum();

    // Pixels available for data segments, after reserving space for the break
    // indicators. Counted per indicator ACTUALLY EMITTED — one between each
    // pair of adjacent segments, which is what `indicator_pixels` below
    // derives from `segments.windows(2)` — not per removed range. A gap that
    // touches `d_lo` or `d_hi` removes a range without separating two
    // segments, so it draws no indicator and must withhold no pixels; before
    // this it withheld one `break_size` for an indicator that never appeared,
    // leaving a dead band at the plot edge.
    let n_breaks = segment_ranges.len().saturating_sub(1) as f64;
    let pixels_for_data = (total_px - n_breaks * break_size).max(1.0);

    // Assign pixel ranges to each data segment proportionally.
    let mut segments: Vec<ScaleSegment> = Vec::new();
    let mut px_cursor = px_lo;

    for (seg_lo, seg_hi) in &segment_ranges {
        let data_fraction = if total_data_span > 0.0 {
            (seg_hi - seg_lo) / total_data_span
        } else {
            1.0 / segment_ranges.len() as f64
        };
        let seg_px = pixels_for_data * data_fraction;
        let seg_px_hi = px_cursor + seg_px * (px_hi - px_lo).signum();
        segments.push(ScaleSegment {
            data_lo: *seg_lo,
            data_hi: *seg_hi,
            pixel_lo: px_cursor,
            pixel_hi: seg_px_hi,
        });
        // Advance past this segment and the following break indicator
        // (unless this is the last segment).
        px_cursor = seg_px_hi;
        if segments.len() < segment_ranges.len() {
            px_cursor += break_size * (px_hi - px_lo).signum();
        }
    }

    // Indicator pixel midpoints: midpoint between each pair of adjacent segments.
    let indicator_pixels: Vec<f64> = segments
        .windows(2)
        .map(|w| (w[0].pixel_hi + w[1].pixel_lo) / 2.0)
        .collect();

    BreakResult { segments, indicator_pixels }
}

/// Map a single data value to its pixel position under the broken scale.
///
/// Returns `None` when `value` falls within a gap.
///
/// Consumed by [`BreakRemap::coord`] to re-position mark coordinates onto the
/// broken axis; also exercised directly by this module's unit tests.
pub fn broken_scale_map(value: f64, result: &BreakResult) -> Option<f64> {
    result
        .segments
        .iter()
        .find(|seg| seg.contains(value))
        .map(|seg| seg.pixel_for(value))
}

// ── Indicator rendering ───────────────────────────────────────────────────────

/// Render break indicators at `result.indicator_pixels` for the given axis and style.
///
/// `plot_area` provides the perpendicular extent (e.g. for a Y-axis break, the
/// x-extent of the plot area determines where the indicators are drawn).
pub fn build_break_indicators(
    result: &BreakResult,
    plot_area: &Rect,
    axis: &str,
    break_size: f64,
    break_style: &str,
    theme: &ThemeInputs,
) -> Vec<SceneNode> {
    let mut nodes = Vec::new();

    let stroke = to_scene_stroke(theme.colors.axis_line_color, theme.sizes.axis_line_width, 1.0, None, None, None);

    for &indicator_px in &result.indicator_pixels {
        let new_nodes = match break_style {
            "gap" => Vec::new(),
            "zigzag" => build_zigzag_indicator(indicator_px, plot_area, axis, break_size, stroke.clone()),
            "wave" => build_wave_indicator(indicator_px, plot_area, axis, break_size, stroke.clone()),
            _ => build_slash_indicator(indicator_px, plot_area, axis, break_size, stroke.clone()),
        };
        nodes.extend(new_nodes);
    }

    nodes
}

/// Two parallel diagonal slashes (`//`) at the break position.
fn build_slash_indicator(
    indicator_px: f64,
    plot_area: &Rect,
    axis: &str,
    break_size: f64,
    stroke: ferrum_scene::StrokeStyle,
) -> Vec<SceneNode> {
    let half = break_size / 2.0;
    let diag = 6.0; // diagonal extension in pixels

    if axis == "y" {
        let x1 = plot_area.x + plot_area.w * 0.35;
        let x2 = plot_area.x + plot_area.w * 0.65;
        let x_mid = (x1 + x2) / 2.0;

        vec![
            // First slash
            SceneNode::Line {
                x1: x_mid - diag / 2.0,
                y1: indicator_px - half,
                x2: x_mid + diag / 2.0,
                y2: indicator_px + half,
                style: stroke.clone(),
            },
            // Second slash (offset to the right)
            SceneNode::Line {
                x1: x_mid - diag / 2.0 + diag,
                y1: indicator_px - half,
                x2: x_mid + diag / 2.0 + diag,
                y2: indicator_px + half,
                style: stroke,
            },
        ]
    } else {
        // x-axis break
        let y1 = plot_area.y + plot_area.h * 0.35;
        let y2 = plot_area.y + plot_area.h * 0.65;
        let y_mid = (y1 + y2) / 2.0;

        vec![
            SceneNode::Line {
                x1: indicator_px - half,
                y1: y_mid - diag / 2.0,
                x2: indicator_px + half,
                y2: y_mid + diag / 2.0,
                style: stroke.clone(),
            },
            SceneNode::Line {
                x1: indicator_px - half,
                y1: y_mid - diag / 2.0 + diag,
                x2: indicator_px + half,
                y2: y_mid + diag / 2.0 + diag,
                style: stroke,
            },
        ]
    }
}

/// A zigzag path at the break position.
fn build_zigzag_indicator(
    indicator_px: f64,
    plot_area: &Rect,
    axis: &str,
    break_size: f64,
    stroke: ferrum_scene::StrokeStyle,
) -> Vec<SceneNode> {
    let half = break_size / 2.0;
    let amplitude = 5.0;
    let n_zags = 4i32;

    let commands = if axis == "y" {
        let x_center = plot_area.x + plot_area.w / 2.0;
        let mut cmds = Vec::new();
        cmds.push(PathCmd::MoveTo { x: x_center - amplitude, y: indicator_px - half });
        for i in 0..=n_zags {
            let t = i as f64 / n_zags as f64;
            let y = indicator_px - half + t * break_size;
            let x = if i % 2 == 0 {
                x_center - amplitude
            } else {
                x_center + amplitude
            };
            cmds.push(PathCmd::LineTo { x, y });
        }
        cmds
    } else {
        let y_center = plot_area.y + plot_area.h / 2.0;
        let mut cmds = Vec::new();
        cmds.push(PathCmd::MoveTo { x: indicator_px - half, y: y_center - amplitude });
        for i in 0..=n_zags {
            let t = i as f64 / n_zags as f64;
            let x = indicator_px - half + t * break_size;
            let y = if i % 2 == 0 {
                y_center - amplitude
            } else {
                y_center + amplitude
            };
            cmds.push(PathCmd::LineTo { x, y });
        }
        cmds
    };

    vec![SceneNode::Path {
        commands,
        style: FillStroke {
            fill: None,
            stroke: Some(stroke.color),
            stroke_width: stroke.width,
            opacity: stroke.opacity,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        },
        closed: false,
    }]
}

/// A wave/tilde path at the break position.
fn build_wave_indicator(
    indicator_px: f64,
    plot_area: &Rect,
    axis: &str,
    break_size: f64,
    stroke: ferrum_scene::StrokeStyle,
) -> Vec<SceneNode> {
    let half = break_size / 2.0;
    let amplitude = 4.0;
    let n_steps = 20i32;

    let commands = if axis == "y" {
        let x_center = plot_area.x + plot_area.w / 2.0;
        let mut cmds = Vec::new();
        cmds.push(PathCmd::MoveTo {
            x: x_center,
            y: indicator_px - half,
        });
        for i in 1..=n_steps {
            let t = i as f64 / n_steps as f64;
            let y = indicator_px - half + t * break_size;
            let x = x_center + amplitude * (std::f64::consts::PI * 2.0 * t).sin();
            cmds.push(PathCmd::LineTo { x, y });
        }
        cmds
    } else {
        let y_center = plot_area.y + plot_area.h / 2.0;
        let mut cmds = Vec::new();
        cmds.push(PathCmd::MoveTo {
            x: indicator_px - half,
            y: y_center,
        });
        for i in 1..=n_steps {
            let t = i as f64 / n_steps as f64;
            let x = indicator_px - half + t * break_size;
            let y = y_center + amplitude * (std::f64::consts::PI * 2.0 * t).sin();
            cmds.push(PathCmd::LineTo { x, y });
        }
        cmds
    };

    vec![SceneNode::Path {
        commands,
        style: FillStroke {
            fill: None,
            stroke: Some(stroke.color),
            stroke_width: stroke.width,
            opacity: stroke.opacity,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        },
        closed: false,
    }]
}

// ── Scene remapping ───────────────────────────────────────────────────────────

/// Coordinate used to park geometry that falls inside a break gap off-screen.
///
/// A large off-screen value rather than NaN, which would panic the SVG
/// serializer. Which geometry is hidden this way, and which is clipped
/// instead, is decided per mark batch by [`BreakPlacement`].
pub const BREAK_HIDDEN: f64 = -99999.0;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakPlacement {
    /// The mark's pixel *extent* along the axis is data (a bar's height, an
    /// area's band, a violin's outline), so it is clipped to the visible
    /// segments: one sub-mark per segment it intersects, none when it lies
    /// wholly inside a gap. A `Polygon` node clips its rings in place, so it
    /// stays one node with one row. See [`BreakRemap::clip_nodes`].
    ClipExtent,
    /// The mark is a fixed-size glyph anchored at one data position (a point's
    /// shape, a tick or rug dash, a text label). It translates to its anchor's
    /// compressed position — keeping its drawn size and shape — and hides whole
    /// when the anchor is inside a gap. Never resized, never split, never
    /// dropped, so glyph batches stay 1:1 with their rows whatever `shape=`
    /// they carry. See [`BreakRemap::remap_glyph_node`].
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
    MoveGlyphAndConnector,
    /// The mark is connected *stroked* geometry whose every vertex is its own
    /// data position (a line, a rule, a segment). Each vertex is placed
    /// independently and in-gap vertices are hidden — spec §4 H4's line-vertex
    /// rule. See [`BreakRemap::remap_node`].
    ///
    /// A vertex parked at [`BREAK_HIDDEN`] leaves the stroke connected to it,
    /// so a line crossing a gap draws a spike off-canvas and back. That is the
    /// honest cost of the sentinel model for connected geometry and the reason
    /// *filled* geometry does not use it: a filled ring treated this way would
    /// paint a column across the whole break. Splitting connected geometry per
    /// segment (the line analogue of the area clip) is the separate follow-up.
    MoveVertices,
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
            // and polar bars), where Cartesian axes — and so Cartesian breaks —
            // are suppressed; it takes the vertex model as the closest honest
            // answer rather than a separate polar one.
            MarkBatchKind::Line
            | MarkBatchKind::Rule
            | MarkBatchKind::Segment
            | MarkBatchKind::Arc => Self::MoveVertices,
        }
    }
}

/// A break resolved against the unbroken scale it replaces.
///
/// Scene nodes are built against the unbroken scale, so re-placing them on the
/// broken axis is a round trip: reverse-map a pixel coordinate to data through
/// the unbroken scale, then forward-map that value through the piecewise
/// [`BreakResult`]. This type carries the four numbers that round trip needs
/// alongside the axis they apply to, so every caller (marks, gridlines, axis
/// chrome) performs it identically.
pub struct BreakRemap<'a> {
    /// `"x"` or `"y"` — which coordinate of a node this break moves.
    axis: &'a str,
    /// `(lo, hi)` data domain of the unbroken scale.
    domain: (f64, f64),
    /// `(lo, hi)` pixel range of the unbroken scale, in that scale's own
    /// orientation (for Y this is `(bottom, top)`).
    pixels: (f64, f64),
    result: &'a BreakResult,
}

/// The outcome of rebuilding a node list through a break.
///
/// Clipping is a 1→N rewrite: a bar crossing one gap becomes two rects, a bar
/// whose whole extent sits inside a gap becomes none. `sources` records, for
/// each emitted node, the index of the node it came from, so the batch's
/// per-node sidecars (`data_indices`, tooltips, hrefs, descriptions, keys)
/// can be rebuilt in lockstep instead of silently drifting out of alignment.
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
    pub fn realign<T: Clone>(&self, sidecar: Option<Vec<T>>) -> Option<Vec<T>> {
        let sidecar = sidecar?;
        debug_assert_eq!(
            sidecar.len(),
            self.source_len,
            "per-node sidecar must have one entry per pre-clip node"
        );
        if sidecar.len() != self.source_len {
            return Some(sidecar);
        }
        Some(self.sources.iter().map(|&i| sidecar[i].clone()).collect())
    }
}

/// Which side of a boundary a half-plane clip keeps.
#[derive(Clone, Copy)]
enum Side {
    AtLeast,
    AtMost,
}

impl<'a> BreakRemap<'a> {
    /// Resolve `result` against the unbroken scale's `domain` and `pixels`.
    pub fn new(
        axis: &'a str,
        domain: (f64, f64),
        pixels: (f64, f64),
        result: &'a BreakResult,
    ) -> Self {
        Self { axis, domain, pixels, result }
    }

    /// `"x"` or `"y"` — the axis this break moves.
    pub fn axis(&self) -> &str {
        self.axis
    }

    fn is_y(&self) -> bool {
        self.axis == "y"
    }

    /// Reverse-map a pixel coordinate to data on the unbroken scale, clamped
    /// to its domain.
    ///
    /// The reverse-map is linear — `data` is `d_lo` plus the pixel's fraction
    /// of `px_hi - px_lo` times `d_hi - d_lo`. That is correct for every
    /// continuous scale type: all ferrum scales produce a linear pixel
    /// mapping, the non-linearity lives in the data transform, not in the
    /// range. Clamping keeps geometry that overshoots the domain (a bar drawn
    /// down to the plot floor, which sits below the scale's zero) attached to
    /// the nearest segment instead of vanishing.
    fn data_at(&self, px: f64) -> f64 {
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
        if a.is_nan() || b.is_nan() {
            return Vec::new();
        }
        let (d0, d1) = (self.data_at(a), self.data_at(b));
        let (lo, hi) = (d0.min(d1), d0.max(d1));
        let mut out = Vec::with_capacity(self.result.segments.len());
        for seg in &self.result.segments {
            let (s_lo, s_hi) = seg.ordered_data();
            if hi <= lo {
                // Zero-length extent: visible iff the single value is in this
                // segment, and in exactly one segment when it is.
                if seg.contains(lo) {
                    let p = seg.pixel_for(lo);
                    out.push((p, p));
                    break;
                }
                continue;
            }
            let (c_lo, c_hi) = (lo.max(s_lo), hi.min(s_hi));
            if c_hi > c_lo {
                let (p0, p1) = (seg.pixel_for(c_lo), seg.pixel_for(c_hi));
                out.push((p0.min(p1), p0.max(p1)));
            }
        }
        out
    }

    /// Rebuild `nodes` onto the broken axis, clipping extent geometry to the
    /// visible segments.
    ///
    /// For [`BreakPlacement::ClipExtent`] batches only. `Rect`s and closed
    /// polygonal `Path`s (the area fill) are clipped per segment, which can
    /// split one node into several or drop it entirely; a `Group` is rebuilt
    /// child-wise but stays one node. Any other node kind found inside such a
    /// batch falls back to [`BreakRemap::remap_node`] and stays 1→1.
    ///
    /// Glyph batches must NOT come through here — extent-clipping a glyph
    /// resizes it and can delete it; they go to
    /// [`BreakRemap::remap_glyph_node`] instead.
    pub fn clip_nodes(&self, nodes: Vec<SceneNode>) -> ClippedNodes {
        let source_len = nodes.len();
        let mut out = Vec::with_capacity(source_len);
        let mut sources = Vec::with_capacity(source_len);
        for (idx, node) in nodes.into_iter().enumerate() {
            let before = out.len();
            self.clip_node_into(node, &mut out);
            sources.resize(out.len(), idx);
            debug_assert!(out.len() >= before);
        }
        ClippedNodes { nodes: out, sources, source_len }
    }

    fn clip_node_into(&self, node: SceneNode, out: &mut Vec<SceneNode>) {
        match node {
            SceneNode::Rect { x, y, w, h, style, corner_radius } => {
                let (start, len) = if self.is_y() { (y, h) } else { (x, w) };
                for (lo, hi) in self.clip_extent(start, start + len) {
                    let (nx, nw, ny, nh) = if self.is_y() {
                        (x, w, lo, hi - lo)
                    } else {
                        (lo, hi - lo, y, h)
                    };
                    out.push(SceneNode::Rect {
                        x: nx,
                        y: ny,
                        w: nw,
                        h: nh,
                        style: style.clone(),
                        corner_radius,
                    });
                }
            }
            SceneNode::Path { commands, style, closed } => {
                match polygon_points(&commands).filter(|_| closed) {
                    Some(points) => self.clip_polygon_into(&points, &style, out),
                    None => {
                        let mut node = SceneNode::Path { commands, style, closed };
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
            let mut commands = Vec::with_capacity(clipped.len() + 1);
            for (i, &(x, y)) in clipped.iter().enumerate() {
                commands.push(if i == 0 {
                    PathCmd::MoveTo { x, y }
                } else {
                    PathCmd::LineTo { x, y }
                });
            }
            commands.push(PathCmd::Close);
            out.push(SceneNode::Path { commands, style: style.clone(), closed: true });
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

    /// Re-place one fixed-size glyph — the [`BreakPlacement::MoveGlyph`] model.
    ///
    /// The glyph translates by the distance its own anchor (the midpoint of
    /// its extent along this axis) moves, so its drawn size and shape survive
    /// the compression: a diamond stays a diamond, a square stays square, and
    /// a `shape=` glyph behaves exactly like the default circle. An anchor
    /// inside a gap hides the whole node at `BREAK_HIDDEN` rather than any
    /// part of it, so a glyph straddling a gap edge cannot emit a spike, and
    /// no glyph is ever dropped — glyph batches stay 1:1 with their rows, and
    /// their `data_indices`/tooltip/key entries stay valid.
    ///
    /// For a single-coordinate node (`Circle`, `Text`) the anchor *is* the
    /// coordinate, so this reduces to the plain assignment those nodes always
    /// had, bit for bit.
    ///
    /// A `Line` node reaching here is a glyph too — a tick's dash, including a
    /// rug's, whose 8px length is drawn size and not data — so it moves
    /// rigidly and keeps that length. The one batch whose `Line`s are NOT
    /// glyphs is `mark_label`'s, and that batch is routed to
    /// [`BreakRemap::remap_label_node`] by its kind, never by a node-shape
    /// test here.
    pub fn remap_glyph_node(&self, node: &mut SceneNode) {
        let Some((lo, hi)) = self.along_extent(node) else {
            return;
        };
        let anchor = (lo + hi) / 2.0;
        // Hiding is the same rule with `BREAK_HIDDEN` as the destination, so a
        // hidden glyph keeps its shape off-screen instead of collapsing to a
        // zero-size artifact — and a one-coordinate node reduces to
        // `coord = BREAK_HIDDEN`, the assignment it has always had.
        let moved = self.coord(anchor).unwrap_or(BREAK_HIDDEN);
        self.for_each_along(node, &mut |c| *c = moved + (*c - anchor));
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
    pub fn remap_label_node(&self, node: &mut SceneNode) {
        match node {
            SceneNode::Line { .. } => self.remap_node(node),
            _ => self.remap_glyph_node(node),
        }
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

    /// Re-place data-positioned axis chrome (domain line, tick marks, tick
    /// labels) onto the broken axis.
    ///
    /// Nodes whose coordinate on this axis sits outside the unbroken scale's
    /// pixel range are cross-axis chrome (an x-axis label below the plot under
    /// a y-break) and are left alone. The axis *title* is never in this list:
    /// it is positioned from the axis's own midpoint rather than from a data
    /// value, so reverse-mapping it is meaningless — and when that midpoint
    /// happened to land in a gap, it teleported the title to `BREAK_HIDDEN`.
    /// `marks::axis::AxisNodes` keeps the two apart at the source.
    pub fn remap_axis_chrome(&self, nodes: &mut [SceneNode]) {
        let (lo, hi) = (self.pixels.0.min(self.pixels.1), self.pixels.0.max(self.pixels.1));
        for node in nodes.iter_mut() {
            if self.node_coord_in_range(node, lo, hi) {
                self.remap_node(node);
            }
        }
    }

    /// True when the node's primary coordinate on this axis falls within
    /// `[lo, hi]` (inclusive, with a 1px margin).
    fn node_coord_in_range(&self, node: &SceneNode, lo: f64, hi: f64) -> bool {
        let margin = 1.0;
        let coord = match node {
            SceneNode::Text { x, y, .. } => {
                if self.is_y() { *y } else { *x }
            }
            SceneNode::Line { x1, y1, x2, y2, .. } => {
                if self.is_y() { (*y1).min(*y2) } else { (*x1).min(*x2) }
            }
            SceneNode::Rect { x, y, .. } => {
                if self.is_y() { *y } else { *x }
            }
            SceneNode::Group { children, .. } => {
                return children.iter().any(|c| self.node_coord_in_range(c, lo, hi));
            }
            _ => return false,
        };
        coord >= lo - margin && coord <= hi + margin
    }

    /// Re-place a single node's coordinates along the broken axis, hiding any
    /// that fall inside a gap — the [`BreakPlacement::MoveVertices`] model,
    /// and the model axis chrome and gridlines use. Recurses into `Group`
    /// children.
    ///
    /// Every coordinate moves independently, so a line whose vertices sit on
    /// both sides of a gap keeps the vertices it can place and parks the rest
    /// at `BREAK_HIDDEN`. That is spec §4 H4's line-vertex rule, and it is
    /// deliberately NOT the extent clip: splitting connected geometry into one
    /// sub-line per segment is a separate piece of work.
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
            SceneNode::Rect { x, y, w, h, .. } => {
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

/// The vertices of a path built only from `MoveTo`/`LineTo` (plus an optional
/// trailing `Close`) — the shape every area fill emits under the linear and
/// step interpolations.
///
/// `None` for anything else (curve commands, multiple subpaths), which routes
/// the node to the per-vertex remap instead of the polygon clip.
fn polygon_points(commands: &[PathCmd]) -> Option<Vec<(f64, f64)>> {
    let mut points = Vec::with_capacity(commands.len());
    for (i, cmd) in commands.iter().enumerate() {
        match cmd {
            PathCmd::MoveTo { x, y } if i == 0 => points.push((*x, *y)),
            PathCmd::LineTo { x, y } => points.push((*x, *y)),
            PathCmd::Close if i + 1 == commands.len() => {}
            _ => return None,
        }
    }
    (points.len() >= 3).then_some(points)
}

/// Clip a polygon of `(along, across)` points to `lo <= along <= hi`.
///
/// Sutherland–Hodgman against the two boundaries; `across` is interpolated
/// linearly at each crossing. Returns fewer than three points when nothing
/// survives.
fn clip_to_band(points: &[(f64, f64)], lo: f64, hi: f64) -> Vec<(f64, f64)> {
    let lower = clip_half_plane(points, lo, Side::AtLeast);
    if lower.len() < 3 {
        return Vec::new();
    }
    clip_half_plane(&lower, hi, Side::AtMost)
}

fn clip_half_plane(points: &[(f64, f64)], bound: f64, side: Side) -> Vec<(f64, f64)> {
    let inside = |along: f64| match side {
        Side::AtLeast => along >= bound,
        Side::AtMost => along <= bound,
    };
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(points.len() + 2);
    for (i, &cur) in points.iter().enumerate() {
        let prev = points[(i + points.len() - 1) % points.len()];
        let (cur_in, prev_in) = (inside(cur.0), inside(prev.0));
        if cur_in != prev_in {
            let denom = cur.0 - prev.0;
            let t = if denom.abs() < f64::EPSILON {
                0.0
            } else {
                (bound - prev.0) / denom
            };
            out.push((bound, prev.1 + t * (cur.1 - prev.1)));
        }
        if cur_in {
            out.push(cur);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::ThemeInputs;

    fn plot() -> Rect {
        Rect { x: 50.0, y: 50.0, w: 300.0, h: 300.0 }
    }

    // ── broken_scale_map tests ───────────────────────────────────────────────

    #[test]
    fn broken_scale_map_value_in_gap_returns_none() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        // 500 is squarely inside the gap [200, 800].
        assert!(broken_scale_map(500.0, &result).is_none());
    }

    #[test]
    fn broken_scale_map_gap_boundary_is_none() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        // 200 is on the gap boundary — in the gap (exclusive of the retained segment end).
        // Values inside gaps should return None; segment boundaries are included in segments.
        // 0 and 1000 should map to finite pixels.
        assert!(broken_scale_map(0.0, &result).is_some());
        assert!(broken_scale_map(1000.0, &result).is_some());
    }

    #[test]
    fn broken_scale_map_value_below_domain_is_none() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        assert!(broken_scale_map(-1.0, &result).is_none());
    }

    #[test]
    fn broken_scale_map_domain_endpoints_map_to_pixel_range() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        let px_min = broken_scale_map(0.0, &result).unwrap();
        let px_max = broken_scale_map(1000.0, &result).unwrap();
        // px_min should be close to pixel_range start, px_max close to end.
        let tolerance = 1.0;
        assert!(
            (px_min - 50.0).abs() < tolerance,
            "data lo should map near pixel_range start; got {px_min}"
        );
        assert!(
            (px_max - 350.0).abs() < tolerance,
            "data hi should map near pixel_range end; got {px_max}"
        );
    }

    #[test]
    fn broken_scale_map_no_gaps_behaves_like_linear() {
        let result = apply_break_to_scale((0.0, 100.0), &[], (50.0, 350.0), 12.0);
        // Single segment, linear mapping.
        let px_mid = broken_scale_map(50.0, &result).unwrap();
        // Should be at the midpoint of [50, 350] = 200.
        assert!((px_mid - 200.0).abs() < 0.01, "midpoint should be ~200; got {px_mid}");
    }

    #[test]
    fn broken_scale_map_segment_monotone() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        let p0 = broken_scale_map(0.0, &result).unwrap();
        let p100 = broken_scale_map(100.0, &result).unwrap();
        let p200 = broken_scale_map(200.0, &result).unwrap();
        // Within first segment [0, 200], values should be monotonically increasing.
        assert!(p0 < p100, "p0={p0} should be < p100={p100}");
        assert!(p100 < p200, "p100={p100} should be < p200={p200}");

        let p800 = broken_scale_map(800.0, &result).unwrap();
        let p900 = broken_scale_map(900.0, &result).unwrap();
        let p1000 = broken_scale_map(1000.0, &result).unwrap();
        // Within second segment [800, 1000], values should be monotonically increasing.
        assert!(p800 < p900, "p800={p800} should be < p900={p900}");
        assert!(p900 < p1000, "p900={p900} should be < p1000={p1000}");
        // Second segment starts after the break (past first segment end).
        assert!(p800 > p200, "second segment should start after first ends");
    }

    // ── apply_break_to_scale tests ───────────────────────────────────────────

    #[test]
    fn apply_break_produces_two_segments_for_one_gap() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.indicator_pixels.len(), 1);
    }

    #[test]
    fn apply_break_no_gaps_produces_one_segment() {
        let result = apply_break_to_scale((0.0, 100.0), &[], (50.0, 350.0), 12.0);
        assert_eq!(result.segments.len(), 1);
        assert!(result.indicator_pixels.is_empty());
    }

    // ── R1 port (bug_hunt_draw.rs): gap clipping edge cases ───────────────────

    #[test]
    fn gap_entirely_outside_domain_is_ignored() {
        let result = apply_break_to_scale((0.0, 100.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        assert_eq!(result.segments.len(), 1, "out-of-domain gap must be ignored");
        assert!(result.indicator_pixels.is_empty());
    }

    #[test]
    fn gap_partially_overlapping_domain_is_clipped() {
        // Gap [80, 200] clips to [80, 100] against domain [0, 100].
        let result = apply_break_to_scale((0.0, 100.0), &[[80.0, 200.0]], (50.0, 350.0), 12.0);
        assert_eq!(result.segments.len(), 1, "clipped gap leaves a single retained segment");
        assert_eq!(result.segments[0].data_lo, 0.0);
        assert_eq!(result.segments[0].data_hi, 80.0);
    }

    #[test]
    fn multiple_gaps_produce_matching_segment_and_indicator_counts() {
        let result = apply_break_to_scale(
            (0.0, 100.0), &[[20.0, 40.0], [60.0, 80.0]], (50.0, 350.0), 12.0,
        );
        assert_eq!(result.segments.len(), 3, "two gaps -> three retained segments");
        assert_eq!(result.indicator_pixels.len(), 2, "two gaps -> two indicators");
    }

    #[test]
    fn gap_covering_entire_domain_leaves_no_segments() {
        let result = apply_break_to_scale((0.0, 100.0), &[[0.0, 100.0]], (50.0, 350.0), 12.0);
        assert!(result.segments.is_empty(), "gap spanning the whole domain leaves nothing retained");
    }

    #[test]
    fn degenerate_domain_with_gap_produces_single_segment() {
        // lo == hi: the gap clips to an empty range and is discarded.
        let result = apply_break_to_scale((50.0, 50.0), &[[40.0, 60.0]], (50.0, 350.0), 12.0);
        assert_eq!(result.segments.len(), 1);
    }

    #[test]
    fn zero_break_size_allocates_all_pixels_to_data() {
        let result = apply_break_to_scale((0.0, 100.0), &[[40.0, 60.0]], (50.0, 350.0), 0.0);
        let total_seg_px: f64 = result.segments.iter().map(|s| (s.pixel_hi - s.pixel_lo).abs()).sum();
        assert!((total_seg_px - 300.0).abs() < 1.0, "zero break_size reserves no indicator space");
    }

    #[test]
    fn nan_gap_bounds_resolve_to_a_domain_covering_gap() {
        // Documents a real, pinned edge case rather than an assumed-good one:
        // `f64::max`/`f64::min` return the *non-NaN* argument (Rust docs), so
        // `NaN.max(d_lo) == d_lo` and `NaN.min(d_hi) == d_hi` — a NaN gap bound
        // resolves to `[d_lo, d_hi]`, which clips to a domain-covering gap and
        // silently erases all data, exactly like an explicit `[d_lo, d_hi]` gap.
        // Tracked as BRK-1 in design-docs/superpowers/followups/
        // 2026-05-15-code-archaeology.md (a NaN gap bound should be rejected
        // loudly, not render a blank chart); this test pins the current
        // behavior until that lands.
        let result = apply_break_to_scale((0.0, 100.0), &[[f64::NAN, f64::NAN]], (50.0, 350.0), 12.0);
        assert!(
            result.segments.is_empty(),
            "NaN gap bounds resolve to a domain-covering gap under IEEE max/min semantics"
        );
    }

    #[test]
    fn broken_scale_map_nan_value_returns_none() {
        let result = apply_break_to_scale((0.0, 100.0), &[[40.0, 60.0]], (50.0, 350.0), 12.0);
        assert!(broken_scale_map(f64::NAN, &result).is_none());
    }

    // ── build_break_indicators tests ─────────────────────────────────────────

    #[test]
    fn build_break_indicators_slash_returns_lines() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        let theme = ThemeInputs::default();
        let nodes = build_break_indicators(&result, &plot(), "y", 12.0, "slash", &theme);
        assert!(!nodes.is_empty(), "slash style should produce nodes");
        let has_line = nodes.iter().any(|n| matches!(n, SceneNode::Line { .. }));
        assert!(has_line, "slash style should produce line nodes");
    }

    #[test]
    fn build_break_indicators_gap_style_returns_empty() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        let theme = ThemeInputs::default();
        let nodes = build_break_indicators(&result, &plot(), "y", 12.0, "gap", &theme);
        assert!(nodes.is_empty(), "gap style should produce no indicator nodes");
    }

    #[test]
    fn build_break_indicators_zigzag_returns_paths() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        let theme = ThemeInputs::default();
        let nodes = build_break_indicators(&result, &plot(), "y", 12.0, "zigzag", &theme);
        let has_path = nodes.iter().any(|n| matches!(n, SceneNode::Path { .. }));
        assert!(has_path, "zigzag style should produce path nodes");
    }

    #[test]
    fn build_break_indicators_wave_returns_paths() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[200.0, 800.0]], (50.0, 350.0), 12.0);
        let theme = ThemeInputs::default();
        let nodes = build_break_indicators(&result, &plot(), "y", 12.0, "wave", &theme);
        let has_path = nodes.iter().any(|n| matches!(n, SceneNode::Path { .. }));
        assert!(has_path, "wave style should produce path nodes");
    }

    #[test]
    fn build_break_indicators_x_axis() {
        let result = apply_break_to_scale((0.0, 100.0), &[[20.0, 80.0]], (50.0, 350.0), 12.0);
        let theme = ThemeInputs::default();
        let nodes = build_break_indicators(&result, &plot(), "x", 12.0, "slash", &theme);
        assert!(!nodes.is_empty(), "x-axis slash should produce nodes");
    }

    // ── Gap normalization (spec §4 H4) ───────────────────────────────────────

    #[test]
    fn nested_gap_cannot_re_admit_the_enclosing_gap_s_range() {
        // [200,300] sits wholly inside [100,900]. The removed range is
        // [100,900]; the pre-normalization walk took the LAST gap's upper
        // bound as its cursor, so it emitted a phantom segment [300,900] of
        // data the enclosing gap had already removed.
        let result = apply_break_to_scale(
            (0.0, 1000.0),
            &[[100.0, 900.0], [200.0, 300.0]],
            (0.0, 400.0),
            12.0,
        );
        assert_eq!(
            result.segments.len(),
            2,
            "nested gaps remove ONE range: segments {:?}",
            result.segments
        );
        assert_eq!(result.indicator_pixels.len(), 1, "one removed range, one indicator");
        assert!(broken_scale_map(500.0, &result).is_none(), "500 is inside the merged gap");
        assert!(broken_scale_map(250.0, &result).is_none(), "250 is inside the merged gap");
        assert_eq!((result.segments[0].data_lo, result.segments[0].data_hi), (0.0, 100.0));
        assert_eq!((result.segments[1].data_lo, result.segments[1].data_hi), (900.0, 1000.0));
    }

    #[test]
    fn overlapping_gaps_merge_into_one_removed_range() {
        let result = apply_break_to_scale(
            (0.0, 100.0),
            &[[20.0, 50.0], [40.0, 70.0]],
            (0.0, 400.0),
            12.0,
        );
        assert_eq!(result.segments.len(), 2);
        assert_eq!((result.segments[0].data_lo, result.segments[0].data_hi), (0.0, 20.0));
        assert_eq!((result.segments[1].data_lo, result.segments[1].data_hi), (70.0, 100.0));
    }

    #[test]
    fn unsorted_gaps_are_ordered_before_segments_are_built() {
        let sorted = apply_break_to_scale(
            (0.0, 100.0),
            &[[20.0, 30.0], [60.0, 70.0]],
            (0.0, 400.0),
            12.0,
        );
        let reversed = apply_break_to_scale(
            (0.0, 100.0),
            &[[60.0, 70.0], [20.0, 30.0]],
            (0.0, 400.0),
            12.0,
        );
        let ranges = |r: &BreakResult| -> Vec<(f64, f64)> {
            r.segments.iter().map(|s| (s.data_lo, s.data_hi)).collect()
        };
        assert_eq!(ranges(&sorted), ranges(&reversed));
        assert_eq!(ranges(&sorted), vec![(0.0, 20.0), (30.0, 60.0), (70.0, 100.0)]);
    }

    #[test]
    fn adjacent_gaps_reserve_one_indicator_s_pixels() {
        // [20,40] and [40,60] remove one contiguous range and draw one
        // indicator, so exactly one break_size may be withheld from the data
        // segments — the pre-merge code reserved two and short-changed them.
        let result = apply_break_to_scale(
            (0.0, 100.0),
            &[[20.0, 40.0], [40.0, 60.0]],
            (0.0, 400.0),
            12.0,
        );
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.indicator_pixels.len(), 1);
        let total: f64 = result.segments.iter().map(|s| (s.pixel_hi - s.pixel_lo).abs()).sum();
        assert!(
            (total - 388.0).abs() < 1e-6,
            "two segments must share 400 - 12 pixels; got {total}"
        );
    }

    // ── BreakRemap: single-value semantics (ported from scene_build) ─────────

    fn one_segment_break(d_lo: f64, d_hi: f64, px_lo: f64, px_hi: f64) -> BreakResult {
        apply_break_to_scale((d_lo, d_hi), &[], (px_lo, px_hi), 12.0)
    }

    fn two_segment_break() -> BreakResult {
        // Domain [0,100] with a gap [40,60] -> two retained segments.
        apply_break_to_scale((0.0, 100.0), &[[40.0, 60.0]], (50.0, 350.0), 12.0)
    }

    fn remap<'a>(axis: &'a str, br: &'a BreakResult) -> BreakRemap<'a> {
        BreakRemap::new(axis, (0.0, 100.0), (50.0, 350.0), br)
    }

    fn fill_stroke() -> FillStroke {
        FillStroke {
            fill: None,
            stroke: None,
            stroke_width: 1.0,
            opacity: 1.0,
            stroke_dash: None,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
            angle: 0.0,
        }
    }

    #[test]
    fn coord_identity_maps_pixel_to_pixel() {
        let br = one_segment_break(0.0, 100.0, 50.0, 350.0);
        let px = remap("x", &br).coord(200.0).unwrap();
        assert!((px - 200.0).abs() < 1e-6, "single-segment remap must be identity; got {px}");
    }

    #[test]
    fn coord_pixel_in_gap_returns_none() {
        let br = two_segment_break();
        // Pixel 200 maps back to data value 50, which falls inside the gap [40, 60].
        assert!(
            remap("x", &br).coord(200.0).is_none(),
            "a pixel that reverse-maps into the gap must return None"
        );
    }

    #[test]
    fn coord_nan_pixel_returns_none() {
        let br = one_segment_break(0.0, 100.0, 50.0, 350.0);
        assert!(remap("x", &br).coord(f64::NAN).is_none());
    }

    #[test]
    fn remap_node_circle_hides_at_break_hidden_when_in_gap() {
        let br = two_segment_break();
        let mut node = SceneNode::Circle { cx: 200.0, cy: 100.0, r: 4.0, style: fill_stroke() };
        remap("x", &br).remap_node(&mut node);
        let SceneNode::Circle { cx, .. } = node else { panic!("expected Circle") };
        assert_eq!(cx, BREAK_HIDDEN, "a mark inside the break gap must be hidden off-screen");
    }

    /// Discriminating fixture (quality-review kill-switch finding, cycle 2):
    /// `one_segment_break()` has no gaps, so remapping through it is the
    /// identity — an assertion of "unchanged" is satisfied whether or not the
    /// real remap dispatch runs at all, which is exactly how the reviewer's
    /// mutation (deleting the x-axis remap call) survived undetected the first
    /// time. `two_segment_break()`'s gap `[40,60]` makes `cx: 110.0` reverse-map
    /// to data value 20.0 (inside the retained segment `[0,40]`), which the
    /// broken scale compresses to pixel 122.0.
    #[test]
    fn remap_node_circle_moves_to_compressed_pixel_across_a_gap() {
        let br = two_segment_break();
        let mut node = SceneNode::Circle { cx: 110.0, cy: 100.0, r: 4.0, style: fill_stroke() };
        remap("x", &br).remap_node(&mut node);
        let SceneNode::Circle { cx, .. } = node else { panic!("expected Circle") };
        assert!(
            (cx - 122.0).abs() < 1e-6,
            "expected cx compressed to 122.0 across the break gap; got {cx}"
        );
    }

    #[test]
    fn remap_path_cmd_hlineto_only_remapped_on_x_axis() {
        // Same 110.0 -> 122.0 discriminating fixture as the Circle test above.
        let br = two_segment_break();
        let mut on_x = SceneNode::Path {
            commands: vec![PathCmd::HLineTo { x: 110.0 }],
            style: fill_stroke(),
            closed: false,
        };
        remap("x", &br).remap_node(&mut on_x);
        let SceneNode::Path { commands, .. } = &on_x else { panic!() };
        let PathCmd::HLineTo { x } = commands[0] else { panic!() };
        assert!(
            (x - 122.0).abs() < 1e-6,
            "HLineTo.x must compress to 122.0 through the broken scale when axis == x; got {x}"
        );

        let mut on_y = SceneNode::Path {
            commands: vec![PathCmd::HLineTo { x: 110.0 }],
            style: fill_stroke(),
            closed: false,
        };
        remap("y", &br).remap_node(&mut on_y);
        let SceneNode::Path { commands, .. } = &on_y else { panic!() };
        let PathCmd::HLineTo { x } = commands[0] else { panic!() };
        assert_eq!(x, 110.0, "HLineTo.x must NOT be touched when axis == y");
    }

    #[test]
    fn remap_path_cmd_vlineto_only_remapped_on_y_axis() {
        let br = two_segment_break();
        let mut on_y = SceneNode::Path {
            commands: vec![PathCmd::VLineTo { y: 110.0 }],
            style: fill_stroke(),
            closed: false,
        };
        remap("y", &br).remap_node(&mut on_y);
        let SceneNode::Path { commands, .. } = &on_y else { panic!() };
        let PathCmd::VLineTo { y } = commands[0] else { panic!() };
        assert!(
            (y - 122.0).abs() < 1e-6,
            "VLineTo.y must compress to 122.0 through the broken scale when axis == y; got {y}"
        );

        let mut on_x = SceneNode::Path {
            commands: vec![PathCmd::VLineTo { y: 110.0 }],
            style: fill_stroke(),
            closed: false,
        };
        remap("x", &br).remap_node(&mut on_x);
        let SceneNode::Path { commands, .. } = &on_x else { panic!() };
        let PathCmd::VLineTo { y } = commands[0] else { panic!() };
        assert_eq!(y, 110.0, "VLineTo.y must NOT be touched when axis == x");
    }

    // ── Extent clipping (spec §4 H4, §9 edge set) ────────────────────────────

    /// The guide's worked example, in pixels: domain 0–1000 over a 400px
    /// y-range (pixel 400 = data 0, pixel 0 = data 1000) with the gap
    /// (150, 850). Numbers below are hand-computed from the segment
    /// allocation, not read back out of `apply_break_to_scale`.
    fn guide_break() -> BreakResult {
        apply_break_to_scale((0.0, 1000.0), &[[150.0, 850.0]], (400.0, 0.0), 12.0)
    }

    fn guide_remap(br: &BreakResult) -> BreakRemap<'_> {
        BreakRemap::new("y", (0.0, 1000.0), (400.0, 0.0), br)
    }

    #[test]
    fn segment_allocation_matches_hand_computed_fractions() {
        let br = guide_break();
        // Retained span = 150 + 150 = 300, both halves equal, so each segment
        // gets (400 - 12) / 2 = 194 pixels.
        assert_eq!(br.segments.len(), 2);
        assert!((br.segments[0].pixel_lo - 400.0).abs() < 1e-9);
        assert!((br.segments[0].pixel_hi - 206.0).abs() < 1e-9);
        assert!((br.segments[1].pixel_lo - 194.0).abs() < 1e-9);
        assert!((br.segments[1].pixel_hi - 0.0).abs() < 1e-9);
    }

    #[test]
    fn bar_crossing_a_gap_clips_into_two_segments() {
        let br = guide_break();
        // A bar from data 0 (pixel 400) to data 900 (pixel 40).
        let pieces = guide_remap(&br).clip_extent(400.0, 40.0);
        assert_eq!(pieces.len(), 2, "0->900 across gap (150,850) is two rects; got {pieces:?}");
        // Lower piece: data 0..150 -> pixels 400..206.
        assert!((pieces[0].0 - 206.0).abs() < 1e-9, "{pieces:?}");
        assert!((pieces[0].1 - 400.0).abs() < 1e-9, "{pieces:?}");
        // Upper piece: data 850..900 -> pixels 194..(194 - 50/150*194).
        assert!((pieces[1].0 - (194.0 - 194.0 / 3.0)).abs() < 1e-9, "{pieces:?}");
        assert!((pieces[1].1 - 194.0).abs() < 1e-9, "{pieces:?}");
    }

    #[test]
    fn bar_ending_inside_a_gap_keeps_its_visible_portion() {
        let br = guide_break();
        // A bar from data 0 to data 400 — its top sits inside the gap.
        let pieces = guide_remap(&br).clip_extent(400.0, 400.0 - 0.4 * 400.0);
        assert_eq!(pieces.len(), 1, "only the below-gap portion is visible; got {pieces:?}");
        assert!((pieces[0].0 - 206.0).abs() < 1e-9, "clipped at the gap edge; got {pieces:?}");
        assert!((pieces[0].1 - 400.0).abs() < 1e-9);
    }

    #[test]
    fn extent_entirely_inside_a_gap_is_dropped_not_flattened() {
        let br = guide_break();
        // Data 300..500, wholly inside the gap.
        let pieces = guide_remap(&br).clip_extent(400.0 - 0.3 * 400.0, 400.0 - 0.5 * 400.0);
        assert!(pieces.is_empty(), "a wholly-hidden extent emits nothing; got {pieces:?}");
    }

    #[test]
    fn multi_gap_extent_yields_one_piece_per_visible_segment() {
        // Two gaps -> three segments; a full-domain extent crosses all three.
        let br = apply_break_to_scale(
            (0.0, 1000.0),
            &[[200.0, 400.0], [600.0, 800.0]],
            (400.0, 0.0),
            12.0,
        );
        let remap = BreakRemap::new("y", (0.0, 1000.0), (400.0, 0.0), &br);
        assert_eq!(remap.clip_extent(400.0, 0.0).len(), 3);
    }

    #[test]
    fn zero_length_extent_survives_as_a_zero_length_piece() {
        let br = guide_break();
        // A zero-height bar at data 0 (a zero-valued category) stays at the
        // baseline instead of disappearing.
        let pieces = guide_remap(&br).clip_extent(400.0, 400.0);
        assert_eq!(pieces.len(), 1);
        assert!((pieces[0].0 - 400.0).abs() < 1e-9);
        assert!((pieces[0].1 - 400.0).abs() < 1e-9);
    }

    #[test]
    fn clip_nodes_splits_one_rect_into_two_and_records_both_sources() {
        let br = guide_break();
        let node = SceneNode::Rect {
            x: 10.0,
            y: 40.0,
            w: 20.0,
            h: 360.0,
            style: fill_stroke(),
            corner_radius: 0.0,
        };
        let clipped = guide_remap(&br).clip_nodes(vec![node]);
        assert_eq!(clipped.nodes.len(), 2, "crossing bar splits: {:?}", clipped.nodes);
        assert!(clipped.is_reindexed());
        assert_eq!(
            clipped.realign(Some(vec![7usize])),
            Some(vec![7usize, 7usize]),
            "both pieces must point back at the row that produced them"
        );
        for node in &clipped.nodes {
            let SceneNode::Rect { x, w, h, .. } = node else { panic!("expected Rect") };
            assert_eq!((*x, *w), (10.0, 20.0), "the cross-axis extent is untouched");
            assert!(*h > 0.5, "no zero-height artifacts; got {h}");
        }
    }

    #[test]
    fn clip_nodes_drops_a_rect_whose_extent_is_wholly_in_a_gap() {
        let br = guide_break();
        let node = SceneNode::Rect {
            x: 10.0,
            y: 200.0,
            w: 20.0,
            h: 80.0,
            style: fill_stroke(),
            corner_radius: 0.0,
        };
        let clipped = guide_remap(&br).clip_nodes(vec![node]);
        assert!(clipped.nodes.is_empty(), "a hidden bar leaves no node: {:?}", clipped.nodes);
        assert!(clipped.is_reindexed());
        assert_eq!(clipped.realign(Some(vec![3usize])), Some(vec![]));
    }

    #[test]
    fn clip_nodes_keeps_point_geometry_one_to_one() {
        let br = guide_break();
        let nodes = vec![
            SceneNode::Circle { cx: 10.0, cy: 40.0, r: 3.0, style: fill_stroke() },
            SceneNode::Circle { cx: 20.0, cy: 200.0, r: 3.0, style: fill_stroke() },
        ];
        let clipped = guide_remap(&br).clip_nodes(nodes);
        assert_eq!(clipped.nodes.len(), 2, "single-value marks are hidden, never removed");
        assert!(!clipped.is_reindexed(), "an unclipped batch keeps its sidecars untouched");
        let SceneNode::Circle { cy, .. } = clipped.nodes[1] else { panic!() };
        assert_eq!(cy, BREAK_HIDDEN, "the in-gap point is hidden off-screen");
    }

    #[test]
    fn closed_area_path_clips_per_segment_without_hidden_spikes() {
        let br = guide_break();
        // A closed area: top edge at data 900 (pixel 40), baseline at data 0
        // (pixel 400), spanning x 0..100.
        let node = SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 0.0, y: 40.0 },
                PathCmd::LineTo { x: 100.0, y: 40.0 },
                PathCmd::LineTo { x: 100.0, y: 400.0 },
                PathCmd::LineTo { x: 0.0, y: 400.0 },
                PathCmd::Close,
            ],
            style: fill_stroke(),
            closed: true,
        };
        let clipped = guide_remap(&br).clip_nodes(vec![node]);
        assert_eq!(clipped.nodes.len(), 2, "the area clips into two bands");
        for node in &clipped.nodes {
            let SceneNode::Path { commands, closed, .. } = node else { panic!("expected Path") };
            assert!(closed, "clipped area pieces stay closed");
            for cmd in commands {
                if let PathCmd::MoveTo { y, .. } | PathCmd::LineTo { y, .. } = cmd {
                    assert!(
                        *y >= -1.0 && *y <= 401.0,
                        "no BREAK_HIDDEN spike in a clipped area; got {y}"
                    );
                }
            }
        }
    }

    #[test]
    fn curved_path_falls_back_to_per_vertex_remap() {
        let br = guide_break();
        let node = SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 0.0, y: 40.0 },
                PathCmd::CubicTo { c1x: 10.0, c1y: 40.0, c2x: 20.0, c2y: 400.0, x: 30.0, y: 400.0 },
                PathCmd::Close,
            ],
            style: fill_stroke(),
            closed: true,
        };
        let clipped = guide_remap(&br).clip_nodes(vec![node]);
        assert_eq!(clipped.nodes.len(), 1, "a curved path is not polygon-clipped");
        assert!(!clipped.is_reindexed());
    }

    #[test]
    fn group_children_are_clipped_without_changing_the_batch_s_node_count() {
        let br = guide_break();
        let group = SceneNode::Group {
            attrs: vec![("k".to_string(), "v".to_string())],
            children: vec![SceneNode::Rect {
                x: 10.0,
                y: 40.0,
                w: 20.0,
                h: 360.0,
                style: fill_stroke(),
                corner_radius: 0.0,
            }],
        };
        let clipped = guide_remap(&br).clip_nodes(vec![group]);
        assert_eq!(clipped.nodes.len(), 1, "a group stays one node");
        assert!(!clipped.is_reindexed());
        let SceneNode::Group { children, .. } = &clipped.nodes[0] else { panic!() };
        assert_eq!(children.len(), 2, "its extent child still splits");
    }

    // ── Placement model per batch kind (quality review F1/F4) ────────────────

    /// All 14 `MarkBatchKind` variants, so no kind's placement is decided by
    /// omission — the cycle-2 miss was exactly the three kinds this list did
    /// not name (`Polygon`, `Arc`, `Image`).
    #[test]
    fn placement_model_is_assigned_for_every_mark_kind() {
        use BreakPlacement::*;
        let table = [
            (MarkBatchKind::Bar, ClipExtent),
            (MarkBatchKind::Area, ClipExtent),
            (MarkBatchKind::Rect, ClipExtent),
            (MarkBatchKind::Ribbon, ClipExtent),
            (MarkBatchKind::Polygon, ClipExtent),
            (MarkBatchKind::Point, MoveGlyph),
            (MarkBatchKind::Tick, MoveGlyph),
            (MarkBatchKind::Text, MoveGlyph),
            (MarkBatchKind::Image, MoveGlyph),
            (MarkBatchKind::Label, MoveGlyphAndConnector),
            (MarkBatchKind::Line, MoveVertices),
            (MarkBatchKind::Rule, MoveVertices),
            (MarkBatchKind::Segment, MoveVertices),
            (MarkBatchKind::Arc, MoveVertices),
        ];
        assert_eq!(table.len(), 14, "every MarkBatchKind variant must be listed");
        for (kind, want) in table {
            assert_eq!(BreakPlacement::for_kind(kind), want, "kind {kind:?}");
        }
    }

    /// A polygon mark (`mark_violin`, `mark_geoshape`) is CLIPPED, not placed
    /// vertex-wise: its rings are filled, so a `BREAK_HIDDEN` vertex would
    /// paint a column across the break. Before this, `remap_node`'s catch-all
    /// left `SceneNode::Polygon` untouched entirely and a violin rendered
    /// against the unbroken scale while its own ticks compressed.
    #[test]
    fn polygon_rings_are_clipped_to_the_visible_segments() {
        let br = guide_break();
        // One ring spanning data 0..900 (pixels 400..40) at x 0..100.
        let mut polygon = SceneNode::Polygon {
            rings: vec![vec![[0.0, 400.0], [100.0, 400.0], [100.0, 40.0], [0.0, 40.0]]],
            style: fill_stroke(),
        };
        guide_remap(&br).remap_node(&mut polygon);
        let SceneNode::Polygon { rings, .. } = &polygon else { panic!() };
        assert_eq!(rings.len(), 2, "the ring is cut into one piece per segment; got {rings:?}");
        for ring in rings {
            for p in ring {
                assert!(
                    p[1] >= -1.0 && p[1] <= 401.0,
                    "no BREAK_HIDDEN vertex may survive a ring clip; got {p:?}"
                );
            }
        }
        // Lower piece sits below the break, upper piece above it.
        let lows: Vec<f64> = rings[0].iter().map(|p| p[1]).collect();
        let highs: Vec<f64> = rings[1].iter().map(|p| p[1]).collect();
        assert!(
            lows.iter().cloned().fold(f64::INFINITY, f64::min)
                > highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            "the two pieces are separated by the break; {lows:?} vs {highs:?}"
        );
    }

    #[test]
    fn polygon_wholly_inside_a_gap_keeps_its_node_and_loses_its_rings() {
        let br = guide_break();
        let mut polygon = SceneNode::Polygon {
            rings: vec![vec![[0.0, 280.0], [100.0, 280.0], [100.0, 200.0], [0.0, 200.0]]],
            style: fill_stroke(),
        };
        let clipped = guide_remap(&br).clip_nodes(vec![polygon.clone()]);
        assert_eq!(clipped.nodes.len(), 1, "a polygon stays one node — one row, one entry");
        assert!(!clipped.is_reindexed(), "so its sidecars are never rebuilt");
        guide_remap(&br).remap_node(&mut polygon);
        let SceneNode::Polygon { rings, .. } = &polygon else { panic!() };
        assert!(rings.is_empty(), "nothing visible is left to draw; got {rings:?}");
    }

    fn thin_stroke() -> ferrum_scene::StrokeStyle {
        ferrum_scene::StrokeStyle {
            color: ferrum_scene::Color::rgb(0, 0, 0),
            width: 0.75,
            opacity: 1.0,
            dash: None,
            stroke_cap: None,
            stroke_join: None,
            stroke_opacity: 1.0,
        }
    }

    /// A label's leader line runs from its data point to the label, so its two
    /// ends are two different positions. Moving it rigidly (as a glyph) pulled
    /// the data end off the point it exists to touch.
    #[test]
    fn label_leader_keeps_each_end_on_its_own_position() {
        let br = guide_break();
        let remap = guide_remap(&br);
        // Data point at pixel 360 (data 100), label end 20px above it.
        let mut leader = SceneNode::Line {
            x1: 10.0,
            y1: 360.0,
            x2: 10.0,
            y2: 340.0,
            style: thin_stroke(),
        };
        remap.remap_label_node(&mut leader);
        let SceneNode::Line { y1, y2, .. } = leader else { panic!() };
        assert_eq!(
            y1,
            remap.coord(360.0).unwrap(),
            "the data end lands exactly where the data point lands"
        );
        assert_eq!(y2, remap.coord(340.0).unwrap(), "the label end lands where the label lands");
    }

    /// The other producer of `SceneNode::Line` in a glyph batch is a tick's
    /// dash, whose length is drawn size, not data. Placing ITS two ends
    /// independently collapses every rug dash lying along the broken axis —
    /// which is why the connector rule is scoped by batch kind, not by node
    /// kind.
    #[test]
    fn tick_dash_along_the_broken_axis_keeps_its_length() {
        let br = guide_break();
        let remap = guide_remap(&br);
        // An 8px rug dash sitting on the plot floor: both ends reverse-map to
        // (or below) the domain minimum, so per-endpoint placement would map
        // them to the same pixel and erase the dash.
        let mut dash = SceneNode::Line {
            x1: 10.0,
            y1: 400.0,
            x2: 10.0,
            y2: 392.0,
            style: thin_stroke(),
        };
        remap.remap_glyph_node(&mut dash);
        let SceneNode::Line { y1, y2, .. } = dash else { panic!() };
        assert!(
            ((y1 - y2).abs() - 8.0).abs() < 1e-9,
            "a rug dash keeps its 8px drawn length through the break; got {}",
            (y1 - y2).abs()
        );
        // ... while the same node under the connector model is re-LENGTHED by
        // the segment's scale factor, which is exactly the semantic difference
        // the batch kind carries. (A dash drawn below the scale's data floor —
        // where a rug sits — has both ends clamp to the same value there and
        // vanishes outright; this fixture shows the general case.)
        let mut connector = SceneNode::Line {
            x1: 10.0,
            y1: 400.0,
            x2: 10.0,
            y2: 392.0,
            style: thin_stroke(),
        };
        remap.remap_label_node(&mut connector);
        let SceneNode::Line { y1, y2, .. } = connector else { panic!() };
        assert!(
            ((y1 - y2).abs() - 8.0).abs() > 1.0,
            "per-endpoint placement rescales the drawn length; got {}",
            (y1 - y2).abs()
        );
    }

    /// The rug's own shape: a dash drawn on the plot floor, below the scale's
    /// data range. Both ends reverse-map past the domain minimum and clamp to
    /// it, so per-endpoint placement maps them to one pixel and the dash
    /// disappears — 200 of them did, on a `displot(rug=True)` under a break.
    #[test]
    fn rug_dash_below_the_data_floor_survives_the_break() {
        let br = guide_break();
        // Pixel range stops at 400 (data 0); the dash hangs below it.
        let remap = guide_remap(&br);
        let mut dash = SceneNode::Line {
            x1: 10.0,
            y1: 412.0,
            x2: 10.0,
            y2: 404.0,
            style: thin_stroke(),
        };
        let mut collapsed = dash.clone();
        remap.remap_glyph_node(&mut dash);
        remap.remap_node(&mut collapsed);
        let SceneNode::Line { y1, y2, .. } = dash else { panic!() };
        assert!(
            ((y1 - y2).abs() - 8.0).abs() < 1e-9,
            "the glyph model keeps the dash drawable; got {}",
            (y1 - y2).abs()
        );
        let SceneNode::Line { y1: c1, y2: c2, .. } = collapsed else { panic!() };
        assert_eq!(c1, c2, "per-endpoint placement is what erased it (both ends clamp)");
    }

    #[test]
    fn measuring_a_glyph_rect_does_not_perturb_its_size() {
        let br = one_segment_break(0.0, 100.0, 50.0, 350.0);
        let remap = remap("y", &br);
        let mut rect = SceneNode::Rect {
            x: 1.0,
            y: 100.3,
            w: 5.416,
            h: 5.416,
            style: fill_stroke(),
            corner_radius: 0.0,
        };
        let before = rect.clone();
        // `along_extent` is a measurement: it must leave the node alone.
        let extent = remap.along_extent(&mut rect);
        assert_eq!(extent, Some((100.3, 105.716)));
        assert_eq!(rect, before, "a measuring pass must not write to the node");
    }

    /// A glyph is a fixed-size symbol: the break moves it, never resizes it.
    /// Extent-clipping one (which is what a `shape="square"`/`"diamond"` point
    /// used to get) cut it flat at the gap edge, or deleted it outright.
    #[test]
    fn glyph_keeps_its_size_when_the_break_moves_it() {
        let br = guide_break();
        let remap = guide_remap(&br);
        // A 6px square glyph anchored at data 100 (pixel 360).
        let mut square = SceneNode::Rect {
            x: 10.0,
            y: 357.0,
            w: 6.0,
            h: 6.0,
            style: fill_stroke(),
            corner_radius: 0.0,
        };
        remap.remap_glyph_node(&mut square);
        let SceneNode::Rect { y, h, w, .. } = square else { panic!() };
        assert!((h - 6.0).abs() < 1e-9, "a moved glyph keeps its height; got {h}");
        assert!((w - 6.0).abs() < 1e-9);
        // Anchor 360 -> data 100 -> segment 1 pixel 400 - (100/150)*194.
        let want_anchor = 400.0 - 194.0 * 100.0 / 150.0;
        assert!((y + 3.0 - want_anchor).abs() < 1e-6, "glyph centre lands on the anchor; got {y}");
    }

    #[test]
    fn glyph_inside_a_gap_hides_whole_and_is_never_dropped() {
        let br = guide_break();
        let remap = guide_remap(&br);
        // A diamond centred on data 500 — inside the gap.
        let mut diamond = SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 10.0, y: 196.0 },
                PathCmd::LineTo { x: 14.0, y: 200.0 },
                PathCmd::LineTo { x: 10.0, y: 204.0 },
                PathCmd::LineTo { x: 6.0, y: 200.0 },
                PathCmd::Close,
            ],
            style: fill_stroke(),
            closed: true,
        };
        remap.remap_glyph_node(&mut diamond);
        let SceneNode::Path { commands, .. } = &diamond else { panic!() };
        let ys: Vec<f64> = commands
            .iter()
            .filter_map(|c| match c {
                PathCmd::MoveTo { y, .. } | PathCmd::LineTo { y, .. } => Some(*y),
                _ => None,
            })
            .collect();
        assert_eq!(ys.len(), 4, "every vertex survives — the glyph is hidden, not clipped");
        assert!(
            ys.iter().all(|y| *y < -1000.0),
            "the WHOLE glyph goes off-screen, so no edge can spike into the plot; got {ys:?}"
        );
        // ... and its shape is intact off-screen (no zero-size collapse).
        let (lo, hi) = (
            ys.iter().cloned().fold(f64::INFINITY, f64::min),
            ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        assert!((hi - lo - 8.0).abs() < 1e-9, "hidden glyph keeps its extent; got {}", hi - lo);
    }

    #[test]
    fn glyph_circle_hide_and_move_are_byte_identical_to_the_plain_assignment() {
        // The glyph model must not perturb the default (circle) shape by so
        // much as an ULP: it is the byte-stability anchor for every existing
        // point-mark break golden.
        let br = guide_break();
        let remap = guide_remap(&br);
        for cy in [40.0_f64, 200.0, 360.0, 400.0] {
            let mut glyph = SceneNode::Circle { cx: 5.0, cy, r: 3.0, style: fill_stroke() };
            let mut plain = glyph.clone();
            remap.remap_glyph_node(&mut glyph);
            remap.remap_node(&mut plain);
            assert_eq!(glyph, plain, "glyph and plain remap must agree exactly for cy={cy}");
        }
    }

    #[test]
    fn polyline_vertices_are_placed_on_the_broken_scale() {
        // `mark_line` emits a Polyline; before this arm existed it was the one
        // mark kind a break left entirely alone, drawing the line against the
        // unbroken scale while its own ticks compressed.
        let br = guide_break();
        let mut line = SceneNode::Polyline {
            points: vec![(0.0, 400.0), (10.0, 200.0), (20.0, 40.0)],
            style: ferrum_scene::StrokeStyle {
                color: ferrum_scene::Color::rgb(0, 0, 0),
                width: 1.0,
                opacity: 1.0,
                dash: None,
                stroke_cap: None,
                stroke_join: None,
                stroke_opacity: 1.0,
            },
        };
        guide_remap(&br).remap_node(&mut line);
        let SceneNode::Polyline { points, .. } = &line else { panic!() };
        assert!((points[0].1 - 400.0).abs() < 1e-9, "data 0 stays at the segment floor");
        assert_eq!(points[1].1, BREAK_HIDDEN, "the in-gap vertex is hidden");
        assert!((points[2].1 - (194.0 - 194.0 / 3.0)).abs() < 1e-6, "data 900 compresses");
        assert_eq!(points[0].0, 0.0, "the cross-axis coordinate is untouched");
    }

    // ── Indicator-pixel reservation (quality review F2) ──────────────────────

    #[test]
    fn a_gap_touching_a_domain_edge_reserves_no_indicator_pixels() {
        // The gap [0,500] removes a range but separates nothing, so
        // `segments.windows(2)` draws no indicator — reserving a `break_size`
        // for it left a 12px dead band at the plot edge.
        let edge = apply_break_to_scale((0.0, 1000.0), &[[0.0, 500.0]], (400.0, 0.0), 12.0);
        assert_eq!(edge.segments.len(), 1);
        assert!(edge.indicator_pixels.is_empty(), "no indicator between one segment");
        assert!(
            (edge.segments[0].pixel_hi - 0.0).abs() < 1e-9,
            "the single segment gets the whole pixel range; got {:?}",
            edge.segments[0]
        );
        // A mid-domain gap still reserves exactly one indicator's pixels.
        let mid = apply_break_to_scale((0.0, 1000.0), &[[400.0, 500.0]], (400.0, 0.0), 12.0);
        let total: f64 = mid.segments.iter().map(|s| (s.pixel_hi - s.pixel_lo).abs()).sum();
        assert!((total - 388.0).abs() < 1e-9, "400 - 12 pixels of data; got {total}");
    }

    #[test]
    fn axis_chrome_outside_the_pixel_range_is_left_alone() {
        let br = guide_break();
        // An x-axis label below the plot (y = 430) is cross-axis chrome for a
        // y-break and must not be reverse-mapped.
        let mut nodes = vec![SceneNode::Text {
            x: 50.0,
            y: 430.0,
            content: "A".to_string(),
            style: crate::render::draw::to_scene_text_style(
                palette::Srgba::new(0, 0, 0, 255),
                11.0,
                crate::layout::TextAnchor::Middle,
                0.0,
                "Inter",
                None,
                None,
                1.0,
            ),
            slot: None,
        }];
        guide_remap(&br).remap_axis_chrome(&mut nodes);
        let SceneNode::Text { y, .. } = &nodes[0] else { panic!() };
        assert_eq!(*y, 430.0);
    }
}
