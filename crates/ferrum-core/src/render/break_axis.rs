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

use std::collections::HashSet;

use ferrum_scene::{FillStroke, MarkBatchKind, PathCmd, SceneNode};

use crate::layout::{Rect, ThemeInputs};
use crate::render::draw::to_scene_stroke;
use crate::render::marks;

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
    /// How many of the requested gaps survived [`normalize_gaps`] — i.e. how
    /// many removed a real slice of the domain, after clipping to it and
    /// merging overlaps.
    ///
    /// Zero means the `BreakAxis` is a no-op: every gap fell outside the
    /// domain, or the axis had no numeric domain to clip against at all.
    /// That case is otherwise invisible (no indicator, no moved mark), so
    /// `scene_build::build_structural_nodes` reads this to emit
    /// [`RenderWarning::BreakAxisIneffective`](crate::render::RenderWarning::BreakAxisIneffective)
    /// rather than letting a structural feature disappear in silence.
    /// A gap that merely touches a domain edge still counts here — it
    /// removes data even though it draws no indicator.
    pub effective_gaps: usize,
}

// ── Scale computation ─────────────────────────────────────────────────────────

/// Clip `gaps` to the domain `d_lo`/`d_hi` — **in either order** — then sort
/// and merge them into disjoint, ascending ranges.
///
/// The domain bounds are read as an unordered pair. A reversed scale
/// (`LinearScale(reverse=True)`) reports a DESCENDING `data_domain()`, and
/// clipping against it endpoint-wise discarded every gap: `g[0].max(1000.0)`
/// and `g[1].min(0.0)` can never satisfy `hi > lo`, so a `BreakAxis` on a
/// reversed axis silently rendered an unbroken chart. Which data range a gap
/// removes is orientation-free; the orientation belongs to the pixel
/// assignment in [`apply_break_to_scale`].
///
/// A GAP's own bounds are read as written, not sorted: `[850, 150]` describes
/// an empty interval and removes nothing. That stays a no-op — but no longer
/// a silent one, since `effective_gaps` then reaches zero and the caller
/// warns.
///
/// Merging is what makes overlapping and nested gaps well-defined: `[100, 900]`
/// plus `[200, 300]` is one removed range `[100, 900]`, not two. Without it the
/// segment walk below took the *last* gap's upper bound as its cursor, so a
/// nested gap moved the cursor backwards and re-admitted data the enclosing gap
/// had removed; and the reserved indicator space counted a break the scale never
/// draws. Adjacent gaps (`hi == lo`) merge too — they remove one contiguous
/// range and get one indicator, so they must reserve one indicator's pixels.
///
/// A non-finite bound (`NaN`/`±inf`) is dropped before clipping rather than
/// clipped against the domain: `f64::max`/`f64::min` return the *non-NaN*
/// argument (Rust docs), so without this guard `[NaN, NaN].max(d_lo).min(d_hi)`
/// resolved to `[d_lo, d_hi]` — a domain-covering gap that silently erased
/// every datum (BRK-1, `design-docs/superpowers/followups/
/// 2026-05-15-code-archaeology.md`). #92/#148 refuse a non-finite gap at
/// `BreakAxis` construction and at the wire gate, so this normal path no
/// longer sees one; this filter is defense-in-depth for any caller that
/// builds a `BreakAxisSpec` directly (e.g. this module's own unit tests) —
/// `apply_break_to_scale` stays infallible per the `chart_config.rs:184-195`
/// layering doctrine, so a malformed gap is treated as absent (dropping
/// `effective_gaps`, which the caller already warns on) rather than erroring.
fn normalize_gaps(gaps: &[[f64; 2]], d_lo: f64, d_hi: f64) -> Vec<(f64, f64)> {
    let (d_lo, d_hi) = (d_lo.min(d_hi), d_lo.max(d_hi));
    let mut clipped: Vec<(f64, f64)> = gaps
        .iter()
        .filter(|g| g[0].is_finite() && g[1].is_finite())
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
/// `pixel_range` is the BROKEN SCALE'S OWN pixel range
/// ([`ScaleKind::pixel_range`](crate::render::scale_resolve::ScaleKind::pixel_range)),
/// i.e. the padded one the unbroken scale places data in, as
/// `(axis_start_px, axis_end_px)` in that scale's orientation — for a
/// continuous Y that is `(bottom, top)`. It is NOT the raw plot rect: the
/// segments built here must tile the same span [`BreakRemap`] reverse-maps
/// from, because re-placing a coordinate is a round trip through both
/// (`BreakRemap::data_at` divides by the scale's range, `ScaleSegment::
/// pixel_for` multiplies by a slice of THIS range). Handing the raw rect in
/// while the remap read the padded range made that round trip lossy in
/// exactly the padding band: every effective break silently deleted up to
/// `SCALE_PADDING_MAX_PX` of axis padding, so a broken axis's ticks, marks
/// and gridlines sat up to 8px off their unbroken positions and a gap
/// stepping over the domain boundary moved the whole chart (#226).
/// `break_size` is the pixel height/width reserved for each indicator.
///
/// `original_domain` may be given in EITHER order. A reversed scale reports a
/// descending `(lo, hi)`; the domain is normalized to ascending here and the
/// orientation is carried into the pixel assignment instead — `d_hi` simply
/// starts at the far end of `pixel_range`. Segments therefore always run
/// ascending in data, whichever way the axis is drawn, and every downstream
/// reader (`contains`, `pixel_for`, `gap_containing`) stays orientation-free.
///
/// Segments are allocated proportionally to their data-domain fraction of the
/// total non-gap span. Gaps that fall entirely outside `original_domain` are
/// ignored (and counted: see [`BreakResult::effective_gaps`], which the caller
/// turns into a warning rather than a silent no-op), and gaps that overlap (or
/// merely touch) each other are merged into one before segments are computed —
/// see [`normalize_gaps`].
pub fn apply_break_to_scale(
    original_domain: (f64, f64),
    gaps: &[[f64; 2]],
    pixel_range: (f64, f64),
    break_size: f64,
) -> BreakResult {
    let (raw_lo, raw_hi) = original_domain;
    let descending = raw_hi < raw_lo;
    let (d_lo, d_hi) = (raw_lo.min(raw_hi), raw_lo.max(raw_hi));
    // The pixel end `d_lo` sits at, and the direction the walk runs in. For an
    // ascending domain this is `pixel_range` unchanged (byte-identical to
    // every break rendered before reversed scales were handled); for a
    // descending one the two ends swap, so the smallest data value keeps the
    // pixel the unbroken scale put it at.
    let (px_lo, px_hi) = if descending {
        (pixel_range.1, pixel_range.0)
    } else {
        pixel_range
    };
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
            effective_gaps: 0,
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

    BreakResult {
        segments,
        indicator_pixels,
        effective_gaps: valid_gaps.len(),
    }
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

/// The gap in `result` that contains `value`, as a `(lo, hi)` data-space
/// pair — the band between two consecutive visible segments.
///
/// `None` when `value` maps onto a visible segment (including exactly on a
/// gap boundary, which belongs to its adjacent segment — [`ScaleSegment`]'s
/// bounds are inclusive, so gaps are open intervals) or lies outside the
/// domain entirely.
///
/// Consumed by the BreakAxis annotation break router (spec §4 H5, F-L08-07,
/// #228) to tell an annotation a gap swallowed from one it merely reaches, and
/// to name the mechanism in `RenderWarning::AnnotationSuppressedByBreak`.
pub fn gap_containing(value: f64, result: &BreakResult) -> Option<(f64, f64)> {
    if broken_scale_map(value, result).is_some() {
        return None;
    }
    result.segments.windows(2).find_map(|w| {
        let (_, hi0) = w[0].ordered_data();
        let (lo1, _) = w[1].ordered_data();
        (value > hi0 && value < lo1).then_some((hi0, lo1))
    })
}

/// The first gap in `result` that the closed interval between `a` and `b`
/// (given in either order) covers ENTIRELY.
///
/// The companion of [`gap_containing`], which answers the same question for a
/// single point. Between them they decide whether an interval's closure meets
/// any gap's interior: either an endpoint is inside a gap (`gap_containing`),
/// or neither is and the interval reaches across one (here). An extent that
/// straddles a gap has no in-gap endpoint at all, which is why it survived
/// the endpoint-only suppression test and painted straight across the break.
///
/// Consumed by the BreakAxis annotation break router (spec §4 H5, #228) for the
/// extent-shaped annotation kinds, where it answers "does this break reach the
/// interval at all"; whether anything of it SURVIVES is
/// [`BreakRemap::data_extent_visible`]'s question.
pub fn gap_spanned_by(a: f64, b: f64, result: &BreakResult) -> Option<(f64, f64)> {
    let (lo, hi) = (a.min(b), a.max(b));
    result.segments.windows(2).find_map(|w| {
        let (_, hi0) = w[0].ordered_data();
        let (lo1, _) = w[1].ordered_data();
        (lo <= hi0 && hi >= lo1).then_some((hi0, lo1))
    })
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
fn is_hidden(c: f64) -> bool {
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

/// Which side of a boundary a half-plane clip keeps.
#[derive(Clone, Copy)]
enum Side {
    AtLeast,
    AtMost,
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
        self.axis == "y"
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
    /// `scene_build::build_structural_nodes` emits
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
    /// merge above (`build_structural_nodes`) makes unreachable for the
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
            SceneNode::Path { commands, style, closed: true } => {
                match polygon_points(&commands) {
                    Some(points) => self.clip_polygon_into(&points, &style, out),
                    // A closed ring the polygon reader will not take is CURVED,
                    // and neither the clip nor the split can cut a placed curve
                    // faithfully, so it keeps the per-vertex fallback.
                    None => {
                        let mut node = SceneNode::Path { commands, style, closed: true };
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
            SceneNode::Path { commands, style, closed: true } => {
                let mut head = SceneNode::Path { commands, style, closed: true };
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
            SceneNode::Polyline { points, style } => {
                for piece in self.split_polyline(&points).pieces {
                    out.push(SceneNode::Polyline { points: piece, style: style.clone() });
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
            SceneNode::Path { commands, style, closed: false } => {
                match straight_path_points(&commands) {
                    Some(points) => {
                        let split = self.split_polyline(&points);
                        if split.cut {
                            for piece in split.pieces {
                                out.push(SceneNode::Path {
                                    commands: path_commands(&piece, false),
                                    style: style.clone(),
                                    closed: false,
                                });
                            }
                        } else {
                            let mut node = SceneNode::Path { commands, style, closed: false };
                            self.remap_node(&mut node);
                            out.push(node);
                        }
                    }
                    None => {
                        let mut node = SceneNode::Path { commands, style, closed: false };
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
    /// (`draw::arrowhead_path_with_opacity` writes it as the leading
    /// `MoveTo`) and is the shaft's own `(x2, y2)`; `head_size` is the
    /// distance from that tip to the midpoint of the other two vertices,
    /// which is how `draw::arrowhead_triangle` built them.
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
                crate::render::draw::arrowhead_triangle(
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

/// The three vertices of an arrowhead triangle node, tip first — the shape
/// `draw::arrowhead_path_with_opacity` emits (`MoveTo` tip, `LineTo` left,
/// `LineTo` right, `Close`).
///
/// `None` for any other closed path, which routes the node to the per-vertex
/// remap instead of the arrowhead rebuild.
fn triangle_vertices(node: &SceneNode) -> Option<[(f64, f64); 3]> {
    let SceneNode::Path { commands, .. } = node else {
        return None;
    };
    let points = polygon_points(commands)?;
    match points[..] {
        [tip, left, right] => Some([tip, left, right]),
        _ => None,
    }
}

/// Overwrite an arrowhead triangle node's three vertices in place, keeping
/// its style and command shape. The inverse of [`triangle_vertices`].
fn set_triangle_vertices(node: &mut SceneNode, points: [(f64, f64); 3]) {
    let SceneNode::Path { commands, .. } = node else {
        return;
    };
    let mut vertices = points.iter();
    for cmd in commands.iter_mut() {
        if let PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } = cmd {
            if let Some(&(nx, ny)) = vertices.next() {
                *x = nx;
                *y = ny;
            }
        }
    }
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

/// Push one `Line` node per piece of a split two-vertex stroke.
///
/// A two-vertex source is one edge, and one edge contributes exactly two points
/// to each piece it reaches (its entry and its exit), so every piece here is a
/// straight `Line` — the node kind the source had. Shared by the plain split and
/// the segment-batch split so a shaft and a rule cannot be re-spelled
/// differently.
///
/// That two-point invariant is ENFORCED rather than assumed: a `Line` is the one
/// node kind with no way to spell an interior vertex, so a three-point piece
/// arriving here could only be rendered by discarding its middle — a silently
/// wrong stroke. The `debug_assert` costs nothing in release and turns a future
/// caller handing in a longer polyline into a test failure instead.
fn push_line_pieces(
    pieces: &[Vec<(f64, f64)>],
    style: &ferrum_scene::StrokeStyle,
    out: &mut Vec<SceneNode>,
) {
    for piece in pieces {
        debug_assert_eq!(
            piece.len(),
            2,
            "a `Line` piece is one edge's entry and exit; got {piece:?}"
        );
        let (a, b) = (piece[0], piece[piece.len() - 1]);
        out.push(SceneNode::Line {
            x1: a.0,
            y1: a.1,
            x2: b.0,
            y2: b.1,
            style: style.clone(),
        });
    }
}

/// Spell `points` as a path: a leading `MoveTo`, then `LineTo` per vertex, then
/// `Close` when the path is closed.
///
/// The inverse of [`polygon_points`]/[`straight_path_points`], and the one
/// spelling both rewrites emit — the clipped area ring and the split step line.
fn path_commands(points: &[(f64, f64)], closed: bool) -> Vec<PathCmd> {
    let mut commands = Vec::with_capacity(points.len() + usize::from(closed));
    for (i, &(x, y)) in points.iter().enumerate() {
        commands.push(if i == 0 {
            PathCmd::MoveTo { x, y }
        } else {
            PathCmd::LineTo { x, y }
        });
    }
    if closed {
        commands.push(PathCmd::Close);
    }
    commands
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

/// Build one clipped rect piece's scene node — a `Path` with arcs on only
/// the UNCUT corners when a cut edge would otherwise round a corner the gap
/// itself produced, a plain `Rect` in every other case (#230-2).
///
/// A piece stays a `Rect` rather than converting to a `Path` in THREE cases,
/// each because the conversion would either add no information or would
/// actively cost the piece something only `Rect`-ness carries (SVG
/// centre-anchored rotation, WASM hit-testing, R-tree membership, batch
/// packing eligibility):
///
/// - **No cut edge, or no radius.** The common case — an unbroken chart, or
///   a break whose gaps miss the piece entirely — and byte-identical to what
///   this code emitted before #230-2.
/// - **Every surviving corner is squared** (quality-review finding, cycle
///   3): a piece cut on both edges of one axis (the middle slice of a
///   two-gap break) has no corner left to round, so its outline is already a
///   plain four-cornered box. It emits `Rect { corner_radius: 0.0 }` rather
///   than a `Path` that would spell the identical geometry while forfeiting
///   `Rect`-only behavior for no visual gain.
/// - **`style.angle != 0.0`** (quality-review finding, cycle 3):
///   `SceneNode::Rect` rotates about its own centre
///   (`svg_walk.rs::emit_node`'s `Rect` arm), while `SceneNode::Path` rotates
///   about its first `MoveTo` — a converted piece would silently jump to a
///   different rotation anchor than its unbroken rounded sibling (hundreds
///   of pixels, measured), and lose the rotation entirely in WASM, which
///   never reads `style.angle` for a `Path`. A rotated piece keeps the plain
///   `Rect`, uncut radius and all, rather than corrupt its anchor: the
///   un-squared cut edge in that specific rotated+rounded+broken three-way
///   combination is a documented residual (`ferrum-spec.md`'s #230-2 dated
///   note), not a silent regression.
fn rounded_rect_piece(
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    style: FillStroke,
    corner_radius: f64,
    cut: CutEdges,
) -> SceneNode {
    if corner_radius <= 0.0 || !cut.any() || style.angle != 0.0 {
        return SceneNode::Rect { x, y, w, h, style, corner_radius };
    }
    if cut.corners().iter().all(|&squared| squared) {
        return SceneNode::Rect { x, y, w, h, style, corner_radius: 0.0 };
    }
    SceneNode::Path {
        commands: square_cut_rounded_rect_path(x, y, w, h, corner_radius, cut),
        style,
        closed: true,
    }
}

/// A rounded rect's outline, clockwise from the top edge, with the radius
/// zeroed on every corner `cut` touches — a corner squares whenever EITHER
/// of its two adjacent edges is cut. Never called with all four corners
/// squared; [`rounded_rect_piece`] takes the plain-`Rect` branch for that.
///
/// `radius` is clamped per axis, matching SVG `<rect rx ry>` semantics
/// exactly: `rx` clamps to `w / 2` and `ry` clamps to `h / 2`, INDEPENDENTLY
/// of each other. An SVG `<rect>` gets that clamp for free from the
/// rendering rules, but a hand-built arc path does not, so a piece shorter
/// than the requested radius on only one axis (a thin sliver left after a
/// gap eats most of a bar) needs it spelled out here to match what the
/// plain-`Rect` branch would have rendered for the same dimensions — a
/// single shared scalar clamped by `min(w/2, h/2)` would round the uncut
/// corners of a narrow-but-tall piece more tightly than an unsplit rect of
/// the same size does, since `rx` and `ry` never actually collapse to one
/// number unless `w == h`.
fn square_cut_rounded_rect_path(
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    radius: f64,
    cut: CutEdges,
) -> Vec<PathCmd> {
    let radius = radius.max(0.0);
    let rx = radius.min(w / 2.0);
    let ry = radius.min(h / 2.0);
    let [squared_tl, squared_tr, squared_br, squared_bl] = cut.corners();
    let corner = |squared: bool| -> (f64, f64) { if squared { (0.0, 0.0) } else { (rx, ry) } };
    let (rx_tl, ry_tl) = corner(squared_tl);
    let (rx_tr, ry_tr) = corner(squared_tr);
    let (rx_br, ry_br) = corner(squared_br);
    let (rx_bl, ry_bl) = corner(squared_bl);

    let mut cmds = Vec::with_capacity(9);
    cmds.push(PathCmd::MoveTo { x: x + rx_tl, y });
    cmds.push(PathCmd::LineTo { x: x + w - rx_tr, y });
    if rx_tr > 0.0 || ry_tr > 0.0 {
        cmds.push(PathCmd::ArcTo {
            rx: rx_tr,
            ry: ry_tr,
            rotation: 0.0,
            large_arc: false,
            sweep: true,
            x: x + w,
            y: y + ry_tr,
        });
    }
    cmds.push(PathCmd::LineTo { x: x + w, y: y + h - ry_br });
    if rx_br > 0.0 || ry_br > 0.0 {
        cmds.push(PathCmd::ArcTo {
            rx: rx_br,
            ry: ry_br,
            rotation: 0.0,
            large_arc: false,
            sweep: true,
            x: x + w - rx_br,
            y: y + h,
        });
    }
    cmds.push(PathCmd::LineTo { x: x + rx_bl, y: y + h });
    if rx_bl > 0.0 || ry_bl > 0.0 {
        cmds.push(PathCmd::ArcTo {
            rx: rx_bl,
            ry: ry_bl,
            rotation: 0.0,
            large_arc: false,
            sweep: true,
            x,
            y: y + h - ry_bl,
        });
    }
    cmds.push(PathCmd::LineTo { x, y: y + ry_tl });
    if rx_tl > 0.0 || ry_tl > 0.0 {
        cmds.push(PathCmd::ArcTo {
            rx: rx_tl,
            ry: ry_tl,
            rotation: 0.0,
            large_arc: false,
            sweep: true,
            x: x + rx_tl,
            y,
        });
    }
    cmds.push(PathCmd::Close);
    cmds
}

/// The vertices of an OPEN path whose every command draws a straight line —
/// `LineTo` plus the `HLineTo`/`VLineTo` shorthand `mark_line`'s three `step*`
/// interpolations are written with.
///
/// `None` for anything a break cannot cut faithfully: a curve command
/// (`QuadTo`/`CubicTo`/`ArcTo`), a second subpath, or a `Close` (a closed ring
/// belongs to [`polygon_points`] and the polygon clip). That is the documented
/// split-vs-fallback boundary of [`BreakRemap::split_nodes`].
fn straight_path_points(commands: &[PathCmd]) -> Option<Vec<(f64, f64)>> {
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(commands.len());
    for (i, cmd) in commands.iter().enumerate() {
        let (x, y) = match (i, cmd) {
            (0, PathCmd::MoveTo { x, y }) => (*x, *y),
            (_, PathCmd::LineTo { x, y }) => (*x, *y),
            // The shorthand carries only the coordinate that moves; the other
            // one is the current point's, which is the previous vertex.
            (_, PathCmd::HLineTo { x }) => (*x, points.last()?.1),
            (_, PathCmd::VLineTo { y }) => (points.last()?.0, *y),
            _ => return None,
        };
        points.push((x, y));
    }
    (points.len() >= 2).then_some(points)
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

    // ── gap_containing tests ─────────────────────────────────────────────────

    #[test]
    fn gap_containing_value_inside_gap() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[150.0, 850.0]], (50.0, 350.0), 12.0);
        assert_eq!(gap_containing(500.0, &result), Some((150.0, 850.0)));
    }

    #[test]
    fn gap_containing_visible_value_is_none() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[150.0, 850.0]], (50.0, 350.0), 12.0);
        assert_eq!(gap_containing(100.0, &result), None);
        assert_eq!(gap_containing(900.0, &result), None);
    }

    #[test]
    fn gap_containing_boundary_value_is_none() {
        // Gaps are open intervals: a value exactly on the gap edge belongs to
        // the adjacent segment, per `ScaleSegment::contains`'s inclusive bounds.
        let result = apply_break_to_scale((0.0, 1000.0), &[[150.0, 850.0]], (50.0, 350.0), 12.0);
        assert_eq!(gap_containing(150.0, &result), None);
        assert_eq!(gap_containing(850.0, &result), None);
    }

    #[test]
    fn gap_containing_picks_the_right_gap_among_several() {
        let result = apply_break_to_scale(
            (0.0, 1000.0),
            &[[100.0, 200.0], [700.0, 800.0]],
            (50.0, 350.0),
            12.0,
        );
        assert_eq!(gap_containing(150.0, &result), Some((100.0, 200.0)));
        assert_eq!(gap_containing(750.0, &result), Some((700.0, 800.0)));
    }

    #[test]
    fn gap_containing_value_outside_domain_is_none() {
        let result = apply_break_to_scale((0.0, 1000.0), &[[150.0, 850.0]], (50.0, 350.0), 12.0);
        // Outside the domain entirely — not a gap, just out of range.
        assert_eq!(gap_containing(-50.0, &result), None);
        assert_eq!(gap_containing(1500.0, &result), None);
    }

    #[test]
    fn gap_containing_no_gaps_is_always_none() {
        let result = apply_break_to_scale((0.0, 1000.0), &[], (50.0, 350.0), 12.0);
        assert_eq!(gap_containing(500.0, &result), None);
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
    fn nan_gap_bounds_are_dropped_not_domain_covering() {
        // BRK-1 (design-docs/superpowers/followups/2026-05-15-code-archaeology.md)
        // closed: a non-finite gap bound used to resolve to `[d_lo, d_hi]` under
        // IEEE `f64::max`/`f64::min` semantics (`NaN.max(d_lo) == d_lo`), which
        // clipped to a domain-covering gap and silently erased all data. #92/#148
        // now refuse a non-finite gap at `BreakAxis` construction and at the wire
        // gate, so this path is unreachable through the normal pipeline; this
        // test inverts the former pin to assert `normalize_gaps`'s defense-in-depth
        // filter, which drops a non-finite bound instead of clipping it — the gap
        // contributes nothing, so the full domain survives as one segment (the
        // same "gap outside the domain" shape `gap_covering_entire_domain_leaves_
        // no_segments`'s sibling `degenerate_domain_with_gap_produces_single_
        // segment` test already exercises for a different malformed-gap shape).
        let result = apply_break_to_scale((0.0, 100.0), &[[f64::NAN, f64::NAN]], (50.0, 350.0), 12.0);
        assert_eq!(
            result.segments.len(),
            1,
            "a non-finite gap bound must be dropped, not clipped into a domain-covering gap"
        );
        assert_eq!(result.effective_gaps, 0, "a dropped non-finite gap must not count as effective");
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

    /// `BREAK_HIDDEN` is poison (#230-9): a pixel already parked at the
    /// sentinel must never reverse-map back into the domain, even against a
    /// single-segment (no-op-looking) break whose degenerate span or
    /// permissive clamp would otherwise treat it as any other out-of-range
    /// pixel and clamp it to a domain edge — which is exactly the mechanism
    /// that resurrected a hidden mark at the plot's top/bottom edge before
    /// the same-axis merge (#230-4) made the double-remap that produced it
    /// unreachable. Checked directly against `data_at`'s poison guard via
    /// `coord`, its only production-reachable caller.
    #[test]
    fn coord_of_break_hidden_pixel_stays_hidden_never_clamped_into_domain() {
        let br = one_segment_break(0.0, 100.0, 50.0, 350.0);
        assert!(
            remap("x", &br).coord(BREAK_HIDDEN).is_none(),
            "a BREAK_HIDDEN pixel must reverse-map to nothing, not clamp to a domain edge"
        );

        // Same claim against a real multi-segment break, where the clamp
        // would otherwise land the sentinel on the FIRST segment's own
        // pixel range (since BREAK_HIDDEN is far below every real pixel) —
        // the precise shape of the pre-fix resurrection.
        let two_seg = two_segment_break();
        assert!(
            remap("x", &two_seg).coord(BREAK_HIDDEN).is_none(),
            "a BREAK_HIDDEN pixel must stay hidden across a real break's segments too"
        );
    }

    /// `data_at`'s poison guard, not just `coord`'s NaN short-circuit: a
    /// hidden pixel handed to `data_at` directly must come back non-finite
    /// (never a real, in-domain data value), so every caller of `data_at`
    /// — not only `coord` — inherits the poison rather than each having to
    /// re-derive its own guard.
    #[test]
    fn data_at_of_break_hidden_pixel_is_never_a_finite_domain_value() {
        let br = two_segment_break();
        let remap = remap("x", &br);
        assert!(
            remap.data_at(BREAK_HIDDEN).is_nan(),
            "BREAK_HIDDEN must not reverse-map to a finite data value"
        );
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

    /// `data_extent_visible` answers, in DATA space, the question `clip_extent`
    /// answers in pixels — "is there anything left to draw" — for the
    /// annotation router, whose extents are spec values rather than placed
    /// pixels (#228). Its edge rules are `data_spans`': a point ON a segment is
    /// visible, a point in a gap is not, and an interval that merely touches a
    /// segment's edge from inside the gap contributes no length.
    #[test]
    fn data_extent_visible_matches_the_clip_it_gates() {
        let br = guide_break();
        let remap = guide_remap(&br);
        assert!(remap.data_extent_visible(0.0, 1000.0), "straddling: two pieces survive");
        assert!(remap.data_extent_visible(100.0, 400.0), "in-gap endpoint: 100..150 survives");
        assert!(!remap.data_extent_visible(300.0, 700.0), "wholly inside the gap");
        assert!(
            !remap.data_extent_visible(150.0, 850.0),
            "exactly the gap: both ends are visible values, but no length is"
        );
        assert!(remap.data_extent_visible(100.0, 100.0), "a visible point has a position");
        assert!(!remap.data_extent_visible(500.0, 500.0), "an in-gap point does not");
        // Order-independent: the router hands in spec order, not sorted order.
        assert!(remap.data_extent_visible(1000.0, 0.0));
    }

    #[test]
    fn extent_entirely_inside_a_gap_is_dropped_not_flattened() {
        let br = guide_break();
        // Data 300..500, wholly inside the gap.
        let pieces = guide_remap(&br).clip_extent(400.0 - 0.3 * 400.0, 400.0 - 0.5 * 400.0);
        assert!(pieces.is_empty(), "a wholly-hidden extent emits nothing; got {pieces:?}");
    }

    /// #230-9 belt-and-braces (cycle-2 fix): `clip_extent` must treat a
    /// `BREAK_HIDDEN` endpoint as poison, not as an ordinary out-of-range
    /// pixel. Pre-fix, a hidden endpoint's `data_at` NaN fed straight into
    /// `f64::min`/`max` to order `lo`/`hi`, and `min`/`max` silently DROP a
    /// NaN operand rather than propagate it — verified empirically pre-fix:
    /// both endpoints hidden returned the segment's FULL pixel extent
    /// (`c_lo = NaN.max(s_lo) == s_lo`, `c_hi = NaN.min(s_hi) == s_hi`), and
    /// one endpoint hidden returned a zero-length pair at the surviving
    /// endpoint. RED-proven by file-copy revert to the pre-cycle-2 source:
    /// both assertions below failed with exactly that shape before the
    /// `is_hidden(a) || is_hidden(b)` early-return was added.
    #[test]
    fn clip_extent_with_a_break_hidden_endpoint_stays_hidden_not_full_extent() {
        let br = guide_break();
        // Both endpoints already hidden: emits nothing, never the segment's
        // full extent.
        assert!(
            guide_remap(&br).clip_extent(BREAK_HIDDEN, BREAK_HIDDEN).is_empty(),
            "both-ends-hidden must stay empty, not resurrect the full segment extent"
        );
        // One endpoint hidden, the other a real pixel: also empty, never a
        // zero-length pair parked at the surviving endpoint.
        assert!(
            guide_remap(&br).clip_extent(BREAK_HIDDEN, 40.0).is_empty(),
            "one-end-hidden must stay empty, not a zero-length pair at the surviving endpoint"
        );
        assert!(
            guide_remap(&br).clip_extent(40.0, BREAK_HIDDEN).is_empty(),
            "hidden-endpoint order must not matter"
        );
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

    // ── Corner radius on cut edges (#230-2) ───────────────────────────────────

    /// The guide's worked example with a corner radius: `mark_bar(value=900,
    /// corner_radius=8)` under a `(150, 850)` gap — the measured repro (prior
    /// disposition `sdd-batch-h/task-3-report.md:125`, F5/S1): the old code
    /// copied `corner_radius` onto both pieces verbatim, so `rx="8"` appeared
    /// on the cut edge each piece shares with the gap, not just the two edges
    /// that were always there. Fixed: each piece is a `Path` with arcs ONLY
    /// on its uncut corners; the cut edge is flat.
    #[test]
    fn corner_radius_split_rect_squares_only_the_cut_edge() {
        let br = guide_break();
        let node = SceneNode::Rect {
            x: 10.0,
            y: 40.0,
            w: 20.0,
            h: 360.0,
            style: fill_stroke(),
            corner_radius: 8.0,
        };
        let clipped = guide_remap(&br).clip_nodes(vec![node]);
        assert_eq!(clipped.nodes.len(), 2, "crossing bar splits: {:?}", clipped.nodes);

        // Piece 0 (data 0..150, pixels 206..400): the baseline (y=400, data
        // 0) is the ORIGINAL rect's real bottom edge — uncut, rounded. The
        // top (y=206, data 150) sits against the gap — cut, squared.
        let lower = path_commands_of(&clipped.nodes[0]);
        assert_eq!(
            arc_count(lower),
            2,
            "only the two uncut (bottom) corners get an arc; got {lower:?}"
        );
        assert!(
            matches!(lower.first(), Some(PathCmd::MoveTo { x, y })
                if (*x - 10.0).abs() < 1e-9 && (*y - 206.0).abs() < 1e-9),
            "the squared top-left corner starts flush at the rect's corner point; got {lower:?}"
        );
        assert!(
            matches!(lower[1], PathCmd::LineTo { x, y }
                if (x - 30.0).abs() < 1e-9 && (y - 206.0).abs() < 1e-9),
            "the top edge runs corner to corner with no arc; got {lower:?}"
        );
        for cmd in lower {
            if let PathCmd::ArcTo { rx, ry, y, .. } = cmd {
                assert!((*rx - 8.0).abs() < 1e-9 && (*ry - 8.0).abs() < 1e-9);
                assert!(*y > 206.0 + 1e-6, "an arc must not sit on the cut (top) edge; got {cmd:?}");
            }
        }

        // Piece 1 (data 850..900, pixels 129.33..194): the true top of the
        // original rect (y≈129.33, data 900) is uncut — rounded. Its bottom
        // (y=194, data 850) sits against the gap — cut, squared.
        let upper = path_commands_of(&clipped.nodes[1]);
        assert_eq!(
            arc_count(upper),
            2,
            "only the two uncut (top) corners get an arc; got {upper:?}"
        );
        let bottom_y = 194.0;
        for cmd in upper {
            if let PathCmd::ArcTo { rx, ry, y, .. } = cmd {
                assert!((*rx - 8.0).abs() < 1e-9 && (*ry - 8.0).abs() < 1e-9);
                assert!(*y < bottom_y - 1e-6, "an arc must not sit on the cut (bottom) edge; got {cmd:?}");
            }
        }
        assert!(
            matches!(upper.last(), Some(PathCmd::Close)),
            "a rounded-piece path must close; got {upper:?}"
        );
    }

    /// Mirror of the above on the X axis: a rect split by an X break squares
    /// its left/right cut edge instead of top/bottom, proving the corner
    /// treatment is not accidentally Y-only.
    #[test]
    fn corner_radius_split_rect_on_x_axis_squares_the_cut_edge() {
        // Domain [0, 1000] on x pixels 0 -> 400 (increasing with data, unlike
        // the Y guide fixture), gap (150, 850), same 12px indicator.
        let result = apply_break_to_scale((0.0, 1000.0), &[[150.0, 850.0]], (0.0, 400.0), 12.0);
        let remap = BreakRemap::new("x", (0.0, 1000.0), (0.0, 400.0), &result);
        // A rect from data 0 (pixel 0) to data 900 (pixel 360), like the Y
        // case rotated 90 degrees.
        let node = SceneNode::Rect {
            x: 0.0,
            y: 10.0,
            w: 360.0,
            h: 20.0,
            style: fill_stroke(),
            corner_radius: 8.0,
        };
        let clipped = remap.clip_nodes(vec![node]);
        assert_eq!(clipped.nodes.len(), 2, "crossing bar splits: {:?}", clipped.nodes);

        // Piece 0 (data 0..150): its LEFT edge (x=0) is the original rect's
        // real left edge — uncut, rounded. Its RIGHT edge sits against the
        // gap — cut, squared.
        // `path_commands_of` itself asserts the node is a `Path`, not a
        // `Rect` — the cut edge means this piece must not take the
        // plain-`Rect` branch at all.
        let left_piece = path_commands_of(&clipped.nodes[0]);
        assert_eq!(arc_count(left_piece), 2, "only the two uncut (left) corners get an arc; got {left_piece:?}");

        // Piece 1's LEFT edge sits against the gap — cut, squared; its RIGHT
        // edge (x=360, data 900) is the original rect's real right edge —
        // uncut, rounded.
        let right_piece = path_commands_of(&clipped.nodes[1]);
        assert_eq!(arc_count(right_piece), 2, "only the two uncut (right) corners get an arc; got {right_piece:?}");
    }

    /// #230-2's clamp clause: a piece SHORTER than the requested radius on
    /// its uncut side gets the same clamp an SVG `<rect rx ry>` gets for
    /// free (`rx`/`ry` each capped at half of their own dimension) — a
    /// hand-built arc path does not get that clamp automatically, so it must
    /// be applied explicitly.
    #[test]
    fn corner_radius_clamps_when_the_piece_is_shorter_than_the_radius() {
        let br = guide_break();
        // A rect from data 0 to data 900, but only 4px wide — narrower than
        // the requested 8px radius.
        let node = SceneNode::Rect {
            x: 10.0,
            y: 40.0,
            w: 4.0,
            h: 360.0,
            style: fill_stroke(),
            corner_radius: 8.0,
        };
        let clipped = guide_remap(&br).clip_nodes(vec![node]);
        assert_eq!(clipped.nodes.len(), 2);
        for node in &clipped.nodes {
            let cmds = path_commands_of(node);
            for cmd in cmds {
                if let PathCmd::ArcTo { rx, ry, .. } = cmd {
                    // SVG `<rect rx ry>` clamps each axis independently:
                    // rx to w/2 (2.0, since w=4.0), ry to h/2 (180.0, but
                    // the 8px requested radius never reaches that ceiling,
                    // so ry stays 8.0). A single shared min(w/2, h/2)
                    // clamp would wrongly squeeze ry down to 2.0 too.
                    assert!(
                        (*rx - 2.0).abs() < 1e-9,
                        "rx must clamp to half the 4px width (2.0); got {cmd:?}"
                    );
                    assert!(
                        (*ry - 8.0).abs() < 1e-9,
                        "ry must stay at the requested 8px radius (well under half the 360px height); got {cmd:?}"
                    );
                }
            }
        }
    }

    /// A piece with no cut edge (a rect that fits wholly in one segment)
    /// keeps the plain `Rect` node — byte-identical to pre-#230-2 output,
    /// which is also what keeps every unbroken chart's bar goldens
    /// untouched by this fix.
    #[test]
    fn corner_radius_on_an_uncut_piece_stays_a_plain_rect() {
        let br = guide_break();
        // Data 0..100, wholly inside segment 0 (data 0..150) — no gap
        // touches this bar at all. Positions are on the UNBROKEN scale (the
        // domain-0..1000-over-pixel-400..0 mapping `clip_node_into` receives
        // its node coordinates in, before any break remap runs): data 100 ->
        // pixel 400 + (100/1000)*(0-400) = 360.
        let node = SceneNode::Rect {
            x: 10.0,
            y: 360.0,
            w: 20.0,
            h: 40.0, // up to pixel 400 (data 0)
            style: fill_stroke(),
            corner_radius: 8.0,
        };
        let clipped = guide_remap(&br).clip_nodes(vec![node]);
        assert_eq!(clipped.nodes.len(), 1, "no gap touches this bar; got {:?}", clipped.nodes);
        match &clipped.nodes[0] {
            SceneNode::Rect { corner_radius, .. } => assert_eq!(*corner_radius, 8.0),
            other => panic!("an uncut piece must stay a plain Rect, not a Path; got {other:?}"),
        }
    }

    /// Quality-review cycle-3 finding 3 (S2): a piece cut on BOTH edges of
    /// the split axis — the middle slice of a two-gap break — has no corner
    /// left to round. Folding "no corner survives" into the Rect-vs-Path
    /// gate keeps it a plain `Rect { corner_radius: 0.0 }` rather than a
    /// `Path` that would draw the identical four-line box while forfeiting
    /// hit-testing, R-tree membership, packing eligibility and the centre
    /// rotation anchor for no visual gain.
    #[test]
    fn corner_radius_fully_cut_middle_piece_stays_a_plain_rect_with_zero_radius() {
        // Same domain/pixel-range/break-size convention as `guide_break`,
        // with a second gap so the middle retained segment (data 300..400)
        // is bounded on both sides by a gap rather than by the domain edge.
        let br = apply_break_to_scale(
            (0.0, 1000.0),
            &[[100.0, 300.0], [400.0, 800.0]],
            (400.0, 0.0),
            12.0,
        );
        let remap = BreakRemap::new("y", (0.0, 1000.0), (400.0, 0.0), &br);
        // Data 0..900, same shape as the guide's own worked example.
        let node = SceneNode::Rect {
            x: 10.0,
            y: 40.0,
            w: 20.0,
            h: 360.0,
            style: fill_stroke(),
            corner_radius: 8.0,
        };
        let clipped = remap.clip_nodes(vec![node]);
        assert_eq!(
            clipped.nodes.len(),
            3,
            "three visible segments intersect the bar (0..100, 300..400, 800..900); got {:?}",
            clipped.nodes
        );

        // Order-independent: exactly one piece (the fully-cut middle slice)
        // must be a plain zero-radius `Rect`; the other two (cut on only one
        // edge) must be `Path`s with their two uncut corners still rounded.
        let (mut fully_cut, mut partly_cut) = (0, 0);
        for piece in &clipped.nodes {
            match piece {
                SceneNode::Rect { corner_radius, .. } => {
                    assert_eq!(
                        *corner_radius, 0.0,
                        "the only Rect piece must be the fully-cut one; got {piece:?}"
                    );
                    fully_cut += 1;
                }
                SceneNode::Path { commands, .. } => {
                    assert_eq!(
                        arc_count(commands),
                        2,
                        "a piece cut on only one edge keeps two rounded corners; got {commands:?}"
                    );
                    partly_cut += 1;
                }
                other => panic!("expected Rect or Path; got {other:?}"),
            }
        }
        assert_eq!(
            (fully_cut, partly_cut),
            (1, 2),
            "exactly one fully-cut piece (Rect) and two partly-cut pieces (Path); got {:?}",
            clipped.nodes
        );
    }

    /// Quality-review cycle-3 finding 1 (S3): `SceneNode::Rect` rotates
    /// about its own centre (`svg_walk.rs::emit_node`'s `Rect` arm), while
    /// `SceneNode::Path` rotates about its first `MoveTo`. Converting a
    /// rotated, rounded, cut piece to a `Path` would silently relocate its
    /// rotation anchor relative to its unbroken rounded sibling (measured:
    /// hundreds of pixels), and WASM drops the rotation for a `Path`
    /// entirely (`tessellate.rs` never reads `style.angle`). The gate keeps
    /// a rotated piece a plain `Rect` — original, uncut radius and all —
    /// rather than corrupt its anchor; the un-squared cut edge that results
    /// is a documented residual (`ferrum-spec.md`'s #230-2 dated note), not
    /// a silent regression.
    #[test]
    fn corner_radius_rotated_piece_keeps_the_plain_rect_and_its_anchor() {
        let br = guide_break();
        let mut style = fill_stroke();
        style.angle = 30.0;
        let node = SceneNode::Rect {
            x: 10.0,
            y: 40.0,
            w: 20.0,
            h: 360.0,
            style,
            corner_radius: 8.0,
        };
        let clipped = guide_remap(&br).clip_nodes(vec![node]);
        assert_eq!(
            clipped.nodes.len(),
            2,
            "a rotated bar still splits into two pieces; got {:?}",
            clipped.nodes
        );
        for piece in &clipped.nodes {
            match piece {
                SceneNode::Rect { corner_radius, style, .. } => {
                    assert_eq!(
                        *corner_radius, 8.0,
                        "a rotated piece keeps its original, un-squared radius; got {piece:?}"
                    );
                    assert!((style.angle - 30.0).abs() < 1e-9, "rotation must survive the clip");
                }
                other => panic!(
                    "a rotated cut piece must stay a Rect (centre-anchored rotation), not a Path; got {other:?}"
                ),
            }
        }
    }

    /// The number of `ArcTo` commands in a path — how many of a clipped
    /// rect piece's four corners kept their rounding.
    fn arc_count(cmds: &[PathCmd]) -> usize {
        cmds.iter().filter(|c| matches!(c, PathCmd::ArcTo { .. })).count()
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

    /// An `Area` batch carries BOTH halves of one mark: the closed fill, which
    /// clips, and `mark_area(line=True)`/`borders=True`'s open border stroke,
    /// which CUTS. They have to agree — a clipped fill under an outline running
    /// straight through the break band is one mark contradicting itself
    /// (quality-review cycle-3 finding).
    #[test]
    fn an_open_border_path_in_an_extent_batch_is_cut_like_its_fill() {
        let br = guide_break();
        // The fill: top edge at data 900 (pixel 40), baseline at data 0.
        let fill = SceneNode::Path {
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
        // The border: the top line, rising from data 0 to data 900 across the
        // gap, so it is cut at data 150 (pixel 206) and resumes at data 850
        // (pixel 194) with `x` interpolated along the stroke at each crossing.
        let border = SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 0.0, y: 400.0 },
                PathCmd::LineTo { x: 100.0, y: 40.0 },
            ],
            style: fill_stroke(),
            closed: false,
        };
        let clipped = guide_remap(&br).clip_nodes(vec![fill, border]);
        assert_eq!(clipped.nodes.len(), 4, "2 fill bands + 2 border pieces");
        assert_eq!(clipped.sources, vec![0, 0, 1, 1], "provenance names both halves");
        for node in &clipped.nodes[..2] {
            let SceneNode::Path { closed: true, .. } = node else { panic!("fill stays closed") };
        }
        assert_path_commands(
            &clipped.nodes[2],
            &[
                PathCmd::MoveTo { x: 0.0, y: 400.0 },
                PathCmd::LineTo { x: 100.0 / 6.0, y: 206.0 },
            ],
            "the border's lower piece ends ON the near gap edge",
        );
        assert_path_commands(
            &clipped.nodes[3],
            &[
                PathCmd::MoveTo { x: 100.0 * 850.0 / 900.0, y: 194.0 },
                PathCmd::LineTo { x: 100.0, y: 194.0 - 50.0 * 194.0 / 150.0 },
            ],
            "and its upper piece resumes on the far edge",
        );
        for node in &clipped.nodes[2..] {
            let SceneNode::Path { closed: false, .. } = node else { panic!("border stays open") };
            for coord in path_coords_of(node) {
                assert!(!is_hidden(coord), "no sentinel in a cut border; got {coord}");
            }
        }
    }

    /// The border shape the old fallback spiked on: a vertex INSIDE the gap.
    /// `remap_node` placed it at `BREAK_HIDDEN`, so the outline detoured far
    /// off-canvas and back while the fill beneath it was clipped correctly.
    #[test]
    fn an_open_border_path_with_a_vertex_inside_a_gap_is_cut_not_spiked() {
        let br = guide_break();
        let border = SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 0.0, y: 400.0 },
                // Data 500: inside the (150, 850) gap.
                PathCmd::LineTo { x: 50.0, y: 200.0 },
                PathCmd::LineTo { x: 100.0, y: 40.0 },
            ],
            style: fill_stroke(),
            closed: false,
        };
        let clipped = guide_remap(&br).clip_nodes(vec![border]);
        assert_eq!(clipped.nodes.len(), 2, "cut into two; got {:?}", clipped.nodes);
        assert!(clipped.is_reindexed(), "so the batch's sidecars realign through the same record");
        assert_path_commands(
            &clipped.nodes[0],
            &[
                PathCmd::MoveTo { x: 0.0, y: 400.0 },
                // Crossing at data 150, 150/500 of the way along the first edge.
                PathCmd::LineTo { x: 15.0, y: 206.0 },
            ],
            "the piece before the in-gap vertex ends at the near edge",
        );
        assert_path_commands(
            &clipped.nodes[1],
            &[
                // Crossing at data 850, 350/400 of the way along the second edge.
                PathCmd::MoveTo { x: 93.75, y: 194.0 },
                PathCmd::LineTo { x: 100.0, y: 194.0 - 50.0 * 194.0 / 150.0 },
            ],
            "and the piece after it starts at the far edge",
        );
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
            (MarkBatchKind::Segment, MoveVerticesAndGlyph),
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
        // The connector `Line` branch ignores `anchor` entirely — it places
        // both of its own ends directly — so this value is inert.
        remap.remap_label_node(&mut leader, (10.0, 360.0));
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
        // The dash's exact data anchor: its two drawn ends are symmetric
        // around their midpoint, so this equals the pre-#230-3 bbox anchor.
        remap.remap_glyph_node(&mut dash, (10.0, 396.0));
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
        remap.remap_label_node(&mut connector, (10.0, 396.0));
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
        // Exact data anchor: symmetric drawn ends, so it matches the
        // pre-#230-3 bbox midpoint.
        remap.remap_glyph_node(&mut dash, (10.0, 408.0));
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
        // The square's own exact data anchor: symmetric around its center,
        // pixel 360 (data 100).
        remap.remap_glyph_node(&mut square, (10.0, 360.0));
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
        // The diamond's own exact data anchor: symmetric, centered on data 500.
        remap.remap_glyph_node(&mut diamond, (10.0, 200.0));
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
            // A circle's own coordinate IS its exact data anchor.
            remap.remap_glyph_node(&mut glyph, (5.0, cy));
            remap.remap_node(&mut plain);
            assert_eq!(glyph, plain, "glyph and plain remap must agree exactly for cy={cy}");
        }
    }

    /// #230-3, all 8 `ShapeKind`s, exercised through the real shape-emission
    /// path (`marks::point::emit_shape_nodes` — the same function
    /// `marks::point::build` calls per row, not a hand-mirrored geometry).
    ///
    /// A shape whose drawn vertices are symmetric around its data position —
    /// `Circle`/`Square`/`Diamond`/`Cross`/`VLine`/`HLine` — has an exact data
    /// anchor equal to its own bbox midpoint, so remapping on either one must
    /// be byte-identical: this is `#230-3`'s "unchanged" contract, and the
    /// reason every symmetric-shape point-mark break golden (the slash/
    /// zigzag/wave/bar/area goldens all use circles) stays byte-stable.
    /// `TriangleUp`/`TriangleDown` are NOT symmetric (their bbox midpoint
    /// sits `h/4` off the true anchor — the exact defect the measured repro
    /// in `scene_build.rs`'s `triangle_up_renders_wherever_a_circle_at_the_\
    /// same_y_renders` catches), so their two remaps must DIFFER.
    #[test]
    fn every_point_shape_kind_is_covered_by_the_exact_anchor_contract() {
        use crate::render::marks::point::{emit_shape_nodes, ShapeStyle};
        use crate::render::scale_resolve::ShapeKind;

        let br = guide_break();
        let remap = guide_remap(&br);
        let style = || ShapeStyle {
            fill: None, fill_cleared: false, stroke: None, stroke_cleared: false,
            stroke_width: 1.0, opacity: 1.0, stroke_opacity: 1.0, fill_opacity: 1.0,
            stroke_dash: None, angle: 0.0,
        };
        // Anchored at data 100 (pixel 360, per `guide_break`'s worked example).
        let (cx, cy, r) = (10.0, 360.0, 5.0);

        let remap_via_exact_anchor = |kind: ShapeKind| -> Vec<SceneNode> {
            let mut nodes = emit_shape_nodes(kind, cx, cy, r, style());
            for node in nodes.iter_mut() {
                remap.remap_glyph_node(node, (cx, cy));
            }
            nodes
        };
        let remap_via_bbox_anchor = |kind: ShapeKind| -> Vec<SceneNode> {
            let mut nodes = emit_shape_nodes(kind, cx, cy, r, style());
            for node in nodes.iter_mut() {
                let anchor = remap.bbox_anchor_fallback(node);
                remap.remap_glyph_node(node, anchor);
            }
            nodes
        };

        for kind in [
            ShapeKind::Circle, ShapeKind::Square, ShapeKind::Diamond,
            ShapeKind::Cross, ShapeKind::VLine, ShapeKind::HLine,
        ] {
            assert_eq!(
                remap_via_exact_anchor(kind),
                remap_via_bbox_anchor(kind),
                "{kind:?}: a symmetric shape's exact-anchor remap must be byte-identical \
                 to its (bbox-midpoint) pre-#230-3 remap"
            );
        }
        for kind in [ShapeKind::TriangleUp, ShapeKind::TriangleDown] {
            assert_ne!(
                remap_via_exact_anchor(kind),
                remap_via_bbox_anchor(kind),
                "{kind:?}: an asymmetric shape's exact-anchor remap must DIFFER from its \
                 bbox-midpoint remap — this asymmetry is #230-3's bug"
            );
        }
    }

    /// The per-vertex FALLBACK, pinned on the node kind that first needed it.
    ///
    /// `mark_line` emits a Polyline; before this arm existed it was the one mark
    /// kind a break left entirely alone, drawing the line against the unbroken
    /// scale while its own ticks compressed. A line batch no longer comes
    /// through `remap_node` at all — it splits (`split_nodes`, #230-1) — but
    /// this is still the model a curved `Path` falls back to, and this is what
    /// that model does to connected geometry: the in-gap vertex parks at the
    /// sentinel and the stroke stays attached to it. The
    /// `a_curved_path_falls_back_to_the_per_vertex_model` test below is the
    /// boundary between the two.
    #[test]
    fn polyline_vertices_are_placed_on_the_broken_scale() {
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

    // ── Vertex splitting (#230-1) ────────────────────────────────────────────
    //
    // Fixture arithmetic, all against `guide_break()` (domain 0..1000, pixels
    // 400..0, gap (150, 850), break_size 12):
    //   data_at(px) = (400 - px) * 2.5      pixel_at(d) = 400 - 0.4 * d
    //   segment 0: data [0, 150]   pixels 400 -> 206
    //   segment 1: data [850,1000] pixels 194 -> 0
    // So 206 and 194 are the two gap EDGES in placed pixel space, and every
    // crossing point below is expected to land exactly on one of them.

    fn polyline(points: Vec<(f64, f64)>) -> SceneNode {
        SceneNode::Polyline { points, style: stroke() }
    }

    fn polyline_points(node: &SceneNode) -> Vec<(f64, f64)> {
        let SceneNode::Polyline { points, .. } = node else { panic!("expected a Polyline") };
        points.clone()
    }

    fn line_ends(node: &SceneNode) -> (f64, f64, f64, f64) {
        let SceneNode::Line { x1, y1, x2, y2, .. } = node else { panic!("expected a Line") };
        (*x1, *y1, *x2, *y2)
    }

    /// The measured #230-1 repro: a polyline whose middle vertex is inside the
    /// gap emitted `points="…,420.961 …,-99999 …,16"`, a spike out of the
    /// viewport and back. It now becomes one sub-polyline per visible piece,
    /// cut at the gap edges, and the sentinel appears nowhere.
    #[test]
    fn a_polyline_crossing_a_gap_splits_instead_of_spiking_through_the_sentinel() {
        let br = guide_break();
        // data 0 (px 400) -> data 500 (px 200, mid-gap) -> data 900 (px 40).
        let split = guide_remap(&br).split_nodes(vec![polyline(vec![
            (0.0, 400.0),
            (10.0, 200.0),
            (20.0, 40.0),
        ])]);
        assert_eq!(split.nodes.len(), 2, "one sub-polyline per visible piece");
        let lower = polyline_points(&split.nodes[0]);
        let upper = polyline_points(&split.nodes[1]);
        // No connecting geometry and no sentinel: every coordinate is a real
        // pixel inside the two visible bands.
        for (x, y) in lower.iter().chain(upper.iter()) {
            assert!(*y >= 0.0 && *y <= 400.0, "placed inside the pixel range; got ({x}, {y})");
            assert_ne!(*y, BREAK_HIDDEN);
        }
        // The lower piece runs from the first vertex to the gap's near edge…
        assert_eq!(lower.len(), 2);
        assert!((lower[0].1 - 400.0).abs() < 1e-9, "data 0 keeps the segment floor");
        assert!((lower[1].1 - 206.0).abs() < 1e-9, "it stops at the gap edge; got {:?}", lower[1]);
        // …and the upper piece from the gap's far edge to the last vertex.
        assert_eq!(upper.len(), 2);
        assert!((upper[0].1 - 194.0).abs() < 1e-9, "it resumes at the far edge; got {:?}", upper[0]);
        assert!(
            (upper[1].1 - (194.0 - 194.0 / 3.0)).abs() < 1e-6,
            "data 900 lands where the compressed scale puts it; got {:?}",
            upper[1]
        );
        assert!(upper[1].1 < lower[0].1, "the two pieces are separated by the break");
    }

    /// The geometry contract for a vertex inside a gap: the crossing point is
    /// INTERPOLATED along the edge, not dropped back to the last visible
    /// vertex. Dropping back would bend the line towards that vertex and invent
    /// a slope the data never had.
    #[test]
    fn a_gap_crossing_interpolates_the_cross_axis_coordinate_at_the_gap_edge() {
        let br = guide_break();
        let split = guide_remap(&br).split_nodes(vec![polyline(vec![
            (0.0, 400.0),
            (10.0, 200.0),
            (20.0, 40.0),
        ])]);
        let lower = polyline_points(&split.nodes[0]);
        let upper = polyline_points(&split.nodes[1]);
        // Leaving: data runs 0 -> 500 while x runs 0 -> 10, so the crossing at
        // data 150 sits 150/500 of the way along: x = 3.
        assert!((lower[1].0 - 3.0).abs() < 1e-9, "x is interpolated at the exit; got {:?}", lower[1]);
        // Re-entering: data runs 500 -> 900 while x runs 10 -> 20, so the
        // crossing at data 850 sits 350/400 of the way along: x = 18.75.
        assert!(
            (upper[0].0 - 18.75).abs() < 1e-9,
            "x is interpolated at the re-entry; got {:?}",
            upper[0]
        );
        // The dishonest alternative, pinned as NOT what happens.
        assert_ne!(lower[1].0, 0.0, "the exit is not the last visible vertex's x");
        assert_ne!(upper[0].0, 20.0, "the re-entry is not the next visible vertex's x");
    }

    /// One edge can cross SEVERAL gaps: the whole of a segment it passes
    /// straight through is still drawn, so a two-gap break yields three pieces
    /// from a single two-vertex line.
    #[test]
    fn one_edge_spanning_two_gaps_emits_a_piece_per_segment_it_traverses() {
        let br = apply_break_to_scale(
            (0.0, 1000.0),
            &[[150.0, 400.0], [600.0, 850.0]],
            (400.0, 0.0),
            12.0,
        );
        let remap = BreakRemap::new("y", (0.0, 1000.0), (400.0, 0.0), &br);
        // One straight line from data 0 (px 400) to data 1000 (px 0).
        let split = remap.split_nodes(vec![SceneNode::Line {
            x1: 0.0,
            y1: 400.0,
            x2: 100.0,
            y2: 0.0,
            style: stroke(),
        }]);
        assert_eq!(split.nodes.len(), 3, "three visible segments, three pieces");
        let mut spans = Vec::new();
        for node in &split.nodes {
            let (x1, y1, x2, y2) = line_ends(node);
            for c in [y1, y2] {
                assert!((-1e-9..=400.0).contains(&c), "no sentinel, no overshoot: {c}");
            }
            assert!(x1 < x2, "each piece runs in the line's own direction");
            spans.push((y1, y2));
        }
        // Pieces come in walk order (data ascending = pixel descending).
        assert!(spans[0].0 > spans[1].0 && spans[1].0 > spans[2].0, "walk order: {spans:?}");
    }

    /// A stroke wholly inside a gap is DROPPED, not parked at the sentinel: a
    /// hidden `Line` at `BREAK_HIDDEN` is still a node in the scene, still
    /// serialized, still hit-testable off-canvas.
    #[test]
    fn a_stroke_wholly_inside_a_gap_is_dropped() {
        let br = guide_break();
        // data 400 -> 600, both inside the (150, 850) gap.
        let split = guide_remap(&br).split_nodes(vec![SceneNode::Line {
            x1: 10.0,
            y1: 240.0,
            x2: 90.0,
            y2: 160.0,
            style: stroke(),
        }]);
        assert!(split.nodes.is_empty(), "nothing survives; got {:?}", split.nodes);
        assert!(split.is_reindexed(), "so the batch's sidecar row goes with it");
        assert_eq!(split.realign(Some(vec![7usize])), Some(vec![]), "the row is dropped");
    }

    /// A rule PERPENDICULAR to the break (both ends at one data value, the
    /// whisker-cap and hline shape) has a zero-length data extent: it survives
    /// whole when visible and vanishes when its value is in a gap. Its drawn
    /// length across the axis is untouched.
    #[test]
    fn a_rule_perpendicular_to_the_break_survives_whole_or_not_at_all() {
        let br = guide_break();
        let remap = guide_remap(&br);
        let visible = remap.split_nodes(vec![SceneNode::Line {
            x1: 10.0,
            y1: 360.0,
            x2: 90.0,
            y2: 360.0,
            style: stroke(),
        }]);
        assert_eq!(visible.nodes.len(), 1, "one piece for a visible rule");
        let (x1, y1, x2, y2) = line_ends(&visible.nodes[0]);
        assert_eq!((x1, x2), (10.0, 90.0), "the cross-axis extent is untouched");
        assert_eq!(y1, y2, "it stays perpendicular");
        assert!((y1 - remap.coord(360.0).unwrap()).abs() < 1e-9, "on the compressed position");

        let in_gap = remap.split_nodes(vec![SceneNode::Line {
            x1: 10.0,
            y1: 200.0,
            x2: 90.0,
            y2: 200.0,
            style: stroke(),
        }]);
        assert!(in_gap.nodes.is_empty(), "an in-gap rule is dropped, not hidden off-canvas");
    }

    /// A whisker-shaped rule (two data values, spanning the gap) splits the way
    /// the polyline does — the boxplot case, since `mark_boxplot`'s whiskers are
    /// a `rule` layer with `y`/`y2`.
    #[test]
    fn a_ranged_rule_spanning_a_gap_splits_into_one_line_per_segment() {
        let br = guide_break();
        let split = guide_remap(&br).split_nodes(vec![SceneNode::Line {
            x1: 30.0,
            y1: 360.0,
            x2: 30.0,
            y2: 40.0,
            style: stroke(),
        }]);
        assert_eq!(split.nodes.len(), 2, "one line per visible segment");
        for node in &split.nodes {
            let (x1, y1, x2, y2) = line_ends(node);
            assert_eq!((x1, x2), (30.0, 30.0), "a vertical rule stays vertical");
            for c in [y1, y2] {
                assert!(c >= 0.0 && c <= 400.0, "no sentinel spike: {c}");
            }
        }
        let (_, _, _, lower_end) = line_ends(&split.nodes[0]);
        let (_, upper_start, _, _) = line_ends(&split.nodes[1]);
        assert!((lower_end - 206.0).abs() < 1e-9, "cut at the near gap edge; got {lower_end}");
        assert!((upper_start - 194.0).abs() < 1e-9, "resumed at the far edge; got {upper_start}");
    }

    /// The split boundary, side one: a straight-line open `Path` — the spelling
    /// `mark_line`'s three `step*` interpolations emit — splits like a polyline,
    /// re-spelled as explicit `MoveTo`/`LineTo`.
    #[test]
    fn a_step_path_of_straight_commands_splits() {
        let br = guide_break();
        // MoveTo data 0, HLineTo (the riser's horizontal half), VLineTo to data
        // 900: the step spelling, crossing the gap on its vertical leg.
        let step = SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 0.0, y: 400.0 },
                PathCmd::HLineTo { x: 10.0 },
                PathCmd::VLineTo { y: 40.0 },
            ],
            style: fill_stroke(),
            closed: false,
        };
        let split = guide_remap(&br).split_nodes(vec![step]);
        assert_eq!(split.nodes.len(), 2, "the vertical leg is cut at the gap; got {:?}", split.nodes);
        for node in &split.nodes {
            let SceneNode::Path { commands, closed, .. } = node else { panic!("expected a Path") };
            assert!(!closed, "an open path stays open");
            assert!(
                commands.iter().all(|c| matches!(c, PathCmd::MoveTo { .. } | PathCmd::LineTo { .. })),
                "pieces are re-spelled with explicit vertices; got {commands:?}"
            );
            for cmd in commands {
                let (PathCmd::MoveTo { y, .. } | PathCmd::LineTo { y, .. }) = cmd else { continue };
                assert!(*y >= 0.0 && *y <= 400.0, "no sentinel in a split step line: {y}");
            }
        }
    }

    /// A step path spelled with `H`/`V` shorthand.
    ///
    /// `guide_break`'s scale: domain `[0, 1000]` over pixels `400..0`, gap
    /// `(150, 850)`, so segment 0 carries data `0..150` over pixels `400..206`
    /// and data `d` there places at `400 - d * 194 / 150`.
    fn step_path(commands: Vec<PathCmd>) -> SceneNode {
        SceneNode::Path { commands, style: fill_stroke(), closed: false }
    }

    fn path_commands_of(node: &SceneNode) -> &[PathCmd] {
        let SceneNode::Path { commands, .. } = node else { panic!("expected a Path; got {node:?}") };
        commands
    }

    /// The coordinates one straight-line command carries.
    fn cmd_coords(cmd: &PathCmd) -> Vec<f64> {
        match *cmd {
            PathCmd::MoveTo { x, y } | PathCmd::LineTo { x, y } => vec![x, y],
            PathCmd::HLineTo { x } => vec![x],
            PathCmd::VLineTo { y } => vec![y],
            ref other => panic!("unexpected command in a split step path: {other:?}"),
        }
    }

    /// Every coordinate a path's commands carry, whatever the spelling.
    fn path_coords_of(node: &SceneNode) -> Vec<f64> {
        path_commands_of(node).iter().flat_map(cmd_coords).collect()
    }

    /// Assert a path's commands match `expected` — spelling exactly, pixels to
    /// within float noise, since a hand-computed pixel and a placed one reach
    /// the same value by different arithmetic and the last ULP is not
    /// contractual.
    fn assert_path_commands(node: &SceneNode, expected: &[PathCmd], ctx: &str) {
        let actual = path_commands_of(node);
        assert_eq!(actual.len(), expected.len(), "{ctx}; got {actual:?}");
        for (got, want) in actual.iter().zip(expected) {
            assert_eq!(
                std::mem::discriminant(got),
                std::mem::discriminant(want),
                "{ctx}; got {actual:?}"
            );
            for (g, w) in cmd_coords(got).iter().zip(cmd_coords(want)) {
                assert!((g - w).abs() < 1e-9, "{ctx}; got {actual:?}");
            }
        }
    }

    /// The uncut case: a step path no gap reaches keeps its `H`/`V` shorthand
    /// and its coordinates are the BROKEN scale's — pinned against
    /// hand-computed pixels, not against `remap_node`'s own output.
    ///
    /// Cycle 2 pinned this by asserting equality with `remap_node`'s output on
    /// the same source. That guard is worthless: `remap_node` IS the arm the
    /// uncut branch takes, so the assertion held whatever the fallback emitted
    /// — sentinel included (spec-review cycle-3 finding). Hand-computed pixels
    /// cannot be satisfied by a sentinel.
    #[test]
    fn an_uncut_step_path_keeps_its_shorthand_on_the_broken_scale() {
        let br = guide_break();
        // Vertices at data 0 and data 50, both interior to segment 0: no edge
        // of this path reaches the gap at data 150.
        let split = guide_remap(&br).split_nodes(vec![step_path(vec![
            PathCmd::MoveTo { x: 0.0, y: 400.0 },
            PathCmd::HLineTo { x: 10.0 },
            PathCmd::VLineTo { y: 380.0 },
        ])]);
        assert_eq!(split.nodes.len(), 1, "an uncut path is not split into pieces");
        assert!(!split.is_reindexed(), "and no sidecar row is disturbed");
        let commands = path_commands_of(&split.nodes[0]);
        // Data 0 places at pixel 400 (segment 0's own low edge) and data 50 at
        // 400 - 50 * 194 / 150 = 335.333…, which is NOT where the unbroken
        // scale put it (380) — the shorthand survives a real compression.
        let (PathCmd::MoveTo { x, y }, PathCmd::HLineTo { x: hx }, PathCmd::VLineTo { y: vy }) =
            (&commands[0], &commands[1], &commands[2])
        else {
            panic!("H/V shorthand must survive when nothing cut the path; got {commands:?}")
        };
        assert_eq!(commands.len(), 3, "and no command is added or dropped; got {commands:?}");
        assert_eq!((*x, *y, *hx), (0.0, 400.0, 10.0), "got {commands:?}");
        assert!((*vy - (400.0 - 50.0 * 194.0 / 150.0)).abs() < 1e-9, "got {commands:?}");
    }

    /// A cut that DROPS a boundary vertex and INSERTS a crossing in the same
    /// edge nets to zero point-count change, so the piece comes back one-piece
    /// with the source's own vertex count — indistinguishable from an uncut
    /// walk by counting, or by comparing the output vertices to the input's.
    /// `split_polyline` reports cut-ness itself for exactly this reason, and
    /// what it buys is that the in-gap vertex is CUT away at the gap edge
    /// instead of being handed to `remap_node`, which parks it at
    /// `BREAK_HIDDEN` (spec-review cycle-3 finding: the #230-1 spike,
    /// reproduced from ordinary `mark_line(interpolate="step-after")` user
    /// code).
    ///
    /// Both boundaries, since the count nets to zero at either end: a riser
    /// LAST carries the final vertex into the gap, a riser FIRST starts inside
    /// one.
    #[test]
    fn a_step_path_whose_boundary_vertex_is_in_a_gap_is_cut_not_hidden() {
        let br = guide_break();
        let remap = guide_remap(&br);
        // Near gap edge: data 150 places at pixel 206; data 50 at 335.333….
        let (near_edge, data_50) = (206.0, 400.0 - 50.0 * 194.0 / 150.0);

        // Last vertex in the gap: data 0 → 0 → 500, the last inside (150, 850).
        let last_in_gap = remap.split_nodes(vec![step_path(vec![
            PathCmd::MoveTo { x: 0.0, y: 400.0 },
            PathCmd::HLineTo { x: 10.0 },
            PathCmd::VLineTo { y: 200.0 },
        ])]);
        assert_eq!(last_in_gap.nodes.len(), 1, "one surviving piece; got {:?}", last_in_gap.nodes);
        assert_path_commands(
            &last_in_gap.nodes[0],
            &[
                PathCmd::MoveTo { x: 0.0, y: 400.0 },
                PathCmd::LineTo { x: 10.0, y: 400.0 },
                PathCmd::LineTo { x: 10.0, y: near_edge },
            ],
            "the in-gap last vertex is cut away at the near gap edge, and the \
             surviving piece is re-spelled with explicit vertices",
        );

        // First vertex in the gap: data 500 → 50 → 50, the first inside it.
        let first_in_gap = remap.split_nodes(vec![step_path(vec![
            PathCmd::MoveTo { x: 0.0, y: 200.0 },
            PathCmd::VLineTo { y: 380.0 },
            PathCmd::HLineTo { x: 10.0 },
        ])]);
        assert_eq!(first_in_gap.nodes.len(), 1, "one surviving piece; got {:?}", first_in_gap.nodes);
        assert_path_commands(
            &first_in_gap.nodes[0],
            &[
                PathCmd::MoveTo { x: 0.0, y: near_edge },
                PathCmd::LineTo { x: 0.0, y: data_50 },
                PathCmd::LineTo { x: 10.0, y: data_50 },
            ],
            "the in-gap first vertex is cut away at the near gap edge too",
        );

        for split in [&last_in_gap, &first_in_gap] {
            for coord in path_coords_of(&split.nodes[0]) {
                assert!(
                    !is_hidden(coord) && (0.0..=400.0).contains(&coord),
                    "no coordinate of a split piece may be the sentinel; got {coord}"
                );
            }
        }
    }

    /// The other half of the pair: a path a gap cuts into SEVERAL pieces still
    /// re-spells each of them, so the shorthand's survival above is a property
    /// of cut-ness rather than of the path never being split.
    #[test]
    fn a_step_path_cut_into_pieces_re_spells_every_piece() {
        let br = guide_break();
        let split = guide_remap(&br).split_nodes(vec![step_path(vec![
            PathCmd::MoveTo { x: 0.0, y: 400.0 },
            PathCmd::HLineTo { x: 10.0 },
            PathCmd::VLineTo { y: 40.0 },
        ])]);
        assert_eq!(split.nodes.len(), 2, "the vertical leg is cut at the gap");
        for node in &split.nodes {
            assert!(
                path_commands_of(node)
                    .iter()
                    .all(|c| matches!(c, PathCmd::MoveTo { .. } | PathCmd::LineTo { .. })),
                "a cut piece is re-spelled with explicit vertices; got {node:?}"
            );
            for coord in path_coords_of(node) {
                assert!(!is_hidden(coord), "and carries no sentinel; got {coord}");
            }
        }
    }

    /// The split boundary, side two: a CURVED path keeps the per-vertex
    /// fallback, sentinel and all. Cutting one honestly would need de
    /// Casteljau subdivision of the placed curve; guessing a straight cut
    /// through a `CubicTo` would silently redraw the curve.
    #[test]
    fn a_curved_path_falls_back_to_the_per_vertex_model() {
        let br = guide_break();
        let curved = SceneNode::Path {
            commands: vec![
                PathCmd::MoveTo { x: 0.0, y: 400.0 },
                PathCmd::CubicTo { c1x: 3.0, c1y: 300.0, c2x: 7.0, c2y: 200.0, x: 10.0, y: 200.0 },
                PathCmd::CubicTo { c1x: 13.0, c1y: 150.0, c2x: 17.0, c2y: 80.0, x: 20.0, y: 40.0 },
            ],
            style: fill_stroke(),
            closed: false,
        };
        let mut expected = curved.clone();
        let remap = guide_remap(&br);
        remap.remap_node(&mut expected);
        let split = remap.split_nodes(vec![curved]);
        assert_eq!(split.nodes.len(), 1, "a curve is not split");
        assert!(!split.is_reindexed(), "so no sidecar is disturbed");
        assert_eq!(split.nodes[0], expected, "it is exactly the per-vertex remap");
    }

    /// `BREAK_HIDDEN` is inert under the split too (#230-9): a coordinate a
    /// prior step already hid has no data position, so the runs around it END
    /// there rather than being clamped back into the domain.
    ///
    /// The fixture has real runs on both sides of the poisoned vertex, so the
    /// "absent, not clamped" assertions run against actual pieces. A three-point
    /// fixture would not: both of its edges touch the sentinel, so both are
    /// dropped, no piece exists, and every per-point assertion passes
    /// vacuously (quality-review cycle-3 finding). That all-poisoned shape is
    /// pinned below as its own outcome instead.
    #[test]
    fn a_poisoned_vertex_is_never_clamped_back_into_view_by_the_split() {
        let br = guide_break();
        // Data 0 / 50 / — / 100 / 150, all inside segment 0 (pixels 400..206,
        // where data `d` places at `400 - d * 194 / 150`), with the middle
        // vertex already parked at the sentinel by an earlier fold.
        let split = guide_remap(&br).split_nodes(vec![polyline(vec![
            (0.0, 400.0),
            (5.0, 380.0),
            (10.0, BREAK_HIDDEN),
            (15.0, 360.0),
            (20.0, 340.0),
        ])]);
        assert_eq!(split.nodes.len(), 2, "one run each side of it; got {:?}", split.nodes);
        let runs: Vec<Vec<(f64, f64)>> = split.nodes.iter().map(polyline_points).collect();
        let want = [
            // Data 0 → 50: ends AT the last live vertex, not at the sentinel.
            vec![(0.0, 400.0), (5.0, 400.0 - 50.0 * 194.0 / 150.0)],
            // Data 100 → 150: resumes at the next live vertex.
            vec![(15.0, 400.0 - 100.0 * 194.0 / 150.0), (20.0, 206.0)],
        ];
        for (got, expected) in runs.iter().zip(&want) {
            assert_eq!(got.len(), expected.len(), "runs: {runs:?}");
            for (&(gx, gy), &(wx, wy)) in got.iter().zip(expected) {
                assert!(
                    (gx - wx).abs() < 1e-9 && (gy - wy).abs() < 1e-9,
                    "hand-computed placement; got {runs:?}"
                );
            }
        }
        // The poisoned vertex contributes no point of its own — the positive
        // form of "never clamped back into view": x == 10 is its cross-axis
        // coordinate, and no run carries it.
        let placed: Vec<(f64, f64)> = runs.concat();
        assert!(
            placed.iter().all(|&(x, _)| x != 10.0),
            "the hidden vertex is absent, not clamped; got {placed:?}"
        );
        assert!(
            placed.iter().all(|&(_, y)| y >= 0.0 && y <= 400.0),
            "and no run resurrects the sentinel; got {placed:?}"
        );

        // The degenerate shape: every edge touches the sentinel, so every edge
        // is dropped and NOTHING survives — the same fate, and the same sidecar
        // bookkeeping, as a stroke wholly inside a gap.
        let all_poisoned = guide_remap(&br).split_nodes(vec![polyline(vec![
            (0.0, 400.0),
            (10.0, BREAK_HIDDEN),
            (20.0, 40.0),
        ])]);
        assert!(all_poisoned.nodes.is_empty(), "got {:?}", all_poisoned.nodes);
        assert!(all_poisoned.is_reindexed(), "so its sidecar row goes with it");
        assert_eq!(all_poisoned.realign(Some(vec![7usize])), Some(vec![]), "the row is dropped");
    }

    /// A 1→N split realigns the batch's sidecars through the same
    /// [`ClippedNodes`] bookkeeping the extent clip uses — one mechanism, one
    /// provenance record.
    #[test]
    fn a_split_realigns_its_sidecars_through_the_clip_machinery() {
        let br = guide_break();
        let nodes = vec![
            // Crosses the gap: two pieces.
            polyline(vec![(0.0, 400.0), (20.0, 40.0)]),
            // Wholly inside the gap: no pieces.
            polyline(vec![(0.0, 240.0), (20.0, 160.0)]),
            // Wholly visible: one piece.
            polyline(vec![(0.0, 400.0), (20.0, 360.0)]),
        ];
        let split = guide_remap(&br).split_nodes(nodes);
        assert_eq!(split.nodes.len(), 3, "2 + 0 + 1 pieces");
        assert!(split.is_reindexed());
        assert_eq!(
            split.realign(Some(vec!["a", "b", "c"])),
            Some(vec!["a", "a", "c"]),
            "each piece inherits its source row; the dropped row vanishes"
        );
        // The `keys` sidecar is the one exception: it suffixes rather than
        // copying verbatim, on this exact fixture (dropped-source shape,
        // quality-review cycle-2 fix 4b).
        assert_eq!(
            split.realign_keys(Some(vec!["a".to_string(), "b".to_string(), "c".to_string()])),
            Some(vec!["a".to_string(), "a#1".to_string(), "c".to_string()]),
            "the crossing row's second piece is suffixed; the dropped row's key vanishes with it"
        );
    }

    /// An unbroken axis must leave vertex geometry exactly where it was — the
    /// byte-stability anchor for every non-break line chart.
    #[test]
    fn a_break_with_no_effective_gap_leaves_a_polyline_untouched() {
        let br = one_segment_break(0.0, 1000.0, 400.0, 0.0);
        let source = polyline(vec![(0.0, 400.0), (10.0, 200.0), (20.0, 40.0)]);
        let split = BreakRemap::new("y", (0.0, 1000.0), (400.0, 0.0), &br)
            .split_nodes(vec![source.clone()]);
        assert_eq!(split.nodes, vec![source], "no gap, no movement, no re-spelling");
        assert!(!split.is_reindexed(), "and no sidecar is disturbed");
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

    /// Cross-axis chrome is excluded BY THE CALLER (an axis knows which
    /// dimension it was placed along — `marks::axis::AxisNodes::placed_along`),
    /// not by this function or by [`BreakRemap::remap_axis_tick_node`]: neither
    /// checks that a node's `anchor` actually belongs to this break's axis: it
    /// trusts whatever anchor it is handed and translates rigidly on it. Pinned
    /// here as the characteristic the caller must respect — passing a foreign
    /// anchor (here: an x-tick's OWN 50px tick position, reused as if it were
    /// a y anchor) does not error; it silently produces a corrupted position.
    #[test]
    fn axis_chrome_handed_a_foreign_anchor_is_placed_without_axis_validation() {
        let br = guide_break();
        // A downward x-axis tick dash: drawn from the plot floor (y=400) to
        // 4px below it (its drawn length, not data). Its own tick anchor is
        // an X pixel (50.0) — nonsense as a Y anchor, which is the point.
        let mut nodes = vec![marks::axis::AxisTickNode {
            node: SceneNode::Line {
                x1: 50.0,
                y1: 400.0,
                x2: 50.0,
                y2: 404.0,
                style: crate::render::draw::to_scene_stroke(
                    palette::Srgba::new(0, 0, 0, 255),
                    1.0,
                    1.0,
                    None,
                    None,
                    None,
                ),
            },
            anchor: 50.0,
        }];
        guide_remap(&br).remap_axis_chrome(&mut nodes);
        let SceneNode::Line { y1, y2, .. } = nodes[0].node else { panic!() };
        // data_at(50) = 0 + (50 - 400)/(0 - 400) * 1000 = 875, which sits in
        // the upper segment [850, 1000] at pixel 194 + (25/150)*(0-194) ≈
        // 161.667; both ends translate by that displacement minus the anchor
        // (≈ 111.667), landing well past the plot's own pixel range [0, 400].
        assert!(y1 > 400.0, "a foreign-axis anchor drives the dash off the plot; got y1={y1}");
        assert!(
            (y2 - y1 - 4.0).abs() < 1e-9,
            "the dash's own 4px drawn length must survive even a corrupted placement; got {} vs {}",
            y2, y1
        );
    }

    // ── Reversed scales (close-review S3) ────────────────────────────────────

    /// A reversed scale reports a DESCENDING `data_domain()`. Clipping the
    /// gaps against it endpoint-wise discarded every one of them, so
    /// `LinearScale(reverse=True)` + `BreakAxis(gap=(150, 850))` drew an
    /// unbroken chart: all three points rendered, no indicator. The domain's
    /// orientation now belongs to the pixel assignment only.
    #[test]
    fn a_descending_domain_still_breaks() {
        // Reversed y: data 0 at the TOP pixel (20), data 1000 at the BOTTOM
        // (320) — the pixel range is the axis's own, unchanged.
        let br = apply_break_to_scale((1000.0, 0.0), &[[150.0, 850.0]], (320.0, 20.0), 12.0);

        assert_eq!(br.effective_gaps, 1, "the gap is inside the domain either way");
        assert_eq!(br.segments.len(), 2, "one segment per side of the gap");
        assert_eq!(br.indicator_pixels.len(), 1, "the break draws its indicator");
        assert!(
            broken_scale_map(500.0, &br).is_none(),
            "a value inside the gap has no pixel on a reversed axis either"
        );
        // Retained span 300, so each half takes (300 - 12) / 2 = 144 pixels,
        // laid out from the pixel the SMALLEST data value owns.
        assert!((broken_scale_map(0.0, &br).unwrap() - 20.0).abs() < 1e-9);
        assert!((broken_scale_map(1000.0, &br).unwrap() - 320.0).abs() < 1e-9);
        assert!((broken_scale_map(150.0, &br).unwrap() - 164.0).abs() < 1e-9);
        assert!((broken_scale_map(850.0, &br).unwrap() - 176.0).abs() < 1e-9);
    }

    /// The ascending case is untouched by the orientation handling — the same
    /// numbers the guide's worked example has always produced.
    #[test]
    fn an_ascending_domain_is_unchanged_by_the_orientation_handling() {
        let br = guide_break();
        assert_eq!(br.effective_gaps, 1);
        assert!((br.segments[0].pixel_lo - 400.0).abs() < 1e-9);
        assert!((br.segments[1].pixel_hi - 0.0).abs() < 1e-9);
    }

    /// Gaps entirely outside the domain leave a scale that is not broken at
    /// all. `effective_gaps` is how the caller tells that apart from a real
    /// break and warns instead of rendering a silently ignored feature.
    #[test]
    fn out_of_domain_gaps_report_zero_effective_gaps() {
        let br = apply_break_to_scale((0.0, 100.0), &[[150.0, 850.0]], (320.0, 20.0), 12.0);
        assert_eq!(br.effective_gaps, 0);
        assert_eq!(br.segments.len(), 1, "one segment spanning the whole domain");
        assert!(br.indicator_pixels.is_empty());
    }

    /// A domain-edge-touching gap removes real data without drawing an
    /// indicator — effective, and must NOT be reported as a no-op.
    #[test]
    fn an_edge_touching_gap_is_still_effective() {
        let br = apply_break_to_scale((0.0, 1000.0), &[[0.0, 200.0]], (400.0, 0.0), 12.0);
        assert_eq!(br.effective_gaps, 1);
        assert!(br.indicator_pixels.is_empty(), "no indicator, but the data is gone");
        assert!(broken_scale_map(100.0, &br).is_none());
    }

    // ── gap_spanned_by (close-review S3: containment inversion) ──────────────

    /// The interval that CONTAINS a gap is the case `gap_containing` cannot
    /// see: neither endpoint is inside, yet the interval's interior is.
    #[test]
    fn gap_spanned_by_finds_the_gap_an_interval_reaches_across() {
        let br = guide_break();
        assert_eq!(gap_spanned_by(100.0, 900.0, &br), Some((150.0, 850.0)));
        assert_eq!(gap_spanned_by(900.0, 100.0, &br), Some((150.0, 850.0)), "either order");
        assert_eq!(gap_spanned_by(0.0, 100.0, &br), None, "wholly on one side");
        assert_eq!(gap_spanned_by(900.0, 1000.0, &br), None, "wholly on the other");
        assert_eq!(
            gap_spanned_by(100.0, 400.0, &br),
            None,
            "a partial overlap has an in-gap endpoint; that is gap_containing's question"
        );
    }

    // ── Segment arrowheads (close-review S3 size, cycle-2 N3 direction) ──────

    fn stroke() -> ferrum_scene::StrokeStyle {
        crate::render::draw::to_scene_stroke(
            palette::Srgba::new(0, 0, 0, 255),
            1.0,
            1.0,
            None,
            None,
            None,
        )
    }

    /// One `mark_segment(arrow=True)` row as the batch carries it: the shaft
    /// `Line` followed by its arrowhead `Path`, built by the same shared
    /// helper `marks::segment::build` uses.
    fn segment_row(
        start: (f64, f64),
        tip: (f64, f64),
        head_size: f64,
    ) -> Vec<SceneNode> {
        let points = crate::render::draw::arrowhead_triangle(
            start.0, start.1, tip.0, tip.1, head_size,
        )
        .expect("non-degenerate segment yields a triangle");
        vec![
            SceneNode::Line {
                x1: start.0,
                y1: start.1,
                x2: tip.0,
                y2: tip.1,
                style: stroke(),
            },
            crate::render::draw::arrowhead_path(points, ferrum_scene::Color::rgb(0, 0, 0)),
        ]
    }

    fn triangle(node: &SceneNode) -> [(f64, f64); 3] {
        triangle_vertices(node).expect("arrowhead triangle")
    }

    /// `(head_size, angle)` of an arrowhead: the distance and direction from
    /// the base midpoint to the tip, i.e. the two properties a break must not
    /// change and must keep aligned with the shaft.
    fn head_size_and_angle(node: &SceneNode) -> (f64, f64) {
        let [tip, left, right] = triangle(node);
        let base = ((left.0 + right.0) / 2.0, (left.1 + right.1) / 2.0);
        let (dx, dy) = (tip.0 - base.0, tip.1 - base.1);
        (dx.hypot(dy), dy.atan2(dx))
    }

    fn shaft_angle(node: &SceneNode) -> f64 {
        let SceneNode::Line { x1, y1, x2, y2, .. } = node else { panic!("expected the shaft") };
        (y2 - y1).atan2(x2 - x1)
    }

    /// The head must keep its drawn SIZE — placing its three vertices one by
    /// one scaled it by the segment's own compression ratio (a 20px head
    /// became ~90px and sheared) — AND keep pointing along the shaft it
    /// terminates, which rigid translation did not: it froze the pre-break
    /// direction, leaving a measured 16.19° head-vs-shaft mismatch.
    ///
    /// Under the split (#230-1) the shaft it terminates is the PIECE that
    /// reaches the tip, not the whole compressed shaft: the head rides that
    /// piece's own direction, which is the geometry a reader sees it attached
    /// to.
    #[test]
    fn a_segment_arrowhead_keeps_its_size_and_follows_its_shaft_under_a_break() {
        let br = guide_break();
        // The reviewer's fixture in pixel space: data y=100 (pixel 360) up to
        // data y=900 (pixel 40), crossing the (150, 850) gap.
        let nodes = segment_row((10.0, 360.0), (100.0, 40.0), 20.0);
        let (size_before, angle_before) = head_size_and_angle(&nodes[1]);
        let angle_unbroken_shaft = shaft_angle(&nodes[0]);
        assert!(
            (angle_before - angle_unbroken_shaft).abs() < 1e-9,
            "test sanity: the unbroken head already points along its shaft"
        );

        let placed = guide_remap(&br).split_segment_nodes(nodes).nodes;
        assert_eq!(placed.len(), 3, "two shaft pieces plus one head; got {placed:?}");
        let head = &placed[2];
        let tip_piece = &placed[1];
        let (size_after, angle_after) = head_size_and_angle(head);
        let placed_shaft_angle = shaft_angle(tip_piece);

        assert!(
            (size_after - size_before).abs() < 1e-9,
            "the head must keep its drawn size: {size_before} -> {size_after}"
        );
        assert!(
            (placed_shaft_angle - angle_unbroken_shaft).abs() > 0.05,
            "test sanity: the break must actually change the shaft's slope, or \
             the direction pin proves nothing ({angle_unbroken_shaft} -> {placed_shaft_angle})"
        );
        assert!(
            (angle_after - placed_shaft_angle).abs() < 1e-9,
            "the head must point along the PLACED piece it terminates, not its \
             pre-break direction: head={angle_after} shaft={placed_shaft_angle}"
        );
        // And it still sits at that piece's placed end.
        let [tip, _, _] = triangle(head);
        let (_, _, x2, y2) = line_ends(tip_piece);
        assert!(
            (tip.0 - x2).abs() < 1e-9 && (tip.1 - y2).abs() < 1e-9,
            "the head's tip is the tip piece's own end: {tip:?} vs ({x2}, {y2})"
        );
        // Neither shaft piece carries a sentinel: the shaft is cut, not spiked.
        for piece in &placed[..2] {
            let (_, y1, _, y2) = line_ends(piece);
            assert!(y1 >= 0.0 && y1 <= 400.0 && y2 >= 0.0 && y2 <= 400.0, "cut, not spiked");
        }
    }

    /// A head whose tip lands in a gap is DROPPED: the shaft's pieces stop at
    /// the gap edge, so there is no visible tip for a head to terminate. It used
    /// to be parked at `BREAK_HIDDEN` — a full triangle still in the scene,
    /// still serialized, off-canvas.
    #[test]
    fn a_segment_arrowhead_is_dropped_when_its_tip_is_in_a_gap() {
        let br = guide_break();
        // Data 500 (mid-gap) sits at pixel 200 on the unbroken 0..1000 scale;
        // the shaft starts at data 50 (pixel 380), so it has one visible piece.
        let nodes = segment_row((10.0, 380.0), (100.0, 200.0), 20.0);
        let split = guide_remap(&br).split_segment_nodes(nodes);
        assert_eq!(split.nodes.len(), 1, "the shaft's visible piece alone; got {:?}", split.nodes);
        assert!(
            matches!(split.nodes[0], SceneNode::Line { .. }),
            "and it is the shaft, not the head"
        );
        let (_, y1, _, y2) = line_ends(&split.nodes[0]);
        assert!((y2 - 206.0).abs() < 1e-9, "the shaft stops at the gap edge; got {y2}");
        assert!(y1 > y2, "running up towards it");
        // The row's sidecars follow: the head's entry goes with the head.
        assert_eq!(split.realign(Some(vec!["shaft", "head"])), Some(vec!["shaft"]));
    }

    /// The knife edge behind the "does a piece REACH the tip" test: a tip lying
    /// exactly on a segment's edge, approached from inside the adjacent gap,
    /// places as VISIBLE (segment bounds are inclusive) while contributing no
    /// length to that segment. The shaft's only piece therefore belongs to a
    /// different part of it, and its direction is not the tip's — so the head is
    /// dropped rather than rebuilt on a direction it has nothing to do with, or
    /// floated at a pixel no stroke reaches.
    #[test]
    fn an_arrowhead_whose_tip_only_touches_a_segment_edge_is_dropped() {
        let br = guide_break();
        let remap = guide_remap(&br);
        // Pixel 60 is data 850 — segment 1's lower bound exactly. The shaft
        // starts at data 100 (pixel 360), so it crosses segment 0 and the gap.
        assert!(remap.coord(60.0).is_some(), "test sanity: the tip places as visible");
        let split = remap.split_segment_nodes(segment_row((10.0, 360.0), (100.0, 60.0), 20.0));
        assert_eq!(split.nodes.len(), 1, "one shaft piece, no head; got {:?}", split.nodes);
        assert!(matches!(split.nodes[0], SceneNode::Line { .. }), "and it is the shaft piece");
        let (_, _, _, end) = line_ends(&split.nodes[0]);
        assert!(
            (end - 206.0).abs() < 1e-9,
            "which stops at segment 0's edge — 12px short of the placed tip; got {end}"
        );
    }

    /// A head with no shaft in front of it (a malformed batch) still lands on
    /// its own placed tip, at its own size — the documented fallback.
    #[test]
    fn a_lone_arrowhead_falls_back_to_rigid_translation() {
        let br = guide_break();
        let mut nodes = segment_row((10.0, 360.0), (100.0, 40.0), 20.0);
        nodes.remove(0);
        let (size_before, angle_before) = head_size_and_angle(&nodes[0]);
        let placed = guide_remap(&br).split_segment_nodes(nodes).nodes;
        assert_eq!(placed.len(), 1);
        let (size_after, angle_after) = head_size_and_angle(&placed[0]);
        assert!((size_after - size_before).abs() < 1e-9, "size survives");
        assert!((angle_after - angle_before).abs() < 1e-9, "its own direction survives");
        let [tip, _, _] = triangle(&placed[0]);
        assert!(
            (tip.1 - guide_remap(&br).coord(40.0).unwrap()).abs() < 1e-9,
            "and it rides its placed tip"
        );
    }

    /// A shaft splits like any other stroke — that is what lets a segment span a
    /// break without drawing through it. Both of its ends are still their own
    /// data positions; what changed is that the gap between them is a CUT rather
    /// than a sentinel detour.
    #[test]
    fn a_segment_shaft_splits_at_the_gap() {
        let br = guide_break();
        let remap = guide_remap(&br);
        let split = remap.split_segment_nodes(vec![SceneNode::Line {
            x1: 10.0,
            y1: 380.0,
            x2: 100.0,
            y2: 40.0,
            style: stroke(),
        }]);
        assert_eq!(split.nodes.len(), 2, "one piece per visible segment");
        let (_, first_start, _, first_end) = line_ends(&split.nodes[0]);
        let (_, last_start, _, last_end) = line_ends(&split.nodes[1]);
        assert_eq!(first_start, remap.coord(380.0).unwrap(), "the start keeps its own position");
        assert_eq!(last_end, remap.coord(40.0).unwrap(), "so does the end");
        assert!((first_end - 206.0).abs() < 1e-9, "cut at the near gap edge; got {first_end}");
        assert!((last_start - 194.0).abs() < 1e-9, "resumed at the far edge; got {last_start}");
    }

    // ── Sidecar realignment reporting (close-review "still feels wrong") ─────

    /// A wrong-length sidecar still passes through untouched (a release build
    /// must not panic), and the realigner REMEMBERS that it did, so the caller
    /// can warn instead of silently shipping misattributed tooltips.
    ///
    /// Runs in the DEBUG gate: the realigner answers the alignment question
    /// itself and returns the sidecar without calling
    /// [`ClippedNodes::realign`], whose `debug_assert` still guards a direct
    /// caller. Before that split, the only reachable branch of
    /// `BreakClipSidecarMisaligned` was release-only, i.e. covered by no gate
    /// the project runs.
    #[test]
    fn the_realigner_reports_a_misaligned_sidecar() {
        let br = guide_break();
        // A bar crossing the gap: one source node, two clipped pieces.
        let node = SceneNode::Rect {
            x: 10.0,
            y: 40.0,
            w: 20.0,
            h: 340.0,
            style: fill_stroke(),
            corner_radius: 0.0,
        };
        let clipped = guide_remap(&br).clip_nodes(vec![node]);
        assert!(clipped.is_reindexed());
        assert!(clipped.accepts_sidecar_len(1), "one entry per PRE-clip node");
        assert!(
            !clipped.accepts_sidecar_len(2),
            "post-clip lengths are exactly the misalignment this reports"
        );

        let mut aligned = clipped.realigner();
        assert_eq!(aligned.realign(Some(vec![7usize])), Some(vec![7usize, 7usize]));
        assert!(!aligned.misaligned(), "a right-length sidecar realigns cleanly");

        let mut broken = clipped.realigner();
        assert_eq!(
            broken.realign(Some(vec![7usize, 8usize])),
            Some(vec![7usize, 8usize]),
            "a wrong-length sidecar is passed through, never indexed into"
        );
        assert!(broken.misaligned(), "and the pass-through is recorded");
    }

    /// [`SidecarRealign::realign_keys`]'s own wrong-length guard (duplicated
    /// from [`SidecarRealign::realign`] before quality-review cycle-2 fix 2,
    /// now the shared [`SidecarRealign::note_if_misaligned`]) had zero
    /// coverage in either language — this pins the `keys` sidecar going
    /// through the exact same pass-through-and-record path as every other
    /// sidecar.
    #[test]
    fn the_realigner_reports_a_misaligned_keys_sidecar() {
        let br = guide_break();
        let node = SceneNode::Rect {
            x: 10.0,
            y: 40.0,
            w: 20.0,
            h: 340.0,
            style: fill_stroke(),
            corner_radius: 0.0,
        };
        let clipped = guide_remap(&br).clip_nodes(vec![node]);
        assert!(clipped.is_reindexed());

        let mut aligned = clipped.realigner();
        assert_eq!(
            aligned.realign_keys(Some(vec!["k".to_string()])),
            Some(vec!["k".to_string(), "k#1".to_string()]),
            "a right-length keys sidecar realigns (and suffixes) cleanly"
        );
        assert!(!aligned.misaligned(), "a right-length sidecar realigns cleanly");

        let mut broken = clipped.realigner();
        assert_eq!(
            broken.realign_keys(Some(vec!["k".to_string(), "j".to_string()])),
            Some(vec!["k".to_string(), "j".to_string()]),
            "a wrong-length keys sidecar is passed through, never indexed into"
        );
        assert!(broken.misaligned(), "and the pass-through is recorded");
    }

    /// A batch broken on BOTH axes runs `install_rebuilt_nodes` — and
    /// [`ClippedNodes::realign_keys`] — once per axis (`scene_build.rs`'s
    /// `break_remaps` loop), so the second pass's `keys` sidecar is the
    /// FIRST pass's already-suffixed output. Quality-review cycle-2 finding
    /// 1 (S3): resuming the `#1, #2, ...` counter naively in the second call
    /// reproduces `"ka#1"` from two different pieces — measured pre-fix as
    /// `["ka", "ka#1", "ka#1", "ka#1#1"]`. This pins the two-call
    /// composition directly (no scene/scale machinery needed: two
    /// back-to-back splits of the same node, the second call fed the
    /// first's output, is exactly what two axis breaks compose into).
    #[test]
    fn realign_keys_composes_across_two_remap_passes_without_colliding() {
        let br = guide_break();
        // First pass (e.g. the x-axis break): one source node splits into two
        // pieces.
        let first_pass = guide_remap(&br).split_nodes(vec![polyline(vec![
            (0.0, 400.0),
            (20.0, 40.0),
        ])]);
        let first_keys = first_pass
            .realign_keys(Some(vec!["ka".to_string()]))
            .expect("keyed sidecar");
        assert_eq!(first_keys, vec!["ka".to_string(), "ka#1".to_string()]);

        // Second pass (e.g. the y-axis break): each of the two pieces from
        // pass one splits again, so `sources` groups two runs of two.
        let second_pass = guide_remap(&br).split_nodes(vec![
            polyline(vec![(0.0, 400.0), (20.0, 40.0)]),
            polyline(vec![(0.0, 400.0), (20.0, 40.0)]),
        ]);
        let second_keys = second_pass
            .realign_keys(Some(first_keys))
            .expect("keyed sidecar");
        assert_eq!(
            second_keys.len(),
            4,
            "two runs of two pieces each; got {second_keys:?}"
        );
        let unique: std::collections::HashSet<&String> = second_keys.iter().collect();
        assert_eq!(
            unique.len(),
            4,
            "every piece must be a distinct key, even composed across two \
             remap passes; got {second_keys:?}"
        );
        assert_eq!(
            second_keys,
            vec![
                "ka".to_string(),
                "ka#2".to_string(),
                "ka#1".to_string(),
                "ka#1#1".to_string(),
            ],
            "pass two's suffix search skips \"ka#1\" because pass one's own \
             output already used it; got {second_keys:?}"
        );
    }
}
